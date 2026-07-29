from __future__ import annotations

import io

import numpy as np
import torch
import pytest

from training.fork_policy_iteration import fork_run
from training.batch_sweep import SweepConfig
from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import POLICY_EXECUTION_VERSION
from training.policy_iteration import PolicyStateBatch, policy_outputs
from training.progress import Progress
from training.search_distillation import build_rank_lcb_mirror_target
from training.tests.test_world_outcomes import tiny_world_batch
from training.train import (
    CHECKPOINT_VERSION,
    LEGACY_CHECKPOINT_VERSION,
    STATELESS_CHECKPOINT_VERSION,
    OPPONENT_POOL_CAPACITY,
    OPPONENT_REFRESH_INTERVAL,
    OpponentPool,
    SELF_PLAY_RANK_DELTA_STEP,
    RunConfig,
    SelfPlayCurriculum,
    _advance_self_play,
    _advance_opponent_pool,
    _cleanup_committed_pending,
    _committed_iteration,
    _checkpoint_payload,
    _initial_opponent_pool,
    _load_checkpoint,
    _load_opponent_actor,
    _model_digest,
    _arena_safety_guard,
    _early_promotion_rejection,
    _promotion_decision,
    _RankLcbOptimizationBatch,
    _optimize_rank_lcb_generation,
    _save_checkpoint,
    _select_opponent_snapshot,
    build_parser,
)


def tiny_actor() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=8,
        )
    )


def changed_actor(offset: float) -> BloodFlowTransformer:
    actor = tiny_actor()
    with torch.no_grad():
        next(actor.parameters()).add_(offset)
    return actor


def test_training_defaults_encode_the_validated_update() -> None:
    config = RunConfig()
    assert config.queries_per_category == 256
    assert 9 * config.queries_per_category == 2304
    assert config.worlds == 16
    assert config.world_chunk == 64
    assert config.envs == 2048
    assert config.target_shard_size == 128
    assert config.target_query_batch_size == 128
    assert config.rollout_inference_batch_size == 128
    assert config.target_kl == pytest.approx(0.001)
    assert config.direction_optimizer == "adamw"
    assert config.direction_momentum == pytest.approx(0.9)
    assert config.direction_gradient_clip_norm == pytest.approx(1.0)
    assert config.policy_objective == "expected_q"
    assert config.world_sampling == "live_wall"
    assert config.kl_control == "target"
    assert config.split_consensus_margin == pytest.approx(0.125)
    assert config.validation_worlds == 64
    assert config.audit_worlds == 32
    assert config.generation_batches == 4
    assert config.target_fdr == pytest.approx(0.05)
    assert config.mirror_temperature == pytest.approx(0.05)
    assert config.mirror_prior_floor == pytest.approx(1e-6)
    assert config.arena_games == 65_536
    assert config.evaluation_games == 16_384
    assert config.evaluation_envs == 4096
    assert config.self_play_start_first_rate == pytest.approx(0.55)
    assert config.self_play_increment == pytest.approx(0.10)
    assert config.maximum_self_play_fraction == pytest.approx(2 / 3)
    assert config.anchor_rule_fast is False


def evaluation(
    rank_mean: float,
    *,
    rank_ci95_high: float = -0.001,
    score_ci95_high: float = 1.0,
    first_rate: float = 0.30,
) -> dict[str, object]:
    return {
        "actor": {"first_rate": first_rate},
        "paired_rank_delta": {
            "mean": rank_mean,
            "ci95_high": rank_ci95_high,
        },
        "paired_score_delta": {"ci95_high": score_ci95_high},
    }


def arena_result(
    rank_low: float,
    rank_high: float,
    *,
    score_high: float = 1.0,
) -> dict[str, object]:
    return {
        "paired_rank_delta": {
            "mean": 0.5 * (rank_low + rank_high),
            "ci95_low": rank_low,
            "ci95_high": rank_high,
        },
        "paired_score_delta": {"ci95_high": score_high},
    }


def test_promotion_needs_pooled_improvement_and_all_safety_guards() -> None:
    fixed = arena_result(-0.003, 0.001)
    historical = arena_result(-0.004, 0.001)
    pooled = arena_result(-0.003, -0.0001)
    audit = {"ci95_high": 0.001}

    decision = _promotion_decision(fixed, historical, pooled, audit)

    assert decision["promoted"] is True
    harmful = _promotion_decision(
        arena_result(0.001, 0.003), historical, pooled, audit
    )
    assert harmful["promoted"] is False
    assert harmful["reason"] == "safety_guard_failed"
    uncertain = _promotion_decision(
        fixed, historical, arena_result(-0.001, 0.001), audit
    )
    assert uncertain["promoted"] is False
    assert uncertain["reason"] == "insufficient_pooled_rank_evidence"


def test_irrevocable_safety_failure_has_an_explicit_early_rejection() -> None:
    guard = _arena_safety_guard(arena_result(0.001, 0.003, score_high=-1.0))
    assert guard == {"rank_harm": True, "score_harm": True}

    decision = _early_promotion_rejection(
        "fixed_arena_guard_failed",
        audit_harm=False,
        arena_guards={"fixed": guard},
    )

    assert decision["promoted"] is False
    assert decision["pooled_rank_improvement"] is None
    assert decision["early_rejection"] is True
    assert decision["arena_guards"] == {"fixed": guard}


def test_rank_lcb_recipe_rejects_old_protocol_combinations() -> None:
    with pytest.raises(ValueError, match="Nesterov"):
        RunConfig(policy_objective="rank_lcb_mirror_ce")
    with pytest.raises(ValueError, match="KL cap"):
        RunConfig(
            policy_objective="rank_lcb_mirror_ce",
            direction_optimizer="nesterov",
        )
    with pytest.raises(ValueError, match="live-wall"):
        RunConfig(
            policy_objective="rank_lcb_mirror_ce",
            direction_optimizer="nesterov",
            kl_control="cap",
            world_sampling="information_set",
        )
    config = RunConfig(
        policy_objective="rank_lcb_mirror_ce",
        direction_optimizer="nesterov",
        kl_control="cap",
        world_sampling="live_wall",
    )
    assert config.generation_batches == 4


def test_generation_inner_steps_share_one_cumulative_champion_kl_ball() -> None:
    champion = tiny_actor()
    world = tiny_world_batch()
    probabilities, reference_actions = policy_outputs(
        champion, world, torch.device("cpu"), batch_size=16
    )
    rank = world.rank_outcomes.copy()
    rows = np.arange(len(world))
    rank[rows, reference_actions] = -3
    alternatives = world.legal.copy()
    alternatives[rows, reference_actions] = False
    selected = alternatives.argmax(axis=1)
    rank[rows, selected] = 3
    improved = world.__class__(
        **{
            **{
                name: getattr(world, name)
                for name in world.__class__.__dataclass_fields__
            },
            "rank_outcomes": rank,
        }
    )
    target, target_metrics = build_rank_lcb_mirror_target(
        improved,
        improved,
        probabilities,
        reference_actions,
        temperature=0.5,
    )
    generation_batch = _RankLcbOptimizationBatch(
        training=improved.counterfactual_batch(),
        target=target,
        reference_actions=reference_actions,
        source_metrics={},
        target_metrics=target_metrics,
    )
    calibration = PolicyStateBatch(
        **{
            name: getattr(world, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
    )
    config = RunConfig(
        policy_objective="rank_lcb_mirror_ce",
        direction_optimizer="nesterov",
        direction_learning_rate=0.01,
        direction_gradient_clip_norm=0.0,
        kl_control="cap",
        world_sampling="live_wall",
        generation_batches=2,
        target_kl=1e-4,
    )
    candidate, optimizer, calibration_metrics = (
        _optimize_rank_lcb_generation(
            champion,
            (generation_batch, generation_batch),
            calibration,
            torch.device("cpu"),
            Progress(stream=io.StringIO(), tty=False, log_interval=1e9),
            attempt=62,
            config=config,
            visit_weights=np.full(9, 1 / 9),
        )
    )

    assert candidate is not champion
    assert optimizer["generation_inner_steps"] == 2
    assert optimizer["checkpoint_state_reset"] is True
    assert len(calibration_metrics["steps"]) == 2
    assert all(
        0 <= step["final_kl"] <= config.target_kl
        for step in calibration_metrics["steps"]
    )
    assert all(step["wall_seconds"] >= 0 for step in optimizer["steps"])
    assert all(
        step["wall_seconds"] >= 0 for step in calibration_metrics["steps"]
    )
    assert calibration_metrics["final_kl"] <= config.target_kl
    assert set(calibration_metrics["policy_change"]["categories"]) == {
        "exchange_first",
        "exchange_second",
        "exchange_third",
        "choose_missing",
        "turn_early",
        "turn_middle",
        "turn_late",
        "hu_response",
        "meld_response",
    }


def test_self_play_curriculum_uses_paired_rank_milestones_and_is_capped() -> None:
    config = RunConfig()
    state = SelfPlayCurriculum()
    state = _advance_self_play(
        state,
        evaluation(-0.006, first_rate=0.27),
        next_iteration=2,
        config=config,
    )
    assert state.fraction == pytest.approx(0.1)
    assert state.activation_iteration == 2
    assert state.last_fixed_first_rate == pytest.approx(0.27)

    state = _advance_self_play(
        state, evaluation(-0.006), next_iteration=3, config=config
    )
    assert state.fraction == pytest.approx(0.1)
    state = _advance_self_play(
        state, evaluation(-0.010), next_iteration=4, config=config
    )
    assert state.fraction == pytest.approx(0.2)
    state = _advance_self_play(
        state, evaluation(-0.022), next_iteration=5, config=config
    )
    assert state.fraction == pytest.approx(0.3)

    state = _advance_self_play(
        state, evaluation(-0.20), next_iteration=6, config=config
    )
    assert state.fraction == pytest.approx(2 / 3)
    assert state.activation_iteration == 2


def test_self_play_curriculum_requires_rank_evidence_and_score_guard() -> None:
    config = RunConfig()
    state = SelfPlayCurriculum(last_fixed_first_rate=0.27)
    state = _advance_self_play(
        state,
        evaluation(-0.02, rank_ci95_high=0.001),
        next_iteration=2,
        config=config,
    )
    assert state.fraction == 0.0
    state = _advance_self_play(
        state,
        evaluation(-0.02, score_ci95_high=-1.0),
        next_iteration=3,
        config=config,
    )
    assert state.fraction == 0.0


def test_self_play_curriculum_does_not_downgrade_legacy_checkpoint_state() -> None:
    config = RunConfig()
    state = SelfPlayCurriculum(
        fraction=0.4,
        last_fixed_first_rate=0.55,
        activation_iteration=3,
    )
    state = _advance_self_play(
        state,
        evaluation(-0.006, rank_ci95_high=0.01, score_ci95_high=-1.0),
        next_iteration=10,
        config=config,
    )
    assert state.fraction == pytest.approx(0.4)
    assert state.activation_iteration == 3
    assert state.last_fixed_first_rate == pytest.approx(0.30)


def test_self_play_rank_delta_step_is_one_hundredth_of_average_rank() -> None:
    assert SELF_PLAY_RANK_DELTA_STEP == pytest.approx(0.01)


def test_checkpoint_roundtrip_persists_explicit_optimizer_state(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    config = RunConfig()
    payload = _checkpoint_payload(
        actor,
        config=config,
        root_seed=7,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(
            fraction=0.2,
            last_fixed_first_rate=0.57,
            activation_iteration=2,
        ),
        opponent_pool=_initial_opponent_pool(actor),
        last_metrics={"iteration": 2, "policy_version_after": 2},
    )
    assert set(payload) == {
        "version",
        "engine_rules_version",
        "policy_execution_version",
        "config",
        "root_seed",
        "next_iteration",
        "champion_iteration",
        "sl_checkpoint",
        "sl_sha256",
        "self_play",
        "opponent_pool",
        "model_config",
        "actor",
        "direction_optimizer_state",
        "last_metrics",
    }
    path = tmp_path / "latest.pt"
    _save_checkpoint(path, payload)
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()
    loaded_actor, loaded_config, loaded = _load_checkpoint(
        path, torch.device("cpu")
    )
    assert loaded["version"] == CHECKPOINT_VERSION
    assert loaded["next_iteration"] == 3
    assert loaded["self_play"] == SelfPlayCurriculum(
        fraction=0.2,
        last_fixed_first_rate=0.57,
        activation_iteration=2,
    )
    assert loaded_config == config
    assert isinstance(loaded["opponent_pool"], OpponentPool)
    assert loaded["direction_optimizer_state"] == {}
    assert len(loaded["opponent_pool"].snapshots) == 1
    for expected, actual in zip(actor.parameters(), loaded_actor.parameters()):
        torch.testing.assert_close(expected, actual)

    payload["version"] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="production format"):
        _load_checkpoint(path, torch.device("cpu"))


def test_v3_checkpoint_loads_without_a_pool_for_direct_resume(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    current = _checkpoint_payload(
        actor,
        config=RunConfig(),
        root_seed=7,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(
            fraction=0.2,
            last_fixed_first_rate=0.30,
            activation_iteration=2,
        ),
        opponent_pool=_initial_opponent_pool(actor),
        last_metrics={"iteration": 2, "policy_version_after": 2},
    )
    legacy = dict(current)
    legacy["version"] = LEGACY_CHECKPOINT_VERSION
    legacy.pop("opponent_pool")
    legacy.pop("direction_optimizer_state")
    legacy.pop("champion_iteration")
    path = tmp_path / "latest.pt"
    torch.save(legacy, path)

    _loaded_actor, _loaded_config, loaded = _load_checkpoint(
        path, torch.device("cpu")
    )
    assert loaded["opponent_pool"] is None
    assert loaded["direction_optimizer_state"] == {}
    assert loaded["legacy_self_play_opponent"] is True


def test_v4_checkpoint_migrates_to_stateless_adamw(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    payload = _checkpoint_payload(
        actor,
        config=RunConfig(),
        root_seed=7,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(),
        opponent_pool=_initial_opponent_pool(actor),
        last_metrics={"iteration": 2, "policy_version_after": 2},
    )
    payload["version"] = STATELESS_CHECKPOINT_VERSION
    payload.pop("direction_optimizer_state")
    payload.pop("champion_iteration")
    path = tmp_path / "latest.pt"
    torch.save(payload, path)

    _loaded_actor, config, loaded = _load_checkpoint(path, torch.device("cpu"))

    assert config.direction_optimizer == "adamw"
    assert config.direction_momentum == pytest.approx(0.9)
    assert loaded["direction_optimizer_state"] == {}


def test_nesterov_checkpoint_roundtrip_restores_velocity(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    config = RunConfig(
        direction_optimizer="nesterov",
        direction_learning_rate=0.1,
    )
    state = {
        name: torch.full_like(parameter, 1e-5)
        for name, parameter in actor.named_parameters()
    }
    payload = _checkpoint_payload(
        actor,
        config=config,
        root_seed=7,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(),
        opponent_pool=_initial_opponent_pool(actor),
        last_metrics={"iteration": 2, "policy_version_after": 2},
        direction_optimizer_state=state,
    )
    path = tmp_path / "latest.pt"
    _save_checkpoint(path, payload)

    _loaded_actor, loaded_config, loaded = _load_checkpoint(
        path, torch.device("cpu")
    )

    assert loaded_config.direction_optimizer == "nesterov"
    assert set(loaded["direction_optimizer_state"]) == set(state)
    for name, value in state.items():
        torch.testing.assert_close(loaded["direction_optimizer_state"][name], value)


def test_resume_parser_does_not_invent_training_overrides() -> None:
    args = build_parser().parse_args(["--resume", "runs/example/latest.pt"])
    assert args.resume.name == "latest.pt"
    assert args.seed is None
    assert args.queries_per_category is None
    assert args.anchor_rule_fast is None
    anchored = build_parser().parse_args(["--anchor-rule-fast"])
    assert anchored.anchor_rule_fast is True


def test_checkpoint_rejects_inconsistent_last_metrics(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    payload = _checkpoint_payload(
        actor,
        config=RunConfig(),
        root_seed=1,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(),
        opponent_pool=_initial_opponent_pool(actor),
        last_metrics={"iteration": 1, "policy_version_after": 2},
    )
    path = tmp_path / "latest.pt"
    _save_checkpoint(path, payload)
    with pytest.raises(ValueError, match="last_metrics"):
        _load_checkpoint(path, torch.device("cpu"))


def test_resume_cleanup_only_removes_committed_pending(tmp_path) -> None:
    pending = tmp_path / "pending"
    for name in ("iteration-000001", "iteration-000002", "keep-me"):
        (pending / name).mkdir(parents=True)
        (pending / name / "value").write_text(name)
    _cleanup_committed_pending(tmp_path, next_iteration=2)
    assert not (pending / "iteration-000001").exists()
    assert (pending / "iteration-000002").exists()
    assert (pending / "keep-me").exists()


def test_committed_iteration_is_read_from_atomic_checkpoint(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    path = tmp_path / "latest.pt"
    _save_checkpoint(
        path,
        _checkpoint_payload(
            actor,
            config=RunConfig(),
            root_seed=1,
            next_iteration=8,
            sl_checkpoint=sl,
            sl_sha256="hash",
            self_play=SelfPlayCurriculum(),
            opponent_pool=_initial_opponent_pool(actor),
            last_metrics={"iteration": 7, "policy_version_after": 7},
        ),
    )
    assert _committed_iteration(path, fallback=3) == 7
    path.write_text("broken")
    assert _committed_iteration(path, fallback=3) == 3


def test_opponent_pool_refreshes_periodically_rotates_and_excludes_current() -> None:
    reference = tiny_actor()
    pool = _initial_opponent_pool(reference)
    current = changed_actor(0.01)
    pool = _advance_opponent_pool(
        pool,
        current,
        completed_iteration=OPPONENT_REFRESH_INTERVAL,
        used_historical_opponent=False,
    )
    assert len(pool.snapshots) == 2
    assert pool.last_refresh_iteration == OPPONENT_REFRESH_INTERVAL
    selected = _select_opponent_snapshot(pool, current_digest=_model_digest(current))
    assert selected.digest != _model_digest(current)
    restored = _load_opponent_actor(selected, current.config, torch.device("cpu"))
    assert _model_digest(restored) == selected.digest

    for iteration in range(
        2 * OPPONENT_REFRESH_INTERVAL,
        (OPPONENT_POOL_CAPACITY + 3) * OPPONENT_REFRESH_INTERVAL,
        OPPONENT_REFRESH_INTERVAL,
    ):
        candidate = changed_actor(float(iteration) / 1000)
        pool = _advance_opponent_pool(
            pool,
            candidate,
            completed_iteration=iteration,
            used_historical_opponent=True,
        )
    assert len(pool.snapshots) == OPPONENT_POOL_CAPACITY
    assert pool.rotations == OPPONENT_POOL_CAPACITY + 1
    assert pool.last_refresh_iteration == (
        (OPPONENT_POOL_CAPACITY + 2) * OPPONENT_REFRESH_INTERVAL
    )


def test_rejected_generation_rotates_without_refreshing_opponent_snapshots() -> None:
    champion = tiny_actor()
    pool = _initial_opponent_pool(champion)

    rotated = _advance_opponent_pool(
        pool,
        changed_actor(0.01),
        completed_iteration=OPPONENT_REFRESH_INTERVAL,
        used_historical_opponent=True,
        promoted=False,
    )

    assert [snapshot.digest for snapshot in rotated.snapshots] == [
        snapshot.digest for snapshot in pool.snapshots
    ]
    assert rotated.last_refresh_iteration == pool.last_refresh_iteration
    assert rotated.rotations == pool.rotations + 1


def test_fork_run_restores_iteration_state_and_enables_fast_anchor(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    reference = tiny_actor()
    candidate = changed_actor(0.01)
    sl = source_dir / "sl.pt"
    torch.save(
        {"model_config": reference.config.__dict__, "model": reference.state_dict()},
        sl,
    )
    pool = _advance_opponent_pool(
        _initial_opponent_pool(reference),
        candidate,
        completed_iteration=OPPONENT_REFRESH_INTERVAL,
        used_historical_opponent=False,
    )
    source_checkpoint = source_dir / "latest.pt"
    source_config = RunConfig(direction_optimizer="nesterov")
    source_velocity = {
        name: torch.full_like(parameter, 1e-4).cpu()
        for name, parameter in candidate.named_parameters()
    }
    source_payload = _checkpoint_payload(
        candidate,
        config=source_config,
        root_seed=17,
        next_iteration=OPPONENT_REFRESH_INTERVAL + 1,
        sl_checkpoint=sl,
        sl_sha256="source-hash",
        self_play=SelfPlayCurriculum(
            fraction=0.5,
            last_fixed_first_rate=0.3,
            activation_iteration=2,
        ),
        opponent_pool=pool,
        last_metrics={
            "iteration": OPPONENT_REFRESH_INTERVAL,
            "policy_version_after": OPPONENT_REFRESH_INTERVAL,
        },
        direction_optimizer_state=source_velocity,
    )
    source_payload["policy_execution_version"] = POLICY_EXECUTION_VERSION - 1
    _save_checkpoint(source_checkpoint, source_payload)
    with pytest.raises(
        ValueError,
        match=(
            rf"policy execution version {POLICY_EXECUTION_VERSION - 1} "
            rf"does not match trainer version {POLICY_EXECUTION_VERSION}.*"
            r"training\.fork_policy_iteration"
        ),
    ):
        _load_checkpoint(source_checkpoint, torch.device("cpu"))

    output_dir = tmp_path / "fork"
    provenance = fork_run(
        source_checkpoint,
        output_dir,
        iteration=OPPONENT_REFRESH_INTERVAL,
        anchor_rule_fast=True,
        direction_optimizer="nesterov",
        direction_learning_rate=8e-4,
        direction_momentum=0.9,
        direction_gradient_clip_norm=0.0,
        queries_per_category=128,
        worlds=32,
        target_kl=1e-4,
        policy_objective="holdout_consensus_ce",
        world_sampling="information_set",
        kl_control="cap",
        split_consensus_margin=0.1875,
        envs=2048,
        evaluation_envs=2048,
        target_shard_size=128,
        target_query_batch_size=128,
    )
    fork_actor, config, payload = _load_checkpoint(
        output_dir / "latest.pt", torch.device("cpu")
    )
    assert provenance["source_iteration"] == OPPONENT_REFRESH_INTERVAL
    assert config.anchor_rule_fast is True
    assert config.direction_optimizer == "nesterov"
    assert config.direction_learning_rate == pytest.approx(8e-4)
    assert config.direction_momentum == pytest.approx(0.9)
    assert config.direction_gradient_clip_norm == 0.0
    assert config.queries_per_category == 128
    assert config.worlds == 32
    assert config.target_kl == pytest.approx(1e-4)
    assert config.policy_objective == "holdout_consensus_ce"
    assert config.world_sampling == "information_set"
    assert config.kl_control == "cap"
    assert config.split_consensus_margin == pytest.approx(0.1875)
    assert config.envs == 2048
    assert config.evaluation_envs == 2048
    assert config.target_shard_size == 128
    assert config.target_query_batch_size == 128
    assert payload["next_iteration"] == OPPONENT_REFRESH_INTERVAL + 1
    assert payload["direction_optimizer_state"] == {}
    assert payload["self_play"].fraction == pytest.approx(0.5)
    assert len(payload["opponent_pool"].snapshots) == 2
    assert _model_digest(fork_actor) == _model_digest(candidate)
    assert (output_dir / "reference-u008.pt").exists()
    assert (output_dir / "fork.json").exists()
    assert provenance["optimizer"]["state_reset"] is True

    preserved_dir = tmp_path / "preserved-fork"
    preserved = fork_run(
        source_checkpoint,
        preserved_dir,
        iteration=OPPONENT_REFRESH_INTERVAL,
        anchor_rule_fast=False,
        direction_optimizer="nesterov",
        direction_learning_rate=1e-4,
        direction_momentum=0.9,
        direction_gradient_clip_norm=0.0,
        preserve_direction_optimizer_state=True,
        policy_objective="holdout_consensus_ce",
        split_consensus_margin=0.1875,
    )
    _actor, preserved_config, preserved_payload = _load_checkpoint(
        preserved_dir / "latest.pt", torch.device("cpu")
    )
    assert preserved_config.direction_optimizer == "nesterov"
    assert preserved_config.direction_gradient_clip_norm == 0.0
    assert preserved["optimizer"]["state_reset"] is False
    assert preserved["optimizer"]["state_preserved"] is True
    assert set(preserved_payload["direction_optimizer_state"]) == set(
        source_velocity
    )
    for name, value in source_velocity.items():
        torch.testing.assert_close(
            preserved_payload["direction_optimizer_state"][name], value
        )


def test_batch_sweep_is_nested_and_reuses_a_maximum_corpus() -> None:
    config = SweepConfig()
    assert config.batch_queries_per_category == (64, 128, 256, 512)
    assert tuple(9 * qpc for qpc in config.batch_queries_per_category) == (
        576,
        1152,
        2304,
        4608,
    )
    assert config.calibration_queries_per_category == 128
    assert 9 * config.calibration_queries_per_category == 1152
    assert config.source_games >= 9 * max(config.batch_queries_per_category)
    with pytest.raises(ValueError, match="unique and increasing"):
        SweepConfig(batch_queries_per_category=(64, 32))
    with pytest.raises(ValueError, match="calibration games"):
        SweepConfig(
            batch_queries_per_category=(1, 2),
            source_games=18,
            calibration_source_games=8,
        )
    with pytest.raises(ValueError, match="heldout games"):
        SweepConfig(
            batch_queries_per_category=(1, 2),
            source_games=18,
            heldout_source_games=8,
        )
