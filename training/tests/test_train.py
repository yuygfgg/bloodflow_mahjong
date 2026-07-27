from __future__ import annotations

import torch
import pytest

from training.batch_sweep import SweepConfig
from training.model import BloodFlowTransformer, TransformerConfig
from training.train import (
    CHECKPOINT_VERSION,
    LEGACY_CHECKPOINT_VERSION,
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
    assert config.target_shard_size == 64
    assert config.target_query_batch_size == 64
    assert config.rollout_inference_batch_size == 128
    assert config.target_kl == pytest.approx(0.001)
    assert config.evaluation_games == 16_384
    assert config.self_play_start_first_rate == pytest.approx(0.55)
    assert config.self_play_increment == pytest.approx(0.10)
    assert config.maximum_self_play_fraction == pytest.approx(2 / 3)


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


def test_checkpoint_roundtrip_has_no_old_optimizer_or_migration_state(tmp_path) -> None:
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
        last_metrics={"iteration": 2},
    )
    assert set(payload) == {
        "version",
        "engine_rules_version",
        "policy_execution_version",
        "config",
        "root_seed",
        "next_iteration",
        "sl_checkpoint",
        "sl_sha256",
        "self_play",
        "opponent_pool",
        "model_config",
        "actor",
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
        last_metrics={"iteration": 2},
    )
    legacy = dict(current)
    legacy["version"] = LEGACY_CHECKPOINT_VERSION
    legacy.pop("opponent_pool")
    path = tmp_path / "latest.pt"
    torch.save(legacy, path)

    _loaded_actor, _loaded_config, loaded = _load_checkpoint(
        path, torch.device("cpu")
    )
    assert loaded["opponent_pool"] is None
    assert loaded["legacy_self_play_opponent"] is True


def test_resume_parser_does_not_invent_training_overrides() -> None:
    args = build_parser().parse_args(["--resume", "runs/example/latest.pt"])
    assert args.resume.name == "latest.pt"
    assert args.seed is None
    assert args.queries_per_category is None


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
        last_metrics={"iteration": 1},
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
            last_metrics={"iteration": 7},
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
