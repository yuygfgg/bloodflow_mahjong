from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest
import torch

from training.iql import CriticConfig, IndependentCritics
from training.learner import (
    LearningConfig,
    actor_update,
    critic_update,
    make_optimizers,
    validate_critics,
)
from training.model import BloodFlowTransformer, TransformerConfig
from training.observation import unpack_action_masks
from training.oracle import OracleCritics
from training.pipeline import (
    CollectionConfig,
    ExecutablePolicyPool,
    FullTrajectoryCollector,
    better_on_both_panels,
    clone_policy,
    load_policy,
    save_policy,
    _bucket_inference_rows,
)
from training.policy_pool import (
    BehaviorSampler,
    OpponentMixtureConfig,
    PolicyPool,
    ReplaySource,
)
from training.replay import ReplayConfig, TrajectoryReplay
from training.train import (
    CHECKPOINT_VERSION,
    RunConfig,
    _load_checkpoint,
    _mc_validation_corpus_gate,
    _mc_validation_corpus_status,
    _should_evaluate,
    build_parser,
)
from training.trajectory import replay_trajectory


def tiny_actor() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
        )
    )


def tiny_critic_config() -> CriticConfig:
    return CriticConfig(
        d_model=32,
        num_heads=4,
        static_layers=1,
        history_layers=1,
        ffn_dim=64,
        head_dim=48,
        max_history=192,
    )


def tiny_critics() -> IndependentCritics:
    return IndependentCritics(tiny_critic_config())


def collector(seed: int = 1) -> FullTrajectoryCollector:
    device = torch.device("cpu")
    actor = tiny_actor().eval()
    reference = clone_policy(actor, device)
    pool = PolicyPool("in-memory-sl.pt", seed=seed)
    return FullTrajectoryCollector(
        CollectionConfig(envs=4, history=192),
        pool,
        ExecutablePolicyPool(actor, reference, device),
        BehaviorSampler(seed=seed + 1),
        device,
        seed=seed + 2,
    )


def test_inference_rows_use_stable_power_of_two_buckets() -> None:
    rows = np.asarray([2, 5, 9], dtype=np.int64)
    padded = _bucket_inference_rows(rows)
    assert len(padded) == 32
    np.testing.assert_array_equal(padded[:3], rows)
    np.testing.assert_array_equal(padded[3:], 9)
    assert len(_bucket_inference_rows(np.arange(33))) == 64


def test_executable_pool_releases_evicted_historical_model() -> None:
    device = torch.device("cpu")
    actor = tiny_actor().eval()
    reference = clone_policy(actor, device)
    pool = PolicyPool(
        "in-memory-sl.pt",
        seed=5,
        config=replace(
            OpponentMixtureConfig(),
            recent_history_count=2,
            max_history=2,
            snapshot_interval=1,
        ),
    )
    executables = ExecutablePolicyPool(actor, reference, device)

    evicted_id = ""
    evicted_model: weakref.ReferenceType[BloodFlowTransformer] | None = None
    for update in range(1, 4):
        descriptor = pool.add_snapshot(
            update=update,
            artifact=f"snapshot-{update}.pt",
        )
        executables.register_snapshot(descriptor, actor)
        if update == 1:
            evicted_id = descriptor.policy_id
            evicted_model = weakref.ref(executables.historical[evicted_id])
        executables.sync(pool)

    assert set(executables.historical) == {
        descriptor.policy_id for descriptor in pool.history
    }
    assert evicted_id not in executables.historical
    gc.collect()
    assert evicted_model is not None
    assert evicted_model() is None


def test_complete_collection_covers_four_seats_and_strict_replay() -> None:
    result = collector(7).collect(4, mode="rules", deterministic=True)
    assert len(result.trajectories) == 4
    assert result.environment_steps > 0
    assert result.policy_actions > 0
    assert result.source_counts[ReplaySource.RULE_FAST.label] > 0
    assert result.source_counts[ReplaySource.RULE_SAFE.label] > 0
    for trajectory in result.trajectories:
        assert set(trajectory.actors.tolist()) == {0, 1, 2, 3}
        assert np.all(trajectory.action_probabilities == 1.0)
        replayed = replay_trajectory(trajectory)
        np.testing.assert_array_equal(
            replayed.trajectory.terminal_scores, trajectory.terminal_scores
        )


def test_stochastic_collection_records_real_behavior_metadata() -> None:
    result = collector(11).collect(4, mode="mixed", deterministic=False)
    probabilities = np.concatenate(
        [trajectory.action_probabilities for trajectory in result.trajectories]
    )
    temperatures = np.concatenate(
        [trajectory.temperatures for trajectory in result.trajectories]
    )
    sources = np.concatenate(
        [trajectory.sources for trajectory in result.trajectories]
    )
    assert np.all((probabilities > 0) & (probabilities <= 1))
    assert np.all(temperatures > 0)
    assert int(ReplaySource.CURRENT) in sources
    assert int(ReplaySource.SL) in sources
    assert np.isin(
        sources,
        [
            int(ReplaySource.RULE_FAST),
            int(ReplaySource.RULE_SAFE),
            int(ReplaySource.SL),
            int(ReplaySource.CURRENT),
        ],
    ).all()


def test_deterministic_collection_rejects_nonfinite_policy_logits() -> None:
    device = torch.device("cpu")
    actor = tiny_actor().eval()
    reference = clone_policy(actor, device)
    with torch.no_grad():
        actor.actor[-1].weight.fill_(torch.nan)
        reference.actor[-1].weight.fill_(torch.nan)
    invalid = FullTrajectoryCollector(
        CollectionConfig(envs=1),
        PolicyPool("in-memory-sl.pt", seed=31),
        ExecutablePolicyPool(actor, reference, device),
        BehaviorSampler(seed=37),
        device,
        seed=41,
    )
    with pytest.raises(RuntimeError, match="non-finite logits"):
        invalid.collect(1, mode="sl", deterministic=True)


def test_batched_replay_materialization_matches_strict_actor_returns(tmp_path) -> None:
    result = collector(17).collect(3, mode="rules", deterministic=True)
    replay = TrajectoryReplay(
        tmp_path,
        seed=19,
        config=ReplayConfig(validation_fraction=0.34),
    )
    replay.add_trajectories(result.trajectories, anchor=True)
    entry = replay.entries[0]
    index = replay.index(entry.split, include_mc=False)
    selected = np.flatnonzero(index.trajectory_ids == entry.trajectory_id)
    chosen = selected[[0, len(selected) // 2, -1]]
    batch = replay.materialize(
        index,
        chosen,
        include_oracle=True,
        include_rule_actions=True,
    )
    strict = replay_trajectory(entry.trajectory)
    for output, row in enumerate(chosen):
        step = int(index.step_indices[row])
        actor = int(entry.trajectory.actors[step])
        np.testing.assert_array_equal(batch.tile_obs[output], strict.tile_obs[step])
        np.testing.assert_array_equal(batch.oracle_tiles[output], strict.oracle_tiles[step])
        assert batch.rule_actions is not None
        assert batch.legal[output, int(batch.rule_actions[output])]
        np.testing.assert_allclose(
            batch.returns[output], strict.returns_to_go[step, actor], atol=1e-6
        )


def test_collected_replay_drives_independent_critic_and_actor_updates(tmp_path) -> None:
    device = torch.device("cpu")
    actor = tiny_actor()
    reference = clone_policy(actor, device)
    pool = PolicyPool("in-memory-sl.pt", seed=23)
    result = FullTrajectoryCollector(
        CollectionConfig(envs=4),
        pool,
        ExecutablePolicyPool(actor, reference, device),
        BehaviorSampler(seed=29),
        device,
        seed=31,
    ).collect(6, mode="mixed")
    replay = TrajectoryReplay(
        tmp_path,
        seed=37,
        config=ReplayConfig(validation_fraction=0.34),
    )
    replay.add_trajectories(result.trajectories, anchor=True)
    train_index = replay.index("train", include_mc=False)
    rows = np.arange(min(16, len(train_index)))
    batch = replay.materialize(train_index, rows)
    critics = tiny_critics()
    config = replace(
        LearningConfig(),
        microbatch_size=4,
        critic_batch_size=len(batch),
        actor_batch_size=len(batch),
        minimum_critic_steps=1,
    )
    optimizers = make_optimizers(actor, critics, config)
    q_before = next(critics.q1.parameters()).detach().clone()
    critic_stats = critic_update(
        critics,
        optimizers,
        batch,
        config,
        device,
        cql_scale=0.1,
    )
    assert not torch.equal(q_before, next(critics.q1.parameters()).detach())
    assert np.isfinite(list(critic_stats.values())).all()

    actor_before = actor.actor[-1].weight.detach().clone()
    actor_stats = actor_update(
        actor,
        reference,
        critics,
        optimizers["actor"],
        batch,
        config,
        device,
    )
    assert not torch.equal(actor_before, actor.actor[-1].weight.detach())
    assert np.isfinite(list(actor_stats.values())).all()
    assert "actor_reference_kl" in actor_stats

    deferred_stats = actor_update(
        actor,
        reference,
        critics,
        optimizers["actor"],
        batch,
        config,
        device,
        measure_post_update_kl=False,
    )
    assert "actor_reference_kl" not in deferred_stats
    assert np.isfinite(list(deferred_stats.values())).all()


def test_sparse_materialize_matches_strict_replay_with_duplicate_rows(tmp_path) -> None:
    result = collector(35).collect(4, mode="mixed")
    replay = TrajectoryReplay(
        tmp_path,
        seed=37,
        config=ReplayConfig(validation_fraction=0.25),
    )
    replay.add_trajectories(result.trajectories, anchor=True)
    entry = replay.entries[0]
    index = replay.index(entry.split, include_mc=False)
    trajectory_rows = np.flatnonzero(index.trajectory_ids == entry.trajectory_id)
    selected_steps = np.asarray(
        [0, len(trajectory_rows) // 2, len(trajectory_rows) - 1, len(trajectory_rows) // 2]
    )
    selected_rows = trajectory_rows[selected_steps]
    batch = replay.materialize(index, selected_rows, include_oracle=True)
    strict = replay_trajectory(entry.trajectory)

    np.testing.assert_array_equal(batch.tile_obs, strict.tile_obs[selected_steps])
    np.testing.assert_array_equal(batch.melds, strict.melds[selected_steps])
    np.testing.assert_array_equal(batch.meta, strict.meta[selected_steps])
    np.testing.assert_array_equal(batch.events, strict.events[selected_steps])
    np.testing.assert_array_equal(
        batch.event_lengths, strict.event_lengths[selected_steps]
    )
    np.testing.assert_array_equal(
        batch.legal,
        unpack_action_masks(strict.legal_mask_words[selected_steps]),
    )
    np.testing.assert_allclose(
        batch.returns, strict.actor_returns[selected_steps], atol=1e-6
    )
    np.testing.assert_array_equal(batch.oracle_tiles, strict.oracle_tiles[selected_steps])


def test_validation_reports_progress_and_all_decision_categories(tmp_path) -> None:
    result = collector(41).collect(8, mode="rules", deterministic=True)
    replay = TrajectoryReplay(
        tmp_path,
        seed=43,
        config=ReplayConfig(validation_fraction=0.5),
    )
    replay.add_trajectories(result.trajectories, anchor=True)
    batch = replay.validation_batch(256, seed=47)
    metrics = validate_critics(
        tiny_critics(), batch, torch.device("cpu"), microbatch_size=32
    )
    assert set(metrics["progress"]) == {"early", "middle", "late"}
    assert len(metrics["categories"]) == 9
    assert set(metrics["categories"]) == {
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


def test_validation_reports_full_oracle_diagnostics(tmp_path) -> None:
    result = collector(49).collect(8, mode="rules", deterministic=True)
    replay = TrajectoryReplay(
        tmp_path,
        seed=51,
        config=ReplayConfig(validation_fraction=0.5),
    )
    replay.add_trajectories(result.trajectories, anchor=True)
    batch = replay.validation_batch(128, seed=53, include_oracle=True)
    metrics = validate_critics(
        tiny_critics(),
        batch,
        torch.device("cpu"),
        microbatch_size=32,
        oracle=OracleCritics(tiny_critic_config()),
    )
    oracle = metrics["oracle"]
    assert set(oracle["progress"]) == {"early", "middle", "late"}
    assert len(oracle["categories"]) == 9
    assert "q" in oracle and "v" in oracle
    assert "q_disagreement" in oracle
    assert "expectile_balance_error" in oracle
    assert "q_relative_mae_gain" in metrics["oracle_vs_partial"]


def test_actor_policy_checkpoint_is_exact_and_has_no_old_migration(tmp_path) -> None:
    model = tiny_actor()
    path = tmp_path / "actor.pt"
    save_policy(path, model)
    restored = load_policy(path, torch.device("cpu"))
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)

    torch.save({"model": model.state_dict(), "optimizer": {}}, tmp_path / "old.pt")
    try:
        load_policy(tmp_path / "old.pt", torch.device("cpu"))
    except ValueError as error:
        assert "Actor-only" in str(error)
    else:
        raise AssertionError("old checkpoint was silently migrated")


def test_pre_teacher_gate_checkpoint_is_rejected(tmp_path) -> None:
    assert CHECKPOINT_VERSION == 3
    path = tmp_path / "old-iql.pt"
    torch.save({"checkpoint_version": 1}, path)
    with pytest.raises(ValueError, match="pre-gate IQL/AWR"):
        _load_checkpoint(path, torch.device("cpu"))


def test_v2_checkpoint_is_explicitly_rejected(tmp_path) -> None:
    path = tmp_path / "v2-iql.pt"
    torch.save({"checkpoint_version": 2}, path)
    with pytest.raises(ValueError, match="v2 resume is unsupported"):
        _load_checkpoint(path, torch.device("cpu"))


def test_current_config_does_not_fill_missing_fields() -> None:
    state = RunConfig().state_dict()
    learning = state["learning"]
    assert isinstance(learning, dict)
    learning.pop("mc_critic_batch_size")
    with pytest.raises(ValueError, match="missing fields:.*mc_critic_batch_size"):
        RunConfig.from_state_dict(state)


@pytest.mark.parametrize(
    ("targets", "groups", "pairs", "frozen", "gate"),
    (
        (511, 128, 128, False, "mc_validation_targets"),
        (512, 127, 128, False, "mc_validation_reliable_groups"),
        (512, 128, 127, False, "mc_validation_reliable_pairs"),
        (512, 128, 128, True, "ready"),
    ),
)
def test_mc_validation_corpus_freezes_only_after_all_evidence_thresholds(
    targets: int, groups: int, pairs: int, frozen: bool, gate: str
) -> None:
    class MetadataReplay:
        def mc_target_count(self, split: str, *, anchor_only: bool = False) -> int:
            assert split == "validation"
            assert anchor_only
            return targets

        def reliable_mc_counts(
            self, split: str, *, anchor_only: bool = False
        ) -> dict[str, int]:
            assert split == "validation"
            assert anchor_only
            return {"targets": min(targets, pairs * 2), "groups": groups, "pairs": pairs}

    status = _mc_validation_corpus_status(
        MetadataReplay(), LearningConfig()  # type: ignore[arg-type]
    )
    assert status["validation_frozen"] is frozen
    assert status["validation_reliable_groups"] == groups
    assert status["validation_reliable_pairs"] == pairs
    assert _mc_validation_corpus_gate(status, LearningConfig()) == gate


def test_best_selection_requires_fixed_and_fresh_improvement() -> None:
    incumbent = {"mean_rank": 2.5, "mean_score_delta": 0.0}
    better = {"mean_rank": 2.4, "mean_score_delta": 10.0}
    worse = {"mean_rank": 2.6, "mean_score_delta": 100.0}
    assert better_on_both_panels(better, better, incumbent, incumbent)
    assert not better_on_both_panels(better, worse, incumbent, incumbent)


def test_experiment_argument_is_explicit_on_resume() -> None:
    parser = build_parser()
    assert parser.parse_args([]).experiment is None
    assert parser.parse_args([]).output_dir.name == "iql-awr-v3"
    assert parser.parse_args(["--experiment", "b"]).experiment == "b"
    assert RunConfig().mc_validation_every == 2


def test_evaluation_stays_on_schedule_after_actor_updates() -> None:
    assert not _should_evaluate(1, 5, smoke=False)
    assert _should_evaluate(5, 5, smoke=False)
    assert _should_evaluate(1, 5, smoke=True)


def test_old_counterfactual_and_ppo_symbols_are_removed() -> None:
    import training.learner as learner
    import training.pipeline as pipeline

    for name in (
        "CounterfactualCollector",
        "counterfactual_update",
        "compile_policy",
        "PPOConfig",
        "RolloutCollector",
        "advance_self_play_curriculum",
    ):
        assert not hasattr(pipeline, name)
    assert not hasattr(learner, "compile_critic_models")
