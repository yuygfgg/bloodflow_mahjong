from __future__ import annotations

import numpy as np
import pytest
import torch

from training.iql import CriticConfig, IndependentCritics
from training.learner import (
    LearningConfig,
    make_optimizers,
    mc_critic_update_deferred,
)
from training.learner import LearningBatch
from training.mc_teacher import (
    MonteCarloConfig,
    _query_candidates,
    _reliable_action_graph,
    collect_mc_targets,
)
from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import (
    CollectionConfig,
    ExecutablePolicyPool,
    FullTrajectoryCollector,
    clone_policy,
)
from training.policy_pool import BehaviorSampler, PolicyPool, ReplaySource
from training.replay import ReplayConfig, TrajectoryReplay


def test_mc_budget_defaults_match_selective_teacher_scale() -> None:
    config = MonteCarloConfig()
    rollouts = (
        config.candidate_actions
        * config.hidden_worlds
        * config.continuations_per_world
    )
    assert 24 <= rollouts <= 96
    assert config.hidden_worlds == 32
    assert config.maximum_confidence_half_width == 0.25
    assert config.minimum_reliable_action_gap == 0.02
    assert config.queries_per_iteration < config.candidate_pool_states


def test_mc_budget_rejects_broad_or_invalid_candidate_enumeration() -> None:
    with pytest.raises(ValueError, match="four"):
        MonteCarloConfig(candidate_actions=5)
    with pytest.raises(ValueError, match="two"):
        MonteCarloConfig(candidate_actions=1)
    with pytest.raises(ValueError, match="minimum_reliable_action_gap"):
        MonteCarloConfig(minimum_reliable_action_gap=-0.01)


def test_paired_common_noise_produces_a_reliable_action_relation() -> None:
    config = MonteCarloConfig()
    shared_noise = np.linspace(-8.0, 8.0, config.hidden_worlds, dtype=np.float32)
    outcomes = np.stack((shared_noise, shared_noise + 0.10))
    marginal_half_width = config.confidence_z * np.sqrt(
        outcomes[0].var(ddof=1) / config.hidden_worlds
    )
    assert marginal_half_width > config.maximum_confidence_half_width

    reliable = _reliable_action_graph([3, 7], outcomes, config)

    assert reliable == ((7,), (3,))


def test_paired_zero_mean_difference_is_not_reliable() -> None:
    config = MonteCarloConfig()
    shared_noise = np.linspace(-8.0, 8.0, config.hidden_worlds, dtype=np.float32)
    alternating = np.tile(np.asarray([-0.5, 0.5], dtype=np.float32), 16)
    outcomes = np.stack((shared_noise, shared_noise + alternating))

    reliable = _reliable_action_graph([3, 7], outcomes, config)

    assert reliable == ((), ())


def test_mc_candidate_selection_skips_forced_action_states() -> None:
    device = torch.device("cpu")
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
        )
    ).eval()
    reference = clone_policy(actor, device)
    critics = IndependentCritics(
        CriticConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
            head_dim=48,
        )
    ).eval()
    size = 3
    legal = np.zeros((size, 115), dtype=np.bool_)
    legal[0, 0] = True
    legal[1, :2] = True
    legal[2, 2] = True
    batch = LearningBatch(
        tile_obs=np.zeros((size, 10, 27), dtype=np.uint8),
        melds=np.full((size, 4, 4, 3), 255, dtype=np.uint8),
        meta=np.zeros((size, 34), dtype=np.int32),
        events=np.zeros((size, 8, 8), dtype=np.int32),
        event_lengths=np.zeros(size, dtype=np.uint16),
        legal=legal,
        actions=np.asarray([0, 0, 2], dtype=np.uint8),
        returns=np.zeros(size, dtype=np.float32),
        categories=np.zeros(size, dtype=np.uint8),
        sources=np.zeros(size, dtype=np.uint8),
        policy_versions=np.zeros(size, dtype=np.uint32),
        behavior_probabilities=np.ones(size, dtype=np.float32),
        temperatures=np.ones(size, dtype=np.float32),
        trajectory_ids=np.arange(size, dtype=np.uint64),
        step_indices=np.zeros(size, dtype=np.uint16),
        rule_actions=np.asarray([0, 0, 2], dtype=np.uint8),
    )

    selected, candidates = _query_candidates(
        batch,
        actor,
        reference,
        critics,
        device,
        count=3,
        candidate_count=3,
    )

    assert selected.tolist() == [1]
    assert candidates.shape == (1, 3)
    assert set(candidates[0][candidates[0] >= 0]) == {0, 1}


def test_information_set_mc_round_trip_uses_rule_candidate(
    tmp_path, monkeypatch
) -> None:
    device = torch.device("cpu")
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
        )
    ).eval()
    reference = clone_policy(actor, device)
    pool = PolicyPool("in-memory-sl.pt", seed=101)
    collector = FullTrajectoryCollector(
        CollectionConfig(envs=4),
        pool,
        ExecutablePolicyPool(actor, reference, device),
        BehaviorSampler(seed=103),
        device,
        seed=107,
    )
    replay = TrajectoryReplay(
        tmp_path,
        seed=109,
        config=ReplayConfig(validation_fraction=0.5),
    )
    replay.add_trajectories(
        collector.collect(8, mode="mixed").trajectories,
        anchor=True,
    )
    critics = IndependentCritics(
        CriticConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
            head_dim=48,
        )
    ).eval()

    def decisive_rollouts(
        _batch, _seat, candidates, _model, _device, config, *, seed
    ):
        del seed
        worlds = config.hidden_worlds * config.continuations_per_world
        shared = np.linspace(-2.0, 2.0, worlds, dtype=np.float32)
        outcomes = np.stack(
            [shared + 0.20 * index for index in range(len(candidates))]
        )
        return outcomes, int(outcomes.size)

    monkeypatch.setattr(
        "training.mc_teacher._rollout_candidates", decisive_rollouts
    )
    targets, statistics = collect_mc_targets(
        replay,
        actor,
        reference,
        critics,
        device,
        MonteCarloConfig(
            queries_per_iteration=1,
            candidate_actions=3,
            hidden_worlds=2,
            continuations_per_world=1,
            candidate_pool_states=16,
            maximum_confidence_half_width=1_000_000.0,
        ),
        split="validation",
        seed=113,
        anchor_only=True,
        exclude_existing_states=True,
    )
    assert statistics.queries == 1
    assert statistics.accepted_queries == 1
    assert statistics.terminal_rollouts == len(targets) * 2
    assert targets
    assert len({target.query_id for target in targets}) == 1
    assert {target.candidate_count for target in targets} == {len(targets)}
    by_action = {target.action: target for target in targets}
    assert all(target.reliable_actions for target in targets)
    assert all(
        target.action in by_action[counterpart].reliable_actions
        for target in targets
        for counterpart in target.reliable_actions
    )
    entries = {entry.trajectory_id: entry for entry in replay.entries}
    assert all(entries[target.trajectory_id].anchor for target in targets)

    index = replay.index("validation", include_mc=False)
    first = targets[0]
    row = np.flatnonzero(
        (index.trajectory_ids == first.trajectory_id)
        & (index.step_indices == first.step_index)
    )
    state = replay.materialize(index, row[:1], include_rule_actions=True)
    assert state.rule_actions is not None
    assert int(state.rule_actions[0]) in {target.action for target in targets}

    replay.add_mc_targets(targets)
    mc_batch = replay.mc_validation_batch(16, seed=127)
    assert mc_batch is not None
    assert np.all(mc_batch.sources == int(ReplaySource.MC_TEACHER))
    assert len(mc_batch) == len(targets)

    train_targets, train_statistics = collect_mc_targets(
        replay,
        actor,
        reference,
        critics,
        device,
        MonteCarloConfig(
            queries_per_iteration=2,
            candidate_actions=3,
            hidden_worlds=2,
            continuations_per_world=1,
            candidate_pool_states=16,
            maximum_confidence_half_width=1_000_000.0,
        ),
        split="train",
        seed=131,
    )
    assert train_statistics.accepted_queries > 0
    replay.add_mc_targets(train_targets)
    train_batch = replay.mc_training_batch(16, seed=137)
    assert train_batch is not None
    assert train_batch.mc_query_ids is not None
    assert train_batch.mc_candidate_counts is not None
    assert train_batch.mc_reliable_actions is not None
    for query_id in np.unique(train_batch.mc_query_ids):
        rows = train_batch.mc_query_ids == query_id
        expected = np.unique(train_batch.mc_candidate_counts[rows])
        assert len(expected) == 1
        assert int(rows.sum()) == int(expected[0])
        assert len(np.unique(train_batch.actions[rows])) == int(expected[0])

    learning = LearningConfig(microbatch_size=16)
    optimizers = make_optimizers(actor, critics, learning)
    q_before = next(critics.q1.parameters()).detach().clone()
    value_before = [parameter.detach().clone() for parameter in critics.v.parameters()]
    summary = mc_critic_update_deferred(
        critics,
        optimizers["q"],
        train_batch,
        learning,
        device,
    )
    assert torch.isfinite(summary).all()
    assert not torch.equal(q_before, next(critics.q1.parameters()).detach())
    assert all(
        torch.equal(before, after)
        for before, after in zip(value_before, critics.v.parameters())
    )


def test_mc_rejects_the_entire_query_when_one_candidate_is_uncertain(
    tmp_path, monkeypatch
) -> None:
    device = torch.device("cpu")
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
        )
    ).eval()
    reference = clone_policy(actor, device)
    pool = PolicyPool("in-memory-sl.pt", seed=131)
    collector = FullTrajectoryCollector(
        CollectionConfig(envs=4),
        pool,
        ExecutablePolicyPool(actor, reference, device),
        BehaviorSampler(seed=137),
        device,
        seed=139,
    )
    replay = TrajectoryReplay(
        tmp_path,
        seed=149,
        config=ReplayConfig(validation_fraction=0.5),
    )
    replay.add_trajectories(
        collector.collect(8, mode="mixed").trajectories,
        anchor=True,
    )
    critics = IndependentCritics(
        CriticConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
            head_dim=48,
        )
    ).eval()

    def uncertain_rollouts(
        _batch, _seat, candidates, _model, _device, config, *, seed
    ):
        del seed
        worlds = config.hidden_worlds * config.continuations_per_world
        outcomes = np.zeros((len(candidates), worlds), dtype=np.float32)
        outcomes[0, -1] = 100.0
        return outcomes, int(outcomes.size)

    monkeypatch.setattr(
        "training.mc_teacher._rollout_candidates", uncertain_rollouts
    )
    targets, statistics = collect_mc_targets(
        replay,
        actor,
        reference,
        critics,
        device,
        MonteCarloConfig(
            queries_per_iteration=1,
            candidate_actions=3,
            hidden_worlds=2,
            candidate_pool_states=16,
            maximum_confidence_half_width=1.0,
        ),
        split="validation",
        seed=151,
    )
    assert targets == []
    assert statistics.accepted_queries == 0
    assert statistics.accepted_targets == 0
