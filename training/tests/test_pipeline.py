from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import (
    EngineBuffers,
    HistoryCacheStore,
    OpponentPool,
    PPOConfig,
    RolloutCollector,
    TransitionStorage,
    cosine_learning_rate,
    infer_actions,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)


def tiny_model() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
            value_atoms=17,
        )
    )


def tiny_ppo(**changes: object) -> PPOConfig:
    config = PPOConfig(
        envs=4,
        rollout_transitions=16,
        ppo_epochs=1,
        minibatch=8,
        microbatch=4,
        opponent_refresh_updates=1,
        frozen_snapshot_limit=2,
    )
    return replace(config, **changes)


def test_four_opponents_are_assigned_per_seat_at_schedule_boundaries() -> None:
    config = tiny_ppo()
    pool = OpponentPool(config, seed=19)
    learner_seats = np.array([0, 1, 2, 3] * 64, dtype=np.uint8)

    assert pool.stage(0.0) == "bootstrap"
    assert pool.stage(config.rule_only_fraction) == "mixed"
    assert pool.stage(config.mixed_opponent_fraction) == "league"
    assert pool.stage(config.self_play_fraction) == "self_play"
    np.testing.assert_allclose(pool.probabilities(0.0), [0.30, 0.50, 0.20, 0.0])

    bootstrap = pool.assign_seats(learner_seats, 0.0)
    assert bootstrap.shape == (len(learner_seats), 4)
    assert np.all(bootstrap[np.arange(len(learner_seats)), learner_seats] == -1)
    assert set(np.unique(bootstrap[bootstrap >= 0])) == {
        OpponentPool.RANDOM_HU,
        OpponentPool.RULE_FAST,
        OpponentPool.RULE_SAFE,
    }

    pool.set_frozen(tiny_model())
    np.testing.assert_allclose(
        pool.probabilities(config.rule_only_fraction), [0.10, 0.35, 0.20, 0.35]
    )
    np.testing.assert_allclose(
        pool.probabilities(config.mixed_opponent_fraction), [0.05, 0.15, 0.15, 0.65]
    )
    np.testing.assert_allclose(
        pool.probabilities(config.self_play_fraction), [0.0, 0.0, 0.0, 1.0]
    )
    league = pool.assign_seats(learner_seats, 1.0)
    assert OpponentPool.FROZEN_TRANSFORMER in league
    assert np.all(league[league >= 0] == OpponentPool.FROZEN_TRANSFORMER)


def test_rollout_has_legal_finite_transitions_and_mixed_seat_counts() -> None:
    config = tiny_ppo()
    model = tiny_model()
    collector = RolloutCollector(config, torch.device("cpu"), seed=23)
    rollout = collector.collect(model, config.rollout_transitions, progress=0.0)
    storage = rollout.storage
    slots = rollout.indices

    assert len(rollout) == config.rollout_transitions
    assert rollout.opponent_stage == "bootstrap"
    assert rollout.opponent_counts.sum() >= config.envs * 3
    assert rollout.opponent_counts[OpponentPool.FROZEN_TRANSFORMER] == 0
    assert np.all(storage.finalized[slots])
    assert np.all(storage.legal[slots, storage.actions[slots]])
    assert np.isfinite(storage.reward[slots]).all()
    assert np.isfinite(rollout.advantages[slots]).all()
    assert np.isfinite(rollout.returns[slots]).all()


def test_gae_keeps_environment_reward_streams_separate() -> None:
    storage = TransitionStorage(4, history=1)
    storage.next_slot = 4
    storage.finalized[:] = True
    storage.env[:] = [0, 1, 0, 1]
    storage.episode[:] = 0
    storage.reward[:] = [1.0, 10.0, 2.0, 20.0]
    storage.value[:] = 0.0
    storage.next_value[:] = 0.0
    storage.done[:] = [False, False, True, True]

    advantages, returns = storage.compute_gae(
        np.arange(4, dtype=np.int64), gamma=1.0, gae_lambda=1.0
    )

    np.testing.assert_allclose(advantages, [3.0, 30.0, 2.0, 20.0])
    np.testing.assert_allclose(returns, advantages)


def test_rollout_cache_append_matches_full_history() -> None:
    model = tiny_model().eval()
    buffers = EngineBuffers.create(1, history=192)
    buffers.batch.reset_all(47)
    buffers.refresh()
    buffers.events[0, :3] = np.arange(24, dtype=np.int32).reshape(3, 8)
    buffers.event_lengths[0] = 3
    cache = HistoryCacheStore(max_history=192)

    cached_first = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
        history_cache=cache,
    )
    full_first = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
    )
    np.testing.assert_array_equal(cached_first[0], full_first[0])
    np.testing.assert_allclose(cached_first[1:], full_first[1:], rtol=1e-5, atol=1e-5)

    buffers.events[0, 3:5] = np.arange(16, dtype=np.int32).reshape(2, 8) + 100
    buffers.event_lengths[0] = 5
    cached_append = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
        history_cache=cache,
    )
    full_append = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
    )
    np.testing.assert_array_equal(cached_append[0], full_append[0])
    np.testing.assert_allclose(cached_append[1:], full_append[1:], rtol=1e-5, atol=1e-5)


def test_ppo_update_changes_parameters() -> None:
    torch.manual_seed(29)
    config = tiny_ppo()
    model = tiny_model()
    collector = RolloutCollector(config, torch.device("cpu"), seed=31)
    rollout = collector.collect(model, config.rollout_transitions, progress=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    before = model.actor[-1].weight.detach().clone()

    statistics = ppo_update(
        model, optimizer, rollout, config, torch.device("cpu"), progress=0.0
    )

    assert not torch.equal(before, model.actor[-1].weight.detach())
    assert all(np.isfinite(value) for value in statistics.values())


def test_checkpoint_restores_model_optimizer_and_opponent_snapshots(tmp_path) -> None:
    config = tiny_ppo()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    pool = OpponentPool(config, seed=37)
    pool.refresh_snapshot(model, torch.device("cpu"), progress=0.6)
    pool.refresh_snapshot(model, torch.device("cpu"), progress=0.7)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 7, 1234, config, pool)

    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=config.learning_rate
    )
    restored_pool = OpponentPool(config, seed=41)
    update, transitions = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
        restored_pool,
    )

    assert (update, transitions) == (7, 1234)
    assert len(restored_pool.snapshots) == 2
    assert restored_pool.frozen_ready
    assert all(
        not parameter.requires_grad
        for snapshot in restored_pool.snapshots
        for parameter in snapshot.parameters()
    )
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_cosine_learning_rate_hits_both_endpoints() -> None:
    config = tiny_ppo()
    assert cosine_learning_rate(config, 0.0) == config.learning_rate
    assert cosine_learning_rate(config, 1.0) == config.final_learning_rate
    assert (
        config.final_learning_rate
        < cosine_learning_rate(config, 0.5)
        < config.learning_rate
    )
