from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bloodflow_mahjong as bm

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import (
    CollectionConfig,
    ExecutablePolicyPool,
    FullTrajectoryCollector,
    clone_policy,
)
from training.policy_pool import BehaviorSampler, PolicyPool, ReplaySource, decision_categories
from training.trajectory import (
    CompactTrajectory,
    TrajectoryBuilder,
    TrajectoryFormatError,
    TrajectoryReplayError,
    decode_trajectory_shard,
    encode_trajectory_shard,
    replay_trajectory,
    split_trajectories,
)


def rule_trajectory(seed: int) -> CompactTrajectory:
    game = bm.Game(seed=seed)
    builder = TrajectoryBuilder(seed, game.exchange_direction)
    tile_obs = np.empty((bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
    melds = np.empty((4, bm.MELD_SLOTS, bm.MELD_FIELDS), dtype=np.uint8)
    river = np.empty((bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8)
    meta = np.empty(bm.META_OBSERVATION_WIDTH, dtype=np.int32)
    steps = 0
    while game.decision is not None:
        actor, phase = game.decision
        game.observe_into(actor, tile_obs, melds, river, meta)
        action = game.simple_rule_action()
        assert action is not None
        builder.append(
            action=action,
            actor=actor,
            phase=phase,
            category=int(decision_categories(meta[None, :])[0]),
            source=ReplaySource.RULE_FAST,
            policy_version=0,
            action_probability=1.0,
            temperature=0.0,
        )
        game.step_id(action)
        steps += 1
        assert steps < 1024
    assert game.termination_reason is not None
    return builder.finish(
        terminal_scores=game.scores(),
        ranking_order=game.rankings(),
        termination_reason=game.termination_reason,
    )


def test_compact_trajectory_round_trip_and_strict_replay() -> None:
    trajectory = rule_trajectory(47)
    encoded = trajectory.to_bytes()
    decoded = CompactTrajectory.from_bytes(encoded)
    assert decoded.seed == trajectory.seed
    assert decoded.nbytes == len(encoded)
    np.testing.assert_array_equal(decoded.actions, trajectory.actions)
    np.testing.assert_array_equal(decoded.actors, trajectory.actors)
    np.testing.assert_array_equal(decoded.sources, trajectory.sources)

    replayed = replay_trajectory(decoded, history_capacity=64)
    assert replayed.tile_obs.shape[0] == len(decoded)
    assert replayed.legal_mask_words.shape == (len(decoded), 2)
    assert replayed.events.shape == (len(decoded), 64, bm.EVENT_RECORD_WIDTH)
    assert replayed.oracle_tiles.shape == (len(decoded), 9, 27)
    assert np.all(replayed.oracle_tiles[:, 4:8] <= replayed.oracle_tiles[:, :4])
    assert set(decoded.actors.tolist()) == {0, 1, 2, 3}
    np.testing.assert_allclose(
        replayed.returns_to_go[0],
        (decoded.terminal_scores.astype(np.float32) - 10_000) / 10_000,
        atol=1e-6,
    )
    rows = np.arange(len(decoded))
    np.testing.assert_array_equal(
        replayed.actor_returns,
        replayed.returns_to_go[rows, decoded.actors],
    )
    for row, action in enumerate(decoded.actions):
        assert int(replayed.legal_mask_words[row, int(action) // 64]) & (
            1 << (int(action) % 64)
        )


def test_trajectory_corruption_and_replay_mismatch_fail_loudly() -> None:
    trajectory = rule_trajectory(59)
    corrupted = bytearray(trajectory.to_bytes())
    corrupted[-1] ^= 0x40
    with pytest.raises(TrajectoryFormatError, match="checksum"):
        CompactTrajectory.from_bytes(corrupted)

    actions = trajectory.actions.copy()
    actions[0] = bm.ACTION_PASS
    mismatched = replace(trajectory, actions=actions)
    with pytest.raises(TrajectoryReplayError, match="illegal"):
        replay_trajectory(mismatched)

    categories = trajectory.categories.copy()
    categories[0] = (int(categories[0]) + 1) % 9
    wrong_category = replace(trajectory, categories=categories)
    with pytest.raises(TrajectoryReplayError, match="category"):
        replay_trajectory(wrong_category)


def test_shards_and_splits_keep_complete_games_disjoint() -> None:
    trajectories = tuple(rule_trajectory(seed) for seed in range(4, 10))
    decoded = decode_trajectory_shard(encode_trajectory_shard(trajectories))
    assert [trajectory.seed for trajectory in decoded] == list(range(4, 10))

    train, validation = split_trajectories(
        decoded, validation_fraction=0.34, seed=7
    )
    train_seeds = {trajectory.seed for trajectory in train}
    validation_seeds = {trajectory.seed for trajectory in validation}
    assert train_seeds
    assert validation_seeds
    assert train_seeds.isdisjoint(validation_seeds)
    assert train_seeds | validation_seeds == set(range(4, 10))


def test_full_collector_outputs_strictly_replayable_four_seat_games() -> None:
    import torch

    device = torch.device("cpu")
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
        )
    ).eval()
    reference = clone_policy(actor, device)
    policy_pool = PolicyPool("unused-in-memory-reference.pt", seed=3)
    collector = FullTrajectoryCollector(
        CollectionConfig(envs=2, history=32),
        policy_pool,
        ExecutablePolicyPool(actor, reference, device),
        BehaviorSampler(seed=5),
        device,
        seed=7,
    )

    result = collector.collect(2, mode="rules", deterministic=True)

    assert len(result.trajectories) == 2
    for trajectory in result.trajectories:
        assert set(trajectory.actors.tolist()) == {0, 1, 2, 3}
        replay_trajectory(trajectory, history_capacity=32)
