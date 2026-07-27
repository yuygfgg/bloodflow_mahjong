from __future__ import annotations

import numpy as np
import pytest
import torch

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import (
    CollectionConfig,
    EngineBuffers,
    TrajectoryCollector,
    _lineup_history_seat_masks,
    clone_policy,
    load_policy,
    save_policy,
)
from training.policy_pool import ReplaySource


def tiny_actor() -> BloodFlowTransformer:
    torch.manual_seed(7)
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=32,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=64,
            max_history=32,
        )
    ).eval()


@pytest.fixture(scope="module")
def collection():
    actor = tiny_actor()
    collector = TrajectoryCollector(
        CollectionConfig(envs=4, history=32), actor, torch.device("cpu"), seed=19
    )
    return actor, collector.collect(8)


def test_rule_collection_records_independent_complete_games(collection) -> None:
    _actor, result = collection
    assert len(result.trajectories) == 8
    assert result.focal_seats.shape == (8,)
    assert len({trajectory.seed for trajectory in result.trajectories}) == 8
    assert result.environment_steps > 0
    assert result.policy_actions > 0
    assert sum(result.opponent_seat_counts.values()) == 3 * len(result.trajectories)
    for trajectory, focal in zip(result.trajectories, result.focal_seats):
        assert len(trajectory) > 0
        assert np.all(trajectory.legal_counts >= 1)
        for seat in range(4):
            sources = np.unique(trajectory.sources[trajectory.actors == seat])
            assert len(sources) == 1
            if seat == focal:
                assert sources[0] == int(ReplaySource.CURRENT)
            else:
                assert sources[0] in (
                    int(ReplaySource.RULE_FAST),
                    int(ReplaySource.RULE_SAFE),
                )


def test_seeded_collection_is_exact_and_ordered() -> None:
    actor = tiny_actor()
    seeds = np.asarray([11, 23, 37, 41], dtype=np.uint64)
    left = TrajectoryCollector(
        CollectionConfig(envs=4, history=32), actor, torch.device("cpu"), seed=1
    ).collect_seeded(seeds, focal_offset=3)
    right = TrajectoryCollector(
        CollectionConfig(envs=4, history=32), actor, torch.device("cpu"), seed=999
    ).collect_seeded(seeds, focal_offset=3)
    assert [trajectory.seed for trajectory in left.trajectories] == seeds.tolist()
    np.testing.assert_array_equal(left.focal_seats, [3, 0, 1, 2])
    for first, second in zip(left.trajectories, right.trajectories):
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(first.terminal_scores, second.terminal_scores)


def test_collection_accepts_full_uint64_seed_domain() -> None:
    actor = tiny_actor()
    collector = TrajectoryCollector(
        CollectionConfig(envs=2, history=32),
        actor,
        torch.device("cpu"),
        seed=(1 << 63) + 17,
    )
    result = collector.collect(2)
    assert len(result.trajectories) == 2
    assert len({trajectory.seed for trajectory in result.trajectories}) == 2


def test_zero_self_play_fraction_has_only_rule_opponents() -> None:
    collector = TrajectoryCollector(
        CollectionConfig(envs=8, history=32),
        tiny_actor(),
        torch.device("cpu"),
        seed=17,
        self_play_fraction=0.0,
    )
    lineups = collector._lineups(8, focal_offset=1)
    rows = np.arange(len(lineups.sources))
    assert np.all(
        lineups.sources[rows, lineups.focal_seats] == int(ReplaySource.CURRENT)
    )
    opponents = lineups.sources[
        lineups.sources != int(ReplaySource.CURRENT)
    ].reshape(len(lineups.sources), 3)
    assert np.isin(
        opponents,
        (int(ReplaySource.RULE_FAST), int(ReplaySource.RULE_SAFE)),
    ).all()
    assert not np.any(opponents == int(ReplaySource.SELF_PLAY))


def test_two_thirds_self_play_is_exact_and_seed_deterministic() -> None:
    config = CollectionConfig(envs=32, history=32)
    left = TrajectoryCollector(
        config,
        tiny_actor(),
        torch.device("cpu"),
        seed=23,
        self_play_fraction=2.0 / 3.0,
    )._lineups(32, focal_offset=2)
    right = TrajectoryCollector(
        config,
        tiny_actor(),
        torch.device("cpu"),
        seed=23,
        self_play_fraction=2.0 / 3.0,
    )._lineups(32, focal_offset=2)
    np.testing.assert_array_equal(left.focal_seats, right.focal_seats)
    np.testing.assert_array_equal(left.sources, right.sources)
    assert np.all(
        np.count_nonzero(left.sources == int(ReplaySource.SELF_PLAY), axis=1) == 2
    )
    assert np.all(
        np.count_nonzero(left.sources == int(ReplaySource.CURRENT), axis=1) == 1
    )
    assert np.all(
        np.count_nonzero(
            np.isin(
                left.sources,
                (int(ReplaySource.RULE_FAST), int(ReplaySource.RULE_SAFE)),
            ),
            axis=1,
        )
        == 1
    )


def test_history_masks_only_include_model_controlled_seats() -> None:
    sources = np.asarray(
        [
            [
                ReplaySource.CURRENT,
                ReplaySource.RULE_FAST,
                ReplaySource.SELF_PLAY,
                ReplaySource.RULE_SAFE,
            ],
            [
                ReplaySource.RULE_SAFE,
                ReplaySource.SELF_PLAY,
                ReplaySource.CURRENT,
                ReplaySource.SELF_PLAY,
            ],
        ],
        dtype=np.uint8,
    )
    masks = _lineup_history_seat_masks(sources)
    np.testing.assert_array_equal(masks, np.asarray([0b0101, 0b1110], dtype=np.uint8))
    assert masks.flags.c_contiguous
    with pytest.raises(ValueError, match="shape"):
        _lineup_history_seat_masks(sources[:, :3])


def test_policy_checkpoint_is_actor_only_and_strict(tmp_path) -> None:
    actor = tiny_actor()
    path = tmp_path / "actor.pt"
    save_policy(path, actor)
    loaded = load_policy(path, torch.device("cpu"), frozen=False)
    for expected, actual in zip(actor.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual)
        assert actual.requires_grad
    frozen = clone_policy(loaded, torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in frozen.parameters())

    torch.save({"model": actor.state_dict()}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Actor-only"):
        load_policy(tmp_path / "bad.pt", torch.device("cpu"))


def test_engine_buffers_can_wrap_cloned_batches() -> None:
    import bloodflow_mahjong as bm

    batch = bm.Batch(3, seed=5)
    clone = batch.clone_indices(np.asarray([2, 0], dtype=np.uint32))
    buffers = EngineBuffers.for_batch(clone, history=16)
    buffers.observe()
    assert buffers.tile_obs.shape == (2, 10, 27)
    assert buffers.legal.shape == (2, 115)
    assert np.all(buffers.legal.sum(axis=1) >= 1)


def test_engine_buffers_observe_can_skip_unneeded_history() -> None:
    import bloodflow_mahjong as bm

    batch = bm.Batch(2, seed=13)
    buffers = EngineBuffers.for_batch(batch, history=16)
    batch.observe_into(buffers.tile_obs, buffers.melds, buffers.river, buffers.meta)
    actors = buffers.meta[:, 1].astype(np.uint8)
    masks = np.left_shift(np.uint8(1), actors)
    masks[1] = np.left_shift(np.uint8(1), (actors[1] + 1) % 4)

    buffers.observe(masks)

    assert buffers.event_lengths[0] > 0
    assert buffers.event_lengths[1] == 0
    with pytest.raises(ValueError, match="shape"):
        buffers.observe(masks[:1])
    with pytest.raises(TypeError, match="uint8"):
        buffers.observe(masks.astype(np.int8))


def test_engine_buffers_remove_rows_reuses_fresh_step_observation() -> None:
    import bloodflow_mahjong as bm

    batch = bm.Batch(4, seed=71)
    buffers = EngineBuffers.for_batch(batch, history=16)
    masks = np.full(4, 0b1111, dtype=np.uint8)
    buffers.batch.reset_and_observe_history_into(
        np.ones(4, dtype=np.uint8),
        np.asarray([71, 73, 79, 83], dtype=np.uint64),
        masks,
        buffers.masks,
        buffers.tile_obs,
        buffers.melds,
        buffers.river,
        buffers.meta,
        buffers.events,
        buffers.event_lengths,
    )
    buffers.refresh_legal()
    actions = np.asarray(
        [np.flatnonzero(row)[0] for row in buffers.legal], dtype=np.uint8
    )
    buffers.batch.step_and_observe_history_into(
        actions,
        masks,
        buffers.records,
        buffers.masks,
        buffers.tile_obs,
        buffers.melds,
        buffers.river,
        buffers.meta,
        buffers.events,
        buffers.event_lengths,
    )
    buffers.refresh_legal()
    reference_batch = batch.clone_indices(np.arange(4, dtype=np.uint32))
    selected, order = buffers.remove_rows(np.asarray([0, 2], dtype=np.int64))
    fresh = EngineBuffers.for_batch(
        reference_batch.clone_indices(order.astype(np.uint32)), history=16
    )
    fresh.observe(masks[order])
    for name in (
        "tile_obs",
        "melds",
        "river",
        "meta",
        "event_lengths",
        "masks",
        "legal",
    ):
        np.testing.assert_array_equal(getattr(selected, name), getattr(fresh, name))
    for row, length in enumerate(fresh.event_lengths.astype(np.int64)):
        np.testing.assert_array_equal(
            selected.events[row, :length], fresh.events[row, :length]
        )
