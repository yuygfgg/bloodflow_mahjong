from __future__ import annotations

import numpy as np
import pytest
import bloodflow_mahjong as bm

from training.policy_pool import ReplaySource
from training.observation import unpack_action_masks
from training.replay import (
    REPLAY_FORMAT_VERSION,
    MonteCarloTarget,
    ReplayConfig,
    TrajectoryReplay,
)
from training.trajectory import CompactTrajectory


def _trajectory(seed: int, source: ReplaySource, length: int = 2) -> CompactTrajectory:
    # Serialization/index tests intentionally do not replay these synthetic steps.
    return CompactTrajectory(
        seed=seed,
        exchange_direction=1,
        actions=np.zeros(length, dtype=np.uint8),
        actors=np.zeros(length, dtype=np.uint8),
        phases=np.zeros(length, dtype=np.uint8),
        categories=np.arange(length, dtype=np.uint8),
        sources=np.full(length, int(source), dtype=np.uint8),
        policy_versions=np.zeros(length, dtype=np.uint32),
        action_probabilities=np.ones(length, dtype=np.float32),
        temperatures=np.ones(length, dtype=np.float16),
        terminal_scores=np.full(4, 10_000, dtype=np.int32),
        terminal_ranks=np.arange(1, 5, dtype=np.uint8),
        termination_reason=0,
    )


def _legal_trajectory(seed: int, source: ReplaySource) -> CompactTrajectory:
    batch = bm.Batch(1, seed=seed)
    words = np.empty((1, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    batch.legal_action_masks_into(words)
    action = int(np.flatnonzero(unpack_action_masks(words)[0])[0])
    return CompactTrajectory(
        seed=seed,
        exchange_direction=1,
        actions=np.asarray([action], dtype=np.uint8),
        actors=np.zeros(1, dtype=np.uint8),
        phases=np.zeros(1, dtype=np.uint8),
        categories=np.zeros(1, dtype=np.uint8),
        sources=np.full(1, int(source), dtype=np.uint8),
        policy_versions=np.zeros(1, dtype=np.uint32),
        action_probabilities=np.ones(1, dtype=np.float32),
        temperatures=np.ones(1, dtype=np.float16),
        terminal_scores=np.full(4, 10_000, dtype=np.int32),
        terminal_ranks=np.arange(1, 5, dtype=np.uint8),
        termination_reason=0,
    )


def test_replay_persists_whole_trajectory_splits_and_anchor_window(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=17,
        config=ReplayConfig(
            validation_fraction=0.25,
            maximum_online_transitions=4,
            maximum_mc_targets=10,
            strict_validation=False,
        ),
    )
    anchors = [_trajectory(seed, ReplaySource.SL) for seed in range(20, 24)]
    replay.add_trajectories(anchors, anchor=True)
    replay.add_trajectories(
        [_trajectory(seed, ReplaySource.CURRENT) for seed in range(30, 35)],
        anchor=False,
    )
    assert sum(entry.anchor for entry in replay.entries) == len(anchors)
    assert sum(not entry.anchor for entry in replay.entries) <= 2
    for entry in replay.entries:
        index = replay.index(entry.split, include_mc=False)
        rows = index.trajectory_ids == entry.trajectory_id
        assert int(rows.sum()) == len(entry.trajectory)

    loaded = TrajectoryReplay.load(tmp_path)
    assert loaded.state_dict()["next_trajectory_id"] == replay.next_trajectory_id
    assert len(loaded.entries) == len(replay.entries)
    assert loaded.composition("train")["states"] == replay.composition("train")["states"]


def test_replay_index_keeps_source_category_and_policy_version(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path, seed=1, config=ReplayConfig(strict_validation=False)
    )
    trajectory = _trajectory(99, ReplaySource.RULE_SAFE, length=2)
    replay.add_trajectories([trajectory], anchor=True)
    entry = replay.entries[0]
    index = replay.index(entry.split, include_mc=False)
    np.testing.assert_array_equal(index.sources, int(ReplaySource.RULE_SAFE))
    np.testing.assert_array_equal(index.categories, [0, 1])
    np.testing.assert_array_equal(index.policy_versions, 0)


def test_replay_index_is_cached_and_invalidated_on_append(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path, seed=3, config=ReplayConfig(strict_validation=False)
    )
    replay.add_trajectories([_trajectory(101, ReplaySource.SL)], anchor=True)
    first = replay.index("train", include_mc=False)
    assert replay.index("train", include_mc=False) is first
    assert not first.actions.flags.writeable

    replay.add_trajectories([_trajectory(102, ReplaySource.CURRENT)], anchor=True)
    second = replay.index("train", include_mc=False)
    assert second is not first


def test_mc_queries_are_admitted_and_evicted_atomically(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=5,
        config=ReplayConfig(
            maximum_mc_targets=3,
            strict_validation=False,
        ),
    )
    replay.add_trajectories([_trajectory(103, ReplaySource.SL)], anchor=True)
    entry = replay.entries[0]

    def target(action: int, mean: float) -> MonteCarloTarget:
        return MonteCarloTarget(
            target_id=0,
            query_id=0,
            candidate_count=2,
            trajectory_id=entry.trajectory_id,
            step_index=0,
            action=action,
            mean_return=mean,
            variance=0.01,
            samples=8,
            confidence_low=mean - 0.1,
            confidence_high=mean + 0.1,
            split=entry.split,
            reliable_actions=(1 - action,),
        )

    with pytest.raises(ValueError, match="incomplete"):
        replay.add_mc_targets([target(0, 0.1)])
    assert replay.mc_targets == []
    assert replay.next_query_id == 0

    replay.add_mc_targets([target(0, 0.1), target(1, 0.2)])
    assert {value.query_id for value in replay.mc_targets} == {0}
    replay.add_mc_targets([target(0, 0.3), target(1, 0.4)])
    assert len(replay.mc_targets) == 2
    assert {value.query_id for value in replay.mc_targets} == {1}
    assert {value.candidate_count for value in replay.mc_targets} == {2}


def _mc_group(entry, *, query_id: int = 0) -> list[MonteCarloTarget]:
    return [
        MonteCarloTarget(
            target_id=0,
            query_id=query_id,
            candidate_count=2,
            trajectory_id=entry.trajectory_id,
            step_index=0,
            action=action,
            mean_return=0.1 * action,
            variance=0.01,
            samples=8,
            confidence_low=0.1 * action - 0.1,
            confidence_high=0.1 * action + 0.1,
            split=entry.split,
            reliable_actions=(1 - action,),
        )
        for action in (0, 1)
    ]


def _legal_actions_for_entry(replay: TrajectoryReplay, entry) -> np.ndarray:
    """Return four legal actions from one real replay state."""
    index = replay.index(entry.split, include_mc=False)
    rows = np.flatnonzero(index.trajectory_ids == entry.trajectory_id)
    state = replay.materialize(index, rows[:1])
    return np.flatnonzero(state.legal[0])[:4].astype(np.uint8)


def _variable_mc_group(
    entry,
    actions: np.ndarray,
    *,
    query_id: int,
    step_index: int = 0,
) -> list[MonteCarloTarget]:
    return [
        MonteCarloTarget(
            target_id=0,
            query_id=query_id,
            candidate_count=len(actions),
            trajectory_id=entry.trajectory_id,
            step_index=step_index,
            action=int(action),
            mean_return=0.1 * index,
            variance=0.01,
            samples=8,
            confidence_low=0.1 * index - 0.1,
            confidence_high=0.1 * index + 0.1,
            split=entry.split,
            reliable_actions=tuple(
                int(other) for other in actions if int(other) != int(action)
            ),
        )
        for index, action in enumerate(actions)
    ]


def test_mc_training_batch_keeps_mixed_candidate_groups_atomic(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=37,
        config=ReplayConfig(maximum_mc_targets=32, strict_validation=False),
    )
    replay.add_trajectories([_legal_trajectory(601, ReplaySource.SL)], anchor=True)
    entry = replay.entries[0]
    legal_actions = _legal_actions_for_entry(replay, entry)
    assert len(legal_actions) >= 4
    replay.add_mc_targets(
        _variable_mc_group(entry, legal_actions[:2], query_id=11)
        + _variable_mc_group(entry, legal_actions[:3], query_id=12)
        + _variable_mc_group(entry, legal_actions[:4], query_id=13)
    )

    batch = replay.mc_training_batch(9, seed=41)
    assert batch is not None
    assert batch.mc_query_ids is not None
    assert batch.mc_candidate_counts is not None
    assert batch.mc_reliable_actions is not None
    assert len(batch) == 9
    assert sorted(
        int(batch.mc_candidate_counts[batch.mc_query_ids == query_id][0])
        for query_id in np.unique(batch.mc_query_ids)
    ) == [2, 3, 4]
    for query_id in np.unique(batch.mc_query_ids):
        rows = batch.mc_query_ids == query_id
        expected = np.unique(batch.mc_candidate_counts[rows])
        assert len(expected) == 1
        assert int(rows.sum()) == int(expected[0])
        assert len(np.unique(batch.actions[rows])) == int(expected[0])
        action_rows = batch.actions[rows].astype(np.int64, copy=False)
        reliable = batch.mc_reliable_actions[rows][:, action_rows]
        assert np.array_equal(reliable, ~np.eye(len(action_rows), dtype=np.bool_))


def test_mc_training_batch_keeps_same_state_queries_separate(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=43,
        config=ReplayConfig(maximum_mc_targets=32, strict_validation=False),
    )
    replay.add_trajectories([_legal_trajectory(701, ReplaySource.SL)], anchor=True)
    entry = replay.entries[0]
    legal_actions = _legal_actions_for_entry(replay, entry)
    assert len(legal_actions) >= 3
    replay.add_mc_targets(
        _variable_mc_group(entry, legal_actions[:3], query_id=21)
        + _variable_mc_group(entry, legal_actions[:3], query_id=22)
    )

    batch = replay.mc_training_batch(6, seed=47)
    assert batch is not None
    assert batch.mc_query_ids is not None
    assert batch.mc_reliable_actions is not None
    assert len(np.unique(batch.mc_query_ids)) == 2
    assert len(np.unique(batch.trajectory_ids)) == 1
    assert len(np.unique(batch.step_indices)) == 1
    for query_id in np.unique(batch.mc_query_ids):
        assert int(np.sum(batch.mc_query_ids == query_id)) == 3


def test_anchor_validation_mc_is_stable_when_online_replay_trims(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=23,
        config=ReplayConfig(
            validation_fraction=0.5,
            maximum_online_transitions=2,
            maximum_mc_targets=20,
            strict_validation=False,
        ),
    )
    replay.add_trajectories(
        [_trajectory(seed, ReplaySource.SL) for seed in range(200, 220)],
        anchor=True,
    )
    anchor = next(
        entry for entry in replay.entries if entry.anchor and entry.split == "validation"
    )
    replay.add_mc_targets(_mc_group(anchor))

    online = None
    for seed in range(300, 340):
        replay.add_trajectories(
            [_trajectory(seed, ReplaySource.CURRENT)], anchor=False
        )
        candidate = replay.entries[-1]
        if candidate.split == "validation":
            online = candidate
            break
    assert online is not None
    replay.add_mc_targets(_mc_group(online))
    assert replay.mc_target_count("validation") == 4
    assert replay.mc_target_count("validation", anchor_only=True) == 2

    anchor_index = replay.index(
        "validation", include_mc=True, anchor_only=True
    )
    anchor_mc = anchor_index.mc_target_ids >= 0
    assert int(anchor_mc.sum()) == 2
    assert set(anchor_index.trajectory_ids[anchor_mc]) == {anchor.trajectory_id}

    replay.add_trajectories(
        [_trajectory(400, ReplaySource.CURRENT)], anchor=False
    )
    assert online.trajectory_id not in {
        entry.trajectory_id for entry in replay.entries
    }
    assert replay.mc_target_count("validation") == 2
    assert replay.mc_target_count("validation", anchor_only=True) == 2


def test_mc_reliable_relations_round_trip_and_materialize(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=53,
        config=ReplayConfig(maximum_mc_targets=16, strict_validation=False),
    )
    replay.add_trajectories([_legal_trajectory(809, ReplaySource.SL)], anchor=True)
    entry = replay.entries[0]
    actions = _legal_actions_for_entry(replay, entry)[:3]
    assert len(actions) == 3
    replay.add_mc_targets(_variable_mc_group(entry, actions, query_id=31))

    counts = replay.reliable_mc_counts(entry.split, anchor_only=True)
    assert counts == {"targets": 3, "groups": 1, "pairs": 3}
    loaded = TrajectoryReplay.load(tmp_path)
    assert loaded.reliable_mc_counts(entry.split, anchor_only=True) == counts
    assert all(
        isinstance(target.reliable_actions, tuple) for target in loaded.mc_targets
    )

    batch = loaded._mc_group_batch(
        entry.split,
        3,
        seed=59,
        anchor_only=True,
    )
    assert batch is not None
    assert batch.mc_reliable_actions is not None
    for row, action in enumerate(batch.actions.astype(np.int64, copy=False)):
        expected = set(batch.actions.astype(np.int64, copy=False)) - {int(action)}
        assert set(np.flatnonzero(batch.mc_reliable_actions[row])) == expected


def test_mc_reliable_relations_are_strictly_validated(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=61,
        config=ReplayConfig(maximum_mc_targets=16, strict_validation=False),
    )
    replay.add_trajectories([_trajectory(907, ReplaySource.SL)], anchor=True)
    entry = replay.entries[0]

    with pytest.raises(ValueError, match="unique"):
        MonteCarloTarget(
            target_id=0,
            query_id=0,
            candidate_count=3,
            trajectory_id=entry.trajectory_id,
            step_index=0,
            action=0,
            mean_return=0.0,
            variance=0.0,
            samples=8,
            confidence_low=0.0,
            confidence_high=0.0,
            split=entry.split,
            reliable_actions=(1, 1),
        )

    outside = _mc_group(entry)
    outside = [
        MonteCarloTarget(
            **{
                **target.__dict__,
                "reliable_actions": (2,),
            }
        )
        for target in outside
    ]
    with pytest.raises(ValueError, match="outside its query group"):
        replay.add_mc_targets(outside)

    three_actions = [0, 1, 2]
    asymmetric = _variable_mc_group(
        entry,
        np.asarray(three_actions, dtype=np.uint8),
        query_id=1,
    )
    asymmetric[1] = MonteCarloTarget(
        **{
            **asymmetric[1].__dict__,
            "reliable_actions": (2,),
        }
    )
    with pytest.raises(ValueError, match="asymmetric"):
        replay.add_mc_targets(asymmetric)


def test_replay_v3_rejects_v2_manifest(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=67,
        config=ReplayConfig(strict_validation=False),
    )
    replay.add_trajectories([_trajectory(1009, ReplaySource.SL)], anchor=True)
    manifest = replay.manifest_path.read_text(encoding="utf-8")
    replay.manifest_path.write_text(
        manifest.replace(
            f'"format_version": {REPLAY_FORMAT_VERSION}',
            '"format_version": 2',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported replay manifest version"):
        TrajectoryReplay.load(tmp_path)


def test_mc_capacity_protects_anchor_validation_queries(tmp_path) -> None:
    replay = TrajectoryReplay(
        tmp_path,
        seed=29,
        config=ReplayConfig(
            validation_fraction=0.5,
            maximum_mc_targets=2,
            strict_validation=False,
        ),
    )
    replay.add_trajectories(
        [_trajectory(seed, ReplaySource.SL) for seed in range(500, 540)],
        anchor=True,
    )
    validation = next(
        entry for entry in replay.entries if entry.split == "validation"
    )
    train = next(entry for entry in replay.entries if entry.split == "train")
    replay.add_mc_targets(_mc_group(validation))
    protected_query = replay.mc_targets[0].query_id
    replay.add_mc_targets(_mc_group(train))

    assert len(replay.mc_targets) == 2
    assert {target.query_id for target in replay.mc_targets} == {protected_query}
    assert replay.mc_target_count("validation", anchor_only=True) == 2
