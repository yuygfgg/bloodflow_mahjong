from __future__ import annotations

import numpy as np
import pytest
import torch

import bloodflow_mahjong as bm

from training.model import BloodFlowTransformer, TransformerConfig
from training.observation import unpack_action_masks
from training.pipeline import EngineBuffers
from training.policy_pool import ReplaySource
from training.search_rollout import (
    SearchRolloutResult,
    _lineup_actions,
    infer_policy_lineup,
    rollout_all_actions_chunked,
    rollout_query_group_chunked,
)
from training.trajectory import CompactTrajectory


def lineup_trajectory() -> CompactTrajectory:
    return CompactTrajectory(
        seed=1,
        exchange_direction=1,
        actions=np.zeros(4, dtype=np.uint8),
        actors=np.arange(4, dtype=np.uint8),
        phases=np.zeros(4, dtype=np.uint8),
        categories=np.zeros(4, dtype=np.uint8),
        sources=np.asarray(
            [
                int(ReplaySource.CURRENT),
                int(ReplaySource.RULE_FAST),
                int(ReplaySource.RULE_SAFE),
                int(ReplaySource.RULE_FAST),
            ],
            dtype=np.uint8,
        ),
        legal_counts=np.full(4, 2, dtype=np.uint8),
        terminal_scores=np.full(4, 10_000, dtype=np.int32),
        terminal_ranks=np.arange(1, 5, dtype=np.uint8),
        termination_reason=int(bm.TERMINATION_WALL_EXHAUSTED),
    )


def test_policy_lineup_reconstruction_accepts_self_play_opponents() -> None:
    trajectory = lineup_trajectory()
    np.testing.assert_array_equal(
        infer_policy_lineup(trajectory, 0), trajectory.sources
    )
    self_play = CompactTrajectory(
        **{
            **trajectory.__dict__,
            "sources": np.asarray(
                [
                    int(ReplaySource.CURRENT),
                    int(ReplaySource.SELF_PLAY),
                    int(ReplaySource.RULE_SAFE),
                    int(ReplaySource.SELF_PLAY),
                ],
                dtype=np.uint8,
            ),
        }
    )
    np.testing.assert_array_equal(
        infer_policy_lineup(self_play, 0), self_play.sources
    )


@pytest.mark.parametrize(
    "sources, message",
    [
        (
            [
                ReplaySource.CURRENT,
                ReplaySource.CURRENT,
                ReplaySource.RULE_SAFE,
                ReplaySource.RULE_FAST,
            ],
            "non-focal",
        ),
        (
            [
                ReplaySource.SELF_PLAY,
                ReplaySource.RULE_FAST,
                ReplaySource.RULE_SAFE,
                ReplaySource.RULE_FAST,
            ],
            "focal seat",
        ),
        (
            [
                ReplaySource.CURRENT,
                ReplaySource.SL,
                ReplaySource.RULE_SAFE,
                ReplaySource.RULE_FAST,
            ],
            "non-focal",
        ),
    ],
)
def test_policy_lineup_reconstruction_rejects_invalid_model_roles(
    sources: list[ReplaySource], message: str
) -> None:
    trajectory = lineup_trajectory()
    broken = CompactTrajectory(
        **{
            **trajectory.__dict__,
            "sources": np.asarray([int(source) for source in sources], dtype=np.uint8),
        }
    )
    with pytest.raises(ValueError, match=message):
        infer_policy_lineup(broken, 0)


def test_lineup_actions_overlap_model_and_masked_rule_rows(monkeypatch) -> None:
    batch = bm.Batch(4, seed=29)
    buffers = EngineBuffers.for_batch(batch, history=16)
    buffers.observe()
    actors = buffers.meta[:, 1].astype(np.int64)
    focal_seats = np.asarray(
        [actors[0], *(((actors[1:] + 1) % 4).tolist())], dtype=np.int64
    )
    lineups = np.full((4, 4), int(ReplaySource.RULE_FAST), dtype=np.int8)
    lineups[np.arange(4), focal_seats] = int(ReplaySource.CURRENT)
    lineups[1, actors[1]] = int(ReplaySource.SELF_PLAY)
    lineups[3, actors[3]] = int(ReplaySource.RULE_SAFE)
    calls: list[tuple[str, np.ndarray]] = []

    def fake_model_actions(model, engine_buffers, rows, device, **kwargs):
        calls.append(("model", rows.copy()))
        return torch.as_tensor(
            [np.flatnonzero(engine_buffers.legal[row])[0] for row in rows],
            dtype=torch.uint8,
        )

    class CapturingBatch:
        def simple_rule_actions_masked_into(self, enabled, output):
            calls.append(("rule", enabled.copy()))
            for row in np.flatnonzero(enabled):
                output[row] = np.flatnonzero(buffers.legal[row])[0]

    buffers.batch = CapturingBatch()
    monkeypatch.setattr(
        "training.search_rollout._launch_frozen_policy_actions", fake_model_actions
    )
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=16,
        )
    ).eval()
    actions = _lineup_actions(
        buffers,
        actor,
        focal_seats,
        lineups,
        torch.device("cpu"),
    )
    assert [call[0] for call in calls] == ["model", "rule"]
    np.testing.assert_array_equal(calls[0][1], np.asarray([0, 1]))
    np.testing.assert_array_equal(calls[1][1], np.asarray([0, 0, 1, 1]))
    assert buffers.legal[np.arange(4), actions.astype(np.int64)].all()


def test_lineup_actions_routes_self_play_to_a_separate_model(monkeypatch) -> None:
    batch = bm.Batch(3, seed=29)
    buffers = EngineBuffers.for_batch(batch, history=16)
    buffers.observe()
    actors = buffers.meta[:, 1].astype(np.int64)
    focal_seats = (actors + 1) % 4
    focal_seats[0] = actors[0]
    lineups = np.full((3, 4), int(ReplaySource.RULE_FAST), dtype=np.int8)
    lineups[np.arange(3), focal_seats] = int(ReplaySource.CURRENT)
    lineups[1, actors[1]] = int(ReplaySource.SELF_PLAY)
    lineups[2, actors[2]] = int(ReplaySource.SELF_PLAY)
    calls: list[tuple[str, np.ndarray]] = []

    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=16,
        )
    ).eval()
    opponent = BloodFlowTransformer(actor.config).eval()

    def fake_model_actions(model, engine_buffers, rows, device, **kwargs):
        calls.append(("focal" if model is actor else "opponent", rows.copy()))
        return torch.as_tensor(
            [np.flatnonzero(engine_buffers.legal[row])[0] for row in rows],
            dtype=torch.uint8,
        )

    monkeypatch.setattr(
        "training.search_rollout._launch_frozen_policy_actions", fake_model_actions
    )
    actions = _lineup_actions(
        buffers,
        actor,
        focal_seats,
        lineups,
        torch.device("cpu"),
        self_play_model=opponent,
    )
    assert [name for name, _rows in calls] == ["focal", "opponent"]
    np.testing.assert_array_equal(calls[0][1], np.asarray([0]))
    np.testing.assert_array_equal(calls[1][1], np.asarray([1, 2]))
    assert buffers.legal[np.arange(3), actions.astype(np.int64)].all()


def test_search_result_validates_action_major_alignment() -> None:
    result = SearchRolloutResult(
        actions=np.asarray([2, 3], dtype=np.uint8),
        score_delta=np.zeros((2, 4), dtype=np.float32),
        rank_utility=np.zeros((2, 4), dtype=np.float32),
        final_scores=np.full((2, 4, 4), 10_000, dtype=np.int64),
        rollout_states=8,
        elapsed_seconds=0.1,
    )
    assert result.worlds == 4
    with pytest.raises(ValueError, match="rank_utility"):
        SearchRolloutResult(
            actions=result.actions,
            score_delta=result.score_delta,
            rank_utility=np.zeros((1, 4), dtype=np.float32),
            final_scores=result.final_scores,
            rollout_states=8,
            elapsed_seconds=0.1,
        )


def test_chunked_rollout_runs_all_actions_on_common_worlds() -> None:
    source = bm.Batch(1, seed=29)
    buffers = EngineBuffers.for_batch(source, history=16)
    buffers.observe()
    focal = int(buffers.meta[0, 1])
    actions = np.flatnonzero(buffers.legal[0])[:2].astype(np.uint8)
    assert len(actions) == 2
    worlds = source.resample_live_walls(
        np.zeros(2, dtype=np.uint32), np.asarray([101, 103], dtype=np.uint64)
    )
    lineup = np.asarray(
        [
            int(ReplaySource.RULE_FAST),
            int(ReplaySource.RULE_SAFE),
            int(ReplaySource.RULE_FAST),
            int(ReplaySource.RULE_SAFE),
        ],
        dtype=np.int8,
    )
    lineup[focal] = int(ReplaySource.CURRENT)
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=16,
        )
    ).eval()
    result = rollout_all_actions_chunked(
        worlds,
        actions,
        actor,
        torch.device("cpu"),
        focal_seat=focal,
        lineup=lineup,
        world_chunk=1,
    )
    assert result.score_delta.shape == (2, 2)
    assert result.rank_utility.shape == (2, 2)
    assert result.final_scores.shape == (2, 2, 4)
    assert result.rollout_states >= 4


def test_grouped_rollout_matches_serial_query_world_pairing() -> None:
    source = bm.Batch(2, seed=29)
    source.reset_many(
        np.arange(2, dtype=np.uint32),
        np.asarray([29, 31], dtype=np.uint64),
    )
    source_buffers = EngineBuffers.for_batch(source, history=16)
    source_buffers.observe()
    focal_seats = source_buffers.meta[:, 1].astype(np.int64)
    action_sets = [
        np.flatnonzero(source_buffers.legal[row])[:2].astype(np.uint8)
        for row in range(2)
    ]
    lineups = np.empty((2, 4), dtype=np.int8)
    for row, focal in enumerate(focal_seats):
        lineups[row] = np.asarray(
            [
                int(ReplaySource.RULE_FAST),
                int(ReplaySource.RULE_SAFE),
                int(ReplaySource.RULE_FAST),
                int(ReplaySource.RULE_SAFE),
            ],
            dtype=np.int8,
        )
        lineups[row, focal] = int(ReplaySource.CURRENT)
        lineups[row, (focal + 1) % 4] = int(ReplaySource.SELF_PLAY)
    world_seeds = np.asarray([101, 103, 107, 109], dtype=np.uint64)
    worlds = source.resample_live_walls(
        np.asarray([0, 0, 1, 1], dtype=np.uint32), world_seeds
    )
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=16,
        )
    ).eval()

    progress = []
    grouped = rollout_query_group_chunked(
        worlds,
        action_sets,
        actor,
        torch.device("cpu"),
        focal_seats=focal_seats,
        lineups=lineups,
        world_chunk=2,
        inference_batch_size=1,
        on_progress=progress.append,
    )
    serial = []
    for row in range(2):
        serial.append(
            rollout_all_actions_chunked(
                worlds.clone_indices(
                    np.arange(row * 2, row * 2 + 2, dtype=np.uint32)
                ),
                action_sets[row],
                actor,
                torch.device("cpu"),
                focal_seat=int(focal_seats[row]),
                lineup=lineups[row],
                world_chunk=2,
            )
        )
    for actual, expected in zip(grouped.queries, serial):
        np.testing.assert_array_equal(actual.actions, expected.actions)
        np.testing.assert_array_equal(actual.score_delta, expected.score_delta)
        np.testing.assert_array_equal(actual.rank_utility, expected.rank_utility)
        np.testing.assert_array_equal(actual.final_scores, expected.final_scores)
    assert grouped.rollout_states == sum(row.rollout_states for row in serial)
    assert progress
    assert progress[-1]["group_rollout_states"] == grouped.rollout_states
    assert progress[-1]["active_branches"] >= 0
