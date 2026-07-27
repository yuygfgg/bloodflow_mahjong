"""Paired all-action continuations for conservative policy iteration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import bloodflow_mahjong as bm

from .model import BloodFlowTransformer
from .pipeline import (
    EngineBuffers,
    _PinnedPolicyStager,
    _lineup_history_seat_masks,
    _launch_policy_actions,
    _safe_rule_actions,
)
from .policy_pool import ReplaySource
from .trajectory import CompactTrajectory


RolloutProgress = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class SearchRolloutResult:
    """Terminal outcomes with action-major, common-world alignment."""

    actions: np.ndarray
    score_delta: np.ndarray
    rank_utility: np.ndarray
    final_scores: np.ndarray
    rollout_states: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.uint8)
        score_delta = np.asarray(self.score_delta, dtype=np.float32)
        rank_utility = np.asarray(self.rank_utility, dtype=np.float32)
        final_scores = np.asarray(self.final_scores, dtype=np.int64)
        if actions.ndim != 1 or not len(actions):
            raise ValueError("rollout actions must be a non-empty vector")
        if score_delta.ndim != 2 or score_delta.shape[0] != len(actions):
            raise ValueError("score_delta must have shape [actions, worlds]")
        if rank_utility.shape != score_delta.shape:
            raise ValueError("rank_utility must match score_delta")
        if final_scores.shape != (*score_delta.shape, 4):
            raise ValueError("final_scores must have shape [actions, worlds, 4]")
        if not np.isfinite(score_delta).all() or not np.isfinite(rank_utility).all():
            raise ValueError("rollout utilities must be finite")
        if self.rollout_states <= 0 or self.elapsed_seconds < 0:
            raise ValueError("rollout accounting must be non-negative")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "score_delta", score_delta)
        object.__setattr__(self, "rank_utility", rank_utility)
        object.__setattr__(self, "final_scores", final_scores)

    @property
    def worlds(self) -> int:
        return self.score_delta.shape[1]


@dataclass(frozen=True)
class GroupedSearchRolloutResult:
    """Per-query results from one shared engine and Actor rollout batch."""

    queries: tuple[SearchRolloutResult, ...]
    rollout_states: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError("grouped rollout must contain at least one query")
        if self.rollout_states != sum(row.rollout_states for row in self.queries):
            raise ValueError("grouped rollout state accounting does not match")
        if self.elapsed_seconds < 0:
            raise ValueError("grouped rollout elapsed time must be non-negative")


def infer_policy_lineup(
    trajectory: CompactTrajectory, focal_seat: int
) -> np.ndarray:
    """Recover one fixed per-seat policy lineup from a collected trajectory."""

    if not 0 <= focal_seat < 4:
        raise ValueError("focal_seat must be in 0..3")
    lineup = np.full(4, -1, dtype=np.int8)
    for seat in range(4):
        sources = np.unique(trajectory.sources[trajectory.actors == seat])
        if len(sources) != 1:
            raise ValueError(f"seat {seat} does not have one fixed policy source")
        lineup[seat] = int(sources[0])
    if lineup[focal_seat] != int(ReplaySource.CURRENT):
        raise ValueError("focal seat must be controlled by CURRENT")
    opponents = np.delete(lineup, focal_seat)
    if not np.isin(
        opponents,
        (
            int(ReplaySource.RULE_FAST),
            int(ReplaySource.RULE_SAFE),
            int(ReplaySource.SELF_PLAY),
        ),
    ).all():
        raise ValueError(
            "non-focal seats must use rule or SELF_PLAY controllers"
        )
    return lineup


def _validate_policy_lineups(
    lineups: np.ndarray, focal_seats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize one policy lineup per active rollout row."""

    lineups = np.asarray(lineups, dtype=np.int8)
    focal_seats = np.asarray(focal_seats, dtype=np.int64)
    if lineups.ndim != 2 or lineups.shape[1] != 4:
        raise ValueError("lineups must have shape [batch, 4]")
    if focal_seats.shape != (len(lineups),) or np.any(
        (focal_seats < 0) | (focal_seats >= 4)
    ):
        raise ValueError("focal_seats must contain one valid seat per row")
    rows = np.arange(len(lineups), dtype=np.int64)
    if not np.all(lineups[rows, focal_seats] == int(ReplaySource.CURRENT)):
        raise ValueError("each focal seat must use the CURRENT controller")
    opponent_mask = np.ones_like(lineups, dtype=np.bool_)
    opponent_mask[rows, focal_seats] = False
    if not np.isin(
        lineups[opponent_mask],
        (
            int(ReplaySource.RULE_FAST),
            int(ReplaySource.RULE_SAFE),
            int(ReplaySource.SELF_PLAY),
        ),
    ).all():
        raise ValueError("non-focal seats must use rule or SELF_PLAY controllers")
    return lineups, focal_seats


def reconstruct_state(trajectory: CompactTrajectory, step: int) -> Any:
    """Replay one compact trajectory to the state immediately before ``step``."""

    if not 0 <= step < len(trajectory):
        raise ValueError("step is outside the trajectory")
    batch = bm.Batch(1, seed=int(trajectory.seed))
    records = np.empty((1, int(bm.STEP_RECORD_WIDTH)), dtype=np.int64)
    for action in trajectory.actions[:step]:
        batch.step_into(np.asarray([action], dtype=np.uint8), records)
        if records[0, 11]:
            raise RuntimeError("trajectory terminated before the requested state")
    return batch


def _absolute_scores(meta: np.ndarray, focal_seat: int) -> np.ndarray:
    if meta.ndim != 2 or meta.shape[1] < 16:
        raise ValueError("meta must have shape [batch, >=16]")
    relative = meta[:, 12:16].astype(np.int64, copy=False)
    absolute = np.empty_like(relative)
    for offset in range(4):
        absolute[:, (focal_seat + offset) % 4] = relative[:, offset]
    return absolute


def _launch_frozen_policy_actions(
    model: BloodFlowTransformer,
    buffers: EngineBuffers,
    rows: np.ndarray,
    device: torch.device,
    *,
    inference_batch_size: int | None = None,
    stager: _PinnedPolicyStager | None = None,
) -> torch.Tensor:
    if not len(rows):
        return torch.empty(0, dtype=torch.uint8, device=device)
    if inference_batch_size is not None and inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    return _launch_policy_actions(
        model,
        buffers,
        rows,
        device,
        inference_batch_size=inference_batch_size or len(rows),
        stager=stager,
    )


def _lineup_actions(
    buffers: EngineBuffers,
    model: BloodFlowTransformer,
    focal_seat: int | np.ndarray,
    lineup: np.ndarray,
    device: torch.device,
    *,
    inference_batch_size: int | None = None,
    stager: _PinnedPolicyStager | None = None,
) -> np.ndarray:
    actors = buffers.meta[:, 1].astype(np.int64)
    if np.any((actors < 0) | (actors >= 4)):
        raise RuntimeError("active rollout contains a terminal actor")
    focal_seats = np.asarray(focal_seat, dtype=np.int64)
    if focal_seats.ndim == 0:
        focal_seats = np.full(len(actors), int(focal_seats), dtype=np.int64)
    if focal_seats.shape != (len(actors),) or np.any(
        (focal_seats < 0) | (focal_seats >= 4)
    ):
        raise ValueError("focal_seat must be a seat scalar or one seat per row")
    lineups = np.asarray(lineup, dtype=np.int8)
    if lineups.shape == (4,):
        lineups = np.broadcast_to(lineups, (len(actors), 4))
    if lineups.shape != (len(actors), 4):
        raise ValueError("lineup must have shape [4] or [batch, 4]")
    rows = np.arange(len(actors))
    sources = lineups[rows, actors]
    focal_rows = actors == focal_seats
    if not np.array_equal(
        sources == int(ReplaySource.CURRENT), focal_rows
    ):
        raise ValueError("CURRENT must control exactly the focal decision rows")
    if not np.isin(
        sources,
        (
            int(ReplaySource.CURRENT),
            int(ReplaySource.RULE_FAST),
            int(ReplaySource.RULE_SAFE),
            int(ReplaySource.SELF_PLAY),
        ),
    ).all():
        raise ValueError("active rows use an unknown policy controller")
    model_rows = np.flatnonzero(
        np.isin(
            sources,
            (int(ReplaySource.CURRENT), int(ReplaySource.SELF_PLAY)),
        )
    )
    pending_model_actions = (
        _launch_frozen_policy_actions(
            model,
            buffers,
            model_rows,
            device,
            inference_batch_size=inference_batch_size,
            stager=stager,
        )
        if len(model_rows)
        else None
    )

    actions = buffers.actions
    rule_enabled = np.isin(
        sources,
        (int(ReplaySource.RULE_FAST), int(ReplaySource.RULE_SAFE)),
    ).astype(np.uint8)
    if np.any(rule_enabled):
        buffers.batch.simple_rule_actions_masked_into(rule_enabled, actions)
        safe_rows = np.flatnonzero(sources == int(ReplaySource.RULE_SAFE))
        _safe_rule_actions(
            actions,
            buffers.legal,
            buffers.tile_obs,
            safe_rows,
        )
    if len(model_rows):
        model_actions = pending_model_actions.cpu().numpy()
        if np.any(model_actions == np.iinfo(np.uint8).max):
            raise RuntimeError("Actor produced non-finite logits during continuation")
        actions[model_rows] = model_actions

    rows = np.arange(len(actions))
    if not buffers.legal[rows, actions.astype(np.int64)].all():
        raise RuntimeError("continuation selected an illegal action")
    return actions


def _rank_utility(final_scores: np.ndarray, focal_seat: int) -> np.ndarray:
    focal = final_scores[:, focal_seat]
    rank = np.ones(len(final_scores), dtype=np.int8)
    for seat in range(4):
        if seat == focal_seat:
            continue
        rank += (
            (final_scores[:, seat] > focal)
            | ((final_scores[:, seat] == focal) & (seat < focal_seat))
        ).astype(np.int8)
    return 2.5 - rank.astype(np.float32)


def _rollout_query_group(
    worlds: Any,
    action_sets: Sequence[np.ndarray],
    model: BloodFlowTransformer,
    device: torch.device,
    *,
    focal_seats: np.ndarray,
    lineups: np.ndarray,
    inference_batch_size: int,
    maximum_steps: int,
    on_progress: RolloutProgress | None = None,
    inference_stager: _PinnedPolicyStager | None = None,
) -> GroupedSearchRolloutResult:
    """Run several queries together while retaining query-local world pairing."""

    started = time.perf_counter()
    query_count = len(action_sets)
    if query_count <= 0 or len(worlds) % query_count:
        raise ValueError("worlds must contain an equal non-empty block per query")
    world_count = len(worlds) // query_count
    if world_count <= 0:
        raise ValueError("each query needs at least one world")
    focal_seats = np.asarray(focal_seats, dtype=np.int64)
    lineups = np.asarray(lineups, dtype=np.int8)
    if focal_seats.shape != (query_count,) or np.any(
        (focal_seats < 0) | (focal_seats >= 4)
    ):
        raise ValueError("focal_seats must contain one valid seat per query")
    if lineups.shape != (query_count, 4):
        raise ValueError("lineups must have shape [queries, 4]")
    lineups, focal_seats = _validate_policy_lineups(lineups, focal_seats)
    model.eval()
    query_history_masks = _lineup_history_seat_masks(lineups)
    if inference_batch_size <= 0 or maximum_steps <= 0:
        raise ValueError("rollout sizes must be positive")

    normalized_actions: list[np.ndarray] = []
    for actions in action_sets:
        actions = np.asarray(actions, dtype=np.uint8)
        if actions.ndim != 1 or not len(actions) or len(np.unique(actions)) != len(actions):
            raise ValueError("each query needs unique non-empty actions")
        normalized_actions.append(actions)

    source = EngineBuffers.for_batch(worlds, history=model.config.max_history)
    source.observe(np.zeros(len(worlds), dtype=np.uint8))
    world_queries = np.repeat(np.arange(query_count, dtype=np.int64), world_count)
    if not np.array_equal(source.meta[:, 1], focal_seats[world_queries]):
        raise ValueError("worlds are not at their query's focal-seat decision")
    start_scores = np.empty((query_count, 4), dtype=np.int64)
    for query, actions in enumerate(normalized_actions):
        rows = slice(query * world_count, (query + 1) * world_count)
        legal = source.legal[rows]
        if not np.all(legal == legal[0]):
            raise ValueError("query worlds do not share one legal action set")
        if not legal[0, actions.astype(np.int64)].all():
            raise ValueError("candidate set contains an illegal action")
        scores = _absolute_scores(source.meta[rows], int(focal_seats[query]))
        if not np.all(scores == scores[0]):
            raise ValueError("query worlds do not share the same public scores")
        start_scores[query] = scores[0]

    branch_world_rows: list[np.ndarray] = []
    first_action_rows: list[np.ndarray] = []
    branch_query_rows: list[np.ndarray] = []
    offsets = [0]
    for query, actions in enumerate(normalized_actions):
        world_rows = np.arange(
            query * world_count, (query + 1) * world_count, dtype=np.uint32
        )
        count = len(actions) * world_count
        branch_world_rows.append(np.tile(world_rows, len(actions)))
        first_action_rows.append(np.repeat(actions, world_count))
        branch_query_rows.append(np.full(count, query, dtype=np.int64))
        offsets.append(offsets[-1] + count)
    branch_indices = np.concatenate(branch_world_rows)
    first_actions = np.concatenate(first_action_rows)
    branch_queries = np.concatenate(branch_query_rows)
    branches = worlds.clone_indices(branch_indices)
    branch_count = len(first_actions)
    buffers = EngineBuffers.for_batch(branches, history=model.config.max_history)
    history_masks = query_history_masks[branch_queries]
    buffers.batch.step_and_observe_history_into(
        first_actions,
        history_masks,
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
    records = buffers.records
    cumulative = records[:, 5:9].astype(np.int64, copy=True)
    final_scores = np.empty((branch_count, 4), dtype=np.int64)
    terminal = records[:, 11].astype(np.bool_)
    if np.any(terminal):
        final_scores[terminal] = (
            start_scores[branch_queries[terminal]] + cumulative[terminal]
        )
    active_ids = np.flatnonzero(~terminal).astype(np.int64)
    if not len(active_ids):
        buffers = None
    elif len(active_ids) != branch_count:
        buffers, active_ids = buffers.remove_rows(np.flatnonzero(terminal))
    rollout_states_by_query = np.bincount(
        branch_queries, minlength=query_count
    ).astype(np.int64)

    steps = 1
    if inference_stager is None and device.type == "cuda":
        inference_stager = _PinnedPolicyStager(
            device, model.config.max_history
        )
    if on_progress is not None:
        on_progress(
            {
                "rollout_step": steps,
                "active_branches": len(active_ids),
                "group_rollout_states": int(rollout_states_by_query.sum()),
                "group_elapsed_seconds": time.perf_counter() - started,
            }
        )
    while buffers is not None and len(active_ids):
        if steps >= maximum_steps:
            raise RuntimeError("search continuation exceeded the engine step limit")
        active_queries = branch_queries[active_ids]
        next_actions = _lineup_actions(
            buffers,
            model,
            focal_seats[active_queries],
            lineups[active_queries],
            device,
            inference_batch_size=inference_batch_size,
            stager=inference_stager,
        )
        history_masks = query_history_masks[active_queries]
        buffers.batch.step_and_observe_history_into(
            next_actions,
            history_masks,
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
        next_records = buffers.records
        cumulative[active_ids] += next_records[:, 5:9]
        rollout_states_by_query += np.bincount(
            active_queries, minlength=query_count
        )
        terminal = next_records[:, 11].astype(np.bool_)
        if np.any(terminal):
            finished = active_ids[terminal]
            final_scores[finished] = (
                start_scores[branch_queries[finished]] + cumulative[finished]
            )
        terminal_rows = np.flatnonzero(terminal)
        if len(terminal_rows) == len(active_ids):
            break
        if len(terminal_rows):
            buffers, survivor_order = buffers.remove_rows(terminal_rows)
            active_ids = active_ids[survivor_order]
        steps += 1
        if on_progress is not None:
            on_progress(
                {
                    "rollout_step": steps,
                    "active_branches": len(active_ids),
                    "group_rollout_states": int(rollout_states_by_query.sum()),
                    "group_elapsed_seconds": time.perf_counter() - started,
                }
            )

    if on_progress is not None:
        on_progress(
            {
                "rollout_step": steps,
                "active_branches": 0,
                "group_rollout_states": int(rollout_states_by_query.sum()),
                "group_elapsed_seconds": time.perf_counter() - started,
            }
        )

    results: list[SearchRolloutResult] = []
    for query, actions in enumerate(normalized_actions):
        rows = slice(offsets[query], offsets[query + 1])
        shape = (len(actions), world_count)
        query_scores = final_scores[rows]
        query_cumulative = cumulative[rows]
        results.append(
            SearchRolloutResult(
                actions=actions,
                score_delta=(
                    query_cumulative[:, int(focal_seats[query])].astype(np.float32)
                    / 10_000.0
                ).reshape(shape),
                rank_utility=_rank_utility(
                    query_scores, int(focal_seats[query])
                ).reshape(shape),
                final_scores=query_scores.reshape(*shape, 4),
                rollout_states=int(rollout_states_by_query[query]),
                elapsed_seconds=0.0,
            )
        )
    return GroupedSearchRolloutResult(
        queries=tuple(results),
        rollout_states=int(rollout_states_by_query.sum()),
        elapsed_seconds=time.perf_counter() - started,
    )


def rollout_query_group_chunked(
    worlds: Any,
    action_sets: Sequence[np.ndarray],
    model: BloodFlowTransformer,
    device: torch.device,
    *,
    focal_seats: np.ndarray,
    lineups: np.ndarray,
    world_chunk: int,
    inference_batch_size: int,
    maximum_steps: int = 2048,
    on_progress: RolloutProgress | None = None,
    inference_stager: _PinnedPolicyStager | None = None,
) -> GroupedSearchRolloutResult:
    """Run query-major worlds in bounded world-axis chunks."""

    query_count = len(action_sets)
    if query_count <= 0 or len(worlds) % query_count:
        raise ValueError("worlds must contain equal query-major blocks")
    if world_chunk <= 0:
        raise ValueError("world_chunk must be positive")
    worlds_per_query = len(worlds) // query_count
    if worlds_per_query <= world_chunk:
        return _rollout_query_group(
            worlds,
            action_sets,
            model,
            device,
            focal_seats=focal_seats,
            lineups=lineups,
            inference_batch_size=inference_batch_size,
            maximum_steps=maximum_steps,
            on_progress=on_progress,
            inference_stager=inference_stager,
        )

    started = time.perf_counter()
    parts: list[GroupedSearchRolloutResult] = []
    chunk_count = (worlds_per_query + world_chunk - 1) // world_chunk
    for chunk_index, start in enumerate(range(0, worlds_per_query, world_chunk)):
        stop = min(start + world_chunk, worlds_per_query)
        indices = np.concatenate(
            [
                np.arange(
                    query * worlds_per_query + start,
                    query * worlds_per_query + stop,
                    dtype=np.uint32,
                )
                for query in range(query_count)
            ]
        )
        completed_states = sum(part.rollout_states for part in parts)
        completed_seconds = sum(part.elapsed_seconds for part in parts)

        def chunk_progress(fields: Mapping[str, object]) -> None:
            if on_progress is None:
                return
            merged = dict(fields)
            merged["group_rollout_states"] = completed_states + int(
                fields["group_rollout_states"]
            )
            merged["group_elapsed_seconds"] = completed_seconds + float(
                fields["group_elapsed_seconds"]
            )
            merged["world_chunk"] = f"{chunk_index + 1}/{chunk_count}"
            on_progress(merged)

        parts.append(
            _rollout_query_group(
                worlds.clone_indices(indices),
                action_sets,
                model,
                device,
                focal_seats=focal_seats,
                lineups=lineups,
                inference_batch_size=inference_batch_size,
                maximum_steps=maximum_steps,
                on_progress=chunk_progress if on_progress is not None else None,
                inference_stager=inference_stager,
            )
        )
    results: list[SearchRolloutResult] = []
    for query, actions in enumerate(action_sets):
        results.append(
            SearchRolloutResult(
                actions=np.asarray(actions, dtype=np.uint8),
                score_delta=np.concatenate(
                    [part.queries[query].score_delta for part in parts], axis=1
                ),
                rank_utility=np.concatenate(
                    [part.queries[query].rank_utility for part in parts], axis=1
                ),
                final_scores=np.concatenate(
                    [part.queries[query].final_scores for part in parts], axis=1
                ),
                rollout_states=sum(
                    part.queries[query].rollout_states for part in parts
                ),
                elapsed_seconds=0.0,
            )
        )
    return GroupedSearchRolloutResult(
        queries=tuple(results),
        rollout_states=sum(part.rollout_states for part in parts),
        elapsed_seconds=time.perf_counter() - started,
    )


def rollout_all_actions(
    worlds: Any,
    actions: np.ndarray,
    model: BloodFlowTransformer,
    device: torch.device,
    *,
    focal_seat: int,
    lineup: np.ndarray,
    maximum_steps: int = 2048,
) -> SearchRolloutResult:
    """Run every action on the same ordered worlds and preserve the source lineup."""

    started = time.perf_counter()
    actions = np.asarray(actions, dtype=np.uint8)
    lineup = np.asarray(lineup, dtype=np.int8)
    if actions.ndim != 1 or not len(actions) or len(np.unique(actions)) != len(actions):
        raise ValueError("actions must be a non-empty vector of unique action ids")
    if lineup.shape != (4,):
        raise ValueError("lineup must have shape [4]")
    if maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive")
    world_count = len(worlds)
    if world_count <= 0:
        raise ValueError("world batch must be non-empty")
    _validate_policy_lineups(
        lineup[None, :], np.asarray([focal_seat], dtype=np.int64)
    )
    model.eval()

    source = EngineBuffers.for_batch(worlds, history=model.config.max_history)
    source.observe(np.zeros(world_count, dtype=np.uint8))
    if not np.all(source.meta[:, 1] == focal_seat):
        raise ValueError("all worlds must be at a focal-seat decision")
    if not np.all(source.legal == source.legal[0]):
        raise ValueError("worlds do not share one legal action set")
    if not source.legal[0, actions.astype(np.int64)].all():
        raise ValueError("candidate set contains an illegal action")
    start_scores = _absolute_scores(source.meta, focal_seat)
    if not np.all(start_scores == start_scores[0]):
        raise ValueError("worlds do not share the same public scores")

    branch_indices = np.tile(np.arange(world_count, dtype=np.uint32), len(actions))
    branches = worlds.clone_indices(branch_indices)
    first_actions = np.repeat(actions, world_count)
    branch_count = len(first_actions)
    records = np.empty((branch_count, int(bm.STEP_RECORD_WIDTH)), dtype=np.int64)
    branches.step_into(first_actions, records)
    cumulative = records[:, 5:9].astype(np.int64, copy=True)
    final_scores = np.empty((branch_count, 4), dtype=np.int64)
    terminal = records[:, 11].astype(np.bool_)
    final_scores[terminal] = start_scores[0] + cumulative[terminal]
    active_ids = np.flatnonzero(~terminal).astype(np.int64)
    active = (
        None
        if not len(active_ids)
        else branches.clone_indices(active_ids.astype(np.uint32))
    )
    rollout_states = branch_count
    history_mask = _lineup_history_seat_masks(lineup[None, :])[0]

    steps = 1
    while active is not None and len(active_ids):
        if steps >= maximum_steps:
            raise RuntimeError("search continuation exceeded the engine step limit")
        buffers = EngineBuffers.for_batch(active, history=model.config.max_history)
        buffers.observe(
            np.full(len(active_ids), history_mask, dtype=np.uint8)
        )
        next_actions = _lineup_actions(
            buffers, model, focal_seat, lineup, device
        )
        next_records = np.empty(
            (len(active_ids), int(bm.STEP_RECORD_WIDTH)), dtype=np.int64
        )
        active.step_into(next_actions, next_records)
        cumulative[active_ids] += next_records[:, 5:9]
        terminal = next_records[:, 11].astype(np.bool_)
        if np.any(terminal):
            finished = active_ids[terminal]
            final_scores[finished] = start_scores[0] + cumulative[finished]
        rollout_states += len(active_ids)
        keep = np.flatnonzero(~terminal).astype(np.uint32)
        if not len(keep):
            break
        active_ids = active_ids[keep]
        active = active.clone_indices(keep)
        steps += 1

    score_delta = cumulative[:, focal_seat].astype(np.float32) / 10_000.0
    rank_utility = _rank_utility(final_scores, focal_seat)
    shape = (len(actions), world_count)
    return SearchRolloutResult(
        actions=actions,
        score_delta=score_delta.reshape(shape),
        rank_utility=rank_utility.reshape(shape),
        final_scores=final_scores.reshape(*shape, 4),
        rollout_states=rollout_states,
        elapsed_seconds=time.perf_counter() - started,
    )


def rollout_all_actions_chunked(
    worlds: Any,
    actions: np.ndarray,
    model: BloodFlowTransformer,
    device: torch.device,
    *,
    focal_seat: int,
    lineup: np.ndarray,
    world_chunk: int,
    maximum_steps: int = 2048,
) -> SearchRolloutResult:
    """Run an action sweep in world-axis chunks without changing alignment."""

    if world_chunk <= 0:
        raise ValueError("world_chunk must be positive")
    if len(worlds) <= world_chunk:
        return rollout_all_actions(
            worlds,
            actions,
            model,
            device,
            focal_seat=focal_seat,
            lineup=lineup,
            maximum_steps=maximum_steps,
        )

    started = time.perf_counter()
    parts: list[SearchRolloutResult] = []
    for start in range(0, len(worlds), world_chunk):
        stop = min(start + world_chunk, len(worlds))
        indices = np.arange(start, stop, dtype=np.uint32)
        parts.append(
            rollout_all_actions(
                worlds.clone_indices(indices),
                actions,
                model,
                device,
                focal_seat=focal_seat,
                lineup=lineup,
                maximum_steps=maximum_steps,
            )
        )
    return SearchRolloutResult(
        actions=np.asarray(actions, dtype=np.uint8),
        score_delta=np.concatenate([part.score_delta for part in parts], axis=1),
        rank_utility=np.concatenate(
            [part.rank_utility for part in parts], axis=1
        ),
        final_scores=np.concatenate(
            [part.final_scores for part in parts], axis=1
        ),
        rollout_states=sum(part.rollout_states for part in parts),
        elapsed_seconds=time.perf_counter() - started,
    )
