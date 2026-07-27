"""Large-independent-batch conservative policy iteration primitives.

Each update estimates every legal action on many independent hidden deals,
accumulates one full-batch gradient, performs exactly one AdamW step, and then
scales that direction to a KL trust region on disjoint calibration states.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import bloodflow_mahjong as bm

from .model import ACTION_SPACE_SIZE, BloodFlowTransformer
from .observation import bucket_history_width
from .pipeline import EngineBuffers, _PinnedPolicyStager, _autocast
from .policy_pool import CATEGORY_COUNT, CATEGORY_NAMES, ReplaySource, decision_categories
from .search_rollout import (
    infer_policy_lineup,
    rollout_query_group_chunked,
)
from .trajectory import CompactTrajectory


TARGET_CACHE_VERSION = 4
STATE_CACHE_VERSION = 1
KL_TARGET_RELATIVE_TOLERANCE = 0.05
WORLD_SEED_DOMAIN = 0xC71E_0001
UINT64_MASK = (1 << 64) - 1
TargetProgress = Callable[[int, Mapping[str, object]], None]


def require_cuda(device: str | torch.device | None = None) -> torch.device:
    resolved = torch.device(device or "cuda")
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA; CPU fallback is disabled")
    return resolved


def require_deterministic_actor(actor: BloodFlowTransformer) -> None:
    if actor.config.dropout != 0:
        raise ValueError("policy iteration requires an Actor with dropout=0")


def mix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def domain_seed(root: int, domain: int, index: int = 0) -> int:
    return mix64(int(root) ^ mix64(domain) ^ mix64(index))


def world_seeds(seed: int, query_id: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("world count must be positive")
    root = domain_seed(seed, WORLD_SEED_DOMAIN, query_id)
    return np.asarray([mix64(root + row) for row in range(count)], dtype=np.uint64)


@dataclass(frozen=True)
class PolicyQuery:
    query_id: int
    trajectory_index: int
    trajectory: CompactTrajectory
    step: int
    category: int


@dataclass(frozen=True)
class PolicyStateBatch:
    query_ids: np.ndarray
    tile_obs: np.ndarray
    melds: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray
    legal: np.ndarray
    categories: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.query_ids)
        aligned = (
            self.tile_obs,
            self.melds,
            self.meta,
            self.events,
            self.event_lengths,
            self.legal,
            self.categories,
        )
        if size <= 0 or any(len(value) != size for value in aligned):
            raise ValueError("policy-state arrays must have one positive length")
        if self.legal.shape != (size, ACTION_SPACE_SIZE):
            raise ValueError("legal must have shape [states, ACTION_SPACE_SIZE]")
        if self.legal.dtype != np.bool_ or np.any(self.legal.sum(axis=1) < 2):
            raise ValueError("each policy state needs at least two legal actions")
        if np.any(self.categories >= CATEGORY_COUNT):
            raise ValueError("policy-state batch contains an invalid category")
        if len(np.unique(self.query_ids)) != size:
            raise ValueError("query ids must be unique")

    def __len__(self) -> int:
        return len(self.query_ids)


@dataclass(frozen=True)
class CounterfactualBatch(PolicyStateBatch):
    rank_q: np.ndarray
    score_q: np.ndarray
    centered_rank_q: np.ndarray
    behavior_actions: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        size = len(self)
        for name, value in (
            ("rank_q", self.rank_q),
            ("score_q", self.score_q),
            ("centered_rank_q", self.centered_rank_q),
        ):
            if value.shape != (size, ACTION_SPACE_SIZE):
                raise ValueError(f"{name} has the wrong shape")
            if not np.isfinite(value[self.legal]).all():
                raise ValueError(f"{name} has non-finite legal values")
            if np.any(value[~self.legal] != 0):
                raise ValueError(f"{name} must be zero on illegal actions")
        if self.behavior_actions.shape != (size,):
            raise ValueError("behavior_actions must have shape [states]")
        rows = np.arange(size)
        if np.any(~self.legal[rows, self.behavior_actions.astype(np.int64)]):
            raise ValueError("a behavior action is illegal")
        means = self.centered_rank_q.sum(axis=1) / self.legal.sum(axis=1)
        if not np.allclose(means, 0.0, atol=1e-6):
            raise ValueError("centered rank Q must have zero legal-action mean")


@dataclass(frozen=True)
class _CalibrationChunk:
    state: tuple[Tensor, ...]
    legal: Tensor
    weights: Tensor
    reference_log_probs: Tensor
    reference_actions: Tensor


def center_legal_values(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    legal = np.asarray(legal, dtype=np.bool_)
    if values.ndim != 2 or values.shape != legal.shape:
        raise ValueError("values and legal must be aligned matrices")
    counts = legal.sum(axis=1)
    if np.any(counts <= 0) or not np.isfinite(values[legal]).all():
        raise ValueError("each row needs finite values for a legal action")
    means = np.where(legal, values, 0.0).sum(axis=1) / counts
    return np.where(legal, values - means[:, None], 0.0).astype(np.float32)


def source_visit_frequencies(
    trajectories: Sequence[CompactTrajectory],
) -> dict[str, object]:
    if not trajectories:
        raise ValueError("source trajectories must not be empty")
    counts = np.zeros(CATEGORY_COUNT, dtype=np.int64)
    current_states = 0
    eligible_states = 0
    for trajectory in trajectories:
        current = trajectory.sources == int(ReplaySource.CURRENT)
        eligible = current & (trajectory.legal_counts >= 2)
        current_states += int(current.sum())
        eligible_states += int(eligible.sum())
        counts += np.bincount(
            trajectory.categories[eligible].astype(np.int64),
            minlength=CATEGORY_COUNT,
        )
    if eligible_states <= 0 or np.any(counts == 0):
        missing = [
            CATEGORY_NAMES[index] for index in np.flatnonzero(counts == 0).tolist()
        ]
        raise RuntimeError(f"source corpus cannot estimate categories: {missing}")
    fractions = counts.astype(np.float64) / float(eligible_states)
    return {
        "games": len(trajectories),
        "current_states": current_states,
        "eligible_multi_action_states": eligible_states,
        "excluded_single_action_states": current_states - eligible_states,
        "counts": {
            CATEGORY_NAMES[index]: int(counts[index])
            for index in range(CATEGORY_COUNT)
        },
        "fractions": {
            CATEGORY_NAMES[index]: float(fractions[index])
            for index in range(CATEGORY_COUNT)
        },
        "vector": fractions.tolist(),
    }


def select_independent_queries(
    trajectories: Sequence[CompactTrajectory],
    *,
    queries_per_category: int,
    seed: int,
) -> list[PolicyQuery]:
    """Sample eligible decisions with at most one state per hidden deal.

    Scarce categories claim games first.  Within each category the shuffled
    unit is an eligible decision, not a game, so long games are not silently
    down-weighted before the one-deal independence constraint is applied.
    """

    if queries_per_category <= 0:
        raise ValueError("queries_per_category must be positive")
    if len(trajectories) < queries_per_category * CATEGORY_COUNT:
        raise ValueError("not enough source games for independent category quotas")
    random = np.random.default_rng(seed)
    candidates: dict[int, list[tuple[int, int]]] = {}
    scarcity: list[tuple[int, int]] = []
    for category in range(CATEGORY_COUNT):
        rows: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            steps = np.flatnonzero(
                (trajectory.sources == int(ReplaySource.CURRENT))
                & (trajectory.categories == category)
                & (trajectory.legal_counts >= 2)
            )
            rows.extend((trajectory_index, int(step)) for step in steps)
        if not rows:
            raise RuntimeError(f"no eligible states for {CATEGORY_NAMES[category]}")
        order = random.permutation(len(rows))
        candidates[category] = [rows[int(index)] for index in order]
        unique_games = len({trajectory_index for trajectory_index, _step in rows})
        scarcity.append((unique_games, category))

    used_games: set[int] = set()
    selected: dict[int, list[tuple[int, int]]] = {
        category: [] for category in range(CATEGORY_COUNT)
    }
    for _available, category in sorted(scarcity):
        for trajectory_index, step in candidates[category]:
            if trajectory_index in used_games:
                continue
            selected[category].append((trajectory_index, step))
            used_games.add(trajectory_index)
            if len(selected[category]) == queries_per_category:
                break
        if len(selected[category]) != queries_per_category:
            raise RuntimeError(
                f"independent quota for {CATEGORY_NAMES[category]} is short: "
                f"{len(selected[category])}/{queries_per_category}"
            )

    queries: list[PolicyQuery] = []
    # Interleave categories so cache shards and progress samples are mixed.
    for position in range(queries_per_category):
        for category in range(CATEGORY_COUNT):
            trajectory_index, step = selected[category][position]
            queries.append(
                PolicyQuery(
                    query_id=len(queries),
                    trajectory_index=trajectory_index,
                    trajectory=trajectories[trajectory_index],
                    step=step,
                    category=category,
                )
            )
    return queries


def nested_category_indices(
    categories: np.ndarray, sizes: Sequence[int]
) -> dict[int, np.ndarray]:
    categories = np.asarray(categories)
    normalized = tuple(int(value) for value in sizes)
    if categories.ndim != 1 or not len(categories):
        raise ValueError("categories must be a non-empty vector")
    if not normalized or tuple(sorted(set(normalized))) != normalized:
        raise ValueError("nested sizes must be unique and increasing")
    result: dict[int, np.ndarray] = {}
    per_category = {
        category: np.flatnonzero(categories == category)
        for category in range(CATEGORY_COUNT)
    }
    if any(len(rows) < normalized[-1] for rows in per_category.values()):
        raise ValueError("corpus cannot cover the largest nested size")
    for size in normalized:
        rows = np.stack(
            [per_category[category][:size] for category in range(CATEGORY_COUNT)],
            axis=1,
        ).reshape(-1)
        result[size] = np.ascontiguousarray(rows, dtype=np.int64)
    return result


def _capture_query(
    query: PolicyQuery, *, history: int
) -> tuple[Any, tuple[np.ndarray, ...], np.ndarray]:
    trajectory = query.trajectory
    source = bm.Batch(1, seed=int(trajectory.seed))
    records = np.empty((1, int(bm.STEP_RECORD_WIDTH)), dtype=np.int64)
    action = np.empty(1, dtype=np.uint8)
    for previous in trajectory.actions[: query.step]:
        action[0] = previous
        source.step_into(action, records)
        if records[0, 11]:
            raise RuntimeError("trajectory terminated before a selected query")
    buffers = EngineBuffers.for_batch(source, history)
    buffers.observe()
    actor = int(trajectory.actors[query.step])
    if (
        int(buffers.meta[0, 1]) != actor
        or int(trajectory.categories[query.step]) != query.category
        or int(decision_categories(buffers.meta)[0]) != query.category
    ):
        raise RuntimeError("reconstructed query metadata does not match trajectory")
    legal = buffers.legal[0].copy()
    if int(legal.sum()) != int(trajectory.legal_counts[query.step]):
        raise RuntimeError("reconstructed query legal count does not match")
    behavior = int(trajectory.actions[query.step])
    if legal.sum() < 2 or not legal[behavior]:
        raise RuntimeError("selected query is not a valid multi-action decision")
    state = (
        buffers.tile_obs[0].copy(),
        buffers.melds[0].copy(),
        buffers.meta[0].copy(),
        buffers.events[0].copy(),
        np.asarray(buffers.event_lengths[0], dtype=np.uint16),
    )
    return source, state, legal


def _capture_query_group(
    queries: Sequence[PolicyQuery], *, history: int
) -> tuple[Any, list[tuple[np.ndarray, ...]], list[np.ndarray]]:
    """Reconstruct ragged trajectory positions in one engine batch."""

    if not queries:
        raise ValueError("query group must not be empty")
    count = len(queries)
    source = bm.Batch(count, seed=0)
    indices = np.arange(count, dtype=np.uint32)
    seeds = np.asarray(
        [int(query.trajectory.seed) for query in queries], dtype=np.uint64
    )
    source.reset_many(indices, seeds)
    records = np.empty((count, int(bm.STEP_RECORD_WIDTH)), dtype=np.int64)
    actions = np.zeros(count, dtype=np.uint8)
    maximum_step = max(query.step for query in queries)
    for step in range(maximum_step):
        enabled = np.asarray(
            [step < query.step for query in queries], dtype=np.uint8
        )
        active_rows = np.flatnonzero(enabled)
        for row in active_rows:
            actions[row] = queries[int(row)].trajectory.actions[step]
        source.step_masked_into(enabled, actions, records)
        if np.any(records[active_rows, 11]):
            raise RuntimeError("trajectory terminated before a selected query")

    buffers = EngineBuffers.for_batch(source, history)
    buffers.observe()
    states: list[tuple[np.ndarray, ...]] = []
    legal_rows: list[np.ndarray] = []
    categories = decision_categories(buffers.meta)
    for row, query in enumerate(queries):
        trajectory = query.trajectory
        actor = int(trajectory.actors[query.step])
        if (
            int(buffers.meta[row, 1]) != actor
            or int(trajectory.categories[query.step]) != query.category
            or int(categories[row]) != query.category
        ):
            raise RuntimeError("reconstructed query metadata does not match trajectory")
        legal = buffers.legal[row].copy()
        if int(legal.sum()) != int(trajectory.legal_counts[query.step]):
            raise RuntimeError("reconstructed query legal count does not match")
        behavior = int(trajectory.actions[query.step])
        if legal.sum() < 2 or not legal[behavior]:
            raise RuntimeError("selected query is not a valid multi-action decision")
        states.append(
            (
                buffers.tile_obs[row].copy(),
                buffers.melds[row].copy(),
                buffers.meta[row].copy(),
                buffers.events[row].copy(),
                np.asarray(buffers.event_lengths[row], dtype=np.uint16),
            )
        )
        legal_rows.append(legal)
    return source, states, legal_rows


def _live_wall_worlds(source: Any, seeds: np.ndarray) -> Any:
    return _live_wall_world_group(
        source, np.ascontiguousarray(seeds, dtype=np.uint64)[None, :]
    )


def _live_wall_world_group(source: Any, seeds: np.ndarray) -> Any:
    seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
    if seeds.ndim != 2 or not seeds.shape[0] or not seeds.shape[1]:
        raise ValueError("world seeds must have shape [queries, worlds]")
    if len(source) != seeds.shape[0]:
        raise ValueError("source batch and world-seed queries do not match")
    source_indices = np.repeat(
        np.arange(len(source), dtype=np.uint32), seeds.shape[1]
    )
    worlds = source.resample_live_walls(source_indices, seeds.reshape(-1))
    planes = int(bm.ORACLE_TILE_COUNT_PLANES)
    tiles = int(bm.TILE_KIND_COUNT)
    source_oracle = np.empty((len(source), planes, tiles), dtype=np.uint8)
    world_oracle = np.empty((len(worlds), planes, tiles), dtype=np.uint8)
    source.oracle_tile_counts_into(source_oracle)
    worlds.oracle_tile_counts_into(world_oracle)
    if not np.all(world_oracle == source_oracle[source_indices]):
        raise RuntimeError("live-wall resampling changed a current hidden hand")
    return worlds


def estimate_counterfactual_batch(
    queries: Sequence[PolicyQuery],
    actor: BloodFlowTransformer,
    device: torch.device,
    *,
    worlds: int,
    world_chunk: int,
    seed: int,
    query_batch_size: int = 64,
    inference_batch_size: int = 512,
    on_progress: TargetProgress | None = None,
) -> tuple[CounterfactualBatch, dict[str, object]]:
    if (
        not queries
        or worlds < 2
        or world_chunk <= 0
        or query_batch_size <= 0
        or inference_batch_size <= 0
    ):
        raise ValueError("queries and rollout batch sizes must be valid")
    query_ids: list[int] = []
    tile_rows: list[np.ndarray] = []
    meld_rows: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    length_rows: list[np.uint16] = []
    legal_rows: list[np.ndarray] = []
    rank_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    category_rows: list[int] = []
    rollout_states = 0
    rollout_seconds = 0.0
    paired_standard_errors: list[float] = []
    inference_stager = (
        _PinnedPolicyStager(device, actor.config.max_history)
        if device.type == "cuda"
        else None
    )

    actor.eval()
    for group_start in range(0, len(queries), query_batch_size):
        group = queries[group_start : group_start + query_batch_size]
        source, states, group_legal = _capture_query_group(
            group, history=actor.config.max_history
        )
        action_sets = [np.flatnonzero(legal).astype(np.uint8) for legal in group_legal]
        focal_seats = np.asarray(
            [query.trajectory.actors[query.step] for query in group], dtype=np.int64
        )
        lineups = np.stack(
            [
                infer_policy_lineup(query.trajectory, int(focal_seat))
                for query, focal_seat in zip(group, focal_seats)
            ]
        )
        seed_matrix = np.stack(
            [world_seeds(seed, query.query_id, worlds) for query in group]
        )

        def rollout_progress(fields: Mapping[str, object]) -> None:
            if on_progress is None:
                return
            group_states = int(fields["group_rollout_states"])
            group_seconds = float(fields["group_elapsed_seconds"])
            on_progress(
                group_start,
                {
                    "query_batch": len(group),
                    "rollout_step": int(fields["rollout_step"]),
                    "active_branches": int(fields["active_branches"]),
                    "world_chunk": fields.get("world_chunk"),
                    "rollout_states": rollout_states + group_states,
                    "rollout_states_per_second": (
                        rollout_states + group_states
                    )
                    / max(rollout_seconds + group_seconds, 1e-9),
                },
            )

        grouped_result = rollout_query_group_chunked(
            _live_wall_world_group(source, seed_matrix),
            action_sets,
            actor,
            device,
            focal_seats=focal_seats,
            lineups=lineups,
            world_chunk=world_chunk,
            inference_batch_size=inference_batch_size,
            on_progress=rollout_progress if on_progress is not None else None,
            inference_stager=inference_stager,
        )
        rollout_seconds += grouped_result.elapsed_seconds
        for local, (query, state, legal, actions, result) in enumerate(
            zip(group, states, group_legal, action_sets, grouped_result.queries)
        ):
            rank_q = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            score_q = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            action_indices = actions.astype(np.int64)
            rank_q[action_indices] = result.rank_utility.mean(axis=1)
            score_q[action_indices] = result.score_delta.mean(axis=1)
            behavior = int(query.trajectory.actions[query.step])
            behavior_index = int(np.flatnonzero(actions == behavior)[0])
            best_index = int(np.argmax(rank_q[action_indices]))
            paired = (
                result.rank_utility[best_index].astype(np.float64)
                - result.rank_utility[behavior_index].astype(np.float64)
            )
            paired_standard_errors.append(
                float(paired.std(ddof=1) / math.sqrt(worlds))
            )
            rollout_states += result.rollout_states

            tile, melds, meta, events, length = state
            query_ids.append(query.query_id)
            tile_rows.append(tile)
            meld_rows.append(melds)
            meta_rows.append(meta)
            event_rows.append(events)
            length_rows.append(length)
            legal_rows.append(legal)
            rank_rows.append(rank_q)
            score_rows.append(score_q)
            action_rows.append(behavior)
            category_rows.append(query.category)
        if on_progress is not None:
            on_progress(
                group_start + len(group),
                {
                    "query_batch": len(group),
                    "rollout_step": None,
                    "active_branches": 0,
                    "world_chunk": None,
                    "rollout_states": rollout_states,
                    "rollout_states_per_second": rollout_states
                    / max(rollout_seconds, 1e-9),
                },
            )

    legal_matrix = np.stack(legal_rows)
    rank_matrix = np.stack(rank_rows)
    score_matrix = np.stack(score_rows)
    batch = CounterfactualBatch(
        query_ids=np.asarray(query_ids, dtype=np.int64),
        tile_obs=np.stack(tile_rows),
        melds=np.stack(meld_rows),
        meta=np.stack(meta_rows),
        events=np.stack(event_rows),
        event_lengths=np.asarray(length_rows, dtype=np.uint16),
        legal=legal_matrix,
        categories=np.asarray(category_rows, dtype=np.uint8),
        rank_q=rank_matrix,
        score_q=score_matrix,
        centered_rank_q=center_legal_values(rank_matrix, legal_matrix),
        behavior_actions=np.asarray(action_rows, dtype=np.uint8),
    )
    rows = np.arange(len(batch))
    behavior_q = batch.rank_q[rows, batch.behavior_actions.astype(np.int64)]
    best_q = np.where(batch.legal, batch.rank_q, -np.inf).max(axis=1)
    return batch, {
        "states": len(batch),
        "worlds": worlds,
        "rollout_states": rollout_states,
        "rollout_seconds": rollout_seconds,
        "rollout_states_per_second": rollout_states / max(rollout_seconds, 1e-9),
        "mean_legal_actions": float(batch.legal.sum(axis=1).mean()),
        "mean_best_rank_gain": float(np.mean(best_q - behavior_q)),
        "mean_paired_standard_error": float(np.mean(paired_standard_errors)),
    }


def build_state_batch(queries: Sequence[PolicyQuery], *, history: int) -> PolicyStateBatch:
    if not queries:
        raise ValueError("state queries must not be empty")
    query_ids: list[int] = []
    tile_rows: list[np.ndarray] = []
    meld_rows: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    length_rows: list[np.uint16] = []
    legal_rows: list[np.ndarray] = []
    categories: list[int] = []
    for query in queries:
        _source, state, legal = _capture_query(query, history=history)
        tile, melds, meta, events, length = state
        query_ids.append(query.query_id)
        tile_rows.append(tile)
        meld_rows.append(melds)
        meta_rows.append(meta)
        event_rows.append(events)
        length_rows.append(length)
        legal_rows.append(legal)
        categories.append(query.category)
    return PolicyStateBatch(
        query_ids=np.asarray(query_ids, dtype=np.int64),
        tile_obs=np.stack(tile_rows),
        melds=np.stack(meld_rows),
        meta=np.stack(meta_rows),
        events=np.stack(event_rows),
        event_lengths=np.asarray(length_rows, dtype=np.uint16),
        legal=np.stack(legal_rows),
        categories=np.asarray(categories, dtype=np.uint8),
    )


def concatenate_counterfactual_batches(
    batches: Sequence[CounterfactualBatch],
) -> CounterfactualBatch:
    if not batches:
        raise ValueError("at least one counterfactual batch is required")
    fields = CounterfactualBatch.__dataclass_fields__
    values = {
        name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
        for name in fields
    }
    return CounterfactualBatch(**values)


def subset_counterfactual_batch(
    batch: CounterfactualBatch, indices: np.ndarray
) -> CounterfactualBatch:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("subset indices must be a non-empty vector")
    values = {
        name: getattr(batch, name)[indices]
        for name in CounterfactualBatch.__dataclass_fields__
    }
    return CounterfactualBatch(**values)


def save_counterfactual_batch(path: Path, batch: CounterfactualBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **{
            "version": np.asarray([TARGET_CACHE_VERSION], dtype=np.int64),
            **{
                name: getattr(batch, name)
                for name in CounterfactualBatch.__dataclass_fields__
            },
        })
    temporary.replace(path)


def save_policy_state_batch(path: Path, batch: PolicyStateBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            version=np.asarray([STATE_CACHE_VERSION], dtype=np.int64),
            **{
                name: getattr(batch, name)
                for name in PolicyStateBatch.__dataclass_fields__
            },
        )
    temporary.replace(path)


def load_policy_state_batch(path: Path) -> PolicyStateBatch:
    expected = {"version", *PolicyStateBatch.__dataclass_fields__}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("policy-state cache fields do not match")
        if int(payload["version"][0]) != STATE_CACHE_VERSION:
            raise ValueError("unsupported policy-state cache version")
        return PolicyStateBatch(
            **{
                name: payload[name].copy()
                for name in PolicyStateBatch.__dataclass_fields__
            }
        )


def load_counterfactual_batch(path: Path) -> CounterfactualBatch:
    expected = {"version", *CounterfactualBatch.__dataclass_fields__}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("counterfactual cache fields do not match")
        if int(payload["version"][0]) != TARGET_CACHE_VERSION:
            raise ValueError("unsupported counterfactual cache version")
        return CounterfactualBatch(
            **{
                name: payload[name].copy()
                for name in CounterfactualBatch.__dataclass_fields__
            }
        )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def query_signature(queries: Sequence[PolicyQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        trajectory = query.trajectory
        digest.update(
            np.asarray(
                [
                    query.query_id,
                    trajectory.seed,
                    trajectory.exchange_direction,
                    query.step,
                    query.category,
                    trajectory.termination_reason,
                ],
                dtype="<u8",
            ).tobytes()
        )
        # The controller lineup is inferred from the complete source trajectory, so
        # hash every compact field rather than only seed/step identifiers.
        for value in (
            trajectory.actions,
            trajectory.actors,
            trajectory.phases,
            trajectory.categories,
            trajectory.sources,
            trajectory.legal_counts,
            trajectory.terminal_scores,
            trajectory.terminal_ranks,
        ):
            contiguous = np.ascontiguousarray(value)
            digest.update(np.asarray([contiguous.nbytes], dtype="<u8").tobytes())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def cached_counterfactual_corpus(
    directory: Path,
    queries: Sequence[PolicyQuery],
    actor: BloodFlowTransformer,
    device: torch.device,
    *,
    fingerprint: str,
    worlds: int,
    world_chunk: int,
    world_seed: int,
    shard_size: int,
    query_batch_size: int = 64,
    inference_batch_size: int = 512,
    on_progress: TargetProgress | None = None,
) -> tuple[CounterfactualBatch, dict[str, object]]:
    """Build or resume immutable target shards for one frozen policy version."""

    if shard_size <= 0 or not fingerprint:
        raise ValueError("target cache needs a shard size and fingerprint")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    expected = {
        "version": TARGET_CACHE_VERSION,
        "fingerprint": fingerprint,
        "query_signature": query_signature(queries),
        "queries": len(queries),
        "worlds": worlds,
        "world_chunk": world_chunk,
        "world_seed": int(world_seed),
        "shard_size": shard_size,
        "query_batch_size": query_batch_size,
        "inference_batch_size": inference_batch_size,
    }
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text())
        if actual != expected:
            raise ValueError("target cache manifest does not match this run")
    else:
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        entries = list(directory.iterdir())
        if entries == [temporary]:
            temporary.unlink()
        elif entries:
            raise ValueError("target cache directory is non-empty without a manifest")
        _atomic_json(manifest_path, expected)

    batches: list[CounterfactualBatch] = []
    rollout_states = 0
    rollout_seconds = 0.0
    for start in range(0, len(queries), shard_size):
        stop = min(start + shard_size, len(queries))
        path = directory / f"targets-{start:06d}-{stop:06d}.npz"
        expected_ids = np.asarray(
            [query.query_id for query in queries[start:stop]], dtype=np.int64
        )
        if path.exists():
            batch = load_counterfactual_batch(path)
            if not np.array_equal(batch.query_ids, expected_ids):
                raise ValueError(f"cached target shard {path} has wrong query ids")
            if on_progress is not None:
                on_progress(stop, {"cached": True})
        else:
            def shard_progress(done: int, fields: Mapping[str, object]) -> None:
                if on_progress is not None:
                    on_progress(start + done, fields)

            batch, metrics = estimate_counterfactual_batch(
                queries[start:stop],
                actor,
                device,
                worlds=worlds,
                world_chunk=world_chunk,
                seed=world_seed,
                query_batch_size=query_batch_size,
                inference_batch_size=inference_batch_size,
                on_progress=shard_progress,
            )
            save_counterfactual_batch(path, batch)
            rollout_states += int(metrics["rollout_states"])
            rollout_seconds += float(metrics["rollout_seconds"])
        batches.append(batch)
    result = concatenate_counterfactual_batches(batches)
    if not np.array_equal(
        result.query_ids,
        np.asarray([query.query_id for query in queries], dtype=np.int64),
    ):
        raise RuntimeError("target shards do not reconstruct the query order")
    return result, {
        "states": len(result),
        "worlds": worlds,
        "new_rollout_states": rollout_states,
        "new_rollout_seconds": rollout_seconds,
        "new_rollout_states_per_second": rollout_states
        / max(rollout_seconds, 1e-9),
    }


def _state_tensors(
    batch: PolicyStateBatch,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, ...]:
    lengths_np = batch.event_lengths[indices].astype(np.int64, copy=False)
    width = bucket_history_width(lengths_np, batch.events.shape[1])
    return (
        torch.as_tensor(batch.tile_obs[indices], device=device),
        torch.as_tensor(batch.melds[indices], device=device),
        torch.as_tensor(batch.meta[indices], device=device),
        torch.as_tensor(batch.events[indices, :width], device=device),
        torch.as_tensor(lengths_np, device=device),
    )


def _device_state_batch(
    batch: PolicyStateBatch,
    device: torch.device,
) -> tuple[Tensor, ...]:
    """Stage immutable policy inputs once for repeated sliced forwards."""

    lengths = batch.event_lengths.astype(np.int64, copy=False)
    return (
        torch.as_tensor(batch.tile_obs, device=device),
        torch.as_tensor(batch.melds, device=device),
        torch.as_tensor(batch.meta, device=device),
        torch.as_tensor(batch.events, device=device),
        torch.as_tensor(lengths, device=device),
    )


def _slice_device_state(
    tensors: tuple[Tensor, ...],
    start: int,
    stop: int,
    width: int,
) -> tuple[Tensor, ...]:
    tile_obs, melds, meta, events, lengths = tensors
    return (
        tile_obs[start:stop],
        melds[start:stop],
        meta[start:stop],
        events[start:stop, :width],
        lengths[start:stop],
    )


def category_row_weights(
    categories: np.ndarray, category_weights: np.ndarray
) -> np.ndarray:
    categories = np.asarray(categories)
    weights = np.asarray(category_weights, dtype=np.float64)
    if categories.ndim != 1 or not len(categories):
        raise ValueError("categories must be a non-empty vector")
    if weights.shape != (CATEGORY_COUNT,) or np.any(weights < 0):
        raise ValueError("category weights must be non-negative and complete")
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("category weights must be finite and sum to one")
    counts = np.bincount(categories.astype(np.int64), minlength=CATEGORY_COUNT)
    if np.any((weights > 0) & (counts == 0)):
        raise ValueError("a positive-weight category has no rows")
    result = weights[categories.astype(np.int64)] / counts[categories.astype(np.int64)]
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError("row weights do not sum to one")
    return result


def policy_direction_row_loss(
    logits: Tensor,
    legal: Tensor,
    centered_q: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    legal = legal.bool()
    log_probs = F.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=-1)
    probabilities = log_probs.exp()
    safe_log = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    q = torch.where(legal, centered_q.float(), torch.zeros_like(centered_q.float()))
    expected_q = (probabilities * q).sum(dim=-1)
    entropy = -(probabilities * safe_log).sum(dim=-1)
    return -expected_q, expected_q, entropy


def cpu_model_state(model: BloodFlowTransformer) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        detached = value.detach()
        result[name] = (
            detached.clone() if detached.device.type == "cpu" else detached.cpu()
        )
    return result


def one_step_direction(
    reference: BloodFlowTransformer,
    batch: CounterfactualBatch,
    device: torch.device,
    *,
    category_weights: np.ndarray,
    learning_rate: float,
    microbatch_size: int,
    on_progress: TargetProgress | None = None,
) -> tuple[BloodFlowTransformer, dict[str, Tensor], dict[str, Tensor], dict[str, object]]:
    if learning_rate <= 0 or microbatch_size <= 0:
        raise ValueError("direction optimizer arguments must be positive")
    actor = copy.deepcopy(reference).to(device)
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    initial = cpu_model_state(actor)
    row_weights = category_row_weights(batch.categories, category_weights)
    weight_tensor = torch.as_tensor(row_weights, device=device, dtype=torch.float32)
    device_states = _device_state_batch(batch, device)
    legal_tensor = torch.as_tensor(batch.legal, device=device)
    q_tensor = torch.as_tensor(batch.centered_rank_q, device=device)
    parameters = list(actor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    microbatches = math.ceil(len(batch) / microbatch_size)
    actor.train()
    for index, start in enumerate(range(0, len(batch), microbatch_size), start=1):
        stop = min(start + microbatch_size, len(batch))
        width = bucket_history_width(
            batch.event_lengths[start:stop], batch.events.shape[1]
        )
        state = _slice_device_state(device_states, start, stop, width)
        legal = legal_tensor[start:stop]
        q = q_tensor[start:stop]
        with _autocast(device):
            output = actor(*state, legal)
            row_loss, expected_q, entropy = policy_direction_row_loss(
                output.raw_logits,
                legal,
                q,
            )
            weights = weight_tensor[start:stop]
            loss = (row_loss * weights).sum()
        loss.backward()
        totals.add_(
            torch.stack(
                [
                    (value.detach() * weights).sum().double()
                    for value in (row_loss, expected_q, entropy)
                ]
            )
        )
        if on_progress is not None:
            on_progress(index, {"microbatches": microbatches})
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    metric_values = torch.cat((totals, gradient_norm.detach().double()[None])).cpu()
    candidate = cpu_model_state(actor)
    return actor, initial, candidate, {
        "optimizer_steps": 1,
        "states": len(batch),
        "microbatches": microbatches,
        "loss": float(metric_values[0]),
        "expected_rank_q": float(metric_values[1]),
        "entropy": float(metric_values[2]),
        "gradient_norm": float(metric_values[3]),
        "effective_sample_size": float(1.0 / np.square(row_weights).sum()),
        "row_weight_sum": float(row_weights.sum()),
    }


def load_scaled_direction(
    actor: BloodFlowTransformer,
    initial: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    scale: float,
) -> None:
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("direction scale must be finite and non-negative")
    state: dict[str, Tensor] = {}
    for name, start in initial.items():
        end = candidate[name]
        if start.is_floating_point():
            state[name] = start + (end - start) * scale
        elif torch.equal(start, end):
            state[name] = start
        else:
            raise ValueError(f"non-floating state {name} changed")
    actor.load_state_dict(state, strict=True)


def _prepare_calibration_chunks(
    reference: BloodFlowTransformer,
    states: PolicyStateBatch,
    row_weights: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> list[_CalibrationChunk]:
    device_states = _device_state_batch(states, device)
    legal = torch.as_tensor(states.legal, device=device)
    weights = torch.as_tensor(row_weights, device=device, dtype=torch.float32)
    chunks: list[_CalibrationChunk] = []
    reference.eval()
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        width = bucket_history_width(
            states.event_lengths[start:stop], states.events.shape[1]
        )
        state = _slice_device_state(device_states, start, stop, width)
        chunk_legal = legal[start:stop]
        with torch.inference_mode(), _autocast(device):
            reference_logits = reference(
                *state, chunk_legal
            ).raw_logits.float()
        masked_reference = reference_logits.masked_fill(
            ~chunk_legal, -torch.inf
        )
        chunks.append(
            _CalibrationChunk(
                state=state,
                legal=chunk_legal,
                weights=weights[start:stop],
                reference_log_probs=F.log_softmax(
                    masked_reference, dim=-1
                ),
                reference_actions=masked_reference.argmax(dim=-1),
            )
        )
    return chunks


def _device_model_state(
    state: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device=device, copy=True)
        for name, value in state.items()
    }


def _load_scaled_device_direction(
    actor: BloodFlowTransformer,
    initial: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    scale: float,
) -> None:
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("direction scale must be finite and non-negative")
    target_state = actor.state_dict()
    if set(target_state) != set(initial) or set(initial) != set(candidate):
        raise ValueError("direction state keys do not match the Actor")
    with torch.no_grad():
        for name, target in target_state.items():
            start = initial[name]
            end = candidate[name]
            if start.is_floating_point():
                torch.lerp(start, end, scale, out=target)
            elif torch.equal(start, end):
                target.copy_(start)
            else:
                raise ValueError(f"non-floating state {name} changed")


def _mean_reverse_kl(
    actor: BloodFlowTransformer,
    chunks: Sequence[_CalibrationChunk],
    device: torch.device,
) -> float:
    total = torch.zeros((), device=device, dtype=torch.float64)
    actor.eval()
    for chunk in chunks:
        with torch.inference_mode(), _autocast(device):
            logits = actor(
                *chunk.state, chunk.legal
            ).raw_logits.float()
        log_probs = F.log_softmax(
            logits.masked_fill(~chunk.legal, -torch.inf), dim=-1
        )
        probabilities = log_probs.exp()
        difference = torch.where(
            chunk.legal,
            log_probs - chunk.reference_log_probs,
            torch.zeros_like(log_probs),
        )
        kl = (probabilities * difference).sum(dim=-1)
        total.add_((kl * chunk.weights).sum().double())
    return float(total.item())


def _greedy_change_metrics(
    actor: BloodFlowTransformer,
    chunks: Sequence[_CalibrationChunk],
    device: torch.device,
) -> tuple[float, float]:
    weighted = torch.zeros((), device=device, dtype=torch.float64)
    changed_states = torch.zeros((), device=device, dtype=torch.int64)
    actor.eval()
    for chunk in chunks:
        with torch.inference_mode(), _autocast(device):
            actions = actor(
                *chunk.state, chunk.legal
            ).logits.argmax(dim=-1)
        changed = actions != chunk.reference_actions
        weighted.add_((changed.float() * chunk.weights).sum().double())
        changed_states.add_(changed.sum())
    values = torch.stack((weighted, changed_states.double())).cpu()
    total_states = sum(len(chunk.legal) for chunk in chunks)
    return float(values[0]), float(values[1]) / total_states


def calibrate_direction(
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    initial: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    states: PolicyStateBatch,
    device: torch.device,
    *,
    category_weights: np.ndarray,
    target_kl: float,
    batch_size: int,
    search_steps: int,
    maximum_scale: float,
    on_progress: TargetProgress | None = None,
) -> dict[str, float]:
    if target_kl <= 0 or batch_size <= 0 or search_steps <= 0 or maximum_scale < 1:
        raise ValueError("KL calibration arguments are invalid")
    row_weights = category_row_weights(states.categories, category_weights)
    chunks = _prepare_calibration_chunks(
        reference,
        states,
        row_weights,
        device,
        batch_size=batch_size,
    )
    device_initial = _device_model_state(initial, device)
    device_candidate = _device_model_state(candidate, device)
    evaluations = 0

    def evaluate(scale: float) -> float:
        nonlocal evaluations
        _load_scaled_device_direction(
            actor, device_initial, device_candidate, scale
        )
        result = _mean_reverse_kl(actor, chunks, device)
        evaluations += 1
        if on_progress is not None:
            on_progress(evaluations, {"scale": scale, "kl": result})
        if not math.isfinite(result) or result < -1e-8:
            raise RuntimeError("direction produced an invalid calibration KL")
        return max(result, 0.0)

    zero = evaluate(0.0)
    if abs(zero) > max(1e-8, target_kl * 1e-3):
        raise RuntimeError(f"zero direction has KL {zero}")
    low, low_kl = 0.0, zero
    high, high_kl = 1.0, evaluate(1.0)
    candidate_kl = high_kl
    while high_kl < target_kl:
        low, low_kl = high, high_kl
        high *= 2.0
        if high > maximum_scale:
            raise RuntimeError("direction cannot reach the target KL")
        next_kl = evaluate(high)
        if next_kl + target_kl * 1e-3 < high_kl:
            raise RuntimeError("calibration KL is not monotone")
        high_kl = next_kl
    allowed_shortfall = max(
        2e-7, target_kl * KL_TARGET_RELATIVE_TOLERANCE
    )
    for _ in range(search_steps):
        middle = 0.5 * (low + high)
        middle_kl = evaluate(middle)
        if middle_kl < target_kl:
            low, low_kl = middle, middle_kl
            if target_kl - middle_kl <= allowed_shortfall:
                break
        else:
            high, high_kl = middle, middle_kl
    # Keep the committed policy inside the trust region. BF16 inference makes
    # KL piecewise constant on very small calibration sets, so the nearest
    # point above the target is not an acceptable substitute.
    scale, final_kl = low, low_kl
    _load_scaled_device_direction(
        actor, device_initial, device_candidate, scale
    )
    if target_kl - final_kl > allowed_shortfall:
        raise RuntimeError(
            "KL calibration did not meet its tolerance: "
            f"target={target_kl:.9g}, final={final_kl:.9g}, "
            f"error={abs(final_kl - target_kl):.9g}, scale={scale:.9g}"
        )
    weighted_flips, equal_flips = _greedy_change_metrics(actor, chunks, device)
    return {
        "scale": scale,
        "candidate_kl": candidate_kl,
        "final_kl": final_kl,
        "target_kl": target_kl,
        "absolute_error": abs(final_kl - target_kl),
        "relative_shortfall": (target_kl - final_kl) / target_kl,
        "greedy_flip_rate": weighted_flips,
        "equal_state_greedy_flip_rate": equal_flips,
        "evaluations": float(evaluations),
    }


def direction_cosine(
    initial: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    maximum: Mapping[str, Tensor],
) -> dict[str, float]:
    dot = norm = maximum_norm = 0.0
    for name, start in initial.items():
        if not start.is_floating_point():
            continue
        current = (candidate[name] - start).double().reshape(-1)
        reference = (maximum[name] - start).double().reshape(-1)
        dot += float(torch.dot(current, reference))
        norm += float(torch.dot(current, current))
        maximum_norm += float(torch.dot(reference, reference))
    if norm <= 0 or maximum_norm <= 0:
        raise RuntimeError("optimizer produced a zero direction")
    return {
        "cosine_to_maximum": dot / math.sqrt(norm * maximum_norm),
        "direction_l2": math.sqrt(norm),
        "maximum_direction_l2": math.sqrt(maximum_norm),
    }


def heldout_policy_value(
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    batch: CounterfactualBatch,
    device: torch.device,
    *,
    category_weights: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    row_weights = category_row_weights(batch.categories, category_weights)
    rank_values = np.empty(len(batch), dtype=np.float64)
    score_values = np.empty(len(batch), dtype=np.float64)
    kl_values = np.empty(len(batch), dtype=np.float64)
    actor.eval()
    reference.eval()
    for start in range(0, len(batch), batch_size):
        stop = min(start + batch_size, len(batch))
        rows = np.arange(start, stop)
        state = _state_tensors(batch, rows, device)
        legal = torch.as_tensor(batch.legal[rows], device=device)
        with torch.inference_mode(), _autocast(device):
            logits = actor(*state, legal).raw_logits.float()
            reference_logits = reference(*state, legal).raw_logits.float()
        log_probs = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
        reference_log = F.log_softmax(
            reference_logits.masked_fill(~legal, -torch.inf), dim=-1
        )
        probabilities = log_probs.exp().cpu().numpy()
        reference_probabilities = reference_log.exp().cpu().numpy()
        delta = probabilities - reference_probabilities
        rank_values[start:stop] = (delta * batch.rank_q[rows]).sum(axis=1)
        score_values[start:stop] = (delta * batch.score_q[rows]).sum(axis=1)
        safe_difference = torch.where(
            legal, log_probs - reference_log, torch.zeros_like(log_probs)
        )
        kl_values[start:stop] = (
            log_probs.exp() * safe_difference
        ).sum(dim=-1).cpu().numpy()
    return {
        "states": len(batch),
        "visitation_weighted_rank_value": float(np.dot(row_weights, rank_values)),
        "visitation_weighted_score_value": float(np.dot(row_weights, score_values)),
        "visitation_weighted_kl": float(np.dot(row_weights, kl_values)),
        "equal_state_rank_value": float(rank_values.mean()),
        "effective_sample_size": float(1.0 / np.square(row_weights).sum()),
        "categories": {
            CATEGORY_NAMES[category]: {
                "states": int(np.sum(batch.categories == category)),
                "mean_rank_value": float(
                    rank_values[batch.categories == category].mean()
                ),
            }
            for category in range(CATEGORY_COUNT)
        },
    }


__all__ = [
    "CounterfactualBatch",
    "PolicyQuery",
    "PolicyStateBatch",
    "build_state_batch",
    "cached_counterfactual_corpus",
    "calibrate_direction",
    "category_row_weights",
    "center_legal_values",
    "concatenate_counterfactual_batches",
    "cpu_model_state",
    "direction_cosine",
    "domain_seed",
    "estimate_counterfactual_batch",
    "heldout_policy_value",
    "load_counterfactual_batch",
    "load_scaled_direction",
    "mix64",
    "nested_category_indices",
    "one_step_direction",
    "load_policy_state_batch",
    "query_signature",
    "require_cuda",
    "require_deterministic_actor",
    "save_counterfactual_batch",
    "save_policy_state_batch",
    "select_independent_queries",
    "source_visit_frequencies",
    "subset_counterfactual_batch",
    "world_seeds",
]
