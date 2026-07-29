"""Per-world all-action outcome corpora for search-policy experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np
import torch

from .model import ACTION_SPACE_SIZE, BloodFlowTransformer
from .pipeline import _PinnedPolicyStager
from .policy_iteration import (
    PolicyQuery,
    PolicyStateBatch,
    TargetProgress,
    _atomic_json,
    _capture_query_group,
    _sample_world_group,
    center_legal_values,
    query_signature,
    world_seeds,
)
from .search_rollout import infer_policy_lineup, rollout_query_group_chunked


WORLD_OUTCOME_CACHE_VERSION = 1


def _normalize_action_sets(
    action_sets: Sequence[np.ndarray] | None,
    query_count: int,
) -> tuple[np.ndarray, ...] | None:
    if action_sets is None:
        return None
    if len(action_sets) != query_count:
        raise ValueError("action sets must align with queries")
    normalized: list[np.ndarray] = []
    for actions in action_sets:
        values = np.asarray(actions)
        if (
            values.ndim != 1
            or not len(values)
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError("each action set must contain integer actions")
        integers = values.astype(np.int64, copy=False)
        if (
            np.any((integers < 0) | (integers >= ACTION_SPACE_SIZE))
            or len(np.unique(integers)) != len(integers)
        ):
            raise ValueError("each action set must contain unique valid actions")
        normalized.append(np.ascontiguousarray(integers, dtype=np.uint8))
    return tuple(normalized)


def _action_set_signature(action_sets: tuple[np.ndarray, ...] | None) -> str:
    if action_sets is None:
        return "all_legal"
    digest = hashlib.sha256()
    for actions in action_sets:
        digest.update(len(actions).to_bytes(2, "little"))
        digest.update(actions.tobytes())
    return digest.hexdigest()


def _require_action_set_match(
    batch: "WorldOutcomeBatch",
    action_sets: tuple[np.ndarray, ...] | None,
) -> None:
    if action_sets is None:
        if not batch.is_complete:
            raise ValueError("world outcome shard is missing legal actions")
        return
    if len(action_sets) != len(batch):
        raise ValueError("world outcome shard has the wrong action set count")
    expected = np.zeros_like(batch.legal)
    for row, actions in enumerate(action_sets):
        expected[row, actions.astype(np.int64)] = True
    if not np.array_equal(batch.evaluated_actions, expected):
        raise ValueError("world outcome shard evaluates different actions")


@dataclass(frozen=True)
class WorldOutcomeBatch(PolicyStateBatch):
    """Terminal outcomes with shape ``[states, actions, worlds]``.

    A zero rank code denotes a legal action that this corpus did not evaluate.
    Full counterfactual corpora evaluate every legal action. Paired validation
    and audit corpora evaluate only the actions that their statistic reads.
    """

    rank_outcomes: np.ndarray
    score_outcomes: np.ndarray
    behavior_actions: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        size = len(self)
        if (
            self.rank_outcomes.ndim != 3
            or self.rank_outcomes.shape[:2] != (size, ACTION_SPACE_SIZE)
            or self.rank_outcomes.shape[2] < 2
            or self.rank_outcomes.dtype != np.int8
        ):
            raise ValueError("rank outcomes have the wrong shape or dtype")
        if self.score_outcomes.shape != self.rank_outcomes.shape:
            raise ValueError("score outcomes must match rank outcomes")
        if not np.isfinite(self.score_outcomes).all():
            raise ValueError("score outcomes must be finite")
        expanded_legal = np.broadcast_to(
            self.legal[:, :, None], self.rank_outcomes.shape
        )
        evaluated = self.rank_outcomes != 0
        if np.any(evaluated != evaluated[:, :, :1]):
            raise ValueError("an action must be evaluated on every world or none")
        if np.any(evaluated & ~expanded_legal):
            raise ValueError("illegal actions must have zero world outcomes")
        if np.any(self.score_outcomes[~evaluated] != 0):
            raise ValueError("unevaluated actions must have zero score outcomes")
        if not np.isin(self.rank_outcomes[evaluated], (-3, -1, 1, 3)).all():
            raise ValueError("rank outcome codes are invalid")
        if self.behavior_actions.shape != (size,):
            raise ValueError("behavior actions have the wrong shape")
        rows = np.arange(size)
        if np.any(~self.legal[rows, self.behavior_actions.astype(np.int64)]):
            raise ValueError("a behavior action is illegal")

    @property
    def worlds(self) -> int:
        return self.rank_outcomes.shape[2]

    @property
    def evaluated_actions(self) -> np.ndarray:
        """Return the action mask represented by nonzero rank outcome codes."""

        return self.rank_outcomes[:, :, 0] != 0

    @property
    def is_complete(self) -> bool:
        """Whether this corpus evaluates every legal action."""

        return bool(np.array_equal(self.evaluated_actions, self.legal))

    def require_evaluated(self, actions: np.ndarray) -> None:
        """Validate that one action per state is represented by this corpus."""

        actions = np.asarray(actions, dtype=np.int64)
        rows = np.arange(len(self))
        if (
            actions.shape != (len(self),)
            or np.any((actions < 0) | (actions >= ACTION_SPACE_SIZE))
            or np.any(~self.evaluated_actions[rows, actions])
        ):
            raise ValueError("requested actions are not evaluated by this corpus")

    def counterfactual_batch(self):
        from .policy_iteration import CounterfactualBatch

        if not self.is_complete:
            raise ValueError(
                "counterfactual batches require outcomes for every legal action"
            )

        rank_q = self.rank_outcomes.astype(np.float32).mean(axis=2) / 2.0
        score_q = self.score_outcomes.mean(axis=2, dtype=np.float32)
        rank_q = np.where(self.legal, rank_q, 0.0).astype(np.float32)
        score_q = np.where(self.legal, score_q, 0.0).astype(np.float32)
        values = {
            name: getattr(self, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
        return CounterfactualBatch(
            **values,
            rank_q=rank_q,
            score_q=score_q,
            centered_rank_q=center_legal_values(rank_q, self.legal),
            behavior_actions=self.behavior_actions.copy(),
        )


def concatenate_world_outcome_batches(
    batches: Sequence[WorldOutcomeBatch],
) -> WorldOutcomeBatch:
    if not batches or len({batch.worlds for batch in batches}) != 1:
        raise ValueError("world outcome batches must share one positive world count")
    values = {
        name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
        for name in WorldOutcomeBatch.__dataclass_fields__
    }
    return WorldOutcomeBatch(**values)


def subset_world_outcome_batch(
    batch: WorldOutcomeBatch, indices: np.ndarray
) -> WorldOutcomeBatch:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("world outcome subset indices must be non-empty")
    return WorldOutcomeBatch(
        **{
            name: getattr(batch, name)[indices]
            for name in WorldOutcomeBatch.__dataclass_fields__
        }
    )


def combine_world_replicates(
    batches: Sequence[WorldOutcomeBatch],
) -> WorldOutcomeBatch:
    if len(batches) < 2:
        raise ValueError("combining worlds needs at least two replicates")
    reference = batches[0]
    for batch in batches[1:]:
        for name in PolicyStateBatch.__dataclass_fields__:
            if not np.array_equal(getattr(batch, name), getattr(reference, name)):
                raise ValueError(f"world replicates differ in {name}")
        if not np.array_equal(batch.behavior_actions, reference.behavior_actions):
            raise ValueError("world replicates differ in behavior actions")
        if not np.array_equal(batch.evaluated_actions, reference.evaluated_actions):
            raise ValueError("world replicates evaluate different actions")
    state = {
        name: getattr(reference, name)
        for name in PolicyStateBatch.__dataclass_fields__
    }
    return WorldOutcomeBatch(
        **state,
        rank_outcomes=np.concatenate(
            [batch.rank_outcomes for batch in batches], axis=2
        ),
        score_outcomes=np.concatenate(
            [batch.score_outcomes for batch in batches], axis=2
        ),
        behavior_actions=reference.behavior_actions,
    )


def save_world_outcome_batch(path: Path, batch: WorldOutcomeBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            version=np.asarray([WORLD_OUTCOME_CACHE_VERSION], dtype=np.int64),
            **{
                name: getattr(batch, name)
                for name in WorldOutcomeBatch.__dataclass_fields__
            },
        )
    temporary.replace(path)


def load_world_outcome_batch(path: Path) -> WorldOutcomeBatch:
    expected = {"version", *WorldOutcomeBatch.__dataclass_fields__}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("world outcome cache fields do not match")
        if int(payload["version"][0]) != WORLD_OUTCOME_CACHE_VERSION:
            raise ValueError("unsupported world outcome cache version")
        return WorldOutcomeBatch(
            **{
                name: payload[name].copy()
                for name in WorldOutcomeBatch.__dataclass_fields__
            }
        )


def load_world_outcome_corpus(directory: Path) -> WorldOutcomeBatch:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("version", -1)) != WORLD_OUTCOME_CACHE_VERSION:
        raise ValueError("unsupported world outcome corpus version")
    paths = sorted(directory.glob("outcomes-*.npz"))
    if not paths:
        raise ValueError("world outcome corpus has no shards")
    result = concatenate_world_outcome_batches(
        [load_world_outcome_batch(path) for path in paths]
    )
    if (
        len(result) != int(manifest["queries"])
        or result.worlds != int(manifest["worlds"])
    ):
        raise ValueError("world outcome corpus does not match its manifest")
    return result


def estimate_world_outcome_batch(
    queries: Sequence[PolicyQuery],
    actor: BloodFlowTransformer,
    device: torch.device,
    *,
    self_play_actor: BloodFlowTransformer | None = None,
    worlds: int,
    world_chunk: int,
    seed: int,
    world_sampling: str,
    query_batch_size: int = 64,
    inference_batch_size: int = 512,
    action_sets: Sequence[np.ndarray] | None = None,
    on_progress: TargetProgress | None = None,
) -> tuple[WorldOutcomeBatch, dict[str, object]]:
    if (
        not queries
        or worlds < 2
        or world_chunk <= 0
        or query_batch_size <= 0
        or inference_batch_size <= 0
        or world_sampling not in {"live_wall", "information_set"}
    ):
        raise ValueError("world outcome rollout arguments are invalid")
    normalized_action_sets = _normalize_action_sets(action_sets, len(queries))
    rows: dict[str, list[np.ndarray | np.uint16 | int]] = {
        "query_ids": [],
        "tile_obs": [],
        "melds": [],
        "meta": [],
        "events": [],
        "event_lengths": [],
        "legal": [],
        "categories": [],
        "rank_outcomes": [],
        "score_outcomes": [],
        "behavior_actions": [],
    }
    rollout_states = 0
    rollout_seconds = 0.0
    inference_stager = (
        _PinnedPolicyStager(device, actor.config.max_history)
        if device.type == "cuda"
        else None
    )
    actor.eval()
    if self_play_actor is not None:
        if self_play_actor.config != actor.config:
            raise ValueError("self-play Actor config must match the focal Actor")
        self_play_actor.eval()

    for group_start in range(0, len(queries), query_batch_size):
        group = queries[group_start : group_start + query_batch_size]
        source, states, group_legal = _capture_query_group(
            group, history=actor.config.max_history
        )
        group_action_sets = (
            [
                np.flatnonzero(legal).astype(np.uint8)
                for legal in group_legal
            ]
            if normalized_action_sets is None
            else list(
                normalized_action_sets[group_start : group_start + len(group)]
            )
        )
        for legal, actions in zip(group_legal, group_action_sets):
            if not legal[actions.astype(np.int64)].all():
                raise ValueError("an action set contains an illegal action")
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

        grouped = rollout_query_group_chunked(
            _sample_world_group(source, seed_matrix, world_sampling),
            group_action_sets,
            actor,
            device,
            focal_seats=focal_seats,
            lineups=lineups,
            self_play_model=self_play_actor,
            world_chunk=world_chunk,
            inference_batch_size=inference_batch_size,
            on_progress=rollout_progress if on_progress is not None else None,
            inference_stager=inference_stager,
        )
        rollout_seconds += grouped.elapsed_seconds
        for query, state, legal, actions, result in zip(
            group, states, group_legal, group_action_sets, grouped.queries
        ):
            rank = np.zeros((ACTION_SPACE_SIZE, worlds), dtype=np.int8)
            score = np.zeros((ACTION_SPACE_SIZE, worlds), dtype=np.float32)
            codes = np.rint(result.rank_utility * 2).astype(np.int8)
            if not np.isin(codes, (-3, -1, 1, 3)).all():
                raise RuntimeError("rollout returned a non-rank utility")
            action_indices = actions.astype(np.int64)
            rank[action_indices] = codes
            score[action_indices] = result.score_delta
            tile, melds, meta, events, length = state
            rows["query_ids"].append(int(query.query_id))
            rows["tile_obs"].append(tile)
            rows["melds"].append(melds)
            rows["meta"].append(meta)
            rows["events"].append(events)
            rows["event_lengths"].append(length)
            rows["legal"].append(legal)
            rows["categories"].append(int(query.category))
            rows["rank_outcomes"].append(rank)
            rows["score_outcomes"].append(score)
            rows["behavior_actions"].append(
                int(query.trajectory.actions[query.step])
            )
            rollout_states += result.rollout_states
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

    batch = WorldOutcomeBatch(
        query_ids=np.asarray(rows["query_ids"], dtype=np.int64),
        tile_obs=np.stack(rows["tile_obs"]),
        melds=np.stack(rows["melds"]),
        meta=np.stack(rows["meta"]),
        events=np.stack(rows["events"]),
        event_lengths=np.asarray(rows["event_lengths"], dtype=np.uint16),
        legal=np.stack(rows["legal"]),
        categories=np.asarray(rows["categories"], dtype=np.uint8),
        rank_outcomes=np.stack(rows["rank_outcomes"]),
        score_outcomes=np.stack(rows["score_outcomes"]),
        behavior_actions=np.asarray(rows["behavior_actions"], dtype=np.uint8),
    )
    return batch, {
        "states": len(batch),
        "worlds": worlds,
        "world_sampling": world_sampling,
        "rollout_states": rollout_states,
        "rollout_seconds": rollout_seconds,
        "rollout_states_per_second": rollout_states / max(rollout_seconds, 1e-9),
        "evaluated_actions": int(
            batch.evaluated_actions.sum(dtype=np.int64)
        ),
        "complete_action_corpus": batch.is_complete,
    }


def cached_world_outcome_corpus(
    directory: Path,
    queries: Sequence[PolicyQuery],
    actor: BloodFlowTransformer,
    device: torch.device,
    *,
    self_play_actor: BloodFlowTransformer | None = None,
    fingerprint: str,
    worlds: int,
    world_chunk: int,
    world_seed: int,
    world_sampling: str,
    shard_size: int,
    query_batch_size: int = 64,
    inference_batch_size: int = 512,
    action_sets: Sequence[np.ndarray] | None = None,
    on_progress: TargetProgress | None = None,
    prefix_directory: Path | None = None,
) -> tuple[WorldOutcomeBatch, dict[str, object]]:
    if shard_size <= 0 or not fingerprint:
        raise ValueError("world outcome cache needs a shard size and fingerprint")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    normalized_action_sets = _normalize_action_sets(action_sets, len(queries))
    expected = {
        "version": WORLD_OUTCOME_CACHE_VERSION,
        "fingerprint": fingerprint,
        "query_signature": query_signature(queries),
        "queries": len(queries),
        "worlds": worlds,
        "world_chunk": world_chunk,
        "world_seed": int(world_seed),
        "world_sampling": world_sampling,
        "shard_size": shard_size,
        "query_batch_size": query_batch_size,
        "inference_batch_size": inference_batch_size,
        "action_set_signature": _action_set_signature(normalized_action_sets),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != expected:
            raise ValueError("world outcome cache manifest does not match")
    else:
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        entries = list(directory.iterdir())
        if entries == [temporary]:
            temporary.unlink()
        elif entries:
            raise ValueError("world outcome cache directory is non-empty")
        _atomic_json(manifest_path, expected)

    if prefix_directory is not None:
        prefix_directory = prefix_directory.resolve()
        prefix_manifest_path = prefix_directory / "manifest.json"
        if not prefix_manifest_path.exists():
            raise FileNotFoundError(prefix_manifest_path)
        prefix_manifest = json.loads(prefix_manifest_path.read_text())
        compatible_keys = (
            "version",
            "worlds",
            "world_chunk",
            "world_seed",
            "world_sampling",
            "shard_size",
            "query_batch_size",
            "inference_batch_size",
        )
        if any(prefix_manifest.get(key) != expected[key] for key in compatible_keys):
            raise ValueError("world outcome prefix cache is incompatible")
        expected_action_signature = expected["action_set_signature"]
        prefix_action_signature = prefix_manifest.get("action_set_signature")
        if (
            expected_action_signature == "all_legal"
            and prefix_action_signature != "all_legal"
        ) or (
            expected_action_signature != "all_legal"
            and prefix_action_signature != expected_action_signature
        ):
            raise ValueError("world outcome prefix cache has different action sets")

    batches: list[WorldOutcomeBatch] = []
    rollout_states = 0
    rollout_seconds = 0.0
    reused_queries = 0
    for start in range(0, len(queries), shard_size):
        stop = min(start + shard_size, len(queries))
        path = directory / f"outcomes-{start:06d}-{stop:06d}.npz"
        expected_ids = np.asarray(
            [query.query_id for query in queries[start:stop]], dtype=np.int64
        )
        if not path.exists() and prefix_directory is not None:
            prefix_path = prefix_directory / path.name
            if prefix_path.exists():
                prefix_batch = load_world_outcome_batch(prefix_path)
                if not np.array_equal(prefix_batch.query_ids, expected_ids):
                    raise ValueError("world outcome prefix has wrong query ids")
                _require_action_set_match(
                    prefix_batch,
                    None
                    if normalized_action_sets is None
                    else normalized_action_sets[start:stop],
                )
                try:
                    os.link(prefix_path, path)
                except OSError:
                    shutil.copy2(prefix_path, path)
                reused_queries += len(prefix_batch)
        if path.exists():
            batch = load_world_outcome_batch(path)
            if not np.array_equal(batch.query_ids, expected_ids):
                raise ValueError("cached world outcome shard has wrong query ids")
            _require_action_set_match(
                batch,
                None
                if normalized_action_sets is None
                else normalized_action_sets[start:stop],
            )
            if on_progress is not None:
                on_progress(stop, {"cached": True})
        else:
            def shard_progress(done: int, fields: Mapping[str, object]) -> None:
                if on_progress is not None:
                    on_progress(start + done, fields)

            batch, metrics = estimate_world_outcome_batch(
                queries[start:stop],
                actor,
                device,
                self_play_actor=self_play_actor,
                worlds=worlds,
                world_chunk=world_chunk,
                seed=world_seed,
                world_sampling=world_sampling,
                query_batch_size=query_batch_size,
                inference_batch_size=inference_batch_size,
                action_sets=(
                    None
                    if normalized_action_sets is None
                    else normalized_action_sets[start:stop]
                ),
                on_progress=shard_progress,
            )
            save_world_outcome_batch(path, batch)
            rollout_states += int(metrics["rollout_states"])
            rollout_seconds += float(metrics["rollout_seconds"])
        batches.append(batch)
    result = concatenate_world_outcome_batches(batches)
    return result, {
        "states": len(result),
        "worlds": worlds,
        "world_sampling": world_sampling,
        "new_rollout_states": rollout_states,
        "new_rollout_seconds": rollout_seconds,
        "new_rollout_states_per_second": rollout_states
        / max(rollout_seconds, 1e-9),
        "reused_prefix_queries": reused_queries,
    }
