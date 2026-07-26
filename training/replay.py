"""Persistent compact trajectory replay with strict on-demand materialization."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from .learner import LearningBatch
from .observation import unpack_action_masks
from .policy_pool import (
    BalancedReplaySampler,
    ReplayBatchSelection,
    ReplaySource,
    replay_composition,
)
from .trajectory import (
    CompactTrajectory,
    decode_trajectory_shard,
    encode_trajectory_shard,
    replay_trajectory,
)
import bloodflow_mahjong as bm


REPLAY_FORMAT_VERSION = 3
Split = Literal["train", "validation"]


@dataclass(frozen=True)
class ReplayConfig:
    validation_fraction: float = 0.10
    maximum_online_transitions: int = 2_000_000
    maximum_mc_targets: int = 100_000
    strict_validation: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.maximum_online_transitions <= 0 or self.maximum_mc_targets <= 0:
            raise ValueError("replay capacities must be positive")


@dataclass(frozen=True)
class ReplayEntry:
    trajectory_id: int
    trajectory: CompactTrajectory
    anchor: bool
    split: Split
    shard: str
    shard_index: int


@dataclass(frozen=True)
class MonteCarloTarget:
    target_id: int
    query_id: int
    candidate_count: int
    trajectory_id: int
    step_index: int
    action: int
    mean_return: float
    variance: float
    samples: int
    confidence_low: float
    confidence_high: float
    split: Split
    reliable_actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.target_id < 0
            or self.query_id < 0
            or self.trajectory_id < 0
            or self.step_index < 0
        ):
            raise ValueError("MC identifiers and step must be non-negative")
        if self.candidate_count < 2:
            raise ValueError("MC candidate count must be at least two")
        if not 0 <= self.action < 115:
            raise ValueError("MC action is outside the policy action space")
        if self.samples <= 0:
            raise ValueError("MC sample count must be positive")
        values = (
            self.mean_return,
            self.variance,
            self.confidence_low,
            self.confidence_high,
        )
        if not np.isfinite(values).all() or self.variance < 0:
            raise ValueError("MC statistics must be finite and variance non-negative")
        if self.confidence_low > self.mean_return or self.confidence_high < self.mean_return:
            raise ValueError("MC confidence interval must contain its mean")
        if not isinstance(self.reliable_actions, (list, tuple)):
            raise ValueError("MC reliable actions must be a sequence")
        reliable: list[int] = []
        for action in self.reliable_actions:
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
                raise ValueError("MC reliable actions must be integer action ids")
            reliable.append(int(action))
        if not reliable:
            raise ValueError("MC target must participate in a reliable action pair")
        if len(set(reliable)) != len(reliable):
            raise ValueError("MC reliable actions must be unique")
        if any(not 0 <= action < 115 for action in reliable):
            raise ValueError("MC reliable action is outside the policy action space")
        if self.action in reliable:
            raise ValueError("MC target cannot be reliable with itself")
        if len(reliable) >= self.candidate_count:
            raise ValueError("MC target has too many reliable counterparts")
        object.__setattr__(self, "reliable_actions", tuple(sorted(reliable)))


def _validate_mc_group(
    group: Sequence[MonteCarloTarget],
    *,
    source: str,
) -> None:
    if not group:
        raise ValueError(f"{source} contains an empty MC query")
    first = group[0]
    if any(
        value.candidate_count != first.candidate_count
        or value.trajectory_id != first.trajectory_id
        or value.step_index != first.step_index
        or value.split != first.split
        or value.samples != first.samples
        for value in group
    ):
        raise ValueError(f"{source} MC query group metadata is inconsistent")
    if len(group) != first.candidate_count:
        raise ValueError(f"{source} MC query group is incomplete")
    by_action = {value.action: value for value in group}
    if len(by_action) != first.candidate_count:
        raise ValueError(f"{source} MC query group actions must be unique")
    candidate_actions = set(by_action)
    for value in group:
        if not set(value.reliable_actions) <= candidate_actions:
            raise ValueError(
                f"{source} MC reliable action is outside its query group"
            )
        for counterpart in value.reliable_actions:
            if value.action not in by_action[counterpart].reliable_actions:
                raise ValueError(f"{source} MC reliable action relation is asymmetric")


@dataclass(frozen=True)
class ReplayIndex:
    trajectory_ids: np.ndarray
    step_indices: np.ndarray
    actions: np.ndarray
    returns_override: np.ndarray
    sources: np.ndarray
    categories: np.ndarray
    policy_versions: np.ndarray
    duplicate_keys: np.ndarray
    mc_target_ids: np.ndarray
    mc_query_ids: np.ndarray
    mc_candidate_counts: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


class TrajectoryReplay:
    """Compact disk-backed replay; observations are rebuilt only when sampled."""

    def __init__(
        self,
        root: str | Path,
        *,
        seed: int,
        config: ReplayConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or ReplayConfig()
        self.split_seed = int(seed)
        self.entries: list[ReplayEntry] = []
        self.mc_targets: list[MonteCarloTarget] = []
        self.next_trajectory_id = 0
        self.next_target_id = 0
        self.next_query_id = 0
        self.next_shard_id = 0
        self.cursor = 0
        self._index_cache: dict[tuple[Split, bool, bool], ReplayIndex] = {}

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _split(self, trajectory: CompactTrajectory) -> Split:
        value = (
            int(trajectory.seed)
            ^ self.split_seed
            ^ (len(trajectory) * 0x9E3779B97F4A7C15)
        ) & ((1 << 64) - 1)
        # SplitMix64 finalizer makes adjacent engine seeds independent here.
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
        value ^= value >> 31
        threshold = int(self.config.validation_fraction * (1 << 32))
        return "validation" if (value & 0xFFFFFFFF) < threshold else "train"

    def add_trajectories(
        self,
        trajectories: Sequence[CompactTrajectory],
        *,
        anchor: bool,
        trusted: bool = False,
    ) -> tuple[int, ...]:
        if not trajectories:
            return ()
        # External trajectories are replayed before admission.  The training
        # collector just produced its trajectories from this exact engine, so
        # repeating every game here only stalls the next CUDA block.
        if self.config.strict_validation and not trusted:
            for trajectory in trajectories:
                replay_trajectory(trajectory, history_capacity=1)
        shard_name = f"shard-{self.next_shard_id:08d}.bfsh"
        self.next_shard_id += 1
        path = self.root / shard_name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encode_trajectory_shard(trajectories))
        temporary.replace(path)
        identifiers: list[int] = []
        for index, trajectory in enumerate(trajectories):
            identifier = self.next_trajectory_id
            self.next_trajectory_id += 1
            identifiers.append(identifier)
            self.entries.append(
                ReplayEntry(
                    trajectory_id=identifier,
                    trajectory=trajectory,
                    anchor=anchor,
                    split=self._split(trajectory),
                    shard=shard_name,
                    shard_index=index,
                )
            )
        self._trim_online()
        self._index_cache.clear()
        self.save_manifest()
        return tuple(identifiers)

    def _trim_online(self) -> None:
        online = [entry for entry in self.entries if not entry.anchor]
        total = sum(len(entry.trajectory) for entry in online)
        remove: set[int] = set()
        for entry in online:
            if total <= self.config.maximum_online_transitions:
                break
            remove.add(entry.trajectory_id)
            total -= len(entry.trajectory)
        if remove:
            self.entries = [
                entry for entry in self.entries if entry.trajectory_id not in remove
            ]
            self.mc_targets = [
                target for target in self.mc_targets if target.trajectory_id not in remove
            ]

    def add_mc_targets(self, targets: Iterable[MonteCarloTarget]) -> None:
        incoming = list(targets)
        if not incoming:
            return
        known = {entry.trajectory_id: entry for entry in self.entries}
        groups: dict[int, list[MonteCarloTarget]] = {}
        for value in incoming:
            entry = known.get(value.trajectory_id)
            if entry is None:
                raise ValueError(f"unknown MC trajectory {value.trajectory_id}")
            if value.step_index >= len(entry.trajectory):
                raise ValueError("MC target step is outside its trajectory")
            if value.split != entry.split:
                raise ValueError("MC target split must match its source trajectory")
            groups.setdefault(value.query_id, []).append(value)
        for group in groups.values():
            _validate_mc_group(group, source="incoming")

        for group in groups.values():
            assigned_query_id = self.next_query_id
            self.next_query_id += 1
            for value in group:
                assigned = MonteCarloTarget(
                    target_id=self.next_target_id,
                    query_id=assigned_query_id,
                    candidate_count=value.candidate_count,
                    trajectory_id=value.trajectory_id,
                    step_index=value.step_index,
                    action=value.action,
                    mean_return=value.mean_return,
                    variance=value.variance,
                    samples=value.samples,
                    confidence_low=value.confidence_low,
                    confidence_high=value.confidence_high,
                    split=value.split,
                    reliable_actions=value.reliable_actions,
                )
                self.next_target_id += 1
                self.mc_targets.append(assigned)
        while len(self.mc_targets) > self.config.maximum_mc_targets:
            oldest_query = next(
                (
                    target.query_id
                    for target in self.mc_targets
                    if not (
                        target.split == "validation"
                        and known[target.trajectory_id].anchor
                    )
                ),
                None,
            )
            if oldest_query is None:
                raise RuntimeError(
                    "MC capacity is smaller than the protected validation corpus"
                )
            self.mc_targets = [
                target
                for target in self.mc_targets
                if target.query_id != oldest_query
            ]
        self._index_cache.clear()
        self.save_manifest()

    def index(
        self,
        split: Split = "train",
        *,
        include_mc: bool = True,
        anchor_only: bool = False,
    ) -> ReplayIndex:
        # With no teacher rows both variants are identical and share one cache.
        cache_key = (split, bool(include_mc and self.mc_targets), anchor_only)
        cached = self._index_cache.get(cache_key)
        if cached is not None:
            return cached
        trajectory_ids: list[int] = []
        step_indices: list[int] = []
        actions: list[int] = []
        returns_override: list[float] = []
        sources: list[int] = []
        categories: list[int] = []
        versions: list[int] = []
        duplicate_keys: list[int] = []
        target_ids: list[int] = []
        query_ids: list[int] = []
        candidate_counts: list[int] = []
        entries = {entry.trajectory_id: entry for entry in self.entries}
        for entry in self.entries:
            if entry.split != split or (anchor_only and not entry.anchor):
                continue
            trajectory = entry.trajectory
            for step in range(len(trajectory)):
                trajectory_ids.append(entry.trajectory_id)
                step_indices.append(step)
                actions.append(int(trajectory.actions[step]))
                returns_override.append(np.nan)
                sources.append(int(trajectory.sources[step]))
                categories.append(int(trajectory.categories[step]))
                versions.append(int(trajectory.policy_versions[step]))
                duplicate_keys.append(hash((int(trajectory.seed), step)) & ((1 << 63) - 1))
                target_ids.append(-1)
                query_ids.append(-1)
                candidate_counts.append(0)
        if include_mc:
            for target in self.mc_targets:
                if target.split != split:
                    continue
                entry = entries.get(target.trajectory_id)
                if entry is None:
                    raise RuntimeError("MC target references an evicted trajectory")
                if anchor_only and not entry.anchor:
                    continue
                trajectory_ids.append(target.trajectory_id)
                step_indices.append(target.step_index)
                actions.append(target.action)
                returns_override.append(target.mean_return)
                sources.append(int(ReplaySource.MC_TEACHER))
                categories.append(int(entry.trajectory.categories[target.step_index]))
                versions.append(int(entry.trajectory.policy_versions[target.step_index]))
                duplicate_keys.append(
                    hash((int(entry.trajectory.seed), target.step_index))
                    & ((1 << 63) - 1)
                )
                target_ids.append(target.target_id)
                query_ids.append(target.query_id)
                candidate_counts.append(target.candidate_count)
        result = ReplayIndex(
            trajectory_ids=np.asarray(trajectory_ids, dtype=np.uint64),
            step_indices=np.asarray(step_indices, dtype=np.uint16),
            actions=np.asarray(actions, dtype=np.uint8),
            returns_override=np.asarray(returns_override, dtype=np.float32),
            sources=np.asarray(sources, dtype=np.uint8),
            categories=np.asarray(categories, dtype=np.uint8),
            policy_versions=np.asarray(versions, dtype=np.uint32),
            duplicate_keys=np.asarray(duplicate_keys, dtype=np.uint64),
            mc_target_ids=np.asarray(target_ids, dtype=np.int64),
            mc_query_ids=np.asarray(query_ids, dtype=np.int64),
            mc_candidate_counts=np.asarray(candidate_counts, dtype=np.uint8),
        )
        for value in result.__dict__.values():
            value.setflags(write=False)
        self._index_cache[cache_key] = result
        return result

    def sample(
        self,
        sampler: BalancedReplaySampler,
        batch_size: int,
        *,
        split: Split = "train",
        include_mc: bool = True,
        actor_only: bool = False,
        include_oracle: bool = False,
    ) -> tuple[LearningBatch, ReplayBatchSelection]:
        index = self.index(split, include_mc=include_mc and not actor_only)
        if actor_only:
            keep = index.sources != int(ReplaySource.MC_TEACHER)
            index = _filter_index(index, keep)
        selection = sampler.sample(
            index.sources,
            index.categories,
            batch_size,
            duplicate_keys=index.duplicate_keys,
            policy_versions=index.policy_versions,
        )
        return self.materialize(index, selection.indices, include_oracle=include_oracle), selection

    def validation_batch(
        self,
        maximum_states: int,
        *,
        seed: int,
        include_mc: bool = False,
        include_oracle: bool = False,
    ) -> LearningBatch:
        index = self.index("validation", include_mc=include_mc)
        if not len(index):
            raise RuntimeError("validation replay is empty")
        random = np.random.default_rng(seed)
        if len(index) > maximum_states:
            selected = random.choice(len(index), maximum_states, replace=False)
        else:
            selected = np.arange(len(index))
        return self.materialize(index, selected, include_oracle=include_oracle)

    def materialize(
        self,
        index: ReplayIndex,
        rows: Sequence[int] | np.ndarray,
        *,
        include_oracle: bool = False,
        include_rule_actions: bool = False,
    ) -> LearningBatch:
        rows = np.asarray(rows, dtype=np.int64)
        count = len(rows)
        tile_obs = np.empty((count, 10, 27), dtype=np.uint8)
        melds = np.empty((count, 4, 4, 3), dtype=np.uint8)
        meta = np.empty((count, 34), dtype=np.int32)
        events = np.empty((count, 192, 8), dtype=np.int32)
        lengths = np.empty(count, dtype=np.uint16)
        legal = np.empty((count, 115), dtype=np.bool_)
        actions = index.actions[rows].copy()
        returns = np.empty(count, dtype=np.float32)
        probabilities = np.empty(count, dtype=np.float32)
        temperatures = np.empty(count, dtype=np.float32)
        oracle_tiles = (
            np.empty((count, 9, 27), dtype=np.uint8) if include_oracle else None
        )
        rule_actions = (
            np.empty(count, dtype=np.uint8) if include_rule_actions else None
        )
        selected_target_ids = index.mc_target_ids[rows]
        has_mc_targets = bool(np.any(selected_target_ids >= 0))
        mc_reliable_actions = None
        if has_mc_targets:
            targets_by_id = {target.target_id: target for target in self.mc_targets}
            mc_reliable_actions = np.zeros((count, 115), dtype=np.bool_)
            for output, target_id in enumerate(selected_target_ids):
                if int(target_id) < 0:
                    continue
                target = targets_by_id.get(int(target_id))
                if target is None:
                    raise RuntimeError("replay index references an unknown MC target")
                mc_reliable_actions[output, target.reliable_actions] = True
        entries = {entry.trajectory_id: entry for entry in self.entries}
        requested: dict[int, list[tuple[int, int, int]]] = {}
        requested_by_step: dict[int, list[tuple[int, int, int]]] = {}
        for output, row in enumerate(rows):
            trajectory_id = int(index.trajectory_ids[row])
            step = int(index.step_indices[row])
            requested.setdefault(trajectory_id, []).append((step, output, int(row)))
            requested_by_step.setdefault(step, []).append(
                (trajectory_id, output, int(row))
            )
        trajectory_ids = np.asarray(sorted(requested), dtype=np.uint64)
        active_entries = [entries[int(identifier)] for identifier in trajectory_ids]
        engine = bm.Batch(len(active_entries), seed=0)
        engine.reset_many(
            np.arange(len(active_entries), dtype=np.uint32),
            np.asarray(
                [entry.trajectory.seed for entry in active_entries], dtype=np.uint64
            ),
        )
        cumulative = np.zeros((len(active_entries), 4), dtype=np.float32)
        maximum_steps = np.asarray(
            [
                max(step for step, _output, _row in requested[int(identifier)])
                for identifier in trajectory_ids
            ],
            dtype=np.int64,
        )
        active_ids = trajectory_ids.copy()
        records = np.empty((len(active_entries), 12), dtype=np.int64)
        masks = np.empty((len(active_entries), 2), dtype=np.uint64)
        current_tile = np.empty((len(active_entries), 10, 27), dtype=np.uint8)
        current_melds = np.empty((len(active_entries), 4, 4, 3), dtype=np.uint8)
        current_river = np.empty((len(active_entries), 108, 2), dtype=np.uint8)
        current_meta = np.empty((len(active_entries), 34), dtype=np.int32)
        current_events = np.empty((len(active_entries), 192, 8), dtype=np.int32)
        current_lengths = np.empty(len(active_entries), dtype=np.uint16)
        current_oracle = (
            np.empty((len(active_entries), 9, 27), dtype=np.uint8)
            if include_oracle
            else None
        )
        current_rule_actions = (
            np.empty(len(active_entries), dtype=np.uint8)
            if include_rule_actions
            else None
        )

        for step in range(int(maximum_steps.max(initial=0)) + 1):
            step_requests = requested_by_step.get(step)
            if step_requests:
                observed_ids = np.asarray(
                    sorted({trajectory_id for trajectory_id, _output, _row in step_requests}),
                    dtype=np.uint64,
                )
                engine_rows = np.searchsorted(active_ids, observed_ids)
                if np.any(engine_rows >= len(active_ids)) or not np.array_equal(
                    active_ids[engine_rows], observed_ids
                ):
                    raise RuntimeError("requested replay trajectory is no longer active")
                observed = engine.clone_indices(engine_rows.astype(np.uint32))
                observed_size = len(observed_ids)
                observed.observe_into(
                    current_tile[:observed_size],
                    current_melds[:observed_size],
                    current_river[:observed_size],
                    current_meta[:observed_size],
                )
                observed.legal_action_masks_into(masks[:observed_size])
                observed.events_into(
                    current_events[:observed_size], current_lengths[:observed_size]
                )
                if include_oracle:
                    observed.oracle_tile_counts_into(current_oracle[:observed_size])
                if include_rule_actions:
                    observed.simple_rule_actions_into(
                        current_rule_actions[:observed_size]
                    )
                dense = unpack_action_masks(masks[:observed_size])

                for observed_row, identifier in enumerate(observed_ids):
                    trajectory = entries[int(identifier)].trajectory
                    if int(current_meta[observed_row, 1]) != int(
                        trajectory.actors[step]
                    ):
                        raise RuntimeError(
                            "batched replay reconstructed a different actor"
                        )
                    if int(current_meta[observed_row, 0]) != int(
                        trajectory.phases[step]
                    ):
                        raise RuntimeError(
                            "batched replay reconstructed a different phase"
                        )

                for trajectory_id, output, index_row in step_requests:
                    observed_row = int(np.searchsorted(observed_ids, trajectory_id))
                    engine_row = int(engine_rows[observed_row])
                    trajectory = entries[trajectory_id].trajectory
                    tile_obs[output] = current_tile[observed_row]
                    melds[output] = current_melds[observed_row]
                    meta[output] = current_meta[observed_row]
                    events[output] = current_events[observed_row]
                    lengths[output] = current_lengths[observed_row]
                    legal[output] = dense[observed_row]
                    action = int(actions[output])
                    if not legal[output, action]:
                        raise RuntimeError(
                            "MC or behavior action is illegal during batched replay"
                        )
                    override = float(index.returns_override[index_row])
                    actor = int(trajectory.actors[step])
                    terminal_delta = (
                        float(trajectory.terminal_scores[actor]) - 10_000.0
                    ) / 10_000.0
                    returns[output] = (
                        terminal_delta - cumulative[engine_row, actor]
                        if np.isnan(override)
                        else override
                    )
                    if int(index.mc_target_ids[index_row]) >= 0:
                        probabilities[output] = 1.0
                        temperatures[output] = 0.0
                    else:
                        probabilities[output] = trajectory.action_probabilities[step]
                        temperatures[output] = trajectory.temperatures[step]
                    if include_oracle:
                        oracle_tiles[output] = current_oracle[observed_row]
                    if include_rule_actions:
                        rule_actions[output] = current_rule_actions[observed_row]

            keep = np.flatnonzero(maximum_steps > step).astype(np.uint32)
            if not len(keep):
                break
            next_ids = active_ids[keep]
            next_entries = [entries[int(identifier)] for identifier in next_ids]
            next_engine = engine.clone_indices(keep)
            step_actions = np.asarray(
                [entry.trajectory.actions[step] for entry in next_entries],
                dtype=np.uint8,
            )
            next_records = records[: len(keep)]
            next_engine.step_into(step_actions, next_records)
            if np.any(next_records[:, 11]):
                raise RuntimeError("trajectory terminated before a requested replay step")
            cumulative = cumulative[keep] + (
                next_records[:, 5:9].astype(np.float32) / 10_000.0
            )
            maximum_steps = maximum_steps[keep]
            active_ids = next_ids
            engine = next_engine
        return LearningBatch(
            tile_obs=tile_obs,
            melds=melds,
            meta=meta,
            events=events,
            event_lengths=lengths,
            legal=legal,
            actions=actions,
            returns=returns,
            categories=index.categories[rows].copy(),
            sources=index.sources[rows].copy(),
            policy_versions=index.policy_versions[rows].copy(),
            behavior_probabilities=probabilities,
            temperatures=temperatures,
            trajectory_ids=index.trajectory_ids[rows].copy(),
            step_indices=index.step_indices[rows].copy(),
            oracle_tiles=oracle_tiles,
            rule_actions=rule_actions,
            mc_query_ids=(
                index.mc_query_ids[rows].copy() if has_mc_targets else None
            ),
            mc_candidate_counts=(
                index.mc_candidate_counts[rows].copy() if has_mc_targets else None
            ),
            mc_reliable_actions=mc_reliable_actions,
        )

    def mc_target_count(self, split: Split, *, anchor_only: bool = False) -> int:
        entries = {entry.trajectory_id: entry for entry in self.entries}
        count = 0
        for target in self.mc_targets:
            if target.split != split:
                continue
            entry = entries.get(target.trajectory_id)
            if entry is None:
                raise RuntimeError("MC target references an evicted trajectory")
            if not anchor_only or entry.anchor:
                count += 1
        return count

    def reliable_mc_counts(
        self,
        split: Split,
        *,
        anchor_only: bool = False,
    ) -> dict[str, int]:
        entries = {entry.trajectory_id: entry for entry in self.entries}
        groups: dict[int, list[MonteCarloTarget]] = {}
        for target in self.mc_targets:
            if target.split != split:
                continue
            entry = entries.get(target.trajectory_id)
            if entry is None:
                raise RuntimeError("MC target references an evicted trajectory")
            if anchor_only and not entry.anchor:
                continue
            groups.setdefault(target.query_id, []).append(target)

        target_count = 0
        pair_count = 0
        for group in groups.values():
            _validate_mc_group(group, source="replay")
            target_count += len(group)
            pair_count += sum(len(target.reliable_actions) for target in group) // 2
        return {
            "targets": target_count,
            "groups": len(groups),
            "pairs": pair_count,
        }

    def _mc_group_batch(
        self,
        split: Split,
        maximum_states: int,
        *,
        seed: int,
        anchor_only: bool,
    ) -> LearningBatch | None:
        if maximum_states <= 0:
            raise ValueError("maximum_states must be positive")
        index = self.index(split, include_mc=True, anchor_only=anchor_only)
        rows = np.flatnonzero(index.mc_target_ids >= 0)
        if not len(rows):
            return None
        # Sample atomic paired-world queries. Repeated queries of the same state
        # remain separate because their common-random-number worlds differ.
        groups: dict[int, list[int]] = {}
        for row in rows:
            query_id = int(index.mc_query_ids[row])
            if query_id < 0:
                raise RuntimeError("MC teacher row has no query id")
            groups.setdefault(query_id, []).append(int(row))
        group_rows: list[np.ndarray] = []
        for query_rows in groups.values():
            group = np.asarray(query_rows, dtype=np.int64)
            expected = np.unique(index.mc_candidate_counts[group])
            if (
                len(expected) != 1
                or int(expected[0]) != len(group)
                or len(np.unique(index.actions[group])) != len(group)
                or len(np.unique(index.trajectory_ids[group])) != 1
                or len(np.unique(index.step_indices[group])) != 1
            ):
                raise RuntimeError("MC query group is incomplete")
            group_rows.append(group)
        random = np.random.default_rng(seed)
        random.shuffle(group_rows)
        selected: list[np.ndarray] = []
        selected_count = 0
        for group in group_rows:
            if selected_count + len(group) > maximum_states:
                continue
            selected.append(group)
            selected_count += len(group)
        if not selected:
            return None
        rows = np.concatenate(selected)
        return self.materialize(index, rows)

    def mc_training_batch(
        self,
        maximum_states: int,
        *,
        seed: int,
    ) -> LearningBatch | None:
        return self._mc_group_batch(
            "train", maximum_states, seed=seed, anchor_only=False
        )

    def mc_validation_batch(
        self,
        maximum_states: int,
        *,
        seed: int,
    ) -> LearningBatch | None:
        return self._mc_group_batch(
            "validation", maximum_states, seed=seed, anchor_only=True
        )

    def composition(self, split: Split = "train") -> dict[str, object]:
        index = self.index(split)
        result: dict[str, object] = replay_composition(index.sources, index.categories)
        result.update(
            {
                "trajectories": sum(entry.split == split for entry in self.entries),
                "states": len(index),
                "anchor_trajectories": sum(
                    entry.split == split and entry.anchor for entry in self.entries
                ),
                "online_trajectories": sum(
                    entry.split == split and not entry.anchor for entry in self.entries
                ),
                "mc_targets": sum(target.split == split for target in self.mc_targets),
            }
        )
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": REPLAY_FORMAT_VERSION,
            "root": str(self.root),
            "config": asdict(self.config),
            "split_seed": self.split_seed,
            "next_trajectory_id": self.next_trajectory_id,
            "next_target_id": self.next_target_id,
            "next_query_id": self.next_query_id,
            "next_shard_id": self.next_shard_id,
            "cursor": self.cursor,
            "manifest": self.manifest_path.name,
        }

    def save_manifest(self) -> None:
        data = {
            **self.state_dict(),
            "entries": [
                {
                    "trajectory_id": entry.trajectory_id,
                    "anchor": entry.anchor,
                    "split": entry.split,
                    "shard": entry.shard,
                    "shard_index": entry.shard_index,
                }
                for entry in self.entries
            ],
            "mc_targets": [asdict(target) for target in self.mc_targets],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)

    @classmethod
    def load(cls, root: str | Path) -> TrajectoryReplay:
        root = Path(root)
        state = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(state["format_version"]) != REPLAY_FORMAT_VERSION:
            raise ValueError("unsupported replay manifest version")
        replay = cls(
            root,
            seed=int(state["split_seed"]),
            config=ReplayConfig(**state["config"]),
        )
        shard_cache: dict[str, tuple[CompactTrajectory, ...]] = {}
        for item in state["entries"]:
            shard = str(item["shard"])
            if shard not in shard_cache:
                shard_cache[shard] = decode_trajectory_shard((root / shard).read_bytes())
            shard_index = int(item["shard_index"])
            trajectories = shard_cache[shard]
            if not 0 <= shard_index < len(trajectories):
                raise ValueError("replay manifest shard index is invalid")
            replay.entries.append(
                ReplayEntry(
                    trajectory_id=int(item["trajectory_id"]),
                    trajectory=trajectories[shard_index],
                    anchor=bool(item["anchor"]),
                    split=str(item["split"]),  # type: ignore[arg-type]
                    shard=shard,
                    shard_index=shard_index,
                )
            )
            # Admission already performed semantic replay for external data;
            # collector-produced data was valid by construction.  Decoding
            # verifies format, engine rules version, record shapes and CRC.
        replay.mc_targets = [MonteCarloTarget(**item) for item in state["mc_targets"]]
        replay.next_trajectory_id = int(state["next_trajectory_id"])
        replay.next_target_id = int(state["next_target_id"])
        replay.next_query_id = int(state["next_query_id"])
        replay.next_shard_id = int(state["next_shard_id"])
        replay.cursor = int(state["cursor"])
        loaded_groups: dict[int, list[MonteCarloTarget]] = {}
        for target in replay.mc_targets:
            loaded_groups.setdefault(target.query_id, []).append(target)
        entries = {entry.trajectory_id: entry for entry in replay.entries}
        for group in loaded_groups.values():
            _validate_mc_group(group, source="replay manifest")
            first = group[0]
            entry = entries.get(first.trajectory_id)
            if entry is None:
                raise ValueError("replay manifest MC query has no source trajectory")
            if first.split != entry.split or first.step_index >= len(entry.trajectory):
                raise ValueError("replay manifest MC query source metadata is invalid")
        if loaded_groups and replay.next_query_id <= max(loaded_groups):
            raise ValueError("replay manifest MC query counter is invalid")
        return replay


def _filter_index(index: ReplayIndex, keep: np.ndarray) -> ReplayIndex:
    return ReplayIndex(
        trajectory_ids=index.trajectory_ids[keep],
        step_indices=index.step_indices[keep],
        actions=index.actions[keep],
        returns_override=index.returns_override[keep],
        sources=index.sources[keep],
        categories=index.categories[keep],
        policy_versions=index.policy_versions[keep],
        duplicate_keys=index.duplicate_keys[keep],
        mc_target_ids=index.mc_target_ids[keep],
        mc_query_ids=index.mc_query_ids[keep],
        mc_candidate_counts=index.mc_candidate_counts[keep],
    )
