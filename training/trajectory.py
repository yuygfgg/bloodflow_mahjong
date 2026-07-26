"""Compact, versioned full-game trajectories and deterministic reconstruction."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import bloodflow_mahjong as bm

from .policy_pool import (
    CATEGORY_COUNT,
    DecisionCategory,
    ReplaySource,
    decision_categories,
)


TRAJECTORY_FORMAT_VERSION = 1
TRAJECTORY_MAGIC = b"BFTR"
SHARD_MAGIC = b"BFSH"
_HEADER = struct.Struct("<4sHHIQIBBHI4i4B")
_SHARD_HEADER = struct.Struct("<4sHHI")
_LENGTH = struct.Struct("<I")
_STEP_DTYPE = np.dtype(
    [
        ("action", "u1"),
        ("actor", "u1"),
        ("phase", "u1"),
        ("category", "u1"),
        ("source", "u1"),
        ("temperature", "<f2"),
        ("action_probability", "<f4"),
        ("policy_version", "<u4"),
    ],
    align=False,
)


class TrajectoryFormatError(ValueError):
    """A persisted trajectory is corrupt or uses an unsupported schema."""


class TrajectoryReplayError(ValueError):
    """A seed/action replay disagrees with its persisted trajectory metadata."""


def _array(value: object, dtype: np.dtype[object], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array)


def _seat_ranks(ranking_order: Sequence[int]) -> np.ndarray:
    order = np.asarray(ranking_order, dtype=np.uint8)
    if order.shape != (4,) or sorted(order.tolist()) != [0, 1, 2, 3]:
        raise ValueError("ranking_order must be a permutation of seats 0..3")
    ranks = np.empty(4, dtype=np.uint8)
    ranks[order] = np.arange(1, 5, dtype=np.uint8)
    return ranks


@dataclass(frozen=True)
class CompactTrajectory:
    """One complete game without persisted observations or legal masks."""

    seed: int
    exchange_direction: int
    actions: np.ndarray
    actors: np.ndarray
    phases: np.ndarray
    categories: np.ndarray
    sources: np.ndarray
    policy_versions: np.ndarray
    action_probabilities: np.ndarray
    temperatures: np.ndarray
    terminal_scores: np.ndarray
    terminal_ranks: np.ndarray
    termination_reason: int
    format_version: int = TRAJECTORY_FORMAT_VERSION
    engine_rules_version: int = field(
        default_factory=lambda: int(bm.ENGINE_RULES_VERSION)
    )

    def __post_init__(self) -> None:
        arrays = {
            "actions": _array(self.actions, np.dtype(np.uint8), "actions"),
            "actors": _array(self.actors, np.dtype(np.uint8), "actors"),
            "phases": _array(self.phases, np.dtype(np.uint8), "phases"),
            "categories": _array(
                self.categories, np.dtype(np.uint8), "categories"
            ),
            "sources": _array(self.sources, np.dtype(np.uint8), "sources"),
            "policy_versions": _array(
                self.policy_versions, np.dtype("<u4"), "policy_versions"
            ),
            "action_probabilities": _array(
                self.action_probabilities,
                np.dtype("<f4"),
                "action_probabilities",
            ),
            "temperatures": _array(
                self.temperatures, np.dtype("<f2"), "temperatures"
            ),
        }
        lengths = {len(array) for array in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("all step arrays must have the same positive length")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

        scores = _array(self.terminal_scores, np.dtype("<i4"), "terminal_scores")
        ranks = _array(self.terminal_ranks, np.dtype(np.uint8), "terminal_ranks")
        if scores.shape != (4,):
            raise ValueError("terminal_scores must have shape [4]")
        if ranks.shape != (4,) or sorted(ranks.tolist()) != [1, 2, 3, 4]:
            raise ValueError("terminal_ranks must be a permutation of ranks 1..4")
        if int(scores.sum(dtype=np.int64)) != 40_000:
            raise ValueError("terminal scores must sum to 40000")
        object.__setattr__(self, "terminal_scores", scores)
        object.__setattr__(self, "terminal_ranks", ranks)

        if not 0 <= int(self.seed) <= np.iinfo(np.uint64).max:
            raise ValueError("seed must fit uint64")
        if self.exchange_direction not in (1, 2, 3):
            raise ValueError("exchange_direction must be 1, 2, or 3")
        if self.format_version != TRAJECTORY_FORMAT_VERSION:
            raise ValueError("unsupported trajectory format version")
        if self.engine_rules_version != int(bm.ENGINE_RULES_VERSION):
            raise ValueError("trajectory engine rule version does not match this engine")
        if self.termination_reason not in (
            int(bm.TERMINATION_WALL_EXHAUSTED),
            int(bm.TERMINATION_THREE_PLAYERS_BANKRUPT),
        ):
            raise ValueError("invalid termination reason")
        if np.any(self.actions >= int(bm.ACTION_SPACE_SIZE)):
            raise ValueError("actions contain an invalid action ID")
        if np.any(self.actors >= 4):
            raise ValueError("actors contain an invalid seat")
        if np.any(self.phases >= int(bm.PHASE_FINISHED)):
            raise ValueError("phases contain a terminal or invalid phase")
        if np.any(self.categories >= CATEGORY_COUNT):
            raise ValueError("categories contain an invalid decision category")
        valid_sources = np.asarray([int(source) for source in ReplaySource], dtype=np.uint8)
        if not np.isin(self.sources, valid_sources).all():
            raise ValueError("sources contain an invalid replay source")
        if not np.isfinite(self.action_probabilities).all() or np.any(
            (self.action_probabilities <= 0) | (self.action_probabilities > 1)
        ):
            raise ValueError("action probabilities must be finite and in (0, 1]")
        if not np.isfinite(self.temperatures).all() or np.any(self.temperatures < 0):
            raise ValueError("temperatures must be finite and non-negative")

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def nbytes(self) -> int:
        return _HEADER.size + len(self) * _STEP_DTYPE.itemsize

    def to_bytes(self) -> bytes:
        steps = np.empty(len(self), dtype=_STEP_DTYPE)
        steps["action"] = self.actions
        steps["actor"] = self.actors
        steps["phase"] = self.phases
        steps["category"] = self.categories
        steps["source"] = self.sources
        steps["temperature"] = self.temperatures
        steps["action_probability"] = self.action_probabilities
        steps["policy_version"] = self.policy_versions
        payload = steps.tobytes(order="C")
        checksum = zlib.crc32(payload)
        header = _HEADER.pack(
            TRAJECTORY_MAGIC,
            self.format_version,
            _HEADER.size,
            self.engine_rules_version,
            int(self.seed),
            len(self),
            int(self.exchange_direction),
            int(self.termination_reason),
            0,
            checksum,
            *(int(value) for value in self.terminal_scores),
            *(int(value) for value in self.terminal_ranks),
        )
        return header + payload

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> CompactTrajectory:
        view = memoryview(data)
        if len(view) < _HEADER.size:
            raise TrajectoryFormatError("trajectory is shorter than its header")
        unpacked = _HEADER.unpack_from(view)
        (
            magic,
            format_version,
            header_size,
            rules_version,
            seed,
            step_count,
            exchange_direction,
            termination_reason,
            _reserved,
            checksum,
            *terminal,
        ) = unpacked
        if magic != TRAJECTORY_MAGIC:
            raise TrajectoryFormatError("invalid trajectory magic")
        if format_version != TRAJECTORY_FORMAT_VERSION or header_size != _HEADER.size:
            raise TrajectoryFormatError("unsupported trajectory format")
        if rules_version != int(bm.ENGINE_RULES_VERSION):
            raise TrajectoryFormatError(
                f"engine rules version {rules_version} != {bm.ENGINE_RULES_VERSION}"
            )
        expected = _HEADER.size + step_count * _STEP_DTYPE.itemsize
        if len(view) != expected:
            raise TrajectoryFormatError(
                f"trajectory length {len(view)} does not match expected {expected}"
            )
        payload = view[_HEADER.size:]
        if zlib.crc32(payload) != checksum:
            raise TrajectoryFormatError("trajectory payload checksum mismatch")
        steps = np.frombuffer(payload, dtype=_STEP_DTYPE, count=step_count).copy()
        return cls(
            seed=seed,
            exchange_direction=exchange_direction,
            actions=steps["action"],
            actors=steps["actor"],
            phases=steps["phase"],
            categories=steps["category"],
            sources=steps["source"],
            policy_versions=steps["policy_version"],
            action_probabilities=steps["action_probability"],
            temperatures=steps["temperature"],
            terminal_scores=np.asarray(terminal[:4], dtype=np.int32),
            terminal_ranks=np.asarray(terminal[4:], dtype=np.uint8),
            termination_reason=termination_reason,
            format_version=format_version,
            engine_rules_version=rules_version,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: str | Path) -> CompactTrajectory:
        return cls.from_bytes(Path(path).read_bytes())


class TrajectoryBuilder:
    """Incrementally records policy metadata alongside a live engine game."""

    def __init__(self, seed: int, exchange_direction: int) -> None:
        self.seed = int(seed)
        self.exchange_direction = int(exchange_direction)
        self._actions: list[int] = []
        self._actors: list[int] = []
        self._phases: list[int] = []
        self._categories: list[int] = []
        self._sources: list[int] = []
        self._policy_versions: list[int] = []
        self._action_probabilities: list[float] = []
        self._temperatures: list[float] = []

    def append(
        self,
        *,
        action: int,
        actor: int,
        phase: int,
        category: DecisionCategory | int,
        source: ReplaySource | str | int,
        policy_version: int,
        action_probability: float,
        temperature: float,
    ) -> None:
        self._actions.append(int(action))
        self._actors.append(int(actor))
        self._phases.append(int(phase))
        self._categories.append(int(category))
        self._sources.append(int(ReplaySource.parse(source)))
        self._policy_versions.append(int(policy_version))
        self._action_probabilities.append(float(action_probability))
        self._temperatures.append(float(temperature))

    def finish(
        self,
        *,
        terminal_scores: Sequence[int],
        ranking_order: Sequence[int],
        termination_reason: int,
    ) -> CompactTrajectory:
        return CompactTrajectory(
            seed=self.seed,
            exchange_direction=self.exchange_direction,
            actions=np.asarray(self._actions, dtype=np.uint8),
            actors=np.asarray(self._actors, dtype=np.uint8),
            phases=np.asarray(self._phases, dtype=np.uint8),
            categories=np.asarray(self._categories, dtype=np.uint8),
            sources=np.asarray(self._sources, dtype=np.uint8),
            policy_versions=np.asarray(self._policy_versions, dtype=np.uint32),
            action_probabilities=np.asarray(
                self._action_probabilities, dtype=np.float32
            ),
            temperatures=np.asarray(self._temperatures, dtype=np.float16),
            terminal_scores=np.asarray(terminal_scores, dtype=np.int32),
            terminal_ranks=_seat_ranks(ranking_order),
            termination_reason=int(termination_reason),
        )


@dataclass(frozen=True)
class ReplayedTrajectory:
    trajectory: CompactTrajectory
    legal_mask_words: np.ndarray
    tile_obs: np.ndarray
    melds: np.ndarray
    river: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray
    oracle_tiles: np.ndarray
    score_deltas: np.ndarray
    returns_to_go: np.ndarray

    @property
    def actor_returns(self) -> np.ndarray:
        rows = np.arange(len(self.trajectory), dtype=np.int64)
        return self.returns_to_go[rows, self.trajectory.actors]


def replay_trajectory(
    trajectory: CompactTrajectory, *, history_capacity: int = 192
) -> ReplayedTrajectory:
    """Strictly reconstruct observations, masks, histories, and MC returns."""

    if not 0 <= history_capacity <= int(bm.EVENT_HISTORY_CAPACITY):
        raise ValueError(
            f"history_capacity must be in 0..{bm.EVENT_HISTORY_CAPACITY}"
        )
    game = bm.Game(seed=int(trajectory.seed))
    if int(game.exchange_direction) != trajectory.exchange_direction:
        raise TrajectoryReplayError("seed reconstructed a different exchange direction")

    steps = len(trajectory)
    legal_mask_words = np.empty(
        (steps, int(bm.LEGAL_ACTION_MASK_WORDS)), dtype=np.uint64
    )
    tile_obs = np.empty(
        (steps, int(bm.TILE_OBSERVATION_PLANES), int(bm.TILE_KIND_COUNT)),
        dtype=np.uint8,
    )
    melds = np.empty(
        (steps, int(bm.PLAYER_COUNT), int(bm.MELD_SLOTS), int(bm.MELD_FIELDS)),
        dtype=np.uint8,
    )
    river = np.empty(
        (steps, int(bm.RIVER_TILE_CAPACITY), int(bm.RIVER_FIELDS)),
        dtype=np.uint8,
    )
    meta = np.empty((steps, int(bm.META_OBSERVATION_WIDTH)), dtype=np.int32)
    events = np.empty(
        (steps, history_capacity, int(bm.EVENT_RECORD_WIDTH)), dtype=np.int32
    )
    event_lengths = np.empty(steps, dtype=np.uint16)
    oracle_tiles = np.empty(
        (steps, int(bm.ORACLE_TILE_COUNT_PLANES), int(bm.TILE_KIND_COUNT)),
        dtype=np.uint8,
    )
    score_deltas = np.empty((steps, 4), dtype=np.float32)
    transition = np.empty(int(bm.STEP_RECORD_WIDTH), dtype=np.int64)

    for step in range(steps):
        decision = game.decision
        expected = (int(trajectory.actors[step]), int(trajectory.phases[step]))
        if decision != expected:
            raise TrajectoryReplayError(
                f"step {step}: decision {decision} does not match {expected}"
            )
        words = np.asarray(game.legal_action_mask, dtype=np.uint64)
        legal_mask_words[step] = words
        action = int(trajectory.actions[step])
        if not int(words[action // 64]) & (1 << (action % 64)):
            raise TrajectoryReplayError(f"step {step}: action {action} is illegal")

        actor = expected[0]
        game.observe_into(
            actor,
            tile_obs[step],
            melds[step],
            river[step],
            meta[step],
        )
        category = int(decision_categories(meta[step : step + 1])[0])
        if category != int(trajectory.categories[step]):
            raise TrajectoryReplayError(
                f"step {step}: category {category} != {trajectory.categories[step]}"
            )
        event_lengths[step] = game.events_into(actor, events[step])
        game.oracle_tile_counts_into(oracle_tiles[step])
        game.step_into(action, transition)
        score_deltas[step] = transition[5:9].astype(np.float32) / 10_000.0
        terminal = bool(transition[11])
        if terminal != (step + 1 == steps):
            state = "early terminal" if terminal else "trajectory ended before terminal"
            raise TrajectoryReplayError(f"step {step}: {state}")

    scores = np.asarray(game.scores(), dtype=np.int32)
    if not np.array_equal(scores, trajectory.terminal_scores):
        raise TrajectoryReplayError(
            f"terminal scores {scores.tolist()} != {trajectory.terminal_scores.tolist()}"
        )
    ranks = _seat_ranks(game.rankings())
    if not np.array_equal(ranks, trajectory.terminal_ranks):
        raise TrajectoryReplayError(
            f"terminal ranks {ranks.tolist()} != {trajectory.terminal_ranks.tolist()}"
        )
    if game.termination_reason != trajectory.termination_reason:
        raise TrajectoryReplayError(
            f"termination reason {game.termination_reason} "
            f"!= {trajectory.termination_reason}"
        )
    returns_to_go = np.cumsum(score_deltas[::-1], axis=0)[::-1].copy()
    return ReplayedTrajectory(
        trajectory=trajectory,
        legal_mask_words=legal_mask_words,
        tile_obs=tile_obs,
        melds=melds,
        river=river,
        meta=meta,
        events=events,
        event_lengths=event_lengths,
        oracle_tiles=oracle_tiles,
        score_deltas=score_deltas,
        returns_to_go=returns_to_go,
    )


def encode_trajectory_shard(trajectories: Iterable[CompactTrajectory]) -> bytes:
    records = [trajectory.to_bytes() for trajectory in trajectories]
    output = bytearray(_SHARD_HEADER.pack(SHARD_MAGIC, TRAJECTORY_FORMAT_VERSION, 0, len(records)))
    for record in records:
        output += _LENGTH.pack(len(record))
        output += record
    return bytes(output)


def decode_trajectory_shard(data: bytes | bytearray | memoryview) -> tuple[CompactTrajectory, ...]:
    view = memoryview(data)
    if len(view) < _SHARD_HEADER.size:
        raise TrajectoryFormatError("trajectory shard is shorter than its header")
    magic, version, _reserved, count = _SHARD_HEADER.unpack_from(view)
    if magic != SHARD_MAGIC or version != TRAJECTORY_FORMAT_VERSION:
        raise TrajectoryFormatError("invalid or unsupported trajectory shard")
    cursor = _SHARD_HEADER.size
    trajectories: list[CompactTrajectory] = []
    for index in range(count):
        if cursor + _LENGTH.size > len(view):
            raise TrajectoryFormatError(f"trajectory shard ends before record {index}")
        (length,) = _LENGTH.unpack_from(view, cursor)
        cursor += _LENGTH.size
        end = cursor + length
        if end > len(view):
            raise TrajectoryFormatError(f"trajectory shard truncates record {index}")
        trajectories.append(CompactTrajectory.from_bytes(view[cursor:end]))
        cursor = end
    if cursor != len(view):
        raise TrajectoryFormatError("trajectory shard has trailing bytes")
    return tuple(trajectories)


def split_trajectories(
    trajectories: Sequence[CompactTrajectory],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[CompactTrajectory, ...], tuple[CompactTrajectory, ...]]:
    """Split whole games so no observations from one game cross the boundary."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    count = len(trajectories)
    validation_count = int(round(count * validation_fraction))
    if validation_fraction > 0 and count > 1:
        validation_count = min(max(validation_count, 1), count - 1)
    random = np.random.default_rng(seed)
    order = random.permutation(count)
    validation_indices = set(order[:validation_count].tolist())
    train = tuple(
        trajectory
        for index, trajectory in enumerate(trajectories)
        if index not in validation_indices
    )
    validation = tuple(
        trajectory
        for index, trajectory in enumerate(trajectories)
        if index in validation_indices
    )
    return train, validation


__all__ = [
    "CompactTrajectory",
    "ReplayedTrajectory",
    "TRAJECTORY_FORMAT_VERSION",
    "TrajectoryBuilder",
    "TrajectoryFormatError",
    "TrajectoryReplayError",
    "decode_trajectory_shard",
    "encode_trajectory_shard",
    "replay_trajectory",
    "split_trajectories",
]
