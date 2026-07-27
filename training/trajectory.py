"""Minimal complete-game records and strict viewer-state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import bloodflow_mahjong as bm

from .policy_pool import CATEGORY_COUNT, ReplaySource, decision_categories


class TrajectoryReplayError(ValueError):
    """A seed/action replay disagrees with the recorded game."""


def _vector(value: object, dtype: np.dtype[object], name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def seat_ranks(ranking_order: Sequence[int]) -> np.ndarray:
    order = np.asarray(ranking_order, dtype=np.uint8)
    if order.shape != (4,) or sorted(order.tolist()) != [0, 1, 2, 3]:
        raise ValueError("ranking_order must be a permutation of seats 0..3")
    ranks = np.empty(4, dtype=np.uint8)
    ranks[order] = np.arange(1, 5, dtype=np.uint8)
    return ranks


@dataclass(frozen=True)
class CompactTrajectory:
    """One in-memory game, retaining only fields used by policy improvement."""

    seed: int
    exchange_direction: int
    actions: np.ndarray
    actors: np.ndarray
    phases: np.ndarray
    categories: np.ndarray
    sources: np.ndarray
    legal_counts: np.ndarray
    terminal_scores: np.ndarray
    terminal_ranks: np.ndarray
    termination_reason: int

    def __post_init__(self) -> None:
        arrays = {
            "actions": _vector(self.actions, np.dtype(np.uint8), "actions"),
            "actors": _vector(self.actors, np.dtype(np.uint8), "actors"),
            "phases": _vector(self.phases, np.dtype(np.uint8), "phases"),
            "categories": _vector(
                self.categories, np.dtype(np.uint8), "categories"
            ),
            "sources": _vector(self.sources, np.dtype(np.uint8), "sources"),
            "legal_counts": _vector(
                self.legal_counts, np.dtype(np.uint8), "legal_counts"
            ),
        }
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("trajectory step arrays must have one positive length")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

        scores = _vector(self.terminal_scores, np.dtype(np.int32), "terminal_scores")
        ranks = _vector(self.terminal_ranks, np.dtype(np.uint8), "terminal_ranks")
        if scores.shape != (4,) or int(scores.sum(dtype=np.int64)) != 40_000:
            raise ValueError("terminal_scores must have shape [4] and sum to 40000")
        if ranks.shape != (4,) or sorted(ranks.tolist()) != [1, 2, 3, 4]:
            raise ValueError("terminal_ranks must be a permutation of 1..4")
        object.__setattr__(self, "terminal_scores", scores)
        object.__setattr__(self, "terminal_ranks", ranks)

        if not 0 <= int(self.seed) <= np.iinfo(np.uint64).max:
            raise ValueError("seed must fit uint64")
        if self.exchange_direction not in (1, 2, 3):
            raise ValueError("exchange_direction must be 1, 2, or 3")
        if np.any(self.actions >= int(bm.ACTION_SPACE_SIZE)):
            raise ValueError("trajectory contains an invalid action")
        if np.any(self.actors >= int(bm.PLAYER_COUNT)):
            raise ValueError("trajectory contains an invalid actor")
        if np.any(self.phases >= int(bm.PHASE_FINISHED)):
            raise ValueError("trajectory contains an invalid phase")
        if np.any(self.categories >= CATEGORY_COUNT):
            raise ValueError("trajectory contains an invalid category")
        valid_sources = np.asarray([int(value) for value in ReplaySource])
        if not np.isin(self.sources, valid_sources).all():
            raise ValueError("trajectory contains an invalid source")
        if np.any((self.legal_counts < 1) | (self.legal_counts > bm.ACTION_SPACE_SIZE)):
            raise ValueError("legal_counts must be in [1, ACTION_SPACE_SIZE]")
        if self.termination_reason not in (
            int(bm.TERMINATION_WALL_EXHAUSTED),
            int(bm.TERMINATION_THREE_PLAYERS_BANKRUPT),
        ):
            raise ValueError("trajectory has an invalid termination reason")

    def __len__(self) -> int:
        return len(self.actions)


class TrajectoryBuilder:
    def __init__(self, seed: int, exchange_direction: int) -> None:
        self.seed = int(seed)
        self.exchange_direction = int(exchange_direction)
        self._actions: list[int] = []
        self._actors: list[int] = []
        self._phases: list[int] = []
        self._categories: list[int] = []
        self._sources: list[int] = []
        self._legal_counts: list[int] = []

    def append(
        self,
        *,
        action: int,
        actor: int,
        phase: int,
        category: int,
        source: int,
        legal_count: int,
    ) -> None:
        self._actions.append(int(action))
        self._actors.append(int(actor))
        self._phases.append(int(phase))
        self._categories.append(int(category))
        self._sources.append(int(source))
        self._legal_counts.append(int(legal_count))

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
            legal_counts=np.asarray(self._legal_counts, dtype=np.uint8),
            terminal_scores=np.asarray(terminal_scores, dtype=np.int32),
            terminal_ranks=seat_ranks(ranking_order),
            termination_reason=int(termination_reason),
        )


@dataclass(frozen=True)
class ReplayedTrajectory:
    trajectory: CompactTrajectory
    legal_mask_words: np.ndarray
    tile_obs: np.ndarray
    melds: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray


def replay_trajectory(
    trajectory: CompactTrajectory, *, history_capacity: int = 192
) -> ReplayedTrajectory:
    """Strictly reconstruct all viewer inputs for an in-memory trajectory."""

    if not 0 < history_capacity <= int(bm.EVENT_HISTORY_CAPACITY):
        raise ValueError(
            f"history_capacity must be in 1..{bm.EVENT_HISTORY_CAPACITY}"
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
        (int(bm.RIVER_TILE_CAPACITY), int(bm.RIVER_FIELDS)), dtype=np.uint8
    )
    meta = np.empty((steps, int(bm.META_OBSERVATION_WIDTH)), dtype=np.int32)
    events = np.empty(
        (steps, history_capacity, int(bm.EVENT_RECORD_WIDTH)), dtype=np.int32
    )
    event_lengths = np.empty(steps, dtype=np.uint16)
    transition = np.empty(int(bm.STEP_RECORD_WIDTH), dtype=np.int64)

    for step in range(steps):
        expected = (int(trajectory.actors[step]), int(trajectory.phases[step]))
        if game.decision != expected:
            raise TrajectoryReplayError(
                f"step {step}: decision {game.decision} does not match {expected}"
            )
        words = np.asarray(game.legal_action_mask, dtype=np.uint64)
        legal_mask_words[step] = words
        legal_count = sum(int(word).bit_count() for word in words)
        if legal_count != int(trajectory.legal_counts[step]):
            raise TrajectoryReplayError(
                f"step {step}: legal count {legal_count} != "
                f"{trajectory.legal_counts[step]}"
            )
        action = int(trajectory.actions[step])
        if not int(words[action // 64]) & (1 << (action % 64)):
            raise TrajectoryReplayError(f"step {step}: action {action} is illegal")
        actor = expected[0]
        game.observe_into(
            actor, tile_obs[step], melds[step], river, meta[step]
        )
        category = int(decision_categories(meta[step : step + 1])[0])
        if category != int(trajectory.categories[step]):
            raise TrajectoryReplayError(
                f"step {step}: category {category} != {trajectory.categories[step]}"
            )
        event_lengths[step] = game.events_into(actor, events[step])
        game.step_into(action, transition)
        terminal = bool(transition[11])
        if terminal != (step + 1 == steps):
            state = "early terminal" if terminal else "trajectory ended before terminal"
            raise TrajectoryReplayError(f"step {step}: {state}")

    scores = np.asarray(game.scores(), dtype=np.int32)
    ranks = seat_ranks(game.rankings())
    if not np.array_equal(scores, trajectory.terminal_scores):
        raise TrajectoryReplayError("terminal scores do not match")
    if not np.array_equal(ranks, trajectory.terminal_ranks):
        raise TrajectoryReplayError("terminal ranks do not match")
    if int(game.termination_reason) != trajectory.termination_reason:
        raise TrajectoryReplayError("termination reason does not match")
    return ReplayedTrajectory(
        trajectory=trajectory,
        legal_mask_words=legal_mask_words,
        tile_obs=tile_obs,
        melds=melds,
        meta=meta,
        events=events,
        event_lengths=event_lengths,
    )


__all__ = [
    "CompactTrajectory",
    "ReplayedTrajectory",
    "TrajectoryBuilder",
    "TrajectoryReplayError",
    "replay_trajectory",
    "seat_ranks",
]
