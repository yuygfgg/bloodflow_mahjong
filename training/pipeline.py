"""CUDA Actor execution and high-throughput mixed-opponent collection."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import bloodflow_mahjong as bm

from .model import ACTION_SPACE_SIZE, BloodFlowTransformer, TransformerConfig
from .observation import bucket_history_width, unpack_action_masks
from .policy_pool import ReplaySource, decision_categories
from .trajectory import CompactTrajectory, TrajectoryBuilder


CollectionProgress = Callable[[int, int, float], None]

# Execution ordering can affect BF16 tie-breaking even with identical weights
# and seeds. Persistent run and cache identities include this value.
POLICY_EXECUTION_VERSION = 2


@dataclass(frozen=True)
class CollectionConfig:
    envs: int = 512
    history: int = 192
    inference_batch_size: int = 128
    maximum_steps_per_game: int = 4096

    def __post_init__(self) -> None:
        if (
            self.envs <= 0
            or self.history <= 0
            or self.inference_batch_size <= 0
            or self.maximum_steps_per_game <= 0
        ):
            raise ValueError("collection sizes must be positive")
        if self.history > int(bm.EVENT_HISTORY_CAPACITY):
            raise ValueError("history exceeds engine event capacity")


@dataclass
class EngineBuffers:
    batch: Any
    tile_obs: np.ndarray
    melds: np.ndarray
    river: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray
    masks: np.ndarray
    legal: np.ndarray
    records: np.ndarray
    actions: np.ndarray

    @classmethod
    def create(cls, batch_size: int, history: int = 192) -> EngineBuffers:
        return cls.for_batch(bm.Batch(batch_size, seed=1), history)

    @classmethod
    def for_batch(cls, batch: Any, history: int = 192) -> EngineBuffers:
        batch_size = len(batch)
        return cls(
            batch=batch,
            tile_obs=np.empty((batch_size, 10, 27), dtype=np.uint8),
            melds=np.empty((batch_size, 4, 4, 3), dtype=np.uint8),
            river=np.empty((batch_size, 108, 2), dtype=np.uint8),
            meta=np.empty((batch_size, 34), dtype=np.int32),
            events=np.empty((batch_size, history, 8), dtype=np.int32),
            event_lengths=np.empty(batch_size, dtype=np.uint16),
            masks=np.empty((batch_size, 2), dtype=np.uint64),
            legal=np.empty((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_),
            records=np.empty((batch_size, 12), dtype=np.int64),
            actions=np.empty(batch_size, dtype=np.uint8),
        )

    def refresh_legal(self, rows: np.ndarray | None = None) -> None:
        if rows is None:
            unpack_action_masks(self.masks, out=self.legal)
        elif len(rows):
            self.legal[rows] = unpack_action_masks(self.masks[rows])

    def remove_rows(
        self, rows: np.ndarray
    ) -> tuple[EngineBuffers, np.ndarray]:
        """Remove terminal rows and return buffers plus the survivor order.

        ``step_and_observe_history_into`` has populated every observation
        buffer before terminal rows are removed.  Re-running ``observe`` for
        the survivors needlessly traverses the same Rust game states and
        re-encodes their event histories. Rust uses swap-removal so each
        terminal row moves at most one large game state. The returned order
        maps every new row to its old row and keeps Python metadata aligned.
        Only the currently needed history prefix is copied; the remaining
        capacity is left uninitialized for the next fused step.
        """

        rows = np.asarray(rows, dtype=np.int64)
        if (
            rows.ndim != 1
            or np.any((rows < 0) | (rows >= len(self.batch)))
            or np.any(rows[:-1] >= rows[1:])
        ):
            raise ValueError(
                "removed rows must be valid and strictly increasing"
            )
        indices = np.ascontiguousarray(rows, dtype=np.uint32)
        original_rows = list(range(len(self.batch)))
        for row in reversed(rows.tolist()):
            original_rows[row] = original_rows[-1]
            original_rows.pop()
        order = np.asarray(original_rows, dtype=np.int64)
        lengths = np.ascontiguousarray(self.event_lengths[order])
        width = bucket_history_width(lengths, self.events.shape[1])
        event_prefix = np.ascontiguousarray(self.events[order, :width])
        if width == self.events.shape[1]:
            events = event_prefix
        else:
            events = np.empty(
                (len(order), self.events.shape[1], self.events.shape[2]),
                dtype=self.events.dtype,
            )
            events[:, :width] = event_prefix
        compacted = type(self)(
            batch=self.batch,
            tile_obs=np.ascontiguousarray(self.tile_obs[order]),
            melds=np.ascontiguousarray(self.melds[order]),
            river=np.ascontiguousarray(self.river[order]),
            meta=np.ascontiguousarray(self.meta[order]),
            events=events,
            event_lengths=lengths,
            masks=np.ascontiguousarray(self.masks[order]),
            legal=np.ascontiguousarray(self.legal[order]),
            records=np.ascontiguousarray(self.records[order]),
            actions=np.ascontiguousarray(self.actions[order]),
        )
        if self.batch.remove_indices_swap(indices) != original_rows:
            raise RuntimeError("engine compaction returned an invalid row order")
        return compacted, order

    def observe(self, history_seat_masks: np.ndarray | None = None) -> None:
        self.batch.observe_into(self.tile_obs, self.melds, self.river, self.meta)
        self.batch.legal_action_masks_into(self.masks)
        if history_seat_masks is None:
            self.batch.events_into(self.events, self.event_lengths)
        else:
            history_seat_masks = np.asarray(history_seat_masks)
            if history_seat_masks.shape != (len(self.batch),):
                raise ValueError("history_seat_masks must have shape [batch]")
            if history_seat_masks.dtype != np.uint8:
                raise TypeError("history_seat_masks must use uint8")
            self.batch.events_masked_into(
                np.ascontiguousarray(history_seat_masks),
                self.events,
                self.event_lengths,
            )
        self.refresh_legal()


def _bucket_inference_rows(rows: np.ndarray, minimum: int = 32) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows):
        raise ValueError("inference rows must be a non-empty vector")
    bucket = max(minimum, 1 << (len(rows) - 1).bit_length())
    return rows if bucket == len(rows) else np.pad(rows, (0, bucket - len(rows)), mode="edge")


def _lineup_history_seat_masks(sources: np.ndarray) -> np.ndarray:
    sources = np.asarray(sources)
    if sources.ndim != 2 or sources.shape[1] != 4:
        raise ValueError("lineup sources must have shape [games, 4]")
    model_seats = np.isin(
        sources,
        (int(ReplaySource.CURRENT), int(ReplaySource.SELF_PLAY)),
    )
    seat_bits = np.left_shift(
        np.uint8(1), np.arange(4, dtype=np.uint8)
    )
    return np.bitwise_or.reduce(
        np.where(model_seats, seat_bits, np.uint8(0)), axis=1
    ).astype(np.uint8, copy=False)


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


class _PinnedPolicyStager:
    """Reusable pinned host and device buffers for one rollout stream."""

    def __init__(self, device: torch.device, history: int) -> None:
        if device.type != "cuda" or history <= 0:
            raise ValueError("pinned policy staging requires CUDA and history")
        self.device = device
        self.history = history
        self.capacity = 0
        self.host: tuple[torch.Tensor, ...] = ()
        self.host_numpy: tuple[np.ndarray, ...] = ()
        self.staged: tuple[torch.Tensor, ...] = ()

    def _allocate(self, required: int) -> None:
        capacity = max(required, 32, 2 * self.capacity)
        specifications = (
            ((capacity, 10, 27), torch.uint8),
            ((capacity, 4, 4, 3), torch.uint8),
            ((capacity, 34), torch.int32),
            ((capacity, self.history, 8), torch.int32),
            ((capacity,), torch.int64),
            ((capacity, ACTION_SPACE_SIZE), torch.bool),
        )
        self.host = tuple(
            torch.empty(shape, dtype=dtype, pin_memory=True)
            for shape, dtype in specifications
        )
        self.host_numpy = tuple(value.numpy() for value in self.host)
        self.staged = tuple(
            torch.empty(shape, dtype=dtype, device=self.device)
            for shape, dtype in specifications
        )
        self.capacity = capacity

    def stage(
        self, buffers: EngineBuffers, rows: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        count = len(rows)
        if not count:
            raise ValueError("cannot stage an empty inference batch")
        if buffers.events.shape[1] != self.history:
            raise ValueError("inference history capacity changed")
        if count > self.capacity:
            self._allocate(count)
        sources = (
            buffers.tile_obs,
            buffers.melds,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
            buffers.legal,
        )
        for index, (source, target) in enumerate(
            zip(sources, self.host_numpy)
        ):
            if index == 4:
                target[:count] = source[rows]
            else:
                np.take(source, rows, axis=0, out=target[:count])
        for host, staged in zip(self.host, self.staged):
            staged[:count].copy_(host[:count], non_blocking=True)
        return tuple(value[:count] for value in self.staged)


def _launch_policy_actions(
    model: BloodFlowTransformer,
    buffers: EngineBuffers,
    rows: np.ndarray,
    device: torch.device,
    *,
    inference_batch_size: int,
    stager: _PinnedPolicyStager | None = None,
) -> torch.Tensor:
    """Launch chunked policy inference with one staged transfer per step."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or inference_batch_size <= 0:
        raise ValueError("policy inference rows and batch size must be valid")
    chunks: list[tuple[int, int, int, int, np.ndarray]] = []
    staged_rows: list[np.ndarray] = []
    offset = 0
    for start in range(0, len(rows), inference_batch_size):
        chunk = rows[start : start + inference_batch_size]
        inference_rows = _bucket_inference_rows(chunk)
        lengths = buffers.event_lengths[inference_rows]
        width = bucket_history_width(lengths, buffers.events.shape[1])
        chunks.append(
            (offset, len(inference_rows), len(chunk), width, inference_rows)
        )
        staged_rows.append(inference_rows)
        offset += len(inference_rows)

    staged: tuple[torch.Tensor, ...] | None = None
    if device.type == "cuda":
        if stager is None:
            stager = _PinnedPolicyStager(device, buffers.events.shape[1])
        staged = stager.stage(buffers, np.concatenate(staged_rows))

    actions: list[torch.Tensor] = []
    with torch.inference_mode(), _autocast(device):
        for offset, padded, valid, width, inference_rows in chunks:
            if staged is None:
                lengths = buffers.event_lengths[inference_rows].astype(np.int64)
                inputs = (
                    torch.as_tensor(buffers.tile_obs[inference_rows], device=device),
                    torch.as_tensor(buffers.melds[inference_rows], device=device),
                    torch.as_tensor(buffers.meta[inference_rows], device=device),
                    torch.as_tensor(
                        buffers.events[inference_rows, :width], device=device
                    ),
                    torch.as_tensor(lengths, device=device),
                    torch.as_tensor(buffers.legal[inference_rows], device=device),
                )
            else:
                stop = offset + padded
                inputs = (
                    staged[0][offset:stop],
                    staged[1][offset:stop],
                    staged[2][offset:stop],
                    staged[3][offset:stop, :width],
                    staged[4][offset:stop],
                    staged[5][offset:stop],
                )
            logits = model(*inputs).logits[:valid]
            selected = logits.argmax(dim=-1).to(torch.uint8)
            actions.append(
                torch.where(
                    torch.isfinite(logits).all(dim=-1),
                    selected,
                    torch.full_like(selected, np.iinfo(np.uint8).max),
                )
            )
    return actions[0] if len(actions) == 1 else torch.cat(actions)


def clone_policy(
    model: BloodFlowTransformer, device: torch.device
) -> BloodFlowTransformer:
    result = copy.deepcopy(model).to(device).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def save_policy(path: Path, model: BloodFlowTransformer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"model_config": model.config.__dict__, "model": model.state_dict()},
        temporary,
    )
    temporary.replace(path)


def load_policy(
    path: str | Path, device: torch.device, *, frozen: bool = True
) -> BloodFlowTransformer:
    payload = torch.load(path, map_location=device, weights_only=False)
    if set(payload) != {"model_config", "model"}:
        raise ValueError(f"{path} is not an Actor-only policy checkpoint")
    model = BloodFlowTransformer(TransformerConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen)
    return model


def _safe_rule_actions(
    actions: np.ndarray,
    legal: np.ndarray,
    tile_obs: np.ndarray,
    rows: np.ndarray,
) -> None:
    if not len(rows):
        return
    selected_actions = actions[rows]
    selected_legal = legal[rows]
    optional = np.isin(
        selected_actions, (bm.ACTION_PONG, bm.ACTION_EXPOSED_KONG)
    ) & selected_legal[:, bm.ACTION_PASS]
    actions[rows[optional]] = bm.ACTION_PASS

    discard_start = int(bm.ACTION_DISCARD_OFFSET)
    discard_stop = int(bm.ACTION_HU)
    discard_rows = rows[
        (~optional)
        & (selected_actions >= discard_start)
        & (selected_actions < discard_stop)
    ]
    if not len(discard_rows):
        return
    exposure = tile_obs[discard_rows, 2:10].sum(axis=1, dtype=np.int16)
    legal_discards = legal[discard_rows, discard_start:discard_stop]
    safest = np.where(legal_discards, exposure, -1).argmax(axis=1)
    current = actions[discard_rows].astype(np.int64) - discard_start
    positions = np.arange(len(discard_rows))
    improve = exposure[positions, safest] >= exposure[positions, current] + 2
    actions[discard_rows[improve]] = (safest[improve] + discard_start).astype(
        np.uint8
    )


@dataclass(frozen=True)
class PolicyLineups:
    sources: np.ndarray
    focal_seats: np.ndarray

    def __post_init__(self) -> None:
        if self.sources.ndim != 2 or self.sources.shape[1] != 4:
            raise ValueError("lineup sources must have shape [games, 4]")
        if self.focal_seats.shape != (len(self.sources),):
            raise ValueError("focal seats must have shape [games]")


@dataclass(frozen=True)
class CollectionResult:
    trajectories: tuple[CompactTrajectory, ...]
    focal_seats: np.ndarray
    environment_steps: int
    policy_actions: int
    elapsed_seconds: float
    source_counts: dict[str, int]
    opponent_seat_counts: dict[str, int]


class TrajectoryCollector:
    """Collect a rotating focal Actor against rules and a frozen opponent."""

    def __init__(
        self,
        config: CollectionConfig,
        actor: BloodFlowTransformer,
        device: torch.device,
        *,
        seed: int,
        self_play_fraction: float = 0.0,
        self_play_actor: BloodFlowTransformer | None = None,
    ) -> None:
        if not 0.0 <= self_play_fraction <= 2.0 / 3.0:
            raise ValueError("self_play_fraction must be in [0, 2/3]")
        self.config = config
        self.actor = actor.eval()
        self.self_play_actor = (
            self.actor if self_play_actor is None else self_play_actor.eval()
        )
        if self.self_play_actor.config != self.actor.config:
            raise ValueError("self-play Actor config must match the focal Actor")
        self.device = device
        self.next_seed = int(seed)
        self.lineup_cursor = 0
        self.lineup_random = np.random.default_rng(
            self._seed(int(seed) ^ 0x5E1F_504C_4159)
        )
        self.self_play_fraction = float(self_play_fraction)
        self.inference_stager = (
            _PinnedPolicyStager(device, config.history)
            if device.type == "cuda"
            else None
        )

    @staticmethod
    def _seed(value: int) -> int:
        return (int(value) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

    def _lineups(self, games: int, *, focal_offset: int | None = None) -> PolicyLineups:
        if focal_offset is None:
            focal_offset = self.lineup_cursor
            self.lineup_cursor = (self.lineup_cursor + games) % 4
        focal = (
            np.arange(games, dtype=np.uint64) + int(focal_offset)
        ).astype(np.uint8) % 4
        sources = np.empty((games, 4), dtype=np.uint8)
        for row in range(games):
            for seat in range(4):
                sources[row, seat] = int(
                    ReplaySource.RULE_SAFE
                    if (row + seat) % 2
                    else ReplaySource.RULE_FAST
                )
        sources[np.arange(games), focal] = int(ReplaySource.CURRENT)
        expected_self_play_seats = 3.0 * self.self_play_fraction
        guaranteed = int(np.floor(expected_self_play_seats))
        fractional = expected_self_play_seats - guaranteed
        for row in range(games):
            self_play_seats = guaranteed
            if fractional and self.lineup_random.random() < fractional:
                self_play_seats += 1
            if not self_play_seats:
                continue
            opponents = np.flatnonzero(np.arange(4) != int(focal[row]))
            selected = self.lineup_random.choice(
                opponents, size=self_play_seats, replace=False
            )
            sources[row, selected] = int(ReplaySource.SELF_PLAY)
        return PolicyLineups(sources, focal)

    def _launch_model_actions(
        self,
        model: BloodFlowTransformer,
        buffers: EngineBuffers,
        rows: np.ndarray,
    ) -> torch.Tensor:
        if not len(rows):
            return torch.empty(0, dtype=torch.uint8, device=self.device)
        return _launch_policy_actions(
            model,
            buffers,
            rows,
            self.device,
            inference_batch_size=self.config.inference_batch_size,
            stager=self.inference_stager,
        )

    def _actions(
        self, buffers: EngineBuffers, lineups: PolicyLineups
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        actors = buffers.meta[:, 1].astype(np.int64)
        if np.any((actors < 0) | (actors >= 4)):
            raise RuntimeError("collector encountered a terminal row before reset")
        rows = np.arange(len(actors))
        sources = lineups.sources[rows, actors].astype(np.uint8)
        current_rows = np.flatnonzero(sources == int(ReplaySource.CURRENT))
        self_play_rows = np.flatnonzero(sources == int(ReplaySource.SELF_PLAY))
        if self.self_play_actor is self.actor:
            model_rows = np.flatnonzero(
                np.isin(
                    sources,
                    (int(ReplaySource.CURRENT), int(ReplaySource.SELF_PLAY)),
                )
            )
            pending_model_actions = (
                self._launch_model_actions(self.actor, buffers, model_rows)
                if len(model_rows)
                else None
            )
            pending_self_play_actions = None
        else:
            model_rows = current_rows
            pending_model_actions = (
                self._launch_model_actions(self.actor, buffers, current_rows)
                if len(current_rows)
                else None
            )
            pending_self_play_actions = (
                self._launch_model_actions(
                    self.self_play_actor, buffers, self_play_rows
                )
                if len(self_play_rows)
                else None
            )
        model_count = len(current_rows) + len(self_play_rows)

        actions = buffers.actions
        rule_enabled = np.isin(
            sources,
            (int(ReplaySource.RULE_FAST), int(ReplaySource.RULE_SAFE)),
        ).astype(np.uint8)
        if np.any(rule_enabled):
            buffers.batch.simple_rule_actions_masked_into(rule_enabled, actions)
            safe_rows = np.flatnonzero(sources == int(ReplaySource.RULE_SAFE))
            _safe_rule_actions(actions, buffers.legal, buffers.tile_obs, safe_rows)
        if len(model_rows):
            model_actions = pending_model_actions.cpu().numpy()
            if np.any(model_actions == np.iinfo(np.uint8).max):
                raise RuntimeError("Actor produced non-finite logits during collection")
            actions[model_rows] = model_actions
        if pending_self_play_actions is not None:
            model_actions = pending_self_play_actions.cpu().numpy()
            if np.any(model_actions == np.iinfo(np.uint8).max):
                raise RuntimeError("self-play Actor produced non-finite logits")
            actions[self_play_rows] = model_actions
        if not buffers.legal[rows, actions.astype(np.int64)].all():
            raise RuntimeError("collector selected an illegal action")
        return actions, decision_categories(buffers.meta), sources, model_count

    @staticmethod
    def _finish(
        builder: TrajectoryBuilder,
        score_delta: np.ndarray,
        meta: np.ndarray,
    ) -> CompactTrajectory:
        scores = 10_000 + score_delta
        ranking = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
        reason = (
            int(bm.TERMINATION_WALL_EXHAUSTED)
            if int(meta[4]) == 0
            else int(bm.TERMINATION_THREE_PLAYERS_BANKRUPT)
        )
        return builder.finish(
            terminal_scores=scores,
            ranking_order=ranking,
            termination_reason=reason,
        )

    def collect_seeded(
        self,
        seeds: np.ndarray,
        *,
        focal_offset: int = 0,
        on_progress: CollectionProgress | None = None,
    ) -> CollectionResult:
        """Collect one exact non-replenished batch in supplied seed order."""

        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        if seeds.ndim != 1 or not len(seeds):
            raise ValueError("seeded collection needs a non-empty seed vector")
        if len(seeds) > self.config.envs:
            raise ValueError("seeded collection exceeds configured envs")
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("seeded collection requires unique seeds")

        buffers = EngineBuffers.create(len(seeds), self.config.history)
        reset = np.ones(len(seeds), dtype=np.uint8)
        lineups = self._lineups(len(seeds), focal_offset=focal_offset)
        history_masks = _lineup_history_seat_masks(lineups.sources)
        builders = [
            TrajectoryBuilder(int(seed), int(bm.Game(seed=int(seed)).exchange_direction))
            for seed in seeds
        ]
        original_rows = np.arange(len(seeds), dtype=np.int64)
        cumulative = np.zeros((len(seeds), 4), dtype=np.int64)
        step_counts = np.zeros(len(seeds), dtype=np.int32)
        buffers.batch.reset_and_observe_history_into(
            reset,
            seeds,
            history_masks,
            buffers.masks,
            buffers.tile_obs,
            buffers.melds,
            buffers.river,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
        )
        buffers.refresh_legal()

        completed: dict[int, tuple[CompactTrajectory, int]] = {}
        environment_steps = 0
        policy_actions = 0
        source_counts = np.zeros(len(ReplaySource), dtype=np.int64)
        opponent_seat_counts = np.bincount(
            lineups.sources[lineups.sources != int(ReplaySource.CURRENT)],
            minlength=len(ReplaySource),
        )
        started = time.perf_counter()
        while builders:
            actors = buffers.meta[:, 1].astype(np.int64)
            phases = buffers.meta[:, 0].astype(np.int64)
            actions, categories, sources, model_count = self._actions(buffers, lineups)
            legal_counts = buffers.legal.sum(axis=1)
            for row, builder in enumerate(builders):
                builder.append(
                    action=int(actions[row]),
                    actor=int(actors[row]),
                    phase=int(phases[row]),
                    category=int(categories[row]),
                    source=int(sources[row]),
                    legal_count=int(legal_counts[row]),
                )
            source_counts += np.bincount(sources, minlength=len(ReplaySource))
            buffers.batch.step_and_observe_history_into(
                actions,
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
            cumulative += buffers.records[:, 5:9]
            step_counts += 1
            environment_steps += len(builders)
            policy_actions += model_count
            if np.any(step_counts > self.config.maximum_steps_per_game):
                raise RuntimeError("seeded collection exceeded the game step limit")
            terminal = buffers.records[:, 11].astype(np.bool_)
            if not np.any(terminal):
                continue
            for row in np.flatnonzero(terminal):
                original = int(original_rows[row])
                completed[original] = (
                    self._finish(builders[row], cumulative[row], buffers.meta[row]),
                    int(lineups.focal_seats[row]),
                )
            if np.any(terminal) and on_progress is not None:
                on_progress(len(completed), environment_steps, time.perf_counter() - started)
            terminal_rows = np.flatnonzero(terminal)
            if len(terminal_rows) == len(builders):
                break
            buffers, keep = buffers.remove_rows(terminal_rows)
            lineups = PolicyLineups(
                lineups.sources[keep].copy(), lineups.focal_seats[keep].copy()
            )
            history_masks = history_masks[keep].copy()
            builders = [builders[int(row)] for row in keep]
            original_rows = original_rows[keep].copy()
            cumulative = cumulative[keep].copy()
            step_counts = step_counts[keep].copy()

        ordered = [completed[index] for index in range(len(seeds))]
        return CollectionResult(
            trajectories=tuple(value[0] for value in ordered),
            focal_seats=np.asarray([value[1] for value in ordered], dtype=np.int8),
            environment_steps=environment_steps,
            policy_actions=policy_actions,
            elapsed_seconds=time.perf_counter() - started,
            source_counts={
                source.label: int(source_counts[int(source)]) for source in ReplaySource
            },
            opponent_seat_counts={
                source.label: int(opponent_seat_counts[int(source)])
                for source in ReplaySource
            },
        )

    def collect(
        self,
        games: int,
        *,
        on_progress: CollectionProgress | None = None,
    ) -> CollectionResult:
        """Collect complete games with immediate replacement of terminal rows."""

        if games <= 0:
            raise ValueError("games must be positive")
        envs = min(self.config.envs, games)
        buffers = EngineBuffers.create(envs, self.config.history)
        rows = np.arange(envs, dtype=np.int64)
        reset_flags = np.ones(envs, dtype=np.uint8)
        reset_seeds = np.asarray(
            [self._seed(self.next_seed + offset) for offset in range(envs)],
            dtype=np.uint64,
        )
        self.next_seed += envs
        lineups = self._lineups(envs)
        history_masks = _lineup_history_seat_masks(lineups.sources)
        builders = [
            TrajectoryBuilder(int(seed), int(bm.Game(seed=int(seed)).exchange_direction))
            for seed in reset_seeds
        ]
        cumulative = np.zeros((envs, 4), dtype=np.int64)
        step_counts = np.zeros(envs, dtype=np.int32)
        buffers.batch.reset_and_observe_history_into(
            reset_flags,
            reset_seeds,
            history_masks,
            buffers.masks,
            buffers.tile_obs,
            buffers.melds,
            buffers.river,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
        )
        reset_flags.fill(0)
        buffers.refresh_legal()

        completed: list[CompactTrajectory] = []
        focal_seats: list[int] = []
        environment_steps = 0
        policy_actions = 0
        source_counts = np.zeros(len(ReplaySource), dtype=np.int64)
        opponent_seat_counts = np.zeros(len(ReplaySource), dtype=np.int64)
        started = time.perf_counter()
        while len(completed) < games:
            actors = buffers.meta[:, 1].astype(np.int64)
            phases = buffers.meta[:, 0].astype(np.int64)
            actions, categories, sources, model_count = self._actions(buffers, lineups)
            legal_counts = buffers.legal.sum(axis=1)
            for row in rows:
                builders[row].append(
                    action=int(actions[row]),
                    actor=int(actors[row]),
                    phase=int(phases[row]),
                    category=int(categories[row]),
                    source=int(sources[row]),
                    legal_count=int(legal_counts[row]),
                )
            source_counts += np.bincount(sources, minlength=len(ReplaySource))
            buffers.batch.step_and_observe_history_into(
                actions,
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
            cumulative += buffers.records[:, 5:9]
            step_counts += 1
            environment_steps += envs
            policy_actions += model_count
            if np.any(step_counts > self.config.maximum_steps_per_game):
                stuck = np.flatnonzero(
                    step_counts > self.config.maximum_steps_per_game
                ).tolist()
                raise RuntimeError(f"games exceeded the step limit: {stuck[:8]}")
            terminal_rows = np.flatnonzero(buffers.records[:, 11])
            if not len(terminal_rows):
                continue
            for row in terminal_rows:
                if len(completed) >= games:
                    break
                completed.append(
                    self._finish(builders[row], cumulative[row], buffers.meta[row])
                )
                focal_seats.append(int(lineups.focal_seats[row]))
                opponents = lineups.sources[row][
                    lineups.sources[row] != int(ReplaySource.CURRENT)
                ]
                opponent_seat_counts += np.bincount(
                    opponents, minlength=len(ReplaySource)
                )
            if on_progress is not None:
                on_progress(len(completed), environment_steps, time.perf_counter() - started)
            if len(completed) >= games:
                break

            reset_count = len(terminal_rows)
            reset_lineups = self._lineups(reset_count)
            reset_history_masks = _lineup_history_seat_masks(
                reset_lineups.sources
            )
            new_seeds = np.asarray(
                [self._seed(self.next_seed + offset) for offset in range(reset_count)],
                dtype=np.uint64,
            )
            self.next_seed += reset_count
            for position, row in enumerate(terminal_rows):
                lineups.sources[row] = reset_lineups.sources[position]
                lineups.focal_seats[row] = reset_lineups.focal_seats[position]
                history_masks[row] = reset_history_masks[position]
                seed = int(new_seeds[position])
                builders[row] = TrajectoryBuilder(
                    seed, int(bm.Game(seed=seed).exchange_direction)
                )
                cumulative[row] = 0
                step_counts[row] = 0
                reset_flags[row] = 1
                reset_seeds[row] = seed
            buffers.batch.reset_and_observe_history_into(
                reset_flags,
                reset_seeds,
                history_masks,
                buffers.masks,
                buffers.tile_obs,
                buffers.melds,
                buffers.river,
                buffers.meta,
                buffers.events,
                buffers.event_lengths,
            )
            reset_flags[terminal_rows] = 0
            buffers.refresh_legal(terminal_rows)

        return CollectionResult(
            trajectories=tuple(completed),
            focal_seats=np.asarray(focal_seats, dtype=np.int8),
            environment_steps=environment_steps,
            policy_actions=policy_actions,
            elapsed_seconds=time.perf_counter() - started,
            source_counts={
                source.label: int(source_counts[int(source)]) for source in ReplaySource
            },
            opponent_seat_counts={
                source.label: int(opponent_seat_counts[int(source)])
                for source in ReplaySource
            },
        )


__all__ = [
    "CollectionConfig",
    "CollectionResult",
    "EngineBuffers",
    "PolicyLineups",
    "TrajectoryCollector",
    "_autocast",
    "_bucket_inference_rows",
    "_safe_rule_actions",
    "clone_policy",
    "load_policy",
    "save_policy",
]
