"""Full-trajectory synthetic data collection and fixed-panel evaluation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

import bloodflow_mahjong as bm

from .model import ACTION_SPACE_SIZE, BloodFlowTransformer, TransformerConfig
from .observation import bucket_history_width, unpack_action_masks
from .policy_pool import (
    BehaviorSampler,
    CATEGORY_NAMES,
    PolicyDescriptor,
    PolicyLineupBatch,
    PolicyPool,
    ReplaySource,
    decision_categories,
)
from .trajectory import CompactTrajectory, TrajectoryBuilder


SEAT_BITS = np.asarray([1, 2, 4, 8], dtype=np.uint8)
ALL_SEATS_MASK = np.uint8(15)
LineupMode = Literal["mixed", "rules", "sl", "history"]


@dataclass(frozen=True)
class CollectionConfig:
    envs: int = 512
    history: int = 192
    maximum_steps_per_game: int = 4096

    def __post_init__(self) -> None:
        if self.envs <= 0 or self.history <= 0 or self.maximum_steps_per_game <= 0:
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
        return cls(
            batch=bm.Batch(batch_size, seed=1),
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


def _history_prefix(events: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    width = bucket_history_width(lengths, events.shape[1])
    return events[:, :width]


def _bucket_inference_rows(rows: np.ndarray, minimum: int = 32) -> np.ndarray:
    """Pad by repeating a row so inference uses a small set of batch shapes."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows):
        raise ValueError("inference rows must be a non-empty vector")
    bucket = max(minimum, 1 << (len(rows) - 1).bit_length())
    if bucket == len(rows):
        return rows
    return np.pad(rows, (0, bucket - len(rows)), mode="edge")


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def clone_policy(
    model: BloodFlowTransformer, device: torch.device
) -> BloodFlowTransformer:
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


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
        raise ValueError(f"{path} is not a current Actor-only policy checkpoint")
    model = BloodFlowTransformer(TransformerConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


class ExecutablePolicyPool:
    """Resolve control-plane policy IDs to CUDA Actor modules."""

    def __init__(
        self,
        actor: BloodFlowTransformer,
        reference: BloodFlowTransformer,
        device: torch.device,
    ) -> None:
        self.actor = actor
        self.reference = reference
        self.device = device
        self.historical: dict[str, BloodFlowTransformer] = {}

    def update_actor(self, actor: BloodFlowTransformer) -> None:
        self.actor = actor

    def register_snapshot(
        self, descriptor: PolicyDescriptor, model: BloodFlowTransformer | None = None
    ) -> None:
        if descriptor.source != ReplaySource.FROZEN_POLICY:
            raise ValueError("only frozen historical policies can be registered")
        if model is None:
            if descriptor.artifact is None:
                raise ValueError("historical policy has no artifact")
            model = load_policy(descriptor.artifact, self.device)
        else:
            model = clone_policy(model, self.device)
        self.historical[descriptor.policy_id] = model

    def sync(self, pool: PolicyPool) -> None:
        """Mirror retained history exactly, releasing evicted CUDA modules."""

        active = {descriptor.policy_id for descriptor in pool.history}
        for policy_id in list(self.historical):
            if policy_id not in active:
                del self.historical[policy_id]
        for descriptor in pool.history:
            if descriptor.policy_id not in self.historical:
                self.register_snapshot(descriptor)

    def resolve(self, policy_id: str) -> BloodFlowTransformer:
        if policy_id == "current":
            return self.actor
        if policy_id == "frozen_sl":
            return self.reference
        try:
            return self.historical[policy_id]
        except KeyError as error:
            raise KeyError(f"policy {policy_id!r} is not executable") from error


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
    optional_call = np.isin(
        selected_actions, (bm.ACTION_PONG, bm.ACTION_EXPOSED_KONG)
    ) & selected_legal[:, bm.ACTION_PASS]
    actions[rows[optional_call]] = bm.ACTION_PASS

    discard_start = int(bm.ACTION_DISCARD_OFFSET)
    discard_stop = int(bm.ACTION_HU)
    discard_rows = rows[
        (~optional_call)
        & (selected_actions >= discard_start)
        & (selected_actions < discard_stop)
    ]
    if not len(discard_rows):
        return
    exposure = tile_obs[discard_rows, 2:10].sum(axis=1, dtype=np.int16)
    legal_discards = legal[discard_rows, discard_start:discard_stop]
    safest_tiles = np.where(legal_discards, exposure, -1).argmax(axis=1)
    current_tiles = actions[discard_rows].astype(np.int64) - discard_start
    positions = np.arange(len(discard_rows))
    improve = exposure[positions, safest_tiles] >= exposure[positions, current_tiles] + 2
    actions[discard_rows[improve]] = (
        safest_tiles[improve] + discard_start
    ).astype(np.uint8)


@dataclass(frozen=True)
class CollectionResult:
    trajectories: tuple[CompactTrajectory, ...]
    focal_seats: np.ndarray
    environment_steps: int
    policy_actions: int
    elapsed_seconds: float
    source_counts: dict[str, int]


class FullTrajectoryCollector:
    """Generate stochastic complete games and retain all four policy views."""

    def __init__(
        self,
        config: CollectionConfig,
        policy_pool: PolicyPool,
        executables: ExecutablePolicyPool,
        behavior: BehaviorSampler,
        device: torch.device,
        *,
        seed: int,
    ) -> None:
        self.config = config
        self.policy_pool = policy_pool
        self.executables = executables
        self.behavior = behavior
        self.device = device
        self.next_seed = int(seed)

    @staticmethod
    def _seed(value: int) -> int:
        return (int(value) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

    def _lineups(self, games: int, mode: LineupMode) -> PolicyLineupBatch:
        if mode == "mixed":
            return self.policy_pool.sample_lineups(games, ensure_current=True)
        focal = np.arange(games, dtype=np.uint8) % 4
        ids = np.empty((games, 4), dtype=object)
        sources = np.empty((games, 4), dtype=np.uint8)
        versions = np.empty((games, 4), dtype=np.uint64)
        if mode == "rules":
            for row in range(games):
                for seat in range(4):
                    safe = (row + seat) % 2 == 1
                    ids[row, seat] = "rule_safe" if safe else "rule_fast"
                    sources[row, seat] = int(
                        ReplaySource.RULE_SAFE if safe else ReplaySource.RULE_FAST
                    )
                    versions[row, seat] = 0
        elif mode == "sl":
            ids.fill("frozen_sl")
            sources.fill(int(ReplaySource.SL))
            versions.fill(0)
        elif mode == "history":
            descriptor = (
                self.policy_pool.history[-1]
                if self.policy_pool.history
                else self.policy_pool.frozen_sl
            )
            ids.fill(descriptor.policy_id)
            sources.fill(int(descriptor.source))
            versions.fill(descriptor.version)
        else:
            raise ValueError(f"unsupported lineup mode {mode}")
        ids[np.arange(games), focal] = "current"
        sources[np.arange(games), focal] = int(ReplaySource.CURRENT)
        versions[np.arange(games), focal] = self.policy_pool.current.version
        return PolicyLineupBatch(ids, sources, versions, focal)

    def _model_logits(
        self,
        buffers: EngineBuffers,
        policy_ids: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        logits = np.zeros((len(rows), ACTION_SPACE_SIZE), dtype=np.float32)
        if not len(rows):
            return logits
        positions = {int(row): index for index, row in enumerate(rows.tolist())}
        for policy_id in sorted(set(str(policy_ids[row]) for row in rows)):
            selected = np.asarray(
                [row for row in rows if str(policy_ids[row]) == policy_id],
                dtype=np.int64,
            )
            inference_rows = _bucket_inference_rows(selected)
            model = self.executables.resolve(policy_id)
            lengths = buffers.event_lengths[inference_rows].astype(np.int64)
            history = _history_prefix(buffers.events[inference_rows], lengths)
            model.eval()
            with torch.inference_mode(), _autocast(self.device):
                output = model(
                    torch.as_tensor(
                        buffers.tile_obs[inference_rows], device=self.device
                    ),
                    torch.as_tensor(buffers.melds[inference_rows], device=self.device),
                    torch.as_tensor(buffers.meta[inference_rows], device=self.device),
                    torch.as_tensor(history, device=self.device),
                    torch.as_tensor(lengths, device=self.device),
                    torch.as_tensor(buffers.legal[inference_rows], device=self.device),
                )
            values = output.raw_logits[: len(selected)].float().cpu().numpy()
            if not np.isfinite(values).all():
                bad = int((~np.isfinite(values)).sum())
                selected_lengths = lengths[: len(selected)]
                raise RuntimeError(
                    f"policy {policy_id!r} produced {bad} non-finite logits "
                    f"(rows={len(selected)}, inference_batch={len(inference_rows)}, "
                    f"history_width={history.shape[1]}, "
                    f"history_length={int(selected_lengths.min())}.."
                    f"{int(selected_lengths.max())})"
                )
            logits[[positions[int(row)] for row in selected]] = values
        return logits

    def _actions(
        self,
        buffers: EngineBuffers,
        lineups: PolicyLineupBatch,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        actors = buffers.meta[:, 1].astype(np.int64)
        rows = np.flatnonzero(actors >= 0)
        if len(rows) != len(buffers.actions):
            raise RuntimeError("collector encountered a terminal row before reset")
        sources = lineups.sources[rows, actors[rows]].astype(np.uint8)
        versions = lineups.versions[rows, actors[rows]].astype(np.uint32)
        policy_ids = lineups.policy_ids[rows, actors[rows]]
        rule_actions = np.empty(len(rows), dtype=np.uint8)
        buffers.batch.simple_rule_actions_into(rule_actions)
        safe_positions = np.flatnonzero(sources == int(ReplaySource.RULE_SAFE))
        if len(safe_positions):
            _safe_rule_actions(
                rule_actions,
                buffers.legal,
                buffers.tile_obs,
                safe_positions,
            )
        logits = np.zeros((len(rows), ACTION_SPACE_SIZE), dtype=np.float32)
        rule = np.isin(
            sources,
            (int(ReplaySource.RULE_FAST), int(ReplaySource.RULE_SAFE)),
        )
        logits[rule] = 0.0
        logits[np.flatnonzero(rule), rule_actions[rule].astype(np.int64)] = 12.0
        model_positions = np.flatnonzero(~rule)
        if len(model_positions):
            model_rows = rows[model_positions]
            model_logits = self._model_logits(
                buffers,
                np.asarray(policy_ids, dtype=object),
                model_rows,
            )
            logits[model_positions] = model_logits
        categories = decision_categories(buffers.meta[rows])
        if deterministic:
            masked = np.where(buffers.legal[rows], logits, -np.inf)
            actions = masked.argmax(axis=1).astype(np.uint8)
            probabilities = np.ones(len(rows), dtype=np.float32)
            temperatures = np.zeros(len(rows), dtype=np.float32)
        else:
            sampled = self.behavior.sample(logits, buffers.legal[rows], categories)
            actions = sampled.actions
            probabilities = sampled.action_probabilities
            temperatures = sampled.temperatures
        return actions, probabilities, temperatures, categories, sources, len(model_positions)

    def collect(
        self,
        games: int,
        *,
        mode: LineupMode = "mixed",
        deterministic: bool = False,
    ) -> CollectionResult:
        if games <= 0:
            raise ValueError("games must be positive")
        envs = min(self.config.envs, games)
        buffers = EngineBuffers.create(envs, self.config.history)
        rows = np.arange(envs, dtype=np.int64)
        reset_flags = np.ones(envs, dtype=np.uint8)
        reset_seeds = np.asarray(
            [self._seed(self.next_seed + row) for row in rows], dtype=np.uint64
        )
        self.next_seed += envs
        history_masks = np.full(envs, ALL_SEATS_MASK, dtype=np.uint8)
        lineups = self._lineups(envs, mode)
        builders = [
            TrajectoryBuilder(
                int(seed), int(bm.Game(seed=int(seed)).exchange_direction)
            )
            for seed in reset_seeds
        ]
        cumulative = np.zeros((envs, 4), dtype=np.int64)
        steps_per_game = np.zeros(envs, dtype=np.int32)
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
        started = time.perf_counter()
        while len(completed) < games:
            actors = buffers.meta[:, 1].astype(np.int64)
            phases = buffers.meta[:, 0].astype(np.int64)
            (
                actions,
                action_probabilities,
                temperatures,
                categories,
                sources,
                model_count,
            ) = self._actions(buffers, lineups, deterministic=deterministic)
            versions = lineups.versions[rows, actors].astype(np.uint32)
            for row in rows:
                builders[row].append(
                    action=int(actions[row]),
                    actor=int(actors[row]),
                    phase=int(phases[row]),
                    category=int(categories[row]),
                    source=int(sources[row]),
                    policy_version=int(versions[row]),
                    action_probability=float(action_probabilities[row]),
                    temperature=float(temperatures[row]),
                )
            source_counts += np.bincount(sources, minlength=len(ReplaySource))
            buffers.actions[:] = actions
            buffers.batch.step_and_observe_history_into(
                buffers.actions,
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
            steps_per_game += 1
            environment_steps += envs
            policy_actions += model_count
            if np.any(steps_per_game > self.config.maximum_steps_per_game):
                stuck = np.flatnonzero(
                    steps_per_game > self.config.maximum_steps_per_game
                ).tolist()
                raise RuntimeError(f"engine games exceeded the step limit: {stuck[:8]}")
            terminal_rows = np.flatnonzero(buffers.records[:, 11])
            if not len(terminal_rows):
                continue
            for row in terminal_rows:
                scores = 10_000 + cumulative[row]
                ranking = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
                reason = (
                    int(bm.TERMINATION_WALL_EXHAUSTED)
                    if int(buffers.meta[row, 4]) == 0
                    else int(bm.TERMINATION_THREE_PLAYERS_BANKRUPT)
                )
                if len(completed) < games:
                    completed.append(
                        builders[row].finish(
                            terminal_scores=scores,
                            ranking_order=ranking,
                            termination_reason=reason,
                        )
                    )
                    focal = (
                        int(lineups.focal_seats[row])
                        if lineups.focal_seats is not None
                        else -1
                    )
                    focal_seats.append(focal)
            if len(completed) >= games:
                break
            reset_count = len(terminal_rows)
            reset_lineups = self._lineups(reset_count, mode)
            new_seeds = np.asarray(
                [
                    self._seed(self.next_seed + offset)
                    for offset in range(reset_count)
                ],
                dtype=np.uint64,
            )
            self.next_seed += reset_count
            for position, row in enumerate(terminal_rows):
                lineups.policy_ids[row] = reset_lineups.policy_ids[position]
                lineups.sources[row] = reset_lineups.sources[position]
                lineups.versions[row] = reset_lineups.versions[position]
                if lineups.focal_seats is not None and reset_lineups.focal_seats is not None:
                    lineups.focal_seats[row] = reset_lineups.focal_seats[position]
                seed = int(new_seeds[position])
                builders[row] = TrajectoryBuilder(
                    seed, int(bm.Game(seed=seed).exchange_direction)
                )
                cumulative[row] = 0
                steps_per_game[row] = 0
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
        )


def _evaluation(result: CollectionResult) -> dict[str, float]:
    if np.any(result.focal_seats < 0):
        raise ValueError("evaluation collection is missing focal seats")
    scores = np.asarray(
        [
            trajectory.terminal_scores[int(seat)] - 10_000
            for trajectory, seat in zip(result.trajectories, result.focal_seats)
        ],
        dtype=np.float64,
    )
    ranks = np.asarray(
        [
            trajectory.terminal_ranks[int(seat)]
            for trajectory, seat in zip(result.trajectories, result.focal_seats)
        ],
        dtype=np.float64,
    )
    return {
        "games": float(len(scores)),
        "mean_score_delta": float(scores.mean()),
        "score_std": float(scores.std()),
        "first_rate": float(np.mean(ranks == 1)),
        "last_rate": float(np.mean(ranks == 4)),
        "mean_rank": float(ranks.mean()),
        "environment_steps_per_second": float(
            result.environment_steps / max(result.elapsed_seconds, 1e-9)
        ),
    }


def evaluate_policy(
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    policy_pool: PolicyPool,
    executables: ExecutablePolicyPool,
    behavior: BehaviorSampler,
    device: torch.device,
    *,
    games: int,
    envs: int,
    seed: int,
    mode: LineupMode,
) -> dict[str, float]:
    copied_pool = PolicyPool.from_state_dict(policy_pool.state_dict())
    collector = FullTrajectoryCollector(
        CollectionConfig(envs=min(envs, games)),
        copied_pool,
        executables,
        BehaviorSampler.from_state_dict(behavior.state_dict()),
        device,
        seed=seed,
    )
    return _evaluation(collector.collect(games, mode=mode, deterministic=True))


def evaluate_panel(
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    policy_pool: PolicyPool,
    executables: ExecutablePolicyPool,
    behavior: BehaviorSampler,
    device: torch.device,
    *,
    seeds: tuple[int, ...] | list[int],
    games: int,
    envs: int,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("evaluation panel requires at least one seed")
    modes: tuple[LineupMode, ...] = ("rules", "sl", "mixed", "history")
    panels: dict[str, dict[str, object]] = {}
    for mode in modes:
        runs = [
            evaluate_policy(
                actor,
                reference,
                policy_pool,
                executables,
                behavior,
                device,
                games=games,
                envs=envs,
                seed=int(seed),
                mode=mode,
            )
            for seed in seeds
        ]
        panels[mode] = {
            "games": float(games * len(runs)),
            "mean_score_delta": float(
                np.mean([run["mean_score_delta"] for run in runs])
            ),
            "score_std": float(np.mean([run["score_std"] for run in runs])),
            "first_rate": float(np.mean([run["first_rate"] for run in runs])),
            "last_rate": float(np.mean([run["last_rate"] for run in runs])),
            "mean_rank": float(np.mean([run["mean_rank"] for run in runs])),
            "seed_score_std": float(
                np.std([run["mean_score_delta"] for run in runs])
            ),
            "seed_rank_std": float(np.std([run["mean_rank"] for run in runs])),
            "runs": runs,
        }
    rules = panels["rules"]
    return {
        "mean_score_delta": rules["mean_score_delta"],
        "score_std": rules["score_std"],
        "first_rate": rules["first_rate"],
        "last_rate": rules["last_rate"],
        "mean_rank": rules["mean_rank"],
        "panels": panels,
    }


def better_on_both_panels(
    fixed: dict[str, object],
    fresh: dict[str, object],
    incumbent_fixed: dict[str, object],
    incumbent_fresh: dict[str, object],
) -> bool:
    def better(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
        candidate_rank = float(candidate["mean_rank"])
        incumbent_rank = float(incumbent["mean_rank"])
        if candidate_rank < incumbent_rank - 1e-12:
            return True
        return abs(candidate_rank - incumbent_rank) <= 1e-12 and float(
            candidate["mean_score_delta"]
        ) > float(incumbent["mean_score_delta"])

    return better(fixed, incumbent_fixed) and better(fresh, incumbent_fresh)


__all__ = [
    "CATEGORY_NAMES",
    "CollectionConfig",
    "CollectionResult",
    "EngineBuffers",
    "ExecutablePolicyPool",
    "FullTrajectoryCollector",
    "better_on_both_panels",
    "clone_policy",
    "evaluate_panel",
    "load_policy",
    "save_policy",
]
