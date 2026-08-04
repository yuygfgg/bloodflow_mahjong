"""End-to-end rollout, opponent scheduling, and PPO utilities."""

from __future__ import annotations

import copy
import math
import random
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import bloodflow_mahjong as bm

from .contracts import TRAINING_INPUT_SCHEMA
from .device import stage_numpy_batch
from .model import BloodFlowTransformer, HistoryKVCache, LayerKV, TransformerConfig
from .observation import unpack_action_masks


CHECKPOINT_VERSION = 1
_SEED_MULTIPLIER = 0x9E3779B97F4A7C15
_U64_MASK = (1 << 64) - 1


def _seed_sequence(start: int, count: int) -> np.ndarray:
    """Map a contiguous counter range to unique deterministic engine seeds."""
    if count < 0:
        raise ValueError("seed count cannot be negative")
    return np.fromiter(
        (
            ((int(start) + offset) * _SEED_MULTIPLIER) & _U64_MASK
            for offset in range(count)
        ),
        dtype=np.uint64,
        count=count,
    )


@dataclass(frozen=True)
class PPOConfig:
    envs: int = 2048
    rollout_transitions: int = 65_536
    ppo_epochs: int = 2
    minibatch: int = 4_096
    microbatch: int = 512
    # Small cached groups lose to one padded full forward because each group
    # needs Python bookkeeping, KV concatenation, and separate GPU launches.
    # Keep cache available for genuinely large groups without fragmenting the
    # rollout into hundreds of tiny dynamic shapes.
    history_cache_min_batch: int = 1_024
    learning_rate: float = 2e-4
    final_learning_rate: float = 3e-5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    final_entropy_coefficient: float = 0.002
    schedule_hours: float = 24.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    kl_control: str = "monitor"
    score_reward_weight: float = 1.0
    rank_reward_weight: float = 1.0
    shanten_coefficient: float = 0.02
    improving_coefficient: float = 0.01
    auxiliary_decay_fraction: float = 0.10
    self_play_enabled: bool = False
    self_play_fraction: float = 0.25
    self_play_gate_score_delta: float = 0.0
    self_play_gate_consecutive_evals: int = 3
    historical_snapshot_probability: float = 0.50
    opponent_refresh_updates: int = 200
    frozen_snapshot_limit: int = 4

    def __post_init__(self) -> None:
        for name in (
            "envs",
            "rollout_transitions",
            "ppo_epochs",
            "minibatch",
            "microbatch",
            "history_cache_min_batch",
            "self_play_gate_consecutive_evals",
            "opponent_refresh_updates",
            "frozen_snapshot_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        reward_weights = (self.score_reward_weight, self.rank_reward_weight)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in reward_weights):
            raise ValueError("reward weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in reward_weights):
            raise ValueError("at least one reward weight must be positive")
        if self.kl_control not in {"off", "monitor", "rollback"}:
            raise ValueError("KL control must be off, monitor, or rollback")
        if not math.isfinite(self.target_kl) or (
            self.kl_control == "rollback" and self.target_kl <= 0.0
        ):
            raise ValueError("target KL must be finite and positive in rollback mode")
        if not math.isfinite(self.self_play_fraction) or not (
            0.0 <= self.self_play_fraction < 1.0
        ):
            raise ValueError("self-play fraction must be in [0, 1)")
        if self.self_play_enabled and self.self_play_fraction == 0.0:
            raise ValueError("enabled self-play requires a positive fraction")
        if not math.isfinite(self.self_play_gate_score_delta):
            raise ValueError("self-play gate score delta must be finite")
        if not math.isfinite(self.historical_snapshot_probability) or not (
            0.0 <= self.historical_snapshot_probability <= 1.0
        ):
            raise ValueError("historical snapshot probability must be in [0, 1]")


def focal_ranks(scores: np.ndarray, seats: np.ndarray) -> np.ndarray:
    """Rank each focal seat using the evaluation tie convention."""
    scores = np.asarray(scores)
    seats = np.asarray(seats)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("scores must have shape [batch, 4]")
    if seats.shape != (len(scores),):
        raise ValueError("seats must have shape [batch]")
    if np.any((seats < 0) | (seats >= 4)):
        raise ValueError("seats contain an out-of-range value")
    own_scores = scores[np.arange(len(scores)), seats.astype(np.int64, copy=False)]
    return 1 + np.count_nonzero(scores > own_scores[:, None], axis=1)


def rank_utilities(scores: np.ndarray, seats: np.ndarray) -> np.ndarray:
    """Map focal ranks to +1, +1/3, -1/3, and -1."""
    ranks = focal_ranks(scores, seats)
    return ((2.5 - ranks) / 1.5).astype(np.float32)


def hybrid_rewards(
    score_deltas: np.ndarray,
    cumulative_scores: np.ndarray,
    learner_seats: np.ndarray,
    terminal: np.ndarray,
    config: PPOConfig,
) -> np.ndarray:
    """Return per-step learner rewards with terminal placement utility."""
    if (
        score_deltas.shape != cumulative_scores.shape
        or score_deltas.shape[1:] != (4,)
    ):
        raise ValueError("score deltas and cumulative scores must have shape [batch, 4]")
    if learner_seats.shape != (len(score_deltas),):
        raise ValueError("learner seats must have shape [batch]")
    if terminal.shape != (len(score_deltas),):
        raise ValueError("terminal flags must have shape [batch]")
    rows = np.arange(len(score_deltas))
    rewards = (
        config.score_reward_weight
        * score_deltas[rows, learner_seats.astype(np.int64, copy=False)].astype(np.float32)
        / 10_000.0
    )
    terminal_rows = np.flatnonzero(terminal)
    if len(terminal_rows):
        rewards[terminal_rows] += config.rank_reward_weight * rank_utilities(
            cumulative_scores[terminal_rows], learner_seats[terminal_rows]
        )
    return rewards


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
            legal=np.empty((batch_size, 115), dtype=np.bool_),
            records=np.empty((batch_size, 12), dtype=np.int64),
            actions=np.empty(batch_size, dtype=np.uint8),
        )

    def refresh(self) -> None:
        self.batch.observe_into(self.tile_obs, self.melds, self.river, self.meta)
        self.batch.legal_action_masks_into(self.masks)
        self.batch.events_into(self.events, self.event_lengths)
        self.refresh_legal()

    def refresh_legal(self, rows: np.ndarray | None = None) -> None:
        if rows is None:
            unpack_action_masks(self.masks, out=self.legal)
        elif len(rows):
            self.legal[rows] = unpack_action_masks(self.masks[rows])

    @property
    def legal_dense(self) -> np.ndarray:
        return self.legal


def _history_prefix(
    events: np.ndarray | Tensor, lengths: np.ndarray | Tensor
) -> np.ndarray | Tensor:
    """Drop right-padding before a model forward while preserving positions.

    Event histories are stored at the fixed 192-token window so the engine and
    rollout storage stay contiguous.  The history encoder masks right padding,
    so slicing to the largest valid length in the current batch is equivalent
    and avoids quadratic attention work on unused tokens.
    """
    if events.shape[1] == 0:
        return events
    if isinstance(lengths, np.ndarray):
        width = int(lengths.max(initial=0))
    else:
        width = int(lengths.max().item()) if lengths.numel() else 0
    return events[:, : max(width, 1)]


class TransitionStorage:
    """Preallocated learner transitions; pending slots are finalized in-place."""

    def __init__(self, capacity: int, history: int = 192) -> None:
        self.capacity = capacity
        self.next_slot = 0
        self.finalized = np.zeros(capacity, dtype=np.bool_)
        self.env = np.empty(capacity, dtype=np.int32)
        self.episode = np.empty(capacity, dtype=np.int64)
        self.tile_obs = np.empty((capacity, 10, 27), dtype=np.uint8)
        self.melds = np.empty((capacity, 4, 4, 3), dtype=np.uint8)
        self.meta = np.empty((capacity, 34), dtype=np.int32)
        self.events = np.empty((capacity, history, 8), dtype=np.int32)
        self.event_lengths = np.empty(capacity, dtype=np.int64)
        self.legal = np.empty((capacity, 115), dtype=np.bool_)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.logprob = np.empty(capacity, dtype=np.float32)
        self.value = np.empty(capacity, dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_value = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.shanten = np.full(capacity, 127, dtype=np.int8)
        self.improving = np.zeros(capacity, dtype=np.uint32)

    def add_many(
        self,
        envs: np.ndarray,
        episode: np.ndarray,
        buffers: EngineBuffers,
        legal: np.ndarray,
        actions: np.ndarray,
        logprob: np.ndarray,
        value: np.ndarray,
        shanten: np.ndarray,
        improving: np.ndarray,
    ) -> np.ndarray:
        count = len(envs)
        end = self.next_slot + count
        if end > self.capacity:
            raise RuntimeError("rollout transition storage capacity was exceeded")
        slots = np.arange(self.next_slot, end, dtype=np.int64)
        self.next_slot = end
        self.env[slots] = envs
        self.episode[slots] = episode
        self.tile_obs[slots] = buffers.tile_obs[envs]
        self.melds[slots] = buffers.melds[envs]
        self.meta[slots] = buffers.meta[envs]
        self.events[slots] = buffers.events[envs]
        self.event_lengths[slots] = buffers.event_lengths[envs]
        self.legal[slots] = legal
        self.actions[slots] = actions
        self.logprob[slots] = logprob
        self.value[slots] = value
        self.shanten[slots] = shanten
        self.improving[slots] = improving
        return slots

    def finalize(self, slots: np.ndarray, next_value: np.ndarray, done: bool) -> None:
        valid = slots >= 0
        selected = slots[valid]
        self.next_value[selected] = next_value[valid]
        self.done[selected] = done
        self.finalized[selected] = True

    def indices(self, count: int) -> np.ndarray:
        selected = np.flatnonzero(self.finalized)
        if len(selected) < count:
            raise RuntimeError(f"only finalized {len(selected)} of {count} transitions")
        return selected[:count]

    def compute_gae(
        self,
        indices: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(self.capacity, dtype=np.float32)
        returns = np.zeros(self.capacity, dtype=np.float32)
        last: dict[tuple[int, int], float] = {}
        for slot in indices[::-1]:
            key = (int(self.env[slot]), int(self.episode[slot]))
            continuation = 0.0 if self.done[slot] else last.get(key, 0.0)
            delta = (
                float(self.reward[slot])
                + gamma * (1.0 - float(self.done[slot])) * float(self.next_value[slot])
                - float(self.value[slot])
            )
            advantage = (
                delta
                + gamma * gae_lambda * (1.0 - float(self.done[slot])) * continuation
            )
            advantages[slot] = advantage
            returns[slot] = advantage + self.value[slot]
            last[key] = advantage
        return advantages, returns


@dataclass(frozen=True)
class RolloutBatch:
    storage: TransitionStorage
    indices: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    opponent_counts: np.ndarray
    opponent_stage: str
    active_snapshot: int | None
    cache_stats: dict[str, int | float]

    def __len__(self) -> int:
        return len(self.indices)

    def tensors(
        self,
        slots: np.ndarray,
        device: torch.device,
        history_width: int,
    ) -> dict[str, Tensor]:
        storage = self.storage
        if not 1 <= history_width <= storage.events.shape[1]:
            raise ValueError("rollout history width is invalid")
        return stage_numpy_batch(
            {
                "tile_obs": storage.tile_obs[slots],
                "melds": storage.melds[slots],
                "meta": storage.meta[slots],
                "events": storage.events[slots, :history_width],
                "event_lengths": storage.event_lengths[slots],
                "legal": storage.legal[slots],
                "actions": storage.actions[slots],
                "old_logprob": storage.logprob[slots],
                "advantages": self.advantages[slots],
                "returns": self.returns[slots],
                "shanten": storage.shanten[slots],
                "improving": storage.improving[slots].astype(np.int64),
            },
            device,
        )


@dataclass(frozen=True)
class _InferenceChunk:
    positions: np.ndarray
    actions: Tensor
    logprobs: Tensor
    values: Tensor


@dataclass(frozen=True)
class PendingInference:
    """Device-side policy results that synchronize only when resolved."""

    size: int
    chunks: tuple[_InferenceChunk, ...]

    @classmethod
    def empty(cls) -> PendingInference:
        return cls(0, ())

    def resolve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.empty(self.size, dtype=np.uint8)
        logprobs = np.empty(self.size, dtype=np.float32)
        values = np.empty(self.size, dtype=np.float32)
        for chunk in self.chunks:
            actions[chunk.positions] = chunk.actions.cpu().numpy().astype(np.uint8)
            logprobs[chunk.positions] = chunk.logprobs.float().cpu().numpy()
            values[chunk.positions] = chunk.values.float().cpu().numpy()
        return actions, logprobs, values


def _inference_chunk(
    positions: np.ndarray,
    output: Any,
    deterministic: bool,
) -> _InferenceChunk:
    log_probs = F.log_softmax(output.logits.float(), dim=-1)
    actions = (
        output.logits.argmax(dim=-1)
        if deterministic
        else torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
    )
    return _InferenceChunk(
        positions=positions,
        actions=actions,
        logprobs=log_probs.gather(1, actions[:, None]).squeeze(1),
        values=output.value,
    )


class OpponentPool:
    """Rule opponents and an optional fixed-ratio frozen-policy pool.

    Opponent types are assigned per *absolute seat*, not once per game.  This
    gives the learner mixed three-player tables while keeping every policy
    action actor-viewer scoped.  A rollout uses one selected frozen snapshot,
    so the transformer inference path remains batched.
    """

    RULE_FAST = 0
    RULE_EV = 1
    FROZEN_TRANSFORMER = 2
    NAMES = ("rule_fast", "rule_ev", "frozen_transformer")

    def __init__(self, config: PPOConfig, seed: int = 1) -> None:
        self.config = config
        self.random = np.random.default_rng(seed)
        self.rule_ev_config = bm.RuleEvConfig.standard()
        self.snapshots: list[BloodFlowTransformer] = []
        self.active_snapshot: int | None = None
        self.last_snapshot_update: int | None = None
        self.rule_first_rate: float | None = None
        self.rule_score_delta: float | None = None
        self.rule_gate_streak = 0

    @property
    def frozen_model(self) -> BloodFlowTransformer | None:
        if self.active_snapshot is None:
            return None
        return self.snapshots[self.active_snapshot]

    @property
    def frozen_ready(self) -> bool:
        return self.frozen_model is not None

    @property
    def gate_ready(self) -> bool:
        return (
            self.rule_gate_streak
            >= self.config.self_play_gate_consecutive_evals
        )

    def stage(self) -> str:
        """Return the executable opponent stage from rule competence.

        Transition count is deliberately absent from this decision.  A few
        consecutive evaluations are required so a noisy small evaluation
        cannot promote the learner into a harder opponent distribution.
        """
        if (
            not self.config.self_play_enabled
            or not self.gate_ready
            or not self.frozen_ready
        ):
            return "bootstrap"
        return "self_play"

    def probabilities(self) -> np.ndarray:
        """Probability of each policy for one non-learner seat."""
        if self.stage() == "bootstrap":
            return np.array([1.0 / 3.0, 2.0 / 3.0, 0.0], dtype=np.float64)
        rule_fraction = 1.0 - self.config.self_play_fraction
        return np.array(
            [
                rule_fraction / 3.0,
                2.0 * rule_fraction / 3.0,
                self.config.self_play_fraction,
            ],
            dtype=np.float64,
        )

    def assign_seats(self, learner_seats: np.ndarray) -> np.ndarray:
        """Return opponent kinds with shape ``[env, absolute_seat]``.

        The learner slot is ``-1`` and must never be consumed by
        :meth:`actions`.  Sampling the remaining seats independently is
        intentional: each environment contains a small mixed opponent table.
        """
        learner_seats = np.asarray(learner_seats, dtype=np.int64)
        kinds = np.full((len(learner_seats), 4), -1, dtype=np.int8)
        probabilities = self.probabilities()
        for seat in range(4):
            rows = np.flatnonzero(learner_seats != seat)
            kinds[rows, seat] = self.random.choice(
                len(self.NAMES), size=len(rows), p=probabilities
            ).astype(np.int8)
        return kinds

    def refresh_snapshot(
        self,
        model: BloodFlowTransformer,
        device: torch.device,
        *,
        update: int | None = None,
    ) -> int | None:
        """Add a frozen learner snapshot after the rule competence gate."""
        if not self.config.self_play_enabled or not self.gate_ready:
            return None
        self.snapshots.append(clone_model(model, device))
        overflow = len(self.snapshots) - self.config.frozen_snapshot_limit
        if self.config.frozen_snapshot_limit == 1:
            self.snapshots[:] = self.snapshots[-1:]
        elif overflow > 0:
            # Keep the fork anchor and evict the oldest post-fork policies.
            # This leaves one stable reference while the remaining slots track
            # recent learner generations.
            del self.snapshots[1 : 1 + overflow]
        self.last_snapshot_update = update
        return self.select_snapshot()

    def snapshot_due(self, update: int) -> bool:
        """Return whether the learner should create a new frozen policy."""
        return (
            not self.frozen_ready
            or self.last_snapshot_update is None
            or update - self.last_snapshot_update
            >= self.config.opponent_refresh_updates
        )

    def select_snapshot(self) -> int | None:
        """Select one batched frozen opponent for the next rollout."""
        if not self.snapshots:
            self.active_snapshot = None
            return None
        latest = len(self.snapshots) - 1
        if latest > 0 and (
            self.random.random() < self.config.historical_snapshot_probability
        ):
            self.active_snapshot = int(self.random.integers(0, latest))
        else:
            self.active_snapshot = latest
        return self.active_snapshot

    def update_rule_evaluation(self, evaluation: dict[str, float]) -> None:
        """Update competence gates from a rule-anchor evaluation."""
        first_rate = float(evaluation["first_rate"])
        score_delta = float(evaluation["mean_score_delta"])
        self.rule_first_rate = first_rate
        self.rule_score_delta = score_delta
        required = self.config.self_play_gate_consecutive_evals
        if score_delta >= self.config.self_play_gate_score_delta:
            self.rule_gate_streak += 1
        else:
            self.rule_gate_streak = 0
        self.rule_gate_streak = min(self.rule_gate_streak, required)

    def set_frozen(self, model: BloodFlowTransformer | None) -> None:
        """Install a frozen model, primarily for tests and explicit resume."""
        self.snapshots.clear()
        self.active_snapshot = None
        self.last_snapshot_update = None
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.snapshots.append(model)
            self.active_snapshot = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.random.bit_generator.state,
            "active_snapshot": self.active_snapshot,
            "last_snapshot_update": self.last_snapshot_update,
            "snapshots": [snapshot.state_dict() for snapshot in self.snapshots],
            "rule_first_rate": self.rule_first_rate,
            "rule_score_delta": self.rule_score_delta,
            "rule_gate_streak": self.rule_gate_streak,
        }

    def load_state_dict(
        self,
        state: dict[str, Any] | None,
        model_config: Any,
        device: torch.device,
    ) -> None:
        if not state:
            return
        self.random.bit_generator.state = state["rng_state"]
        self.snapshots = []
        last_snapshot_update = state.get("last_snapshot_update")
        self.last_snapshot_update = (
            int(last_snapshot_update) if last_snapshot_update is not None else None
        )
        self.rule_first_rate = state.get("rule_first_rate")
        self.rule_score_delta = state.get("rule_score_delta")
        self.rule_gate_streak = int(
            state.get("rule_gate_streak", state.get("rule_league_streak", 0))
        )
        for snapshot_state in state.get("snapshots", []):
            snapshot = BloodFlowTransformer(model_config).to(device)
            snapshot.load_state_dict(snapshot_state)
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
            self.snapshots.append(snapshot)
        active = state.get("active_snapshot")
        self.active_snapshot = (
            int(active)
            if active is not None and int(active) < len(self.snapshots)
            else None
        )

    def action_kinds(
        self,
        buffers: EngineBuffers,
        seat_kinds: np.ndarray,
    ) -> np.ndarray:
        actors = buffers.meta[:, 1].astype(np.int64)
        active = actors >= 0
        rows = np.flatnonzero(active)
        kinds = np.full(len(buffers.batch), -1, dtype=np.int8)
        kinds[rows] = seat_kinds[rows, actors[rows]]
        return kinds

    def rule_actions(
        self,
        buffers: EngineBuffers,
        kinds: np.ndarray,
    ) -> np.ndarray:
        """Calculate only rows controlled by a rule opponent."""
        actions = np.empty(len(buffers.batch), dtype=np.uint8)
        enabled = np.zeros(len(actions), dtype=np.uint8)
        fast_rows = np.flatnonzero(kinds == self.RULE_FAST)
        if len(fast_rows):
            enabled[fast_rows] = 1
            buffers.batch.simple_rule_actions_masked_into(enabled, actions)

        ev_rows = np.flatnonzero(kinds == self.RULE_EV)
        if len(ev_rows):
            enabled.fill(0)
            enabled[ev_rows] = 1
            buffers.batch.rule_ev_actions_masked_into(
                enabled, actions, self.rule_ev_config
            )
        return actions


class HistoryCacheStore:
    """Per-environment viewer caches for one fixed model version.

    Histories are append-only until the fixed window fills. Exact cache shapes
    are batched only when the group is large enough to keep the GPU busy;
    fragmented groups fall back to one padded full-history forward.
    """

    def __init__(self, max_history: int = 192, min_cache_batch: int = 32) -> None:
        self.max_history = max_history
        self.min_cache_batch = min_cache_batch
        self.caches: dict[tuple[int, int], HistoryKVCache] = {}
        self.hit_rows = 0
        self.cached_rows = 0
        self.full_rows = 0
        self.cached_groups = 0

    def clear_rows(self, rows: np.ndarray) -> None:
        row_set = {int(row) for row in rows}
        for key in list(self.caches):
            if key[0] in row_set:
                self.caches.pop(key, None)

    def clear(self) -> None:
        self.caches.clear()

    def statistics(self) -> dict[str, int]:
        return {
            "hit_rows": self.hit_rows,
            "cached_rows": self.cached_rows,
            "full_rows": self.full_rows,
            "cached_groups": self.cached_groups,
        }

    @staticmethod
    def _split_cache(cache: HistoryKVCache, index: int) -> HistoryKVCache:
        return HistoryKVCache(
            tuple(
                LayerKV(layer.key[index : index + 1], layer.value[index : index + 1])
                for layer in cache.layers
            ),
            cache.length,
            cache.summary[index : index + 1],
        )

    def launch(
        self,
        model: BloodFlowTransformer,
        buffers: EngineBuffers,
        rows: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> PendingInference:
        if len(rows) == 0:
            return PendingInference.empty()

        actors = buffers.meta[rows, 1].astype(np.int64)
        requests: dict[tuple[int, int, int], list[tuple[int, int, int, np.ndarray]]] = (
            {}
        )
        for row, actor in zip(rows.tolist(), actors.tolist()):
            key = (int(row), int(actor))
            length = int(buffers.event_lengths[row])
            cache = self.caches.get(key)
            can_append = (
                cache is not None
                and 0 < length <= self.max_history
                and length >= cache.length
                and cache.length < self.max_history
            )
            if can_append:
                start = cache.length
                request_key = (cache.length, length - start, 1)
                new_events = buffers.events[row, start:length]
                self.hit_rows += 1
            else:
                self.caches.pop(key, None)
                request_key = (-1, length, 0)
                new_events = buffers.events[row, :length]
            requests.setdefault(request_key, []).append(
                (row, actor, length, new_events)
            )

        cached_requests: list[
            tuple[tuple[int, int, int], list[tuple[int, int, int, np.ndarray]]]
        ] = []
        full_rows: list[int] = []
        for request_key, entries in requests.items():
            _, length, cached = request_key
            if length > 0 and len(entries) >= self.min_cache_batch:
                cached_requests.append((request_key, entries))
                continue
            full_rows.extend(entry[0] for entry in entries)
            for row, actor, _, _ in entries:
                self.caches.pop((row, actor), None)

        chunks: list[_InferenceChunk] = []
        row_positions = {int(row): index for index, row in enumerate(rows.tolist())}

        def retain(
            result_rows: list[int], output: Any, result_cache: HistoryKVCache | None
        ) -> None:
            positions = np.fromiter(
                (row_positions[row] for row in result_rows),
                dtype=np.int64,
                count=len(result_rows),
            )
            chunks.append(_inference_chunk(positions, output, deterministic))
            if result_cache is not None:
                for index, row in enumerate(result_rows):
                    actor = int(buffers.meta[row, 1])
                    key = (row, actor)
                    self.caches[key] = self._split_cache(result_cache, index)

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if full_rows:
                self.full_rows += len(full_rows)
                output = model(
                    torch.as_tensor(buffers.tile_obs[full_rows], device=device),
                    torch.as_tensor(buffers.melds[full_rows], device=device),
                    torch.as_tensor(buffers.meta[full_rows], device=device),
                    torch.as_tensor(
                        _history_prefix(
                            buffers.events[full_rows],
                            buffers.event_lengths[full_rows],
                        ),
                        device=device,
                    ),
                    torch.as_tensor(
                        buffers.event_lengths[full_rows].astype(np.int64), device=device
                    ),
                    torch.as_tensor(buffers.legal_dense[full_rows], device=device),
                )
                retain(full_rows, output, None)

            for (past_length, _delta_length, cached), entries in cached_requests:
                result_rows = [entry[0] for entry in entries]
                self.cached_rows += len(result_rows)
                self.cached_groups += 1
                event_batch = torch.as_tensor(
                    np.stack([entry[3] for entry in entries]), device=device
                )
                if cached:
                    past_caches = [
                        self.caches[(row, int(buffers.meta[row, 1]))]
                        for row in result_rows
                    ]
                    stacked = HistoryKVCache(
                        tuple(
                            LayerKV(
                                torch.cat(
                                    [cache.layers[layer].key for cache in past_caches],
                                    dim=0,
                                ),
                                torch.cat(
                                    [
                                        cache.layers[layer].value
                                        for cache in past_caches
                                    ],
                                    dim=0,
                                ),
                            )
                            for layer in range(len(past_caches[0].layers))
                        ),
                        past_length,
                        torch.cat([cache.summary for cache in past_caches], dim=0),
                    )
                else:
                    stacked = None
                output, next_cache = model.forward_cached(
                    torch.as_tensor(buffers.tile_obs[result_rows], device=device),
                    torch.as_tensor(buffers.melds[result_rows], device=device),
                    torch.as_tensor(buffers.meta[result_rows], device=device),
                    event_batch,
                    stacked,
                    torch.as_tensor(buffers.legal_dense[result_rows], device=device),
                )
                retain(result_rows, output, next_cache)

        return PendingInference(len(rows), tuple(chunks))

    def infer(
        self,
        model: BloodFlowTransformer,
        buffers: EngineBuffers,
        rows: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.launch(model, buffers, rows, device, deterministic).resolve()


def launch_inference(
    model: BloodFlowTransformer,
    buffers: EngineBuffers,
    rows: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
    history_cache: HistoryCacheStore | None = None,
) -> PendingInference:
    if len(rows) == 0:
        return PendingInference.empty()
    if history_cache is not None:
        return history_cache.launch(model, buffers, rows, device, deterministic)
    dense = buffers.legal_dense[rows]
    history_lengths = buffers.event_lengths[rows].astype(np.int64)
    inputs = (
        torch.as_tensor(buffers.tile_obs[rows], device=device),
        torch.as_tensor(buffers.melds[rows], device=device),
        torch.as_tensor(buffers.meta[rows], device=device),
        torch.as_tensor(
            _history_prefix(buffers.events[rows], history_lengths), device=device
        ),
        torch.as_tensor(history_lengths, device=device),
        torch.as_tensor(dense, device=device),
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(*inputs)
        log_probs = F.log_softmax(output.logits.float(), dim=-1)
        actions = (
            output.logits.argmax(dim=-1)
            if deterministic
            else torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        )
        logprob = log_probs.gather(1, actions[:, None]).squeeze(1)
    return PendingInference(
        len(rows),
        (
            _InferenceChunk(
                positions=np.arange(len(rows), dtype=np.int64),
                actions=actions,
                logprobs=logprob,
                values=output.value,
            ),
        ),
    )


def infer_actions(
    model: BloodFlowTransformer,
    buffers: EngineBuffers,
    rows: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
    history_cache: HistoryCacheStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return launch_inference(
        model,
        buffers,
        rows,
        device,
        deterministic,
        history_cache,
    ).resolve()


class RolloutCollector:
    SEAT_BITS = np.asarray([1, 2, 4, 8], dtype=np.uint8)

    def __init__(
        self,
        config: PPOConfig,
        device: torch.device,
        seed: int = 1,
    ) -> None:
        self.config = config
        self.device = device
        self.buffers = EngineBuffers.create(config.envs, history=192)
        self.pool = OpponentPool(config, seed)
        self.learner_seats = np.arange(config.envs, dtype=np.uint8) % 4
        self.episode_ids = np.zeros(config.envs, dtype=np.int64)
        self.opponent_kinds = self.pool.assign_seats(self.learner_seats)
        self._opponent_counts = np.zeros(len(OpponentPool.NAMES), dtype=np.int64)
        self.history_seat_masks = np.empty(config.envs, dtype=np.uint8)
        self.reset_flags = np.zeros(config.envs, dtype=np.uint8)
        self.reset_seeds = np.zeros(config.envs, dtype=np.uint64)
        self._rows = np.arange(config.envs)
        self.learner_history_cache = HistoryCacheStore(
            max_history=192, min_cache_batch=config.history_cache_min_batch
        )
        self.frozen_history_cache = HistoryCacheStore(
            max_history=192, min_cache_batch=config.history_cache_min_batch
        )
        self.next_seed = int(seed) & _U64_MASK
        self._update_history_seat_masks(self._rows)

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_seed": self.next_seed,
            "opponent_pool": self.pool.state_dict(),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        model_config: Any,
    ) -> None:
        next_seed = int(state["next_seed"])
        if next_seed < 0:
            raise ValueError("collector next seed cannot be negative")
        self.next_seed = next_seed
        self.pool.load_state_dict(state["opponent_pool"], model_config, self.device)

    def _update_history_seat_masks(self, rows: np.ndarray) -> None:
        masks = self.SEAT_BITS[self.learner_seats[rows]].copy()
        if self.pool.frozen_ready:
            frozen = self.opponent_kinds[rows] == OpponentPool.FROZEN_TRANSFORMER
            frozen_bits = np.bitwise_or.reduce(
                np.where(frozen, self.SEAT_BITS[None, :], np.uint8(0)), axis=1
            )
            masks |= frozen_bits
        self.history_seat_masks[rows] = masks

    def _assign_opponents(self, rows: np.ndarray) -> None:
        assigned = self.pool.assign_seats(self.learner_seats[rows])
        self.opponent_kinds[rows] = assigned
        selected = assigned[assigned >= 0]
        self._opponent_counts += np.bincount(
            selected, minlength=len(OpponentPool.NAMES)
        )
        self._update_history_seat_masks(rows)

    def _take_seeds(self, count: int) -> np.ndarray:
        seeds = _seed_sequence(self.next_seed, count)
        self.next_seed += count
        return seeds

    def _reset_rows(self, rows: np.ndarray) -> None:
        if len(rows) == 0:
            return
        seeds = self._take_seeds(len(rows))
        self.episode_ids[rows] += 1
        self.learner_seats[rows] = (
            (rows + self.episode_ids[rows]) % len(self.SEAT_BITS)
        ).astype(np.uint8)
        self._assign_opponents(rows)
        self.reset_flags[rows] = 1
        self.reset_seeds[rows] = seeds
        self.buffers.batch.reset_and_observe_history_into(
            self.reset_flags,
            self.reset_seeds,
            self.history_seat_masks,
            self.buffers.masks,
            self.buffers.tile_obs,
            self.buffers.melds,
            self.buffers.river,
            self.buffers.meta,
            self.buffers.events,
            self.buffers.event_lengths,
        )
        self.reset_flags[rows] = 0
        self.buffers.refresh_legal(rows)

    def collect(
        self,
        model: BloodFlowTransformer,
        transitions: int,
        collect_auxiliary: bool = True,
    ) -> RolloutBatch:
        model.eval()
        self.learner_seats = np.arange(self.config.envs, dtype=np.uint8) % 4
        self.episode_ids.fill(0)
        self.opponent_kinds = np.full((self.config.envs, 4), -1, dtype=np.int8)
        self._opponent_counts.fill(0)
        self.learner_history_cache = HistoryCacheStore(
            max_history=192, min_cache_batch=self.config.history_cache_min_batch
        )
        self.frozen_history_cache = HistoryCacheStore(
            max_history=192, min_cache_batch=self.config.history_cache_min_batch
        )
        self._assign_opponents(self._rows)
        self.reset_flags.fill(1)
        self.reset_seeds[:] = self._take_seeds(self.config.envs)
        self.buffers.batch.reset_and_observe_history_into(
            self.reset_flags,
            self.reset_seeds,
            self.history_seat_masks,
            self.buffers.masks,
            self.buffers.tile_obs,
            self.buffers.melds,
            self.buffers.river,
            self.buffers.meta,
            self.buffers.events,
            self.buffers.event_lengths,
        )
        self.reset_flags.fill(0)
        self.buffers.refresh_legal()

        storage = TransitionStorage(transitions + self.config.envs * 2)
        pending = np.full(self.config.envs, -1, dtype=np.int64)
        cumulative_scores = np.zeros((self.config.envs, 4), dtype=np.int64)
        phase_seconds = {
            "policy_launch_seconds": 0.0,
            "rule_action_seconds": 0.0,
            "auxiliary_seconds": 0.0,
            "policy_resolve_seconds": 0.0,
        }
        finished = 0
        while finished < transitions:
            actors = self.buffers.meta[:, 1]
            learner_rows = np.flatnonzero(actors == self.learner_seats.astype(np.int32))
            kinds = self.pool.action_kinds(self.buffers, self.opponent_kinds)
            frozen_rows = np.flatnonzero(kinds == OpponentPool.FROZEN_TRANSFORMER)

            phase_start = time.perf_counter()
            learner_inference = launch_inference(
                model,
                self.buffers,
                learner_rows,
                self.device,
                history_cache=self.learner_history_cache,
            )
            frozen_inference = PendingInference.empty()
            if len(frozen_rows):
                frozen_model = self.pool.frozen_model
                if frozen_model is None:
                    raise RuntimeError("frozen opponent rows have no frozen model")
                frozen_inference = launch_inference(
                    frozen_model,
                    self.buffers,
                    frozen_rows,
                    self.device,
                    history_cache=self.frozen_history_cache,
                )
            phase_seconds["policy_launch_seconds"] += (
                time.perf_counter() - phase_start
            )

            phase_start = time.perf_counter()
            next_actions = self.pool.rule_actions(self.buffers, kinds)
            phase_seconds["rule_action_seconds"] += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            dense = self.buffers.legal_dense[learner_rows]
            shanten = np.full(len(learner_rows), 127, dtype=np.int8)
            improving = np.zeros(len(learner_rows), dtype=np.uint32)
            if collect_auxiliary and len(learner_rows):
                valid_rows = self.buffers.meta[learner_rows, 20] == 0
                analysis_positions = np.flatnonzero(valid_rows)
                if len(analysis_positions):
                    analysis_rows = learner_rows[analysis_positions].astype(np.uint32)
                    analysis_shanten = np.empty(len(analysis_rows), dtype=np.int8)
                    analysis_improving = np.empty(
                        len(analysis_rows), dtype=np.uint32
                    )
                    self.buffers.batch.hand_analysis_indices_into(
                        analysis_rows, analysis_shanten, analysis_improving
                    )
                    shanten[analysis_positions] = analysis_shanten
                    improving[analysis_positions] = analysis_improving
            phase_seconds["auxiliary_seconds"] += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            learner_actions, learner_logprob, learner_value = (
                learner_inference.resolve()
            )
            frozen_actions, _, _ = frozen_inference.resolve()
            phase_seconds["policy_resolve_seconds"] += (
                time.perf_counter() - phase_start
            )
            next_actions[frozen_rows] = frozen_actions

            if len(learner_rows):
                previous = pending[learner_rows]
                storage.finalize(previous, learner_value, False)
                finished += int(np.count_nonzero(previous >= 0))
            if finished >= transitions:
                break

            if len(learner_rows):
                pending[learner_rows] = storage.add_many(
                    learner_rows,
                    self.episode_ids[learner_rows],
                    self.buffers,
                    dense,
                    learner_actions.astype(np.int64),
                    learner_logprob,
                    learner_value,
                    shanten,
                    improving,
                )

            self.buffers.actions[:] = next_actions
            self.buffers.actions[learner_rows] = learner_actions
            self.buffers.batch.step_and_observe_history_into(
                self.buffers.actions,
                self.history_seat_masks,
                self.buffers.records,
                self.buffers.masks,
                self.buffers.tile_obs,
                self.buffers.melds,
                self.buffers.river,
                self.buffers.meta,
                self.buffers.events,
                self.buffers.event_lengths,
            )
            self.buffers.refresh_legal()

            score_deltas = self.buffers.records[:, 5:9]
            cumulative_scores += score_deltas
            terminal = self.buffers.records[:, 11].astype(bool)
            reward = hybrid_rewards(
                score_deltas,
                cumulative_scores,
                self.learner_seats,
                terminal,
                self.config,
            )
            active_pending = pending >= 0
            storage.reward[pending[active_pending]] += reward[active_pending]

            terminal_slots = pending[terminal]
            storage.finalize(
                terminal_slots, np.zeros(len(terminal_slots), dtype=np.float32), True
            )
            finished += int(np.count_nonzero(terminal_slots >= 0))
            pending[terminal] = -1
            terminal_rows = np.flatnonzero(terminal)
            cumulative_scores[terminal_rows] = 0
            self.learner_history_cache.clear_rows(terminal_rows)
            self.frozen_history_cache.clear_rows(terminal_rows)
            self._reset_rows(terminal_rows)

        indices = storage.indices(transitions)
        advantages, returns = storage.compute_gae(
            indices, self.config.gamma, self.config.gae_lambda
        )
        selected_advantages = advantages[indices]
        selected_advantages = (selected_advantages - selected_advantages.mean()) / (
            selected_advantages.std() + 1e-8
        )
        advantages[indices] = selected_advantages
        cache_stats = {
            f"{prefix}_{key}": value
            for prefix, cache in (
                ("learner_cache", self.learner_history_cache),
                ("frozen_cache", self.frozen_history_cache),
            )
            for key, value in cache.statistics().items()
        } | phase_seconds
        self.learner_history_cache.clear()
        self.frozen_history_cache.clear()
        return RolloutBatch(
            storage,
            indices,
            advantages,
            returns,
            self._opponent_counts.copy(),
            self.pool.stage(),
            self.pool.active_snapshot,
            cache_stats,
        )


def categorical_return_targets(returns: Tensor, support: Tensor) -> Tensor:
    values = torch.minimum(
        torch.maximum(returns.float(), support[0].float()), support[-1].float()
    )
    scale = (values - support[0]) / (support[1] - support[0])
    lower = scale.floor().long().clamp(0, len(support) - 1)
    upper = scale.ceil().long().clamp(0, len(support) - 1)
    upper_weight = scale - lower.float()
    lower_weight = 1.0 - upper_weight
    result = torch.zeros(
        (len(values), len(support)), device=returns.device, dtype=torch.float32
    )
    result.scatter_add_(1, lower[:, None], lower_weight[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


def cosine_learning_rate(config: PPOConfig, progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.final_learning_rate + weight * (
        config.learning_rate - config.final_learning_rate
    )


def _shuffled_minibatches(
    history_lengths: np.ndarray, minibatch_size: int
) -> list[np.ndarray]:
    """Shuffle globally, then sort only within each fixed minibatch."""
    if minibatch_size <= 0:
        raise ValueError("minibatch size must be positive")
    permutation = torch.randperm(len(history_lengths)).numpy()
    batches: list[np.ndarray] = []
    for start in range(0, len(permutation), minibatch_size):
        batch = permutation[start : start + minibatch_size]
        order = np.argsort(history_lengths[batch], kind="stable")
        batches.append(batch[order])
    return batches


@torch.no_grad()
def _policy_log_probabilities(
    model: BloodFlowTransformer,
    data: dict[str, Tensor],
    microbatch_size: int,
    device: torch.device,
) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, len(data["event_lengths"]), microbatch_size):
        stop = min(start + microbatch_size, len(data["event_lengths"]))
        lengths = data["event_lengths"][start:stop]
        history_width = max(int(lengths.max().item()), 1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                data["tile_obs"][start:stop],
                data["melds"][start:stop],
                data["meta"][start:stop],
                data["events"][start:stop, :history_width],
                lengths,
                data["legal"][start:stop],
            )
        chunks.append(F.log_softmax(output.logits.float(), dim=-1))
    return torch.cat(chunks)


def _mean_policy_kl(
    old_log_probs: Tensor, new_log_probs: Tensor, legal: Tensor
) -> float:
    terms = old_log_probs.exp() * (old_log_probs - new_log_probs)
    per_state = terms.masked_fill(~legal, 0.0).sum(dim=-1)
    return float(per_state.mean().cpu())


def ppo_update(
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PPOConfig,
    device: torch.device,
    progress: float,
) -> dict[str, float]:
    model.train()
    count = len(rollout)
    if count == 0:
        raise ValueError("cannot update PPO from an empty rollout")
    aux_scale = max(0.0, 1.0 - progress / max(config.auxiliary_decay_fraction, 1e-6))
    entropy_scale = config.entropy_coefficient + progress * (
        config.final_entropy_coefficient - config.entropy_coefficient
    )
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    completed_epochs = 0
    rolled_back_epochs = 0
    policy_kl = 0.0
    max_attempted_kl = 0.0
    improving_bits = torch.arange(27, device=device)
    history_lengths = rollout.storage.event_lengths[rollout.indices].astype(
        np.int64, copy=False
    )

    monitor_data: dict[str, Tensor] | None = None
    old_monitor_log_probs: Tensor | None = None
    if config.kl_control != "off":
        monitor_size = min(config.minibatch, max(config.microbatch, 1_024), count)
        monitor_order = _shuffled_minibatches(
            history_lengths, monitor_size
        )[0]
        monitor_lengths = history_lengths[monitor_order]
        monitor_width = max(int(monitor_lengths.max(initial=0)), 1)
        monitor_slots = rollout.indices[monitor_order]
        monitor_data = rollout.tensors(monitor_slots, device, monitor_width)
        model.eval()
        old_monitor_log_probs = _policy_log_probabilities(
            model, monitor_data, config.microbatch, device
        )
        model.train()

    for _ in range(config.ppo_epochs):
        model_backup = None
        optimizer_backup = None
        if config.kl_control == "rollback":
            model_backup = copy.deepcopy(model.state_dict())
            optimizer_backup = copy.deepcopy(optimizer.state_dict())
        epoch_sums = {key: 0.0 for key in sums}
        epoch_updates = 0
        for batch_order in _shuffled_minibatches(history_lengths, config.minibatch):
            minibatch_lengths = history_lengths[batch_order]
            minibatch_width = max(int(minibatch_lengths.max(initial=0)), 1)
            slots = rollout.indices[batch_order]
            minibatch_data = rollout.tensors(slots, device, minibatch_width)
            optimizer.zero_grad(set_to_none=True)
            minibatch_policy = torch.zeros((), device=device)
            minibatch_value = torch.zeros((), device=device)
            minibatch_entropy = torch.zeros((), device=device)
            minibatch_size = len(batch_order)
            for micro_start in range(0, minibatch_size, config.microbatch):
                micro_stop = min(micro_start + config.microbatch, minibatch_size)
                rows = slice(micro_start, micro_stop)
                history_width = max(
                    int(minibatch_lengths[micro_start:micro_stop].max(initial=0)), 1
                )
                data = {name: value[rows] for name, value in minibatch_data.items()}
                data["events"] = minibatch_data["events"][rows, :history_width]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(
                        data["tile_obs"],
                        data["melds"],
                        data["meta"],
                        data["events"],
                        data["event_lengths"],
                        data["legal"],
                    )
                    log_probs = F.log_softmax(output.logits.float(), dim=-1)
                    logprob = log_probs.gather(
                        1, data["actions"].long()[:, None]
                    ).squeeze(1)
                    ratio = (logprob - data["old_logprob"]).exp()
                    clipped = ratio.clamp(
                        1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                    )
                    policy_loss = -torch.minimum(
                        ratio * data["advantages"], clipped * data["advantages"]
                    ).mean()
                    target_distribution = categorical_return_targets(
                        data["returns"], model.value_support
                    )
                    value_loss = (
                        -(
                            target_distribution
                            * torch.log_softmax(output.value_logits.float(), dim=-1)
                        )
                        .sum(dim=-1)
                        .mean()
                    )
                    legal_count = data["legal"].sum(dim=-1).clamp_min(2).float()
                    entropy = (
                        -(log_probs.exp() * log_probs).sum(dim=-1)
                        / legal_count.log()
                    ).mean()
                    valid_shanten = (
                        (data["shanten"] >= -1)
                        & (data["shanten"] <= 8)
                        & (data["meta"][:, 20] == 0)
                    )
                    valid_weight = valid_shanten.float()
                    valid_count = valid_weight.sum().clamp_min(1.0)
                    shanten_targets = (data["shanten"].long() + 1).clamp(0, 9)
                    shanten_per_row = torch.nn.functional.cross_entropy(
                        output.shanten_logits, shanten_targets, reduction="none"
                    )
                    shanten_loss = (shanten_per_row * valid_weight).sum() / valid_count
                    target_improving = (
                        data["improving"].long()[:, None] >> improving_bits
                    ) & 1
                    improving_per_tile = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            output.improving_logits,
                            target_improving.float(),
                            reduction="none",
                        )
                    )
                    improving_loss = (
                        improving_per_tile.mean(dim=1) * valid_weight
                    ).sum() / valid_count
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - entropy_scale * entropy
                        + aux_scale
                        * (
                            config.shanten_coefficient * shanten_loss
                            + config.improving_coefficient * improving_loss
                        )
                    )
                scale = (micro_stop - micro_start) / minibatch_size
                (loss * scale).backward()
                minibatch_policy += policy_loss.detach() * scale
                minibatch_value += value_loss.detach() * scale
                minibatch_entropy += entropy.detach() * scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            minibatch_metrics = torch.stack(
                (
                    minibatch_policy,
                    minibatch_value,
                    minibatch_entropy,
                )
            ).float().cpu().numpy()
            epoch_sums["policy_loss"] += float(minibatch_metrics[0])
            epoch_sums["value_loss"] += float(minibatch_metrics[1])
            epoch_sums["entropy"] += float(minibatch_metrics[2])
            epoch_updates += 1

        attempted_kl = 0.0
        if monitor_data is not None and old_monitor_log_probs is not None:
            model.eval()
            new_monitor_log_probs = _policy_log_probabilities(
                model, monitor_data, config.microbatch, device
            )
            model.train()
            attempted_kl = _mean_policy_kl(
                old_monitor_log_probs,
                new_monitor_log_probs,
                monitor_data["legal"],
            )
            max_attempted_kl = max(max_attempted_kl, attempted_kl)

        if config.kl_control == "rollback" and attempted_kl > config.target_kl:
            if model_backup is None or optimizer_backup is None:
                raise RuntimeError("KL rollback has no saved optimizer state")
            model.load_state_dict(model_backup)
            optimizer.load_state_dict(optimizer_backup)
            rolled_back_epochs += 1
            policy_kl = attempted_kl
            break

        for key in sums:
            sums[key] += epoch_sums[key]
        updates += epoch_updates
        completed_epochs += 1
        policy_kl = attempted_kl

    return {key: value / max(updates, 1) for key, value in sums.items()} | {
        "approx_kl": policy_kl,
        "max_attempted_kl": max_attempted_kl,
        "updates": float(updates),
        "epochs": float(completed_epochs),
        "rolled_back_epochs": float(rolled_back_epochs),
        "kl_monitor_samples": float(
            0 if monitor_data is None else len(monitor_data["event_lengths"])
        ),
        "aux_scale": aux_scale,
        "entropy_scale": entropy_scale,
    }


def clone_model(
    model: BloodFlowTransformer, device: torch.device
) -> BloodFlowTransformer:
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def _validate_checkpoint_header(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint format version does not match")
    if checkpoint.get("engine_rules_version") != int(bm.ENGINE_RULES_VERSION):
        raise ValueError("checkpoint engine rules version does not match")
    if checkpoint.get("training_input_schema") != TRAINING_INPUT_SCHEMA:
        raise ValueError("checkpoint training input schema does not match")


def _model_config_from_state(state: Any) -> TransformerConfig:
    expected = {field.name for field in fields(TransformerConfig)}
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("checkpoint model configuration is invalid")
    return TransformerConfig(**state)


def _ppo_config_from_state(state: Any) -> PPOConfig:
    if not isinstance(state, dict):
        raise ValueError("checkpoint PPO configuration is invalid")
    state = dict(state)
    expected = {field.name for field in fields(PPOConfig)}
    legacy_curriculum = {
        "rule_mix_score_delta",
        "rule_league_score_delta",
        "rule_gate_consecutive_evals",
    }
    current_curriculum = {
        "self_play_fraction",
        "self_play_gate_score_delta",
        "self_play_gate_consecutive_evals",
        "historical_snapshot_probability",
    }
    if legacy_curriculum <= set(state) and not (current_curriculum & set(state)):
        league_score_delta = state.pop("rule_league_score_delta")
        gate_evaluations = state.pop("rule_gate_consecutive_evals")
        state.pop("rule_mix_score_delta")
        state.update(
            {
                "self_play_fraction": 0.25,
                "self_play_gate_score_delta": league_score_delta,
                "self_play_gate_consecutive_evals": gate_evaluations,
                "historical_snapshot_probability": 0.50,
            }
        )
    legacy_additions = {
        "kl_control": "monitor",
        "score_reward_weight": 1.0,
        "rank_reward_weight": 0.0,
    }
    if set(state) == expected - set(legacy_additions):
        state = state | legacy_additions
        warnings.warn(
            "resuming a legacy score-only PPO checkpoint with KL monitoring; "
            "start a new run to use hybrid rewards",
            stacklevel=3,
        )
    if set(state) != expected:
        raise ValueError("checkpoint PPO configuration is invalid")
    return PPOConfig(**state)


def checkpoint_configs(path: Path) -> tuple[TransformerConfig, PPOConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_header(checkpoint)
    return (
        _model_config_from_state(checkpoint.get("model_config")),
        _ppo_config_from_state(checkpoint.get("ppo_config")),
    )


def checkpoint_model_config(path: Path) -> TransformerConfig:
    return checkpoint_configs(path)[0]


def load_checkpoint_model(
    path: Path,
    device: torch.device,
) -> BloodFlowTransformer:
    """Load and validate the complete policy from a PPO checkpoint.

    The arena evaluator needs the policy weights but does not need the
    optimizer or rollout state. Keeping this loader next to the checkpoint
    validators prevents evaluation tools from silently accepting a checkpoint
    from a different engine or observation schema.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_header(checkpoint)
    model = BloodFlowTransformer(
        _model_config_from_state(checkpoint.get("model_config"))
    )
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint model state is invalid")
    model.load_state_dict(state)
    return model.to(device).eval()


def save_checkpoint(
    path: Path,
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    update: int,
    transitions: int,
    ppo_elapsed_seconds: float,
    config: PPOConfig,
    collector: RolloutCollector,
) -> None:
    if ppo_elapsed_seconds < 0.0:
        raise ValueError("PPO elapsed time cannot be negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
            "training_input_schema": TRAINING_INPUT_SCHEMA,
            "model_config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "transitions": transitions,
            "ppo_elapsed_seconds": ppo_elapsed_seconds,
            "ppo_config": asdict(config),
            "collector": collector.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if next(model.parameters()).device.type == "cuda"
                else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: PPOConfig,
    collector: RolloutCollector,
    *,
    expected_checkpoint_config: PPOConfig | None = None,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _validate_checkpoint_header(checkpoint)
    if _model_config_from_state(checkpoint.get("model_config")) != model.config:
        raise ValueError("checkpoint model configuration does not match")
    stored_config = _ppo_config_from_state(checkpoint.get("ppo_config"))
    expected_config = expected_checkpoint_config or config
    if stored_config != expected_config:
        raise ValueError("checkpoint PPO configuration does not match")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    collector.load_state_dict(checkpoint["collector"], model.config)
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
    elapsed = float(checkpoint["ppo_elapsed_seconds"])
    if elapsed < 0.0:
        raise ValueError("checkpoint PPO elapsed time cannot be negative")
    return int(checkpoint["update"]), int(checkpoint["transitions"]), elapsed


def evaluation_panel(games: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return paired seeds with each focal seat represented once per seed."""
    if games <= 0 or games % 4 != 0:
        raise ValueError("evaluation games must be a positive multiple of four")
    base_seeds = _seed_sequence(seed, games // 4)
    return np.repeat(base_seeds, 4), np.tile(np.arange(4, dtype=np.uint8), games // 4)


def evaluate_against_rule_ev(
    model: BloodFlowTransformer,
    device: torch.device,
    games: int = 256,
    envs: int = 64,
    seed: int = 0xA51CE,
) -> dict[str, float]:
    """Evaluate one deterministic learner against three Rule-EV opponents.

    Every requested ``(seed, focal_seat)`` is consumed exactly once. Finished
    rows stay disabled, which avoids the short-game bias caused by refilling a
    batch until an aggregate game count is reached.
    """
    panel_seeds, panel_seats = evaluation_panel(games, seed)
    envs = max(1, min(envs, games))
    completed_scores: list[tuple[int, int]] = []
    rule_ev_config = bm.RuleEvConfig.standard()
    model.eval()
    for start in range(0, games, envs):
        stop = min(start + envs, games)
        chunk_seeds = panel_seeds[start:stop]
        learner_seats = panel_seats[start:stop]
        size = len(chunk_seeds)
        buffers = EngineBuffers.create(size, history=192)
        reset_flags = np.ones(size, dtype=np.uint8)
        history_seat_masks = np.full(size, 0x0F, dtype=np.uint8)
        buffers.batch.reset_and_observe_history_into(
            reset_flags,
            chunk_seeds,
            history_seat_masks,
            buffers.masks,
            buffers.tile_obs,
            buffers.melds,
            buffers.river,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
        )
        buffers.refresh_legal()
        active = np.ones(size, dtype=np.bool_)
        cumulative = np.zeros((size, 4), dtype=np.int64)
        step_enabled = np.ones(size, dtype=np.uint8)
        rule_enabled = np.ones(size, dtype=np.uint8)
        while active.any():
            actors = buffers.meta[:, 1]
            learner_rows = np.flatnonzero(
                active & (actors == learner_seats.astype(np.int32))
            )
            learner_inference = launch_inference(
                model, buffers, learner_rows, device, deterministic=True
            )
            rule_enabled[:] = active
            rule_enabled[learner_rows] = 0
            buffers.batch.rule_ev_actions_masked_into(
                rule_enabled, buffers.actions, rule_ev_config
            )
            learner_actions, _, _ = learner_inference.resolve()
            buffers.actions[learner_rows] = learner_actions
            step_enabled[:] = active
            buffers.batch.step_masked_into(
                step_enabled, buffers.actions, buffers.records
            )
            cumulative[active] += buffers.records[active, 5:9]
            terminal = active & buffers.records[:, 11].astype(bool)
            terminal_rows = np.flatnonzero(terminal)
            ranks = focal_ranks(
                cumulative[terminal_rows], learner_seats[terminal_rows]
            )
            for row, rank in zip(terminal_rows, ranks, strict=True):
                completed_scores.append(
                    (int(cumulative[row, learner_seats[row]]), int(rank))
                )
            active[terminal] = False
            if active.any():
                buffers.refresh()

    values = np.asarray(
        [score for score, _ in completed_scores], dtype=np.float64
    )
    ranks = np.asarray([rank for _, rank in completed_scores], dtype=np.float64)
    if len(values) != games:
        raise RuntimeError("fixed evaluation did not finish every panel game")
    return {
        "games": float(len(values)),
        "mean_score_delta": float(values.mean()),
        "score_std": float(values.std()),
        "first_rate": float(np.mean(ranks == 1)),
        "last_rate": float(np.mean(ranks == 4)),
        "mean_rank": float(ranks.mean()),
    }
