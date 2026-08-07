"""End-to-end rollout, opponent scheduling, and PPO utilities."""

from __future__ import annotations

import copy
import math
import random
import time
import warnings
from collections.abc import Mapping
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
    minimum_learning_rate: float = 3e-5
    learning_rate_decay: float = 0.5
    learning_rate_patience_evaluations: int = 6
    learning_rate_rank_improvement: float = 0.02
    learning_rate_score_improvement: float = 50.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    minimum_entropy_coefficient: float = 0.001
    maximum_entropy_coefficient: float = 0.02
    entropy_target_low: float = 0.07
    entropy_target_high: float = 0.18
    entropy_adjustment: float = 1.25
    entropy_patience_evaluations: int = 3
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    kl_control: str = "monitor"
    score_reward_weight: float = 1.0
    rank_reward_weight: float = 1.0
    shanten_coefficient: float = 0.02
    improving_coefficient: float = 0.01
    self_play_enabled: bool = True
    self_play_max_fraction: float = 0.45
    self_play_fraction_step: float = 0.15
    self_play_gate_score_delta: float = 75.0
    self_play_gate_mean_rank: float = 2.45
    self_play_fallback_score_delta: float = -75.0
    self_play_fallback_mean_rank: float = 2.55
    self_play_gate_confidence_z: float = 1.96
    self_play_gate_window: int = 3
    self_play_gate_required_passes: int = 2
    historical_snapshot_probability: float = 0.50
    frozen_snapshot_limit: int = 4

    def __post_init__(self) -> None:
        for name in (
            "envs",
            "rollout_transitions",
            "ppo_epochs",
            "minibatch",
            "microbatch",
            "history_cache_min_batch",
            "learning_rate_patience_evaluations",
            "entropy_patience_evaluations",
            "self_play_gate_window",
            "self_play_gate_required_passes",
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
        if not (
            math.isfinite(self.learning_rate)
            and math.isfinite(self.minimum_learning_rate)
            and 0.0 < self.minimum_learning_rate <= self.learning_rate
        ):
            raise ValueError("minimum learning rate must be in (0, learning_rate]")
        if not math.isfinite(self.learning_rate_decay) or not (
            0.0 < self.learning_rate_decay < 1.0
        ):
            raise ValueError("learning rate decay must be in (0, 1)")
        if not (
            math.isfinite(self.minimum_entropy_coefficient)
            and math.isfinite(self.maximum_entropy_coefficient)
            and 0.0 <= self.minimum_entropy_coefficient
            <= self.entropy_coefficient
            <= self.maximum_entropy_coefficient
        ):
            raise ValueError("entropy coefficient limits are invalid")
        if not (
            0.0 <= self.entropy_target_low < self.entropy_target_high <= 1.0
        ):
            raise ValueError("entropy targets must be ordered within [0, 1]")
        if not math.isfinite(self.entropy_adjustment) or self.entropy_adjustment <= 1.0:
            raise ValueError("entropy adjustment must be greater than one")
        if not math.isfinite(self.self_play_max_fraction) or not (
            0.0 < self.self_play_max_fraction < 1.0
        ):
            raise ValueError("maximum self-play fraction must be in (0, 1)")
        if not math.isfinite(self.self_play_fraction_step) or not (
            0.0 < self.self_play_fraction_step <= self.self_play_max_fraction
        ):
            raise ValueError("self-play fraction step is invalid")
        for name in (
            "self_play_gate_score_delta",
            "self_play_gate_mean_rank",
            "self_play_fallback_score_delta",
            "self_play_fallback_mean_rank",
            "self_play_gate_confidence_z",
            "learning_rate_rank_improvement",
            "learning_rate_score_improvement",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.self_play_gate_confidence_z <= 0.0:
            raise ValueError("self-play gate confidence z must be positive")
        if (
            self.learning_rate_rank_improvement < 0.0
            or self.learning_rate_score_improvement < 0.0
        ):
            raise ValueError("learning-rate improvement margins cannot be negative")
        if self.self_play_gate_required_passes > self.self_play_gate_window:
            raise ValueError("required gate passes cannot exceed the gate window")
        if not (
            self.self_play_gate_mean_rank < self.self_play_fallback_mean_rank
            and self.self_play_fallback_score_delta < self.self_play_gate_score_delta
        ):
            raise ValueError("self-play fallback thresholds must provide hysteresis")
        if not math.isfinite(self.historical_snapshot_probability) or not (
            0.0 <= self.historical_snapshot_probability <= 1.0
        ):
            raise ValueError("historical snapshot probability must be in [0, 1]")


@dataclass(frozen=True)
class TrainingControls:
    learning_rate: float
    entropy_coefficient: float
    auxiliary_scale: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("controlled learning rate must be finite and positive")
        if (
            not math.isfinite(self.entropy_coefficient)
            or self.entropy_coefficient < 0.0
        ):
            raise ValueError("controlled entropy coefficient must be finite and non-negative")
        if not math.isfinite(self.auxiliary_scale) or not (
            0.0 <= self.auxiliary_scale <= 1.0
        ):
            raise ValueError("controlled auxiliary scale must be within [0, 1]")


def _confidence_bounds(mean: float, standard_error: float, z: float) -> tuple[float, float]:
    radius = z * standard_error
    return mean - radius, mean + radius


def _pooled_panel_statistics(
    evaluations: list[dict[str, float]],
) -> dict[str, float]:
    """Combine independent, seat-balanced evaluation panels."""
    if not evaluations:
        raise ValueError("cannot pool an empty evaluation window")

    def combine(prefix: str) -> tuple[float, float, float]:
        count = sum(int(item["panel_count"]) for item in evaluations)
        if count <= 0:
            raise ValueError("evaluation panel count must be positive")
        mean = sum(
            int(item["panel_count"]) * float(item[f"mean_{prefix}"])
            for item in evaluations
        ) / count
        m2 = 0.0
        for item in evaluations:
            item_count = int(item["panel_count"])
            item_mean = float(item[f"mean_{prefix}"])
            item_std = float(item[f"{prefix}_panel_std"])
            m2 += max(item_count - 1, 0) * item_std * item_std
            m2 += item_count * (item_mean - mean) ** 2
        standard_deviation = math.sqrt(m2 / (count - 1)) if count > 1 else 0.0
        standard_error = standard_deviation / math.sqrt(count) if count > 1 else 0.0
        return mean, standard_deviation, standard_error

    score_mean, score_std, score_se = combine("score_delta")
    rank_mean, rank_std, rank_se = combine("rank")
    return {
        "panel_count": float(sum(int(item["panel_count"]) for item in evaluations)),
        "mean_score_delta": score_mean,
        "score_delta_panel_std": score_std,
        "score_se": score_se,
        "mean_rank": rank_mean,
        "rank_panel_std": rank_std,
        "rank_se": rank_se,
    }


class TrainingController:
    """Metric-driven learning controls and Rule-EV curriculum evidence."""

    AUXILIARY_SCALES = (1.0, 0.5, 0.25, 0.0)
    EVALUATION_FIELDS = frozenset(
        {
            "panel_count",
            "mean_score_delta",
            "score_delta_panel_std",
            "score_se",
            "mean_rank",
            "rank_panel_std",
            "rank_se",
        }
    )

    def __init__(self, config: PPOConfig, evaluation_seed: int = 0) -> None:
        self.config = config
        self.current_learning_rate = config.learning_rate
        self.current_entropy_coefficient = config.entropy_coefficient
        self.learning_rate_plateau = 0
        self.entropy_low_streak = 0
        self.entropy_high_streak = 0
        self.best_rank: float | None = None
        self.best_score_delta: float | None = None
        self.champion_rank: float | None = None
        self.champion_score_delta: float | None = None
        self.last_decision_evaluation: dict[str, float] | None = None
        self.evaluation_window: list[dict[str, float]] = []
        self.next_evaluation_seed = int(evaluation_seed)
        self.evaluation_count = 0
        self.entropy_sum = 0.0
        self.entropy_updates = 0
        self.last_decision = "hold"

    def controls(self, self_play_level: int) -> TrainingControls:
        level = min(max(int(self_play_level), 0), len(self.AUXILIARY_SCALES) - 1)
        return TrainingControls(
            learning_rate=self.current_learning_rate,
            entropy_coefficient=self.current_entropy_coefficient,
            auxiliary_scale=self.AUXILIARY_SCALES[level],
        )

    def reset_learning_rate_schedule(self) -> None:
        """Start a new learning-rate schedule from the configured initial rate."""
        self.current_learning_rate = self.config.learning_rate
        self.learning_rate_plateau = 0
        self.best_rank = None
        self.best_score_delta = None

    def take_evaluation_seed(self, games: int) -> int:
        if games <= 0 or games % 4:
            raise ValueError("evaluation games must be a positive multiple of four")
        seed = self.next_evaluation_seed
        self.next_evaluation_seed += games // 4
        return seed

    def observe_update(self, statistics: dict[str, float]) -> None:
        entropy = float(statistics["entropy"])
        if not math.isfinite(entropy):
            raise ValueError("training entropy must be finite")
        self.entropy_sum += entropy
        self.entropy_updates += 1

    @classmethod
    def _normalize_evaluation(
        cls, evaluation: Mapping[str, Any]
    ) -> dict[str, float]:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Rule-EV evaluation must be a mapping")
        missing = cls.EVALUATION_FIELDS - evaluation.keys()
        if missing:
            raise ValueError(
                "Rule-EV evaluation is missing: " + ", ".join(sorted(missing))
            )
        values = {
            name: float(evaluation[name]) for name in cls.EVALUATION_FIELDS
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Rule-EV evaluation contains a non-finite value")
        panel_count = values["panel_count"]
        if panel_count < 1.0 or not panel_count.is_integer():
            raise ValueError("Rule-EV evaluation panel count must be a positive integer")
        for name in (
            "score_delta_panel_std",
            "score_se",
            "rank_panel_std",
            "rank_se",
        ):
            if values[name] < 0.0:
                raise ValueError("Rule-EV evaluation uncertainty cannot be negative")
        if not 1.0 <= values["mean_rank"] <= 4.0:
            raise ValueError("Rule-EV mean rank must be within [1, 4]")
        return values

    def _single_passes(self, evaluation: dict[str, float]) -> bool:
        if int(evaluation["panel_count"]) < 2:
            return False
        score_low, _ = _confidence_bounds(
            float(evaluation["mean_score_delta"]),
            float(evaluation["score_se"]),
            self.config.self_play_gate_confidence_z,
        )
        _, rank_high = _confidence_bounds(
            float(evaluation["mean_rank"]),
            float(evaluation["rank_se"]),
            self.config.self_play_gate_confidence_z,
        )
        return (
            score_low > self.config.self_play_gate_score_delta
            and rank_high < self.config.self_play_gate_mean_rank
        )

    def _single_fails(self, evaluation: dict[str, float]) -> bool:
        if int(evaluation["panel_count"]) < 2:
            return False
        _, score_high = _confidence_bounds(
            float(evaluation["mean_score_delta"]),
            float(evaluation["score_se"]),
            self.config.self_play_gate_confidence_z,
        )
        rank_low, _ = _confidence_bounds(
            float(evaluation["mean_rank"]),
            float(evaluation["rank_se"]),
            self.config.self_play_gate_confidence_z,
        )
        return (
            score_high < self.config.self_play_fallback_score_delta
            or rank_low > self.config.self_play_fallback_mean_rank
        )

    def _update_learning_rate(self, evaluation: dict[str, float]) -> None:
        rank = float(evaluation["mean_rank"])
        score = float(evaluation["mean_score_delta"])
        improved = self.best_rank is None or self.best_score_delta is None
        if not improved:
            improved = rank < self.best_rank - self.config.learning_rate_rank_improvement
            if not improved and rank <= self.best_rank + self.config.learning_rate_rank_improvement:
                improved = (
                    score
                    > self.best_score_delta
                    + self.config.learning_rate_score_improvement
                )
        if improved:
            self.best_rank = rank
            self.best_score_delta = score
            self.learning_rate_plateau = 0
            return
        self.learning_rate_plateau += 1
        if (
            self.learning_rate_plateau
            >= self.config.learning_rate_patience_evaluations
        ):
            self.current_learning_rate = max(
                self.config.minimum_learning_rate,
                self.current_learning_rate * self.config.learning_rate_decay,
            )
            self.learning_rate_plateau = 0

    def _update_entropy(self) -> None:
        if self.entropy_updates == 0:
            return
        entropy = self.entropy_sum / self.entropy_updates
        self.entropy_sum = 0.0
        self.entropy_updates = 0
        if entropy < self.config.entropy_target_low:
            self.entropy_low_streak += 1
            self.entropy_high_streak = 0
        elif entropy > self.config.entropy_target_high:
            self.entropy_high_streak += 1
            self.entropy_low_streak = 0
        else:
            self.entropy_low_streak = 0
            self.entropy_high_streak = 0
        patience = self.config.entropy_patience_evaluations
        if self.entropy_low_streak >= patience:
            self.current_entropy_coefficient = min(
                self.config.maximum_entropy_coefficient,
                self.current_entropy_coefficient * self.config.entropy_adjustment,
            )
            self.entropy_low_streak = 0
        elif self.entropy_high_streak >= patience:
            self.current_entropy_coefficient = max(
                self.config.minimum_entropy_coefficient,
                self.current_entropy_coefficient / self.config.entropy_adjustment,
            )
            self.entropy_high_streak = 0

    def observe_rule_ev(
        self,
        evaluation: dict[str, Any],
        *,
        self_play_level: int,
        maximum_self_play_level: int,
    ) -> str:
        if evaluation.get("opponent") != "rule-ev":
            raise ValueError("training controls accept only Rule-EV evaluations")
        normalized = self._normalize_evaluation(evaluation)

        self.evaluation_count += 1
        self._update_learning_rate(normalized)
        self._update_entropy()
        self.evaluation_window.append(normalized)
        self.evaluation_window = self.evaluation_window[
            -self.config.self_play_gate_window :
        ]
        self.last_decision = "hold"
        self.last_decision_evaluation = None
        if len(self.evaluation_window) < self.config.self_play_gate_window:
            return self.last_decision

        pooled = _pooled_panel_statistics(self.evaluation_window)
        latest_passes = self._single_passes(self.evaluation_window[-1])
        pass_count = sum(self._single_passes(item) for item in self.evaluation_window)
        pooled_passes = self._single_passes(pooled)
        latest_fails = self._single_fails(self.evaluation_window[-1])
        fail_count = sum(self._single_fails(item) for item in self.evaluation_window)
        pooled_fails = self._single_fails(pooled)

        if (
            self.config.self_play_enabled
            and latest_passes
            and pooled_passes
            and pass_count >= self.config.self_play_gate_required_passes
        ):
            if self_play_level < maximum_self_play_level:
                self.last_decision = "promote"
            elif self._improves_champion(pooled):
                self.last_decision = "refresh"
        elif (
            self_play_level > 0
            and latest_fails
            and pooled_fails
            and fail_count >= self.config.self_play_gate_required_passes
        ):
            self.last_decision = "demote"

        if self.last_decision != "hold":
            self.last_decision_evaluation = pooled
            self.evaluation_window.clear()
            self.learning_rate_plateau = 0
        return self.last_decision

    def _improves_champion(self, evaluation: dict[str, float]) -> bool:
        if self.champion_rank is None or self.champion_score_delta is None:
            return True
        rank = float(evaluation["mean_rank"])
        score = float(evaluation["mean_score_delta"])
        return (
            rank < self.champion_rank - self.config.learning_rate_rank_improvement
            or (
                rank <= self.champion_rank + self.config.learning_rate_rank_improvement
                and score
                > self.champion_score_delta
                + self.config.learning_rate_score_improvement
            )
        )

    def mark_champion(self) -> None:
        if self.last_decision_evaluation is None:
            raise RuntimeError("cannot mark a champion without gate evidence")
        self.champion_rank = float(self.last_decision_evaluation["mean_rank"])
        self.champion_score_delta = float(
            self.last_decision_evaluation["mean_score_delta"]
        )

    def gate_statistics(self) -> dict[str, Any]:
        pooled = (
            _pooled_panel_statistics(self.evaluation_window)
            if self.evaluation_window
            else None
        )
        return {
            "window_size": len(self.evaluation_window),
            "required_window_size": self.config.self_play_gate_window,
            "single_passes": sum(
                self._single_passes(item) for item in self.evaluation_window
            ),
            "single_fails": sum(
                self._single_fails(item) for item in self.evaluation_window
            ),
            "pooled": pooled,
            "last_decision": self.last_decision,
            "decision_evaluation": self.last_decision_evaluation,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_learning_rate": self.current_learning_rate,
            "current_entropy_coefficient": self.current_entropy_coefficient,
            "learning_rate_plateau": self.learning_rate_plateau,
            "entropy_low_streak": self.entropy_low_streak,
            "entropy_high_streak": self.entropy_high_streak,
            "best_rank": self.best_rank,
            "best_score_delta": self.best_score_delta,
            "champion_rank": self.champion_rank,
            "champion_score_delta": self.champion_score_delta,
            "last_decision_evaluation": self.last_decision_evaluation,
            "evaluation_window": self.evaluation_window,
            "next_evaluation_seed": self.next_evaluation_seed,
            "evaluation_count": self.evaluation_count,
            "entropy_sum": self.entropy_sum,
            "entropy_updates": self.entropy_updates,
            "last_decision": self.last_decision,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            warnings.warn(
                "checkpoint has no metric-controller state; gate evidence was reset",
                stacklevel=3,
            )
            return
        self.current_learning_rate = float(state["current_learning_rate"])
        self.current_entropy_coefficient = float(
            state["current_entropy_coefficient"]
        )
        self.learning_rate_plateau = int(state["learning_rate_plateau"])
        self.entropy_low_streak = int(state["entropy_low_streak"])
        self.entropy_high_streak = int(state["entropy_high_streak"])
        self.best_rank = state.get("best_rank")
        self.best_score_delta = state.get("best_score_delta")
        self.champion_rank = state.get("champion_rank")
        self.champion_score_delta = state.get("champion_score_delta")
        decision_evaluation = state.get("last_decision_evaluation")
        self.last_decision_evaluation = (
            self._normalize_evaluation(decision_evaluation)
            if isinstance(decision_evaluation, dict)
            else None
        )
        self.evaluation_window = [
            self._normalize_evaluation(item)
            for item in state.get("evaluation_window", [])
        ][-self.config.self_play_gate_window :]
        self.next_evaluation_seed = int(state["next_evaluation_seed"])
        self.evaluation_count = int(state["evaluation_count"])
        self.entropy_sum = float(state.get("entropy_sum", 0.0))
        self.entropy_updates = int(state.get("entropy_updates", 0))
        self.last_decision = str(state.get("last_decision", "hold"))


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
            tile_obs=np.empty(
                (batch_size, bm.TILE_OBSERVATION_PLANES, bm.TILE_KIND_COUNT),
                dtype=np.uint8,
            ),
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
        self.tile_obs = np.empty(
            (capacity, bm.TILE_OBSERVATION_PLANES, bm.TILE_KIND_COUNT),
            dtype=np.uint8,
        )
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
    """Rule opponents and a metric-controlled frozen-policy pool.

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
        self.self_play_level = 0

    @property
    def frozen_model(self) -> BloodFlowTransformer | None:
        if self.active_snapshot is None:
            return None
        return self.snapshots[self.active_snapshot]

    @property
    def frozen_ready(self) -> bool:
        return self.frozen_model is not None

    @property
    def maximum_self_play_level(self) -> int:
        return int(
            math.ceil(
                self.config.self_play_max_fraction
                / self.config.self_play_fraction_step
                - 1e-12
            )
        )

    @property
    def frozen_fraction(self) -> float:
        if not self.config.self_play_enabled or self.self_play_level <= 0:
            return 0.0
        return min(
            self.self_play_level * self.config.self_play_fraction_step,
            self.config.self_play_max_fraction,
        )

    def stage(self) -> str:
        """Return the executable opponent stage from rule competence.

        Transition count is deliberately absent from this decision. The
        metric controller changes ``self_play_level`` only after Rule-EV
        confidence gates have passed.
        """
        if (
            not self.config.self_play_enabled
            or self.self_play_level == 0
            or not self.frozen_ready
        ):
            return "bootstrap"
        return "self_play"

    def probabilities(self) -> np.ndarray:
        """Probability of each policy for one non-learner seat."""
        if self.stage() == "bootstrap":
            return np.array([1.0 / 3.0, 2.0 / 3.0, 0.0], dtype=np.float64)
        rule_fraction = 1.0 - self.frozen_fraction
        return np.array(
            [
                rule_fraction / 3.0,
                2.0 * rule_fraction / 3.0,
                self.frozen_fraction,
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
        """Add a frozen learner snapshot after a metric-controller decision."""
        if not self.config.self_play_enabled or self.self_play_level == 0:
            return None
        self.snapshots.append(clone_model(model, device))
        retained = self._retained_snapshot_indices(len(self.snapshots))
        if len(retained) < len(self.snapshots):
            self.snapshots[:] = [self.snapshots[index] for index in retained]
        self.last_snapshot_update = update
        return self.select_snapshot()

    def _retained_snapshot_indices(self, count: int) -> list[int]:
        limit = self.config.frozen_snapshot_limit
        if count <= limit:
            return list(range(count))
        if limit == 1:
            return [count - 1]
        recent = range(count - limit + 1, count)
        return [0, *recent]

    def promote(
        self,
        model: BloodFlowTransformer,
        device: torch.device,
        *,
        update: int,
    ) -> int | None:
        """Advance one self-play level and install the current champion."""
        if not self.config.self_play_enabled:
            return None
        self.self_play_level = min(
            self.self_play_level + 1, self.maximum_self_play_level
        )
        return self.refresh_snapshot(model, device, update=update)

    def demote(self) -> int:
        """Reduce frozen-policy exposure without deleting league history."""
        self.self_play_level = max(self.self_play_level - 1, 0)
        return self.self_play_level

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
            if self.config.self_play_enabled:
                self.self_play_level = 1
        else:
            self.self_play_level = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.random.bit_generator.state,
            "active_snapshot": self.active_snapshot,
            "last_snapshot_update": self.last_snapshot_update,
            "snapshots": [snapshot.state_dict() for snapshot in self.snapshots],
            "self_play_level": self.self_play_level,
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
        snapshot_states = list(state.get("snapshots", []))
        retained = self._retained_snapshot_indices(len(snapshot_states))
        for source_index in retained:
            snapshot = BloodFlowTransformer(model_config).to(device)
            snapshot.load_state_dict(snapshot_states[source_index])
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
            self.snapshots.append(snapshot)
        active = state.get("active_snapshot")
        source_active = int(active) if active is not None else None
        retained_positions = {
            source_index: position for position, source_index in enumerate(retained)
        }
        if source_active is None or not 0 <= source_active < len(snapshot_states):
            self.active_snapshot = None
        else:
            self.active_snapshot = retained_positions.get(
                source_active, len(self.snapshots) - 1
            )
        if not self.config.self_play_enabled:
            self.self_play_level = 0
        elif "self_play_level" in state:
            self.self_play_level = min(
                max(int(state["self_play_level"]), 0),
                self.maximum_self_play_level,
            )
        else:
            legacy_streak = int(
                state.get("rule_gate_streak", state.get("rule_league_streak", 0))
            )
            self.self_play_level = int(bool(self.snapshots and legacy_streak))

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
    controls: TrainingControls,
) -> dict[str, float]:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = controls.learning_rate
    model.train()
    count = len(rollout)
    if count == 0:
        raise ValueError("cannot update PPO from an empty rollout")
    aux_scale = controls.auxiliary_scale
    entropy_scale = controls.entropy_coefficient
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
    defaults = asdict(PPOConfig())
    expected = set(defaults)
    migrated = False
    legacy_curriculum = {
        "rule_mix_score_delta",
        "rule_league_score_delta",
        "rule_gate_consecutive_evals",
    }
    if legacy_curriculum <= set(state) and "self_play_gate_score_delta" not in state:
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
        migrated = True

    for name, value in {
        "kl_control": "monitor",
        "score_reward_weight": 1.0,
        "rank_reward_weight": 0.0,
    }.items():
        if name not in state:
            state[name] = value
            migrated = True

    if "final_learning_rate" in state:
        state["minimum_learning_rate"] = state.pop("final_learning_rate")
        migrated = True
    if "final_entropy_coefficient" in state:
        state["minimum_entropy_coefficient"] = state.pop(
            "final_entropy_coefficient"
        )
        migrated = True
    if "self_play_fraction" in state:
        fraction = float(state.pop("self_play_fraction"))
        state["self_play_max_fraction"] = max(fraction, 1e-6)
        state["self_play_fraction_step"] = min(
            defaults["self_play_fraction_step"], max(fraction, 1e-6)
        )
        migrated = True
    if "self_play_gate_consecutive_evals" in state:
        evaluations = max(int(state.pop("self_play_gate_consecutive_evals")), 1)
        state["self_play_gate_window"] = evaluations
        state["self_play_gate_required_passes"] = max(1, evaluations - 1)
        migrated = True
    for name in ("schedule_hours", "auxiliary_decay_fraction", "opponent_refresh_updates"):
        if name in state:
            state.pop(name)
            migrated = True

    unknown = set(state) - expected
    if unknown:
        raise ValueError(
            "checkpoint PPO configuration has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    for name, value in defaults.items():
        if name not in state:
            state[name] = value
            migrated = True
    if migrated:
        warnings.warn(
            "migrated a legacy PPO configuration to metric-driven scheduling; "
            "gate evidence will restart when controller state is absent",
            stacklevel=3,
        )
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
    controller: TrainingController | None = None,
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
            "training_controller": (
                controller.state_dict() if controller is not None else None
            ),
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
    controller: TrainingController | None = None,
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
    if controller is not None:
        controller.load_state_dict(checkpoint.get("training_controller"))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = controller.current_learning_rate
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


def _panel_statistics(values: np.ndarray) -> tuple[float, float]:
    panel_means = values.reshape(-1, 4).mean(axis=1)
    if len(panel_means) < 2:
        return 0.0, 0.0
    standard_deviation = float(panel_means.std(ddof=1))
    return standard_deviation, standard_deviation / math.sqrt(len(panel_means))


def evaluate_against_rule_policy(
    model: BloodFlowTransformer,
    device: torch.device,
    *,
    opponent: str,
    rule_nn: Any | None = None,
    games: int = 256,
    envs: int = 64,
    seed: int = 0xA51CE,
) -> dict[str, Any]:
    """Evaluate one deterministic learner against one fixed rule policy.

    Every requested ``(seed, focal_seat)`` is consumed exactly once. Finished
    rows stay disabled, which avoids the short-game bias caused by refilling a
    batch until an aggregate game count is reached.
    """
    supported = {"rule-fast", "rule-ev", "rule-nn"}
    if opponent not in supported:
        raise ValueError(f"unsupported evaluation opponent: {opponent}")
    if opponent == "rule-nn" and rule_nn is None:
        raise ValueError("rule-nn evaluation requires an ONNX policy")
    panel_seeds, panel_seats = evaluation_panel(games, seed)
    envs = max(1, min(envs, games))
    score_values = np.empty(games, dtype=np.float64)
    rank_values = np.empty(games, dtype=np.float64)
    completed = np.zeros(games, dtype=np.bool_)
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
            if opponent == "rule-fast":
                buffers.batch.simple_rule_actions_masked_into(
                    rule_enabled, buffers.actions
                )
            elif opponent == "rule-ev":
                buffers.batch.rule_ev_actions_masked_into(
                    rule_enabled, buffers.actions, rule_ev_config
                )
            else:
                buffers.batch.rule_nn_actions_masked_into(
                    rule_enabled, buffers.actions, rule_nn
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
            game_indices = start + terminal_rows
            score_values[game_indices] = cumulative[
                terminal_rows, learner_seats[terminal_rows]
            ]
            rank_values[game_indices] = ranks
            completed[game_indices] = True
            active[terminal] = False
            if active.any():
                buffers.refresh()

    if not completed.all():
        raise RuntimeError("fixed evaluation did not finish every panel game")
    score_panel_std, score_se = _panel_statistics(score_values)
    rank_panel_std, rank_se = _panel_statistics(rank_values)
    return {
        "opponent": opponent,
        "games": float(games),
        "panel_count": float(games // 4),
        "mean_score_delta": float(score_values.mean()),
        "score_std": float(score_values.std()),
        "score_delta_panel_std": score_panel_std,
        "score_se": score_se,
        "first_rate": float(np.mean(rank_values == 1)),
        "last_rate": float(np.mean(rank_values == 4)),
        "mean_rank": float(rank_values.mean()),
        "rank_std": float(rank_values.std()),
        "rank_panel_std": rank_panel_std,
        "rank_se": rank_se,
    }


def evaluate_against_rule_ev(
    model: BloodFlowTransformer,
    device: torch.device,
    games: int = 256,
    envs: int = 64,
    seed: int = 0xA51CE,
) -> dict[str, Any]:
    """Return the only evaluation result accepted by the internal gate."""
    return evaluate_against_rule_policy(
        model,
        device,
        opponent="rule-ev",
        games=games,
        envs=envs,
        seed=seed,
    )
