"""Policy mixtures, category-aware exploration, and balanced replay sampling.

The objects in this module deliberately contain no model instances.  They are
small, JSON-serializable control-plane state that can be checkpointed beside
the Actor and Critics while the training loop owns the actual modules.
"""

from __future__ import annotations

import copy
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from os import PathLike
from typing import Hashable, Mapping, Sequence

import numpy as np


ACTION_SPACE_SIZE = 115
POLICY_POOL_STATE_VERSION = 1
EXPLORATION_STATE_VERSION = 1
REPLAY_SAMPLER_STATE_VERSION = 1


class DecisionCategory(IntEnum):
    EXCHANGE_FIRST = 0
    EXCHANGE_SECOND = 1
    EXCHANGE_THIRD = 2
    CHOOSE_MISSING = 3
    TURN_EARLY = 4
    TURN_MIDDLE = 5
    TURN_LATE = 6
    HU_RESPONSE = 7
    MELD_RESPONSE = 8


CATEGORY_NAMES = tuple(category.name.lower() for category in DecisionCategory)
CATEGORY_COUNT = len(CATEGORY_NAMES)
DEFAULT_CATEGORY_WEIGHTS = (
    0.05,
    0.03,
    0.04,
    0.05,
    0.17,
    0.22,
    0.18,
    0.12,
    0.14,
)


class ReplaySource(IntEnum):
    """Compact trajectory source code.  Integer values are a stable schema."""

    SL = 0
    RULE_FAST = 1
    RULE_SAFE = 2
    CURRENT = 3
    FROZEN_POLICY = 4
    MC_TEACHER = 5

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: ReplaySource | str | int | np.integer) -> ReplaySource:
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, np.integer)):
            return cls(int(value))
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError as error:
                raise ValueError(f"unknown replay source {value!r}") from error
        raise TypeError(f"unsupported replay source {value!r}")


OPPONENT_SOURCES = (
    ReplaySource.RULE_FAST,
    ReplaySource.RULE_SAFE,
    ReplaySource.SL,
    ReplaySource.CURRENT,
    ReplaySource.FROZEN_POLICY,
)


@dataclass(frozen=True)
class PolicyDescriptor:
    """A resolvable policy identity, without loading its model weights."""

    policy_id: str
    source: ReplaySource
    version: int
    artifact: str | None = None
    created_update: int = 0

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        object.__setattr__(self, "policy_id", str(self.policy_id))
        object.__setattr__(self, "source", ReplaySource.parse(self.source))
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "created_update", int(self.created_update))
        if self.artifact is not None:
            object.__setattr__(self, "artifact", str(self.artifact))
        if self.source not in OPPONENT_SOURCES:
            raise ValueError(f"{self.source.label} cannot be used as an opponent")
        if self.version < 0 or self.created_update < 0:
            raise ValueError("policy version and update must be non-negative")
        if self.source in (ReplaySource.RULE_FAST, ReplaySource.RULE_SAFE):
            if self.artifact is not None:
                raise ValueError("built-in rule policies cannot have an artifact")

    def state_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "source": self.source.label,
            "version": self.version,
            "artifact": self.artifact,
            "created_update": self.created_update,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> PolicyDescriptor:
        return cls(
            policy_id=str(state["policy_id"]),
            source=ReplaySource.parse(str(state["source"])),
            version=int(state["version"]),
            artifact=None if state["artifact"] is None else str(state["artifact"]),
            created_update=int(state["created_update"]),
        )


@dataclass(frozen=True)
class OpponentMixtureConfig:
    """Always-on opponent mixture; there is intentionally no unlock stage."""

    rule_fast_weight: float = 0.15
    rule_safe_weight: float = 0.15
    frozen_sl_weight: float = 0.20
    current_weight: float = 0.30
    historical_weight: float = 0.20
    recent_history_fraction: float = 0.65
    recent_history_count: int = 4
    max_history: int = 16
    snapshot_interval: int = 50

    def __post_init__(self) -> None:
        weights = self.group_weights
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("opponent weights must be finite and non-negative")
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("opponent weights must sum to one")
        if min(
            self.rule_fast_weight,
            self.rule_safe_weight,
            self.frozen_sl_weight,
        ) <= 0:
            raise ValueError("rule_fast, rule_safe, and frozen SL must remain present")
        if self.current_weight <= 0:
            raise ValueError("the current policy must remain present")
        if not 0 <= self.recent_history_fraction <= 1:
            raise ValueError("recent_history_fraction must be in [0, 1]")
        if self.recent_history_count <= 0 or self.max_history <= 0:
            raise ValueError("history counts must be positive")
        if self.recent_history_count > self.max_history:
            raise ValueError("recent_history_count cannot exceed max_history")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")

    @property
    def group_weights(self) -> dict[ReplaySource, float]:
        return {
            ReplaySource.RULE_FAST: self.rule_fast_weight,
            ReplaySource.RULE_SAFE: self.rule_safe_weight,
            ReplaySource.SL: self.frozen_sl_weight,
            ReplaySource.CURRENT: self.current_weight,
            ReplaySource.FROZEN_POLICY: self.historical_weight,
        }


@dataclass(frozen=True)
class PolicyLineupBatch:
    policy_ids: np.ndarray
    sources: np.ndarray
    versions: np.ndarray
    focal_seats: np.ndarray | None

    def __post_init__(self) -> None:
        if self.policy_ids.ndim != 2 or self.policy_ids.shape[1] != 4:
            raise ValueError("policy_ids must have shape [games, 4]")
        if self.sources.shape != self.policy_ids.shape:
            raise ValueError("sources must match policy_ids")
        if self.versions.shape != self.policy_ids.shape:
            raise ValueError("versions must match policy_ids")
        if self.focal_seats is not None and self.focal_seats.shape != (
            self.policy_ids.shape[0],
        ):
            raise ValueError("focal_seats must have shape [games]")


class PolicyPool:
    """Checkpointable opponent pool with permanent rule and SL anchors."""

    def __init__(
        self,
        frozen_sl_artifact: str | PathLike[str],
        *,
        seed: int,
        current_version: int = 0,
        current_artifact: str | None = None,
        config: OpponentMixtureConfig | None = None,
    ) -> None:
        if not frozen_sl_artifact:
            raise ValueError("frozen_sl_artifact must not be empty")
        self.config = config or OpponentMixtureConfig()
        self.rule_fast = PolicyDescriptor(
            "rule_fast", ReplaySource.RULE_FAST, version=0
        )
        self.rule_safe = PolicyDescriptor(
            "rule_safe", ReplaySource.RULE_SAFE, version=0
        )
        self.frozen_sl = PolicyDescriptor(
            "frozen_sl",
            ReplaySource.SL,
            version=0,
            artifact=frozen_sl_artifact,
        )
        self.current = PolicyDescriptor(
            "current",
            ReplaySource.CURRENT,
            version=current_version,
            artifact=current_artifact,
        )
        self._history: list[PolicyDescriptor] = []
        self._rng = np.random.default_rng(seed)
        self._last_snapshot_update = 0
        self._lineup_cursor = 0
        self._lineups_sampled = 0

    @property
    def history(self) -> tuple[PolicyDescriptor, ...]:
        return tuple(self._history)

    @property
    def lineups_sampled(self) -> int:
        return self._lineups_sampled

    def update_current(
        self,
        version: int,
        *,
        artifact: str | None = None,
        update: int,
    ) -> None:
        if version < self.current.version:
            raise ValueError("current policy version cannot move backwards")
        self.current = PolicyDescriptor(
            "current",
            ReplaySource.CURRENT,
            version=version,
            artifact=artifact,
            created_update=update,
        )

    def snapshot_due(self, update: int) -> bool:
        if update < 0:
            raise ValueError("update must be non-negative")
        return update - self._last_snapshot_update >= self.config.snapshot_interval

    def add_snapshot(
        self,
        *,
        update: int,
        artifact: str,
        policy_id: str | None = None,
    ) -> PolicyDescriptor:
        """Freeze the current version after its artifact has been written."""
        if update < self._last_snapshot_update:
            raise ValueError("snapshot update cannot move backwards")
        if not artifact:
            raise ValueError("snapshot artifact must not be empty")
        descriptor = PolicyDescriptor(
            policy_id or f"frozen_v{self.current.version}_u{update}",
            ReplaySource.FROZEN_POLICY,
            version=self.current.version,
            artifact=artifact,
            created_update=update,
        )
        if any(item.policy_id == descriptor.policy_id for item in self._history):
            raise ValueError(f"duplicate policy_id {descriptor.policy_id!r}")
        self._history.append(descriptor)
        if len(self._history) > self.config.max_history:
            del self._history[: len(self._history) - self.config.max_history]
        self._last_snapshot_update = update
        return descriptor

    def maybe_add_snapshot(
        self,
        *,
        update: int,
        artifact: str,
        policy_id: str | None = None,
    ) -> PolicyDescriptor | None:
        if not self.snapshot_due(update):
            return None
        return self.add_snapshot(
            update=update,
            artifact=artifact,
            policy_id=policy_id,
        )

    def distribution(self) -> dict[PolicyDescriptor, float]:
        """Return exact member probabilities after history stratification."""
        config = self.config
        probabilities: dict[PolicyDescriptor, float] = {
            self.rule_fast: config.rule_fast_weight,
            self.rule_safe: config.rule_safe_weight,
            self.frozen_sl: config.frozen_sl_weight,
            self.current: config.current_weight,
        }
        if not self._history:
            probabilities[self.current] += config.historical_weight
            return probabilities

        recent_count = min(config.recent_history_count, len(self._history))
        old = self._history[:-recent_count]
        recent = self._history[-recent_count:]
        if not old:
            recent_mass = config.historical_weight
            old_mass = 0.0
        else:
            recent_mass = (
                config.historical_weight * config.recent_history_fraction
            )
            old_mass = config.historical_weight - recent_mass
        for descriptor in recent:
            probabilities[descriptor] = recent_mass / len(recent)
        for descriptor in old:
            probabilities[descriptor] = old_mass / len(old)
        return probabilities

    def sample_descriptors(
        self, shape: int | tuple[int, ...]
    ) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        if not shape or any(dimension < 0 for dimension in shape):
            raise ValueError("sample shape must contain non-negative dimensions")
        distribution = self.distribution()
        descriptors = tuple(distribution)
        probabilities = np.fromiter(distribution.values(), dtype=np.float64)
        probabilities /= probabilities.sum()
        indices = self._rng.choice(len(descriptors), size=shape, p=probabilities)
        members = np.asarray(descriptors, dtype=object)
        return members[indices]

    def sample_lineups(
        self, games: int, *, ensure_current: bool = True
    ) -> PolicyLineupBatch:
        """Sample four-seat lineups and rotate a focal current-policy seat."""
        if games <= 0:
            raise ValueError("games must be positive")
        descriptors = self.sample_descriptors((games, 4))
        focal_seats: np.ndarray | None = None
        if ensure_current:
            focal_seats = (
                np.arange(games, dtype=np.uint64) + self._lineup_cursor
            ).astype(np.uint8) % 4
            descriptors[np.arange(games), focal_seats] = self.current
            self._lineup_cursor = (self._lineup_cursor + games) % 4
        self._lineups_sampled += games
        policy_ids = np.empty((games, 4), dtype=object)
        sources = np.empty((games, 4), dtype=np.uint8)
        versions = np.empty((games, 4), dtype=np.uint64)
        for row in range(games):
            for seat in range(4):
                descriptor = descriptors[row, seat]
                policy_ids[row, seat] = descriptor.policy_id
                sources[row, seat] = int(descriptor.source)
                versions[row, seat] = descriptor.version
        return PolicyLineupBatch(policy_ids, sources, versions, focal_seats)

    def state_dict(self) -> dict[str, object]:
        return {
            "state_version": POLICY_POOL_STATE_VERSION,
            "config": asdict(self.config),
            "frozen_sl": self.frozen_sl.state_dict(),
            "current": self.current.state_dict(),
            "history": [item.state_dict() for item in self._history],
            "last_snapshot_update": self._last_snapshot_update,
            "lineup_cursor": self._lineup_cursor,
            "lineups_sampled": self._lineups_sampled,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> PolicyPool:
        version = int(state["state_version"])
        if version != POLICY_POOL_STATE_VERSION:
            raise ValueError(
                f"unsupported policy pool state version {version}; "
                f"expected {POLICY_POOL_STATE_VERSION}"
            )
        frozen_sl = PolicyDescriptor.from_state_dict(state["frozen_sl"])  # type: ignore[arg-type]
        current = PolicyDescriptor.from_state_dict(state["current"])  # type: ignore[arg-type]
        config = OpponentMixtureConfig(**state["config"])  # type: ignore[arg-type]
        pool = cls(
            frozen_sl.artifact or "",
            seed=0,
            current_version=current.version,
            current_artifact=current.artifact,
            config=config,
        )
        pool.frozen_sl = frozen_sl
        pool.current = current
        pool._history = [
            PolicyDescriptor.from_state_dict(item)
            for item in state["history"]  # type: ignore[union-attr]
        ]
        if len(pool._history) > config.max_history:
            raise ValueError("checkpoint history exceeds configured maximum")
        if pool.frozen_sl.source != ReplaySource.SL:
            raise ValueError("checkpoint frozen_sl descriptor has the wrong source")
        if pool.current.source != ReplaySource.CURRENT:
            raise ValueError("checkpoint current descriptor has the wrong source")
        if any(
            item.source != ReplaySource.FROZEN_POLICY for item in pool._history
        ):
            raise ValueError("checkpoint history contains a non-frozen policy")
        policy_ids = [item.policy_id for item in pool._history]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("checkpoint history contains duplicate policy ids")
        pool._last_snapshot_update = int(state["last_snapshot_update"])
        if pool._last_snapshot_update < 0 or any(
            item.created_update > pool._last_snapshot_update
            for item in pool._history
        ):
            raise ValueError("checkpoint snapshot update metadata is inconsistent")
        pool._lineup_cursor = int(state["lineup_cursor"])
        pool._lineups_sampled = int(state["lineups_sampled"])
        pool._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return pool


@dataclass(frozen=True)
class ExplorationRule:
    temperature: float
    top_k: int | None
    random_action_probability: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("exploration temperature must be positive and finite")
        if self.top_k is not None and not 1 <= self.top_k <= ACTION_SPACE_SIZE:
            raise ValueError("top_k must be in [1, 115] or None")
        if not 0 <= self.random_action_probability < 1:
            raise ValueError("random_action_probability must be in [0, 1)")


def _default_exploration_rules() -> tuple[ExplorationRule, ...]:
    return (
        ExplorationRule(temperature=1.15, top_k=6, random_action_probability=0.020),
        ExplorationRule(temperature=1.10, top_k=6, random_action_probability=0.020),
        ExplorationRule(temperature=1.05, top_k=5, random_action_probability=0.015),
        ExplorationRule(temperature=1.00, top_k=3, random_action_probability=0.010),
        ExplorationRule(temperature=1.10, top_k=8, random_action_probability=0.020),
        ExplorationRule(temperature=1.00, top_k=6, random_action_probability=0.015),
        ExplorationRule(temperature=0.90, top_k=4, random_action_probability=0.010),
        ExplorationRule(temperature=0.50, top_k=2, random_action_probability=0.001),
        ExplorationRule(temperature=0.75, top_k=3, random_action_probability=0.005),
    )


@dataclass(frozen=True)
class ExplorationConfig:
    rules: tuple[ExplorationRule, ...] = field(
        default_factory=_default_exploration_rules
    )

    def __post_init__(self) -> None:
        if len(self.rules) != CATEGORY_COUNT:
            raise ValueError("exploration rules must cover all decision categories")
        if any(not isinstance(rule, ExplorationRule) for rule in self.rules):
            raise TypeError("rules must contain ExplorationRule values")
        if self.rules[DecisionCategory.HU_RESPONSE].random_action_probability <= 0:
            raise ValueError("hu-response exploration must be non-zero")

    def state_dict(self) -> dict[str, object]:
        return {"rules": [asdict(rule) for rule in self.rules]}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> ExplorationConfig:
        return cls(
            rules=tuple(
                ExplorationRule(**item)
                for item in state["rules"]  # type: ignore[union-attr,arg-type]
            )
        )


@dataclass(frozen=True)
class BehaviorActionBatch:
    actions: np.ndarray
    action_probabilities: np.ndarray
    temperatures: np.ndarray
    distributions: np.ndarray


def decision_categories(meta: np.ndarray) -> np.ndarray:
    """Map engine metadata to the canonical nine training categories."""
    meta = np.asarray(meta)
    if meta.ndim != 2 or meta.shape[1] < 11:
        raise ValueError("meta must have shape [batch, >=11]")
    categories = np.full(len(meta), -1, dtype=np.int8)
    phase = meta[:, 0]
    exchange = phase == 0
    categories[exchange & (meta[:, 10] == 0)] = DecisionCategory.EXCHANGE_FIRST
    categories[exchange & (meta[:, 10] == 1)] = DecisionCategory.EXCHANGE_SECOND
    categories[exchange & (meta[:, 10] == 2)] = DecisionCategory.EXCHANGE_THIRD
    categories[phase == 1] = DecisionCategory.CHOOSE_MISSING
    turn = phase == 2
    categories[turn & (meta[:, 4] >= 40)] = DecisionCategory.TURN_EARLY
    categories[turn & (meta[:, 4] >= 20) & (meta[:, 4] < 40)] = (
        DecisionCategory.TURN_MIDDLE
    )
    categories[turn & (meta[:, 4] < 20)] = DecisionCategory.TURN_LATE
    categories[phase == 3] = DecisionCategory.HU_RESPONSE
    categories[phase == 4] = DecisionCategory.MELD_RESPONSE
    if np.any(categories < 0):
        bad_rows = np.flatnonzero(categories < 0)[:8].tolist()
        raise ValueError(f"metadata contains unclassifiable decisions at rows {bad_rows}")
    return categories.astype(np.uint8)


class BehaviorSampler:
    """Samples legal actions and exposes their exact behavior probabilities."""

    def __init__(
        self, *, seed: int, config: ExplorationConfig | None = None
    ) -> None:
        self.config = config or ExplorationConfig()
        self._rng = np.random.default_rng(seed)
        self.actions_sampled = 0

    def probabilities(
        self,
        logits: np.ndarray,
        legal: np.ndarray,
        categories: np.ndarray,
    ) -> np.ndarray:
        logits = np.asarray(logits)
        legal = np.asarray(legal)
        categories = np.asarray(categories)
        if logits.ndim != 2 or logits.shape[1] != ACTION_SPACE_SIZE:
            raise ValueError("logits must have shape [batch, 115]")
        if legal.shape != logits.shape or legal.dtype != np.bool_:
            raise ValueError("legal must be a bool array matching logits")
        if categories.shape != (len(logits),):
            raise ValueError("categories must have shape [batch]")
        if not np.issubdtype(categories.dtype, np.integer):
            raise ValueError("categories must contain integer codes")
        categories = categories.astype(np.int64, copy=False)
        if np.any(categories < 0) or np.any(categories >= CATEGORY_COUNT):
            raise ValueError("categories contain an invalid code")
        if np.any(legal.sum(axis=1) == 0):
            raise ValueError("every sampled decision must have a legal action")
        if np.any(~np.isfinite(logits[legal])):
            raise ValueError("legal-action logits must be finite")

        output = np.zeros(logits.shape, dtype=np.float64)
        for category, rule in enumerate(self.config.rules):
            rows = np.flatnonzero(categories == category)
            if len(rows) == 0:
                continue
            group_legal = legal[rows]
            scores = logits[rows].astype(np.float64) / rule.temperature
            scores[~group_legal] = -np.inf
            support = group_legal.copy()
            if rule.top_k is not None:
                legal_counts = group_legal.sum(axis=1)
                limited = legal_counts > rule.top_k
                if np.any(limited):
                    limited_scores = scores[limited]
                    top = np.argpartition(
                        limited_scores,
                        -rule.top_k,
                        axis=1,
                    )[:, -rule.top_k :]
                    limited_support = np.zeros_like(limited_scores, dtype=np.bool_)
                    np.put_along_axis(limited_support, top, True, axis=1)
                    support[limited] = limited_support & group_legal[limited]
            policy_scores = np.where(support, scores, -np.inf)
            maximum = policy_scores.max(axis=1, keepdims=True)
            policy = np.exp(policy_scores - maximum)
            policy *= support
            policy /= policy.sum(axis=1, keepdims=True)
            epsilon = rule.random_action_probability
            random_policy = group_legal / group_legal.sum(axis=1, keepdims=True)
            output[rows] = (1.0 - epsilon) * policy + epsilon * random_policy
        output /= output.sum(axis=1, keepdims=True)
        return output

    def sample(
        self,
        logits: np.ndarray,
        legal: np.ndarray,
        categories: np.ndarray,
    ) -> BehaviorActionBatch:
        categories = np.asarray(categories)
        probabilities = self.probabilities(logits, legal, categories)
        draws = self._rng.random(len(probabilities))
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        actions = (cumulative < draws[:, None]).sum(axis=1).astype(np.uint8)
        action_probabilities = probabilities[
            np.arange(len(probabilities)), actions.astype(np.int64)
        ].astype(np.float32)
        temperatures = np.asarray(
            [self.config.rules[int(category)].temperature for category in categories],
            dtype=np.float32,
        )
        self.actions_sampled += len(actions)
        return BehaviorActionBatch(
            actions=actions,
            action_probabilities=action_probabilities,
            temperatures=temperatures,
            distributions=probabilities.astype(np.float32),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "state_version": EXPLORATION_STATE_VERSION,
            "config": self.config.state_dict(),
            "actions_sampled": self.actions_sampled,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> BehaviorSampler:
        version = int(state["state_version"])
        if version != EXPLORATION_STATE_VERSION:
            raise ValueError(
                f"unsupported exploration state version {version}; "
                f"expected {EXPLORATION_STATE_VERSION}"
            )
        sampler = cls(
            seed=0,
            config=ExplorationConfig.from_state_dict(state["config"]),  # type: ignore[arg-type]
        )
        sampler.actions_sampled = int(state["actions_sampled"])
        sampler._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return sampler


def _default_source_floors() -> tuple[tuple[ReplaySource, float], ...]:
    return (
        (ReplaySource.SL, 0.12),
        (ReplaySource.RULE_FAST, 0.08),
        (ReplaySource.RULE_SAFE, 0.08),
        (ReplaySource.CURRENT, 0.10),
        (ReplaySource.FROZEN_POLICY, 0.05),
    )


def _default_source_weights() -> tuple[tuple[ReplaySource, float], ...]:
    return (
        (ReplaySource.SL, 0.18),
        (ReplaySource.RULE_FAST, 0.14),
        (ReplaySource.RULE_SAFE, 0.14),
        (ReplaySource.CURRENT, 0.24),
        (ReplaySource.FROZEN_POLICY, 0.20),
        (ReplaySource.MC_TEACHER, 0.10),
    )


@dataclass(frozen=True)
class ReplayBalanceConfig:
    source_minimum_fractions: tuple[tuple[ReplaySource, float], ...] = field(
        default_factory=_default_source_floors
    )
    source_target_weights: tuple[tuple[ReplaySource, float], ...] = field(
        default_factory=_default_source_weights
    )
    category_minimum_fractions: tuple[float, ...] = (0.01,) * CATEGORY_COUNT
    category_target_weights: tuple[float, ...] = DEFAULT_CATEGORY_WEIGHTS
    required_sources: tuple[ReplaySource, ...] = (
        ReplaySource.SL,
        ReplaySource.RULE_FAST,
        ReplaySource.RULE_SAFE,
        ReplaySource.CURRENT,
    )
    require_all_categories: bool = True
    duplicate_downsample_power: float = 0.5
    policy_version_downsample_power: float = 0.25

    def __post_init__(self) -> None:
        normalized_floors = tuple(
            (ReplaySource.parse(source), float(value))
            for source, value in self.source_minimum_fractions
        )
        normalized_weights = tuple(
            (ReplaySource.parse(source), float(value))
            for source, value in self.source_target_weights
        )
        object.__setattr__(self, "source_minimum_fractions", normalized_floors)
        object.__setattr__(self, "source_target_weights", normalized_weights)
        source_floors = _source_mapping(normalized_floors)
        source_weights = _source_mapping(normalized_weights)
        if any(not np.isfinite(value) or value < 0 for value in source_floors.values()):
            raise ValueError("source minimum fractions must be finite and non-negative")
        if sum(source_floors.values()) > 1 + 1e-12:
            raise ValueError("source minimum fractions cannot sum above one")
        if set(source_weights) != set(ReplaySource):
            raise ValueError("source target weights must cover every replay source")
        if any(not np.isfinite(value) or value <= 0 for value in source_weights.values()):
            raise ValueError("source target weights must be positive and finite")
        if not np.isclose(sum(source_weights.values()), 1.0):
            raise ValueError("source target weights must sum to one")
        if len(self.category_minimum_fractions) != CATEGORY_COUNT:
            raise ValueError("category minimum fractions must cover every category")
        if any(
            not np.isfinite(value) or value < 0
            for value in self.category_minimum_fractions
        ):
            raise ValueError("category minimum fractions must be non-negative")
        if sum(self.category_minimum_fractions) > 1 + 1e-12:
            raise ValueError("category minimum fractions cannot sum above one")
        if len(self.category_target_weights) != CATEGORY_COUNT or any(
            not np.isfinite(value) or value <= 0
            for value in self.category_target_weights
        ):
            raise ValueError("category target weights must be positive and complete")
        if not np.isclose(sum(self.category_target_weights), 1.0):
            raise ValueError("category target weights must sum to one")
        required = tuple(ReplaySource.parse(value) for value in self.required_sources)
        if len(set(required)) != len(required):
            raise ValueError("required_sources contains duplicates")
        object.__setattr__(self, "required_sources", required)
        if self.duplicate_downsample_power < 0:
            raise ValueError("duplicate_downsample_power must be non-negative")
        if self.policy_version_downsample_power < 0:
            raise ValueError("policy_version_downsample_power must be non-negative")

    @property
    def source_floors(self) -> dict[ReplaySource, float]:
        return _source_mapping(self.source_minimum_fractions)

    @property
    def source_weights(self) -> dict[ReplaySource, float]:
        return _source_mapping(self.source_target_weights)

    def state_dict(self) -> dict[str, object]:
        return {
            "source_minimum_fractions": [
                [source.label, fraction]
                for source, fraction in self.source_minimum_fractions
            ],
            "source_target_weights": [
                [source.label, weight]
                for source, weight in self.source_target_weights
            ],
            "category_minimum_fractions": list(self.category_minimum_fractions),
            "category_target_weights": list(self.category_target_weights),
            "required_sources": [source.label for source in self.required_sources],
            "require_all_categories": self.require_all_categories,
            "duplicate_downsample_power": self.duplicate_downsample_power,
            "policy_version_downsample_power": self.policy_version_downsample_power,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> ReplayBalanceConfig:
        return cls(
            source_minimum_fractions=tuple(
                (ReplaySource.parse(source), float(value))
                for source, value in state["source_minimum_fractions"]  # type: ignore[union-attr]
            ),
            source_target_weights=tuple(
                (ReplaySource.parse(source), float(value))
                for source, value in state["source_target_weights"]  # type: ignore[union-attr]
            ),
            category_minimum_fractions=tuple(
                float(value)
                for value in state["category_minimum_fractions"]  # type: ignore[union-attr]
            ),
            category_target_weights=tuple(
                float(value)
                for value in state["category_target_weights"]  # type: ignore[union-attr]
            ),
            required_sources=tuple(
                ReplaySource.parse(value)
                for value in state["required_sources"]  # type: ignore[union-attr]
            ),
            require_all_categories=bool(state["require_all_categories"]),
            duplicate_downsample_power=float(state["duplicate_downsample_power"]),
            policy_version_downsample_power=float(
                state["policy_version_downsample_power"]
            ),
        )


def _source_mapping(
    pairs: Sequence[tuple[ReplaySource, float]],
) -> dict[ReplaySource, float]:
    output: dict[ReplaySource, float] = {}
    for source, value in pairs:
        parsed = ReplaySource.parse(source)
        if parsed in output:
            raise ValueError(f"duplicate source {parsed.label}")
        output[parsed] = float(value)
    return output


@dataclass(frozen=True)
class ReplayBatchSelection:
    indices: np.ndarray
    source_counts: np.ndarray
    category_counts: np.ndarray


@dataclass(frozen=True)
class ReplaySamplingIndex:
    """Pre-indexed immutable replay metadata for repeated minibatch draws."""

    sources: np.ndarray
    categories: np.ndarray
    cells: Mapping[tuple[int, int], np.ndarray]
    row_weights: np.ndarray | None

    def __post_init__(self) -> None:
        if self.sources.ndim != 1 or self.categories.shape != self.sources.shape:
            raise ValueError("indexed replay metadata must be one-dimensional and aligned")
        if self.row_weights is not None and self.row_weights.shape != self.sources.shape:
            raise ValueError("row_weights must align with replay metadata")

    def __len__(self) -> int:
        return len(self.sources)


class ReplayCoverageError(RuntimeError):
    """Raised when a requested replay floor cannot be represented."""


class BalancedReplaySampler:
    """Samples with replacement while enforcing source/category batch floors."""

    def __init__(
        self, *, seed: int, config: ReplayBalanceConfig | None = None
    ) -> None:
        self.config = config or ReplayBalanceConfig()
        self._rng = np.random.default_rng(seed)
        self.batches_sampled = 0
        self.rows_sampled = 0

    def sample(
        self,
        sources: Sequence[ReplaySource | str | int] | np.ndarray,
        categories: Sequence[int] | np.ndarray,
        batch_size: int,
        *,
        duplicate_keys: Sequence[Hashable] | np.ndarray | None = None,
        policy_versions: Sequence[Hashable] | np.ndarray | None = None,
    ) -> ReplayBatchSelection:
        index = self.prepare(
            sources,
            categories,
            duplicate_keys=duplicate_keys,
            policy_versions=policy_versions,
        )
        return self.sample_index(index, batch_size)

    def prepare(
        self,
        sources: Sequence[ReplaySource | str | int] | np.ndarray,
        categories: Sequence[int] | np.ndarray,
        *,
        duplicate_keys: Sequence[Hashable] | np.ndarray | None = None,
        policy_versions: Sequence[Hashable] | np.ndarray | None = None,
    ) -> ReplaySamplingIndex:
        """Build an index once per replay revision, then reuse it for updates."""
        source_codes = _source_codes(sources).copy()
        category_codes = _category_codes(categories).copy()
        if category_codes.shape != source_codes.shape:
            raise ValueError("sources and categories must be one-dimensional and aligned")
        if len(source_codes) == 0:
            raise ReplayCoverageError("cannot sample an empty replay")
        duplicate_values = _optional_hashable_array(
            duplicate_keys, len(source_codes), "duplicate_keys"
        )
        version_values = _optional_hashable_array(
            policy_versions, len(source_codes), "policy_versions"
        )
        row_weights: np.ndarray | None = None
        if duplicate_values is not None or version_values is not None:
            row_weights = np.ones(len(source_codes), dtype=np.float64)
            if duplicate_values is not None:
                row_weights *= _inverse_frequency_weights(
                    duplicate_values,
                    self.config.duplicate_downsample_power,
                )
            if version_values is not None:
                row_weights *= _inverse_frequency_weights(
                    version_values,
                    self.config.policy_version_downsample_power,
                )
            row_weights.setflags(write=False)

        cell_codes = (
            source_codes.astype(np.int64) * CATEGORY_COUNT + category_codes
        )
        order = np.argsort(cell_codes, kind="stable")
        sorted_codes = cell_codes[order]
        unique, starts, counts = np.unique(
            sorted_codes, return_index=True, return_counts=True
        )
        cells = {
            (int(code // CATEGORY_COUNT), int(code % CATEGORY_COUNT)): order[
                start : start + count
            ]
            for code, start, count in zip(unique, starts, counts)
        }
        source_codes.setflags(write=False)
        category_codes.setflags(write=False)
        for indices in cells.values():
            indices.setflags(write=False)
        return ReplaySamplingIndex(
            source_codes,
            category_codes,
            cells,
            row_weights,
        )

    def sample_index(
        self,
        index: ReplaySamplingIndex,
        batch_size: int,
    ) -> ReplayBatchSelection:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(index) == 0:
            raise ReplayCoverageError("cannot sample an empty replay")
        source_quotas = self._source_quotas(index.sources, batch_size)
        category_quotas = self._category_quotas(index.categories, batch_size)
        allocations = _minimum_cell_allocations(
            index.cells, source_quotas, category_quotas, batch_size
        )
        allocated = sum(allocations.values())
        remaining = batch_size - allocated
        if remaining:
            keys = tuple(index.cells)
            source_weights = self.config.source_weights
            category_weights = np.asarray(
                self.config.category_target_weights, dtype=np.float64
            )
            weights = np.asarray(
                [
                    source_weights[ReplaySource(source)] * category_weights[category]
                    for source, category in keys
                ],
                dtype=np.float64,
            )
            weights /= weights.sum()
            additions = self._rng.multinomial(remaining, weights)
            for key, count in zip(keys, additions):
                allocations[key] = allocations.get(key, 0) + int(count)

        selected_parts: list[np.ndarray] = []
        for key, count in allocations.items():
            if count == 0:
                continue
            candidates = index.cells[key]
            probabilities: np.ndarray | None = None
            if index.row_weights is not None:
                weights = index.row_weights[candidates]
                probabilities = weights / weights.sum()
            selected_parts.append(
                self._rng.choice(
                    candidates,
                    size=count,
                    replace=True,
                    p=probabilities,
                )
            )
        indices = np.concatenate(selected_parts).astype(np.int64, copy=False)
        self._rng.shuffle(indices)
        selected_sources = index.sources[indices]
        selected_categories = index.categories[indices]
        source_counts = np.bincount(
            selected_sources, minlength=len(ReplaySource)
        ).astype(np.int64)
        category_counts = np.bincount(
            selected_categories, minlength=CATEGORY_COUNT
        ).astype(np.int64)
        self.batches_sampled += 1
        self.rows_sampled += batch_size
        return ReplayBatchSelection(indices, source_counts, category_counts)

    def _source_quotas(
        self, sources: np.ndarray, batch_size: int
    ) -> dict[int, int]:
        available = set(int(value) for value in np.unique(sources))
        missing = [
            source.label
            for source in self.config.required_sources
            if int(source) not in available
        ]
        if missing:
            raise ReplayCoverageError(
                "replay is missing required sources: " + ", ".join(missing)
            )
        quotas = {
            int(source): int(np.ceil(fraction * batch_size))
            for source, fraction in self.config.source_floors.items()
            if int(source) in available and fraction > 0
        }
        if sum(quotas.values()) > batch_size:
            raise ReplayCoverageError("rounded source floors exceed batch size")
        return quotas

    def _category_quotas(
        self, categories: np.ndarray, batch_size: int
    ) -> dict[int, int]:
        available = set(int(value) for value in np.unique(categories))
        if self.config.require_all_categories:
            missing = [
                CATEGORY_NAMES[category]
                for category in range(CATEGORY_COUNT)
                if category not in available
            ]
            if missing:
                raise ReplayCoverageError(
                    "replay is missing required categories: " + ", ".join(missing)
                )
        quotas = {
            category: int(np.ceil(fraction * batch_size))
            for category, fraction in enumerate(
                self.config.category_minimum_fractions
            )
            if category in available and fraction > 0
        }
        if sum(quotas.values()) > batch_size:
            raise ReplayCoverageError("rounded category floors exceed batch size")
        return quotas

    def state_dict(self) -> dict[str, object]:
        return {
            "state_version": REPLAY_SAMPLER_STATE_VERSION,
            "config": self.config.state_dict(),
            "batches_sampled": self.batches_sampled,
            "rows_sampled": self.rows_sampled,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> BalancedReplaySampler:
        version = int(state["state_version"])
        if version != REPLAY_SAMPLER_STATE_VERSION:
            raise ValueError(
                f"unsupported replay sampler state version {version}; "
                f"expected {REPLAY_SAMPLER_STATE_VERSION}"
            )
        sampler = cls(
            seed=0,
            config=ReplayBalanceConfig.from_state_dict(state["config"]),  # type: ignore[arg-type]
        )
        sampler.batches_sampled = int(state["batches_sampled"])
        sampler.rows_sampled = int(state["rows_sampled"])
        sampler._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return sampler


def replay_composition(
    sources: Sequence[ReplaySource | str | int] | np.ndarray,
    categories: Sequence[int] | np.ndarray,
) -> dict[str, dict[str, int]]:
    source_codes = _source_codes(sources)
    categories = _category_codes(categories)
    if source_codes.shape != categories.shape:
        raise ValueError("sources and categories must be one-dimensional and aligned")
    if np.any(categories < 0) or np.any(categories >= CATEGORY_COUNT):
        raise ValueError("categories contain an invalid code")
    source_counts = np.bincount(source_codes, minlength=len(ReplaySource))
    category_counts = np.bincount(categories, minlength=CATEGORY_COUNT)
    return {
        "sources": {
            source.label: int(source_counts[int(source)]) for source in ReplaySource
        },
        "categories": {
            name: int(category_counts[index])
            for index, name in enumerate(CATEGORY_NAMES)
        },
    }


def _source_codes(
    values: Sequence[ReplaySource | str | int] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("sources must be one-dimensional")
    if np.issubdtype(array.dtype, np.integer):
        numeric = array.astype(np.int64, copy=False)
        if np.any(numeric < 0) or np.any(numeric >= len(ReplaySource)):
            raise ValueError("sources contain an invalid code")
        return numeric.astype(np.uint8, copy=False)
    return np.fromiter(
        (int(ReplaySource.parse(value)) for value in array),
        dtype=np.uint8,
        count=len(array),
    )


def _category_codes(values: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("categories must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("categories must contain integer codes")
    output = array.astype(np.int64, copy=False)
    if np.any(output < 0) or np.any(output >= CATEGORY_COUNT):
        raise ValueError("categories contain an invalid code")
    return output


def _optional_hashable_array(
    values: Sequence[Hashable] | np.ndarray | None,
    length: int,
    name: str,
) -> list[Hashable] | None:
    if values is None:
        return None
    if len(values) != length:
        raise ValueError(f"{name} must align with replay rows")
    output = list(values)
    for value in output:
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(f"{name} values must be hashable") from error
    return output


def _inverse_frequency_weights(
    values: Sequence[Hashable], power: float
) -> np.ndarray:
    if power == 0:
        return np.ones(len(values), dtype=np.float64)
    array = np.asarray(values)
    if array.ndim == 1 and array.dtype.kind in "biufSU":
        _, inverse, counts = np.unique(
            array, return_inverse=True, return_counts=True
        )
        return counts[inverse].astype(np.float64) ** -power
    counts = Counter(values)
    return np.asarray([counts[value] ** -power for value in values], dtype=np.float64)


def _minimum_cell_allocations(
    cells: Mapping[tuple[int, int], np.ndarray],
    source_quotas: Mapping[int, int],
    category_quotas: Mapping[int, int],
    batch_size: int,
) -> dict[tuple[int, int], int]:
    """Use max-flow to overlap source and category quota draws optimally."""
    sources = tuple(sorted(source_quotas))
    categories = tuple(sorted(category_quotas))
    source_index = {source: index + 1 for index, source in enumerate(sources)}
    category_offset = 1 + len(sources)
    category_index = {
        category: category_offset + index
        for index, category in enumerate(categories)
    }
    sink = category_offset + len(categories)
    node_count = sink + 1
    capacity = np.zeros((node_count, node_count), dtype=np.int64)
    for source, quota in source_quotas.items():
        capacity[0, source_index[source]] = quota
    for category, quota in category_quotas.items():
        capacity[category_index[category], sink] = quota
    for source in sources:
        for category in categories:
            if (source, category) in cells:
                capacity[source_index[source], category_index[category]] = batch_size
    original = capacity.copy()
    adjacency = [set() for _ in range(node_count)]
    for start, end in zip(*np.nonzero(capacity)):
        adjacency[int(start)].add(int(end))
        adjacency[int(end)].add(int(start))

    while True:
        parent = np.full(node_count, -1, dtype=np.int64)
        parent[0] = 0
        queue: deque[int] = deque([0])
        while queue and parent[sink] < 0:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if parent[neighbor] < 0 and capacity[node, neighbor] > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if parent[sink] < 0:
            break
        amount = batch_size
        node = sink
        while node != 0:
            previous = int(parent[node])
            amount = min(amount, int(capacity[previous, node]))
            node = previous
        node = sink
        while node != 0:
            previous = int(parent[node])
            capacity[previous, node] -= amount
            capacity[node, previous] += amount
            adjacency[node].add(previous)
            adjacency[previous].add(node)
            node = previous

    allocations: dict[tuple[int, int], int] = {}
    source_matched = Counter()
    category_matched = Counter()
    for source in sources:
        for category in categories:
            start = source_index[source]
            end = category_index[category]
            flow = int(original[start, end] - capacity[start, end])
            if flow > 0:
                allocations[(source, category)] = flow
                source_matched[source] += flow
                category_matched[category] += flow

    for source, quota in source_quotas.items():
        remaining = quota - source_matched[source]
        if remaining:
            candidates = sorted(key for key in cells if key[0] == source)
            if not candidates:
                raise ReplayCoverageError(f"source {ReplaySource(source).label} has no rows")
            for offset in range(remaining):
                key = candidates[offset % len(candidates)]
                allocations[key] = allocations.get(key, 0) + 1
    for category, quota in category_quotas.items():
        remaining = quota - category_matched[category]
        if remaining:
            candidates = sorted(key for key in cells if key[1] == category)
            if not candidates:
                raise ReplayCoverageError(
                    f"category {CATEGORY_NAMES[category]} has no rows"
                )
            for offset in range(remaining):
                key = candidates[offset % len(candidates)]
                allocations[key] = allocations.get(key, 0) + 1

    required = sum(allocations.values())
    if required > batch_size:
        raise ReplayCoverageError(
            "source/category floors are jointly infeasible for this replay and batch size"
        )
    return allocations
