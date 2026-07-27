"""Decision taxonomy and controller identities used by policy iteration."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


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


class ReplaySource(IntEnum):
    """Stable per-step controller codes used by in-memory trajectories."""

    SL = 0
    RULE_FAST = 1
    RULE_SAFE = 2
    CURRENT = 3
    SELF_PLAY = 4

    @property
    def label(self) -> str:
        return self.name.lower()


def decision_categories(meta: np.ndarray) -> np.ndarray:
    """Map engine decision metadata to the canonical nine categories."""

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
        rows = np.flatnonzero(categories < 0)[:8].tolist()
        raise ValueError(f"metadata has unclassifiable decisions at rows {rows}")
    return categories.astype(np.uint8)


__all__ = [
    "CATEGORY_COUNT",
    "CATEGORY_NAMES",
    "DecisionCategory",
    "ReplaySource",
    "decision_categories",
]
