from __future__ import annotations

import numpy as np
import pytest

from training.policy_pool import (
    CATEGORY_COUNT,
    CATEGORY_NAMES,
    DecisionCategory,
    ReplaySource,
    decision_categories,
)


def test_decision_taxonomy_is_stable_and_complete() -> None:
    assert CATEGORY_COUNT == 9
    assert CATEGORY_NAMES == (
        "exchange_first",
        "exchange_second",
        "exchange_third",
        "choose_missing",
        "turn_early",
        "turn_middle",
        "turn_late",
        "hu_response",
        "meld_response",
    )
    assert [int(value) for value in ReplaySource] == [0, 1, 2, 3, 4]
    assert ReplaySource.SELF_PLAY.label == "self_play"


def test_engine_metadata_maps_to_all_nine_categories() -> None:
    meta = np.zeros((9, 34), dtype=np.int32)
    meta[:3, 0] = 0
    meta[:3, 10] = np.arange(3)
    meta[3, 0] = 1
    meta[4:7, 0] = 2
    meta[4:7, 4] = (40, 20, 19)
    meta[7, 0] = 3
    meta[8, 0] = 4
    np.testing.assert_array_equal(
        decision_categories(meta), np.arange(9, dtype=np.uint8)
    )


def test_unclassifiable_metadata_is_rejected() -> None:
    meta = np.zeros((1, 34), dtype=np.int32)
    meta[0, 0] = 9
    with pytest.raises(ValueError, match="unclassifiable"):
        decision_categories(meta)
