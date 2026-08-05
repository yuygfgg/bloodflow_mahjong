"""Runtime contract between the Python trainers and the engine binding."""

from __future__ import annotations

import bloodflow_mahjong as bm


TRAINING_INPUT_SCHEMA = "viewer-public-win-tiles-v1"
MODEL_HISTORY_CAPACITY = 192

_EXACT_CONSTANTS = {
    "ENGINE_RULES_VERSION": 7,
    "ACTION_SPACE_SIZE": 115,
    "LEGAL_ACTION_MASK_WORDS": 2,
    "STEP_RECORD_WIDTH": 12,
    "EVENT_RECORD_WIDTH": 8,
    "TILE_OBSERVATION_WIDTH": 270,
    "TILE_OBSERVATION_PLANES": 10,
    "TILE_KIND_COUNT": 27,
    "MELD_OBSERVATION_WIDTH": 48,
    "MELD_SLOTS": 4,
    "MELD_FIELDS": 3,
    "RIVER_OBSERVATION_WIDTH": 216,
    "RIVER_TILE_CAPACITY": 108,
    "RIVER_FIELDS": 2,
    "META_OBSERVATION_WIDTH": 34,
    "PLAYER_COUNT": 4,
}


def validate_engine_contract() -> None:
    """Reject a binding whose fixed buffers do not match the trainers."""
    mismatches = []
    for name, expected in _EXACT_CONSTANTS.items():
        actual = getattr(bm, name, None)
        if actual != expected:
            mismatches.append(f"{name}={actual!r}, expected {expected}")
    history_capacity = getattr(bm, "EVENT_HISTORY_CAPACITY", None)
    if not isinstance(history_capacity, int) or history_capacity < MODEL_HISTORY_CAPACITY:
        mismatches.append(
            "EVENT_HISTORY_CAPACITY="
            f"{history_capacity!r}, expected at least {MODEL_HISTORY_CAPACITY}"
        )
    if mismatches:
        raise RuntimeError("engine binding contract mismatch: " + "; ".join(mismatches))
