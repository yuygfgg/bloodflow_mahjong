"""Conversion helpers for caller-owned engine buffers."""

from __future__ import annotations

import numpy as np

ACTION_SPACE_SIZE = 115
LEGAL_ACTION_MASK_WORDS = 2


def unpack_action_masks(mask_words: np.ndarray) -> np.ndarray:
    """Expand packed engine masks to a boolean ``[batch, 115]`` array."""
    words = np.asarray(mask_words)
    if words.dtype != np.uint64:
        raise TypeError(f"mask_words must have dtype uint64, got {words.dtype}")
    if words.ndim != 2 or words.shape[1] != LEGAL_ACTION_MASK_WORDS:
        raise ValueError(
            "mask_words must have shape [batch, 2], " f"got {tuple(words.shape)}"
        )
    bits = np.arange(64, dtype=np.uint64)
    dense = ((words[..., None] >> bits) & np.uint64(1)).astype(np.bool_)
    return dense.reshape(words.shape[0], -1)[:, :ACTION_SPACE_SIZE]
