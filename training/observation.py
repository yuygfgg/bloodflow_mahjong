"""Conversion helpers for caller-owned engine buffers."""

from __future__ import annotations

import numpy as np

ACTION_SPACE_SIZE = 115
LEGAL_ACTION_MASK_WORDS = 2


def bucket_history_width(lengths: np.ndarray, capacity: int) -> int:
    """Bucket a padded history width to keep CUDA attention shapes stable."""

    values = np.asarray(lengths)
    if values.ndim != 1:
        raise ValueError("history lengths must be one-dimensional")
    if capacity <= 0:
        raise ValueError("history capacity must be positive")
    maximum = max(int(values.max(initial=0)), 1)
    if maximum > capacity:
        raise ValueError("history length exceeds its allocated capacity")
    bucket = max(8, 1 << (maximum - 1).bit_length())
    return min(bucket, capacity)


def unpack_action_masks(
    mask_words: np.ndarray, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Expand packed engine masks to a boolean ``[batch, 115]`` array."""
    words = np.asarray(mask_words)
    if words.dtype != np.uint64:
        raise TypeError(f"mask_words must have dtype uint64, got {words.dtype}")
    if words.ndim != 2 or words.shape[1] != LEGAL_ACTION_MASK_WORDS:
        raise ValueError(
            "mask_words must have shape [batch, 2], " f"got {tuple(words.shape)}"
        )
    # The engine stores bit zero as each word's least-significant bit.
    # Byte-wise expansion avoids a temporary uint64 tensor eight times larger
    # than the boolean result.
    little_endian = words.astype("<u8", copy=False)
    byte_rows = np.ascontiguousarray(little_endian).view(np.uint8).reshape(
        words.shape[0], -1
    )
    dense = np.unpackbits(byte_rows, axis=1, bitorder="little")[
        :, :ACTION_SPACE_SIZE
    ]
    if out is None:
        return dense.astype(np.bool_)
    if out.dtype != np.bool_ or out.shape != (words.shape[0], ACTION_SPACE_SIZE):
        raise ValueError(
            "out must be a bool array with shape "
            f"[{words.shape[0]}, {ACTION_SPACE_SIZE}], got {out.dtype} {out.shape}"
        )
    out[...] = dense
    return out
