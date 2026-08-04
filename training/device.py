"""Small host-to-device helpers shared by training loops."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


def stage_numpy_batch(
    arrays: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, Tensor]:
    """Pack contiguous host arrays once before a minibatch is consumed."""
    return {
        name: torch.from_numpy(np.ascontiguousarray(array)).to(device=device)
        for name, array in arrays.items()
    }
