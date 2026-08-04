from __future__ import annotations

import numpy as np
import torch

from training.device import stage_numpy_batch


def test_stage_numpy_batch_packs_noncontiguous_cpu_arrays() -> None:
    source = np.arange(48, dtype=np.int32).reshape(8, 6)
    expected = source[::2, 1:5]

    staged = stage_numpy_batch({"values": expected}, torch.device("cpu"))

    assert staged["values"].is_contiguous()
    np.testing.assert_array_equal(staged["values"].numpy(), expected)
