from __future__ import annotations

import numpy as np
import pytest

from training import information_set_sampling_sweep as sweep
from training.tests.test_policy_iteration import tiny_batch


def test_identical_q_batches_have_perfect_pair_metrics() -> None:
    batch = tiny_batch()
    metrics = sweep.q_pair_metrics(batch, batch, np.full(9, 1 / 9))
    assert metrics["centered_q_cosine"] == pytest.approx(1.0)
    assert metrics["centered_q_delta_l2"] == pytest.approx(0.0)
    assert metrics["best_action_agreement"] == pytest.approx(1.0)
    assert metrics["pairwise_preference_agreement"] == pytest.approx(1.0)


def test_information_set_parser_defaults_to_short_paired_run() -> None:
    args = sweep.build_parser().parse_args(
        ["--batch-sweep-dir", "batch", "--output-dir", "output"]
    )
    assert args.qpc == 16
    assert args.worlds is None
    assert args.objective == "softmax_ce"
    assert args.temperature == pytest.approx(0.05)
