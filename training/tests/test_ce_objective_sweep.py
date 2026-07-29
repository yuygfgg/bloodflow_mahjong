from __future__ import annotations

import pytest

from training import ce_objective_sweep as sweep


def test_default_objective_specs_cover_pg_and_search_ce() -> None:
    specs = sweep.objective_specs(
        (0.05, 0.1, 0.2),
        (0.02, 0.05, 0.1),
        mirror_prior_floor=1e-4,
    )
    assert specs[0].objective == "expected_q"
    assert specs[1].objective == "uniform_ce"
    assert specs[2].objective == "hard_ce"
    assert [spec.temperature for spec in specs if spec.objective == "softmax_ce"] == [
        0.05,
        0.1,
        0.2,
    ]
    assert all(
        spec.prior_floor == pytest.approx(1e-4)
        for spec in specs
        if spec.objective == "mirror_ce"
    )


def test_ce_sweep_parser_defaults_to_shared_small_kl() -> None:
    args = sweep.build_parser().parse_args(
        ["--batch-sweep-dir", "batch", "--output-dir", "output"]
    )
    assert args.qpc == 256
    assert args.hard_ce is True
    assert args.uniform_control is True
    assert args.target_kl == pytest.approx(1e-4)
    assert args.learning_rate == pytest.approx(0.1)
