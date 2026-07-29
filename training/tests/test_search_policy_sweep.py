from __future__ import annotations

import numpy as np
import pytest

from training import search_policy_sweep as sweep
from training.tests.test_world_outcomes import tiny_world_batch


def reference_policy(states: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((states, 115), dtype=np.float32)
    probabilities[:, 0] = 0.9
    probabilities[:, 1] = 0.1
    return probabilities, np.zeros(states, dtype=np.int64)


def test_world_targets_are_legal_distributions() -> None:
    outcomes = tiny_world_batch()
    probabilities, actions = reference_policy(len(outcomes))
    for spec in sweep.default_target_specs():
        target, metrics = sweep.build_search_policy_target(
            spec, outcomes, probabilities, actions
        )
        np.testing.assert_allclose(target.sum(axis=1), 1.0)
        assert np.all(target[~outcomes.legal] == 0)
        assert np.all(target >= 0)
        assert 0.0 <= metrics["changed_state_rate"] <= 1.0


def test_rank_lcb_moves_only_on_positive_paired_evidence() -> None:
    outcomes = tiny_world_batch()
    rank = outcomes.rank_outcomes.copy()
    rank[:, 0] = -3
    rank[:, 1] = 3
    outcomes = outcomes.__class__(
        **{
            **{
                name: getattr(outcomes, name)
                for name in outcomes.__class__.__dataclass_fields__
            },
            "rank_outcomes": rank,
        }
    )
    probabilities, actions = reference_policy(len(outcomes))
    target, metrics = sweep.build_search_policy_target(
        sweep.TargetSpec("rank-lcb", "rank_lcb", 1.0),
        outcomes,
        probabilities,
        actions,
    )
    np.testing.assert_array_equal(target[:, 1], np.ones(len(outcomes)))
    assert metrics["changed_state_rate"] == pytest.approx(1.0)


def test_search_policy_parser_defaults_to_information_sets() -> None:
    args = sweep.build_parser().parse_args(
        ["--batch-sweep-dir", "batch", "--output-dir", "output"]
    )
    assert args.qpc == 16
    assert args.worlds == 16
    assert args.world_sampling == "information_set"
    assert args.targets is None
    assert args.kl_search_steps == 28
    assert args.reuse_outcome_prefix_dir is None


def test_select_target_specs_preserves_requested_order() -> None:
    specs = sweep.select_target_specs(
        ["vote-lex-p0p625", "uniform-control", "world-best-lex"]
    )
    assert [spec.name for spec in specs] == [
        "vote-lex-p0p625",
        "uniform-control",
        "world-best-lex",
    ]
    assert sweep.target_specs_from_identity(
        [spec.identity() for spec in specs]
    ) == specs


@pytest.mark.parametrize(
    "names", [[], ["uniform-control", "uniform-control"], ["missing-target"]]
)
def test_select_target_specs_rejects_invalid_names(names: list[str]) -> None:
    with pytest.raises(ValueError):
        sweep.select_target_specs(names)
