from __future__ import annotations

import numpy as np

from training import search_policy_combined_sweep as combined
from training.search_distillation import (
    benjamini_hochberg,
    build_holdout_win_target,
    build_rank_lcb_mirror_target,
    build_split_win_consensus_target,
    select_rank_lcb_challengers,
    summarize_greedy_rank_audit,
    summarize_greedy_rank_advantages,
)
from training.search_policy_sweep import TargetSpec
from training.tests.test_search_policy_sweep import reference_policy
from training.tests.test_world_outcomes import tiny_world_batch
from training.world_outcomes import combine_world_replicates


def _constant_action_values(action_zero: int, action_one: int):
    source = tiny_world_batch()
    rank = source.rank_outcomes.copy()
    rank[:, 0] = action_zero
    rank[:, 1] = action_one
    return source.__class__(
        **{
            **{
                name: getattr(source, name)
                for name in source.__class__.__dataclass_fields__
            },
            "rank_outcomes": rank,
        }
    )


def test_split_rank_target_requires_both_replicates_to_improve() -> None:
    left = _constant_action_values(-3, 3)
    right = _constant_action_values(-3, 3)
    outcomes = combine_world_replicates((left, right))
    probabilities, actions = reference_policy(len(outcomes))
    target, metrics = combined.build_split_consensus_target(
        TargetSpec("split", "split_rank_both", 0.0),
        outcomes,
        left,
        right,
        probabilities,
        actions,
    )
    np.testing.assert_array_equal(target[:, 1], np.ones(len(target)))
    assert metrics["split_selected_states"] == len(target)

    disagreeing = _constant_action_values(3, -3)
    outcomes = combine_world_replicates((left, disagreeing))
    target, metrics = combined.build_split_consensus_target(
        TargetSpec("split", "split_rank_both", 0.0),
        outcomes,
        left,
        disagreeing,
        probabilities,
        actions,
    )
    np.testing.assert_allclose(target, probabilities)
    assert metrics["split_selected_states"] == 0

def test_split_target_selection_preserves_order() -> None:
    specs = combined.select_split_target_specs(
        ["split-win-both-p0", "split-rank-agree-m0"]
    )
    assert [spec.name for spec in specs] == [
        "split-win-both-p0",
        "split-rank-agree-m0",
    ]


def test_production_split_win_target_matches_the_validated_sweep_target() -> None:
    left = _constant_action_values(-3, 3)
    right = _constant_action_values(-3, 3)
    outcomes = combine_world_replicates((left, right))
    probabilities, actions = reference_policy(len(outcomes))

    expected, expected_metrics = combined.build_split_consensus_target(
        TargetSpec("split", "split_win_both", 0.125),
        outcomes,
        left,
        right,
        probabilities,
        actions,
    )
    actual, actual_metrics = build_split_win_consensus_target(
        outcomes,
        left,
        right,
        probabilities,
        actions,
        margin=0.125,
    )

    np.testing.assert_allclose(actual, expected)
    assert actual_metrics["split_selected_states"] == expected_metrics[
        "split_selected_states"
    ]


def test_holdout_target_selects_and_validates_on_disjoint_replicates() -> None:
    selection = _constant_action_values(-3, 3)
    validation = _constant_action_values(-3, 3)
    outcomes = combine_world_replicates((selection, validation))
    probabilities, actions = reference_policy(len(outcomes))

    target, metrics = build_holdout_win_target(
        outcomes,
        selection,
        validation,
        probabilities,
        actions,
        margin=0.1875,
    )

    np.testing.assert_array_equal(target[:, 1], np.ones(len(target)))
    assert metrics["kind"] == "holdout_win"
    assert metrics["validation_win_rate_threshold"] == 0.6875
    assert metrics["split_selected_states"] == len(target)

    disagreeing_selection = _constant_action_values(3, -3)
    outcomes = combine_world_replicates((disagreeing_selection, validation))
    target, metrics = build_holdout_win_target(
        outcomes,
        disagreeing_selection,
        validation,
        probabilities,
        actions,
        margin=0.1875,
    )
    np.testing.assert_allclose(target, probabilities)
    assert metrics["split_selected_states"] == 0

    disagreeing_validation = _constant_action_values(3, -3)
    outcomes = combine_world_replicates((selection, disagreeing_validation))
    target, metrics = build_holdout_win_target(
        outcomes,
        selection,
        disagreeing_validation,
        probabilities,
        actions,
        margin=0.1875,
    )
    np.testing.assert_allclose(target, probabilities)
    assert metrics["split_selected_states"] == 0


def test_bh_fdr_uses_the_largest_passing_prefix() -> None:
    accepted = benjamini_hochberg(
        np.asarray([0.001, 0.04, 0.009, 0.5]), fdr=0.05
    )
    np.testing.assert_array_equal(accepted, [True, False, True, False])


def test_rank_lcb_mirror_target_separates_evidence_from_distribution() -> None:
    selection = _constant_action_values(-3, 3)
    validation = _constant_action_values(-3, 3)
    probabilities, actions = reference_policy(len(selection))

    target, metrics = build_rank_lcb_mirror_target(
        selection,
        validation,
        probabilities,
        actions,
        fdr=0.05,
        temperature=0.5,
    )

    assert np.all(target.row_confidence == 1)
    assert np.all(target.selected_actions == 1)
    assert np.all(target.distribution[:, 1] > probabilities[:, 1])
    assert np.all(target.distribution[:, 1] < 1)
    assert metrics["accepted_states"] == len(selection)

    harmful_validation = _constant_action_values(3, -3)
    rejected, rejected_metrics = build_rank_lcb_mirror_target(
        selection,
        harmful_validation,
        probabilities,
        actions,
    )
    assert np.all(rejected.row_confidence == 0)
    np.testing.assert_allclose(rejected.distribution, probabilities, atol=1e-6)
    assert rejected_metrics["accepted_states"] == 0


def test_rank_lcb_target_reads_only_the_selected_validation_pair() -> None:
    selection = _constant_action_values(-3, 3)
    legal = selection.legal.copy()
    legal[:, 2] = True
    rank_complete = selection.rank_outcomes.copy()
    rank_complete[:, 2] = -1
    selection = selection.__class__(
        **{
            **{
                name: getattr(selection, name)
                for name in selection.__class__.__dataclass_fields__
                if name not in {"legal", "rank_outcomes"}
            },
            "legal": legal,
            "rank_outcomes": rank_complete,
        }
    )
    probabilities, actions = reference_policy(len(selection))
    selected = select_rank_lcb_challengers(selection, actions)
    rank = np.zeros_like(selection.rank_outcomes)
    score = np.zeros_like(selection.score_outcomes)
    rows = np.arange(len(selection))
    rank[rows, actions] = selection.rank_outcomes[rows, actions]
    rank[rows, selected] = selection.rank_outcomes[rows, selected]
    score[rows, actions] = selection.score_outcomes[rows, actions]
    score[rows, selected] = selection.score_outcomes[rows, selected]
    validation = selection.__class__(
        **{
            **{
                name: getattr(selection, name)
                for name in selection.__class__.__dataclass_fields__
                if name not in {"rank_outcomes", "score_outcomes"}
            },
            "rank_outcomes": rank,
            "score_outcomes": score,
        }
    )

    target, metrics = build_rank_lcb_mirror_target(
        selection,
        validation,
        probabilities,
        actions,
        selected_actions=selected,
        temperature=0.5,
    )

    assert np.all(target.row_confidence == 1)
    assert metrics["target_q_metric_corpus"] == "selection"


def test_greedy_rank_audit_uses_only_audit_worlds() -> None:
    audit = _constant_action_values(-3, 3)
    reference = np.zeros(len(audit), dtype=np.int64)
    candidate = np.ones(len(audit), dtype=np.int64)

    metrics = summarize_greedy_rank_audit((audit,), (reference,), (candidate,))

    assert metrics["mean_rank_utility_advantage"] == 3.0
    assert metrics["ci95_low"] == 3.0
    assert metrics["greedy_flip_rate"] == 1.0


def test_sparse_greedy_rank_audit_preserves_zero_advantage_rows() -> None:
    audit = _constant_action_values(-3, 3)
    reference = np.zeros(len(audit), dtype=np.int64)
    candidate = reference.copy()
    candidate[0] = 1
    full = summarize_greedy_rank_audit((audit,), (reference,), (candidate,))

    rows = np.arange(len(audit))
    utility = audit.rank_outcomes.astype(np.float64) / 2.0
    advantage = np.zeros(len(audit), dtype=np.float64)
    changed = candidate != reference
    advantage[changed] = (
        utility[rows[changed], candidate[changed]]
        - utility[rows[changed], reference[changed]]
    ).mean(axis=1)
    sparse = summarize_greedy_rank_advantages(
        (advantage,), (audit.categories,), (changed,)
    )

    assert sparse == full


def test_sparse_greedy_rank_audit_rejects_nonzero_unchanged_rows() -> None:
    audit = _constant_action_values(-3, 3)
    with np.testing.assert_raises_regex(ValueError, "advantage values"):
        summarize_greedy_rank_advantages(
            (np.ones(len(audit)),),
            (audit.categories,),
            (np.zeros(len(audit), dtype=np.bool_),),
        )
