from __future__ import annotations

import numpy as np
import pytest
import torch

from training.evaluation import (
    bootstrap_mean_interval,
    collect_fixed_panel,
    evaluation_seeds,
    load_reference_panel,
    save_reference_panel,
    summarize_paired,
)
from training.model import BloodFlowTransformer, TransformerConfig


def tiny_actor() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=16,
        )
    ).eval()


def test_evaluation_seed_panel_is_unique_and_reproducible() -> None:
    first = evaluation_seeds(17, 1024)
    second = evaluation_seeds(17, 1024)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert not np.array_equal(first, evaluation_seeds(18, 1024))


def test_fixed_panel_never_uses_self_play_opponents() -> None:
    result = collect_fixed_panel(
        tiny_actor(),
        torch.device("cpu"),
        evaluation_seeds(23, 4),
        envs=4,
    )
    assert result.opponent_seat_counts["self_play"] == 0
    assert (
        result.opponent_seat_counts["rule_fast"]
        + result.opponent_seat_counts["rule_safe"]
        == 12
    )


def test_paired_summary_uses_lower_rank_as_improvement() -> None:
    reference_ranks = np.asarray([2, 3, 2, 3] * 64, dtype=np.float64)
    actor_ranks = reference_ranks - 1
    reference_scores = np.zeros_like(reference_ranks)
    actor_scores = np.full_like(reference_ranks, 500)
    result = summarize_paired(
        actor_ranks,
        actor_scores,
        reference_ranks,
        reference_scores,
        seed=9,
        bootstrap_samples=200,
    )
    assert result["paired_rank_delta"]["mean"] == -1
    assert result["rank_significant_positive"]
    assert result["score_significant_positive"]


def test_reference_panel_cache_rejects_another_identity(tmp_path) -> None:
    path = tmp_path / "panel.npz"
    seeds = evaluation_seeds(5, 8)
    ranks = np.arange(8, dtype=np.float64) % 4 + 1
    scores = np.arange(8, dtype=np.float64)
    save_reference_panel(
        path, seeds=seeds, ranks=ranks, scores=scores, fingerprint="same"
    )
    loaded = load_reference_panel(path, fingerprint="same")
    for expected, actual in zip((seeds, ranks, scores), loaded):
        np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ValueError, match="fingerprint"):
        load_reference_panel(path, fingerprint="different")


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval(np.asarray([]), seed=1)
