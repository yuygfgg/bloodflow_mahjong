from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from training.history_tournament import (
    TournamentGames,
    _parse_extra_checkpoint,
    _parse_extra_policy,
    _plackett_luce_ratings,
    _stratified_block_counts,
    balanced_schedule,
    summarize_tournament,
)


def test_parse_extra_checkpoint() -> None:
    assert _parse_extra_checkpoint("original_u063=runs/original/latest.pt") == (
        "original_u063",
        Path("runs/original/latest.pt"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_extra_checkpoint("missing-path=")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_extra_checkpoint("ambiguous__name=checkpoint.pt")

    assert _parse_extra_policy("search=runs/search.pt") == (
        "search",
        Path("runs/search.pt"),
    )
    with pytest.raises(argparse.ArgumentTypeError, match="extra policy"):
        _parse_extra_policy("missing-path=")


def test_balanced_schedule_covers_every_seat_and_pair_equally() -> None:
    lineups, blocks, seeds = balanced_schedule(5, rounds_per_combination=2, seed=17)
    assert len(lineups) == 5 * 2 * 24
    assert len(np.unique(blocks)) == 5 * 2
    for agent in range(5):
        assert np.sum(lineups == agent) == 4 * 24 * 2
        for seat in range(4):
            assert np.sum(lineups[:, seat] == agent) == 24 * 2
    for left in range(5):
        for right in range(left + 1, 5):
            both = np.any(lineups == left, axis=1) & np.any(lineups == right, axis=1)
            assert both.sum() == 3 * 24 * 2
    for block in np.unique(blocks):
        assert len(np.unique(seeds[blocks == block])) == 1


def test_plackett_luce_orders_a_dominant_agent_above_others() -> None:
    lineups = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.int8), (128, 1))
    ranks = np.tile(np.asarray([[1, 2, 3, 4]], dtype=np.int8), (128, 1))
    ratings = _plackett_luce_ratings(lineups, ranks, anchor=3)
    assert ratings[0] > ratings[1] > ratings[2] > ratings[3]


def test_bootstrap_preserves_every_fixed_agent_combination() -> None:
    lineups, blocks, seeds = balanced_schedule(5, rounds_per_combination=3, seed=19)
    ranks = np.tile(np.arange(1, 5, dtype=np.int8), (len(lineups), 1))
    games = TournamentGames(lineups, ranks, np.zeros_like(lineups), blocks, seeds)
    counts, inverse = _stratified_block_counts(games, samples=8, seed=23)
    for sample_counts in counts:
        for combination in range(5):
            combination_blocks = np.arange(combination * 3, combination * 3 + 3)
            assert sample_counts[combination_blocks].sum() == 3
        assert sample_counts[inverse].sum() == len(lineups)


def test_summary_uses_block_bootstrap_and_sl_anchor() -> None:
    lineups, blocks, seeds = balanced_schedule(4, rounds_per_combination=2, seed=23)
    ranks = np.empty_like(lineups)
    scores = np.empty_like(lineups, dtype=np.int64)
    for row, lineup in enumerate(lineups):
        order = np.argsort(lineup)
        ranks[row, order] = np.arange(1, 5, dtype=np.int8)
        scores[row, order] = np.asarray([600, 200, -200, -600])
    summary = summarize_tournament(
        TournamentGames(lineups, ranks, scores, blocks, seeds),
        ("sl", "u001", "rule_fast", "rule_safe"),
        anchor=0,
        bootstrap_samples=20,
        seed=29,
    )
    assert summary["anchor"] == "sl"
    assert summary["elo_like_order"][0] == "sl"
    assert summary["agents"]["sl"]["elo_like"] == 0.0
    assert summary["pairwise"]["sl__u001"]["games"] == 48
    assert summary["rating_comparisons"]["sl__u001"][
        "left_stronger_probability"
    ] == 1.0
