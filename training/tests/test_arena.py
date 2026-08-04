from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.arena import (
    Snapshot,
    _matchup_panel,
    _seat_panel_se,
    evaluate_matchup,
)
from training.model import BloodFlowTransformer, TransformerConfig


def tiny_model() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
            value_atoms=17,
        )
    )


def test_matchup_panel_reuses_each_seed_for_all_focal_seats() -> None:
    seeds, seats = _matchup_panel(12, seed=41)

    assert seeds.shape == (12,)
    assert seats.shape == (12,)
    for start in range(0, 12, 4):
        assert len(set(seeds[start : start + 4])) == 1
        np.testing.assert_array_equal(seats[start : start + 4], [0, 1, 2, 3])


@pytest.mark.parametrize("games", [0, 1, 6])
def test_matchup_panel_rejects_unbalanced_game_counts(games: int) -> None:
    with pytest.raises(ValueError, match="positive multiple of four"):
        _matchup_panel(games, seed=41)


def test_standard_error_uses_independent_four_seat_panels() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0])

    expected = np.std([2.5, 4.5], ddof=1) / np.sqrt(2)
    assert _seat_panel_se(values) == pytest.approx(expected)


def test_same_policy_matchup_is_neutral_and_chunk_independent() -> None:
    model = tiny_model()
    snapshot = Snapshot("same", Path("same.pt"), model)
    one_at_a_time = evaluate_matchup(
        snapshot,
        snapshot,
        games=4,
        envs=1,
        seed=43,
        device=torch.device("cpu"),
    )
    together = evaluate_matchup(
        snapshot,
        snapshot,
        games=4,
        envs=4,
        seed=43,
        device=torch.device("cpu"),
    )

    assert one_at_a_time == together
    assert together.mean_score_delta == 0.0
    assert together.score_se == 0.0
    assert together.mean_rank == 2.5
    assert together.rank_se == 0.0
