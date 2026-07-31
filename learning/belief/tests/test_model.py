from __future__ import annotations

import torch

from learning.belief.model import BeliefModelConfig, BeliefResidualModel
from learning.belief.objective import grouped_loss, metrics
from learning.belief.train import (
    _choose_beta,
    _collision_counts,
    _prepare_output_dir,
    _validate_disjoint_blocks,
)


def _inputs(batch: int = 2, candidates: int = 4):
    tile_obs = torch.zeros(batch, 10, 27, dtype=torch.uint8)
    melds = torch.full((batch, 4, 4, 3), 255, dtype=torch.uint8)
    meta = torch.zeros(batch, 34, dtype=torch.int32)
    events = torch.zeros(batch, 192, 8, dtype=torch.int32)
    lengths = torch.zeros(batch, dtype=torch.int64)
    worlds = torch.zeros(batch, candidates, 4, 27, dtype=torch.uint8)
    return tile_obs, melds, meta, events, lengths, worlds


def test_zero_initialized_model_preserves_handwritten_posterior() -> None:
    model = BeliefResidualModel(BeliefModelConfig())
    residuals = model(*_inputs())
    torch.testing.assert_close(residuals, torch.zeros_like(residuals))


def test_candidate_permutation_only_permutes_outputs() -> None:
    model = BeliefResidualModel(BeliefModelConfig())
    inputs = list(_inputs(candidates=5))
    inputs[-1].random_(0, 5)
    with torch.no_grad():
        first = model(*inputs)
        order = torch.tensor([3, 0, 4, 1, 2])
        inputs[-1] = inputs[-1][:, order]
        second = model(*inputs)
    torch.testing.assert_close(second, first[:, order])


def test_grouped_objective_supports_duplicate_positives() -> None:
    handwritten = torch.tensor([[0.0, -1.0, 0.0, -2.0]])
    residuals = torch.zeros_like(handwritten, requires_grad=True)
    positive = torch.tensor([[1, 0, 1, 0]], dtype=torch.uint8)
    loss, _ = grouped_loss(
        handwritten, residuals, positive, variance_weight=1e-3
    )
    loss.backward()
    assert torch.isfinite(loss)
    expected = torch.logsumexp(handwritten, dim=1) - torch.logsumexp(
        handwritten[:, [0, 2]], dim=1
    )
    torch.testing.assert_close(loss, expected.mean())
    summary = metrics(handwritten, residuals.detach(), positive, beta=1.0).summary()
    assert abs(summary["mrr"] - 2.0 / 3.0) < 1e-6


def test_unsupported_negative_worlds_keep_zero_probability() -> None:
    handwritten = torch.tensor([[0.0, -torch.inf, -1.0]])
    residuals = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[1, 0, 0]], dtype=torch.uint8)

    loss, _ = grouped_loss(
        handwritten, residuals, positive, variance_weight=0.0
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert residuals.grad is not None
    assert residuals.grad[0, 1] == 0.0


def test_invalid_handwritten_support_is_rejected() -> None:
    residuals = torch.zeros(1, 2)
    positive = torch.tensor([[1, 0]], dtype=torch.uint8)

    for handwritten in (
        torch.tensor([[torch.inf, 0.0]]),
        torch.tensor([[torch.nan, 0.0]]),
        torch.tensor([[-torch.inf, 0.0]]),
    ):
        try:
            grouped_loss(
                handwritten, residuals, positive, variance_weight=0.0
            )
        except ValueError:
            continue
        raise AssertionError("invalid handwritten support was accepted")


def test_ess_fallback_does_not_report_an_unusable_nll_gain() -> None:
    handwritten = torch.zeros(1, 8)
    residuals = torch.tensor([[0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    positive = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)

    summary = metrics(handwritten, residuals, positive, beta=1.0).summary()

    assert summary["nll"] > 10.0
    assert abs(summary["deployment_fallback_nll"] - torch.log(torch.tensor(8.0)).item()) < 1e-6
    assert summary["deployment_nll_delta"] == 0.0
    assert summary["atomic_fallback_fraction"] == 1.0


def test_collision_audit_counts_duplicate_positive_worlds() -> None:
    worlds = torch.zeros(1, 3, 4, 27, dtype=torch.uint8)
    worlds[:, 1] = 1
    positive = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    assert _collision_counts(worlds, positive) == (1, 1)


def test_training_output_directory_must_be_absent_or_empty(tmp_path) -> None:
    output = tmp_path / "model"

    _prepare_output_dir(output)
    _prepare_output_dir(output)
    (output / "stale-manifest.json").write_text("{}")

    try:
        _prepare_output_dir(output)
    except FileExistsError as error:
        assert "not empty" in str(error)
        return
    raise AssertionError("non-empty training output was accepted")


def test_beta_selection_uses_the_ess_fallback_metric() -> None:
    values = {
        0.0: {
            "nll": 2.0,
            "deployment_fallback_nll": 2.0,
            "atomic_fallback_fraction": 0.1,
        },
        0.25: {
            "nll": 1.9,
            "deployment_fallback_nll": 1.9,
            "atomic_fallback_fraction": 0.1,
        },
        1.0: {
            "nll": 1.0,
            "deployment_fallback_nll": 2.0,
            "atomic_fallback_fraction": 0.1,
        },
    }

    assert _choose_beta(values) == 0.25


def test_split_blocks_must_be_disjoint() -> None:
    try:
        _validate_disjoint_blocks(
            {
                "train": frozenset({0, 1}),
                "calibration": frozenset({2}),
                "development": frozenset({1, 3}),
            }
        )
    except ValueError as error:
        assert "cross 'train' and 'development'" in str(error)
        return
    raise AssertionError("a block shared by two splits was accepted")
