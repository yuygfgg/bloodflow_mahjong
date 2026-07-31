"""Grouped density-ratio objective and offline metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GroupMetrics:
    loss_sum: float
    fallback_loss_sum: float
    fallback_delta_sum: float
    roots: int
    reciprocal_rank_sum: float
    top1_sum: float
    top5_sum: float
    ess_sum: float
    low_ess: int

    @classmethod
    def empty(cls) -> "GroupMetrics":
        return cls(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0)

    def __add__(self, other: "GroupMetrics") -> "GroupMetrics":
        return GroupMetrics(
            self.loss_sum + other.loss_sum,
            self.fallback_loss_sum + other.fallback_loss_sum,
            self.fallback_delta_sum + other.fallback_delta_sum,
            self.roots + other.roots,
            self.reciprocal_rank_sum + other.reciprocal_rank_sum,
            self.top1_sum + other.top1_sum,
            self.top5_sum + other.top5_sum,
            self.ess_sum + other.ess_sum,
            self.low_ess + other.low_ess,
        )

    def summary(self) -> dict[str, float]:
        denominator = max(self.roots, 1)
        return {
            "nll": self.loss_sum / denominator,
            "deployment_fallback_nll": self.fallback_loss_sum / denominator,
            "deployment_nll_delta": self.fallback_delta_sum / denominator,
            "mrr": self.reciprocal_rank_sum / denominator,
            "top1": self.top1_sum / denominator,
            "top5": self.top5_sum / denominator,
            "mean_min_stream_ess": self.ess_sum / denominator,
            "atomic_fallback_fraction": self.low_ess / denominator,
        }


def centered_handwritten(log_weights: Tensor) -> Tensor:
    if log_weights.ndim != 2:
        raise ValueError("handwritten log weights must have shape [roots, candidates]")
    if torch.any(torch.isnan(log_weights) | torch.isposinf(log_weights)):
        raise ValueError("handwritten weights may only use -inf for unsupported worlds")
    finite = torch.isfinite(log_weights)
    if not torch.all(finite.any(dim=1)):
        raise ValueError("a belief group has no finite handwritten weight")
    maximum = log_weights.masked_fill(~finite, -torch.inf).amax(dim=1, keepdim=True)
    return log_weights - maximum


def _validated_logits(
    handwritten: Tensor,
    residuals: Tensor,
    positive_mask: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if handwritten.shape != residuals.shape or handwritten.shape != positive_mask.shape:
        raise ValueError("belief logits and labels must have identical shapes")
    if handwritten.ndim != 2:
        raise ValueError("belief groups must have shape [roots, candidates]")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if not torch.all(torch.isfinite(residuals)):
        raise ValueError("belief residuals must be finite")
    positive = positive_mask.bool()
    if not torch.all(positive.any(dim=1)):
        raise ValueError("every belief group needs a positive")
    if not torch.all(torch.isfinite(handwritten)[positive]):
        raise ValueError("positive worlds need finite handwritten weights")
    baseline = centered_handwritten(handwritten.float())
    return baseline, baseline + beta * residuals.float(), positive


def grouped_loss(
    handwritten: Tensor,
    residuals: Tensor,
    positive_mask: Tensor,
    *,
    variance_weight: float,
) -> tuple[Tensor, Tensor]:
    if not math.isfinite(variance_weight) or variance_weight < 0.0:
        raise ValueError("variance_weight must be finite and non-negative")
    _, logits, positive = _validated_logits(
        handwritten, residuals, positive_mask, beta=1.0
    )
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    nll = torch.logsumexp(logits, dim=1) - numerator
    finite = torch.isfinite(handwritten)
    finite_count = finite.sum(dim=1).clamp_min(1)
    finite_residuals = residuals.float().masked_fill(~finite, 0.0)
    finite_mean = finite_residuals.sum(dim=1, keepdim=True) / finite_count[:, None]
    centered_residual = (finite_residuals - finite_mean).masked_fill(~finite, 0.0)
    variance = centered_residual.square().sum(dim=1) / finite_count
    return (nll + variance_weight * variance).mean(), nll


@torch.no_grad()
def metrics(
    handwritten: Tensor,
    residuals: Tensor,
    positive_mask: Tensor,
    proposal_streams: Tensor | None = None,
    *,
    beta: float,
    proposal_stream_count: int = 2,
    low_ess_threshold: float = 8.0,
) -> GroupMetrics:
    if not math.isfinite(low_ess_threshold) or low_ess_threshold <= 0.0:
        raise ValueError("low_ess_threshold must be finite and positive")
    baseline, logits, positive = _validated_logits(
        handwritten, residuals, positive_mask, beta=beta
    )
    if proposal_streams is None:
        # Keep the small standalone objective API useful for legacy unit tests.
        # Production batches always carry two explicit proposal streams.
        proposal_stream_count = 1
        proposal_streams = torch.zeros_like(logits, dtype=torch.long)
        proposal_streams.scatter_(1, positive.to(torch.long).argmax(dim=1, keepdim=True), 1)
    if proposal_streams.shape != logits.shape:
        raise ValueError("proposal stream labels must match belief logits")
    if proposal_streams.ndim != 2 or proposal_stream_count <= 0:
        raise ValueError("proposal streams must be a non-empty rank-two tensor")
    if not torch.all(
        (proposal_streams >= 0) & (proposal_streams <= proposal_stream_count)
    ):
        raise ValueError("proposal stream labels are out of range")
    truth = proposal_streams == proposal_stream_count
    if not torch.all(truth.sum(dim=1) == 1):
        raise ValueError("each belief group must contain one truth candidate")
    if not torch.all((truth & positive).any(dim=1)):
        raise ValueError("the truth candidate must be positive")
    for stream in range(proposal_stream_count):
        if not torch.all((proposal_streams == stream).sum(dim=1) > 0):
            raise ValueError("each belief group must contain every proposal stream")
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    nll = torch.logsumexp(logits, dim=1) - numerator
    positive_score = logits.masked_fill(~positive, -torch.inf).amax(dim=1)
    better = (logits > positive_score[:, None]).sum(dim=1)
    tied = (logits == positive_score[:, None]).sum(dim=1)
    rank = better.float() + (tied.float() + 1.0) / 2.0
    stream_ess = []
    for stream in range(proposal_stream_count):
        stream_logits = logits.masked_fill(proposal_streams != stream, -torch.inf)
        finite = torch.isfinite(stream_logits)
        finite_any = finite.any(dim=1)
        maximum = stream_logits.masked_fill(~finite, -torch.inf).amax(dim=1, keepdim=True)
        maximum = torch.where(finite_any[:, None], maximum, torch.zeros_like(maximum))
        exponent = torch.where(
            finite,
            (stream_logits - maximum).exp(),
            torch.zeros_like(stream_logits),
        )
        total = exponent.sum(dim=1, keepdim=True)
        weights = torch.where(total > 0.0, exponent / total, torch.zeros_like(exponent))
        squared = weights.square().sum(dim=1)
        ess = torch.where(
            finite_any & (squared > 0.0), squared.reciprocal(), torch.zeros_like(squared)
        )
        stream_ess.append(ess)
    ess_matrix = torch.stack(stream_ess, dim=1)
    ess = ess_matrix.min(dim=1).values
    fallback = (ess_matrix < low_ess_threshold).any(dim=1)
    effective_logits = torch.where(fallback[:, None], baseline, logits)
    effective_numerator = torch.logsumexp(
        effective_logits.masked_fill(~positive, -torch.inf), dim=1
    )
    effective_nll = torch.logsumexp(effective_logits, dim=1) - effective_numerator
    baseline_numerator = torch.logsumexp(
        baseline.masked_fill(~positive, -torch.inf), dim=1
    )
    baseline_nll = torch.logsumexp(baseline, dim=1) - baseline_numerator
    fallback_delta = effective_nll - baseline_nll
    return GroupMetrics(
        loss_sum=float(nll.sum().item()),
        fallback_loss_sum=float(effective_nll.sum().item()),
        fallback_delta_sum=float(fallback_delta.sum().item()),
        roots=int(logits.shape[0]),
        reciprocal_rank_sum=float(rank.reciprocal().sum().item()),
        top1_sum=float((rank <= 1.0).sum().item()),
        top5_sum=float((rank <= 5.0).sum().item()),
        ess_sum=float(ess.sum().item()),
        low_ess=int((ess < low_ess_threshold).sum().item()),
    )
