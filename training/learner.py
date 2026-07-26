"""CUDA-oriented IQL/AWR optimization and critic diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .iql import (
    IndependentCritics,
    awr_actor_loss,
    critic_losses,
    gather_action_values,
    policy_reference_kl,
)
from .model import BloodFlowTransformer
from .observation import bucket_history_width
from .oracle import OracleCritics, distillation_loss
from .policy_pool import CATEGORY_COUNT, CATEGORY_NAMES, ReplaySource


CRITIC_METRIC_NAMES = (
    "q_loss",
    "q_regression",
    "cql",
    "v_loss",
    "oracle_q_loss",
    "oracle_v_loss",
    "distillation_loss",
)
ACTOR_METRIC_NAMES = (
    "actor_loss",
    "actor_policy_loss",
    "actor_reference_kl_loss",
    "actor_advantage_mean",
    "actor_weight_mean",
    "actor_effective_sample_size",
)
MC_CRITIC_METRIC_NAMES = (
    "mc_critic_loss",
    "mc_absolute_loss",
    "mc_centered_loss",
    "mc_pairwise_loss",
    "mc_train_pairwise_accuracy",
    "mc_train_groups",
    "mc_train_pairs",
)


@dataclass(frozen=True)
class LearningConfig:
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    actor_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    critic_batch_size: int = 2048
    actor_batch_size: int = 1024
    microbatch_size: int = 256
    initial_critic_steps: int = 500
    critic_steps_per_iteration: int = 64
    actor_steps_per_iteration: int = 8
    mc_critic_batch_size: int = 192
    mc_critic_steps_per_iteration: int = 4
    expectile: float = 0.7
    huber_delta: float = 1.0
    offline_cql_scale: float = 0.10
    minimum_cql_fraction: float = 0.10
    awr_beta: float = 0.10
    awr_max_weight: float = 20.0
    reference_kl_scale: float = 0.05
    max_grad_norm: float = 1.0
    minimum_critic_steps: int = 500
    minimum_middle_late_improvement: float = 0.02
    minimum_middle_late_correlation: float = 0.02
    maximum_q_disagreement: float = 0.50
    oracle_distillation_scale: float = 0.10
    teacher_readiness_streak: int = 3
    minimum_oracle_relative_mae_gain: float = 0.02
    minimum_oracle_early_relative_mae_gain: float = 0.02
    minimum_oracle_early_improvement: float = 0.02
    minimum_oracle_early_correlation: float = 0.02
    minimum_oracle_value_correlation: float = 0.02
    maximum_oracle_q_disagreement: float = 0.15
    maximum_oracle_expectile_balance_error: float = 0.10
    minimum_mc_train_targets: int = 512
    minimum_mc_validation_targets: int = 512
    minimum_mc_validation_groups: int = 128
    minimum_mc_pairwise_pairs: int = 128
    minimum_mc_pairwise_accuracy: float = 0.55
    maximum_mc_mean_regret: float = 0.10
    mc_centered_loss_scale: float = 1.0
    mc_pairwise_loss_scale: float = 0.25
    mc_pairwise_temperature: float = 0.10
    mc_centered_huber_delta: float = 0.25

    def __post_init__(self) -> None:
        positive = (
            self.critic_learning_rate,
            self.value_learning_rate,
            self.actor_learning_rate,
            self.critic_batch_size,
            self.actor_batch_size,
            self.microbatch_size,
            self.initial_critic_steps,
            self.critic_steps_per_iteration,
            self.actor_steps_per_iteration,
            self.mc_critic_batch_size,
            self.mc_critic_steps_per_iteration,
            self.huber_delta,
            self.awr_beta,
            self.awr_max_weight,
            self.max_grad_norm,
            self.minimum_critic_steps,
            self.teacher_readiness_streak,
            self.mc_pairwise_temperature,
            self.mc_centered_huber_delta,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("learning rates, sizes, steps, and scales must be positive")
        if not 0 < self.expectile < 1:
            raise ValueError("expectile must be in (0, 1)")
        if not 0 <= self.minimum_cql_fraction <= 1:
            raise ValueError("minimum_cql_fraction must be in [0, 1]")
        nonnegative = (
            self.offline_cql_scale,
            self.reference_kl_scale,
            self.maximum_q_disagreement,
            self.oracle_distillation_scale,
            self.minimum_mc_pairwise_accuracy,
            self.maximum_mc_mean_regret,
            self.maximum_oracle_q_disagreement,
            self.maximum_oracle_expectile_balance_error,
            self.mc_centered_loss_scale,
            self.mc_pairwise_loss_scale,
        )
        if min(nonnegative) < 0:
            raise ValueError("loss coefficients and gate thresholds cannot be negative")
        counts = (
            self.minimum_mc_train_targets,
            self.minimum_mc_validation_targets,
            self.minimum_mc_validation_groups,
            self.minimum_mc_pairwise_pairs,
        )
        if min(counts) < 0:
            raise ValueError("minimum MC counts cannot be negative")
        if self.minimum_mc_pairwise_accuracy > 1:
            raise ValueError("minimum_mc_pairwise_accuracy must be at most one")
        bounded = (
            self.minimum_middle_late_correlation,
            self.minimum_oracle_early_correlation,
            self.minimum_oracle_value_correlation,
            self.minimum_oracle_relative_mae_gain,
            self.minimum_oracle_early_relative_mae_gain,
        )
        if any(not math.isfinite(value) or value > 1 for value in bounded):
            raise ValueError("correlation and relative-gain thresholds must be finite <= 1")
        lower_bounds = (
            self.minimum_middle_late_improvement,
            self.minimum_oracle_early_improvement,
        )
        if any(not math.isfinite(value) or value > 1 for value in lower_bounds):
            raise ValueError("improvement thresholds must be finite and at most one")


@dataclass(frozen=True)
class LearningBatch:
    tile_obs: np.ndarray
    melds: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray
    legal: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    categories: np.ndarray
    sources: np.ndarray
    policy_versions: np.ndarray
    behavior_probabilities: np.ndarray
    temperatures: np.ndarray
    trajectory_ids: np.ndarray
    step_indices: np.ndarray
    oracle_tiles: np.ndarray | None = None
    rule_actions: np.ndarray | None = None
    mc_query_ids: np.ndarray | None = None
    mc_candidate_counts: np.ndarray | None = None
    mc_reliable_actions: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = len(self.actions)
        aligned = (
            self.tile_obs,
            self.melds,
            self.meta,
            self.events,
            self.event_lengths,
            self.legal,
            self.returns,
            self.categories,
            self.sources,
            self.policy_versions,
            self.behavior_probabilities,
            self.temperatures,
            self.trajectory_ids,
            self.step_indices,
        )
        if any(len(value) != size for value in aligned):
            raise ValueError("learning batch arrays must have the same leading size")
        if self.oracle_tiles is not None and len(self.oracle_tiles) != size:
            raise ValueError("oracle tiles must align with learning rows")
        if self.mc_query_ids is not None and len(self.mc_query_ids) != size:
            raise ValueError("MC query ids must align with learning rows")
        if self.mc_candidate_counts is not None and len(self.mc_candidate_counts) != size:
            raise ValueError("MC candidate counts must align with learning rows")
        if self.mc_reliable_actions is not None:
            if self.mc_reliable_actions.shape != (size, 115):
                raise ValueError(
                    "MC reliable actions must have shape [batch, 115]"
                )
            if self.mc_reliable_actions.dtype != np.bool_:
                raise ValueError("MC reliable actions must be boolean")
        mc_metadata = (
            self.mc_query_ids,
            self.mc_candidate_counts,
            self.mc_reliable_actions,
        )
        if any(value is None for value in mc_metadata) and any(
            value is not None for value in mc_metadata
        ):
            raise ValueError(
                "MC query ids, counts, and reliable actions must be paired"
            )
        if self.legal.shape != (size, 115):
            raise ValueError("legal must have shape [batch, 115]")
        if np.any(~self.legal[np.arange(size), self.actions.astype(np.int64)]):
            raise ValueError("learning batch contains an illegal behavior action")
        if self.rule_actions is not None:
            if self.rule_actions.shape != (size,):
                raise ValueError("rule actions must have shape [batch]")
            if np.any(
                ~self.legal[
                    np.arange(size), self.rule_actions.astype(np.int64, copy=False)
                ]
            ):
                raise ValueError("learning batch contains an illegal rule action")

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, indices: np.ndarray) -> LearningBatch:
        indices = np.asarray(indices, dtype=np.int64)
        return LearningBatch(
            tile_obs=self.tile_obs[indices],
            melds=self.melds[indices],
            meta=self.meta[indices],
            events=self.events[indices],
            event_lengths=self.event_lengths[indices],
            legal=self.legal[indices],
            actions=self.actions[indices],
            returns=self.returns[indices],
            categories=self.categories[indices],
            sources=self.sources[indices],
            policy_versions=self.policy_versions[indices],
            behavior_probabilities=self.behavior_probabilities[indices],
            temperatures=self.temperatures[indices],
            trajectory_ids=self.trajectory_ids[indices],
            step_indices=self.step_indices[indices],
            oracle_tiles=(
                None if self.oracle_tiles is None else self.oracle_tiles[indices]
            ),
            rule_actions=(
                None if self.rule_actions is None else self.rule_actions[indices]
            ),
            mc_query_ids=(
                None if self.mc_query_ids is None else self.mc_query_ids[indices]
            ),
            mc_candidate_counts=(
                None
                if self.mc_candidate_counts is None
                else self.mc_candidate_counts[indices]
            ),
            mc_reliable_actions=(
                None
                if self.mc_reliable_actions is None
                else self.mc_reliable_actions[indices]
            ),
        )

    def tensors(self, device: torch.device) -> dict[str, Tensor]:
        maximum = bucket_history_width(self.event_lengths, self.events.shape[1])
        result = {
            "tile_obs": torch.as_tensor(self.tile_obs, device=device),
            "melds": torch.as_tensor(self.melds, device=device),
            "meta": torch.as_tensor(self.meta, device=device),
            "events": torch.as_tensor(self.events[:, :maximum], device=device),
            "event_lengths": torch.as_tensor(
                self.event_lengths.astype(np.int64, copy=False), device=device
            ),
            "legal": torch.as_tensor(self.legal, device=device),
            "actions": torch.as_tensor(
                self.actions.astype(np.int64, copy=False), device=device
            ),
            "returns": torch.as_tensor(self.returns, device=device),
            "sources": torch.as_tensor(
                self.sources.astype(np.int64, copy=False), device=device
            ),
            "categories": torch.as_tensor(
                self.categories.astype(np.int64, copy=False), device=device
            ),
        }
        if self.oracle_tiles is not None:
            result["oracle_tiles"] = torch.as_tensor(
                self.oracle_tiles, device=device
            )
        if self.mc_query_ids is not None:
            result["mc_query_ids"] = torch.as_tensor(
                self.mc_query_ids.astype(np.int64, copy=False), device=device
            )
            result["mc_candidate_counts"] = torch.as_tensor(
                self.mc_candidate_counts.astype(np.int64, copy=False), device=device
            )
            result["mc_reliable_actions"] = torch.as_tensor(
                self.mc_reliable_actions, device=device
            )
        return result


@dataclass(frozen=True)
class MCGroupLossOutput:
    loss: Tensor
    absolute_loss: Tensor
    centered_loss: Tensor
    pairwise_loss: Tensor
    pairwise_accuracy: Tensor
    group_count: Tensor
    pair_count: Tensor


def mc_group_critic_loss(
    q1_values: Tensor,
    q2_values: Tensor,
    actions: Tensor,
    returns: Tensor,
    query_ids: Tensor,
    reliable_actions: Tensor,
    *,
    huber_delta: float,
    centered_scale: float,
    pairwise_scale: float,
    pairwise_temperature: float,
) -> MCGroupLossOutput:
    """Fit absolute returns and within-query action differences together."""

    if q1_values.shape != q2_values.shape or q1_values.ndim != 2:
        raise ValueError("MC Q tensors must be aligned matrices")
    rows = q1_values.shape[0]
    vectors = (actions, returns, query_ids)
    if any(value.shape != (rows,) for value in vectors):
        raise ValueError("MC actions, returns, and query ids must align with Q rows")
    if reliable_actions.shape != q1_values.shape:
        raise ValueError("MC reliable actions must align with Q rows and actions")
    if reliable_actions.dtype != torch.bool:
        raise ValueError("MC reliable actions must be boolean")
    if rows == 0:
        raise ValueError("MC group loss requires at least one query")
    if huber_delta <= 0 or pairwise_temperature <= 0:
        raise ValueError("MC Huber delta and pairwise temperature must be positive")
    if centered_scale < 0 or pairwise_scale < 0:
        raise ValueError("MC auxiliary loss scales must be non-negative")

    target = returns.detach().float()
    q1 = gather_action_values(q1_values, actions).float()
    q2 = gather_action_values(q2_values, actions).float()
    unique_queries, inverse, counts = torch.unique(
        query_ids.long(), sorted=True, return_inverse=True, return_counts=True
    )
    torch._assert_async((unique_queries >= 0).all(), "MC query ids must be non-negative")
    torch._assert_async((counts >= 2).all(), "MC queries must contain at least two actions")
    group_count = len(unique_queries)
    absolute_loss = 0.5 * (
        F.huber_loss(q1, target, delta=huber_delta)
        + F.huber_loss(q2, target, delta=huber_delta)
    )

    same_query = inverse[:, None] == inverse[None, :]
    upper_triangle = torch.triu(
        torch.ones((rows, rows), dtype=torch.bool, device=target.device), diagonal=1
    )
    target_gaps = target[:, None] - target[None, :]
    pair_mask = same_query & upper_triangle
    reliable_edges = reliable_actions[:, actions.long()]
    pair_mask &= reliable_edges & reliable_edges.T
    pair_count = pair_mask.sum()
    if bool(pair_count):
        selected_target_gaps = target_gaps[pair_mask]
        directions = selected_target_gaps.sign()
        weights = (selected_target_gaps.abs() / pairwise_temperature).clamp(max=1.0)
        q1_gaps = (q1[:, None] - q1[None, :])[pair_mask]
        q2_gaps = (q2[:, None] - q2[None, :])[pair_mask]
        pair_groups = inverse[:, None].expand(rows, rows)[pair_mask]
        pair_counts_by_group = torch.zeros(
            group_count, dtype=target.dtype, device=target.device
        ).scatter_add_(0, pair_groups, torch.ones_like(selected_target_gaps))
        reliable_group_count = (pair_counts_by_group > 0).sum()

        def equal_pair_group_mean(pair_losses: Tensor) -> Tensor:
            group_sums = torch.zeros(
                group_count, dtype=pair_losses.dtype, device=pair_losses.device
            ).scatter_add_(0, pair_groups, pair_losses)
            eligible_groups = pair_counts_by_group > 0
            return (
                group_sums[eligible_groups] / pair_counts_by_group[eligible_groups]
            ).mean()

        centered_pair_q1 = F.huber_loss(
            q1_gaps,
            selected_target_gaps,
            delta=huber_delta,
            reduction="none",
        )
        centered_pair_q2 = F.huber_loss(
            q2_gaps,
            selected_target_gaps,
            delta=huber_delta,
            reduction="none",
        )
        centered_loss = 0.5 * (
            equal_pair_group_mean(centered_pair_q1)
            + equal_pair_group_mean(centered_pair_q2)
        )

        q1_pair_losses = (
            weights
            * pairwise_temperature
            * F.softplus(-directions * q1_gaps / pairwise_temperature)
        )
        q2_pair_losses = (
            weights
            * pairwise_temperature
            * F.softplus(-directions * q2_gaps / pairwise_temperature)
        )
        pairwise_loss = 0.5 * (
            equal_pair_group_mean(q1_pair_losses)
            + equal_pair_group_mean(q2_pair_losses)
        )
        conservative_gaps = (
            torch.minimum(q1, q2)[:, None] - torch.minimum(q1, q2)[None, :]
        )[pair_mask]
        pairwise_accuracy = (
            conservative_gaps * target_gaps[pair_mask] > 0
        ).float().mean()
    else:
        centered_loss = (q1.sum() + q2.sum()) * 0.0
        pairwise_loss = (q1.sum() + q2.sum()) * 0.0
        pairwise_accuracy = pairwise_loss.detach()
        reliable_group_count = pair_count.new_zeros(())
    loss = centered_scale * centered_loss + pairwise_scale * pairwise_loss
    return MCGroupLossOutput(
        loss=loss,
        absolute_loss=absolute_loss,
        centered_loss=centered_loss,
        pairwise_loss=pairwise_loss,
        pairwise_accuracy=pairwise_accuracy,
        group_count=reliable_group_count,
        pair_count=pair_count,
    )


def cql_scale_from_coverage(
    config: LearningConfig, current_source_fraction: float
) -> float:
    """Decay CQL from replay coverage, never from an arbitrary update count."""

    if not 0 <= current_source_fraction <= 1:
        raise ValueError("current source fraction must be in [0, 1]")
    multiplier = max(config.minimum_cql_fraction, 1.0 - current_source_fraction)
    return config.offline_cql_scale * multiplier


def make_optimizers(
    actor: BloodFlowTransformer,
    critics: IndependentCritics,
    config: LearningConfig,
    oracle: OracleCritics | None = None,
) -> dict[str, torch.optim.Optimizer]:
    common = {"betas": (0.9, 0.95), "eps": 1e-5, "weight_decay": config.weight_decay}
    fused = next(actor.parameters()).device.type == "cuda"
    optimizers: dict[str, torch.optim.Optimizer] = {
        "actor": torch.optim.AdamW(
            actor.parameters(), lr=config.actor_learning_rate, fused=fused, **common
        ),
        "q": torch.optim.AdamW(
            [*critics.q1.parameters(), *critics.q2.parameters()],
            lr=config.critic_learning_rate,
            fused=fused,
            **common,
        ),
        "v": torch.optim.AdamW(
            critics.v.parameters(),
            lr=config.value_learning_rate,
            fused=fused,
            **common,
        ),
    }
    if oracle is not None:
        optimizers["oracle_q"] = torch.optim.AdamW(
            [*oracle.q1.parameters(), *oracle.q2.parameters()],
            lr=config.critic_learning_rate,
            fused=fused,
            **common,
        )
        optimizers["oracle_v"] = torch.optim.AdamW(
            oracle.v.parameters(),
            lr=config.value_learning_rate,
            fused=fused,
            **common,
        )
    return optimizers


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def _tensor_microbatches(
    tensors: dict[str, Tensor], size: int
) -> list[dict[str, Tensor]]:
    rows = len(tensors["actions"])
    return [
        {name: value[start : start + size] for name, value in tensors.items()}
        for start in range(0, rows, size)
    ]


def critic_update_deferred(
    critics: IndependentCritics,
    optimizers: dict[str, torch.optim.Optimizer],
    batch: LearningBatch,
    config: LearningConfig,
    device: torch.device,
    *,
    cql_scale: float,
    oracle: OracleCritics | None = None,
    enable_oracle_distillation: bool = False,
) -> Tensor:
    if not len(batch):
        raise ValueError("critic batch cannot be empty")
    if oracle is not None and batch.oracle_tiles is None:
        raise ValueError("oracle training requires oracle tile counts")
    if enable_oracle_distillation and oracle is None:
        raise ValueError("oracle distillation requires oracle critics")
    critics.train()
    if oracle is not None:
        oracle.train()
    active = [optimizers["q"], optimizers["v"]]
    if oracle is not None:
        active += [optimizers["oracle_q"], optimizers["oracle_v"]]
    for optimizer in active:
        optimizer.zero_grad(set_to_none=True)

    metric_rows: list[Tensor] = []
    # Copy the materialized batch once.  NumPy advanced indexing per
    # microbatch used to create eight host copies and eight groups of H2D
    # transfers for the default critic batch.
    tensors = batch.tensors(device)
    microbatches = _tensor_microbatches(tensors, config.microbatch_size)
    for microbatch in microbatches:
        state = (
            microbatch["tile_obs"],
            microbatch["melds"],
            microbatch["meta"],
            microbatch["events"],
            microbatch["event_lengths"],
        )
        with _autocast(device):
            output = critics(*state, microbatch["legal"])
            losses = critic_losses(
                output.q1.raw_values,
                output.q2.raw_values,
                output.value,
                microbatch["actions"],
                microbatch["returns"],
                microbatch["legal"],
                expectile=config.expectile,
                huber_delta=config.huber_delta,
                cql_scale=cql_scale,
                cql_sample_mask=microbatch["sources"]
                != int(ReplaySource.MC_TEACHER),
                value_sample_mask=microbatch["sources"]
                != int(ReplaySource.MC_TEACHER),
            )
            combined = losses.q_loss + losses.value_loss
            oracle_losses = None
            distill = combined.new_zeros(())
            if oracle is not None:
                oracle_output = oracle(
                    *state, microbatch["oracle_tiles"], microbatch["legal"]
                )
                oracle_losses = critic_losses(
                    oracle_output.q1.raw_values,
                    oracle_output.q2.raw_values,
                    oracle_output.value,
                    microbatch["actions"],
                    microbatch["returns"],
                    microbatch["legal"],
                    expectile=config.expectile,
                    huber_delta=config.huber_delta,
                    cql_scale=cql_scale,
                    cql_sample_mask=microbatch["sources"]
                    != int(ReplaySource.MC_TEACHER),
                    value_sample_mask=microbatch["sources"]
                    != int(ReplaySource.MC_TEACHER),
                )
                student_q1 = output.q1.raw_values
                student_q2 = output.q2.raw_values
                teacher_q1 = oracle_output.q1.raw_values
                teacher_q2 = oracle_output.q2.raw_values
                if not enable_oracle_distillation:
                    student_q1 = student_q1.detach()
                    student_q2 = student_q2.detach()
                    teacher_q1 = teacher_q1.detach()
                    teacher_q2 = teacher_q2.detach()
                distill = distillation_loss(
                    student_q1,
                    student_q2,
                    teacher_q1,
                    teacher_q2,
                    microbatch["legal"],
                    microbatch["actions"],
                )
                combined = combined + oracle_losses.q_loss + oracle_losses.value_loss
                if enable_oracle_distillation:
                    combined = combined + config.oracle_distillation_scale * distill
        (combined / len(microbatches)).backward()
        oracle_q = distill.new_zeros(())
        oracle_v = distill.new_zeros(())
        if oracle_losses is not None:
            oracle_q = oracle_losses.q_loss
            oracle_v = oracle_losses.value_loss
        # Keep only compact metric tensors instead of synchronizing the host
        # or retaining full forward outputs between microbatches.
        metric_rows.append(
            torch.stack(
                (
                    losses.q_loss,
                    losses.q1_regression + losses.q2_regression,
                    losses.q1_cql + losses.q2_cql,
                    losses.value_loss,
                    oracle_q,
                    oracle_v,
                    distill,
                )
            )
            .detach()
            .float()
        )

    parameters: list[Tensor] = list(critics.parameters())
    if oracle is not None:
        parameters += list(oracle.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    for optimizer in active:
        optimizer.step()
    return torch.cat(
        (torch.stack(metric_rows).mean(dim=0), grad_norm.detach().float().reshape(1))
    )


def resolve_critic_statistics(summary: Tensor, *, cql_scale: float) -> dict[str, float]:
    values = summary.detach().float().cpu().tolist()
    result = dict(zip(CRITIC_METRIC_NAMES, values[: len(CRITIC_METRIC_NAMES)]))
    result["critic_grad_norm"] = values[-1]
    result["cql_scale"] = float(cql_scale)
    return result


def critic_update(
    critics: IndependentCritics,
    optimizers: dict[str, torch.optim.Optimizer],
    batch: LearningBatch,
    config: LearningConfig,
    device: torch.device,
    *,
    cql_scale: float,
    oracle: OracleCritics | None = None,
    enable_oracle_distillation: bool = False,
) -> dict[str, float]:
    return resolve_critic_statistics(
        critic_update_deferred(
            critics,
            optimizers,
            batch,
            config,
            device,
            cql_scale=cql_scale,
            oracle=oracle,
            enable_oracle_distillation=enable_oracle_distillation,
        ),
        cql_scale=cql_scale,
    )


def _validate_mc_group_batch(batch: LearningBatch) -> None:
    if (
        batch.mc_query_ids is None
        or batch.mc_candidate_counts is None
        or batch.mc_reliable_actions is None
    ):
        raise ValueError("MC group update requires query metadata")
    if np.any(batch.sources != int(ReplaySource.MC_TEACHER)):
        raise ValueError("MC group update accepts only teacher rows")
    for query_id in np.unique(batch.mc_query_ids):
        if query_id < 0:
            raise ValueError("MC group update received a negative query id")
        rows = np.flatnonzero(batch.mc_query_ids == query_id)
        expected = np.unique(batch.mc_candidate_counts[rows])
        if len(expected) != 1 or int(expected[0]) != len(rows):
            raise ValueError("MC group update received an incomplete query")
        if len(np.unique(batch.actions[rows])) != len(rows):
            raise ValueError("MC group update requires unique candidate actions")
        if len(np.unique(batch.trajectory_ids[rows])) != 1:
            raise ValueError("MC query candidates must share one trajectory")
        if len(np.unique(batch.step_indices[rows])) != 1:
            raise ValueError("MC query candidates must share one replay step")
        actions = batch.actions[rows].astype(np.int64, copy=False)
        reliable = batch.mc_reliable_actions[rows]
        if np.any(reliable[np.arange(len(rows)), actions]):
            raise ValueError("MC reliable actions cannot contain self edges")
        candidates = np.zeros(115, dtype=np.bool_)
        candidates[actions] = True
        if np.any(reliable[:, ~candidates]):
            raise ValueError("MC reliable actions must stay within the query")
        query_edges = reliable[:, actions]
        if not np.array_equal(query_edges, query_edges.T):
            raise ValueError("MC reliable action relationships must be symmetric")
        if np.any(~query_edges.any(axis=1)):
            raise ValueError("every MC candidate requires a reliable action edge")


def mc_critic_update_deferred(
    critics: IndependentCritics,
    optimizer: torch.optim.Optimizer,
    batch: LearningBatch,
    config: LearningConfig,
    device: torch.device,
) -> Tensor:
    """Run one Q-only update over complete counterfactual action groups."""

    if not len(batch):
        raise ValueError("MC critic batch cannot be empty")
    _validate_mc_group_batch(batch)
    critics.train()
    optimizer.zero_grad(set_to_none=True)
    tensors = batch.tensors(device)
    state = (
        tensors["tile_obs"],
        tensors["melds"],
        tensors["meta"],
        tensors["events"],
        tensors["event_lengths"],
    )
    with _autocast(device):
        q1 = critics.q1(*state, tensors["legal"])
        q2 = critics.q2(*state, tensors["legal"])
        losses = mc_group_critic_loss(
            q1.raw_values,
            q2.raw_values,
            tensors["actions"],
            tensors["returns"],
            tensors["mc_query_ids"],
            tensors["mc_reliable_actions"],
            huber_delta=config.mc_centered_huber_delta,
            centered_scale=config.mc_centered_loss_scale,
            pairwise_scale=config.mc_pairwise_loss_scale,
            pairwise_temperature=config.mc_pairwise_temperature,
        )
    losses.loss.backward()
    parameters = [*critics.q1.parameters(), *critics.q2.parameters()]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    optimizer.step()
    return torch.stack(
        (
            losses.loss,
            losses.absolute_loss,
            losses.centered_loss,
            losses.pairwise_loss,
            losses.pairwise_accuracy,
            losses.group_count.float(),
            losses.pair_count.float(),
            grad_norm,
        )
    ).detach().float()


def resolve_mc_critic_statistics(summary: Tensor) -> dict[str, float]:
    values = summary.detach().float().cpu().tolist()
    result = dict(zip(MC_CRITIC_METRIC_NAMES, values[:-1]))
    result["mc_critic_grad_norm"] = values[-1]
    return result


def actor_update(
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    critics: IndependentCritics,
    optimizer: torch.optim.Optimizer,
    batch: LearningBatch,
    config: LearningConfig,
    device: torch.device,
    *,
    oracle: OracleCritics | None = None,
    use_oracle_teacher: bool = False,
    measure_post_update_kl: bool = True,
) -> dict[str, float]:
    if not len(batch):
        raise ValueError("actor batch cannot be empty")
    if np.any(batch.sources == int(ReplaySource.MC_TEACHER)):
        raise ValueError("MC teacher rows cannot be used as Actor labels")
    if use_oracle_teacher and (oracle is None or batch.oracle_tiles is None):
        raise ValueError("oracle Actor teacher requires oracle critics and inputs")
    actor.train()
    reference.eval()
    critics.eval()
    if oracle is not None:
        oracle.eval()
    optimizer.zero_grad(set_to_none=True)
    metric_rows: list[Tensor] = []
    # count, advantage sum, weight sum, squared-weight sum by decision category.
    category_totals = torch.zeros(
        (4, CATEGORY_COUNT), dtype=torch.float32, device=device
    )
    tensors = batch.tensors(device)
    microbatches = _tensor_microbatches(tensors, config.microbatch_size)
    for microbatch in microbatches:
        state = (
            microbatch["tile_obs"],
            microbatch["melds"],
            microbatch["meta"],
            microbatch["events"],
            microbatch["event_lengths"],
        )
        with _autocast(device):
            policy = actor(*state, microbatch["legal"])
            with torch.no_grad():
                reference_output = reference(*state, microbatch["legal"])
                if use_oracle_teacher:
                    teacher = oracle(
                        *state, microbatch["oracle_tiles"], microbatch["legal"]
                    )
                else:
                    teacher = critics(*state, microbatch["legal"])
            losses = awr_actor_loss(
                policy.raw_logits,
                microbatch["actions"],
                teacher.q1.raw_values,
                teacher.q2.raw_values,
                teacher.value,
                microbatch["legal"],
                reference_logits=reference_output.raw_logits,
                beta=config.awr_beta,
                max_weight=config.awr_max_weight,
                reference_kl_scale=config.reference_kl_scale,
            )
        (losses.loss / len(microbatches)).backward()
        metric_rows.append(
            torch.stack(
                (
                    losses.loss,
                    losses.policy_loss,
                    losses.reference_kl,
                    losses.advantages.mean(),
                    losses.weights.mean(),
                    losses.effective_sample_size,
                )
            )
            .detach()
            .float()
        )
        with torch.no_grad():
            categories = microbatch["categories"]
            advantages = losses.advantages.detach().float()
            weights = losses.weights.detach().float()
            category_totals[0].scatter_add_(
                0, categories, torch.ones_like(advantages)
            )
            category_totals[1].scatter_add_(0, categories, advantages)
            category_totals[2].scatter_add_(0, categories, weights)
            category_totals[3].scatter_add_(0, categories, weights.square())
    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
    optimizer.step()
    actor.eval()
    post_update_kl: Tensor | None = None
    if measure_post_update_kl:
        post_update_kl = torch.zeros((), dtype=torch.float32, device=device)
        with torch.no_grad():
            for microbatch in microbatches:
                state = (
                    microbatch["tile_obs"],
                    microbatch["melds"],
                    microbatch["meta"],
                    microbatch["events"],
                    microbatch["event_lengths"],
                )
                with _autocast(device):
                    actor_output = actor(*state, microbatch["legal"])
                    reference_output = reference(*state, microbatch["legal"])
                    measured = policy_reference_kl(
                        actor_output.raw_logits,
                        reference_output.raw_logits,
                        microbatch["legal"],
                    )
                post_update_kl.add_(
                    measured.float(),
                    alpha=len(microbatch["actions"]) / len(batch),
                )

    summary_parts = [
        torch.stack(metric_rows).mean(dim=0),
        grad_norm.detach().float().reshape(1),
        category_totals.flatten(),
    ]
    if post_update_kl is not None:
        summary_parts.append(post_update_kl.reshape(1))
    summary = torch.cat(summary_parts).cpu().tolist()
    metric_count = len(ACTOR_METRIC_NAMES)
    result = dict(zip(ACTOR_METRIC_NAMES, summary[:metric_count]))
    result["actor_grad_norm"] = summary[metric_count]
    category_values = np.asarray(
        summary[metric_count + 1 : metric_count + 1 + 4 * CATEGORY_COUNT],
        dtype=np.float64,
    ).reshape(4, CATEGORY_COUNT)
    if post_update_kl is not None:
        result["actor_reference_kl"] = summary[-1]
    for category, name in enumerate(CATEGORY_NAMES):
        count, advantage_sum, weight_sum, squared_weight_sum = category_values[
            :, category
        ]
        if count > 0:
            result[f"advantage_{name}"] = float(advantage_sum / count)
            result[f"weight_{name}"] = float(weight_sum / count)
            result[f"ess_{name}"] = float(
                weight_sum**2 / max(squared_weight_sum, 1e-12)
            )
        else:
            result[f"advantage_{name}"] = 0.0
            result[f"weight_{name}"] = 0.0
            result[f"ess_{name}"] = 0.0
    return result


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 2 or prediction.std() < 1e-8 or target.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def _calibration_error(prediction: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    if not len(prediction):
        return 0.0
    boundaries = np.unique(np.quantile(prediction, np.linspace(0, 1, bins + 1)))
    if len(boundaries) < 2:
        return float(abs(prediction.mean() - target.mean()))
    assignments = np.clip(np.digitize(prediction, boundaries[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(len(boundaries) - 1):
        selected = assignments == index
        if np.any(selected):
            error += float(selected.mean()) * abs(
                float(prediction[selected].mean() - target[selected].mean())
            )
    return error


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if not len(prediction):
        return {
            "count": 0.0,
            "loss": 0.0,
            "mae": 0.0,
            "correlation": 0.0,
            "calibration_error": 0.0,
            "constant_mae": 0.0,
            "improvement": 0.0,
        }
    mae = float(np.abs(prediction - target).mean())
    absolute_error = np.abs(prediction - target)
    loss = float(
        np.where(
            absolute_error <= 1.0,
            0.5 * np.square(absolute_error),
            absolute_error - 0.5,
        ).mean()
    )
    constant = float(np.abs(target - np.median(target)).mean())
    improvement = 0.0 if constant < 1e-8 else 1.0 - mae / constant
    return {
        "count": float(len(prediction)),
        "loss": loss,
        "mae": mae,
        "correlation": _correlation(prediction, target),
        "calibration_error": _calibration_error(prediction, target),
        "constant_mae": constant,
        "improvement": improvement,
    }


def _finite_metric(metrics: dict[str, object], name: str) -> float | None:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _expectile_balance_error(
    prediction: np.ndarray, values: np.ndarray, expectile: float
) -> float:
    if not len(prediction):
        return 0.0
    residual = prediction - values
    weights = np.where(residual < 0.0, 1.0 - expectile, expectile)
    weighted = weights * residual
    denominator = float(np.mean(np.abs(weighted)))
    if denominator < 1e-8:
        return 0.0
    return float(abs(np.mean(weighted)) / denominator)


def grouped_action_ranking_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    trajectory_ids: np.ndarray,
    step_indices: np.ndarray,
    actions: np.ndarray,
    reliable_actions: np.ndarray,
    *,
    query_ids: np.ndarray | None = None,
    expected_candidate_counts: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure action ordering only where the MC teacher marked a reliable edge."""

    arrays = (predictions, targets, trajectory_ids, step_indices, actions)
    if any(len(value) != len(predictions) for value in arrays):
        raise ValueError("ranking arrays must have the same length")
    if reliable_actions.shape != (len(predictions), 115):
        raise ValueError("reliable actions must have shape [rows, 115]")
    if reliable_actions.dtype != np.bool_:
        raise ValueError("reliable actions must be boolean")
    if (query_ids is None) != (expected_candidate_counts is None):
        raise ValueError("query ids and expected candidate counts must be paired")
    if query_ids is not None and (
        len(query_ids) != len(predictions)
        or len(expected_candidate_counts) != len(predictions)
    ):
        raise ValueError("MC query metadata must align with ranking rows")
    grouped: dict[
        tuple[int, ...], list[tuple[int, float, float, np.ndarray]]
    ] = defaultdict(list)
    expected: dict[tuple[int, ...], int] = {}
    for row, (prediction, target, trajectory_id, step_index, action) in enumerate(
        zip(*arrays)
    ):
        key = (
            (int(query_ids[row]),)
            if query_ids is not None
            else (int(trajectory_id), int(step_index))
        )
        grouped[key].append(
            (
                int(action),
                float(prediction),
                float(target),
                reliable_actions[row],
            )
        )
        if expected_candidate_counts is not None:
            candidate_count = int(expected_candidate_counts[row])
            previous = expected.setdefault(key, candidate_count)
            if previous != candidate_count:
                expected[key] = -1

    group_count = 0
    incomplete_group_count = 0
    all_pair_count = 0
    pair_count = 0
    correct_pairs = 0
    top_action_correct = 0
    regrets: list[float] = []
    action_gaps: list[float] = []
    for key, rows in grouped.items():
        # A state can be queried again in a later iteration. Average duplicate
        # estimates for one action so repeated queries do not overweight it.
        by_action: dict[int, list[tuple[float, float, np.ndarray]]] = defaultdict(list)
        for action, prediction, target, reliable in rows:
            by_action[action].append((prediction, target, reliable))
        estimates = [
            (
                action,
                float(np.mean([value[0] for value in values])),
                float(np.mean([value[1] for value in values])),
                np.logical_and.reduce([value[2] for value in values]),
            )
            for action, values in by_action.items()
        ]
        expected_count = expected.get(key)
        if expected_count is not None and (
            expected_count < 2
            or len(rows) != expected_count
            or len(estimates) != expected_count
        ):
            incomplete_group_count += 1
            continue
        if len(estimates) < 2:
            incomplete_group_count += 1
            continue
        all_pair_count += len(estimates) * (len(estimates) - 1) // 2
        candidate_actions = np.zeros(115, dtype=np.bool_)
        candidate_actions[[estimate[0] for estimate in estimates]] = True
        for action, _, _, reliable in estimates:
            if reliable[action]:
                raise ValueError("reliable actions cannot contain self edges")
            if np.any(reliable[~candidate_actions]):
                raise ValueError("reliable actions must stay within the group")
        reliable_pairs: list[tuple[int, int, float]] = []
        for left in range(len(estimates)):
            for right in range(left + 1, len(estimates)):
                target_gap = estimates[left][2] - estimates[right][2]
                left_action = estimates[left][0]
                right_action = estimates[right][0]
                left_marks_right = bool(estimates[left][3][right_action])
                right_marks_left = bool(estimates[right][3][left_action])
                if left_marks_right != right_marks_left:
                    raise ValueError("reliable action relationships must be symmetric")
                if not left_marks_right:
                    continue
                reliable_pairs.append((left, right, target_gap))
        if not reliable_pairs:
            continue
        group_count += 1
        participating = {
            index for left, right, _ in reliable_pairs for index in (left, right)
        }
        reliable_estimates = [estimates[index] for index in sorted(participating)]
        predicted_best = max(
            reliable_estimates, key=lambda value: (value[1], -value[0])
        )
        target_best = max(
            reliable_estimates, key=lambda value: (value[2], -value[0])
        )
        top_action_correct += int(predicted_best[0] == target_best[0])
        regrets.append(max(0.0, target_best[2] - predicted_best[2]))
        for left, right, target_gap in reliable_pairs:
            prediction_gap = estimates[left][1] - estimates[right][1]
            pair_count += 1
            correct_pairs += int(prediction_gap * target_gap > 0)
            action_gaps.append(abs(target_gap))
    return {
        "group_count": float(group_count),
        "incomplete_group_count": float(incomplete_group_count),
        "all_pair_count": float(all_pair_count),
        "pair_count": float(pair_count),
        "pairwise_accuracy": (
            float(correct_pairs / pair_count) if pair_count else 0.0
        ),
        "top_action_accuracy": (
            float(top_action_correct / group_count) if group_count else 0.0
        ),
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "maximum_regret": float(np.max(regrets)) if regrets else 0.0,
        "mean_action_gap": float(np.mean(action_gaps)) if action_gaps else 0.0,
    }


def _critic_diagnostics(
    q1: np.ndarray,
    q2: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    batch: LearningBatch,
    *,
    expectile: float,
) -> dict[str, object]:
    prediction = np.minimum(q1, q2)
    result: dict[str, object] = {
        "q": _metrics(prediction, target),
        "v": _metrics(values, target),
        "q_disagreement": float(np.abs(q1 - q2).mean()),
        "expectile_balance_error": _expectile_balance_error(
            prediction, values, expectile
        ),
        "progress": {},
        "categories": {},
    }
    wall = batch.meta[:, 4]
    progress_masks = {
        "early": wall >= 40,
        "middle": (wall >= 20) & (wall < 40),
        "late": wall < 20,
    }
    progress = result["progress"]
    assert isinstance(progress, dict)
    for name, selected in progress_masks.items():
        progress[name] = {
            "q": _metrics(prediction[selected], target[selected]),
            "v": _metrics(values[selected], target[selected]),
            "q_disagreement": float(
                np.abs(q1[selected] - q2[selected]).mean()
                if np.any(selected)
                else 0.0
            ),
            "expectile_balance_error": _expectile_balance_error(
                prediction[selected], values[selected], expectile
            ),
        }
    categories = result["categories"]
    assert isinstance(categories, dict)
    for category, name in enumerate(CATEGORY_NAMES):
        selected = batch.categories == category
        categories[name] = {
            "q": _metrics(prediction[selected], target[selected]),
            "q_disagreement": float(
                np.abs(q1[selected] - q2[selected]).mean()
                if np.any(selected)
                else 0.0
            ),
            "expectile_balance_error": _expectile_balance_error(
                prediction[selected], values[selected], expectile
            ),
        }
    return result


@torch.no_grad()
def validate_critics(
    critics: IndependentCritics,
    batch: LearningBatch,
    device: torch.device,
    *,
    microbatch_size: int = 256,
    oracle: OracleCritics | None = None,
    expectile: float = 0.7,
) -> dict[str, object]:
    if not len(batch):
        raise ValueError("validation batch cannot be empty")
    if not 0 < expectile < 1:
        raise ValueError("expectile must be in (0, 1)")
    critics.eval()
    if oracle is not None:
        oracle.eval()
    prediction_rows = 6 if oracle is not None else 3
    predictions = torch.empty(
        (prediction_rows, len(batch)), dtype=torch.float32, device=device
    )
    offset = 0
    tensors = batch.tensors(device)
    for microbatch in _tensor_microbatches(tensors, microbatch_size):
        state = (
            microbatch["tile_obs"],
            microbatch["melds"],
            microbatch["meta"],
            microbatch["events"],
            microbatch["event_lengths"],
        )
        with _autocast(device):
            output = critics(*state, microbatch["legal"])
            actions = microbatch["actions"][:, None]
            end = offset + len(microbatch["actions"])
            predictions[0, offset:end].copy_(
                output.q1.raw_values.gather(1, actions).squeeze(1)
            )
            predictions[1, offset:end].copy_(
                output.q2.raw_values.gather(1, actions).squeeze(1)
            )
            predictions[2, offset:end].copy_(output.value)
            if oracle is not None:
                if "oracle_tiles" not in microbatch:
                    raise ValueError("oracle validation requires oracle tile counts")
                oracle_output = oracle(
                    *state, microbatch["oracle_tiles"], microbatch["legal"]
                )
                predictions[3, offset:end].copy_(
                    oracle_output.q1.raw_values.gather(1, actions).squeeze(1)
                )
                predictions[4, offset:end].copy_(
                    oracle_output.q2.raw_values.gather(1, actions).squeeze(1)
                )
                predictions[5, offset:end].copy_(oracle_output.value)
            offset = end
    host_predictions = predictions.cpu().numpy()
    target = batch.returns.astype(np.float64)
    result = _critic_diagnostics(
        host_predictions[0],
        host_predictions[1],
        host_predictions[2],
        target,
        batch,
        expectile=expectile,
    )
    if oracle is not None:
        result["oracle"] = _critic_diagnostics(
            host_predictions[3],
            host_predictions[4],
            host_predictions[5],
            target,
            batch,
            expectile=expectile,
        )
        partial_q = result["q"]
        oracle_q = result["oracle"]["q"]
        assert isinstance(partial_q, dict)
        assert isinstance(oracle_q, dict)
        partial_progress = result["progress"]
        oracle_progress = result["oracle"]["progress"]
        assert isinstance(partial_progress, dict)
        assert isinstance(oracle_progress, dict)

        def relative_gain(partial: dict[str, float], privileged: dict[str, float]) -> float:
            denominator = float(partial.get("mae", 0.0))
            if denominator < 1e-8:
                return 0.0
            return float(1.0 - float(privileged.get("mae", 0.0)) / denominator)

        result["oracle_vs_partial"] = {
            "q_relative_mae_gain": relative_gain(partial_q, oracle_q),
            "progress": {
                stage: {
                    "q_relative_mae_gain": relative_gain(
                        partial_progress[stage]["q"], oracle_progress[stage]["q"]
                    )
                }
                for stage in ("early", "middle", "late")
            },
        }
    if batch.mc_reliable_actions is not None:
        if batch.mc_query_ids is None or batch.mc_candidate_counts is None:
            raise ValueError("action-ranking validation requires MC query metadata")
        result["action_ranking"] = grouped_action_ranking_metrics(
            np.minimum(host_predictions[0], host_predictions[1]),
            target,
            batch.trajectory_ids,
            batch.step_indices,
            batch.actions,
            batch.mc_reliable_actions,
            query_ids=batch.mc_query_ids,
            expected_candidate_counts=batch.mc_candidate_counts,
        )
    return result


def critic_ready(
    validation: dict[str, object],
    critic_steps: int,
    config: LearningConfig,
) -> tuple[bool, str]:
    if critic_steps < config.minimum_critic_steps:
        return False, "minimum_critic_steps"
    progress = validation.get("progress")
    if not isinstance(progress, dict):
        return False, "missing_progress_metrics"
    for stage in ("middle", "late"):
        metrics = progress.get(stage)
        if not isinstance(metrics, dict):
            return False, f"missing_{stage}_metrics"
        q = metrics.get("q")
        if not isinstance(q, dict):
            return False, f"missing_{stage}_q"
        count = _finite_metric(q, "count")
        improvement = _finite_metric(q, "improvement")
        correlation = _finite_metric(q, "correlation")
        if count is None or improvement is None or correlation is None:
            return False, f"nonfinite_{stage}_metrics"
        if count <= 0:
            return False, f"missing_{stage}_samples"
        if improvement < config.minimum_middle_late_improvement:
            return False, f"{stage}_constant_baseline"
        if correlation < config.minimum_middle_late_correlation:
            return False, f"{stage}_correlation"
    disagreement = _finite_metric(validation, "q_disagreement")
    if disagreement is None:
        return False, "nonfinite_q_disagreement"
    if disagreement > config.maximum_q_disagreement:
        return False, "q_disagreement"
    return True, "ready"


def oracle_teacher_ready(
    validation: dict[str, object], config: LearningConfig
) -> tuple[bool, str]:
    """Require the privileged Critic to prove it is a better teacher."""

    oracle = validation.get("oracle")
    if not isinstance(oracle, dict):
        return False, "missing_oracle_metrics"
    oracle_progress = oracle.get("progress")
    partial_progress = validation.get("progress")
    if not isinstance(oracle_progress, dict) or not isinstance(partial_progress, dict):
        return False, "missing_oracle_progress_metrics"
    for stage in ("middle", "late"):
        stage_metrics = oracle_progress.get(stage)
        if not isinstance(stage_metrics, dict):
            return False, f"missing_oracle_{stage}_metrics"
        q = stage_metrics.get("q")
        if not isinstance(q, dict):
            return False, f"missing_oracle_{stage}_q"
        count = _finite_metric(q, "count")
        improvement = _finite_metric(q, "improvement")
        correlation = _finite_metric(q, "correlation")
        if count is None or improvement is None or correlation is None:
            return False, f"nonfinite_oracle_{stage}_metrics"
        if count <= 0:
            return False, f"missing_oracle_{stage}_samples"
        if improvement < config.minimum_middle_late_improvement:
            return False, f"oracle_{stage}_constant_baseline"
        if correlation < config.minimum_middle_late_correlation:
            return False, f"oracle_{stage}_correlation"

    oracle_early = oracle_progress.get("early")
    partial_early = partial_progress.get("early")
    if not isinstance(oracle_early, dict) or not isinstance(partial_early, dict):
        return False, "missing_oracle_early_metrics"
    oracle_early_q = oracle_early.get("q")
    partial_early_q = partial_early.get("q")
    if not isinstance(oracle_early_q, dict) or not isinstance(partial_early_q, dict):
        return False, "missing_oracle_early_q"
    early_count = _finite_metric(oracle_early_q, "count")
    early_improvement = _finite_metric(oracle_early_q, "improvement")
    early_correlation = _finite_metric(oracle_early_q, "correlation")
    if (
        early_count is None
        or early_improvement is None
        or early_correlation is None
    ):
        return False, "nonfinite_oracle_early_metrics"
    if early_count <= 0:
        return False, "missing_oracle_early_samples"
    if early_improvement < config.minimum_oracle_early_improvement:
        return False, "oracle_early_constant_baseline"
    if early_correlation < config.minimum_oracle_early_correlation:
        return False, "oracle_early_correlation"

    # An opening-state V near zero can be correctly calibrated while having
    # little correlation with noisy terminal samples. Gate V only where the
    # realized situation signal is informative; early action quality is
    # covered separately by Oracle Q and relative MAE checks in this gate.
    for stage in ("middle", "late"):
        oracle_stage = oracle_progress.get(stage)
        if not isinstance(oracle_stage, dict):
            return False, f"missing_oracle_{stage}_metrics"
        value_metrics = oracle_stage.get("v")
        if not isinstance(value_metrics, dict):
            return False, f"missing_oracle_{stage}_value_metrics"
        value_correlation = _finite_metric(value_metrics, "correlation")
        if value_correlation is None:
            return False, f"nonfinite_oracle_{stage}_value_correlation"
        if value_correlation < config.minimum_oracle_value_correlation:
            return False, f"oracle_{stage}_value_correlation"

    oracle_q = oracle.get("q")
    partial_q = validation.get("q")
    if not isinstance(oracle_q, dict) or not isinstance(partial_q, dict):
        return False, "missing_oracle_q"
    oracle_vs_partial = validation.get("oracle_vs_partial")
    if not isinstance(oracle_vs_partial, dict):
        return False, "missing_oracle_comparison"
    overall_gain = _finite_metric(oracle_vs_partial, "q_relative_mae_gain")
    if overall_gain is None:
        return False, "nonfinite_oracle_overall_mae_gain"
    if overall_gain < config.minimum_oracle_relative_mae_gain:
        return False, "oracle_overall_mae_gain"
    comparison_progress = oracle_vs_partial.get("progress")
    if not isinstance(comparison_progress, dict):
        return False, "missing_oracle_comparison_progress"
    early_comparison = comparison_progress.get("early")
    if not isinstance(early_comparison, dict):
        return False, "missing_oracle_early_comparison"
    early_gain = _finite_metric(early_comparison, "q_relative_mae_gain")
    if early_gain is None:
        return False, "nonfinite_oracle_early_mae_gain"
    if early_gain < config.minimum_oracle_early_relative_mae_gain:
        return False, "oracle_early_mae_gain"
    disagreement = _finite_metric(oracle, "q_disagreement")
    if disagreement is None:
        return False, "nonfinite_oracle_q_disagreement"
    if disagreement > config.maximum_oracle_q_disagreement:
        return False, "oracle_q_disagreement"
    balance = _finite_metric(oracle, "expectile_balance_error")
    if balance is None:
        return False, "nonfinite_oracle_expectile_balance"
    if balance > config.maximum_oracle_expectile_balance_error:
        return False, "oracle_expectile_balance"
    return True, "ready"


def mc_teacher_ready(
    validation: dict[str, object] | None,
    *,
    train_targets: int,
    validation_targets: int,
    config: LearningConfig,
) -> tuple[bool, str]:
    """Gate Actor extraction on accumulated action-level MC evidence."""

    if train_targets < config.minimum_mc_train_targets:
        return False, "mc_train_targets"
    if validation_targets < config.minimum_mc_validation_targets:
        return False, "mc_validation_targets"
    if not isinstance(validation, dict):
        return False, "missing_mc_validation"
    ranking = validation.get("action_ranking")
    if not isinstance(ranking, dict):
        return False, "missing_mc_action_ranking"
    group_count = _finite_metric(ranking, "group_count")
    pair_count = _finite_metric(ranking, "pair_count")
    pairwise_accuracy = _finite_metric(ranking, "pairwise_accuracy")
    mean_regret = _finite_metric(ranking, "mean_regret")
    if (
        group_count is None
        or pair_count is None
        or pairwise_accuracy is None
        or mean_regret is None
    ):
        return False, "nonfinite_mc_action_ranking"
    if group_count < config.minimum_mc_validation_groups:
        return False, "mc_validation_groups"
    if pair_count < config.minimum_mc_pairwise_pairs:
        return False, "mc_pairwise_pairs"
    if pairwise_accuracy < config.minimum_mc_pairwise_accuracy:
        return False, "mc_pairwise_accuracy"
    if mean_regret > config.maximum_mc_mean_regret:
        return False, "mc_mean_regret"
    return True, "ready"


def resolve_actor_gate(
    experiment: str,
    validation: dict[str, object],
    critic_steps: int,
    config: LearningConfig,
    previous_teacher_streak: int,
    *,
    mc_validation: dict[str, object] | None = None,
    mc_train_targets: int = 0,
    mc_validation_targets: int = 0,
) -> tuple[bool, str, int]:
    """Resolve the base Critic gate and experiment-specific teacher gate."""

    if experiment not in ("a", "b", "c"):
        raise ValueError(f"unknown experiment {experiment!r}")
    if previous_teacher_streak < 0:
        raise ValueError("teacher readiness streak cannot be negative")
    base_ready, base_reason = critic_ready(validation, critic_steps, config)
    if not base_ready:
        return False, base_reason, 0
    if experiment == "a":
        return True, "ready", 0
    if experiment == "b":
        candidate_ready, reason = oracle_teacher_ready(validation, config)
        teacher = "oracle"
    else:
        candidate_ready, reason = mc_teacher_ready(
            mc_validation,
            train_targets=mc_train_targets,
            validation_targets=mc_validation_targets,
            config=config,
        )
        teacher = "mc"
    if not candidate_ready:
        return False, reason, 0
    streak = min(previous_teacher_streak + 1, config.teacher_readiness_streak)
    if streak < config.teacher_readiness_streak:
        return (
            False,
            f"{teacher}_readiness_streak:{streak}/{config.teacher_readiness_streak}",
            streak,
        )
    return True, f"ready_{teacher}", streak
