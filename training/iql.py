"""Independent visible-state critics and IQL/AWR objectives.

The Actor intentionally does not share parameters with these networks.  Q1,
Q2, and V each encode the acting player's observation independently so a
critic update cannot overwrite either the SL policy or another critic's
features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import (
    ACTION_SPACE_SIZE,
    HistoryEncoder,
    RMSNorm,
    StaticStateEncoder,
    TransformerConfig,
)

Reduction = Literal["none", "mean", "sum"]

__all__ = [
    "AWRLossOutput",
    "ActionValueNetwork",
    "ActionValueOutput",
    "CriticConfig",
    "CriticLossOutput",
    "CriticOutput",
    "CriticStateEncoder",
    "IndependentCritics",
    "StateValueNetwork",
    "awr_actor_loss",
    "awr_weights",
    "conservative_advantages",
    "critic_losses",
    "double_q_huber_loss",
    "expectile_loss",
    "gather_action_values",
    "legal_cql_loss",
    "policy_reference_kl",
]


@dataclass(frozen=True)
class CriticConfig(TransformerConfig):
    """A compact Transformer configuration used by each independent critic."""

    d_model: int = 128
    num_heads: int = 4
    static_layers: int = 2
    history_layers: int = 3
    ffn_dim: int = 384
    head_dim: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")


@dataclass(frozen=True)
class ActionValueOutput:
    """Action values before and after applying the legal-action mask."""

    values: Tensor
    raw_values: Tensor


@dataclass(frozen=True)
class CriticOutput:
    q1: ActionValueOutput
    q2: ActionValueOutput
    value: Tensor


@dataclass(frozen=True)
class CriticLossOutput:
    """Loss components kept separate for independent optimizer schedules."""

    q_loss: Tensor
    q1_regression: Tensor
    q2_regression: Tensor
    q1_cql: Tensor
    q2_cql: Tensor
    value_loss: Tensor
    value_target: Tensor


@dataclass(frozen=True)
class AWRLossOutput:
    loss: Tensor
    policy_loss: Tensor
    reference_kl: Tensor
    advantages: Tensor
    weights: Tensor
    effective_sample_size: Tensor


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


class CriticStateEncoder(nn.Module):
    """Encode only information visible to the player taking the action."""

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.static_encoder = StaticStateEncoder(config)
        self.history_encoder = HistoryEncoder(config)

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
    ) -> Tensor:
        static = self.static_encoder(tile_obs, melds, meta)
        history = self.history_encoder(events, event_lengths)
        return torch.cat((static, history), dim=-1)


class ActionValueNetwork(nn.Module):
    """Return Q(s, a) for all actions, masking actions illegal in s."""

    def __init__(self, config: CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or CriticConfig()
        fused = self.config.d_model * 2
        self.encoder = CriticStateEncoder(self.config)
        self.head = nn.Sequential(
            RMSNorm(fused),
            nn.Linear(fused, self.config.head_dim),
            nn.SiLU(),
            nn.Linear(self.config.head_dim, ACTION_SPACE_SIZE),
        )
        self.apply(_initialize)

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        legal_mask: Tensor,
    ) -> ActionValueOutput:
        encoded = self.encoder(tile_obs, melds, meta, events, event_lengths)
        raw_values = self.head(encoded)
        legal_mask = _validate_legal_mask(legal_mask, raw_values)
        values = raw_values.masked_fill(
            ~legal_mask, torch.finfo(raw_values.dtype).min
        )
        return ActionValueOutput(values=values, raw_values=raw_values)


class StateValueNetwork(nn.Module):
    """Estimate V(s) from the same deployable partial observation as Actor."""

    def __init__(self, config: CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or CriticConfig()
        fused = self.config.d_model * 2
        self.encoder = CriticStateEncoder(self.config)
        self.head = nn.Sequential(
            RMSNorm(fused),
            nn.Linear(fused, self.config.head_dim),
            nn.SiLU(),
            nn.Linear(self.config.head_dim, 1),
        )
        self.apply(_initialize)

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
    ) -> Tensor:
        encoded = self.encoder(tile_obs, melds, meta, events, event_lengths)
        return self.head(encoded).squeeze(-1)


class IndependentCritics(nn.Module):
    """Container for independently parameterized Q1, Q2, and V networks."""

    def __init__(self, config: CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or CriticConfig()
        self.q1 = ActionValueNetwork(self.config)
        self.q2 = ActionValueNetwork(self.config)
        self.v = StateValueNetwork(self.config)

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        legal_mask: Tensor,
    ) -> CriticOutput:
        inputs = (tile_obs, melds, meta, events, event_lengths)
        return CriticOutput(
            q1=self.q1(*inputs, legal_mask),
            q2=self.q2(*inputs, legal_mask),
            value=self.v(*inputs),
        )


def gather_action_values(action_values: Tensor, actions: Tensor) -> Tensor:
    """Gather one behavior-action value per batch row."""

    _validate_action_values(action_values)
    if actions.shape != (action_values.shape[0],):
        raise ValueError("actions must have shape [batch]")
    actions = actions.to(device=action_values.device, dtype=torch.long)
    torch._assert_async(
        ((actions >= 0) & (actions < action_values.shape[1])).all(),
        "actions contain an out-of-range index",
    )
    return action_values.gather(1, actions[:, None]).squeeze(1)


def double_q_huber_loss(
    q1_values: Tensor,
    q2_values: Tensor,
    actions: Tensor,
    returns: Tensor,
    *,
    delta: float = 1.0,
    reduction: Reduction = "mean",
) -> tuple[Tensor, Tensor]:
    """Huber regression of both Q networks to Monte-Carlo return-to-go."""

    if delta <= 0:
        raise ValueError("delta must be positive")
    if q1_values.shape != q2_values.shape:
        raise ValueError("q1_values and q2_values must have the same shape")
    if returns.shape != (q1_values.shape[0],):
        raise ValueError("returns must have shape [batch]")
    targets = returns.to(device=q1_values.device, dtype=q1_values.dtype).detach()
    q1 = gather_action_values(q1_values, actions)
    q2 = gather_action_values(q2_values, actions)
    return (
        F.huber_loss(q1, targets, delta=delta, reduction=reduction),
        F.huber_loss(q2, targets, delta=delta, reduction=reduction),
    )


def expectile_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    expectile: float = 0.7,
    reduction: Reduction = "mean",
) -> Tensor:
    """Asymmetric squared loss used to fit V below the high-value actions."""

    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be between zero and one")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    residual = target.detach() - prediction
    weights = torch.where(residual < 0, 1.0 - expectile, expectile)
    return _reduce(weights * residual.square(), reduction)


def legal_cql_loss(
    action_values: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Conservative Q penalty over legal actions only."""

    _validate_action_values(action_values)
    legal_mask = _validate_legal_mask(legal_mask, action_values)
    actions = actions.to(device=action_values.device, dtype=torch.long)
    behavior_values = gather_action_values(action_values, actions)
    behavior_is_legal = legal_mask.gather(1, actions[:, None]).squeeze(1)
    torch._assert_async(
        behavior_is_legal.all(), "behavior action must be legal in its state"
    )
    legal_values = action_values.masked_fill(~legal_mask, -torch.inf)
    penalties = torch.logsumexp(legal_values, dim=-1) - behavior_values
    return _reduce(penalties, reduction)


def critic_losses(
    q1_values: Tensor,
    q2_values: Tensor,
    values: Tensor,
    actions: Tensor,
    returns: Tensor,
    legal_mask: Tensor,
    *,
    expectile: float = 0.7,
    huber_delta: float = 1.0,
    cql_scale: float = 0.0,
    cql_sample_mask: Tensor | None = None,
    value_sample_mask: Tensor | None = None,
) -> CriticLossOutput:
    """Compute independent Double-Q regression, CQL, and expectile-V losses."""

    if cql_scale < 0:
        raise ValueError("cql_scale must be non-negative")
    if values.shape != (q1_values.shape[0],):
        raise ValueError("values must have shape [batch]")
    q1_regression_rows, q2_regression_rows = double_q_huber_loss(
        q1_values,
        q2_values,
        actions,
        returns,
        delta=huber_delta,
        reduction="none",
    )
    q1_regression = q1_regression_rows.mean()
    q2_regression = q2_regression_rows.mean()
    q1_cql = _sample_mean(
        legal_cql_loss(q1_values, legal_mask, actions, reduction="none"),
        cql_sample_mask,
    )
    q2_cql = _sample_mean(
        legal_cql_loss(q2_values, legal_mask, actions, reduction="none"),
        cql_sample_mask,
    )
    q1_taken = gather_action_values(q1_values, actions)
    q2_taken = gather_action_values(q2_values, actions)
    value_target = torch.minimum(q1_taken, q2_taken).detach()
    value_loss = _sample_mean(
        expectile_loss(values, value_target, expectile=expectile, reduction="none"),
        value_sample_mask,
    )
    q_loss = q1_regression + q2_regression + cql_scale * (q1_cql + q2_cql)
    return CriticLossOutput(
        q_loss=q_loss,
        q1_regression=q1_regression,
        q2_regression=q2_regression,
        q1_cql=q1_cql,
        q2_cql=q2_cql,
        value_loss=value_loss,
        value_target=value_target,
    )


def conservative_advantages(
    q1_values: Tensor,
    q2_values: Tensor,
    values: Tensor,
    actions: Tensor,
) -> Tensor:
    """Compute detached min(Q1, Q2) - V advantages for Actor extraction."""

    if q1_values.shape != q2_values.shape:
        raise ValueError("q1_values and q2_values must have the same shape")
    if values.shape != (q1_values.shape[0],):
        raise ValueError("values must have shape [batch]")
    q1 = gather_action_values(q1_values, actions)
    q2 = gather_action_values(q2_values, actions)
    return (torch.minimum(q1, q2) - values).detach()


def awr_weights(
    advantages: Tensor,
    *,
    beta: float = 0.1,
    max_weight: float = 20.0,
) -> Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    maximum_log_weight = math.log(max_weight)
    return torch.exp(
        (advantages.detach() / beta).clamp(max=maximum_log_weight)
    ).clamp(max=max_weight)


def awr_actor_loss(
    actor_logits: Tensor,
    actions: Tensor,
    q1_values: Tensor,
    q2_values: Tensor,
    values: Tensor,
    legal_mask: Tensor,
    *,
    reference_logits: Tensor | None = None,
    beta: float = 0.1,
    max_weight: float = 20.0,
    reference_kl_scale: float = 0.0,
) -> AWRLossOutput:
    """Advantage-weighted behavior cloning with an optional frozen-SL KL."""

    _validate_action_values(actor_logits)
    if q1_values.shape != actor_logits.shape or q2_values.shape != actor_logits.shape:
        raise ValueError("Q values and Actor logits must have the same shape")
    if reference_kl_scale < 0:
        raise ValueError("reference_kl_scale must be non-negative")
    legal_mask = _validate_legal_mask(legal_mask, actor_logits)
    actions = actions.to(device=actor_logits.device, dtype=torch.long)
    behavior_is_legal = legal_mask.gather(1, actions[:, None]).squeeze(1)
    torch._assert_async(
        behavior_is_legal.all(), "behavior action must be legal in its state"
    )

    advantages = conservative_advantages(q1_values, q2_values, values, actions)
    weights = awr_weights(advantages, beta=beta, max_weight=max_weight)
    actor_log_probabilities = _masked_log_softmax(actor_logits, legal_mask)
    behavior_log_probabilities = gather_action_values(
        actor_log_probabilities, actions
    )
    policy_loss = -(weights * behavior_log_probabilities).mean()

    reference_kl = actor_logits.new_zeros(())
    if reference_logits is not None:
        reference_kl = policy_reference_kl(
            actor_logits, reference_logits, legal_mask
        )

    weight_sum = weights.sum()
    effective_sample_size = weight_sum.square() / weights.square().sum().clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    loss = policy_loss + reference_kl_scale * reference_kl
    return AWRLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        advantages=advantages,
        weights=weights,
        effective_sample_size=effective_sample_size,
    )


def policy_reference_kl(
    actor_logits: Tensor,
    reference_logits: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    """Mean legal-action KL(pi_actor || pi_reference)."""

    _validate_action_values(actor_logits)
    if reference_logits.shape != actor_logits.shape:
        raise ValueError("reference_logits must have shape [batch, 115]")
    legal_mask = _validate_legal_mask(legal_mask, actor_logits)
    actor_logs = _masked_log_softmax(actor_logits, legal_mask)
    reference_logs = _masked_log_softmax(reference_logits.detach(), legal_mask)
    actor_probabilities = actor_logs.exp().masked_fill(~legal_mask, 0.0)
    return (
        actor_probabilities
        * (
            actor_logs.masked_fill(~legal_mask, 0.0)
            - reference_logs.masked_fill(~legal_mask, 0.0)
        )
    ).sum(dim=-1).mean()


def _validate_action_values(action_values: Tensor) -> None:
    if action_values.ndim != 2 or action_values.shape[1] != ACTION_SPACE_SIZE:
        raise ValueError("action values must have shape [batch, 115]")


def _validate_legal_mask(legal_mask: Tensor, values: Tensor) -> Tensor:
    _validate_action_values(values)
    if legal_mask.shape != values.shape:
        raise ValueError("legal_mask must have shape [batch, 115]")
    legal_mask = legal_mask.to(device=values.device, dtype=torch.bool)
    torch._assert_async(
        legal_mask.any(dim=-1).all(), "every state must have at least one legal action"
    )
    return legal_mask


def _masked_log_softmax(logits: Tensor, legal_mask: Tensor) -> Tensor:
    masked_logits = logits.float().masked_fill(~legal_mask, -torch.inf)
    return F.log_softmax(masked_logits, dim=-1)


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def _sample_mean(values: Tensor, sample_mask: Tensor | None) -> Tensor:
    if sample_mask is None:
        return values.mean()
    if sample_mask.shape != values.shape:
        raise ValueError("sample mask must have shape [batch]")
    weights = sample_mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)
