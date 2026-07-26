"""Training-only perfect-information critics.

The Actor never imports or owns these modules.  Oracle tile counts are only
used to measure how much critic error comes from partial observability and,
optionally, to provide a lower-variance advantage teacher over replay samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .iql import ActionValueOutput, CriticConfig
from .model import (
    ACTION_SPACE_SIZE,
    HistoryEncoder,
    RMSNorm,
    StaticStateEncoder,
    TILE_KIND_COUNT,
)

ORACLE_TILE_PLANES = 9


@dataclass(frozen=True)
class OracleCriticOutput:
    q1: ActionValueOutput
    q2: ActionValueOutput
    value: Tensor


class OracleStateEncoder(nn.Module):
    """Independent viewer-state encoder augmented with hidden tile counts."""

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.static = StaticStateEncoder(config)
        self.history = HistoryEncoder(config)
        self.oracle = nn.Sequential(
            nn.Linear(ORACLE_TILE_PLANES * TILE_KIND_COUNT, config.d_model),
            nn.SiLU(),
            RMSNorm(config.d_model),
        )

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        oracle_tiles: Tensor,
    ) -> Tensor:
        if oracle_tiles.shape != (
            tile_obs.shape[0],
            ORACLE_TILE_PLANES,
            TILE_KIND_COUNT,
        ):
            raise ValueError("oracle_tiles must have shape [batch, 9, 27]")
        static = self.static(tile_obs, melds, meta)
        history = self.history(events, event_lengths)
        hidden = self.oracle(
            oracle_tiles.to(self.oracle[0].weight.dtype).flatten(1) / 4.0
        )
        return torch.cat((static, history, hidden), dim=-1)


class _OracleQ(nn.Module):
    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.encoder = OracleStateEncoder(config)
        width = config.d_model * 3
        self.head = nn.Sequential(
            RMSNorm(width),
            nn.Linear(width, config.head_dim),
            nn.SiLU(),
            nn.Linear(config.head_dim, ACTION_SPACE_SIZE),
        )

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        oracle_tiles: Tensor,
        legal_mask: Tensor,
    ) -> ActionValueOutput:
        raw = self.head(
            self.encoder(
                tile_obs,
                melds,
                meta,
                events,
                event_lengths,
                oracle_tiles,
            )
        )
        if legal_mask.shape != raw.shape:
            raise ValueError("legal_mask must have shape [batch, 115]")
        legal = legal_mask.to(device=raw.device, dtype=torch.bool)
        torch._assert_async(legal.any(dim=-1).all(), "state has no legal action")
        return ActionValueOutput(
            values=raw.masked_fill(~legal, torch.finfo(raw.dtype).min),
            raw_values=raw,
        )


class _OracleV(nn.Module):
    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.encoder = OracleStateEncoder(config)
        width = config.d_model * 3
        self.head = nn.Sequential(
            RMSNorm(width),
            nn.Linear(width, config.head_dim),
            nn.SiLU(),
            nn.Linear(config.head_dim, 1),
        )

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        oracle_tiles: Tensor,
    ) -> Tensor:
        encoded = self.encoder(
            tile_obs, melds, meta, events, event_lengths, oracle_tiles
        )
        return self.head(encoded).squeeze(-1)


class OracleCritics(nn.Module):
    """Perfect-information Q1/Q2/V with no parameter shared with Actor."""

    def __init__(self, config: CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or CriticConfig()
        self.q1 = _OracleQ(self.config)
        self.q2 = _OracleQ(self.config)
        self.v = _OracleV(self.config)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        oracle_tiles: Tensor,
        legal_mask: Tensor,
    ) -> OracleCriticOutput:
        state = (tile_obs, melds, meta, events, event_lengths, oracle_tiles)
        return OracleCriticOutput(
            q1=self.q1(*state, legal_mask),
            q2=self.q2(*state, legal_mask),
            value=self.v(*state),
        )


def distillation_loss(
    student_q1: Tensor,
    student_q2: Tensor,
    teacher_q1: Tensor,
    teacher_q2: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
) -> Tensor:
    """Distill only Oracle values covered by logged-action validation."""

    if not (
        student_q1.shape
        == student_q2.shape
        == teacher_q1.shape
        == teacher_q2.shape
        == legal_mask.shape
    ):
        raise ValueError("Q tensors and legal_mask must have identical shapes")
    legal = legal_mask.bool()
    if actions.shape != (student_q1.shape[0],):
        raise ValueError("actions must have shape [batch]")
    actions = actions.to(device=student_q1.device, dtype=torch.long)
    torch._assert_async(
        legal.gather(1, actions[:, None]).all(),
        "distillation action must be legal",
    )
    target = torch.minimum(teacher_q1, teacher_q2).detach()
    student = torch.minimum(student_q1, student_q2)
    return torch.nn.functional.huber_loss(
        student.gather(1, actions[:, None]).squeeze(1),
        target.gather(1, actions[:, None]).squeeze(1),
    )
