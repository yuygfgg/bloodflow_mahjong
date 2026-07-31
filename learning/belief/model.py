"""Factorized residual scorer for planner belief particles."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TILE_KIND_COUNT = 27
TILE_OBSERVATION_PLANES = 10
PLAYER_COUNT = 4
MELD_SLOTS = 4
MELD_FIELDS = 3
META_WIDTH = 34
EVENT_RECORD_WIDTH = 8
BELIEF_WORLD_PLANES = 4


@dataclass(frozen=True)
class BeliefModelConfig:
    d_model: int = 128
    num_heads: int = 4
    static_layers: int = 2
    history_layers: int = 3
    ffn_dim: int = 384
    world_hidden: int = 512
    interaction_width: int = 256
    max_history: int = 192
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.static_layers <= 0 or self.history_layers <= 0:
            raise ValueError("encoder depths must be positive")
        if self.max_history != 192:
            raise ValueError("belief schema fixes max_history at 192")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        scale = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(scale + self.eps).to(inputs.dtype)
        return normalized * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.up(inputs).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class SelfAttention(nn.Module):
    def __init__(self, config: BeliefModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        inverse = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inverse = 1.0 / (config.rope_theta ** (inverse / self.head_dim))
        positions = torch.arange(config.max_history, dtype=torch.float32)
        angles = positions[:, None] * inverse[None, :]
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _apply_rope(self, tensor: Tensor, positions: Tensor) -> Tensor:
        cosine = self.rope_cos.index_select(0, positions).to(tensor.dtype)
        sine = self.rope_sin.index_select(0, positions).to(tensor.dtype)
        cosine = cosine[None, None]
        sine = sine[None, None]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def forward(
        self,
        inputs: Tensor,
        *,
        causal: bool,
        key_valid: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if positions is not None:
            query = self._apply_rope(query, positions)
            key = self._apply_rope(key, positions)

        length = inputs.shape[1]
        attention_mask: Tensor | None = None
        native_causal = causal and key_valid is None
        if causal and not native_causal:
            attention_mask = torch.ones(
                (length, length), device=inputs.device, dtype=torch.bool
            ).tril()[None, None]
        if key_valid is not None:
            valid = key_valid[:, None, None, :]
            attention_mask = valid if attention_mask is None else attention_mask & valid
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=native_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(inputs.shape)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: BeliefModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim)

    def forward(
        self,
        inputs: Tensor,
        *,
        causal: bool,
        key_valid: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        hidden = inputs + self.attention(
            self.attention_norm(inputs),
            causal=causal,
            key_valid=key_valid,
            positions=positions,
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


def _optional_index(values: Tensor, missing_index: int, maximum: int) -> Tensor:
    values = values.long()
    return torch.where(values < 0, missing_index, values.clamp(max=maximum))


class StaticStateEncoder(nn.Module):
    def __init__(self, config: BeliefModelConfig) -> None:
        super().__init__()
        width = config.d_model
        self.count_embeddings = nn.ModuleList(
            nn.Embedding(5, width) for _ in range(TILE_OBSERVATION_PLANES)
        )
        self.tile_embedding = nn.Embedding(TILE_KIND_COUNT, width)
        self.suit_embedding = nn.Embedding(3, width)
        self.rank_embedding = nn.Embedding(9, width)
        self.binary_embedding = nn.Embedding(2, width)
        self.phase_embedding = nn.Embedding(6, width)
        self.direction_embedding = nn.Embedding(4, width)
        self.optional_tile_embedding = nn.Embedding(TILE_KIND_COUNT + 1, width)
        self.optional_seat_embedding = nn.Embedding(PLAYER_COUNT + 1, width)
        self.optional_suit_embedding = nn.Embedding(4, width)
        self.reaction_embedding = nn.Embedding(8, width)
        self.global_numeric = nn.Linear(3, width, bias=False)
        self.seat_embedding = nn.Embedding(PLAYER_COUNT, width)
        self.player_numeric = nn.Linear(3, width, bias=False)
        self.meld_kind_embedding = nn.Embedding(5, width)
        self.meld_slot_embedding = nn.Embedding(MELD_SLOTS, width)
        self.token_type_embedding = nn.Embedding(4, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.static_layers)
        )
        self.output_norm = RMSNorm(width)

    def _global_token(self, meta: Tensor) -> Tensor:
        numeric = torch.stack(
            (
                meta[:, 4].float() / 55.0,
                meta[:, 9].float() / 108.0,
                meta[:, 10].float() / 3.0,
            ),
            dim=-1,
        )
        return (
            self.phase_embedding(meta[:, 0].long().clamp(0, 5))
            + self.direction_embedding(meta[:, 3].long().clamp(0, 3))
            + self.optional_tile_embedding(
                _optional_index(meta[:, 5], TILE_KIND_COUNT, TILE_KIND_COUNT - 1)
            )
            + self.optional_seat_embedding(
                _optional_index(meta[:, 7], PLAYER_COUNT, PLAYER_COUNT - 1)
            )
            + self.optional_tile_embedding(
                _optional_index(meta[:, 8], TILE_KIND_COUNT, TILE_KIND_COUNT - 1)
            )
            + self.optional_suit_embedding(_optional_index(meta[:, 11], 3, 2))
            + self.reaction_embedding(meta[:, 29].long().clamp(0, 7))
            + self.binary_embedding(meta[:, 6].long().clamp(0, 1))
            + self.binary_embedding(meta[:, 28].long().clamp(0, 1))
            + self.global_numeric(numeric.to(self.global_numeric.weight.dtype))
            + self.token_type_embedding.weight[0]
        )

    def _tile_tokens(self, tile_obs: Tensor, meta: Tensor) -> Tensor:
        batch = tile_obs.shape[0]
        tile_ids = torch.arange(TILE_KIND_COUNT, device=tile_obs.device)
        tokens = (
            self.tile_embedding(tile_ids)
            + self.suit_embedding(tile_ids // 9)
            + self.rank_embedding(tile_ids % 9)
        ).unsqueeze(0).expand(batch, -1, -1)
        counts = tile_obs.long().clamp(0, 4)
        for plane, embedding in enumerate(self.count_embeddings):
            tokens = tokens + embedding(counts[:, plane])
        draw = meta[:, 5, None].long() == tile_ids[None]
        response = meta[:, 8, None].long() == tile_ids[None]
        missing = meta[:, 16, None].long() == tile_ids[None] // 9
        return (
            tokens
            + self.binary_embedding(draw.long())
            + self.binary_embedding(response.long())
            + self.binary_embedding(missing.long())
            + self.token_type_embedding.weight[1]
        )

    def _player_tokens(self, meta: Tensor) -> Tensor:
        batch = meta.shape[0]
        seats = torch.arange(PLAYER_COUNT, device=meta.device)
        seat_rows = seats.unsqueeze(0).expand(batch, -1)
        dealer = meta[:, 2, None].long() == seat_rows
        maximum = meta[:, 30:34].float().clamp_min(0)
        numeric = torch.stack(
            (
                meta[:, 12:16].float() / 10_000.0,
                meta[:, 24:28].float() / 18.0,
                torch.log2(1.0 + maximum) / 8.0,
            ),
            dim=-1,
        )
        return (
            self.seat_embedding(seat_rows)
            + self.binary_embedding(dealer.long())
            + self.optional_suit_embedding(_optional_index(meta[:, 16:20], 3, 2))
            + self.binary_embedding(meta[:, 20:24].long().clamp(0, 1))
            + self.player_numeric(numeric.to(self.player_numeric.weight.dtype))
            + self.token_type_embedding.weight[2]
        )

    def _meld_tokens(self, melds: Tensor) -> tuple[Tensor, Tensor]:
        batch = melds.shape[0]
        flat = melds.reshape(batch, PLAYER_COUNT * MELD_SLOTS, MELD_FIELDS)
        padding = flat[:, :, 0] == 255
        tile = torch.where(padding, TILE_KIND_COUNT, flat[:, :, 0].long().clamp(0, 26))
        kind = torch.where(padding, 4, flat[:, :, 1].long().clamp(0, 3))
        source = torch.where(padding, PLAYER_COUNT, flat[:, :, 2].long().clamp(0, 3))
        owners = torch.arange(PLAYER_COUNT, device=melds.device).repeat_interleave(MELD_SLOTS)
        slots = torch.arange(MELD_SLOTS, device=melds.device).repeat(PLAYER_COUNT)
        tokens = (
            self.optional_tile_embedding(tile)
            + self.meld_kind_embedding(kind)
            + self.seat_embedding(owners)[None]
            + self.optional_seat_embedding(source)
            + self.meld_slot_embedding(slots)[None]
            + self.token_type_embedding.weight[3]
        )
        return tokens, padding

    def forward(self, tile_obs: Tensor, melds: Tensor, meta: Tensor) -> Tensor:
        global_token = self._global_token(meta).unsqueeze(1)
        meld_tokens, meld_padding = self._meld_tokens(melds)
        hidden = torch.cat(
            (global_token, self._tile_tokens(tile_obs, meta), self._player_tokens(meta), meld_tokens),
            dim=1,
        )
        prefix = torch.ones(
            (hidden.shape[0], 1 + TILE_KIND_COUNT + PLAYER_COUNT),
            device=hidden.device,
            dtype=torch.bool,
        )
        key_valid = torch.cat((prefix, ~meld_padding), dim=1)
        for block in self.blocks:
            hidden = block(hidden, causal=False, key_valid=key_valid)
        return self.output_norm(hidden[:, 0])


class HistoryEncoder(nn.Module):
    EVENT_KIND_COUNT = 11

    def __init__(self, config: BeliefModelConfig) -> None:
        super().__init__()
        self.max_history = config.max_history
        width = config.d_model
        self.kind_embedding = nn.Embedding(self.EVENT_KIND_COUNT + 1, width)
        self.seat_embedding = nn.Embedding(PLAYER_COUNT + 1, width)
        self.tile_embedding = nn.Embedding(TILE_KIND_COUNT + 1, width)
        self.flags_embedding = nn.Embedding(256, width)
        self.numeric = nn.Linear(2, width, bias=False)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.history_layers)
        )
        self.output_norm = RMSNorm(width)
        self.empty_summary = nn.Parameter(torch.zeros(width))

    def forward(self, events: Tensor, lengths: Tensor) -> Tensor:
        if events.shape[1:] != (self.max_history, EVENT_RECORD_WIDTH):
            raise ValueError("events must have shape [batch, 192, 8]")
        kind = events[:, :, 0].long().clamp(0, self.EVENT_KIND_COUNT - 1)
        actor = _optional_index(events[:, :, 1], PLAYER_COUNT, PLAYER_COUNT - 1)
        target = _optional_index(events[:, :, 2], PLAYER_COUNT, PLAYER_COUNT - 1)
        tile = _optional_index(events[:, :, 3], TILE_KIND_COUNT, TILE_KIND_COUNT - 1)
        flags = events[:, :, 4].long().clamp(0, 255)
        numeric = events[:, :, 5:7].float()
        numeric = torch.sign(numeric) * torch.log1p(numeric.abs()) / math.log1p(40_000.0)
        hidden = (
            self.kind_embedding(kind)
            + self.seat_embedding(actor)
            + self.seat_embedding(target)
            + self.tile_embedding(tile)
            + self.flags_embedding(flags)
            + self.numeric(numeric.clamp(-1, 1).to(self.numeric.weight.dtype))
        )
        positions = torch.arange(self.max_history, device=events.device)
        for block in self.blocks:
            hidden = block(hidden, causal=True, positions=positions)
        hidden = self.output_norm(hidden)
        lengths = lengths.long()
        indices = (lengths - 1).clamp_min(0)
        summary = hidden[torch.arange(events.shape[0], device=events.device), indices]
        empty = self.empty_summary.unsqueeze(0).expand(events.shape[0], -1)
        return torch.where((lengths > 0)[:, None], summary, empty)


class BeliefResidualModel(nn.Module):
    """Scores candidate worlds independently under one public root."""

    def __init__(self, config: BeliefModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BeliefModelConfig()
        width = self.config.d_model
        public_width = self.config.interaction_width
        self.static_encoder = StaticStateEncoder(self.config)
        self.history_encoder = HistoryEncoder(self.config)
        self.public_projection = nn.Sequential(
            RMSNorm(width * 2),
            nn.Linear(width * 2, public_width),
            nn.SiLU(),
        )
        self.world_encoder = nn.Sequential(
            nn.Linear(BELIEF_WORLD_PLANES * TILE_KIND_COUNT, self.config.world_hidden),
            nn.SiLU(),
            nn.Linear(self.config.world_hidden, public_width),
            nn.SiLU(),
        )
        interaction_input = public_width * 3
        self.residual_head = nn.Sequential(
            RMSNorm(interaction_input),
            nn.Linear(interaction_input, public_width),
            nn.SiLU(),
            nn.Linear(public_width, 1),
        )
        self.apply(self._initialize)
        output = self.residual_head[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def encode_public(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
    ) -> Tensor:
        static = self.static_encoder(tile_obs, melds, meta)
        history = self.history_encoder(events, event_lengths)
        return self.public_projection(torch.cat((static, history), dim=-1))

    def score_worlds(self, public: Tensor, candidate_worlds: Tensor) -> Tensor:
        if candidate_worlds.ndim != 4 or candidate_worlds.shape[2:] != (
            BELIEF_WORLD_PLANES,
            TILE_KIND_COUNT,
        ):
            raise ValueError("candidate_worlds must have shape [batch, candidates, 4, 27]")
        if candidate_worlds.shape[0] != public.shape[0]:
            raise ValueError("public and candidate batch sizes must match")
        worlds = candidate_worlds.to(self.world_encoder[0].weight.dtype) / 4.0
        worlds = self.world_encoder(worlds.flatten(2))
        expanded = public[:, None, :].expand(-1, worlds.shape[1], -1)
        interaction = torch.cat((expanded, worlds, expanded * worlds), dim=-1)
        return self.residual_head(interaction).squeeze(-1)

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        candidate_worlds: Tensor,
    ) -> Tensor:
        public = self.encode_public(tile_obs, melds, meta, events, event_lengths)
        return self.score_worlds(public, candidate_worlds)
