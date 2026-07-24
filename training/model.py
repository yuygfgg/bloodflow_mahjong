"""Structured actor-critic with a static encoder and cached GPT history."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ACTION_SPACE_SIZE = 115
TILE_KIND_COUNT = 27
TILE_OBSERVATION_PLANES = 10
PLAYER_COUNT = 4
MELD_SLOTS = 4
MELD_FIELDS = 3
META_WIDTH = 34
EVENT_RECORD_WIDTH = 8
SHANTEN_CLASS_COUNT = 10  # -1 through 8


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 192
    num_heads: int = 6
    static_layers: int = 2
    history_layers: int = 4
    ffn_dim: int = 768
    dropout: float = 0.0
    max_history: int = 192
    value_atoms: int = 129
    value_min: float = -4.0
    value_max: float = 4.0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.static_layers <= 0 or self.history_layers <= 0:
            raise ValueError("both encoders need at least one layer")
        if self.max_history <= 0 or self.value_atoms < 2:
            raise ValueError("max_history and value_atoms must be positive")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be less than value_max")


@dataclass(frozen=True)
class LayerKV:
    key: Tensor
    value: Tensor

    def detach(self) -> LayerKV:
        return LayerKV(self.key.detach(), self.value.detach())


@dataclass(frozen=True)
class HistoryKVCache:
    layers: tuple[LayerKV, ...]
    length: int
    summary: Tensor

    def detach(self) -> HistoryKVCache:
        return HistoryKVCache(
            tuple(layer.detach() for layer in self.layers),
            self.length,
            self.summary.detach(),
        )


@dataclass(frozen=True)
class ActorCriticOutput:
    logits: Tensor
    raw_logits: Tensor
    value_logits: Tensor
    value: Tensor
    shanten_logits: Tensor
    improving_logits: Tensor
    static_embedding: Tensor
    history_embedding: Tensor


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
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.up(inputs).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        inputs: Tensor,
        *,
        causal: bool,
        key_valid: Tensor | None = None,
        past: LayerKV | None = None,
    ) -> tuple[Tensor, LayerKV]:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        past_length = 0
        if past is not None:
            if past.key.shape[:2] != key.shape[:2] or past.key.shape[3] != key.shape[3]:
                raise ValueError("KV cache shape does not match the attention layer")
            past_length = past.key.shape[2]
            key = torch.cat((past.key, key), dim=2)
            value = torch.cat((past.value, value), dim=2)

        query_length = query.shape[2]
        key_length = key.shape[2]
        attention_mask: Tensor | None = None
        if causal:
            new_causal = torch.ones(
                (query_length, query_length), device=inputs.device, dtype=torch.bool
            ).tril()
            if past_length:
                prefix = torch.ones(
                    (query_length, past_length), device=inputs.device, dtype=torch.bool
                )
                attention_mask = torch.cat((prefix, new_causal), dim=1)
            else:
                attention_mask = new_causal
            attention_mask = attention_mask.view(1, 1, query_length, key_length)

        if key_valid is not None:
            if past is not None:
                raise ValueError(
                    "key_valid is only supported by full-sequence attention"
                )
            if key_valid.shape != (inputs.shape[0], key_length):
                raise ValueError("key_valid has the wrong shape")
            valid_mask = key_valid[:, None, None, :]
            attention_mask = (
                valid_mask if attention_mask is None else attention_mask & valid_mask
            )

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(inputs.shape)
        return self.output(attended), LayerKV(key, value)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim, config.dropout)

    def forward(
        self,
        inputs: Tensor,
        *,
        causal: bool,
        key_valid: Tensor | None = None,
        past: LayerKV | None = None,
    ) -> tuple[Tensor, LayerKV]:
        attended, present = self.attention(
            self.attention_norm(inputs),
            causal=causal,
            key_valid=key_valid,
            past=past,
        )
        hidden = inputs + attended
        return hidden + self.ffn(self.ffn_norm(hidden)), present


def _optional_index(values: Tensor, missing_index: int, maximum: int) -> Tensor:
    values = values.long()
    return torch.where(values < 0, missing_index, values.clamp(max=maximum))


class StaticStateEncoder(nn.Module):
    """Bidirectional encoder over the current public state and actor hand."""

    def __init__(self, config: TransformerConfig) -> None:
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
        phase = meta[:, 0].long().clamp(0, 5)
        direction = meta[:, 3].long().clamp(0, 3)
        draw = _optional_index(meta[:, 5], TILE_KIND_COUNT, TILE_KIND_COUNT - 1)
        pending = _optional_index(meta[:, 7], PLAYER_COUNT, PLAYER_COUNT - 1)
        response = _optional_index(meta[:, 8], TILE_KIND_COUNT, TILE_KIND_COUNT - 1)
        selected_suit = _optional_index(meta[:, 11], 3, 2)
        flags = meta[:, 29].long().clamp(0, 7)
        numeric = torch.stack(
            (
                meta[:, 4].float() / 55.0,
                meta[:, 9].float() / 108.0,
                meta[:, 10].float() / 3.0,
            ),
            dim=-1,
        ).to(meta.device)
        return (
            self.phase_embedding(phase)
            + self.direction_embedding(direction)
            + self.optional_tile_embedding(draw)
            + self.optional_seat_embedding(pending)
            + self.optional_tile_embedding(response)
            + self.optional_suit_embedding(selected_suit)
            + self.reaction_embedding(flags)
            + self.binary_embedding(meta[:, 6].long().clamp(0, 1))
            + self.binary_embedding(meta[:, 28].long().clamp(0, 1))
            + self.global_numeric(numeric)
            + self.token_type_embedding.weight[0]
        )

    def _tile_tokens(self, tile_obs: Tensor, meta: Tensor) -> Tensor:
        batch = tile_obs.shape[0]
        tile_ids = torch.arange(TILE_KIND_COUNT, device=tile_obs.device)
        tokens = self.tile_embedding(tile_ids) + self.suit_embedding(tile_ids // 9)
        tokens = tokens + self.rank_embedding(tile_ids % 9)
        tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
        counts = tile_obs.long().clamp(0, 4)
        for plane, embedding in enumerate(self.count_embeddings):
            tokens = tokens + embedding(counts[:, plane])

        draw = meta[:, 5, None].long() == tile_ids[None, :]
        response = meta[:, 8, None].long() == tile_ids[None, :]
        own_missing = meta[:, 16, None].long() == (tile_ids[None, :] // 9)
        return (
            tokens
            + self.binary_embedding(draw.long())
            + self.binary_embedding(response.long())
            + self.binary_embedding(own_missing.long())
            + self.token_type_embedding.weight[1]
        )

    def _player_tokens(self, meta: Tensor) -> Tensor:
        batch = meta.shape[0]
        seats = torch.arange(PLAYER_COUNT, device=meta.device)
        seats_batch = seats.unsqueeze(0).expand(batch, -1)
        dealer = meta[:, 2, None].long() == seats_batch
        missing = _optional_index(meta[:, 16:20], 3, 2)
        won = meta[:, 20:24].long().clamp(0, 1)
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
            self.seat_embedding(seats_batch)
            + self.binary_embedding(dealer.long())
            + self.optional_suit_embedding(missing)
            + self.binary_embedding(won)
            + self.player_numeric(numeric)
            + self.token_type_embedding.weight[2]
        )

    def _meld_tokens(self, melds: Tensor) -> tuple[Tensor, Tensor]:
        batch = melds.shape[0]
        flat = melds.reshape(batch, PLAYER_COUNT * MELD_SLOTS, MELD_FIELDS)
        padding = flat[:, :, 0] == 255
        tile = torch.where(padding, TILE_KIND_COUNT, flat[:, :, 0].long().clamp(0, 26))
        kind = torch.where(padding, 4, flat[:, :, 1].long().clamp(0, 3))
        source = torch.where(padding, PLAYER_COUNT, flat[:, :, 2].long().clamp(0, 3))
        owners = torch.arange(PLAYER_COUNT, device=melds.device).repeat_interleave(
            MELD_SLOTS
        )
        slots = torch.arange(MELD_SLOTS, device=melds.device).repeat(PLAYER_COUNT)
        tokens = (
            self.optional_tile_embedding(tile)
            + self.meld_kind_embedding(kind)
            + self.seat_embedding(owners)[None, :, :]
            + self.optional_seat_embedding(source)
            + self.meld_slot_embedding(slots)[None, :, :]
            + self.token_type_embedding.weight[3]
        )
        return tokens, padding

    def forward(self, tile_obs: Tensor, melds: Tensor, meta: Tensor) -> Tensor:
        _validate_static_inputs(tile_obs, melds, meta)
        global_token = self._global_token(meta).unsqueeze(1)
        tile_tokens = self._tile_tokens(tile_obs, meta)
        player_tokens = self._player_tokens(meta)
        meld_tokens, meld_padding = self._meld_tokens(melds)
        hidden = torch.cat(
            (global_token, tile_tokens, player_tokens, meld_tokens), dim=1
        )
        prefix_valid = torch.ones(
            (hidden.shape[0], 1 + TILE_KIND_COUNT + PLAYER_COUNT),
            device=hidden.device,
            dtype=torch.bool,
        )
        key_valid = torch.cat((prefix_valid, ~meld_padding), dim=1)
        for block in self.blocks:
            hidden, _ = block(hidden, causal=False, key_valid=key_valid)
        return self.output_norm(hidden[:, 0])


class HistoryEncoder(nn.Module):
    """GPT-style viewer-scoped event encoder with an incremental KV cache."""

    EVENT_KIND_COUNT = 11

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.kind_embedding = nn.Embedding(self.EVENT_KIND_COUNT + 1, width)
        self.seat_embedding = nn.Embedding(PLAYER_COUNT + 1, width)
        self.tile_embedding = nn.Embedding(TILE_KIND_COUNT + 1, width)
        self.flags_embedding = nn.Embedding(256, width)
        self.position_embedding = nn.Embedding(config.max_history, width)
        self.numeric = nn.Linear(2, width, bias=False)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.history_layers)
        )
        self.output_norm = RMSNorm(width)
        self.empty_summary = nn.Parameter(torch.zeros(width))

    def _embed(self, events: Tensor, positions: Tensor) -> Tensor:
        kind = events[:, :, 0].long().clamp(0, self.EVENT_KIND_COUNT - 1)
        actor = _optional_index(events[:, :, 1], PLAYER_COUNT, PLAYER_COUNT - 1)
        target = _optional_index(events[:, :, 2], PLAYER_COUNT, PLAYER_COUNT - 1)
        tile = _optional_index(events[:, :, 3], TILE_KIND_COUNT, TILE_KIND_COUNT - 1)
        flags = events[:, :, 4].long().clamp(0, 255)
        values = events[:, :, 5:7].float()
        values = torch.sign(values) * torch.log1p(values.abs()) / math.log1p(40_000.0)
        values = values.clamp(-1.0, 1.0)
        return (
            self.kind_embedding(kind)
            + self.seat_embedding(actor)
            + self.seat_embedding(target)
            + self.tile_embedding(tile)
            + self.flags_embedding(flags)
            + self.position_embedding(positions)[None, :, :]
            + self.numeric(values)
        )

    def forward(self, events: Tensor, lengths: Tensor) -> Tensor:
        _validate_history_inputs(events, lengths, self.config.max_history)
        batch, length, _ = events.shape
        positions = torch.arange(length, device=events.device)
        hidden = self._embed(events, positions)
        key_valid = positions[None, :] < lengths[:, None]
        if length:
            empty_rows = lengths == 0
            key_valid[:, 0] |= empty_rows
        for block in self.blocks:
            hidden, _ = block(hidden, causal=True, key_valid=key_valid)
        hidden = self.output_norm(hidden)
        indices = (lengths.long() - 1).clamp_min(0)
        summary = hidden[torch.arange(batch, device=events.device), indices]
        empty = self.empty_summary.unsqueeze(0).expand(batch, -1)
        return torch.where((lengths > 0)[:, None], summary, empty)

    def forward_cached(
        self,
        new_events: Tensor,
        cache: HistoryKVCache | None = None,
    ) -> tuple[Tensor, HistoryKVCache]:
        if new_events.ndim != 3 or new_events.shape[2] != EVENT_RECORD_WIDTH:
            raise ValueError("new_events must have shape [batch, time, 8]")
        if new_events.shape[1] == 0:
            if cache is None:
                raise ValueError("an initial history cache needs at least one event")
            return cache.summary, cache

        past_length = 0 if cache is None else cache.length
        if past_length + new_events.shape[1] > self.config.max_history:
            raise ValueError(
                "history cache would exceed max_history; rebuild the window"
            )
        if cache is not None:
            if len(cache.layers) != len(self.blocks):
                raise ValueError("history cache layer count does not match the model")
            if cache.summary.shape[0] != new_events.shape[0]:
                raise ValueError("history cache batch size does not match new_events")

        positions = torch.arange(
            past_length,
            past_length + new_events.shape[1],
            device=new_events.device,
        )
        hidden = self._embed(new_events, positions)
        present_layers: list[LayerKV] = []
        for index, block in enumerate(self.blocks):
            past = None if cache is None else cache.layers[index]
            hidden, present = block(hidden, causal=True, past=past)
            present_layers.append(present)
        summary = self.output_norm(hidden[:, -1])
        present_cache = HistoryKVCache(
            tuple(present_layers), past_length + new_events.shape[1], summary
        )
        return summary, present_cache


class BloodFlowTransformer(nn.Module):
    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        width = self.config.d_model
        fused = width * 2
        self.static_encoder = StaticStateEncoder(self.config)
        self.history_encoder = HistoryEncoder(self.config)
        self.actor = nn.Sequential(
            RMSNorm(fused),
            nn.Linear(fused, fused),
            nn.SiLU(),
            nn.Linear(fused, ACTION_SPACE_SIZE),
        )
        self.critic = nn.Sequential(
            RMSNorm(fused),
            nn.Linear(fused, fused),
            nn.SiLU(),
            nn.Linear(fused, self.config.value_atoms),
        )
        self.shanten_head = nn.Linear(width, SHANTEN_CLASS_COUNT)
        self.improving_head = nn.Linear(width, TILE_KIND_COUNT)
        self.register_buffer(
            "value_support",
            torch.linspace(
                self.config.value_min,
                self.config.value_max,
                self.config.value_atoms,
            ),
            persistent=True,
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _heads(
        self,
        static: Tensor,
        history: Tensor,
        legal_mask: Tensor | None,
    ) -> ActorCriticOutput:
        fused = torch.cat((static, history), dim=-1)
        raw_logits = self.actor(fused)
        logits = raw_logits
        if legal_mask is not None:
            if legal_mask.shape != raw_logits.shape:
                raise ValueError("legal_mask must have shape [batch, 115]")
            legal_mask = legal_mask.to(device=raw_logits.device, dtype=torch.bool)
            logits = raw_logits.masked_fill(
                ~legal_mask, torch.finfo(raw_logits.dtype).min
            )
        value_logits = self.critic(fused)
        probabilities = torch.softmax(value_logits.float(), dim=-1)
        value = probabilities @ self.value_support.float()
        return ActorCriticOutput(
            logits=logits,
            raw_logits=raw_logits,
            value_logits=value_logits,
            value=value,
            shanten_logits=self.shanten_head(static),
            improving_logits=self.improving_head(static),
            static_embedding=static,
            history_embedding=history,
        )

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
        legal_mask: Tensor | None = None,
    ) -> ActorCriticOutput:
        static = self.static_encoder(tile_obs, melds, meta)
        history = self.history_encoder(events, event_lengths)
        return self._heads(static, history, legal_mask)

    def forward_cached(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        new_events: Tensor,
        cache: HistoryKVCache | None = None,
        legal_mask: Tensor | None = None,
    ) -> tuple[ActorCriticOutput, HistoryKVCache]:
        static = self.static_encoder(tile_obs, melds, meta)
        history, next_cache = self.history_encoder.forward_cached(new_events, cache)
        return self._heads(static, history, legal_mask), next_cache


def _validate_static_inputs(tile_obs: Tensor, melds: Tensor, meta: Tensor) -> None:
    if tile_obs.ndim != 3 or tile_obs.shape[1:] != (
        TILE_OBSERVATION_PLANES,
        TILE_KIND_COUNT,
    ):
        raise ValueError("tile_obs must have shape [batch, 10, 27]")
    if melds.ndim != 4 or melds.shape[1:] != (
        PLAYER_COUNT,
        MELD_SLOTS,
        MELD_FIELDS,
    ):
        raise ValueError("melds must have shape [batch, 4, 4, 3]")
    if meta.ndim != 2 or meta.shape[1] != META_WIDTH:
        raise ValueError("meta must have shape [batch, 34]")
    if not (tile_obs.shape[0] == melds.shape[0] == meta.shape[0]):
        raise ValueError("static input batch sizes must match")


def _validate_history_inputs(events: Tensor, lengths: Tensor, maximum: int) -> None:
    if events.ndim != 3 or events.shape[2] != EVENT_RECORD_WIDTH:
        raise ValueError("events must have shape [batch, time, 8]")
    if events.shape[1] > maximum:
        raise ValueError(f"events exceed max_history={maximum}")
    if events.shape[1] == 0:
        raise ValueError("events must allocate at least one history slot")
    if lengths.shape != (events.shape[0],):
        raise ValueError("event_lengths must have shape [batch]")
    if torch.any(lengths < 0) or torch.any(lengths > events.shape[1]):
        raise ValueError("event_lengths contain an out-of-range value")
