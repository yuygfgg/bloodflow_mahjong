from __future__ import annotations

import numpy as np
import pytest
import torch

from training import (
    BloodFlowTransformer,
    HistoryEncoder,
    TransformerConfig,
    unpack_action_masks,
)


def config() -> TransformerConfig:
    return TransformerConfig(
        d_model=48,
        num_heads=4,
        static_layers=1,
        history_layers=2,
        ffn_dim=96,
        max_history=32,
        value_atoms=17,
    )


def state(batch: int = 3, history: int = 12) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    tile_obs = torch.randint(
        0, 5, (batch, 11, 27), generator=generator, dtype=torch.uint8
    )
    melds = torch.full((batch, 4, 4, 3), 255, dtype=torch.uint8)
    melds[:, 0, 0] = torch.tensor([0, 1, 2], dtype=torch.uint8)[:batch, None]
    melds[:, 0, 0, 1:] = torch.tensor([0, 0], dtype=torch.uint8)
    meta = torch.zeros((batch, 34), dtype=torch.int32)
    meta[:, 4] = 42
    meta[:, 9] = history
    meta[:, 12:16] = 10_000
    meta[:, 24:28] = 14
    events = torch.zeros((batch, history, 8), dtype=torch.int32)
    events[:, :, 0] = torch.arange(history) % 11
    events[:, :, 1] = torch.arange(history) % 4
    events[:, :, 2] = (torch.arange(history) + 1) % 4
    events[:, :, 3] = torch.arange(history) % 27
    events[:, :, 4] = torch.arange(history) % 8
    events[:, :, 5] = torch.arange(history) * 100
    events[:, :, 6] = torch.arange(history)
    lengths = torch.full((batch,), history, dtype=torch.int64)
    legal = torch.ones((batch, 115), dtype=torch.bool)
    legal[:, 5::7] = False
    return tile_obs, melds, meta, events, lengths, legal


def test_mask_unpack_matches_fixed_action_space() -> None:
    words = np.zeros((2, 2), dtype=np.uint64)
    words[0, 0] = np.uint64(1 << 30)
    words[0, 1] = np.uint64(1 << 50)
    words[1, 0] = np.uint64(1)
    dense = unpack_action_masks(words)
    assert dense.shape == (2, 115)
    assert dense[0, 30]
    assert dense[0, 64 + 50]
    assert dense[1, 0]
    assert not dense[0, 31]


def test_default_model_stays_in_the_planned_size() -> None:
    parameters = sum(
        parameter.numel() for parameter in BloodFlowTransformer().parameters()
    )
    assert 3_000_000 <= parameters <= 5_000_000


def test_history_uses_rope_and_requires_even_head_dimensions() -> None:
    model = BloodFlowTransformer(config())
    assert not hasattr(model.history_encoder, "position_embedding")
    assert all(
        hasattr(block.attention, "rope_inverse")
        for block in model.history_encoder.blocks
    )
    with pytest.raises(ValueError, match="even attention head dimension"):
        TransformerConfig(
            d_model=30,
            num_heads=6,
            static_layers=1,
            history_layers=1,
            ffn_dim=60,
        )


def test_transformer_forward_backward_and_masking() -> None:
    model = BloodFlowTransformer(config())
    model.train()
    inputs = state()
    output = model(*inputs)
    assert output.logits.shape == (3, 115)
    assert output.value_logits.shape == (3, 17)
    assert output.value.shape == (3,)
    assert output.shanten_logits.shape == (3, 10)
    assert output.improving_logits.shape == (3, 27)
    assert torch.all(output.logits[~inputs[-1]] == torch.finfo(output.logits.dtype).min)

    loss = (
        output.logits[inputs[-1]].mean()
        + output.value_logits.square().mean()
        + output.shanten_logits.square().mean()
        + output.improving_logits.square().mean()
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cached_history_matches_full_history() -> None:
    model = BloodFlowTransformer(config()).eval()
    tile_obs, melds, meta, events, lengths, legal = state(batch=1, history=7)
    with torch.no_grad():
        full = model(tile_obs, melds, meta, events, lengths, legal)
        first_history, cache = model.history_encoder.forward_cached(events[:, :3])
        assert first_history.shape == (1, 48)
        cached, cache = model.forward_cached(
            tile_obs,
            melds,
            meta,
            events[:, 3:7],
            cache,
            legal,
        )
    assert cache.length == 7
    torch.testing.assert_close(
        cached.history_embedding, full.history_embedding, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(cached.logits, full.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cached.value, full.value, atol=1e-5, rtol=1e-5)


def test_right_padding_can_be_safely_trimmed_before_forward() -> None:
    model = BloodFlowTransformer(config()).eval()
    tile_obs, melds, meta, events, lengths, legal = state(batch=2, history=7)
    padded = torch.zeros((2, 32, 8), dtype=torch.int32)
    padded[:, :7] = events
    with torch.no_grad():
        trimmed = model(tile_obs, melds, meta, events, lengths, legal)
        masked_padding = model(tile_obs, melds, meta, padded, lengths, legal)
    torch.testing.assert_close(
        trimmed.history_embedding, masked_padding.history_embedding, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(trimmed.logits, masked_padding.logits, atol=1e-5, rtol=1e-5)


def test_empty_history_length_uses_finite_learned_summary() -> None:
    encoder = HistoryEncoder(config()).eval()
    events = torch.zeros((1, 4, 8), dtype=torch.int32)
    summary = encoder(events, torch.zeros(1, dtype=torch.int64))
    assert torch.isfinite(summary).all()


def test_history_cache_rejects_overflow() -> None:
    encoder = HistoryEncoder(config()).eval()
    events = torch.zeros((1, 32, 8), dtype=torch.int32)
    _, cache = encoder.forward_cached(events[:, :31])
    with pytest.raises(ValueError, match="exceed"):
        encoder.forward_cached(events[:, :2], cache)


def test_engine_observation_smoke() -> None:
    bm = pytest.importorskip("bloodflow_mahjong")
    batch_size = 4
    batch = bm.Batch(batch_size, seed=23)
    tile_obs = np.empty((batch_size, bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
    melds = np.empty((batch_size, 4, bm.MELD_SLOTS, bm.MELD_FIELDS), dtype=np.uint8)
    river = np.empty((batch_size, bm.RIVER_TILE_CAPACITY, 2), dtype=np.uint8)
    meta = np.empty((batch_size, bm.META_OBSERVATION_WIDTH), dtype=np.int32)
    events = np.empty((batch_size, 32, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    lengths = np.empty(batch_size, dtype=np.uint16)
    masks = np.empty((batch_size, 2), dtype=np.uint64)
    batch.observe_into(tile_obs, melds, river, meta)
    batch.events_into(events, lengths)
    batch.legal_action_masks_into(masks)
    model = BloodFlowTransformer(config()).eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(tile_obs),
            torch.from_numpy(melds),
            torch.from_numpy(meta),
            torch.from_numpy(events),
            torch.from_numpy(lengths.astype(np.int64)),
            torch.from_numpy(unpack_action_masks(masks)),
        )
    assert output.logits.shape == (batch_size, 115)
    assert torch.isfinite(output.value).all()
