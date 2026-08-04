from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.export_onnx import (
    INPUT_NAMES,
    ExportResult,
    RawPolicy,
    build_parser,
    export_model,
    sample_inputs,
)
from training.model import BloodFlowTransformer, TransformerConfig


def small_config(*, max_history: int = 192) -> TransformerConfig:
    return TransformerConfig(
        d_model=24,
        num_heads=3,
        static_layers=1,
        history_layers=1,
        ffn_dim=48,
        max_history=max_history,
        value_atoms=5,
    )


def test_sample_inputs_match_fixed_rule_nn_contract() -> None:
    tile_obs, melds, meta, events, event_lengths = sample_inputs()

    assert INPUT_NAMES == (
        "tile_obs",
        "melds",
        "meta",
        "events",
        "event_lengths",
    )
    assert tile_obs.shape == (1, 10, 27) and tile_obs.dtype == torch.uint8
    assert melds.shape == (1, 4, 4, 3) and melds.dtype == torch.uint8
    assert meta.shape == (1, 34) and meta.dtype == torch.int32
    assert events.shape == (1, 192, 8) and events.dtype == torch.int32
    assert event_lengths.shape == (1,) and event_lengths.dtype == torch.int64


def test_raw_policy_exports_unmasked_actor_logits() -> None:
    model = BloodFlowTransformer(small_config()).eval()
    inputs = sample_inputs()

    with torch.inference_mode():
        expected = model(*inputs).raw_logits
        actual = RawPolicy(model)(*inputs)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 115)


def test_export_model_checks_graph_and_reference_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    output = tmp_path / "policy.onnx"

    result = export_model(BloodFlowTransformer(small_config()), output)

    assert isinstance(result, ExportResult)
    assert result.path == output.resolve()
    assert result.size_bytes == output.stat().st_size
    assert result.max_absolute_error is not None
    assert result.max_absolute_error < 1e-4
    assert result.mean_absolute_error is not None
    assert result.argmax_matches


def test_export_rejects_model_with_insufficient_history(tmp_path: Path) -> None:
    model = BloodFlowTransformer(small_config(max_history=64))

    with pytest.raises(ValueError, match="smaller than the rule-nn history contract"):
        export_model(model, tmp_path / "policy.onnx", check=False, parity=False)


def test_cli_can_disable_optional_validation() -> None:
    args = build_parser().parse_args(
        ["checkpoint.pt", "policy.onnx", "--no-check", "--no-parity"]
    )

    assert args.checkpoint == Path("checkpoint.pt")
    assert args.output == Path("policy.onnx")
    assert args.no_check
    assert args.no_parity
