"""Export a PPO policy checkpoint to the fixed rule-nn ONNX contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import ENGINE_RULES_VERSION, MODEL_HISTORY_CAPACITY
from .model import (
    ACTION_SPACE_SIZE,
    TILE_KIND_COUNT,
    TILE_OBSERVATION_PLANES,
    BloodFlowTransformer,
)
from .pipeline import load_checkpoint_model


ONNX_OPSET = 17
INPUT_NAMES = ("tile_obs", "melds", "meta", "events", "event_lengths")
OUTPUT_NAME = "logits"
MODEL_RULES_VERSION_METADATA = "engine_rules_version"

PolicyInputs: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class ExportResult:
    path: Path
    size_bytes: int
    max_absolute_error: float | None
    mean_absolute_error: float | None
    argmax_matches: bool | None


class RawPolicy(nn.Module):
    """Expose only policy logits and leave legal-action masking to the engine."""

    def __init__(self, model: BloodFlowTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tile_obs: Tensor,
        melds: Tensor,
        meta: Tensor,
        events: Tensor,
        event_lengths: Tensor,
    ) -> Tensor:
        return self.model(
            tile_obs,
            melds,
            meta,
            events,
            event_lengths,
        ).raw_logits


def sample_inputs() -> PolicyInputs:
    """Return one deterministic, valid-shaped input for export and parity checks."""
    generator = torch.Generator().manual_seed(0x4E4E)
    tile_obs = torch.randint(
        0,
        5,
        (1, TILE_OBSERVATION_PLANES, TILE_KIND_COUNT),
        generator=generator,
        dtype=torch.uint8,
    )
    melds = torch.full((1, 4, 4, 3), 255, dtype=torch.uint8)
    melds[0, 0, 0] = torch.tensor((0, 0, 0), dtype=torch.uint8)

    meta = torch.zeros((1, 34), dtype=torch.int32)
    meta[0, 0] = 1
    meta[0, 1] = 0
    meta[0, 2] = 0
    meta[0, 3] = 1
    meta[0, 4] = 42
    meta[0, 5] = 13
    meta[0, 7] = -1
    meta[0, 8] = -1
    meta[0, 9] = 64
    meta[0, 11] = -1
    meta[0, 12:16] = 10_000
    meta[0, 16:20] = -1
    meta[0, 24:28] = 13

    events = torch.zeros(
        (1, MODEL_HISTORY_CAPACITY, 8),
        dtype=torch.int32,
    )
    positions = torch.arange(64, dtype=torch.int32)
    events[0, :64, 0] = positions % 11
    events[0, :64, 1] = positions % 4
    events[0, :64, 2] = (positions + 1) % 4
    events[0, :64, 3] = positions % 27
    events[0, :64, 4] = positions % 8
    events[0, :64, 5] = positions * 100
    events[0, :64, 6] = positions
    event_lengths = torch.tensor((64,), dtype=torch.int64)
    return tile_obs, melds, meta, events, event_lengths


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(
        int(dimension.dim_value) for dimension in value.type.tensor_type.shape.dim
    )


def _check_graph_contract(model: Any) -> None:
    import onnx

    metadata = {
        entry.key: entry.value
        for entry in model.metadata_props
        if entry.key == MODEL_RULES_VERSION_METADATA
    }
    expected_version = str(ENGINE_RULES_VERSION)
    if metadata != {MODEL_RULES_VERSION_METADATA: expected_version}:
        raise ValueError(
            "exported ONNX model does not declare the current engine rules "
            f"version {expected_version}: {metadata}"
        )

    expected_inputs = (
        (
            "tile_obs",
            onnx.TensorProto.UINT8,
            (1, TILE_OBSERVATION_PLANES, TILE_KIND_COUNT),
        ),
        ("melds", onnx.TensorProto.UINT8, (1, 4, 4, 3)),
        ("meta", onnx.TensorProto.INT32, (1, 34)),
        (
            "events",
            onnx.TensorProto.INT32,
            (1, MODEL_HISTORY_CAPACITY, 8),
        ),
        ("event_lengths", onnx.TensorProto.INT64, (1,)),
    )
    actual_inputs = tuple(
        (value.name, value.type.tensor_type.elem_type, _shape(value))
        for value in model.graph.input
    )
    if actual_inputs != expected_inputs:
        raise ValueError(
            f"exported ONNX inputs do not match the rule-nn contract: {actual_inputs}"
        )

    expected_output = (OUTPUT_NAME, onnx.TensorProto.FLOAT, (1, ACTION_SPACE_SIZE))
    actual_outputs = tuple(
        (value.name, value.type.tensor_type.elem_type, _shape(value))
        for value in model.graph.output
    )
    if actual_outputs != (expected_output,):
        raise ValueError(
            "exported ONNX output does not match the rule-nn contract: "
            f"{actual_outputs}"
        )

    default_opsets = {
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    }
    if default_opsets != {ONNX_OPSET}:
        raise ValueError(f"exported ONNX uses unexpected opsets: {default_opsets}")


def _reference_parity(
    onnx_model: Any,
    policy: RawPolicy,
    inputs: PolicyInputs,
) -> tuple[float, float, bool]:
    from onnx.reference import ReferenceEvaluator

    with torch.inference_mode():
        expected = policy(*inputs).detach().cpu().numpy()
    feed = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, inputs, strict=True)
    }
    actual = np.asarray(ReferenceEvaluator(onnx_model).run([OUTPUT_NAME], feed)[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    argmax_matches = bool(
        np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    )
    return float(difference.max()), float(difference.mean()), argmax_matches


def export_model(
    model: BloodFlowTransformer,
    output: Path,
    *,
    check: bool = True,
    parity: bool = True,
) -> ExportResult:
    """Export one fixed-shape policy graph and optionally validate it."""
    if model.config.max_history < MODEL_HISTORY_CAPACITY:
        raise ValueError(
            "model max_history is smaller than the rule-nn history contract "
            f"({model.config.max_history} < {MODEL_HISTORY_CAPACITY})"
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    policy = RawPolicy(model.to(torch.device("cpu")).eval()).eval()
    inputs = sample_inputs()
    maximum_error: float | None = None
    mean_error: float | None = None
    argmax_matches: bool | None = None

    try:
        torch.onnx.export(
            policy,
            inputs,
            temporary,
            input_names=INPUT_NAMES,
            output_names=(OUTPUT_NAME,),
            opset_version=ONNX_OPSET,
            dynamo=False,
            external_data=False,
            export_params=True,
            do_constant_folding=True,
        )

        import onnx

        onnx_model = onnx.load(temporary, load_external_data=False)
        onnx_model.metadata_props.add(
            key=MODEL_RULES_VERSION_METADATA,
            value=str(ENGINE_RULES_VERSION),
        )
        if check:
            onnx.checker.check_model(onnx_model)
            _check_graph_contract(onnx_model)
        if parity:
            maximum_error, mean_error, argmax_matches = _reference_parity(
                onnx_model,
                policy,
                inputs,
            )
        onnx.save(onnx_model, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return ExportResult(
        path=output,
        size_bytes=output.stat().st_size,
        max_absolute_error=maximum_error,
        mean_absolute_error=mean_error,
        argmax_matches=argmax_matches,
    )


def export_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    check: bool = True,
    parity: bool = True,
) -> ExportResult:
    model = load_checkpoint_model(checkpoint, torch.device("cpu"))
    return export_model(model, output, check=check, parity=parity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="PPO checkpoint to export")
    parser.add_argument("output", type=Path, help="destination .onnx path")
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip ONNX checker and fixed-interface validation",
    )
    parser.add_argument(
        "--no-parity",
        action="store_true",
        help="skip PyTorch versus ONNX reference evaluation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_checkpoint(
        args.checkpoint,
        args.output,
        check=not args.no_check,
        parity=not args.no_parity,
    )
    print(f"ONNX  {result.path}  {result.size_bytes:,} bytes")
    if result.max_absolute_error is not None:
        print(
            "PARITY"
            f"  max {result.max_absolute_error:.3e}"
            f"  mean {result.mean_absolute_error:.3e}"
            f"  argmax {'match' if result.argmax_matches else 'DIFFER'}"
        )


if __name__ == "__main__":
    main()
