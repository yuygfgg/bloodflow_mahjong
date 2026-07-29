"""Localize TorchInductor inference errors on real Actor states."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch._inductor import config as inductor_config

from training.benchmarks.inference_compile import (
    _RawLogits,
    _actions,
    _bucket_rows,
    _inputs,
    _load_states,
    _synchronize,
)
from training.model import BloodFlowTransformer, SelfAttention, TransformerBlock
from training.pipeline import _autocast, load_policy


TensorTuple = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class Probe:
    name: str
    module: Callable[..., torch.Tensor | TensorTuple]
    inputs: TensorTuple


class _StaticTokens(nn.Module):
    def __init__(self, actor: BloodFlowTransformer) -> None:
        super().__init__()
        self.encoder = actor.static_encoder

    def forward(
        self,
        tile_obs: torch.Tensor,
        melds: torch.Tensor,
        meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_token = self.encoder._global_token(meta).unsqueeze(1)
        tile_tokens = self.encoder._tile_tokens(tile_obs, meta)
        player_tokens = self.encoder._player_tokens(meta)
        meld_tokens, meld_padding = self.encoder._meld_tokens(melds)
        hidden = torch.cat(
            (global_token, tile_tokens, player_tokens, meld_tokens), dim=1
        )
        prefix_valid = torch.ones(
            (hidden.shape[0], 32), device=hidden.device, dtype=torch.bool
        )
        return hidden, torch.cat((prefix_valid, ~meld_padding), dim=1)


class _StaticAttention(nn.Module):
    def __init__(self, attention: SelfAttention) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, inputs: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        return self.attention(inputs, causal=False, key_valid=key_valid)


class _StaticBlock(nn.Module):
    def __init__(self, block: TransformerBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, inputs: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        return self.block(inputs, causal=False, key_valid=key_valid)


class _HistoryAttention(nn.Module):
    def __init__(self, attention: SelfAttention) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.attention(inputs, causal=True, positions=positions)


class _HistoryBlock(nn.Module):
    def __init__(self, block: TransformerBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.block(inputs, causal=True, positions=positions)


class _HistoryCore(nn.Module):
    """History encoder math without its public input assertions."""

    def __init__(self, actor: BloodFlowTransformer) -> None:
        super().__init__()
        self.encoder = actor.history_encoder

    def forward(self, events: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, length, _ = events.shape
        positions = torch.arange(length, device=events.device)
        hidden = self.encoder._embed(events)
        for block in self.encoder.blocks:
            hidden = block(hidden, causal=True, positions=positions)
        hidden = self.encoder.output_norm(hidden)
        indices = (lengths.long() - 1).clamp_min(0)
        summary = hidden[torch.arange(batch, device=events.device), indices]
        empty = self.encoder.empty_summary.unsqueeze(0).expand(batch, -1)
        return torch.where((lengths > 0)[:, None], summary, empty)


class _HistoryPrefix(nn.Module):
    def __init__(self, actor: BloodFlowTransformer, block_count: int) -> None:
        super().__init__()
        if not 1 <= block_count <= len(actor.history_encoder.blocks):
            raise ValueError("history prefix length is out of range")
        self.encoder = actor.history_encoder
        self.block_count = block_count

    def forward(self, events: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, length, _ = events.shape
        positions = torch.arange(length, device=events.device)
        hidden = self.encoder._embed(events)
        for block in self.encoder.blocks[: self.block_count]:
            hidden = block(hidden, causal=True, positions=positions)
        hidden = self.encoder.output_norm(hidden)
        indices = (lengths.long() - 1).clamp_min(0)
        summary = hidden[torch.arange(batch, device=hidden.device), indices]
        empty = self.encoder.empty_summary.unsqueeze(0).expand_as(summary)
        return torch.where((lengths > 0)[:, None], summary, empty)


class _HistoryPrefixSequence(nn.Module):
    def __init__(
        self,
        actor: BloodFlowTransformer,
        block_count: int,
        *,
        normalize: bool,
    ) -> None:
        super().__init__()
        if not 1 <= block_count <= len(actor.history_encoder.blocks):
            raise ValueError("history prefix length is out of range")
        self.encoder = actor.history_encoder
        self.block_count = block_count
        self.normalize = normalize

    def forward(self, events: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        positions = torch.arange(events.shape[1], device=events.device)
        hidden = self.encoder._embed(events)
        for block in self.encoder.blocks[: self.block_count]:
            hidden = block(hidden, causal=True, positions=positions)
        return self.encoder.output_norm(hidden) if self.normalize else hidden


class _HistorySummary(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        indices = (lengths.long() - 1).clamp_min(0)
        summary = hidden[torch.arange(batch, device=hidden.device), indices]
        empty = self.encoder.empty_summary.unsqueeze(0).expand_as(summary)
        return torch.where((lengths > 0)[:, None], summary, empty)


class _UncheckedRawLogits(nn.Module):
    """Actor math with the tensor-valued history assertions omitted."""

    def __init__(self, actor: BloodFlowTransformer) -> None:
        super().__init__()
        self.actor = actor
        self.history = _HistoryCore(actor)

    def forward(
        self,
        tile_obs: torch.Tensor,
        melds: torch.Tensor,
        meta: torch.Tensor,
        events: torch.Tensor,
        lengths: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        del legal
        static = self.actor.static_encoder(tile_obs, melds, meta)
        history = self.history(events, lengths)
        return self.actor.actor(torch.cat((static, history), dim=-1))


class _ActorStages(nn.Module):
    """Return major intermediates from one full graph.

    Returning intermediates can inhibit fusion. A correct staged graph together
    with an incorrect normal graph is evidence of an Inductor scheduling or
    buffer-reuse error rather than an erroneous model operation.
    """

    def __init__(self, actor: BloodFlowTransformer) -> None:
        super().__init__()
        self.actor = actor
        self.static_tokens = _StaticTokens(actor)

    def forward(
        self,
        tile_obs: torch.Tensor,
        melds: torch.Tensor,
        meta: torch.Tensor,
        events: torch.Tensor,
        lengths: torch.Tensor,
        legal: torch.Tensor,
    ) -> TensorTuple:
        del legal
        static, key_valid = self.static_tokens(tile_obs, melds, meta)
        outputs = [static]
        for block in self.actor.static_encoder.blocks:
            static = block(static, causal=False, key_valid=key_valid)
            outputs.append(static)
        static = self.actor.static_encoder.output_norm(static[:, 0])
        outputs.append(static)

        positions = torch.arange(events.shape[1], device=events.device)
        history = self.actor.history_encoder._embed(events)
        outputs.append(history)
        for block in self.actor.history_encoder.blocks:
            history = block(history, causal=True, positions=positions)
            outputs.append(history)
        history = self.actor.history_encoder.output_norm(history)
        outputs.append(history)
        indices = (lengths.long() - 1).clamp_min(0)
        summary = history[torch.arange(history.shape[0], device=history.device), indices]
        empty = self.actor.history_encoder.empty_summary.unsqueeze(0).expand_as(summary)
        summary = torch.where((lengths > 0)[:, None], summary, empty)
        outputs.append(summary)

        hidden = torch.cat((static, summary), dim=-1)
        for layer in self.actor.actor:
            hidden = layer(hidden)
            outputs.append(hidden)
        return tuple(outputs)


def _evaluation_context(device: torch.device, precision: str) -> Any:
    if precision == "bf16":
        return _autocast(device)
    if precision == "fp32":
        return nullcontext()
    if precision == "native_bf16":
        return nullcontext()
    raise ValueError(f"unsupported precision: {precision}")


def _as_tuple(value: torch.Tensor | TensorTuple) -> TensorTuple:
    return value if isinstance(value, tuple) else (value,)


def _tensor_metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "expected_dtype": str(expected.dtype),
            "actual_dtype": str(actual.dtype),
        }
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    different = expected != actual
    result: dict[str, Any] = {
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "different": int(different.sum().item()),
        "expected_nonfinite": int((~torch.isfinite(expected)).sum().item()),
        "actual_nonfinite": int((~torch.isfinite(actual)).sum().item()),
    }
    if finite.any():
        delta = (expected.float() - actual.float()).abs()
        result["max_abs_delta"] = float(delta[finite].max().item())
        result["mean_abs_delta"] = float(delta[finite].mean().item())
    if different.any():
        result["first_different_flat_index"] = int(
            different.flatten().nonzero()[0].item()
        )
    return result


def _output_metrics(
    expected: torch.Tensor | TensorTuple,
    actual: torch.Tensor | TensorTuple,
) -> dict[str, Any]:
    expected_values = _as_tuple(expected)
    actual_values = _as_tuple(actual)
    if len(expected_values) != len(actual_values):
        return {
            "expected_outputs": len(expected_values),
            "actual_outputs": len(actual_values),
        }
    tensors = [
        _tensor_metrics(expected_tensor, actual_tensor)
        for expected_tensor, actual_tensor in zip(expected_values, actual_values)
    ]
    return {
        "outputs": tensors,
        "different": sum(int(item.get("different", 1)) for item in tensors),
        "actual_nonfinite": sum(
            int(item.get("actual_nonfinite", 0)) for item in tensors
        ),
        "max_abs_delta": max(
            (float(item.get("max_abs_delta", float("inf"))) for item in tensors),
            default=0.0,
        ),
    }


def _stage_names(actor: BloodFlowTransformer) -> list[str]:
    names = ["static.tokens"]
    names.extend(
        f"static.block.{index}" for index in range(len(actor.static_encoder.blocks))
    )
    names.extend(("static.summary", "history.tokens"))
    names.extend(
        f"history.block.{index}"
        for index in range(len(actor.history_encoder.blocks))
    )
    names.extend(
        (
            "history.normalized",
            "history.summary",
            "actor.normalized",
            "actor.hidden",
            "actor.activated",
            "actor.logits",
        )
    )
    return names


def _run_compiled(
    module: Callable[..., torch.Tensor | TensorTuple],
    inputs: TensorTuple,
    *,
    device: torch.device,
    precision: str,
    mode: str,
    options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], torch.Tensor | TensorTuple]:
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    with torch.inference_mode(), _evaluation_context(device, precision):
        expected = module(*inputs)
    expected = tuple(item.detach().clone() for item in expected) if isinstance(
        expected, tuple
    ) else expected.detach().clone()
    patch = inductor_config.patch(dict(options)) if options else nullcontext()
    with patch:
        compiled = torch.compile(
            module,
            mode=mode,
            fullgraph=True,
            dynamic=False,
        )
        with torch.inference_mode(), _evaluation_context(device, precision):
            actual = compiled(*inputs)
        _synchronize(device)
    return _output_metrics(expected, actual), actual


def _component_probes(
    actor: BloodFlowTransformer,
    inputs: TensorTuple,
    device: torch.device,
    precision: str,
) -> Iterator[Probe]:
    tile_obs, melds, meta, events, lengths, legal = inputs
    del lengths, legal
    tokens = _StaticTokens(actor).eval()
    with torch.inference_mode(), _evaluation_context(device, precision):
        static, key_valid = tokens(tile_obs, melds, meta)
    yield Probe("static.tokens", tokens, (tile_obs, melds, meta))
    for index, block in enumerate(actor.static_encoder.blocks):
        yield Probe(
            f"static.block.{index}.attention_norm",
            block.attention_norm,
            (static,),
        )
        with torch.inference_mode(), _evaluation_context(device, precision):
            normalized = block.attention_norm(static)
        yield Probe(
            f"static.block.{index}.attention",
            _StaticAttention(block.attention).eval(),
            (normalized, key_valid),
        )
        yield Probe(
            f"static.block.{index}",
            _StaticBlock(block).eval(),
            (static, key_valid),
        )
        with torch.inference_mode(), _evaluation_context(device, precision):
            attended = block.attention(
                normalized, causal=False, key_valid=key_valid
            )
            hidden = static + attended
            ffn_normalized = block.ffn_norm(hidden)
        yield Probe(
            f"static.block.{index}.ffn_norm", block.ffn_norm, (hidden,)
        )
        yield Probe(f"static.block.{index}.ffn", block.ffn, (ffn_normalized,))
        with torch.inference_mode(), _evaluation_context(device, precision):
            static = hidden + block.ffn(ffn_normalized)

    with torch.inference_mode(), _evaluation_context(device, precision):
        history = actor.history_encoder._embed(events)
        positions = torch.arange(events.shape[1], device=device)
    yield Probe("history.embed", actor.history_encoder._embed, (events,))
    for index, block in enumerate(actor.history_encoder.blocks):
        yield Probe(
            f"history.block.{index}.attention_norm",
            block.attention_norm,
            (history,),
        )
        with torch.inference_mode(), _evaluation_context(device, precision):
            normalized = block.attention_norm(history)
        yield Probe(
            f"history.block.{index}.attention",
            _HistoryAttention(block.attention).eval(),
            (normalized, positions),
        )
        yield Probe(
            f"history.block.{index}",
            _HistoryBlock(block).eval(),
            (history, positions),
        )
        with torch.inference_mode(), _evaluation_context(device, precision):
            attended = block.attention(
                normalized, causal=True, positions=positions
            )
            hidden = history + attended
            ffn_normalized = block.ffn_norm(hidden)
        yield Probe(
            f"history.block.{index}.ffn_norm", block.ffn_norm, (hidden,)
        )
        yield Probe(f"history.block.{index}.ffn", block.ffn, (ffn_normalized,))
        with torch.inference_mode(), _evaluation_context(device, precision):
            history = hidden + block.ffn(ffn_normalized)

    yield Probe("history.hidden_before_output_norm", nn.Identity(), (history,))
    with torch.inference_mode(), _evaluation_context(device, precision):
        history_normalized = actor.history_encoder.output_norm(history)
    yield Probe(
        "history.output_norm_on_hidden",
        actor.history_encoder.output_norm,
        (history,),
    )
    yield Probe(
        "history.summary_on_normalized",
        _HistorySummary(actor.history_encoder).eval(),
        (history_normalized, inputs[4]),
    )

    with torch.inference_mode(), _evaluation_context(device, precision):
        static_summary = actor.static_encoder(tile_obs, melds, meta)
        history_summary = actor.history_encoder(events, inputs[4])
        fused = torch.cat((static_summary, history_summary), dim=-1)
    yield Probe("static.encoder", actor.static_encoder, (tile_obs, melds, meta))
    yield Probe("history.encoder", actor.history_encoder, (events, inputs[4]))
    yield Probe(
        "history.core_without_assertions",
        _HistoryCore(actor).eval(),
        (events, inputs[4]),
    )
    for block_count in range(1, len(actor.history_encoder.blocks) + 1):
        yield Probe(
            f"history.prefix.{block_count}",
            _HistoryPrefix(actor, block_count).eval(),
            (events, inputs[4]),
        )
        yield Probe(
            f"history.prefix_hidden.{block_count}",
            _HistoryPrefixSequence(
                actor, block_count, normalize=False
            ).eval(),
            (events, inputs[4]),
        )
        yield Probe(
            f"history.prefix_normalized.{block_count}",
            _HistoryPrefixSequence(
                actor, block_count, normalize=True
            ).eval(),
            (events, inputs[4]),
        )
    yield Probe("actor.head", actor.actor, (fused,))
    yield Probe("actor.full_without_assertions", _UncheckedRawLogits(actor).eval(), inputs)
    yield Probe("actor.full", _RawLogits(actor).eval(), inputs)


CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no_epilogue_fusion": {
        "epilogue_fusion": False,
        "benchmark_epilogue_fusion": False,
    },
    "no_inplace_buffers": {"inplace_buffers": False},
    "no_layout_optimization": {"layout_optimization": False},
    "no_pattern_matcher": {"pattern_matcher": False},
    "small_fusion": {"max_fusion_size": 1},
    "fusion_2": {"max_fusion_size": 2},
    "fusion_4": {"max_fusion_size": 4},
    "fusion_8": {"max_fusion_size": 8},
    "fusion_16": {"max_fusion_size": 16},
    "fusion_32": {"max_fusion_size": 32},
    "aten_gemm": {
        "max_autotune": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN",
    },
    "triton_gemm": {
        "max_autotune": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
    },
}


def _print_result(name: str, result: Mapping[str, Any]) -> None:
    print(
        f"{name:42s} different={result['different']:9d} "
        f"nonfinite={result['actual_nonfinite']:7d} "
        f"max_delta={result['max_abs_delta']:.7g}",
        flush=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires CUDA")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    states = _load_states(args.corpus)
    rows = _bucket_rows(states, args.history, args.batch_size)
    if rows is None:
        raise ValueError("the corpus has no states in the requested history bucket")
    inputs = _inputs(states, rows, args.history, device)

    rows_out: list[dict[str, Any]] = []
    for precision in args.precisions:
        actor = load_policy(args.policy, device, frozen=True)
        if precision == "native_bf16":
            actor.to(dtype=torch.bfloat16)
        if "configs" in args.phases:
            config_targets: dict[str, tuple[nn.Module, TensorTuple]] = {
                "full": (_RawLogits(actor).eval(), inputs),
                "full_without_assertions": (
                    _UncheckedRawLogits(actor).eval(),
                    inputs,
                ),
                "static_encoder": (actor.static_encoder, inputs[:3]),
                "history_encoder": (actor.history_encoder, inputs[3:5]),
                "history_core": (_HistoryCore(actor).eval(), inputs[3:5]),
            }
            config_targets.update(
                {
                    f"history_prefix_{block_count}": (
                        _HistoryPrefix(actor, block_count).eval(),
                        inputs[3:5],
                    )
                    for block_count in range(1, len(actor.history_encoder.blocks) + 1)
                }
            )
            for target in args.config_targets:
                module, target_inputs = config_targets[target]
                for configuration in args.configurations:
                    metrics, actual = _run_compiled(
                        module,
                        target_inputs,
                        device=device,
                        precision=precision,
                        mode=args.mode,
                        options=CONFIGURATIONS[configuration],
                    )
                    if target == "full":
                        with (
                            torch.inference_mode(),
                            _evaluation_context(device, precision),
                        ):
                            expected = _RawLogits(actor).eval()(*inputs)
                        metrics["action_mismatches"] = int(
                            (
                                _actions(actual, inputs[-1])
                                != _actions(expected, inputs[-1])
                            ).sum().item()
                        )
                    name = f"config.{target}.{configuration}.{precision}"
                    _print_result(name, metrics)
                    if "action_mismatches" in metrics:
                        print(
                            f"{'':42s} "
                            f"action_mismatches={metrics['action_mismatches']}",
                            flush=True,
                        )
                    rows_out.append(
                        {
                            "phase": "configs",
                            "target": target,
                            "name": configuration,
                            "precision": precision,
                            **metrics,
                        }
                    )

        if "stages" in args.phases:
            metrics, _ = _run_compiled(
                _ActorStages(actor).eval(),
                inputs,
                device=device,
                precision=precision,
                mode=args.mode,
            )
            names = _stage_names(actor)
            if len(names) != len(metrics["outputs"]):
                raise RuntimeError("staged output names do not align with tensors")
            for name, output in zip(names, metrics["outputs"]):
                output["stage"] = name
            _print_result(f"stages.full_graph.{precision}", metrics)
            rows_out.append(
                {
                    "phase": "stages",
                    "name": "full_graph",
                    "precision": precision,
                    **metrics,
                }
            )

        if "components" in args.phases:
            for probe in _component_probes(actor, inputs, device, precision):
                if args.component_filters and not any(
                    fragment in probe.name for fragment in args.component_filters
                ):
                    continue
                metrics, _ = _run_compiled(
                    probe.module,
                    probe.inputs,
                    device=device,
                    precision=precision,
                    mode=args.mode,
                )
                _print_result(f"component.{probe.name}.{precision}", metrics)
                rows_out.append(
                    {
                        "phase": "components",
                        "name": probe.name,
                        "precision": precision,
                        **metrics,
                    }
                )

    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "policy": str(args.policy.resolve()),
        "corpus": [str(path.resolve()) for path in args.corpus],
        "batch_size": args.batch_size,
        "history": args.history,
        "mode": args.mode,
        "results": rows_out,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--mode", default="default")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("configs", "stages", "components"),
        default=("configs", "stages", "components"),
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("bf16", "native_bf16", "fp32"),
        default=("bf16",),
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CONFIGURATIONS),
        default=tuple(CONFIGURATIONS),
    )
    parser.add_argument(
        "--config-targets",
        nargs="+",
        choices=(
            "full",
            "full_without_assertions",
            "static_encoder",
            "history_encoder",
            "history_core",
            "history_prefix_1",
            "history_prefix_2",
            "history_prefix_3",
            "history_prefix_4",
            "history_prefix_5",
        ),
        default=("full",),
    )
    parser.add_argument("--component-filters", nargs="+", default=())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.history <= 0:
        raise ValueError("batch size and history must be positive")
    run(args)


if __name__ == "__main__":
    main()
