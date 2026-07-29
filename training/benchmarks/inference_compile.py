"""Benchmark compiled Actor inference on cached real policy states."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from training.observation import bucket_history_width
from training.pipeline import _autocast, load_policy
from training.policy_iteration import (
    CounterfactualBatch,
    PolicyStateBatch,
    load_counterfactual_batch,
    load_policy_state_batch,
)
from training.world_outcomes import (
    WorldOutcomeBatch,
    load_world_outcome_batch,
    load_world_outcome_corpus,
)


class _RawLogits(nn.Module):
    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(
        self,
        tile_obs: torch.Tensor,
        melds: torch.Tensor,
        meta: torch.Tensor,
        events: torch.Tensor,
        lengths: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor(tile_obs, melds, meta, events, lengths, legal).raw_logits


def _load_state_file(path: Path) -> PolicyStateBatch:
    with np.load(path, allow_pickle=False) as payload:
        fields = set(payload.files)
    candidates = (
        (set(WorldOutcomeBatch.__dataclass_fields__) | {"version"}, load_world_outcome_batch),
        (
            set(CounterfactualBatch.__dataclass_fields__) | {"version"},
            load_counterfactual_batch,
        ),
        (set(PolicyStateBatch.__dataclass_fields__) | {"version"}, load_policy_state_batch),
    )
    for expected, loader in candidates:
        if fields == expected:
            return loader(path)
    raise ValueError(f"{path} is not a supported policy-state cache")


def _load_states(paths: Sequence[Path]) -> PolicyStateBatch:
    batches = [
        load_world_outcome_corpus(path) if path.is_dir() else _load_state_file(path)
        for path in paths
    ]
    if len(batches) == 1:
        return batches[0]
    fields = PolicyStateBatch.__dataclass_fields__
    return PolicyStateBatch(
        **{
            name: np.concatenate([getattr(batch, name) for batch in batches])
            for name in fields
        }
    )


def _parse_positive_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return result


def _bucket_rows(
    states: WorldOutcomeBatch, history: int, batch: int
) -> np.ndarray | None:
    widths = np.asarray(
        [
            bucket_history_width(np.asarray([length]), states.events.shape[1])
            for length in states.event_lengths
        ],
        dtype=np.int64,
    )
    candidates = np.flatnonzero(widths == history)
    if not len(candidates):
        return None
    return np.resize(candidates, batch)


def _inputs(
    states: WorldOutcomeBatch,
    rows: np.ndarray,
    history: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(states.tile_obs[rows], device=device),
        torch.as_tensor(states.melds[rows], device=device),
        torch.as_tensor(states.meta[rows], device=device),
        torch.as_tensor(states.events[rows, :history], device=device),
        torch.as_tensor(
            states.event_lengths[rows].astype(np.int64), device=device
        ),
        torch.as_tensor(states.legal[rows], device=device),
    )


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _measure(
    function: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    batch: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    with torch.inference_mode(), _autocast(device):
        for _ in range(warmup):
            function(*inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            function(*inputs)
        _synchronize(device)
    return batch * iterations / (time.perf_counter() - started)


def _actions(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~legal, -torch.inf).argmax(dim=-1)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    actor = load_policy(args.policy, device, frozen=True)
    states = _load_states(args.corpus)
    if states.events.shape[1] != actor.config.max_history:
        raise ValueError("cached states and Actor history capacities differ")
    shapes: list[tuple[int, int, tuple[torch.Tensor, ...]]] = []
    for history in args.histories:
        if history > actor.config.max_history:
            raise ValueError("a requested history exceeds the Actor capacity")
        for batch in args.batch_sizes:
            rows = _bucket_rows(states, history, batch)
            if rows is not None:
                shapes.append((batch, history, _inputs(states, rows, history, device)))
    if not shapes:
        raise ValueError("cached states do not cover any requested history bucket")

    wrapper = _RawLogits(actor).eval()
    eager_logits: dict[tuple[int, int], torch.Tensor] = {}
    eager_rates: dict[tuple[int, int], float] = {}
    for batch, history, inputs in shapes:
        with torch.inference_mode(), _autocast(device):
            eager_logits[(batch, history)] = wrapper(*inputs).detach().clone()
        eager_rates[(batch, history)] = _measure(
            wrapper,
            inputs,
            batch=batch,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )

    results: list[dict[str, object]] = []
    for mode in args.modes:
        for batch, history, inputs in shapes:
            gc.collect()
            torch.cuda.empty_cache()
            torch._dynamo.reset()
            _synchronize(device)
            started = time.perf_counter()
            compiled = None
            try:
                if mode == "trace":
                    with torch.inference_mode(), _autocast(device):
                        compiled = torch.jit.trace(
                            wrapper,
                            inputs,
                            check_trace=False,
                            strict=True,
                        )
                elif mode == "export":
                    with torch.inference_mode(), _autocast(device):
                        compiled = torch.export.export(
                            wrapper,
                            inputs,
                            strict=True,
                        ).module()
                elif mode in {"dynamo-eager", "aot-eager", "cudagraphs"}:
                    backend = {
                        "dynamo-eager": "eager",
                        "aot-eager": "aot_eager",
                        "cudagraphs": "cudagraphs",
                    }[mode]
                    compiled = torch.compile(
                        wrapper,
                        backend=backend,
                        fullgraph=True,
                        dynamic=args.dynamic,
                    )
                elif mode in {"small-fusion", "fusion-2", "fusion-4"}:
                    maximum_fusion = {
                        "small-fusion": 1,
                        "fusion-2": 2,
                        "fusion-4": 4,
                    }[mode]
                    compiled = torch.compile(
                        wrapper,
                        fullgraph=True,
                        dynamic=args.dynamic,
                        options={"max_fusion_size": maximum_fusion},
                    )
                else:
                    compiled = torch.compile(
                        wrapper,
                        mode=mode,
                        fullgraph=True,
                        dynamic=args.dynamic,
                    )
                with torch.inference_mode(), _autocast(device):
                    actual = compiled(*inputs)
                _synchronize(device)
                cold_seconds = time.perf_counter() - started
                expected = eager_logits[(batch, history)]
                mismatch = int(
                    (
                        _actions(actual, inputs[-1])
                        != _actions(expected, inputs[-1])
                    ).sum().item()
                )
                nonfinite = int((~torch.isfinite(actual)).sum().item())
                max_logit_delta = float(
                    (actual.float() - expected.float()).abs().max().item()
                )
                rate = _measure(
                    compiled,
                    inputs,
                    batch=batch,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                row = {
                    "mode": mode,
                    "dynamic": bool(args.dynamic),
                    "batch": batch,
                    "history": history,
                    "cold_seconds": cold_seconds,
                    "eager_states_per_second": eager_rates[(batch, history)],
                    "compiled_states_per_second": rate,
                    "speedup": rate / eager_rates[(batch, history)],
                    "action_mismatches": mismatch,
                    "nonfinite_logits": nonfinite,
                    "max_logit_delta": max_logit_delta,
                }
            except Exception as error:
                row = {
                    "mode": mode,
                    "dynamic": bool(args.dynamic),
                    "batch": batch,
                    "history": history,
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(row)
            if "error" in row:
                print(
                    f"{mode:15s} batch/history={batch:3d}/{history:3d} "
                    f"error={row['error']}",
                    flush=True,
                )
            else:
                print(
                    f"{mode:15s} dynamic={str(args.dynamic):5s} "
                    f"batch/history={batch:3d}/{history:3d} "
                    f"cold={row['cold_seconds']:7.2f}s "
                    f"eager={row['eager_states_per_second']:8.0f}/s "
                    f"compiled={row['compiled_states_per_second']:8.0f}/s "
                    f"speedup={row['speedup']:.2f}x "
                    f"mismatch={row['action_mismatches']} "
                    f"nonfinite={row['nonfinite_logits']} "
                    f"max_delta={row['max_logit_delta']:.6g}",
                    flush=True,
                )
            del compiled
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "policy": str(args.policy.resolve()),
        "states": len(states),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "results": results,
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
    parser.add_argument(
        "--modes", nargs="+", default=("default", "reduce-overhead")
    )
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-sizes", type=_parse_positive_csv, default=(32, 64, 128))
    parser.add_argument(
        "--histories", type=_parse_positive_csv, default=(8, 16, 32, 64, 128, 192)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    run(args)


if __name__ == "__main__":
    main()
