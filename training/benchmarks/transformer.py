#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from training import BloodFlowTransformer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inputs(batch: int, history: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    tile_obs = torch.randint(0, 5, (batch, 10, 27), dtype=torch.uint8, device=device)
    melds = torch.full((batch, 4, 4, 3), 255, dtype=torch.uint8, device=device)
    meta = torch.zeros((batch, 34), dtype=torch.int32, device=device)
    meta[:, 4] = 30
    meta[:, 9] = history
    meta[:, 12:16] = 10_000
    meta[:, 24:28] = 14
    events = torch.zeros((batch, history, 8), dtype=torch.int32, device=device)
    positions = torch.arange(history, device=device)
    events[:, :, 0] = positions % 11
    events[:, :, 1] = positions % 4
    events[:, :, 2] = (positions + 1) % 4
    events[:, :, 3] = positions % 27
    events[:, :, 4] = positions % 8
    events[:, :, 5] = positions * 100
    events[:, :, 6] = positions
    lengths = torch.full((batch,), history, dtype=torch.int64, device=device)
    legal = torch.ones((batch, 115), dtype=torch.bool, device=device)
    return tile_obs, melds, meta, events, lengths, legal


def measure(
    function: Callable[[], object],
    iterations: int,
    warmup: int,
    batch: int,
    device: torch.device,
) -> float:
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for _ in range(warmup):
            function()
        synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        synchronize(device)
    elapsed = time.perf_counter() - started
    return batch * iterations / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history", type=int, default=192)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if not 1 <= args.history <= 192:
        parser.error("--history must be in 1..192")
    if min(args.batch_size, args.iterations, args.warmup) <= 0:
        parser.error("batch size, iterations, and warmup must be positive")

    device = torch.device(args.device)
    model = BloodFlowTransformer().to(device).eval()
    tile_obs, melds, meta, events, lengths, legal = inputs(
        args.batch_size, args.history, device
    )

    def full_forward() -> object:
        return model(tile_obs, melds, meta, events, lengths, legal)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full_rate = measure(
        full_forward, args.iterations, args.warmup, args.batch_size, device
    )
    full_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device:          {device}")
    print(f"parameters:      {parameters:,}")
    print(f"batch/history:   {args.batch_size}/{args.history}")
    print(f"full states/s:   {full_rate:,.0f}")
    if device.type == "cuda":
        print(f"full peak MiB:   {full_memory:,.0f}")


if __name__ == "__main__":
    main()
