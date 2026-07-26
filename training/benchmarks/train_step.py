from __future__ import annotations

import argparse
import time

import torch

from training import BloodFlowTransformer
from training.benchmarks.transformer import inputs, synchronize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = BloodFlowTransformer().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    tile_obs, melds, meta, events, lengths, legal = inputs(
        args.batch_size, args.history, device
    )
    target_actions = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(tile_obs, melds, meta, events, lengths, legal)
            loss = torch.nn.functional.cross_entropy(
                output.raw_logits, target_actions
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(tile_obs, melds, meta, events, lengths, legal)
        loss = torch.nn.functional.cross_entropy(output.raw_logits, target_actions)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"device:       {device}")
    print(f"batch/history:{args.batch_size}/{args.history}")
    print(f"step seconds: {elapsed:.3f}")
    print(f"loss:         {loss.detach().item():.4f}")
    if device.type == "cuda":
        print(f"peak MiB:     {torch.cuda.max_memory_allocated(device) / 1024**2:.0f}")


if __name__ == "__main__":
    main()
