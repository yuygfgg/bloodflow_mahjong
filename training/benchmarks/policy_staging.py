"""Compare legacy bulk staging with double-buffered policy staging."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from training.model import ACTION_SPACE_SIZE
from training.observation import bucket_history_width
from training.pipeline import (
    EngineBuffers,
    _PinnedPolicyStager,
    _autocast,
    _bucket_inference_rows,
    _launch_policy_actions,
    clone_policy,
    load_policy,
)
from training.world_outcomes import (
    WorldOutcomeBatch,
    concatenate_world_outcome_batches,
    load_world_outcome_batch,
    load_world_outcome_corpus,
)


class _LegacyBulkStager:
    """The former one-transfer-per-step implementation for comparison only."""

    def __init__(self, device: torch.device, history: int) -> None:
        self.device = device
        self.history = history
        self.capacity = 0
        self.host: tuple[torch.Tensor, ...] = ()
        self.host_numpy: tuple[np.ndarray, ...] = ()
        self.staged: tuple[torch.Tensor, ...] = ()

    def _allocate(self, required: int) -> None:
        capacity = max(required, 32, 2 * self.capacity)
        specifications = (
            ((capacity, 10, 27), torch.uint8),
            ((capacity, 4, 4, 3), torch.uint8),
            ((capacity, 34), torch.int32),
            ((capacity, self.history, 8), torch.int32),
            ((capacity,), torch.int64),
            ((capacity, ACTION_SPACE_SIZE), torch.bool),
        )
        self.host = tuple(
            torch.empty(shape, dtype=dtype, pin_memory=True)
            for shape, dtype in specifications
        )
        self.host_numpy = tuple(value.numpy() for value in self.host)
        self.staged = tuple(
            torch.empty(shape, dtype=dtype, device=self.device)
            for shape, dtype in specifications
        )
        self.capacity = capacity

    def stage(
        self, buffers: EngineBuffers, rows: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if len(rows) > self.capacity:
            self._allocate(len(rows))
        sources = (
            buffers.tile_obs,
            buffers.melds,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
            buffers.legal,
        )
        for index, (source, target) in enumerate(zip(sources, self.host_numpy)):
            if index == 4:
                target[: len(rows)] = source[rows]
            else:
                np.take(source, rows, axis=0, out=target[: len(rows)])
        for host, staged in zip(self.host, self.staged):
            staged[: len(rows)].copy_(host[: len(rows)], non_blocking=True)
        return tuple(value[: len(rows)] for value in self.staged)


def _legacy_actions(
    model: torch.nn.Module,
    buffers: EngineBuffers,
    rows: np.ndarray,
    device: torch.device,
    *,
    inference_batch_size: int,
    stager: _LegacyBulkStager,
) -> torch.Tensor:
    chunks: list[tuple[int, int, int, int]] = []
    staged_rows: list[np.ndarray] = []
    offset = 0
    for start in range(0, len(rows), inference_batch_size):
        chunk = rows[start : start + inference_batch_size]
        inference_rows = _bucket_inference_rows(chunk)
        width = bucket_history_width(
            buffers.event_lengths[inference_rows], buffers.events.shape[1]
        )
        chunks.append((offset, len(inference_rows), len(chunk), width))
        staged_rows.append(inference_rows)
        offset += len(inference_rows)
    staged = stager.stage(buffers, np.concatenate(staged_rows))
    actions: list[torch.Tensor] = []
    with torch.inference_mode(), _autocast(device):
        for offset, padded, valid, width in chunks:
            stop = offset + padded
            logits = model(
                staged[0][offset:stop],
                staged[1][offset:stop],
                staged[2][offset:stop],
                staged[3][offset:stop, :width],
                staged[4][offset:stop],
                staged[5][offset:stop],
            ).logits[:valid]
            actions.append(logits.argmax(dim=-1).to(torch.uint8))
    return actions[0] if len(actions) == 1 else torch.cat(actions)


def _load_states(paths: Sequence[Path]) -> WorldOutcomeBatch:
    batches = [
        load_world_outcome_corpus(path)
        if path.is_dir()
        else load_world_outcome_batch(path)
        for path in paths
    ]
    return batches[0] if len(batches) == 1 else concatenate_world_outcome_batches(batches)


def _buffers(states: WorldOutcomeBatch, rows: int) -> EngineBuffers:
    indices = np.resize(np.arange(len(states), dtype=np.int64), rows)
    return EngineBuffers(
        batch=None,
        tile_obs=np.ascontiguousarray(states.tile_obs[indices]),
        melds=np.ascontiguousarray(states.melds[indices]),
        river=np.zeros((rows, 108, 2), dtype=np.uint8),
        meta=np.ascontiguousarray(states.meta[indices]),
        events=np.ascontiguousarray(states.events[indices]),
        event_lengths=np.ascontiguousarray(states.event_lengths[indices]),
        masks=np.zeros((rows, 2), dtype=np.uint64),
        legal=np.ascontiguousarray(states.legal[indices]),
        records=np.zeros((rows, 12), dtype=np.int64),
        actions=np.zeros(rows, dtype=np.uint8),
    )


def _measure(function, *, rows: int, warmup: int, iterations: int, device: torch.device):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize(device)
    return rows * iterations / (time.perf_counter() - started)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.rows <= 0 or args.inference_batch_size <= 0:
        raise ValueError("rows and inference batch size must be positive")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    actor = load_policy(args.policy, device, frozen=True)
    states = _load_states(args.corpus)
    buffers = _buffers(states, args.rows)
    rows = np.arange(args.rows, dtype=np.int64)

    legacy_stager = _LegacyBulkStager(device, actor.config.max_history)
    modern_stager = _PinnedPolicyStager(device, actor.config.max_history)
    expected = _legacy_actions(
        actor,
        buffers,
        rows,
        device,
        inference_batch_size=args.inference_batch_size,
        stager=legacy_stager,
    )
    torch.cuda.synchronize(device)
    actual = _launch_policy_actions(
        actor,
        buffers,
        rows,
        device,
        inference_batch_size=args.inference_batch_size,
        stager=modern_stager,
    )
    torch.cuda.synchronize(device)
    mismatch = int((expected != actual).sum().item())
    legacy_rate = _measure(
        lambda: _legacy_actions(
            actor,
            buffers,
            rows,
            device,
            inference_batch_size=args.inference_batch_size,
            stager=legacy_stager,
        ),
        rows=args.rows,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    modern_rate = _measure(
        lambda: _launch_policy_actions(
            actor,
            buffers,
            rows,
            device,
            inference_batch_size=args.inference_batch_size,
            stager=modern_stager,
        ),
        rows=args.rows,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    left_rows = rows[::2]
    right_rows = rows[1::2]
    opponent = clone_policy(actor, device)
    left_legacy = _LegacyBulkStager(device, actor.config.max_history)
    right_legacy = _LegacyBulkStager(device, actor.config.max_history)
    paired_expected = torch.cat(
        (
            _legacy_actions(
                actor,
                buffers,
                left_rows,
                device,
                inference_batch_size=args.inference_batch_size,
                stager=left_legacy,
            ),
            _legacy_actions(
                opponent,
                buffers,
                right_rows,
                device,
                inference_batch_size=args.inference_batch_size,
                stager=right_legacy,
            ),
        )
    )
    torch.cuda.synchronize(device)
    paired_stager = _PinnedPolicyStager(device, actor.config.max_history)
    paired_actual = torch.cat(
        (
            _launch_policy_actions(
                actor,
                buffers,
                left_rows,
                device,
                inference_batch_size=args.inference_batch_size,
                stager=paired_stager,
            ),
            _launch_policy_actions(
                opponent,
                buffers,
                right_rows,
                device,
                inference_batch_size=args.inference_batch_size,
                stager=paired_stager,
            ),
        )
    )
    torch.cuda.synchronize(device)
    paired_mismatch = int((paired_expected != paired_actual).sum().item())
    print(
        f"single rows={args.rows:,} batch={args.inference_batch_size} "
        f"legacy={legacy_rate:,.0f}/s modern={modern_rate:,.0f}/s "
        f"speedup={modern_rate / legacy_rate:.2f}x mismatch={mismatch}",
        flush=True,
    )
    print(
        f"paired rows={args.rows:,} shared-stager mismatch={paired_mismatch}",
        flush=True,
    )
    if mismatch or paired_mismatch:
        raise RuntimeError("double-buffered staging changed greedy actions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
