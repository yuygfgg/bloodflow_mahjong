"""Behavior cloning from the deterministic rule-EV policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import bloodflow_mahjong as bm
import numpy as np
import torch
import torch.nn.functional as F

from .contracts import MODEL_HISTORY_CAPACITY, TRAINING_INPUT_SCHEMA, validate_engine_contract
from .device import stage_numpy_batch
from .model import BloodFlowTransformer, TransformerConfig
from .pipeline import EngineBuffers
from .policy import actor_parameters, save_actor_checkpoint
from .reporting import append_jsonl, format_duration, format_percent, format_rate


@dataclass(frozen=True)
class SupervisedConfig:
    envs: int = 4_096
    batch_labels: int = 65_536
    minibatch: int = 4_096
    microbatch: int = 512
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    exploration: float = 0.15
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        sizes = (self.envs, self.batch_labels, self.minibatch, self.microbatch)
        if any(value <= 0 for value in sizes) or self.epochs <= 0:
            raise ValueError("supervised batch sizes and epochs must be positive")
        if self.microbatch > self.minibatch:
            raise ValueError("microbatch cannot exceed minibatch")
        if self.batch_labels <= self.minibatch:
            raise ValueError("batch_labels must leave one validation minibatch")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer values are invalid")
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")


@dataclass
class SupervisedBatch:
    tile_obs: np.ndarray
    melds: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray
    legal: np.ndarray
    actions: np.ndarray

    @classmethod
    def allocate(cls, size: int, history: int) -> SupervisedBatch:
        return cls(
            tile_obs=np.empty(
                (size, bm.TILE_OBSERVATION_PLANES, bm.TILE_KIND_COUNT),
                dtype=np.uint8,
            ),
            melds=np.empty(
                (size, bm.PLAYER_COUNT, bm.MELD_SLOTS, bm.MELD_FIELDS),
                dtype=np.uint8,
            ),
            meta=np.empty((size, bm.META_OBSERVATION_WIDTH), dtype=np.int32),
            events=np.empty((size, history, bm.EVENT_RECORD_WIDTH), dtype=np.int32),
            event_lengths=np.empty(size, dtype=np.uint16),
            legal=np.empty((size, bm.ACTION_SPACE_SIZE), dtype=np.bool_),
            actions=np.empty(size, dtype=np.uint8),
        )

    def __len__(self) -> int:
        return len(self.actions)

    def tensors(
        self, indices: np.ndarray, device: torch.device
    ) -> dict[str, torch.Tensor]:
        lengths = self.event_lengths[indices].astype(np.int64)
        width = max(int(lengths.max(initial=0)), 1)
        return stage_numpy_batch(
            {
                "tile_obs": self.tile_obs[indices],
                "melds": self.melds[indices],
                "meta": self.meta[indices],
                "events": self.events[indices, :width],
                "event_lengths": lengths,
                "legal": self.legal[indices],
                "actions": self.actions[indices].astype(np.int64),
            },
            device,
        )

    def copy_from(
        self,
        destination: slice,
        buffers: EngineBuffers,
        rows: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        self.tile_obs[destination] = buffers.tile_obs[rows]
        self.melds[destination] = buffers.melds[rows]
        self.meta[destination] = buffers.meta[rows]
        self.events[destination] = buffers.events[rows]
        self.event_lengths[destination] = buffers.event_lengths[rows]
        self.legal[destination] = buffers.legal[rows]
        self.actions[destination] = actions[rows]


@dataclass(frozen=True)
class _CollectedBatch:
    batch: SupervisedBatch
    collect_seconds: float
    collector_wait_seconds: float


def _compact_record(record: dict[str, Any]) -> str:
    labels = int(record["labels"])
    target = int(record["target_labels"])
    elapsed = float(record["elapsed_seconds"])
    rate = labels / elapsed if elapsed > 0.0 else None
    eta = max(target - labels, 0) / rate if rate is not None and rate > 0.0 else None
    progress = 100.0 * labels / target
    return (
        f"SL  {labels:,}/{target:,} {progress:.1f}%"
        f"  {format_rate(rate, 'labels')}"
        f"  elapsed {format_duration(elapsed)}"
        f"  ETA {format_duration(eta)}"
        f"  loss {float(record['loss']):.3f}"
        f"  acc {format_percent(record['train_accuracy'])}"
        f"  val-loss {float(record['validation_loss']):.3f}"
        f"  val-acc {format_percent(record['validation_accuracy'])}"
        f"  grad {float(record['grad_norm']):.3f}"
    )


def _append_record(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)
    print(_compact_record(record), flush=True)


class RuleEvCollector:
    """Collect current-actor states while rule-EV controls all four seats."""

    def __init__(
        self,
        envs: int,
        *,
        history: int = MODEL_HISTORY_CAPACITY,
        seed: int = 1,
    ) -> None:
        validate_engine_contract()
        if envs <= 0 or not 0 < history <= int(bm.EVENT_HISTORY_CAPACITY):
            raise ValueError("collector dimensions are invalid")
        self.buffers = EngineBuffers.create(envs, history=history)
        self.history_seat_masks = np.full(envs, 0x0F, dtype=np.uint8)
        self.reset_flags = np.ones(envs, dtype=np.uint8)
        self.reset_seeds = np.empty(envs, dtype=np.uint64)
        self.teacher_actions = np.empty(envs, dtype=np.uint8)
        self.random = np.random.default_rng(seed)
        self.teacher_config = bm.RuleEvConfig.standard()
        self._reset(np.arange(envs, dtype=np.int64))

    def _reset(self, rows: np.ndarray) -> None:
        if not len(rows):
            return
        self.reset_flags[rows] = 1
        self.reset_seeds[rows] = self.random.integers(
            0, np.iinfo(np.uint64).max, size=len(rows), dtype=np.uint64
        )
        self.buffers.batch.reset_and_observe_history_into(
            self.reset_flags,
            self.reset_seeds,
            self.history_seat_masks,
            self.buffers.masks,
            self.buffers.tile_obs,
            self.buffers.melds,
            self.buffers.river,
            self.buffers.meta,
            self.buffers.events,
            self.buffers.event_lengths,
        )
        self.reset_flags[rows] = 0
        self.buffers.refresh_legal(rows)

    def _behavior_actions(self, exploration: float) -> np.ndarray:
        actions = self.teacher_actions.copy()
        explore = np.flatnonzero(self.random.random(len(actions)) < exploration)
        if len(explore):
            scores = self.random.random((len(explore), bm.ACTION_SPACE_SIZE))
            scores[~self.buffers.legal[explore]] = -1.0
            actions[explore] = scores.argmax(axis=1).astype(np.uint8)
        return actions

    def collect(self, labels: int, exploration: float) -> SupervisedBatch:
        if labels <= 0 or not 0.0 <= exploration <= 1.0:
            raise ValueError("collection settings are invalid")
        result = SupervisedBatch.allocate(labels, history=self.buffers.events.shape[1])
        cursor = 0
        rows = np.arange(len(self.buffers.batch), dtype=np.int64)
        while cursor < labels:
            self.buffers.batch.rule_ev_actions_into(
                self.teacher_actions, self.teacher_config
            )
            valid = self.teacher_actions < bm.ACTION_SPACE_SIZE
            bounded_actions = np.minimum(
                self.teacher_actions, np.uint8(bm.ACTION_SPACE_SIZE - 1)
            )
            legal = valid & self.buffers.legal[
                rows, bounded_actions.astype(np.int64, copy=False)
            ]
            if not np.all(legal):
                invalid = np.flatnonzero(~legal)[:8].tolist()
                raise RuntimeError(f"rule-EV returned invalid actions for rows {invalid}")

            choice_rows = rows[self.buffers.legal.sum(axis=1) > 1]
            count = min(labels - cursor, len(choice_rows))
            selected = choice_rows[:count]
            result.copy_from(
                slice(cursor, cursor + count),
                self.buffers,
                selected,
                self.teacher_actions,
            )
            cursor += count

            self.buffers.actions[:] = self._behavior_actions(exploration)
            self.buffers.batch.step_and_observe_history_into(
                self.buffers.actions,
                self.history_seat_masks,
                self.buffers.records,
                self.buffers.masks,
                self.buffers.tile_obs,
                self.buffers.melds,
                self.buffers.river,
                self.buffers.meta,
                self.buffers.events,
                self.buffers.event_lengths,
            )
            self.buffers.refresh_legal()
            self._reset(np.flatnonzero(self.buffers.records[:, 11]))
        return result


def _batch_sizes(
    labels: int,
    batch_labels: int,
    minibatch: int,
) -> list[int]:
    """Plan batches while keeping the final validation split non-empty."""
    if labels <= 0 or batch_labels <= minibatch or minibatch <= 0:
        raise ValueError("supervised label and batch sizes are invalid")

    result: list[int] = []
    remaining = labels
    while remaining:
        size = min(batch_labels, remaining)
        if remaining > batch_labels and remaining - batch_labels <= minibatch:
            size = remaining
        if size <= minibatch:
            raise ValueError("the final batch must contain more than one minibatch")
        result.append(size)
        remaining -= size
    return result


def _collect_timed(
    collector: RuleEvCollector,
    labels: int,
    exploration: float,
) -> tuple[SupervisedBatch, float]:
    started = time.perf_counter()
    batch = collector.collect(labels, exploration)
    return batch, time.perf_counter() - started


def _prefetched_batches(
    collector: RuleEvCollector,
    batch_sizes: list[int],
    exploration: float,
) -> Iterator[_CollectedBatch]:
    """Collect one batch ahead while the caller trains the current batch."""
    if not batch_sizes:
        return

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="rule-ev-sl"
    ) as executor:
        sizes = iter(batch_sizes)
        pending = executor.submit(_collect_timed, collector, next(sizes), exploration)
        try:
            while pending is not None:
                wait_started = time.perf_counter()
                batch, collect_seconds = pending.result()
                collector_wait_seconds = time.perf_counter() - wait_started
                try:
                    next_size = next(sizes)
                except StopIteration:
                    pending = None
                else:
                    pending = executor.submit(
                        _collect_timed, collector, next_size, exploration
                    )
                yield _CollectedBatch(
                    batch=batch,
                    collect_seconds=collect_seconds,
                    collector_wait_seconds=collector_wait_seconds,
                )
        finally:
            if pending is not None:
                pending.cancel()


def _model_inputs(
    data: dict[str, torch.Tensor], rows: slice, history_width: int
) -> tuple[torch.Tensor, ...]:
    lengths = data["event_lengths"][rows]
    return (
        data["tile_obs"][rows],
        data["melds"][rows],
        data["meta"][rows],
        data["events"][rows, : max(history_width, 1)],
        lengths,
        data["legal"][rows],
    )


def _validation_metrics(
    model: BloodFlowTransformer,
    batch: SupervisedBatch,
    indices: np.ndarray,
    microbatch: int,
    device: torch.device,
) -> dict[str, float]:
    if len(indices) == 0:
        raise ValueError("supervised validation has no decisions with a choice")
    model.eval()
    data = batch.tensors(indices, device)
    host_lengths = batch.event_lengths[indices]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    phase_ids = torch.tensor(
        (
            bm.PHASE_EXCHANGE,
            bm.PHASE_CHOOSE_MISSING,
            bm.PHASE_TURN,
            bm.PHASE_HU_RESPONSE,
            bm.PHASE_MELD_RESPONSE,
        ),
        device=device,
        dtype=torch.int32,
    )
    phase_total = torch.zeros(len(phase_ids), device=device, dtype=torch.int64)
    phase_correct = torch.zeros(len(phase_ids), device=device, dtype=torch.int64)
    with torch.inference_mode():
        for start in range(0, len(indices), microbatch):
            stop = min(start + microbatch, len(indices))
            rows = slice(start, stop)
            history_width = int(host_lengths[start:stop].max(initial=0))
            targets = data["actions"][rows]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(*_model_inputs(data, rows, history_width)).logits.float()
                loss = F.cross_entropy(logits, targets)
            predicted = logits.argmax(dim=-1)
            is_correct = predicted == targets
            loss_sum.add_(loss.detach().double(), alpha=stop - start)
            correct.add_(is_correct.sum())
            phase_matches = data["meta"][rows, 0, None] == phase_ids[None, :]
            phase_total.add_(phase_matches.sum(dim=0))
            phase_correct.add_((phase_matches & is_correct[:, None]).sum(dim=0))

    summary = torch.cat(
        (
            loss_sum[None],
            correct.double()[None],
            phase_total.double(),
            phase_correct.double(),
        )
    ).cpu().numpy()

    result = {
        "validation_loss": float(summary[0]) / len(indices),
        "validation_accuracy": float(summary[1]) / len(indices),
    }
    phase_names = (
        "exchange",
        "choose_missing",
        "turn",
        "hu_response",
        "meld_response",
    )
    totals = summary[2 : 2 + len(phase_names)]
    correct_by_phase = summary[2 + len(phase_names) :]
    for name, total, phase_hits in zip(
        phase_names, totals, correct_by_phase, strict=True
    ):
        if total:
            result[f"validation_accuracy_{name}"] = float(phase_hits / total)
    return result


def supervised_update(
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    batch: SupervisedBatch,
    config: SupervisedConfig,
    device: torch.device,
    random_generator: np.random.Generator,
) -> dict[str, float]:
    if len(batch) <= config.minibatch:
        raise ValueError("a supervised batch must contain more than one minibatch")
    if np.any(batch.legal.sum(axis=1) <= 1):
        raise ValueError("a supervised batch cannot contain forced decisions")
    shuffled = random_generator.permutation(len(batch))
    validation = shuffled[: config.minibatch]
    training = shuffled[config.minibatch :]
    parameters = list(actor_parameters(model))
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    seen = 0
    grad_norm_sum = torch.zeros((), device=device, dtype=torch.float64)
    updates = 0
    model.train()

    for _ in range(config.epochs):
        epoch_indices = random_generator.permutation(training)
        for start in range(0, len(epoch_indices), config.minibatch):
            minibatch = epoch_indices[start : start + config.minibatch]
            data = batch.tensors(minibatch, device)
            host_lengths = batch.event_lengths[minibatch]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(minibatch), config.microbatch):
                micro_stop = min(micro_start + config.microbatch, len(minibatch))
                rows = slice(micro_start, micro_stop)
                history_width = int(
                    host_lengths[micro_start:micro_stop].max(initial=0)
                )
                targets = data["actions"][rows]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(
                        *_model_inputs(data, rows, history_width)
                    ).logits.float()
                    loss = F.cross_entropy(logits, targets)
                micro_size = micro_stop - micro_start
                (loss * (micro_size / len(minibatch))).backward()
                loss_sum.add_(loss.detach().double(), alpha=micro_size)
                correct.add_((logits.detach().argmax(dim=-1) == targets).sum())
                seen += micro_size
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.max_grad_norm
            )
            optimizer.step()
            grad_norm_sum.add_(grad_norm.detach().double())
            updates += 1

    summary = torch.stack((loss_sum, correct.double(), grad_norm_sum)).cpu().numpy()
    return {
        "loss": float(summary[0]) / seen,
        "train_accuracy": float(summary[1]) / seen,
        "train_rows": float(seen),
        "optimizer_steps": float(updates),
        "grad_norm": float(summary[2]) / updates,
    } | _validation_metrics(
        model, batch, validation, config.microbatch, device
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rule-ev-sl"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--labels", type=int, default=10_000_000)
    parser.add_argument("--envs", type=int, default=4_096)
    parser.add_argument("--batch-labels", type=int, default=65_536)
    parser.add_argument("--minibatch", type=int, default=4_096)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--exploration", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--model-size", choices=("base", "large"), default="base")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _model_config(size: str) -> TransformerConfig:
    if size == "large":
        return TransformerConfig(
            d_model=256,
            num_heads=8,
            static_layers=3,
            history_layers=5,
            ffn_dim=1_024,
        )
    return TransformerConfig()


def run(args: argparse.Namespace) -> None:
    validate_engine_contract()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.labels <= 0:
        raise ValueError("labels must be positive")

    config = SupervisedConfig(
        envs=args.envs,
        batch_labels=args.batch_labels,
        minibatch=args.minibatch,
        microbatch=args.microbatch,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        exploration=args.exploration,
        max_grad_norm=args.max_grad_norm,
    )
    model_config = _model_config(args.model_size)
    if args.smoke:
        config = replace(
            config,
            envs=4,
            batch_labels=32,
            minibatch=8,
            microbatch=4,
        )
        model_config = TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            value_atoms=17,
        )
        args.labels = 64

    model = BloodFlowTransformer(model_config).to(device)
    parameters = list(actor_parameters(model))
    optimizer_kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (0.9, 0.95),
        "eps": 1e-5,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
    collector = RuleEvCollector(
        config.envs,
        history=model.config.max_history,
        seed=args.seed + 1,
    )
    training_random = np.random.default_rng(args.seed + 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.iterdir())
    if existing:
        raise FileExistsError(f"new output directory is not empty: {args.output_dir}")
    metrics_path = args.output_dir / "metrics.jsonl"
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "supervised": asdict(config),
                "model": asdict(model.config),
                "training_input_schema": TRAINING_INPUT_SCHEMA,
                "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
                "args": vars(args) | {"output_dir": str(args.output_dir)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    batch_sizes = _batch_sizes(args.labels, config.batch_labels, config.minibatch)
    completed = 0
    start = time.monotonic()
    with closing(
        _prefetched_batches(collector, batch_sizes, config.exploration)
    ) as batches:
        while True:
            pipeline_start = time.perf_counter()
            try:
                collected = next(batches)
            except StopIteration:
                break
            batch = collected.batch
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_start = time.perf_counter()
            statistics = supervised_update(
                model, optimizer, batch, config, device, training_random
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - train_start
            completed += len(batch)
            save_actor_checkpoint(
                args.output_dir / "latest.actor.pt",
                model,
                metadata={
                    "supervised_labels": completed,
                    "teacher": "rule_ev",
                    "exploration": config.exploration,
                },
            )
            pipeline_seconds = time.perf_counter() - pipeline_start
            record = {
                "phase": "supervised",
                "labels": completed,
                "target_labels": args.labels,
                "elapsed_seconds": time.monotonic() - start,
                "collect_seconds": collected.collect_seconds,
                "collector_wait_seconds": collected.collector_wait_seconds,
                "train_seconds": train_seconds,
                "pipeline_seconds": pipeline_seconds,
                "teacher": "rule_ev",
                "exploration": config.exploration,
                **statistics,
            }
            _append_record(metrics_path, record)

    save_actor_checkpoint(
        args.output_dir / "actor.pt",
        model,
        metadata={
            "supervised_labels": completed,
            "teacher": "rule_ev",
            "exploration": config.exploration,
        },
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
