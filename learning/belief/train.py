"""Train and export the learned planner belief residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .data import BeliefDataset, DatasetManifest
from .model import BeliefModelConfig, BeliefResidualModel
from .objective import GroupMetrics, grouped_loss, metrics

BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise FileExistsError(f"training output path is not a directory: {path}")
    if next(path.iterdir(), None) is not None:
        raise FileExistsError(f"training output directory is not empty: {path}")


def _save_safetensors(tensors: dict[str, Tensor], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("install the Python safetensors package") from error
    contiguous = {
        name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()
    }
    save_file(contiguous, path)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collision_counts(worlds: Tensor, positive: Tensor) -> tuple[int, int]:
    proposal_collisions = 0
    flattened = worlds.reshape(worlds.shape[0], worlds.shape[1], -1).cpu().numpy()
    for row in flattened:
        _, counts = np.unique(row, axis=0, return_counts=True)
        proposal_collisions += int((counts - 1).sum())
    positive_collisions = int((positive.sum(dim=1) - 1).sum().item())
    return proposal_collisions, positive_collisions


def _support_audit(
    dataset: BeliefDataset,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float | int], frozenset[int]]:
    roots = 0
    candidates = 0
    nonfinite = 0
    positive_nonfinite = 0
    all_nonfinite = 0
    collisions = 0
    positive_collisions = 0
    block_ids: set[int] = set()
    for batch in dataset.batches(
        split, batch_size, device=device, shuffle=False, seed=0
    ):
        weights = batch["handwritten_log_weights"]
        positive = batch["positive_mask"].bool()
        finite = torch.isfinite(weights)
        roots += weights.shape[0]
        candidates += weights.numel()
        nonfinite += int((~finite).sum().item())
        positive_nonfinite += int((positive & ~finite).sum().item())
        streams = batch["proposal_streams"]
        for stream in range(dataset.manifest.proposal_stream_count):
            stream_finite = finite & (streams == stream)
            all_nonfinite += int((~stream_finite.any(dim=1)).sum().item())
        block_ids.update(int(value) for value in batch["block_ids"].cpu().tolist())

        batch_collisions, batch_positive_collisions = _collision_counts(
            batch["candidate_worlds"], positive
        )
        collisions += batch_collisions
        positive_collisions += batch_positive_collisions
    summary = {
        "roots": roots,
        "blocks": len(block_ids),
        "candidates": candidates,
        "nonfinite": nonfinite,
        "nonfinite_fraction": nonfinite / max(candidates, 1),
        "positive_nonfinite": positive_nonfinite,
        "proposal_streams_without_finite_weight": all_nonfinite,
        "proposal_collisions": collisions,
        "positive_collisions": positive_collisions,
    }
    return summary, frozenset(block_ids)


def _validate_disjoint_blocks(block_ids: dict[str, frozenset[int]]) -> None:
    split_names = tuple(block_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = block_ids[left] & block_ids[right]
            if overlap:
                raise ValueError(
                    f"belief blocks cross {left!r} and {right!r} splits: "
                    f"{sorted(overlap)}"
                )


@torch.no_grad()
def _evaluate(
    model: BeliefResidualModel,
    dataset: BeliefDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[float, dict[str, float]]:
    model.eval()
    totals = {beta: GroupMetrics.empty() for beta in BETA_GRID}
    for batch in dataset.batches(
        split, batch_size, device=device, shuffle=False, seed=seed
    ):
        # Calibration must match the FP32 Candle deployment path. Training may
        # use autocast, but the ESS threshold is a discrete safety decision.
        residuals = model(
            batch["tile_obs"],
            batch["melds"],
            batch["meta"],
            batch["events"],
            batch["event_lengths"],
            batch["candidate_worlds"],
        )
        for beta in BETA_GRID:
            totals[beta] = totals[beta] + metrics(
                batch["handwritten_log_weights"],
                residuals,
                batch["positive_mask"],
                batch["proposal_streams"],
                beta=beta,
            )
    return {beta: total.summary() for beta, total in totals.items()}


def _choose_beta(values: dict[float, dict[str, float]]) -> float:
    baseline_low_ess = values[0.0]["atomic_fallback_fraction"]
    eligible = [
        beta
        for beta, metrics_value in values.items()
        if metrics_value["atomic_fallback_fraction"] <= baseline_low_ess + 0.05
    ]
    if not eligible:
        return 0.0
    return min(
        eligible,
        key=lambda beta: (
            values[beta]["deployment_fallback_nll"],
            values[beta]["nll"],
            beta,
        ),
    )


def _schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _export_golden(
    model: BeliefResidualModel,
    dataset: BeliefDataset,
    output: Path,
) -> None:
    reference = BeliefResidualModel(model.config)
    reference.load_state_dict(model.state_dict())
    batch = next(
        dataset.batches(
            "development", 2, device=torch.device("cpu"), shuffle=False, seed=0
        )
    )
    reference.eval()
    with torch.no_grad():
        public = reference.encode_public(
            batch["tile_obs"],
            batch["melds"],
            batch["meta"],
            batch["events"],
            batch["event_lengths"],
        )
        residuals = reference.score_worlds(public, batch["candidate_worlds"])
    names = (
        "tile_obs",
        "melds",
        "meta",
        "events",
        "event_lengths",
        "candidate_worlds",
    )
    tensors = {name: batch[name] for name in names}
    tensors["event_lengths"] = tensors["event_lengths"].to(torch.int32)
    tensors["public"] = public.float()
    tensors["residuals"] = residuals.float()
    _save_safetensors(tensors, output)


def train(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    _set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest = DatasetManifest.load(args.dataset / "manifest.json")
    dataset = BeliefDataset(manifest)
    _prepare_output_dir(args.output)

    audited = {
        split: _support_audit(dataset, split, args.batch_size, device)
        for split in ("train", "calibration", "development")
    }
    audit = {split: result[0] for split, result in audited.items()}
    block_ids = {split: result[1] for split, result in audited.items()}
    _validate_disjoint_blocks(block_ids)
    if any(value["positive_nonfinite"] for value in audit.values()):
        raise RuntimeError(f"true belief worlds have non-finite offsets: {audit}")
    # A proposal stream can have no finite hand-written support. Deployment
    # treats that stream as ESS=0 and atomically falls back, so retain the root
    # and report the rate instead of rejecting the entire dataset.

    config = BeliefModelConfig()
    model = BeliefResidualModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(dataset.roots("train") / args.batch_size)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(int(total_steps * args.warmup_fraction), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule(step, total_steps, warmup_steps)
    )

    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        roots = 0
        for batch in dataset.batches(
            "train",
            args.batch_size,
            device=device,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                residuals = model(
                    batch["tile_obs"],
                    batch["melds"],
                    batch["meta"],
                    batch["events"],
                    batch["event_lengths"],
                    batch["candidate_worlds"],
                )
                loss, _ = grouped_loss(
                    batch["handwritten_log_weights"],
                    residuals,
                    batch["positive_mask"],
                    variance_weight=args.variance_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(batch["tile_obs"].shape[0])
            loss_sum += float(loss.item()) * count
            roots += count
        calibration = _evaluate(
            model, dataset, "calibration", args.batch_size, device, args.seed
        )
        beta = _choose_beta(calibration)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(roots, 1),
            "beta": beta,
            "calibration": {str(key): value for key, value in calibration.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    calibration = _evaluate(
        model, dataset, "calibration", args.batch_size, device, args.seed
    )
    beta = _choose_beta(calibration)
    model_path = args.output / "model.safetensors"
    _save_safetensors(model.state_dict(), model_path)
    golden_path = args.output / "golden.safetensors"
    _export_golden(model, dataset, golden_path)
    development = _evaluate(
        model, dataset, "development", args.batch_size, device, args.seed
    )
    artifact = {
        "artifact_version": 1,
        "model_kind": "belief_residual",
        "belief_schema_version": manifest.schema_version,
        "belief_target_version": manifest.belief_target_version,
        "engine_rules_version": manifest.engine_rules_version,
        "proposal_stream_count": manifest.proposal_stream_count,
        "calibration_particle_count": manifest.candidate_count,
        "max_history": config.max_history,
        "candidate_world_planes": 4,
        "tile_kind_count": 27,
        "config": config.to_dict(),
        "beta": beta,
        "model_sha256": _sha256(model_path),
        "golden_sha256": _sha256(golden_path),
        "training_dataset": str(args.dataset.resolve()),
        "training_seed": args.seed,
        "audit": audit,
        "calibration": {str(key): value for key, value in calibration.items()},
        "development": {str(key): value for key, value in development.items()},
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
    }
    _atomic_json(args.output / "manifest.json", artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
