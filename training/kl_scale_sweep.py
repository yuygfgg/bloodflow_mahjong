"""Test capped AdamW step scales using a completed batch sweep.

The expensive source collection and counterfactual targets are reused from the
input sweep.  For each selected QPC this module loads the cached raw AdamW
direction, evaluates fixed scales no larger than one, and compares them on the
same fixed panels against both U56 and the original KL-normalized AdamW actor.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from . import batch_sweep, optimizer_sweep
from .evaluation import collect_fixed_panel, outcomes, summarize_paired
from .pipeline import load_policy, save_policy
from .policy_iteration import (
    domain_seed,
    evaluate_direction_scale,
    heldout_policy_value,
    load_counterfactual_batch,
    load_policy_state_batch,
    nested_category_indices,
    require_cuda,
    require_deterministic_actor,
    subset_counterfactual_batch,
)
from .progress import Progress


KL_SCALE_SWEEP_VERSION = 1
KL_SCALE_DOMAIN = 0xB800_0001
DEFAULT_SCALES = (0.5, 1.0)


def _scale_key(scale: float) -> str:
    return format(float(scale), ".12g")


def _scale_slug(scale: float) -> str:
    return _scale_key(scale).replace("-", "m").replace(".", "p")


def _variant_key(qpc: int, scale: float) -> str:
    return f"qpc-{int(qpc)}-scale-{_scale_key(scale)}"


def _panel_path(directory: Path, qpc: int, scale: float) -> Path:
    return directory / "evaluation" / (
        f"adamw-qpc-{int(qpc)}-scale-{_scale_slug(scale)}.npz"
    )


def _actor_path(directory: Path, qpc: int, scale: float) -> Path:
    return directory / f"actor-adamw-qpc{int(qpc)}-scale-{_scale_slug(scale)}.pt"


def _normalize_scales(scales: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in scales)
    if (
        not normalized
        or any(not math.isfinite(value) or value <= 0 or value > 1 for value in normalized)
        or len(set(normalized)) != len(normalized)
        or tuple(sorted(normalized)) != normalized
    ):
        raise ValueError("KL scale sweep scales must be unique, increasing, and in (0, 1]")
    if 1.0 not in normalized:
        raise ValueError("KL scale sweep must include raw AdamW scale 1")
    return normalized


def _normalize_qpc(
    available: Sequence[int], requested: Sequence[int] | None
) -> tuple[int, ...]:
    normalized = (256,) if requested is None else tuple(int(value) for value in requested)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or tuple(sorted(normalized)) != normalized
        or not set(normalized).issubset(set(available))
    ):
        raise ValueError("KL scale sweep QPC values must be an increasing input subset")
    return normalized


def _experiment_identity(
    input_sweep: optimizer_sweep.InputSweep,
    qpc: Sequence[int],
    scales: Sequence[float],
) -> dict[str, object]:
    return {
        "version": KL_SCALE_SWEEP_VERSION,
        "optimizer": "adamw",
        "step_rule": "fixed-raw-scale",
        "input_directory": str(input_sweep.root),
        "input_identity": dict(input_sweep.identity),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "seeds": [int(seed) for seed in input_sweep.seeds],
        "queries_per_category": [int(value) for value in qpc],
        "scales": [float(value) for value in scales],
    }


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kl_scale_config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("KL scale sweep output configuration does not match")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("KL scale sweep output directory is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _child_identity(
    experiment: Mapping[str, object],
    input_identity: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    return {
        "experiment": dict(experiment),
        "input_child_identity": dict(input_identity),
        "seed": int(seed),
    }


def _expected_variants(qpc: Sequence[int], scales: Sequence[float]) -> set[str]:
    return {_variant_key(value, scale) for value in qpc for scale in scales}


def _load_child_summary(
    path: Path,
    identity: Mapping[str, object],
    qpc: Sequence[int],
    scales: Sequence[float],
) -> dict[str, object]:
    if not path.exists():
        return {"identity": dict(identity), "variants": {}}
    summary = optimizer_sweep._json(path)
    variants = summary.get("variants")
    if (
        summary.get("identity") != identity
        or not isinstance(variants, dict)
        or not set(variants).issubset(_expected_variants(qpc, scales))
    ):
        raise ValueError("KL scale sweep child summary identity does not match")
    return summary


def _input_calibrated_scale(
    input_sweep: optimizer_sweep.InputSweep, seed: int, qpc: int
) -> float:
    variant = input_sweep.child_summaries[seed]["variants"].get(str(qpc))
    calibration = variant.get("calibration") if isinstance(variant, dict) else None
    if not isinstance(calibration, dict):
        raise ValueError(f"input seed {seed} QPC {qpc} has no calibration metrics")
    scale = float(calibration["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"input seed {seed} QPC {qpc} has an invalid calibrated scale")
    return scale


def _run_seed(
    input_sweep: optimizer_sweep.InputSweep,
    output_directory: Path,
    *,
    experiment: Mapping[str, object],
    qpc: tuple[int, ...],
    scales: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    config = input_sweep.config
    child_input = input_sweep.children[seed]
    input_identity = input_sweep.child_identities[seed]
    child_identity = _child_identity(experiment, input_identity, seed)
    child_output = output_directory / f"seed-{seed}"
    summary_path = child_output / "summary.json"
    summary = _load_child_summary(summary_path, child_identity, qpc, scales)
    if set(summary["variants"]) == _expected_variants(qpc, scales):
        print(f"SEED {seed} KL scale sweep cached", flush=True)
        return summary

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    reference_path, _self_play_path = optimizer_sweep._checkpoint_paths(input_identity)
    reference = load_policy(reference_path, device, frozen=True)
    require_deterministic_actor(reference)
    reference_digest = batch_sweep._model_digest(reference)
    train = load_counterfactual_batch(child_input / "shared" / "train.npz")
    calibration = load_policy_state_batch(child_input / "shared" / "calibration.npz")
    heldout_batch = load_counterfactual_batch(child_input / "shared" / "heldout.npz")
    maximum = config.batch_queries_per_category[-1]
    if len(train) != 9 * maximum:
        raise ValueError("input training corpus has the wrong state count")
    nested = nested_category_indices(train.categories, config.batch_queries_per_category)
    manifest = optimizer_sweep._json(child_input / "shared" / "manifest.json")
    visit_weights = np.asarray(
        manifest["source_visit_frequencies"]["vector"], dtype=np.float64
    )
    if visit_weights.shape != (9,) or not np.isfinite(visit_weights).all():
        raise ValueError("input source visitation weights are invalid")

    panel_seeds, reference_ranks, reference_scores = optimizer_sweep._reference_panel(
        child_input, input_identity, config, seed
    )
    progress = Progress()
    print(
        f"CUDA {torch.cuda.get_device_name(device)}  AdamW KL scale sweep  "
        f"seed {seed}  QPC {','.join(map(str, qpc))}  "
        f"scales {','.join(_scale_key(value) for value in scales)}",
        flush=True,
    )

    for value in qpc:
        batch = subset_counterfactual_batch(train, nested[value])
        initial, candidate, optimizer_metrics, direction_seconds = (
            optimizer_sweep._input_adam_direction(
                child_input,
                input_identity,
                reference_digest,
                value,
                batch,
            )
        )
        calibrated_scale = _input_calibrated_scale(input_sweep, seed, value)
        calibrated_actor = load_policy(
            child_input / f"actor-qpc{value}.pt", device, frozen=True
        )
        require_deterministic_actor(calibrated_actor)
        (
            calibrated_seeds,
            calibrated_ranks,
            calibrated_scores,
            _calibrated_seconds,
        ) = optimizer_sweep._input_adam_panel(
            child_input, input_identity, value, panel_seeds, calibrated_actor
        )
        if not np.array_equal(calibrated_seeds, panel_seeds):
            raise ValueError("input calibrated AdamW evaluation seeds do not match")

        for scale_index, scale in enumerate(scales):
            key = _variant_key(value, scale)
            if key in summary["variants"]:
                continue
            actor = copy.deepcopy(reference).to(device)
            started = time.perf_counter()
            fixed_scale_metrics = evaluate_direction_scale(
                actor,
                reference,
                initial,
                candidate,
                calibration,
                device,
                category_weights=visit_weights,
                scale=scale,
                batch_size=config.inference_batch_size,
            )
            scale_seconds = time.perf_counter() - started

            started = time.perf_counter()
            heldout = heldout_policy_value(
                actor,
                reference,
                heldout_batch,
                device,
                category_weights=visit_weights,
                batch_size=config.inference_batch_size,
            )
            heldout_seconds = time.perf_counter() - started
            save_policy(_actor_path(child_output, value, scale), actor)

            actor_digest = batch_sweep._model_digest(actor)
            panel_fingerprint = batch_sweep._fingerprint(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "qpc": value,
                    "scale": scale,
                    "actor": actor_digest,
                    "seeds": batch_sweep._fingerprint(panel_seeds.tolist()),
                }
            )
            panel_path = _panel_path(child_output, value, scale)
            if panel_path.exists():
                actor_seeds, actor_ranks, actor_scores, evaluation_seconds = (
                    batch_sweep._load_actor_panel(
                        panel_path, fingerprint=panel_fingerprint
                    )
                )
                if not np.array_equal(actor_seeds, panel_seeds):
                    raise ValueError("cached fixed-scale evaluation seeds do not match")
                print(
                    f"QPC {value:4d} scale {_scale_key(scale)} evaluation cached",
                    flush=True,
                )
            else:
                progress.start(
                    f"QPC{value}_SCALE{_scale_slug(scale)}_EVAL",
                    total=len(panel_seeds),
                    unit="games",
                )
                started = time.perf_counter()
                result = collect_fixed_panel(
                    actor,
                    device,
                    panel_seeds,
                    envs=config.evaluation_envs,
                    on_progress=lambda done, values: progress.update(done, fields=values),
                )
                evaluation_seconds = time.perf_counter() - started
                progress.complete(
                    fields={"games/s": len(panel_seeds) / max(evaluation_seconds, 1e-9)}
                )
                actor_ranks, actor_scores = outcomes(result)
                batch_sweep._save_actor_panel(
                    panel_path,
                    fingerprint=panel_fingerprint,
                    seeds=panel_seeds,
                    ranks=actor_ranks,
                    scores=actor_scores,
                    elapsed_seconds=evaluation_seconds,
                )

            evaluation_vs_reference = summarize_paired(
                actor_ranks,
                actor_scores,
                reference_ranks,
                reference_scores,
                seed=domain_seed(seed, KL_SCALE_DOMAIN + 1, value * 16 + scale_index),
                bootstrap_samples=config.bootstrap_samples,
            )
            evaluation_vs_calibrated = summarize_paired(
                actor_ranks,
                actor_scores,
                calibrated_ranks,
                calibrated_scores,
                seed=domain_seed(seed, KL_SCALE_DOMAIN + 2, value * 16 + scale_index),
                bootstrap_samples=config.bootstrap_samples,
            )
            summary["variants"][key] = {
                "queries_per_category": value,
                "states": len(batch),
                "scale": scale,
                "input_calibrated_scale": calibrated_scale,
                "optimizer": optimizer_metrics,
                "fixed_scale": fixed_scale_metrics,
                "heldout": heldout,
                "evaluation_vs_reference": evaluation_vs_reference,
                "evaluation_vs_calibrated": evaluation_vs_calibrated,
                "timing": {
                    "reused_direction_seconds": direction_seconds,
                    "scale_metrics_seconds": scale_seconds,
                    "heldout_seconds": heldout_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "elapsed_seconds": scale_seconds
                    + heldout_seconds
                    + evaluation_seconds,
                },
            }
            batch_sweep._atomic_json(summary_path, summary)
            rank = evaluation_vs_calibrated["paired_rank_delta"]
            print(
                f"QPC {value:4d} scale {_scale_key(scale):>4s}  "
                f"KL {fixed_scale_metrics['kl']:.6f}  "
                f"vs-calibrated dRank {rank['mean']:+.4f} "
                f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]  "
                f"heldout {heldout['visitation_weighted_rank_value']:+.5f}",
                flush=True,
            )
            del actor
            torch.cuda.empty_cache()
        del calibrated_actor
        torch.cuda.empty_cache()
    return summary


def _aggregate(
    input_sweep: optimizer_sweep.InputSweep,
    output_directory: Path,
    *,
    experiment: Mapping[str, object],
    qpc: Sequence[int],
    scales: Sequence[float],
    child_summaries: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    qpc_results: dict[str, object] = {}
    raw_key = _scale_key(1.0)
    for value in qpc:
        pooled_seeds: list[np.ndarray] = []
        reference_ranks: list[np.ndarray] = []
        reference_scores: list[np.ndarray] = []
        calibrated_ranks: list[np.ndarray] = []
        calibrated_scores: list[np.ndarray] = []
        calibrated_scales: list[float] = []
        scale_ranks = {_scale_key(scale): [] for scale in scales}
        scale_scores = {_scale_key(scale): [] for scale in scales}
        per_scale_seed: dict[str, dict[str, object]] = {
            _scale_key(scale): {} for scale in scales
        }

        for seed in input_sweep.seeds:
            child_input = input_sweep.children[seed]
            input_identity = input_sweep.child_identities[seed]
            child_output = output_directory / f"seed-{seed}"
            ref_seed, ref_rank, ref_score = optimizer_sweep._reference_panel(
                child_input, input_identity, input_sweep.config, seed
            )
            calibrated_seed, calibrated_rank, calibrated_score, _ = (
                batch_sweep._load_actor_panel_unchecked(
                    child_input / "evaluation" / f"qpc-{value}.npz"
                )
            )
            if not np.array_equal(ref_seed, calibrated_seed):
                raise ValueError(f"input panels do not align for seed {seed}")
            pooled_seeds.append(ref_seed)
            reference_ranks.append(ref_rank)
            reference_scores.append(ref_score)
            calibrated_ranks.append(calibrated_rank)
            calibrated_scores.append(calibrated_score)
            calibrated_scales.append(_input_calibrated_scale(input_sweep, seed, value))

            for scale in scales:
                key = _scale_key(scale)
                variant = child_summaries[seed]["variants"][_variant_key(value, scale)]
                actor_seed, actor_rank, actor_score, panel_seconds = (
                    batch_sweep._load_actor_panel_unchecked(
                        _panel_path(child_output, value, scale)
                    )
                )
                if not np.array_equal(actor_seed, ref_seed):
                    raise ValueError(
                        f"fixed-scale panels do not align for seed {seed} scale {key}"
                    )
                reported = float(
                    variant["evaluation_vs_calibrated"]["paired_rank_delta"]["mean"]
                )
                if not math.isclose(
                    float((actor_rank - calibrated_rank).mean()),
                    reported,
                    abs_tol=1e-12,
                ):
                    raise ValueError("fixed-scale panel metrics do not match")
                scale_ranks[key].append(actor_rank)
                scale_scores[key].append(actor_score)
                per_scale_seed[key][str(seed)] = {
                    "input_calibrated_scale": variant["input_calibrated_scale"],
                    "fixed_scale": variant["fixed_scale"],
                    "heldout": variant["heldout"],
                    "evaluation_vs_reference": variant["evaluation_vs_reference"],
                    "evaluation_vs_calibrated": variant[
                        "evaluation_vs_calibrated"
                    ],
                    "timing": variant["timing"],
                    "panel_seconds": panel_seconds,
                }

        all_panel_seeds = np.concatenate(pooled_seeds)
        if len(np.unique(all_panel_seeds)) != len(all_panel_seeds):
            raise ValueError(f"KL scale sweep panels overlap for QPC {value}")
        joined_reference_rank = np.concatenate(reference_ranks)
        joined_reference_score = np.concatenate(reference_scores)
        joined_calibrated_rank = np.concatenate(calibrated_ranks)
        joined_calibrated_score = np.concatenate(calibrated_scores)
        joined_raw_rank = np.concatenate(scale_ranks[raw_key])
        joined_raw_score = np.concatenate(scale_scores[raw_key])
        scale_results: dict[str, object] = {}
        for scale_index, scale in enumerate(scales):
            key = _scale_key(scale)
            joined_rank = np.concatenate(scale_ranks[key])
            joined_score = np.concatenate(scale_scores[key])
            seed_variants = [
                child_summaries[seed]["variants"][_variant_key(value, scale)]
                for seed in input_sweep.seeds
            ]
            scale_results[key] = {
                "scale": float(scale),
                "seeds": per_scale_seed[key],
                "pooled_evaluation_vs_reference": summarize_paired(
                    joined_rank,
                    joined_score,
                    joined_reference_rank,
                    joined_reference_score,
                    seed=batch_sweep._multi_summary_seed(
                        value + 0x10000 * (scale_index + 1), input_sweep.seeds
                    ),
                    bootstrap_samples=input_sweep.config.bootstrap_samples,
                ),
                "pooled_evaluation_vs_calibrated": summarize_paired(
                    joined_rank,
                    joined_score,
                    joined_calibrated_rank,
                    joined_calibrated_score,
                    seed=batch_sweep._multi_summary_seed(
                        value + 0x20000 * (scale_index + 1), input_sweep.seeds
                    ),
                    bootstrap_samples=input_sweep.config.bootstrap_samples,
                ),
                "pooled_evaluation_vs_raw": summarize_paired(
                    joined_rank,
                    joined_score,
                    joined_raw_rank,
                    joined_raw_score,
                    seed=batch_sweep._multi_summary_seed(
                        value + 0x30000 * (scale_index + 1), input_sweep.seeds
                    ),
                    bootstrap_samples=input_sweep.config.bootstrap_samples,
                ),
                "seed_metrics": {
                    "calibration_kl": batch_sweep._numeric_summary(
                        [float(item["fixed_scale"]["kl"]) for item in seed_variants]
                    ),
                    "heldout_visitation_weighted_rank_value": (
                        batch_sweep._numeric_summary(
                            [
                                float(
                                    item["heldout"][
                                        "visitation_weighted_rank_value"
                                    ]
                                )
                                for item in seed_variants
                            ]
                        )
                    ),
                    "paired_rank_delta_vs_calibrated": batch_sweep._numeric_summary(
                        [
                            float(
                                item["evaluation_vs_calibrated"][
                                    "paired_rank_delta"
                                ]["mean"]
                            )
                            for item in seed_variants
                        ]
                    ),
                },
            }
        qpc_results[str(value)] = {
            "queries_per_category": int(value),
            "states": int(9 * value),
            "input_calibrated_scale": batch_sweep._numeric_summary(calibrated_scales),
            "scales": scale_results,
        }
    return {
        "identity": dict(experiment),
        "seeds": [int(seed) for seed in input_sweep.seeds],
        "variants": qpc_results,
    }


def _load_aggregate(
    path: Path,
    identity: Mapping[str, object],
    qpc: Sequence[int],
    scales: Sequence[float],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    value = optimizer_sweep._json(path)
    variants = value.get("variants")
    expected_qpc = {str(item) for item in qpc}
    expected_scales = {_scale_key(item) for item in scales}
    if (
        value.get("identity") != identity
        or not isinstance(variants, dict)
        or set(variants) != expected_qpc
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("scales"), dict)
            or set(item["scales"]) != expected_scales
            for item in variants.values()
        )
    ):
        raise ValueError("KL scale sweep aggregate identity does not match")
    return value


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    qpc: Sequence[int] | None = None,
    scales: Sequence[float] = DEFAULT_SCALES,
    seeds: Sequence[int] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    requested_qpc = _normalize_qpc(
        input_sweep.config.batch_queries_per_category, qpc
    )
    requested_scales = _normalize_scales(scales)
    experiment = _experiment_identity(input_sweep, requested_qpc, requested_scales)
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, experiment)
    aggregate_path = output_directory / "aggregate.json"
    cached = _load_aggregate(
        aggregate_path, experiment, requested_qpc, requested_scales
    )
    if cached is not None:
        print(f"RESULT KL scale sweep cached  {aggregate_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    children: dict[int, Mapping[str, object]] = {}
    for seed in input_sweep.seeds:
        children[seed] = _run_seed(
            input_sweep,
            output_directory,
            experiment=experiment,
            qpc=requested_qpc,
            scales=requested_scales,
            seed=seed,
            device=resolved,
        )
    aggregate = _aggregate(
        input_sweep,
        output_directory,
        experiment=experiment,
        qpc=requested_qpc,
        scales=requested_scales,
        child_summaries=children,
    )
    batch_sweep._atomic_json(aggregate_path, aggregate)
    for value in requested_qpc:
        for scale in requested_scales:
            result = aggregate["variants"][str(value)]["scales"][_scale_key(scale)]
            calibrated = result["pooled_evaluation_vs_calibrated"][
                "paired_rank_delta"
            ]
            raw = result["pooled_evaluation_vs_raw"]["paired_rank_delta"]
            print(
                f"QPC {value:4d} scale {_scale_key(scale):>4s} pooled  "
                f"vs-calibrated dRank {calibrated['mean']:+.4f} "
                f"[{calibrated['ci95_low']:+.4f},{calibrated['ci95_high']:+.4f}]  "
                f"vs-raw {raw['mean']:+.4f} "
                f"[{raw['ci95_low']:+.4f},{raw['ci95_high']:+.4f}]",
                flush=True,
            )
    print(f"RESULT KL scale sweep complete  {aggregate_path}", flush=True)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qpc",
        type=int,
        nargs="+",
        default=[256],
        help="increasing subset of completed input QPC values (default: 256)",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="increasing raw AdamW scales in (0,1], including 1 (default: 0.5 1)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="completed input seeds to use (default: all configured seeds)",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.batch_sweep_dir,
        args.output_dir,
        qpc=args.qpc,
        scales=args.scales,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
