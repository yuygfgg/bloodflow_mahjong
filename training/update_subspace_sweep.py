"""Test whether restricting RL updates improves cross-batch generalization.

The experiment reuses completed counterfactual batches. It compares full-model,
last-block-plus-head, and policy-head-only updates under AdamW and SGD. Each
candidate is normalized to one shared probe KL, then evaluated on both train-Q
and heldout-Q corpora. It also trains one pooled two-seed direction per variant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from . import batch_sweep, direction_generalization as dg, optimizer_sweep
from .pipeline import load_policy
from .policy_iteration import (
    CounterfactualBatch,
    PolicyStateBatch,
    calibrate_direction,
    direction_cosine,
    load_counterfactual_batch,
    load_policy_state_batch,
    nested_category_indices,
    one_step_direction,
    require_cuda,
    require_deterministic_actor,
    subset_counterfactual_batch,
)
from .policy_pool import CATEGORY_COUNT


UPDATE_SUBSPACE_SWEEP_VERSION = 1
DEFAULT_SCOPES = ("full", "last_blocks", "actor")
DEFAULT_OPTIMIZERS = ("adamw", "sgd")


@dataclass(frozen=True)
class SourceBatch:
    name: str
    seed: int | None
    batch: CounterfactualBatch
    category_weights: np.ndarray


def _concatenate_targets(
    batches: Sequence[CounterfactualBatch],
) -> CounterfactualBatch:
    if len(batches) < 2:
        raise ValueError("pooled update needs at least two target batches")
    values = {
        name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
        for name in CounterfactualBatch.__dataclass_fields__
        if name != "query_ids"
    }
    values["query_ids"] = np.arange(
        sum(len(batch) for batch in batches), dtype=np.int64
    )
    return CounterfactualBatch(**values)


def _scope_parameter_names(model) -> dict[str, tuple[str, ...]]:
    all_names = tuple(name for name, _parameter in model.named_parameters())
    actor = tuple(name for name in all_names if name.startswith("actor."))
    static_last = model.config.static_layers - 1
    history_last = model.config.history_layers - 1
    last_prefixes = (
        "actor.",
        f"static_encoder.blocks.{static_last}.",
        "static_encoder.output_norm.",
        f"history_encoder.blocks.{history_last}.",
        "history_encoder.output_norm.",
    )
    last_blocks = tuple(
        name for name in all_names if name.startswith(last_prefixes)
    )
    if not actor or not last_blocks or len(last_blocks) >= len(all_names):
        raise RuntimeError("Actor parameter scopes do not match the model")
    return {"full": all_names, "last_blocks": last_blocks, "actor": actor}


def _normalize_values(
    requested: Sequence[str], available: Sequence[str], label: str
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in requested)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or not set(normalized).issubset(set(available))
    ):
        raise ValueError(f"{label} must be a unique supported subset")
    return normalized


def _experiment_identity(
    input_sweep: optimizer_sweep.InputSweep,
    *,
    qpc: int,
    seeds: Sequence[int],
    scopes: Sequence[str],
    optimizers: Sequence[str],
    target_kl: float,
    adamw_learning_rate: float,
    sgd_learning_rate: float,
) -> dict[str, object]:
    return {
        "version": UPDATE_SUBSPACE_SWEEP_VERSION,
        "input_directory": str(input_sweep.root),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "queries_per_category": int(qpc),
        "seeds": [int(seed) for seed in seeds],
        "scopes": list(scopes),
        "optimizers": list(optimizers),
        "target_kl": float(target_kl),
        "adamw_learning_rate": float(adamw_learning_rate),
        "sgd_learning_rate": float(sgd_learning_rate),
        "microbatch_size": int(input_sweep.config.microbatch_size),
        "inference_batch_size": int(input_sweep.config.inference_batch_size),
    }


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("update-subspace output configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("update-subspace output directory is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _candidate_name(optimizer: str, scope: str, source: str) -> str:
    return f"{optimizer}-{scope}-{source}"


def _numeric(values: Sequence[float]) -> dict[str, float | int]:
    return batch_sweep._numeric_summary([float(value) for value in values])


def _generalization_summary(
    candidates: Mapping[str, Mapping[str, object]],
    pairwise: Mapping[str, Mapping[str, object]],
    *,
    seeds: Sequence[int],
    scopes: Sequence[str],
    optimizers: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for optimizer in optimizers:
        for scope in scopes:
            key = f"{optimizer}-{scope}"
            own_soft: list[float] = []
            other_soft: list[float] = []
            own_greedy: list[float] = []
            other_greedy: list[float] = []
            for seed in seeds:
                candidate = candidates[
                    _candidate_name(optimizer, scope, f"seed-{seed}")
                ]
                for target_seed in seeds:
                    metrics = candidate["heldout"][str(target_seed)]
                    destination_soft = (
                        own_soft if seed == target_seed else other_soft
                    )
                    destination_greedy = (
                        own_greedy if seed == target_seed else other_greedy
                    )
                    destination_soft.append(metrics["soft_rank_value"]["mean"])
                    destination_greedy.append(
                        metrics["greedy_rank_value"]["mean"]
                    )
            pooled = candidates[_candidate_name(optimizer, scope, "pooled")]
            pooled_soft = [
                float(pooled["heldout"][str(seed)]["soft_rank_value"]["mean"])
                for seed in seeds
            ]
            pooled_greedy = [
                float(pooled["heldout"][str(seed)]["greedy_rank_value"]["mean"])
                for seed in seeds
            ]
            cross_key = dg._pair_key(
                _candidate_name(optimizer, scope, f"seed-{seeds[0]}"),
                _candidate_name(optimizer, scope, f"seed-{seeds[1]}"),
            )
            result[key] = {
                "individual_own_soft_rank_value": _numeric(own_soft),
                "individual_other_soft_rank_value": _numeric(other_soft),
                "individual_own_greedy_rank_value": _numeric(own_greedy),
                "individual_other_greedy_rank_value": _numeric(other_greedy),
                "pooled_soft_rank_value": _numeric(pooled_soft),
                "pooled_greedy_rank_value": _numeric(pooled_greedy),
                "pooled_min_soft_rank_value": min(pooled_soft),
                "pooled_min_greedy_rank_value": min(pooled_greedy),
                "cross_seed_parameter_cosine": pairwise[cross_key][
                    "parameter"
                ]["cosine_to_maximum"],
                "cross_seed_policy_cosine": pairwise[cross_key]["policy"][
                    "probability_delta_cosine"
                ],
            }
    return result


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    qpc: int = 256,
    seeds: Sequence[int] | None = None,
    scopes: Sequence[str] = DEFAULT_SCOPES,
    optimizers: Sequence[str] = DEFAULT_OPTIMIZERS,
    target_kl: float = 1e-4,
    adamw_learning_rate: float = 1e-5,
    sgd_learning_rate: float = 0.1,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if (
        not math.isfinite(target_kl)
        or target_kl <= 0
        or not math.isfinite(adamw_learning_rate)
        or adamw_learning_rate <= 0
        or not math.isfinite(sgd_learning_rate)
        or sgd_learning_rate <= 0
    ):
        raise ValueError("update-subspace numeric arguments must be positive")
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = dg._normalize_seeds(input_sweep.seeds, seeds)
    if len(selected_seeds) != 2:
        raise ValueError("update-subspace sweep currently requires exactly two seeds")
    if qpc not in input_sweep.config.batch_queries_per_category:
        raise ValueError("update-subspace QPC is not in the input sweep")
    selected_scopes = _normalize_values(scopes, DEFAULT_SCOPES, "scopes")
    selected_optimizers = _normalize_values(
        optimizers, DEFAULT_OPTIMIZERS, "optimizers"
    )
    identity = _experiment_identity(
        input_sweep,
        qpc=qpc,
        seeds=selected_seeds,
        scopes=selected_scopes,
        optimizers=selected_optimizers,
        target_kl=target_kl,
        adamw_learning_rate=adamw_learning_rate,
        sgd_learning_rate=sgd_learning_rate,
    )
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached update-subspace identity differs")
        print(f"RESULT update subspace sweep cached  {result_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    config = input_sweep.config
    reference = None
    reference_hash = None
    train_batches: dict[int, CounterfactualBatch] = {}
    calibration_batches: list[PolicyStateBatch] = []
    heldout_batches: dict[int, CounterfactualBatch] = {}
    visit_weights: dict[int, np.ndarray] = {}
    for seed in selected_seeds:
        child = input_sweep.children[seed]
        input_identity = input_sweep.child_identities[seed]
        reference_path, _self_play_path = optimizer_sweep._checkpoint_paths(
            input_identity
        )
        if reference is None:
            reference = load_policy(reference_path, resolved, frozen=True)
            require_deterministic_actor(reference)
            reference_hash = input_identity["reference_sha256"]
        elif input_identity["reference_sha256"] != reference_hash:
            raise ValueError("input seeds do not share a reference policy")
        train = load_counterfactual_batch(child / "shared" / "train.npz")
        nested = nested_category_indices(
            train.categories, config.batch_queries_per_category
        )
        train_batches[seed] = subset_counterfactual_batch(train, nested[qpc])
        calibration_batches.append(
            load_policy_state_batch(child / "shared" / "calibration.npz")
        )
        heldout_batches[seed] = load_counterfactual_batch(
            child / "shared" / "heldout.npz"
        )
        manifest = optimizer_sweep._json(child / "shared" / "manifest.json")
        weights = np.asarray(
            manifest["source_visit_frequencies"]["vector"], dtype=np.float64
        )
        if weights.shape != (CATEGORY_COUNT,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("source visitation weights are invalid")
        visit_weights[seed] = weights
    assert reference is not None

    common_probe = dg._concatenate_states(calibration_batches)
    common_weights = np.stack(
        [visit_weights[seed] for seed in selected_seeds]
    ).mean(axis=0)
    common_weights /= common_weights.sum()
    pooled_train = _concatenate_targets(
        [train_batches[seed] for seed in selected_seeds]
    )
    sources = [
        SourceBatch(
            name=f"seed-{seed}",
            seed=seed,
            batch=train_batches[seed],
            category_weights=visit_weights[seed],
        )
        for seed in selected_seeds
    ] + [
        SourceBatch(
            name="pooled",
            seed=None,
            batch=pooled_train,
            category_weights=common_weights,
        )
    ]
    scope_names = _scope_parameter_names(reference)
    scope_counts = {
        scope: int(
            sum(
                dict(reference.named_parameters())[name].numel()
                for name in scope_names[scope]
            )
        )
        for scope in selected_scopes
    }
    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  update subspace sweep  "
        f"QPC {qpc}  seeds {','.join(map(str, selected_seeds))}  "
        f"target KL {target_kl:g}",
        flush=True,
    )
    print(
        "SCOPES "
        + "  ".join(
            f"{scope}={scope_counts[scope]:,}" for scope in selected_scopes
        ),
        flush=True,
    )

    reference_probe_probabilities, reference_probe_actions = dg._policy_outputs(
        reference,
        common_probe,
        resolved,
        batch_size=config.inference_batch_size,
    )
    target_batches: dict[str, CounterfactualBatch] = {}
    target_weights: dict[str, np.ndarray] = {}
    for seed in selected_seeds:
        target_batches[f"train-{seed}"] = train_batches[seed]
        target_weights[f"train-{seed}"] = visit_weights[seed]
        target_batches[f"heldout-{seed}"] = heldout_batches[seed]
        target_weights[f"heldout-{seed}"] = visit_weights[seed]
    reference_outputs = {
        name: dg._policy_outputs(
            reference,
            batch,
            resolved,
            batch_size=config.inference_batch_size,
        )
        for name, batch in target_batches.items()
    }

    candidates: dict[str, dict[str, object]] = {}
    pairwise: dict[str, dict[str, object]] = {}
    probe_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for optimizer in selected_optimizers:
        learning_rate = (
            adamw_learning_rate if optimizer == "adamw" else sgd_learning_rate
        )
        for scope in selected_scopes:
            raw_states: dict[str, tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]] = {}
            for source in sources:
                name = _candidate_name(optimizer, scope, source.name)
                actor, initial, candidate, optimizer_metrics = one_step_direction(
                    reference,
                    source.batch,
                    resolved,
                    category_weights=source.category_weights,
                    learning_rate=learning_rate,
                    microbatch_size=config.microbatch_size,
                    optimizer_name=optimizer,
                    trainable_parameter_names=scope_names[scope],
                )
                calibration = calibrate_direction(
                    actor,
                    reference,
                    initial,
                    candidate,
                    common_probe,
                    resolved,
                    category_weights=common_weights,
                    target_kl=target_kl,
                    batch_size=config.inference_batch_size,
                    search_steps=config.kl_search_steps,
                    maximum_scale=config.maximum_scale,
                )
                probabilities, actions = dg._policy_outputs(
                    actor,
                    common_probe,
                    resolved,
                    batch_size=config.inference_batch_size,
                )
                probe_outputs[name] = (probabilities, actions)
                train_metrics: dict[str, object] = {}
                heldout_metrics: dict[str, object] = {}
                for target_name, target_batch in target_batches.items():
                    reference_probabilities, reference_actions = reference_outputs[
                        target_name
                    ]
                    target_probabilities, target_actions = dg._policy_outputs(
                        actor,
                        target_batch,
                        resolved,
                        batch_size=config.inference_batch_size,
                    )
                    metrics = dg._heldout_metrics(
                        target_probabilities,
                        target_actions,
                        reference_probabilities,
                        reference_actions,
                        target_batch,
                        target_weights[target_name],
                    )
                    target_seed = target_name.split("-", 1)[1]
                    destination = (
                        train_metrics
                        if target_name.startswith("train-")
                        else heldout_metrics
                    )
                    destination[target_seed] = metrics
                candidates[name] = {
                    "optimizer": optimizer,
                    "scope": scope,
                    "source": source.name,
                    "seed": source.seed,
                    "states": len(source.batch),
                    "optimizer_metrics": optimizer_metrics,
                    "calibration": calibration,
                    "common_probe_policy": dg._policy_signature(
                        probabilities,
                        actions,
                        reference_probe_probabilities,
                        reference_probe_actions,
                        common_probe,
                        common_weights,
                    ),
                    "train_q": train_metrics,
                    "heldout": heldout_metrics,
                }
                if source.seed is not None:
                    raw_states[source.name] = (initial, candidate)
                heldout_text = " ".join(
                    f"H{seed}={heldout_metrics[str(seed)]['soft_rank_value']['mean']:+.6f}"
                    for seed in selected_seeds
                )
                print(
                    f"CANDIDATE {name:32s}  scale {calibration['scale']:.5g}  "
                    f"{heldout_text}",
                    flush=True,
                )
                del actor
                torch.cuda.empty_cache()

            left_name = _candidate_name(
                optimizer, scope, f"seed-{selected_seeds[0]}"
            )
            right_name = _candidate_name(
                optimizer, scope, f"seed-{selected_seeds[1]}"
            )
            left_initial, left_candidate = raw_states[f"seed-{selected_seeds[0]}"]
            _right_initial, right_candidate = raw_states[f"seed-{selected_seeds[1]}"]
            left_probabilities, left_actions = probe_outputs[left_name]
            right_probabilities, right_actions = probe_outputs[right_name]
            pairwise[dg._pair_key(left_name, right_name)] = {
                "parameter": direction_cosine(
                    left_initial, left_candidate, right_candidate
                ),
                "policy": dg._policy_pair_metrics(
                    left_probabilities,
                    left_actions,
                    right_probabilities,
                    right_actions,
                    reference_probe_probabilities,
                    reference_probe_actions,
                    common_probe,
                    common_weights,
                ),
            }

    generalization = _generalization_summary(
        candidates,
        pairwise,
        seeds=selected_seeds,
        scopes=selected_scopes,
        optimizers=selected_optimizers,
    )
    result = {
        "identity": identity,
        "scope_parameters": scope_counts,
        "common_probe_states": len(common_probe),
        "candidates": candidates,
        "cross_seed_pairwise": pairwise,
        "generalization": generalization,
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)
    print("SUMMARY pooled directions", flush=True)
    for key, metrics in generalization.items():
        print(
            f"  {key:22s}  policy-cos {metrics['cross_seed_policy_cosine']:+.3f}  "
            f"pooled soft mean {metrics['pooled_soft_rank_value']['mean']:+.6f}  "
            f"min {metrics['pooled_min_soft_rank_value']:+.6f}  "
            f"greedy mean {metrics['pooled_greedy_rank_value']['mean']:+.6f}",
            flush=True,
        )
    print(f"RESULT update subspace sweep complete  {result_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qpc", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES))
    parser.add_argument(
        "--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS)
    )
    parser.add_argument("--target-kl", type=float, default=1e-4)
    parser.add_argument("--adamw-learning-rate", type=float, default=1e-5)
    parser.add_argument("--sgd-learning-rate", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.batch_sweep_dir,
        args.output_dir,
        qpc=args.qpc,
        seeds=args.seeds,
        scopes=args.scopes,
        optimizers=args.optimizers,
        target_kl=args.target_kl,
        adamw_learning_rate=args.adamw_learning_rate,
        sgd_learning_rate=args.sgd_learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
