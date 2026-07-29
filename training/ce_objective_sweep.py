"""Compare policy-gradient and search-policy cross-entropy directions.

The expensive counterfactual Q corpora are reused from a completed batch sweep.
Every raw SGD direction is calibrated to the same reverse-KL on one shared,
disjoint probe.  Seed-specific and pooled directions are then evaluated against
both heldout Q corpora, so a target only wins if its improvement transfers.
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
from .pipeline import load_policy, save_policy
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
from .update_subspace_sweep import _concatenate_targets


CE_OBJECTIVE_SWEEP_VERSION = 1


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    objective: str
    temperature: float = 0.1
    prior_floor: float = 0.0

    def identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "objective": self.objective,
            "temperature": float(self.temperature),
            "prior_floor": float(self.prior_floor),
        }


@dataclass(frozen=True)
class SourceBatch:
    name: str
    seed: int | None
    batch: CounterfactualBatch
    category_weights: np.ndarray


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def objective_specs(
    softmax_temperatures: Sequence[float],
    mirror_temperatures: Sequence[float],
    *,
    mirror_prior_floor: float,
    include_uniform: bool = True,
    include_hard: bool = True,
) -> tuple[ObjectiveSpec, ...]:
    softmax = tuple(float(value) for value in softmax_temperatures)
    mirror = tuple(float(value) for value in mirror_temperatures)
    values = (*softmax, *mirror)
    if (
        any(not math.isfinite(value) or value <= 0 for value in values)
        or len(set(softmax)) != len(softmax)
        or len(set(mirror)) != len(mirror)
        or not math.isfinite(mirror_prior_floor)
        or not 0.0 <= mirror_prior_floor < 1.0
    ):
        raise ValueError("CE temperatures/floor are invalid")
    specs = [ObjectiveSpec("expected-q", "expected_q")]
    if include_uniform:
        specs.append(ObjectiveSpec("uniform-ce-control", "uniform_ce"))
    if include_hard:
        specs.append(ObjectiveSpec("hard-ce", "hard_ce"))
    specs.extend(
        ObjectiveSpec(
            f"softmax-ce-t{_number_token(temperature)}",
            "softmax_ce",
            temperature,
        )
        for temperature in softmax
    )
    specs.extend(
        ObjectiveSpec(
            f"mirror-ce-t{_number_token(temperature)}"
            f"-f{_number_token(mirror_prior_floor)}",
            "mirror_ce",
            temperature,
            mirror_prior_floor,
        )
        for temperature in mirror
    )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("CE objective names are not unique")
    return tuple(specs)


def _experiment_identity(
    input_sweep: optimizer_sweep.InputSweep,
    *,
    qpc: int,
    seeds: Sequence[int],
    specs: Sequence[ObjectiveSpec],
    target_kl: float,
    learning_rate: float,
) -> dict[str, object]:
    return {
        "version": CE_OBJECTIVE_SWEEP_VERSION,
        "input_directory": str(input_sweep.root),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "queries_per_category": int(qpc),
        "seeds": [int(seed) for seed in seeds],
        "objectives": [spec.identity() for spec in specs],
        "optimizer": "sgd",
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "microbatch_size": int(input_sweep.config.microbatch_size),
        "inference_batch_size": int(input_sweep.config.inference_batch_size),
    }


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("CE objective output configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("CE objective output directory is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _candidate_name(spec: ObjectiveSpec, source: str) -> str:
    return f"{spec.name}-{source}"


def _numeric(values: Sequence[float]) -> dict[str, float | int]:
    return batch_sweep._numeric_summary([float(value) for value in values])


def _summary(
    candidates: Mapping[str, Mapping[str, object]],
    pairwise: Mapping[str, Mapping[str, object]],
    *,
    specs: Sequence[ObjectiveSpec],
    seeds: Sequence[int],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for spec in specs:
        own: list[float] = []
        other: list[float] = []
        for seed in seeds:
            candidate = candidates[_candidate_name(spec, f"seed-{seed}")]
            for target_seed in seeds:
                value = float(
                    candidate["heldout"][str(target_seed)]["soft_rank_value"][
                        "mean"
                    ]
                )
                (own if seed == target_seed else other).append(value)
        pooled = candidates[_candidate_name(spec, "pooled")]
        pooled_soft = [
            float(pooled["heldout"][str(seed)]["soft_rank_value"]["mean"])
            for seed in seeds
        ]
        pooled_greedy = [
            float(pooled["heldout"][str(seed)]["greedy_rank_value"]["mean"])
            for seed in seeds
        ]
        left = _candidate_name(spec, f"seed-{seeds[0]}")
        right = _candidate_name(spec, f"seed-{seeds[1]}")
        cross = pairwise[dg._pair_key(left, right)]
        result[spec.name] = {
            "individual_own_soft_rank_value": _numeric(own),
            "individual_other_soft_rank_value": _numeric(other),
            "pooled_soft_rank_value": _numeric(pooled_soft),
            "pooled_greedy_rank_value": _numeric(pooled_greedy),
            "pooled_min_soft_rank_value": min(pooled_soft),
            "pooled_min_greedy_rank_value": min(pooled_greedy),
            "cross_seed_parameter_cosine": cross["parameter"][
                "cosine_to_maximum"
            ],
            "cross_seed_policy_cosine": cross["policy"][
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
    softmax_temperatures: Sequence[float] = (0.05, 0.1, 0.2),
    mirror_temperatures: Sequence[float] = (0.02, 0.05, 0.1),
    mirror_prior_floor: float = 1e-4,
    include_hard: bool = True,
    include_uniform: bool = True,
    target_kl: float = 1e-4,
    learning_rate: float = 0.1,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if (
        not math.isfinite(target_kl)
        or target_kl <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("CE sweep KL and learning rate must be positive")
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = dg._normalize_seeds(input_sweep.seeds, seeds)
    if len(selected_seeds) != 2:
        raise ValueError("CE objective sweep currently requires exactly two seeds")
    if qpc not in input_sweep.config.batch_queries_per_category:
        raise ValueError("CE objective QPC is not in the input sweep")
    specs = objective_specs(
        softmax_temperatures,
        mirror_temperatures,
        mirror_prior_floor=mirror_prior_floor,
        include_uniform=include_uniform,
        include_hard=include_hard,
    )
    identity = _experiment_identity(
        input_sweep,
        qpc=qpc,
        seeds=selected_seeds,
        specs=specs,
        target_kl=target_kl,
        learning_rate=learning_rate,
    )
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached CE objective identity differs")
        print(f"RESULT CE objective sweep cached  {result_path}", flush=True)
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
            reference_hash = str(input_identity["reference_sha256"])
        elif str(input_identity["reference_sha256"]) != reference_hash:
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
        if (
            weights.shape != (CATEGORY_COUNT,)
            or not np.isfinite(weights).all()
            or not np.isclose(weights.sum(), 1.0)
        ):
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
            f"seed-{seed}", seed, train_batches[seed], visit_weights[seed]
        )
        for seed in selected_seeds
    ] + [SourceBatch("pooled", None, pooled_train, common_weights)]
    target_batches: dict[str, CounterfactualBatch] = {}
    target_weights: dict[str, np.ndarray] = {}
    for seed in selected_seeds:
        target_batches[f"train-{seed}"] = train_batches[seed]
        target_weights[f"train-{seed}"] = visit_weights[seed]
        target_batches[f"heldout-{seed}"] = heldout_batches[seed]
        target_weights[f"heldout-{seed}"] = visit_weights[seed]

    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  CE objective sweep  "
        f"QPC {qpc}  seeds {','.join(map(str, selected_seeds))}  "
        f"target KL {target_kl:g}  candidates {len(specs)}",
        flush=True,
    )
    reference_probe_probabilities, reference_probe_actions = dg._policy_outputs(
        reference,
        common_probe,
        resolved,
        batch_size=config.inference_batch_size,
    )
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
    for spec in specs:
        raw_states: dict[
            str, tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
        ] = {}
        probe_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source in sources:
            name = _candidate_name(spec, source.name)
            actor, initial, candidate, optimizer_metrics = one_step_direction(
                reference,
                source.batch,
                resolved,
                category_weights=source.category_weights,
                learning_rate=learning_rate,
                microbatch_size=config.microbatch_size,
                optimizer_name="sgd",
                objective=spec.objective,
                target_temperature=spec.temperature,
                target_prior_floor=spec.prior_floor,
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
                "objective": spec.identity(),
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
            else:
                save_policy(output_directory / f"actor-{spec.name}-pooled.pt", actor)
            heldout_text = " ".join(
                f"H{seed}={heldout_metrics[str(seed)]['soft_rank_value']['mean']:+.6f}"
                for seed in selected_seeds
            )
            print(
                f"CANDIDATE {name:42s}  scale {calibration['scale']:.5g}  "
                f"{heldout_text}",
                flush=True,
            )
            del actor
            torch.cuda.empty_cache()

        left_name = _candidate_name(spec, f"seed-{selected_seeds[0]}")
        right_name = _candidate_name(spec, f"seed-{selected_seeds[1]}")
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

    summary = _summary(
        candidates, pairwise, specs=specs, seeds=selected_seeds
    )
    result = {
        "identity": identity,
        "common_probe_states": len(common_probe),
        "candidates": candidates,
        "cross_seed_pairwise": pairwise,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)
    print("SUMMARY CE objectives", flush=True)
    for name, metrics in summary.items():
        cosine = metrics["cross_seed_policy_cosine"]
        cosine_text = "null" if cosine is None else f"{cosine:+.3f}"
        print(
            f"  {name:32s}  policy-cos {cosine_text}  "
            f"pooled soft {metrics['pooled_soft_rank_value']['mean']:+.6f}  "
            f"min {metrics['pooled_min_soft_rank_value']:+.6f}",
            flush=True,
        )
    print(f"RESULT CE objective sweep complete  {result_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qpc", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--softmax-temperatures", type=float, nargs="+", default=[0.05, 0.1, 0.2]
    )
    parser.add_argument(
        "--mirror-temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.1]
    )
    parser.add_argument("--mirror-prior-floor", type=float, default=1e-4)
    parser.add_argument(
        "--uniform-control", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--hard-ce", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--target-kl", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.batch_sweep_dir,
        args.output_dir,
        qpc=args.qpc,
        seeds=args.seeds,
        softmax_temperatures=args.softmax_temperatures,
        mirror_temperatures=args.mirror_temperatures,
        mirror_prior_floor=args.mirror_prior_floor,
        include_uniform=args.uniform_control,
        include_hard=args.hard_ce,
        target_kl=args.target_kl,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
