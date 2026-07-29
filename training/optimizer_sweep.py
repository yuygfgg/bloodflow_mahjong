"""Compare full-batch raw SGD with the AdamW direction from a batch sweep.

The input batch sweep owns expensive source collection and counterfactual
rollouts. This module is deliberately read-only with respect to that input:
it reuses the completed train, calibration, heldout, and fixed-evaluation
panels, then writes all optimizer-specific artifacts into a separate output
directory.

Nesterov is intentionally not included here. With no prior committed update,
its first direction is exactly raw SGD after fixed-KL calibration. Its useful
test is the persistent training fork, where velocity is carried across
updates.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from . import batch_sweep
from .batch_sweep import SweepConfig
from .evaluation import (
    collect_fixed_panel,
    load_reference_panel,
    outcomes,
    summarize_paired,
)
from .pipeline import POLICY_EXECUTION_VERSION, load_policy, save_policy
from .policy_iteration import (
    calibrate_direction,
    direction_cosine,
    domain_seed,
    heldout_policy_value,
    load_counterfactual_batch,
    load_policy_state_batch,
    nested_category_indices,
    one_step_direction,
    require_cuda,
    require_deterministic_actor,
    subset_counterfactual_batch,
)
from .progress import Progress


OPTIMIZER_SWEEP_VERSION = 1
OPTIMIZER_NAME = "sgd"
OPTIMIZER_DOMAIN = 0xB700_0001


@dataclass(frozen=True)
class InputSweep:
    root: Path
    identity: Mapping[str, object]
    config: SweepConfig
    seeds: tuple[int, ...]
    children: Mapping[int, Path]
    child_identities: Mapping[int, Mapping[str, object]]
    child_summaries: Mapping[int, Mapping[str, object]]


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _config_from_identity(identity: Mapping[str, object]) -> SweepConfig:
    raw = identity.get("config")
    if not isinstance(raw, dict):
        raise ValueError("input batch sweep identity has no config")
    try:
        return SweepConfig(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError("input batch sweep config is invalid") from error


def _checkpoint_paths(
    identity: Mapping[str, object],
) -> tuple[Path, Path | None]:
    reference = identity.get("reference_checkpoint")
    self_play = identity.get("self_play_checkpoint")
    if not isinstance(reference, str):
        raise ValueError("input batch sweep reference checkpoint is invalid")
    if self_play is not None and not isinstance(self_play, str):
        raise ValueError("input batch sweep self-play checkpoint is invalid")
    return Path(reference), None if self_play is None else Path(self_play)


def _completed_child(
    directory: Path,
    identity: Mapping[str, object],
    config: SweepConfig,
) -> Mapping[str, object]:
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"input batch sweep child is incomplete: {directory}")
    summary = batch_sweep._load_summary(summary_path, identity)
    expected_qpc = {str(value) for value in config.batch_queries_per_category}
    if set(summary["variants"]) != expected_qpc:
        raise ValueError(f"input batch sweep child is incomplete: {directory}")

    manifest_path = directory / "shared" / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"input batch sweep shared corpus is missing: {directory}")
    manifest = _json(manifest_path)
    expected_manifest = {
        "version",
        "fingerprint",
        "source_visit_frequencies",
        "train_targets",
        "heldout_targets",
    }
    expected_fingerprint = batch_sweep._fingerprint(
        {"identity": identity, "cache": "shared-corpora"}
    )
    if (
        set(manifest) != expected_manifest
        or int(manifest["version"]) != batch_sweep.SHARED_CACHE_VERSION
        or manifest["fingerprint"] != expected_fingerprint
    ):
        raise ValueError(f"input batch sweep shared corpus is invalid: {directory}")
    for name in ("train.npz", "calibration.npz", "heldout.npz"):
        if not (directory / "shared" / name).exists():
            raise ValueError(f"input batch sweep shared corpus is missing: {name}")
    return summary


def _select_seeds(
    available: Sequence[int], requested: Sequence[int] | None
) -> tuple[int, ...]:
    normalized_available = tuple(int(value) for value in available)
    if requested is None:
        return normalized_available
    normalized = tuple(int(value) for value in requested)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("optimizer sweep input seeds must be unique and non-empty")
    if not set(normalized).issubset(normalized_available):
        raise ValueError("optimizer sweep input seeds are not in the batch sweep")
    return normalized


def load_input_sweep(
    directory: Path,
    *,
    seeds: Sequence[int] | None = None,
) -> InputSweep:
    """Validate a completed single- or multi-seed batch-sweep directory."""
    root = directory.resolve()
    multi_path = root / "multi_config.json"
    if multi_path.exists():
        identity = _json(multi_path)
        config = _config_from_identity(identity)
        reference, self_play = _checkpoint_paths(identity)
        raw_seeds = identity.get("seeds")
        if (
            not isinstance(raw_seeds, list)
            or not raw_seeds
            or len({int(value) for value in raw_seeds}) != len(raw_seeds)
        ):
            raise ValueError("input multi-seed batch sweep seeds are invalid")
        available_seeds = tuple(int(value) for value in raw_seeds)
        expected_identity = batch_sweep._multi_identity(
            reference,
            config,
            available_seeds,
            self_play_checkpoint=self_play,
        )
        if identity != expected_identity:
            raise ValueError("input multi-seed batch sweep identity does not match")
        selected_seeds = _select_seeds(available_seeds, seeds)
        children = {seed: root / f"seed-{seed}" for seed in selected_seeds}
        child_identities: dict[int, Mapping[str, object]] = {}
        child_summaries: dict[int, Mapping[str, object]] = {}
        for seed, child in children.items():
            child_identity = batch_sweep._single_identity(
                reference, self_play, config, seed
            )
            config_path = child / "config.json"
            if not config_path.exists() or _json(config_path) != child_identity:
                raise ValueError(f"input batch sweep child identity does not match: {child}")
            child_identities[seed] = child_identity
            child_summaries[seed] = _completed_child(
                child, child_identity, config
            )
        return InputSweep(
            root=root,
            identity=identity,
            config=config,
            seeds=selected_seeds,
            children=children,
            child_identities=child_identities,
            child_summaries=child_summaries,
        )

    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"expected config.json or multi_config.json in {root}"
        )
    identity = _json(config_path)
    config = _config_from_identity(identity)
    reference, self_play = _checkpoint_paths(identity)
    raw_seed = identity.get("seed")
    if not isinstance(raw_seed, int):
        raise ValueError("input single-seed batch sweep seed is invalid")
    expected_identity = batch_sweep._single_identity(
        reference, self_play, config, raw_seed
    )
    if identity != expected_identity:
        raise ValueError("input single-seed batch sweep identity does not match")
    selected_seeds = _select_seeds((raw_seed,), seeds)
    summary = _completed_child(root, identity, config)
    return InputSweep(
        root=root,
        identity=identity,
        config=config,
        seeds=selected_seeds,
        children={raw_seed: root},
        child_identities={raw_seed: identity},
        child_summaries={raw_seed: summary},
    )


def _experiment_identity(
    input_sweep: InputSweep,
    qpc: Sequence[int],
    sgd_learning_rate: float,
) -> dict[str, object]:
    return {
        "version": OPTIMIZER_SWEEP_VERSION,
        "optimizer": OPTIMIZER_NAME,
        "sgd_learning_rate": float(sgd_learning_rate),
        "input_directory": str(input_sweep.root),
        "input_identity": dict(input_sweep.identity),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "seeds": [int(seed) for seed in input_sweep.seeds],
        "queries_per_category": [int(value) for value in qpc],
    }


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "optimizer_config.json"
    if path.exists():
        if _json(path) != identity:
            raise ValueError("optimizer sweep output configuration does not match")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("optimizer sweep output directory is non-empty")
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


def _load_output_summary(
    path: Path,
    identity: Mapping[str, object],
    qpc: Sequence[int],
) -> dict[str, object]:
    if not path.exists():
        return {"identity": dict(identity), "variants": {}}
    summary = _json(path)
    expected = {str(value) for value in qpc}
    variants = summary.get("variants")
    if (
        summary.get("identity") != identity
        or not isinstance(variants, dict)
        or not set(variants).issubset(expected)
    ):
        raise ValueError("optimizer sweep child summary identity does not match")
    return summary


def _reference_panel(
    child_directory: Path,
    identity: Mapping[str, object],
    config: SweepConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fingerprint = batch_sweep._fingerprint(
        {
            "reference": identity["reference_sha256"],
            "policy_execution_version": POLICY_EXECUTION_VERSION,
            "seed": int(batch_sweep.domain_seed(seed, batch_sweep.FIXED_EVAL)),
            "games": config.evaluation_games,
        }
    )
    return load_reference_panel(
        child_directory / "reference_panel.npz", fingerprint=fingerprint
    )


def _input_adam_direction(
    child_directory: Path,
    identity: Mapping[str, object],
    reference_digest: str,
    qpc: int,
    batch,
):
    fingerprint = batch_sweep._fingerprint(
        {
            "identity": identity,
            "reference": reference_digest,
            "qpc": qpc,
            "query_ids": batch.query_ids.tolist(),
        }
    )
    return batch_sweep._load_direction(
        child_directory / "directions" / f"qpc-{qpc}.pt",
        fingerprint=fingerprint,
    )


def _input_adam_panel(
    child_directory: Path,
    identity: Mapping[str, object],
    qpc: int,
    seeds: np.ndarray,
    actor,
):
    fingerprint = batch_sweep._fingerprint(
        {
            "identity": identity,
            "qpc": qpc,
            "actor": batch_sweep._model_digest(actor),
            "seeds": batch_sweep._fingerprint(seeds.tolist()),
        }
    )
    return batch_sweep._load_actor_panel(
        child_directory / "evaluation" / f"qpc-{qpc}.npz",
        fingerprint=fingerprint,
    )


def _sgd_direction(
    directory: Path,
    *,
    experiment: Mapping[str, object],
    reference,
    reference_digest: str,
    batch,
    qpc: int,
    visit_weights: np.ndarray,
    config: SweepConfig,
    learning_rate: float,
    device: torch.device,
    progress: Progress,
):
    fingerprint = batch_sweep._fingerprint(
        {
            "experiment": experiment,
            "reference": reference_digest,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": learning_rate,
            "qpc": qpc,
            "query_ids": batch.query_ids.tolist(),
        }
    )
    path = directory / "directions" / f"sgd-qpc-{qpc}.pt"
    if path.exists():
        initial, candidate, metrics, elapsed = batch_sweep._load_direction(
            path, fingerprint=fingerprint
        )
        print(
            f"QPC {qpc:4d} SGD direction cached  "
            f"{len(batch) / elapsed:,.1f} states/s",
            flush=True,
        )
        return initial, candidate, metrics, elapsed

    progress.start(
        f"QPC{qpc}_SGD_DIRECTION",
        total=math.ceil(len(batch) / config.microbatch_size),
        unit="microbatches",
    )
    started = time.perf_counter()
    actor, initial, candidate, metrics = one_step_direction(
        reference,
        batch,
        device,
        category_weights=visit_weights,
        learning_rate=learning_rate,
        microbatch_size=config.microbatch_size,
        optimizer_name=OPTIMIZER_NAME,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    elapsed = time.perf_counter() - started
    progress.complete(fields={"states/s": len(batch) / max(elapsed, 1e-9)})
    batch_sweep._save_direction(
        path,
        fingerprint=fingerprint,
        initial=initial,
        candidate=candidate,
        optimizer=metrics,
        elapsed_seconds=elapsed,
    )
    del actor
    torch.cuda.empty_cache()
    return initial, candidate, metrics, elapsed


def _direction_summary(
    initial,
    candidate,
    maximum_candidate,
    adam_candidate,
) -> dict[str, object]:
    return {
        "to_sgd_maximum": direction_cosine(initial, candidate, maximum_candidate),
        "to_adamw": direction_cosine(initial, candidate, adam_candidate),
    }


def _run_seed(
    input_sweep: InputSweep,
    output_directory: Path,
    *,
    experiment: Mapping[str, object],
    qpc: tuple[int, ...],
    sgd_learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    config = input_sweep.config
    child_input = input_sweep.children[seed]
    input_identity = input_sweep.child_identities[seed]
    child_identity = _child_identity(experiment, input_identity, seed)
    child_output = output_directory / f"seed-{seed}"
    summary_path = child_output / "summary.json"
    summary = _load_output_summary(summary_path, child_identity, qpc)
    expected = {str(value) for value in qpc}
    if set(summary["variants"]) == expected:
        print(f"SEED {seed} optimizer sweep cached", flush=True)
        return summary

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    reference_path, _self_play_path = _checkpoint_paths(input_identity)
    reference = load_policy(reference_path, device, frozen=True)
    require_deterministic_actor(reference)
    reference_digest = batch_sweep._model_digest(reference)
    train = load_counterfactual_batch(child_input / "shared" / "train.npz")
    calibration = load_policy_state_batch(child_input / "shared" / "calibration.npz")
    heldout_batch = load_counterfactual_batch(
        child_input / "shared" / "heldout.npz"
    )
    maximum = config.batch_queries_per_category[-1]
    if len(train) != 9 * maximum:
        raise ValueError("input training corpus has the wrong state count")
    if len(calibration) != 9 * config.calibration_queries_per_category:
        raise ValueError("input calibration corpus has the wrong state count")
    if len(heldout_batch) != 9 * config.heldout_queries_per_category:
        raise ValueError("input heldout corpus has the wrong state count")
    manifest = _json(child_input / "shared" / "manifest.json")
    visit_weights = np.asarray(
        manifest["source_visit_frequencies"]["vector"], dtype=np.float64
    )
    if visit_weights.shape != (9,) or not np.isfinite(visit_weights).all():
        raise ValueError("input source visitation weights are invalid")
    nested = nested_category_indices(train.categories, config.batch_queries_per_category)
    progress = Progress()
    print(
        f"CUDA {torch.cuda.get_device_name(device)}  optimizer sweep  "
        f"seed {seed}  QPC {','.join(map(str, qpc))}  SGD lr {sgd_learning_rate:g}",
        flush=True,
    )

    maximum_batch = subset_counterfactual_batch(train, nested[maximum])
    max_initial, max_candidate, _max_metrics, _max_seconds = _sgd_direction(
        child_output,
        experiment=experiment,
        reference=reference,
        reference_digest=reference_digest,
        batch=maximum_batch,
        qpc=maximum,
        visit_weights=visit_weights,
        config=config,
        learning_rate=sgd_learning_rate,
        device=device,
        progress=progress,
    )
    panel_seeds, reference_ranks, reference_scores = _reference_panel(
        child_input, input_identity, config, seed
    )

    for value in qpc:
        key = str(value)
        if key in summary["variants"]:
            continue
        batch = subset_counterfactual_batch(train, nested[value])
        if value == maximum:
            initial, candidate_state, optimizer_metrics, direction_seconds = (
                max_initial,
                max_candidate,
                _max_metrics,
                _max_seconds,
            )
        else:
            initial, candidate_state, optimizer_metrics, direction_seconds = _sgd_direction(
                child_output,
                experiment=experiment,
                reference=reference,
                reference_digest=reference_digest,
                batch=batch,
                qpc=value,
                visit_weights=visit_weights,
                config=config,
                learning_rate=sgd_learning_rate,
                device=device,
                progress=progress,
            )
        _adam_initial, adam_candidate, adam_metrics, _adam_direction_seconds = (
            _input_adam_direction(
                child_input,
                input_identity,
                reference_digest,
                value,
                batch,
            )
        )
        direction = _direction_summary(
            initial, candidate_state, max_candidate, adam_candidate
        )
        actor = copy.deepcopy(reference).to(device)
        actor.load_state_dict(candidate_state, strict=True)
        max_kl_evaluations = (
            config.kl_search_steps
            + 2
            + int(math.ceil(math.log2(config.maximum_scale)))
        )
        progress.start(
            f"QPC{value}_SGD_KL", total=max_kl_evaluations, unit="evaluations"
        )
        started = time.perf_counter()
        calibration_metrics = calibrate_direction(
            actor,
            reference,
            initial,
            candidate_state,
            calibration,
            device,
            category_weights=visit_weights,
            target_kl=config.target_kl,
            batch_size=config.inference_batch_size,
            search_steps=config.kl_search_steps,
            maximum_scale=config.maximum_scale,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        calibration_seconds = time.perf_counter() - started
        progress.complete(
            int(calibration_metrics["evaluations"]),
            fields={
                "scale": calibration_metrics["scale"],
                "kl": calibration_metrics["final_kl"],
            },
        )

        started = time.perf_counter()
        heldout_metrics = heldout_policy_value(
            actor,
            reference,
            heldout_batch,
            device,
            category_weights=visit_weights,
            batch_size=config.inference_batch_size,
        )
        heldout_seconds = time.perf_counter() - started
        save_policy(child_output / f"actor-sgd-qpc{value}.pt", actor)

        actor_digest = batch_sweep._model_digest(actor)
        panel_fingerprint = batch_sweep._fingerprint(
            {
                "experiment": experiment,
                "seed": seed,
                "qpc": value,
                "optimizer": OPTIMIZER_NAME,
                "actor": actor_digest,
                "seeds": batch_sweep._fingerprint(panel_seeds.tolist()),
            }
        )
        panel_path = child_output / "evaluation" / f"sgd-qpc-{value}.npz"
        if panel_path.exists():
            actor_seeds, actor_ranks, actor_scores, evaluation_seconds = (
                batch_sweep._load_actor_panel(panel_path, fingerprint=panel_fingerprint)
            )
            if not np.array_equal(actor_seeds, panel_seeds):
                raise ValueError("cached SGD evaluation seeds do not match")
            print(f"QPC {value:4d} SGD evaluation cached", flush=True)
        else:
            progress.start(
                f"QPC{value}_SGD_EVAL", total=len(panel_seeds), unit="games"
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

        adam = load_policy(child_input / f"actor-qpc{value}.pt", device, frozen=True)
        require_deterministic_actor(adam)
        adam_seeds, adam_ranks, adam_scores, _adam_panel_seconds = _input_adam_panel(
            child_input, input_identity, value, panel_seeds, adam
        )
        if not np.array_equal(adam_seeds, panel_seeds):
            raise ValueError("input AdamW evaluation seeds do not match")
        evaluation_vs_reference = summarize_paired(
            actor_ranks,
            actor_scores,
            reference_ranks,
            reference_scores,
            seed=domain_seed(seed, OPTIMIZER_DOMAIN + 1, value),
            bootstrap_samples=config.bootstrap_samples,
        )
        evaluation_vs_adamw = summarize_paired(
            actor_ranks,
            actor_scores,
            adam_ranks,
            adam_scores,
            seed=domain_seed(seed, OPTIMIZER_DOMAIN + 2, value),
            bootstrap_samples=config.bootstrap_samples,
        )
        elapsed = (
            direction_seconds
            + calibration_seconds
            + heldout_seconds
            + evaluation_seconds
        )
        summary["variants"][key] = {
            "queries_per_category": value,
            "states": len(batch),
            "sgd_optimizer": optimizer_metrics,
            "adamw_optimizer": adam_metrics,
            "direction": direction,
            "calibration": calibration_metrics,
            "heldout": heldout_metrics,
            "evaluation_vs_reference": evaluation_vs_reference,
            "evaluation_vs_adamw": evaluation_vs_adamw,
            "timing": {
                "direction_seconds": direction_seconds,
                "calibration_seconds": calibration_seconds,
                "heldout_seconds": heldout_seconds,
                "evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": elapsed,
                "direction_states_per_second": len(batch)
                / max(direction_seconds, 1e-9),
            },
        }
        batch_sweep._atomic_json(summary_path, summary)
        rank = evaluation_vs_adamw["paired_rank_delta"]
        print(
            f"QPC {value:4d} SGD-vs-Adam dRank {rank['mean']:+.4f} "
            f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]  "
            f"cos {direction['to_adamw']['cosine_to_maximum']:.4f}  "
            f"heldout {heldout_metrics['visitation_weighted_rank_value']:+.5f}",
            flush=True,
        )
        del adam
        del actor
        torch.cuda.empty_cache()

    return summary


def _aggregate(
    input_sweep: InputSweep,
    output_directory: Path,
    *,
    experiment: Mapping[str, object],
    qpc: Sequence[int],
    child_summaries: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    variants: dict[str, object] = {}
    for value in qpc:
        key = str(value)
        sgd_ranks: list[np.ndarray] = []
        sgd_scores: list[np.ndarray] = []
        adam_ranks: list[np.ndarray] = []
        adam_scores: list[np.ndarray] = []
        reference_ranks: list[np.ndarray] = []
        reference_scores: list[np.ndarray] = []
        all_seeds: list[np.ndarray] = []
        rank_means: list[float] = []
        heldout_values: list[float] = []
        per_seed: dict[str, object] = {}
        for seed in input_sweep.seeds:
            input_child = input_sweep.children[seed]
            input_identity = input_sweep.child_identities[seed]
            output_child = output_directory / f"seed-{seed}"
            variant = child_summaries[seed]["variants"][key]
            sgd_panel_seeds, sgd_rank, sgd_score, panel_seconds = (
                batch_sweep._load_actor_panel_unchecked(
                    output_child / "evaluation" / f"sgd-qpc-{value}.npz"
                )
            )
            reference_seed, reference_rank, reference_score = _reference_panel(
                input_child, input_identity, input_sweep.config, seed
            )
            adam_seed, adam_rank, adam_score, _adam_panel_seconds = (
                batch_sweep._load_actor_panel_unchecked(
                    input_child / "evaluation" / f"qpc-{value}.npz"
                )
            )
            if not (
                np.array_equal(sgd_panel_seeds, reference_seed)
                and np.array_equal(sgd_panel_seeds, adam_seed)
            ):
                raise ValueError(f"optimizer sweep panels do not align for seed {seed}")
            raw_rank_mean = float((sgd_rank - adam_rank).mean())
            reported = float(
                variant["evaluation_vs_adamw"]["paired_rank_delta"]["mean"]
            )
            if not math.isclose(raw_rank_mean, reported, abs_tol=1e-12):
                raise ValueError(f"optimizer sweep panel metrics do not match for seed {seed}")
            sgd_ranks.append(sgd_rank)
            sgd_scores.append(sgd_score)
            adam_ranks.append(adam_rank)
            adam_scores.append(adam_score)
            reference_ranks.append(reference_rank)
            reference_scores.append(reference_score)
            all_seeds.append(sgd_panel_seeds)
            rank_means.append(raw_rank_mean)
            heldout_values.append(
                float(variant["heldout"]["visitation_weighted_rank_value"])
            )
            per_seed[str(seed)] = {
                "evaluation_vs_adamw": variant["evaluation_vs_adamw"],
                "evaluation_vs_reference": variant["evaluation_vs_reference"],
                "heldout": variant["heldout"],
                "timing": variant["timing"],
                "panel_seconds": panel_seconds,
            }
        joined_seeds = np.concatenate(all_seeds)
        if len(np.unique(joined_seeds)) != len(joined_seeds):
            raise ValueError(f"optimizer sweep evaluation panels overlap for QPC {value}")
        variants[key] = {
            "queries_per_category": int(value),
            "states": int(9 * value),
            "seeds": per_seed,
            "pooled_evaluation_vs_adamw": summarize_paired(
                np.concatenate(sgd_ranks),
                np.concatenate(sgd_scores),
                np.concatenate(adam_ranks),
                np.concatenate(adam_scores),
                seed=batch_sweep._multi_summary_seed(value, input_sweep.seeds),
                bootstrap_samples=input_sweep.config.bootstrap_samples,
            ),
            "pooled_evaluation_vs_reference": summarize_paired(
                np.concatenate(sgd_ranks),
                np.concatenate(sgd_scores),
                np.concatenate(reference_ranks),
                np.concatenate(reference_scores),
                seed=batch_sweep._multi_summary_seed(value + 0x10000, input_sweep.seeds),
                bootstrap_samples=input_sweep.config.bootstrap_samples,
            ),
            "seed_metrics": {
                "paired_rank_delta_vs_adamw": batch_sweep._numeric_summary(rank_means),
                "heldout_visitation_weighted_rank_value": batch_sweep._numeric_summary(
                    heldout_values
                ),
            },
        }
    return {
        "identity": dict(experiment),
        "seeds": [int(seed) for seed in input_sweep.seeds],
        "variants": variants,
    }


def _load_aggregate(
    path: Path,
    identity: Mapping[str, object],
    qpc: Sequence[int],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    value = _json(path)
    if (
        value.get("identity") != identity
        or not isinstance(value.get("variants"), dict)
        or set(value["variants"]) != {str(item) for item in qpc}
    ):
        raise ValueError("optimizer sweep aggregate identity does not match")
    return value


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    sgd_learning_rate: float = 0.1,
    qpc: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if not math.isfinite(sgd_learning_rate) or sgd_learning_rate <= 0:
        raise ValueError("sgd learning rate must be positive and finite")
    input_sweep = load_input_sweep(batch_sweep_directory, seeds=seeds)
    requested = (
        input_sweep.config.batch_queries_per_category
        if qpc is None
        else tuple(int(value) for value in qpc)
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("optimizer sweep QPC values must be unique and non-empty")
    if tuple(sorted(requested)) != requested or not set(requested).issubset(
        input_sweep.config.batch_queries_per_category
    ):
        raise ValueError("optimizer sweep QPC values must be an increasing input subset")
    experiment = _experiment_identity(input_sweep, requested, sgd_learning_rate)
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, experiment)
    aggregate_path = output_directory / "aggregate.json"
    cached = _load_aggregate(aggregate_path, experiment, requested)
    if cached is not None:
        print(f"RESULT optimizer sweep cached  {aggregate_path}", flush=True)
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
            qpc=requested,
            sgd_learning_rate=sgd_learning_rate,
            seed=seed,
            device=resolved,
        )
    aggregate = _aggregate(
        input_sweep,
        output_directory,
        experiment=experiment,
        qpc=requested,
        child_summaries=children,
    )
    batch_sweep._atomic_json(aggregate_path, aggregate)
    for value in requested:
        result = aggregate["variants"][str(value)]["pooled_evaluation_vs_adamw"]
        rank = result["paired_rank_delta"]
        print(
            f"QPC {value:4d} pooled SGD-vs-Adam dRank {rank['mean']:+.4f} "
            f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]",
            flush=True,
        )
    print(f"RESULT optimizer sweep complete  {aggregate_path}", flush=True)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sgd-learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--qpc",
        type=int,
        nargs="+",
        help="increasing subset of completed input QPC values (default: all)",
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
        sgd_learning_rate=args.sgd_learning_rate,
        qpc=args.qpc,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
