"""Diagnose cross-seed optimizer-direction generalization at one checkpoint.

This experiment is inference-only. It reuses AdamW and SGD directions plus
calibration/heldout corpora from completed sweeps, normalizes every direction
to one small KL on a shared probe, then measures:

* direction-seed x heldout-seed Q generalization;
* parameter-space and policy-space pairwise alignment;
* category-level probability alignment and greedy action conflicts.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import batch_sweep, optimizer_sweep
from .pipeline import _autocast, load_policy
from .policy_iteration import (
    CounterfactualBatch,
    PolicyStateBatch,
    _state_tensors,
    calibrate_direction,
    category_row_weights,
    direction_cosine,
    load_counterfactual_batch,
    load_policy_state_batch,
    nested_category_indices,
    require_cuda,
    require_deterministic_actor,
    subset_counterfactual_batch,
)
from .policy_pool import CATEGORY_COUNT, CATEGORY_NAMES


DIRECTION_GENERALIZATION_VERSION = 2
DEFAULT_TARGET_KL = 1e-4


@dataclass(frozen=True)
class CachedDirection:
    name: str
    optimizer: str
    seed: int
    initial: Mapping[str, torch.Tensor]
    candidate: Mapping[str, torch.Tensor]
    optimizer_metrics: Mapping[str, object]
    elapsed_seconds: float


def _concatenate_states(batches: Sequence[PolicyStateBatch]) -> PolicyStateBatch:
    if len(batches) < 2:
        raise ValueError("shared probe needs at least two state batches")
    fields = PolicyStateBatch.__dataclass_fields__
    values = {
        name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
        for name in fields
        if name != "query_ids"
    }
    values["query_ids"] = np.arange(
        sum(len(batch) for batch in batches), dtype=np.int64
    )
    return PolicyStateBatch(**values)


def _normalize_seeds(
    available: Sequence[int], requested: Sequence[int] | None
) -> tuple[int, ...]:
    normalized = (
        tuple(int(value) for value in available)
        if requested is None
        else tuple(int(value) for value in requested)
    )
    if (
        len(normalized) < 2
        or len(set(normalized)) != len(normalized)
        or not set(normalized).issubset(set(int(value) for value in available))
    ):
        raise ValueError("direction generalization needs at least two unique input seeds")
    return normalized


def _load_sgd_identity(
    directory: Path,
    input_sweep: optimizer_sweep.InputSweep,
    *,
    qpc: int,
    seeds: Sequence[int],
) -> tuple[Path, Mapping[str, object]]:
    root = directory.resolve()
    path = root / "optimizer_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing optimizer_config.json in {root}")
    identity = optimizer_sweep._json(path)
    configured_qpc = identity.get("queries_per_category")
    configured_seeds = identity.get("seeds")
    if (
        identity.get("optimizer") != "sgd"
        or identity.get("input_identity_fingerprint")
        != batch_sweep._fingerprint(input_sweep.identity)
        or Path(str(identity.get("input_directory"))).resolve() != input_sweep.root
        or not isinstance(configured_qpc, list)
        or qpc not in {int(value) for value in configured_qpc}
        or not isinstance(configured_seeds, list)
        or not set(int(seed) for seed in seeds).issubset(
            {int(value) for value in configured_seeds}
        )
    ):
        raise ValueError("SGD optimizer sweep does not match the batch sweep request")
    learning_rate = float(identity.get("sgd_learning_rate", math.nan))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("SGD optimizer sweep has an invalid learning rate")
    return root, identity


def _experiment_identity(
    input_sweep: optimizer_sweep.InputSweep,
    sgd_identity: Mapping[str, object],
    *,
    qpc: int,
    seeds: Sequence[int],
    target_kl: float,
) -> dict[str, object]:
    return {
        "version": DIRECTION_GENERALIZATION_VERSION,
        "batch_sweep_directory": str(input_sweep.root),
        "batch_sweep_identity_fingerprint": batch_sweep._fingerprint(
            input_sweep.identity
        ),
        "sgd_sweep_identity_fingerprint": batch_sweep._fingerprint(sgd_identity),
        "queries_per_category": int(qpc),
        "seeds": [int(seed) for seed in seeds],
        "target_kl": float(target_kl),
        "calibration_queries_per_category_per_seed": int(
            input_sweep.config.calibration_queries_per_category
        ),
        "heldout_queries_per_category_per_seed": int(
            input_sweep.config.heldout_queries_per_category
        ),
        "inference_batch_size": int(input_sweep.config.inference_batch_size),
    }


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("direction-generalization output configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("direction-generalization output directory is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _load_sgd_direction(
    directory: Path,
    identity: Mapping[str, object],
    *,
    reference_digest: str,
    batch: CounterfactualBatch,
    qpc: int,
    seed: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, object],
    float,
]:
    learning_rate = float(identity["sgd_learning_rate"])
    fingerprint = batch_sweep._fingerprint(
        {
            "experiment": identity,
            "reference": reference_digest,
            "optimizer": "sgd",
            "learning_rate": learning_rate,
            "qpc": qpc,
            "query_ids": batch.query_ids.tolist(),
        }
    )
    return batch_sweep._load_direction(
        directory / f"seed-{seed}" / "directions" / f"sgd-qpc-{qpc}.pt",
        fingerprint=fingerprint,
    )


def _state_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _policy_outputs(
    actor,
    states: PolicyStateBatch,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(states), states.legal.shape[1]), dtype=np.float32)
    actions = np.empty(len(states), dtype=np.int64)
    actor.eval()
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        rows = np.arange(start, stop)
        state = _state_tensors(states, rows, device)
        legal = torch.as_tensor(states.legal[rows], device=device)
        with torch.inference_mode(), _autocast(device):
            logits = actor(*state, legal).raw_logits.float()
        masked = logits.masked_fill(~legal, -torch.inf)
        chunk = F.softmax(masked, dim=-1)
        probabilities[start:stop] = chunk.cpu().numpy()
        actions[start:stop] = masked.argmax(dim=-1).cpu().numpy()
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities[~states.legal] != 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise RuntimeError("policy probe produced invalid probabilities")
    return probabilities, actions


def _weighted_delta_stats(
    delta: np.ndarray,
    reference_actions: np.ndarray,
    actor_actions: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    squared = np.square(delta, dtype=np.float64).sum(axis=1)
    l1 = np.abs(delta, dtype=np.float64).sum(axis=1)
    flips = actor_actions != reference_actions
    return {
        "probability_delta_l2": math.sqrt(float(np.dot(weights, squared))),
        "mean_probability_l1": float(np.dot(weights, l1)),
        "greedy_flip_rate": float(np.dot(weights, flips.astype(np.float64))),
    }


def _policy_signature(
    probabilities: np.ndarray,
    actions: np.ndarray,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    states: PolicyStateBatch,
    category_weights: np.ndarray,
) -> dict[str, object]:
    delta = probabilities - reference_probabilities
    weights = category_row_weights(states.categories, category_weights)
    categories: dict[str, object] = {}
    for category, name in enumerate(CATEGORY_NAMES):
        rows = states.categories == category
        row_weights = np.full(int(rows.sum()), 1.0 / int(rows.sum()))
        categories[name] = {
            "states": int(rows.sum()),
            **_weighted_delta_stats(
                delta[rows],
                reference_actions[rows],
                actions[rows],
                row_weights,
            ),
        }
    return {
        "states": len(states),
        **_weighted_delta_stats(
            delta, reference_actions, actions, weights
        ),
        "categories": categories,
    }


def _reference_policy_summary(
    probabilities: np.ndarray,
    states: PolicyStateBatch,
    category_weights: np.ndarray,
) -> dict[str, object]:
    safe = np.clip(probabilities, 1e-30, 1.0)
    entropy = -(probabilities * np.log(safe)).sum(axis=1, dtype=np.float64)
    top = probabilities.max(axis=1)
    legal_counts = states.legal.sum(axis=1).astype(np.float64)

    def summarize(weights: np.ndarray, rows: np.ndarray) -> dict[str, float]:
        return {
            "mean_entropy": float(np.dot(weights, entropy[rows])),
            "mean_top_probability": float(np.dot(weights, top[rows])),
            "top_probability_ge_0_99_rate": float(
                np.dot(weights, (top[rows] >= 0.99).astype(np.float64))
            ),
            "top_probability_ge_0_999_rate": float(
                np.dot(weights, (top[rows] >= 0.999).astype(np.float64))
            ),
            "mean_legal_actions": float(np.dot(weights, legal_counts[rows])),
        }

    all_rows = np.ones(len(states), dtype=np.bool_)
    global_weights = category_row_weights(states.categories, category_weights)
    categories: dict[str, object] = {}
    for category, name in enumerate(CATEGORY_NAMES):
        rows = states.categories == category
        weights = np.full(int(rows.sum()), 1.0 / int(rows.sum()))
        categories[name] = {"states": int(rows.sum()), **summarize(weights, rows)}
    return {
        "states": len(states),
        **summarize(global_weights, all_rows),
        "categories": categories,
    }


def _weighted_pair_stats(
    left_delta: np.ndarray,
    right_delta: np.ndarray,
    reference_actions: np.ndarray,
    left_actions: np.ndarray,
    right_actions: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | None]:
    state_dot = (left_delta.astype(np.float64) * right_delta).sum(axis=1)
    left_squared = np.square(left_delta, dtype=np.float64).sum(axis=1)
    right_squared = np.square(right_delta, dtype=np.float64).sum(axis=1)
    dot = float(np.dot(weights, state_dot))
    left_norm = math.sqrt(float(np.dot(weights, left_squared)))
    right_norm = math.sqrt(float(np.dot(weights, right_squared)))
    denominator = left_norm * right_norm
    left_flip = left_actions != reference_actions
    right_flip = right_actions != reference_actions
    either_flip = left_flip | right_flip
    both_flip = left_flip & right_flip
    disagreement = left_actions != right_actions
    either_rate = float(np.dot(weights, either_flip.astype(np.float64)))
    disagreement_rate = float(np.dot(weights, disagreement.astype(np.float64)))
    return {
        "probability_delta_cosine": dot / denominator if denominator > 0 else None,
        "left_probability_delta_l2": left_norm,
        "right_probability_delta_l2": right_norm,
        "aligned_state_rate": float(np.dot(weights, (state_dot > 0).astype(np.float64))),
        "opposed_state_rate": float(np.dot(weights, (state_dot < 0).astype(np.float64))),
        "left_greedy_flip_rate": float(np.dot(weights, left_flip.astype(np.float64))),
        "right_greedy_flip_rate": float(np.dot(weights, right_flip.astype(np.float64))),
        "both_greedy_flip_rate": float(np.dot(weights, both_flip.astype(np.float64))),
        "action_disagreement_rate": disagreement_rate,
        "action_disagreement_given_either_flip": (
            disagreement_rate / either_rate if either_rate > 0 else None
        ),
    }


def _policy_pair_metrics(
    left_probabilities: np.ndarray,
    left_actions: np.ndarray,
    right_probabilities: np.ndarray,
    right_actions: np.ndarray,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    states: PolicyStateBatch,
    category_weights: np.ndarray,
) -> dict[str, object]:
    left_delta = left_probabilities - reference_probabilities
    right_delta = right_probabilities - reference_probabilities
    weights = category_row_weights(states.categories, category_weights)
    categories: dict[str, object] = {}
    for category, name in enumerate(CATEGORY_NAMES):
        rows = states.categories == category
        row_weights = np.full(int(rows.sum()), 1.0 / int(rows.sum()))
        categories[name] = {
            "states": int(rows.sum()),
            **_weighted_pair_stats(
                left_delta[rows],
                right_delta[rows],
                reference_actions[rows],
                left_actions[rows],
                right_actions[rows],
                row_weights,
            ),
        }
    return {
        "states": len(states),
        **_weighted_pair_stats(
            left_delta,
            right_delta,
            reference_actions,
            left_actions,
            right_actions,
            weights,
        ),
        "categories": categories,
    }


def _mean_standard_error(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("diagnostic values must be finite and non-empty")
    mean = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    )
    return mean, standard_error


def _stratified_metric(
    values: np.ndarray,
    categories: np.ndarray,
    category_weights: np.ndarray,
) -> tuple[float, float]:
    mean = variance = 0.0
    for category in range(CATEGORY_COUNT):
        rows = values[categories == category]
        category_mean, category_se = _mean_standard_error(rows)
        weight = float(category_weights[category])
        mean += weight * category_mean
        variance += weight * weight * category_se * category_se
    return mean, math.sqrt(variance)


def _interval(mean: float, standard_error: float) -> dict[str, float]:
    return {
        "mean": float(mean),
        "standard_error": float(standard_error),
        "ci95_low": float(mean - 1.96 * standard_error),
        "ci95_high": float(mean + 1.96 * standard_error),
    }


def _heldout_metrics(
    probabilities: np.ndarray,
    actions: np.ndarray,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    batch: CounterfactualBatch,
    category_weights: np.ndarray,
) -> dict[str, object]:
    delta = probabilities - reference_probabilities
    rank_values = (delta * batch.rank_q).sum(axis=1, dtype=np.float64)
    score_values = (delta * batch.score_q).sum(axis=1, dtype=np.float64)
    rows = np.arange(len(batch))
    greedy_rank_values = (
        batch.rank_q[rows, actions] - batch.rank_q[rows, reference_actions]
    ).astype(np.float64)
    greedy_score_values = (
        batch.score_q[rows, actions] - batch.score_q[rows, reference_actions]
    ).astype(np.float64)
    legal = batch.legal
    safe_probabilities = np.where(legal, np.clip(probabilities, 1e-30, 1.0), 1.0)
    safe_reference = np.where(
        legal, np.clip(reference_probabilities, 1e-30, 1.0), 1.0
    )
    kl_values = np.where(
        legal,
        probabilities * (np.log(safe_probabilities) - np.log(safe_reference)),
        0.0,
    ).sum(axis=1, dtype=np.float64)
    rank_mean, rank_se = _stratified_metric(
        rank_values, batch.categories, category_weights
    )
    score_mean, score_se = _stratified_metric(
        score_values, batch.categories, category_weights
    )
    greedy_rank_mean, greedy_rank_se = _stratified_metric(
        greedy_rank_values, batch.categories, category_weights
    )
    greedy_score_mean, greedy_score_se = _stratified_metric(
        greedy_score_values, batch.categories, category_weights
    )
    row_weights = category_row_weights(batch.categories, category_weights)
    categories: dict[str, object] = {}
    for category, name in enumerate(CATEGORY_NAMES):
        mask = batch.categories == category
        rank_category, rank_category_se = _mean_standard_error(rank_values[mask])
        greedy_category, greedy_category_se = _mean_standard_error(
            greedy_rank_values[mask]
        )
        categories[name] = {
            "states": int(mask.sum()),
            "soft_rank_value": _interval(rank_category, rank_category_se),
            "greedy_rank_value": _interval(greedy_category, greedy_category_se),
            "greedy_flip_rate": float(
                np.mean(actions[mask] != reference_actions[mask])
            ),
        }
    return {
        "states": len(batch),
        "soft_rank_value": _interval(rank_mean, rank_se),
        "soft_score_value": _interval(score_mean, score_se),
        "greedy_rank_value": _interval(greedy_rank_mean, greedy_rank_se),
        "greedy_score_value": _interval(greedy_score_mean, greedy_score_se),
        "visitation_weighted_kl": float(np.dot(row_weights, kl_values)),
        "greedy_flip_rate": float(
            np.dot(row_weights, (actions != reference_actions).astype(np.float64))
        ),
        "categories": categories,
    }


def _heldout_target_summary(
    batch: CounterfactualBatch,
    reference_actions: np.ndarray,
    category_weights: np.ndarray,
) -> dict[str, object]:
    legal_max = np.where(batch.legal, batch.rank_q, -np.inf).max(axis=1)
    legal_min = np.where(batch.legal, batch.rank_q, np.inf).min(axis=1)
    rows = np.arange(len(batch))
    reference_q = batch.rank_q[rows, reference_actions]
    best_actions = np.where(batch.legal, batch.rank_q, -np.inf).argmax(axis=1)
    q_range = (legal_max - legal_min).astype(np.float64)
    best_advantage = (legal_max - reference_q).astype(np.float64)
    reference_not_best = best_actions != reference_actions
    row_weights = category_row_weights(batch.categories, category_weights)

    def summarize(weights: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        centered = batch.centered_rank_q[mask][batch.legal[mask]]
        return {
            "mean_legal_q_range": float(np.dot(weights, q_range[mask])),
            "mean_best_vs_reference_q": float(
                np.dot(weights, best_advantage[mask])
            ),
            "reference_not_q_best_rate": float(
                np.dot(weights, reference_not_best[mask].astype(np.float64))
            ),
            "mean_absolute_centered_legal_q": float(np.mean(np.abs(centered))),
        }

    all_rows = np.ones(len(batch), dtype=np.bool_)
    categories: dict[str, object] = {}
    for category, name in enumerate(CATEGORY_NAMES):
        mask = batch.categories == category
        weights = np.full(int(mask.sum()), 1.0 / int(mask.sum()))
        categories[name] = {"states": int(mask.sum()), **summarize(weights, mask)}
    return {
        "states": len(batch),
        **summarize(row_weights, all_rows),
        "categories": categories,
    }


def _pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _generalization_summary(
    directions: Mapping[str, Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for optimizer in ("adamw", "sgd"):
        diagonal_soft: list[float] = []
        off_diagonal_soft: list[float] = []
        diagonal_greedy: list[float] = []
        off_diagonal_greedy: list[float] = []
        for name, direction in directions.items():
            if direction["optimizer"] != optimizer:
                continue
            direction_seed = int(direction["seed"])
            for target_seed in seeds:
                heldout = direction["heldout"][str(target_seed)]
                soft = float(heldout["soft_rank_value"]["mean"])
                greedy = float(heldout["greedy_rank_value"]["mean"])
                if direction_seed == target_seed:
                    diagonal_soft.append(soft)
                    diagonal_greedy.append(greedy)
                else:
                    off_diagonal_soft.append(soft)
                    off_diagonal_greedy.append(greedy)
        result[optimizer] = {
            "own_seed_soft_rank_value": batch_sweep._numeric_summary(diagonal_soft),
            "other_seed_soft_rank_value": batch_sweep._numeric_summary(
                off_diagonal_soft
            ),
            "own_seed_greedy_rank_value": batch_sweep._numeric_summary(
                diagonal_greedy
            ),
            "other_seed_greedy_rank_value": batch_sweep._numeric_summary(
                off_diagonal_greedy
            ),
        }
    return result


def run(
    batch_sweep_directory: Path,
    optimizer_sweep_directory: Path,
    output_directory: Path,
    *,
    qpc: int = 256,
    seeds: Sequence[int] | None = None,
    target_kl: float = DEFAULT_TARGET_KL,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if not math.isfinite(target_kl) or target_kl <= 0:
        raise ValueError("direction-generalization target KL must be positive")
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = _normalize_seeds(input_sweep.seeds, seeds)
    if qpc not in input_sweep.config.batch_queries_per_category:
        raise ValueError("direction-generalization QPC is not in the input sweep")
    sgd_root, sgd_identity = _load_sgd_identity(
        optimizer_sweep_directory,
        input_sweep,
        qpc=qpc,
        seeds=selected_seeds,
    )
    identity = _experiment_identity(
        input_sweep,
        sgd_identity,
        qpc=qpc,
        seeds=selected_seeds,
        target_kl=target_kl,
    )
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached direction-generalization identity differs")
        print(f"RESULT direction generalization cached  {result_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    config = input_sweep.config

    reference = None
    reference_digest = ""
    reference_state: Mapping[str, torch.Tensor] | None = None
    calibration_batches: list[PolicyStateBatch] = []
    heldout_batches: dict[int, CounterfactualBatch] = {}
    visit_weights: dict[int, np.ndarray] = {}
    cached_directions: list[CachedDirection] = []
    for seed in selected_seeds:
        child = input_sweep.children[seed]
        input_identity = input_sweep.child_identities[seed]
        reference_path, _self_play_path = optimizer_sweep._checkpoint_paths(
            input_identity
        )
        if reference is None:
            reference = load_policy(reference_path, resolved, frozen=True)
            require_deterministic_actor(reference)
            reference_digest = batch_sweep._model_digest(reference)
            reference_state = {
                name: value.detach().cpu().clone()
                for name, value in reference.state_dict().items()
            }
        elif str(input_identity["reference_sha256"]) != str(
            input_sweep.child_identities[selected_seeds[0]]["reference_sha256"]
        ):
            raise ValueError("input seeds do not share one reference checkpoint")

        train = load_counterfactual_batch(child / "shared" / "train.npz")
        nested = nested_category_indices(
            train.categories, config.batch_queries_per_category
        )
        train_qpc = subset_counterfactual_batch(train, nested[qpc])
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
            raise ValueError(f"seed {seed} source visitation weights are invalid")
        visit_weights[seed] = weights

        adam_initial, adam_candidate, adam_metrics, adam_seconds = (
            optimizer_sweep._input_adam_direction(
                child,
                input_identity,
                reference_digest,
                qpc,
                train_qpc,
            )
        )
        sgd_initial, sgd_candidate, sgd_metrics, sgd_seconds = _load_sgd_direction(
            sgd_root,
            sgd_identity,
            reference_digest=reference_digest,
            batch=train_qpc,
            qpc=qpc,
            seed=seed,
        )
        assert reference_state is not None
        if not _state_equal(adam_initial, reference_state) or not _state_equal(
            sgd_initial, reference_state
        ):
            raise ValueError("cached direction does not start at the shared reference")
        cached_directions.extend(
            (
                CachedDirection(
                    name=f"adamw-seed-{seed}",
                    optimizer="adamw",
                    seed=seed,
                    initial=adam_initial,
                    candidate=adam_candidate,
                    optimizer_metrics=adam_metrics,
                    elapsed_seconds=adam_seconds,
                ),
                CachedDirection(
                    name=f"sgd-seed-{seed}",
                    optimizer="sgd",
                    seed=seed,
                    initial=sgd_initial,
                    candidate=sgd_candidate,
                    optimizer_metrics=sgd_metrics,
                    elapsed_seconds=sgd_seconds,
                ),
            )
        )

    assert reference is not None
    common_probe = _concatenate_states(calibration_batches)
    common_weights = np.stack(
        [visit_weights[seed] for seed in selected_seeds]
    ).mean(axis=0)
    common_weights /= common_weights.sum()
    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  direction generalization  "
        f"QPC {qpc}  seeds {','.join(map(str, selected_seeds))}  "
        f"shared probe {len(common_probe)}  target KL {target_kl:g}",
        flush=True,
    )

    actors: dict[str, object] = {}
    direction_results: dict[str, dict[str, object]] = {}
    max_kl_evaluations = (
        config.kl_search_steps
        + 2
        + int(math.ceil(math.log2(config.maximum_scale)))
    )
    for direction in cached_directions:
        actor = copy.deepcopy(reference).to(resolved)
        actor.load_state_dict(direction.candidate, strict=True)
        calibration = calibrate_direction(
            actor,
            reference,
            direction.initial,
            direction.candidate,
            common_probe,
            resolved,
            category_weights=common_weights,
            target_kl=target_kl,
            batch_size=config.inference_batch_size,
            search_steps=config.kl_search_steps,
            maximum_scale=config.maximum_scale,
        )
        if int(calibration["evaluations"]) > max_kl_evaluations:
            raise RuntimeError("KL calibration exceeded its evaluation bound")
        actors[direction.name] = actor
        direction_results[direction.name] = {
            "optimizer": direction.optimizer,
            "seed": direction.seed,
            "raw_optimizer": dict(direction.optimizer_metrics),
            "raw_direction_seconds": direction.elapsed_seconds,
            "common_probe_calibration": calibration,
            "heldout": {},
        }
        print(
            f"DIRECTION {direction.name:24s}  scale {calibration['scale']:.6g}  "
            f"KL {calibration['final_kl']:.6g}  "
            f"flip {100 * calibration['greedy_flip_rate']:.3f}%",
            flush=True,
        )

    reference_probe_probabilities, reference_probe_actions = _policy_outputs(
        reference,
        common_probe,
        resolved,
        batch_size=config.inference_batch_size,
    )
    reference_probe_summary = _reference_policy_summary(
        reference_probe_probabilities,
        common_probe,
        common_weights,
    )
    probe_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, actor in actors.items():
        probabilities, actions = _policy_outputs(
            actor,
            common_probe,
            resolved,
            batch_size=config.inference_batch_size,
        )
        probe_outputs[name] = (probabilities, actions)
        direction_results[name]["common_probe_policy"] = _policy_signature(
            probabilities,
            actions,
            reference_probe_probabilities,
            reference_probe_actions,
            common_probe,
            common_weights,
        )

    heldout_targets: dict[str, object] = {}
    for target_seed in selected_seeds:
        target_batch = heldout_batches[target_seed]
        target_weights = visit_weights[target_seed]
        reference_probabilities, reference_actions = _policy_outputs(
            reference,
            target_batch,
            resolved,
            batch_size=config.inference_batch_size,
        )
        heldout_targets[str(target_seed)] = _heldout_target_summary(
            target_batch,
            reference_actions,
            target_weights,
        )
        for name, actor in actors.items():
            probabilities, actions = _policy_outputs(
                actor,
                target_batch,
                resolved,
                batch_size=config.inference_batch_size,
            )
            metrics = _heldout_metrics(
                probabilities,
                actions,
                reference_probabilities,
                reference_actions,
                target_batch,
                target_weights,
            )
            direction_results[name]["heldout"][str(target_seed)] = metrics
            rank = metrics["soft_rank_value"]
            greedy = metrics["greedy_rank_value"]
            print(
                f"Q {name:24s} -> heldout {target_seed}  "
                f"soft {rank['mean']:+.6f} "
                f"[{rank['ci95_low']:+.6f},{rank['ci95_high']:+.6f}]  "
                f"greedy {greedy['mean']:+.6f} "
                f"[{greedy['ci95_low']:+.6f},{greedy['ci95_high']:+.6f}]",
                flush=True,
            )

    parameter_pairwise: dict[str, object] = {}
    policy_pairwise: dict[str, object] = {}
    direction_by_name = {item.name: item for item in cached_directions}
    for left_name, right_name in itertools.combinations(direction_by_name, 2):
        left = direction_by_name[left_name]
        right = direction_by_name[right_name]
        parameter_pairwise[_pair_key(left_name, right_name)] = direction_cosine(
            left.initial, left.candidate, right.candidate
        )
        left_probabilities, left_actions = probe_outputs[left_name]
        right_probabilities, right_actions = probe_outputs[right_name]
        policy_pairwise[_pair_key(left_name, right_name)] = _policy_pair_metrics(
            left_probabilities,
            left_actions,
            right_probabilities,
            right_actions,
            reference_probe_probabilities,
            reference_probe_actions,
            common_probe,
            common_weights,
        )

    result = {
        "identity": identity,
        "common_probe": {
            "states": len(common_probe),
            "states_per_seed": [len(batch) for batch in calibration_batches],
            "category_weights": {
                name: float(common_weights[index])
                for index, name in enumerate(CATEGORY_NAMES)
            },
            "reference_policy": reference_probe_summary,
        },
        "heldout_targets": heldout_targets,
        "directions": direction_results,
        "parameter_pairwise": parameter_pairwise,
        "policy_pairwise": policy_pairwise,
        "generalization": _generalization_summary(
            direction_results, selected_seeds
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)

    print("CATEGORY cross-seed policy alignment", flush=True)
    for category in CATEGORY_NAMES:
        values: list[str] = []
        for optimizer in ("adamw", "sgd"):
            names = [
                f"{optimizer}-seed-{seed}" for seed in selected_seeds
            ]
            if len(names) != 2:
                values.append(f"{optimizer}=see-json")
                continue
            pair = policy_pairwise[_pair_key(names[0], names[1])]["categories"][
                category
            ]
            cosine = pair["probability_delta_cosine"]
            cosine_text = "null" if cosine is None else f"{cosine:+.3f}"
            values.append(
                f"{optimizer} cos={cosine_text} "
                f"disagree={100 * pair['action_disagreement_rate']:.2f}%"
            )
        reference_category = reference_probe_summary["categories"][category]
        print(
            f"  {category:18s}  top={reference_category['mean_top_probability']:.5f}  "
            f"{'  '.join(values)}",
            flush=True,
        )
    print(f"RESULT direction generalization complete  {result_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--optimizer-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qpc", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--target-kl", type=float, default=DEFAULT_TARGET_KL)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.batch_sweep_dir,
        args.optimizer_sweep_dir,
        args.output_dir,
        qpc=args.qpc,
        seeds=args.seeds,
        target_kl=args.target_kl,
        device=args.device,
    )


if __name__ == "__main__":
    main()
