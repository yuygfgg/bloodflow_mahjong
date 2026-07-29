"""Paired diagnostic for live-wall versus information-set Q sampling.

The script reconstructs the exact source queries of a completed batch sweep,
then estimates the same states under two independent world-seed replicates for
both sampling modes.  It compares Q-label repeatability and the repeatability
of KL-matched softmax-CE update directions.  No full-game panel is run.
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
    build_state_batch,
    cached_counterfactual_corpus,
    calibrate_direction,
    category_row_weights,
    direction_cosine,
    domain_seed,
    load_counterfactual_batch,
    load_policy_state_batch,
    nested_category_indices,
    one_step_direction,
    require_cuda,
    require_deterministic_actor,
    subset_counterfactual_batch,
)
from .policy_pool import CATEGORY_COUNT, CATEGORY_NAMES
from .progress import Progress
from .update_subspace_sweep import _concatenate_targets


INFORMATION_SET_SWEEP_VERSION = 1
WORLD_REPLICATE_DOMAIN = 0xC71E_1002
SAMPLING_MODES = ("live_wall", "information_set")
REPLICATES = ("a", "b")


@dataclass(frozen=True)
class DirectionSource:
    name: str
    seed: int | None
    mode: str
    replicate: str
    batch: CounterfactualBatch
    category_weights: np.ndarray


def _experiment_identity(
    input_sweep: optimizer_sweep.InputSweep,
    *,
    qpc: int,
    seeds: Sequence[int],
    worlds: int,
    objective: str,
    temperature: float,
    target_kl: float,
    learning_rate: float,
) -> dict[str, object]:
    return {
        "version": INFORMATION_SET_SWEEP_VERSION,
        "input_directory": str(input_sweep.root),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "queries_per_category": int(qpc),
        "seeds": [int(seed) for seed in seeds],
        "worlds": int(worlds),
        "sampling_modes": list(SAMPLING_MODES),
        "replicates": list(REPLICATES),
        "objective": objective,
        "temperature": float(temperature),
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
            raise ValueError("information-set output configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("information-set output directory is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _assert_same_policy_states(
    reconstructed: PolicyStateBatch, cached: PolicyStateBatch
) -> None:
    for name in PolicyStateBatch.__dataclass_fields__:
        if not np.array_equal(getattr(reconstructed, name), getattr(cached, name)):
            raise RuntimeError(f"reconstructed source queries differ in {name}")


def _weighted_q_pair(
    left: CounterfactualBatch,
    right: CounterfactualBatch,
    weights: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float | None]:
    if not len(rows):
        raise ValueError("Q pair metric needs non-empty rows")
    legal = left.legal[rows]
    left_q = left.centered_rank_q[rows].astype(np.float64)
    right_q = right.centered_rank_q[rows].astype(np.float64)
    row_weights = np.asarray(weights, dtype=np.float64)
    if row_weights.shape != (len(rows),) or not np.isclose(row_weights.sum(), 1.0):
        raise ValueError("Q pair row weights are invalid")
    state_dot = (left_q * right_q).sum(axis=1)
    left_squared = np.square(left_q).sum(axis=1)
    right_squared = np.square(right_q).sum(axis=1)
    dot = float(np.dot(row_weights, state_dot))
    left_norm = math.sqrt(float(np.dot(row_weights, left_squared)))
    right_norm = math.sqrt(float(np.dot(row_weights, right_squared)))
    denominator = left_norm * right_norm
    difference = left_q - right_q
    best_left = np.where(legal, left_q, -np.inf).argmax(axis=1)
    best_right = np.where(legal, right_q, -np.inf).argmax(axis=1)
    preference_agreement = np.empty(len(rows), dtype=np.float64)
    for local in range(len(rows)):
        actions = np.flatnonzero(legal[local])
        first, second = np.triu_indices(len(actions), k=1)
        if not len(first):
            preference_agreement[local] = 1.0
            continue
        left_delta = left_q[local, actions[first]] - left_q[local, actions[second]]
        right_delta = (
            right_q[local, actions[first]] - right_q[local, actions[second]]
        )
        comparable = (left_delta != 0) & (right_delta != 0)
        preference_agreement[local] = (
            float(
                np.mean(
                    np.sign(left_delta[comparable])
                    == np.sign(right_delta[comparable])
                )
            )
            if np.any(comparable)
            else 1.0
        )
    return {
        "centered_q_cosine": dot / denominator if denominator > 0 else None,
        "centered_q_delta_l2": math.sqrt(
            float(np.dot(row_weights, np.square(difference).sum(axis=1)))
        ),
        "best_action_agreement": float(
            np.dot(row_weights, (best_left == best_right).astype(np.float64))
        ),
        "pairwise_preference_agreement": float(
            np.dot(row_weights, preference_agreement)
        ),
    }


def q_pair_metrics(
    left: CounterfactualBatch,
    right: CounterfactualBatch,
    category_weights: np.ndarray,
) -> dict[str, object]:
    _assert_same_policy_states(left, right)
    if not np.array_equal(left.legal, right.legal):
        raise RuntimeError("paired Q batches have different legal masks")
    all_rows = np.arange(len(left), dtype=np.int64)
    global_weights = category_row_weights(left.categories, category_weights)
    categories: dict[str, object] = {}
    for category, name in enumerate(CATEGORY_NAMES):
        rows = np.flatnonzero(left.categories == category)
        categories[name] = {
            "states": len(rows),
            **_weighted_q_pair(
                left,
                right,
                np.full(len(rows), 1.0 / len(rows)),
                rows,
            ),
        }
    return {
        "states": len(left),
        **_weighted_q_pair(left, right, global_weights, all_rows),
        "categories": categories,
    }


def _direction_name(mode: str, replicate: str, source: str) -> str:
    return f"{mode}-{replicate}-{source}"


def _add_pair(
    output: dict[str, object],
    left_name: str,
    right_name: str,
    raw_states: Mapping[
        str, tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
    ],
    probe_outputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    probe: PolicyStateBatch,
    category_weights: np.ndarray,
) -> None:
    left_initial, left_candidate = raw_states[left_name]
    _right_initial, right_candidate = raw_states[right_name]
    left_probabilities, left_actions = probe_outputs[left_name]
    right_probabilities, right_actions = probe_outputs[right_name]
    output[dg._pair_key(left_name, right_name)] = {
        "parameter": direction_cosine(
            left_initial, left_candidate, right_candidate
        ),
        "policy": dg._policy_pair_metrics(
            left_probabilities,
            left_actions,
            right_probabilities,
            right_actions,
            reference_probabilities,
            reference_actions,
            probe,
            category_weights,
        ),
    }


def _numeric(values: Sequence[float]) -> dict[str, float | int]:
    return batch_sweep._numeric_summary([float(value) for value in values])


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    qpc: int = 16,
    seeds: Sequence[int] | None = None,
    worlds: int | None = None,
    objective: str = "softmax_ce",
    temperature: float = 0.05,
    target_kl: float = 1e-4,
    learning_rate: float = 0.1,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if (
        qpc <= 0
        or not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(target_kl)
        or target_kl <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or objective not in {"hard_ce", "softmax_ce", "mirror_ce"}
    ):
        raise ValueError("information-set sweep arguments are invalid")
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = dg._normalize_seeds(input_sweep.seeds, seeds)
    if len(selected_seeds) != 2:
        raise ValueError("information-set sweep currently requires exactly two seeds")
    config = input_sweep.config
    maximum_qpc = config.batch_queries_per_category[-1]
    if qpc > maximum_qpc:
        raise ValueError("information-set QPC exceeds the cached source quota")
    selected_worlds = config.train_worlds if worlds is None else int(worlds)
    if selected_worlds != config.train_worlds:
        raise ValueError("world count must match the cached live-wall replicate")
    identity = _experiment_identity(
        input_sweep,
        qpc=qpc,
        seeds=selected_seeds,
        worlds=selected_worlds,
        objective=objective,
        temperature=temperature,
        target_kl=target_kl,
        learning_rate=learning_rate,
    )
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached information-set identity differs")
        print(f"RESULT information-set sweep cached  {result_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    reference = None
    self_play_actor = None
    reference_hash = None
    self_play_hash = None
    cached_live: dict[int, CounterfactualBatch] = {}
    calibration_batches: list[PolicyStateBatch] = []
    visit_weights: dict[int, np.ndarray] = {}
    for seed in selected_seeds:
        child = input_sweep.children[seed]
        input_identity = input_sweep.child_identities[seed]
        reference_path, self_play_path = optimizer_sweep._checkpoint_paths(
            input_identity
        )
        if reference is None:
            reference = load_policy(reference_path, resolved, frozen=True)
            require_deterministic_actor(reference)
            reference_hash = str(input_identity["reference_sha256"])
            self_play_hash = input_identity.get("self_play_sha256")
            if self_play_path is not None:
                self_play_actor = load_policy(self_play_path, resolved, frozen=True)
                require_deterministic_actor(self_play_actor)
        elif (
            str(input_identity["reference_sha256"]) != reference_hash
            or input_identity.get("self_play_sha256") != self_play_hash
        ):
            raise ValueError("input seeds do not share reference/self-play policies")
        train = load_counterfactual_batch(child / "shared" / "train.npz")
        nested = nested_category_indices(train.categories, (qpc,))
        cached_live[seed] = subset_counterfactual_batch(train, nested[qpc])
        calibration_batches.append(
            load_policy_state_batch(child / "shared" / "calibration.npz")
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
    if self_play_actor is not None and self_play_actor.config != reference.config:
        raise ValueError("self-play Actor config differs from the reference")

    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  information-set sweep  "
        f"QPC {qpc}  worlds {selected_worlds}  "
        f"seeds {','.join(map(str, selected_seeds))}",
        flush=True,
    )
    queries_by_seed: dict[int, Sequence[object]] = {}
    for seed in selected_seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed & 0xFFFF_FFFF)
        queries, _trajectories = batch_sweep._collect(
            reference,
            resolved,
            Progress(),
            self_play_actor=self_play_actor,
            self_play_fraction=config.self_play_fraction,
            anchor_rule_fast=config.anchor_rule_fast,
            games=config.source_games,
            envs=config.envs,
            qpc=maximum_qpc,
            source_seed=domain_seed(seed, batch_sweep.TRAIN_SOURCE),
            query_seed=domain_seed(seed, batch_sweep.TRAIN_QUERY),
            phase=f"RECONSTRUCT_{seed}",
        )
        selected = queries[: CATEGORY_COUNT * qpc]
        reconstructed = build_state_batch(
            selected, history=reference.config.max_history
        )
        _assert_same_policy_states(reconstructed, cached_live[seed])
        queries_by_seed[seed] = selected
        print(
            f"RECONSTRUCT seed {seed} exact match  {len(selected)} states",
            flush=True,
        )

    target_batches: dict[int, dict[str, CounterfactualBatch]] = {}
    target_metrics: dict[str, object] = {}
    for seed in selected_seeds:
        target_batches[seed] = {"live_wall-a": cached_live[seed]}
        for mode in SAMPLING_MODES:
            for replicate in REPLICATES:
                key = f"{mode}-{replicate}"
                if key == "live_wall-a":
                    continue
                world_seed = (
                    domain_seed(seed, batch_sweep.TRAIN_WORLD)
                    if replicate == "a"
                    else domain_seed(seed, WORLD_REPLICATE_DOMAIN)
                )
                progress = Progress()
                progress.start(
                    f"{seed}_{mode.upper()}_{replicate.upper()}",
                    total=CATEGORY_COUNT * qpc,
                    unit="queries",
                    fields={"worlds": selected_worlds},
                )
                batch, metrics = cached_counterfactual_corpus(
                    output_directory / f"seed-{seed}" / key,
                    queries_by_seed[seed],
                    reference,
                    resolved,
                    self_play_actor=self_play_actor,
                    fingerprint=batch_sweep._fingerprint(
                        {
                            "identity": identity,
                            "seed": int(seed),
                            "mode": mode,
                            "replicate": replicate,
                        }
                    ),
                    worlds=selected_worlds,
                    world_chunk=config.world_chunk,
                    world_seed=world_seed,
                    shard_size=config.target_shard_size,
                    query_batch_size=config.target_query_batch_size,
                    inference_batch_size=config.rollout_inference_batch_size,
                    world_sampling=mode,
                    on_progress=lambda done, values, progress=progress: progress.update(
                        done, fields=values
                    ),
                )
                progress.complete()
                _assert_same_policy_states(batch, cached_live[seed])
                target_batches[seed][key] = batch
                target_metrics[f"{seed}-{key}"] = metrics

    q_pairs: dict[str, object] = {}
    for seed in selected_seeds:
        for mode in SAMPLING_MODES:
            left = target_batches[seed][f"{mode}-a"]
            right = target_batches[seed][f"{mode}-b"]
            q_pairs[f"{seed}-{mode}-repeat"] = q_pair_metrics(
                left, right, visit_weights[seed]
            )
        q_pairs[f"{seed}-sampling-shift-a"] = q_pair_metrics(
            target_batches[seed]["live_wall-a"],
            target_batches[seed]["information_set-a"],
            visit_weights[seed],
        )

    common_probe = dg._concatenate_states(calibration_batches)
    common_weights = np.stack(
        [visit_weights[seed] for seed in selected_seeds]
    ).mean(axis=0)
    common_weights /= common_weights.sum()
    sources: list[DirectionSource] = []
    for mode in SAMPLING_MODES:
        for replicate in REPLICATES:
            for seed in selected_seeds:
                sources.append(
                    DirectionSource(
                        _direction_name(mode, replicate, f"seed-{seed}"),
                        seed,
                        mode,
                        replicate,
                        target_batches[seed][f"{mode}-{replicate}"],
                        visit_weights[seed],
                    )
                )
            pooled = _concatenate_targets(
                [
                    target_batches[seed][f"{mode}-{replicate}"]
                    for seed in selected_seeds
                ]
            )
            sources.append(
                DirectionSource(
                    _direction_name(mode, replicate, "pooled"),
                    None,
                    mode,
                    replicate,
                    pooled,
                    common_weights,
                )
            )

    reference_probe_probabilities, reference_probe_actions = dg._policy_outputs(
        reference,
        common_probe,
        resolved,
        batch_size=config.inference_batch_size,
    )
    evaluation_batches: dict[str, CounterfactualBatch] = {}
    evaluation_weights: dict[str, np.ndarray] = {}
    reference_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seed in selected_seeds:
        for key, batch in target_batches[seed].items():
            name = f"{seed}-{key}"
            evaluation_batches[name] = batch
            evaluation_weights[name] = visit_weights[seed]
            reference_outputs[name] = dg._policy_outputs(
                reference,
                batch,
                resolved,
                batch_size=config.inference_batch_size,
            )

    candidates: dict[str, object] = {}
    raw_states: dict[
        str, tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
    ] = {}
    probe_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source in sources:
        actor, initial, candidate, optimizer_metrics = one_step_direction(
            reference,
            source.batch,
            resolved,
            category_weights=source.category_weights,
            learning_rate=learning_rate,
            microbatch_size=config.microbatch_size,
            optimizer_name="sgd",
            objective=objective,
            target_temperature=temperature,
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
        probe_outputs[source.name] = (probabilities, actions)
        raw_states[source.name] = (initial, candidate)
        q_values: dict[str, object] = {}
        for target_name, target_batch in evaluation_batches.items():
            reference_probabilities, reference_actions = reference_outputs[target_name]
            target_probabilities, target_actions = dg._policy_outputs(
                actor,
                target_batch,
                resolved,
                batch_size=config.inference_batch_size,
            )
            q_values[target_name] = dg._heldout_metrics(
                target_probabilities,
                target_actions,
                reference_probabilities,
                reference_actions,
                target_batch,
                evaluation_weights[target_name],
            )
        candidates[source.name] = {
            "seed": source.seed,
            "sampling_mode": source.mode,
            "replicate": source.replicate,
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
            "q_value": q_values,
        }
        if source.seed is None:
            save_policy(output_directory / f"actor-{source.name}.pt", actor)
        own_other = [
            float(
                q_values[
                    f"{seed}-{source.mode}-"
                    f"{'b' if source.replicate == 'a' else 'a'}"
                ]["soft_rank_value"]["mean"]
            )
            for seed in selected_seeds
            if source.seed is None or seed == source.seed
        ]
        print(
            f"DIRECTION {source.name:38s}  scale {calibration['scale']:.5g}  "
            f"other-replicate {np.mean(own_other):+.6f}",
            flush=True,
        )
        del actor
        torch.cuda.empty_cache()

    direction_pairs: dict[str, object] = {}
    for mode in SAMPLING_MODES:
        for seed in selected_seeds:
            _add_pair(
                direction_pairs,
                _direction_name(mode, "a", f"seed-{seed}"),
                _direction_name(mode, "b", f"seed-{seed}"),
                raw_states,
                probe_outputs,
                reference_probe_probabilities,
                reference_probe_actions,
                common_probe,
                common_weights,
            )
        for replicate in REPLICATES:
            _add_pair(
                direction_pairs,
                _direction_name(mode, replicate, f"seed-{selected_seeds[0]}"),
                _direction_name(mode, replicate, f"seed-{selected_seeds[1]}"),
                raw_states,
                probe_outputs,
                reference_probe_probabilities,
                reference_probe_actions,
                common_probe,
                common_weights,
            )

    summary: dict[str, object] = {}
    for mode in SAMPLING_MODES:
        q_cosines: list[float] = []
        q_best: list[float] = []
        repeat_policy_cosines: list[float] = []
        for seed in selected_seeds:
            q_pair = q_pairs[f"{seed}-{mode}-repeat"]
            q_cosines.append(float(q_pair["centered_q_cosine"]))
            q_best.append(float(q_pair["best_action_agreement"]))
            left = _direction_name(mode, "a", f"seed-{seed}")
            right = _direction_name(mode, "b", f"seed-{seed}")
            repeat_policy_cosines.append(
                float(
                    direction_pairs[dg._pair_key(left, right)]["policy"][
                        "probability_delta_cosine"
                    ]
                )
            )
        cross_source_cosines: list[float] = []
        for replicate in REPLICATES:
            left = _direction_name(
                mode, replicate, f"seed-{selected_seeds[0]}"
            )
            right = _direction_name(
                mode, replicate, f"seed-{selected_seeds[1]}"
            )
            cross_source_cosines.append(
                float(
                    direction_pairs[dg._pair_key(left, right)]["policy"][
                        "probability_delta_cosine"
                    ]
                )
            )
        pooled_cross_values: list[float] = []
        for replicate, other in (("a", "b"), ("b", "a")):
            candidate = candidates[_direction_name(mode, replicate, "pooled")]
            for seed in selected_seeds:
                pooled_cross_values.append(
                    float(
                        candidate["q_value"][f"{seed}-{mode}-{other}"][
                            "soft_rank_value"
                        ]["mean"]
                    )
                )
        summary[mode] = {
            "q_repeat_centered_cosine": _numeric(q_cosines),
            "q_repeat_best_action_agreement": _numeric(q_best),
            "direction_repeat_policy_cosine": _numeric(repeat_policy_cosines),
            "direction_cross_source_policy_cosine": _numeric(
                cross_source_cosines
            ),
            "pooled_cross_replicate_soft_rank_value": _numeric(
                pooled_cross_values
            ),
            "pooled_cross_replicate_min_soft_rank_value": min(
                pooled_cross_values
            ),
        }

    result = {
        "identity": identity,
        "target_metrics": target_metrics,
        "q_pairs": q_pairs,
        "candidates": candidates,
        "direction_pairs": direction_pairs,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "response_phase_note": (
            "The engine fixes every concealed hand in Hu/Meld response windows "
            "to preserve pending responder legality; information-set sampling "
            "therefore differs from live-wall sampling only outside those windows."
        ),
    }
    batch_sweep._atomic_json(result_path, result)
    print("SUMMARY sampling modes", flush=True)
    for mode, metrics in summary.items():
        print(
            f"  {mode:16s}  Q-cos {metrics['q_repeat_centered_cosine']['mean']:+.3f}  "
            "best-agree "
            f"{100 * metrics['q_repeat_best_action_agreement']['mean']:.1f}%  "
            "direction-repeat "
            f"{metrics['direction_repeat_policy_cosine']['mean']:+.3f}  "
            "cross-source "
            f"{metrics['direction_cross_source_policy_cosine']['mean']:+.3f}  "
            f"cross-Q {metrics['pooled_cross_replicate_soft_rank_value']['mean']:+.6f}",
            flush=True,
        )
    print(f"RESULT information-set sweep complete  {result_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qpc", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--worlds", type=int)
    parser.add_argument(
        "--objective",
        choices=("hard_ce", "softmax_ce", "mirror_ce"),
        default="softmax_ce",
    )
    parser.add_argument("--temperature", type=float, default=0.05)
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
        worlds=args.worlds,
        objective=args.objective,
        temperature=args.temperature,
        target_kl=args.target_kl,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
