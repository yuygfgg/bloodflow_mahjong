"""Screen per-world search-policy targets on independent determinizations."""

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
from .information_set_sampling_sweep import (
    WORLD_REPLICATE_DOMAIN,
    _assert_same_policy_states,
)
from .pipeline import load_policy, save_policy
from .policy_iteration import (
    CounterfactualBatch,
    PolicyStateBatch,
    build_state_batch,
    calibrate_direction,
    category_row_weights,
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
from .world_outcomes import (
    WorldOutcomeBatch,
    cached_world_outcome_corpus,
)


SEARCH_POLICY_SWEEP_VERSION = 2
REPLICATES = ("a", "b")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    value: float = 0.0

    def identity(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "value": float(self.value)}


@dataclass(frozen=True)
class DirectionSource:
    name: str
    seed: int | None
    replicate: str
    outcomes: WorldOutcomeBatch
    counterfactual: CounterfactualBatch
    reference_probabilities: np.ndarray
    reference_actions: np.ndarray
    category_weights: np.ndarray


def default_target_specs() -> tuple[TargetSpec, ...]:
    return (
        TargetSpec("uniform-control", "uniform"),
        TargetSpec("mean-softmax-t0p05", "mean_softmax", 0.05),
        TargetSpec("world-softmax-t0p25", "world_softmax", 0.25),
        TargetSpec("world-softmax-t0p5", "world_softmax", 0.5),
        TargetSpec("world-softmax-t1", "world_softmax", 1.0),
        TargetSpec("world-best-rank", "world_best_rank"),
        TargetSpec("world-best-lex", "world_best_lex"),
        TargetSpec("vote-lex-p0p5", "vote_lex", 0.5),
        TargetSpec("vote-lex-p0p625", "vote_lex", 0.625),
        TargetSpec("vote-lex-p0p75", "vote_lex", 0.75),
        TargetSpec("rank-lcb-z0", "rank_lcb", 0.0),
        TargetSpec("rank-lcb-z0p5", "rank_lcb", 0.5),
        TargetSpec("rank-lcb-z1", "rank_lcb", 1.0),
        TargetSpec("win-lcb-z0", "win_lcb", 0.0),
        TargetSpec("win-lcb-z0p5", "win_lcb", 0.5),
        TargetSpec("win-lcb-z1", "win_lcb", 1.0),
    )


def select_target_specs(names: Sequence[str] | None) -> tuple[TargetSpec, ...]:
    available = {spec.name: spec for spec in default_target_specs()}
    if names is None:
        return tuple(available.values())
    selected = tuple(str(name) for name in names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("search target names must be unique and non-empty")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown search targets: {sorted(unknown)}")
    return tuple(available[name] for name in selected)


def target_specs_from_identity(raw: object) -> tuple[TargetSpec, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("search target identity is invalid")
    names = [str(item.get("name")) for item in raw]
    specs = select_target_specs(names)
    if [spec.identity() for spec in specs] != raw:
        raise ValueError("search target definitions do not match this implementation")
    return specs


def _masked_softmax(values: np.ndarray, legal: np.ndarray, axis: int) -> np.ndarray:
    masked = np.where(legal, values, -np.inf)
    maximum = np.max(masked, axis=axis, keepdims=True)
    exponent = np.where(legal, np.exp(masked - maximum), 0.0)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def _prior_or_one_hot(
    reference_probabilities: np.ndarray,
    actions: np.ndarray,
    changed: np.ndarray,
) -> np.ndarray:
    target = reference_probabilities.astype(np.float32, copy=True)
    rows = np.flatnonzero(changed)
    if len(rows):
        target[rows] = 0.0
        target[rows, actions[rows]] = 1.0
    return target


def _world_best_votes(
    outcomes: WorldOutcomeBatch, *, lexicographic: bool
) -> np.ndarray:
    rank = outcomes.rank_outcomes.astype(np.float32) / 2.0
    legal = outcomes.legal[:, :, None]
    best_rank = np.where(legal, rank, -np.inf).max(axis=1, keepdims=True)
    winners = legal & (rank == best_rank)
    if lexicographic:
        best_score = np.where(winners, outcomes.score_outcomes, -np.inf).max(
            axis=1, keepdims=True
        )
        winners &= outcomes.score_outcomes == best_score
    votes = winners / winners.sum(axis=1, keepdims=True)
    return votes.mean(axis=2, dtype=np.float64).astype(np.float32)


def build_search_policy_target(
    spec: TargetSpec,
    outcomes: WorldOutcomeBatch,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if (
        reference_probabilities.shape != outcomes.legal.shape
        or reference_actions.shape != (len(outcomes),)
    ):
        raise ValueError("reference policy outputs do not match world outcomes")
    rows = np.arange(len(outcomes))
    legal = outcomes.legal
    rank = outcomes.rank_outcomes.astype(np.float32) / 2.0
    expanded_legal = legal[:, :, None]
    if spec.kind == "uniform":
        target = legal / legal.sum(axis=1, keepdims=True)
    elif spec.kind == "mean_softmax":
        target = _masked_softmax(
            rank.mean(axis=2) / spec.value, legal, axis=1
        )
    elif spec.kind == "world_softmax":
        per_world = _masked_softmax(
            rank / spec.value, expanded_legal, axis=1
        )
        target = per_world.mean(axis=2, dtype=np.float64)
    elif spec.kind in {"world_best_rank", "world_best_lex"}:
        target = _world_best_votes(
            outcomes, lexicographic=spec.kind == "world_best_lex"
        )
    elif spec.kind == "vote_lex":
        votes = _world_best_votes(outcomes, lexicographic=True)
        actions = votes.argmax(axis=1)
        confidence = votes[rows, actions]
        changed = (actions != reference_actions) & (confidence > spec.value)
        target = _prior_or_one_hot(reference_probabilities, actions, changed)
    elif spec.kind == "rank_lcb":
        baseline = rank[rows, reference_actions]
        advantage = rank - baseline[:, None, :]
        mean = advantage.mean(axis=2)
        standard_error = advantage.std(axis=2, ddof=1) / math.sqrt(outcomes.worlds)
        lcb = np.where(legal, mean - spec.value * standard_error, -np.inf)
        lcb[rows, reference_actions] = -np.inf
        actions = lcb.argmax(axis=1)
        confidence = lcb[rows, actions]
        changed = confidence > 0.0
        target = _prior_or_one_hot(reference_probabilities, actions, changed)
    elif spec.kind == "win_lcb":
        baseline_rank = rank[rows, reference_actions]
        baseline_score = outcomes.score_outcomes[rows, reference_actions]
        better_rank = rank > baseline_rank[:, None, :]
        worse_rank = rank < baseline_rank[:, None, :]
        equal_rank = ~(better_rank | worse_rank)
        better_score = outcomes.score_outcomes > baseline_score[:, None, :]
        equal_score = outcomes.score_outcomes == baseline_score[:, None, :]
        wins = np.where(
            better_rank | (equal_rank & better_score),
            1.0,
            np.where(equal_rank & equal_score, 0.5, 0.0),
        )
        mean = wins.mean(axis=2)
        standard_error = wins.std(axis=2, ddof=1) / math.sqrt(outcomes.worlds)
        lcb = np.where(legal, mean - spec.value * standard_error, -np.inf)
        lcb[rows, reference_actions] = -np.inf
        actions = lcb.argmax(axis=1)
        confidence = lcb[rows, actions]
        changed = confidence > 0.5
        target = _prior_or_one_hot(reference_probabilities, actions, changed)
    else:
        raise ValueError(f"unsupported search target kind {spec.kind!r}")

    target = np.where(legal, target, 0.0).astype(np.float32)
    target /= target.sum(axis=1, keepdims=True)
    return target, search_policy_target_metrics(
        target, outcomes, reference_probabilities
    )


def search_policy_target_metrics(
    target: np.ndarray,
    outcomes: WorldOutcomeBatch,
    reference_probabilities: np.ndarray,
) -> dict[str, object]:
    legal = outcomes.legal
    if (
        target.shape != legal.shape
        or reference_probabilities.shape != legal.shape
        or np.any(target[~legal] != 0)
        or not np.allclose(target.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("search target metrics received invalid policy arrays")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise RuntimeError("search target is invalid")
    reference_safe = np.clip(reference_probabilities, 1e-30, 1.0)
    target_safe = np.clip(target, 1e-30, 1.0)
    l1 = np.abs(target - reference_probabilities).sum(axis=1)
    changed = l1 > 1e-6
    entropy = -(target * np.log(target_safe)).sum(axis=1)
    reverse_kl = (
        target * (np.log(target_safe) - np.log(reference_safe))
    ).sum(axis=1)
    counterfactual = outcomes.counterfactual_batch()
    q_gain = (
        (target - reference_probabilities) * counterfactual.rank_q
    ).sum(axis=1)
    return {
        "states": len(outcomes),
        "changed_state_rate": float(changed.mean()),
        "mean_target_l1": float(l1.mean()),
        "mean_target_entropy": float(entropy.mean()),
        "mean_target_reverse_kl": float(reverse_kl.mean()),
        "mean_in_sample_rank_q_gain": float(q_gain.mean()),
        "categories": {
            name: {
                "states": int(np.sum(outcomes.categories == category)),
                "changed_state_rate": float(
                    changed[outcomes.categories == category].mean()
                ),
                "mean_target_l1": float(
                    l1[outcomes.categories == category].mean()
                ),
            }
            for category, name in enumerate(CATEGORY_NAMES)
        },
    }


def _pool_outcomes(batches: Sequence[WorldOutcomeBatch]) -> WorldOutcomeBatch:
    if len(batches) < 2 or len({batch.worlds for batch in batches}) != 1:
        raise ValueError("pooled outcomes need aligned independent batches")
    values = {
        name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
        for name in WorldOutcomeBatch.__dataclass_fields__
        if name != "query_ids"
    }
    values["query_ids"] = np.arange(
        sum(len(batch) for batch in batches), dtype=np.int64
    )
    return WorldOutcomeBatch(**values)


def _experiment_identity(
    input_sweep: optimizer_sweep.InputSweep,
    *,
    qpc: int,
    seeds: Sequence[int],
    worlds: int,
    world_sampling: str,
    specs: Sequence[TargetSpec],
    target_kl: float,
    learning_rate: float,
    kl_search_steps: int,
) -> dict[str, object]:
    return {
        "version": SEARCH_POLICY_SWEEP_VERSION,
        "input_directory": str(input_sweep.root),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "queries_per_category": int(qpc),
        "seeds": [int(seed) for seed in seeds],
        "worlds": int(worlds),
        "world_sampling": world_sampling,
        "replicates": list(REPLICATES),
        "targets": [spec.identity() for spec in specs],
        "optimizer": "sgd",
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "kl_search_steps": int(kl_search_steps),
    }


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("search-policy output configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("search-policy output directory is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _source_name(replicate: str, source: str) -> str:
    return f"{replicate}-{source}"


def _policy_pair(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    probe: PolicyStateBatch,
    weights: np.ndarray,
) -> dict[str, object]:
    return dg._policy_pair_metrics(
        left[0],
        left[1],
        right[0],
        right[1],
        reference_probabilities,
        reference_actions,
        probe,
        weights,
    )


def _numeric(values: Sequence[float]) -> dict[str, float | int]:
    return batch_sweep._numeric_summary([float(value) for value in values])


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    qpc: int = 16,
    seeds: Sequence[int] | None = None,
    worlds: int = 16,
    world_sampling: str = "information_set",
    target_kl: float = 1e-4,
    learning_rate: float = 0.1,
    target_names: Sequence[str] | None = None,
    kl_search_steps: int = 28,
    reuse_outcome_prefix_directory: Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if (
        qpc <= 0
        or worlds < 2
        or world_sampling not in {"live_wall", "information_set"}
        or not math.isfinite(target_kl)
        or target_kl <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or kl_search_steps <= 0
    ):
        raise ValueError("search-policy sweep arguments are invalid")
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = dg._normalize_seeds(input_sweep.seeds, seeds)
    if len(selected_seeds) != 2:
        raise ValueError("search-policy sweep currently requires exactly two seeds")
    config = input_sweep.config
    maximum_qpc = config.batch_queries_per_category[-1]
    if qpc > maximum_qpc:
        raise ValueError("search-policy QPC exceeds the source quota")
    specs = select_target_specs(target_names)
    reuse_identity = None
    if reuse_outcome_prefix_directory is not None:
        reuse_outcome_prefix_directory = reuse_outcome_prefix_directory.resolve()
        reuse_identity = optimizer_sweep._json(
            reuse_outcome_prefix_directory / "config.json"
        )
        if (
            reuse_identity.get("input_identity_fingerprint")
            != batch_sweep._fingerprint(input_sweep.identity)
            or reuse_identity.get("seeds") != list(selected_seeds)
            or int(reuse_identity.get("queries_per_category", 0)) >= qpc
            or int(reuse_identity.get("worlds", 0)) != worlds
            or reuse_identity.get("world_sampling") != world_sampling
            or reuse_identity.get("replicates") != list(REPLICATES)
        ):
            raise ValueError("outcome prefix experiment is incompatible")
    identity = _experiment_identity(
        input_sweep,
        qpc=qpc,
        seeds=selected_seeds,
        worlds=worlds,
        world_sampling=world_sampling,
        specs=specs,
        target_kl=target_kl,
        learning_rate=learning_rate,
        kl_search_steps=kl_search_steps,
    )
    identity["reuse_outcome_prefix"] = (
        None
        if reuse_outcome_prefix_directory is None
        else {
            "directory": str(reuse_outcome_prefix_directory),
            "identity_fingerprint": batch_sweep._fingerprint(reuse_identity),
            "queries_per_category": int(reuse_identity["queries_per_category"]),
        }
    )
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached search-policy identity differs")
        print(f"RESULT search-policy sweep cached  {result_path}", flush=True)
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
    heldout: dict[int, CounterfactualBatch] = {}
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
            raise ValueError("input seeds do not share policy checkpoints")
        train = load_counterfactual_batch(child / "shared" / "train.npz")
        nested = nested_category_indices(train.categories, (qpc,))
        cached_live[seed] = subset_counterfactual_batch(train, nested[qpc])
        heldout[seed] = load_counterfactual_batch(child / "shared" / "heldout.npz")
        calibration_batches.append(
            load_policy_state_batch(child / "shared" / "calibration.npz")
        )
        manifest = optimizer_sweep._json(child / "shared" / "manifest.json")
        weights = np.asarray(
            manifest["source_visit_frequencies"]["vector"], dtype=np.float64
        )
        if weights.shape != (CATEGORY_COUNT,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("source visitation weights are invalid")
        visit_weights[seed] = weights
    assert reference is not None

    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  search-policy sweep  "
        f"QPC {qpc}  worlds {worlds}  sampling {world_sampling}  "
        f"targets {len(specs)}",
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
        _assert_same_policy_states(
            build_state_batch(selected, history=reference.config.max_history),
            cached_live[seed],
        )
        queries_by_seed[seed] = selected
        print(f"RECONSTRUCT seed {seed} exact  {len(selected)} states", flush=True)

    outcome_batches: dict[int, dict[str, WorldOutcomeBatch]] = {}
    outcome_metrics: dict[str, object] = {}
    for seed in selected_seeds:
        outcome_batches[seed] = {}
        for replicate in REPLICATES:
            world_seed = (
                domain_seed(seed, batch_sweep.TRAIN_WORLD)
                if replicate == "a"
                else domain_seed(seed, WORLD_REPLICATE_DOMAIN)
            )
            progress = Progress()
            progress.start(
                f"{seed}_{world_sampling.upper()}_{replicate.upper()}",
                total=CATEGORY_COUNT * qpc,
                unit="queries",
                fields={"worlds": worlds},
            )
            outcomes, metrics = cached_world_outcome_corpus(
                output_directory / f"seed-{seed}" / f"outcomes-{replicate}",
                queries_by_seed[seed],
                reference,
                resolved,
                self_play_actor=self_play_actor,
                fingerprint=batch_sweep._fingerprint(
                    {
                        "identity": identity,
                        "seed": int(seed),
                        "replicate": replicate,
                    }
                ),
                worlds=worlds,
                world_chunk=config.world_chunk,
                world_seed=world_seed,
                world_sampling=world_sampling,
                shard_size=config.target_shard_size,
                query_batch_size=config.target_query_batch_size,
                inference_batch_size=config.rollout_inference_batch_size,
                on_progress=lambda done, values, progress=progress: progress.update(
                    done, fields=values
                ),
                prefix_directory=(
                    None
                    if reuse_outcome_prefix_directory is None
                    else reuse_outcome_prefix_directory
                    / f"seed-{seed}"
                    / f"outcomes-{replicate}"
                ),
            )
            progress.complete()
            _assert_same_policy_states(outcomes, cached_live[seed])
            if (
                world_sampling == "live_wall"
                and replicate == "a"
                and worlds == config.train_worlds
            ):
                actual = outcomes.counterfactual_batch()
                if not np.array_equal(actual.rank_q, cached_live[seed].rank_q):
                    raise RuntimeError("raw live-wall outcomes do not match cached Q")
            outcome_batches[seed][replicate] = outcomes
            outcome_metrics[f"{seed}-{replicate}"] = metrics

    common_probe = dg._concatenate_states(calibration_batches)
    common_weights = np.stack(
        [visit_weights[seed] for seed in selected_seeds]
    ).mean(axis=0)
    common_weights /= common_weights.sum()
    reference_probe = dg._policy_outputs(
        reference,
        common_probe,
        resolved,
        batch_size=config.inference_batch_size,
    )
    reference_heldout = {
        seed: dg._policy_outputs(
            reference,
            heldout[seed],
            resolved,
            batch_size=config.inference_batch_size,
        )
        for seed in selected_seeds
    }
    reference_raw: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for seed in selected_seeds:
        for replicate in REPLICATES:
            reference_raw[(seed, replicate)] = dg._policy_outputs(
                reference,
                outcome_batches[seed][replicate].counterfactual_batch(),
                resolved,
                batch_size=config.inference_batch_size,
            )

    sources: list[DirectionSource] = []
    for replicate in REPLICATES:
        for seed in selected_seeds:
            outcomes = outcome_batches[seed][replicate]
            sources.append(
                DirectionSource(
                    _source_name(replicate, f"seed-{seed}"),
                    seed,
                    replicate,
                    outcomes,
                    outcomes.counterfactual_batch(),
                    reference_raw[(seed, replicate)][0],
                    reference_raw[(seed, replicate)][1],
                    visit_weights[seed],
                )
            )
        pooled_outcomes = _pool_outcomes(
            [outcome_batches[seed][replicate] for seed in selected_seeds]
        )
        sources.append(
            DirectionSource(
                _source_name(replicate, "pooled"),
                None,
                replicate,
                pooled_outcomes,
                pooled_outcomes.counterfactual_batch(),
                np.concatenate(
                    [reference_raw[(seed, replicate)][0] for seed in selected_seeds]
                ),
                np.concatenate(
                    [reference_raw[(seed, replicate)][1] for seed in selected_seeds]
                ),
                common_weights,
            )
        )

    candidates: dict[str, object] = {}
    summaries: dict[str, object] = {}
    best_score = -math.inf
    best_name = None
    for spec in specs:
        outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        spec_candidates: dict[str, object] = {}
        for source in sources:
            target, target_metrics = build_search_policy_target(
                spec,
                source.outcomes,
                source.reference_probabilities,
                source.reference_actions,
            )
            actor, initial, candidate, optimizer_metrics = one_step_direction(
                reference,
                source.counterfactual,
                resolved,
                category_weights=source.category_weights,
                learning_rate=learning_rate,
                microbatch_size=config.microbatch_size,
                optimizer_name="sgd",
                objective="search_ce",
                policy_targets=target,
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
                search_steps=kl_search_steps,
                maximum_scale=config.maximum_scale,
            )
            outputs[source.name] = dg._policy_outputs(
                actor,
                common_probe,
                resolved,
                batch_size=config.inference_batch_size,
            )
            raw_q: dict[str, object] = {}
            for target_seed in selected_seeds:
                for target_replicate in REPLICATES:
                    target_batch = outcome_batches[target_seed][
                        target_replicate
                    ].counterfactual_batch()
                    probabilities, actions = dg._policy_outputs(
                        actor,
                        target_batch,
                        resolved,
                        batch_size=config.inference_batch_size,
                    )
                    reference_probabilities, reference_actions = reference_raw[
                        (target_seed, target_replicate)
                    ]
                    raw_q[f"{target_seed}-{target_replicate}"] = dg._heldout_metrics(
                        probabilities,
                        actions,
                        reference_probabilities,
                        reference_actions,
                        target_batch,
                        visit_weights[target_seed],
                    )
            heldout_q: dict[str, object] = {}
            for target_seed in selected_seeds:
                probabilities, actions = dg._policy_outputs(
                    actor,
                    heldout[target_seed],
                    resolved,
                    batch_size=config.inference_batch_size,
                )
                reference_probabilities, reference_actions = reference_heldout[
                    target_seed
                ]
                heldout_q[str(target_seed)] = dg._heldout_metrics(
                    probabilities,
                    actions,
                    reference_probabilities,
                    reference_actions,
                    heldout[target_seed],
                    visit_weights[target_seed],
                )
            spec_candidates[source.name] = {
                "seed": source.seed,
                "replicate": source.replicate,
                "target": target_metrics,
                "optimizer": optimizer_metrics,
                "calibration": calibration,
                "raw_q": raw_q,
                "heldout_q": heldout_q,
            }
            print(
                f"TARGET {spec.name:24s} {source.name:20s}  "
                f"changed {100 * target_metrics['changed_state_rate']:5.1f}%  "
                f"scale {calibration['scale']:.4g}",
                flush=True,
            )
            del actor
            torch.cuda.empty_cache()

        repeat_cosines: list[float] = []
        for seed in (*selected_seeds, None):
            source = "pooled" if seed is None else f"seed-{seed}"
            left = outputs[_source_name("a", source)]
            right = outputs[_source_name("b", source)]
            repeat_cosines.append(
                float(
                    _policy_pair(
                        left,
                        right,
                        reference_probe[0],
                        reference_probe[1],
                        common_probe,
                        common_weights,
                    )["probability_delta_cosine"]
                )
            )
        cross_source_cosines: list[float] = []
        for replicate in REPLICATES:
            left = outputs[
                _source_name(replicate, f"seed-{selected_seeds[0]}")
            ]
            right = outputs[
                _source_name(replicate, f"seed-{selected_seeds[1]}")
            ]
            cross_source_cosines.append(
                float(
                    _policy_pair(
                        left,
                        right,
                        reference_probe[0],
                        reference_probe[1],
                        common_probe,
                        common_weights,
                    )["probability_delta_cosine"]
                )
            )

        individual_cross_world: list[float] = []
        pooled_cross_world: list[float] = []
        pooled_heldout: list[float] = []
        for replicate, other in (("a", "b"), ("b", "a")):
            for seed in selected_seeds:
                individual = spec_candidates[
                    _source_name(replicate, f"seed-{seed}")
                ]
                individual_cross_world.append(
                    float(
                        individual["raw_q"][f"{seed}-{other}"][
                            "soft_rank_value"
                        ]["mean"]
                    )
                )
                pooled = spec_candidates[_source_name(replicate, "pooled")]
                pooled_cross_world.append(
                    float(
                        pooled["raw_q"][f"{seed}-{other}"][
                            "soft_rank_value"
                        ]["mean"]
                    )
                )
                pooled_heldout.append(
                    float(
                        pooled["heldout_q"][str(seed)]["soft_rank_value"][
                            "mean"
                        ]
                    )
                )
        summary = {
            "direction_repeat_policy_cosine": _numeric(repeat_cosines),
            "direction_cross_source_policy_cosine": _numeric(
                cross_source_cosines
            ),
            "individual_cross_world_soft_rank_value": _numeric(
                individual_cross_world
            ),
            "individual_cross_world_min": min(individual_cross_world),
            "pooled_cross_world_soft_rank_value": _numeric(pooled_cross_world),
            "pooled_cross_world_min": min(pooled_cross_world),
            "pooled_heldout_soft_rank_value": _numeric(pooled_heldout),
            "pooled_heldout_min": min(pooled_heldout),
        }
        summaries[spec.name] = summary
        candidates[spec.name] = spec_candidates
        score = min(summary["pooled_cross_world_min"], summary["pooled_heldout_min"])
        if score > best_score:
            best_score = score
            best_name = spec.name
            # Recreate only the current best pooled-A candidate for later panels.
            source = next(item for item in sources if item.name == "a-pooled")
            target, _metrics = build_search_policy_target(
                spec,
                source.outcomes,
                source.reference_probabilities,
                source.reference_actions,
            )
            actor, initial, candidate, _optimizer_metrics = one_step_direction(
                reference,
                source.counterfactual,
                resolved,
                category_weights=source.category_weights,
                learning_rate=learning_rate,
                microbatch_size=config.microbatch_size,
                optimizer_name="sgd",
                objective="search_ce",
                policy_targets=target,
            )
            calibrate_direction(
                actor,
                reference,
                initial,
                candidate,
                common_probe,
                resolved,
                category_weights=common_weights,
                target_kl=target_kl,
                batch_size=config.inference_batch_size,
                search_steps=kl_search_steps,
                maximum_scale=config.maximum_scale,
            )
            save_policy(output_directory / "best-actor.pt", actor)
            del actor
        print(
            f"SUMMARY {spec.name:24s}  repeat "
            f"{summary['direction_repeat_policy_cosine']['mean']:+.3f}  "
            f"cross-world {summary['pooled_cross_world_soft_rank_value']['mean']:+.6f} "
            f"min {summary['pooled_cross_world_min']:+.6f}  "
            f"heldout {summary['pooled_heldout_soft_rank_value']['mean']:+.6f} "
            f"min {summary['pooled_heldout_min']:+.6f}",
            flush=True,
        )

    result = {
        "identity": identity,
        "outcome_metrics": outcome_metrics,
        "candidates": candidates,
        "summary": summaries,
        "best_candidate": best_name,
        "best_minimum_score": best_score,
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)
    print(
        f"RESULT search-policy sweep complete  best {best_name}  "
        f"minimum {best_score:+.6f}  {result_path}",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qpc", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument(
        "--world-sampling",
        choices=("live_wall", "information_set"),
        default="information_set",
    )
    parser.add_argument("--target-kl", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--kl-search-steps", type=int, default=28)
    parser.add_argument("--reuse-outcome-prefix-dir", type=Path)
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
        world_sampling=args.world_sampling,
        target_kl=args.target_kl,
        learning_rate=args.learning_rate,
        target_names=args.targets,
        kl_search_steps=args.kl_search_steps,
        reuse_outcome_prefix_directory=args.reuse_outcome_prefix_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
