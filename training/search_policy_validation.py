"""Blind information-set Q validation for Actor-only policy candidates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from . import batch_sweep, direction_generalization as dg, optimizer_sweep
from .information_set_sampling_sweep import _assert_same_policy_states
from .pipeline import load_policy
from .policy_iteration import (
    CounterfactualBatch,
    build_state_batch,
    domain_seed,
    load_counterfactual_batch,
    nested_category_indices,
    require_cuda,
    require_deterministic_actor,
    subset_counterfactual_batch,
)
from .policy_pool import CATEGORY_COUNT
from .progress import Progress
from .world_outcomes import cached_world_outcome_corpus


SEARCH_POLICY_VALIDATION_VERSION = 2
SEARCH_POLICY_VALIDATION_CORPUS_VERSION = 1
VALIDATION_WORLD = 0xCE20_0001
ACTOR_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _parse_actor(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if (
        not separator
        or not ACTOR_NAME_PATTERN.fullmatch(name)
        or "__" in name
        or not path
    ):
        raise argparse.ArgumentTypeError("actor must be specified as NAME=PATH")
    return name, Path(path)


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("search validation configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("search validation output is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _candidate_identity(
    actors: Sequence[tuple[str, Path]],
) -> tuple[dict[str, object], ...]:
    names = [name for name, _path in actors]
    if not names or len(names) != len(set(names)):
        raise ValueError("validation Actor names must be unique and non-empty")
    result = []
    for name, path in actors:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result.append(
            {
                "name": name,
                "path": str(resolved),
                "sha256": batch_sweep._sha256(resolved),
            }
        )
    return tuple(result)


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    actors: Sequence[tuple[str, Path]],
    qpc: int = 32,
    worlds: int = 64,
    seeds: Sequence[int] | None = None,
    corpus_directory: Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    if qpc <= 0 or worlds < 2:
        raise ValueError("search validation sizes must be positive")
    candidates = _candidate_identity(actors)
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = dg._normalize_seeds(input_sweep.seeds, seeds)
    if len(selected_seeds) != 2:
        raise ValueError("search validation currently requires exactly two seeds")
    config = input_sweep.config
    if qpc > config.heldout_queries_per_category:
        raise ValueError("validation QPC exceeds the heldout source quota")
    output_directory = output_directory.resolve()
    resolved_corpus_directory = (
        output_directory
        if corpus_directory is None
        else corpus_directory.resolve()
    )
    identity = {
        "version": SEARCH_POLICY_VALIDATION_VERSION,
        "input_directory": str(input_sweep.root),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "seeds": [int(seed) for seed in selected_seeds],
        "queries_per_category": int(qpc),
        "worlds": int(worlds),
        "world_sampling": "information_set",
        "world_seed_domain": VALIDATION_WORLD,
        "corpus_directory": str(resolved_corpus_directory),
        "actors": list(candidates),
    }
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached search validation identity differs")
        print(f"RESULT search validation cached  {result_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    reference = None
    reference_hash = None
    self_play_actor = None
    self_play_hash = None
    queries_by_seed: dict[int, Sequence[object]] = {}
    cached_by_seed: dict[int, CounterfactualBatch] = {}
    visit_weights: dict[int, np.ndarray] = {}
    for seed in selected_seeds:
        child = input_sweep.children[seed]
        child_identity = input_sweep.child_identities[seed]
        reference_path, self_play_path = optimizer_sweep._checkpoint_paths(
            child_identity
        )
        if reference is None:
            reference = load_policy(reference_path, resolved, frozen=True)
            require_deterministic_actor(reference)
            reference_hash = str(child_identity["reference_sha256"])
            self_play_hash = child_identity.get("self_play_sha256")
            if self_play_path is not None:
                self_play_actor = load_policy(self_play_path, resolved, frozen=True)
                require_deterministic_actor(self_play_actor)
        elif (
            str(child_identity["reference_sha256"]) != reference_hash
            or child_identity.get("self_play_sha256") != self_play_hash
        ):
            raise ValueError("validation seeds do not share policy checkpoints")
        heldout = load_counterfactual_batch(child / "shared" / "heldout.npz")
        nested = nested_category_indices(heldout.categories, (qpc,))
        cached_by_seed[seed] = subset_counterfactual_batch(heldout, nested[qpc])
        manifest = optimizer_sweep._json(child / "shared" / "manifest.json")
        weights = np.asarray(
            manifest["source_visit_frequencies"]["vector"], dtype=np.float64
        )
        if weights.shape != (CATEGORY_COUNT,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("source visitation weights are invalid")
        visit_weights[seed] = weights
    assert reference is not None

    actor_models = {
        str(candidate["name"]): load_policy(
            Path(str(candidate["path"])), resolved, frozen=True
        )
        for candidate in candidates
    }
    for name, actor in actor_models.items():
        require_deterministic_actor(actor)
        if actor.config != reference.config:
            raise ValueError(f"validation Actor {name!r} uses a different model config")

    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  blind search validation  "
        f"QPC {qpc}  worlds {worlds}  actors {','.join(actor_models)}",
        flush=True,
    )
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
            games=config.heldout_source_games,
            envs=config.envs,
            qpc=config.heldout_queries_per_category,
            source_seed=domain_seed(seed, batch_sweep.HELDOUT_SOURCE),
            query_seed=domain_seed(seed, batch_sweep.HELDOUT_QUERY),
            phase=f"RECONSTRUCT_VALIDATION_{seed}",
        )
        selected = queries[: CATEGORY_COUNT * qpc]
        _assert_same_policy_states(
            build_state_batch(selected, history=reference.config.max_history),
            cached_by_seed[seed],
        )
        queries_by_seed[seed] = selected
        print(f"RECONSTRUCT validation seed {seed} exact  {len(selected)} states", flush=True)

    outcome_metrics: dict[str, object] = {}
    outcomes_by_seed = {}
    corpus_identity = {
        key: identity[key]
        for key in (
            "version",
            "input_identity_fingerprint",
            "seeds",
            "queries_per_category",
            "worlds",
            "world_sampling",
            "world_seed_domain",
        )
    }
    corpus_identity["version"] = SEARCH_POLICY_VALIDATION_CORPUS_VERSION
    for seed in selected_seeds:
        progress = Progress()
        progress.start(
            f"VALIDATION_INFO_{seed}",
            total=CATEGORY_COUNT * qpc,
            unit="queries",
            fields={"worlds": worlds},
        )
        outcomes, metrics = cached_world_outcome_corpus(
            resolved_corpus_directory / f"seed-{seed}" / "outcomes",
            queries_by_seed[seed],
            reference,
            resolved,
            self_play_actor=self_play_actor,
            fingerprint=batch_sweep._fingerprint(
                {"corpus": corpus_identity, "seed": int(seed)}
            ),
            worlds=worlds,
            world_chunk=config.world_chunk,
            world_seed=domain_seed(seed, VALIDATION_WORLD),
            world_sampling="information_set",
            shard_size=config.target_shard_size,
            query_batch_size=config.target_query_batch_size,
            inference_batch_size=config.rollout_inference_batch_size,
            on_progress=lambda done, values, progress=progress: progress.update(
                done, fields=values
            ),
        )
        progress.complete()
        _assert_same_policy_states(outcomes, cached_by_seed[seed])
        outcomes_by_seed[seed] = outcomes
        outcome_metrics[str(seed)] = metrics

    evaluations: dict[str, object] = {}
    summaries: dict[str, object] = {}
    for name, actor in actor_models.items():
        per_seed: dict[str, object] = {}
        soft_values: list[float] = []
        greedy_values: list[float] = []
        for seed in selected_seeds:
            batch = outcomes_by_seed[seed].counterfactual_batch()
            reference_probabilities, reference_actions = dg._policy_outputs(
                reference,
                batch,
                resolved,
                batch_size=config.inference_batch_size,
            )
            probabilities, actions = dg._policy_outputs(
                actor,
                batch,
                resolved,
                batch_size=config.inference_batch_size,
            )
            metrics = dg._heldout_metrics(
                probabilities,
                actions,
                reference_probabilities,
                reference_actions,
                batch,
                visit_weights[seed],
            )
            per_seed[str(seed)] = metrics
            soft_values.append(float(metrics["soft_rank_value"]["mean"]))
            greedy_values.append(float(metrics["greedy_rank_value"]["mean"]))
        evaluations[name] = per_seed
        summaries[name] = {
            "soft_rank_value": batch_sweep._numeric_summary(soft_values),
            "soft_rank_min": min(soft_values),
            "greedy_rank_value": batch_sweep._numeric_summary(greedy_values),
            "greedy_rank_min": min(greedy_values),
        }
        print(
            f"VALIDATE {name:24s} soft {np.mean(soft_values):+.6f} "
            f"min {min(soft_values):+.6f}  greedy {np.mean(greedy_values):+.6f} "
            f"min {min(greedy_values):+.6f}",
            flush=True,
        )

    result = {
        "identity": identity,
        "outcome_metrics": outcome_metrics,
        "evaluations": evaluations,
        "summary": summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)
    print(f"RESULT search validation complete  {result_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--actor", action="append", required=True, type=_parse_actor, metavar="NAME=PATH"
    )
    parser.add_argument("--qpc", type=int, default=32)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.batch_sweep_dir,
        args.output_dir,
        actors=args.actor,
        qpc=args.qpc,
        worlds=args.worlds,
        seeds=args.seeds,
        corpus_directory=args.corpus_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
