"""Evaluate Actor-only search candidates on cached fixed-rule panels."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from . import batch_sweep, direction_generalization as dg, optimizer_sweep
from .evaluation import (
    collect_fixed_panel,
    evaluation_seeds,
    outcomes,
    summarize_paired,
)
from .pipeline import load_policy
from .policy_iteration import domain_seed, require_cuda, require_deterministic_actor
from .progress import Progress


SEARCH_POLICY_PANEL_VERSION = 1
EXTENDED_SEARCH_POLICY_PANEL_VERSION = 2
PANEL_BOOTSTRAP_DOMAIN = 0xCE30_0001
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


def _actor_identity(
    actors: Sequence[tuple[str, Path]],
) -> tuple[dict[str, object], ...]:
    names = [name for name, _path in actors]
    if not names or len(names) != len(set(names)):
        raise ValueError("panel Actor names must be unique and non-empty")
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


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("search panel configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("search panel output is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def _load_prefix_identity(
    directory: Path,
    *,
    candidates: Sequence[Mapping[str, object]],
    input_directory: Path,
    input_fingerprint: str,
    seeds: Sequence[int],
    maximum_games: int,
) -> tuple[dict[str, object], int]:
    identity = optimizer_sweep._json(directory / "config.json")
    try:
        prefix_games = int(identity["evaluation_games_per_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("panel prefix has no valid evaluation size") from error
    if (
        str(identity.get("input_directory")) != str(input_directory)
        or identity.get("input_identity_fingerprint") != input_fingerprint
        or identity.get("seeds") != [int(seed) for seed in seeds]
        or identity.get("actors") != list(candidates)
        or prefix_games <= 0
        or prefix_games >= maximum_games
    ):
        raise ValueError("panel prefix is incompatible")
    return identity, prefix_games


def _panel_fingerprint(
    identity: Mapping[str, object],
    *,
    seed: int,
    name: str,
    actor_digest: str,
    seeds: np.ndarray,
) -> str:
    return batch_sweep._fingerprint(
        {
            "experiment": identity,
            "seed": int(seed),
            "actor": name,
            "actor_digest": actor_digest,
            "seeds": batch_sweep._fingerprint(seeds.tolist()),
        }
    )


def _collect_panel_tail(
    actor,
    device: torch.device,
    seeds: np.ndarray,
    *,
    envs: int,
    phase: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not len(seeds):
        raise ValueError("panel tail must be non-empty")
    progress = Progress()
    progress.start(phase, total=len(seeds), unit="games")
    started = time.perf_counter()
    collected = collect_fixed_panel(
        actor,
        device,
        seeds,
        envs=envs,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    elapsed = time.perf_counter() - started
    progress.complete(fields={"games/s": len(seeds) / elapsed})
    ranks, scores = outcomes(collected)
    return ranks, scores, elapsed


def run(
    batch_sweep_directory: Path,
    output_directory: Path,
    *,
    actors: Sequence[tuple[str, Path]],
    seeds: Sequence[int] | None = None,
    evaluation_games: int | None = None,
    reuse_panel_prefix_directory: Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    candidates = _actor_identity(actors)
    input_sweep = optimizer_sweep.load_input_sweep(
        batch_sweep_directory, seeds=seeds
    )
    selected_seeds = dg._normalize_seeds(input_sweep.seeds, seeds)
    if len(selected_seeds) != 2:
        raise ValueError("search panel currently requires exactly two seeds")
    config = input_sweep.config
    games = config.evaluation_games if evaluation_games is None else int(evaluation_games)
    if games < config.evaluation_games:
        raise ValueError("search panel cannot be shorter than its cached reference panel")
    extended = games != config.evaluation_games
    if reuse_panel_prefix_directory is not None and not extended:
        raise ValueError("panel prefix reuse is only valid for an extended panel")
    input_fingerprint = batch_sweep._fingerprint(input_sweep.identity)
    prefix_directory = (
        None
        if reuse_panel_prefix_directory is None
        else reuse_panel_prefix_directory.resolve()
    )
    prefix_identity = None
    prefix_games = 0
    if prefix_directory is not None:
        prefix_identity, prefix_games = _load_prefix_identity(
            prefix_directory,
            candidates=candidates,
            input_directory=input_sweep.root,
            input_fingerprint=input_fingerprint,
            seeds=selected_seeds,
            maximum_games=games,
        )
    identity = {
        "version": (
            EXTENDED_SEARCH_POLICY_PANEL_VERSION
            if extended
            else SEARCH_POLICY_PANEL_VERSION
        ),
        "input_directory": str(input_sweep.root),
        "input_identity_fingerprint": input_fingerprint,
        "seeds": [int(seed) for seed in selected_seeds],
        "evaluation_games_per_seed": games,
        "evaluation_envs": int(config.evaluation_envs),
        "bootstrap_samples": int(config.bootstrap_samples),
        "actors": list(candidates),
    }
    if extended:
        identity["reuse_panel_prefix"] = (
            None
            if prefix_identity is None
            else {
                "directory": str(prefix_directory),
                "identity_fingerprint": batch_sweep._fingerprint(prefix_identity),
                "games": prefix_games,
            }
        )
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached search panel identity differs")
        print(f"RESULT search panel cached  {result_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    reference = None
    reference_hash = None
    for seed in selected_seeds:
        child_identity = input_sweep.child_identities[seed]
        reference_path, _self_play_path = optimizer_sweep._checkpoint_paths(
            child_identity
        )
        if reference is None:
            reference = load_policy(reference_path, resolved, frozen=True)
            require_deterministic_actor(reference)
            reference_hash = str(child_identity["reference_sha256"])
        elif str(child_identity["reference_sha256"]) != reference_hash:
            raise ValueError("panel seeds do not share a reference Actor")
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
            raise ValueError(f"panel Actor {name!r} uses a different model config")

    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  fixed search panel  "
        f"seeds {','.join(map(str, selected_seeds))}  actors {','.join(actor_models)}  "
        f"games/seed {games:,}  reused {prefix_games:,}",
        flush=True,
    )
    started = time.perf_counter()
    reference_results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    actor_results: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        name: {} for name in actor_models
    }
    per_seed: dict[str, dict[str, object]] = {name: {} for name in actor_models}
    for seed in selected_seeds:
        child = input_sweep.children[seed]
        child_identity = input_sweep.child_identities[seed]
        base_seeds, base_ranks, base_scores = optimizer_sweep._reference_panel(
            child, child_identity, config, seed
        )
        panel_seeds = evaluation_seeds(domain_seed(seed, batch_sweep.FIXED_EVAL), games)
        if not np.array_equal(panel_seeds[: len(base_seeds)], base_seeds):
            raise RuntimeError("extended reference seeds do not preserve the base panel")
        reference_fingerprint = _panel_fingerprint(
            identity,
            seed=seed,
            name="reference",
            actor_digest=str(reference_hash),
            seeds=panel_seeds,
        )
        reference_path = output_directory / "panels" / f"reference-seed-{seed}.npz"
        if not extended:
            reference_ranks, reference_scores = base_ranks, base_scores
        elif reference_path.exists():
            loaded_seeds, reference_ranks, reference_scores, _elapsed = (
                batch_sweep._load_actor_panel(
                    reference_path, fingerprint=reference_fingerprint
                )
            )
            if not np.array_equal(loaded_seeds, panel_seeds):
                raise ValueError("cached extended reference panel seeds differ")
            print(f"PANEL reference seed {seed} cached", flush=True)
        else:
            reference_prefix_seeds = base_seeds
            reference_prefix_ranks = base_ranks
            reference_prefix_scores = base_scores
            reference_prefix_elapsed = 0.0
            if prefix_identity is not None and prefix_games > len(base_seeds):
                prefix_reference_seeds = panel_seeds[:prefix_games]
                prefix_reference_fingerprint = _panel_fingerprint(
                    prefix_identity,
                    seed=seed,
                    name="reference",
                    actor_digest=str(reference_hash),
                    seeds=prefix_reference_seeds,
                )
                (
                    reference_prefix_seeds,
                    reference_prefix_ranks,
                    reference_prefix_scores,
                    reference_prefix_elapsed,
                ) = batch_sweep._load_actor_panel(
                    prefix_directory / "panels" / f"reference-seed-{seed}.npz",
                    fingerprint=prefix_reference_fingerprint,
                )
                if not np.array_equal(
                    reference_prefix_seeds, prefix_reference_seeds
                ):
                    raise ValueError("reference panel prefix seeds differ")
            tail_ranks, tail_scores, tail_elapsed = _collect_panel_tail(
                reference,
                resolved,
                panel_seeds[len(reference_prefix_seeds) :],
                envs=config.evaluation_envs,
                phase=f"PANEL_reference_{seed}_TAIL",
            )
            reference_ranks = np.concatenate((reference_prefix_ranks, tail_ranks))
            reference_scores = np.concatenate((reference_prefix_scores, tail_scores))
            batch_sweep._save_actor_panel(
                reference_path,
                fingerprint=reference_fingerprint,
                seeds=panel_seeds,
                ranks=reference_ranks,
                scores=reference_scores,
                elapsed_seconds=reference_prefix_elapsed + tail_elapsed,
            )
        reference_results[seed] = (
            panel_seeds,
            reference_ranks,
            reference_scores,
        )
        for name, actor in actor_models.items():
            actor_digest = batch_sweep._model_digest(actor)
            fingerprint = _panel_fingerprint(
                identity,
                seed=seed,
                name=name,
                actor_digest=actor_digest,
                seeds=panel_seeds,
            )
            path = output_directory / "panels" / f"{name}-seed-{seed}.npz"
            if path.exists():
                loaded_seeds, ranks, scores, elapsed = batch_sweep._load_actor_panel(
                    path, fingerprint=fingerprint
                )
                if not np.array_equal(loaded_seeds, panel_seeds):
                    raise ValueError("cached candidate panel seeds differ")
                print(f"PANEL {name} seed {seed} cached", flush=True)
            else:
                prefix_seeds = panel_seeds[:0]
                prefix_ranks = np.empty(0, dtype=np.float64)
                prefix_scores = np.empty(0, dtype=np.float64)
                prefix_elapsed = 0.0
                if prefix_identity is not None:
                    prefix_seeds = panel_seeds[:prefix_games]
                    prefix_fingerprint = _panel_fingerprint(
                        prefix_identity,
                        seed=seed,
                        name=name,
                        actor_digest=actor_digest,
                        seeds=prefix_seeds,
                    )
                    (
                        loaded_prefix_seeds,
                        prefix_ranks,
                        prefix_scores,
                        prefix_elapsed,
                    ) = batch_sweep._load_actor_panel(
                        prefix_directory / "panels" / f"{name}-seed-{seed}.npz",
                        fingerprint=prefix_fingerprint,
                    )
                    if not np.array_equal(loaded_prefix_seeds, prefix_seeds):
                        raise ValueError("candidate panel prefix seeds differ")
                tail_ranks, tail_scores, tail_elapsed = _collect_panel_tail(
                    actor,
                    resolved,
                    panel_seeds[len(prefix_seeds) :],
                    envs=config.evaluation_envs,
                    phase=f"PANEL_{name}_{seed}_TAIL",
                )
                elapsed = prefix_elapsed + tail_elapsed
                ranks = np.concatenate((prefix_ranks, tail_ranks))
                scores = np.concatenate((prefix_scores, tail_scores))
                batch_sweep._save_actor_panel(
                    path,
                    fingerprint=fingerprint,
                    seeds=panel_seeds,
                    ranks=ranks,
                    scores=scores,
                    elapsed_seconds=elapsed,
                )
            actor_results[name][seed] = (ranks, scores)
            comparison = summarize_paired(
                ranks,
                scores,
                reference_ranks,
                reference_scores,
                seed=domain_seed(seed, PANEL_BOOTSTRAP_DOMAIN),
                bootstrap_samples=config.bootstrap_samples,
            )
            per_seed[name][str(seed)] = comparison
            rank = comparison["paired_rank_delta"]
            print(
                f"PANEL {name:20s} seed {seed} dRank {rank['mean']:+.5f} "
                f"[{rank['ci95_low']:+.5f},{rank['ci95_high']:+.5f}]",
                flush=True,
            )

    pooled: dict[str, object] = {}
    pooled_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reference_ranks = np.concatenate(
        [reference_results[seed][1] for seed in selected_seeds]
    )
    reference_scores = np.concatenate(
        [reference_results[seed][2] for seed in selected_seeds]
    )
    for index, name in enumerate(actor_models):
        ranks = np.concatenate(
            [actor_results[name][seed][0] for seed in selected_seeds]
        )
        scores = np.concatenate(
            [actor_results[name][seed][1] for seed in selected_seeds]
        )
        pooled_arrays[name] = (ranks, scores)
        pooled[name] = summarize_paired(
            ranks,
            scores,
            reference_ranks,
            reference_scores,
            seed=domain_seed(selected_seeds[0], PANEL_BOOTSTRAP_DOMAIN + 1, index),
            bootstrap_samples=config.bootstrap_samples,
        )
        rank = pooled[name]["paired_rank_delta"]
        print(
            f"POOLED {name:20s} dRank {rank['mean']:+.5f} "
            f"[{rank['ci95_low']:+.5f},{rank['ci95_high']:+.5f}]",
            flush=True,
        )

    pairwise: dict[str, object] = {}
    for index, (left, right) in enumerate(itertools.combinations(actor_models, 2)):
        pairwise[f"{left}__{right}"] = summarize_paired(
            pooled_arrays[left][0],
            pooled_arrays[left][1],
            pooled_arrays[right][0],
            pooled_arrays[right][1],
            seed=domain_seed(
                selected_seeds[0], PANEL_BOOTSTRAP_DOMAIN + 2, index
            ),
            bootstrap_samples=config.bootstrap_samples,
        )

    result = {
        "identity": identity,
        "per_seed": per_seed,
        "pooled_vs_reference": pooled,
        "pooled_pairwise": pairwise,
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)
    print(f"RESULT search panel complete  {result_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--actor", action="append", required=True, type=_parse_actor, metavar="NAME=PATH"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--evaluation-games", type=int)
    parser.add_argument("--reuse-panel-prefix-dir", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.batch_sweep_dir,
        args.output_dir,
        actors=args.actor,
        seeds=args.seeds,
        evaluation_games=args.evaluation_games,
        reuse_panel_prefix_directory=args.reuse_panel_prefix_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
