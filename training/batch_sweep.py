"""Nested independent-state batch-size sweep for policy iteration."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import bloodflow_mahjong as bm

from .evaluation import (
    collect_fixed_panel,
    evaluation_seeds,
    load_reference_panel,
    outcomes,
    save_reference_panel,
    summarize_paired,
)
from .pipeline import (
    CollectionConfig,
    POLICY_EXECUTION_VERSION,
    TrajectoryCollector,
    load_policy,
    save_policy,
)
from .policy_iteration import (
    build_state_batch,
    cached_counterfactual_corpus,
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
    save_counterfactual_batch,
    save_policy_state_batch,
    select_independent_queries,
    source_visit_frequencies,
    subset_counterfactual_batch,
)
from .progress import Progress


SWEEP_VERSION = 4
MULTI_SWEEP_VERSION = 1
SHARED_CACHE_VERSION = 1
DIRECTION_CACHE_VERSION = 1
ACTOR_PANEL_VERSION = 1
TRAIN_SOURCE = 0xB100_0001
TRAIN_QUERY = 0xB100_0002
TRAIN_WORLD = 0xB100_0003
CAL_SOURCE = 0xB200_0001
CAL_QUERY = 0xB200_0002
HELDOUT_SOURCE = 0xB300_0001
HELDOUT_QUERY = 0xB300_0002
HELDOUT_WORLD = 0xB300_0003
FIXED_EVAL = 0xB400_0001


@dataclass(frozen=True)
class SweepConfig:
    source_games: int = 8192
    envs: int = 512
    # Search from the known minimum useful update through a likely
    # diminishing-returns point.  There are nine independent categories, so
    # these correspond to 576, 1152, 2304 and 4608 states.
    batch_queries_per_category: tuple[int, ...] = (64, 128, 256, 512)
    train_worlds: int = 16
    world_chunk: int = 64
    target_shard_size: int = 64
    target_query_batch_size: int = 64
    rollout_inference_batch_size: int = 128
    calibration_source_games: int = 4096
    # Keep calibration noise below the training-direction noise at the
    # smallest candidate (576 states).
    calibration_queries_per_category: int = 128
    heldout_source_games: int = 4096
    heldout_queries_per_category: int = 32
    heldout_worlds: int = 64
    direction_learning_rate: float = 1e-5
    microbatch_size: int = 64
    inference_batch_size: int = 128
    target_kl: float = 1e-3
    kl_search_steps: int = 18
    maximum_scale: float = 64.0
    evaluation_games: int = 16_384
    evaluation_envs: int = 512
    bootstrap_samples: int = 10_000

    def __post_init__(self) -> None:
        nested = tuple(int(value) for value in self.batch_queries_per_category)
        object.__setattr__(self, "batch_queries_per_category", nested)
        if not nested or tuple(sorted(set(nested))) != nested:
            raise ValueError("batch QPC values must be unique and increasing")
        positive = (
            self.source_games,
            self.envs,
            self.train_worlds,
            self.world_chunk,
            self.target_shard_size,
            self.target_query_batch_size,
            self.rollout_inference_batch_size,
            self.calibration_source_games,
            self.calibration_queries_per_category,
            self.heldout_source_games,
            self.heldout_queries_per_category,
            self.heldout_worlds,
            self.direction_learning_rate,
            self.microbatch_size,
            self.inference_batch_size,
            self.target_kl,
            self.kl_search_steps,
            self.maximum_scale,
            self.evaluation_games,
            self.evaluation_envs,
            self.bootstrap_samples,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("sweep sizes and rates must be positive")
        if self.train_worlds < 2 or self.heldout_worlds < 2:
            raise ValueError("target world counts must be at least two")
        if self.source_games < 9 * nested[-1]:
            raise ValueError("source games cannot cover the maximum independent batch")
        if self.calibration_source_games < 9 * self.calibration_queries_per_category:
            raise ValueError("calibration games cannot cover independent queries")
        if self.heldout_source_games < 9 * self.heldout_queries_per_category:
            raise ValueError("heldout games cannot cover independent queries")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _save_direction(
    path: Path,
    *,
    fingerprint: str,
    initial: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    optimizer: Mapping[str, object],
    elapsed_seconds: float,
) -> None:
    _atomic_torch(
        path,
        {
            "version": DIRECTION_CACHE_VERSION,
            "fingerprint": fingerprint,
            "initial": dict(initial),
            "candidate": dict(candidate),
            "optimizer": dict(optimizer),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def _load_direction(
    path: Path, *, fingerprint: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object], float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "version",
        "fingerprint",
        "initial",
        "candidate",
        "optimizer",
        "elapsed_seconds",
    }
    if set(payload) != expected or int(payload["version"]) != DIRECTION_CACHE_VERSION:
        raise ValueError("direction cache format does not match")
    if str(payload["fingerprint"]) != fingerprint:
        raise ValueError("direction cache fingerprint does not match")
    elapsed = float(payload["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("direction cache elapsed time is invalid")
    return (
        dict(payload["initial"]),
        dict(payload["candidate"]),
        dict(payload["optimizer"]),
        elapsed,
    )


def _save_actor_panel(
    path: Path,
    *,
    fingerprint: str,
    seeds: np.ndarray,
    ranks: np.ndarray,
    scores: np.ndarray,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            version=np.asarray([ACTOR_PANEL_VERSION], dtype=np.int64),
            fingerprint=np.asarray([fingerprint]),
            seeds=np.asarray(seeds, dtype=np.uint64),
            ranks=np.asarray(ranks, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            elapsed_seconds=np.asarray([elapsed_seconds], dtype=np.float64),
        )
    temporary.replace(path)


def _read_actor_panel(
    path: Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
    expected = {
        "version",
        "fingerprint",
        "seeds",
        "ranks",
        "scores",
        "elapsed_seconds",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("Actor panel cache fields do not match")
        if int(payload["version"][0]) != ACTOR_PANEL_VERSION:
            raise ValueError("Actor panel cache version does not match")
        fingerprint = str(payload["fingerprint"][0])
        if not fingerprint:
            raise ValueError("Actor panel cache fingerprint is empty")
        elapsed = float(payload["elapsed_seconds"][0])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Actor panel elapsed time is invalid")
        seeds = payload["seeds"].copy()
        ranks = payload["ranks"].copy()
        scores = payload["scores"].copy()
    if (
        seeds.ndim != 1
        or not len(seeds)
        or ranks.shape != seeds.shape
        or scores.shape != seeds.shape
        or len(np.unique(seeds)) != len(seeds)
        or not np.isfinite(ranks).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Actor panel cache arrays are invalid")
    return fingerprint, seeds, ranks, scores, elapsed


def _load_actor_panel(
    path: Path, *, fingerprint: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    actual, seeds, ranks, scores, elapsed = _read_actor_panel(path)
    if actual != fingerprint:
        raise ValueError("Actor panel cache fingerprint does not match")
    return seeds, ranks, scores, elapsed


def _load_summary(path: Path, identity: Mapping[str, object]) -> dict[str, object]:
    if not path.exists():
        return {"identity": identity, "variants": {}}
    summary = json.loads(path.read_text())
    if summary.get("identity") != identity or not isinstance(
        summary.get("variants"), dict
    ):
        raise ValueError("sweep summary identity does not match")
    return summary


def _collect(
    actor,
    device: torch.device,
    progress: Progress,
    *,
    games: int,
    envs: int,
    qpc: int,
    source_seed: int,
    query_seed: int,
    phase: str,
):
    progress.start(phase, total=games, unit="games")
    collector = TrajectoryCollector(
        CollectionConfig(envs=min(envs, games), history=actor.config.max_history),
        actor,
        device,
        seed=source_seed,
        self_play_fraction=0.0,
    )

    def report(done: int, states: int, elapsed: float) -> None:
        progress.update(
            done,
            fields={"env_states/s": states / max(elapsed, 1e-9)},
        )

    result = collector.collect(games, on_progress=report)
    progress.complete(
        fields={
            "env_states/s": result.environment_steps / max(result.elapsed_seconds, 1e-9)
        }
    )
    progress.start(f"{phase}_SELECT", total=9 * qpc, unit="states")
    queries = select_independent_queries(
        result.trajectories, queries_per_category=qpc, seed=query_seed
    )
    progress.complete(len(queries))
    return queries, result.trajectories


def _targets(
    directory: Path,
    queries,
    actor,
    device: torch.device,
    progress: Progress,
    *,
    phase: str,
    fingerprint: str,
    worlds: int,
    world_chunk: int,
    world_seed: int,
    shard_size: int,
    query_batch_size: int,
    inference_batch_size: int,
):
    progress.start(
        phase,
        total=len(queries),
        unit="queries",
        fields={"worlds": worlds},
    )
    batch, metrics = cached_counterfactual_corpus(
        directory,
        queries,
        actor,
        device,
        fingerprint=fingerprint,
        worlds=worlds,
        world_chunk=world_chunk,
        world_seed=world_seed,
        shard_size=shard_size,
        query_batch_size=query_batch_size,
        inference_batch_size=inference_batch_size,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()
    return batch, metrics


def _reference_panel(
    reference,
    device: torch.device,
    output_dir: Path,
    progress: Progress,
    *,
    seed: int,
    config: SweepConfig,
    sl_hash: str,
):
    seeds = evaluation_seeds(domain_seed(seed, FIXED_EVAL), config.evaluation_games)
    fingerprint = _fingerprint(
        {
            "sl": sl_hash,
            "policy_execution_version": POLICY_EXECUTION_VERSION,
            "seed": int(domain_seed(seed, FIXED_EVAL)),
            "games": config.evaluation_games,
        }
    )
    path = output_dir / "reference_panel.npz"
    if path.exists():
        loaded_seeds, ranks, scores = load_reference_panel(
            path, fingerprint=fingerprint
        )
        if not np.array_equal(seeds, loaded_seeds):
            raise ValueError("cached reference panel seeds do not match")
        return seeds, ranks, scores
    progress.start("REFERENCE_EVAL", total=len(seeds), unit="games")
    result = collect_fixed_panel(
        reference,
        device,
        seeds,
        envs=config.evaluation_envs,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()
    ranks, scores = outcomes(result)
    save_reference_panel(
        path,
        seeds=seeds,
        ranks=ranks,
        scores=scores,
        fingerprint=fingerprint,
    )
    return seeds, ranks, scores


def run(
    sl_checkpoint: Path,
    output_dir: Path,
    *,
    config: SweepConfig | None = None,
    seed: int = 20260727,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    config = config or SweepConfig()
    resolved = require_cuda(device)
    if not sl_checkpoint.exists():
        raise FileNotFoundError(sl_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    sl_checkpoint = sl_checkpoint.resolve()
    sl_hash = _sha256(sl_checkpoint)
    identity = json.loads(
        json.dumps(
            {
                "version": SWEEP_VERSION,
                "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
                "policy_execution_version": POLICY_EXECUTION_VERSION,
                "seed": seed,
                "sl_checkpoint": str(sl_checkpoint),
                "sl_sha256": sl_hash,
                "config": asdict(config),
            },
            sort_keys=True,
        )
    )
    config_path = output_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != identity:
            raise ValueError("existing sweep configuration does not match")
    else:
        temporary = config_path.with_suffix(config_path.suffix + ".tmp")
        entries = list(output_dir.iterdir())
        if entries == [temporary]:
            temporary.unlink()
        elif entries:
            raise ValueError("sweep directory is non-empty without config.json")
        _atomic_json(config_path, identity)

    summary_path = output_dir / "summary.json"
    summary = _load_summary(summary_path, identity)
    expected_variants = {
        str(size) for size in config.batch_queries_per_category
    }
    variants = summary["variants"]
    variant_keys = set(variants)
    if not variant_keys.issubset(expected_variants):
        raise ValueError("sweep summary contains an unknown batch variant")
    if variant_keys == expected_variants:
        for size in config.batch_queries_per_category:
            result = variants[str(size)]["evaluation"]
            rank = result["paired_rank_delta"]
            print(
                f"QPC {size:3d} cached  dRank {rank['mean']:+.4f} "
                f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]",
                flush=True,
            )
        print(f"RESULT cached complete  {summary_path}", flush=True)
        return summary

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    reference = load_policy(sl_checkpoint, resolved, frozen=True)
    require_deterministic_actor(reference)
    progress = Progress()
    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  nested batch sweep  "
        f"QPC {','.join(map(str, config.batch_queries_per_category))}",
        flush=True,
    )

    maximum = config.batch_queries_per_category[-1]
    shared_dir = output_dir / "shared"
    shared_manifest = shared_dir / "manifest.json"
    shared_fingerprint = _fingerprint(
        {"identity": identity, "cache": "shared-corpora"}
    )
    if shared_manifest.exists():
        shared = json.loads(shared_manifest.read_text())
        expected_shared = {
            "version",
            "fingerprint",
            "source_visit_frequencies",
            "train_targets",
            "heldout_targets",
        }
        if (
            set(shared) != expected_shared
            or int(shared["version"]) != SHARED_CACHE_VERSION
            or shared["fingerprint"] != shared_fingerprint
        ):
            raise ValueError("shared sweep cache manifest does not match")
        train_batch = load_counterfactual_batch(shared_dir / "train.npz")
        calibration = load_policy_state_batch(shared_dir / "calibration.npz")
        heldout_batch = load_counterfactual_batch(shared_dir / "heldout.npz")
        visit_stats = shared["source_visit_frequencies"]
        train_metrics = shared["train_targets"]
        heldout_metrics = shared["heldout_targets"]
        print(
            f"SHARED cached  train {len(train_batch):,}  "
            f"cal {len(calibration):,}  heldout {len(heldout_batch):,}",
            flush=True,
        )
    else:
        train_queries, train_sources = _collect(
            reference,
            resolved,
            progress,
            games=config.source_games,
            envs=config.envs,
            qpc=maximum,
            source_seed=domain_seed(seed, TRAIN_SOURCE),
            query_seed=domain_seed(seed, TRAIN_QUERY),
            phase="TRAIN_SOURCE",
        )
        visit_stats = source_visit_frequencies(train_sources)
        train_batch, train_metrics = _targets(
            output_dir / "train_targets",
            train_queries,
            reference,
            resolved,
            progress,
            phase="TRAIN_TARGETS",
            fingerprint=_fingerprint(
                {"identity": identity, "corpus": "train", "qpc": maximum}
            ),
            worlds=config.train_worlds,
            world_chunk=config.world_chunk,
            world_seed=domain_seed(seed, TRAIN_WORLD),
            shard_size=config.target_shard_size,
            query_batch_size=config.target_query_batch_size,
            inference_batch_size=config.rollout_inference_batch_size,
        )
        calibration_queries, calibration_sources = _collect(
            reference,
            resolved,
            progress,
            games=config.calibration_source_games,
            envs=config.envs,
            qpc=config.calibration_queries_per_category,
            source_seed=domain_seed(seed, CAL_SOURCE),
            query_seed=domain_seed(seed, CAL_QUERY),
            phase="CAL_SOURCE",
        )
        calibration = build_state_batch(
            calibration_queries, history=reference.config.max_history
        )
        heldout_queries, heldout_sources = _collect(
            reference,
            resolved,
            progress,
            games=config.heldout_source_games,
            envs=config.envs,
            qpc=config.heldout_queries_per_category,
            source_seed=domain_seed(seed, HELDOUT_SOURCE),
            query_seed=domain_seed(seed, HELDOUT_QUERY),
            phase="HELDOUT_SOURCE",
        )
        source_sets = [
            {trajectory.seed for trajectory in group}
            for group in (train_sources, calibration_sources, heldout_sources)
        ]
        if any(
            source_sets[left] & source_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise RuntimeError("sweep source seed domains overlap")
        heldout_batch, heldout_metrics = _targets(
            output_dir / "heldout_targets",
            heldout_queries,
            reference,
            resolved,
            progress,
            phase="HELDOUT_TARGETS",
            fingerprint=_fingerprint(
                {"identity": identity, "corpus": "heldout"}
            ),
            worlds=config.heldout_worlds,
            world_chunk=config.world_chunk,
            world_seed=domain_seed(seed, HELDOUT_WORLD),
            shard_size=config.target_shard_size,
            query_batch_size=config.target_query_batch_size,
            inference_batch_size=config.rollout_inference_batch_size,
        )
        save_counterfactual_batch(shared_dir / "train.npz", train_batch)
        save_policy_state_batch(shared_dir / "calibration.npz", calibration)
        save_counterfactual_batch(shared_dir / "heldout.npz", heldout_batch)
        _atomic_json(
            shared_manifest,
            {
                "version": SHARED_CACHE_VERSION,
                "fingerprint": shared_fingerprint,
                "source_visit_frequencies": visit_stats,
                "train_targets": train_metrics,
                "heldout_targets": heldout_metrics,
            },
        )

    if len(train_batch) != 9 * maximum:
        raise ValueError("shared training cache has the wrong state count")
    if len(calibration) != 9 * config.calibration_queries_per_category:
        raise ValueError("shared calibration cache has the wrong state count")
    if len(heldout_batch) != 9 * config.heldout_queries_per_category:
        raise ValueError("shared heldout cache has the wrong state count")
    visit_weights = np.asarray(visit_stats["vector"], dtype=np.float64)
    nested = nested_category_indices(
        train_batch.categories, config.batch_queries_per_category
    )
    summary["source_visit_frequencies"] = visit_stats
    summary["train_targets"] = train_metrics
    summary["heldout_targets"] = heldout_metrics
    _atomic_json(summary_path, summary)

    evaluation_panel = _reference_panel(
        reference,
        resolved,
        output_dir,
        progress,
        seed=seed,
        config=config,
        sl_hash=sl_hash,
    )

    maximum_batch = subset_counterfactual_batch(train_batch, nested[maximum])
    reference_digest = _model_digest(reference)

    def direction_for(size: int, batch):
        fingerprint = _fingerprint(
            {
                "identity": identity,
                "reference": reference_digest,
                "qpc": size,
                "query_ids": batch.query_ids.tolist(),
            }
        )
        path = output_dir / "directions" / f"qpc-{size}.pt"
        if path.exists():
            initial, candidate_state, optimizer_metrics, elapsed = _load_direction(
                path, fingerprint=fingerprint
            )
            print(
                f"QPC {size:3d} direction cached  "
                f"{len(batch) / elapsed:,.1f} states/s",
                flush=True,
            )
            return initial, candidate_state, optimizer_metrics, elapsed
        count = math.ceil(len(batch) / config.microbatch_size)
        progress.start(
            f"QPC{size}_DIRECTION", total=count, unit="microbatches"
        )
        started = time.perf_counter()
        actor, initial, candidate_state, optimizer_metrics = one_step_direction(
            reference,
            batch,
            resolved,
            category_weights=visit_weights,
            learning_rate=config.direction_learning_rate,
            microbatch_size=config.microbatch_size,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        elapsed = time.perf_counter() - started
        progress.complete(
            fields={"states/s": len(batch) / max(elapsed, 1e-9)}
        )
        _save_direction(
            path,
            fingerprint=fingerprint,
            initial=initial,
            candidate=candidate_state,
            optimizer=optimizer_metrics,
            elapsed_seconds=elapsed,
        )
        del actor
        torch.cuda.empty_cache()
        return initial, candidate_state, optimizer_metrics, elapsed

    (
        maximum_initial,
        maximum_candidate,
        _maximum_optimizer,
        _maximum_direction_seconds,
    ) = direction_for(maximum, maximum_batch)

    seeds, reference_ranks, reference_scores = evaluation_panel
    for size in config.batch_queries_per_category:
        key = str(size)
        if key in variants:
            result = variants[key]["evaluation"]
            rank = result["paired_rank_delta"]
            print(
                f"QPC {size:3d} cached  dRank {rank['mean']:+.4f} "
                f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]",
                flush=True,
            )
            continue
        batch = subset_counterfactual_batch(train_batch, nested[size])
        if size == maximum:
            initial = maximum_initial
            candidate_state = maximum_candidate
            optimizer_metrics = _maximum_optimizer
            direction_seconds = _maximum_direction_seconds
        else:
            (
                initial,
                candidate_state,
                optimizer_metrics,
                direction_seconds,
            ) = direction_for(size, batch)
        comparison = direction_cosine(
            initial, candidate_state, maximum_candidate
        )
        actor = copy.deepcopy(reference).to(resolved)
        actor.load_state_dict(candidate_state, strict=True)

        max_kl_evals = (
            config.kl_search_steps
            + 2
            + int(math.ceil(math.log2(config.maximum_scale)))
        )
        progress.start(
            f"QPC{size}_KL", total=max_kl_evals, unit="evaluations"
        )
        stage_started = time.perf_counter()
        calibration_metrics = calibrate_direction(
            actor,
            reference,
            initial,
            candidate_state,
            calibration,
            resolved,
            category_weights=visit_weights,
            target_kl=config.target_kl,
            batch_size=config.inference_batch_size,
            search_steps=config.kl_search_steps,
            maximum_scale=config.maximum_scale,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        calibration_seconds = time.perf_counter() - stage_started
        progress.complete(
            int(calibration_metrics["evaluations"]),
            fields={
                "scale": calibration_metrics["scale"],
                "kl": calibration_metrics["final_kl"],
            },
        )

        stage_started = time.perf_counter()
        heldout = heldout_policy_value(
            actor,
            reference,
            heldout_batch,
            resolved,
            category_weights=visit_weights,
            batch_size=config.inference_batch_size,
        )
        heldout_seconds = time.perf_counter() - stage_started
        save_policy(output_dir / f"actor-qpc{size}.pt", actor)

        actor_digest = _model_digest(actor)
        panel_fingerprint = _fingerprint(
            {
                "identity": identity,
                "qpc": size,
                "actor": actor_digest,
                "seeds": _fingerprint(seeds.tolist()),
            }
        )
        panel_path = output_dir / "evaluation" / f"qpc-{size}.npz"
        if panel_path.exists():
            panel_seeds, actor_ranks, actor_scores, evaluation_seconds = (
                _load_actor_panel(panel_path, fingerprint=panel_fingerprint)
            )
            if not np.array_equal(panel_seeds, seeds):
                raise ValueError("cached Actor evaluation seeds do not match")
            print(f"QPC {size:3d} evaluation cached", flush=True)
        else:
            progress.start(
                f"QPC{size}_EVAL", total=len(seeds), unit="games"
            )
            stage_started = time.perf_counter()
            actor_result = collect_fixed_panel(
                actor,
                resolved,
                seeds,
                envs=config.evaluation_envs,
                on_progress=lambda done, values: progress.update(done, fields=values),
            )
            evaluation_seconds = time.perf_counter() - stage_started
            progress.complete(
                fields={"games/s": len(seeds) / max(evaluation_seconds, 1e-9)}
            )
            actor_ranks, actor_scores = outcomes(actor_result)
            _save_actor_panel(
                panel_path,
                fingerprint=panel_fingerprint,
                seeds=seeds,
                ranks=actor_ranks,
                scores=actor_scores,
                elapsed_seconds=evaluation_seconds,
            )
        evaluation = summarize_paired(
            actor_ranks,
            actor_scores,
            reference_ranks,
            reference_scores,
            seed=domain_seed(seed, 0xB500_0001, size),
            bootstrap_samples=config.bootstrap_samples,
        )
        elapsed = (
            direction_seconds
            + calibration_seconds
            + heldout_seconds
            + evaluation_seconds
        )
        variants[key] = {
            "queries_per_category": size,
            "states": len(batch),
            "optimizer": optimizer_metrics,
            "direction": comparison,
            "calibration": calibration_metrics,
            "heldout": heldout,
            "evaluation": evaluation,
            "timing": {
                "direction_seconds": direction_seconds,
                "calibration_seconds": calibration_seconds,
                "heldout_seconds": heldout_seconds,
                "evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": elapsed,
                "direction_states_per_second": len(batch)
                / max(direction_seconds, 1e-9),
                "evaluation_games_per_second": len(seeds)
                / max(evaluation_seconds, 1e-9),
            },
        }
        _atomic_json(summary_path, summary)
        rank = evaluation["paired_rank_delta"]
        score = evaluation["paired_score_delta"]
        print(
            f"QPC {size:3d}  states {len(batch):4d}  "
            f"cos {comparison['cosine_to_maximum']:.4f}  "
            f"dRank {rank['mean']:+.4f} "
            f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]  "
            f"score {score['mean']:+.0f}  "
            f"heldout {heldout['visitation_weighted_rank_value']:+.5f}  "
            f"flip {100 * calibration_metrics['greedy_flip_rate']:.1f}%  "
            f"dir {len(batch) / max(direction_seconds, 1e-9):,.1f} states/s",
            flush=True,
        )
        del actor
        torch.cuda.empty_cache()
    print(f"RESULT complete  {summary_path}", flush=True)
    return summary


def _multi_identity(
    sl_checkpoint: Path, config: SweepConfig, seeds: Sequence[int]
) -> dict[str, object]:
    """Return the immutable identity for a multi-seed sweep directory."""
    sl_checkpoint = sl_checkpoint.resolve()
    normalized = [int(seed) for seed in seeds]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("multi-seed sweep needs unique seeds")
    if not sl_checkpoint.exists():
        raise FileNotFoundError(sl_checkpoint)
    return json.loads(
        json.dumps(
            {
                "version": MULTI_SWEEP_VERSION,
                "sweep_version": SWEEP_VERSION,
                "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
                "policy_execution_version": POLICY_EXECUTION_VERSION,
                "seeds": normalized,
                "sl_checkpoint": str(sl_checkpoint),
                "sl_sha256": _sha256(sl_checkpoint),
                "config": asdict(config),
            },
            sort_keys=True,
        )
    )


def _prepare_multi_directory(
    output_dir: Path, identity: Mapping[str, object]
) -> None:
    """Create or validate the root manifest before starting child sweeps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "multi_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != identity:
            raise ValueError("existing multi-seed configuration does not match")
        return
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    entries = [path for path in output_dir.iterdir() if path != temporary]
    if entries:
        raise ValueError(
            "multi-seed sweep directory is non-empty without multi_config.json"
        )
    if temporary.exists():
        temporary.unlink()
    _atomic_json(config_path, identity)


def _multi_summary_seed(seed: int, seeds: Sequence[int]) -> int:
    payload = json.dumps([int(value) for value in seeds], separators=(",", ":"))
    digest = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
    return domain_seed(digest, 0xB600_0001, int(seed))


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("multi-seed metric values must be finite and non-empty")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _load_actor_panel_unchecked(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a panel for aggregation while still validating its on-disk format."""
    _fingerprint_value, seeds, ranks, scores, elapsed = _read_actor_panel(path)
    return seeds, ranks, scores, elapsed


def _aggregate_multi_seed(
    output_dir: Path,
    *,
    identity: Mapping[str, object],
    seeds: Sequence[int],
    config: SweepConfig,
    sl_hash: str,
    child_summaries: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Pool raw paired panels and retain per-seed metrics for variance checks."""
    variants: dict[str, object] = {}
    for qpc in config.batch_queries_per_category:
        print(
            f"AGGREGATE QPC {qpc:3d}  "
            f"{len(seeds) * config.evaluation_games:,} paired games",
            flush=True,
        )
        key = str(qpc)
        per_seed: dict[str, object] = {}
        pooled_actor_ranks: list[np.ndarray] = []
        pooled_actor_scores: list[np.ndarray] = []
        pooled_reference_ranks: list[np.ndarray] = []
        pooled_reference_scores: list[np.ndarray] = []
        pooled_seeds: list[np.ndarray] = []
        rank_means: list[float] = []
        score_means: list[float] = []
        heldout_rank_values: list[float] = []
        heldout_score_values: list[float] = []
        elapsed_values: list[float] = []

        for seed in seeds:
            child_dir = output_dir / f"seed-{int(seed)}"
            child = child_summaries[int(seed)]
            child_variants = child.get("variants")
            if not isinstance(child_variants, dict) or key not in child_variants:
                raise ValueError(f"seed {seed} is missing QPC {qpc} summary")
            child_variant = child_variants[key]
            if not isinstance(child_variant, dict):
                raise ValueError(f"seed {seed} has an invalid QPC {qpc} summary")

            panel_path = child_dir / "evaluation" / f"qpc-{qpc}.npz"
            actor_seeds, actor_ranks, actor_scores, panel_seconds = (
                _load_actor_panel_unchecked(panel_path)
            )
            reference_fingerprint = _fingerprint(
                {
                    "sl": sl_hash,
                    "policy_execution_version": POLICY_EXECUTION_VERSION,
                    "seed": int(domain_seed(int(seed), FIXED_EVAL)),
                    "games": config.evaluation_games,
                }
            )
            reference_seeds, reference_ranks, reference_scores = load_reference_panel(
                child_dir / "reference_panel.npz",
                fingerprint=reference_fingerprint,
            )
            if not np.array_equal(actor_seeds, reference_seeds):
                raise ValueError(f"seed {seed} QPC {qpc} panel seeds do not match")
            if len(actor_seeds) != config.evaluation_games:
                raise ValueError(f"seed {seed} QPC {qpc} panel size does not match")

            evaluation = child_variant.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError(f"seed {seed} QPC {qpc} lacks evaluation metrics")
            raw_rank_mean = float((actor_ranks - reference_ranks).mean())
            raw_score_mean = float((actor_scores - reference_scores).mean())
            if not math.isclose(
                raw_rank_mean,
                float(evaluation["paired_rank_delta"]["mean"]),
                abs_tol=1e-12,
            ) or not math.isclose(
                raw_score_mean,
                float(evaluation["paired_score_delta"]["mean"]),
                abs_tol=1e-12,
            ):
                raise ValueError(f"seed {seed} QPC {qpc} panel metrics do not match")
            rank_means.append(raw_rank_mean)
            score_means.append(raw_score_mean)
            heldout = child_variant.get("heldout")
            if not isinstance(heldout, dict):
                raise ValueError(f"seed {seed} QPC {qpc} lacks heldout metrics")
            heldout_rank_values.append(
                float(heldout["visitation_weighted_rank_value"])
            )
            heldout_score_values.append(
                float(heldout["visitation_weighted_score_value"])
            )
            timing = child_variant.get("timing")
            if not isinstance(timing, dict):
                raise ValueError(f"seed {seed} QPC {qpc} lacks timing metrics")
            elapsed_values.append(float(timing["elapsed_seconds"]))
            per_seed[str(int(seed))] = {
                "states": int(child_variant["states"]),
                "evaluation": evaluation,
                "heldout": heldout,
                "timing": timing,
                "panel_seconds": float(panel_seconds),
            }
            pooled_seeds.append(actor_seeds)
            pooled_actor_ranks.append(actor_ranks)
            pooled_actor_scores.append(actor_scores)
            pooled_reference_ranks.append(reference_ranks)
            pooled_reference_scores.append(reference_scores)

        all_panel_seeds = np.concatenate(pooled_seeds)
        if len(np.unique(all_panel_seeds)) != len(all_panel_seeds):
            raise ValueError(f"QPC {qpc} multi-seed evaluation panels overlap")
        pooled_evaluation = summarize_paired(
            np.concatenate(pooled_actor_ranks),
            np.concatenate(pooled_actor_scores),
            np.concatenate(pooled_reference_ranks),
            np.concatenate(pooled_reference_scores),
            seed=_multi_summary_seed(qpc, seeds),
            bootstrap_samples=config.bootstrap_samples,
        )
        seed_metrics: dict[str, object] = {
            "paired_rank_delta": _numeric_summary(rank_means),
            "paired_score_delta": _numeric_summary(score_means),
        }
        seed_metrics["heldout_visitation_weighted_rank_value"] = _numeric_summary(
            heldout_rank_values
        )
        seed_metrics["heldout_visitation_weighted_score_value"] = _numeric_summary(
            heldout_score_values
        )
        seed_metrics["elapsed_seconds"] = _numeric_summary(elapsed_values)
        variants[key] = {
            "queries_per_category": int(qpc),
            "states": int(9 * qpc),
            "seeds": per_seed,
            "seed_metrics": seed_metrics,
            "pooled_evaluation": pooled_evaluation,
        }
    return {
        "identity": dict(identity),
        "seeds": [int(seed) for seed in seeds],
        "variants": variants,
    }


def _load_multi_summary(
    path: Path,
    *,
    identity: Mapping[str, object],
    config: SweepConfig,
    seeds: Sequence[int],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    expected = {str(qpc) for qpc in config.batch_queries_per_category}
    if (
        summary.get("identity") != identity
        or summary.get("seeds") != [int(seed) for seed in seeds]
        or not isinstance(summary.get("variants"), dict)
        or set(summary["variants"]) != expected
    ):
        raise ValueError("multi-seed summary identity does not match")
    return summary


def run_many(
    sl_checkpoint: Path,
    output_dir: Path,
    *,
    seeds: Sequence[int],
    config: SweepConfig | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Run independent sweeps and aggregate their raw fixed-evaluation panels.

    Each child uses the existing single-seed ``run`` implementation, so a
    stopped child resumes from its own caches without invalidating siblings.
    """
    config = config or SweepConfig()
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("multi-seed sweep needs unique seeds")
    if len(normalized) == 1:
        return run(
            sl_checkpoint,
            output_dir,
            config=config,
            seed=normalized[0],
            device=device,
        )
    identity = _multi_identity(sl_checkpoint, config, normalized)
    _prepare_multi_directory(output_dir, identity)
    aggregate_path = output_dir / "aggregate.json"
    cached = _load_multi_summary(
        aggregate_path,
        identity=identity,
        config=config,
        seeds=normalized,
    )
    if cached is not None:
        print(f"RESULT multi-seed cached  {aggregate_path}", flush=True)
        return cached

    child_summaries: dict[int, Mapping[str, object]] = {}
    for index, seed in enumerate(normalized, start=1):
        child_dir = output_dir / f"seed-{seed}"
        print(
            f"SEED {index}/{len(normalized)}  seed {seed}  output {child_dir}",
            flush=True,
        )
        child_summaries[seed] = run(
            sl_checkpoint,
            child_dir,
            config=config,
            seed=seed,
            device=device,
        )

    aggregate = _aggregate_multi_seed(
        output_dir,
        identity=identity,
        seeds=normalized,
        config=config,
        sl_hash=str(identity["sl_sha256"]),
        child_summaries=child_summaries,
    )
    _atomic_json(aggregate_path, aggregate)
    for qpc in config.batch_queries_per_category:
        pooled = aggregate["variants"][str(qpc)]["pooled_evaluation"]
        rank = pooled["paired_rank_delta"]
        score = pooled["paired_score_delta"]
        seed_rank = aggregate["variants"][str(qpc)]["seed_metrics"][
            "paired_rank_delta"
        ]
        print(
            f"QPC {qpc:3d} multi  pooled dRank {rank['mean']:+.4f} "
            f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]  "
            f"score {score['mean']:+.0f}  "
            f"seed dRank {seed_rank['mean']:+.4f} +/- {seed_rank['std']:.4f}",
            flush=True,
        )
    print(f"RESULT multi-seed complete  {aggregate_path}", flush=True)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sl-checkpoint",
        type=Path,
        default=Path("runs/counterfactual-larger/sl_reference.pt"),
    )
    parser.add_argument("--output-dir", type=Path)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=20260727)
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="independent sweep seeds; two or more create a pooled aggregate",
    )
    parser.add_argument("--source-games", type=int, default=SweepConfig.source_games)
    parser.add_argument("--envs", type=int, default=SweepConfig.envs)
    parser.add_argument(
        "--batch-qpc",
        type=int,
        nargs="+",
        default=list(SweepConfig.batch_queries_per_category),
    )
    parser.add_argument("--train-worlds", type=int, default=SweepConfig.train_worlds)
    parser.add_argument("--world-chunk", type=int, default=SweepConfig.world_chunk)
    parser.add_argument(
        "--target-shard-size", type=int, default=SweepConfig.target_shard_size
    )
    parser.add_argument(
        "--target-query-batch-size",
        type=int,
        default=SweepConfig.target_query_batch_size,
    )
    parser.add_argument(
        "--rollout-inference-batch-size",
        type=int,
        default=SweepConfig.rollout_inference_batch_size,
    )
    parser.add_argument(
        "--calibration-source-games",
        type=int,
        default=SweepConfig.calibration_source_games,
    )
    parser.add_argument(
        "--calibration-qpc",
        type=int,
        default=SweepConfig.calibration_queries_per_category,
    )
    parser.add_argument(
        "--heldout-source-games",
        type=int,
        default=SweepConfig.heldout_source_games,
    )
    parser.add_argument(
        "--heldout-qpc",
        type=int,
        default=SweepConfig.heldout_queries_per_category,
    )
    parser.add_argument(
        "--heldout-worlds", type=int, default=SweepConfig.heldout_worlds
    )
    parser.add_argument(
        "--evaluation-games", type=int, default=SweepConfig.evaluation_games
    )
    parser.add_argument(
        "--evaluation-envs", type=int, default=SweepConfig.evaluation_envs
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = SweepConfig(
        source_games=args.source_games,
        envs=args.envs,
        batch_queries_per_category=tuple(args.batch_qpc),
        train_worlds=args.train_worlds,
        world_chunk=args.world_chunk,
        target_shard_size=args.target_shard_size,
        target_query_batch_size=args.target_query_batch_size,
        rollout_inference_batch_size=args.rollout_inference_batch_size,
        calibration_source_games=args.calibration_source_games,
        calibration_queries_per_category=args.calibration_qpc,
        heldout_source_games=args.heldout_source_games,
        heldout_queries_per_category=args.heldout_qpc,
        heldout_worlds=args.heldout_worlds,
        evaluation_games=args.evaluation_games,
        evaluation_envs=args.evaluation_envs,
    )
    if args.smoke:
        config = replace(
            config,
            source_games=512,
            envs=128,
            batch_queries_per_category=(1, 2),
            train_worlds=2,
            world_chunk=2,
            target_shard_size=3,
            target_query_batch_size=3,
            rollout_inference_batch_size=16,
            calibration_source_games=256,
            calibration_queries_per_category=8,
            heldout_source_games=256,
            heldout_queries_per_category=1,
            heldout_worlds=2,
            microbatch_size=8,
            inference_batch_size=16,
            kl_search_steps=18,
            evaluation_games=128,
            evaluation_envs=128,
            bootstrap_samples=200,
        )
    if args.smoke:
        default_output = (
            "/tmp/batch-sweep-smoke-multiseed"
            if args.seeds is not None
            else "/tmp/batch-sweep-smoke"
        )
    else:
        default_output = (
            "runs/batch-sweep-v3-multiseed"
            if args.seeds is not None
            else "runs/batch-sweep-v3"
        )
    output_dir = args.output_dir or Path(default_output)
    try:
        if args.seeds is None:
            run(
                args.sl_checkpoint,
                output_dir,
                config=config,
                seed=args.seed,
            )
        else:
            run_many(
                args.sl_checkpoint,
                output_dir,
                config=config,
                seeds=args.seeds,
            )
    except KeyboardInterrupt:
        print(
            f"\nStopped. Re-run the same command to resume {output_dir}.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
