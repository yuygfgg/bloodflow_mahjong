"""Exact seeded fixed-rule evaluation with paired confidence intervals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from .model import BloodFlowTransformer
from .pipeline import CollectionConfig, CollectionResult, TrajectoryCollector
from .policy_iteration import mix64


EvaluationProgress = Callable[[int, Mapping[str, object]], None]
REFERENCE_PANEL_VERSION = 2


def evaluation_seeds(root: int, games: int) -> np.ndarray:
    if games <= 0:
        raise ValueError("evaluation games must be positive")
    seeds = np.asarray([mix64(root + index) for index in range(games)], dtype=np.uint64)
    if len(np.unique(seeds)) != games:
        raise RuntimeError("evaluation seed generation produced a collision")
    return seeds


def collect_fixed_panel(
    actor: BloodFlowTransformer,
    device: torch.device,
    seeds: np.ndarray,
    *,
    envs: int,
    on_progress: EvaluationProgress | None = None,
) -> CollectionResult:
    return collect_policy_panel(
        actor,
        device,
        seeds,
        envs=envs,
        self_play_fraction=0.0,
        on_progress=on_progress,
    )


def collect_policy_panel(
    actor: BloodFlowTransformer,
    device: torch.device,
    seeds: np.ndarray,
    *,
    envs: int,
    self_play_fraction: float,
    self_play_actor: BloodFlowTransformer | None = None,
    anchor_rule_fast: bool = False,
    lineup_seed: int = 0,
    on_progress: EvaluationProgress | None = None,
) -> CollectionResult:
    """Collect a seeded focal-policy panel against one reproducible lineup mix."""
    seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
    if seeds.ndim != 1 or not len(seeds) or envs <= 0:
        raise ValueError("policy panel needs seeds and positive envs")
    collector = TrajectoryCollector(
        CollectionConfig(envs=min(envs, len(seeds)), history=actor.config.max_history),
        actor,
        device,
        seed=lineup_seed,
        self_play_fraction=self_play_fraction,
        self_play_actor=self_play_actor,
        anchor_rule_fast=anchor_rule_fast,
    )
    trajectories = []
    focal_rows = []
    environment_steps = 0
    policy_actions = 0
    elapsed_seconds = 0.0
    source_counts: dict[str, int] = {}
    opponent_seat_counts: dict[str, int] = {}
    for start in range(0, len(seeds), envs):
        stop = min(start + envs, len(seeds))

        def chunk_progress(
            completed: int, steps: int, elapsed: float
        ) -> None:
            if on_progress is not None:
                on_progress(
                    start + completed,
                    {
                        "environment_steps": environment_steps + steps,
                        "environment_states_per_second": (environment_steps + steps)
                        / max(elapsed_seconds + elapsed, 1e-9),
                    },
                )

        result = collector.collect_seeded(
            seeds[start:stop],
            focal_offset=start,
            on_progress=chunk_progress,
        )
        trajectories.extend(result.trajectories)
        focal_rows.append(result.focal_seats)
        environment_steps += result.environment_steps
        policy_actions += result.policy_actions
        elapsed_seconds += result.elapsed_seconds
        for source, count in result.source_counts.items():
            source_counts[source] = source_counts.get(source, 0) + count
        for source, count in result.opponent_seat_counts.items():
            opponent_seat_counts[source] = (
                opponent_seat_counts.get(source, 0) + count
            )
    return CollectionResult(
        trajectories=tuple(trajectories),
        focal_seats=np.concatenate(focal_rows),
        environment_steps=environment_steps,
        policy_actions=policy_actions,
        elapsed_seconds=elapsed_seconds,
        source_counts=source_counts,
        opponent_seat_counts=opponent_seat_counts,
    )


def outcomes(result: CollectionResult) -> tuple[np.ndarray, np.ndarray]:
    if len(result.trajectories) != len(result.focal_seats):
        raise ValueError("evaluation trajectories and focal seats are misaligned")
    ranks = np.asarray(
        [
            trajectory.terminal_ranks[int(seat)]
            for trajectory, seat in zip(result.trajectories, result.focal_seats)
        ],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            trajectory.terminal_scores[int(seat)] - 10_000
            for trajectory, seat in zip(result.trajectories, result.focal_seats)
        ],
        dtype=np.float64,
    )
    return ranks, scores


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or samples <= 0:
        raise ValueError("bootstrap needs values and positive samples")
    random = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    support, frequencies = np.unique(values, return_counts=True)
    probabilities = frequencies.astype(np.float64) / len(values)
    chunk = min(256, samples)
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        bootstrap_counts = random.multinomial(
            len(values), probabilities, size=stop - start
        )
        means[start:stop] = bootstrap_counts @ support / len(values)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize_paired(
    actor_ranks: np.ndarray,
    actor_scores: np.ndarray,
    reference_ranks: np.ndarray,
    reference_scores: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    if not (
        actor_ranks.shape
        == actor_scores.shape
        == reference_ranks.shape
        == reference_scores.shape
    ):
        raise ValueError("paired outcome arrays must have identical shapes")
    rank_delta = actor_ranks - reference_ranks
    score_delta = actor_scores - reference_scores
    rank = bootstrap_mean_interval(
        rank_delta, seed=seed, samples=bootstrap_samples
    )
    score = bootstrap_mean_interval(
        score_delta, seed=seed + 1, samples=bootstrap_samples
    )
    return {
        "games": int(len(actor_ranks)),
        "actor": {
            "mean_rank": float(actor_ranks.mean()),
            "mean_score_delta": float(actor_scores.mean()),
            "first_rate": float(np.mean(actor_ranks == 1)),
            "last_rate": float(np.mean(actor_ranks == 4)),
        },
        "reference": {
            "mean_rank": float(reference_ranks.mean()),
            "mean_score_delta": float(reference_scores.mean()),
            "first_rate": float(np.mean(reference_ranks == 1)),
            "last_rate": float(np.mean(reference_ranks == 4)),
        },
        "paired_rank_delta": rank,
        "paired_score_delta": score,
        "rank_significant_positive": rank["ci95_high"] < 0,
        "rank_significant_harmful": rank["ci95_low"] > 0,
        "score_significant_positive": score["ci95_low"] > 0,
    }


def save_reference_panel(
    path: Path,
    *,
    seeds: np.ndarray,
    ranks: np.ndarray,
    scores: np.ndarray,
    fingerprint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            version=np.asarray([REFERENCE_PANEL_VERSION], dtype=np.int64),
            seeds=np.asarray(seeds, dtype=np.uint64),
            ranks=np.asarray(ranks, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            fingerprint=np.asarray([fingerprint]),
        )
    temporary.replace(path)


def load_reference_panel(
    path: Path, *, fingerprint: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = {"version", "seeds", "ranks", "scores", "fingerprint"}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("reference panel cache fields do not match")
        if int(payload["version"][0]) != REFERENCE_PANEL_VERSION:
            raise ValueError("unsupported reference panel version")
        if str(payload["fingerprint"][0]) != fingerprint:
            raise ValueError("reference panel fingerprint does not match")
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
        raise ValueError("reference panel cache arrays are invalid")
    return seeds, ranks, scores


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


__all__ = [
    "atomic_json",
    "bootstrap_mean_interval",
    "collect_fixed_panel",
    "collect_policy_panel",
    "evaluation_seeds",
    "load_reference_panel",
    "outcomes",
    "save_reference_panel",
    "summarize_paired",
]
