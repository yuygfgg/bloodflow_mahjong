"""Validate paired world rollouts against a full all-action corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from training.pipeline import CollectionConfig, TrajectoryCollector, clone_policy, load_policy
from training.policy_iteration import policy_outputs, select_independent_queries
from training.search_distillation import select_rank_lcb_challengers
from training.world_outcomes import estimate_world_outcome_batch


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.games < 9 * args.queries_per_category:
        raise ValueError("source games cannot cover the query quotas")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    actor = load_policy(args.policy, device, frozen=True)
    opponent = clone_policy(actor, device)
    collector = TrajectoryCollector(
        CollectionConfig(
            envs=min(args.envs, args.games),
            history=actor.config.max_history,
            inference_batch_size=args.inference_batch_size,
        ),
        actor,
        device,
        seed=args.seed,
        self_play_fraction=args.self_play_fraction,
        self_play_actor=opponent,
    )
    collection = collector.collect(args.games)
    print(
        f"source games={args.games} states={collection.environment_steps:,} "
        f"seconds={collection.elapsed_seconds:.3f}",
        flush=True,
    )
    queries = select_independent_queries(
        collection.trajectories,
        queries_per_category=args.queries_per_category,
        seed=args.seed + 1,
    )
    print(f"full rollout queries={len(queries)}", flush=True)
    full, full_metrics = estimate_world_outcome_batch(
        queries,
        actor,
        device,
        self_play_actor=opponent,
        worlds=args.worlds,
        world_chunk=args.world_chunk,
        seed=args.seed + 2,
        world_sampling="live_wall",
        query_batch_size=args.query_batch_size,
        inference_batch_size=args.inference_batch_size,
    )
    print(
        f"full actions={full_metrics['evaluated_actions']} "
        f"states={full_metrics['rollout_states']:,} "
        f"seconds={full_metrics['rollout_seconds']:.3f} "
        f"states/s={full_metrics['rollout_states_per_second']:.0f}",
        flush=True,
    )
    if args.skip_paired:
        return
    probabilities, reference_actions = policy_outputs(
        actor, full, device, batch_size=args.inference_batch_size
    )
    del probabilities
    selected_actions = select_rank_lcb_challengers(full, reference_actions)
    action_sets = tuple(
        np.asarray((reference, selected), dtype=np.uint8)
        for reference, selected in zip(reference_actions, selected_actions)
    )
    print("paired rollout", flush=True)
    paired, paired_metrics = estimate_world_outcome_batch(
        queries,
        actor,
        device,
        self_play_actor=opponent,
        worlds=args.worlds,
        world_chunk=args.world_chunk,
        seed=args.seed + 2,
        world_sampling="live_wall",
        query_batch_size=args.query_batch_size,
        inference_batch_size=args.inference_batch_size,
        action_sets=action_sets,
    )
    rows = np.arange(len(full))
    full_rank = np.stack(
        (
            full.rank_outcomes[rows, reference_actions],
            full.rank_outcomes[rows, selected_actions],
        ),
        axis=1,
    )
    paired_rank = np.stack(
        (
            paired.rank_outcomes[rows, reference_actions],
            paired.rank_outcomes[rows, selected_actions],
        ),
        axis=1,
    )
    full_score = np.stack(
        (
            full.score_outcomes[rows, reference_actions],
            full.score_outcomes[rows, selected_actions],
        ),
        axis=1,
    )
    paired_score = np.stack(
        (
            paired.score_outcomes[rows, reference_actions],
            paired.score_outcomes[rows, selected_actions],
        ),
        axis=1,
    )
    rank_mismatches = int(np.count_nonzero(full_rank != paired_rank))
    score_mismatches = int(np.count_nonzero(full_score != paired_score))
    maximum_score_delta = float(np.max(np.abs(full_score - paired_score)))
    print(
        f"queries={len(queries)} worlds={args.worlds} "
        f"full_actions={full_metrics['evaluated_actions']} "
        f"paired_actions={paired_metrics['evaluated_actions']} "
        f"full_states={full_metrics['rollout_states']:,} "
        f"paired_states={paired_metrics['rollout_states']:,} "
        f"state_reduction="
        f"{1.0 - paired_metrics['rollout_states'] / full_metrics['rollout_states']:.1%}",
        flush=True,
    )
    print(
        f"full_seconds={full_metrics['rollout_seconds']:.3f} "
        f"paired_seconds={paired_metrics['rollout_seconds']:.3f} "
        f"speedup={full_metrics['rollout_seconds'] / paired_metrics['rollout_seconds']:.2f}x "
        f"rank_mismatches={rank_mismatches} score_mismatches={score_mismatches} "
        f"max_score_delta={maximum_score_delta:.6g}",
        flush=True,
    )
    if rank_mismatches:
        raise RuntimeError("paired rollout changed an evaluated rank outcome")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--queries-per-category", type=int, default=1)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--world-chunk", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--self-play-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--skip-paired", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.query_batch_size <= 0:
        raise ValueError("query batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
