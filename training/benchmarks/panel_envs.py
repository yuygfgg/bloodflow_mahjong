"""Benchmark policy-panel environment counts on one exact seed panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from training.evaluation import collect_policy_panel, evaluation_seeds, outcomes
from training.pipeline import load_policy


def _parse_positive_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("envs must be positive comma-separated integers")
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    actor = load_policy(args.policy, device, frozen=True)
    self_play_actor = (
        None
        if args.self_play_policy is None
        else load_policy(args.self_play_policy, device, frozen=True)
    )
    seeds = evaluation_seeds(args.seed, args.games)
    reference: tuple[np.ndarray, np.ndarray] | None = None
    for envs in args.envs:
        result = collect_policy_panel(
            actor,
            device,
            seeds,
            envs=envs,
            self_play_fraction=args.self_play_fraction,
            self_play_actor=self_play_actor,
            anchor_rule_fast=args.anchor_rule_fast,
            lineup_seed=args.lineup_seed,
        )
        rank, score = outcomes(result)
        if reference is None:
            rank_mismatches = 0
            score_mismatches = 0
            reference = rank, score
        else:
            rank_mismatches = int(np.count_nonzero(rank != reference[0]))
            score_mismatches = int(np.count_nonzero(score != reference[1]))
        digest = hashlib.sha256(rank.tobytes() + score.tobytes()).hexdigest()[:16]
        elapsed = result.elapsed_seconds
        environment_steps = result.environment_steps
        print(
            f"envs={envs:4d} games/s={args.games / elapsed:8.1f} "
            f"states/s={environment_steps / elapsed:9.0f} "
            f"rank_mismatches={rank_mismatches} score_mismatches={score_mismatches} "
            f"digest={digest}",
            flush=True,
        )
        del result, rank, score
        gc.collect()
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--self-play-policy", type=Path)
    parser.add_argument("--self-play-fraction", type=float, default=0.0)
    parser.add_argument(
        "--anchor-rule-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lineup-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--envs", type=_parse_positive_csv, default=(256, 512, 1024, 2048))
    parser.add_argument("--seed", type=int, default=20260729)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.games <= 0 or max(args.envs) > args.games:
        raise ValueError("games must be positive and cover the largest env count")
    if not 0.0 <= args.self_play_fraction <= 2.0 / 3.0:
        raise ValueError("self-play fraction must be in [0, 2/3]")
    if args.self_play_fraction > 0 and args.self_play_policy is None:
        raise ValueError("self-play needs a policy checkpoint")
    run(args)


if __name__ == "__main__":
    main()
