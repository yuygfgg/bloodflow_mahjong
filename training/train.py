"""Command-line PPO training entry point for the Transformer pipeline."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .model import BloodFlowTransformer
from .pipeline import (
    OpponentPool,
    PPOConfig,
    RolloutCollector,
    cosine_learning_rate,
    evaluate_against_rules,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/transformer"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--total-transitions", type=int, default=200_000_000)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--microbatch",
        type=int,
        help="PPO forward/backward batch size; increase until GPU memory is nearly full",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke", action="store_true", help="small CPU/GPU end-to-end check"
    )
    return parser


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    config = PPOConfig()
    if args.smoke:
        config = replace(
            config,
            envs=8,
            rollout_transitions=64,
            ppo_epochs=1,
            minibatch=32,
            microbatch=16,
            opponent_refresh_updates=2,
        )
        args.hours = 0.02
        args.total_transitions = 128
        args.max_updates = 2
        args.eval_every = 1
        args.eval_games = 8
        args.checkpoint_every = 1
    if args.microbatch is not None:
        if args.microbatch <= 0:
            raise ValueError("--microbatch must be positive")
        config = replace(config, microbatch=args.microbatch)

    model = BloodFlowTransformer().to(device)
    optimizer_kwargs = {"lr": config.learning_rate, "betas": (0.9, 0.95), "eps": 1e-5}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    collector = RolloutCollector(config, device, seed=args.seed + 1)
    update = 0
    transitions = 0
    if args.resume:
        update, transitions = load_checkpoint(
            args.resume, model, optimizer, device, collector.pool
        )
    initial_transitions = transitions

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "ppo": asdict(config),
                "model": asdict(model.config),
                "args": vars(args)
                | {
                    "output_dir": str(args.output_dir),
                    "resume": str(args.resume) if args.resume else None,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    deadline = time.monotonic() + args.hours * 3600.0
    start = time.monotonic()
    log_path = args.output_dir / "metrics.jsonl"
    milestone_hours = (2, 6, 12, 24)
    saved_milestones = {
        hour
        for hour in milestone_hours
        if (args.output_dir / f"snapshot_{hour}h.pt").exists()
    }

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    while transitions < args.total_transitions and time.monotonic() < deadline:
        if args.max_updates and update >= args.max_updates:
            break
        progress = transitions / max(args.total_transitions, 1)
        if collector.pool.rule_mix_streak >= config.rule_gate_consecutive_evals and (
            not collector.pool.frozen_ready
            or update % config.opponent_refresh_updates == 0
        ):
            collector.pool.refresh_snapshot(model, device)

        learning_rate = cosine_learning_rate(config, progress)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        synchronize()
        rollout_start = time.perf_counter()
        rollout = collector.collect(
            model,
            config.rollout_transitions,
            collect_auxiliary=progress < config.auxiliary_decay_fraction,
        )
        synchronize()
        rollout_seconds = time.perf_counter() - rollout_start
        ppo_start = time.perf_counter()
        stats = ppo_update(model, optimizer, rollout, config, device, progress)
        synchronize()
        ppo_seconds = time.perf_counter() - ppo_start
        transitions += len(rollout)
        update += 1
        elapsed = time.monotonic() - start
        record = {
            "update": update,
            "transitions": transitions,
            "elapsed_seconds": elapsed,
            "states_per_second": (transitions - initial_transitions)
            / max(elapsed, 1e-6),
            "rollout_seconds": rollout_seconds,
            "rollout_states_per_second": len(rollout)
            / max(rollout_seconds, 1e-6),
            "ppo_seconds": ppo_seconds,
            "learning_rate": learning_rate,
            "opponent_stage": rollout.opponent_stage,
            "opponent_assignments": {
                name: int(count)
                for name, count in zip(OpponentPool.NAMES, rollout.opponent_counts)
            },
            "frozen_opponent_seat_fraction": float(
                rollout.opponent_counts[OpponentPool.FROZEN_TRANSFORMER]
                / max(rollout.opponent_counts.sum(), 1)
            ),
            "active_snapshot": rollout.active_snapshot,
            "snapshot_count": len(collector.pool.snapshots),
            "rule_first_rate": collector.pool.rule_first_rate,
            "rule_mix_streak": collector.pool.rule_mix_streak,
            "rule_league_streak": collector.pool.rule_league_streak,
            **rollout.cache_stats,
            **stats,
        }
        if update % args.eval_every == 0 or args.smoke:
            synchronize()
            evaluation_start = time.perf_counter()
            record["evaluation"] = evaluate_against_rules(
                model,
                device,
                games=args.eval_games,
                envs=min(config.envs, args.eval_games),
                seed=args.seed + update,
            )
            synchronize()
            record["evaluation_seconds"] = time.perf_counter() - evaluation_start
            collector.pool.update_rule_evaluation(record["evaluation"])
            record["rule_first_rate"] = collector.pool.rule_first_rate
            record["rule_mix_streak"] = collector.pool.rule_mix_streak
            record["rule_league_streak"] = collector.pool.rule_league_streak
            record["next_opponent_stage"] = collector.pool.stage()
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        elapsed_hours = elapsed / 3600.0
        for milestone in milestone_hours:
            if milestone in saved_milestones or elapsed_hours < milestone:
                continue
            save_checkpoint(
                args.output_dir / f"snapshot_{milestone}h.pt",
                model,
                optimizer,
                update,
                transitions,
                config,
                collector.pool,
            )
            saved_milestones.add(milestone)
        if update % args.checkpoint_every == 0 or transitions >= args.total_transitions:
            save_checkpoint(
                args.output_dir / "latest.pt",
                model,
                optimizer,
                update,
                transitions,
                config,
                collector.pool,
            )

    save_checkpoint(
        args.output_dir / "latest.pt",
        model,
        optimizer,
        update,
        transitions,
        config,
        collector.pool,
    )
    final = evaluate_against_rules(
        model,
        device,
        games=args.eval_games,
        envs=min(config.envs, args.eval_games),
        seed=args.seed + 0x10000,
    )
    print(
        json.dumps(
            {"final": final, "update": update, "transitions": transitions},
            ensure_ascii=False,
        )
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
