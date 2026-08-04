"""Wall-clock PPO training from a random or supervised Actor."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import bloodflow_mahjong as bm
import numpy as np
import torch

from .contracts import TRAINING_INPUT_SCHEMA, validate_engine_contract
from .model import BloodFlowTransformer
from .pipeline import (
    OpponentPool,
    PPOConfig,
    RolloutCollector,
    checkpoint_configs,
    cosine_learning_rate,
    evaluate_against_rule_ev,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from .policy import load_actor_checkpoint
from .reporting import append_jsonl, format_duration, format_percent, format_rate


_PERIODIC_EVALUATION_SEED = 0xE000
_FINAL_EVALUATION_SEED = 0x1E000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--microbatch",
        type=int,
        help="PPO forward/backward batch size",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        default=None,
        help="enable gated fixed-ratio frozen-policy opponents",
    )
    parser.add_argument(
        "--self-play-fraction",
        type=float,
        help="fraction of non-learner seats controlled by frozen policies",
    )
    parser.add_argument(
        "--historical-snapshot-probability",
        type=float,
        help="chance to use retained history instead of the newest snapshot",
    )
    parser.add_argument(
        "--opponent-refresh-updates",
        type=int,
        help="PPO updates between frozen snapshot creations",
    )
    parser.add_argument(
        "--frozen-snapshot-limit",
        type=int,
        help="maximum retained frozen policies",
    )
    parser.add_argument(
        "--score-reward-weight",
        type=float,
        help="weight for score deltas divided by 10,000 (default: 1)",
    )
    parser.add_argument(
        "--rank-reward-weight",
        type=float,
        help="weight for terminal rank utility (default: 1)",
    )
    parser.add_argument(
        "--kl-control",
        choices=("off", "monitor", "rollback"),
        help="measure KL only by default; rollback is an explicit experiment",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        help="categorical KL threshold used by rollback mode",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path)
    checkpoint.add_argument(
        "--fork",
        type=Path,
        help="branch complete PPO state into a new opponent curriculum",
    )
    checkpoint.add_argument(
        "--init-actor",
        type=Path,
        help="initialize the policy path from a supervised Actor checkpoint",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="small CPU/GPU end-to-end check"
    )
    return parser


def _optimizer(
    model: BloodFlowTransformer,
    config: PPOConfig,
    device: torch.device,
) -> torch.optim.AdamW:
    options: dict[str, Any] = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-5,
    }
    if device.type == "cuda":
        options["fused"] = True
    return torch.optim.AdamW(model.parameters(), **options)


def _compact_evaluation(evaluation: dict[str, Any]) -> str:
    return (
        f"rank {float(evaluation['mean_rank']):.2f}"
        f"  score {float(evaluation['mean_score_delta']):+,.0f}"
        f"  first {format_percent(evaluation['first_rate'])}"
        f"  last {format_percent(evaluation['last_rate'])}"
    )


def _compact_opponents(record: dict[str, Any]) -> str:
    assignments = record.get("opponent_assignments")
    if not isinstance(assignments, dict):
        return ""
    fast = int(assignments.get("rule_fast", 0))
    ev = int(assignments.get("rule_ev", 0))
    frozen = int(assignments.get("frozen_transformer", 0))
    total = fast + ev + frozen
    if total <= 0:
        return ""
    if frozen:
        return (
            "opp fast/EV/self "
            f"{format_percent(fast / total)}/"
            f"{format_percent(ev / total)}/"
            f"{format_percent(frozen / total)}"
        )
    return (
        "opp fast/EV "
        f"{format_percent(fast / total)}/{format_percent(ev / total)}"
    )


def _compact_record(record: dict[str, Any]) -> str:
    phase = str(record.get("phase", "event"))
    if phase == "ppo_start":
        evaluation = record["evaluation"]
        assert isinstance(evaluation, dict)
        return (
            f"BASE EV  {_compact_evaluation(evaluation)}"
            f"  games {int(evaluation['games']):,}"
            "  opp fast/EV 33.3%/66.7%"
            f"  time {format_duration(float(record['evaluation_seconds']))}"
        )
    if phase == "resume":
        return (
            f"RESUME  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  elapsed {format_duration(float(record['previous_run_elapsed_seconds']))}"
            f"/{float(record['target_hours']):g}h"
        )
    if phase == "fork":
        evaluation = record["evaluation"]
        assert isinstance(evaluation, dict)
        return (
            f"FORK  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  elapsed {format_duration(float(record['previous_run_elapsed_seconds']))}"
            f"/{float(record['target_hours']):g}h"
            f"  EV {_compact_evaluation(evaluation)}"
        )
    if phase == "ppo":
        kl_summary = (
            "  KL off"
            if record.get("kl_control") == "off"
            else f"  KL {float(record['approx_kl']):+.5f}"
        )
        message = (
            f"PPO u{int(record['update']):>5}"
            f"  {format_duration(float(record['ppo_elapsed_seconds']))}"
            f"/{float(record['target_hours']):g}h"
            f"  {int(record['transitions']):,} states"
            f"  {format_rate(float(record['rollout_states_per_second']), 'states')}"
            f"  pi {float(record['policy_loss']):+.4f}"
            f"  value {float(record['value_loss']):.3f}"
            f"  ent {float(record['entropy']):.3f}"
            f"{kl_summary}"
            f"  lr {float(record['learning_rate']):.2e}"
        )
        if int(record.get("rolled_back_epochs", 0)):
            message += f"  rollback {int(record['rolled_back_epochs'])} epoch"
        opponents = _compact_opponents(record)
        if opponents:
            message += f"  {opponents}"
        evaluation = record.get("evaluation")
        if isinstance(evaluation, dict):
            message += f"  EV {_compact_evaluation(evaluation)}"
        return message
    if phase == "complete":
        evaluation = record["final"]
        assert isinstance(evaluation, dict)
        return (
            f"DONE  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  {format_duration(float(record['ppo_elapsed_seconds']))}"
            f"/{float(record['target_hours']):g}h"
            f"  EV {_compact_evaluation(evaluation)}"
            f"  games {int(evaluation['games']):,}"
        )
    return phase.upper()


def _append_record(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)
    print(_compact_record(record), flush=True)


def _requested_overrides(
    args: argparse.Namespace,
) -> dict[str, tuple[str, Any]]:
    return {
        "--self-play": ("self_play_enabled", args.self_play),
        "--self-play-fraction": (
            "self_play_fraction",
            args.self_play_fraction,
        ),
        "--historical-snapshot-probability": (
            "historical_snapshot_probability",
            args.historical_snapshot_probability,
        ),
        "--opponent-refresh-updates": (
            "opponent_refresh_updates",
            args.opponent_refresh_updates,
        ),
        "--frozen-snapshot-limit": (
            "frozen_snapshot_limit",
            args.frozen_snapshot_limit,
        ),
        "--microbatch": ("microbatch", args.microbatch),
        "--score-reward-weight": (
            "score_reward_weight",
            args.score_reward_weight,
        ),
        "--rank-reward-weight": (
            "rank_reward_weight",
            args.rank_reward_weight,
        ),
        "--kl-control": ("kl_control", args.kl_control),
        "--target-kl": ("target_kl", args.target_kl),
    }


def _fresh_config(args: argparse.Namespace) -> PPOConfig:
    overrides = {
        name: value
        for name, value in _requested_overrides(args).values()
        if value is not None
    }
    return replace(
        PPOConfig(),
        **overrides,
    )


def _check_resume_overrides(args: argparse.Namespace, config: PPOConfig) -> None:
    for option, (name, requested) in _requested_overrides(args).items():
        checkpoint_value = getattr(config, name)
        if requested is not None and requested != checkpoint_value:
            raise ValueError(
                f"{option} cannot override the resumed checkpoint value "
                f"{checkpoint_value!r}"
            )


_FORKABLE_CONFIG = {
    "self_play_enabled",
    "self_play_fraction",
    "historical_snapshot_probability",
    "opponent_refresh_updates",
    "frozen_snapshot_limit",
}


def _fork_config(args: argparse.Namespace, source: PPOConfig) -> PPOConfig:
    overrides: dict[str, Any] = {}
    for option, (name, requested) in _requested_overrides(args).items():
        if requested is None:
            continue
        source_value = getattr(source, name)
        if name not in _FORKABLE_CONFIG and requested != source_value:
            raise ValueError(
                f"{option} cannot override the forked checkpoint value "
                f"{source_value!r}"
            )
        if name in _FORKABLE_CONFIG:
            overrides[name] = requested
    return replace(source, **overrides)


def run(args: argparse.Namespace) -> None:
    validate_engine_contract()
    if args.hours <= 0.0:
        raise ValueError("--hours must be positive")
    if args.eval_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("evaluation and checkpoint intervals must be positive")
    if args.eval_games <= 0 or args.eval_games % 4 != 0:
        raise ValueError("--eval-games must be a positive multiple of four")
    checkpoint_path = args.resume if args.resume is not None else args.fork
    if args.smoke and checkpoint_path is not None:
        raise ValueError("--smoke cannot be combined with --resume or --fork")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    restored_model_config = None
    source_config = None
    if checkpoint_path is not None:
        restored_model_config, source_config = checkpoint_configs(checkpoint_path)
    if args.resume is not None:
        if source_config is None:
            raise RuntimeError("resumed run has no PPO configuration")
        config = source_config
        _check_resume_overrides(args, config)
    elif args.fork is not None:
        if source_config is None:
            raise RuntimeError("forked run has no PPO configuration")
        config = _fork_config(args, source_config)
    else:
        config = _fresh_config(args)
    smoke_updates = 0
    if args.smoke:
        config = replace(
            config,
            envs=8,
            rollout_transitions=64,
            ppo_epochs=1,
            minibatch=32,
            microbatch=16,
            opponent_refresh_updates=2,
            self_play_enabled=False,
            schedule_hours=0.02,
        )
        args.hours = 0.02
        args.eval_every = 1
        args.eval_games = 8
        args.checkpoint_every = 1
        smoke_updates = 2
    if checkpoint_path is None and args.microbatch is not None:
        if args.microbatch <= 0:
            raise ValueError("--microbatch must be positive")

    if checkpoint_path is not None:
        if restored_model_config is None:
            raise RuntimeError("restored run has no model configuration")
        model = BloodFlowTransformer(restored_model_config).to(device)
    elif args.init_actor is not None:
        model = load_actor_checkpoint(args.init_actor, device)
    else:
        model = BloodFlowTransformer().to(device)
    optimizer = _optimizer(model, config, device)
    collector = RolloutCollector(config, device, seed=args.seed + 1)

    if args.output_dir is None:
        if args.resume is not None:
            args.output_dir = args.resume.parent
        elif args.fork is not None:
            raise ValueError("--fork requires a new --output-dir")
        else:
            args.output_dir = Path("runs/transformer")
    if args.resume is not None and args.output_dir.resolve() != args.resume.parent.resolve():
        raise ValueError("a resumed run must use the checkpoint directory")
    if args.fork is not None and args.output_dir.resolve() == args.fork.parent.resolve():
        raise ValueError("a fork must use a new output directory")

    update = 0
    transitions = 0
    previous_ppo_seconds = 0.0
    if checkpoint_path is not None:
        update, transitions, previous_ppo_seconds = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device,
            config,
            collector,
            expected_checkpoint_config=source_config,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.resume is None:
        existing = list(args.output_dir.iterdir())
        if existing:
            raise FileExistsError(f"new output directory is not empty: {args.output_dir}")
        config_path.write_text(
            json.dumps(
                {
                    "ppo": asdict(config),
                    "model": asdict(model.config),
                    "training_input_schema": TRAINING_INPUT_SCHEMA,
                    "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
                    "args": vars(args)
                    | {
                        "output_dir": str(args.output_dir),
                        "resume": None,
                        "fork": str(args.fork) if args.fork is not None else None,
                        "init_actor": (
                            str(args.init_actor) if args.init_actor is not None else None
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif not config_path.exists():
        raise FileNotFoundError(f"resumed run has no {config_path}")

    milestone_hours = (2, 6, 12, 24)
    saved_milestones = {
        hour
        for hour in milestone_hours
        if (args.output_dir / f"snapshot_{hour}h.pt").exists()
        or previous_ppo_seconds >= hour * 3600.0
    }
    initial_transitions = transitions
    initial_update = update
    session_start = time.monotonic()
    budget_seconds = args.hours * 3600.0
    schedule_seconds = config.schedule_hours * 3600.0

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def elapsed_seconds() -> float:
        return previous_ppo_seconds + (time.monotonic() - session_start)

    if args.resume is not None:
        _append_record(
            metrics_path,
            {
                "phase": "resume",
                "checkpoint": str(args.resume),
                "update": update,
                "transitions": transitions,
                "previous_run_elapsed_seconds": previous_ppo_seconds,
                "target_hours": args.hours,
                "schedule_hours": config.schedule_hours,
                "eval_every": args.eval_every,
                "eval_games": args.eval_games,
                "checkpoint_every": args.checkpoint_every,
            },
        )
    elif args.fork is not None:
        synchronize()
        evaluation_start = time.perf_counter()
        baseline = evaluate_against_rule_ev(
            model,
            device,
            games=args.eval_games,
            envs=min(config.envs, args.eval_games),
            seed=_PERIODIC_EVALUATION_SEED,
        )
        synchronize()
        collector.pool.update_rule_evaluation(baseline)
        _append_record(
            metrics_path,
            {
                "phase": "fork",
                "checkpoint": str(args.fork),
                "update": update,
                "transitions": transitions,
                "previous_run_elapsed_seconds": previous_ppo_seconds,
                "target_hours": args.hours,
                "schedule_hours": config.schedule_hours,
                "evaluation": baseline,
                "evaluation_seconds": time.perf_counter() - evaluation_start,
            },
        )
    else:
        synchronize()
        evaluation_start = time.perf_counter()
        baseline = evaluate_against_rule_ev(
            model,
            device,
            games=args.eval_games,
            envs=min(config.envs, args.eval_games),
            seed=_PERIODIC_EVALUATION_SEED,
        )
        synchronize()
        _append_record(
            metrics_path,
            {
                "phase": "ppo_start",
                "update": update,
                "transitions": transitions,
                "target_hours": args.hours,
                "schedule_hours": config.schedule_hours,
                "evaluation": baseline,
                "evaluation_seconds": time.perf_counter() - evaluation_start,
            },
        )
        collector.pool.update_rule_evaluation(baseline)

    while elapsed_seconds() < budget_seconds:
        if smoke_updates and update - initial_update >= smoke_updates:
            break
        progress = min(elapsed_seconds() / schedule_seconds, 1.0)
        if config.self_play_enabled and collector.pool.gate_ready:
            if collector.pool.snapshot_due(update):
                collector.pool.refresh_snapshot(model, device, update=update)
            else:
                collector.pool.select_snapshot()

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
        update_start = time.perf_counter()
        statistics = ppo_update(model, optimizer, rollout, config, device, progress)
        synchronize()
        update_seconds = time.perf_counter() - update_start
        transitions += len(rollout)
        update += 1
        elapsed = elapsed_seconds()
        session_elapsed = time.monotonic() - session_start
        record: dict[str, Any] = {
            "phase": "ppo",
            "update": update,
            "transitions": transitions,
            "ppo_elapsed_seconds": elapsed,
            "target_hours": args.hours,
            "states_per_second": (transitions - initial_transitions)
            / max(session_elapsed, 1e-6),
            "rollout_seconds": rollout_seconds,
            "rollout_states_per_second": len(rollout)
            / max(rollout_seconds, 1e-6),
            "update_seconds": update_seconds,
            "learning_rate": learning_rate,
            "kl_control": config.kl_control,
            "reward_weights": {
                "score": config.score_reward_weight,
                "rank": config.rank_reward_weight,
            },
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
            "last_snapshot_update": collector.pool.last_snapshot_update,
            "rule_ev_score_delta": collector.pool.rule_score_delta,
            "rule_ev_first_rate": collector.pool.rule_first_rate,
            "rule_gate_streak": collector.pool.rule_gate_streak,
            **rollout.cache_stats,
            **statistics,
        }
        if update % args.eval_every == 0 or args.smoke:
            synchronize()
            evaluation_start = time.perf_counter()
            record["evaluation"] = evaluate_against_rule_ev(
                model,
                device,
                games=args.eval_games,
                envs=min(config.envs, args.eval_games),
                seed=_PERIODIC_EVALUATION_SEED,
            )
            synchronize()
            record["evaluation_seconds"] = time.perf_counter() - evaluation_start
            collector.pool.update_rule_evaluation(record["evaluation"])
            record["rule_ev_score_delta"] = collector.pool.rule_score_delta
            record["rule_ev_first_rate"] = collector.pool.rule_first_rate
            record["rule_gate_streak"] = collector.pool.rule_gate_streak
            record["next_opponent_stage"] = collector.pool.stage()
        _append_record(metrics_path, record)

        elapsed = elapsed_seconds()
        for milestone in milestone_hours:
            if milestone in saved_milestones or elapsed < milestone * 3600.0:
                continue
            save_checkpoint(
                args.output_dir / f"snapshot_{milestone}h.pt",
                model,
                optimizer,
                update,
                transitions,
                elapsed,
                config,
                collector,
            )
            saved_milestones.add(milestone)
        if update % args.checkpoint_every == 0:
            save_checkpoint(
                args.output_dir / "latest.pt",
                model,
                optimizer,
                update,
                transitions,
                elapsed,
                config,
                collector,
            )

    elapsed = elapsed_seconds()
    save_checkpoint(
        args.output_dir / "latest.pt",
        model,
        optimizer,
        update,
        transitions,
        elapsed,
        config,
        collector,
    )
    synchronize()
    evaluation_start = time.perf_counter()
    final = evaluate_against_rule_ev(
        model,
        device,
        games=args.eval_games,
        envs=min(config.envs, args.eval_games),
        seed=_FINAL_EVALUATION_SEED,
    )
    synchronize()
    _append_record(
        metrics_path,
        {
            "phase": "complete",
            "final": final,
            "evaluation_seconds": time.perf_counter() - evaluation_start,
            "update": update,
            "transitions": transitions,
            "ppo_elapsed_seconds": elapsed,
            "target_hours": args.hours,
            "schedule_hours": config.schedule_hours,
        },
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
