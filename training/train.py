"""Metric-controlled PPO training from a random or supervised Actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
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
    TrainingController,
    checkpoint_configs,
    evaluate_against_rule_ev,
    evaluate_against_rule_policy,
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
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        help="write a numbered checkpoint every N PPO updates",
    )
    parser.add_argument(
        "--microbatch",
        type=int,
        help="PPO forward/backward batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="initial learning rate; resets the rate when used with --fork",
    )
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        help="minimum learning rate allowed by the plateau controller",
    )
    parser.add_argument(
        "--learning-rate-decay",
        type=float,
        help="multiplicative decay applied after a learning-rate plateau",
    )
    parser.add_argument(
        "--learning-rate-patience-evaluations",
        type=int,
        help="evaluations without a metric improvement before decay",
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="finish after this many PPO updates in the current process",
    )
    parser.add_argument(
        "--self-play",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable the metric-gated frozen-policy curriculum (default: enabled)",
    )
    parser.add_argument(
        "--self-play-max-fraction",
        type=float,
        help="maximum fraction of opponent seats controlled by frozen policies",
    )
    parser.add_argument(
        "--self-play-fraction-step",
        type=float,
        help="frozen-opponent fraction added after each stable gate",
    )
    parser.add_argument(
        "--self-play-gate-score",
        type=float,
        help="required lower confidence bound for Rule-EV score delta",
    )
    parser.add_argument(
        "--self-play-gate-rank",
        type=float,
        help="required upper confidence bound for Rule-EV mean rank",
    )
    parser.add_argument(
        "--self-play-gate-window",
        type=int,
        help="number of independent Rule-EV evaluations combined by the gate",
    )
    parser.add_argument(
        "--historical-snapshot-probability",
        type=float,
        help="chance to use retained history instead of the newest snapshot",
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
    parser.add_argument(
        "--analysis-opponent",
        choices=("rule-fast", "rule-ev", "rule-nn"),
        default="rule-ev",
        help="additional human-facing evaluation opponent",
    )
    parser.add_argument(
        "--analysis-nn-model",
        type=Path,
        help="ONNX model used when --analysis-opponent=rule-nn",
    )
    parser.add_argument(
        "--analysis-games",
        type=int,
        help="games per human-facing evaluation (default: --eval-games)",
    )
    parser.add_argument(
        "--analysis-every",
        type=int,
        default=1,
        help="run human analysis every N internal gate evaluations",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path)
    checkpoint.add_argument(
        "--fork",
        type=Path,
        help="branch complete PPO state into a new run with controlled overrides",
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
        evaluation = record["gate_evaluation"]
        assert isinstance(evaluation, dict)
        return (
            f"BASE EV  {_compact_evaluation(evaluation)}"
            f"  games {int(evaluation['games']):,}"
            "  opp fast/EV 33.3%/66.7%"
            f"  time {format_duration(float(record['gate_evaluation_seconds']))}"
        )
    if phase == "resume":
        return (
            f"RESUME  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  elapsed {format_duration(float(record['previous_run_elapsed_seconds']))}"
        )
    if phase == "fork":
        evaluation = record["gate_evaluation"]
        assert isinstance(evaluation, dict)
        return (
            f"FORK  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  elapsed {format_duration(float(record['previous_run_elapsed_seconds']))}"
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
        evaluation = record.get("gate_evaluation")
        if isinstance(evaluation, dict):
            message += f"  EV {_compact_evaluation(evaluation)}"
        analysis = record.get("analysis_evaluation")
        if isinstance(analysis, dict) and analysis.get("opponent") != "rule-ev":
            label = str(analysis.get("opponent", "analysis")).upper()
            message += f"  {label} {_compact_evaluation(analysis)}"
        return message
    if phase == "complete":
        evaluation = record["gate_evaluation"]
        assert isinstance(evaluation, dict)
        return (
            f"DONE  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  {format_duration(float(record['ppo_elapsed_seconds']))}"
            f"  EV {_compact_evaluation(evaluation)}"
            f"  games {int(evaluation['games']):,}"
        )
    if phase == "interrupted":
        return (
            f"STOP  u{int(record['update']):>5}"
            f"  {int(record['transitions']):,} states"
            f"  elapsed {format_duration(float(record['ppo_elapsed_seconds']))}"
            f"  checkpoint {record['checkpoint']}"
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
        "--self-play-max-fraction": (
            "self_play_max_fraction",
            args.self_play_max_fraction,
        ),
        "--self-play-fraction-step": (
            "self_play_fraction_step",
            args.self_play_fraction_step,
        ),
        "--self-play-gate-score": (
            "self_play_gate_score_delta",
            args.self_play_gate_score,
        ),
        "--self-play-gate-rank": (
            "self_play_gate_mean_rank",
            args.self_play_gate_rank,
        ),
        "--self-play-gate-window": (
            "self_play_gate_window",
            args.self_play_gate_window,
        ),
        "--historical-snapshot-probability": (
            "historical_snapshot_probability",
            args.historical_snapshot_probability,
        ),
        "--frozen-snapshot-limit": (
            "frozen_snapshot_limit",
            args.frozen_snapshot_limit,
        ),
        "--microbatch": ("microbatch", args.microbatch),
        "--learning-rate": ("learning_rate", args.learning_rate),
        "--minimum-learning-rate": (
            "minimum_learning_rate",
            args.minimum_learning_rate,
        ),
        "--learning-rate-decay": (
            "learning_rate_decay",
            args.learning_rate_decay,
        ),
        "--learning-rate-patience-evaluations": (
            "learning_rate_patience_evaluations",
            args.learning_rate_patience_evaluations,
        ),
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


_LEARNING_RATE_CONFIG = frozenset(
    {
        "learning_rate",
        "minimum_learning_rate",
        "learning_rate_decay",
        "learning_rate_patience_evaluations",
    }
)

_FORKABLE_CONFIG = _LEARNING_RATE_CONFIG | {
    "self_play_enabled",
    "self_play_max_fraction",
    "self_play_fraction_step",
    "self_play_gate_score_delta",
    "self_play_gate_mean_rank",
    "self_play_gate_window",
    "historical_snapshot_probability",
    "frozen_snapshot_limit",
}


def _fork_config(args: argparse.Namespace, source: PPOConfig) -> PPOConfig:
    requested_learning_rate_config = {
        name
        for name, requested in _requested_overrides(args).values()
        if name in _LEARNING_RATE_CONFIG and requested is not None
    }
    if requested_learning_rate_config and args.learning_rate is None:
        raise ValueError(
            "forked learning-rate overrides require an explicit --learning-rate"
        )
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


def _serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def _load_analysis_policy(
    args: argparse.Namespace,
) -> tuple[Any | None, dict[str, str]]:
    if args.analysis_opponent != "rule-nn":
        if args.analysis_nn_model is not None:
            raise ValueError(
                "--analysis-nn-model requires --analysis-opponent=rule-nn"
            )
        return None, {}
    if args.analysis_nn_model is None:
        raise ValueError("rule-nn analysis requires --analysis-nn-model")
    if not args.analysis_nn_model.is_file():
        raise FileNotFoundError(args.analysis_nn_model)
    with args.analysis_nn_model.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    policy = bm.RuleNn.from_file(str(args.analysis_nn_model))
    return policy, {
        "model_path": str(args.analysis_nn_model),
        "model_sha256": digest,
    }


def _analysis_evaluation(
    model: BloodFlowTransformer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    seed: int,
    envs: int,
    gate_evaluation: dict[str, Any],
    rule_nn: Any | None,
    metadata: dict[str, str],
) -> dict[str, Any]:
    games = int(args.analysis_games)
    if args.analysis_opponent == "rule-ev" and games == args.eval_games:
        result = dict(gate_evaluation)
    else:
        result = evaluate_against_rule_policy(
            model,
            device,
            opponent=args.analysis_opponent,
            rule_nn=rule_nn,
            games=games,
            envs=min(envs, games),
            seed=seed,
        )
    return result | metadata


def run(args: argparse.Namespace) -> None:
    validate_engine_contract()
    if args.eval_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("evaluation and checkpoint intervals must be positive")
    if args.snapshot_every is not None and args.snapshot_every <= 0:
        raise ValueError("--snapshot-every must be positive")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        raise ValueError("--stop-after-updates must be positive")
    if args.eval_games <= 0 or args.eval_games % 4 != 0:
        raise ValueError("--eval-games must be a positive multiple of four")
    if args.analysis_every <= 0:
        raise ValueError("--analysis-every must be positive")
    checkpoint_path = args.resume if args.resume is not None else args.fork
    if args.smoke and checkpoint_path is not None:
        raise ValueError("--smoke cannot be combined with --resume or --fork")
    if args.smoke and args.stop_after_updates is not None:
        raise ValueError("--smoke cannot be combined with --stop-after-updates")

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
            self_play_enabled=False,
        )
        args.eval_every = 1
        args.eval_games = 8
        args.checkpoint_every = 1
        smoke_updates = 2
    if args.analysis_games is None:
        args.analysis_games = args.eval_games
    if args.analysis_games <= 0 or args.analysis_games % 4 != 0:
        raise ValueError("--analysis-games must be a positive multiple of four")
    if checkpoint_path is None and args.microbatch is not None:
        if args.microbatch <= 0:
            raise ValueError("--microbatch must be positive")

    analysis_policy, analysis_metadata = _load_analysis_policy(args)

    if checkpoint_path is not None:
        if restored_model_config is None:
            raise RuntimeError("restored run has no model configuration")
        model = BloodFlowTransformer(restored_model_config).to(device)
    elif args.init_actor is not None:
        model = load_actor_checkpoint(args.init_actor, device)
    else:
        model = BloodFlowTransformer().to(device)
    optimizer = _optimizer(model, config, device)
    controller = TrainingController(config, evaluation_seed=_PERIODIC_EVALUATION_SEED)
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
            controller,
            expected_checkpoint_config=source_config,
        )
        if args.fork is not None and args.learning_rate is not None:
            controller.reset_learning_rate_schedule()
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = controller.current_learning_rate

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
                    "args": _serialized_args(args)
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
    initial_transitions = transitions
    initial_update = update
    update_budget = smoke_updates or args.stop_after_updates or 0
    session_start = time.monotonic()
    stop_requested = False

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def elapsed_seconds() -> float:
        return previous_ppo_seconds + (time.monotonic() - session_start)

    def save(path: Path) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            update,
            transitions,
            elapsed_seconds(),
            config,
            collector,
            controller,
        )

    def apply_curriculum_decision(decision: str) -> None:
        if decision == "promote":
            collector.pool.promote(model, device, update=update)
            controller.mark_champion()
        elif decision == "refresh":
            collector.pool.refresh_snapshot(model, device, update=update)
            controller.mark_champion()
        elif decision == "demote":
            collector.pool.demote()

    def evaluate_gate_and_analysis(*, run_analysis: bool) -> dict[str, Any]:
        seed = controller.take_evaluation_seed(args.eval_games)
        synchronize()
        gate_start = time.perf_counter()
        gate = evaluate_against_rule_ev(
            model,
            device,
            games=args.eval_games,
            envs=min(config.envs, args.eval_games),
            seed=seed,
        )
        synchronize()
        result: dict[str, Any] = {
            "gate_evaluation": gate,
            "gate_evaluation_seed": seed,
            "gate_evaluation_seconds": time.perf_counter() - gate_start,
        }
        decision = controller.observe_rule_ev(
            gate,
            self_play_level=collector.pool.self_play_level,
            maximum_self_play_level=collector.pool.maximum_self_play_level,
        )
        apply_curriculum_decision(decision)
        result["curriculum_decision"] = decision
        result["gate_statistics"] = controller.gate_statistics()
        if run_analysis:
            synchronize()
            analysis_start = time.perf_counter()
            result["analysis_evaluation"] = _analysis_evaluation(
                model,
                device,
                args,
                seed=seed,
                envs=config.envs,
                gate_evaluation=gate,
                rule_nn=analysis_policy,
                metadata=analysis_metadata,
            )
            synchronize()
            result["analysis_evaluation_seconds"] = (
                time.perf_counter() - analysis_start
            )
        return result

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        if stop_requested:
            print("Stop already requested; waiting for the current update.", flush=True)
            return
        stop_requested = True
        print("Stop requested; saving after the current PPO update.", flush=True)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        if args.resume is not None:
            _append_record(
                metrics_path,
                {
                    "phase": "resume",
                    "checkpoint": str(args.resume),
                    "update": update,
                    "transitions": transitions,
                    "previous_run_elapsed_seconds": previous_ppo_seconds,
                    "eval_every": args.eval_every,
                    "eval_games": args.eval_games,
                    "checkpoint_every": args.checkpoint_every,
                    "analysis_opponent": args.analysis_opponent,
                    **analysis_metadata,
                },
            )
        else:
            baseline = evaluate_gate_and_analysis(run_analysis=True)
            phase = "fork" if args.fork is not None else "ppo_start"
            _append_record(
                metrics_path,
                {
                    "phase": phase,
                    "checkpoint": str(args.fork) if args.fork is not None else None,
                    "update": update,
                    "transitions": transitions,
                    "previous_run_elapsed_seconds": previous_ppo_seconds,
                    **baseline,
                },
            )

        while not stop_requested:
            if update_budget and update - initial_update >= update_budget:
                break
            if collector.pool.stage() == "self_play":
                collector.pool.select_snapshot()

            controls = controller.controls(collector.pool.self_play_level)

            synchronize()
            rollout_start = time.perf_counter()
            rollout = collector.collect(
                model,
                config.rollout_transitions,
                collect_auxiliary=controls.auxiliary_scale > 0.0,
            )
            synchronize()
            rollout_seconds = time.perf_counter() - rollout_start
            update_start = time.perf_counter()
            statistics = ppo_update(
                model, optimizer, rollout, config, device, controls
            )
            synchronize()
            update_seconds = time.perf_counter() - update_start
            if statistics["updates"] > 0.0:
                controller.observe_update(statistics)
            transitions += len(rollout)
            update += 1
            elapsed = elapsed_seconds()
            session_elapsed = time.monotonic() - session_start
            rollout_probabilities = collector.pool.probabilities()
            record: dict[str, Any] = {
                "phase": "ppo",
                "update": update,
                "transitions": transitions,
                "ppo_elapsed_seconds": elapsed,
                "states_per_second": (transitions - initial_transitions)
                / max(session_elapsed, 1e-6),
                "rollout_seconds": rollout_seconds,
                "rollout_states_per_second": len(rollout)
                / max(rollout_seconds, 1e-6),
                "update_seconds": update_seconds,
                "learning_rate": controls.learning_rate,
                "entropy_coefficient": controls.entropy_coefficient,
                "auxiliary_scale": controls.auxiliary_scale,
                "kl_control": config.kl_control,
                "reward_weights": {
                    "score": config.score_reward_weight,
                    "rank": config.rank_reward_weight,
                },
                "opponent_stage": rollout.opponent_stage,
                "self_play_level": collector.pool.self_play_level,
                "opponent_probabilities": {
                    name: float(probability)
                    for name, probability in zip(
                        OpponentPool.NAMES, rollout_probabilities, strict=True
                    )
                },
                "opponent_assignments": {
                    name: int(count)
                    for name, count in zip(
                        OpponentPool.NAMES, rollout.opponent_counts, strict=True
                    )
                },
                "frozen_opponent_seat_fraction": float(
                    rollout.opponent_counts[OpponentPool.FROZEN_TRANSFORMER]
                    / max(rollout.opponent_counts.sum(), 1)
                ),
                "active_snapshot": rollout.active_snapshot,
                "snapshot_count": len(collector.pool.snapshots),
                "last_snapshot_update": collector.pool.last_snapshot_update,
                "gate_statistics": controller.gate_statistics(),
                **rollout.cache_stats,
                **statistics,
            }
            if not stop_requested and (update % args.eval_every == 0 or args.smoke):
                next_evaluation = controller.evaluation_count + 1
                record |= evaluate_gate_and_analysis(
                    run_analysis=next_evaluation % args.analysis_every == 0
                )
            next_controls = controller.controls(collector.pool.self_play_level)
            next_probabilities = collector.pool.probabilities()
            record["next_learning_rate"] = next_controls.learning_rate
            record["next_entropy_coefficient"] = next_controls.entropy_coefficient
            record["next_auxiliary_scale"] = next_controls.auxiliary_scale
            record["next_self_play_level"] = collector.pool.self_play_level
            record["next_opponent_stage"] = collector.pool.stage()
            record["next_opponent_probabilities"] = {
                name: float(probability)
                for name, probability in zip(
                    OpponentPool.NAMES, next_probabilities, strict=True
                )
            }
            record["next_snapshot_count"] = len(collector.pool.snapshots)
            _append_record(metrics_path, record)

            if not stop_requested and update % args.checkpoint_every == 0:
                save(args.output_dir / "latest.pt")
            if (
                not stop_requested
                and args.snapshot_every is not None
                and update % args.snapshot_every == 0
            ):
                save(args.output_dir / f"snapshot_u{update}.pt")

        save(args.output_dir / "latest.pt")
        elapsed = elapsed_seconds()
        if stop_requested:
            _append_record(
                metrics_path,
                {
                    "phase": "interrupted",
                    "checkpoint": str(args.output_dir / "latest.pt"),
                    "update": update,
                    "transitions": transitions,
                    "ppo_elapsed_seconds": elapsed,
                },
            )
        else:
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
                    "gate_evaluation": final,
                    "gate_evaluation_seconds": time.perf_counter()
                    - evaluation_start,
                    "update": update,
                    "transitions": transitions,
                    "ppo_elapsed_seconds": elapsed,
                },
            )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
