"""Fork a completed policy-iteration snapshot into an independent run."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .model import BloodFlowTransformer
from .pipeline import save_policy
from .train import (
    OpponentPool,
    RunConfig,
    SelfPlayCurriculum,
    _atomic_json,
    _checkpoint_payload,
    _ensure_run_identity,
    _load_checkpoint,
    _model_digest,
    _run_identity,
    _save_checkpoint,
    _sha256,
)


def _iteration_metric(checkpoint: Path, iteration: int) -> dict[str, object]:
    metrics_path = checkpoint.resolve().parent / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    matches: list[dict[str, object]] = []
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid metrics line {line_number} in {metrics_path}"
            ) from error
        if int(value.get("iteration", -1)) == iteration:
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"expected one metrics row for iteration {iteration}, found {len(matches)}"
        )
    return matches[0]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"iteration metrics have no valid {name}")
    return value


def fork_run(
    source_checkpoint: Path,
    output_dir: Path,
    *,
    iteration: int,
    anchor_rule_fast: bool,
    direction_optimizer: str | None = None,
    direction_learning_rate: float | None = None,
    direction_momentum: float | None = None,
    direction_gradient_clip_norm: float | None = None,
    preserve_direction_optimizer_state: bool = False,
    queries_per_category: int | None = None,
    worlds: int | None = None,
    target_kl: float | None = None,
    policy_objective: str | None = None,
    world_sampling: str | None = None,
    kl_control: str | None = None,
    split_consensus_margin: float | None = None,
    validation_worlds: int | None = None,
    audit_worlds: int | None = None,
    generation_batches: int | None = None,
    target_fdr: float | None = None,
    mirror_temperature: float | None = None,
    mirror_prior_floor: float | None = None,
    arena_games: int | None = None,
    envs: int | None = None,
    evaluation_envs: int | None = None,
    target_shard_size: int | None = None,
    target_query_batch_size: int | None = None,
) -> dict[str, object]:
    if iteration < 1:
        raise ValueError("fork iteration must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("fork output directory must be empty")

    source_checkpoint = source_checkpoint.resolve()
    source_actor, source_config, payload = _load_checkpoint(
        source_checkpoint,
        torch.device("cpu"),
        allow_policy_execution_mismatch=True,
    )
    current_champion = int(payload["champion_iteration"])
    if iteration > current_champion:
        raise ValueError(
            f"fork iteration {iteration} exceeds current champion {current_champion}"
        )
    source_pool = payload["opponent_pool"]
    if not isinstance(source_pool, OpponentPool):
        raise ValueError("source checkpoint has no historical opponent pool")
    snapshots = tuple(
        snapshot
        for snapshot in source_pool.snapshots
        if snapshot.iteration <= iteration
    )
    if not snapshots:
        raise ValueError("source checkpoint retains no snapshot for the fork iteration")

    if iteration == current_champion:
        fork_actor = source_actor
    else:
        candidates = [
            snapshot for snapshot in snapshots if snapshot.iteration == iteration
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"iteration {iteration} is not retained in the source opponent pool"
            )
        fork_actor = BloodFlowTransformer(source_actor.config)
        fork_actor.load_state_dict(candidates[0].actor, strict=True)
        fork_actor.eval()

    if iteration == current_champion:
        opponent_pool = OpponentPool(
            snapshots=snapshots,
            rotations=source_pool.rotations,
            last_refresh_iteration=source_pool.last_refresh_iteration,
        )
        source_self_play = payload["self_play"]
        if not isinstance(source_self_play, SelfPlayCurriculum):
            raise ValueError("source checkpoint has no self-play curriculum")
        self_play = source_self_play
    else:
        metric = _iteration_metric(source_checkpoint, iteration)
        if int(metric.get("policy_version_after", -1)) != iteration:
            raise ValueError("fork metrics policy version does not match the iteration")
        pool_metrics = _mapping(metric.get("opponent_pool"), "opponent_pool")
        expected_size = int(pool_metrics["next_size"])
        if len(snapshots) != expected_size:
            raise ValueError(
                "retained source snapshots do not match the iteration's committed pool"
            )
        opponent_pool = OpponentPool(
            snapshots=snapshots,
            rotations=int(pool_metrics["next_rotations"]),
            last_refresh_iteration=int(pool_metrics["next_last_refresh_iteration"]),
        )
        self_play_metrics = _mapping(metric.get("self_play"), "self_play")
        activation = self_play_metrics["activation_iteration"]
        self_play = SelfPlayCurriculum(
            fraction=float(self_play_metrics["next_fraction"]),
            last_fixed_first_rate=float(self_play_metrics["last_fixed_first_rate"]),
            activation_iteration=None if activation is None else int(activation),
        )
    config_changes: dict[str, object] = {
        "anchor_rule_fast": bool(anchor_rule_fast),
    }
    if direction_optimizer is not None:
        config_changes["direction_optimizer"] = direction_optimizer
    if direction_learning_rate is not None:
        config_changes["direction_learning_rate"] = direction_learning_rate
    if direction_momentum is not None:
        config_changes["direction_momentum"] = direction_momentum
    if direction_gradient_clip_norm is not None:
        config_changes["direction_gradient_clip_norm"] = (
            direction_gradient_clip_norm
        )
    for name, value in (
        ("queries_per_category", queries_per_category),
        ("worlds", worlds),
        ("target_kl", target_kl),
        ("policy_objective", policy_objective),
        ("world_sampling", world_sampling),
        ("kl_control", kl_control),
        ("split_consensus_margin", split_consensus_margin),
        ("validation_worlds", validation_worlds),
        ("audit_worlds", audit_worlds),
        ("generation_batches", generation_batches),
        ("target_fdr", target_fdr),
        ("mirror_temperature", mirror_temperature),
        ("mirror_prior_floor", mirror_prior_floor),
        ("arena_games", arena_games),
        ("envs", envs),
        ("evaluation_envs", evaluation_envs),
        ("target_shard_size", target_shard_size),
        ("target_query_batch_size", target_query_batch_size),
    ):
        if value is not None:
            config_changes[name] = value
    config: RunConfig = replace(source_config, **config_changes)
    direction_optimizer_state: Mapping[str, torch.Tensor] = {}
    if preserve_direction_optimizer_state:
        if config.policy_objective == "rank_lcb_mirror_ce":
            raise ValueError("rank-LCB forks must reset generation-local state")
        if iteration != current_champion:
            raise ValueError(
                "optimizer state can only be preserved from the current champion"
            )
        if (
            source_config.direction_optimizer != config.direction_optimizer
            or config.direction_optimizer not in {"momentum", "nesterov"}
        ):
            raise ValueError(
                "optimizer state preservation needs the same stateful optimizer"
            )
        raw_state = payload.get("direction_optimizer_state")
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("source checkpoint has no direction optimizer state")
        direction_optimizer_state = {
            str(name): value.detach().cpu().clone()
            for name, value in raw_state.items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = output_dir / f"reference-u{iteration:03d}.pt"
    save_policy(reference, fork_actor)
    reference = reference.resolve()
    reference_sha256 = _sha256(reference)
    root_seed = int(payload["root_seed"])
    next_iteration = (
        int(payload["next_iteration"])
        if iteration == current_champion
        else iteration + 1
    )
    _save_checkpoint(
        output_dir / "latest.pt",
        _checkpoint_payload(
            fork_actor,
            config=config,
            root_seed=root_seed,
            next_iteration=next_iteration,
            champion_iteration=iteration,
            sl_checkpoint=reference,
            sl_sha256=reference_sha256,
            self_play=self_play,
            opponent_pool=opponent_pool,
            last_metrics=None,
            direction_optimizer_state=direction_optimizer_state,
        ),
    )
    _ensure_run_identity(
        output_dir / "config.json",
        _run_identity(
            config=config,
            root_seed=root_seed,
            sl_checkpoint=reference,
            sl_sha256=reference_sha256,
        ),
    )
    save_policy(output_dir / "actor.pt", fork_actor)

    provenance: dict[str, object] = {
        "version": 1,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "source_iteration": iteration,
        "source_actor_digest": _model_digest(fork_actor),
        "next_iteration": next_iteration,
        "root_seed": root_seed,
        "anchor_rule_fast": anchor_rule_fast,
        "optimizer": {
            "direction_optimizer": config.direction_optimizer,
            "direction_learning_rate": config.direction_learning_rate,
            "direction_momentum": config.direction_momentum,
            "direction_gradient_clip_norm": (
                config.direction_gradient_clip_norm
            ),
            "state_reset": not preserve_direction_optimizer_state,
            "state_preserved": preserve_direction_optimizer_state,
        },
        "policy_update": {
            "queries_per_category": config.queries_per_category,
            "worlds": config.worlds,
            "policy_objective": config.policy_objective,
            "world_sampling": config.world_sampling,
            "split_consensus_margin": config.split_consensus_margin,
            "kl_control": config.kl_control,
            "target_kl": config.target_kl,
            "validation_worlds": config.validation_worlds,
            "audit_worlds": config.audit_worlds,
            "generation_batches": config.generation_batches,
            "target_fdr": config.target_fdr,
            "mirror_temperature": config.mirror_temperature,
            "mirror_prior_floor": config.mirror_prior_floor,
            "arena_games": config.arena_games,
            "envs": config.envs,
            "evaluation_envs": config.evaluation_envs,
            "target_shard_size": config.target_shard_size,
            "target_query_batch_size": config.target_query_batch_size,
        },
        "self_play": {
            "fraction": self_play.fraction,
            "activation_iteration": self_play.activation_iteration,
        },
        "opponent_pool": {
            "iterations": [snapshot.iteration for snapshot in snapshots],
            "rotations": opponent_pool.rotations,
            "last_refresh_iteration": opponent_pool.last_refresh_iteration,
        },
    }
    _atomic_json(output_dir / "fork.json", provenance)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-rule-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--direction-optimizer",
        choices=("adamw", "sgd", "momentum", "nesterov"),
    )
    parser.add_argument("--direction-learning-rate", type=float)
    parser.add_argument("--direction-momentum", type=float)
    parser.add_argument("--direction-gradient-clip-norm", type=float)
    parser.add_argument(
        "--preserve-direction-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--queries-per-category", type=int)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument(
        "--policy-objective",
        choices=(
            "expected_q",
            "holdout_consensus_ce",
            "split_consensus_ce",
            "rank_lcb_mirror_ce",
        ),
    )
    parser.add_argument(
        "--world-sampling", choices=("live_wall", "information_set")
    )
    parser.add_argument("--kl-control", choices=("target", "cap"))
    parser.add_argument("--split-consensus-margin", type=float)
    parser.add_argument("--validation-worlds", type=int)
    parser.add_argument("--audit-worlds", type=int)
    parser.add_argument("--generation-batches", type=int)
    parser.add_argument("--target-fdr", type=float)
    parser.add_argument("--mirror-temperature", type=float)
    parser.add_argument("--mirror-prior-floor", type=float)
    parser.add_argument("--arena-games", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--evaluation-envs", type=int)
    parser.add_argument("--target-shard-size", type=int)
    parser.add_argument("--target-query-batch-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    provenance = fork_run(
        args.source_checkpoint,
        args.output_dir,
        iteration=args.iteration,
        anchor_rule_fast=args.anchor_rule_fast,
        direction_optimizer=args.direction_optimizer,
        direction_learning_rate=args.direction_learning_rate,
        direction_momentum=args.direction_momentum,
        direction_gradient_clip_norm=args.direction_gradient_clip_norm,
        preserve_direction_optimizer_state=(
            args.preserve_direction_optimizer_state
        ),
        queries_per_category=args.queries_per_category,
        worlds=args.worlds,
        target_kl=args.target_kl,
        policy_objective=args.policy_objective,
        world_sampling=args.world_sampling,
        kl_control=args.kl_control,
        split_consensus_margin=args.split_consensus_margin,
        validation_worlds=args.validation_worlds,
        audit_worlds=args.audit_worlds,
        generation_batches=args.generation_batches,
        target_fdr=args.target_fdr,
        mirror_temperature=args.mirror_temperature,
        mirror_prior_floor=args.mirror_prior_floor,
        arena_games=args.arena_games,
        envs=args.envs,
        evaluation_envs=args.evaluation_envs,
        target_shard_size=args.target_shard_size,
        target_query_batch_size=args.target_query_batch_size,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    print(
        "resume with: python -m training.train "
        f"--resume {args.output_dir / 'latest.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
