"""Infinite CUDA conservative policy iteration for Blood Flow Mahjong."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bloodflow_mahjong as bm

from .evaluation import (
    collect_fixed_panel,
    collect_policy_panel,
    evaluation_seeds,
    load_reference_panel,
    outcomes,
    save_reference_panel,
    summarize_paired,
)
from .model import ACTION_SPACE_SIZE, BloodFlowTransformer, TransformerConfig
from .pipeline import (
    CollectionConfig,
    POLICY_EXECUTION_VERSION,
    TrajectoryCollector,
    clone_policy,
    load_policy,
    save_policy,
)
from .policy_iteration import (
    CounterfactualBatch,
    PolicyQuery,
    build_state_batch,
    cached_counterfactual_corpus,
    calibrate_direction,
    cap_direction,
    committed_optimizer_state,
    cpu_model_state,
    domain_seed,
    optimizer_direction,
    policy_change_metrics,
    policy_outputs,
    require_cuda,
    require_deterministic_actor,
    select_independent_queries,
    source_visit_frequencies,
)
from .progress import Progress
from .search_distillation import (
    PolicyImprovementTarget,
    build_holdout_win_target,
    build_rank_lcb_mirror_target,
    select_rank_lcb_challengers,
    build_split_win_consensus_target,
    summarize_greedy_rank_advantages,
)
from .world_outcomes import (
    WorldOutcomeBatch,
    cached_world_outcome_corpus,
    combine_world_replicates,
)


CHECKPOINT_VERSION = 6
MOMENTUM_CHECKPOINT_VERSION = 5
STATELESS_CHECKPOINT_VERSION = 4
LEGACY_CHECKPOINT_VERSION = 3
RUN_IDENTITY_VERSION = 3
SOURCE_DOMAIN = 0x7100_0001
SOURCE_QUERY_DOMAIN = 0x7100_0002
SOURCE_WORLD_DOMAIN = 0x7100_0003
SOURCE_WORLD_A_DOMAIN = 0x7100_0013
SOURCE_WORLD_B_DOMAIN = 0x7100_0014
SOURCE_WORLD_C_DOMAIN = 0x7100_0015
CALIBRATION_DOMAIN = 0x7200_0001
CALIBRATION_QUERY_DOMAIN = 0x7200_0002
EVALUATION_DOMAIN = 0x7300_0001
ARENA_FIXED_DOMAIN = 0x7500_0001
ARENA_HISTORY_DOMAIN = 0x7500_0002
ARENA_LINEUP_DOMAIN = 0x7500_0003
SELF_PLAY_RANK_DELTA_STEP = 0.01
OPPONENT_POOL_VERSION = 1
OPPONENT_POOL_CAPACITY = 4
OPPONENT_REFRESH_INTERVAL = 8
POLICY_OBJECTIVES = frozenset(
    {
        "expected_q",
        "holdout_consensus_ce",
        "split_consensus_ce",
        "rank_lcb_mirror_ce",
    }
)
WORLD_SAMPLING_MODES = frozenset({"live_wall", "information_set"})
KL_CONTROL_MODES = frozenset({"target", "cap"})


@dataclass(frozen=True)
class RunConfig:
    source_games: int = 4096
    calibration_source_games: int = 4096
    envs: int = 2048
    queries_per_category: int = 256
    calibration_queries_per_category: int = 64
    worlds: int = 16
    world_chunk: int = 64
    target_shard_size: int = 128
    target_query_batch_size: int = 128
    rollout_inference_batch_size: int = 128
    direction_learning_rate: float = 1e-5
    direction_optimizer: str = "adamw"
    direction_momentum: float = 0.9
    direction_gradient_clip_norm: float = 1.0
    microbatch_size: int = 64
    inference_batch_size: int = 128
    target_kl: float = 1e-3
    kl_search_steps: int = 18
    maximum_scale: float = 64.0
    policy_objective: str = "expected_q"
    world_sampling: str = "live_wall"
    kl_control: str = "target"
    split_consensus_margin: float = 0.125
    validation_worlds: int = 64
    audit_worlds: int = 32
    generation_batches: int = 4
    target_fdr: float = 0.05
    mirror_temperature: float = 0.05
    mirror_prior_floor: float = 1e-6
    arena_games: int = 65_536
    evaluation_games: int = 16_384
    evaluation_envs: int = 4096
    bootstrap_samples: int = 10_000
    # Serialized for v3 checkpoint identity compatibility; the paired gate ignores it.
    self_play_start_first_rate: float = 0.55
    self_play_increment: float = 0.10
    maximum_self_play_fraction: float = 2.0 / 3.0
    anchor_rule_fast: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.source_games,
            self.calibration_source_games,
            self.envs,
            self.queries_per_category,
            self.calibration_queries_per_category,
            self.worlds,
            self.world_chunk,
            self.target_shard_size,
            self.target_query_batch_size,
            self.rollout_inference_batch_size,
            self.direction_learning_rate,
            self.microbatch_size,
            self.inference_batch_size,
            self.target_kl,
            self.kl_search_steps,
            self.maximum_scale,
            self.evaluation_games,
            self.evaluation_envs,
            self.bootstrap_samples,
            self.validation_worlds,
            self.audit_worlds,
            self.generation_batches,
            self.target_fdr,
            self.mirror_temperature,
            self.arena_games,
            self.self_play_increment,
            self.maximum_self_play_fraction,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all training sizes and rates must be positive")
        required = 9 * self.queries_per_category
        calibration_required = 9 * self.calibration_queries_per_category
        if self.source_games < required:
            raise ValueError("source_games cannot cover independent train queries")
        if self.calibration_source_games < calibration_required:
            raise ValueError("calibration games cannot cover independent queries")
        if self.worlds < 2 or self.validation_worlds < 2 or self.audit_worlds < 2:
            raise ValueError("counterfactual targets need at least two worlds")
        if self.maximum_scale < 1:
            raise ValueError("maximum_scale must be at least one")
        if self.policy_objective not in POLICY_OBJECTIVES:
            raise ValueError("unsupported policy objective")
        if self.world_sampling not in WORLD_SAMPLING_MODES:
            raise ValueError("unsupported world sampling mode")
        if self.kl_control not in KL_CONTROL_MODES:
            raise ValueError("unsupported KL control mode")
        if not 0.0 <= self.split_consensus_margin < 0.5:
            raise ValueError("split consensus margin must be in [0, 0.5)")
        if not 0.0 < self.target_fdr < 1.0:
            raise ValueError("target FDR must be in (0, 1)")
        if not 0.0 <= self.mirror_prior_floor < 1.0:
            raise ValueError("mirror prior floor must be in [0, 1)")
        if self.direction_optimizer not in {"adamw", "sgd", "momentum", "nesterov"}:
            raise ValueError(
                "direction_optimizer must be one of adamw, sgd, momentum, nesterov"
            )
        if not 0.0 <= self.direction_momentum < 1.0:
            raise ValueError("direction_momentum must be in [0, 1)")
        if (
            not math.isfinite(self.direction_gradient_clip_norm)
            or self.direction_gradient_clip_norm < 0.0
        ):
            raise ValueError("direction_gradient_clip_norm must be non-negative")
        if self.direction_optimizer in {"momentum", "nesterov"} and not (
            self.direction_momentum > 0.0
        ):
            raise ValueError("stateful direction optimizers need positive momentum")
        if not 0.0 <= self.self_play_start_first_rate <= 1.0:
            raise ValueError("self-play start first rate must be in [0, 1]")
        if self.self_play_increment > self.maximum_self_play_fraction:
            raise ValueError("self-play increment cannot exceed its maximum fraction")
        if self.maximum_self_play_fraction > 2.0 / 3.0:
            raise ValueError("self-play must always leave at least one rule opponent")
        if not isinstance(self.anchor_rule_fast, bool):
            raise TypeError("anchor_rule_fast must be a bool")
        if self.policy_objective == "rank_lcb_mirror_ce":
            if self.direction_optimizer != "nesterov":
                raise ValueError("rank-LCB generations require Nesterov")
            if self.kl_control != "cap":
                raise ValueError("rank-LCB generations require a KL cap")
            if self.world_sampling != "live_wall":
                raise ValueError(
                    "rank-LCB generations require live-wall sampling until "
                    "history-posterior information sets are available"
                )


def _serialized_run_config(config: RunConfig) -> dict[str, object]:
    state = asdict(config)
    if not config.anchor_rule_fast:
        state.pop("anchor_rule_fast")
    # Keep the serialized default byte-for-byte compatible with existing
    # AdamW checkpoints and run identities.
    if config.direction_optimizer == "adamw":
        state.pop("direction_optimizer")
        state.pop("direction_momentum")
    elif config.direction_optimizer == "sgd":
        state.pop("direction_momentum")
    if config.direction_gradient_clip_norm == 1.0:
        state.pop("direction_gradient_clip_norm")
    if config.policy_objective == "expected_q":
        state.pop("policy_objective")
        state.pop("split_consensus_margin")
    elif config.policy_objective == "rank_lcb_mirror_ce":
        state.pop("split_consensus_margin")
    if config.policy_objective != "rank_lcb_mirror_ce":
        for name in (
            "validation_worlds",
            "audit_worlds",
            "generation_batches",
            "target_fdr",
            "mirror_temperature",
            "mirror_prior_floor",
            "arena_games",
        ):
            state.pop(name)
    if config.world_sampling == "live_wall":
        state.pop("world_sampling")
    if config.kl_control == "target":
        state.pop("kl_control")
    return state


@dataclass(frozen=True)
class SelfPlayCurriculum:
    fraction: float = 0.0
    last_fixed_first_rate: float | None = None
    activation_iteration: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 2.0 / 3.0:
            raise ValueError("self-play fraction must be in [0, 2/3]")
        if self.last_fixed_first_rate is not None and not (
            0.0 <= self.last_fixed_first_rate <= 1.0
        ):
            raise ValueError("fixed-rule first rate must be in [0, 1]")
        if self.activation_iteration is not None and self.activation_iteration < 1:
            raise ValueError("self-play activation iteration must be positive")
        if (self.fraction > 0.0) != (self.activation_iteration is not None):
            raise ValueError("self-play fraction and activation iteration disagree")


@dataclass(frozen=True)
class OpponentSnapshot:
    iteration: int
    digest: str
    actor: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("opponent snapshot iteration must be non-negative")
        if len(self.digest) != 64:
            raise ValueError("opponent snapshot digest must be a SHA-256 hex string")
        if not self.actor or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in self.actor.items()
        ):
            raise ValueError("opponent snapshot Actor state is invalid")


@dataclass(frozen=True)
class OpponentPool:
    snapshots: tuple[OpponentSnapshot, ...]
    rotations: int = 0
    last_refresh_iteration: int = 0

    def __post_init__(self) -> None:
        if not 1 <= len(self.snapshots) <= OPPONENT_POOL_CAPACITY:
            raise ValueError("opponent pool size is invalid")
        if self.rotations < 0 or self.last_refresh_iteration < 0:
            raise ValueError("opponent pool counters must be non-negative")
        digests = [snapshot.digest for snapshot in self.snapshots]
        if len(set(digests)) != len(digests):
            raise ValueError("opponent pool snapshots must have unique digests")


def _self_play_decision(
    evaluation: Mapping[str, object], config: RunConfig
) -> dict[str, float | bool]:
    try:
        actor = evaluation["actor"]
        rank = evaluation["paired_rank_delta"]
        score = evaluation["paired_score_delta"]
        if not isinstance(actor, Mapping):
            raise TypeError
        if not isinstance(rank, Mapping) or not isinstance(score, Mapping):
            raise TypeError
        fixed_first_rate = float(actor["first_rate"])
        rank_mean = float(rank["mean"])
        rank_ci95_high = float(rank["ci95_high"])
        score_ci95_high = float(score["ci95_high"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("self-play curriculum needs paired evaluation metrics") from error
    values = (fixed_first_rate, rank_mean, rank_ci95_high, score_ci95_high)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("self-play curriculum metrics must be finite")
    if not 0.0 <= fixed_first_rate <= 1.0:
        raise ValueError("fixed-rule first rate must be in [0, 1]")

    rank_gate_passed = rank_ci95_high < 0.0
    score_guard_passed = score_ci95_high >= 0.0
    evidence_passed = rank_gate_passed and score_guard_passed
    target_fraction = 0.0
    if evidence_passed:
        completed_milestones = math.floor(
            (-rank_mean + 1e-12) / SELF_PLAY_RANK_DELTA_STEP
        )
        tiers = max(1, 1 + completed_milestones)
        target_fraction = min(
            config.maximum_self_play_fraction,
            round(tiers * config.self_play_increment, 12),
        )
    return {
        "fixed_first_rate": fixed_first_rate,
        "rank_mean": rank_mean,
        "rank_ci95_high": rank_ci95_high,
        "score_ci95_high": score_ci95_high,
        "rank_gate_passed": rank_gate_passed,
        "score_guard_passed": score_guard_passed,
        "evidence_passed": evidence_passed,
        "target_fraction": target_fraction,
    }


def _advance_self_play(
    state: SelfPlayCurriculum,
    evaluation: Mapping[str, object],
    *,
    next_iteration: int,
    config: RunConfig,
) -> SelfPlayCurriculum:
    if next_iteration < 1:
        raise ValueError("next iteration must be positive")
    decision = _self_play_decision(evaluation, config)
    fraction = max(state.fraction, float(decision["target_fraction"]))
    activation = state.activation_iteration
    if state.fraction == 0.0 and fraction > 0.0:
        activation = next_iteration
    return SelfPlayCurriculum(
        fraction=fraction,
        last_fixed_first_rate=float(decision["fixed_first_rate"]),
        activation_iteration=activation,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_digest(model: BloodFlowTransformer) -> str:
    return _model_state_digest(model.state_dict())


def _opponent_snapshot(
    model: BloodFlowTransformer, *, iteration: int
) -> OpponentSnapshot:
    state = cpu_model_state(model)
    return OpponentSnapshot(
        iteration=iteration,
        digest=_model_state_digest(state),
        actor=state,
    )


def _initial_opponent_pool(model: BloodFlowTransformer) -> OpponentPool:
    return OpponentPool(snapshots=(_opponent_snapshot(model, iteration=0),))


def _select_opponent_snapshot(
    pool: OpponentPool, *, current_digest: str
) -> OpponentSnapshot:
    eligible = tuple(
        snapshot for snapshot in pool.snapshots if snapshot.digest != current_digest
    )
    if not eligible:
        raise RuntimeError("opponent pool has no snapshot distinct from the learner")
    return eligible[pool.rotations % len(eligible)]


def _load_opponent_actor(
    snapshot: OpponentSnapshot,
    config: TransformerConfig,
    device: torch.device,
) -> BloodFlowTransformer:
    if _model_state_digest(snapshot.actor) != snapshot.digest:
        raise ValueError("opponent snapshot digest does not match its Actor state")
    actor = BloodFlowTransformer(config).to(device)
    actor.load_state_dict(snapshot.actor, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _advance_opponent_pool(
    pool: OpponentPool,
    candidate: BloodFlowTransformer,
    *,
    completed_iteration: int,
    used_historical_opponent: bool,
    promoted: bool = True,
) -> OpponentPool:
    if completed_iteration < 1:
        raise ValueError("completed iteration must be positive")
    snapshots = list(pool.snapshots)
    last_refresh = pool.last_refresh_iteration
    if promoted and completed_iteration - last_refresh >= OPPONENT_REFRESH_INTERVAL:
        candidate_snapshot = _opponent_snapshot(
            candidate, iteration=completed_iteration
        )
        if all(
            snapshot.digest != candidate_snapshot.digest for snapshot in snapshots
        ):
            snapshots.append(candidate_snapshot)
            snapshots = snapshots[-OPPONENT_POOL_CAPACITY:]
        last_refresh = completed_iteration
    return OpponentPool(
        snapshots=tuple(snapshots),
        rotations=pool.rotations + int(used_historical_opponent),
        last_refresh_iteration=last_refresh,
    )


def _opponent_pool_payload(pool: OpponentPool) -> dict[str, object]:
    return {
        "version": OPPONENT_POOL_VERSION,
        "capacity": OPPONENT_POOL_CAPACITY,
        "refresh_interval": OPPONENT_REFRESH_INTERVAL,
        "rotations": pool.rotations,
        "last_refresh_iteration": pool.last_refresh_iteration,
        "snapshots": [
            {
                "iteration": snapshot.iteration,
                "digest": snapshot.digest,
                "actor": dict(snapshot.actor),
            }
            for snapshot in pool.snapshots
        ],
    }


def _load_opponent_pool(value: object) -> OpponentPool:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "capacity",
        "refresh_interval",
        "rotations",
        "last_refresh_iteration",
        "snapshots",
    }:
        raise ValueError("checkpoint opponent pool fields do not match")
    if int(value["version"]) != OPPONENT_POOL_VERSION:
        raise ValueError("unsupported checkpoint opponent pool version")
    if (
        int(value["capacity"]) != OPPONENT_POOL_CAPACITY
        or int(value["refresh_interval"]) != OPPONENT_REFRESH_INTERVAL
    ):
        raise ValueError("checkpoint opponent pool schedule does not match")
    raw_snapshots = value["snapshots"]
    if not isinstance(raw_snapshots, list):
        raise ValueError("checkpoint opponent snapshots must be a list")
    snapshots: list[OpponentSnapshot] = []
    for raw in raw_snapshots:
        if not isinstance(raw, dict) or set(raw) != {"iteration", "digest", "actor"}:
            raise ValueError("checkpoint opponent snapshot fields do not match")
        actor = raw["actor"]
        if not isinstance(actor, dict):
            raise ValueError("checkpoint opponent snapshot Actor is invalid")
        snapshot = OpponentSnapshot(
            iteration=int(raw["iteration"]),
            digest=str(raw["digest"]),
            actor=actor,
        )
        if _model_state_digest(snapshot.actor) != snapshot.digest:
            raise ValueError("checkpoint opponent snapshot digest does not match")
        snapshots.append(snapshot)
    return OpponentPool(
        snapshots=tuple(snapshots),
        rotations=int(value["rotations"]),
        last_refresh_iteration=int(value["last_refresh_iteration"]),
    )


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_metric(path: Path, value: dict[str, object]) -> None:
    existing: list[dict[str, object]] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metrics line {line_number}") from error
    iteration = int(value["iteration"])
    existing = [row for row in existing if int(row["iteration"]) < iteration]
    existing.append(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in existing)
    )
    temporary.replace(path)


def _run_identity(
    *,
    config: RunConfig,
    root_seed: int,
    sl_checkpoint: Path,
    sl_sha256: str,
) -> dict[str, object]:
    return {
        "version": RUN_IDENTITY_VERSION,
        "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
        "policy_execution_version": POLICY_EXECUTION_VERSION,
        "config": _serialized_run_config(config),
        "root_seed": int(root_seed),
        "sl_checkpoint": str(sl_checkpoint.resolve()),
        "sl_sha256": sl_sha256,
    }


def _ensure_run_identity(path: Path, identity: dict[str, object]) -> None:
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("run config.json does not match the checkpoint")
    _atomic_json(path, identity)


_PENDING_ITERATION = re.compile(r"iteration-(\d{6,})")


def _cleanup_committed_pending(output_dir: Path, next_iteration: int) -> None:
    root = output_dir / "pending"
    if not root.exists():
        return
    for path in root.iterdir():
        match = _PENDING_ITERATION.fullmatch(path.name)
        if path.is_dir() and match and int(match.group(1)) < next_iteration:
            shutil.rmtree(path)
    try:
        root.rmdir()
    except OSError:
        pass


def _committed_iteration(path: Path, fallback: int) -> int:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            isinstance(payload, dict)
            and int(payload.get("version", -1))
            in (
                LEGACY_CHECKPOINT_VERSION,
                STATELESS_CHECKPOINT_VERSION,
                MOMENTUM_CHECKPOINT_VERSION,
                CHECKPOINT_VERSION,
            )
            and "next_iteration" in payload
        ):
            return int(payload["next_iteration"]) - 1
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        pass
    return fallback


def _checkpoint_payload(
    actor: BloodFlowTransformer,
    *,
    config: RunConfig,
    root_seed: int,
    next_iteration: int,
    champion_iteration: int | None = None,
    sl_checkpoint: Path,
    sl_sha256: str,
    self_play: SelfPlayCurriculum,
    opponent_pool: OpponentPool,
    last_metrics: dict[str, object] | None,
    direction_optimizer_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    if champion_iteration is None:
        champion_iteration = next_iteration - 1
    if not 0 <= champion_iteration < next_iteration:
        raise ValueError("champion_iteration must precede next_iteration")
    optimizer_state = {
        name: value.detach().cpu()
        for name, value in (direction_optimizer_state or {}).items()
    }
    return {
        "version": CHECKPOINT_VERSION,
        "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
        "policy_execution_version": POLICY_EXECUTION_VERSION,
        "config": _serialized_run_config(config),
        "root_seed": int(root_seed),
        "next_iteration": int(next_iteration),
        "champion_iteration": int(champion_iteration),
        "sl_checkpoint": str(sl_checkpoint.resolve()),
        "sl_sha256": sl_sha256,
        "self_play": asdict(self_play),
        "opponent_pool": _opponent_pool_payload(opponent_pool),
        "model_config": actor.config.__dict__,
        "actor": cpu_model_state(actor),
        "direction_optimizer_state": optimizer_state,
        "last_metrics": last_metrics,
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    allow_policy_execution_mismatch: bool = False,
) -> tuple[BloodFlowTransformer, RunConfig, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    legacy_expected = {
        "version",
        "engine_rules_version",
        "policy_execution_version",
        "config",
        "root_seed",
        "next_iteration",
        "sl_checkpoint",
        "sl_sha256",
        "self_play",
        "model_config",
        "actor",
        "last_metrics",
    }
    version = int(payload.get("version", -1))
    stateless_expected = {*legacy_expected, "opponent_pool"}
    momentum_expected = {*stateless_expected, "direction_optimizer_state"}
    current_expected = {*momentum_expected, "champion_iteration"}
    if version == LEGACY_CHECKPOINT_VERSION:
        expected = legacy_expected
    elif version == STATELESS_CHECKPOINT_VERSION:
        expected = stateless_expected
    elif version == MOMENTUM_CHECKPOINT_VERSION:
        expected = momentum_expected
    elif version == CHECKPOINT_VERSION:
        expected = current_expected
    else:
        expected = set()
    if set(payload) != expected:
        raise ValueError(
            f"checkpoint is not a supported production format v{CHECKPOINT_VERSION}"
        )
    if int(payload["engine_rules_version"]) != int(bm.ENGINE_RULES_VERSION):
        raise ValueError("checkpoint engine rules version does not match")
    checkpoint_execution_version = int(payload["policy_execution_version"])
    if (
        checkpoint_execution_version != POLICY_EXECUTION_VERSION
        and not allow_policy_execution_mismatch
    ):
        raise ValueError(
            "checkpoint policy execution version "
            f"{checkpoint_execution_version} does not match trainer version "
            f"{POLICY_EXECUTION_VERSION}; migrate it into a new run with "
            "python -m training.fork_policy_iteration"
        )
    if int(payload["next_iteration"]) < 1:
        raise ValueError("checkpoint next_iteration must be positive")
    champion_iteration = int(
        payload.get("champion_iteration", int(payload["next_iteration"]) - 1)
    )
    if not 0 <= champion_iteration < int(payload["next_iteration"]):
        raise ValueError("checkpoint champion_iteration is invalid")
    payload["champion_iteration"] = champion_iteration
    last_metrics = payload["last_metrics"]
    if last_metrics is not None:
        if (
            not isinstance(last_metrics, dict)
            or int(last_metrics.get("iteration", -1))
            != int(payload["next_iteration"]) - 1
        ):
            raise ValueError("checkpoint last_metrics does not match next_iteration")
        if version == CHECKPOINT_VERSION and int(
            last_metrics.get("policy_version_after", -1)
        ) != champion_iteration:
            raise ValueError("checkpoint last_metrics does not match champion_iteration")
    config_state = dict(payload["config"])
    config_fields = {field.name for field in fields(RunConfig)}
    optional_defaults = {
        "anchor_rule_fast": False,
        "direction_optimizer": "adamw",
        "direction_momentum": 0.9,
        "direction_gradient_clip_norm": 1.0,
        "policy_objective": "expected_q",
        "world_sampling": "live_wall",
        "kl_control": "target",
        "split_consensus_margin": 0.125,
        "validation_worlds": 64,
        "audit_worlds": 32,
        "generation_batches": 4,
        "target_fdr": 0.05,
        "mirror_temperature": 0.05,
        "mirror_prior_floor": 1e-6,
        "arena_games": 65_536,
    }
    missing = config_fields - set(config_state)
    if set(config_state) - config_fields or not missing.issubset(optional_defaults):
        raise ValueError("checkpoint RunConfig fields do not match this trainer")
    for name in missing:
        config_state[name] = optional_defaults[name]
    config = RunConfig(**config_state)
    self_play_state = dict(payload["self_play"])
    if set(self_play_state) != {field.name for field in fields(SelfPlayCurriculum)}:
        raise ValueError("checkpoint self-play state fields do not match this trainer")
    self_play = SelfPlayCurriculum(**self_play_state)
    if self_play.fraction > config.maximum_self_play_fraction:
        raise ValueError("checkpoint self-play fraction exceeds the configured maximum")
    payload["self_play"] = self_play
    payload["opponent_pool"] = (
        None
        if version == LEGACY_CHECKPOINT_VERSION
        else _load_opponent_pool(payload["opponent_pool"])
    )
    raw_optimizer_state = (
        payload["direction_optimizer_state"]
        if version in {MOMENTUM_CHECKPOINT_VERSION, CHECKPOINT_VERSION}
        else {}
    )
    if not isinstance(raw_optimizer_state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in raw_optimizer_state.items()
    ):
        raise ValueError("checkpoint direction optimizer state is invalid")
    if config.direction_optimizer not in {"momentum", "nesterov"} and raw_optimizer_state:
        raise ValueError("stateless direction optimizer has checkpoint state")
    if config.policy_objective == "rank_lcb_mirror_ce" and raw_optimizer_state:
        raise ValueError("rank-LCB checkpoint cannot carry generation-local state")
    payload["direction_optimizer_state"] = raw_optimizer_state
    payload["legacy_self_play_opponent"] = version == LEGACY_CHECKPOINT_VERSION
    actor = BloodFlowTransformer(TransformerConfig(**payload["model_config"])).to(
        device
    )
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    if raw_optimizer_state:
        parameter_shapes = {
            name: parameter.shape for name, parameter in actor.named_parameters()
        }
        if set(raw_optimizer_state) != set(parameter_shapes) or any(
            value.shape != parameter_shapes[name]
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
            for name, value in raw_optimizer_state.items()
        ):
            raise ValueError("checkpoint direction optimizer state does not match Actor")
    return actor, config, payload


def _collection_progress(progress: Progress):
    def update(completed: int, states: int, elapsed: float) -> None:
        progress.update(
            completed,
            fields={
                "env_states": states,
                "env_states/s": states / max(elapsed, 1e-9),
            },
        )

    return update


def _collect_queries(
    actor: BloodFlowTransformer,
    self_play_actor: BloodFlowTransformer | None,
    device: torch.device,
    progress: Progress,
    *,
    games: int,
    envs: int,
    qpc: int,
    source_seed: int,
    query_seed: int,
    phase: str,
    self_play_fraction: float,
    opponent_snapshot: OpponentSnapshot | None,
    anchor_rule_fast: bool,
) -> tuple[list[Any], tuple[Any, ...], dict[str, object]]:
    progress.start(phase, total=games, unit="games")
    collector = TrajectoryCollector(
        CollectionConfig(envs=min(envs, games), history=actor.config.max_history),
        actor,
        device,
        seed=source_seed,
        self_play_fraction=self_play_fraction,
        self_play_actor=self_play_actor,
        anchor_rule_fast=anchor_rule_fast,
    )
    collection = collector.collect(
        games, on_progress=_collection_progress(progress)
    )
    progress.complete(
        fields={
            "env_states/s": collection.environment_steps
            / max(collection.elapsed_seconds, 1e-9)
        }
    )
    progress.start(f"{phase}_SELECT", total=9 * qpc, unit="states")
    queries = select_independent_queries(
        collection.trajectories,
        queries_per_category=qpc,
        seed=query_seed,
    )
    progress.complete(len(queries))
    return (
        queries,
        collection.trajectories,
        {
            "games": games,
            "environment_steps": collection.environment_steps,
            "elapsed_seconds": collection.elapsed_seconds,
            "environment_states_per_second": collection.environment_steps
            / max(collection.elapsed_seconds, 1e-9),
            "configured_self_play_fraction": self_play_fraction,
            "anchor_rule_fast": anchor_rule_fast,
            "self_play_opponent_iteration": (
                None if opponent_snapshot is None else opponent_snapshot.iteration
            ),
            "self_play_opponent_digest": (
                None if opponent_snapshot is None else opponent_snapshot.digest
            ),
            "source_counts": collection.source_counts,
            "opponent_seat_counts": collection.opponent_seat_counts,
            "actual_self_play_opponent_fraction": collection.opponent_seat_counts[
                "self_play"
            ]
            / max(sum(collection.opponent_seat_counts.values()), 1),
        },
    )


def _reference_panel(
    reference: BloodFlowTransformer | None,
    device: torch.device,
    output_dir: Path,
    progress: Progress,
    *,
    root_seed: int,
    config: RunConfig,
    sl_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds = evaluation_seeds(
        domain_seed(root_seed, EVALUATION_DOMAIN), config.evaluation_games
    )
    fingerprint = _json_fingerprint(
        {
            "sl_sha256": sl_sha256,
            "policy_execution_version": POLICY_EXECUTION_VERSION,
            "seed_root": int(domain_seed(root_seed, EVALUATION_DOMAIN)),
            "games": config.evaluation_games,
        }
    )
    path = output_dir / "reference_panel.npz"
    if path.exists():
        loaded_seeds, ranks, scores = load_reference_panel(
            path, fingerprint=fingerprint
        )
        if not np.array_equal(loaded_seeds, seeds):
            raise ValueError("reference panel seeds do not match")
        print(
            f"BASE  rank {ranks.mean():.4f}  score {scores.mean():+.0f}  "
            f"games {len(ranks):,}  cached",
            flush=True,
        )
        return seeds, ranks, scores

    if reference is None:
        raise RuntimeError("the SL reference model is required to rebuild its panel")

    progress.start("BASE_EVAL", total=len(seeds), unit="games")
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
    print(
        f"BASE  rank {ranks.mean():.4f}  score {scores.mean():+.0f}  "
        f"games {len(ranks):,}",
        flush=True,
    )
    return seeds, ranks, scores


@dataclass(frozen=True)
class _RankLcbOptimizationBatch:
    training: CounterfactualBatch
    target: PolicyImprovementTarget
    reference_actions: np.ndarray
    source_metrics: dict[str, object]
    target_metrics: dict[str, object]


@dataclass(frozen=True)
class _RankLcbAuditSpec:
    queries: tuple[PolicyQuery, ...]
    directory: Path
    fingerprint: str
    world_seed: int


@dataclass(frozen=True)
class _RankLcbGenerationBatch(_RankLcbOptimizationBatch):
    audit: _RankLcbAuditSpec


def _generation_seed(root: int, domain: int, attempt: int, batch: int) -> int:
    return domain_seed(domain_seed(root, domain, attempt), 0x6E00_0001, batch)


def _combine_visit_stats(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not values:
        raise ValueError("visit statistics must not be empty")
    counts = {
        name: sum(int(value["counts"][name]) for value in values)
        for name in next(iter(values))["counts"]
    }
    eligible = sum(int(value["eligible_multi_action_states"]) for value in values)
    current = sum(int(value["current_states"]) for value in values)
    fractions = {name: count / eligible for name, count in counts.items()}
    return {
        "games": sum(int(value["games"]) for value in values),
        "current_states": current,
        "eligible_multi_action_states": eligible,
        "excluded_single_action_states": current - eligible,
        "counts": counts,
        "fractions": fractions,
        "vector": list(fractions.values()),
    }


def _collect_rank_lcb_batches(
    champion: BloodFlowTransformer,
    self_play_actor: BloodFlowTransformer | None,
    device: torch.device,
    pending: Path,
    progress: Progress,
    *,
    attempt: int,
    root_seed: int,
    config: RunConfig,
    self_play: SelfPlayCurriculum,
    opponent_snapshot: OpponentSnapshot | None,
) -> tuple[list[_RankLcbGenerationBatch], set[int], dict[str, object]]:
    batches: list[_RankLcbGenerationBatch] = []
    training_seeds: set[int] = set()
    visit_batches: list[dict[str, object]] = []
    champion_digest = _model_digest(champion)
    for batch_index in range(config.generation_batches):
        label = f"U{attempt}_G{batch_index + 1}"
        phase_started = time.perf_counter()
        queries, trajectories, source_metrics = _collect_queries(
            champion,
            self_play_actor,
            device,
            progress,
            games=config.source_games,
            envs=config.envs,
            qpc=config.queries_per_category,
            source_seed=_generation_seed(
                root_seed, SOURCE_DOMAIN, attempt, batch_index
            ),
            query_seed=_generation_seed(
                root_seed, SOURCE_QUERY_DOMAIN, attempt, batch_index
            ),
            phase=f"{label}_SOURCE",
            self_play_fraction=self_play.fraction,
            opponent_snapshot=opponent_snapshot,
            anchor_rule_fast=config.anchor_rule_fast,
        )
        source_metrics["wall_seconds"] = time.perf_counter() - phase_started
        batch_seeds = {int(trajectory.seed) for trajectory in trajectories}
        if training_seeds & batch_seeds:
            raise RuntimeError("generation source batches share game seeds")
        training_seeds.update(batch_seeds)
        visit_batches.append(source_visit_frequencies(trajectories))
        identity = {
            "actor": champion_digest,
            "policy_execution_version": POLICY_EXECUTION_VERSION,
            "attempt": attempt,
            "generation_batch": batch_index,
            "policy_objective": config.policy_objective,
            "world_sampling": config.world_sampling,
            "self_play_fraction": self_play.fraction,
            "selection_worlds": config.worlds,
            "validation_worlds": config.validation_worlds,
            "audit_worlds": config.audit_worlds,
            "target_fdr": config.target_fdr,
            "mirror_temperature": config.mirror_temperature,
            "mirror_prior_floor": config.mirror_prior_floor,
            "self_play_opponent": (
                None if opponent_snapshot is None else opponent_snapshot.digest
            ),
        }
        progress.start(
            f"{label}_SELECTION",
            total=len(queries),
            unit="queries",
            fields={"worlds": config.worlds},
        )
        phase_started = time.perf_counter()
        selection, selection_metrics = cached_world_outcome_corpus(
            pending / f"generation-{batch_index:02d}" / "selection",
            queries,
            champion,
            device,
            self_play_actor=self_play_actor,
            fingerprint=_json_fingerprint(
                {
                    **identity,
                    "replicate": "selection",
                    "worlds": config.worlds,
                }
            ),
            worlds=config.worlds,
            world_chunk=config.world_chunk,
            world_seed=_generation_seed(
                root_seed, SOURCE_WORLD_A_DOMAIN, attempt, batch_index
            ),
            world_sampling=config.world_sampling,
            shard_size=config.target_shard_size,
            query_batch_size=config.target_query_batch_size,
            inference_batch_size=config.rollout_inference_batch_size,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        progress.complete()
        selection_metrics["wall_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        reference_probabilities, reference_actions = policy_outputs(
            champion,
            selection,
            device,
            batch_size=config.inference_batch_size,
        )
        selected_actions = select_rank_lcb_challengers(selection, reference_actions)
        validation_action_sets = tuple(
            np.asarray((reference, selected), dtype=np.uint8)
            for reference, selected in zip(reference_actions, selected_actions)
        )
        selection_policy_seconds = time.perf_counter() - phase_started
        progress.start(
            f"{label}_VALIDATION",
            total=len(queries),
            unit="queries",
            fields={"worlds": config.validation_worlds, "actions": 2},
        )
        phase_started = time.perf_counter()
        validation, validation_metrics = cached_world_outcome_corpus(
            pending / f"generation-{batch_index:02d}" / "validation-paired",
            queries,
            champion,
            device,
            self_play_actor=self_play_actor,
            fingerprint=_json_fingerprint(
                {
                    **identity,
                    "replicate": "validation_paired",
                    "worlds": config.validation_worlds,
                }
            ),
            worlds=config.validation_worlds,
            world_chunk=config.world_chunk,
            world_seed=_generation_seed(
                root_seed, SOURCE_WORLD_B_DOMAIN, attempt, batch_index
            ),
            world_sampling=config.world_sampling,
            shard_size=config.target_shard_size,
            query_batch_size=config.target_query_batch_size,
            inference_batch_size=config.rollout_inference_batch_size,
            action_sets=validation_action_sets,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        progress.complete()
        validation_metrics["wall_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        target, target_metrics = build_rank_lcb_mirror_target(
            selection,
            validation,
            reference_probabilities,
            reference_actions,
            fdr=config.target_fdr,
            temperature=config.mirror_temperature,
            prior_floor=config.mirror_prior_floor,
            selected_actions=selected_actions,
        )
        target_build_seconds = time.perf_counter() - phase_started
        target_metrics.update(
            {
                "generation_batch": batch_index,
                "world_sampling": config.world_sampling,
                "selection_policy_seconds": selection_policy_seconds,
                "target_build_seconds": target_build_seconds,
                "replicates": {
                    "selection": selection_metrics,
                    "validation": validation_metrics,
                    "audit": {
                        "worlds": config.audit_worlds,
                        "mode": "deferred_paired",
                    },
                },
            }
        )
        batches.append(
            _RankLcbGenerationBatch(
                training=selection.counterfactual_batch(),
                target=target,
                reference_actions=reference_actions,
                audit=_RankLcbAuditSpec(
                    queries=tuple(queries),
                    directory=(
                        pending
                        / f"generation-{batch_index:02d}"
                        / "audit-changed-paired"
                    ),
                    fingerprint=_json_fingerprint(
                        {
                            **identity,
                            "replicate": "audit_changed_paired",
                            "worlds": config.audit_worlds,
                        }
                    ),
                    world_seed=_generation_seed(
                        root_seed, SOURCE_WORLD_C_DOMAIN, attempt, batch_index
                    ),
                ),
                source_metrics=source_metrics,
                target_metrics=target_metrics,
            )
        )
    return batches, training_seeds, _combine_visit_stats(visit_batches)


def _optimizer_state_l2(state: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(
        sum(
            float(torch.sum(value.detach().double() * value.detach().double()))
            for value in state.values()
        )
    )


def _optimize_rank_lcb_generation(
    champion: BloodFlowTransformer,
    batches: Sequence[_RankLcbOptimizationBatch],
    calibration: Any,
    device: torch.device,
    progress: Progress,
    *,
    attempt: int,
    config: RunConfig,
    visit_weights: np.ndarray,
) -> tuple[BloodFlowTransformer, dict[str, object], dict[str, object]]:
    candidate = clone_policy(champion, device)
    champion_state = cpu_model_state(champion)
    velocity: dict[str, torch.Tensor] = {}
    optimizer_steps: list[dict[str, object]] = []
    kl_steps: list[dict[str, object]] = []
    for batch_index, batch in enumerate(batches):
        microbatches = math.ceil(len(batch.training) / config.microbatch_size)
        progress.start(
            f"U{attempt}_G{batch_index + 1}_ACTOR",
            total=microbatches,
            unit="microbatches",
        )
        phase_started = time.perf_counter()
        raw_candidate, step_initial, raw_state, optimizer_metrics = (
            optimizer_direction(
                candidate,
                batch.training,
                device,
                category_weights=visit_weights,
                learning_rate=config.direction_learning_rate,
                microbatch_size=config.microbatch_size,
                optimizer_name="nesterov",
                momentum=config.direction_momentum,
                gradient_clip_norm=(
                    None
                    if config.direction_gradient_clip_norm == 0.0
                    else config.direction_gradient_clip_norm
                ),
                optimizer_state=velocity,
                objective="search_ce",
                policy_targets=batch.target.distribution,
                policy_row_confidence=batch.target.row_confidence,
                on_progress=lambda done, values: progress.update(
                    done, fields=values
                ),
            )
        )
        progress.complete()
        optimizer_metrics["wall_seconds"] = time.perf_counter() - phase_started

        progress.start(
            f"U{attempt}_G{batch_index + 1}_CUMULATIVE_KL",
            total=config.kl_search_steps + 2,
            unit="evaluations",
        )
        phase_started = time.perf_counter()
        kl_metrics = cap_direction(
            raw_candidate,
            champion,
            champion_state,
            raw_state,
            calibration,
            device,
            category_weights=visit_weights,
            kl_cap=config.target_kl,
            batch_size=config.inference_batch_size,
            search_steps=config.kl_search_steps,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        progress.complete(
            int(kl_metrics["evaluations"]),
            fields={"scale": kl_metrics["scale"], "kl": kl_metrics["final_kl"]},
        )
        kl_metrics["wall_seconds"] = time.perf_counter() - phase_started
        velocity = committed_optimizer_state(
            "nesterov", step_initial, raw_candidate
        )
        optimizer_metrics["generation_batch"] = batch_index
        optimizer_metrics["committed_state_l2"] = _optimizer_state_l2(velocity)
        optimizer_steps.append(optimizer_metrics)
        kl_steps.append({"generation_batch": batch_index, **kl_metrics})
        candidate = raw_candidate

    change_metrics = policy_change_metrics(
        candidate,
        champion,
        calibration,
        device,
        category_weights=visit_weights,
        batch_size=config.inference_batch_size,
    )
    optimizer = {
        "optimizer": "nesterov",
        "momentum": config.direction_momentum,
        "generation_inner_steps": len(batches),
        "state_scope": "generation",
        "checkpoint_state_reset": True,
        "steps": optimizer_steps,
    }
    calibration_metrics = {
        **kl_steps[-1],
        "reference": "generation_champion",
        "cumulative_cap": config.target_kl,
        "steps": kl_steps,
        "policy_change": change_metrics,
    }
    return candidate, optimizer, calibration_metrics


def _rank_lcb_paired_action_sets(
    reference_actions: np.ndarray, candidate_actions: np.ndarray
) -> tuple[np.ndarray, ...]:
    reference_actions = np.asarray(reference_actions, dtype=np.int64)
    candidate_actions = np.asarray(candidate_actions, dtype=np.int64)
    if (
        reference_actions.shape != candidate_actions.shape
        or reference_actions.ndim != 1
        or np.any(
            (reference_actions < 0) | (reference_actions >= ACTION_SPACE_SIZE)
        )
        or np.any(
            (candidate_actions < 0) | (candidate_actions >= ACTION_SPACE_SIZE)
        )
    ):
        raise ValueError("rank-LCB audit actions are invalid")
    return tuple(
        np.asarray(
            (reference,)
            if reference == candidate
            else (reference, candidate),
            dtype=np.uint8,
        )
        for reference, candidate in zip(reference_actions, candidate_actions)
    )


def _collect_rank_lcb_audit(
    candidate: BloodFlowTransformer,
    champion: BloodFlowTransformer,
    batches: Sequence[_RankLcbGenerationBatch],
    self_play_actor: BloodFlowTransformer | None,
    device: torch.device,
    progress: Progress,
    *,
    attempt: int,
    config: RunConfig,
) -> dict[str, object]:
    row_advantages: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    flips: list[np.ndarray] = []
    audit_metrics: list[dict[str, object]] = []
    for batch_index, batch in enumerate(batches):
        phase_started = time.perf_counter()
        actions = policy_outputs(
            candidate,
            batch.training,
            device,
            batch_size=config.inference_batch_size,
        )[1]
        changed = actions != batch.reference_actions
        changed_indices = np.flatnonzero(changed)
        advantage = np.zeros(len(actions), dtype=np.float64)
        label = f"U{attempt}_G{batch_index + 1}_AUDIT"
        if len(changed_indices):
            changed_queries = tuple(
                batch.audit.queries[int(index)] for index in changed_indices
            )
            changed_reference = batch.reference_actions[changed_indices]
            changed_actions = actions[changed_indices]
            action_sets = _rank_lcb_paired_action_sets(
                changed_reference, changed_actions
            )
            progress.start(
                label,
                total=len(changed_queries),
                unit="queries",
                fields={
                    "worlds": config.audit_worlds,
                    "actions": 2,
                    "skipped": len(actions) - len(changed_queries),
                },
            )
            outcomes, metrics = cached_world_outcome_corpus(
                batch.audit.directory,
                changed_queries,
                champion,
                device,
                self_play_actor=self_play_actor,
                fingerprint=batch.audit.fingerprint,
                worlds=config.audit_worlds,
                world_chunk=config.world_chunk,
                world_seed=batch.audit.world_seed,
                world_sampling=config.world_sampling,
                shard_size=config.target_shard_size,
                query_batch_size=config.target_query_batch_size,
                inference_batch_size=config.rollout_inference_batch_size,
                action_sets=action_sets,
                on_progress=lambda done, values: progress.update(
                    done, fields=values
                ),
            )
            progress.complete()
            rows = np.arange(len(outcomes))
            utility = outcomes.rank_outcomes.astype(np.float64) / 2.0
            outcomes.require_evaluated(changed_reference)
            outcomes.require_evaluated(changed_actions)
            advantage[changed_indices] = (
                utility[rows, changed_actions]
                - utility[rows, changed_reference]
            ).mean(axis=1)
        else:
            metrics = {
                "states": 0,
                "worlds": config.audit_worlds,
                "world_sampling": config.world_sampling,
                "new_rollout_states": 0,
                "new_rollout_seconds": 0.0,
                "new_rollout_states_per_second": 0.0,
                "reused_prefix_queries": 0,
            }
        metrics["wall_seconds"] = time.perf_counter() - phase_started
        metrics["evaluated_states"] = int(len(changed_indices))
        metrics["skipped_unchanged_states"] = int(
            len(actions) - len(changed_indices)
        )
        metrics["total_states"] = int(len(actions))
        row_advantages.append(advantage)
        categories.append(batch.training.categories)
        flips.append(changed)
        audit_metrics.append(metrics)
    summary = summarize_greedy_rank_advantages(
        row_advantages, categories, flips
    )
    summary["replicates"] = audit_metrics
    summary["corpus"] = "deferred_sparse_paired"
    summary["evaluated_states"] = int(
        sum(np.count_nonzero(value) for value in flips)
    )
    summary["skipped_unchanged_states"] = int(
        summary["states"] - summary["evaluated_states"]
    )
    summary["wall_seconds"] = sum(
        float(metrics["wall_seconds"]) for metrics in audit_metrics
    )
    return summary


def _collect_arena_pair(
    candidate: BloodFlowTransformer,
    champion: BloodFlowTransformer,
    device: torch.device,
    progress: Progress,
    *,
    label: str,
    seeds: np.ndarray,
    envs: int,
    self_play_fraction: float,
    self_play_actor: BloodFlowTransformer | None,
    anchor_rule_fast: bool,
    lineup_seed: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    collected = []
    for policy_label, policy in (("CANDIDATE", candidate), ("CHAMPION", champion)):
        progress.start(
            f"{label}_{policy_label}", total=len(seeds), unit="games"
        )
        result = collect_policy_panel(
            policy,
            device,
            seeds,
            envs=envs,
            self_play_fraction=self_play_fraction,
            self_play_actor=self_play_actor,
            anchor_rule_fast=anchor_rule_fast,
            lineup_seed=lineup_seed,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        progress.complete()
        collected.append((result, *outcomes(result)))
    candidate_result, candidate_ranks, candidate_scores = collected[0]
    champion_result, champion_ranks, champion_scores = collected[1]
    if candidate_result.opponent_seat_counts != champion_result.opponent_seat_counts:
        raise RuntimeError("paired arena policies received different opponent lineups")
    summary = summarize_paired(
        candidate_ranks,
        candidate_scores,
        champion_ranks,
        champion_scores,
        seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    summary["opponent_seat_counts"] = candidate_result.opponent_seat_counts
    return summary, (
        candidate_ranks,
        candidate_scores,
        champion_ranks,
        champion_scores,
    )


def _arena_safety_guard(result: Mapping[str, object]) -> dict[str, bool]:
    rank = result["paired_rank_delta"]
    score = result["paired_score_delta"]
    return {
        "rank_harm": bool(rank["ci95_low"] > 0),
        "score_harm": bool(score["ci95_high"] < 0),
    }


def _skipped_phase(reason: str) -> dict[str, object]:
    return {"skipped": True, "reason": reason, "wall_seconds": 0.0}


def _early_promotion_rejection(
    reason: str,
    *,
    audit_harm: bool,
    arena_guards: Mapping[str, Mapping[str, bool]],
) -> dict[str, object]:
    return {
        "promoted": False,
        "pooled_rank_improvement": None,
        "guards_passed": False,
        "audit_harm": audit_harm,
        "arena_guards": dict(arena_guards),
        "reason": reason,
        "early_rejection": True,
    }


def _promotion_decision(
    fixed: Mapping[str, object],
    historical: Mapping[str, object],
    pooled: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    arena_guards = {
        "fixed": _arena_safety_guard(fixed),
        "historical": _arena_safety_guard(historical),
    }
    pooled_improvement = bool(pooled["paired_rank_delta"]["ci95_high"] < 0)
    audit_harm = bool(audit["ci95_high"] < 0)
    guards_passed = not audit_harm and not any(
        value for guard in arena_guards.values() for value in guard.values()
    )
    promoted = pooled_improvement and guards_passed
    return {
        "promoted": promoted,
        "pooled_rank_improvement": pooled_improvement,
        "guards_passed": guards_passed,
        "audit_harm": audit_harm,
        "arena_guards": arena_guards,
        "early_rejection": False,
        "reason": (
            "promoted"
            if promoted
            else "safety_guard_failed"
            if not guards_passed
            else "insufficient_pooled_rank_evidence"
        ),
    }


def _run_rank_lcb_iteration(
    actor: BloodFlowTransformer,
    device: torch.device,
    output_dir: Path,
    progress: Progress,
    *,
    iteration: int,
    champion_iteration: int,
    root_seed: int,
    config: RunConfig,
    evaluation_panel: tuple[np.ndarray, np.ndarray, np.ndarray],
    self_play: SelfPlayCurriculum,
    opponent_pool: OpponentPool,
    legacy_same_policy_opponent: bool,
    direction_optimizer_state: Mapping[str, torch.Tensor],
) -> tuple[
    BloodFlowTransformer,
    dict[str, object],
    Path,
    SelfPlayCurriculum,
    OpponentPool,
    dict[str, torch.Tensor],
]:
    if direction_optimizer_state:
        raise ValueError("rank-LCB optimizer state must be empty at generation boundaries")
    started = time.perf_counter()
    champion = clone_policy(actor, device)
    champion_digest = _model_digest(champion)
    opponent_snapshot: OpponentSnapshot | None = None
    self_play_actor: BloodFlowTransformer | None = None
    if self_play.fraction > 0.0:
        if legacy_same_policy_opponent:
            self_play_actor = champion
        else:
            opponent_snapshot = _select_opponent_snapshot(
                opponent_pool, current_digest=champion_digest
            )
            self_play_actor = _load_opponent_actor(
                opponent_snapshot, champion.config, device
            )

    pending = output_dir / "pending" / f"iteration-{iteration:06d}"
    batches, training_seeds, visit_stats = _collect_rank_lcb_batches(
        champion,
        self_play_actor,
        device,
        pending,
        progress,
        attempt=iteration,
        root_seed=root_seed,
        config=config,
        self_play=self_play,
        opponent_snapshot=opponent_snapshot,
    )
    visit_weights = np.asarray(visit_stats["vector"], dtype=np.float64)
    phase_started = time.perf_counter()
    calibration_queries, calibration_trajectories, calibration_source_metrics = (
        _collect_queries(
            champion,
            self_play_actor,
            device,
            progress,
            games=config.calibration_source_games,
            envs=config.envs,
            qpc=config.calibration_queries_per_category,
            source_seed=domain_seed(root_seed, CALIBRATION_DOMAIN, iteration),
            query_seed=domain_seed(
                root_seed, CALIBRATION_QUERY_DOMAIN, iteration
            ),
            phase=f"U{iteration}_CAL_SOURCE",
            self_play_fraction=self_play.fraction,
            opponent_snapshot=opponent_snapshot,
            anchor_rule_fast=config.anchor_rule_fast,
        )
    )
    if training_seeds & {
        int(trajectory.seed) for trajectory in calibration_trajectories
    }:
        raise RuntimeError("training and calibration source games overlap")
    calibration = build_state_batch(
        calibration_queries, history=champion.config.max_history
    )
    calibration_source_metrics["wall_seconds"] = (
        time.perf_counter() - phase_started
    )
    candidate, optimizer_metrics, calibration_metrics = _optimize_rank_lcb_generation(
        champion,
        batches,
        calibration,
        device,
        progress,
        attempt=iteration,
        config=config,
        visit_weights=visit_weights,
    )
    audit_metrics = _collect_rank_lcb_audit(
        candidate,
        champion,
        batches,
        self_play_actor,
        device,
        progress,
        attempt=iteration,
        config=config,
    )

    seeds, reference_ranks, reference_scores = evaluation_panel
    phase_started = time.perf_counter()
    progress.start(f"U{iteration}_EVAL", total=len(seeds), unit="games")
    evaluation_result = collect_fixed_panel(
        candidate,
        device,
        seeds,
        envs=config.evaluation_envs,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()
    actor_ranks, actor_scores = outcomes(evaluation_result)
    evaluation = summarize_paired(
        actor_ranks,
        actor_scores,
        reference_ranks,
        reference_scores,
        seed=domain_seed(root_seed, 0x7400_0001, iteration),
        bootstrap_samples=config.bootstrap_samples,
    )
    evaluation["opponent_seat_counts"] = evaluation_result.opponent_seat_counts
    evaluation["wall_seconds"] = time.perf_counter() - phase_started

    audit_harm = bool(audit_metrics["ci95_high"] < 0)
    if audit_harm:
        reason = "audit_guard_failed"
        fixed_arena = _skipped_phase(reason)
        historical_arena = _skipped_phase(reason)
        pooled_arena = _skipped_phase(reason)
        promotion = _early_promotion_rejection(
            reason,
            audit_harm=True,
            arena_guards={},
        )
    else:
        fixed_seeds = evaluation_seeds(
            domain_seed(root_seed, ARENA_FIXED_DOMAIN, iteration),
            config.arena_games,
        )
        phase_started = time.perf_counter()
        fixed_arena, fixed_arrays = _collect_arena_pair(
            candidate,
            champion,
            device,
            progress,
            label=f"U{iteration}_ARENA_FIXED",
            seeds=fixed_seeds,
            envs=config.evaluation_envs,
            self_play_fraction=0.0,
            self_play_actor=None,
            anchor_rule_fast=False,
            lineup_seed=domain_seed(root_seed, ARENA_LINEUP_DOMAIN, iteration),
            bootstrap_seed=domain_seed(root_seed, 0x7510_0001, iteration),
            bootstrap_samples=config.bootstrap_samples,
        )
        fixed_arena["wall_seconds"] = time.perf_counter() - phase_started
        fixed_guard = _arena_safety_guard(fixed_arena)
        if any(fixed_guard.values()):
            reason = "fixed_arena_guard_failed"
            historical_arena = _skipped_phase(reason)
            pooled_arena = _skipped_phase(reason)
            promotion = _early_promotion_rejection(
                reason,
                audit_harm=False,
                arena_guards={"fixed": fixed_guard},
            )
        else:
            historical_seeds = evaluation_seeds(
                domain_seed(root_seed, ARENA_HISTORY_DOMAIN, iteration),
                config.arena_games,
            )
            phase_started = time.perf_counter()
            historical_arena, historical_arrays = _collect_arena_pair(
                candidate,
                champion,
                device,
                progress,
                label=f"U{iteration}_ARENA_HISTORY",
                seeds=historical_seeds,
                envs=config.evaluation_envs,
                self_play_fraction=self_play.fraction,
                self_play_actor=self_play_actor,
                anchor_rule_fast=config.anchor_rule_fast,
                lineup_seed=domain_seed(
                    root_seed, ARENA_LINEUP_DOMAIN, iteration + 1
                ),
                bootstrap_seed=domain_seed(root_seed, 0x7510_0002, iteration),
                bootstrap_samples=config.bootstrap_samples,
            )
            historical_arena["wall_seconds"] = (
                time.perf_counter() - phase_started
            )
            pooled_arrays = tuple(
                np.concatenate((fixed_arrays[index], historical_arrays[index]))
                for index in range(4)
            )
            phase_started = time.perf_counter()
            pooled_arena = summarize_paired(
                *pooled_arrays,
                seed=domain_seed(root_seed, 0x7510_0003, iteration),
                bootstrap_samples=config.bootstrap_samples,
            )
            pooled_arena["wall_seconds"] = time.perf_counter() - phase_started
            promotion = _promotion_decision(
                fixed_arena, historical_arena, pooled_arena, audit_metrics
            )
    promoted = bool(promotion["promoted"])
    committed_actor = candidate if promoted else champion
    self_play_decision = _self_play_decision(evaluation, config)
    next_self_play = (
        _advance_self_play(
            self_play,
            evaluation,
            next_iteration=iteration + 1,
            config=config,
        )
        if promoted
        else self_play
    )
    next_opponent_pool = _advance_opponent_pool(
        opponent_pool,
        candidate,
        completed_iteration=iteration,
        used_historical_opponent=opponent_snapshot is not None,
        promoted=promoted,
    )
    target_batches = [batch.target_metrics for batch in batches]
    phase_seconds = {
        "source": sum(
            float(batch.source_metrics["wall_seconds"]) for batch in batches
        ),
        "selection_rollout": sum(
            float(batch.target_metrics["replicates"]["selection"]["wall_seconds"])
            for batch in batches
        ),
        "selection_policy": sum(
            float(batch.target_metrics["selection_policy_seconds"])
            for batch in batches
        ),
        "validation_rollout": sum(
            float(batch.target_metrics["replicates"]["validation"]["wall_seconds"])
            for batch in batches
        ),
        "target_build": sum(
            float(batch.target_metrics["target_build_seconds"])
            for batch in batches
        ),
        "calibration_source": float(calibration_source_metrics["wall_seconds"]),
        "optimizer": sum(
            float(step["wall_seconds"]) for step in optimizer_metrics["steps"]
        ),
        "kl_calibration": sum(
            float(step["wall_seconds"]) for step in calibration_metrics["steps"]
        ),
        "audit": float(audit_metrics["wall_seconds"]),
        "evaluation": float(evaluation["wall_seconds"]),
        "fixed_arena": float(fixed_arena["wall_seconds"]),
        "historical_arena": float(historical_arena["wall_seconds"]),
        "pooled_bootstrap": float(pooled_arena["wall_seconds"]),
    }
    elapsed_seconds = time.perf_counter() - started
    phase_seconds["other"] = max(0.0, elapsed_seconds - sum(phase_seconds.values()))
    phase_seconds["total"] = elapsed_seconds
    metrics: dict[str, object] = {
        "iteration": iteration,
        "attempt": iteration,
        "policy_version_before": champion_iteration,
        "policy_version_after": iteration if promoted else champion_iteration,
        "elapsed_seconds": elapsed_seconds,
        "phase_seconds": phase_seconds,
        "source": {
            "generation_batches": [batch.source_metrics for batch in batches],
            "games": config.generation_batches * config.source_games,
        },
        "source_visit_frequencies": visit_stats,
        "targets": {
            "objective": config.policy_objective,
            "generation_batches": target_batches,
            "accepted_states": int(
                sum(int(value["accepted_states"]) for value in target_batches)
            ),
            "tested_states": int(
                sum(int(value["tested_states"]) for value in target_batches)
            ),
        },
        "calibration_source": calibration_source_metrics,
        "optimizer": optimizer_metrics,
        "calibration": calibration_metrics,
        "audit": audit_metrics,
        "evaluation": evaluation,
        "arena": {
            "games_per_lineup": config.arena_games,
            "fixed": fixed_arena,
            "historical": historical_arena,
            "pooled": pooled_arena,
        },
        "promotion": promotion,
        "self_play": {
            "fraction": self_play.fraction,
            "next_fraction": next_self_play.fraction,
            "last_fixed_first_rate": next_self_play.last_fixed_first_rate,
            "activation_iteration": next_self_play.activation_iteration,
            "rank_delta_step": SELF_PLAY_RANK_DELTA_STEP,
            "rank_gate_passed": self_play_decision["rank_gate_passed"],
            "score_guard_passed": self_play_decision["score_guard_passed"],
            "evidence_passed": self_play_decision["evidence_passed"],
            "evidence_target_fraction": self_play_decision["target_fraction"],
            "snapshot_advanced": promoted,
            "rotation_advanced": opponent_snapshot is not None,
        },
        "opponent_pool": {
            "version": OPPONENT_POOL_VERSION,
            "capacity": OPPONENT_POOL_CAPACITY,
            "refresh_interval": OPPONENT_REFRESH_INTERVAL,
            "size": len(opponent_pool.snapshots),
            "next_size": len(next_opponent_pool.snapshots),
            "rotations": opponent_pool.rotations,
            "next_rotations": next_opponent_pool.rotations,
            "last_refresh_iteration": opponent_pool.last_refresh_iteration,
            "next_last_refresh_iteration": next_opponent_pool.last_refresh_iteration,
            "selected_iteration": (
                None if opponent_snapshot is None else opponent_snapshot.iteration
            ),
            "selected_digest": (
                None if opponent_snapshot is None else opponent_snapshot.digest
            ),
            "advanced": promoted,
        },
    }
    return (
        committed_actor,
        metrics,
        pending,
        next_self_play,
        next_opponent_pool,
        {},
    )


def _run_iteration(
    actor: BloodFlowTransformer,
    device: torch.device,
    output_dir: Path,
    progress: Progress,
    *,
    iteration: int,
    champion_iteration: int,
    root_seed: int,
    config: RunConfig,
    evaluation_panel: tuple[np.ndarray, np.ndarray, np.ndarray],
    self_play: SelfPlayCurriculum,
    opponent_pool: OpponentPool,
    legacy_same_policy_opponent: bool,
    direction_optimizer_state: Mapping[str, torch.Tensor],
) -> tuple[
    BloodFlowTransformer,
    dict[str, object],
    Path,
    SelfPlayCurriculum,
    OpponentPool,
    dict[str, torch.Tensor],
]:
    if config.policy_objective == "rank_lcb_mirror_ce":
        return _run_rank_lcb_iteration(
            actor,
            device,
            output_dir,
            progress,
            iteration=iteration,
            champion_iteration=champion_iteration,
            root_seed=root_seed,
            config=config,
            evaluation_panel=evaluation_panel,
            self_play=self_play,
            opponent_pool=opponent_pool,
            legacy_same_policy_opponent=legacy_same_policy_opponent,
            direction_optimizer_state=direction_optimizer_state,
        )
    started = time.perf_counter()
    frozen = clone_policy(actor, device)
    version_digest = _model_digest(frozen)
    opponent_snapshot: OpponentSnapshot | None = None
    self_play_actor: BloodFlowTransformer | None = None
    if self_play.fraction > 0.0:
        if legacy_same_policy_opponent:
            self_play_actor = frozen
        else:
            opponent_snapshot = _select_opponent_snapshot(
                opponent_pool, current_digest=version_digest
            )
            self_play_actor = _load_opponent_actor(
                opponent_snapshot, frozen.config, device
            )
    source_seed = domain_seed(root_seed, SOURCE_DOMAIN, iteration)
    queries, trajectories, source_metrics = _collect_queries(
        frozen,
        self_play_actor,
        device,
        progress,
        games=config.source_games,
        envs=config.envs,
        qpc=config.queries_per_category,
        source_seed=source_seed,
        query_seed=domain_seed(root_seed, SOURCE_QUERY_DOMAIN, iteration),
        phase=f"U{iteration}_SOURCE",
        self_play_fraction=self_play.fraction,
        opponent_snapshot=opponent_snapshot,
        anchor_rule_fast=config.anchor_rule_fast,
    )
    visit_stats = source_visit_frequencies(trajectories)
    visit_weights = np.asarray(visit_stats["vector"], dtype=np.float64)

    pending = output_dir / "pending" / f"iteration-{iteration:06d}"
    target_identity: dict[str, object] = {
        "actor": version_digest,
        "policy_execution_version": POLICY_EXECUTION_VERSION,
        "iteration": iteration,
        "root_seed": root_seed,
        "worlds": config.worlds,
        "policy_objective": config.policy_objective,
        "world_sampling": config.world_sampling,
        "self_play_fraction": self_play.fraction,
    }
    if config.policy_objective != "expected_q":
        target_identity.update(
            {
                "world_replicates": 2,
                "split_consensus_margin": config.split_consensus_margin,
            }
        )
    if config.anchor_rule_fast:
        target_identity["anchor_rule_fast"] = True
    if not legacy_same_policy_opponent:
        target_identity.update(
            {
                "opponent_pool_version": OPPONENT_POOL_VERSION,
                "self_play_opponent": (
                    None if opponent_snapshot is None else opponent_snapshot.digest
                ),
            }
        )
    policy_targets = None
    direction_objective = "expected_q"
    if config.policy_objective == "expected_q":
        target_fingerprint = _json_fingerprint(target_identity)
        progress.start(
            f"U{iteration}_TARGETS",
            total=len(queries),
            unit="queries",
            fields={"worlds": config.worlds},
        )
        targets, target_metrics = cached_counterfactual_corpus(
            pending / "targets",
            queries,
            frozen,
            device,
            self_play_actor=self_play_actor,
            fingerprint=target_fingerprint,
            worlds=config.worlds,
            world_chunk=config.world_chunk,
            world_seed=domain_seed(root_seed, SOURCE_WORLD_DOMAIN, iteration),
            world_sampling=config.world_sampling,
            shard_size=config.target_shard_size,
            query_batch_size=config.target_query_batch_size,
            inference_batch_size=config.rollout_inference_batch_size,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
        progress.complete()
    else:
        replicate_batches = []
        replicate_metrics: dict[str, object] = {}
        for replicate, world_domain in (
            ("a", SOURCE_WORLD_A_DOMAIN),
            ("b", SOURCE_WORLD_B_DOMAIN),
        ):
            progress.start(
                f"U{iteration}_TARGETS_{replicate.upper()}",
                total=len(queries),
                unit="queries",
                fields={"worlds": config.worlds},
            )
            outcomes_batch, outcomes_metrics = cached_world_outcome_corpus(
                pending / f"outcomes-{replicate}",
                queries,
                frozen,
                device,
                self_play_actor=self_play_actor,
                fingerprint=_json_fingerprint(
                    {**target_identity, "replicate": replicate}
                ),
                worlds=config.worlds,
                world_chunk=config.world_chunk,
                world_seed=domain_seed(root_seed, world_domain, iteration),
                world_sampling=config.world_sampling,
                shard_size=config.target_shard_size,
                query_batch_size=config.target_query_batch_size,
                inference_batch_size=config.rollout_inference_batch_size,
                on_progress=lambda done, values: progress.update(
                    done, fields=values
                ),
            )
            progress.complete()
            replicate_batches.append(outcomes_batch)
            replicate_metrics[replicate] = outcomes_metrics
        combined_outcomes = combine_world_replicates(replicate_batches)
        reference_probabilities, reference_actions = policy_outputs(
            frozen,
            combined_outcomes,
            device,
            batch_size=config.inference_batch_size,
        )
        if config.policy_objective == "holdout_consensus_ce":
            selection_index = iteration % 2
            validation_index = 1 - selection_index
            policy_targets, consensus_metrics = build_holdout_win_target(
                combined_outcomes,
                replicate_batches[selection_index],
                replicate_batches[validation_index],
                reference_probabilities,
                reference_actions,
                margin=config.split_consensus_margin,
            )
            consensus_metrics.update(
                {
                    "selection_replicate": ("a", "b")[selection_index],
                    "validation_replicate": ("a", "b")[validation_index],
                }
            )
        else:
            policy_targets, consensus_metrics = build_split_win_consensus_target(
                combined_outcomes,
                replicate_batches[0],
                replicate_batches[1],
                reference_probabilities,
                reference_actions,
                margin=config.split_consensus_margin,
            )
        targets = combined_outcomes.counterfactual_batch()
        target_metrics = {
            **consensus_metrics,
            "objective": config.policy_objective,
            "world_sampling": config.world_sampling,
            "replicates": replicate_metrics,
        }
        direction_objective = "search_ce"

    calibration_queries, calibration_trajectories, calibration_source_metrics = (
        _collect_queries(
            frozen,
            self_play_actor,
            device,
            progress,
            games=config.calibration_source_games,
            envs=config.envs,
            qpc=config.calibration_queries_per_category,
            source_seed=domain_seed(root_seed, CALIBRATION_DOMAIN, iteration),
            query_seed=domain_seed(
                root_seed, CALIBRATION_QUERY_DOMAIN, iteration
            ),
            phase=f"U{iteration}_CAL_SOURCE",
            self_play_fraction=self_play.fraction,
            opponent_snapshot=opponent_snapshot,
            anchor_rule_fast=config.anchor_rule_fast,
        )
    )
    if {trajectory.seed for trajectory in trajectories} & {
        trajectory.seed for trajectory in calibration_trajectories
    }:
        raise RuntimeError("training and calibration source games overlap")
    calibration = build_state_batch(
        calibration_queries, history=frozen.config.max_history
    )

    microbatches = math.ceil(len(targets) / config.microbatch_size)
    progress.start(
        f"U{iteration}_ACTOR",
        total=microbatches,
        unit="microbatches",
    )
    candidate, initial_state, candidate_state, optimizer_metrics = optimizer_direction(
        frozen,
        targets,
        device,
        category_weights=visit_weights,
        learning_rate=config.direction_learning_rate,
        microbatch_size=config.microbatch_size,
        optimizer_name=config.direction_optimizer,
        momentum=config.direction_momentum,
        gradient_clip_norm=(
            None
            if config.direction_gradient_clip_norm == 0.0
            else config.direction_gradient_clip_norm
        ),
        optimizer_state=direction_optimizer_state,
        objective=direction_objective,
        policy_targets=policy_targets,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()

    maximum_kl_evaluations = config.kl_search_steps + 2
    if config.kl_control == "target":
        maximum_kl_evaluations += int(math.ceil(math.log2(config.maximum_scale)))
    progress.start(
        f"U{iteration}_KL",
        total=maximum_kl_evaluations,
        unit="evaluations",
    )
    if config.kl_control == "cap":
        calibration_metrics = cap_direction(
            candidate,
            frozen,
            initial_state,
            candidate_state,
            calibration,
            device,
            category_weights=visit_weights,
            kl_cap=config.target_kl,
            batch_size=config.inference_batch_size,
            search_steps=config.kl_search_steps,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
    else:
        calibration_metrics = calibrate_direction(
            candidate,
            frozen,
            initial_state,
            candidate_state,
            calibration,
            device,
            category_weights=visit_weights,
            target_kl=config.target_kl,
            batch_size=config.inference_batch_size,
            search_steps=config.kl_search_steps,
            maximum_scale=config.maximum_scale,
            on_progress=lambda done, values: progress.update(done, fields=values),
        )
    progress.complete(
        int(calibration_metrics["evaluations"]),
        fields={
            "scale": calibration_metrics["scale"],
            "kl": calibration_metrics["final_kl"],
        },
    )

    seeds, reference_ranks, reference_scores = evaluation_panel
    progress.start(
        f"U{iteration}_EVAL", total=len(seeds), unit="games"
    )
    evaluation_result = collect_fixed_panel(
        candidate,
        device,
        seeds,
        envs=config.evaluation_envs,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()
    actor_ranks, actor_scores = outcomes(evaluation_result)
    evaluation = summarize_paired(
        actor_ranks,
        actor_scores,
        reference_ranks,
        reference_scores,
        seed=domain_seed(root_seed, 0x7400_0001, iteration),
        bootstrap_samples=config.bootstrap_samples,
    )
    if evaluation_result.opponent_seat_counts.get("self_play", 0) != 0:
        raise RuntimeError("fixed-rule evaluation unexpectedly used self-play")
    evaluation["opponent_seat_counts"] = evaluation_result.opponent_seat_counts
    self_play_decision = _self_play_decision(evaluation, config)
    next_self_play = _advance_self_play(
        self_play,
        evaluation,
        next_iteration=iteration + 1,
        config=config,
    )
    next_opponent_pool = _advance_opponent_pool(
        opponent_pool,
        candidate,
        completed_iteration=iteration,
        used_historical_opponent=opponent_snapshot is not None,
    )
    next_direction_optimizer_state = committed_optimizer_state(
        config.direction_optimizer,
        initial_state,
        candidate,
    )
    optimizer_metrics["committed_state_l2"] = float(
        math.sqrt(
            sum(
                float(torch.sum(value.double() * value.double()))
                for value in next_direction_optimizer_state.values()
            )
        )
    )
    metrics: dict[str, object] = {
        "iteration": iteration,
        "policy_version_before": champion_iteration,
        "policy_version_after": iteration,
        "elapsed_seconds": time.perf_counter() - started,
        "source": source_metrics,
        "source_visit_frequencies": visit_stats,
        "targets": target_metrics,
        "calibration_source": calibration_source_metrics,
        "optimizer": optimizer_metrics,
        "calibration": calibration_metrics,
        "evaluation": evaluation,
        "self_play": {
            "fraction": self_play.fraction,
            "next_fraction": next_self_play.fraction,
            "last_fixed_first_rate": next_self_play.last_fixed_first_rate,
            "activation_iteration": next_self_play.activation_iteration,
            "rank_delta_step": SELF_PLAY_RANK_DELTA_STEP,
            "rank_gate_passed": self_play_decision["rank_gate_passed"],
            "score_guard_passed": self_play_decision["score_guard_passed"],
            "evidence_passed": self_play_decision["evidence_passed"],
            "evidence_target_fraction": self_play_decision["target_fraction"],
        },
        "opponent_pool": {
            "version": OPPONENT_POOL_VERSION,
            "capacity": OPPONENT_POOL_CAPACITY,
            "refresh_interval": OPPONENT_REFRESH_INTERVAL,
            "size": len(opponent_pool.snapshots),
            "next_size": len(next_opponent_pool.snapshots),
            "rotations": opponent_pool.rotations,
            "next_rotations": next_opponent_pool.rotations,
            "last_refresh_iteration": opponent_pool.last_refresh_iteration,
            "next_last_refresh_iteration": next_opponent_pool.last_refresh_iteration,
            "selected_iteration": (
                None if opponent_snapshot is None else opponent_snapshot.iteration
            ),
            "selected_digest": (
                None if opponent_snapshot is None else opponent_snapshot.digest
            ),
            "legacy_same_policy_opponent": (
                legacy_same_policy_opponent and self_play.fraction > 0.0
            ),
        },
    }
    return (
        candidate,
        metrics,
        pending,
        next_self_play,
        next_opponent_pool,
        next_direction_optimizer_state,
    )


def run(args: argparse.Namespace) -> None:
    device = require_cuda(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    progress = Progress()
    output_dir: Path | None = None
    iteration = 1

    config_overrides = {
        name: getattr(args, name)
        for name in (field.name for field in fields(RunConfig))
        if getattr(args, name) is not None
    }
    if args.resume is not None:
        if config_overrides or args.seed is not None:
            raise ValueError("resume does not accept seed or config overrides")
        if args.resume.name != "latest.pt":
            raise ValueError("resume accepts only the canonical latest.pt checkpoint")
        actor, config, checkpoint = _load_checkpoint(args.resume, device)
        require_deterministic_actor(actor)
        output_dir = args.resume.resolve().parent
        if args.output_dir is not None and args.output_dir.resolve() != output_dir:
            raise ValueError("resume output directory does not match checkpoint")
        root_seed = int(checkpoint["root_seed"])
        next_iteration = int(checkpoint["next_iteration"])
        champion_iteration = int(
            checkpoint.get("champion_iteration", next_iteration - 1)
        )
        direction_optimizer_state = dict(
            checkpoint.get("direction_optimizer_state", {})
        )
        self_play = checkpoint["self_play"]
        sl_checkpoint = Path(str(checkpoint["sl_checkpoint"]))
        sl_sha256 = str(checkpoint["sl_sha256"])
        if not sl_checkpoint.exists() or _sha256(sl_checkpoint) != sl_sha256:
            raise ValueError("the frozen SL checkpoint changed or is missing")
        if args.sl_checkpoint is not None and args.sl_checkpoint.resolve() != sl_checkpoint:
            raise ValueError("resume SL checkpoint path does not match")
        opponent_pool = checkpoint["opponent_pool"]
        legacy_same_policy_opponent = bool(
            checkpoint["legacy_self_play_opponent"]
        )
        if opponent_pool is None:
            legacy_reference = load_policy(
                sl_checkpoint, torch.device("cpu"), frozen=True
            )
            require_deterministic_actor(legacy_reference)
            opponent_pool = _initial_opponent_pool(legacy_reference)
            if legacy_reference.config != actor.config:
                raise ValueError("legacy SL model config does not match checkpoint Actor")
            del legacy_reference
        last_metrics = checkpoint["last_metrics"]
        _ensure_run_identity(
            output_dir / "config.json",
            _run_identity(
                config=config,
                root_seed=root_seed,
                sl_checkpoint=sl_checkpoint,
                sl_sha256=sl_sha256,
            ),
        )
        if last_metrics is not None:
            _append_metric(output_dir / "metrics.jsonl", last_metrics)
        save_policy(output_dir / "actor.pt", actor)
        _cleanup_committed_pending(output_dir, next_iteration)
    else:
        config = RunConfig(**config_overrides)
        root_seed = 20260727 if args.seed is None else int(args.seed)
        output_dir = args.output_dir or Path("runs/policy-iteration-v3")
        initial_temporary = output_dir / "latest.pt.tmp"
        if output_dir.exists():
            entries = list(output_dir.iterdir())
            if entries == [initial_temporary]:
                initial_temporary.unlink()
            elif entries:
                raise ValueError("new output directory must be empty")
        sl_checkpoint = args.sl_checkpoint or Path(
            "runs/counterfactual-larger/sl_reference.pt"
        )
        if not sl_checkpoint.exists():
            raise FileNotFoundError(sl_checkpoint)
        sl_checkpoint = sl_checkpoint.resolve()
        sl_sha256 = _sha256(sl_checkpoint)
        actor = load_policy(sl_checkpoint, device, frozen=False)
        require_deterministic_actor(actor)
        next_iteration = 1
        champion_iteration = 0
        self_play = SelfPlayCurriculum()
        opponent_pool = _initial_opponent_pool(actor)
        legacy_same_policy_opponent = False
        direction_optimizer_state: dict[str, torch.Tensor] = {}
        last_metrics = None
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(
            output_dir / "latest.pt",
            _checkpoint_payload(
                actor,
                config=config,
                root_seed=root_seed,
                next_iteration=next_iteration,
                champion_iteration=champion_iteration,
                sl_checkpoint=sl_checkpoint,
                sl_sha256=sl_sha256,
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
                sl_checkpoint=sl_checkpoint,
                sl_sha256=sl_sha256,
            ),
        )
        save_policy(output_dir / "actor.pt", actor)

    torch.manual_seed(root_seed)
    torch.cuda.manual_seed_all(root_seed)
    np.random.seed(root_seed & 0xFFFF_FFFF)
    states_per_attempt = 9 * config.queries_per_category
    worlds_label = str(config.worlds)
    if config.policy_objective == "rank_lcb_mirror_ce":
        states_per_attempt *= config.generation_batches
        worlds_label = (
            f"{config.worlds}/{config.validation_worlds}/{config.audit_worlds}"
            f"x{config.generation_batches}"
        )
    elif config.policy_objective != "expected_q":
        worlds_label += "x2"
    print(
        f"CUDA {torch.cuda.get_device_name(device)}  "
        "gradients eager  inference eager  "
        f"states/attempt {states_per_attempt:,}  "
        f"worlds {worlds_label}  "
        f"KL-{config.kl_control} {config.target_kl:g}  "
        f"objective {config.policy_objective}  "
        f"sampling {config.world_sampling}  "
        f"optimizer {config.direction_optimizer}  "
        "grad-clip "
        + (
            "off"
            if config.direction_gradient_clip_norm == 0
            else f"{config.direction_gradient_clip_norm:g}"
        ),
        flush=True,
    )
    iteration = next_iteration
    try:
        panel_reference: BloodFlowTransformer | None
        temporary_reference = False
        if args.resume is None:
            panel_reference = actor
        elif (output_dir / "reference_panel.npz").exists():
            panel_reference = None
        else:
            panel_reference = load_policy(sl_checkpoint, device, frozen=True)
            require_deterministic_actor(panel_reference)
            temporary_reference = True
        panel = _reference_panel(
            panel_reference,
            device,
            output_dir,
            progress,
            root_seed=root_seed,
            config=config,
            sl_sha256=sl_sha256,
        )
        if self_play.last_fixed_first_rate is None:
            if next_iteration != 1 or last_metrics is not None:
                raise RuntimeError("initialized run is missing its self-play baseline")
            self_play = SelfPlayCurriculum(
                last_fixed_first_rate=float(np.mean(panel[1] == 1))
            )
            _save_checkpoint(
                output_dir / "latest.pt",
                _checkpoint_payload(
                    actor,
                    config=config,
                    root_seed=root_seed,
                    next_iteration=next_iteration,
                    champion_iteration=champion_iteration,
                    sl_checkpoint=sl_checkpoint,
                    sl_sha256=sl_sha256,
                    self_play=self_play,
                    opponent_pool=opponent_pool,
                    last_metrics=None,
                    direction_optimizer_state=direction_optimizer_state,
                ),
            )
        print(
            f"OPPONENTS self {100 * self_play.fraction:.0f}%  "
            f"fixed-first {100 * self_play.last_fixed_first_rate:.1f}%  "
            f"paired-dRank step {SELF_PLAY_RANK_DELTA_STEP:.2f}  "
            f"pool {len(opponent_pool.snapshots)}/{OPPONENT_POOL_CAPACITY}  "
            f"max {100 * config.maximum_self_play_fraction:.0f}%  "
            f"fast-anchor {'on' if config.anchor_rule_fast else 'off'}",
            flush=True,
        )
        if temporary_reference:
            del panel_reference
            torch.cuda.empty_cache()
        while True:
            (
                candidate,
                metrics,
                pending,
                next_self_play,
                next_opponent_pool,
                next_direction_optimizer_state,
            ) = _run_iteration(
                actor,
                device,
                output_dir,
                progress,
                iteration=iteration,
                champion_iteration=champion_iteration,
                root_seed=root_seed,
                config=config,
                evaluation_panel=panel,
                self_play=self_play,
                opponent_pool=opponent_pool,
                legacy_same_policy_opponent=legacy_same_policy_opponent,
                direction_optimizer_state=direction_optimizer_state,
            )
            next_champion_iteration = int(metrics["policy_version_after"])
            payload = _checkpoint_payload(
                candidate,
                config=config,
                root_seed=root_seed,
                next_iteration=iteration + 1,
                champion_iteration=next_champion_iteration,
                sl_checkpoint=sl_checkpoint,
                sl_sha256=sl_sha256,
                self_play=next_self_play,
                opponent_pool=next_opponent_pool,
                last_metrics=metrics,
                direction_optimizer_state=next_direction_optimizer_state,
            )
            progress.start(
                f"U{iteration}_CHECKPOINT", total=1, unit="commits"
            )
            _save_checkpoint(output_dir / "latest.pt", payload)
            actor = candidate
            champion_iteration = next_champion_iteration
            self_play = next_self_play
            opponent_pool = next_opponent_pool
            direction_optimizer_state = next_direction_optimizer_state
            legacy_same_policy_opponent = False
            committed_iteration = iteration
            iteration += 1
            save_policy(output_dir / "actor.pt", candidate)
            _append_metric(output_dir / "metrics.jsonl", metrics)
            if pending.exists():
                shutil.rmtree(pending)
            progress.complete(1)
            evaluation = metrics["evaluation"]
            rank = evaluation["paired_rank_delta"]
            score = evaluation["paired_score_delta"]
            if "promotion" in metrics:
                pooled_arena = metrics["arena"]["pooled"]
                if pooled_arena.get("skipped", False):
                    arena_text = f"arena skipped:{pooled_arena['reason']}"
                else:
                    arena_rank = pooled_arena["paired_rank_delta"]
                    arena_text = (
                        f"arena dRank {arena_rank['mean']:+.5f} "
                        f"[{arena_rank['ci95_low']:+.5f},"
                        f"{arena_rank['ci95_high']:+.5f}]"
                    )
                print(
                    f"attempt {committed_iteration:4d}  "
                    f"{'PROMOTE' if metrics['promotion']['promoted'] else 'REJECT '}  "
                    f"champion u{metrics['policy_version_after']:03d}  "
                    f"{arena_text}  "
                    f"audit {metrics['audit']['mean_rank_utility_advantage']:+.5f}  "
                    f"KL {metrics['calibration']['final_kl']:.6f}  "
                    f"time {metrics['elapsed_seconds'] / 60:.1f}m",
                    flush=True,
                )
                continue
            print(
                f"u {committed_iteration:4d}  rank {evaluation['actor']['mean_rank']:.4f}  "
                f"dRank {rank['mean']:+.4f} "
                f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]  "
                f"score {score['mean']:+.0f}  "
                f"KL {metrics['calibration']['final_kl']:.6f}  "
                f"flip {100 * metrics['calibration']['greedy_flip_rate']:.1f}%  "
                f"self {100 * metrics['self_play']['fraction']:.0f}%"
                f"->{100 * metrics['self_play']['next_fraction']:.0f}%  "
                f"pool {metrics['opponent_pool']['size']}"
                f"->{metrics['opponent_pool']['next_size']}  "
                f"time {metrics['elapsed_seconds'] / 60:.1f}m",
                flush=True,
            )
    except KeyboardInterrupt:
        if progress.active:
            snapshot = progress.snapshot()
            progress.complete(snapshot.current, fields={"interrupted": True})
        if output_dir is not None and (output_dir / "latest.pt").exists():
            committed = _committed_iteration(
                output_dir / "latest.pt", iteration - 1
            )
            print(
                f"Stopped. latest.pt contains complete iteration {committed}; "
                f"resume with --resume {output_dir / 'latest.pt'}",
                flush=True,
            )
        else:
            print("Stopped before the initial checkpoint was committed.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sl-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    for field in fields(RunConfig):
        option = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(
                option,
                dest=field.name,
                action=argparse.BooleanOptionalAction,
                default=None,
            )
        else:
            parser.add_argument(option, dest=field.name, type=type(field.default))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
