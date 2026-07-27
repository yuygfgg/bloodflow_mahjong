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
from typing import Any, Sequence

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
from .model import BloodFlowTransformer, TransformerConfig
from .pipeline import (
    CollectionConfig,
    POLICY_EXECUTION_VERSION,
    TrajectoryCollector,
    clone_policy,
    load_policy,
    save_policy,
)
from .policy_iteration import (
    build_state_batch,
    cached_counterfactual_corpus,
    calibrate_direction,
    cpu_model_state,
    domain_seed,
    one_step_direction,
    require_cuda,
    require_deterministic_actor,
    select_independent_queries,
    source_visit_frequencies,
)
from .progress import Progress


CHECKPOINT_VERSION = 3
SOURCE_DOMAIN = 0x7100_0001
SOURCE_QUERY_DOMAIN = 0x7100_0002
SOURCE_WORLD_DOMAIN = 0x7100_0003
CALIBRATION_DOMAIN = 0x7200_0001
CALIBRATION_QUERY_DOMAIN = 0x7200_0002
EVALUATION_DOMAIN = 0x7300_0001


@dataclass(frozen=True)
class RunConfig:
    source_games: int = 4096
    calibration_source_games: int = 4096
    envs: int = 512
    queries_per_category: int = 256
    calibration_queries_per_category: int = 64
    worlds: int = 16
    world_chunk: int = 64
    target_shard_size: int = 64
    target_query_batch_size: int = 64
    rollout_inference_batch_size: int = 128
    direction_learning_rate: float = 1e-5
    microbatch_size: int = 64
    inference_batch_size: int = 128
    target_kl: float = 1e-3
    kl_search_steps: int = 18
    maximum_scale: float = 64.0
    evaluation_games: int = 16_384
    evaluation_envs: int = 512
    bootstrap_samples: int = 10_000
    self_play_start_first_rate: float = 0.55
    self_play_increment: float = 0.10
    maximum_self_play_fraction: float = 2.0 / 3.0

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
        if self.worlds < 2:
            raise ValueError("counterfactual targets need at least two worlds")
        if self.maximum_scale < 1:
            raise ValueError("maximum_scale must be at least one")
        if not 0.0 <= self.self_play_start_first_rate <= 1.0:
            raise ValueError("self-play start first rate must be in [0, 1]")
        if self.self_play_increment > self.maximum_self_play_fraction:
            raise ValueError("self-play increment cannot exceed its maximum fraction")
        if self.maximum_self_play_fraction > 2.0 / 3.0:
            raise ValueError("self-play must always leave at least one rule opponent")


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


def _advance_self_play(
    state: SelfPlayCurriculum,
    fixed_first_rate: float,
    *,
    next_iteration: int,
    config: RunConfig,
) -> SelfPlayCurriculum:
    if not 0.0 <= fixed_first_rate <= 1.0:
        raise ValueError("fixed-rule first rate must be in [0, 1]")
    if next_iteration < 1:
        raise ValueError("next iteration must be positive")
    fraction = state.fraction
    activation = state.activation_iteration
    if fixed_first_rate >= config.self_play_start_first_rate:
        fraction = min(
            config.maximum_self_play_fraction,
            round(fraction + config.self_play_increment, 12),
        )
        if state.fraction == 0.0 and fraction > 0.0:
            activation = next_iteration
    return SelfPlayCurriculum(
        fraction=fraction,
        last_fixed_first_rate=fixed_first_rate,
        activation_iteration=activation,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_digest(model: BloodFlowTransformer) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
        "version": CHECKPOINT_VERSION,
        "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
        "policy_execution_version": POLICY_EXECUTION_VERSION,
        "config": asdict(config),
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
            and int(payload.get("version", -1)) == CHECKPOINT_VERSION
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
    sl_checkpoint: Path,
    sl_sha256: str,
    self_play: SelfPlayCurriculum,
    last_metrics: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "version": CHECKPOINT_VERSION,
        "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
        "policy_execution_version": POLICY_EXECUTION_VERSION,
        "config": asdict(config),
        "root_seed": int(root_seed),
        "next_iteration": int(next_iteration),
        "sl_checkpoint": str(sl_checkpoint.resolve()),
        "sl_sha256": sl_sha256,
        "self_play": asdict(self_play),
        "model_config": actor.config.__dict__,
        "actor": cpu_model_state(actor),
        "last_metrics": last_metrics,
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path, device: torch.device
) -> tuple[BloodFlowTransformer, RunConfig, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
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
    if set(payload) != expected or int(payload["version"]) != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint is not current production format v{CHECKPOINT_VERSION}"
        )
    if int(payload["engine_rules_version"]) != int(bm.ENGINE_RULES_VERSION):
        raise ValueError("checkpoint engine rules version does not match")
    if int(payload["policy_execution_version"]) != POLICY_EXECUTION_VERSION:
        raise ValueError("checkpoint policy execution version does not match")
    if int(payload["next_iteration"]) < 1:
        raise ValueError("checkpoint next_iteration must be positive")
    last_metrics = payload["last_metrics"]
    if last_metrics is not None:
        if (
            not isinstance(last_metrics, dict)
            or int(last_metrics.get("iteration", -1))
            != int(payload["next_iteration"]) - 1
        ):
            raise ValueError("checkpoint last_metrics does not match next_iteration")
    config_state = dict(payload["config"])
    if set(config_state) != {field.name for field in fields(RunConfig)}:
        raise ValueError("checkpoint RunConfig fields do not match this trainer")
    config = RunConfig(**config_state)
    self_play_state = dict(payload["self_play"])
    if set(self_play_state) != {field.name for field in fields(SelfPlayCurriculum)}:
        raise ValueError("checkpoint self-play state fields do not match this trainer")
    self_play = SelfPlayCurriculum(**self_play_state)
    if self_play.fraction > config.maximum_self_play_fraction:
        raise ValueError("checkpoint self-play fraction exceeds the configured maximum")
    payload["self_play"] = self_play
    actor = BloodFlowTransformer(TransformerConfig(**payload["model_config"])).to(
        device
    )
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
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
) -> tuple[list[Any], tuple[Any, ...], dict[str, object]]:
    progress.start(phase, total=games, unit="games")
    collector = TrajectoryCollector(
        CollectionConfig(envs=min(envs, games), history=actor.config.max_history),
        actor,
        device,
        seed=source_seed,
        self_play_fraction=self_play_fraction,
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


def _run_iteration(
    actor: BloodFlowTransformer,
    device: torch.device,
    output_dir: Path,
    progress: Progress,
    *,
    iteration: int,
    root_seed: int,
    config: RunConfig,
    evaluation_panel: tuple[np.ndarray, np.ndarray, np.ndarray],
    self_play: SelfPlayCurriculum,
) -> tuple[
    BloodFlowTransformer, dict[str, object], Path, SelfPlayCurriculum
]:
    started = time.perf_counter()
    frozen = clone_policy(actor, device)
    version_digest = _model_digest(frozen)
    source_seed = domain_seed(root_seed, SOURCE_DOMAIN, iteration)
    queries, trajectories, source_metrics = _collect_queries(
        frozen,
        device,
        progress,
        games=config.source_games,
        envs=config.envs,
        qpc=config.queries_per_category,
        source_seed=source_seed,
        query_seed=domain_seed(root_seed, SOURCE_QUERY_DOMAIN, iteration),
        phase=f"U{iteration}_SOURCE",
        self_play_fraction=self_play.fraction,
    )
    visit_stats = source_visit_frequencies(trajectories)
    visit_weights = np.asarray(visit_stats["vector"], dtype=np.float64)

    pending = output_dir / "pending" / f"iteration-{iteration:06d}"
    target_fingerprint = _json_fingerprint(
        {
            "actor": version_digest,
            "policy_execution_version": POLICY_EXECUTION_VERSION,
            "iteration": iteration,
            "root_seed": root_seed,
            "worlds": config.worlds,
            "self_play_fraction": self_play.fraction,
        }
    )
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
        fingerprint=target_fingerprint,
        worlds=config.worlds,
        world_chunk=config.world_chunk,
        world_seed=domain_seed(root_seed, SOURCE_WORLD_DOMAIN, iteration),
        shard_size=config.target_shard_size,
        query_batch_size=config.target_query_batch_size,
        inference_batch_size=config.rollout_inference_batch_size,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()

    calibration_queries, calibration_trajectories, calibration_source_metrics = (
        _collect_queries(
            frozen,
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
    candidate, initial_state, candidate_state, optimizer_metrics = one_step_direction(
        frozen,
        targets,
        device,
        category_weights=visit_weights,
        learning_rate=config.direction_learning_rate,
        microbatch_size=config.microbatch_size,
        on_progress=lambda done, values: progress.update(done, fields=values),
    )
    progress.complete()

    maximum_kl_evaluations = (
        config.kl_search_steps
        + 2
        + int(math.ceil(math.log2(config.maximum_scale)))
    )
    progress.start(
        f"U{iteration}_KL",
        total=maximum_kl_evaluations,
        unit="evaluations",
    )
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
    next_self_play = _advance_self_play(
        self_play,
        float(evaluation["actor"]["first_rate"]),
        next_iteration=iteration + 1,
        config=config,
    )
    metrics: dict[str, object] = {
        "iteration": iteration,
        "policy_version_before": iteration - 1,
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
        },
    }
    return candidate, metrics, pending, next_self_play


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
        self_play = checkpoint["self_play"]
        sl_checkpoint = Path(str(checkpoint["sl_checkpoint"]))
        sl_sha256 = str(checkpoint["sl_sha256"])
        if not sl_checkpoint.exists() or _sha256(sl_checkpoint) != sl_sha256:
            raise ValueError("the frozen SL checkpoint changed or is missing")
        if args.sl_checkpoint is not None and args.sl_checkpoint.resolve() != sl_checkpoint:
            raise ValueError("resume SL checkpoint path does not match")
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
        self_play = SelfPlayCurriculum()
        last_metrics = None
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(
            output_dir / "latest.pt",
            _checkpoint_payload(
                actor,
                config=config,
                root_seed=root_seed,
                next_iteration=next_iteration,
                sl_checkpoint=sl_checkpoint,
                sl_sha256=sl_sha256,
                self_play=self_play,
                last_metrics=None,
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
    print(
        f"CUDA {torch.cuda.get_device_name(device)}  eager mode  "
        f"states/update {9 * config.queries_per_category:,}  "
        f"worlds {config.worlds}  target_KL {config.target_kl:g}",
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
            self_play = _advance_self_play(
                self_play,
                float(np.mean(panel[1] == 1)),
                next_iteration=1,
                config=config,
            )
            _save_checkpoint(
                output_dir / "latest.pt",
                _checkpoint_payload(
                    actor,
                    config=config,
                    root_seed=root_seed,
                    next_iteration=next_iteration,
                    sl_checkpoint=sl_checkpoint,
                    sl_sha256=sl_sha256,
                    self_play=self_play,
                    last_metrics=None,
                ),
            )
        print(
            f"OPPONENTS self {100 * self_play.fraction:.0f}%  "
            f"fixed-first {100 * self_play.last_fixed_first_rate:.1f}%  "
            f"gate {100 * config.self_play_start_first_rate:.0f}%  "
            f"max {100 * config.maximum_self_play_fraction:.0f}%",
            flush=True,
        )
        if temporary_reference:
            del panel_reference
            torch.cuda.empty_cache()
        while True:
            candidate, metrics, pending, next_self_play = _run_iteration(
                actor,
                device,
                output_dir,
                progress,
                iteration=iteration,
                root_seed=root_seed,
                config=config,
                evaluation_panel=panel,
                self_play=self_play,
            )
            payload = _checkpoint_payload(
                candidate,
                config=config,
                root_seed=root_seed,
                next_iteration=iteration + 1,
                sl_checkpoint=sl_checkpoint,
                sl_sha256=sl_sha256,
                self_play=next_self_play,
                last_metrics=metrics,
            )
            progress.start(
                f"U{iteration}_CHECKPOINT", total=1, unit="commits"
            )
            _save_checkpoint(output_dir / "latest.pt", payload)
            actor = candidate
            self_play = next_self_play
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
            print(
                f"u {committed_iteration:4d}  rank {evaluation['actor']['mean_rank']:.4f}  "
                f"dRank {rank['mean']:+.4f} "
                f"[{rank['ci95_low']:+.4f},{rank['ci95_high']:+.4f}]  "
                f"score {score['mean']:+.0f}  "
                f"KL {metrics['calibration']['final_kl']:.6f}  "
                f"flip {100 * metrics['calibration']['greedy_flip_rate']:.1f}%  "
                f"self {100 * metrics['self_play']['fraction']:.0f}%"
                f"->{100 * metrics['self_play']['next_fraction']:.0f}%  "
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
        parser.add_argument(option, dest=field.name, type=type(field.default))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
