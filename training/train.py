"""CUDA-only synthetic trajectory IQL/AWR training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dashboard import render_dashboard
from .iql import CriticConfig, IndependentCritics
from .learner import (
    LearningConfig,
    actor_update,
    cql_scale_from_coverage,
    critic_ready,
    critic_update_deferred,
    make_optimizers,
    mc_critic_update_deferred,
    mc_teacher_ready,
    oracle_teacher_ready,
    resolve_actor_gate,
    resolve_critic_statistics,
    resolve_mc_critic_statistics,
    validate_critics,
)
from .mc_teacher import MonteCarloConfig, collect_mc_targets
from .model import BloodFlowTransformer, TransformerConfig
from .oracle import OracleCritics
from .pipeline import (
    CollectionConfig,
    ExecutablePolicyPool,
    FullTrajectoryCollector,
    better_on_both_panels,
    clone_policy,
    evaluate_panel,
    load_policy,
    save_policy,
)
from .policy_pool import (
    BalancedReplaySampler,
    BehaviorSampler,
    OpponentMixtureConfig,
    PolicyPool,
    ReplaySource,
)
from .replay import ReplayConfig, TrajectoryReplay


CHECKPOINT_VERSION = 3


@dataclass(frozen=True)
class RunConfig:
    collection: CollectionConfig = CollectionConfig()
    replay: ReplayConfig = ReplayConfig()
    learning: LearningConfig = LearningConfig()
    critics: CriticConfig = CriticConfig()
    opponents: OpponentMixtureConfig = OpponentMixtureConfig()
    mc: MonteCarloConfig = MonteCarloConfig()
    anchor_games: int = 8192
    games_per_iteration: int = 1024
    validation_states: int = 8192
    evaluation_every: int = 5
    evaluation_games: int = 256
    evaluation_seed_count: int = 2
    checkpoint_every: int = 5
    mc_validation_every: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.anchor_games,
            self.games_per_iteration,
            self.validation_states,
            self.evaluation_every,
            self.evaluation_games,
            self.evaluation_seed_count,
            self.checkpoint_every,
            self.mc_validation_every,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("run sizes and intervals must be positive")

    def state_dict(self) -> dict[str, object]:
        return {
            "collection": asdict(self.collection),
            "replay": asdict(self.replay),
            "learning": asdict(self.learning),
            "critics": asdict(self.critics),
            "opponents": asdict(self.opponents),
            "mc": asdict(self.mc),
            "anchor_games": self.anchor_games,
            "games_per_iteration": self.games_per_iteration,
            "validation_states": self.validation_states,
            "evaluation_every": self.evaluation_every,
            "evaluation_games": self.evaluation_games,
            "evaluation_seed_count": self.evaluation_seed_count,
            "checkpoint_every": self.checkpoint_every,
            "mc_validation_every": self.mc_validation_every,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> RunConfig:
        learning_state = dict(state["learning"])  # type: ignore[arg-type]
        expected_learning_fields = set(asdict(LearningConfig()))
        actual_learning_fields = set(learning_state)
        if actual_learning_fields != expected_learning_fields:
            missing = sorted(expected_learning_fields - actual_learning_fields)
            unexpected = sorted(actual_learning_fields - expected_learning_fields)
            raise ValueError(
                f"checkpoint LearningConfig does not match v{CHECKPOINT_VERSION}; "
                f"missing fields: {missing}, unexpected fields: {unexpected}"
            )
        return cls(
            collection=CollectionConfig(**state["collection"]),  # type: ignore[arg-type]
            replay=ReplayConfig(**state["replay"]),  # type: ignore[arg-type]
            learning=LearningConfig(**learning_state),
            critics=CriticConfig(**state["critics"]),  # type: ignore[arg-type]
            opponents=OpponentMixtureConfig(**state["opponents"]),  # type: ignore[arg-type]
            mc=MonteCarloConfig(**state["mc"]),  # type: ignore[arg-type]
            anchor_games=int(state["anchor_games"]),
            games_per_iteration=int(state["games_per_iteration"]),
            validation_states=int(state["validation_states"]),
            evaluation_every=int(state["evaluation_every"]),
            evaluation_games=int(state["evaluation_games"]),
            evaluation_seed_count=int(state["evaluation_seed_count"]),
            checkpoint_every=int(state["checkpoint_every"]),
            mc_validation_every=int(state["mc_validation_every"]),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/iql-awr-v3"))
    parser.add_argument(
        "--sl-checkpoint",
        type=Path,
        default=Path("runs/counterfactual-larger/sl_reference.pt"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--experiment",
        choices=("a", "b", "c"),
        default=None,
        help="a=partial IQL/AWR, b=oracle critic, c=selective information-set MC",
    )
    parser.add_argument("--envs", type=int)
    parser.add_argument("--anchor-games", type=int)
    parser.add_argument("--games-per-iteration", type=int)
    parser.add_argument("--critic-batch-size", type=int)
    parser.add_argument("--actor-batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--initial-critic-steps", type=int)
    parser.add_argument("--critic-steps", type=int)
    parser.add_argument("--actor-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose-console", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig()
    collection_changes = {
        key: value
        for key, value in (("envs", args.envs),)
        if value is not None
    }
    learning_changes = {
        key: value
        for key, value in (
            ("critic_batch_size", args.critic_batch_size),
            ("actor_batch_size", args.actor_batch_size),
            ("microbatch_size", args.microbatch_size),
            ("initial_critic_steps", args.initial_critic_steps),
            ("critic_steps_per_iteration", args.critic_steps),
            ("actor_steps_per_iteration", args.actor_steps),
        )
        if value is not None
    }
    top_changes = {
        key: value
        for key, value in (
            ("anchor_games", args.anchor_games),
            ("games_per_iteration", args.games_per_iteration),
            ("evaluation_every", args.eval_every),
            ("evaluation_games", args.eval_games),
            ("checkpoint_every", args.checkpoint_every),
        )
        if value is not None
    }
    if collection_changes:
        config = replace(config, collection=replace(config.collection, **collection_changes))
    if learning_changes:
        config = replace(config, learning=replace(config.learning, **learning_changes))
    if top_changes:
        config = replace(config, **top_changes)
    if args.smoke:
        config = replace(
            config,
            collection=replace(config.collection, envs=16),
            replay=replace(
                config.replay,
                validation_fraction=0.25,
                maximum_online_transitions=4096,
            ),
            learning=replace(
                config.learning,
                critic_batch_size=64,
                actor_batch_size=64,
                microbatch_size=16,
                initial_critic_steps=2,
                critic_steps_per_iteration=1,
                actor_steps_per_iteration=1,
                mc_critic_batch_size=16,
                mc_critic_steps_per_iteration=1,
                minimum_critic_steps=1,
                minimum_middle_late_improvement=-1_000_000.0,
                minimum_middle_late_correlation=-1.0,
                maximum_q_disagreement=10.0,
                teacher_readiness_streak=1,
                minimum_oracle_relative_mae_gain=-1_000_000.0,
                minimum_oracle_early_relative_mae_gain=-1_000_000.0,
                minimum_oracle_early_improvement=-1_000_000.0,
                minimum_oracle_early_correlation=-1.0,
                minimum_oracle_value_correlation=-1.0,
                maximum_oracle_q_disagreement=10.0,
                maximum_oracle_expectile_balance_error=1.0,
                minimum_mc_train_targets=1,
                minimum_mc_validation_targets=1,
                minimum_mc_validation_groups=1,
                minimum_mc_pairwise_pairs=0,
                minimum_mc_pairwise_accuracy=0.0,
                maximum_mc_mean_regret=1_000_000.0,
            ),
            critics=replace(
                config.critics,
                d_model=32,
                num_heads=4,
                static_layers=1,
                history_layers=1,
                ffn_dim=64,
                head_dim=48,
            ),
            mc=replace(
                config.mc,
                queries_per_iteration=1,
                hidden_worlds=2,
                candidate_pool_states=16,
                maximum_confidence_half_width=1_000_000.0,
                minimum_reliable_action_gap=0.0,
            ),
            anchor_games=16,
            games_per_iteration=8,
            validation_states=64,
            evaluation_every=1,
            evaluation_games=8,
            evaluation_seed_count=1,
            checkpoint_every=1,
            mc_validation_every=1,
        )
    return config


def _validate_args(args: argparse.Namespace) -> None:
    if args.resume is not None and args.sl_checkpoint != Path(
        "runs/counterfactual-larger/sl_reference.pt"
    ):
        raise ValueError("--resume cannot be combined with --sl-checkpoint")
    for name in (
        "envs",
        "anchor_games",
        "games_per_iteration",
        "critic_batch_size",
        "actor_batch_size",
        "microbatch_size",
        "initial_critic_steps",
        "critic_steps",
        "actor_steps",
        "eval_every",
        "eval_games",
        "checkpoint_every",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _synchronize() -> None:
    torch.cuda.synchronize()


def _compact_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{100.0 * number:.1f}%" if np.isfinite(number) else "-"


def _compact_record(record: dict[str, object]) -> str:
    phase = str(record.get("phase", "event"))
    if phase == "baseline":
        evaluation = record["fixed_evaluation"]
        assert isinstance(evaluation, dict)
        return (
            f"BASE  rank {float(evaluation['mean_rank']):.2f}"
            f"  score {float(evaluation['mean_score_delta']):.0f}"
            f"  replay {int(record.get('replay_states', 0)):,}"
        )
    if phase == "critic_warmup":
        validation = record.get("critic_validation", {})
        q = validation.get("q", {}) if isinstance(validation, dict) else {}
        return (
            f"CRITIC  step {int(record.get('critic_steps', 0)):>5}"
            f"  qMAE {float(q.get('mae', 0.0)):.3f}"
            f"  corr {float(q.get('correlation', 0.0)):.3f}"
            f"  ready {str(record.get('actor_gate', 'unknown'))}"
        )
    if phase == "iteration":
        evaluation = record.get("fixed_evaluation")
        validation = record.get("critic_validation", {})
        q = validation.get("q", {}) if isinstance(validation, dict) else {}
        message = (
            f"u{int(record.get('iteration', 0)):>5}"
            f"  replay {int(record.get('replay_states', 0)):>9,}"
            f"  qMAE {float(q.get('mae', 0.0)):.3f}"
            f"  dis {float(validation.get('q_disagreement', 0.0)):.3f}"
            f"  actor {record.get('actor_gate', 'held')}"
        )
        mc_critic = record.get("mc_critic")
        mc = record.get("mc")
        if isinstance(mc_critic, dict) or isinstance(mc, dict):
            mc_critic = mc_critic if isinstance(mc_critic, dict) else {}
            mc = mc if isinstance(mc, dict) else {}
            validation_metrics = mc.get("validation_metrics")
            validation_metrics = (
                validation_metrics if isinstance(validation_metrics, dict) else {}
            )
            ranking = validation_metrics.get("action_ranking")
            ranking = ranking if isinstance(ranking, dict) else {}
            train_targets = mc.get(
                "train_targets_after_trim", mc.get("train_targets", 0)
            )
            validation_targets = mc.get("validation_targets", 0)
            reliable_pairs = ranking.get(
                "pair_count", mc.get("validation_reliable_pairs", 0)
            )
            all_pairs = ranking.get("all_pair_count", ranking.get("pair_count", 0))
            reliable_groups = mc.get(
                "validation_reliable_groups",
                ranking.get("group_count", 0),
            )
            message += (
                "  MC "
                f"{int(train_targets):,}/{int(validation_targets):,}"
                f" diff {float(mc_critic.get('mc_centered_loss', 0.0)):.3f}"
                f" pair {float(mc_critic.get('mc_pairwise_loss', 0.0)):.3f}"
                f" train_acc {_compact_percent(mc_critic.get('mc_train_pairwise_accuracy'))}"
                f" val_acc {_compact_percent(ranking.get('pairwise_accuracy'))}"
                f" sig_pairs {int(reliable_pairs):,}/{int(all_pairs):,}"
                f" rel_groups {int(reliable_groups):,}"
                f" frozen {'yes' if mc.get('validation_frozen') is True else 'no'}"
                f" {float(record.get('mc_critic_seconds', 0.0)):.1f}s"
            )
        actor = record.get("actor")
        if isinstance(actor, dict):
            message += f"  KL {float(actor.get('actor_reference_kl', 0.0)):.4f}"
        if isinstance(evaluation, dict):
            message += (
                f"  rank {float(evaluation.get('mean_rank', 0.0)):.2f}"
                f"  score {float(evaluation.get('mean_score_delta', 0.0)):.0f}"
            )
        return message
    if phase == "stopped":
        return (
            f"STOP  u{int(record.get('iteration', 0))}"
            f"  critic {int(record.get('critic_steps', 0))}"
            f"  reason {record.get('reason', 'unknown')}"
        )
    return phase


def _should_evaluate(iteration: int, every: int, *, smoke: bool) -> bool:
    return smoke or iteration % every == 0


def _mc_validation_corpus_status(
    replay: TrajectoryReplay, config: LearningConfig
) -> dict[str, object]:
    """Return cheap, corpus-wide evidence counts used to freeze MC validation."""

    validation_targets = replay.mc_target_count("validation", anchor_only=True)
    reliable = replay.reliable_mc_counts("validation", anchor_only=True)
    reliable_targets = int(reliable["targets"])
    reliable_groups = int(reliable["groups"])
    reliable_pairs = int(reliable["pairs"])
    frozen = (
        validation_targets >= config.minimum_mc_validation_targets
        and reliable_groups >= config.minimum_mc_validation_groups
        and reliable_pairs >= config.minimum_mc_pairwise_pairs
    )
    return {
        "validation_targets": validation_targets,
        "validation_reliable_targets": reliable_targets,
        "validation_reliable_groups": reliable_groups,
        "validation_reliable_pairs": reliable_pairs,
        "validation_frozen": frozen,
    }


def _mc_validation_corpus_gate(
    status: dict[str, object], config: LearningConfig
) -> str:
    if int(status["validation_targets"]) < config.minimum_mc_validation_targets:
        return "mc_validation_targets"
    if (
        int(status["validation_reliable_groups"])
        < config.minimum_mc_validation_groups
    ):
        return "mc_validation_reliable_groups"
    if (
        int(status["validation_reliable_pairs"])
        < config.minimum_mc_pairwise_pairs
    ):
        return "mc_validation_reliable_pairs"
    return "ready"


def _write_record(
    metrics_path: Path,
    dashboard_path: Path,
    record: dict[str, object],
    *,
    verbose: bool,
) -> None:
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(_compact_record(record), flush=True)
    if verbose:
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if record.get("phase") != "iteration" or "fixed_evaluation" in record:
        render_dashboard(metrics_path, dashboard_path)


def _mean_statistics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = set.intersection(*(set(value) for value in values))
    return {
        key: float(np.mean([value[key] for value in values])) for key in sorted(keys)
    }


def _train_critics(
    replay: TrajectoryReplay,
    sampler: BalancedReplaySampler,
    critics: IndependentCritics,
    optimizers: dict[str, torch.optim.Optimizer],
    config: RunConfig,
    device: torch.device,
    steps: int,
    *,
    oracle: OracleCritics | None,
    enable_oracle_distillation: bool,
    show_progress: bool = False,
) -> tuple[dict[str, float], float]:
    index = replay.index("train", include_mc=False)
    prepared = sampler.prepare(
        index.sources,
        index.categories,
        duplicate_keys=index.duplicate_keys,
        policy_versions=index.policy_versions,
    )
    current_fraction = float(
        np.mean(index.sources == int(ReplaySource.CURRENT))
    )
    cql_scale = cql_scale_from_coverage(config.learning, current_fraction)
    statistics: list[torch.Tensor] = []
    started = time.perf_counter()

    def submit_batch(executor: ThreadPoolExecutor):
        selected = sampler.sample_index(prepared, config.learning.critic_batch_size)
        replay.cursor += len(selected.indices)
        return executor.submit(
            replay.materialize,
            index,
            selected.indices,
            include_oracle=oracle is not None,
        )

    # Engine replay runs on a detached Rust thread.  Materialize batch N+1
    # while CUDA trains batch N so the GPU no longer waits between updates.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-critic") as executor:
        pending = submit_batch(executor)
        for step in range(steps):
            batch = pending.result()
            if step + 1 < steps:
                pending = submit_batch(executor)
            statistics.append(
                critic_update_deferred(
                    critics,
                    optimizers,
                    batch,
                    config.learning,
                    device,
                    cql_scale=cql_scale,
                    oracle=oracle,
                    enable_oracle_distillation=enable_oracle_distillation,
                )
            )
            completed = step + 1
            progress_every = max(steps // 5, 1)
            if show_progress and (
                completed % progress_every == 0 or completed == steps
            ):
                elapsed = time.perf_counter() - started
                states_per_second = (
                    completed * config.learning.critic_batch_size
                ) / max(elapsed, 1e-9)
                print(
                    f"WARM  {completed:>4}/{steps:<4}"
                    f"  {states_per_second:,.0f} states/s",
                    flush=True,
                )
    result = resolve_critic_statistics(
        torch.stack(statistics).mean(dim=0), cql_scale=cql_scale
    )
    return result, time.perf_counter() - started


def _train_mc_critics(
    replay: TrajectoryReplay,
    critics: IndependentCritics,
    optimizers: dict[str, torch.optim.Optimizer],
    config: RunConfig,
    device: torch.device,
    *,
    seed: int,
) -> tuple[dict[str, float] | None, float]:
    """Train Q heads on complete MC query groups without touching V."""

    batch_size = min(
        config.learning.mc_critic_batch_size,
        config.learning.microbatch_size,
    )
    if batch_size < 2:
        raise ValueError("MC critic batch must fit at least two candidate actions")
    started = time.perf_counter()
    statistics: list[torch.Tensor] = []

    def submit(executor: ThreadPoolExecutor, step: int):
        return executor.submit(
            replay.mc_training_batch,
            batch_size,
            seed=seed + step * 0x9E3779B1,
        )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-mc") as executor:
        pending = submit(executor, 0)
        for step in range(config.learning.mc_critic_steps_per_iteration):
            batch = pending.result()
            if batch is None:
                break
            if step + 1 < config.learning.mc_critic_steps_per_iteration:
                pending = submit(executor, step + 1)
            replay.cursor += len(batch)
            statistics.append(
                mc_critic_update_deferred(
                    critics,
                    optimizers["q"],
                    batch,
                    config.learning,
                    device,
                )
            )
    if not statistics:
        return None, time.perf_counter() - started
    result = resolve_mc_critic_statistics(torch.stack(statistics).mean(dim=0))
    result["mc_critic_updates"] = float(len(statistics))
    return result, time.perf_counter() - started


def _train_actor(
    replay: TrajectoryReplay,
    sampler: BalancedReplaySampler,
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    critics: IndependentCritics,
    optimizers: dict[str, torch.optim.Optimizer],
    config: RunConfig,
    device: torch.device,
    *,
    oracle: OracleCritics | None,
    use_oracle_teacher: bool,
) -> tuple[dict[str, float], float]:
    index = replay.index("train", include_mc=False)
    prepared = sampler.prepare(
        index.sources,
        index.categories,
        duplicate_keys=index.duplicate_keys,
        policy_versions=index.policy_versions,
    )
    statistics: list[dict[str, float]] = []
    started = time.perf_counter()
    actor_steps = config.learning.actor_steps_per_iteration

    def submit_batch(executor: ThreadPoolExecutor):
        selected = sampler.sample_index(prepared, config.learning.actor_batch_size)
        replay.cursor += len(selected.indices)
        return executor.submit(
            replay.materialize,
            index,
            selected.indices,
            include_oracle=use_oracle_teacher,
        )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-actor") as executor:
        pending = submit_batch(executor)
        for step in range(actor_steps):
            batch = pending.result()
            if step + 1 < actor_steps:
                pending = submit_batch(executor)
            statistics.append(
                actor_update(
                    actor,
                    reference,
                    critics,
                    optimizers["actor"],
                    batch,
                    config.learning,
                    device,
                    oracle=oracle,
                    use_oracle_teacher=use_oracle_teacher,
                    measure_post_update_kl=step + 1 == actor_steps,
                )
            )
    _synchronize()
    result = _mean_statistics(statistics)
    if "actor_reference_kl" in statistics[-1]:
        result["actor_reference_kl"] = statistics[-1]["actor_reference_kl"]
    return result, time.perf_counter() - started


def _critic_validation(
    replay: TrajectoryReplay,
    critics: IndependentCritics,
    config: RunConfig,
    device: torch.device,
    *,
    seed: int,
    oracle: OracleCritics | None,
) -> dict[str, object]:
    batch = replay.validation_batch(
        config.validation_states,
        seed=seed,
        include_mc=False,
        include_oracle=oracle is not None,
    )
    return validate_critics(
        critics,
        batch,
        device,
        microbatch_size=config.learning.microbatch_size,
        oracle=oracle,
        expectile=config.learning.expectile,
    )


def _checkpoint_payload(
    *,
    experiment: str,
    config: RunConfig,
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    critics: IndependentCritics,
    oracle: OracleCritics | None,
    optimizers: dict[str, torch.optim.Optimizer],
    policy_pool: PolicyPool,
    behavior: BehaviorSampler,
    sampler: BalancedReplaySampler,
    replay: TrajectoryReplay,
    iteration: int,
    critic_steps: int,
    actor_updates: int,
    policy_version: int,
    teacher_ready_streak: int,
    collector_next_seed: int,
    run_seed: int,
    fixed_seeds: tuple[int, ...],
    fresh_seeds: tuple[int, ...],
    baseline_fixed: dict[str, object],
    baseline_fresh: dict[str, object],
    best_fixed: dict[str, object],
    best_fresh: dict[str, object],
) -> dict[str, object]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "experiment": experiment,
        "run_config": config.state_dict(),
        "model_config": asdict(actor.config),
        "actor": actor.state_dict(),
        "reference": reference.state_dict(),
        "critics": critics.state_dict(),
        "oracle": None if oracle is None else oracle.state_dict(),
        "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
        "policy_pool": policy_pool.state_dict(),
        "behavior": behavior.state_dict(),
        "replay_sampler": sampler.state_dict(),
        "replay": replay.state_dict(),
        "iteration": iteration,
        "critic_steps": critic_steps,
        "actor_updates": actor_updates,
        "policy_version": policy_version,
        "teacher_ready_streak": teacher_ready_streak,
        "collector_next_seed": collector_next_seed,
        "run_seed": run_seed,
        "fixed_seeds": fixed_seeds,
        "fresh_seeds": fresh_seeds,
        "baseline_fixed": baseline_fixed,
        "baseline_fresh": baseline_fresh,
        "best_fixed": best_fixed,
        "best_fresh": best_fresh,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }


def _save_checkpoint(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replay = values.get("replay")
    if isinstance(replay, TrajectoryReplay):
        replay.save_manifest()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_checkpoint_payload(**values), temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = int(checkpoint.get("checkpoint_version", -1))
    if version == 2:
        raise ValueError(
            f"{path} is a v2 IQL/AWR checkpoint; v2 resume is unsupported "
            "because it predates reliable MC evidence and v3 validation state"
        )
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} is not an IQL/AWR teacher-gated v{CHECKPOINT_VERSION} "
            "checkpoint; pre-gate IQL/AWR and old PPO/counterfactual "
            "checkpoints are unsupported"
        )
    return checkpoint


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA; CPU fallback is unsupported")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    output_dir = args.resume.parent if args.resume is not None else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    dashboard_path = output_dir / "dashboard.html"
    if args.resume is None and metrics_path.exists():
        raise FileExistsError(
            f"{metrics_path} exists; select a new directory or use --resume"
        )

    def write(record: dict[str, object]) -> None:
        _write_record(
            metrics_path,
            dashboard_path,
            record,
            verbose=args.verbose_console,
        )

    if args.resume is not None:
        checkpoint = _load_checkpoint(args.resume, device)
        experiment = str(checkpoint["experiment"])
        if args.experiment is not None and args.experiment != experiment:
            raise ValueError("--experiment does not match the resumed checkpoint")
        config = RunConfig.from_state_dict(checkpoint["run_config"])  # type: ignore[arg-type]
        model_config = TransformerConfig(**checkpoint["model_config"])  # type: ignore[arg-type]
        actor = BloodFlowTransformer(model_config).to(device)
        reference = BloodFlowTransformer(model_config).to(device)
        actor.load_state_dict(checkpoint["actor"], strict=True)
        reference.load_state_dict(checkpoint["reference"], strict=True)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        critics = IndependentCritics(config.critics).to(device)
        critics.load_state_dict(checkpoint["critics"], strict=True)
        oracle = OracleCritics(config.critics).to(device) if experiment == "b" else None
        if oracle is not None:
            if checkpoint["oracle"] is None:
                raise ValueError("oracle experiment checkpoint has no oracle state")
            oracle.load_state_dict(checkpoint["oracle"], strict=True)
        optimizers = make_optimizers(actor, critics, config.learning, oracle)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])  # type: ignore[index]
        policy_pool = PolicyPool.from_state_dict(checkpoint["policy_pool"])  # type: ignore[arg-type]
        behavior = BehaviorSampler.from_state_dict(checkpoint["behavior"])  # type: ignore[arg-type]
        sampler = BalancedReplaySampler.from_state_dict(
            checkpoint["replay_sampler"]  # type: ignore[arg-type]
        )
        replay = TrajectoryReplay.load(output_dir / "replay")
        replay_state = checkpoint["replay"]
        if not isinstance(replay_state, dict):
            raise ValueError("checkpoint replay state is invalid")
        for key in (
            "next_trajectory_id",
            "next_target_id",
            "next_query_id",
            "next_shard_id",
        ):
            if int(replay.state_dict()[key]) != int(replay_state[key]):
                raise ValueError(f"checkpoint and replay manifest disagree on {key}")
        replay.cursor = int(replay_state["cursor"])
        iteration = int(checkpoint["iteration"])
        critic_steps = int(checkpoint["critic_steps"])
        actor_updates = int(checkpoint["actor_updates"])
        policy_version = int(checkpoint["policy_version"])
        teacher_ready_streak = int(checkpoint["teacher_ready_streak"])
        run_seed = int(checkpoint["run_seed"])
        fixed_seeds = tuple(int(value) for value in checkpoint["fixed_seeds"])
        fresh_seeds = tuple(int(value) for value in checkpoint["fresh_seeds"])
        baseline_fixed = checkpoint["baseline_fixed"]
        baseline_fresh = checkpoint["baseline_fresh"]
        best_fixed = checkpoint["best_fixed"]
        best_fresh = checkpoint["best_fresh"]
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in checkpoint["cuda_random_state"]]
        )
    else:
        experiment = args.experiment or "a"
        config = _config_from_args(args)
        if not args.sl_checkpoint.exists():
            raise FileNotFoundError(args.sl_checkpoint)
        reference = load_policy(args.sl_checkpoint, device)
        actor = clone_policy(reference, device)
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
        critics = IndependentCritics(config.critics).to(device)
        oracle = OracleCritics(config.critics).to(device) if experiment == "b" else None
        optimizers = make_optimizers(actor, critics, config.learning, oracle)
        policy_pool = PolicyPool(
            str(args.sl_checkpoint.resolve()),
            seed=args.seed + 11,
            config=config.opponents,
        )
        behavior = BehaviorSampler(seed=args.seed + 17)
        sampler = BalancedReplaySampler(seed=args.seed + 23)
        replay = TrajectoryReplay(
            output_dir / "replay", seed=args.seed + 29, config=config.replay
        )
        iteration = 0
        critic_steps = 0
        actor_updates = 0
        policy_version = 0
        teacher_ready_streak = 0
        run_seed = args.seed
        fixed_seeds = tuple(
            run_seed + 0xA51CE + index * 0x10001
            for index in range(config.evaluation_seed_count)
        )
        fresh_seeds = tuple(
            run_seed + 0xF12E5 + index * 0x20003
            for index in range(config.evaluation_seed_count)
        )

    executables = ExecutablePolicyPool(actor, reference, device)
    executables.sync(policy_pool)
    collector = FullTrajectoryCollector(
        config.collection,
        policy_pool,
        executables,
        behavior,
        device,
        seed=(
            int(checkpoint["collector_next_seed"])
            if args.resume is not None
            else run_seed + 0x100000
        ),
    )

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "run": config.state_dict(),
                "actor": asdict(actor.config),
                "fixed_seeds": fixed_seeds,
                "fresh_seeds": fresh_seeds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.resume is None:
        _synchronize()
        baseline_started = time.perf_counter()
        baseline_fixed = evaluate_panel(
            actor,
            reference,
            policy_pool,
            executables,
            behavior,
            device,
            seeds=fixed_seeds,
            games=config.evaluation_games,
            envs=config.collection.envs,
        )
        baseline_fresh = evaluate_panel(
            actor,
            reference,
            policy_pool,
            executables,
            behavior,
            device,
            seeds=fresh_seeds,
            games=config.evaluation_games,
            envs=config.collection.envs,
        )
        best_fixed = baseline_fixed
        best_fresh = baseline_fresh
        anchor = collector.collect(config.anchor_games)
        replay.add_trajectories(anchor.trajectories, anchor=True, trusted=True)
        replay_summary = replay.composition("train")
        write(
            {
                "phase": "baseline",
                "experiment": experiment,
                "fixed_evaluation": baseline_fixed,
                "fresh_evaluation": baseline_fresh,
                "replay_states": replay_summary["states"],
                "replay": replay_summary,
                "collection_seconds": anchor.elapsed_seconds,
                "environment_steps_per_second": anchor.environment_steps
                / max(anchor.elapsed_seconds, 1e-9),
                "elapsed_seconds": time.perf_counter() - baseline_started,
            }
        )
        warmup, warmup_seconds = _train_critics(
            replay,
            sampler,
            critics,
            optimizers,
            config,
            device,
            config.learning.initial_critic_steps,
            oracle=oracle,
            enable_oracle_distillation=False,
            show_progress=config.learning.initial_critic_steps >= 50,
        )
        critic_steps += config.learning.initial_critic_steps
        validation = _critic_validation(
            replay,
            critics,
            config,
            device,
            seed=run_seed + 31,
            oracle=oracle,
        )
        partial_ready, partial_gate = critic_ready(
            validation, critic_steps, config.learning
        )
        ready, gate, teacher_ready_streak = resolve_actor_gate(
            experiment,
            validation,
            critic_steps,
            config.learning,
            teacher_ready_streak,
        )
        write(
            {
                "phase": "critic_warmup",
                "critic_steps": critic_steps,
                "critic": warmup,
                "critic_validation": validation,
                "actor_ready": ready,
                "actor_gate": gate,
                "partial_critic_ready": partial_ready,
                "partial_critic_gate": partial_gate,
                "teacher_ready_streak": teacher_ready_streak,
                "oracle_distillation_active": False,
                "training_seconds": warmup_seconds,
            }
        )

    def checkpoint_values() -> dict[str, object]:
        return {
            "experiment": experiment,
            "config": config,
            "actor": actor,
            "reference": reference,
            "critics": critics,
            "oracle": oracle,
            "optimizers": optimizers,
            "policy_pool": policy_pool,
            "behavior": behavior,
            "sampler": sampler,
            "replay": replay,
            "iteration": iteration,
            "critic_steps": critic_steps,
            "actor_updates": actor_updates,
            "policy_version": policy_version,
            "teacher_ready_streak": teacher_ready_streak,
            "collector_next_seed": collector.next_seed,
            "run_seed": run_seed,
            "fixed_seeds": fixed_seeds,
            "fresh_seeds": fresh_seeds,
            "baseline_fixed": baseline_fixed,
            "baseline_fresh": baseline_fresh,
            "best_fixed": best_fixed,
            "best_fresh": best_fresh,
        }

    if args.resume is None:
        save_policy(output_dir / "best.pt", actor)
        _save_checkpoint(output_dir / "latest.pt", **checkpoint_values())

    interrupted = False
    try:
        while True:
            iteration += 1
            iteration_started = time.perf_counter()
            oracle_distillation_active = (
                experiment == "b"
                and teacher_ready_streak >= config.learning.teacher_readiness_streak
            )
            critic_stats, critic_seconds = _train_critics(
                replay,
                sampler,
                critics,
                optimizers,
                config,
                device,
                config.learning.critic_steps_per_iteration,
                oracle=oracle,
                enable_oracle_distillation=oracle_distillation_active,
            )
            critic_steps += config.learning.critic_steps_per_iteration
            validation = _critic_validation(
                replay,
                critics,
                config,
                device,
                seed=run_seed + 31,
                oracle=oracle,
            )
            partial_ready, partial_gate = critic_ready(
                validation, critic_steps, config.learning
            )

            # Experiment C must first accumulate and validate counterfactual
            # action evidence while Actor remains frozen. These targets join
            # Critic replay immediately and can only unlock a later Actor step.
            mc_record: dict[str, object] | None = None
            mc_validation_metrics: dict[str, object] | None = None
            mc_critic_stats: dict[str, float] | None = None
            mc_critic_seconds = 0.0
            mc_validation_status: dict[str, object] | None = None
            if experiment == "c" and partial_ready:
                targets, mc_stats = collect_mc_targets(
                    replay,
                    actor,
                    reference,
                    critics,
                    device,
                    config.mc,
                    split="train",
                    seed=run_seed + iteration * 0x100003,
                )
                replay.add_mc_targets(targets)
                mc_record = asdict(mc_stats)
                mc_validation_status = _mc_validation_corpus_status(
                    replay, config.learning
                )
                if (
                    iteration % config.mc_validation_every == 0
                    and not bool(mc_validation_status["validation_frozen"])
                ):
                    validation_targets, validation_mc = collect_mc_targets(
                        replay,
                        actor,
                        reference,
                        critics,
                        device,
                        config.mc,
                        split="validation",
                        seed=run_seed + iteration * 0x100019,
                        anchor_only=True,
                        exclude_existing_states=True,
                    )
                    replay.add_mc_targets(validation_targets)
                    mc_record["validation"] = asdict(validation_mc)
                    mc_validation_status = _mc_validation_corpus_status(
                        replay, config.learning
                    )

                mc_critic_stats, mc_critic_seconds = _train_mc_critics(
                    replay,
                    critics,
                    optimizers,
                    config,
                    device,
                    seed=run_seed + iteration * 0x10002D,
                )
                mc_record["critic_update"] = mc_critic_stats
                if mc_critic_stats is not None:
                    validation = _critic_validation(
                        replay,
                        critics,
                        config,
                        device,
                        seed=run_seed + 31,
                        oracle=oracle,
                    )
                    partial_ready, partial_gate = critic_ready(
                        validation, critic_steps, config.learning
                    )

            mc_train_targets = replay.mc_target_count("train")
            if experiment == "c":
                if mc_validation_status is None:
                    mc_validation_status = _mc_validation_corpus_status(
                        replay, config.learning
                    )
                mc_validation_targets = int(
                    mc_validation_status["validation_targets"]
                )
                mc_batch = replay.mc_validation_batch(
                    config.validation_states,
                    seed=run_seed + iteration * 0x100021,
                )
                if mc_batch is not None:
                    mc_validation_metrics = validate_critics(
                        critics,
                        mc_batch,
                        device,
                        microbatch_size=config.learning.microbatch_size,
                        expectile=config.learning.expectile,
                    )
                if mc_record is None:
                    mc_record = {}
                mc_record["train_targets"] = mc_train_targets
                mc_record.update(mc_validation_status)
                mc_record["validation_metrics"] = mc_validation_metrics
            else:
                mc_validation_targets = 0

            ready, gate, teacher_ready_streak = resolve_actor_gate(
                experiment,
                validation,
                critic_steps,
                config.learning,
                teacher_ready_streak,
                mc_validation=mc_validation_metrics,
                mc_train_targets=mc_train_targets,
                mc_validation_targets=mc_validation_targets,
            )
            if (
                experiment == "c"
                and partial_ready
                and mc_validation_status is not None
                and not bool(mc_validation_status["validation_frozen"])
            ):
                ready = False
                gate = _mc_validation_corpus_gate(
                    mc_validation_status, config.learning
                )
                teacher_ready_streak = 0
            teacher_candidate_ready: bool | None = None
            teacher_candidate_gate: str | None = None
            if experiment == "b":
                teacher_candidate_ready, teacher_candidate_gate = oracle_teacher_ready(
                    validation, config.learning
                )
            elif experiment == "c":
                teacher_candidate_ready, teacher_candidate_gate = mc_teacher_ready(
                    mc_validation_metrics,
                    train_targets=mc_train_targets,
                    validation_targets=mc_validation_targets,
                    config=config.learning,
                )
                if (
                    mc_validation_status is not None
                    and not bool(mc_validation_status["validation_frozen"])
                ):
                    teacher_candidate_ready = False
                    teacher_candidate_gate = _mc_validation_corpus_gate(
                        mc_validation_status, config.learning
                    )
            actor_stats: dict[str, float] | None = None
            actor_seconds = 0.0
            if ready:
                actor_stats, actor_seconds = _train_actor(
                    replay,
                    sampler,
                    actor,
                    reference,
                    critics,
                    optimizers,
                    config,
                    device,
                    oracle=oracle,
                    use_oracle_teacher=experiment == "b",
                )
                actor_updates += 1
                policy_version += 1
                policy_pool.update_current(
                    policy_version, artifact=None, update=actor_updates
                )
                executables.update_actor(actor)

            # New on-policy-ish trajectories are generated immediately after
            # every trusted Actor extraction (and also while Critic is held).
            collected = collector.collect(config.games_per_iteration)
            replay.add_trajectories(
                collected.trajectories, anchor=False, trusted=True
            )
            if mc_record is not None:
                mc_record["train_targets_after_trim"] = replay.mc_target_count(
                    "train"
                )

            snapshot = None
            if actor_stats is not None and policy_pool.snapshot_due(actor_updates):
                snapshot_path = (
                    output_dir
                    / "snapshots"
                    / f"policy-v{policy_version:06d}-a{actor_updates:06d}.pt"
                )
                save_policy(snapshot_path, actor)
                descriptor = policy_pool.add_snapshot(
                    update=actor_updates,
                    artifact=str(snapshot_path.resolve()),
                )
                executables.register_snapshot(descriptor, actor)
                executables.sync(policy_pool)
                snapshot = descriptor.state_dict()

            replay_summary = replay.composition("train")
            record: dict[str, object] = {
                "phase": "iteration",
                "experiment": experiment,
                "iteration": iteration,
                "critic_steps": critic_steps,
                "actor_updates": actor_updates,
                "policy_version": policy_version,
                "actor_ready": ready,
                "actor_gate": gate,
                "partial_critic_ready": partial_ready,
                "partial_critic_gate": partial_gate,
                "teacher_candidate_ready": teacher_candidate_ready,
                "teacher_candidate_gate": teacher_candidate_gate,
                "teacher_ready_streak": teacher_ready_streak,
                "oracle_distillation_active": oracle_distillation_active,
                "critic": critic_stats,
                "mc_critic": mc_critic_stats,
                "critic_validation": validation,
                "actor": actor_stats,
                "collection": {
                    "trajectories": len(collected.trajectories),
                    "environment_steps": collected.environment_steps,
                    "seconds": collected.elapsed_seconds,
                    "states_per_second": collected.environment_steps
                    / max(collected.elapsed_seconds, 1e-9),
                    "source_counts": collected.source_counts,
                },
                "replay_states": replay_summary["states"],
                "replay": replay_summary,
                "critic_seconds": critic_seconds,
                "mc_critic_seconds": mc_critic_seconds,
                "actor_seconds": actor_seconds,
                "training_states_per_second": (
                    config.learning.critic_batch_size
                    * config.learning.critic_steps_per_iteration
                    + (
                        config.learning.mc_critic_batch_size
                        * config.learning.mc_critic_steps_per_iteration
                        if mc_critic_stats is not None
                        else 0
                    )
                    + (
                        config.learning.actor_batch_size
                        * config.learning.actor_steps_per_iteration
                        if ready
                        else 0
                    )
                )
                / max(critic_seconds + mc_critic_seconds + actor_seconds, 1e-9),
                "mc": mc_record,
                "snapshot": snapshot,
                "iteration_seconds": time.perf_counter() - iteration_started,
            }

            evaluated = _should_evaluate(
                iteration,
                config.evaluation_every,
                smoke=args.smoke,
            )
            improved = False
            if evaluated:
                fixed_evaluation = evaluate_panel(
                    actor,
                    reference,
                    policy_pool,
                    executables,
                    behavior,
                    device,
                    seeds=fixed_seeds,
                    games=config.evaluation_games,
                    envs=config.collection.envs,
                )
                fresh_evaluation = evaluate_panel(
                    actor,
                    reference,
                    policy_pool,
                    executables,
                    behavior,
                    device,
                    seeds=fresh_seeds,
                    games=config.evaluation_games,
                    envs=config.collection.envs,
                )
                record["fixed_evaluation"] = fixed_evaluation
                record["fresh_evaluation"] = fresh_evaluation
                improved = better_on_both_panels(
                    fixed_evaluation,
                    fresh_evaluation,
                    best_fixed,
                    best_fresh,
                )
                if improved:
                    best_fixed = fixed_evaluation
                    best_fresh = fresh_evaluation
                record["best_fixed_evaluation"] = best_fixed
                record["best_fresh_evaluation"] = best_fresh

            write(record)
            if improved:
                save_policy(output_dir / "best.pt", actor)
            if iteration % config.checkpoint_every == 0 or evaluated:
                _save_checkpoint(output_dir / "latest.pt", **checkpoint_values())
            if args.smoke:
                break
    except KeyboardInterrupt:
        interrupted = True
    _save_checkpoint(output_dir / "latest.pt", **checkpoint_values())
    write(
        {
            "phase": "stopped",
            "iteration": iteration,
            "critic_steps": critic_steps,
            "actor_updates": actor_updates,
            "reason": "user_interrupt" if interrupted else "smoke_complete",
            "best_fixed_evaluation": best_fixed,
            "best_fresh_evaluation": best_fresh,
        }
    )


def main() -> None:
    args = build_parser().parse_args()
    print("CUDA eager mode; torch.compile disabled", flush=True)
    run(args)


if __name__ == "__main__":
    main()
