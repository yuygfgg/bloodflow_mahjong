"""Combine cached world replicates and sweep search-target KL scales."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from . import batch_sweep, direction_generalization as dg, optimizer_sweep
from .pipeline import load_policy, save_policy
from .policy_iteration import (
    CounterfactualBatch,
    PolicyStateBatch,
    calibrate_direction,
    load_counterfactual_batch,
    load_policy_state_batch,
    one_step_direction,
    require_cuda,
    require_deterministic_actor,
)
from .policy_pool import CATEGORY_COUNT
from .search_policy_sweep import (
    DirectionSource,
    TargetSpec,
    _numeric,
    _prior_or_one_hot,
    _policy_pair,
    _pool_outcomes,
    build_search_policy_target,
    search_policy_target_metrics,
    target_specs_from_identity,
)
from .world_outcomes import (
    combine_world_replicates,
    load_world_outcome_corpus,
)


COMBINED_SWEEP_VERSION = 2


def default_split_target_specs() -> tuple[TargetSpec, ...]:
    return (
        TargetSpec("split-rank-both-m0", "split_rank_both", 0.0),
        TargetSpec("split-rank-both-m0p03125", "split_rank_both", 0.03125),
        TargetSpec("split-rank-both-m0p0625", "split_rank_both", 0.0625),
        TargetSpec("split-rank-agree-m0", "split_rank_agree", 0.0),
        TargetSpec("split-win-both-p0", "split_win_both", 0.0),
        TargetSpec("split-win-both-p0p0625", "split_win_both", 0.0625),
        TargetSpec("split-win-both-p0p125", "split_win_both", 0.125),
        TargetSpec("split-win-both-p0p1875", "split_win_both", 0.1875),
        TargetSpec("split-win-both-p0p25", "split_win_both", 0.25),
    )


def select_split_target_specs(names: Sequence[str]) -> tuple[TargetSpec, ...]:
    available = {spec.name: spec for spec in default_split_target_specs()}
    selected = tuple(str(name) for name in names)
    if len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError("split target names must be a unique supported subset")
    return tuple(available[name] for name in selected)


def build_split_consensus_target(
    spec: TargetSpec,
    combined,
    left,
    right,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if left.worlds != right.worlds or combined.worlds != left.worlds + right.worlds:
        raise ValueError("split targets need two equal world replicates")
    for name in PolicyStateBatch.__dataclass_fields__:
        if not (
            np.array_equal(getattr(left, name), getattr(right, name))
            and np.array_equal(getattr(left, name), getattr(combined, name))
        ):
            raise ValueError(f"split target states differ in {name}")
    if reference_probabilities.shape != combined.legal.shape or reference_actions.shape != (
        len(combined),
    ):
        raise ValueError("split target reference policy does not match outcomes")

    rows = np.arange(len(combined))
    legal = combined.legal
    alternatives = legal.copy()
    alternatives[rows, reference_actions] = False
    has_alternative = alternatives.any(axis=1)
    left_rank = left.rank_outcomes.astype(np.float32).mean(axis=2) / 2.0
    right_rank = right.rank_outcomes.astype(np.float32).mean(axis=2) / 2.0
    left_advantage = left_rank - left_rank[rows, reference_actions][:, None]
    right_advantage = right_rank - right_rank[rows, reference_actions][:, None]

    if spec.kind in {"split_rank_both", "split_rank_agree"}:
        pooled = 0.5 * (left_advantage + right_advantage)
        actions = np.where(alternatives, pooled, -np.inf).argmax(axis=1)
        changed = has_alternative
        if spec.kind == "split_rank_agree":
            left_actions = np.where(alternatives, left_advantage, -np.inf).argmax(
                axis=1
            )
            right_actions = np.where(alternatives, right_advantage, -np.inf).argmax(
                axis=1
            )
            changed &= (actions == left_actions) & (actions == right_actions)
        changed &= left_advantage[rows, actions] > spec.value
        changed &= right_advantage[rows, actions] > spec.value
    elif spec.kind == "split_win_both":
        def paired_win_rates(outcomes) -> np.ndarray:
            rank = outcomes.rank_outcomes.astype(np.float32) / 2.0
            baseline_rank = rank[rows, reference_actions]
            score = outcomes.score_outcomes
            baseline_score = score[rows, reference_actions]
            better_rank = rank > baseline_rank[:, None, :]
            worse_rank = rank < baseline_rank[:, None, :]
            equal_rank = ~(better_rank | worse_rank)
            better_score = score > baseline_score[:, None, :]
            equal_score = score == baseline_score[:, None, :]
            wins = np.where(
                better_rank | (equal_rank & better_score),
                1.0,
                np.where(equal_rank & equal_score, 0.5, 0.0),
            )
            return wins.mean(axis=2)

        left_win = paired_win_rates(left)
        right_win = paired_win_rates(right)
        pooled = 0.5 * (left_win + right_win)
        actions = np.where(alternatives, pooled, -np.inf).argmax(axis=1)
        threshold = 0.5 + spec.value
        changed = has_alternative
        changed &= left_win[rows, actions] > threshold
        changed &= right_win[rows, actions] > threshold
    else:
        raise ValueError(f"unsupported split target kind {spec.kind!r}")

    target = _prior_or_one_hot(reference_probabilities, actions, changed)
    target = np.where(legal, target, 0.0).astype(np.float32)
    target /= target.sum(axis=1, keepdims=True)
    metrics = search_policy_target_metrics(
        target, combined, reference_probabilities
    )
    metrics["split_selected_states"] = int(changed.sum())
    return target, metrics


def _prepare_output(directory: Path, identity: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if optimizer_sweep._json(path) != identity:
            raise ValueError("combined search-policy configuration differs")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    entries = [entry for entry in directory.iterdir() if entry != temporary]
    if entries:
        raise ValueError("combined search-policy output is non-empty")
    if temporary.exists():
        temporary.unlink()
    batch_sweep._atomic_json(path, identity)


def run(
    source_directory: Path,
    output_directory: Path,
    *,
    target_kls: Sequence[float] = (1e-5, 3e-5, 1e-4),
    learning_rate: float = 0.1,
    kl_search_steps: int = 28,
    split_target_names: Sequence[str] = (),
    only_split_targets: bool = False,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    source_directory = source_directory.resolve()
    source_identity = optimizer_sweep._json(source_directory / "config.json")
    kls = tuple(float(value) for value in target_kls)
    if (
        not kls
        or len(set(kls)) != len(kls)
        or any(not math.isfinite(value) or value <= 0 for value in kls)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or kl_search_steps <= 0
    ):
        raise ValueError("combined search-policy scales are invalid")
    batch_directory = Path(str(source_identity["input_directory"]))
    seeds = tuple(int(seed) for seed in source_identity["seeds"])
    if len(seeds) != 2 or source_identity.get("replicates") != ["a", "b"]:
        raise ValueError("source sweep must contain two seeds and A/B replicates")
    input_sweep = optimizer_sweep.load_input_sweep(batch_directory, seeds=seeds)
    source_specs = target_specs_from_identity(source_identity.get("targets"))
    split_specs = select_split_target_specs(split_target_names)
    specs = (*(() if only_split_targets else source_specs), *split_specs)
    if not specs:
        raise ValueError("combined search-policy sweep needs at least one target")
    identity = {
        "version": COMBINED_SWEEP_VERSION,
        "source_directory": str(source_directory),
        "source_identity_fingerprint": batch_sweep._fingerprint(source_identity),
        "input_identity_fingerprint": batch_sweep._fingerprint(input_sweep.identity),
        "seeds": list(seeds),
        "queries_per_category": int(source_identity["queries_per_category"]),
        "worlds_per_replicate": int(source_identity["worlds"]),
        "combined_worlds": 2 * int(source_identity["worlds"]),
        "world_sampling": source_identity["world_sampling"],
        "targets": [spec.identity() for spec in specs],
        "target_kls": list(kls),
        "optimizer": "sgd",
        "learning_rate": float(learning_rate),
        "kl_search_steps": int(kl_search_steps),
    }
    output_directory = output_directory.resolve()
    _prepare_output(output_directory, identity)
    result_path = output_directory / "result.json"
    if result_path.exists():
        cached = optimizer_sweep._json(result_path)
        if cached.get("identity") != identity:
            raise ValueError("cached combined search-policy identity differs")
        print(f"RESULT combined search-policy cached  {result_path}", flush=True)
        return cached

    resolved = require_cuda(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    config = input_sweep.config
    reference = None
    reference_hash = None
    outcomes = {}
    outcome_pairs = {}
    heldout: dict[int, CounterfactualBatch] = {}
    calibration_batches: list[PolicyStateBatch] = []
    visit_weights: dict[int, np.ndarray] = {}
    for seed in seeds:
        child = input_sweep.children[seed]
        child_identity = input_sweep.child_identities[seed]
        reference_path, _self_play_path = optimizer_sweep._checkpoint_paths(
            child_identity
        )
        if reference is None:
            reference = load_policy(reference_path, resolved, frozen=True)
            require_deterministic_actor(reference)
            reference_hash = str(child_identity["reference_sha256"])
        elif str(child_identity["reference_sha256"]) != reference_hash:
            raise ValueError("source seeds do not share a reference Actor")
        outcome_pairs[seed] = (
            load_world_outcome_corpus(
                source_directory / f"seed-{seed}" / "outcomes-a"
            ),
            load_world_outcome_corpus(
                source_directory / f"seed-{seed}" / "outcomes-b"
            ),
        )
        outcomes[seed] = combine_world_replicates(outcome_pairs[seed])
        heldout[seed] = load_counterfactual_batch(child / "shared" / "heldout.npz")
        calibration_batches.append(
            load_policy_state_batch(child / "shared" / "calibration.npz")
        )
        manifest = optimizer_sweep._json(child / "shared" / "manifest.json")
        weights = np.asarray(
            manifest["source_visit_frequencies"]["vector"], dtype=np.float64
        )
        if weights.shape != (CATEGORY_COUNT,) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("source visitation weights are invalid")
        visit_weights[seed] = weights
    assert reference is not None

    common_probe = dg._concatenate_states(calibration_batches)
    common_weights = np.stack([visit_weights[seed] for seed in seeds]).mean(axis=0)
    common_weights /= common_weights.sum()
    reference_probe = dg._policy_outputs(
        reference,
        common_probe,
        resolved,
        batch_size=config.inference_batch_size,
    )
    reference_heldout = {
        seed: dg._policy_outputs(
            reference,
            heldout[seed],
            resolved,
            batch_size=config.inference_batch_size,
        )
        for seed in seeds
    }
    reference_train = {
        seed: dg._policy_outputs(
            reference,
            outcomes[seed].counterfactual_batch(),
            resolved,
            batch_size=config.inference_batch_size,
        )
        for seed in seeds
    }
    pooled_outcomes = _pool_outcomes([outcomes[seed] for seed in seeds])
    pooled_pair = (
        _pool_outcomes([outcome_pairs[seed][0] for seed in seeds]),
        _pool_outcomes([outcome_pairs[seed][1] for seed in seeds]),
    )
    source_pairs = {
        **{f"seed-{seed}": outcome_pairs[seed] for seed in seeds},
        "pooled": pooled_pair,
    }
    sources = [
        DirectionSource(
            f"seed-{seed}",
            seed,
            "combined",
            outcomes[seed],
            outcomes[seed].counterfactual_batch(),
            reference_train[seed][0],
            reference_train[seed][1],
            visit_weights[seed],
        )
        for seed in seeds
    ] + [
        DirectionSource(
            "pooled",
            None,
            "combined",
            pooled_outcomes,
            pooled_outcomes.counterfactual_batch(),
            np.concatenate([reference_train[seed][0] for seed in seeds]),
            np.concatenate([reference_train[seed][1] for seed in seeds]),
            common_weights,
        )
    ]
    print(
        f"CUDA {torch.cuda.get_device_name(resolved)}  combined search-policy  "
        f"worlds {pooled_outcomes.worlds}  targets {len(specs)}  "
        f"KL {','.join(f'{value:g}' for value in kls)}",
        flush=True,
    )

    candidates: dict[str, object] = {}
    summaries: dict[str, object] = {}
    best_score = -math.inf
    best_name = None
    for spec in specs:
        spec_candidates: dict[str, object] = {}
        probe_outputs: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            kl: {} for kl in kls
        }
        best_actor_for_spec = None
        for source in sources:
            if spec.kind.startswith("split_"):
                left, right = source_pairs[source.name]
                target, target_metrics = build_split_consensus_target(
                    spec,
                    source.outcomes,
                    left,
                    right,
                    source.reference_probabilities,
                    source.reference_actions,
                )
            else:
                target, target_metrics = build_search_policy_target(
                    spec,
                    source.outcomes,
                    source.reference_probabilities,
                    source.reference_actions,
                )
            raw_actor, initial, candidate, optimizer_metrics = one_step_direction(
                reference,
                source.counterfactual,
                resolved,
                category_weights=source.category_weights,
                learning_rate=learning_rate,
                microbatch_size=config.microbatch_size,
                optimizer_name="sgd",
                objective="search_ce",
                policy_targets=target,
            )
            per_kl: dict[str, object] = {}
            for kl in kls:
                actor = copy.deepcopy(raw_actor)
                calibration = calibrate_direction(
                    actor,
                    reference,
                    initial,
                    candidate,
                    common_probe,
                    resolved,
                    category_weights=common_weights,
                    target_kl=kl,
                    batch_size=config.inference_batch_size,
                    search_steps=kl_search_steps,
                    maximum_scale=config.maximum_scale,
                )
                probe_outputs[kl][source.name] = dg._policy_outputs(
                    actor,
                    common_probe,
                    resolved,
                    batch_size=config.inference_batch_size,
                )
                heldout_q: dict[str, object] = {}
                for seed in seeds:
                    probabilities, actions = dg._policy_outputs(
                        actor,
                        heldout[seed],
                        resolved,
                        batch_size=config.inference_batch_size,
                    )
                    heldout_q[str(seed)] = dg._heldout_metrics(
                        probabilities,
                        actions,
                        reference_heldout[seed][0],
                        reference_heldout[seed][1],
                        heldout[seed],
                        visit_weights[seed],
                    )
                per_kl[f"{kl:g}"] = {
                    "calibration": calibration,
                    "heldout_q": heldout_q,
                }
                if source.seed is None:
                    save_policy(
                        output_directory
                        / "actors"
                        / f"{spec.name}-kl{kl:g}.pt",
                        actor,
                    )
                    score = min(
                        float(
                            heldout_q[str(seed)]["soft_rank_value"]["mean"]
                        )
                        for seed in seeds
                    )
                    if score > best_score:
                        best_score = score
                        best_name = f"{spec.name}-kl{kl:g}"
                        best_actor_for_spec = copy.deepcopy(actor)
                del actor
            spec_candidates[source.name] = {
                "target": target_metrics,
                "optimizer": optimizer_metrics,
                "scales": per_kl,
            }
            del raw_actor
            torch.cuda.empty_cache()

        per_kl_summary: dict[str, object] = {}
        for kl in kls:
            own: list[float] = []
            other: list[float] = []
            pooled: list[float] = []
            for source_seed in seeds:
                source = spec_candidates[f"seed-{source_seed}"]["scales"][f"{kl:g}"]
                for target_seed in seeds:
                    value = float(
                        source["heldout_q"][str(target_seed)]["soft_rank_value"][
                            "mean"
                        ]
                    )
                    (own if source_seed == target_seed else other).append(value)
            pooled_source = spec_candidates["pooled"]["scales"][f"{kl:g}"]
            pooled = [
                float(
                    pooled_source["heldout_q"][str(seed)]["soft_rank_value"][
                        "mean"
                    ]
                )
                for seed in seeds
            ]
            cross = _policy_pair(
                probe_outputs[kl][f"seed-{seeds[0]}"],
                probe_outputs[kl][f"seed-{seeds[1]}"],
                reference_probe[0],
                reference_probe[1],
                common_probe,
                common_weights,
            )["probability_delta_cosine"]
            per_kl_summary[f"{kl:g}"] = {
                "own_heldout": _numeric(own),
                "other_heldout": _numeric(other),
                "pooled_heldout": _numeric(pooled),
                "pooled_min": min(pooled),
                "cross_source_policy_cosine": cross,
            }
            print(
                f"SUMMARY {spec.name:24s} KL {kl:g}  "
                f"pooled {np.mean(pooled):+.6f} min {min(pooled):+.6f}  "
                f"cross-seed {cross:+.3f}",
                flush=True,
            )
        candidates[spec.name] = spec_candidates
        summaries[spec.name] = per_kl_summary
        if best_actor_for_spec is not None:
            save_policy(output_directory / "best-actor.pt", best_actor_for_spec)
            del best_actor_for_spec

    result = {
        "identity": identity,
        "candidates": candidates,
        "summary": summaries,
        "best_candidate": best_name,
        "best_minimum_heldout": best_score,
        "elapsed_seconds": time.perf_counter() - started,
    }
    batch_sweep._atomic_json(result_path, result)
    print(
        f"RESULT combined search-policy complete  best {best_name}  "
        f"minimum {best_score:+.6f}  {result_path}",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-kls", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4]
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--kl-search-steps", type=int, default=28)
    parser.add_argument(
        "--split-targets",
        nargs="+",
        default=[],
        choices=[spec.name for spec in default_split_target_specs()],
    )
    parser.add_argument("--only-split-targets", action="store_true")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        args.source_dir,
        args.output_dir,
        target_kls=args.target_kls,
        learning_rate=args.learning_rate,
        kl_search_steps=args.kl_search_steps,
        split_target_names=args.split_targets,
        only_split_targets=args.only_split_targets,
        device=args.device,
    )


if __name__ == "__main__":
    main()
