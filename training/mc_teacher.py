"""Selective information-set Monte Carlo targets for uncertain replay states."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

import bloodflow_mahjong as bm

from .iql import IndependentCritics
from .learner import LearningBatch
from .model import BloodFlowTransformer
from .observation import unpack_action_masks
from .pipeline import ALL_SEATS_MASK, EngineBuffers, _autocast, _history_prefix
from .replay import MonteCarloTarget, ReplayIndex, TrajectoryReplay


@dataclass(frozen=True)
class MonteCarloConfig:
    queries_per_iteration: int = 16
    candidate_actions: int = 3
    hidden_worlds: int = 32
    continuations_per_world: int = 1
    candidate_pool_states: int = 2048
    confidence_z: float = 1.96
    maximum_confidence_half_width: float = 0.25
    minimum_reliable_action_gap: float = 0.02
    maximum_rollout_steps: int = 2048

    def __post_init__(self) -> None:
        positive = (
            self.queries_per_iteration,
            self.candidate_actions,
            self.hidden_worlds,
            self.continuations_per_world,
            self.candidate_pool_states,
            self.confidence_z,
            self.maximum_confidence_half_width,
            self.maximum_rollout_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Monte Carlo budgets and confidence values must be positive")
        if not 2 <= self.candidate_actions <= 4:
            raise ValueError("candidate_actions must be between two and four")
        if (
            not math.isfinite(self.minimum_reliable_action_gap)
            or self.minimum_reliable_action_gap < 0
        ):
            raise ValueError(
                "minimum_reliable_action_gap must be finite and non-negative"
            )


@dataclass(frozen=True)
class MonteCarloStatistics:
    queries: int
    accepted_queries: int
    accepted_targets: int
    terminal_rollouts: int
    rollout_states: int
    elapsed_seconds: float
    mean_variance: float
    mean_confidence_half_width: float


def _batch_for_engine(batch: object, history: int = 192) -> EngineBuffers:
    size = len(batch)  # type: ignore[arg-type]
    buffers = EngineBuffers.create(size, history)
    buffers.batch = batch
    buffers.batch.observe_into(
        buffers.tile_obs, buffers.melds, buffers.river, buffers.meta
    )
    buffers.batch.legal_action_masks_into(buffers.masks)
    buffers.batch.events_into(buffers.events, buffers.event_lengths)
    buffers.refresh_legal()
    return buffers


@torch.no_grad()
def _model_actions(
    model: BloodFlowTransformer,
    buffers: EngineBuffers,
    device: torch.device,
) -> np.ndarray:
    lengths = buffers.event_lengths.astype(np.int64)
    with _autocast(device):
        output = model(
            torch.as_tensor(buffers.tile_obs, device=device),
            torch.as_tensor(buffers.melds, device=device),
            torch.as_tensor(buffers.meta, device=device),
            torch.as_tensor(
                _history_prefix(buffers.events, lengths), device=device
            ),
            torch.as_tensor(lengths, device=device),
            torch.as_tensor(buffers.legal, device=device),
        )
    return output.logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)


def _reconstruct_batch(replay: TrajectoryReplay, trajectory_id: int, step: int) -> object:
    entry = next(
        (item for item in replay.entries if item.trajectory_id == trajectory_id),
        None,
    )
    if entry is None:
        raise ValueError(f"unknown trajectory {trajectory_id}")
    if not 0 <= step < len(entry.trajectory):
        raise ValueError("query step is outside its trajectory")
    batch = bm.Batch(1, seed=int(entry.trajectory.seed))
    records = np.empty((1, int(bm.STEP_RECORD_WIDTH)), dtype=np.int64)
    for action in entry.trajectory.actions[:step]:
        batch.step_into(np.asarray([action], dtype=np.uint8), records)
    return batch


@torch.no_grad()
def _query_candidates(
    batch: LearningBatch,
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    critics: IndependentCritics,
    device: torch.device,
    count: int,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = batch.tensors(device)
    state = (
        tensors["tile_obs"],
        tensors["melds"],
        tensors["meta"],
        tensors["events"],
        tensors["event_lengths"],
    )
    actor.eval()
    reference.eval()
    critics.eval()
    with _autocast(device):
        policy = actor(*state, tensors["legal"])
        frozen = reference(*state, tensors["legal"])
        values = critics(*state, tensors["legal"])
    legal = tensors["legal"]
    q1 = values.q1.raw_values.float().masked_fill(~legal, -torch.inf)
    q2 = values.q2.raw_values.float().masked_fill(~legal, -torch.inf)
    conservative = torch.minimum(q1, q2)
    finite_disagreement = (q1 - q2).abs().masked_fill(~legal, 0.0)
    disagreement = finite_disagreement.max(dim=-1).values
    actor_actions = policy.logits.argmax(dim=-1)
    reference_actions = frozen.logits.argmax(dim=-1)
    if batch.rule_actions is None:
        raise ValueError("MC candidate selection requires rule actions")
    rule_actions = torch.as_tensor(
        batch.rule_actions.astype(np.int64, copy=False), device=device
    )
    rule_is_legal = legal.gather(1, rule_actions[:, None]).squeeze(1)
    torch._assert_async(rule_is_legal.all(), "rule candidate must be legal")
    policy_disagreement = (
        (actor_actions != reference_actions).float()
        + (actor_actions != rule_actions).float()
        + (reference_actions != rule_actions).float()
    ) / 3.0
    best_advantage = conservative.max(dim=-1).values - values.value.float()
    near_zero = torch.exp(-best_advantage.abs() / 0.10)
    impact = (tensors["meta"][:, 4] < 40).float()
    score = disagreement + 0.5 * policy_disagreement + 0.25 * near_zero + 0.25 * impact
    # A counterfactual query only carries ranking information when the player
    # has at least two legal choices. Forced states are common (especially in
    # exchange/response transitions), but cannot form a valid MC query group.
    decision_rows = legal.sum(dim=-1) >= 2
    query_count = min(count, int(decision_rows.sum().item()))
    if query_count == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, candidate_count), dtype=np.int64),
        )
    score = score.masked_fill(~decision_rows, -torch.inf)
    selected = score.topk(query_count).indices
    candidates = torch.full(
        (query_count, candidate_count), -1, dtype=torch.int64, device=device
    )
    for output, row in enumerate(selected.tolist()):
        choices: list[int] = []
        for action in (
            int(actor_actions[row]),
            int(reference_actions[row]),
            int(rule_actions[row]),
        ):
            if action not in choices:
                choices.append(action)
        for action in conservative[row].argsort(descending=True).tolist():
            if bool(legal[row, action]) and action not in choices:
                choices.append(int(action))
            if len(choices) >= candidate_count:
                break
        choices = choices[:candidate_count]
        candidates[output, : len(choices)] = torch.as_tensor(
            choices, device=device
        )
    return selected.cpu().numpy(), candidates.cpu().numpy()


def _rollout_candidates(
    source_batch: object,
    actor_seat: int,
    candidates: np.ndarray,
    model: BloodFlowTransformer,
    device: torch.device,
    config: MonteCarloConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    candidates = candidates[candidates >= 0].astype(np.uint8)
    worlds = config.hidden_worlds * config.continuations_per_world
    world_seeds = np.asarray(
        [
            ((seed + world) * 0xD1B54A32D192ED03) & ((1 << 64) - 1)
            for world in range(worlds)
        ],
        dtype=np.uint64,
    )
    indices = np.zeros(len(candidates) * worlds, dtype=np.uint32)
    seeds = np.tile(world_seeds, len(candidates))
    branches = source_batch.resample_information_sets(indices, seeds)
    first_actions = np.repeat(candidates, worlds)
    records = np.empty((len(first_actions), 12), dtype=np.int64)
    branches.step_into(first_actions, records)
    returns = records[:, 5 + actor_seat].astype(np.float32) / 10_000.0
    terminal = records[:, 11].astype(np.bool_)
    branch_ids = np.arange(len(first_actions), dtype=np.int64)
    active_ids = branch_ids[~terminal]
    active = (
        None
        if not len(active_ids)
        else branches.clone_indices(np.flatnonzero(~terminal).astype(np.uint32))
    )
    rollout_states = len(first_actions)
    steps = 0
    while active is not None and len(active_ids):
        steps += 1
        if steps > config.maximum_rollout_steps:
            raise RuntimeError("Monte Carlo continuation exceeded engine step limit")
        buffers = _batch_for_engine(active)
        actions = _model_actions(model, buffers, device)
        records = np.empty((len(actions), 12), dtype=np.int64)
        active.step_into(actions, records)
        returns[active_ids] += (
            records[:, 5 + actor_seat].astype(np.float32) / 10_000.0
        )
        terminal = records[:, 11].astype(np.bool_)
        rollout_states += len(actions)
        if terminal.all():
            break
        keep = np.flatnonzero(~terminal).astype(np.uint32)
        active_ids = active_ids[keep]
        active = active.clone_indices(keep)
    return returns.reshape(len(candidates), worlds), rollout_states


def _reliable_action_graph(
    candidates: list[int],
    outcomes: np.ndarray,
    config: MonteCarloConfig,
) -> tuple[tuple[int, ...], ...]:
    """Return statistically reliable counterparts for every candidate action."""

    if outcomes.ndim != 2 or outcomes.shape[0] != len(candidates):
        raise ValueError("candidate outcomes must have shape [candidates, worlds]")
    if outcomes.shape[1] < 2:
        return tuple(() for _ in candidates)
    if len(set(candidates)) != len(candidates):
        raise ValueError("MC candidate actions must be unique")

    values = outcomes.astype(np.float64, copy=False)
    reliable: list[set[int]] = [set() for _ in candidates]
    worlds = values.shape[1]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            differences = values[left] - values[right]
            mean_difference = float(differences.mean())
            variance = float(differences.var(ddof=1))
            half_width = config.confidence_z * math.sqrt(variance / worlds)
            if (
                half_width <= config.maximum_confidence_half_width
                and abs(mean_difference) - half_width
                >= config.minimum_reliable_action_gap
            ):
                reliable[left].add(candidates[right])
                reliable[right].add(candidates[left])
    return tuple(tuple(sorted(actions)) for actions in reliable)


def collect_mc_targets(
    replay: TrajectoryReplay,
    actor: BloodFlowTransformer,
    reference: BloodFlowTransformer,
    critics: IndependentCritics,
    device: torch.device,
    config: MonteCarloConfig,
    *,
    split: str,
    seed: int,
    anchor_only: bool = False,
    exclude_existing_states: bool = False,
) -> tuple[list[MonteCarloTarget], MonteCarloStatistics]:
    """Query only high-uncertainty non-response states using paired worlds."""

    if split not in ("train", "validation"):
        raise ValueError("MC split must be train or validation")
    index = replay.index(  # type: ignore[arg-type]
        split,
        include_mc=False,
        anchor_only=anchor_only,
    )
    if not len(index):
        raise RuntimeError(f"{split} replay is empty")
    # Response windows fix all concealed hands in the engine sampler to preserve
    # pending legality. Skip them so every MC query truly varies hidden hands.
    eligible = ~np.isin(index.categories, (7, 8))
    eligible_rows = np.flatnonzero(eligible)
    if exclude_existing_states and len(eligible_rows):
        entries = {entry.trajectory_id: entry for entry in replay.entries}
        existing = {
            (target.trajectory_id, target.step_index)
            for target in replay.mc_targets
            if target.split == split
            and (
                not anchor_only
                or entries[target.trajectory_id].anchor
            )
        }
        if existing:
            eligible_rows = np.asarray(
                [
                    row
                    for row in eligible_rows
                    if (
                        int(index.trajectory_ids[row]),
                        int(index.step_indices[row]),
                    )
                    not in existing
                ],
                dtype=np.int64,
            )
    if not len(eligible_rows):
        return [], MonteCarloStatistics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
    random = np.random.default_rng(seed)
    pool_size = min(config.candidate_pool_states, len(eligible_rows))
    pool_rows = random.choice(eligible_rows, pool_size, replace=False)
    pool_batch = replay.materialize(
        index,
        pool_rows,
        include_oracle=False,
        include_rule_actions=True,
    )
    selected, candidate_matrix = _query_candidates(
        pool_batch,
        actor,
        reference,
        critics,
        device,
        config.queries_per_iteration,
        config.candidate_actions,
    )
    targets: list[MonteCarloTarget] = []
    variances: list[float] = []
    half_widths: list[float] = []
    rollout_states = 0
    terminal_rollouts = 0
    accepted_queries = 0
    started = time.perf_counter()
    for query_offset, selected_row in enumerate(selected):
        original_row = int(pool_rows[selected_row])
        trajectory_id = int(index.trajectory_ids[original_row])
        step = int(index.step_indices[original_row])
        source_batch = _reconstruct_batch(replay, trajectory_id, step)
        actor_seat = int(pool_batch.meta[selected_row, 1])
        candidates = [
            int(action)
            for action in candidate_matrix[query_offset]
            if action >= 0
        ]
        # Keep this boundary defensive even though _query_candidates filters
        # forced states. No incomplete query may reach replay construction.
        if len(candidates) < 2:
            continue
        outcomes, states = _rollout_candidates(
            source_batch,
            actor_seat,
            np.asarray(candidates, dtype=np.int64),
            actor,
            device,
            config,
            seed=seed + query_offset * 0x10001,
        )
        rollout_states += states
        terminal_rollouts += outcomes.size
        entry = next(
            item for item in replay.entries if item.trajectory_id == trajectory_id
        )
        reliable_actions = _reliable_action_graph(candidates, outcomes, config)
        participating = [
            candidate
            for candidate, counterparts in zip(candidates, reliable_actions)
            if counterparts
        ]
        if len(participating) < 2:
            continue
        participating_set = set(participating)
        query_targets: list[MonteCarloTarget] = []
        for action, samples in zip(candidates, outcomes):
            mean = float(samples.mean())
            variance = float(samples.var(ddof=1)) if len(samples) > 1 else 0.0
            half_width = config.confidence_z * math.sqrt(
                variance / max(len(samples), 1)
            )
            variances.append(variance)
            half_widths.append(half_width)
            if action not in participating_set:
                continue
            action_index = candidates.index(action)
            query_targets.append(
                MonteCarloTarget(
                    target_id=0,
                    query_id=query_offset,
                    candidate_count=len(participating),
                    trajectory_id=trajectory_id,
                    step_index=step,
                    action=action,
                    mean_return=mean,
                    variance=variance,
                    samples=len(samples),
                    confidence_low=mean - half_width,
                    confidence_high=mean + half_width,
                    split=entry.split,
                    reliable_actions=tuple(
                        counterpart
                        for counterpart in reliable_actions[action_index]
                        if counterpart in participating_set
                    ),
                )
            )
        if len(query_targets) == len(participating):
            targets.extend(query_targets)
            accepted_queries += 1
    elapsed = time.perf_counter() - started
    return targets, MonteCarloStatistics(
        queries=len(selected),
        accepted_queries=accepted_queries,
        accepted_targets=len(targets),
        terminal_rollouts=terminal_rollouts,
        rollout_states=rollout_states,
        elapsed_seconds=elapsed,
        mean_variance=float(np.mean(variances)) if variances else 0.0,
        mean_confidence_half_width=(
            float(np.mean(half_widths)) if half_widths else 0.0
        ),
    )
