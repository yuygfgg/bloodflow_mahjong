"""End-to-end rollout, opponent scheduling, and PPO utilities."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import bloodflow_mahjong as bm

from .model import BloodFlowTransformer, HistoryKVCache, LayerKV
from .observation import unpack_action_masks


@dataclass(frozen=True)
class PPOConfig:
    envs: int = 2048
    rollout_transitions: int = 65_536
    ppo_epochs: int = 2
    minibatch: int = 4_096
    microbatch: int = 256
    learning_rate: float = 2e-4
    final_learning_rate: float = 3e-5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    final_entropy_coefficient: float = 0.002
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    shanten_coefficient: float = 0.02
    improving_coefficient: float = 0.01
    auxiliary_decay_fraction: float = 0.10
    rule_only_fraction: float = 0.10
    mixed_opponent_fraction: float = 0.35
    self_play_fraction: float = 0.75
    opponent_refresh_updates: int = 10
    frozen_snapshot_limit: int = 4


@dataclass
class EngineBuffers:
    batch: Any
    tile_obs: np.ndarray
    melds: np.ndarray
    river: np.ndarray
    meta: np.ndarray
    events: np.ndarray
    event_lengths: np.ndarray
    masks: np.ndarray
    records: np.ndarray
    actions: np.ndarray

    @classmethod
    def create(cls, batch_size: int, history: int = 192) -> EngineBuffers:
        return cls(
            batch=bm.Batch(batch_size, seed=1),
            tile_obs=np.empty((batch_size, 10, 27), dtype=np.uint8),
            melds=np.empty((batch_size, 4, 4, 3), dtype=np.uint8),
            river=np.empty((batch_size, 108, 2), dtype=np.uint8),
            meta=np.empty((batch_size, 34), dtype=np.int32),
            events=np.empty((batch_size, history, 8), dtype=np.int32),
            event_lengths=np.empty(batch_size, dtype=np.uint16),
            masks=np.empty((batch_size, 2), dtype=np.uint64),
            records=np.empty((batch_size, 12), dtype=np.int64),
            actions=np.empty(batch_size, dtype=np.uint8),
        )

    def refresh(self) -> None:
        self.batch.observe_into(self.tile_obs, self.melds, self.river, self.meta)
        self.batch.legal_action_masks_into(self.masks)
        self.batch.events_into(self.events, self.event_lengths)

    @property
    def legal_dense(self) -> np.ndarray:
        return unpack_action_masks(self.masks)


class TransitionStorage:
    """Preallocated learner transitions; pending slots are finalized in-place."""

    def __init__(self, capacity: int, history: int = 192) -> None:
        self.capacity = capacity
        self.next_slot = 0
        self.finalized = np.zeros(capacity, dtype=np.bool_)
        self.env = np.empty(capacity, dtype=np.int32)
        self.episode = np.empty(capacity, dtype=np.int64)
        self.tile_obs = np.empty((capacity, 10, 27), dtype=np.uint8)
        self.melds = np.empty((capacity, 4, 4, 3), dtype=np.uint8)
        self.meta = np.empty((capacity, 34), dtype=np.int32)
        self.events = np.empty((capacity, history, 8), dtype=np.int32)
        self.event_lengths = np.empty(capacity, dtype=np.int64)
        self.legal = np.empty((capacity, 115), dtype=np.bool_)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.logprob = np.empty(capacity, dtype=np.float32)
        self.value = np.empty(capacity, dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_value = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.shanten = np.full(capacity, 127, dtype=np.int8)
        self.improving = np.zeros(capacity, dtype=np.uint32)

    def add_many(
        self,
        envs: np.ndarray,
        episode: np.ndarray,
        buffers: EngineBuffers,
        legal: np.ndarray,
        actions: np.ndarray,
        logprob: np.ndarray,
        value: np.ndarray,
        shanten: np.ndarray,
        improving: np.ndarray,
    ) -> np.ndarray:
        count = len(envs)
        end = self.next_slot + count
        if end > self.capacity:
            raise RuntimeError("rollout transition storage capacity was exceeded")
        slots = np.arange(self.next_slot, end, dtype=np.int64)
        self.next_slot = end
        self.env[slots] = envs
        self.episode[slots] = episode
        self.tile_obs[slots] = buffers.tile_obs[envs]
        self.melds[slots] = buffers.melds[envs]
        self.meta[slots] = buffers.meta[envs]
        self.events[slots] = buffers.events[envs]
        self.event_lengths[slots] = buffers.event_lengths[envs]
        self.legal[slots] = legal
        self.actions[slots] = actions
        self.logprob[slots] = logprob
        self.value[slots] = value
        self.shanten[slots] = shanten
        self.improving[slots] = improving
        return slots

    def finalize(self, slots: np.ndarray, next_value: np.ndarray, done: bool) -> None:
        valid = slots >= 0
        selected = slots[valid]
        self.next_value[selected] = next_value[valid]
        self.done[selected] = done
        self.finalized[selected] = True

    def indices(self, count: int) -> np.ndarray:
        selected = np.flatnonzero(self.finalized)
        if len(selected) < count:
            raise RuntimeError(f"only finalized {len(selected)} of {count} transitions")
        return selected[:count]

    def compute_gae(
        self,
        indices: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(self.capacity, dtype=np.float32)
        returns = np.zeros(self.capacity, dtype=np.float32)
        last: dict[tuple[int, int], float] = {}
        for slot in indices[::-1]:
            key = (int(self.env[slot]), int(self.episode[slot]))
            continuation = 0.0 if self.done[slot] else last.get(key, 0.0)
            delta = (
                float(self.reward[slot])
                + gamma * (1.0 - float(self.done[slot])) * float(self.next_value[slot])
                - float(self.value[slot])
            )
            advantage = (
                delta
                + gamma * gae_lambda * (1.0 - float(self.done[slot])) * continuation
            )
            advantages[slot] = advantage
            returns[slot] = advantage + self.value[slot]
            last[key] = advantage
        return advantages, returns


@dataclass(frozen=True)
class RolloutBatch:
    storage: TransitionStorage
    indices: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    opponent_counts: np.ndarray
    opponent_stage: str
    active_snapshot: int | None

    def __len__(self) -> int:
        return len(self.indices)

    def tensors(self, slots: np.ndarray, device: torch.device) -> dict[str, Tensor]:
        storage = self.storage
        return {
            "tile_obs": torch.as_tensor(storage.tile_obs[slots], device=device),
            "melds": torch.as_tensor(storage.melds[slots], device=device),
            "meta": torch.as_tensor(storage.meta[slots], device=device),
            "events": torch.as_tensor(storage.events[slots], device=device),
            "event_lengths": torch.as_tensor(
                storage.event_lengths[slots], device=device
            ),
            "legal": torch.as_tensor(storage.legal[slots], device=device),
            "actions": torch.as_tensor(storage.actions[slots], device=device),
            "old_logprob": torch.as_tensor(storage.logprob[slots], device=device),
            "old_value": torch.as_tensor(storage.value[slots], device=device),
            "advantages": torch.as_tensor(self.advantages[slots], device=device),
            "returns": torch.as_tensor(self.returns[slots], device=device),
            "shanten": torch.as_tensor(storage.shanten[slots], device=device),
            "improving": torch.as_tensor(storage.improving[slots], device=device),
        }


class OpponentPool:
    """Four executable opponent policies and their transition curriculum.

    Opponent types are assigned per *absolute seat*, not once per game.  This
    gives the learner mixed three-player tables while keeping every policy
    action actor-viewer scoped.  A rollout uses one selected frozen snapshot,
    so the transformer inference path remains batched.
    """

    RANDOM_HU = 0
    RULE_FAST = 1
    RULE_SAFE = 2
    FROZEN_TRANSFORMER = 3
    NAMES = ("random_hu", "rule_fast", "rule_safe", "frozen_transformer")

    def __init__(self, config: PPOConfig, seed: int = 1) -> None:
        self.config = config
        self.random = np.random.default_rng(seed)
        self.snapshots: list[BloodFlowTransformer] = []
        self.active_snapshot: int | None = None

    @property
    def frozen_model(self) -> BloodFlowTransformer | None:
        if self.active_snapshot is None:
            return None
        return self.snapshots[self.active_snapshot]

    @property
    def frozen_ready(self) -> bool:
        return self.frozen_model is not None

    def stage(self, progress: float) -> str:
        if progress < self.config.rule_only_fraction:
            return "bootstrap"
        if progress < self.config.mixed_opponent_fraction:
            return "mixed"
        if progress < self.config.self_play_fraction:
            return "league"
        return "self_play"

    def probabilities(self, progress: float) -> np.ndarray:
        """Probability of the four policies for each non-learner seat."""
        if not self.frozen_ready or progress < self.config.rule_only_fraction:
            return np.array([0.30, 0.50, 0.20, 0.0], dtype=np.float64)
        if progress < self.config.mixed_opponent_fraction:
            return np.array([0.10, 0.35, 0.20, 0.35], dtype=np.float64)
        if progress < self.config.self_play_fraction:
            return np.array([0.05, 0.15, 0.15, 0.65], dtype=np.float64)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def assign_seats(self, learner_seats: np.ndarray, progress: float) -> np.ndarray:
        """Return opponent kinds with shape ``[env, absolute_seat]``.

        The learner slot is ``-1`` and must never be consumed by
        :meth:`actions`.  Sampling the remaining seats independently is
        intentional: each environment contains a small mixed opponent table.
        """
        learner_seats = np.asarray(learner_seats, dtype=np.int64)
        kinds = np.full((len(learner_seats), 4), -1, dtype=np.int8)
        probabilities = self.probabilities(progress)
        for seat in range(4):
            rows = np.flatnonzero(learner_seats != seat)
            kinds[rows, seat] = self.random.choice(
                len(self.NAMES), size=len(rows), p=probabilities
            ).astype(np.int8)
        return kinds

    def refresh_snapshot(
        self,
        model: BloodFlowTransformer,
        device: torch.device,
        progress: float,
    ) -> int | None:
        """Freeze the learner and select a current or historical snapshot.

        Snapshot creation starts only after the rule-only stage.  During the
        mixed stage the latest snapshot is preferred.  In the league stage a
        quarter of rollouts deliberately use a retained historical snapshot;
        this protects against immediately overfitting the newest policy while
        requiring only one frozen forward batch per rollout.
        """
        if progress < self.config.rule_only_fraction:
            return None
        self.snapshots.append(clone_model(model, device))
        overflow = len(self.snapshots) - self.config.frozen_snapshot_limit
        if overflow > 0:
            del self.snapshots[:overflow]

        latest = len(self.snapshots) - 1
        if self.stage(progress) in ("league", "self_play") and len(self.snapshots) > 1:
            if self.random.random() < 0.25:
                self.active_snapshot = int(self.random.integers(0, latest))
            else:
                self.active_snapshot = latest
        else:
            self.active_snapshot = latest
        return self.active_snapshot

    def set_frozen(self, model: BloodFlowTransformer | None) -> None:
        """Install a frozen model, primarily for tests and explicit resume."""
        self.snapshots.clear()
        self.active_snapshot = None
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.snapshots.append(model)
            self.active_snapshot = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.random.bit_generator.state,
            "active_snapshot": self.active_snapshot,
            "snapshots": [snapshot.state_dict() for snapshot in self.snapshots],
        }

    def load_state_dict(
        self,
        state: dict[str, Any] | None,
        model_config: Any,
        device: torch.device,
    ) -> None:
        if not state:
            return
        self.random.bit_generator.state = state["rng_state"]
        self.snapshots = []
        for snapshot_state in state.get("snapshots", []):
            snapshot = BloodFlowTransformer(model_config).to(device)
            snapshot.load_state_dict(snapshot_state)
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
            self.snapshots.append(snapshot)
        active = state.get("active_snapshot")
        self.active_snapshot = (
            int(active)
            if active is not None and int(active) < len(self.snapshots)
            else None
        )

    def random_actions(self, masks: np.ndarray, dense: np.ndarray) -> np.ndarray:
        actions = np.empty(len(masks), dtype=np.uint8)
        for row, legal in enumerate(dense):
            choices = np.flatnonzero(legal)
            if bm.ACTION_HU < len(legal) and legal[bm.ACTION_HU]:
                actions[row] = bm.ACTION_HU
            else:
                actions[row] = self.random.choice(choices)
        return actions

    def rule_actions(self, batch: Any) -> np.ndarray:
        actions = np.empty(len(batch), dtype=np.uint8)
        batch.simple_rule_actions_into(actions)
        return actions

    def safe_actions(
        self,
        actions: np.ndarray,
        dense: np.ndarray,
        tile_obs: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        """Make R2 more defensive without hidden information.

        The engine R2 action remains the structural default.  This variant
        declines optional exposed calls and only overrides a discard when a
        substantially more public tile is available, using locked tiles and
        visible rivers as a crude safety signal.
        """
        for row in rows:
            action = int(actions[row])
            legal = dense[row]
            if (
                action in (bm.ACTION_PONG, bm.ACTION_EXPOSED_KONG)
                and legal[bm.ACTION_PASS]
            ):
                actions[row] = bm.ACTION_PASS
                continue
            discard_start = bm.ACTION_DISCARD_OFFSET
            discard_stop = bm.ACTION_HU
            legal_discards = (
                np.flatnonzero(legal[discard_start:discard_stop]) + discard_start
            )
            if not len(legal_discards) or not (discard_start <= action < discard_stop):
                continue
            exposure = tile_obs[row, 2:10].sum(axis=0, dtype=np.int16)
            base_exposure = exposure[action - discard_start]
            tiles = legal_discards - discard_start
            safest = legal_discards[np.argmax(exposure[tiles])]
            if exposure[safest - discard_start] >= base_exposure + 2:
                actions[row] = safest

    def actions(
        self,
        buffers: EngineBuffers,
        seat_kinds: np.ndarray,
        device: torch.device,
        history_cache: HistoryCacheStore | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        dense = buffers.legal_dense
        actions = self.rule_actions(buffers.batch)
        actors = buffers.meta[:, 1].astype(np.int64)
        active = actors >= 0
        rows = np.flatnonzero(active)
        kinds = np.full(len(actions), -1, dtype=np.int8)
        kinds[rows] = seat_kinds[rows, actors[rows]]
        random_rows = np.flatnonzero(kinds == self.RANDOM_HU)
        if len(random_rows):
            actions[random_rows] = self.random_actions(
                buffers.masks[random_rows], dense[random_rows]
            )

        safe_rows = np.flatnonzero(kinds == self.RULE_SAFE)
        if len(safe_rows):
            self.safe_actions(actions, dense, buffers.tile_obs, safe_rows)

        frozen = np.flatnonzero(kinds == self.FROZEN_TRANSFORMER)
        model = self.frozen_model
        if len(frozen) and model is not None:
            sampled, _, _ = infer_actions(
                model,
                buffers,
                frozen,
                device,
                deterministic,
                history_cache,
            )
            actions[frozen] = sampled
        return actions


class HistoryCacheStore:
    """Per-environment viewer caches for one fixed model version.

    Event lengths differ between asynchronous games, so cached rows are
    grouped by exact ``(past_length, delta_length)``.  A row whose viewer
    history was truncated or changed is rebuilt with a full forward.  Keeping
    this state for one rollout only also guarantees that updated learner
    weights never consume stale K/V tensors.
    """

    def __init__(self, max_history: int = 192) -> None:
        self.max_history = max_history
        self.caches: dict[tuple[int, int], HistoryKVCache] = {}
        self.last_events: dict[tuple[int, int], np.ndarray] = {}

    def clear_rows(self, rows: np.ndarray) -> None:
        row_set = {int(row) for row in rows}
        for key in list(self.caches):
            if key[0] in row_set:
                self.caches.pop(key, None)
                self.last_events.pop(key, None)

    @staticmethod
    def _split_cache(cache: HistoryKVCache, index: int) -> HistoryKVCache:
        return HistoryKVCache(
            tuple(
                LayerKV(layer.key[index : index + 1], layer.value[index : index + 1])
                for layer in cache.layers
            ),
            cache.length,
            cache.summary[index : index + 1],
        )

    def infer(
        self,
        model: BloodFlowTransformer,
        buffers: EngineBuffers,
        rows: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(rows) == 0:
            empty = np.empty(0, dtype=np.uint8)
            return empty, empty.astype(np.float32), empty.astype(np.float32)

        dense = buffers.legal_dense[rows]
        actors = buffers.meta[rows, 1].astype(np.int64)
        requests: dict[tuple[int, int, int], list[tuple[int, int, int, np.ndarray]]] = (
            {}
        )
        full_rows: list[int] = []
        for row, actor in zip(rows.tolist(), actors.tolist()):
            key = (int(row), int(actor))
            length = int(buffers.event_lengths[row])
            cache = self.caches.get(key)
            previous = self.last_events.get(key)
            can_append = (
                cache is not None
                and previous is not None
                and length <= self.max_history
                and cache.length == len(previous)
                and length >= cache.length
                and np.array_equal(buffers.events[row, : cache.length], previous)
            )
            if length == 0 or length > self.max_history:
                full_rows.append(row)
                self.caches.pop(key, None)
                self.last_events.pop(key, None)
                continue
            if can_append:
                start = cache.length
                request_key = (cache.length, length - start, 1)
                new_events = buffers.events[row, start:length]
            else:
                self.caches.pop(key, None)
                request_key = (-1, length, 0)
                new_events = buffers.events[row, :length]
            requests.setdefault(request_key, []).append(
                (row, actor, length, new_events)
            )

        actions = np.empty(len(rows), dtype=np.uint8)
        logprobs = np.empty(len(rows), dtype=np.float32)
        values = np.empty(len(rows), dtype=np.float32)
        row_positions = {int(row): index for index, row in enumerate(rows.tolist())}

        def consume(
            result_rows: list[int], output: Any, result_cache: HistoryKVCache | None
        ) -> None:
            distribution = torch.distributions.Categorical(logits=output.logits)
            sampled = (
                output.logits.argmax(dim=-1) if deterministic else distribution.sample()
            )
            positions = [row_positions[row] for row in result_rows]
            actions[positions] = sampled.cpu().numpy().astype(np.uint8)
            logprobs[positions] = distribution.log_prob(sampled).float().cpu().numpy()
            values[positions] = output.value.float().cpu().numpy()
            if result_cache is not None:
                for index, row in enumerate(result_rows):
                    actor = int(buffers.meta[row, 1])
                    key = (row, actor)
                    self.caches[key] = self._split_cache(result_cache, index)
                    length = int(buffers.event_lengths[row])
                    self.last_events[key] = buffers.events[row, :length].copy()

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if full_rows:
                output = model(
                    torch.as_tensor(buffers.tile_obs[full_rows], device=device),
                    torch.as_tensor(buffers.melds[full_rows], device=device),
                    torch.as_tensor(buffers.meta[full_rows], device=device),
                    torch.as_tensor(buffers.events[full_rows], device=device),
                    torch.as_tensor(
                        buffers.event_lengths[full_rows].astype(np.int64), device=device
                    ),
                    torch.as_tensor(buffers.legal_dense[full_rows], device=device),
                )
                consume(full_rows, output, None)

            for (past_length, delta_length, cached), entries in requests.items():
                result_rows = [entry[0] for entry in entries]
                event_batch = torch.as_tensor(
                    np.stack([entry[3] for entry in entries]), device=device
                )
                if cached:
                    past_caches = [
                        self.caches[(row, int(buffers.meta[row, 1]))]
                        for row in result_rows
                    ]
                    stacked = HistoryKVCache(
                        tuple(
                            LayerKV(
                                torch.cat(
                                    [cache.layers[layer].key for cache in past_caches],
                                    dim=0,
                                ),
                                torch.cat(
                                    [
                                        cache.layers[layer].value
                                        for cache in past_caches
                                    ],
                                    dim=0,
                                ),
                            )
                            for layer in range(len(past_caches[0].layers))
                        ),
                        past_length,
                        torch.cat([cache.summary for cache in past_caches], dim=0),
                    )
                else:
                    stacked = None
                output, next_cache = model.forward_cached(
                    torch.as_tensor(buffers.tile_obs[result_rows], device=device),
                    torch.as_tensor(buffers.melds[result_rows], device=device),
                    torch.as_tensor(buffers.meta[result_rows], device=device),
                    event_batch,
                    stacked,
                    torch.as_tensor(buffers.legal_dense[result_rows], device=device),
                )
                consume(result_rows, output, next_cache)

        return actions, logprobs, values


def infer_actions(
    model: BloodFlowTransformer,
    buffers: EngineBuffers,
    rows: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
    history_cache: HistoryCacheStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(rows) == 0:
        return (
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    if history_cache is not None:
        return history_cache.infer(model, buffers, rows, device, deterministic)
    dense = buffers.legal_dense[rows]
    inputs = (
        torch.as_tensor(buffers.tile_obs[rows], device=device),
        torch.as_tensor(buffers.melds[rows], device=device),
        torch.as_tensor(buffers.meta[rows], device=device),
        torch.as_tensor(buffers.events[rows], device=device),
        torch.as_tensor(buffers.event_lengths[rows].astype(np.int64), device=device),
        torch.as_tensor(dense, device=device),
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(*inputs)
        distribution = torch.distributions.Categorical(logits=output.logits)
        actions = (
            output.logits.argmax(dim=-1) if deterministic else distribution.sample()
        )
        logprob = distribution.log_prob(actions)
    return (
        actions.cpu().numpy().astype(np.uint8),
        logprob.float().cpu().numpy(),
        output.value.float().cpu().numpy(),
    )


class RolloutCollector:
    def __init__(
        self,
        config: PPOConfig,
        device: torch.device,
        seed: int = 1,
    ) -> None:
        self.config = config
        self.device = device
        self.buffers = EngineBuffers.create(config.envs, history=192)
        self.pool = OpponentPool(config, seed)
        self.learner_seats = np.arange(config.envs, dtype=np.uint8) % 4
        self.episode_ids = np.zeros(config.envs, dtype=np.int64)
        self.opponent_kinds = self.pool.assign_seats(self.learner_seats, 0.0)
        self._opponent_counts = np.zeros(len(OpponentPool.NAMES), dtype=np.int64)
        self.learner_history_cache = HistoryCacheStore(max_history=192)
        self.frozen_history_cache = HistoryCacheStore(max_history=192)
        self.next_seed = 1

    def _assign_opponents(self, rows: np.ndarray, progress: float) -> None:
        assigned = self.pool.assign_seats(self.learner_seats[rows], progress)
        self.opponent_kinds[rows] = assigned
        selected = assigned[assigned >= 0]
        self._opponent_counts += np.bincount(
            selected, minlength=len(OpponentPool.NAMES)
        )

    def _reset_rows(self, rows: np.ndarray, progress: float) -> None:
        if len(rows) == 0:
            return
        seeds = np.asarray(
            [
                ((self.next_seed + int(row)) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                for row in rows
            ],
            dtype=np.uint64,
        )
        self.next_seed += len(rows)
        self.buffers.batch.reset_many(rows.astype(np.uint32), seeds)
        self.episode_ids[rows] += 1
        self.learner_seats[rows] = np.asarray(
            [int(episode) % 4 for episode in self.episode_ids[rows]], dtype=np.uint8
        )
        self._assign_opponents(rows, progress)

    def collect(
        self,
        model: BloodFlowTransformer,
        transitions: int,
        progress: float,
    ) -> RolloutBatch:
        model.eval()
        self.buffers.batch.reset_all(self.next_seed)
        self.next_seed += self.config.envs
        self.learner_seats = np.arange(self.config.envs, dtype=np.uint8) % 4
        self.episode_ids.fill(0)
        self.opponent_kinds = np.full((self.config.envs, 4), -1, dtype=np.int8)
        self._opponent_counts.fill(0)
        self.learner_history_cache = HistoryCacheStore(max_history=192)
        self.frozen_history_cache = HistoryCacheStore(max_history=192)
        self._assign_opponents(np.arange(self.config.envs), progress)
        self.buffers.refresh()

        storage = TransitionStorage(transitions + self.config.envs * 2)
        pending = np.full(self.config.envs, -1, dtype=np.int64)
        finished = 0
        while finished < transitions:
            actors = self.buffers.meta[:, 1]
            learner_rows = np.flatnonzero(actors == self.learner_seats.astype(np.int32))
            learner_actions, learner_logprob, learner_value = infer_actions(
                model,
                self.buffers,
                learner_rows,
                self.device,
                history_cache=self.learner_history_cache,
            )

            if len(learner_rows):
                previous = pending[learner_rows]
                storage.finalize(previous, learner_value, False)
                finished += int(np.count_nonzero(previous >= 0))
            if finished >= transitions:
                break

            dense = self.buffers.legal_dense[learner_rows]
            shanten = np.full(len(learner_rows), 127, dtype=np.int8)
            improving = np.zeros(len(learner_rows), dtype=np.uint32)
            if len(learner_rows):
                all_shanten = np.empty(self.config.envs, dtype=np.int8)
                all_improving = np.empty(self.config.envs, dtype=np.uint32)
                self.buffers.batch.hand_analysis_into(all_shanten, all_improving)
                shanten[:] = all_shanten[learner_rows]
                improving[:] = all_improving[learner_rows]
                pending[learner_rows] = storage.add_many(
                    learner_rows,
                    self.episode_ids[learner_rows],
                    self.buffers,
                    dense,
                    learner_actions.astype(np.int64),
                    learner_logprob,
                    learner_value,
                    shanten,
                    improving,
                )

            self.buffers.actions[:] = self.pool.actions(
                self.buffers,
                self.opponent_kinds,
                self.device,
                history_cache=self.frozen_history_cache,
            )
            self.buffers.actions[learner_rows] = learner_actions
            self.buffers.batch.step_and_observe_into(
                self.buffers.actions,
                self.buffers.records,
                self.buffers.masks,
                self.buffers.tile_obs,
                self.buffers.melds,
                self.buffers.river,
                self.buffers.meta,
            )

            rows = np.arange(self.config.envs)
            reward = (
                self.buffers.records[rows, 5 + self.learner_seats].astype(np.float32)
                / 10_000.0
            )
            active_pending = pending >= 0
            storage.reward[pending[active_pending]] += reward[active_pending]

            terminal = self.buffers.records[:, 11].astype(bool)
            terminal_slots = pending[terminal]
            storage.finalize(
                terminal_slots, np.zeros(len(terminal_slots), dtype=np.float32), True
            )
            finished += int(np.count_nonzero(terminal_slots >= 0))
            pending[terminal] = -1
            terminal_rows = np.flatnonzero(terminal)
            self.learner_history_cache.clear_rows(terminal_rows)
            self.frozen_history_cache.clear_rows(terminal_rows)
            self._reset_rows(terminal_rows, progress)
            self.buffers.batch.events_into(
                self.buffers.events, self.buffers.event_lengths
            )
            if len(terminal_rows):
                self.buffers.batch.observe_into(
                    self.buffers.tile_obs,
                    self.buffers.melds,
                    self.buffers.river,
                    self.buffers.meta,
                )
                self.buffers.batch.legal_action_masks_into(self.buffers.masks)
                self.buffers.batch.events_into(
                    self.buffers.events, self.buffers.event_lengths
                )

        indices = storage.indices(transitions)
        advantages, returns = storage.compute_gae(
            indices, self.config.gamma, self.config.gae_lambda
        )
        selected_advantages = advantages[indices]
        selected_advantages = (selected_advantages - selected_advantages.mean()) / (
            selected_advantages.std() + 1e-8
        )
        advantages[indices] = selected_advantages
        return RolloutBatch(
            storage,
            indices,
            advantages,
            returns,
            self._opponent_counts.copy(),
            self.pool.stage(progress),
            self.pool.active_snapshot,
        )


def categorical_return_targets(returns: Tensor, support: Tensor) -> Tensor:
    values = returns.float().clamp(float(support[0]), float(support[-1]))
    scale = (values - support[0]) / (support[1] - support[0])
    lower = scale.floor().long().clamp(0, len(support) - 1)
    upper = scale.ceil().long().clamp(0, len(support) - 1)
    upper_weight = scale - lower.float()
    lower_weight = 1.0 - upper_weight
    result = torch.zeros(
        (len(values), len(support)), device=returns.device, dtype=torch.float32
    )
    result.scatter_add_(1, lower[:, None], lower_weight[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


def cosine_learning_rate(config: PPOConfig, progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.final_learning_rate + weight * (
        config.learning_rate - config.final_learning_rate
    )


def ppo_update(
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PPOConfig,
    device: torch.device,
    progress: float,
) -> dict[str, float]:
    model.train()
    count = len(rollout)
    permutation = np.random.permutation(count)
    aux_scale = max(0.0, 1.0 - progress / max(config.auxiliary_decay_fraction, 1e-6))
    entropy_scale = config.entropy_coefficient + progress * (
        config.final_entropy_coefficient - config.entropy_coefficient
    )
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0

    for _ in range(config.ppo_epochs):
        for start in range(0, count, config.minibatch):
            batch_positions = permutation[start : start + config.minibatch]
            if len(batch_positions) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            minibatch_kl = 0.0
            minibatch_size = len(batch_positions)
            for micro_start in range(0, minibatch_size, config.microbatch):
                positions = batch_positions[
                    micro_start : micro_start + config.microbatch
                ]
                slots = rollout.indices[positions]
                data = rollout.tensors(slots, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(
                        data["tile_obs"],
                        data["melds"],
                        data["meta"],
                        data["events"],
                        data["event_lengths"],
                        data["legal"],
                    )
                    distribution = torch.distributions.Categorical(logits=output.logits)
                    logprob = distribution.log_prob(data["actions"])
                    ratio = (logprob - data["old_logprob"]).exp()
                    clipped = ratio.clamp(
                        1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                    )
                    policy_loss = -torch.minimum(
                        ratio * data["advantages"], clipped * data["advantages"]
                    ).mean()
                    target_distribution = categorical_return_targets(
                        data["returns"], model.value_support
                    )
                    value_loss = (
                        -(
                            target_distribution
                            * torch.log_softmax(output.value_logits.float(), dim=-1)
                        )
                        .sum(dim=-1)
                        .mean()
                    )
                    legal_count = data["legal"].sum(dim=-1).clamp_min(2).float()
                    entropy = (distribution.entropy() / legal_count.log()).mean()
                    valid_shanten = (
                        (data["shanten"] >= -1)
                        & (data["shanten"] <= 8)
                        & (data["meta"][:, 20] == 0)
                    )
                    shanten_loss = torch.zeros((), device=device)
                    improving_loss = torch.zeros((), device=device)
                    if valid_shanten.any():
                        shanten_loss = torch.nn.functional.cross_entropy(
                            output.shanten_logits[valid_shanten],
                            (data["shanten"][valid_shanten] + 1).long(),
                        )
                        bits = torch.arange(27, device=device)
                        target_improving = (
                            data["improving"].long()[valid_shanten, None] >> bits
                        ) & 1
                        improving_loss = (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                output.improving_logits[valid_shanten],
                                target_improving.float(),
                            )
                        )
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - entropy_scale * entropy
                        + aux_scale
                        * (
                            config.shanten_coefficient * shanten_loss
                            + config.improving_coefficient * improving_loss
                        )
                    )
                scale = len(positions) / minibatch_size
                (loss * scale).backward()
                minibatch_kl += (
                    float((data["old_logprob"] - logprob).mean().detach()) * scale
                )
                sums["policy_loss"] += float(policy_loss.detach()) * scale
                sums["value_loss"] += float(value_loss.detach()) * scale
                sums["entropy"] += float(entropy.detach()) * scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            sums["approx_kl"] += minibatch_kl
            updates += 1
            if minibatch_kl > config.target_kl:
                break
        if sums["approx_kl"] / max(updates, 1) > config.target_kl:
            break

    return {key: value / max(updates, 1) for key, value in sums.items()} | {
        "updates": float(updates),
        "aux_scale": aux_scale,
        "entropy_scale": entropy_scale,
    }


def clone_model(
    model: BloodFlowTransformer, device: torch.device
) -> BloodFlowTransformer:
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def save_checkpoint(
    path: Path,
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    update: int,
    transitions: int,
    config: PPOConfig,
    opponent_pool: OpponentPool | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "transitions": transitions,
            "ppo_config": asdict(config),
            "opponent_pool": (
                opponent_pool.state_dict() if opponent_pool is not None else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if next(model.parameters()).device.type == "cuda"
                else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: BloodFlowTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    opponent_pool: OpponentPool | None = None,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if opponent_pool is not None:
        opponent_pool.load_state_dict(
            checkpoint.get("opponent_pool"), model.config, device
        )
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
    return int(checkpoint["update"]), int(checkpoint["transitions"])


def evaluate_against_rules(
    model: BloodFlowTransformer,
    device: torch.device,
    games: int = 256,
    envs: int = 64,
    seed: int = 0xA51CE,
) -> dict[str, float]:
    """Deterministic learner evaluation against the engine rule policy."""
    envs = max(1, min(envs, games))
    buffers = EngineBuffers.create(envs, history=192)
    buffers.batch.reset_all(seed)
    buffers.refresh()
    learner_seats = np.arange(envs, dtype=np.uint8) % 4
    cumulative = np.zeros((envs, 4), dtype=np.int64)
    completed_scores: list[tuple[int, int]] = []
    pool = OpponentPool(PPOConfig(envs=envs), seed + 1)
    model.eval()
    next_seed = seed + envs
    while len(completed_scores) < games:
        actors = buffers.meta[:, 1]
        learner_rows = np.flatnonzero(actors == learner_seats.astype(np.int32))
        actions = pool.rule_actions(buffers.batch)
        learner_actions, _, _ = infer_actions(
            model, buffers, learner_rows, device, deterministic=True
        )
        actions[learner_rows] = learner_actions
        buffers.actions[:] = actions
        buffers.batch.step_and_observe_into(
            buffers.actions,
            buffers.records,
            buffers.masks,
            buffers.tile_obs,
            buffers.melds,
            buffers.river,
            buffers.meta,
        )
        cumulative += buffers.records[:, 5:9]
        terminal = buffers.records[:, 11].astype(bool)
        for row in np.flatnonzero(terminal):
            scores = 10_000 + cumulative[row]
            own = int(scores[learner_seats[row]])
            rank = 1 + int(np.count_nonzero(scores > own))
            completed_scores.append((int(cumulative[row, learner_seats[row]]), rank))
        rows = np.flatnonzero(terminal)
        if len(rows):
            reset_seeds = np.asarray(
                [
                    ((next_seed + int(row)) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                    for row in rows
                ],
                dtype=np.uint64,
            )
            next_seed += len(rows)
            buffers.batch.reset_many(rows.astype(np.uint32), reset_seeds)
            learner_seats[rows] = (learner_seats[rows] + 1) % 4
            cumulative[rows] = 0
            buffers.batch.observe_into(
                buffers.tile_obs, buffers.melds, buffers.river, buffers.meta
            )
            buffers.batch.legal_action_masks_into(buffers.masks)
        buffers.batch.events_into(buffers.events, buffers.event_lengths)

    values = np.asarray(
        [score for score, _ in completed_scores[:games]], dtype=np.float64
    )
    ranks = np.asarray([rank for _, rank in completed_scores[:games]], dtype=np.float64)
    return {
        "games": float(len(values)),
        "mean_score_delta": float(values.mean()),
        "score_std": float(values.std()),
        "first_rate": float(np.mean(ranks == 1)),
        "mean_rank": float(ranks.mean()),
    }
