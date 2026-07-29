"""Balanced four-player tournament for policy-iteration history snapshots.

The reported rating is a Plackett-Luce skill, scaled to an Elo-like unit and
anchored at the frozen SL policy.  It is a relative, tournament-local rating:
the accompanying mean rank, score delta, and pairwise records remain the
primary diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import math
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import bloodflow_mahjong as bm

from .evaluation import atomic_json
from .model import BloodFlowTransformer, TransformerConfig
from .pipeline import (
    EngineBuffers,
    _PinnedPolicyStager,
    _launch_policy_actions,
    _safe_rule_actions,
)
from .policy_iteration import mix64, require_cuda
from .train import (
    CHECKPOINT_VERSION,
    MOMENTUM_CHECKPOINT_VERSION,
    STATELESS_CHECKPOINT_VERSION,
)


TOURNAMENT_VERSION = 1
ELO_SCALE = 400.0 / math.log(10.0)
RULE_FAST = -1
RULE_SAFE = -2
SUPPORTED_CHECKPOINT_VERSIONS = frozenset(
    {STATELESS_CHECKPOINT_VERSION, MOMENTUM_CHECKPOINT_VERSION, CHECKPOINT_VERSION}
)
AGENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class TournamentConfig:
    rounds_per_combination: int = 12
    envs: int = 512
    inference_batch_size: int = 128
    bootstrap_samples: int = 400
    seed: int = 20260728

    def __post_init__(self) -> None:
        if (
            self.rounds_per_combination <= 0
            or self.envs <= 0
            or self.inference_batch_size <= 0
            or self.bootstrap_samples <= 0
        ):
            raise ValueError("tournament sizes must be positive")


@dataclass(frozen=True)
class TournamentAgent:
    name: str
    digest: str
    model: BloodFlowTransformer | None
    rule: int | None = None

    def __post_init__(self) -> None:
        if not self.name or len(self.digest) != 64:
            raise ValueError("tournament agent identity is invalid")
        if (self.model is None) == (self.rule is None):
            raise ValueError("agent must be either a model or a rule")
        if self.rule is not None and self.rule not in (RULE_FAST, RULE_SAFE):
            raise ValueError("unknown tournament rule")


@dataclass(frozen=True)
class TournamentGames:
    lineups: np.ndarray
    ranks: np.ndarray
    scores: np.ndarray
    block_ids: np.ndarray
    seeds: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.lineups)
        if (
            self.lineups.shape != (count, 4)
            or self.ranks.shape != (count, 4)
            or self.scores.shape != (count, 4)
            or self.block_ids.shape != (count,)
            or self.seeds.shape != (count,)
        ):
            raise ValueError("tournament result shapes do not match")
        if not np.all(np.sort(self.ranks, axis=1) == np.arange(1, 5)):
            raise ValueError("every tournament game needs four unique ranks")


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_version(payload: Mapping[str, object], name: str) -> int:
    version = int(payload.get("version", -1))
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        supported = "/".join(f"v{item}" for item in sorted(SUPPORTED_CHECKPOINT_VERSIONS))
        raise ValueError(
            f"{name} requires a supported {supported} policy-iteration checkpoint"
        )
    return version


def _load_model_state(
    state: Mapping[str, torch.Tensor],
    config: TransformerConfig,
    device: torch.device,
) -> BloodFlowTransformer:
    model = BloodFlowTransformer(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _parse_extra_checkpoint(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if (
        not separator
        or not AGENT_NAME_PATTERN.fullmatch(name)
        or "__" in name
        or not path
    ):
        raise argparse.ArgumentTypeError(
            "extra checkpoint must be NAME=PATH; NAME may use letters, digits, "
            "underscore, dot, and dash"
        )
    return name, Path(path)


def _parse_extra_policy(value: str) -> tuple[str, Path]:
    try:
        return _parse_extra_checkpoint(value)
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(
            "extra policy must be NAME=PATH; NAME may use letters, digits, "
            "underscore, dot, and dash"
        ) from error


def load_history_agents(
    checkpoint: Path,
    device: torch.device,
    *,
    extra_checkpoints: Sequence[tuple[str, Path]] = (),
    extra_policies: Sequence[tuple[str, Path]] = (),
    selected_agents: Sequence[str] = (),
) -> tuple[tuple[TournamentAgent, ...], dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_version = _checkpoint_version(payload, "history tournament")
    required = {"actor", "model_config", "opponent_pool", "sl_checkpoint"}
    if not required <= set(payload):
        raise ValueError("checkpoint has no complete history pool")
    config = TransformerConfig(**payload["model_config"])
    pool = payload["opponent_pool"]
    snapshots = pool.get("snapshots") if isinstance(pool, dict) else None
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("checkpoint opponent pool is invalid")
    sl_path = Path(str(payload["sl_checkpoint"]))
    sl_payload = torch.load(sl_path, map_location="cpu", weights_only=False)
    if set(sl_payload) != {"model_config", "model"}:
        raise ValueError("frozen SL checkpoint is not Actor-only")
    if TransformerConfig(**sl_payload["model_config"]) != config:
        raise ValueError("frozen SL model config differs from the history pool")

    models: list[TournamentAgent] = []
    sl_state = sl_payload["model"]
    models.append(
        TournamentAgent(
            name="sl",
            digest=_state_digest(sl_state),
            model=_load_model_state(sl_state, config, device),
        )
    )
    seen = {models[0].digest}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "iteration",
            "digest",
            "actor",
        }:
            raise ValueError("checkpoint opponent snapshot is invalid")
        iteration = int(snapshot["iteration"])
        state = snapshot["actor"]
        digest = str(snapshot["digest"])
        if _state_digest(state) != digest:
            raise ValueError("checkpoint opponent digest does not match weights")
        if digest in seen:
            continue
        seen.add(digest)
        models.append(
            TournamentAgent(
                name=f"u{iteration:03d}",
                digest=digest,
                model=_load_model_state(state, config, device),
            )
        )
    actor_state = payload["actor"]
    actor_digest = _state_digest(actor_state)
    if actor_digest not in seen:
        seen.add(actor_digest)
        models.append(
            TournamentAgent(
                name=f"u{int(payload['next_iteration']) - 1:03d}",
                digest=actor_digest,
                model=_load_model_state(actor_state, config, device),
            )
        )
    extra_identity: list[dict[str, object]] = []
    for name, path in extra_checkpoints:
        extra_payload = torch.load(path, map_location="cpu", weights_only=False)
        extra_version = _checkpoint_version(extra_payload, f"extra checkpoint {name!r}")
        required_extra = {
            "actor",
            "model_config",
            "next_iteration",
            "engine_rules_version",
            "policy_execution_version",
        }
        if not required_extra <= set(extra_payload):
            raise ValueError(f"extra checkpoint {name!r} has no complete Actor")
        if TransformerConfig(**extra_payload["model_config"]) != config:
            raise ValueError(f"extra checkpoint {name!r} uses a different model config")
        if int(extra_payload["engine_rules_version"]) != int(
            payload["engine_rules_version"]
        ):
            raise ValueError(f"extra checkpoint {name!r} uses different engine rules")
        if int(extra_payload["policy_execution_version"]) != int(
            payload["policy_execution_version"]
        ):
            raise ValueError(
                f"extra checkpoint {name!r} uses different policy execution semantics"
            )
        state = extra_payload["actor"]
        digest = _state_digest(state)
        if digest in seen:
            raise ValueError(
                f"extra checkpoint {name!r} duplicates an existing tournament Actor"
            )
        seen.add(digest)
        models.append(
            TournamentAgent(
                name=name,
                digest=digest,
                model=_load_model_state(state, config, device),
            )
        )
        extra_identity.append(
            {
                "name": name,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": _sha256(path),
                "checkpoint_version": extra_version,
                "next_iteration": int(extra_payload["next_iteration"]),
                "actor_digest": digest,
            }
        )
    extra_policy_identity: list[dict[str, object]] = []
    for name, path in extra_policies:
        policy_payload = torch.load(path, map_location="cpu", weights_only=False)
        if set(policy_payload) != {"model_config", "model"}:
            raise ValueError(f"extra policy {name!r} is not Actor-only")
        if TransformerConfig(**policy_payload["model_config"]) != config:
            raise ValueError(f"extra policy {name!r} uses a different model config")
        state = policy_payload["model"]
        digest = _state_digest(state)
        if digest in seen:
            raise ValueError(
                f"extra policy {name!r} duplicates an existing tournament Actor"
            )
        seen.add(digest)
        models.append(
            TournamentAgent(
                name=name,
                digest=digest,
                model=_load_model_state(state, config, device),
            )
        )
        extra_policy_identity.append(
            {
                "name": name,
                "policy": str(path.resolve()),
                "policy_sha256": _sha256(path),
                "actor_digest": digest,
            }
        )
    agents = tuple(
        [
            *models,
            TournamentAgent(
                name="rule_fast",
                digest=hashlib.sha256(b"rule_fast/v1").hexdigest(),
                model=None,
                rule=RULE_FAST,
            ),
            TournamentAgent(
                name="rule_safe",
                digest=hashlib.sha256(b"rule_safe/v1").hexdigest(),
                model=None,
                rule=RULE_SAFE,
            ),
        ]
    )
    if selected_agents:
        requested = tuple(selected_agents)
        if len(requested) != len(set(requested)):
            raise ValueError("selected tournament agents must be unique")
        if "sl" not in requested:
            raise ValueError("selected tournament agents must include sl as the anchor")
        available = {agent.name for agent in agents}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                "selected tournament agents are unavailable: " + ", ".join(missing)
            )
        requested_set = set(requested)
        agents = tuple(agent for agent in agents if agent.name in requested_set)
    names = [agent.name for agent in agents]
    if len(names) != len(set(names)) or len(agents) < 4:
        raise ValueError("tournament agents must have unique names")
    identity = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_version": checkpoint_version,
        "next_iteration": int(payload["next_iteration"]),
        "engine_rules_version": int(payload["engine_rules_version"]),
        "policy_execution_version": int(payload["policy_execution_version"]),
        "extra_checkpoints": extra_identity,
        "extra_policies": extra_policy_identity,
        "agents": [
            {"name": agent.name, "digest": agent.digest} for agent in agents
        ],
    }
    return agents, identity


def balanced_schedule(
    agents: int, rounds_per_combination: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if agents < 4 or rounds_per_combination <= 0:
        raise ValueError("tournament schedule needs at least four agents and rounds")
    lineups: list[tuple[int, int, int, int]] = []
    block_ids: list[int] = []
    seeds: list[int] = []
    block = 0
    for combination in itertools.combinations(range(agents), 4):
        for repeat in range(rounds_per_combination):
            game_seed = mix64(seed + block)
            for lineup in itertools.permutations(combination):
                lineups.append(lineup)
                block_ids.append(block)
                seeds.append(game_seed)
            block += 1
    return (
        np.asarray(lineups, dtype=np.int8),
        np.asarray(block_ids, dtype=np.int32),
        np.asarray(seeds, dtype=np.uint64),
    )


def _history_masks(lineups: np.ndarray, model_agents: np.ndarray) -> np.ndarray:
    model_seats = model_agents[lineups]
    bits = np.left_shift(np.uint8(1), np.arange(4, dtype=np.uint8))
    return np.bitwise_or.reduce(
        np.where(model_seats, bits, np.uint8(0)), axis=1
    ).astype(np.uint8, copy=False)


def _model_actions(
    buffers: EngineBuffers,
    lineups: np.ndarray,
    agents: Sequence[TournamentAgent],
    device: torch.device,
    config: TournamentConfig,
    stagers: dict[int, _PinnedPolicyStager],
) -> np.ndarray:
    acting = buffers.meta[:, 1].astype(np.int64)
    if np.any((acting < 0) | (acting >= 4)):
        raise RuntimeError("tournament encountered a terminal row before reset")
    rows = np.arange(len(acting))
    controller = lineups[rows, acting].astype(np.int64)
    actions = buffers.actions
    rule_rows = np.asarray(
        [row for row, value in enumerate(controller) if agents[value].rule is not None],
        dtype=np.int64,
    )
    if len(rule_rows):
        enabled = np.zeros(len(actions), dtype=np.uint8)
        enabled[rule_rows] = 1
        buffers.batch.simple_rule_actions_masked_into(enabled, actions)
        safe_rows = rule_rows[
            np.asarray(
                [agents[controller[row]].rule == RULE_SAFE for row in rule_rows]
            )
        ]
        _safe_rule_actions(actions, buffers.legal, buffers.tile_obs, safe_rows)
    for agent_index, agent in enumerate(agents):
        if agent.model is None:
            continue
        selected = np.flatnonzero(controller == agent_index)
        if not len(selected):
            continue
        model_actions = _launch_policy_actions(
            agent.model,
            buffers,
            selected,
            device,
            inference_batch_size=config.inference_batch_size,
            stager=stagers[agent_index],
        ).cpu().numpy()
        if np.any(model_actions == np.iinfo(np.uint8).max):
            raise RuntimeError(f"{agent.name} produced non-finite logits")
        actions[selected] = model_actions
    if not buffers.legal[rows, actions.astype(np.int64)].all():
        raise RuntimeError("tournament controller selected an illegal action")
    return actions


def _play_batch(
    lineups: np.ndarray,
    seeds: np.ndarray,
    agents: Sequence[TournamentAgent],
    device: torch.device,
    config: TournamentConfig,
    stagers: dict[int, _PinnedPolicyStager],
) -> tuple[np.ndarray, np.ndarray]:
    history = agents[0].model.config.max_history
    buffers = EngineBuffers.create(len(seeds), history)
    masks = _history_masks(
        lineups, np.asarray([agent.model is not None for agent in agents])
    )
    reset = np.ones(len(seeds), dtype=np.uint8)
    buffers.batch.reset_and_observe_history_into(
        reset,
        seeds,
        masks,
        buffers.masks,
        buffers.tile_obs,
        buffers.melds,
        buffers.river,
        buffers.meta,
        buffers.events,
        buffers.event_lengths,
    )
    buffers.refresh_legal()
    original = np.arange(len(seeds), dtype=np.int64)
    active_lineups = lineups.copy()
    active_masks = masks.copy()
    scores = np.empty((len(seeds), 4), dtype=np.int64)
    cumulative = np.zeros((len(seeds), 4), dtype=np.int64)
    steps = np.zeros(len(seeds), dtype=np.int32)
    while len(original):
        actions = _model_actions(
            buffers, active_lineups, agents, device, config, stagers
        )
        buffers.batch.step_and_observe_history_into(
            actions,
            active_masks,
            buffers.records,
            buffers.masks,
            buffers.tile_obs,
            buffers.melds,
            buffers.river,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
        )
        buffers.refresh_legal()
        cumulative += buffers.records[:, 5:9]
        steps += 1
        if np.any(steps > 4096):
            raise RuntimeError("tournament game exceeded the step limit")
        terminal = buffers.records[:, 11].astype(bool)
        for row in np.flatnonzero(terminal):
            scores[original[row]] = 10_000 + cumulative[row]
        if not np.any(terminal):
            continue
        terminal_rows = np.flatnonzero(terminal)
        if len(terminal_rows) == len(original):
            break
        buffers, keep = buffers.remove_rows(terminal_rows)
        active_lineups = active_lineups[keep].copy()
        active_masks = active_masks[keep].copy()
        original = original[keep].copy()
        cumulative = cumulative[keep].copy()
        steps = steps[keep].copy()
    seats = np.arange(4)
    order = np.asarray(
        [np.lexsort((seats, -score)) for score in scores], dtype=np.int8
    )
    ranks = np.empty_like(order)
    ranks[np.arange(len(scores))[:, None], order] = np.arange(1, 5, dtype=np.int8)
    return ranks, scores - 10_000


def run_tournament(
    agents: Sequence[TournamentAgent],
    identity: dict[str, object],
    output_dir: Path,
    device: torch.device,
    config: TournamentConfig,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("tournament output directory must be empty")
    if any(
        agent.model is not None and agent.model.config != agents[0].model.config
        for agent in agents
    ):
        raise ValueError("all tournament models must share one configuration")
    lineups, block_ids, seeds = balanced_schedule(
        len(agents), config.rounds_per_combination, config.seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "version": TOURNAMENT_VERSION,
        "config": {
            "rounds_per_combination": config.rounds_per_combination,
            "envs": config.envs,
            "inference_batch_size": config.inference_batch_size,
            "bootstrap_samples": config.bootstrap_samples,
            "seed": config.seed,
        },
        **identity,
    }
    atomic_json(output_dir / "config.json", identity)
    _save_agent_archive(output_dir / "agents.pt", agents)
    stagers = {
        index: _PinnedPolicyStager(device, agent.model.config.max_history)
        for index, agent in enumerate(agents)
        if agent.model is not None and device.type == "cuda"
    }
    ranks = np.empty_like(lineups)
    scores = np.empty(lineups.shape, dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, len(lineups), config.envs):
        stop = min(start + config.envs, len(lineups))
        batch_ranks, batch_scores = _play_batch(
            lineups[start:stop],
            seeds[start:stop],
            agents,
            device,
            config,
            stagers,
        )
        ranks[start:stop] = batch_ranks
        scores[start:stop] = batch_scores
        print(f"TOURNAMENT {stop:,}/{len(lineups):,}", flush=True)
    games = TournamentGames(lineups, ranks, scores, block_ids, seeds)
    _save_games(output_dir / "games.npz", games)
    summary = summarize_tournament(
        games,
        [agent.name for agent in agents],
        anchor=0,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )
    summary["identity"] = identity
    summary["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(output_dir / "summary.json", summary)
    return summary


def _save_games(path: Path, games: TournamentGames) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            version=np.asarray([TOURNAMENT_VERSION], dtype=np.int64),
            lineups=games.lineups,
            ranks=games.ranks,
            scores=games.scores,
            block_ids=games.block_ids,
            seeds=games.seeds,
        )
    temporary.replace(path)


def _save_agent_archive(
    path: Path, agents: Sequence[TournamentAgent]
) -> None:
    entries: list[dict[str, object]] = []
    for agent in agents:
        if agent.model is None:
            continue
        entries.append(
            {
                "name": agent.name,
                "digest": agent.digest,
                "model_config": agent.model.config.__dict__,
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in agent.model.state_dict().items()
                },
            }
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"version": TOURNAMENT_VERSION, "agents": entries}, temporary)
    temporary.replace(path)


def load_tournament_games(path: Path) -> TournamentGames:
    expected = {"version", "lineups", "ranks", "scores", "block_ids", "seeds"}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("tournament game cache fields do not match")
        if int(payload["version"][0]) != TOURNAMENT_VERSION:
            raise ValueError("unsupported tournament game cache version")
        return TournamentGames(
            lineups=payload["lineups"].copy(),
            ranks=payload["ranks"].copy(),
            scores=payload["scores"].copy(),
            block_ids=payload["block_ids"].copy(),
            seeds=payload["seeds"].copy(),
        )


def _plackett_luce_ratings(
    lineups: np.ndarray,
    ranks: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    anchor: int,
) -> np.ndarray:
    lineups = np.asarray(lineups, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.int64)
    if lineups.shape != ranks.shape or lineups.ndim != 2 or lineups.shape[1] != 4:
        raise ValueError("ratings need [games, 4] lineups and ranks")
    agents = int(lineups.max()) + 1
    if not 0 <= anchor < agents:
        raise ValueError("rating anchor is invalid")
    game_weights = (
        np.ones(len(lineups), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if game_weights.shape != (len(lineups),) or np.any(game_weights < 0):
        raise ValueError("rating weights are invalid")
    order = np.argsort(ranks, axis=1)
    choices = np.take_along_axis(lineups, order, axis=1)
    free = np.asarray([index for index in range(agents) if index != anchor])
    ratings = np.zeros(agents, dtype=np.float64)
    ridge = 1e-8
    for _ in range(32):
        gradient = np.zeros(agents, dtype=np.float64)
        hessian = np.zeros((agents, agents), dtype=np.float64)
        for position in range(3):
            remaining = choices[:, position:]
            logits = ratings[remaining]
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            np.add.at(gradient, remaining[:, 0], game_weights)
            for left in range(remaining.shape[1]):
                np.add.at(
                    gradient,
                    remaining[:, left],
                    -game_weights * probabilities[:, left],
                )
                for right in range(remaining.shape[1]):
                    value = -game_weights * probabilities[:, left] * (
                        (left == right) - probabilities[:, right]
                    )
                    np.add.at(
                        hessian,
                        (remaining[:, left], remaining[:, right]),
                        value,
                    )
        gradient[free] -= ridge * ratings[free]
        hessian[np.ix_(free, free)] -= ridge * np.eye(len(free))
        step = np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
        ratings[free] -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    ratings -= ratings[anchor]
    return ratings


def _stratified_block_counts(
    games: TournamentGames, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    blocks, inverse = np.unique(games.block_ids, return_inverse=True)
    strata: dict[tuple[int, ...], list[int]] = {}
    for block_index in range(len(blocks)):
        rows = np.flatnonzero(inverse == block_index)
        lineups = np.sort(games.lineups[rows], axis=1)
        if not np.all(lineups == lineups[0]):
            raise ValueError("one seed block contains multiple agent combinations")
        key = tuple(int(value) for value in lineups[0])
        strata.setdefault(key, []).append(block_index)
    random = np.random.default_rng(seed)
    counts = np.zeros((samples, len(blocks)), dtype=np.int16)
    for sample in range(samples):
        for block_indices in strata.values():
            picked = random.choice(block_indices, size=len(block_indices), replace=True)
            counts[sample] += np.bincount(
                picked, minlength=len(blocks)
            ).astype(np.int16)
    return counts, inverse


def _block_bootstrap(
    games: TournamentGames,
    agents: int,
    anchor: int,
    block_counts: np.ndarray,
    inverse_blocks: np.ndarray,
) -> np.ndarray:
    results = np.empty((len(block_counts), agents), dtype=np.float64)
    for sample, counts in enumerate(block_counts):
        results[sample] = _plackett_luce_ratings(
            games.lineups,
            games.ranks,
            counts[inverse_blocks],
            anchor=anchor,
        )
    return results


def _bootstrap_mean(
    values: np.ndarray,
    inverse_blocks: np.ndarray,
    block_counts: np.ndarray,
) -> tuple[float, float, float]:
    means = np.empty(len(block_counts), dtype=np.float64)
    for sample, counts in enumerate(block_counts):
        means[sample] = np.average(values, weights=counts[inverse_blocks])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def summarize_tournament(
    games: TournamentGames,
    names: Sequence[str],
    *,
    anchor: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    agents = len(names)
    if agents < 4 or len(set(names)) != agents:
        raise ValueError("tournament names are invalid")
    if games.lineups.max() >= agents or games.lineups.min() < 0:
        raise ValueError("tournament lineup references an unknown agent")
    blocks = np.unique(games.block_ids)
    block_counts, inverse = _stratified_block_counts(
        games, bootstrap_samples, seed + 1
    )
    ratings = _plackett_luce_ratings(games.lineups, games.ranks, anchor=anchor)
    bootstrap = _block_bootstrap(
        games, agents, anchor, block_counts, inverse
    )
    entries: dict[str, dict[str, object]] = {}
    for index, name in enumerate(names):
        seat_rows, seats = np.where(games.lineups == index)
        rank_values = games.ranks[seat_rows, seats].astype(np.float64)
        score_values = games.scores[seat_rows, seats].astype(np.float64)
        rank_mean, rank_low, rank_high = _bootstrap_mean(
            rank_values,
            inverse[seat_rows],
            block_counts,
        )
        score_mean, score_low, score_high = _bootstrap_mean(
            score_values,
            inverse[seat_rows],
            block_counts,
        )
        elo = ratings[index] * ELO_SCALE
        elo_samples = bootstrap[:, index] * ELO_SCALE
        entries[name] = {
            "games": int(len(seat_rows)),
            "mean_rank": rank_mean,
            "mean_rank_ci95": [rank_low, rank_high],
            "mean_score_delta": score_mean,
            "mean_score_delta_ci95": [score_low, score_high],
            "first_rate": float(np.mean(rank_values == 1)),
            "last_rate": float(np.mean(rank_values == 4)),
            "elo_like": float(elo),
            "elo_like_ci95": [
                float(np.quantile(elo_samples, 0.025)),
                float(np.quantile(elo_samples, 0.975)),
            ],
        }
    pairwise: dict[str, dict[str, object]] = {}
    rating_comparisons: dict[str, dict[str, object]] = {}
    for left in range(agents):
        for right in range(left + 1, agents):
            rows = np.flatnonzero(
                np.any(games.lineups == left, axis=1)
                & np.any(games.lineups == right, axis=1)
            )
            left_seats = np.argmax(games.lineups[rows] == left, axis=1)
            right_seats = np.argmax(games.lineups[rows] == right, axis=1)
            left_ranks = games.ranks[rows, left_seats]
            right_ranks = games.ranks[rows, right_seats]
            score_delta = (
                games.scores[rows, left_seats] - games.scores[rows, right_seats]
            ).astype(np.float64)
            pairwise[f"{names[left]}__{names[right]}"] = {
                "games": int(len(rows)),
                "left_win_rate": float(np.mean(left_ranks < right_ranks)),
                "left_mean_rank_delta": float(np.mean(left_ranks - right_ranks)),
                "left_mean_score_delta": float(score_delta.mean()),
            }
            elo_delta = (ratings[left] - ratings[right]) * ELO_SCALE
            bootstrap_delta = (
                bootstrap[:, left] - bootstrap[:, right]
            ) * ELO_SCALE
            rating_comparisons[f"{names[left]}__{names[right]}"] = {
                "left_elo_like_delta": float(elo_delta),
                "left_elo_like_delta_ci95": [
                    float(np.quantile(bootstrap_delta, 0.025)),
                    float(np.quantile(bootstrap_delta, 0.975)),
                ],
                "left_stronger_probability": float(
                    np.mean(bootstrap_delta > 0.0)
                ),
            }
    order = sorted(range(agents), key=lambda index: ratings[index], reverse=True)
    tiers: list[list[str]] = []
    current = [order[0]]
    for previous, next_index in zip(order, order[1:]):
        probability = float(np.mean(bootstrap[:, previous] > bootstrap[:, next_index]))
        if probability >= 0.975:
            tiers.append([names[index] for index in current])
            current = [next_index]
        else:
            current.append(next_index)
    tiers.append([names[index] for index in current])
    return {
        "games": int(len(games.lineups)),
        "seed_blocks": int(len(blocks)),
        "anchor": names[anchor],
        "agents": entries,
        "pairwise": pairwise,
        "rating_comparisons": rating_comparisons,
        "elo_like_order": [names[index] for index in order],
        "stable_tiers": tiers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/policy-iteration-v3/latest.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/history-tournament-v1")
    )
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        type=_parse_extra_checkpoint,
        metavar="NAME=PATH",
        help="add the current Actor from another v4 policy-iteration checkpoint",
    )
    parser.add_argument(
        "--extra-policy",
        action="append",
        default=[],
        type=_parse_extra_policy,
        metavar="NAME=PATH",
        help="add an Actor-only policy produced by training.pipeline.save_policy",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=[],
        metavar="NAME",
        help="run only these agents; sl is required as the rating anchor",
    )
    parser.add_argument("--rounds-per-combination", type=int, default=12)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TournamentConfig(
        rounds_per_combination=args.rounds_per_combination,
        envs=args.envs,
        inference_batch_size=args.inference_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    device = require_cuda(args.device)
    agents, identity = load_history_agents(
        args.checkpoint,
        device,
        extra_checkpoints=args.extra_checkpoint,
        extra_policies=args.extra_policy,
        selected_agents=args.agents,
    )
    summary = run_tournament(agents, identity, args.output_dir, device, config)
    print(
        "TOURNAMENT complete  "
        + "  ".join(
            f"{name} {summary['agents'][name]['elo_like']:+.0f}"
            for name in summary["elo_like_order"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
