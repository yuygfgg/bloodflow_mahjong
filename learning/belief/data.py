"""Safetensors dataset reader for grouped belief examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

MAX_HISTORY = 192
WORLD_PLANES = 4
TILE_KIND_COUNT = 27
PROPOSAL_STREAM_COUNT = 2
TRUTH_STREAM = PROPOSAL_STREAM_COUNT
BELIEF_FEATURE_SCHEMA_VERSION = 2
BELIEF_TARGET_VERSION = 2
ENGINE_RULES_VERSION = 3


@dataclass(frozen=True)
class Shard:
    path: Path
    split: str
    roots: int
    sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    schema_version: int
    belief_target_version: int
    engine_rules_version: int
    proposal_stream_count: int
    candidate_count: int
    shards: tuple[Shard, ...]

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        value = json.loads(path.read_text())
        root = path.parent
        shards = tuple(
            Shard(
                root / item["path"],
                item["split"],
                int(item["roots"]),
                str(item["sha256"]),
            )
            for item in value["shards"]
        )
        manifest = cls(
            root=root,
            schema_version=int(value["schema_version"]),
            belief_target_version=int(value["belief_target_version"]),
            engine_rules_version=int(value["engine_rules_version"]),
            proposal_stream_count=int(value["proposal_stream_count"]),
            candidate_count=int(value["candidate_count"]),
            shards=shards,
        )
        if manifest.schema_version != BELIEF_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported belief schema {manifest.schema_version}")
        if manifest.belief_target_version != BELIEF_TARGET_VERSION:
            raise ValueError(
                "unsupported belief target version "
                f"{manifest.belief_target_version}"
            )
        if manifest.engine_rules_version != ENGINE_RULES_VERSION:
            raise ValueError(
                "unsupported engine rules version "
                f"{manifest.engine_rules_version}"
            )
        if manifest.proposal_stream_count != PROPOSAL_STREAM_COUNT:
            raise ValueError(
                "unsupported belief proposal stream count "
                f"{manifest.proposal_stream_count}"
            )
        if manifest.candidate_count < 2:
            raise ValueError("belief groups need at least two candidates")
        allowed_splits = {"train", "calibration", "development"}
        unknown_splits = {shard.split for shard in shards} - allowed_splits
        if unknown_splits:
            raise ValueError(f"unknown belief dataset splits: {sorted(unknown_splits)}")
        if any(shard.roots <= 0 for shard in shards):
            raise ValueError("belief shards must contain at least one root")
        paths = [shard.path for shard in shards]
        if len(set(paths)) != len(paths):
            raise ValueError("belief manifest contains duplicate shard paths")
        for shard in shards:
            if len(shard.sha256) != 64 or any(
                character not in "0123456789abcdef" for character in shard.sha256
            ):
                raise ValueError(f"invalid SHA-256 for belief shard {shard.path}")
        return manifest

    def for_split(self, split: str) -> tuple[Shard, ...]:
        selected = tuple(shard for shard in self.shards if shard.split == split)
        if not selected:
            raise ValueError(f"dataset has no {split!r} shards")
        return selected


def _load_safetensors(
    path: Path, expected_metadata: dict[str, str]
) -> dict[str, np.ndarray]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("install the Python safetensors package") from error
    with safe_open(path, framework="numpy") as source:
        metadata = source.metadata() or {}
        for name, expected in expected_metadata.items():
            actual = metadata.get(name)
            if actual != expected:
                raise ValueError(
                    f"belief shard {path} has {name}={actual!r}, "
                    f"expected {expected!r}"
                )
        return {name: source.get_tensor(name) for name in source.keys()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read belief shard {path}: {error}") from error
    return digest.hexdigest()


def _validate_tensors(
    tensors: dict[str, np.ndarray], candidate_count: int, proposal_stream_count: int
) -> int:
    required = {
        "tile_obs",
        "melds",
        "river",
        "meta",
        "events",
        "event_lengths",
        "candidate_worlds",
        "handwritten_log_weights",
        "positive_mask",
        "proposal_streams",
        "block_ids",
        "root_ids",
    }
    missing = required - tensors.keys()
    if missing:
        raise ValueError(f"belief shard is missing tensors: {sorted(missing)}")
    roots = int(tensors["tile_obs"].shape[0])
    if roots <= 0:
        raise ValueError("belief shards must contain at least one root")
    group_count = candidate_count * proposal_stream_count + 1
    expected = {
        "tile_obs": (roots, 10, TILE_KIND_COUNT),
        "melds": (roots, 4, 4, 3),
        "river": (roots, 108, 2),
        "meta": (roots, 34),
        "events": (roots, MAX_HISTORY, 8),
        "event_lengths": (roots,),
        "candidate_worlds": (roots, group_count, WORLD_PLANES, TILE_KIND_COUNT),
        "handwritten_log_weights": (roots, group_count),
        "positive_mask": (roots, group_count),
        "proposal_streams": (roots, group_count),
        "block_ids": (roots,),
        "root_ids": (roots,),
    }
    for name, shape in expected.items():
        if tensors[name].shape != shape:
            raise ValueError(f"{name} has shape {tensors[name].shape}, expected {shape}")

    expected_dtypes = {
        "tile_obs": np.dtype(np.uint8),
        "melds": np.dtype(np.uint8),
        "river": np.dtype(np.uint8),
        "meta": np.dtype(np.int32),
        "events": np.dtype(np.int32),
        "event_lengths": np.dtype(np.uint16),
        "candidate_worlds": np.dtype(np.uint8),
        "handwritten_log_weights": np.dtype(np.float32),
        "positive_mask": np.dtype(np.uint8),
        "proposal_streams": np.dtype(np.uint8),
        "block_ids": np.dtype(np.uint64),
        "root_ids": np.dtype(np.uint64),
    }
    for name, dtype in expected_dtypes.items():
        if tensors[name].dtype != dtype:
            raise ValueError(f"{name} has dtype {tensors[name].dtype}, expected {dtype}")

    positive = tensors["positive_mask"]
    if not np.all((positive == 0) | (positive == 1)):
        raise ValueError("positive_mask must be binary")
    if not np.all(positive.sum(axis=1) > 0):
        raise ValueError("every belief group needs at least one positive")
    streams = tensors["proposal_streams"]
    if not np.all(streams <= proposal_stream_count):
        raise ValueError("proposal_streams contains an unknown stream")
    if not np.all((streams == proposal_stream_count).sum(axis=1) == 1):
        raise ValueError("every belief group needs exactly one truth candidate")
    for stream in range(proposal_stream_count):
        if not np.all((streams == stream).sum(axis=1) == candidate_count):
            raise ValueError("every belief group needs balanced proposal streams")
    worlds = tensors["candidate_worlds"]
    reference_indices = positive.argmax(axis=1)
    references = worlds[np.arange(roots), reference_indices]
    matching = np.all(worlds == references[:, None], axis=(2, 3))
    if not np.array_equal(matching, positive.astype(bool)):
        raise ValueError("positive_mask must mark every duplicate of the target world")
    truth_indices = (streams == proposal_stream_count).argmax(axis=1)
    if not np.all(positive[np.arange(roots), truth_indices] != 0):
        raise ValueError("the truth stream must be marked positive")

    weights = tensors["handwritten_log_weights"]
    if np.isnan(weights).any() or np.isposinf(weights).any():
        raise ValueError("handwritten weights may only use -inf for unsupported worlds")
    finite = np.isfinite(weights)
    truth = streams == proposal_stream_count
    if not np.all(finite[truth]):
        raise ValueError("truth worlds need finite handwritten weights")
    if not np.all(finite[positive.astype(bool)]):
        raise ValueError("positive worlds need finite handwritten weights")
    if np.any(tensors["event_lengths"] > MAX_HISTORY):
        raise ValueError("event_lengths exceeds the fixed history capacity")
    return roots


def _tensor(array: np.ndarray, indices: np.ndarray, device: torch.device) -> Tensor:
    selected = np.ascontiguousarray(array[indices])
    return torch.from_numpy(selected).to(device=device, non_blocking=True)


class BeliefDataset:
    def __init__(self, manifest: DatasetManifest) -> None:
        self.manifest = manifest
        for shard in manifest.shards:
            actual = _sha256(shard.path)
            if actual != shard.sha256:
                raise ValueError(
                    f"belief shard SHA-256 mismatch for {shard.path}: "
                    f"expected {shard.sha256}, got {actual}"
                )

    def roots(self, split: str) -> int:
        return sum(shard.roots for shard in self.manifest.for_split(split))

    def _shard_metadata(self) -> dict[str, str]:
        return {
            "belief_schema_version": str(self.manifest.schema_version),
            "belief_target_version": str(self.manifest.belief_target_version),
            "engine_rules_version": str(self.manifest.engine_rules_version),
            "proposal_stream_count": str(self.manifest.proposal_stream_count),
        }

    def batches(
        self,
        split: str,
        batch_size: int,
        *,
        device: torch.device,
        shuffle: bool,
        seed: int,
    ) -> Iterator[dict[str, Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        shards = list(self.manifest.for_split(split))
        if shuffle:
            rng.shuffle(shards)
        for shard in shards:
            tensors = _load_safetensors(shard.path, self._shard_metadata())
            roots = _validate_tensors(
                tensors,
                self.manifest.candidate_count,
                self.manifest.proposal_stream_count,
            )
            if roots != shard.roots:
                raise ValueError(
                    f"{shard.path} contains {roots} roots, manifest records {shard.roots}"
                )
            order = np.arange(roots)
            if shuffle:
                rng.shuffle(order)
            for start in range(0, roots, batch_size):
                indices = order[start : start + batch_size]
                yield {
                    name: _tensor(tensors[name], indices, device)
                    for name in (
                        "tile_obs",
                        "melds",
                        "meta",
                        "events",
                        "event_lengths",
                        "candidate_worlds",
                        "handwritten_log_weights",
                        "positive_mask",
                        "proposal_streams",
                        "block_ids",
                        "root_ids",
                    )
                }
