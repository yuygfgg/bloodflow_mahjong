from __future__ import annotations

import hashlib

import numpy as np

from learning.belief.data import (
    BELIEF_FEATURE_SCHEMA_VERSION,
    BELIEF_TARGET_VERSION,
    ENGINE_RULES_VERSION,
    MAX_HISTORY,
    BeliefDataset,
    DatasetManifest,
    Shard,
    _load_safetensors,
    _validate_tensors,
)


def _valid_tensors() -> dict[str, np.ndarray]:
    roots = 1
    candidates = 3
    group_count = candidates * 2 + 1
    worlds = np.zeros((roots, group_count, 4, 27), dtype=np.uint8)
    worlds[:, 1] = 1
    worlds[:, 3] = 3
    worlds[:, candidates + 1] = 2
    worlds[:, 5] = 4
    return {
        "tile_obs": np.zeros((roots, 10, 27), dtype=np.uint8),
        "melds": np.full((roots, 4, 4, 3), 255, dtype=np.uint8),
        "river": np.full((roots, 108, 2), 255, dtype=np.uint8),
        "meta": np.zeros((roots, 34), dtype=np.int32),
        "events": np.full((roots, MAX_HISTORY, 8), -1, dtype=np.int32),
        "event_lengths": np.zeros(roots, dtype=np.uint16),
        "candidate_worlds": worlds,
        "handwritten_log_weights": np.array(
            [[0.0, -np.inf, 0.0, -np.inf, -np.inf, -np.inf, 0.0]],
            dtype=np.float32,
        ),
        "positive_mask": np.array([[1, 0, 1, 0, 0, 0, 1]], dtype=np.uint8),
        "proposal_streams": np.array([[0, 0, 0, 1, 1, 1, 2]], dtype=np.uint8),
        "block_ids": np.zeros(roots, dtype=np.uint64),
        "root_ids": np.zeros(roots, dtype=np.uint64),
    }


def _expect_invalid(tensors: dict[str, np.ndarray], message: str) -> None:
    try:
        _validate_tensors(tensors, 3, 2)
    except ValueError as error:
        assert message in str(error)
        return
    raise AssertionError("invalid belief tensors were accepted")


def test_validation_accepts_duplicate_positives_and_negative_infinity() -> None:
    assert _validate_tensors(_valid_tensors(), 3, 2) == 1


def test_validation_requires_the_full_target_equivalence_class() -> None:
    tensors = _valid_tensors()
    tensors["positive_mask"][0, 2] = 0

    _expect_invalid(tensors, "every duplicate")


def test_validation_rejects_nonfinite_positive_weights() -> None:
    tensors = _valid_tensors()
    tensors["handwritten_log_weights"][0, 0] = -np.inf

    _expect_invalid(tensors, "positive worlds")


def test_validation_rejects_nan_and_positive_infinity() -> None:
    for value in (np.nan, np.inf):
        tensors = _valid_tensors()
        tensors["handwritten_log_weights"][0, 1] = value
        _expect_invalid(tensors, "may only use -inf")


def test_validation_rejects_history_overflow() -> None:
    tensors = _valid_tensors()
    tensors["event_lengths"][0] = MAX_HISTORY + 1

    _expect_invalid(tensors, "history capacity")


def test_dataset_verifies_each_shard_digest_once_at_construction(tmp_path) -> None:
    shard_path = tmp_path / "train-00000.safetensors"
    shard_path.write_bytes(b"valid shard bytes")
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest = DatasetManifest(
        root=tmp_path,
        schema_version=BELIEF_FEATURE_SCHEMA_VERSION,
        belief_target_version=BELIEF_TARGET_VERSION,
        engine_rules_version=ENGINE_RULES_VERSION,
        proposal_stream_count=2,
        candidate_count=3,
        shards=(Shard(shard_path, "train", 1, digest),),
    )

    BeliefDataset(manifest)
    shard_path.write_bytes(b"corrupted shard bytes")

    try:
        BeliefDataset(manifest)
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
        return
    raise AssertionError("corrupted belief shard was accepted")


def test_manifest_rejects_wrong_belief_target_version(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
            "schema_version": 2,
            "belief_target_version": 3,
            "engine_rules_version": 3,
            "proposal_stream_count": 2,
            "candidate_count": 3,
            "shards": []
        }"""
    )

    try:
        DatasetManifest.load(manifest_path)
    except ValueError as error:
        assert "unsupported belief target version 3" in str(error)
        return
    raise AssertionError("unsupported belief target version was accepted")


def test_shard_metadata_must_match_manifest_contract(tmp_path) -> None:
    try:
        from safetensors.numpy import save_file
    except ImportError as error:
        raise AssertionError("test requires safetensors") from error

    shard_path = tmp_path / "train-00000.safetensors"
    save_file(
        {"value": np.zeros(1, dtype=np.uint8)},
        shard_path,
        metadata={
            "belief_schema_version": str(BELIEF_FEATURE_SCHEMA_VERSION),
            "belief_target_version": str(BELIEF_TARGET_VERSION + 1),
            "engine_rules_version": "3",
            "proposal_stream_count": "2",
        },
    )

    try:
        _load_safetensors(
            shard_path,
            {
                "belief_schema_version": str(BELIEF_FEATURE_SCHEMA_VERSION),
                "belief_target_version": str(BELIEF_TARGET_VERSION),
                "engine_rules_version": "3",
                "proposal_stream_count": "2",
            },
        )
    except ValueError as error:
        assert "belief_target_version='3'" in str(error)
        return
    raise AssertionError("mismatched belief shard metadata was accepted")
