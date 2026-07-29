from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import training.batch_sweep as batch_sweep
from training.batch_sweep import SweepConfig


def test_direction_cache_roundtrip_is_strict(tmp_path) -> None:
    path = tmp_path / "direction.pt"
    initial = {"weight": torch.asarray([1.0])}
    candidate = {"weight": torch.asarray([2.0])}
    batch_sweep._save_direction(
        path,
        fingerprint="direction",
        initial=initial,
        candidate=candidate,
        optimizer={"optimizer_steps": 1},
        elapsed_seconds=2.0,
    )
    loaded_initial, loaded_candidate, optimizer, elapsed = (
        batch_sweep._load_direction(path, fingerprint="direction")
    )
    torch.testing.assert_close(loaded_initial["weight"], initial["weight"])
    torch.testing.assert_close(loaded_candidate["weight"], candidate["weight"])
    assert optimizer == {"optimizer_steps": 1}
    assert elapsed == pytest.approx(2.0)
    with pytest.raises(ValueError, match="fingerprint"):
        batch_sweep._load_direction(path, fingerprint="other")


def test_actor_panel_cache_roundtrip_is_strict(tmp_path) -> None:
    path = tmp_path / "panel.npz"
    seeds = np.asarray([1, 2], dtype=np.uint64)
    ranks = np.asarray([1, 3], dtype=np.float64)
    scores = np.asarray([100, -50], dtype=np.float64)
    batch_sweep._save_actor_panel(
        path,
        fingerprint="actor",
        seeds=seeds,
        ranks=ranks,
        scores=scores,
        elapsed_seconds=3.0,
    )
    loaded_seeds, loaded_ranks, loaded_scores, elapsed = (
        batch_sweep._load_actor_panel(path, fingerprint="actor")
    )
    np.testing.assert_array_equal(loaded_seeds, seeds)
    np.testing.assert_array_equal(loaded_ranks, ranks)
    np.testing.assert_array_equal(loaded_scores, scores)
    assert elapsed == pytest.approx(3.0)


def test_completed_sweep_returns_before_collection(monkeypatch, tmp_path) -> None:
    config = SweepConfig(batch_queries_per_category=(1, 2), source_games=18)
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"actor")
    output = tmp_path / "sweep"
    output.mkdir()
    identity = batch_sweep._single_identity(sl, None, config, 7)
    (output / "config.json").write_text(json.dumps(identity))
    result = {
        "identity": identity,
        "variants": {
            str(size): {
                "evaluation": {
                    "paired_rank_delta": {
                        "mean": 0.0,
                        "ci95_low": -0.1,
                        "ci95_high": 0.1,
                    }
                }
            }
            for size in config.batch_queries_per_category
        },
    }
    (output / "summary.json").write_text(json.dumps(result))
    monkeypatch.setattr(
        batch_sweep, "require_cuda", lambda _device: torch.device("cpu")
    )
    monkeypatch.setattr(
        batch_sweep,
        "_collect",
        lambda *_args, **_kwargs: pytest.fail("completed sweep recollected data"),
    )
    assert batch_sweep.run(sl, output, config=config, seed=7) == result


def test_smoke_has_an_isolated_default_output_directory() -> None:
    args = batch_sweep.build_parser().parse_args(["--smoke"])
    assert args.output_dir is None
    assert str(
        args.output_dir
        or ("/tmp/batch-sweep-smoke" if args.smoke else "runs/batch-sweep-v3")
    ) == "/tmp/batch-sweep-smoke"


def test_parser_defaults_cover_the_full_batch_search_range() -> None:
    args = batch_sweep.build_parser().parse_args([])
    assert args.batch_qpc == [64, 128, 256, 512]
    assert args.calibration_qpc == 128
    assert args.target_shard_size == 64
    assert args.target_query_batch_size == 64
    assert args.rollout_inference_batch_size == 128
    assert args.self_play_checkpoint is None
    assert args.self_play_fraction == 0.0
    assert args.anchor_rule_fast is False


def test_parser_accepts_reference_alias_and_late_stage_lineup() -> None:
    args = batch_sweep.build_parser().parse_args(
        [
            "--sl-checkpoint",
            "u56.pt",
            "--self-play-checkpoint",
            "u40.pt",
            "--self-play-fraction",
            "0.5",
            "--anchor-rule-fast",
        ]
    )
    assert args.reference_checkpoint.name == "u56.pt"
    assert args.self_play_checkpoint.name == "u40.pt"
    assert args.self_play_fraction == 0.5
    assert args.anchor_rule_fast is True


def test_lineup_identity_hashes_both_checkpoints(tmp_path) -> None:
    reference = tmp_path / "u56.pt"
    opponent = tmp_path / "u40.pt"
    reference.write_bytes(b"reference")
    opponent.write_bytes(b"opponent")
    config = SweepConfig(self_play_fraction=0.5, anchor_rule_fast=True)

    identity = batch_sweep._single_identity(reference, opponent, config, 7)

    assert identity["reference_checkpoint"] == str(reference.resolve())
    assert identity["reference_sha256"] == batch_sweep._sha256(reference)
    assert identity["self_play_checkpoint"] == str(opponent.resolve())
    assert identity["self_play_sha256"] == batch_sweep._sha256(opponent)
    assert identity["config"]["self_play_fraction"] == 0.5
    assert identity["config"]["anchor_rule_fast"] is True


def test_positive_self_play_requires_a_checkpoint(tmp_path) -> None:
    reference = tmp_path / "u56.pt"
    reference.write_bytes(b"reference")
    config = SweepConfig(self_play_fraction=0.5)
    with pytest.raises(ValueError, match="requires --self-play-checkpoint"):
        batch_sweep._single_identity(reference, None, config, 7)


def test_source_collection_propagates_late_stage_lineup(monkeypatch) -> None:
    calls = {}

    class StubCollector:
        def __init__(self, collection_config, actor, device, **kwargs):
            calls["collection_config"] = collection_config
            calls["actor"] = actor
            calls["device"] = device
            calls["kwargs"] = kwargs

        def collect(self, games, *, on_progress):
            calls["games"] = games
            on_progress(games, 20, 2.0)
            return SimpleNamespace(
                trajectories=("trajectory",),
                environment_steps=20,
                elapsed_seconds=2.0,
            )

    class StubProgress:
        def start(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs):
            pass

        def complete(self, *_args, **_kwargs):
            pass

    actor = SimpleNamespace(config=SimpleNamespace(max_history=192))
    opponent = object()
    monkeypatch.setattr(batch_sweep, "TrajectoryCollector", StubCollector)
    monkeypatch.setattr(
        batch_sweep,
        "select_independent_queries",
        lambda trajectories, **_kwargs: [trajectories[0]],
    )

    queries, trajectories = batch_sweep._collect(
        actor,
        torch.device("cpu"),
        StubProgress(),
        self_play_actor=opponent,
        self_play_fraction=0.5,
        anchor_rule_fast=True,
        games=8,
        envs=4,
        qpc=1,
        source_seed=11,
        query_seed=12,
        phase="SOURCE",
    )

    assert queries == ["trajectory"]
    assert trajectories == ("trajectory",)
    assert calls["kwargs"] == {
        "seed": 11,
        "self_play_fraction": 0.5,
        "self_play_actor": opponent,
        "anchor_rule_fast": True,
    }


def test_target_collection_propagates_grouped_rollout_sizes(
    monkeypatch, tmp_path
) -> None:
    calls = {}

    def cached(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        kwargs["on_progress"](1, {"rollout_states/s": 10.0})
        return "batch", {"states": 1}

    class StubProgress:
        def start(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs):
            pass

        def complete(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(batch_sweep, "cached_counterfactual_corpus", cached)
    batch, metrics = batch_sweep._targets(
        tmp_path,
        ["query"],
        "actor",
        torch.device("cpu"),
        StubProgress(),
        self_play_actor="opponent",
        phase="TARGETS",
        fingerprint="identity",
        worlds=16,
        world_chunk=8,
        world_seed=7,
        shard_size=16,
        query_batch_size=16,
        inference_batch_size=512,
    )

    assert batch == "batch"
    assert metrics == {"states": 1}
    assert calls["kwargs"]["self_play_actor"] == "opponent"
    assert calls["kwargs"]["query_batch_size"] == 16
    assert calls["kwargs"]["inference_batch_size"] == 512


@pytest.mark.parametrize(
    "override",
    (
        {"target_query_batch_size": 0},
        {"rollout_inference_batch_size": 0},
    ),
)
def test_grouped_rollout_sizes_must_be_positive(override) -> None:
    with pytest.raises(ValueError, match="positive"):
        SweepConfig(**override)


def test_multi_seed_aggregate_pools_raw_panels(tmp_path) -> None:
    config = SweepConfig(
        source_games=9,
        batch_queries_per_category=(1,),
        calibration_source_games=9,
        calibration_queries_per_category=1,
        heldout_source_games=9,
        heldout_queries_per_category=1,
        train_worlds=2,
        heldout_worlds=2,
        evaluation_games=2,
        evaluation_envs=1,
        bootstrap_samples=50,
    )
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"actor")
    seeds = (7, 8)
    identity = batch_sweep._multi_identity(sl, config, seeds)
    child_summaries = {}
    for seed in seeds:
        child_dir = tmp_path / f"seed-{seed}"
        child_dir.mkdir()
        panel_seeds = batch_sweep.evaluation_seeds(
            batch_sweep.domain_seed(seed, batch_sweep.FIXED_EVAL), 2
        )
        reference_ranks = np.asarray([2.0, 3.0])
        reference_scores = np.asarray([0.0, 0.0])
        reference_fingerprint = batch_sweep._fingerprint(
            {
                "reference": identity["reference_sha256"],
                "policy_execution_version": batch_sweep.POLICY_EXECUTION_VERSION,
                "seed": int(batch_sweep.domain_seed(seed, batch_sweep.FIXED_EVAL)),
                "games": config.evaluation_games,
            }
        )
        batch_sweep.save_reference_panel(
            child_dir / "reference_panel.npz",
            seeds=panel_seeds,
            ranks=reference_ranks,
            scores=reference_scores,
            fingerprint=reference_fingerprint,
        )
        batch_sweep._save_actor_panel(
            child_dir / "evaluation" / "qpc-1.npz",
            fingerprint=f"actor-{seed}",
            seeds=panel_seeds,
            ranks=reference_ranks - 0.5,
            scores=reference_scores + 100.0,
            elapsed_seconds=2.0,
        )
        child_summaries[seed] = {
            "variants": {
                "1": {
                    "states": 9,
                    "heldout": {
                        "visitation_weighted_rank_value": 0.1,
                        "visitation_weighted_score_value": 10.0,
                    },
                    "timing": {"elapsed_seconds": 2.0},
                    "evaluation": {
                        "paired_rank_delta": {"mean": -0.5},
                        "paired_score_delta": {"mean": 100.0},
                    },
                }
            }
        }

    aggregate = batch_sweep._aggregate_multi_seed(
        tmp_path,
        identity=identity,
        seeds=seeds,
        config=config,
        reference_hash=str(identity["reference_sha256"]),
        child_summaries=child_summaries,
    )
    variant = aggregate["variants"]["1"]
    assert variant["pooled_evaluation"]["games"] == 4
    assert variant["pooled_evaluation"]["paired_rank_delta"]["mean"] == pytest.approx(
        -0.5
    )
    assert variant["seed_metrics"]["paired_score_delta"]["mean"] == pytest.approx(
        100.0
    )
    assert set(variant["seeds"]) == {"7", "8"}


def test_multi_seed_identity_rejects_duplicate_seeds(tmp_path) -> None:
    config = SweepConfig(batch_queries_per_category=(1,), source_games=9)
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"actor")
    with pytest.raises(ValueError, match="unique"):
        batch_sweep._multi_identity(sl, config, (1, 1))


def test_completed_multi_seed_sweep_returns_before_child_runs(
    monkeypatch, tmp_path
) -> None:
    config = SweepConfig(batch_queries_per_category=(1,), source_games=9)
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"actor")
    output = tmp_path / "multi"
    identity = batch_sweep._multi_identity(sl, config, (7, 8))
    batch_sweep._prepare_multi_directory(output, identity)
    aggregate = {
        "identity": identity,
        "seeds": [7, 8],
        "variants": {"1": {"pooled_evaluation": {}}},
    }
    batch_sweep._atomic_json(output / "aggregate.json", aggregate)
    monkeypatch.setattr(
        batch_sweep,
        "run",
        lambda *_args, **_kwargs: pytest.fail("completed sweep reran a child"),
    )
    assert (
        batch_sweep.run_many(sl, output, config=config, seeds=(7, 8)) == aggregate
    )


def test_multi_seed_parser_accepts_explicit_seed_list() -> None:
    args = batch_sweep.build_parser().parse_args(["--seeds", "7", "8"])
    assert args.seeds == [7, 8]
    assert args.seed == 20260727
