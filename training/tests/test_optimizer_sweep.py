from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import training.batch_sweep as batch_sweep
import training.kl_scale_sweep as kl_scale_sweep
import training.optimizer_sweep as optimizer_sweep
from training.batch_sweep import SweepConfig
from training.evaluation import evaluation_seeds, save_reference_panel
from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import POLICY_EXECUTION_VERSION, save_policy
from training.policy_iteration import (
    CounterfactualBatch,
    PolicyStateBatch,
    center_legal_values,
    domain_seed,
    one_step_direction,
    save_counterfactual_batch,
    save_policy_state_batch,
)


def tiny_config() -> SweepConfig:
    return SweepConfig(
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
        bootstrap_samples=20,
    )


def write_completed_single_sweep(tmp_path):
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    config = tiny_config()
    identity = batch_sweep._single_identity(reference, None, config, 7)
    (tmp_path / "config.json").write_text(json.dumps(identity))
    shared = tmp_path / "shared"
    shared.mkdir()
    fingerprint = batch_sweep._fingerprint(
        {"identity": identity, "cache": "shared-corpora"}
    )
    (shared / "manifest.json").write_text(
        json.dumps(
            {
                "version": batch_sweep.SHARED_CACHE_VERSION,
                "fingerprint": fingerprint,
                "source_visit_frequencies": {"vector": [1 / 9] * 9},
                "train_targets": {},
                "heldout_targets": {},
            }
        )
    )
    for name in ("train.npz", "calibration.npz", "heldout.npz"):
        (shared / name).write_bytes(b"complete")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "variants": {"1": {}},
            }
        )
    )
    return tmp_path


def tiny_actor() -> BloodFlowTransformer:
    torch.manual_seed(3)
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=8,
        )
    ).eval()


def tiny_batch() -> CounterfactualBatch:
    size = 9
    legal = np.zeros((size, 115), dtype=np.bool_)
    legal[:, :2] = True
    rank_q = np.zeros((size, 115), dtype=np.float32)
    rank_q[:, 0] = -0.2
    rank_q[:, 1] = 0.2
    score_q = rank_q * 100
    meta = np.zeros((size, 34), dtype=np.int32)
    meta[:, 4] = 30
    meta[:, 12:16] = 10_000
    meta[:, 24:28] = 14
    return CounterfactualBatch(
        query_ids=np.arange(size, dtype=np.int64),
        tile_obs=np.zeros((size, 10, 27), dtype=np.uint8),
        melds=np.full((size, 4, 4, 3), 255, dtype=np.uint8),
        meta=meta,
        events=np.zeros((size, 8, 8), dtype=np.int32),
        event_lengths=np.zeros(size, dtype=np.uint16),
        legal=legal,
        categories=np.arange(size, dtype=np.uint8),
        rank_q=rank_q,
        score_q=score_q,
        centered_rank_q=center_legal_values(rank_q, legal),
        behavior_actions=np.zeros(size, dtype=np.uint8),
    )


def write_runnable_single_sweep(tmp_path):
    config = tiny_config()
    actor = tiny_actor()
    reference = tmp_path / "reference.pt"
    save_policy(reference, actor)
    identity = batch_sweep._single_identity(reference, None, config, 7)
    (tmp_path / "config.json").write_text(json.dumps(identity))
    shared = tmp_path / "shared"
    shared.mkdir()
    batch = tiny_batch()
    save_counterfactual_batch(shared / "train.npz", batch)
    save_counterfactual_batch(shared / "heldout.npz", batch)
    states = PolicyStateBatch(
        **{
            name: getattr(batch, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
    )
    save_policy_state_batch(shared / "calibration.npz", states)
    (shared / "manifest.json").write_text(
        json.dumps(
            {
                "version": batch_sweep.SHARED_CACHE_VERSION,
                "fingerprint": batch_sweep._fingerprint(
                    {"identity": identity, "cache": "shared-corpora"}
                ),
                "source_visit_frequencies": {"vector": [1 / 9] * 9},
                "train_targets": {},
                "heldout_targets": {},
            }
        )
    )
    adam_actor, initial, candidate, metrics = one_step_direction(
        actor,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=config.direction_learning_rate,
        microbatch_size=config.microbatch_size,
    )
    direction_fingerprint = batch_sweep._fingerprint(
        {
            "identity": identity,
            "reference": batch_sweep._model_digest(actor),
            "qpc": 1,
            "query_ids": batch.query_ids.tolist(),
        }
    )
    batch_sweep._save_direction(
        tmp_path / "directions" / "qpc-1.pt",
        fingerprint=direction_fingerprint,
        initial=initial,
        candidate=candidate,
        optimizer=metrics,
        elapsed_seconds=1.0,
    )
    save_policy(tmp_path / "actor-qpc1.pt", adam_actor)
    seeds = evaluation_seeds(
        domain_seed(7, batch_sweep.FIXED_EVAL), config.evaluation_games
    )
    reference_fingerprint = batch_sweep._fingerprint(
        {
            "reference": identity["reference_sha256"],
            "policy_execution_version": POLICY_EXECUTION_VERSION,
            "seed": int(domain_seed(7, batch_sweep.FIXED_EVAL)),
            "games": config.evaluation_games,
        }
    )
    save_reference_panel(
        tmp_path / "reference_panel.npz",
        seeds=seeds,
        ranks=np.asarray([2.0, 3.0]),
        scores=np.asarray([0.0, -100.0]),
        fingerprint=reference_fingerprint,
    )
    adam_panel_fingerprint = batch_sweep._fingerprint(
        {
            "identity": identity,
            "qpc": 1,
            "actor": batch_sweep._model_digest(adam_actor),
            "seeds": batch_sweep._fingerprint(seeds.tolist()),
        }
    )
    batch_sweep._save_actor_panel(
        tmp_path / "evaluation" / "qpc-1.npz",
        fingerprint=adam_panel_fingerprint,
        seeds=seeds,
        ranks=np.asarray([2.0, 2.0]),
        scores=np.asarray([10.0, 20.0]),
        elapsed_seconds=1.0,
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"identity": identity, "variants": {"1": {}}})
    )
    return tmp_path


def test_load_input_sweep_rejects_an_incomplete_child(tmp_path) -> None:
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    config = tiny_config()
    identity = batch_sweep._single_identity(reference, None, config, 7)
    (tmp_path / "config.json").write_text(json.dumps(identity))

    with pytest.raises(ValueError, match="incomplete"):
        optimizer_sweep.load_input_sweep(tmp_path)


def test_load_input_sweep_validates_completed_shared_identity(tmp_path) -> None:
    directory = write_completed_single_sweep(tmp_path)
    loaded = optimizer_sweep.load_input_sweep(directory)

    assert loaded.seeds == (7,)
    assert loaded.config == tiny_config()
    assert loaded.children[7] == directory.resolve()


def test_experiment_identity_includes_learning_rate_and_input(tmp_path) -> None:
    directory = write_completed_single_sweep(tmp_path)
    loaded = optimizer_sweep.load_input_sweep(directory)
    identity = optimizer_sweep._experiment_identity(loaded, (1,), 0.1)

    assert identity["optimizer"] == "sgd"
    assert identity["sgd_learning_rate"] == pytest.approx(0.1)
    assert identity["input_directory"] == str(directory.resolve())
    assert identity["queries_per_category"] == [1]


def test_input_seed_selection_is_strict() -> None:
    assert optimizer_sweep._select_seeds((7, 8, 9), (8, 7)) == (8, 7)
    with pytest.raises(ValueError, match="unique"):
        optimizer_sweep._select_seeds((7, 8), (7, 7))
    with pytest.raises(ValueError, match="not in"):
        optimizer_sweep._select_seeds((7, 8), (9,))


def test_optimizer_parser_requires_independent_input_and_output() -> None:
    parser = optimizer_sweep.build_parser()
    args = parser.parse_args(
        ["--batch-sweep-dir", "input", "--output-dir", "output"]
    )
    assert args.sgd_learning_rate == pytest.approx(0.1)
    assert args.qpc is None
    assert args.seeds is None


def test_optimizer_sweep_runs_the_reused_corpus_pipeline(
    tmp_path, monkeypatch
) -> None:
    source = write_runnable_single_sweep(tmp_path / "input")
    output = tmp_path / "output"
    monkeypatch.setattr(
        optimizer_sweep, "require_cuda", lambda _device: torch.device("cpu")
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "CPU")
    monkeypatch.setattr(
        optimizer_sweep,
        "calibrate_direction",
        lambda *_args, **_kwargs: {
            "evaluations": 1.0,
            "scale": 1.0,
            "candidate_kl": 0.001,
            "final_kl": 0.001,
            "target_kl": 0.001,
            "absolute_error": 0.0,
            "relative_shortfall": 0.0,
            "greedy_flip_rate": 0.1,
            "equal_state_greedy_flip_rate": 0.1,
        },
    )
    monkeypatch.setattr(
        optimizer_sweep,
        "heldout_policy_value",
        lambda *_args, **_kwargs: {
            "visitation_weighted_rank_value": 0.01,
            "visitation_weighted_score_value": 1.0,
        },
    )
    monkeypatch.setattr(
        optimizer_sweep,
        "collect_fixed_panel",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        optimizer_sweep,
        "outcomes",
        lambda _result: (
            np.asarray([1.0, 2.0]),
            np.asarray([30.0, 40.0]),
        ),
    )

    result = optimizer_sweep.run(source, output, device="cuda")

    variant = result["variants"]["1"]
    assert variant["pooled_evaluation_vs_adamw"]["games"] == 2
    assert (output / "aggregate.json").exists()
    assert (output / "seed-7" / "actor-sgd-qpc1.pt").exists()


def test_kl_scale_parser_defaults_to_raw_and_half_scale() -> None:
    parser = kl_scale_sweep.build_parser()
    args = parser.parse_args(
        ["--batch-sweep-dir", "input", "--output-dir", "output"]
    )
    assert args.qpc == [256]
    assert args.scales == [0.5, 1.0]


def test_kl_scale_sweep_runs_the_reused_adamw_direction(
    tmp_path, monkeypatch
) -> None:
    source = write_runnable_single_sweep(tmp_path / "input")
    output = tmp_path / "output"
    input_summary = json.loads((source / "summary.json").read_text())
    input_summary["variants"]["1"]["calibration"] = {"scale": 2.0}
    (source / "summary.json").write_text(json.dumps(input_summary))
    monkeypatch.setattr(
        kl_scale_sweep, "require_cuda", lambda _device: torch.device("cpu")
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "CPU")

    def fixed_scale(actor, _reference, initial, candidate, *_args, scale, **_kwargs):
        from training.policy_iteration import load_scaled_direction

        load_scaled_direction(actor, initial, candidate, scale)
        return {
            "scale": scale,
            "kl": 0.0002 * scale * scale,
            "greedy_flip_rate": 0.01,
            "equal_state_greedy_flip_rate": 0.01,
            "evaluations": 1.0,
        }

    monkeypatch.setattr(kl_scale_sweep, "evaluate_direction_scale", fixed_scale)
    monkeypatch.setattr(
        kl_scale_sweep,
        "heldout_policy_value",
        lambda *_args, **_kwargs: {
            "visitation_weighted_rank_value": 0.01,
            "visitation_weighted_score_value": 1.0,
        },
    )
    monkeypatch.setattr(
        kl_scale_sweep,
        "collect_fixed_panel",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    panels = iter(
        [
            (np.asarray([1.0, 2.0]), np.asarray([30.0, 40.0])),
            (np.asarray([1.0, 1.0]), np.asarray([40.0, 50.0])),
        ]
    )
    monkeypatch.setattr(kl_scale_sweep, "outcomes", lambda _result: next(panels))

    result = kl_scale_sweep.run(
        source,
        output,
        qpc=(1,),
        scales=(0.5, 1.0),
        device="cuda",
    )

    variants = result["variants"]["1"]["scales"]
    assert set(variants) == {"0.5", "1"}
    assert variants["0.5"]["pooled_evaluation_vs_raw"]["games"] == 2
    assert (output / "aggregate.json").exists()
    assert (output / "seed-7" / "actor-adamw-qpc1-scale-1.pt").exists()
