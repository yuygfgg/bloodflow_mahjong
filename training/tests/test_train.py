from __future__ import annotations

import torch
import pytest

from training.batch_sweep import SweepConfig
from training.model import BloodFlowTransformer, TransformerConfig
from training.train import (
    CHECKPOINT_VERSION,
    RunConfig,
    SelfPlayCurriculum,
    _advance_self_play,
    _cleanup_committed_pending,
    _committed_iteration,
    _checkpoint_payload,
    _load_checkpoint,
    _save_checkpoint,
    build_parser,
)


def tiny_actor() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=8,
        )
    )


def test_training_defaults_encode_the_validated_update() -> None:
    config = RunConfig()
    assert config.queries_per_category == 256
    assert 9 * config.queries_per_category == 2304
    assert config.worlds == 16
    assert config.world_chunk == 64
    assert config.target_shard_size == 64
    assert config.target_query_batch_size == 64
    assert config.rollout_inference_batch_size == 128
    assert config.target_kl == pytest.approx(0.001)
    assert config.evaluation_games == 16_384
    assert config.self_play_start_first_rate == pytest.approx(0.55)
    assert config.self_play_increment == pytest.approx(0.10)
    assert config.maximum_self_play_fraction == pytest.approx(2 / 3)


def test_self_play_curriculum_is_thresholded_monotonic_and_capped() -> None:
    config = RunConfig()
    state = SelfPlayCurriculum()
    state = _advance_self_play(state, 0.549, next_iteration=2, config=config)
    assert state == SelfPlayCurriculum(last_fixed_first_rate=0.549)

    state = _advance_self_play(state, 0.55, next_iteration=3, config=config)
    assert state.fraction == pytest.approx(0.1)
    assert state.activation_iteration == 3
    state = _advance_self_play(state, 0.40, next_iteration=4, config=config)
    assert state.fraction == pytest.approx(0.1)
    assert state.activation_iteration == 3

    for iteration in range(5, 20):
        state = _advance_self_play(state, 0.80, next_iteration=iteration, config=config)
    assert state.fraction == pytest.approx(2 / 3)
    assert state.activation_iteration == 3


def test_checkpoint_roundtrip_has_no_old_optimizer_or_migration_state(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    config = RunConfig()
    payload = _checkpoint_payload(
        actor,
        config=config,
        root_seed=7,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(
            fraction=0.2,
            last_fixed_first_rate=0.57,
            activation_iteration=2,
        ),
        last_metrics={"iteration": 2},
    )
    assert set(payload) == {
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
    path = tmp_path / "latest.pt"
    _save_checkpoint(path, payload)
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()
    loaded_actor, loaded_config, loaded = _load_checkpoint(
        path, torch.device("cpu")
    )
    assert loaded["version"] == CHECKPOINT_VERSION
    assert loaded["next_iteration"] == 3
    assert loaded["self_play"] == SelfPlayCurriculum(
        fraction=0.2,
        last_fixed_first_rate=0.57,
        activation_iteration=2,
    )
    assert loaded_config == config
    for expected, actual in zip(actor.parameters(), loaded_actor.parameters()):
        torch.testing.assert_close(expected, actual)

    payload["version"] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="production format"):
        _load_checkpoint(path, torch.device("cpu"))


def test_resume_parser_does_not_invent_training_overrides() -> None:
    args = build_parser().parse_args(["--resume", "runs/example/latest.pt"])
    assert args.resume.name == "latest.pt"
    assert args.seed is None
    assert args.queries_per_category is None


def test_checkpoint_rejects_inconsistent_last_metrics(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    payload = _checkpoint_payload(
        actor,
        config=RunConfig(),
        root_seed=1,
        next_iteration=3,
        sl_checkpoint=sl,
        sl_sha256="hash",
        self_play=SelfPlayCurriculum(),
        last_metrics={"iteration": 1},
    )
    path = tmp_path / "latest.pt"
    _save_checkpoint(path, payload)
    with pytest.raises(ValueError, match="last_metrics"):
        _load_checkpoint(path, torch.device("cpu"))


def test_resume_cleanup_only_removes_committed_pending(tmp_path) -> None:
    pending = tmp_path / "pending"
    for name in ("iteration-000001", "iteration-000002", "keep-me"):
        (pending / name).mkdir(parents=True)
        (pending / name / "value").write_text(name)
    _cleanup_committed_pending(tmp_path, next_iteration=2)
    assert not (pending / "iteration-000001").exists()
    assert (pending / "iteration-000002").exists()
    assert (pending / "keep-me").exists()


def test_committed_iteration_is_read_from_atomic_checkpoint(tmp_path) -> None:
    actor = tiny_actor()
    sl = tmp_path / "sl.pt"
    sl.write_bytes(b"identity")
    path = tmp_path / "latest.pt"
    _save_checkpoint(
        path,
        _checkpoint_payload(
            actor,
            config=RunConfig(),
            root_seed=1,
            next_iteration=8,
            sl_checkpoint=sl,
            sl_sha256="hash",
            self_play=SelfPlayCurriculum(),
            last_metrics={"iteration": 7},
        ),
    )
    assert _committed_iteration(path, fallback=3) == 7
    path.write_text("broken")
    assert _committed_iteration(path, fallback=3) == 3


def test_batch_sweep_is_nested_and_reuses_a_maximum_corpus() -> None:
    config = SweepConfig()
    assert config.batch_queries_per_category == (64, 128, 256, 512)
    assert tuple(9 * qpc for qpc in config.batch_queries_per_category) == (
        576,
        1152,
        2304,
        4608,
    )
    assert config.calibration_queries_per_category == 128
    assert 9 * config.calibration_queries_per_category == 1152
    assert config.source_games >= 9 * max(config.batch_queries_per_category)
    with pytest.raises(ValueError, match="unique and increasing"):
        SweepConfig(batch_queries_per_category=(64, 32))
    with pytest.raises(ValueError, match="calibration games"):
        SweepConfig(
            batch_queries_per_category=(1, 2),
            source_games=18,
            calibration_source_games=8,
        )
    with pytest.raises(ValueError, match="heldout games"):
        SweepConfig(
            batch_queries_per_category=(1, 2),
            source_games=18,
            heldout_source_games=8,
        )
