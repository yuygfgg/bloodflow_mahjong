from __future__ import annotations

import hashlib
import json
import os
import signal

import pytest
import torch

import training.train as train_module
from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import PPOConfig
from training.train import (
    _check_resume_overrides,
    _fork_config,
    _fresh_config,
    _load_analysis_policy,
    build_parser,
)


def tiny_transformer(_config: object = None) -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
            value_atoms=17,
        )
    )


def test_new_run_defaults_to_metric_controlled_training_without_time_limit() -> None:
    args = build_parser().parse_args([])

    config = _fresh_config(args)

    assert not hasattr(args, "hours")
    assert not args.smoke
    assert config.score_reward_weight == 1.0
    assert config.rank_reward_weight == 1.0
    assert config.kl_control == "monitor"
    assert config.self_play_enabled
    assert config.self_play_max_fraction == 0.45
    assert config.self_play_fraction_step == 0.15
    assert config.self_play_gate_score_delta == 75.0
    assert config.self_play_gate_mean_rank == 2.45
    assert config.self_play_gate_window == 3


def test_smoke_run_stops_after_two_updates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_module, "BloodFlowTransformer", tiny_transformer)
    output = tmp_path / "smoke"
    args = build_parser().parse_args(
        ["--smoke", "--device", "cpu", "--output-dir", str(output)]
    )

    train_module.run(args)

    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    updates = [record for record in records if record["phase"] == "ppo"]
    assert [record["update"] for record in updates] == [1, 2]
    assert records[-1]["phase"] == "complete"
    assert (output / "latest.pt").is_file()


def test_fully_rolled_back_updates_do_not_reach_entropy_control(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(train_module, "BloodFlowTransformer", tiny_transformer)

    def fully_rolled_back_update(*args, **kwargs):
        controls = args[-1]
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 1.0,
            "max_attempted_kl": 1.0,
            "updates": 0.0,
            "epochs": 0.0,
            "rolled_back_epochs": 1.0,
            "kl_monitor_samples": 64.0,
            "aux_scale": controls.auxiliary_scale,
            "entropy_scale": controls.entropy_coefficient,
        }

    monkeypatch.setattr(train_module, "ppo_update", fully_rolled_back_update)
    output = tmp_path / "rolled-back"

    train_module.run(
        build_parser().parse_args(
            ["--smoke", "--device", "cpu", "--output-dir", str(output)]
        )
    )

    checkpoint = torch.load(
        output / "latest.pt", map_location="cpu", weights_only=False
    )
    controller = checkpoint["training_controller"]
    assert controller["entropy_updates"] == 0
    assert controller["entropy_low_streak"] == 0
    assert controller["entropy_high_streak"] == 0


def test_fork_can_run_a_finite_learning_rate_experiment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(train_module, "BloodFlowTransformer", tiny_transformer)
    source = tmp_path / "source"
    train_module.run(
        build_parser().parse_args(
            ["--smoke", "--device", "cpu", "--output-dir", str(source)]
        )
    )

    target = tmp_path / "fork"
    train_module.run(
        build_parser().parse_args(
            [
                "--fork",
                str(source / "latest.pt"),
                "--output-dir",
                str(target),
                "--device",
                "cpu",
                "--eval-games",
                "8",
                "--analysis-games",
                "8",
                "--eval-every",
                "1",
                "--checkpoint-every",
                "1",
                "--stop-after-updates",
                "1",
                "--learning-rate",
                "5e-5",
                "--minimum-learning-rate",
                "5e-5",
                "--learning-rate-decay",
                "0.7",
                "--learning-rate-patience-evaluations",
                "18",
            ]
        )
    )

    records = [
        json.loads(line)
        for line in (target / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    updates = [record for record in records if record["phase"] == "ppo"]
    checkpoint = torch.load(target / "latest.pt", map_location="cpu", weights_only=False)
    assert [record["update"] for record in updates] == [3]
    assert records[-1]["phase"] == "complete"
    assert updates[0]["learning_rate"] == 5e-5
    assert checkpoint["training_controller"]["current_learning_rate"] == 5e-5
    assert checkpoint["training_controller"]["best_rank"] is not None


def test_sigint_saves_only_after_the_active_update(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_module, "BloodFlowTransformer", tiny_transformer)
    original_ppo_update = train_module.ppo_update
    signal_sent = False

    def interrupting_ppo_update(*args, **kwargs):
        nonlocal signal_sent
        if not signal_sent:
            signal_sent = True
            os.kill(os.getpid(), signal.SIGINT)
        return original_ppo_update(*args, **kwargs)

    monkeypatch.setattr(train_module, "ppo_update", interrupting_ppo_update)
    output = tmp_path / "interrupted"
    args = build_parser().parse_args(
        ["--smoke", "--device", "cpu", "--output-dir", str(output)]
    )

    train_module.run(args)

    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    updates = [record for record in records if record["phase"] == "ppo"]
    assert signal_sent
    assert [record["update"] for record in updates] == [1]
    assert "gate_evaluation" not in updates[0]
    assert records[-1]["phase"] == "interrupted"
    checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["update"] == 1
    assert checkpoint["transitions"] == 64


def test_new_run_accepts_explicit_reward_and_kl_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--score-reward-weight",
            "0.5",
            "--rank-reward-weight",
            "1.5",
            "--kl-control",
            "off",
        ]
    )

    config = _fresh_config(args)

    assert config.score_reward_weight == 0.5
    assert config.rank_reward_weight == 1.5
    assert config.kl_control == "off"


def test_analysis_evaluator_options_do_not_change_training_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--analysis-opponent",
            "rule-nn",
            "--analysis-nn-model",
            "model/anchor.onnx",
            "--analysis-games",
            "2048",
            "--analysis-every",
            "3",
        ]
    )

    config = _fresh_config(args)

    assert args.analysis_opponent == "rule-nn"
    assert str(args.analysis_nn_model) == "model/anchor.onnx"
    assert args.analysis_games == 2048
    assert args.analysis_every == 3
    assert config == PPOConfig()


def test_rule_nn_analysis_requires_an_explicit_model() -> None:
    args = build_parser().parse_args(["--analysis-opponent", "rule-nn"])

    with pytest.raises(ValueError, match="requires --analysis-nn-model"):
        _load_analysis_policy(args)


def test_analysis_model_is_rejected_for_non_nn_opponents(tmp_path) -> None:
    path = tmp_path / "anchor.onnx"
    args = build_parser().parse_args(["--analysis-nn-model", str(path)])

    with pytest.raises(ValueError, match="requires --analysis-opponent=rule-nn"):
        _load_analysis_policy(args)


def test_rule_nn_analysis_records_model_identity(tmp_path, monkeypatch) -> None:
    path = tmp_path / "anchor.onnx"
    payload = b"fixed analysis anchor"
    path.write_bytes(payload)
    sentinel = object()

    class FakeRuleNn:
        @staticmethod
        def from_file(model_path: str):
            assert model_path == str(path)
            return sentinel

    monkeypatch.setattr(train_module.bm, "RuleNn", FakeRuleNn)
    args = build_parser().parse_args(
        [
            "--analysis-opponent",
            "rule-nn",
            "--analysis-nn-model",
            str(path),
        ]
    )

    policy, metadata = _load_analysis_policy(args)

    assert policy is sentinel
    assert metadata == {
        "model_path": str(path),
        "model_sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_kl_threshold_is_irrelevant_when_kl_control_is_off() -> None:
    args = build_parser().parse_args(["--kl-control", "off", "--target-kl", "0"])

    config = _fresh_config(args)

    assert config.kl_control == "off"
    assert config.target_kl == 0.0


def test_resume_uses_checkpoint_configuration_without_repeated_flags() -> None:
    args = build_parser().parse_args(["--resume", "run/latest.pt"])
    config = PPOConfig(
        microbatch=128,
        self_play_enabled=True,
        score_reward_weight=0.75,
        rank_reward_weight=1.25,
        kl_control="rollback",
    )

    _check_resume_overrides(args, config)


def test_resume_rejects_an_explicit_configuration_change() -> None:
    args = build_parser().parse_args(
        ["--resume", "run/latest.pt", "--rank-reward-weight", "2"]
    )

    with pytest.raises(ValueError, match="cannot override"):
        _check_resume_overrides(args, PPOConfig())


def test_fork_can_change_the_opponent_curriculum() -> None:
    args = build_parser().parse_args(
        [
            "--fork",
            "run/latest.pt",
            "--no-self-play",
            "--self-play-max-fraction",
            "0.30",
            "--self-play-fraction-step",
            "0.10",
            "--self-play-gate-score",
            "100",
            "--self-play-gate-rank",
            "2.40",
            "--self-play-gate-window",
            "4",
            "--historical-snapshot-probability",
            "0.5",
        ]
    )
    source = PPOConfig(historical_snapshot_probability=0.25)

    forked = _fork_config(args, source)

    assert not forked.self_play_enabled
    assert forked.self_play_max_fraction == 0.30
    assert forked.self_play_fraction_step == 0.10
    assert forked.self_play_gate_score_delta == 100.0
    assert forked.self_play_gate_mean_rank == 2.40
    assert forked.self_play_gate_window == 4
    assert forked.historical_snapshot_probability == 0.5
    assert forked.score_reward_weight == source.score_reward_weight
    assert forked.rank_reward_weight == source.rank_reward_weight
    assert forked.kl_control == source.kl_control


def test_fork_can_override_learning_schedule() -> None:
    args = build_parser().parse_args(
        [
            "--fork",
            "run/latest.pt",
            "--learning-rate",
            "5e-5",
            "--minimum-learning-rate",
            "5e-5",
            "--learning-rate-decay",
            "0.7",
            "--learning-rate-patience-evaluations",
            "18",
            "--stop-after-updates",
            "200",
        ]
    )
    source = PPOConfig()

    forked = _fork_config(args, source)

    assert forked.learning_rate == 5e-5
    assert forked.minimum_learning_rate == 5e-5
    assert forked.learning_rate_decay == 0.7
    assert forked.learning_rate_patience_evaluations == 18
    assert args.stop_after_updates == 200


def test_fork_schedule_override_requires_explicit_learning_rate() -> None:
    args = build_parser().parse_args(
        [
            "--fork",
            "run/latest.pt",
            "--learning-rate-patience-evaluations",
            "18",
        ]
    )

    with pytest.raises(ValueError, match="require an explicit --learning-rate"):
        _fork_config(args, PPOConfig())


def test_fork_rejects_a_reward_change() -> None:
    args = build_parser().parse_args(
        ["--fork", "run/latest.pt", "--rank-reward-weight", "2"]
    )

    with pytest.raises(ValueError, match="cannot override"):
        _fork_config(args, PPOConfig())
