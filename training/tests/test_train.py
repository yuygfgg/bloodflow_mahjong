from __future__ import annotations

import pytest

from training.pipeline import PPOConfig
from training.train import (
    _check_resume_overrides,
    _fork_config,
    _fresh_config,
    build_parser,
)


def test_new_run_defaults_to_hybrid_reward_and_kl_monitoring() -> None:
    args = build_parser().parse_args([])

    config = _fresh_config(args)

    assert config.score_reward_weight == 1.0
    assert config.rank_reward_weight == 1.0
    assert config.kl_control == "monitor"
    assert not config.self_play_enabled
    assert config.self_play_fraction == 0.25
    assert config.opponent_refresh_updates == 200


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


def test_fork_changes_only_the_opponent_curriculum() -> None:
    args = build_parser().parse_args(
        [
            "--fork",
            "run/latest.pt",
            "--self-play",
            "--self-play-fraction",
            "0.25",
            "--historical-snapshot-probability",
            "0.5",
            "--opponent-refresh-updates",
            "200",
        ]
    )
    source = PPOConfig(opponent_refresh_updates=10)

    forked = _fork_config(args, source)

    assert forked.self_play_enabled
    assert forked.self_play_fraction == 0.25
    assert forked.historical_snapshot_probability == 0.5
    assert forked.opponent_refresh_updates == 200
    assert forked.score_reward_weight == source.score_reward_weight
    assert forked.rank_reward_weight == source.rank_reward_weight
    assert forked.kl_control == source.kl_control


def test_fork_rejects_a_reward_change() -> None:
    args = build_parser().parse_args(
        ["--fork", "run/latest.pt", "--rank-reward-weight", "2"]
    )

    with pytest.raises(ValueError, match="cannot override"):
        _fork_config(args, PPOConfig())
