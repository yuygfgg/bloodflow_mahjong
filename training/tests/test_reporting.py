from __future__ import annotations

import json

from training.reporting import append_jsonl, format_duration, format_rate
from training.supervised import _compact_record as compact_supervised_record
from training.train import _append_record, _compact_record as compact_ppo_record


def evaluation() -> dict[str, float]:
    return {
        "games": 8.0,
        "mean_score_delta": -725.0,
        "score_std": 100.0,
        "first_rate": 0.125,
        "last_rate": 0.5,
        "mean_rank": 3.25,
    }


def test_historical_duration_and_rate_style() -> None:
    assert format_duration(3_661.0) == "1h01m"
    assert format_rate(12_345.0, "states") == "12.35k states/s"


def test_append_jsonl_preserves_the_complete_record(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    record = {"phase": "ppo", "nested": {"values": [1, 2, 3]}}

    append_jsonl(path, record)

    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_supervised_summary_is_compact() -> None:
    record = {
        "labels": 50,
        "target_labels": 100,
        "elapsed_seconds": 2.0,
        "loss": 1.25,
        "train_accuracy": 0.6,
        "validation_loss": 1.5,
        "validation_accuracy": 0.55,
        "grad_norm": 0.42,
    }

    assert compact_supervised_record(record) == (
        "SL  50/100 50.0%  25.0 labels/s  elapsed 2s  ETA 2s"
        "  loss 1.250  acc 60.0%  val-loss 1.500  val-acc 55.0%  grad 0.420"
    )


def test_ppo_summaries_cover_each_phase() -> None:
    baseline = {
        "phase": "ppo_start",
        "evaluation": evaluation(),
        "evaluation_seconds": 0.75,
    }
    assert compact_ppo_record(baseline) == (
        "BASE EV  rank 3.25  score -725  first 12.5%  last 50.0%"
        "  games 8  opp fast/EV 33.3%/66.7%  time 1s"
    )

    resume = {
        "phase": "resume",
        "update": 2,
        "transitions": 128,
        "previous_run_elapsed_seconds": 7_200.0,
        "target_hours": 24.0,
    }
    assert compact_ppo_record(resume) == (
        "RESUME  u    2  128 states  elapsed 2h00m/24h"
    )

    update = {
        "phase": "ppo",
        "update": 3,
        "transitions": 196_608,
        "ppo_elapsed_seconds": 3_661.0,
        "target_hours": 24.0,
        "rollout_states_per_second": 12_345.0,
        "policy_loss": -0.01234,
        "value_loss": 2.41,
        "entropy": 0.72,
        "approx_kl": 0.0061,
        "learning_rate": 2.8e-4,
        "opponent_assignments": {
            "rule_fast": 8,
            "rule_ev": 16,
            "frozen_transformer": 0,
        },
        "evaluation": evaluation(),
    }
    assert compact_ppo_record(update) == (
        "PPO u    3  1h01m/24h  196,608 states  12.35k states/s"
        "  pi -0.0123  value 2.410  ent 0.720  KL +0.00610  lr 2.80e-04"
        "  opp fast/EV 33.3%/66.7%"
        "  EV rank 3.25  score -725  first 12.5%  last 50.0%"
    )
    off_update = update | {"kl_control": "off"}
    assert "  KL off  lr" in compact_ppo_record(off_update)

    complete = {
        "phase": "complete",
        "update": 4,
        "transitions": 262_144,
        "ppo_elapsed_seconds": 86_400.0,
        "target_hours": 24.0,
        "final": evaluation(),
    }
    assert compact_ppo_record(complete) == (
        "DONE  u    4  262,144 states  1d00h/24h"
        "  EV rank 3.25  score -725  first 12.5%  last 50.0%  games 8"
    )


def test_ppo_record_writes_json_but_prints_only_the_summary(tmp_path, capsys) -> None:
    path = tmp_path / "metrics.jsonl"
    record = {
        "phase": "ppo_start",
        "evaluation": evaluation(),
        "evaluation_seconds": 0.75,
    }

    _append_record(path, record)

    output = capsys.readouterr().out
    assert output.startswith("BASE EV  rank 3.25")
    assert not output.startswith("{")
    assert json.loads(path.read_text(encoding="utf-8")) == record
