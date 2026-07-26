from __future__ import annotations

import json
from pathlib import Path

from training.dashboard import CATEGORY_NAMES, SOURCE_NAMES, render_dashboard
from training.train import _compact_record


def _evaluation(rank: float, score: float) -> dict[str, object]:
    panels: dict[str, object] = {}
    for offset, name in enumerate(("rules", "sl", "mixed", "history")):
        panels[name] = {
            "games": 256.0,
            "mean_score_delta": score + offset * 100,
            "score_std": 3000.0,
            "first_rate": 0.30 - offset * 0.01,
            "last_rate": 0.20 + offset * 0.01,
            "mean_rank": rank + offset * 0.02,
        }
    return {
        "mean_score_delta": score,
        "score_std": 3000.0,
        "first_rate": 0.30,
        "last_rate": 0.20,
        "mean_rank": rank,
        "panels": panels,
    }


def _validation(scale: float) -> dict[str, object]:
    metric = {
        "count": 512.0,
        "loss": 0.12 * scale,
        "mae": 0.24 * scale,
        "correlation": 0.31 / scale,
        "calibration_error": 0.08 * scale,
        "constant_mae": 0.4,
        "improvement": 0.4,
    }
    return {
        "q": metric,
        "v": metric,
        "q_disagreement": 0.06 * scale,
        "progress": {
            name: {"q": metric, "v": metric, "q_disagreement": 0.05 * scale}
            for name in ("early", "middle", "late")
        },
        "categories": {
            name: {"q": metric, "q_disagreement": 0.04 * scale}
            for name in CATEGORY_NAMES
        },
    }


def _replay() -> dict[str, object]:
    sources = {name: (index + 1) * 1_000 for index, name in enumerate(SOURCE_NAMES)}
    categories = {
        name: (index + 2) * 400 for index, name in enumerate(CATEGORY_NAMES)
    }
    return {
        "states": sum(sources.values()),
        "trajectories": 800,
        "anchor_trajectories": 300,
        "online_trajectories": 500,
        "mc_targets": 120,
        "sources": sources,
        "categories": categories,
    }


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n{unfinished",
        encoding="utf-8",
    )


def test_dashboard_renders_iql_metrics_and_ignores_partial_json(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    dashboard = tmp_path / "dashboard.html"
    baseline = _evaluation(2.55, -900.0)
    latest = _evaluation(2.31, -420.0)
    fresh = _evaluation(2.38, -510.0)
    actor = {
        "actor_reference_kl": 0.0123,
        "actor_advantage_mean": 0.15,
        "actor_weight_mean": 1.2,
        "actor_effective_sample_size": 612.0,
    }
    for index, name in enumerate(CATEGORY_NAMES):
        actor[f"advantage_{name}"] = 0.01 * index
        actor[f"ess_{name}"] = 20.0 + index
    records = [
        {
            "phase": "baseline",
            "fixed_evaluation": baseline,
            "fresh_evaluation": baseline,
            "replay_states": 10_000,
            "replay": _replay(),
        },
        {
            "phase": "critic_warmup",
            "critic_steps": 500,
            "critic_validation": _validation(1.2),
            "actor_gate": "middle_correlation",
        },
        {
            "phase": "iteration",
            "iteration": 1,
            "critic_steps": 564,
            "actor_updates": 1,
            "policy_version": 1,
            "actor_gate": "ready",
            "critic_validation": _validation(1.0),
            "actor": actor,
            "collection": {
                "trajectories": 128,
                "states_per_second": 18_250.0,
            },
            "replay": _replay(),
            "training_states_per_second": 31_500.0,
            "critic_seconds": 8.2,
            "actor_seconds": 1.1,
            "iteration_seconds": 15.4,
            "mc_critic_seconds": 1.7,
            "mc_critic": {
                "mc_centered_loss": 0.123,
                "mc_pairwise_loss": 0.045,
                "mc_train_pairwise_accuracy": 0.573,
                "mc_train_groups": 24.0,
                "mc_train_pairs": 48.0,
            },
            "mc": {
                "train_targets": 680.0,
                "train_targets_after_trim": 640.0,
                "validation_targets": 512.0,
                "validation_reliable_targets": 500.0,
                "validation_reliable_groups": 128.0,
                "validation_reliable_pairs": 160.0,
                "validation_frozen": True,
                "accepted_targets": 32.0,
                "terminal_rollouts": 192.0,
                "mean_variance": 0.21,
                "mean_confidence_half_width": 0.17,
                "validation_metrics": {
                    "q": {"mae": 0.19},
                    "action_ranking": {
                        "group_count": 128.0,
                        "pair_count": 160.0,
                        "all_pair_count": 240.0,
                        "pairwise_accuracy": 0.5625,
                    },
                },
            },
            "fixed_evaluation": latest,
            "fresh_evaluation": fresh,
            "best_fixed_evaluation": latest,
            "best_fresh_evaluation": fresh,
        },
        {
            "phase": "stopped",
            "iteration": 1,
            "critic_steps": 564,
            "actor_updates": 1,
            "reason": "user_interrupt",
            "best_fixed_evaluation": latest,
            "best_fresh_evaluation": fresh,
        },
    ]
    _write_records(metrics, records)

    render_dashboard(metrics, dashboard)
    output = dashboard.read_text(encoding="utf-8")
    assert "Blood Flow Mahjong IQL" in output
    assert "iteration 1" in output
    assert "2.31" in output
    assert "-420" in output
    assert "Q MAE" in output
    assert "Q loss" in output
    assert "Q correlation" in output
    assert "Calibration" in output
    assert "Q disagreement" in output
    assert "Actor KL" in output
    assert "Replay composition" in output
    assert "Decision coverage and diagnostics" in output
    assert "31.5k/s" in output
    assert "18.2k/s" in output
    assert "exchange_first" in output
    assert "frozen_policy" in output
    assert "MC train targets (trimmed)" in output
    assert "MC anchor validation targets" in output
    assert "MC reliable validation targets" in output
    assert "MC action-difference loss" in output
    assert "MC pairwise loss" in output
    assert "57.3%" in output
    assert "56.2%" in output
    assert "MC significant pairs (reliable / all)" in output
    assert "160 / 240" in output
    assert "MC reliable validation groups" in output
    assert "MC validation frozen" in output
    assert ">yes<" in output
    assert "MC critic seconds" in output
    assert "counterfactual" not in output.lower()
    assert "informative state" not in output.lower()
    assert not dashboard.with_suffix(".html.tmp").exists()


def test_dashboard_handles_empty_and_warmup_only_logs(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    dashboard = tmp_path / "dashboard.html"
    render_dashboard(metrics, dashboard)
    empty = dashboard.read_text(encoding="utf-8")
    assert "starting" in empty
    assert "No data yet" in empty

    metrics.write_text(
        json.dumps(
            {
                "phase": "critic_warmup",
                "critic_steps": 500,
                "critic_validation": _validation(1.0),
                "actor_gate": "late_correlation",
            }
        ),
        encoding="utf-8",
    )
    render_dashboard(metrics, dashboard)
    warmup = dashboard.read_text(encoding="utf-8")
    assert "0.240" in warmup
    assert "0.310" in warmup
    assert "iteration 0" in warmup
    assert "critic 500" in warmup
    assert "late_correlation" in warmup


def test_compact_iteration_line_includes_mc_progress() -> None:
    line = _compact_record(
        {
            "phase": "iteration",
            "iteration": 7,
            "replay_states": 12_345,
            "critic_validation": {"q": {"mae": 0.321}, "q_disagreement": 0.044},
            "actor_gate": "mc_pairwise_accuracy",
            "mc_critic_seconds": 1.7,
            "mc_critic": {
                "mc_centered_loss": 0.123,
                "mc_pairwise_loss": 0.045,
                "mc_train_pairwise_accuracy": 0.573,
            },
            "mc": {
                "train_targets_after_trim": 640,
                "validation_targets": 512,
                "validation_reliable_groups": 128,
                "validation_reliable_pairs": 160,
                "validation_frozen": True,
                "validation_metrics": {
                    "action_ranking": {
                        "pairwise_accuracy": 0.5625,
                        "pair_count": 160,
                        "all_pair_count": 240,
                    }
                },
            },
        }
    )
    assert "MC 640/512" in line
    assert "diff 0.123" in line
    assert "pair 0.045" in line
    assert "train_acc 57.3%" in line
    assert "val_acc 56.2%" in line
    assert "sig_pairs 160/240" in line
    assert "rel_groups 128" in line
    assert "frozen yes" in line
    assert "1.7s" in line
