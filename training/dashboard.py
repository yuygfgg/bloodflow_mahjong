"""Dependency-free HTML dashboard for the offline-to-online IQL trainer."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


CATEGORY_NAMES = (
    "exchange_first",
    "exchange_second",
    "exchange_third",
    "choose_missing",
    "turn_early",
    "turn_middle",
    "turn_late",
    "hu_response",
    "meld_response",
)
SOURCE_NAMES = (
    "sl",
    "rule_fast",
    "rule_safe",
    "current",
    "frozen_policy",
    "mc_teacher",
)
PANEL_NAMES = ("rules", "sl", "mixed", "history")
PROGRESS_NAMES = ("early", "middle", "late")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_number(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _format_integer(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{int(number):,}"


def _format_percent(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{100.0 * number:.1f}%"


def _format_boolean(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def _format_rate(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M/s"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}k/s"
    return f"{number:.0f}/s"


def _format_delta(value: Any, baseline: Any, digits: int) -> str:
    current = _number(value)
    reference = _number(baseline)
    if current is None or reference is None:
        return "baseline unavailable"
    delta = current - reference
    return f"{delta:+.{digits}f} vs SL baseline"


def _latest(
    records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any] | None:
    return next((record for record in reversed(records) if predicate(record)), None)


def _record_step(record: dict[str, Any]) -> float | None:
    iteration = _number(record.get("iteration"))
    if iteration is not None:
        return iteration
    if record.get("phase") in {"baseline", "critic_warmup"}:
        return 0.0
    return None


def _series(
    records: list[dict[str, Any]], extractor: Callable[[dict[str, Any]], Any]
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for record in records:
        step = _record_step(record)
        value = _number(extractor(record))
        if step is not None and value is not None:
            points.append((step, value))
    return points


def _card(label: str, value: str, detail: str = "", tone: str = "neutral") -> str:
    return (
        f'<div class="metric tone-{tone}"><div class="metric-label">'
        + html.escape(label)
        + '</div><div class="metric-value">'
        + html.escape(value)
        + '</div><div class="metric-detail">'
        + html.escape(detail)
        + "</div></div>"
    )


def _line_chart(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    lower_is_better: bool = False,
    y_bounds: tuple[float, float] | None = None,
) -> str:
    width, height = 700, 238
    left, right, top, bottom = 52, 18, 22, 34
    populated = [(label, color, points) for label, color, points in series if points]
    if not populated:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img">'
            '<text x="52" y="118" class="empty">No data yet</text></svg>'
        )
    all_points = [point for _, _, points in populated for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    values = [point[1] for point in all_points]
    if y_bounds is None:
        y_min, y_max = min(values), max(values)
        if math.isclose(y_min, y_max):
            margin = max(abs(y_min) * 0.05, 0.05)
        else:
            margin = (y_max - y_min) * 0.10
        y_min -= margin
        y_max += margin
    else:
        y_min, y_max = y_bounds
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-9)
    plot_width = width - left - right
    plot_height = height - top - bottom

    def coordinate(point: tuple[float, float]) -> tuple[float, float]:
        x = left + (point[0] - x_min) * plot_width / x_span
        y = top + (y_max - point[1]) * plot_height / y_span
        return x, y

    paths: list[str] = []
    legends: list[str] = []
    legend_x = left
    for label, color, points in populated:
        coordinates = [coordinate(point) for point in points]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
        paths.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
        last_x, last_y = coordinates[-1]
        paths.append(
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="{color}" />'
        )
        escaped = html.escape(label)
        legends.append(
            f'<line x1="{legend_x}" y1="10" x2="{legend_x + 15}" y2="10" '
            f'stroke="{color}" stroke-width="3" /><text x="{legend_x + 20}" '
            f'y="14" class="legend">{escaped}</text>'
        )
        legend_x += 30 + 7 * len(label)
    direction = "lower is better" if lower_is_better else "higher is better"
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{direction}">
  {''.join(legends)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis" />
  <text x="{left - 8}" y="{top + 4}" text-anchor="end" class="axis-label">{y_max:.2f}</text>
  <text x="{left - 8}" y="{height - bottom + 4}" text-anchor="end" class="axis-label">{y_min:.2f}</text>
  <text x="{left}" y="{height - 9}" class="axis-label">u{int(x_min)}</text>
  <text x="{width - right}" y="{height - 9}" text-anchor="end" class="axis-label">u{int(x_max)}</text>
  {''.join(paths)}
</svg>
"""


def _evaluation_rows(evaluation: dict[str, Any]) -> str:
    panels = _mapping(evaluation.get("panels"))
    rows: list[str] = []
    for name in PANEL_NAMES:
        metrics = _mapping(panels.get(name))
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_format_number(metrics.get('mean_rank'), 2)}</td>"
            f"<td>{_format_number(metrics.get('mean_score_delta'), 0)}</td>"
            f"<td>{_format_percent(metrics.get('first_rate'))}</td>"
            f"<td>{_format_percent(metrics.get('last_rate'))}</td>"
            f"<td>{_format_integer(metrics.get('games'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _progress_rows(validation: dict[str, Any]) -> str:
    progress = _mapping(validation.get("progress"))
    rows: list[str] = []
    for name in PROGRESS_NAMES:
        metrics = _mapping(progress.get(name))
        q = _mapping(metrics.get("q"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_format_integer(q.get('count'))}</td>"
            f"<td>{_format_number(q.get('loss'))}</td>"
            f"<td>{_format_number(q.get('mae'))}</td>"
            f"<td>{_format_number(q.get('correlation'))}</td>"
            f"<td>{_format_number(q.get('calibration_error'))}</td>"
            f"<td>{_format_percent(q.get('improvement'))}</td>"
            f"<td>{_format_number(metrics.get('q_disagreement'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _bar(fraction: float, tone: str) -> str:
    width = max(0.0, min(100.0, 100.0 * fraction))
    return (
        '<div class="bar-track" aria-hidden="true"><span class="bar-fill '
        + html.escape(tone)
        + f'" style="width:{width:.2f}%"></span></div>'
    )


def _source_rows(replay: dict[str, Any]) -> str:
    counts = _mapping(replay.get("sources"))
    total = sum(max(_number(counts.get(name)) or 0.0, 0.0) for name in SOURCE_NAMES)
    tones = ("cyan", "green", "amber", "red", "blue", "violet")
    rows: list[str] = []
    for name, tone in zip(SOURCE_NAMES, tones):
        count = max(_number(counts.get(name)) or 0.0, 0.0)
        fraction = count / total if total else 0.0
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_format_integer(count)}</td>"
            f"<td>{_format_percent(fraction)}</td>"
            f'<td class="bar-cell">{_bar(fraction, tone)}</td>'
            "</tr>"
        )
    return "".join(rows)


def _category_rows(
    replay: dict[str, Any], validation: dict[str, Any], actor: dict[str, Any]
) -> str:
    counts = _mapping(replay.get("categories"))
    metrics = _mapping(validation.get("categories"))
    total = sum(max(_number(counts.get(name)) or 0.0, 0.0) for name in CATEGORY_NAMES)
    rows: list[str] = []
    for name in CATEGORY_NAMES:
        count = max(_number(counts.get(name)) or 0.0, 0.0)
        fraction = count / total if total else 0.0
        category = _mapping(metrics.get(name))
        q = _mapping(category.get("q"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_format_integer(count)}</td>"
            f"<td>{_format_percent(fraction)}</td>"
            f"<td>{_format_number(q.get('loss'))}</td>"
            f"<td>{_format_number(q.get('mae'))}</td>"
            f"<td>{_format_number(q.get('correlation'))}</td>"
            f"<td>{_format_number(q.get('calibration_error'))}</td>"
            f"<td>{_format_number(category.get('q_disagreement'))}</td>"
            f"<td>{_format_number(actor.get('advantage_' + name))}</td>"
            f"<td>{_format_number(actor.get('ess_' + name), 0)}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_dashboard(metrics_path: Path, dashboard_path: Path) -> None:
    """Render a self-contained dashboard from the append-only JSONL metrics."""

    records = _read_records(metrics_path)
    latest_record = _latest(records, lambda record: record.get("phase") == "iteration")
    latest_validation_record = _latest(
        records, lambda record: isinstance(record.get("critic_validation"), dict)
    )
    latest_evaluation_record = _latest(
        records, lambda record: isinstance(record.get("fixed_evaluation"), dict)
    )
    baseline_record = _latest(records, lambda record: record.get("phase") == "baseline")
    latest_actor_record = _latest(records, lambda record: isinstance(record.get("actor"), dict))
    best_record = _latest(
        records, lambda record: isinstance(record.get("best_fixed_evaluation"), dict)
    )

    latest = latest_record or {}
    validation = _mapping(
        latest_validation_record.get("critic_validation")
        if latest_validation_record
        else None
    )
    evaluation = _mapping(
        latest_evaluation_record.get("fixed_evaluation")
        if latest_evaluation_record
        else None
    )
    fresh_evaluation = _mapping(
        latest_evaluation_record.get("fresh_evaluation")
        if latest_evaluation_record
        else None
    )
    baseline = _mapping(baseline_record.get("fixed_evaluation") if baseline_record else None)
    best = _mapping(
        best_record.get("best_fixed_evaluation") if best_record else baseline
    )
    actor = _mapping(latest_actor_record.get("actor") if latest_actor_record else None)
    replay = _mapping(latest.get("replay"))
    mc = _mapping(latest.get("mc"))
    mc_validation = _mapping(mc.get("validation_metrics"))
    mc_q = _mapping(mc_validation.get("q"))
    q = _mapping(validation.get("q"))
    oracle = _mapping(validation.get("oracle"))
    oracle_q = _mapping(oracle.get("q"))
    oracle_comparison = _mapping(validation.get("oracle_vs_partial"))
    mc_ranking = _mapping(mc_validation.get("action_ranking"))
    mc_critic = _mapping(latest.get("mc_critic"))
    mc_train_targets = mc.get("train_targets_after_trim", mc.get("train_targets"))
    mc_validation_targets = mc.get("validation_targets")
    mc_reliable_targets = mc.get("validation_reliable_targets")
    mc_reliable_pairs = mc_ranking.get(
        "pair_count", mc.get("validation_reliable_pairs")
    )
    mc_all_pairs = mc_ranking.get("all_pair_count", mc_ranking.get("pair_count"))
    mc_reliable_groups = mc.get(
        "validation_reliable_groups",
        mc_ranking.get("group_count"),
    )

    status_record = latest_record or latest_validation_record or {}
    iteration = int(_number(latest.get("iteration")) or 0)
    critic_steps = int(_number(status_record.get("critic_steps")) or 0)
    actor_updates = int(_number(latest.get("actor_updates")) or 0)
    status = str(records[-1].get("phase", "starting")) if records else "starting"
    gate = str(status_record.get("actor_gate", "waiting"))
    updated = (
        datetime.fromtimestamp(metrics_path.stat().st_mtime).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        if metrics_path.exists()
        else "not started"
    )

    fixed_rank = evaluation.get("mean_rank")
    fixed_score = evaluation.get("mean_score_delta")
    baseline_rank = baseline.get("mean_rank")
    baseline_score = baseline.get("mean_score_delta")
    best_rank = best.get("mean_rank")
    fresh_rank = fresh_evaluation.get("mean_rank")

    rank_series = _series(
        records, lambda record: _nested(record, "fixed_evaluation", "mean_rank")
    )
    fresh_rank_series = _series(
        records, lambda record: _nested(record, "fresh_evaluation", "mean_rank")
    )
    score_series = _series(
        records, lambda record: _nested(record, "fixed_evaluation", "mean_score_delta")
    )
    fresh_score_series = _series(
        records, lambda record: _nested(record, "fresh_evaluation", "mean_score_delta")
    )
    q_mae_series = _series(
        records, lambda record: _nested(record, "critic_validation", "q", "mae")
    )
    calibration_series = _series(
        records,
        lambda record: _nested(
            record, "critic_validation", "q", "calibration_error"
        ),
    )
    disagreement_series = _series(
        records,
        lambda record: _nested(record, "critic_validation", "q_disagreement"),
    )
    correlation_series = _series(
        records,
        lambda record: _nested(record, "critic_validation", "q", "correlation"),
    )
    actor_kl_series = _series(
        records, lambda record: _nested(record, "actor", "actor_reference_kl")
    )

    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blood Flow Mahjong IQL Training</title>
<style>
:root {{ color-scheme:dark; --bg:#111210; --surface:#1a1c19; --surface-2:#20231f; --line:#343831; --text:#eef0eb; --muted:#9da49a; --cyan:#56c5d0; --green:#75c98f; --amber:#e4b45e; --red:#e77f79; --blue:#81a9e0; --violet:#bd95db; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; letter-spacing:0; }}
main {{ max-width:1500px; margin:0 auto; padding:22px; }} header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin-bottom:16px; }}
h1 {{ margin:0; font-size:21px; font-weight:650; }} h2 {{ margin:0 0 11px; font-size:14px; font-weight:650; }} code {{ color:var(--cyan); }}
.subhead,.metric-label,.metric-detail,.axis-label,.legend,.empty {{ color:var(--muted); }} .subhead {{ margin-top:3px; font-variant-numeric:tabular-nums; }}
.metrics {{ display:grid; grid-template-columns:repeat(8,minmax(116px,1fr)); gap:8px; margin-bottom:12px; }} .metric {{ min-width:0; min-height:82px; padding:11px 12px; background:var(--surface); border:1px solid var(--line); border-top:2px solid var(--line); border-radius:6px; }}
.tone-cyan {{ border-top-color:var(--cyan); }} .tone-green {{ border-top-color:var(--green); }} .tone-amber {{ border-top-color:var(--amber); }} .tone-red {{ border-top-color:var(--red); }}
.metric-label {{ font-size:11px; }} .metric-value {{ margin-top:3px; overflow-wrap:anywhere; font-size:20px; font-variant-numeric:tabular-nums; }} .metric-detail {{ margin-top:3px; overflow-wrap:anywhere; font-size:10px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }} .panel {{ min-width:0; padding:14px; overflow:hidden; background:var(--surface); border:1px solid var(--line); border-radius:6px; }} .wide {{ grid-column:1 / -1; }}
svg {{ display:block; width:100%; height:auto; }} .axis {{ stroke:var(--line); stroke-width:1; }} .axis-label,.legend {{ font-size:10px; }}
.table-scroll {{ width:100%; overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; white-space:nowrap; }} th,td {{ padding:6px 7px; border-bottom:1px solid var(--line); text-align:right; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:10px; font-weight:550; }} tbody tr:last-child td {{ border-bottom:0; }}
.bar-cell {{ width:38%; }} .bar-track {{ width:100%; height:5px; overflow:hidden; background:var(--surface-2); border-radius:2px; }} .bar-fill {{ display:block; height:100%; }} .cyan {{ background:var(--cyan); }} .green {{ background:var(--green); }} .amber {{ background:var(--amber); }} .red {{ background:var(--red); }} .blue {{ background:var(--blue); }} .violet {{ background:var(--violet); }}
@media (max-width:1150px) {{ .metrics {{ grid-template-columns:repeat(4,minmax(120px,1fr)); }} }} @media (max-width:760px) {{ main {{ padding:12px; }} header {{ display:block; }} header .subhead:last-child {{ margin-top:7px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} .metric-value {{ font-size:18px; }} }}
</style></head><body><main>
<header><div><h1>Blood Flow Mahjong IQL</h1><div class="subhead">{html.escape(status)} · iteration {iteration:,} · critic {critic_steps:,} · actor {actor_updates:,}</div></div><div class="subhead">updated {html.escape(updated)} · refresh 15s</div></header>
<section class="metrics">
{_card("Fixed rank", _format_number(fixed_rank, 2), _format_delta(fixed_rank, baseline_rank, 2), "cyan")}
{_card("Fixed score", _format_number(fixed_score, 0), _format_delta(fixed_score, baseline_score, 0), "green")}
{_card("Fresh rank", _format_number(fresh_rank, 2), "secondary seed panel", "cyan")}
{_card("Best rank", _format_number(best_rank, 2), "fixed panel", "green")}
{_card("Q loss", _format_number(q.get('loss')), f"MAE {_format_number(q.get('mae'))}", "amber")}
{_card("Calibration", _format_number(q.get('calibration_error')), f"disagreement {_format_number(validation.get('q_disagreement'))}", "amber")}
{_card("Actor KL", _format_number(actor.get('actor_reference_kl'), 4), f"gate {gate} · streak {_format_integer(status_record.get('teacher_ready_streak'))}", "red")}
{_card("Replay", _format_integer(replay.get('states')), f"{_format_integer(replay.get('trajectories'))} trajectories", "neutral")}
</section>
<section class="metrics">
{_card("First rate", _format_percent(evaluation.get('first_rate')), "fixed rules", "green")}
{_card("Last rate", _format_percent(evaluation.get('last_rate')), "fixed rules", "red")}
{_card("Q correlation", _format_number(q.get('correlation')), f"improvement {_format_percent(q.get('improvement'))}", "amber")}
{_card("Q disagreement", _format_number(validation.get('q_disagreement')), "mean |Q1-Q2|", "amber")}
{_card("Actor advantage", _format_number(actor.get('actor_advantage_mean')), f"weight {_format_number(actor.get('actor_weight_mean'))}", "red")}
{_card("Actor ESS", _format_number(actor.get('actor_effective_sample_size'), 0), f"updates {actor_updates:,}", "red")}
{_card("Collection", _format_rate(_nested(latest, 'collection', 'states_per_second')), f"{_format_integer(_nested(latest, 'collection', 'trajectories'))} games", "cyan")}
{_card("Training", _format_rate(latest.get('training_states_per_second')), f"iteration {_format_number(latest.get('iteration_seconds'), 1)}s", "cyan")}
</section>
<section class="grid">
<div class="panel"><h2>Evaluation rank</h2>{_line_chart([('fixed', 'var(--cyan)', rank_series), ('fresh', 'var(--blue)', fresh_rank_series)], lower_is_better=True, y_bounds=(1.0, 4.0))}</div>
<div class="panel"><h2>Evaluation score</h2>{_line_chart([('fixed', 'var(--green)', score_series), ('fresh', 'var(--amber)', fresh_score_series)])}</div>
<div class="panel"><h2>Critic error</h2>{_line_chart([('Q MAE', 'var(--amber)', q_mae_series), ('calibration', 'var(--red)', calibration_series), ('disagreement', 'var(--violet)', disagreement_series)], lower_is_better=True)}</div>
<div class="panel"><h2>Critic correlation and Actor KL</h2>{_line_chart([('Q correlation', 'var(--green)', correlation_series), ('Actor KL', 'var(--cyan)', actor_kl_series)])}</div>
<div class="panel"><h2>Fixed evaluation panels</h2><div class="table-scroll"><table><thead><tr><th>opponents</th><th>rank</th><th>score</th><th>first</th><th>last</th><th>games</th></tr></thead><tbody>{_evaluation_rows(evaluation)}</tbody></table></div></div>
<div class="panel"><h2>Critic by game progress</h2><div class="table-scroll"><table><thead><tr><th>stage</th><th>n</th><th>Q loss</th><th>Q MAE</th><th>corr</th><th>calib</th><th>vs const</th><th>disagree</th></tr></thead><tbody>{_progress_rows(validation)}</tbody></table></div></div>
<div class="panel"><h2>Replay composition</h2><div class="table-scroll"><table><thead><tr><th>source</th><th>states</th><th>share</th><th></th></tr></thead><tbody>{_source_rows(replay)}</tbody></table></div></div>
<div class="panel"><h2>Replay and teacher diagnostics</h2><div class="table-scroll"><table><tbody><tr><th>anchor trajectories</th><td>{_format_integer(replay.get('anchor_trajectories'))}</td></tr><tr><th>online trajectories</th><td>{_format_integer(replay.get('online_trajectories'))}</td></tr><tr><th>MC targets (all)</th><td>{_format_integer(replay.get('mc_targets'))}</td></tr><tr><th>MC train targets (trimmed)</th><td>{_format_integer(mc_train_targets)}</td></tr><tr><th>MC anchor validation targets</th><td>{_format_integer(mc_validation_targets)}</td></tr><tr><th>MC reliable validation targets</th><td>{_format_integer(mc_reliable_targets)}</td></tr><tr><th>MC accepted</th><td>{_format_integer(mc.get('accepted_targets'))}</td></tr><tr><th>MC terminal rollouts</th><td>{_format_integer(mc.get('terminal_rollouts'))}</td></tr><tr><th>MC variance / CI half-width</th><td>{_format_number(mc.get('mean_variance'))} / {_format_number(mc.get('mean_confidence_half_width'))}</td></tr><tr><th>MC validation Q MAE (diagnostic)</th><td>{_format_number(mc_q.get('mae'))}</td></tr><tr><th>MC train pairwise accuracy</th><td>{_format_percent(mc_critic.get('mc_train_pairwise_accuracy'))}</td></tr><tr><th>MC validation pairwise accuracy</th><td>{_format_percent(mc_ranking.get('pairwise_accuracy'))}</td></tr><tr><th>MC significant pairs (reliable / all)</th><td>{_format_integer(mc_reliable_pairs)} / {_format_integer(mc_all_pairs)}</td></tr><tr><th>MC reliable validation groups</th><td>{_format_integer(mc_reliable_groups)}</td></tr><tr><th>MC validation frozen</th><td>{_format_boolean(mc.get('validation_frozen'))}</td></tr><tr><th>MC action-difference loss</th><td>{_format_number(mc_critic.get('mc_centered_loss'))}</td></tr><tr><th>MC pairwise loss</th><td>{_format_number(mc_critic.get('mc_pairwise_loss'))}</td></tr><tr><th>MC critic groups / pairs</th><td>{_format_integer(mc_critic.get('mc_train_groups'))} / {_format_integer(mc_critic.get('mc_train_pairs'))}</td></tr><tr><th>MC critic seconds</th><td>{_format_number(latest.get('mc_critic_seconds'), 1)}</td></tr><tr><th>Oracle Q MAE</th><td>{_format_number(oracle_q.get('mae'))}</td></tr><tr><th>Oracle relative MAE gain</th><td>{_format_percent(oracle_comparison.get('q_relative_mae_gain'))}</td></tr><tr><th>policy version</th><td>{_format_integer(latest.get('policy_version'))}</td></tr><tr><th>critic / actor seconds</th><td>{_format_number(latest.get('critic_seconds'), 1)} / {_format_number(latest.get('actor_seconds'), 1)}</td></tr></tbody></table></div></div>
<div class="panel wide"><h2>Decision coverage and diagnostics</h2><div class="table-scroll"><table><thead><tr><th>category</th><th>states</th><th>share</th><th>Q loss</th><th>Q MAE</th><th>corr</th><th>calib</th><th>disagree</th><th>advantage</th><th>ESS</th></tr></thead><tbody>{_category_rows(replay, validation, actor)}</tbody></table></div></div>
</section></main></body></html>"""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dashboard_path.with_suffix(dashboard_path.suffix + ".tmp")
    temporary.write_text(html_text, encoding="utf-8")
    temporary.replace(dashboard_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="metrics.jsonl path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics.with_name("dashboard.html")
    render_dashboard(args.metrics, output)
    print(output)


if __name__ == "__main__":
    main()
