"""Compact terminal formatting with complete JSONL metric persistence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    rounded = int(max(float(seconds), 0.0) + 0.5)
    if rounded < 60:
        return f"{rounded}s"
    minutes, remaining = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m{remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def format_rate(value: float | None, unit: str) -> str:
    suffix = f" {unit}/s" if unit else "/s"
    if value is None or not math.isfinite(value):
        return f"--{suffix}"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        number = f"{value / 1_000_000:.2f}M"
    elif magnitude >= 1_000:
        number = f"{value / 1_000:.2f}k"
    elif magnitude >= 100:
        number = f"{value:.0f}"
    elif magnitude >= 10:
        number = f"{value:.1f}"
    else:
        number = f"{value:.2f}"
    return f"{number}{suffix}"


def format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{100.0 * number:.1f}%" if math.isfinite(number) else "--"
