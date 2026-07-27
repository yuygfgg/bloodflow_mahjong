"""Small, thread-free terminal progress reporting.

``Progress`` writes in place when attached to a TTY and emits periodic normal
lines when output is redirected to a log.  Callers drive it synchronously from
their existing loop; there is no background refresh thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import time
from typing import Callable, Mapping, TextIO


FieldValue = str | int | float | bool | None


@dataclass(frozen=True)
class ProgressSnapshot:
    phase: str
    current: float
    total: float | None
    rate: float | None
    elapsed: float
    eta: float | None
    unit: str
    fields: Mapping[str, FieldValue]
    completed: bool = False


@dataclass
class _Phase:
    name: str
    current: float
    initial: float
    total: float | None
    unit: str
    started_at: float
    fields: dict[str, FieldValue]


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _format_count(value: float) -> str:
    normalized = float(value)
    if normalized.is_integer():
        return f"{int(normalized):,}"
    magnitude = abs(normalized)
    if magnitude >= 1_000:
        return f"{normalized:,.1f}"
    return f"{normalized:.2f}".rstrip("0").rstrip(".")


def _format_rate(value: float | None, unit: str) -> str:
    suffix = f" {unit}/s" if unit else "/s"
    if value is None:
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


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(float(seconds), 0.0)
    rounded = int(seconds + 0.5)
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


def _format_field(value: FieldValue) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4g}"
    return str(value).replace("\r", " ").replace("\n", " ")


def _fit_terminal_line(line: str, columns: int) -> str:
    # Leave the final column unused: writing into it can trigger an automatic
    # wrap before the following erase-to-end-of-line control sequence.
    limit = max(columns - 1, 1)
    if len(line) <= limit:
        return line
    if limit <= 3:
        return line[:limit]
    return line[: limit - 3] + "..."


def format_snapshot(snapshot: ProgressSnapshot) -> str:
    """Render one compact, single-line progress snapshot."""

    current = _format_count(snapshot.current)
    if snapshot.total is None:
        amount = current
    else:
        total = _format_count(snapshot.total)
        percent = 100.0 * snapshot.current / snapshot.total
        amount = f"{current}/{total} {percent:.1f}%"
    parts = [
        snapshot.phase,
        amount,
        _format_rate(snapshot.rate, snapshot.unit),
        f"elapsed {_format_duration(snapshot.elapsed)}",
        f"ETA {_format_duration(snapshot.eta)}",
    ]
    parts.extend(
        f"{str(name).replace(chr(13), ' ').replace(chr(10), ' ')}="
        f"{_format_field(value)}"
        for name, value in snapshot.fields.items()
    )
    if snapshot.completed:
        parts.append("done")
    return "  ".join(parts)


class Progress:
    """Synchronous progress reporter for one phase at a time.

    A phase must be explicitly completed before another phase is started.  TTY
    output is refreshed at ``refresh_interval`` and finalized with a newline.
    Non-TTY output is emitted at ``log_interval`` so redirected training logs
    stay useful without becoming noisy.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        tty: bool | None = None,
        refresh_interval: float = 0.1,
        log_interval: float = 30.0,
    ) -> None:
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.refresh_interval = _finite_nonnegative(
            refresh_interval, "refresh_interval"
        )
        self.log_interval = _finite_nonnegative(log_interval, "log_interval")
        if tty is None:
            try:
                tty = bool(self.stream.isatty())
            except (AttributeError, OSError):
                tty = False
        self.tty = bool(tty)
        self._phase: _Phase | None = None
        self._last_emit: float | None = None

    @property
    def active(self) -> bool:
        return self._phase is not None

    def start(
        self,
        phase: str,
        *,
        total: float | None,
        current: float = 0,
        unit: str = "items",
        fields: Mapping[str, FieldValue] | None = None,
    ) -> ProgressSnapshot:
        if self._phase is not None:
            raise RuntimeError("complete the active progress phase before starting another")
        name = str(phase).strip()
        if not name or "\n" in name or "\r" in name:
            raise ValueError("phase must be a non-empty single-line name")
        initial = _finite_nonnegative(current, "current")
        normalized_total = (
            None if total is None else _finite_nonnegative(total, "total")
        )
        if normalized_total == 0:
            raise ValueError("total must be positive when provided")
        now = float(self.clock())
        if not math.isfinite(now):
            raise ValueError("clock must return a finite value")
        self._phase = _Phase(
            name=name,
            current=initial,
            initial=initial,
            total=normalized_total,
            unit=str(unit).strip(),
            started_at=now,
            fields=dict(fields or {}),
        )
        snapshot = self._snapshot(now, completed=False)
        self._emit(snapshot, now)
        return snapshot

    def update(
        self,
        current: float | None = None,
        *,
        advance: float | None = None,
        fields: Mapping[str, FieldValue] | None = None,
        force: bool = False,
    ) -> ProgressSnapshot:
        phase = self._require_phase()
        if current is not None and advance is not None:
            raise ValueError("provide current or advance, not both")
        if advance is not None:
            delta = _finite_nonnegative(advance, "advance")
            next_current = phase.current + delta
        elif current is not None:
            next_current = _finite_nonnegative(current, "current")
        else:
            next_current = phase.current
        if next_current < phase.current:
            raise ValueError("progress current cannot move backwards")
        phase.current = next_current
        if fields:
            phase.fields.update(fields)
        now = float(self.clock())
        snapshot = self._snapshot(now, completed=False)
        if force or self._due(now):
            self._emit(snapshot, now)
        return snapshot

    def complete(
        self,
        current: float | None = None,
        *,
        fields: Mapping[str, FieldValue] | None = None,
    ) -> ProgressSnapshot:
        phase = self._require_phase()
        if current is None:
            if phase.total is not None and phase.current <= phase.total:
                phase.current = phase.total
        else:
            next_current = _finite_nonnegative(current, "current")
            if next_current < phase.current:
                raise ValueError("progress current cannot move backwards")
            phase.current = next_current
        if fields:
            phase.fields.update(fields)
        now = float(self.clock())
        snapshot = self._snapshot(now, completed=True)
        self._emit(snapshot, now, final=True)
        self._phase = None
        self._last_emit = None
        return snapshot

    def snapshot(self) -> ProgressSnapshot:
        return self._snapshot(float(self.clock()), completed=False)

    def _require_phase(self) -> _Phase:
        if self._phase is None:
            raise RuntimeError("no active progress phase")
        return self._phase

    def _snapshot(self, now: float, *, completed: bool) -> ProgressSnapshot:
        phase = self._require_phase()
        elapsed = max(now - phase.started_at, 0.0)
        completed_amount = phase.current - phase.initial
        rate = completed_amount / elapsed if elapsed > 0 and completed_amount > 0 else None
        eta: float | None = None
        if phase.total is not None:
            remaining = max(phase.total - phase.current, 0.0)
            if remaining == 0:
                eta = 0.0
            elif rate is not None and rate > 0:
                eta = remaining / rate
        return ProgressSnapshot(
            phase=phase.name,
            current=phase.current,
            total=phase.total,
            rate=rate,
            elapsed=elapsed,
            eta=eta,
            unit=phase.unit,
            fields=dict(phase.fields),
            completed=completed,
        )

    def _due(self, now: float) -> bool:
        if self._last_emit is None:
            return True
        interval = self.refresh_interval if self.tty else self.log_interval
        return now - self._last_emit >= interval

    def _emit(
        self, snapshot: ProgressSnapshot, now: float, *, final: bool = False
    ) -> None:
        line = format_snapshot(snapshot)
        if self.tty:
            try:
                columns = os.get_terminal_size(self.stream.fileno()).columns
            except (AttributeError, OSError, ValueError):
                columns = None
            if columns is not None:
                line = _fit_terminal_line(line, columns)
            self.stream.write(f"\r{line}\x1b[K")
            if final:
                self.stream.write("\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self._last_emit = now


__all__ = ["FieldValue", "Progress", "ProgressSnapshot", "format_snapshot"]
