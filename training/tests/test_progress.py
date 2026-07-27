from __future__ import annotations

import io
import os

import pytest

from training.progress import Progress, ProgressSnapshot, format_snapshot


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty
        self.flushes = 0

    def isatty(self) -> bool:
        return self._tty

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def test_tty_updates_in_place_and_completion_finishes_the_line() -> None:
    clock = FakeClock()
    stream = FakeStream(True)
    progress = Progress(
        stream=stream,
        clock=clock,
        refresh_interval=0.1,
        log_interval=10.0,
    )

    progress.start("COLLECT", total=100, unit="states", fields={"mode": "rules"})
    clock.advance(2.0)
    snapshot = progress.update(20, fields={"loss": 0.125})

    assert snapshot.rate == pytest.approx(10.0)
    assert snapshot.eta == pytest.approx(8.0)
    line = format_snapshot(snapshot)
    assert "COLLECT" in line
    assert "20/100 20.0%" in line
    assert "10.0 states/s" in line
    assert "elapsed 2s" in line
    assert "ETA 8s" in line
    assert "mode=rules" in line
    assert "loss=0.125" in line

    clock.advance(8.0)
    completed = progress.complete(fields={"saved": True})
    output = stream.getvalue()
    assert completed.current == 100
    assert completed.completed
    assert not progress.active
    assert output.count("\r") == 3
    assert output.count("\x1b[K") == 3
    assert output.endswith("  done\x1b[K\n")
    assert "saved=yes" in output


def test_tty_long_progress_is_truncated_before_it_can_wrap(monkeypatch) -> None:
    class TerminalStream(FakeStream):
        def fileno(self) -> int:
            return 123

    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((40, 24)))
    stream = TerminalStream(True)
    progress = Progress(stream=stream, clock=FakeClock(), refresh_interval=0)

    progress.start(
        "U1_TARGETS",
        total=2_304,
        unit="queries",
        fields={
            "worlds": 16,
            "rollout_step": 42,
            "active_branches": 512,
            "rollout_states_per_second": 12_345.6,
        },
    )
    progress.update(64)
    progress.complete()

    output = stream.getvalue()
    rendered_lines = [part.split("\x1b[K", 1)[0] for part in output.split("\r")[1:]]
    assert len(rendered_lines) == 3
    assert all(len(line) <= 39 for line in rendered_lines)
    assert all(line.endswith("...") for line in rendered_lines)
    assert output.count("\n") == 1


def test_non_tty_emits_periodically_and_completion_is_never_suppressed() -> None:
    clock = FakeClock()
    stream = FakeStream(False)
    progress = Progress(stream=stream, clock=clock, log_interval=10.0)

    progress.start("SEARCH", total=50, unit="worlds")
    initial = stream.getvalue()
    assert initial.count("\n") == 1
    assert "\r" not in initial

    clock.advance(5.0)
    progress.update(10)
    assert stream.getvalue() == initial

    clock.advance(5.0)
    progress.update(20)
    assert stream.getvalue().count("\n") == 2

    clock.advance(1.0)
    progress.complete(current=25, fields={"reason": "quota"})
    output = stream.getvalue()
    assert output.count("\n") == 3
    assert "25/50 50.0%" in output.splitlines()[-1]
    assert "reason=quota" in output.splitlines()[-1]
    assert output.splitlines()[-1].endswith("done")


def test_unknown_total_has_no_eta_and_fields_are_merged() -> None:
    clock = FakeClock()
    stream = FakeStream(False)
    progress = Progress(stream=stream, clock=clock, log_interval=100.0)

    progress.start("TRAIN", total=None, current=3, fields={"rank": 2.5})
    clock.advance(4.0)
    snapshot = progress.update(advance=8, fields={"score": -120})

    assert snapshot.current == 11
    assert snapshot.rate == pytest.approx(2.0)
    assert snapshot.eta is None
    assert snapshot.fields == {"rank": 2.5, "score": -120}
    completed = progress.complete()
    assert completed.current == 11
    assert "ETA --" in stream.getvalue().splitlines()[-1]


def test_phase_lifecycle_and_progress_validation() -> None:
    progress = Progress(stream=FakeStream(False), clock=FakeClock())

    with pytest.raises(RuntimeError, match="no active"):
        progress.update(1)
    with pytest.raises(ValueError, match="positive"):
        progress.start("bad", total=0)

    progress.start("GOOD", total=10, current=2)
    with pytest.raises(RuntimeError, match="complete"):
        progress.start("OTHER", total=1)
    with pytest.raises(ValueError, match="backwards"):
        progress.update(1)
    with pytest.raises(ValueError, match="not both"):
        progress.update(3, advance=1)
    progress.complete(current=5)

    progress.start("NEXT", total=1)
    progress.complete()


def test_public_snapshot_formatter_accepts_integer_counts_and_stays_single_line() -> None:
    line = format_snapshot(
        ProgressSnapshot(
            phase="PHASE",
            current=1,
            total=2,
            rate=1.0,
            elapsed=1.0,
            eta=1.0,
            unit="item",
            fields={"note\nname": "line one\nline two"},
        )
    )

    assert "1/2 50.0%" in line
    assert "note name=line one line two" in line
    assert "\n" not in line
