"""Deterministic cross-play evaluation for PPO checkpoints.

The normal PPO evaluation measures a policy against Rule-EV only. This module
keeps the same engine panel and action semantics, but replaces the three
opponents with another checkpoint. The resulting matrix shows whether later
snapshots improve against earlier snapshots, instead of only exploiting a
fixed rule policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .model import BloodFlowTransformer
from .pipeline import (
    EngineBuffers,
    evaluation_panel,
    focal_ranks,
    launch_inference,
    load_checkpoint_model,
)


@dataclass(frozen=True)
class Snapshot:
    label: str
    path: Path
    model: BloodFlowTransformer


@dataclass(frozen=True)
class MatchResult:
    focal: str
    opponent: str
    games: int
    mean_score_delta: float
    score_std: float
    score_se: float
    mean_rank: float
    rank_std: float
    rank_se: float
    first_rate: float
    last_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "focal": self.focal,
            "opponent": self.opponent,
            "games": self.games,
            "mean_score_delta": self.mean_score_delta,
            "score_std": self.score_std,
            "score_se": self.score_se,
            "mean_rank": self.mean_rank,
            "rank_std": self.rank_std,
            "rank_se": self.rank_se,
            "first_rate": self.first_rate,
            "last_rate": self.last_rate,
        }


def _validate_games(games: int) -> None:
    if games <= 0 or games % 4:
        raise ValueError("games must be a positive multiple of four")


def _matchup_panel(games: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed, seat-balanced panel used by Rule-EV evaluation."""
    _validate_games(games)
    return evaluation_panel(games, seed)


def _average_focal_ranks(scores: np.ndarray, seats: np.ndarray) -> np.ndarray:
    """Return average ranks, splitting a tied placement between its seats."""
    rows = np.arange(len(scores))
    own_scores = scores[rows, seats.astype(np.int64, copy=False)]
    greater = np.count_nonzero(scores > own_scores[:, None], axis=1)
    equal = np.count_nonzero(scores == own_scores[:, None], axis=1)
    return 1.0 + greater + 0.5 * (equal - 1)


def _seat_panel_se(values: np.ndarray) -> float:
    """Estimate error from independent seeds after averaging four seats."""
    panel_means = values.reshape(-1, 4).mean(axis=1)
    if len(panel_means) < 2:
        return 0.0
    return float(panel_means.std(ddof=1) / math.sqrt(len(panel_means)))


def evaluate_matchup(
    focal: Snapshot,
    opponent: Snapshot,
    *,
    games: int = 1024,
    envs: int = 256,
    seed: int = 0xA51CE,
    device: torch.device,
) -> MatchResult:
    """Evaluate one focal policy against three copies of another policy.

    Each requested ``(seed, focal seat)`` is run once. Finished rows remain
    disabled, so short games cannot be counted more than once. Both policies
    use deterministic (argmax) actions, matching the existing Rule-EV panel.
    """
    _validate_games(games)
    if envs <= 0:
        raise ValueError("envs must be positive")
    panel_seeds, focal_seats = _matchup_panel(games, seed)
    envs = min(envs, games)
    score_values = np.empty(games, dtype=np.float64)
    rank_values = np.empty(games, dtype=np.float64)
    competition_rank_values = np.empty(games, dtype=np.int8)
    completed = np.zeros(games, dtype=np.bool_)
    focal.model.eval()
    opponent.model.eval()

    for start in range(0, games, envs):
        stop = min(start + envs, games)
        chunk_seeds = panel_seeds[start:stop]
        chunk_focal_seats = focal_seats[start:stop]
        size = len(chunk_seeds)
        buffers = EngineBuffers.create(size, history=192)
        reset_flags = np.ones(size, dtype=np.uint8)
        history_seat_masks = np.full(size, 0x0F, dtype=np.uint8)
        buffers.batch.reset_and_observe_history_into(
            reset_flags,
            chunk_seeds,
            history_seat_masks,
            buffers.masks,
            buffers.tile_obs,
            buffers.melds,
            buffers.river,
            buffers.meta,
            buffers.events,
            buffers.event_lengths,
        )
        buffers.refresh_legal()
        active = np.ones(size, dtype=np.bool_)
        cumulative = np.zeros((size, 4), dtype=np.int64)
        chunk_focal_seats_i32 = chunk_focal_seats.astype(np.int32)

        while active.any():
            actors = buffers.meta[:, 1].astype(np.int32, copy=False)
            if focal.model is opponent.model:
                active_rows = np.flatnonzero(active)
                pending = launch_inference(
                    focal.model,
                    buffers,
                    active_rows,
                    device,
                    deterministic=True,
                )
                actions, _, _ = pending.resolve()
                buffers.actions.fill(0)
                buffers.actions[active_rows] = actions
            else:
                focal_rows = np.flatnonzero(
                    active & (actors == chunk_focal_seats_i32)
                )
                opponent_rows = np.flatnonzero(
                    active & (actors != chunk_focal_seats_i32)
                )
                focal_pending = launch_inference(
                    focal.model,
                    buffers,
                    focal_rows,
                    device,
                    deterministic=True,
                )
                opponent_pending = launch_inference(
                    opponent.model,
                    buffers,
                    opponent_rows,
                    device,
                    deterministic=True,
                )
                focal_actions, _, _ = focal_pending.resolve()
                opponent_actions, _, _ = opponent_pending.resolve()
                buffers.actions.fill(0)
                buffers.actions[focal_rows] = focal_actions
                buffers.actions[opponent_rows] = opponent_actions
            buffers.batch.step_masked_into(
                active.astype(np.uint8), buffers.actions, buffers.records
            )
            cumulative[active] += buffers.records[active, 5:9]
            terminal = active & buffers.records[:, 11].astype(bool)
            terminal_rows = np.flatnonzero(terminal)
            terminal_scores = cumulative[terminal_rows]
            terminal_average_ranks = _average_focal_ranks(
                terminal_scores, chunk_focal_seats[terminal_rows]
            )
            terminal_competition_ranks = focal_ranks(
                terminal_scores, chunk_focal_seats[terminal_rows]
            )
            for row, average_rank, competition_rank in zip(
                terminal_rows,
                terminal_average_ranks,
                terminal_competition_ranks,
                strict=True,
            ):
                game = start + int(row)
                score_values[game] = cumulative[row, chunk_focal_seats[row]]
                rank_values[game] = average_rank
                competition_rank_values[game] = competition_rank
                completed[game] = True
            active[terminal] = False
            if active.any():
                buffers.refresh()

    if not completed.all():
        raise RuntimeError("cross-play evaluation did not finish every game")
    return MatchResult(
        focal=focal.label,
        opponent=opponent.label,
        games=games,
        mean_score_delta=float(score_values.mean()),
        score_std=float(score_values.std()),
        score_se=_seat_panel_se(score_values),
        mean_rank=float(rank_values.mean()),
        rank_std=float(rank_values.std()),
        rank_se=_seat_panel_se(rank_values),
        first_rate=float(np.mean(competition_rank_values == 1)),
        last_rate=float(np.mean(competition_rank_values == 4)),
    )


def _parse_snapshot(spec: str) -> tuple[str, Path]:
    label, separator, raw_path = spec.partition("=")
    if not separator:
        path = Path(spec)
        return path.stem, path
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("snapshot must be PATH or LABEL=PATH")
    return label, Path(raw_path)


def _load_snapshots(
    specs: Iterable[str], device: torch.device
) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    labels: set[str] = set()
    for spec in specs:
        label, path = _parse_snapshot(spec)
        if label in labels:
            raise ValueError(f"duplicate snapshot label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        print(f"Loading {label}: {path}", flush=True)
        snapshots.append(Snapshot(label, path, load_checkpoint_model(path, device)))
    if len(snapshots) < 2:
        raise ValueError("at least two snapshots are required")
    return snapshots


def _format_result(result: MatchResult) -> str:
    return (
        f"{result.focal:>8} vs {result.opponent:<8} "
        f"rank {result.mean_rank:.3f} +/- {1.96 * result.rank_se:.3f}  "
        f"score {result.mean_score_delta:+.0f} +/- {1.96 * result.score_se:.0f}  "
        f"first {result.first_rate:.1%}"
    )


def _print_matrix(results: list[MatchResult], labels: list[str]) -> None:
    by_pair = {(item.focal, item.opponent): item for item in results}
    print("\nMean rank (focal policy vs three copies of column policy)")
    print("focal\\opponent  " + "  ".join(f"{label:>8}" for label in labels))
    for focal in labels:
        cells = []
        for opponent in labels:
            item = by_pair[(focal, opponent)]
            cells.append(f"{item.mean_rank:8.3f}")
        print(f"{focal:>15}  " + "  ".join(cells))
    print("\nMean score delta")
    print("focal\\opponent  " + "  ".join(f"{label:>8}" for label in labels))
    for focal in labels:
        cells = []
        for opponent in labels:
            item = by_pair[(focal, opponent)]
            cells.append(f"{item.mean_score_delta:+8.0f}")
        print(f"{focal:>15}  " + "  ".join(cells))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshots",
        nargs="+",
        help="checkpoint paths, optionally LABEL=PATH",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0xA51CE)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_games(args.games)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    snapshots = _load_snapshots(args.snapshots, device)
    results: list[MatchResult] = []
    total = len(snapshots) ** 2
    for focal in snapshots:
        for opponent in snapshots:
            print(
                f"[{len(results) + 1}/{total}] {focal.label} vs {opponent.label}",
                flush=True,
            )
            result = evaluate_matchup(
                focal,
                opponent,
                games=args.games,
                envs=args.envs,
                seed=args.seed,
                device=device,
            )
            results.append(result)
            print(_format_result(result), flush=True)
    _print_matrix(results, [snapshot.label for snapshot in snapshots])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "games": args.games,
            "envs": args.envs,
            "seed": args.seed,
            "rank_tie_method": "average",
            "standard_error_unit": "four-seat seed panel",
            "snapshots": [
                {"label": snapshot.label, "path": str(snapshot.path)}
                for snapshot in snapshots
            ],
            "results": [result.as_dict() for result in results],
        }
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
