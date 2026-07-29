from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from training import search_policy_panel as panel


def test_parse_actor() -> None:
    assert panel._parse_actor("candidate=runs/candidate.pt") == (
        "candidate",
        Path("runs/candidate.pt"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        panel._parse_actor("missing=")
    with pytest.raises(argparse.ArgumentTypeError):
        panel._parse_actor("bad__name=actor.pt")


def test_parser_accepts_extended_panel_options() -> None:
    args = panel.build_parser().parse_args(
        [
            "--batch-sweep-dir",
            "batch",
            "--output-dir",
            "output",
            "--actor",
            "candidate=actor.pt",
            "--evaluation-games",
            "65536",
            "--reuse-panel-prefix-dir",
            "prefix",
        ]
    )
    assert args.evaluation_games == 65_536
    assert args.reuse_panel_prefix_dir == Path("prefix")


def test_load_prefix_identity_validates_actors_and_size(tmp_path: Path) -> None:
    candidates = (
        {"name": "candidate", "path": "/actor.pt", "sha256": "abc"},
    )
    identity = {
        "input_directory": "/batch",
        "input_identity_fingerprint": "input-hash",
        "seeds": [7, 8],
        "evaluation_games_per_seed": 16_384,
        "actors": list(candidates),
    }
    (tmp_path / "config.json").write_text(json.dumps(identity))

    loaded, games = panel._load_prefix_identity(
        tmp_path,
        candidates=candidates,
        input_directory=Path("/batch"),
        input_fingerprint="input-hash",
        seeds=(7, 8),
        maximum_games=65_536,
    )
    assert loaded == identity
    assert games == 16_384

    with pytest.raises(ValueError, match="incompatible"):
        panel._load_prefix_identity(
            tmp_path,
            candidates=candidates,
            input_directory=Path("/batch"),
            input_fingerprint="input-hash",
            seeds=(7, 8),
            maximum_games=16_384,
        )
