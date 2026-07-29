from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from training import search_policy_validation as validation


def test_parse_actor() -> None:
    assert validation._parse_actor("candidate=runs/candidate.pt") == (
        "candidate",
        Path("runs/candidate.pt"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        validation._parse_actor("missing-path=")
    with pytest.raises(argparse.ArgumentTypeError):
        validation._parse_actor("ambiguous__name=actor.pt")


def test_parser_defaults() -> None:
    args = validation.build_parser().parse_args(
        [
            "--batch-sweep-dir",
            "batch",
            "--output-dir",
            "output",
            "--actor",
            "candidate=actor.pt",
        ]
    )
    assert args.qpc == 32
    assert args.worlds == 64
    assert args.corpus_dir is None
