from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import CollectionConfig, TrajectoryCollector
from training.trajectory import TrajectoryReplayError, replay_trajectory


def _trajectory():
    torch.manual_seed(3)
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=24,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=48,
            max_history=24,
        )
    ).eval()
    result = TrajectoryCollector(
        CollectionConfig(envs=1, history=24), actor, torch.device("cpu"), seed=31
    ).collect(1)
    return result.trajectories[0]


def test_strict_replay_reconstructs_viewer_inputs() -> None:
    trajectory = _trajectory()
    replayed = replay_trajectory(trajectory, history_capacity=24)
    assert replayed.tile_obs.shape == (len(trajectory), 10, 27)
    assert replayed.melds.shape == (len(trajectory), 4, 4, 3)
    assert replayed.meta.shape == (len(trajectory), 34)
    assert replayed.events.shape == (len(trajectory), 24, 8)
    assert replayed.legal_mask_words.shape == (len(trajectory), 2)
    assert np.all(replayed.event_lengths <= 24)


def test_strict_replay_rejects_corrupt_action_and_legal_count() -> None:
    trajectory = _trajectory()
    counts = trajectory.legal_counts.copy()
    counts[0] += 1
    with pytest.raises(TrajectoryReplayError, match="legal count"):
        replay_trajectory(replace(trajectory, legal_counts=counts), history_capacity=24)

    actions = trajectory.actions.copy()
    actions[0] = 114
    corrupt = replace(trajectory, actions=actions)
    with pytest.raises(TrajectoryReplayError, match="illegal|decision"):
        replay_trajectory(corrupt, history_capacity=24)
