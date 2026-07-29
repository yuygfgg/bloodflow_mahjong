from __future__ import annotations

import numpy as np
import pytest

import training.direction_generalization as diagnostic
import training.update_subspace_sweep as subspace
from training.model import BloodFlowTransformer, TransformerConfig
from training.policy_iteration import CounterfactualBatch, PolicyStateBatch


def tiny_batch() -> CounterfactualBatch:
    size = 9
    legal = np.zeros((size, 115), dtype=np.bool_)
    legal[:, :2] = True
    rank_q = np.zeros((size, 115), dtype=np.float32)
    rank_q[:, 1] = 0.2
    centered_rank_q = np.zeros_like(rank_q)
    centered_rank_q[:, 0] = -0.1
    centered_rank_q[:, 1] = 0.1
    score_q = rank_q * 100
    meta = np.zeros((size, 34), dtype=np.int32)
    meta[:, 4] = 30
    meta[:, 12:16] = 10_000
    meta[:, 24:28] = 14
    return CounterfactualBatch(
        query_ids=np.arange(size, dtype=np.int64),
        tile_obs=np.zeros((size, 10, 27), dtype=np.uint8),
        melds=np.full((size, 4, 4, 3), 255, dtype=np.uint8),
        meta=meta,
        events=np.zeros((size, 8, 8), dtype=np.int32),
        event_lengths=np.zeros(size, dtype=np.uint16),
        legal=legal,
        categories=np.arange(size, dtype=np.uint8),
        rank_q=rank_q,
        score_q=score_q,
        centered_rank_q=centered_rank_q,
        behavior_actions=np.zeros(size, dtype=np.uint8),
    )


def state_batch(batch: CounterfactualBatch) -> PolicyStateBatch:
    return PolicyStateBatch(
        **{
            name: getattr(batch, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
    )


def probabilities(first: float) -> np.ndarray:
    result = np.zeros((9, 115), dtype=np.float32)
    result[:, 0] = first
    result[:, 1] = 1.0 - first
    return result


def test_concatenated_probe_resets_overlapping_query_ids() -> None:
    states = state_batch(tiny_batch())

    joined = diagnostic._concatenate_states((states, states))

    assert len(joined) == 18
    np.testing.assert_array_equal(joined.query_ids, np.arange(18))
    np.testing.assert_array_equal(joined.categories, np.tile(np.arange(9), 2))


def test_policy_pair_metrics_detect_opposed_probability_directions() -> None:
    states = state_batch(tiny_batch())
    reference = probabilities(0.6)
    left = probabilities(0.4)
    right = probabilities(0.8)
    reference_actions = np.zeros(9, dtype=np.int64)
    left_actions = np.ones(9, dtype=np.int64)
    right_actions = np.zeros(9, dtype=np.int64)

    metrics = diagnostic._policy_pair_metrics(
        left,
        left_actions,
        right,
        right_actions,
        reference,
        reference_actions,
        states,
        np.full(9, 1 / 9),
    )

    assert metrics["probability_delta_cosine"] == pytest.approx(-1.0)
    assert metrics["action_disagreement_rate"] == pytest.approx(1.0)
    assert metrics["categories"]["turn_early"][
        "probability_delta_cosine"
    ] == pytest.approx(-1.0)


def test_heldout_metrics_report_soft_and_greedy_q_values() -> None:
    batch = tiny_batch()
    reference = probabilities(0.6)
    actor = probabilities(0.4)
    reference_actions = np.zeros(9, dtype=np.int64)
    actor_actions = np.ones(9, dtype=np.int64)

    metrics = diagnostic._heldout_metrics(
        actor,
        actor_actions,
        reference,
        reference_actions,
        batch,
        np.full(9, 1 / 9),
    )

    assert metrics["soft_rank_value"]["mean"] == pytest.approx(0.04)
    assert metrics["greedy_rank_value"]["mean"] == pytest.approx(0.2)
    assert metrics["greedy_flip_rate"] == pytest.approx(1.0)
    assert metrics["visitation_weighted_kl"] > 0


def test_reference_saturation_and_q_target_summaries_are_category_aware() -> None:
    batch = tiny_batch()
    states = state_batch(batch)
    reference = probabilities(0.9995)
    reference_actions = np.zeros(9, dtype=np.int64)

    policy = diagnostic._reference_policy_summary(
        reference, states, np.full(9, 1 / 9)
    )
    targets = diagnostic._heldout_target_summary(
        batch, reference_actions, np.full(9, 1 / 9)
    )

    assert policy["top_probability_ge_0_999_rate"] == pytest.approx(1.0)
    assert policy["categories"]["hu_response"]["mean_entropy"] > 0
    assert targets["mean_legal_q_range"] == pytest.approx(0.2)
    assert targets["reference_not_q_best_rate"] == pytest.approx(1.0)


def test_parser_defaults_to_small_common_kl() -> None:
    args = diagnostic.build_parser().parse_args(
        [
            "--batch-sweep-dir",
            "batch",
            "--optimizer-sweep-dir",
            "optimizer",
            "--output-dir",
            "output",
        ]
    )
    assert args.qpc == 256
    assert args.target_kl == pytest.approx(1e-4)


def test_seed_selection_requires_two_unique_seeds() -> None:
    assert diagnostic._normalize_seeds((7, 8, 9), (8, 7)) == (8, 7)
    with pytest.raises(ValueError, match="at least two"):
        diagnostic._normalize_seeds((7, 8), (7,))


def test_update_subspace_scopes_and_pooled_targets() -> None:
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=2,
            history_layers=2,
            ffn_dim=32,
            max_history=8,
        )
    )
    scopes = subspace._scope_parameter_names(actor)

    assert set(scopes) == {"full", "last_blocks", "actor"}
    assert len(scopes["actor"]) < len(scopes["last_blocks"]) < len(scopes["full"])
    assert all(name.startswith("actor.") for name in scopes["actor"])
    pooled = subspace._concatenate_targets((tiny_batch(), tiny_batch()))
    assert len(pooled) == 18
    np.testing.assert_array_equal(pooled.query_ids, np.arange(18))


def test_update_subspace_parser_defaults_cover_both_optimizers() -> None:
    args = subspace.build_parser().parse_args(
        ["--batch-sweep-dir", "batch", "--output-dir", "output"]
    )
    assert args.scopes == ["full", "last_blocks", "actor"]
    assert args.optimizers == ["adamw", "sgd"]
    assert args.target_kl == pytest.approx(1e-4)
