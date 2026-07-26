from __future__ import annotations

import numpy as np
import torch

from training.learner import (
    LearningBatch,
    LearningConfig,
    _validate_mc_group_batch,
    cql_scale_from_coverage,
    critic_ready,
    grouped_action_ranking_metrics,
    mc_group_critic_loss,
    oracle_teacher_ready,
    resolve_actor_gate,
)
from training.observation import bucket_history_width
from training.policy_pool import ReplaySource


def _batch() -> LearningBatch:
    size = 3
    legal = np.zeros((size, 115), dtype=np.bool_)
    legal[:, :3] = True
    return LearningBatch(
        tile_obs=np.zeros((size, 10, 27), dtype=np.uint8),
        melds=np.full((size, 4, 4, 3), 255, dtype=np.uint8),
        meta=np.zeros((size, 34), dtype=np.int32),
        events=np.zeros((size, 4, 8), dtype=np.int32),
        event_lengths=np.asarray([0, 2, 4], dtype=np.uint16),
        legal=legal,
        actions=np.asarray([0, 1, 2], dtype=np.uint8),
        returns=np.asarray([0.0, 0.5, -0.5], dtype=np.float32),
        categories=np.asarray([0, 5, 8], dtype=np.uint8),
        sources=np.asarray([0, 1, 3], dtype=np.uint8),
        policy_versions=np.zeros(size, dtype=np.uint32),
        behavior_probabilities=np.ones(size, dtype=np.float32),
        temperatures=np.ones(size, dtype=np.float32),
        trajectory_ids=np.arange(size, dtype=np.uint64),
        step_indices=np.zeros(size, dtype=np.uint16),
    )


def _reliable_edges(
    actions: np.ndarray | torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> np.ndarray | torch.Tensor:
    if isinstance(actions, torch.Tensor):
        result = torch.zeros((len(actions), 115), dtype=torch.bool)
        for left, right in pairs:
            result[left, actions[right]] = True
            result[right, actions[left]] = True
        return result
    result = np.zeros((len(actions), 115), dtype=np.bool_)
    for left, right in pairs:
        result[left, actions[right]] = True
        result[right, actions[left]] = True
    return result


def _mc_batch(reliable_actions: np.ndarray) -> LearningBatch:
    batch = _batch()
    return LearningBatch(
        **{
            **batch.__dict__,
            "sources": np.full(
                len(batch), int(ReplaySource.MC_TEACHER), dtype=np.uint8
            ),
            "trajectory_ids": np.full(len(batch), 7, dtype=np.uint64),
            "step_indices": np.full(len(batch), 11, dtype=np.uint16),
            "mc_query_ids": np.full(len(batch), 5, dtype=np.int64),
            "mc_candidate_counts": np.full(len(batch), 3, dtype=np.uint8),
            "mc_reliable_actions": reliable_actions,
        }
    )


def test_learning_batch_slices_history_and_checks_actions() -> None:
    batch = _batch()
    tensors = batch.tensors(torch.device("cpu"))
    assert tensors["events"].shape == (3, 4, 8)
    assert len(batch.subset(np.asarray([2, 0]))) == 2
    illegal = np.zeros((1, 115), dtype=np.bool_)
    illegal[0, 1] = True
    try:
        LearningBatch(
            **{
                **batch.subset(np.asarray([0])).__dict__,
                "legal": illegal,
            }
        )
    except ValueError as error:
        assert "illegal" in str(error)
    else:
        raise AssertionError("illegal action was accepted")


def test_mc_group_batch_rejects_invalid_reliable_action_graphs() -> None:
    actions = _batch().actions
    valid = _reliable_edges(actions, ((0, 1), (0, 2), (1, 2)))
    assert isinstance(valid, np.ndarray)
    _validate_mc_group_batch(_mc_batch(valid))

    self_edge = valid.copy()
    self_edge[0, actions[0]] = True
    outside = valid.copy()
    outside[0, 114] = True
    asymmetric = valid.copy()
    asymmetric[1, actions[0]] = False
    isolated = _reliable_edges(actions, ((0, 1),))
    assert isinstance(isolated, np.ndarray)
    cases = (
        (self_edge, "self edges"),
        (outside, "within the query"),
        (asymmetric, "symmetric"),
        (isolated, "requires a reliable action edge"),
    )
    for reliable, expected in cases:
        try:
            _validate_mc_group_batch(_mc_batch(reliable))
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"invalid reliable graph accepted: {expected}")


def test_history_width_uses_stable_buckets() -> None:
    assert bucket_history_width(np.asarray([0, 3]), 192) == 8
    assert bucket_history_width(np.asarray([9, 17]), 192) == 32
    assert bucket_history_width(np.asarray([129]), 192) == 192


def test_cql_uses_current_replay_coverage_not_update_number() -> None:
    config = LearningConfig()
    assert cql_scale_from_coverage(config, 0.0) == config.offline_cql_scale
    assert cql_scale_from_coverage(config, 0.5) == config.offline_cql_scale * 0.5
    assert cql_scale_from_coverage(config, 1.0) == (
        config.offline_cql_scale * config.minimum_cql_fraction
    )


def test_actor_gate_requires_middle_and_late_calibration() -> None:
    config = LearningConfig(minimum_critic_steps=10)
    good_q = {"count": 10.0, "improvement": 0.1, "correlation": 0.2}
    validation = {
        "q_disagreement": 0.1,
        "progress": {
            "middle": {"q": good_q},
            "late": {"q": good_q},
        },
    }
    assert critic_ready(validation, 9, config) == (False, "minimum_critic_steps")
    assert critic_ready(validation, 10, config) == (True, "ready")
    validation["progress"]["late"] = {
        "q": {**good_q, "improvement": -0.1}
    }
    assert critic_ready(validation, 10, config)[0] is False


def _q_metrics(
    *, improvement: float = 0.20, correlation: float = 0.30, mae: float = 0.30
) -> dict[str, float]:
    return {
        "count": 100.0,
        "improvement": improvement,
        "correlation": correlation,
        "mae": mae,
    }


def _oracle_validation() -> dict[str, object]:
    partial_progress = {
        stage: {"q": _q_metrics(improvement=0.10, mae=0.40)}
        for stage in ("early", "middle", "late")
    }
    oracle_progress = {
        stage: {
            "q": _q_metrics(improvement=0.20, mae=0.30),
            "v": _q_metrics(correlation=0.20),
        }
        for stage in ("early", "middle", "late")
    }
    return {
        "q": _q_metrics(improvement=0.10, mae=0.40),
        "q_disagreement": 0.10,
        "progress": partial_progress,
        "oracle": {
            "q": _q_metrics(improvement=0.20, mae=0.30),
            "q_disagreement": 0.10,
            "expectile_balance_error": 0.05,
            "progress": oracle_progress,
        },
        "oracle_vs_partial": {
            "q_relative_mae_gain": 0.25,
            "progress": {
                stage: {"q_relative_mae_gain": 0.25}
                for stage in ("early", "middle", "late")
            },
        },
    }


def test_oracle_teacher_must_outperform_partial() -> None:
    config = LearningConfig(minimum_critic_steps=10)
    validation = _oracle_validation()
    validation["oracle_vs_partial"]["q_relative_mae_gain"] = 0.0
    assert oracle_teacher_ready(validation, config) == (
        False,
        "oracle_overall_mae_gain",
    )


def test_all_actor_gates_fail_closed_on_nonfinite_metrics() -> None:
    config = LearningConfig(minimum_critic_steps=10)
    partial = _oracle_validation()
    partial["progress"]["middle"]["q"]["improvement"] = float("nan")
    assert critic_ready(partial, 10, config) == (
        False,
        "nonfinite_middle_metrics",
    )

    oracle = _oracle_validation()
    oracle["oracle_vs_partial"]["q_relative_mae_gain"] = float("nan")
    assert oracle_teacher_ready(oracle, config) == (
        False,
        "nonfinite_oracle_overall_mae_gain",
    )

    mc_base = _oracle_validation()
    del mc_base["oracle"]
    del mc_base["oracle_vs_partial"]
    mc_metrics = {
        "q": _q_metrics(),
        "action_ranking": {
            "group_count": 100.0,
            "pair_count": 100.0,
            "pairwise_accuracy": float("nan"),
            "mean_regret": 0.0,
        },
    }
    ready, reason, streak = resolve_actor_gate(
        "c",
        mc_base,
        10,
        config,
        2,
        mc_validation=mc_metrics,
        mc_train_targets=1000,
        mc_validation_targets=1000,
    )
    assert (ready, reason, streak) == (
        False,
        "nonfinite_mc_action_ranking",
        0,
    )


def test_oracle_actor_gate_requires_a_consecutive_streak_and_resets() -> None:
    config = LearningConfig(minimum_critic_steps=10, teacher_readiness_streak=3)
    validation = _oracle_validation()
    ready, gate, streak = resolve_actor_gate(
        "b", validation, 10, config, 0
    )
    assert (ready, gate, streak) == (
        False,
        "oracle_readiness_streak:1/3",
        1,
    )
    ready, gate, streak = resolve_actor_gate(
        "b", validation, 10, config, streak
    )
    assert (ready, gate, streak) == (
        False,
        "oracle_readiness_streak:2/3",
        2,
    )
    ready, gate, streak = resolve_actor_gate(
        "b", validation, 10, config, streak
    )
    assert (ready, gate, streak) == (True, "ready_oracle", 3)

    validation["oracle_vs_partial"]["progress"]["early"][
        "q_relative_mae_gain"
    ] = 0.0
    assert resolve_actor_gate("b", validation, 10, config, streak) == (
        False,
        "oracle_early_mae_gain",
        0,
    )


def test_grouped_mc_action_ranking_reports_accuracy_and_regret() -> None:
    actions = np.asarray([0, 1, 0, 1])
    metrics = grouped_action_ranking_metrics(
        np.asarray([0.9, 0.1, 0.9, 0.1]),
        np.asarray([1.0, 0.0, 0.0, 1.0]),
        np.asarray([10, 10, 20, 20]),
        np.asarray([3, 3, 4, 4]),
        actions,
        _reliable_edges(actions, ((0, 1), (2, 3))),
    )
    assert metrics["group_count"] == 2.0
    assert metrics["all_pair_count"] == 2.0
    assert metrics["pair_count"] == 2.0
    assert metrics["pairwise_accuracy"] == 0.5
    assert metrics["top_action_accuracy"] == 0.5
    assert metrics["mean_regret"] == 0.5

    incomplete_actions = np.asarray([0, 1])
    incomplete = grouped_action_ranking_metrics(
        np.asarray([0.9, 0.1]),
        np.asarray([1.0, 0.0]),
        np.asarray([10, 10]),
        np.asarray([3, 3]),
        incomplete_actions,
        _reliable_edges(incomplete_actions, ((0, 1),)),
        query_ids=np.asarray([7, 7]),
        expected_candidate_counts=np.asarray([3, 3]),
    )
    assert incomplete["group_count"] == 0.0
    assert incomplete["incomplete_group_count"] == 1.0
    assert incomplete["all_pair_count"] == 0.0
    assert incomplete["pair_count"] == 0.0


def test_grouped_mc_action_ranking_counts_only_reliable_pairs_and_groups() -> None:
    actions = np.asarray([0, 1, 0, 1])
    reliable = _reliable_edges(actions, ((0, 1),))
    metrics = grouped_action_ranking_metrics(
        np.asarray([0.9, 0.1, 0.1, 0.9]),
        np.asarray([1.0, 0.0, 1.0, 0.0]),
        np.asarray([10, 10, 20, 20]),
        np.asarray([3, 3, 4, 4]),
        actions,
        reliable,
    )
    assert metrics["group_count"] == 1.0
    assert metrics["all_pair_count"] == 2.0
    assert metrics["pair_count"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["top_action_accuracy"] == 1.0
    assert metrics["mean_regret"] == 0.0


def test_mc_group_loss_learns_action_differences_not_state_offsets() -> None:
    actions = torch.tensor([0, 1, 0, 1])
    returns = torch.tensor([1.0, -1.0, -0.5, 0.5])
    query_ids = torch.tensor([10, 10, 20, 20])
    q1 = torch.zeros((4, 115), requires_grad=True)
    q2 = torch.zeros((4, 115), requires_grad=True)
    reliable = _reliable_edges(actions, ((0, 1), (2, 3)))
    with torch.no_grad():
        q1[torch.arange(4), actions] = returns
        q2[torch.arange(4), actions] = returns * 0.8

    correct = mc_group_critic_loss(
        q1,
        q2,
        actions,
        returns,
        query_ids,
        reliable,
        huber_delta=0.25,
        centered_scale=1.0,
        pairwise_scale=0.25,
        pairwise_temperature=0.10,
    )
    shifted = mc_group_critic_loss(
        q1,
        q2,
        actions,
        returns + torch.tensor([10.0, 10.0, -4.0, -4.0]),
        query_ids,
        reliable,
        huber_delta=0.25,
        centered_scale=1.0,
        pairwise_scale=0.25,
        pairwise_temperature=0.10,
    )
    reversed_values = q1.detach().clone().requires_grad_(True)
    with torch.no_grad():
        reversed_values[torch.arange(4), actions] *= -1
    reversed_loss = mc_group_critic_loss(
        reversed_values,
        reversed_values,
        actions,
        returns,
        query_ids,
        reliable,
        huber_delta=0.25,
        centered_scale=1.0,
        pairwise_scale=0.25,
        pairwise_temperature=0.10,
    )

    torch.testing.assert_close(correct.loss, shifted.loss)
    torch.testing.assert_close(correct.centered_loss, shifted.centered_loss)
    torch.testing.assert_close(correct.pairwise_loss, shifted.pairwise_loss)
    assert correct.loss < reversed_loss.loss
    assert correct.pairwise_accuracy == 1.0
    assert correct.group_count == 2
    assert correct.pair_count == 2
    correct.loss.backward()
    assert q1.grad is not None and torch.isfinite(q1.grad).all()
    assert q2.grad is not None and torch.isfinite(q2.grad).all()


def test_mc_group_loss_handles_queries_without_significant_pairs() -> None:
    actions = torch.tensor([0, 1])
    q1 = torch.zeros((2, 115), requires_grad=True)
    q2 = torch.zeros((2, 115), requires_grad=True)
    output = mc_group_critic_loss(
        q1,
        q2,
        actions,
        torch.tensor([0.0, 0.01]),
        torch.tensor([7, 7]),
        _reliable_edges(actions, ()),
        huber_delta=0.25,
        centered_scale=1.0,
        pairwise_scale=0.25,
        pairwise_temperature=0.10,
    )
    assert output.pair_count == 0
    torch.testing.assert_close(output.pairwise_loss, torch.tensor(0.0))
    output.loss.backward()
    assert q1.grad is not None and torch.isfinite(q1.grad).all()
    assert q2.grad is not None and torch.isfinite(q2.grad).all()
    assert torch.count_nonzero(q1.grad) == 0
    assert torch.count_nonzero(q2.grad) == 0


def test_mc_group_loss_ignores_unreliable_edges_and_their_gradients() -> None:
    actions = torch.tensor([0, 1, 2])
    returns = torch.tensor([1.0, 0.0, -100.0])
    query_ids = torch.tensor([7, 7, 7])
    reliable = _reliable_edges(actions, ((0, 1),))
    q1 = torch.zeros((3, 115), requires_grad=True)
    q2 = torch.zeros((3, 115), requires_grad=True)
    output = mc_group_critic_loss(
        q1,
        q2,
        actions,
        returns,
        query_ids,
        reliable,
        huber_delta=0.25,
        centered_scale=1.0,
        pairwise_scale=0.25,
        pairwise_temperature=0.10,
    )
    changed_unreliable_target = mc_group_critic_loss(
        q1,
        q2,
        actions,
        torch.tensor([1.0, 0.0, 100.0]),
        query_ids,
        reliable,
        huber_delta=0.25,
        centered_scale=1.0,
        pairwise_scale=0.25,
        pairwise_temperature=0.10,
    )
    torch.testing.assert_close(output.loss, changed_unreliable_target.loss)
    assert output.group_count == 1
    assert output.pair_count == 1
    output.loss.backward()
    assert q1.grad is not None and torch.count_nonzero(q1.grad[2]) == 0
    assert q2.grad is not None and torch.count_nonzero(q2.grad[2]) == 0


def test_mc_actor_gate_requires_target_coverage_and_action_quality() -> None:
    config = LearningConfig(
        minimum_critic_steps=10,
        teacher_readiness_streak=2,
        minimum_mc_train_targets=10,
        minimum_mc_validation_targets=8,
        minimum_mc_validation_groups=3,
        minimum_mc_pairwise_pairs=4,
    )
    validation = _oracle_validation()
    del validation["oracle"]
    del validation["oracle_vs_partial"]
    mc_validation = {
        "action_ranking": {
            "group_count": 3.0,
            "pair_count": 6.0,
            "pairwise_accuracy": 0.75,
            "mean_regret": 0.05,
        },
    }
    assert resolve_actor_gate(
        "c",
        validation,
        10,
        config,
        0,
        mc_validation=mc_validation,
        mc_train_targets=9,
        mc_validation_targets=8,
    ) == (False, "mc_train_targets", 0)
    first = resolve_actor_gate(
        "c",
        validation,
        10,
        config,
        0,
        mc_validation=mc_validation,
        mc_train_targets=10,
        mc_validation_targets=8,
    )
    assert first == (False, "mc_readiness_streak:1/2", 1)
    assert resolve_actor_gate(
        "c",
        validation,
        10,
        config,
        first[2],
        mc_validation=mc_validation,
        mc_train_targets=10,
        mc_validation_targets=8,
    ) == (True, "ready_mc", 2)
