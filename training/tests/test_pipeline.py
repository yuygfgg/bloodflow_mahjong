from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import (
    _pooled_panel_statistics,
    _shuffled_minibatches,
    EngineBuffers,
    HistoryCacheStore,
    OpponentPool,
    PPOConfig,
    RolloutCollector,
    TrainingController,
    TrainingControls,
    TransitionStorage,
    checkpoint_configs,
    evaluate_against_rule_ev,
    focal_ranks,
    hybrid_rewards,
    infer_actions,
    load_checkpoint,
    ppo_update,
    rank_utilities,
    save_checkpoint,
)


def tiny_model() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
            value_atoms=17,
        )
    )


def tiny_ppo(**changes: object) -> PPOConfig:
    config = PPOConfig(
        envs=4,
        rollout_transitions=16,
        ppo_epochs=1,
        minibatch=8,
        microbatch=4,
        frozen_snapshot_limit=2,
    )
    return replace(config, **changes)


def panel_evaluation(
    *,
    score: float = 200.0,
    rank: float = 2.30,
    score_panel_std: float = 20.0,
    rank_panel_std: float = 0.04,
    panel_count: int = 16,
) -> dict[str, Any]:
    divisor = np.sqrt(panel_count) if panel_count > 1 else 1.0
    return {
        "opponent": "rule-ev",
        "panel_count": float(panel_count),
        "mean_score_delta": score,
        "score_delta_panel_std": score_panel_std,
        "score_se": score_panel_std / divisor if panel_count > 1 else 0.0,
        "mean_rank": rank,
        "rank_panel_std": rank_panel_std,
        "rank_se": rank_panel_std / divisor if panel_count > 1 else 0.0,
    }


def evaluation_from_panels(
    scores: list[float], ranks: list[float]
) -> dict[str, float]:
    score_values = np.asarray(scores, dtype=np.float64)
    rank_values = np.asarray(ranks, dtype=np.float64)
    if score_values.shape != rank_values.shape or len(score_values) < 2:
        raise ValueError("test panels must have matching lengths of at least two")
    count = len(score_values)
    score_std = float(score_values.std(ddof=1))
    rank_std = float(rank_values.std(ddof=1))
    return {
        "panel_count": float(count),
        "mean_score_delta": float(score_values.mean()),
        "score_delta_panel_std": score_std,
        "score_se": score_std / np.sqrt(count),
        "mean_rank": float(rank_values.mean()),
        "rank_panel_std": rank_std,
        "rank_se": rank_std / np.sqrt(count),
    }


def test_rule_opponents_are_assigned_per_seat_after_metric_promotion() -> None:
    config = tiny_ppo()
    pool = OpponentPool(config, seed=19)
    learner_seats = np.array([0, 1, 2, 3] * 64, dtype=np.uint8)

    assert pool.stage() == "bootstrap"
    np.testing.assert_allclose(pool.probabilities(), [1.0 / 3.0, 2.0 / 3.0, 0.0])

    bootstrap = pool.assign_seats(learner_seats)
    assert bootstrap.shape == (len(learner_seats), 4)
    assert np.all(bootstrap[np.arange(len(learner_seats)), learner_seats] == -1)
    assert set(np.unique(bootstrap[bootstrap >= 0])) == {
        OpponentPool.RULE_FAST,
        OpponentPool.RULE_EV,
    }
    assert pool.refresh_snapshot(tiny_model(), torch.device("cpu")) is None

    pool.promote(tiny_model(), torch.device("cpu"), update=1)
    assert pool.stage() == "self_play"
    np.testing.assert_allclose(
        pool.probabilities(), [0.85 / 3.0, 1.70 / 3.0, 0.15]
    )
    self_play = pool.assign_seats(learner_seats)
    assert OpponentPool.FROZEN_TRANSFORMER in self_play
    assert set(np.unique(self_play[self_play >= 0])) <= {
        OpponentPool.RULE_FAST,
        OpponentPool.RULE_EV,
        OpponentPool.FROZEN_TRANSFORMER,
    }


def test_self_play_levels_adjust_fraction_and_snapshot_selection() -> None:
    config = tiny_ppo(
        historical_snapshot_probability=1.0,
    )
    pool = OpponentPool(config, seed=19)

    pool.promote(tiny_model(), torch.device("cpu"), update=100)
    pool.refresh_snapshot(tiny_model(), torch.device("cpu"), update=110)

    assert pool.self_play_level == 1
    assert pool.frozen_fraction == pytest.approx(0.15)
    assert pool.last_snapshot_update == 110
    assert pool.select_snapshot() == 0

    for update in range(111, 115):
        pool.promote(tiny_model(), torch.device("cpu"), update=update)
    assert pool.self_play_level == pool.maximum_self_play_level == 3
    assert pool.frozen_fraction == pytest.approx(0.45)
    assert pool.demote() == 2
    assert pool.frozen_fraction == pytest.approx(0.30)


def test_disabling_self_play_resets_a_restored_curriculum_level() -> None:
    source = tiny_ppo(self_play_enabled=True)
    pool = OpponentPool(source, seed=21)
    model = tiny_model()
    pool.promote(model, torch.device("cpu"), update=1)

    disabled = replace(source, self_play_enabled=False)
    restored = OpponentPool(disabled, seed=23)
    restored.load_state_dict(pool.state_dict(), model.config, torch.device("cpu"))

    assert restored.self_play_level == 0
    assert restored.stage() == "bootstrap"
    np.testing.assert_allclose(restored.probabilities(), [1.0 / 3.0, 2.0 / 3.0, 0.0])
    assert TrainingController(disabled).controls(restored.self_play_level).auxiliary_scale == 1.0


def test_snapshot_pool_keeps_the_fork_anchor() -> None:
    config = tiny_ppo(
        frozen_snapshot_limit=2,
    )
    pool = OpponentPool(config, seed=23)
    model = tiny_model()
    anchor = next(model.parameters()).detach().clone()
    pool.promote(model, torch.device("cpu"), update=1)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    pool.refresh_snapshot(model, torch.device("cpu"), update=2)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    pool.refresh_snapshot(model, torch.device("cpu"), update=3)

    assert len(pool.snapshots) == 2
    torch.testing.assert_close(next(pool.snapshots[0].parameters()), anchor)


@pytest.mark.parametrize(
    ("target_limit", "source_active", "expected_indices", "expected_active"),
    [
        (1, 2, (2,), 0),
        (2, 1, (0, 2), 1),
    ],
)
def test_restoring_snapshot_pool_enforces_the_current_limit(
    target_limit: int,
    source_active: int,
    expected_indices: tuple[int, ...],
    expected_active: int,
) -> None:
    source = OpponentPool(tiny_ppo(frozen_snapshot_limit=3), seed=127)
    model = tiny_model()
    source.promote(model, torch.device("cpu"), update=1)
    for update in (2, 3):
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        source.refresh_snapshot(model, torch.device("cpu"), update=update)
    source.active_snapshot = source_active
    expected = [
        next(source.snapshots[index].parameters()).detach().clone()
        for index in expected_indices
    ]

    restored = OpponentPool(
        tiny_ppo(frozen_snapshot_limit=target_limit), seed=131
    )
    restored.load_state_dict(
        source.state_dict(), model.config, torch.device("cpu")
    )

    assert len(restored.snapshots) == target_limit
    assert restored.active_snapshot == expected_active
    assert restored.frozen_model is restored.snapshots[expected_active]
    for snapshot, expected_parameter in zip(
        restored.snapshots, expected, strict=True
    ):
        torch.testing.assert_close(
            next(snapshot.parameters()), expected_parameter
        )


def test_pooled_panel_statistics_match_the_combined_panels() -> None:
    first_scores = [80.0, 120.0]
    second_scores = [160.0, 240.0]
    first_ranks = [2.4, 2.2]
    second_ranks = [2.3, 2.1]

    pooled = _pooled_panel_statistics(
        [
            evaluation_from_panels(first_scores, first_ranks),
            evaluation_from_panels(second_scores, second_ranks),
        ]
    )
    combined = evaluation_from_panels(
        first_scores + second_scores,
        first_ranks + second_ranks,
    )

    assert pooled == pytest.approx(combined)


@pytest.mark.parametrize(
    "evaluation",
    [
        panel_evaluation(panel_count=1),
        panel_evaluation(
            panel_count=4,
            score_panel_std=600.0,
            rank_panel_std=0.8,
        ),
    ],
)
def test_controller_rejects_insufficient_or_uncertain_single_evaluations(
    evaluation: dict[str, float],
) -> None:
    config = tiny_ppo(
        self_play_gate_window=1,
        self_play_gate_required_passes=1,
    )
    controller = TrainingController(config)

    decision = controller.observe_rule_ev(
        evaluation,
        self_play_level=0,
        maximum_self_play_level=3,
    )

    assert decision == "hold"
    assert controller.gate_statistics()["single_passes"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("panel_count", 1.5, "positive integer"),
        ("score_se", -1.0, "uncertainty"),
        ("rank_panel_std", -0.1, "uncertainty"),
        ("mean_rank", 4.1, "mean rank"),
        ("mean_score_delta", float("nan"), "non-finite"),
    ],
)
def test_controller_rejects_invalid_rule_ev_statistics(
    field: str,
    value: float,
    message: str,
) -> None:
    controller = TrainingController(tiny_ppo())
    evaluation = panel_evaluation()
    evaluation[field] = value

    with pytest.raises(ValueError, match=message):
        controller.observe_rule_ev(
            evaluation,
            self_play_level=0,
            maximum_self_play_level=3,
        )

    assert controller.evaluation_count == 0
    assert not controller.evaluation_window


def test_controller_rejects_human_analysis_results() -> None:
    controller = TrainingController(tiny_ppo())
    analysis = panel_evaluation()
    analysis["opponent"] = "rule-nn"

    with pytest.raises(ValueError, match="only Rule-EV"):
        controller.observe_rule_ev(
            analysis,
            self_play_level=0,
            maximum_self_play_level=3,
        )

    assert controller.state_dict() == TrainingController(tiny_ppo()).state_dict()


def test_controller_requires_a_full_window_and_enough_individual_passes() -> None:
    config = tiny_ppo(
        self_play_gate_window=3,
        self_play_gate_required_passes=2,
    )
    controller = TrainingController(config)
    neutral = panel_evaluation(score=50.0, rank=2.35)
    passing = panel_evaluation()

    for evaluation in (neutral, neutral):
        assert (
            controller.observe_rule_ev(
                evaluation,
                self_play_level=0,
                maximum_self_play_level=3,
            )
            == "hold"
        )
    assert (
        controller.observe_rule_ev(
            passing,
            self_play_level=0,
            maximum_self_play_level=3,
        )
        == "hold"
    )
    statistics = controller.gate_statistics()
    assert statistics["window_size"] == 3
    assert statistics["single_passes"] == 1
    assert controller._single_passes(statistics["pooled"])

    assert (
        controller.observe_rule_ev(
            passing,
            self_play_level=0,
            maximum_self_play_level=3,
        )
        == "promote"
    )
    assert controller.gate_statistics()["window_size"] == 0


def test_controller_promotes_and_demotes_with_hysteresis() -> None:
    config = tiny_ppo(
        self_play_gate_window=2,
        self_play_gate_required_passes=2,
    )
    controller = TrainingController(config)
    pool = OpponentPool(config, seed=29)
    model = tiny_model()

    assert (
        controller.observe_rule_ev(
            panel_evaluation(),
            self_play_level=pool.self_play_level,
            maximum_self_play_level=pool.maximum_self_play_level,
        )
        == "hold"
    )
    decision = controller.observe_rule_ev(
        panel_evaluation(),
        self_play_level=pool.self_play_level,
        maximum_self_play_level=pool.maximum_self_play_level,
    )
    assert decision == "promote"
    pool.promote(model, torch.device("cpu"), update=1)
    controller.mark_champion()
    assert pool.self_play_level == 1
    assert controller.last_decision_evaluation is not None
    assert controller.champion_rank == pytest.approx(
        controller.last_decision_evaluation["mean_rank"]
    )
    assert controller.champion_score_delta == pytest.approx(
        controller.last_decision_evaluation["mean_score_delta"]
    )

    failing = panel_evaluation(score=-200.0, rank=2.70)
    assert (
        controller.observe_rule_ev(
            failing,
            self_play_level=pool.self_play_level,
            maximum_self_play_level=pool.maximum_self_play_level,
        )
        == "hold"
    )
    decision = controller.observe_rule_ev(
        failing,
        self_play_level=pool.self_play_level,
        maximum_self_play_level=pool.maximum_self_play_level,
    )
    assert decision == "demote"
    assert pool.demote() == 0
    assert pool.stage() == "bootstrap"
    assert pool.frozen_ready


def test_controller_state_restores_the_next_evaluation_seed() -> None:
    config = tiny_ppo()
    controller = TrainingController(config, evaluation_seed=100)
    assert controller.take_evaluation_seed(8) == 100
    controller.observe_update({"entropy": 0.05})
    controller.observe_rule_ev(
        panel_evaluation(score=50.0, rank=2.35),
        self_play_level=0,
        maximum_self_play_level=3,
    )

    restored = TrainingController(config, evaluation_seed=999)
    restored.load_state_dict(controller.state_dict())

    assert restored.state_dict() == controller.state_dict()
    assert (
        restored.take_evaluation_seed(12)
        == controller.take_evaluation_seed(12)
        == 102
    )


def test_training_controls_follow_self_play_level() -> None:
    controller = TrainingController(tiny_ppo())

    assert [controller.controls(level).auxiliary_scale for level in range(5)] == [
        1.0,
        0.5,
        0.25,
        0.0,
        0.0,
    ]


def test_controller_adjusts_entropy_only_after_sustained_observations() -> None:
    config = tiny_ppo(
        self_play_enabled=False,
        entropy_patience_evaluations=2,
        entropy_adjustment=1.25,
    )
    controller = TrainingController(config)
    evaluation = panel_evaluation(score=0.0, rank=2.5)

    controller.observe_update({"entropy": config.entropy_target_low - 0.01})
    controller.observe_rule_ev(
        evaluation,
        self_play_level=0,
        maximum_self_play_level=3,
    )
    assert controller.current_entropy_coefficient == config.entropy_coefficient

    controller.observe_update({"entropy": config.entropy_target_low - 0.01})
    controller.observe_rule_ev(
        evaluation,
        self_play_level=0,
        maximum_self_play_level=3,
    )
    assert controller.current_entropy_coefficient == pytest.approx(
        config.entropy_coefficient * config.entropy_adjustment
    )

    for _ in range(config.entropy_patience_evaluations):
        controller.observe_update({"entropy": config.entropy_target_high + 0.01})
        controller.observe_rule_ev(
            evaluation,
            self_play_level=0,
            maximum_self_play_level=3,
        )
    assert controller.current_entropy_coefficient == pytest.approx(
        config.entropy_coefficient
    )


def test_rule_opponents_only_calculate_their_assigned_rows() -> None:
    class RecordingBatch:
        def __init__(self) -> None:
            self.fast_masks: list[np.ndarray] = []
            self.ev_masks: list[np.ndarray] = []

        def __len__(self) -> int:
            return 4

        def simple_rule_actions_masked_into(
            self, enabled: np.ndarray, actions: np.ndarray
        ) -> None:
            self.fast_masks.append(enabled.copy())
            actions[enabled.astype(bool)] = 10

        def rule_ev_actions_masked_into(
            self, enabled: np.ndarray, actions: np.ndarray, config: object
        ) -> None:
            self.ev_masks.append(enabled.copy())
            actions[enabled.astype(bool)] = 20

    batch = RecordingBatch()
    buffers = SimpleNamespace(
        batch=batch,
        meta=np.array(
            [[0, 0], [0, 1], [0, 2], [0, 3]],
            dtype=np.int32,
        ),
    )
    seat_kinds = np.array(
        [
            [-1, 0, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 1, 0],
            [2, 2, 2, 2],
        ],
        dtype=np.int8,
    )
    pool = OpponentPool(tiny_ppo(), seed=21)

    kinds = pool.action_kinds(buffers, seat_kinds)  # type: ignore[arg-type]
    actions = pool.rule_actions(buffers, kinds)  # type: ignore[arg-type]

    np.testing.assert_array_equal(kinds, [-1, 0, 1, 2])
    np.testing.assert_array_equal(batch.fast_masks, [[0, 1, 0, 0]])
    np.testing.assert_array_equal(batch.ev_masks, [[0, 0, 1, 0]])
    assert actions[1] == 10
    assert actions[2] == 20


def test_rollout_has_legal_finite_transitions_and_mixed_seat_counts() -> None:
    config = tiny_ppo()
    model = tiny_model()
    collector = RolloutCollector(config, torch.device("cpu"), seed=23)
    rollout = collector.collect(model, config.rollout_transitions)
    storage = rollout.storage
    slots = rollout.indices

    assert len(rollout) == config.rollout_transitions
    assert rollout.opponent_stage == "bootstrap"
    assert rollout.opponent_counts.sum() >= config.envs * 3
    assert rollout.opponent_counts[OpponentPool.FROZEN_TRANSFORMER] == 0
    assert np.all(storage.finalized[slots])
    assert np.all(storage.legal[slots, storage.actions[slots]])
    assert np.isfinite(storage.reward[slots]).all()
    assert np.isfinite(rollout.advantages[slots]).all()
    assert np.isfinite(rollout.returns[slots]).all()


def test_rank_utility_uses_the_evaluation_tie_convention() -> None:
    descending = np.array([[400, 300, 200, 100]] * 4, dtype=np.int64)
    seats = np.arange(4, dtype=np.uint8)

    np.testing.assert_array_equal(focal_ranks(descending, seats), [1, 2, 3, 4])
    np.testing.assert_allclose(
        rank_utilities(descending, seats),
        [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0],
    )

    ties = np.array([[100, 100, -100, -100]] * 4, dtype=np.int64)
    np.testing.assert_array_equal(focal_ranks(ties, seats), [1, 1, 3, 3])


def test_hybrid_reward_keeps_score_and_terminal_rank_signals() -> None:
    config = tiny_ppo(score_reward_weight=0.5, rank_reward_weight=2.0)
    score_deltas = np.array(
        [[100, -100, 0, 0], [0, -300, 300, 0]], dtype=np.int64
    )
    cumulative_scores = np.array(
        [[100, -100, 0, 0], [1_000, 500, -500, -1_000]], dtype=np.int64
    )

    rewards = hybrid_rewards(
        score_deltas,
        cumulative_scores,
        np.array([0, 1], dtype=np.uint8),
        np.array([False, True]),
        config,
    )

    np.testing.assert_allclose(rewards, [0.005, -0.015 + 2.0 / 3.0])


def test_sparse_resets_consume_unique_sequential_seeds() -> None:
    collector = RolloutCollector(tiny_ppo(), torch.device("cpu"), seed=100)

    collector._reset_rows(np.array([3], dtype=np.int64))
    first_seed = collector.reset_seeds[3]
    assert collector.learner_seats[3] == 0
    collector._reset_rows(np.array([2], dtype=np.int64))
    second_seed = collector.reset_seeds[2]
    assert collector.learner_seats[2] == 3

    assert first_seed != second_seed


def test_learner_seats_rotate_from_each_environment_offset() -> None:
    collector = RolloutCollector(tiny_ppo(), torch.device("cpu"), seed=101)
    rows = np.arange(4, dtype=np.int64)

    collector._reset_rows(rows)
    np.testing.assert_array_equal(collector.learner_seats, [1, 2, 3, 0])
    collector._reset_rows(rows)
    np.testing.assert_array_equal(collector.learner_seats, [2, 3, 0, 1])


def test_rollout_can_disable_auxiliary_labels() -> None:
    config = tiny_ppo(rollout_transitions=8)
    collector = RolloutCollector(config, torch.device("cpu"), seed=24)
    rollout = collector.collect(
        tiny_model(), config.rollout_transitions, collect_auxiliary=False
    )

    assert len(rollout) == config.rollout_transitions
    assert np.all(rollout.storage.shanten[rollout.indices] == 127)
    assert np.all(rollout.storage.improving[rollout.indices] == 0)


def test_gae_keeps_environment_reward_streams_separate() -> None:
    storage = TransitionStorage(4, history=1)
    storage.next_slot = 4
    storage.finalized[:] = True
    storage.env[:] = [0, 1, 0, 1]
    storage.episode[:] = 0
    storage.reward[:] = [1.0, 10.0, 2.0, 20.0]
    storage.value[:] = 0.0
    storage.next_value[:] = 0.0
    storage.done[:] = [False, False, True, True]

    advantages, returns = storage.compute_gae(
        np.arange(4, dtype=np.int64), gamma=1.0, gae_lambda=1.0
    )

    np.testing.assert_allclose(advantages, [3.0, 30.0, 2.0, 20.0])
    np.testing.assert_allclose(returns, advantages)


def test_rollout_cache_append_matches_full_history() -> None:
    model = tiny_model().eval()
    buffers = EngineBuffers.create(1, history=192)
    buffers.batch.reset_all(47)
    buffers.refresh()
    buffers.events[0, :3] = np.arange(24, dtype=np.int32).reshape(3, 8)
    buffers.event_lengths[0] = 3
    cache = HistoryCacheStore(max_history=192, min_cache_batch=1)

    cached_first = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
        history_cache=cache,
    )
    full_first = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
    )
    np.testing.assert_array_equal(cached_first[0], full_first[0])
    np.testing.assert_allclose(cached_first[1:], full_first[1:], rtol=1e-5, atol=1e-5)

    buffers.events[0, 3:5] = np.arange(16, dtype=np.int32).reshape(2, 8) + 100
    buffers.event_lengths[0] = 5
    cached_append = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
        history_cache=cache,
    )
    full_append = infer_actions(
        model,
        buffers,
        np.array([0], dtype=np.int64),
        torch.device("cpu"),
        deterministic=True,
    )
    np.testing.assert_array_equal(cached_append[0], full_append[0])
    np.testing.assert_allclose(cached_append[1:], full_append[1:], rtol=1e-5, atol=1e-5)
    assert cache.statistics()["hit_rows"] == 1
    assert cache.statistics()["cached_groups"] == 2


def test_ppo_monitor_completes_every_random_minibatch() -> None:
    torch.manual_seed(29)
    config = tiny_ppo(ppo_epochs=2, target_kl=1e-12, kl_control="monitor")
    model = tiny_model()
    collector = RolloutCollector(config, torch.device("cpu"), seed=31)
    rollout = collector.collect(model, config.rollout_transitions)
    controls = TrainingControls(
        learning_rate=1.5e-4,
        entropy_coefficient=0.007,
        auxiliary_scale=0.25,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=controls.learning_rate)
    before = model.actor[-1].weight.detach().clone()

    statistics = ppo_update(
        model, optimizer, rollout, config, torch.device("cpu"), controls
    )

    assert not torch.equal(before, model.actor[-1].weight.detach())
    assert all(np.isfinite(value) for value in statistics.values())
    assert statistics["updates"] == 4.0
    assert statistics["epochs"] == 2.0
    assert statistics["rolled_back_epochs"] == 0.0
    assert statistics["entropy_scale"] == controls.entropy_coefficient
    assert statistics["aux_scale"] == controls.auxiliary_scale


def test_history_sorting_does_not_change_random_minibatch_membership() -> None:
    lengths = np.array([80, 1, 70, 2, 60, 3, 50, 4, 40, 5], dtype=np.int64)
    torch.manual_seed(101)
    expected = torch.randperm(len(lengths)).numpy()
    torch.manual_seed(101)

    batches = _shuffled_minibatches(lengths, minibatch_size=4)

    for start, batch in zip(range(0, len(lengths), 4), batches, strict=True):
        expected_members = expected[start : start + 4]
        assert set(batch) == set(expected_members)
        assert np.all(np.diff(lengths[batch]) >= 0)


def test_ppo_rollback_restores_the_rejected_epoch() -> None:
    torch.manual_seed(103)
    config = tiny_ppo(
        ppo_epochs=2,
        learning_rate=0.05,
        target_kl=1e-12,
        kl_control="rollback",
    )
    model = tiny_model()
    collector = RolloutCollector(config, torch.device("cpu"), seed=107)
    rollout = collector.collect(model, config.rollout_transitions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    controls = TrainingController(config).controls(self_play_level=0)

    statistics = ppo_update(
        model, optimizer, rollout, config, torch.device("cpu"), controls
    )

    assert statistics["updates"] == 0.0
    assert statistics["epochs"] == 0.0
    assert statistics["rolled_back_epochs"] == 1.0
    assert statistics["approx_kl"] == statistics["max_attempted_kl"]
    assert statistics["max_attempted_kl"] > config.target_kl
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_checkpoint_restores_model_optimizer_and_opponent_snapshots(tmp_path) -> None:
    config = tiny_ppo(
        self_play_enabled=True,
        score_reward_weight=0.75,
        rank_reward_weight=1.25,
        kl_control="off",
        learning_rate_patience_evaluations=1,
    )
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(config, torch.device("cpu"), seed=37)
    controller = TrainingController(config, evaluation_seed=500)
    controller.take_evaluation_seed(8)
    controller.observe_update({"entropy": 0.05})
    controller.observe_rule_ev(
        panel_evaluation(),
        self_play_level=0,
        maximum_self_play_level=collector.pool.maximum_self_play_level,
    )
    controller.observe_rule_ev(
        panel_evaluation(),
        self_play_level=0,
        maximum_self_play_level=collector.pool.maximum_self_play_level,
    )
    collector.pool.promote(model, torch.device("cpu"), update=5)
    collector.pool.refresh_snapshot(model, torch.device("cpu"), update=6)
    collector.next_seed = 9_999
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        7,
        1234,
        321.5,
        config,
        collector,
        controller,
    )
    restored_model_config, restored_ppo_config = checkpoint_configs(path)

    assert restored_model_config == model.config
    assert restored_ppo_config == config

    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=config.learning_rate
    )
    restored_collector = RolloutCollector(config, torch.device("cpu"), seed=41)
    restored_controller = TrainingController(config, evaluation_seed=999)
    update, transitions, elapsed = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
        config,
        restored_collector,
        restored_controller,
    )

    assert (update, transitions, elapsed) == (7, 1234, 321.5)
    assert restored_collector.next_seed == 9_999
    assert len(restored_collector.pool.snapshots) == 2
    assert restored_collector.pool.frozen_ready
    assert restored_collector.pool.self_play_level == 1
    assert restored_collector.pool.last_snapshot_update == 6
    assert restored_controller.state_dict() == controller.state_dict()
    assert all(
        group["lr"] == controller.current_learning_rate
        for group in restored_optimizer.param_groups
    )
    assert all(
        not parameter.requires_grad
        for snapshot in restored_collector.pool.snapshots
        for parameter in snapshot.parameters()
    )
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_can_restore_into_a_forked_opponent_curriculum(tmp_path) -> None:
    source = tiny_ppo(self_play_enabled=False)
    target = replace(
        source,
        self_play_enabled=True,
        self_play_max_fraction=0.30,
        self_play_fraction_step=0.15,
    )
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=source.learning_rate)
    collector = RolloutCollector(source, torch.device("cpu"), seed=43)
    path = tmp_path / "source.pt"
    save_checkpoint(path, model, optimizer, 9, 4321, 654.0, source, collector)

    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=target.learning_rate
    )
    restored_collector = RolloutCollector(target, torch.device("cpu"), seed=47)
    state = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
        target,
        restored_collector,
        expected_checkpoint_config=source,
    )

    assert state == (9, 4321, 654.0)
    assert restored_collector.config == target
    assert restored_collector.pool.config == target


def test_legacy_checkpoint_keeps_score_only_reward_semantics(tmp_path) -> None:
    config = tiny_ppo()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(config, torch.device("cpu"), seed=109)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, optimizer, 1, 16, 1.0, config, collector)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("score_reward_weight", "rank_reward_weight", "kl_control"):
        checkpoint["ppo_config"].pop(name)
    torch.save(checkpoint, path)

    with pytest.warns(UserWarning, match="migrated a legacy PPO configuration"):
        _, restored = checkpoint_configs(path)

    assert restored.score_reward_weight == 1.0
    assert restored.rank_reward_weight == 0.0
    assert restored.kl_control == "monitor"


def test_checkpoint_migrates_wall_clock_and_fixed_self_play_configuration(
    tmp_path,
) -> None:
    config = tiny_ppo()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(config, torch.device("cpu"), seed=113)
    path = tmp_path / "staged-self-play.pt"
    save_checkpoint(path, model, optimizer, 1, 16, 1.0, config, collector)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["ppo_config"]
    state["final_learning_rate"] = state.pop("minimum_learning_rate")
    state["final_entropy_coefficient"] = state.pop(
        "minimum_entropy_coefficient"
    )
    state["self_play_fraction"] = 0.25
    state.pop("self_play_max_fraction")
    state.pop("self_play_fraction_step")
    state["self_play_gate_consecutive_evals"] = 3
    state.pop("self_play_gate_window")
    state.pop("self_play_gate_required_passes")
    state.update(
        {
            "schedule_hours": 24.0,
            "auxiliary_decay_fraction": 0.1,
            "opponent_refresh_updates": 200,
        }
    )
    torch.save(checkpoint, path)

    with pytest.warns(UserWarning, match="metric-driven scheduling"):
        _, restored = checkpoint_configs(path)

    assert restored.minimum_learning_rate == config.minimum_learning_rate
    assert restored.minimum_entropy_coefficient == config.minimum_entropy_coefficient
    assert restored.self_play_max_fraction == 0.25
    assert restored.self_play_fraction_step == 0.15
    assert restored.self_play_gate_window == 3
    assert restored.self_play_gate_required_passes == 2
    assert restored.historical_snapshot_probability == 0.5


def test_controller_reduces_learning_rate_only_after_metric_plateau() -> None:
    config = tiny_ppo(
        self_play_enabled=False,
        learning_rate_patience_evaluations=2,
        learning_rate_decay=0.5,
    )
    controller = TrainingController(config)
    evaluation = panel_evaluation()

    controller.observe_rule_ev(
        evaluation,
        self_play_level=0,
        maximum_self_play_level=3,
    )
    controller.observe_rule_ev(
        evaluation,
        self_play_level=0,
        maximum_self_play_level=3,
    )
    assert controller.current_learning_rate == config.learning_rate

    controller.observe_rule_ev(
        evaluation,
        self_play_level=0,
        maximum_self_play_level=3,
    )
    assert controller.current_learning_rate == config.learning_rate * 0.5


def test_controller_can_restart_learning_rate_schedule() -> None:
    config = tiny_ppo(learning_rate=5e-5, minimum_learning_rate=3e-5)
    controller = TrainingController(config)
    controller.current_learning_rate = config.minimum_learning_rate
    controller.learning_rate_plateau = 4
    controller.best_rank = 2.4
    controller.best_score_delta = 100.0

    controller.reset_learning_rate_schedule()

    assert controller.current_learning_rate == config.learning_rate
    assert controller.learning_rate_plateau == 0
    assert controller.best_rank is None
    assert controller.best_score_delta is None


def test_fixed_rule_ev_evaluation_is_independent_of_chunk_size() -> None:
    model = tiny_model()
    one_at_a_time = evaluate_against_rule_ev(
        model, torch.device("cpu"), games=4, envs=1, seed=43
    )
    together = evaluate_against_rule_ev(
        model, torch.device("cpu"), games=4, envs=4, seed=43
    )

    assert one_at_a_time == together
