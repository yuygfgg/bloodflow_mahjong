from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import (
    _shuffled_minibatches,
    EngineBuffers,
    HistoryCacheStore,
    OpponentPool,
    PPOConfig,
    RolloutCollector,
    TransitionStorage,
    checkpoint_configs,
    evaluate_against_rule_ev,
    cosine_learning_rate,
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
        opponent_refresh_updates=1,
        frozen_snapshot_limit=2,
    )
    return replace(config, **changes)


def test_rule_opponents_are_assigned_per_seat_and_self_play_is_explicit() -> None:
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

    config = tiny_ppo(self_play_enabled=True, self_play_fraction=0.25)
    pool = OpponentPool(config, seed=19)
    pool.set_frozen(tiny_model())
    for _ in range(config.self_play_gate_consecutive_evals - 1):
        pool.update_rule_evaluation({"first_rate": 0.24, "mean_score_delta": 0.0})
    assert pool.stage() == "bootstrap"
    pool.update_rule_evaluation({"first_rate": 0.25, "mean_score_delta": 0.0})
    assert pool.stage() == "self_play"
    np.testing.assert_allclose(
        pool.probabilities(), [0.25, 0.50, 0.25]
    )
    self_play = pool.assign_seats(learner_seats)
    assert OpponentPool.FROZEN_TRANSFORMER in self_play
    assert set(np.unique(self_play[self_play >= 0])) <= {
        OpponentPool.RULE_FAST,
        OpponentPool.RULE_EV,
        OpponentPool.FROZEN_TRANSFORMER,
    }


def test_snapshot_refresh_and_selection_have_separate_cadence() -> None:
    config = tiny_ppo(
        self_play_enabled=True,
        self_play_gate_consecutive_evals=1,
        historical_snapshot_probability=1.0,
        opponent_refresh_updates=10,
    )
    pool = OpponentPool(config, seed=19)
    pool.update_rule_evaluation({"first_rate": 0.25, "mean_score_delta": 0.0})

    pool.refresh_snapshot(tiny_model(), torch.device("cpu"), update=100)
    pool.refresh_snapshot(tiny_model(), torch.device("cpu"), update=110)

    assert not pool.snapshot_due(119)
    assert pool.snapshot_due(120)
    assert pool.select_snapshot() == 0


def test_snapshot_pool_keeps_the_fork_anchor() -> None:
    config = tiny_ppo(
        self_play_enabled=True,
        self_play_gate_consecutive_evals=1,
        frozen_snapshot_limit=2,
    )
    pool = OpponentPool(config, seed=23)
    pool.update_rule_evaluation({"first_rate": 0.25, "mean_score_delta": 0.0})
    model = tiny_model()
    anchor = next(model.parameters()).detach().clone()
    pool.refresh_snapshot(model, torch.device("cpu"), update=1)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    pool.refresh_snapshot(model, torch.device("cpu"), update=2)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    pool.refresh_snapshot(model, torch.device("cpu"), update=3)

    assert len(pool.snapshots) == 2
    torch.testing.assert_close(next(pool.snapshots[0].parameters()), anchor)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    before = model.actor[-1].weight.detach().clone()

    statistics = ppo_update(
        model, optimizer, rollout, config, torch.device("cpu"), progress=0.0
    )

    assert not torch.equal(before, model.actor[-1].weight.detach())
    assert all(np.isfinite(value) for value in statistics.values())
    assert statistics["updates"] == 4.0
    assert statistics["epochs"] == 2.0
    assert statistics["rolled_back_epochs"] == 0.0


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

    statistics = ppo_update(
        model, optimizer, rollout, config, torch.device("cpu"), progress=0.0
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
    )
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(config, torch.device("cpu"), seed=37)
    for _ in range(config.self_play_gate_consecutive_evals):
        collector.pool.update_rule_evaluation(
            {"first_rate": 0.24, "mean_score_delta": 0.0}
        )
    collector.pool.refresh_snapshot(model, torch.device("cpu"), update=5)
    collector.pool.refresh_snapshot(model, torch.device("cpu"), update=6)
    collector.next_seed = 9_999
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 7, 1234, 321.5, config, collector)
    restored_model_config, restored_ppo_config = checkpoint_configs(path)

    assert restored_model_config == model.config
    assert restored_ppo_config == config

    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=config.learning_rate
    )
    restored_collector = RolloutCollector(config, torch.device("cpu"), seed=41)
    update, transitions, elapsed = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
        config,
        restored_collector,
    )

    assert (update, transitions, elapsed) == (7, 1234, 321.5)
    assert restored_collector.next_seed == 9_999
    assert len(restored_collector.pool.snapshots) == 2
    assert restored_collector.pool.frozen_ready
    assert restored_collector.pool.last_snapshot_update == 6
    assert all(
        not parameter.requires_grad
        for snapshot in restored_collector.pool.snapshots
        for parameter in snapshot.parameters()
    )
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_can_restore_into_a_forked_opponent_curriculum(tmp_path) -> None:
    source = tiny_ppo(self_play_enabled=False, opponent_refresh_updates=10)
    target = replace(
        source,
        self_play_enabled=True,
        self_play_fraction=0.25,
        opponent_refresh_updates=200,
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

    with pytest.warns(UserWarning, match="legacy score-only"):
        _, restored = checkpoint_configs(path)

    assert restored.score_reward_weight == 1.0
    assert restored.rank_reward_weight == 0.0
    assert restored.kl_control == "monitor"


def test_checkpoint_migrates_the_staged_self_play_curriculum(tmp_path) -> None:
    config = tiny_ppo()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(config, torch.device("cpu"), seed=113)
    path = tmp_path / "staged-self-play.pt"
    save_checkpoint(path, model, optimizer, 1, 16, 1.0, config, collector)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for name in (
        "self_play_fraction",
        "self_play_gate_score_delta",
        "self_play_gate_consecutive_evals",
        "historical_snapshot_probability",
    ):
        checkpoint["ppo_config"].pop(name)
    checkpoint["ppo_config"].update(
        {
            "rule_mix_score_delta": -500.0,
            "rule_league_score_delta": 0.0,
            "rule_gate_consecutive_evals": 3,
        }
    )
    torch.save(checkpoint, path)

    _, restored = checkpoint_configs(path)

    assert restored.self_play_fraction == 0.25
    assert restored.self_play_gate_score_delta == 0.0
    assert restored.self_play_gate_consecutive_evals == 3
    assert restored.historical_snapshot_probability == 0.5


def test_cosine_learning_rate_hits_both_endpoints() -> None:
    config = tiny_ppo()
    assert cosine_learning_rate(config, 0.0) == config.learning_rate
    assert cosine_learning_rate(config, 1.0) == config.final_learning_rate
    assert (
        config.final_learning_rate
        < cosine_learning_rate(config, 0.5)
        < config.learning_rate
    )


def test_fixed_rule_ev_evaluation_is_independent_of_chunk_size() -> None:
    model = tiny_model()
    one_at_a_time = evaluate_against_rule_ev(
        model, torch.device("cpu"), games=4, envs=1, seed=43
    )
    together = evaluate_against_rule_ev(
        model, torch.device("cpu"), games=4, envs=4, seed=43
    )

    assert one_at_a_time == together
