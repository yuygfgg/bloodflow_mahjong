from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import bloodflow_mahjong as bm

from training.model import BloodFlowTransformer, TransformerConfig
from training.pipeline import EngineBuffers
from training.policy_iteration import (
    CounterfactualBatch,
    PolicyQuery,
    PolicyStateBatch,
    _information_set_world_group,
    cached_counterfactual_corpus,
    calibrate_direction,
    cap_direction,
    committed_optimizer_state,
    category_row_weights,
    center_legal_values,
    direction_cosine,
    evaluate_direction_scale,
    load_counterfactual_batch,
    load_policy_state_batch,
    load_scaled_direction,
    nested_category_indices,
    one_step_direction,
    optimizer_direction,
    policy_cross_entropy_row_loss,
    policy_direction_row_loss,
    policy_improvement_target,
    policy_target_cross_entropy_row_loss,
    query_signature,
    require_deterministic_actor,
    save_counterfactual_batch,
    save_policy_state_batch,
    select_independent_queries,
    source_visit_frequencies,
)
from training.policy_pool import ReplaySource
from training.trajectory import CompactTrajectory


def fake_trajectory(seed: int, category: int) -> CompactTrajectory:
    return CompactTrajectory(
        seed=seed,
        exchange_direction=1 + seed % 3,
        actions=np.asarray([0], dtype=np.uint8),
        actors=np.asarray([0], dtype=np.uint8),
        phases=np.asarray([0], dtype=np.uint8),
        categories=np.asarray([category], dtype=np.uint8),
        sources=np.asarray([int(ReplaySource.CURRENT)], dtype=np.uint8),
        legal_counts=np.asarray([2], dtype=np.uint8),
        terminal_scores=np.asarray([10_000] * 4, dtype=np.int32),
        terminal_ranks=np.asarray([1, 2, 3, 4], dtype=np.uint8),
        termination_reason=int(bm.TERMINATION_WALL_EXHAUSTED),
    )


def tiny_batch() -> CounterfactualBatch:
    size = 9
    legal = np.zeros((size, 115), dtype=np.bool_)
    legal[:, :2] = True
    rank_q = np.zeros((size, 115), dtype=np.float32)
    rank_q[:, 0] = -0.2
    rank_q[:, 1] = 0.2
    score_q = rank_q * 0.5
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
        centered_rank_q=center_legal_values(rank_q, legal),
        behavior_actions=np.zeros(size, dtype=np.uint8),
    )


def test_ce_target_escapes_a_saturated_policy_gradient() -> None:
    legal = torch.tensor([[True, True, False]])
    q = torch.tensor([[-0.5, 0.5, 0.0]])
    pg_logits = torch.tensor([[20.0, -20.0, 0.0]], requires_grad=True)
    pg_loss, _expected_q, _entropy = policy_direction_row_loss(
        pg_logits, legal, q
    )
    pg_loss.sum().backward()

    ce_logits = torch.tensor([[20.0, -20.0, 0.0]], requires_grad=True)
    ce_loss, *_metrics = policy_cross_entropy_row_loss(
        ce_logits,
        legal,
        q,
        objective="hard_ce",
        temperature=0.1,
        prior_floor=0.0,
    )
    ce_loss.sum().backward()

    assert abs(float(pg_logits.grad[0, 1])) < 1e-12
    assert float(ce_logits.grad[0, 1]) == pytest.approx(-1.0)
    assert float(ce_logits.grad[0, 0]) == pytest.approx(1.0)
    assert float(ce_logits.grad[0, 2]) == pytest.approx(0.0)


def test_search_policy_targets_respect_legality_ties_and_prior() -> None:
    logits = torch.tensor([[2.0, 0.0, 7.0]])
    legal = torch.tensor([[True, True, False]])
    tied_q = torch.tensor([[0.25, 0.25, 0.0]])
    hard = policy_improvement_target(
        logits,
        legal,
        tied_q,
        objective="hard_ce",
        temperature=0.1,
        prior_floor=0.0,
    )
    np.testing.assert_allclose(hard.numpy(), [[0.5, 0.5, 0.0]])

    flat_q = torch.zeros_like(tied_q)
    mirror = policy_improvement_target(
        logits,
        legal,
        flat_q,
        objective="mirror_ce",
        temperature=0.1,
        prior_floor=0.0,
    )
    expected = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
    torch.testing.assert_close(mirror[0, :2], expected)
    assert float(mirror[0, 2]) == 0.0


def test_explicit_search_policy_target_drives_cross_entropy() -> None:
    logits = torch.tensor([[4.0, -4.0, 2.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    q = torch.tensor([[-0.25, 0.25, 0.0]])
    target = torch.tensor([[0.25, 0.75, 0.0]])
    loss, *_metrics = policy_target_cross_entropy_row_loss(
        logits, legal, q, target
    )
    loss.sum().backward()
    expected = torch.softmax(torch.tensor([4.0, -4.0]), dim=0) - target[0, :2]
    torch.testing.assert_close(logits.grad[0, :2], expected)
    assert float(logits.grad[0, 2]) == 0.0


def test_information_set_world_group_preserves_current_actor_view() -> None:
    source = bm.Batch(2, seed=31)
    # Use the same public buffer helpers as policy iteration.  The initial
    # exchange decisions already have hidden opponent hands to resample.
    source_engine = EngineBuffers.for_batch(source, history=8)
    source_engine.observe()
    seeds = np.asarray([[101, 102, 103], [201, 202, 203]], dtype=np.uint64)
    sampled = _information_set_world_group(source, seeds)
    sampled_engine = EngineBuffers.for_batch(sampled, history=8)
    sampled_engine.observe()
    repeated = np.repeat(np.arange(2), 3)
    np.testing.assert_array_equal(
        sampled_engine.tile_obs, source_engine.tile_obs[repeated]
    )
    np.testing.assert_array_equal(
        sampled_engine.melds, source_engine.melds[repeated]
    )
    np.testing.assert_array_equal(sampled_engine.meta, source_engine.meta[repeated])
    np.testing.assert_array_equal(sampled_engine.legal, source_engine.legal[repeated])


def tiny_actor() -> BloodFlowTransformer:
    torch.manual_seed(13)
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            max_history=8,
        )
    ).eval()


def test_policy_iteration_rejects_dropout_actor() -> None:
    actor = BloodFlowTransformer(
        TransformerConfig(
            d_model=16,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=32,
            dropout=0.1,
            max_history=8,
        )
    )
    with pytest.raises(ValueError, match="dropout=0"):
        require_deterministic_actor(actor)
    require_deterministic_actor(tiny_actor())


def test_independent_query_selection_and_nested_prefixes() -> None:
    trajectories = [
        fake_trajectory(category * 10 + copy, category)
        for category in range(9)
        for copy in range(2)
    ]
    queries = select_independent_queries(
        trajectories, queries_per_category=2, seed=17
    )
    assert len(queries) == 18
    assert len({query.trajectory_index for query in queries}) == 18
    np.testing.assert_array_equal(
        np.bincount([query.category for query in queries], minlength=9),
        np.full(9, 2),
    )
    categories = np.asarray([query.category for query in queries], dtype=np.uint8)
    nested = nested_category_indices(categories, (1, 2))
    assert len(nested[1]) == 9
    assert len(nested[2]) == 18
    assert set(nested[1]).issubset(set(nested[2]))

    stats = source_visit_frequencies(trajectories)
    np.testing.assert_allclose(stats["vector"], np.full(9, 1 / 9))


def test_query_selection_excludes_self_play_opponent_states() -> None:
    trajectories = []
    for category in range(9):
        trajectory = fake_trajectory(category + 100, category)
        trajectories.append(
            replace(
                trajectory,
                actions=np.asarray([0, 1], dtype=np.uint8),
                actors=np.asarray([0, 1], dtype=np.uint8),
                phases=np.asarray([0, 0], dtype=np.uint8),
                categories=np.asarray([category, category], dtype=np.uint8),
                sources=np.asarray(
                    [int(ReplaySource.CURRENT), int(ReplaySource.SELF_PLAY)],
                    dtype=np.uint8,
                ),
                legal_counts=np.asarray([2, 2], dtype=np.uint8),
            )
        )

    queries = select_independent_queries(
        trajectories, queries_per_category=1, seed=19
    )
    assert [query.step for query in queries] == [0] * 9
    stats = source_visit_frequencies(trajectories)
    assert stats["current_states"] == 9
    assert stats["eligible_multi_action_states"] == 9


def test_visitation_weights_are_unbiased_and_report_effective_mass() -> None:
    categories = np.repeat(np.arange(9, dtype=np.uint8), [1, 2, 3, 4, 5, 6, 7, 8, 9])
    category_weights = np.arange(1, 10, dtype=np.float64)
    category_weights /= category_weights.sum()
    weights = category_row_weights(categories, category_weights)
    assert weights.sum() == pytest.approx(1.0)
    for category in range(9):
        assert weights[categories == category].sum() == pytest.approx(
            category_weights[category]
        )


def test_counterfactual_cache_roundtrip_is_strict(tmp_path) -> None:
    batch = tiny_batch()
    path = tmp_path / "targets.npz"
    save_counterfactual_batch(path, batch)
    loaded = load_counterfactual_batch(path)
    for name in CounterfactualBatch.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(batch, name))

    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    payload["version"][:] = 99
    with path.open("wb") as stream:
        np.savez(stream, **payload)
    with pytest.raises(ValueError, match="version"):
        load_counterfactual_batch(path)

    state_path = tmp_path / "states.npz"
    save_policy_state_batch(state_path, batch)
    state = load_policy_state_batch(state_path)
    for name in state.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(state, name), getattr(batch, name))


def test_query_signature_covers_trajectory_contents() -> None:
    trajectory = fake_trajectory(7, 0)
    query = PolicyQuery(0, 0, trajectory, 0, 0)
    changed = replace(trajectory, actions=np.asarray([1], dtype=np.uint8))
    changed_query = replace(query, trajectory=changed)
    assert query_signature([query]) != query_signature([changed_query])


def test_target_cache_recovers_interrupted_manifest_write(
    tmp_path, monkeypatch
) -> None:
    import training.policy_iteration as policy_iteration

    batch = tiny_batch()
    queries = [
        PolicyQuery(index, index, fake_trajectory(index + 1, index), 0, index)
        for index in range(9)
    ]
    directory = tmp_path / "targets"
    directory.mkdir()
    (directory / "manifest.json.tmp").write_text("partial")

    def estimate(*_args, **_kwargs):
        return batch, {"rollout_states": 1, "rollout_seconds": 1.0}

    monkeypatch.setattr(policy_iteration, "estimate_counterfactual_batch", estimate)
    loaded, _metrics = cached_counterfactual_corpus(
        directory,
        queries,
        tiny_actor(),
        torch.device("cpu"),
        fingerprint="test",
        worlds=2,
        world_chunk=2,
        world_seed=3,
        shard_size=9,
    )
    np.testing.assert_array_equal(loaded.query_ids, batch.query_ids)
    assert (directory / "manifest.json").exists()
    assert not (directory / "manifest.json.tmp").exists()


def test_target_cache_manifest_identifies_batching_path(
    tmp_path, monkeypatch
) -> None:
    import training.policy_iteration as policy_iteration

    batch = tiny_batch()
    queries = [
        PolicyQuery(index, index, fake_trajectory(index + 1, index), 0, index)
        for index in range(9)
    ]
    directory = tmp_path / "targets"
    estimate_calls = 0

    def estimate(*_args, **_kwargs):
        nonlocal estimate_calls
        estimate_calls += 1
        return batch, {"rollout_states": 1, "rollout_seconds": 1.0}

    monkeypatch.setattr(policy_iteration, "estimate_counterfactual_batch", estimate)
    arguments = {
        "fingerprint": "test",
        "worlds": 2,
        "world_chunk": 2,
        "world_seed": 3,
        "shard_size": 9,
        "query_batch_size": 4,
        "inference_batch_size": 32,
    }
    cached_counterfactual_corpus(
        directory,
        queries,
        tiny_actor(),
        torch.device("cpu"),
        **arguments,
    )
    cached_counterfactual_corpus(
        directory,
        queries,
        tiny_actor(),
        torch.device("cpu"),
        **arguments,
    )
    assert estimate_calls == 1

    for changed in (
        {**arguments, "world_chunk": 1},
        {**arguments, "query_batch_size": 8},
        {**arguments, "inference_batch_size": 64},
    ):
        with pytest.raises(ValueError, match="manifest does not match"):
            cached_counterfactual_corpus(
                directory,
                queries,
                tiny_actor(),
                torch.device("cpu"),
                **changed,
            )
    assert estimate_calls == 1


def test_full_batch_microbatching_executes_exactly_one_adam_step(monkeypatch) -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    calls = 0
    original = torch.optim.AdamW.step

    def counted(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted)
    actor, initial, candidate, metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=1e-3,
        microbatch_size=4,
    )
    assert calls == 1
    assert metrics["optimizer_steps"] == 1
    assert metrics["microbatches"] == 3
    assert any(
        not torch.equal(initial[name], candidate[name])
        for name in initial
        if initial[name].is_floating_point()
    )

    endpoint = tiny_actor()
    load_scaled_direction(endpoint, initial, candidate, 1.0)
    for name, value in endpoint.state_dict().items():
        torch.testing.assert_close(value.cpu(), candidate[name])
    assert direction_cosine(initial, candidate, candidate)["cosine_to_maximum"] == pytest.approx(1.0)
    del actor


def test_full_batch_sgd_direction_executes_exactly_one_step(monkeypatch) -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    calls = 0
    original = torch.optim.SGD.step

    def counted(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", counted)
    _actor, initial, candidate, metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=0.1,
        microbatch_size=4,
        optimizer_name="sgd",
    )

    assert calls == 1
    assert metrics["optimizer"] == "sgd"
    assert metrics["optimizer_steps"] == 1
    assert any(
        not torch.equal(initial[name], candidate[name])
        for name in initial
        if initial[name].is_floating_point()
    )


def test_direction_optimizer_name_is_strict() -> None:
    with pytest.raises(ValueError, match="unsupported direction optimizer"):
        one_step_direction(
            tiny_actor(),
            tiny_batch(),
            torch.device("cpu"),
            category_weights=np.full(9, 1 / 9),
            learning_rate=0.1,
            microbatch_size=4,
            optimizer_name="sign-sgd",
        )


def test_direction_can_restrict_updates_to_the_policy_head() -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    head_names = tuple(
        name for name, _parameter in reference.named_parameters()
        if name.startswith("actor.")
    )

    _actor, initial, candidate, metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=0.1,
        microbatch_size=4,
        optimizer_name="sgd",
        trainable_parameter_names=head_names,
    )

    assert metrics["trainable_parameters"] < metrics["total_parameters"]
    assert any(not torch.equal(initial[name], candidate[name]) for name in head_names)
    assert all(
        torch.equal(initial[name], candidate[name])
        for name in initial
        if name not in set(head_names)
    )
    with pytest.raises(ValueError, match="trainable parameter names"):
        one_step_direction(
            reference,
            batch,
            torch.device("cpu"),
            category_weights=np.full(9, 1 / 9),
            learning_rate=0.1,
            microbatch_size=4,
            trainable_parameter_names=("missing",),
    )


def test_gradient_clip_can_be_disabled_without_hiding_the_raw_norm() -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    kwargs = {
        "category_weights": np.full(9, 1 / 9),
        "learning_rate": 0.01,
        "microbatch_size": 4,
        "optimizer_name": "sgd",
    }
    _clipped_actor, clipped_initial, clipped_candidate, clipped_metrics = (
        one_step_direction(
            reference,
            batch,
            torch.device("cpu"),
            gradient_clip_norm=0.05,
            **kwargs,
        )
    )
    _raw_actor, raw_initial, raw_candidate, raw_metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        gradient_clip_norm=None,
        **kwargs,
    )

    def displacement(initial, candidate) -> float:
        return float(
            sum(
                torch.sum((candidate[name].double() - value.double()).square())
                for name, value in initial.items()
            ).sqrt()
        )

    assert clipped_metrics["gradient_was_clipped"] is True
    assert raw_metrics["gradient_was_clipped"] is False
    assert raw_metrics["gradient_clip_norm"] is None
    assert raw_metrics["gradient_norm"] == pytest.approx(
        clipped_metrics["gradient_norm"]
    )
    assert displacement(raw_initial, raw_candidate) > displacement(
        clipped_initial, clipped_candidate
    )


def test_nesterov_cold_start_is_the_same_raw_direction_as_sgd() -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    kwargs = {
        "category_weights": np.full(9, 1 / 9),
        "learning_rate": 0.1,
        "microbatch_size": 4,
    }
    _sgd_actor, sgd_initial, sgd_candidate, _sgd_metrics = optimizer_direction(
        reference, batch, torch.device("cpu"), optimizer_name="sgd", **kwargs
    )
    _nag_actor, nag_initial, nag_candidate, nag_metrics = optimizer_direction(
        reference,
        batch,
        torch.device("cpu"),
        optimizer_name="nesterov",
        momentum=0.9,
        optimizer_state={},
        **kwargs,
    )
    for name in sgd_initial:
        torch.testing.assert_close(sgd_initial[name], nag_initial[name])
        torch.testing.assert_close(sgd_candidate[name], nag_candidate[name])
    assert nag_metrics["gradient_evaluation"] == "lookahead"
    assert nag_metrics["optimizer_state_parameters"] == 0


def test_search_ce_zero_confidence_has_exactly_zero_fresh_gradient() -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    target = np.zeros_like(batch.rank_q)
    target[:, 1] = 1.0
    actor, initial, candidate, metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=0.1,
        microbatch_size=4,
        optimizer_name="sgd",
        objective="search_ce",
        policy_targets=target,
        policy_row_confidence=np.zeros(len(batch), dtype=np.float32),
    )

    for name in initial:
        torch.testing.assert_close(candidate[name], initial[name])
    assert metrics["row_weight_sum"] == 0.0
    assert metrics["effective_sample_size"] == 0.0
    assert metrics["supervised_states"] == 0
    for expected, actual in zip(reference.parameters(), actor.parameters()):
        torch.testing.assert_close(expected, actual)


def test_stateful_optimizer_keeps_the_kl_committed_displacement() -> None:
    reference = tiny_actor()
    initial = {
        name: value.detach().cpu().clone()
        for name, value in reference.state_dict().items()
    }
    with torch.no_grad():
        next(reference.parameters()).add_(0.01)

    state = committed_optimizer_state("nesterov", initial, reference)

    assert set(state) == {name for name, _parameter in reference.named_parameters()}
    first_name = next(iter(state))
    expected = reference.state_dict()[first_name].cpu() - initial[first_name]
    torch.testing.assert_close(state[first_name], expected)
    with pytest.raises(ValueError, match="parameter names"):
        optimizer_direction(
            tiny_actor(),
            tiny_batch(),
            torch.device("cpu"),
            category_weights=np.full(9, 1 / 9),
            learning_rate=0.1,
            microbatch_size=4,
            optimizer_name="nesterov",
            optimizer_state={"missing": torch.ones(1)},
        )


def test_calibration_caches_reference_forwards_and_stops_within_tolerance(
    monkeypatch,
) -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    actor, initial, candidate, _metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=1e-3,
        microbatch_size=4,
    )
    states = PolicyStateBatch(
        **{
            name: getattr(batch, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
    )
    reference_forwards = 0
    original_forward = reference.forward

    def counted_forward(*args, **kwargs):
        nonlocal reference_forwards
        reference_forwards += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(reference, "forward", counted_forward)
    target = 1e-4
    metrics = calibrate_direction(
        actor,
        reference,
        initial,
        candidate,
        states,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        target_kl=target,
        batch_size=4,
        search_steps=18,
        maximum_scale=64,
    )
    assert reference_forwards == 3
    assert 0 <= metrics["final_kl"] <= target
    assert metrics["relative_shortfall"] <= 0.05
    assert metrics["evaluations"] < 15


def test_kl_cap_never_enlarges_a_raw_direction() -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    actor, initial, candidate, _metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=0.1,
        microbatch_size=4,
        optimizer_name="sgd",
    )
    states = PolicyStateBatch(
        **{
            name: getattr(batch, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
    )
    raw = evaluate_direction_scale(
        actor,
        reference,
        initial,
        candidate,
        states,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        scale=1.0,
        batch_size=4,
    )
    assert raw["kl"] > 0

    accepted = cap_direction(
        actor,
        reference,
        initial,
        candidate,
        states,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        kl_cap=2 * raw["kl"],
        batch_size=4,
        search_steps=12,
    )
    assert accepted["scale"] == pytest.approx(1.0)
    assert accepted["cap_activated"] is False
    expected = tiny_actor()
    expected.load_state_dict(candidate)
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected.state_dict()[name].cpu())

    capped = cap_direction(
        actor,
        reference,
        initial,
        candidate,
        states,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        kl_cap=raw["kl"] / 4,
        batch_size=4,
        search_steps=18,
    )
    assert capped["cap_activated"] is True
    assert 0 < capped["scale"] < 1
    assert capped["final_kl"] <= capped["kl_cap"]


def test_fixed_direction_scale_loads_the_requested_endpoint() -> None:
    reference = tiny_actor()
    batch = tiny_batch()
    actor, initial, candidate, _metrics = one_step_direction(
        reference,
        batch,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        learning_rate=1e-3,
        microbatch_size=4,
    )
    states = PolicyStateBatch(
        **{
            name: getattr(batch, name)
            for name in PolicyStateBatch.__dataclass_fields__
        }
    )

    metrics = evaluate_direction_scale(
        actor,
        reference,
        initial,
        candidate,
        states,
        torch.device("cpu"),
        category_weights=np.full(9, 1 / 9),
        scale=0.5,
        batch_size=4,
    )

    assert metrics["scale"] == pytest.approx(0.5)
    assert metrics["kl"] >= 0
    assert 0 <= metrics["greedy_flip_rate"] <= 1
    expected = tiny_actor()
    load_scaled_direction(expected, initial, candidate, 0.5)
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected.state_dict()[name].cpu())
