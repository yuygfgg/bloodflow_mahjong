from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import bloodflow_mahjong as bm

from training.model import BloodFlowTransformer, TransformerConfig
from training.policy_iteration import (
    CounterfactualBatch,
    PolicyQuery,
    PolicyStateBatch,
    cached_counterfactual_corpus,
    calibrate_direction,
    category_row_weights,
    center_legal_values,
    direction_cosine,
    load_counterfactual_batch,
    load_policy_state_batch,
    load_scaled_direction,
    nested_category_indices,
    one_step_direction,
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
