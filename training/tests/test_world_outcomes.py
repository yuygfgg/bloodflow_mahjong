from __future__ import annotations

import numpy as np
import pytest

from training.tests.test_policy_iteration import tiny_batch
from training.policy_iteration import PolicyStateBatch
from training.world_outcomes import (
    WorldOutcomeBatch,
    _action_set_signature,
    _normalize_action_sets,
    _require_action_set_match,
    combine_world_replicates,
    load_world_outcome_batch,
    save_world_outcome_batch,
)


def tiny_world_batch() -> WorldOutcomeBatch:
    base = tiny_batch()
    rank = np.zeros((len(base), base.legal.shape[1], 4), dtype=np.int8)
    score = np.zeros_like(rank, dtype=np.float32)
    rank[:, 0] = np.asarray([-3, -1, 1, 3], dtype=np.int8)
    rank[:, 1] = np.asarray([3, 1, -1, -3], dtype=np.int8)
    score[:, 0] = np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float32)
    score[:, 1] = -score[:, 0]
    state = {
        name: getattr(base, name)
        for name in PolicyStateBatch.__dataclass_fields__
    }
    return WorldOutcomeBatch(
        **state,
        rank_outcomes=rank,
        score_outcomes=score,
        behavior_actions=base.behavior_actions,
    )


def test_world_outcome_batch_converts_to_counterfactual_means() -> None:
    world = tiny_world_batch()
    counterfactual = world.counterfactual_batch()
    assert world.worlds == 4
    np.testing.assert_allclose(counterfactual.rank_q[:, :2], 0.0)
    np.testing.assert_allclose(counterfactual.score_q[:, :2], 0.0)


def test_world_outcome_cache_round_trip(tmp_path) -> None:
    expected = tiny_world_batch()
    path = tmp_path / "worlds.npz"
    save_world_outcome_batch(path, expected)
    actual = load_world_outcome_batch(path)
    for name in WorldOutcomeBatch.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_world_replicates_combine_on_world_axis() -> None:
    batch = tiny_world_batch()
    combined = combine_world_replicates((batch, batch))
    assert combined.worlds == 8
    np.testing.assert_array_equal(combined.rank_outcomes[:, :, :4], batch.rank_outcomes)


def test_sparse_world_outcomes_keep_only_requested_legal_actions() -> None:
    complete = tiny_world_batch()
    legal = complete.legal.copy()
    legal[:, 2] = True
    rank_complete = complete.rank_outcomes.copy()
    rank_complete[:, 2] = 1
    complete = WorldOutcomeBatch(
        **{
            **{
                name: getattr(complete, name)
                for name in PolicyStateBatch.__dataclass_fields__
                if name != "legal"
            },
            "legal": legal,
            "rank_outcomes": rank_complete,
            "score_outcomes": complete.score_outcomes,
            "behavior_actions": complete.behavior_actions,
        }
    )
    rank = np.zeros_like(complete.rank_outcomes)
    score = np.zeros_like(complete.score_outcomes)
    rank[:, :2] = complete.rank_outcomes[:, :2]
    score[:, :2] = complete.score_outcomes[:, :2]
    sparse = WorldOutcomeBatch(
        **{
            **{
                name: getattr(complete, name)
                for name in PolicyStateBatch.__dataclass_fields__
            },
            "rank_outcomes": rank,
            "score_outcomes": score,
            "behavior_actions": complete.behavior_actions,
        }
    )

    assert not sparse.is_complete
    assert np.all(sparse.evaluated_actions[:, :2])
    sparse.require_evaluated(np.ones(len(sparse), dtype=np.int64))
    with pytest.raises(ValueError, match="every legal action"):
        sparse.counterfactual_batch()
    with pytest.raises(ValueError, match="not evaluated"):
        sparse.require_evaluated(np.full(len(sparse), 2, dtype=np.int64))

    action_sets = tuple(np.asarray([0, 1], dtype=np.int64) for _ in range(len(sparse)))
    normalized = _normalize_action_sets(action_sets, len(sparse))
    assert normalized is not None
    _require_action_set_match(sparse, normalized)
    assert _action_set_signature(normalized) != "all_legal"


@pytest.mark.parametrize(
    "actions",
    (
        np.asarray([-1], dtype=np.int64),
        np.asarray([256], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([1.0], dtype=np.float64),
    ),
)
def test_action_sets_reject_values_that_uint8_would_hide(actions) -> None:
    with pytest.raises(ValueError, match="action"):
        _normalize_action_sets((actions,), 1)
