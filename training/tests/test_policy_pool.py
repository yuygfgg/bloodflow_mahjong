from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.policy_pool import (
    CATEGORY_COUNT,
    BehaviorSampler,
    BalancedReplaySampler,
    DecisionCategory,
    ExplorationConfig,
    ExplorationRule,
    OpponentMixtureConfig,
    PolicyPool,
    ReplayBalanceConfig,
    ReplayCoverageError,
    ReplaySource,
    decision_categories,
    replay_composition,
)


def all_replay_groups(repetitions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    categories: list[int] = []
    for source in ReplaySource:
        for category in range(CATEGORY_COUNT):
            sources.extend([int(source)] * repetitions)
            categories.extend([category] * repetitions)
    return np.asarray(sources, dtype=np.uint8), np.asarray(categories, dtype=np.uint8)


def test_policy_pool_starts_mixed_and_never_removes_anchors() -> None:
    pool = PolicyPool("sl.pt", seed=7)
    distribution = pool.distribution()
    by_source: dict[ReplaySource, float] = {}
    for policy, probability in distribution.items():
        by_source[policy.source] = by_source.get(policy.source, 0.0) + probability

    assert np.isclose(sum(distribution.values()), 1.0)
    assert by_source[ReplaySource.RULE_FAST] == pytest.approx(0.15)
    assert by_source[ReplaySource.RULE_SAFE] == pytest.approx(0.15)
    assert by_source[ReplaySource.SL] == pytest.approx(0.20)
    # Historical mass goes to current until the first snapshot exists.
    assert by_source[ReplaySource.CURRENT] == pytest.approx(0.50)

    pool.update_current(8, artifact="current-8.pt", update=50)
    pool.add_snapshot(update=50, artifact="history-8.pt")
    distribution = pool.distribution()
    by_source.clear()
    for policy, probability in distribution.items():
        by_source[policy.source] = by_source.get(policy.source, 0.0) + probability
    assert by_source == pytest.approx(
        {
            ReplaySource.RULE_FAST: 0.15,
            ReplaySource.RULE_SAFE: 0.15,
            ReplaySource.SL: 0.20,
            ReplaySource.CURRENT: 0.30,
            ReplaySource.FROZEN_POLICY: 0.20,
        }
    )


def test_policy_pool_stratifies_history_and_bounds_snapshots() -> None:
    config = OpponentMixtureConfig(
        recent_history_fraction=0.75,
        recent_history_count=2,
        max_history=4,
        snapshot_interval=10,
    )
    pool = PolicyPool("sl.pt", seed=11, config=config)
    assert not pool.snapshot_due(9)
    assert pool.snapshot_due(10)
    for update in (10, 20, 30, 40, 50):
        pool.update_current(update, update=update)
        assert pool.maybe_add_snapshot(
            update=update, artifact=f"snapshot-{update}.pt"
        ) is not None

    assert [item.version for item in pool.history] == [20, 30, 40, 50]
    history_distribution = {
        item.version: probability
        for item, probability in pool.distribution().items()
        if item.source == ReplaySource.FROZEN_POLICY
    }
    assert history_distribution[40] == pytest.approx(0.075)
    assert history_distribution[50] == pytest.approx(0.075)
    assert history_distribution[20] == pytest.approx(0.025)
    assert history_distribution[30] == pytest.approx(0.025)


def test_policy_pool_lineups_rotate_current_and_resume_exactly() -> None:
    pool = PolicyPool(Path("sl.pt"), seed=19)
    first = pool.sample_lineups(6)
    np.testing.assert_array_equal(first.focal_seats, [0, 1, 2, 3, 0, 1])
    rows = np.arange(6)
    assert np.all(first.sources[rows, first.focal_seats] == ReplaySource.CURRENT)

    state = json.loads(json.dumps(pool.state_dict()))
    expected = pool.sample_lineups(13)
    restored = PolicyPool.from_state_dict(state)
    actual = restored.sample_lineups(13)
    np.testing.assert_array_equal(actual.policy_ids, expected.policy_ids)
    np.testing.assert_array_equal(actual.sources, expected.sources)
    np.testing.assert_array_equal(actual.versions, expected.versions)
    np.testing.assert_array_equal(actual.focal_seats, expected.focal_seats)


def test_policy_pool_rejects_mixtures_without_permanent_anchors() -> None:
    with pytest.raises(ValueError, match="must remain present"):
        OpponentMixtureConfig(
            rule_fast_weight=0.0,
            current_weight=0.45,
        )


def test_decision_categories_cover_engine_phases_and_reject_terminal() -> None:
    meta = np.zeros((9, 34), dtype=np.int32)
    meta[:3, 0] = 0
    meta[:3, 10] = [0, 1, 2]
    meta[3, 0] = 1
    meta[4:7, 0] = 2
    meta[4:7, 4] = [45, 30, 10]
    meta[7, 0] = 3
    meta[8, 0] = 4
    np.testing.assert_array_equal(decision_categories(meta), np.arange(9))

    meta[0, 0] = 5
    with pytest.raises(ValueError, match="unclassifiable"):
        decision_categories(meta)


def test_behavior_sampler_is_legal_and_reports_exact_probability() -> None:
    logits = np.zeros((CATEGORY_COUNT, 115), dtype=np.float32)
    legal = np.zeros_like(logits, dtype=np.bool_)
    legal[:, [5, 12, 40, 57, 114]] = True
    logits[:, 5] = 5.0
    logits[:, 12] = 3.0
    categories = np.arange(CATEGORY_COUNT, dtype=np.uint8)
    sampler = BehaviorSampler(seed=23)
    sampled = sampler.sample(logits, legal, categories)

    assert np.all(legal[np.arange(CATEGORY_COUNT), sampled.actions])
    np.testing.assert_allclose(
        sampled.distributions.sum(axis=1), 1.0, atol=2e-7
    )
    np.testing.assert_allclose(
        sampled.action_probabilities,
        sampled.distributions[np.arange(CATEGORY_COUNT), sampled.actions],
    )
    assert np.all(sampled.distributions[~legal] == 0)
    # Even top-k-pruned legal actions keep a small, category-specific chance.
    assert np.all(sampled.distributions[:, 114] > 0)
    hu_random_mass = sampled.distributions[DecisionCategory.HU_RESPONSE, 114]
    exchange_random_mass = sampled.distributions[DecisionCategory.EXCHANGE_FIRST, 114]
    assert hu_random_mass < exchange_random_mass


def test_behavior_sampler_top_k_and_checkpoint_are_deterministic() -> None:
    rules = list(ExplorationConfig().rules)
    rules[DecisionCategory.TURN_MIDDLE] = ExplorationRule(
        temperature=1.0, top_k=2, random_action_probability=0.0
    )
    sampler = BehaviorSampler(
        seed=29, config=ExplorationConfig(rules=tuple(rules))
    )
    logits = np.arange(115, dtype=np.float64)[None, :].repeat(16, axis=0)
    legal = np.ones_like(logits, dtype=np.bool_)
    categories = np.full(16, DecisionCategory.TURN_MIDDLE, dtype=np.uint8)
    probabilities = sampler.probabilities(logits, legal, categories)
    assert np.all(probabilities[:, :-2] == 0)

    sampler.sample(logits, legal, categories)
    state = json.loads(json.dumps(sampler.state_dict()))
    expected = sampler.sample(logits, legal, categories)
    restored = BehaviorSampler.from_state_dict(state)
    actual = restored.sample(logits, legal, categories)
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_array_equal(
        actual.action_probabilities, expected.action_probabilities
    )


def test_behavior_sampler_rejects_illegal_or_nonfinite_rows() -> None:
    sampler = BehaviorSampler(seed=31)
    logits = np.zeros((1, 115), dtype=np.float64)
    legal = np.zeros((1, 115), dtype=np.bool_)
    with pytest.raises(ValueError, match="legal action"):
        sampler.probabilities(logits, legal, np.asarray([0]))

    legal[0, 3] = True
    logits[0, 3] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        sampler.probabilities(logits, legal, np.asarray([0]))


def test_balanced_replay_enforces_source_and_category_floors() -> None:
    sources, categories = all_replay_groups()
    sampler = BalancedReplaySampler(seed=37)
    selection = sampler.sample(
        sources,
        categories,
        512,
        duplicate_keys=np.arange(len(sources)) % 17,
        policy_versions=np.arange(len(sources)) % 5,
    )

    assert selection.indices.shape == (512,)
    assert np.all((selection.indices >= 0) & (selection.indices < len(sources)))
    for source, fraction in sampler.config.source_floors.items():
        assert selection.source_counts[source] >= np.ceil(fraction * 512)
    for category, fraction in enumerate(
        sampler.config.category_minimum_fractions
    ):
        assert selection.category_counts[category] >= np.ceil(fraction * 512)

    composition = replay_composition(
        sources[selection.indices], categories[selection.indices]
    )
    assert sum(composition["sources"].values()) == 512
    assert sum(composition["categories"].values()) == 512


def test_balanced_replay_max_flow_avoids_greedy_quota_trap() -> None:
    # Feasible in two draws only via (SL, category 1) and (RULE_FAST, category 0).
    source_floors = (
        (ReplaySource.SL, 0.5),
        (ReplaySource.RULE_FAST, 0.5),
    )
    category_floors = (0.5, 0.5) + (0.0,) * (CATEGORY_COUNT - 2)
    config = ReplayBalanceConfig(
        source_minimum_fractions=source_floors,
        category_minimum_fractions=category_floors,
        required_sources=(ReplaySource.SL, ReplaySource.RULE_FAST),
        require_all_categories=False,
    )
    sources = np.asarray(
        [ReplaySource.SL, ReplaySource.SL, ReplaySource.RULE_FAST], dtype=np.uint8
    )
    categories = np.asarray([0, 1, 0], dtype=np.uint8)
    result = BalancedReplaySampler(seed=41, config=config).sample(
        sources, categories, 2
    )
    assert result.source_counts[ReplaySource.SL] == 1
    assert result.source_counts[ReplaySource.RULE_FAST] == 1
    assert result.category_counts[0] == 1
    assert result.category_counts[1] == 1


def test_balanced_replay_reports_missing_coverage() -> None:
    sources = np.full(32, ReplaySource.CURRENT, dtype=np.uint8)
    categories = np.arange(32, dtype=np.uint8) % CATEGORY_COUNT
    sampler = BalancedReplaySampler(seed=43)
    with pytest.raises(ReplayCoverageError, match="missing required sources"):
        sampler.sample(sources, categories, 16)


def test_balanced_replay_checkpoint_resumes_exactly() -> None:
    sources, categories = all_replay_groups()
    sampler = BalancedReplaySampler(seed=47)
    index = sampler.prepare(sources, categories)
    sampler.sample_index(index, 128)
    state = json.loads(json.dumps(sampler.state_dict()))
    expected = sampler.sample_index(index, 128)
    restored = BalancedReplaySampler.from_state_dict(state)
    actual = restored.sample_index(index, 128)

    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.source_counts, expected.source_counts)
    np.testing.assert_array_equal(actual.category_counts, expected.category_counts)
    assert restored.batches_sampled == sampler.batches_sampled
    assert restored.rows_sampled == sampler.rows_sampled


def test_replay_sampling_index_owns_immutable_metadata() -> None:
    sources, categories = all_replay_groups()
    sampler = BalancedReplaySampler(seed=49)
    index = sampler.prepare(sources, categories)
    sources[:] = ReplaySource.MC_TEACHER
    categories[:] = DecisionCategory.MELD_RESPONSE

    assert not index.sources.flags.writeable
    assert not index.categories.flags.writeable
    assert set(np.unique(index.sources)) == set(range(len(ReplaySource)))
    assert set(np.unique(index.categories)) == set(range(CATEGORY_COUNT))


def test_state_versions_are_strict() -> None:
    pool_state = PolicyPool("sl.pt", seed=53).state_dict()
    pool_state["state_version"] = 99
    with pytest.raises(ValueError, match="unsupported policy pool"):
        PolicyPool.from_state_dict(pool_state)

    sampler_state = BalancedReplaySampler(seed=59).state_dict()
    sampler_state["state_version"] = 99
    with pytest.raises(ValueError, match="unsupported replay sampler"):
        BalancedReplaySampler.from_state_dict(sampler_state)


def test_config_rejects_incomplete_or_oversubscribed_floors() -> None:
    with pytest.raises(ValueError, match="sum above one"):
        ReplayBalanceConfig(
            category_minimum_fractions=(0.2,) * CATEGORY_COUNT
        )
    with pytest.raises(ValueError, match="cover every replay source"):
        ReplayBalanceConfig(
            source_target_weights=((ReplaySource.CURRENT, 1.0),)
        )

    config = ReplayBalanceConfig()
    restored = ReplayBalanceConfig.from_state_dict(config.state_dict())
    assert restored == config
