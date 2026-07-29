"""Production search-distillation targets for policy iteration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .policy_iteration import PolicyStateBatch
from .policy_pool import CATEGORY_NAMES
from .world_outcomes import WorldOutcomeBatch


@dataclass(frozen=True)
class PolicyImprovementTarget:
    """A policy target plus explicit evidence for every source state."""

    distribution: np.ndarray
    row_confidence: np.ndarray
    selected_actions: np.ndarray
    advantage: np.ndarray
    standard_error: np.ndarray
    p_value: np.ndarray
    lower_confidence_bound: np.ndarray

    def __post_init__(self) -> None:
        size, actions = self.distribution.shape
        vectors = (
            self.row_confidence,
            self.selected_actions,
            self.advantage,
            self.standard_error,
            self.p_value,
            self.lower_confidence_bound,
        )
        if size <= 0 or actions <= 0 or any(value.shape != (size,) for value in vectors):
            raise ValueError("policy improvement target arrays are misaligned")
        if (
            not np.isfinite(self.distribution).all()
            or np.any(self.distribution < 0)
            or not np.allclose(self.distribution.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("policy improvement distributions are invalid")
        if (
            not np.isfinite(self.row_confidence).all()
            or np.any((self.row_confidence < 0) | (self.row_confidence > 1))
        ):
            raise ValueError("policy improvement confidence is invalid")
        if np.any((self.selected_actions < 0) | (self.selected_actions >= actions)):
            raise ValueError("policy improvement actions are invalid")
        for value in (
            self.advantage,
            self.standard_error,
            self.p_value,
            self.lower_confidence_bound,
        ):
            if not np.isfinite(value).all():
                raise ValueError("policy improvement statistics are not finite")
        if np.any(self.standard_error < 0) or np.any(
            (self.p_value < 0) | (self.p_value > 1)
        ):
            raise ValueError("policy improvement uncertainty is invalid")

    @property
    def accepted(self) -> np.ndarray:
        return self.row_confidence > 0


def benjamini_hochberg(p_values: np.ndarray, *, fdr: float) -> np.ndarray:
    """Return hypotheses accepted by the Benjamini-Hochberg procedure."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if (
        p_values.ndim != 1
        or not len(p_values)
        or not np.isfinite(p_values).all()
        or np.any((p_values < 0) | (p_values > 1))
        or not math.isfinite(fdr)
        or not 0 < fdr < 1
    ):
        raise ValueError("BH-FDR arguments are invalid")
    order = np.argsort(p_values, kind="stable")
    thresholds = fdr * np.arange(1, len(p_values) + 1) / len(p_values)
    passing = np.flatnonzero(p_values[order] <= thresholds)
    accepted = np.zeros(len(p_values), dtype=np.bool_)
    if len(passing):
        accepted[order[: int(passing[-1]) + 1]] = True
    return accepted


def _one_sided_positive_p_value(
    mean: np.ndarray, standard_error: np.ndarray
) -> np.ndarray:
    result = np.ones(mean.shape, dtype=np.float64)
    deterministic = standard_error == 0
    result[deterministic & (mean > 0)] = 0.0
    stochastic = ~deterministic
    if np.any(stochastic):
        z = mean[stochastic] / standard_error[stochastic]
        result[stochastic] = np.asarray(
            [0.5 * math.erfc(float(value) / math.sqrt(2.0)) for value in z],
            dtype=np.float64,
        )
    return result


def _paired_rank_advantage(
    outcomes: WorldOutcomeBatch,
    actions: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outcomes.require_evaluated(actions)
    outcomes.require_evaluated(baseline)
    rows = np.arange(len(outcomes))
    utility = outcomes.rank_outcomes.astype(np.float64) / 2.0
    delta = utility[rows, actions] - utility[rows, baseline]
    mean = delta.mean(axis=1)
    standard_error = delta.std(axis=1, ddof=1) / math.sqrt(outcomes.worlds)
    return mean, standard_error


def select_rank_lcb_challengers(
    selection: WorldOutcomeBatch, reference_actions: np.ndarray
) -> np.ndarray:
    """Choose one non-reference action per state from a complete selection corpus."""

    if not selection.is_complete:
        raise ValueError("rank-LCB selection needs every legal action")
    size = len(selection)
    rows = np.arange(size)
    reference_actions = np.asarray(reference_actions, dtype=np.int64)
    if (
        reference_actions.shape != (size,)
        or np.any((reference_actions < 0) | (reference_actions >= selection.legal.shape[1]))
        or np.any(~selection.legal[rows, reference_actions])
    ):
        raise ValueError("reference actions do not match rank-LCB selection")
    alternatives = selection.legal.copy()
    alternatives[rows, reference_actions] = False
    if np.any(~alternatives.any(axis=1)):
        raise ValueError("rank-LCB selection needs a non-reference legal action")
    utility = selection.rank_outcomes.astype(np.float64).mean(axis=2) / 2.0
    baseline = utility[rows, reference_actions]
    advantage = utility - baseline[:, None]
    return np.where(alternatives, advantage, -np.inf).argmax(axis=1).astype(np.int64)


def build_rank_lcb_mirror_target(
    selection: WorldOutcomeBatch,
    validation: WorldOutcomeBatch,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    *,
    fdr: float = 0.05,
    temperature: float = 0.05,
    prior_floor: float = 1e-6,
    selected_actions: np.ndarray | None = None,
) -> tuple[PolicyImprovementTarget, dict[str, object]]:
    """Build a mean-rank mirror target from disjoint selection and validation worlds."""
    if (
        selection.worlds < 2
        or validation.worlds < 2
        or not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(prior_floor)
        or not 0 <= prior_floor < 1
    ):
        raise ValueError("rank-LCB mirror target arguments are invalid")
    for name in PolicyStateBatch.__dataclass_fields__:
        if not np.array_equal(getattr(selection, name), getattr(validation, name)):
            raise ValueError(f"rank-LCB states differ in {name}")
    if not np.array_equal(selection.behavior_actions, validation.behavior_actions):
        raise ValueError("rank-LCB behavior actions differ")
    legal = selection.legal
    size = len(selection)
    rows = np.arange(size)
    reference_actions = np.asarray(reference_actions, dtype=np.int64)
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64)
    if (
        reference_probabilities.shape != legal.shape
        or reference_actions.shape != (size,)
        or np.any(~legal[rows, reference_actions])
        or not np.isfinite(reference_probabilities).all()
        or np.any(reference_probabilities < 0)
        or np.any(reference_probabilities[~legal] != 0)
        or not np.allclose(reference_probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("reference policy does not match rank-LCB outcomes")

    if selected_actions is None:
        selected_actions = select_rank_lcb_challengers(selection, reference_actions)
    else:
        selected_actions = np.asarray(selected_actions, dtype=np.int64)
        if (
            selected_actions.shape != (size,)
            or np.any((selected_actions < 0) | (selected_actions >= legal.shape[1]))
            or np.any(~legal[rows, selected_actions])
            or np.any(selected_actions == reference_actions)
        ):
            raise ValueError("selected actions do not match rank-LCB outcomes")

    advantage, standard_error = _paired_rank_advantage(
        validation, selected_actions, reference_actions
    )
    p_value = _one_sided_positive_p_value(advantage, standard_error)
    accepted = benjamini_hochberg(p_value, fdr=fdr)
    z = NormalDist().inv_cdf(1.0 - fdr)
    lower = advantage - z * standard_error
    accepted &= lower > 0

    prior = np.where(legal, np.maximum(reference_probabilities, prior_floor), 0.0)
    prior /= prior.sum(axis=1, keepdims=True)
    logits = np.where(legal, np.log(np.maximum(prior, np.finfo(np.float64).tiny)), -np.inf)
    selected = np.flatnonzero(accepted)
    if len(selected):
        logits[selected, selected_actions[selected]] += lower[selected] / temperature
    maximum = np.max(logits, axis=1, keepdims=True)
    distribution = np.where(legal, np.exp(logits - maximum), 0.0)
    distribution /= distribution.sum(axis=1, keepdims=True)
    distribution[~accepted] = reference_probabilities[~accepted]
    row_confidence = accepted.astype(np.float32)
    target = PolicyImprovementTarget(
        distribution=distribution.astype(np.float32),
        row_confidence=row_confidence,
        selected_actions=selected_actions.astype(np.int64),
        advantage=advantage,
        standard_error=standard_error,
        p_value=p_value,
        lower_confidence_bound=lower,
    )
    metric_outcomes = validation if validation.is_complete else selection
    metrics = _target_metrics(
        target.distribution, metric_outcomes, reference_probabilities
    )
    metrics.update(
        {
            "kind": "rank_lcb_mirror",
            "selection_worlds": int(selection.worlds),
            "validation_worlds": int(validation.worlds),
            "fdr": float(fdr),
            "temperature": float(temperature),
            "prior_floor": float(prior_floor),
            "tested_states": int(size),
            "accepted_states": int(accepted.sum()),
            "accepted_state_rate": float(accepted.mean()),
            "mean_validation_advantage": float(advantage.mean()),
            "mean_accepted_advantage": (
                float(advantage[accepted].mean()) if np.any(accepted) else 0.0
            ),
            "mean_accepted_lcb": (
                float(lower[accepted].mean()) if np.any(accepted) else 0.0
            ),
            "target_q_metric_corpus": (
                "validation" if validation.is_complete else "selection"
            ),
        }
    )
    for category, name in enumerate(CATEGORY_NAMES):
        category_rows = selection.categories == category
        metrics["categories"][name].update(
            {
                "accepted_states": int(np.sum(accepted & category_rows)),
                "accepted_state_rate": float(accepted[category_rows].mean()),
            }
        )
    return target, metrics


def summarize_greedy_rank_audit(
    outcomes: Sequence[WorldOutcomeBatch],
    reference_actions: Sequence[np.ndarray],
    candidate_actions: Sequence[np.ndarray],
) -> dict[str, object]:
    """Summarize action-only improvement on worlds excluded from target construction."""
    if not outcomes or not (
        len(outcomes) == len(reference_actions) == len(candidate_actions)
    ):
        raise ValueError("audit batches are misaligned")
    row_advantages: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    flips: list[np.ndarray] = []
    for batch, reference, candidate in zip(
        outcomes, reference_actions, candidate_actions
    ):
        reference = np.asarray(reference, dtype=np.int64)
        candidate = np.asarray(candidate, dtype=np.int64)
        rows = np.arange(len(batch))
        if (
            reference.shape != (len(batch),)
            or candidate.shape != (len(batch),)
            or np.any(~batch.legal[rows, reference])
            or np.any(~batch.legal[rows, candidate])
        ):
            raise ValueError("audit policy actions are invalid")
        batch.require_evaluated(reference)
        batch.require_evaluated(candidate)
        utility = batch.rank_outcomes.astype(np.float64) / 2.0
        row_advantages.append(
            (utility[rows, candidate] - utility[rows, reference]).mean(axis=1)
        )
        categories.append(batch.categories)
        flips.append(candidate != reference)
    return summarize_greedy_rank_advantages(
        row_advantages, categories, flips
    )


def summarize_greedy_rank_advantages(
    row_advantages: Sequence[np.ndarray],
    categories: Sequence[np.ndarray],
    flips: Sequence[np.ndarray],
) -> dict[str, object]:
    """Summarize per-state audit evidence, including deterministic zero rows."""
    if not row_advantages or not (
        len(row_advantages) == len(categories) == len(flips)
    ):
        raise ValueError("audit advantage batches are misaligned")
    normalized_advantages: list[np.ndarray] = []
    normalized_categories: list[np.ndarray] = []
    normalized_flips: list[np.ndarray] = []
    for advantage, category, changed in zip(
        row_advantages, categories, flips
    ):
        advantage = np.asarray(advantage, dtype=np.float64)
        category = np.asarray(category)
        changed = np.asarray(changed)
        if (
            advantage.ndim != 1
            or category.shape != advantage.shape
            or changed.shape != advantage.shape
            or not np.issubdtype(category.dtype, np.integer)
            or changed.dtype != np.bool_
            or not np.isfinite(advantage).all()
            or np.any((category < 0) | (category >= len(CATEGORY_NAMES)))
            or np.any((~changed) & (advantage != 0.0))
        ):
            raise ValueError("audit advantage values are invalid")
        normalized_advantages.append(advantage)
        normalized_categories.append(category.astype(np.uint8, copy=False))
        normalized_flips.append(changed)

    advantage = np.concatenate(normalized_advantages)
    category_ids = np.concatenate(normalized_categories)
    changed = np.concatenate(normalized_flips)
    standard_error = (
        float(advantage.std(ddof=1) / math.sqrt(len(advantage)))
        if len(advantage) > 1
        else 0.0
    )
    mean = float(advantage.mean())
    radius = NormalDist().inv_cdf(0.975) * standard_error
    return {
        "states": int(len(advantage)),
        "mean_rank_utility_advantage": mean,
        "standard_error": standard_error,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "greedy_flip_rate": float(changed.mean()),
        "significant_harm": mean + radius < 0,
        "categories": {
            name: {
                "states": int(np.sum(category_ids == category)),
                "mean_rank_utility_advantage": float(
                    advantage[category_ids == category].mean()
                ),
                "greedy_flip_rate": float(changed[category_ids == category].mean()),
            }
            for category, name in enumerate(CATEGORY_NAMES)
        },
    }


def _assert_matching_replicates(
    combined: WorldOutcomeBatch,
    left: WorldOutcomeBatch,
    right: WorldOutcomeBatch,
) -> None:
    if left.worlds != right.worlds or combined.worlds != left.worlds + right.worlds:
        raise ValueError("split consensus needs two equal world replicates")
    for name in PolicyStateBatch.__dataclass_fields__:
        if not (
            np.array_equal(getattr(left, name), getattr(right, name))
            and np.array_equal(getattr(left, name), getattr(combined, name))
        ):
            raise ValueError(f"split consensus states differ in {name}")
    if not (
        np.array_equal(left.behavior_actions, right.behavior_actions)
        and np.array_equal(left.behavior_actions, combined.behavior_actions)
    ):
        raise ValueError("split consensus behavior actions differ")


def _paired_win_rates(outcomes: WorldOutcomeBatch, baseline: np.ndarray) -> np.ndarray:
    rows = np.arange(len(outcomes))
    rank = outcomes.rank_outcomes.astype(np.float32) / 2.0
    baseline_rank = rank[rows, baseline]
    score = outcomes.score_outcomes
    baseline_score = score[rows, baseline]
    better_rank = rank > baseline_rank[:, None, :]
    worse_rank = rank < baseline_rank[:, None, :]
    equal_rank = ~(better_rank | worse_rank)
    better_score = score > baseline_score[:, None, :]
    equal_score = score == baseline_score[:, None, :]
    wins = np.where(
        better_rank | (equal_rank & better_score),
        1.0,
        np.where(equal_rank & equal_score, 0.5, 0.0),
    )
    return wins.mean(axis=2)


def _target_metrics(
    target: np.ndarray,
    outcomes: WorldOutcomeBatch,
    reference_probabilities: np.ndarray,
) -> dict[str, object]:
    legal = outcomes.legal
    if (
        target.shape != legal.shape
        or reference_probabilities.shape != legal.shape
        or np.any(target[~legal] != 0)
        or not np.allclose(target.sum(axis=1), 1.0, atol=1e-6)
        or not np.isfinite(target).all()
        or np.any(target < 0)
    ):
        raise ValueError("search distillation target is invalid")
    reference_safe = np.clip(reference_probabilities, 1e-30, 1.0)
    target_safe = np.clip(target, 1e-30, 1.0)
    l1 = np.abs(target - reference_probabilities).sum(axis=1)
    changed = l1 > 1e-6
    entropy = -(target * np.log(target_safe)).sum(axis=1)
    reverse_kl = (
        target * (np.log(target_safe) - np.log(reference_safe))
    ).sum(axis=1)
    rank_q = outcomes.counterfactual_batch().rank_q
    q_gain = ((target - reference_probabilities) * rank_q).sum(axis=1)
    return {
        "states": len(outcomes),
        "changed_state_rate": float(changed.mean()),
        "mean_target_l1": float(l1.mean()),
        "mean_target_entropy": float(entropy.mean()),
        "mean_target_reverse_kl": float(reverse_kl.mean()),
        "mean_in_sample_rank_q_gain": float(q_gain.mean()),
        "categories": {
            name: {
                "states": int(np.sum(outcomes.categories == category)),
                "changed_state_rate": float(
                    changed[outcomes.categories == category].mean()
                ),
                "mean_target_l1": float(
                    l1[outcomes.categories == category].mean()
                ),
            }
            for category, name in enumerate(CATEGORY_NAMES)
        },
    }


def build_split_win_consensus_target(
    combined: WorldOutcomeBatch,
    left: WorldOutcomeBatch,
    right: WorldOutcomeBatch,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    *,
    margin: float = 0.125,
) -> tuple[np.ndarray, dict[str, object]]:
    """Use one-hot improvements only when both world halves beat the baseline.

    ``margin=0.125`` means that the selected alternative must have a paired
    lexicographic win rate strictly above 62.5% in each independent half.
    Unselected rows retain the frozen current-policy distribution, so they
    contribute no artificial entropy pressure to the CE direction.
    """
    if not math.isfinite(margin) or not 0.0 <= margin < 0.5:
        raise ValueError("split consensus margin must be in [0, 0.5)")
    _assert_matching_replicates(combined, left, right)
    if (
        reference_probabilities.shape != combined.legal.shape
        or reference_actions.shape != (len(combined),)
    ):
        raise ValueError("reference policy does not match split outcomes")
    rows = np.arange(len(combined))
    legal = combined.legal
    alternatives = legal.copy()
    alternatives[rows, reference_actions] = False
    left_win = _paired_win_rates(left, reference_actions)
    right_win = _paired_win_rates(right, reference_actions)
    pooled = 0.5 * (left_win + right_win)
    actions = np.where(alternatives, pooled, -np.inf).argmax(axis=1)
    threshold = 0.5 + margin
    changed = alternatives.any(axis=1)
    changed &= left_win[rows, actions] > threshold
    changed &= right_win[rows, actions] > threshold

    target = reference_probabilities.astype(np.float32, copy=True)
    selected = np.flatnonzero(changed)
    if len(selected):
        target[selected] = 0.0
        target[selected, actions[selected]] = 1.0
    target = np.where(legal, target, 0.0).astype(np.float32)
    target /= target.sum(axis=1, keepdims=True)
    metrics = _target_metrics(target, combined, reference_probabilities)
    metrics.update(
        {
            "kind": "split_win_both",
            "margin": float(margin),
            "win_rate_threshold": float(threshold),
            "worlds_per_replicate": int(left.worlds),
            "split_selected_states": int(changed.sum()),
        }
    )
    return target, metrics


def build_holdout_win_target(
    combined: WorldOutcomeBatch,
    selection: WorldOutcomeBatch,
    validation: WorldOutcomeBatch,
    reference_probabilities: np.ndarray,
    reference_actions: np.ndarray,
    *,
    margin: float = 0.1875,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select on one world replicate and validate on the untouched replicate.

    The validation half never participates in choosing the alternative action.
    With 32 binary worlds, ``margin=0.1875`` requires at least 23 wins and has
    a one-sided null pass probability of about one percent before ties.
    """
    if not math.isfinite(margin) or not 0.0 <= margin < 0.5:
        raise ValueError("holdout validation margin must be in [0, 0.5)")
    _assert_matching_replicates(combined, selection, validation)
    if (
        reference_probabilities.shape != combined.legal.shape
        or reference_actions.shape != (len(combined),)
    ):
        raise ValueError("reference policy does not match holdout outcomes")
    rows = np.arange(len(combined))
    alternatives = combined.legal.copy()
    alternatives[rows, reference_actions] = False
    selection_win = _paired_win_rates(selection, reference_actions)
    validation_win = _paired_win_rates(validation, reference_actions)
    actions = np.where(alternatives, selection_win, -np.inf).argmax(axis=1)
    threshold = 0.5 + margin
    changed = alternatives.any(axis=1)
    changed &= selection_win[rows, actions] > 0.5
    changed &= validation_win[rows, actions] > threshold

    target = reference_probabilities.astype(np.float32, copy=True)
    selected = np.flatnonzero(changed)
    if len(selected):
        target[selected] = 0.0
        target[selected, actions[selected]] = 1.0
    target = np.where(combined.legal, target, 0.0).astype(np.float32)
    target /= target.sum(axis=1, keepdims=True)
    metrics = _target_metrics(target, combined, reference_probabilities)
    metrics.update(
        {
            "kind": "holdout_win",
            "margin": float(margin),
            "selection_win_rate_threshold": 0.5,
            "validation_win_rate_threshold": float(threshold),
            "worlds_per_replicate": int(selection.worlds),
            "split_selected_states": int(changed.sum()),
        }
    )
    return target, metrics


__all__ = [
    "PolicyImprovementTarget",
    "benjamini_hochberg",
    "build_holdout_win_target",
    "build_rank_lcb_mirror_target",
    "build_split_win_consensus_target",
    "summarize_greedy_rank_audit",
    "summarize_greedy_rank_advantages",
]
