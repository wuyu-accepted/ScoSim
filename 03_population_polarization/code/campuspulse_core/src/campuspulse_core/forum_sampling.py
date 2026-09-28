"""ForumTwin PPS recovery, scheduler audit and idempotent writeback.

The estimators operate only on the 32 rotating PPS residents.  Longitudinal
anchors and interaction activations never enter the HT/GREG correction term.
Public speech is never synthesized here; missing residents remain missing
semantic observations and contribute only their frozen surrogate prediction.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .governance_arena_v2 import (
    ACTIVITY,
    ATTENTION,
    CONCERN,
    OPPOSITION,
    SATISFACTION,
    SUPPORT,
    TRUST,
    VOICE,
    BranchState,
)
from .mass_population import canonical_hash


class ForumSamplingError(ValueError):
    pass


PUBLIC_FORUM_ACTIONS = {
    "post", "reply", "quote", "repost", "verify", "correct",
    "help_request", "seek_help", "risk_report", "comment", "report_risk",
}


def audit_scheduler_recall(
    *,
    pps_observed_actions: Mapping[int, str],
    scheduled_interaction_indices: Sequence[int],
) -> dict[str, Any]:
    """Estimate scheduler recall without exposing PPS resident identities."""

    scheduled = set(scheduled_interaction_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in scheduled
    ):
        raise ForumSamplingError("scheduler interaction identity is invalid")
    should_act = {
        index
        for index, action in pps_observed_actions.items()
        if action in PUBLIC_FORUM_ACTIONS
    }
    recalled = should_act & scheduled
    false_negative_count = len(should_act - recalled)
    recall = 1.0 if not should_act else len(recalled) / len(should_act)
    return {
        "schema_version": "campus-pulse-forum-scheduler-recall-v1",
        "pps_observed_public_actions": len(should_act),
        "scheduler_recalled_public_actions": len(recalled),
        "scheduler_false_negative_count": false_negative_count,
        "public_action_recall": round(recall, 6),
        "gate_threshold": 0.90,
        "gate_passed": recall >= 0.90,
        "identities_public": False,
    }


def _validate_probabilities(
    predictions: Sequence[Mapping[str, float]], categories: Sequence[str]
) -> np.ndarray:
    category_set = set(categories)
    matrix = np.zeros((len(predictions), len(categories)), dtype=np.float64)
    for row_index, row in enumerate(predictions):
        if not isinstance(row, Mapping) or set(row) != category_set:
            raise ForumSamplingError("surrogate probability categories drifted")
        for column, category in enumerate(categories):
            value = row[category]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ForumSamplingError("surrogate probability is invalid")
            matrix[row_index, column] = float(value)
        total = float(matrix[row_index].sum())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ForumSamplingError("surrogate probabilities do not sum to one")
        matrix[row_index] /= total
    return matrix


def generalized_difference_categorical(
    *,
    design_weights: Sequence[float],
    surrogate_probabilities: Sequence[Mapping[str, float]],
    observed_categories: Mapping[int, str],
    inclusion_probabilities: Mapping[int, float],
    categories: Sequence[str],
    anchor_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Return GREG/HT shares, residual intervals and PPS Kish ESS."""

    weights = np.asarray(design_weights, dtype=np.float64)
    if (
        weights.ndim != 1
        or len(weights) != len(surrogate_probabilities)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
        or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9)
        or not categories
        or len(set(categories)) != len(categories)
    ):
        raise ForumSamplingError("population design weights are invalid")
    matrix = _validate_probabilities(surrogate_probabilities, categories)
    selected = set(observed_categories)
    if selected != set(inclusion_probabilities):
        raise ForumSamplingError("observations and inclusion probabilities differ")
    if selected & set(anchor_indices):
        raise ForumSamplingError("anchors cannot enter the population estimator")
    if not selected:
        raise ForumSamplingError("PPS observations cannot be empty")
    category_index = {category: index for index, category in enumerate(categories)}
    base = weights @ matrix
    correction = np.zeros(len(categories), dtype=np.float64)
    variance = np.zeros(len(categories), dtype=np.float64)
    analysis_weights = []
    for index in sorted(selected):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(weights)
            or observed_categories[index] not in category_index
        ):
            raise ForumSamplingError("PPS observation is invalid")
        pi = inclusion_probabilities[index]
        if (
            isinstance(pi, bool)
            or not isinstance(pi, (int, float))
            or not math.isfinite(float(pi))
            or not 0 < float(pi) <= 1
        ):
            raise ForumSamplingError("PPS inclusion probability must be positive")
        pi = float(pi)
        observed = np.zeros(len(categories), dtype=np.float64)
        observed[category_index[observed_categories[index]]] = 1.0
        residual = observed - matrix[index]
        factor = float(weights[index]) / pi
        correction += factor * residual
        variance += (1.0 - pi) * (factor * residual) ** 2
        analysis_weights.append(factor)
    raw = base + correction
    projected = np.maximum(raw, 0.0)
    if float(projected.sum()) <= 1e-12:
        projected = np.full(len(categories), 1.0 / len(categories))
    else:
        projected /= projected.sum()
    analysis = np.asarray(analysis_weights, dtype=np.float64)
    kish = float(analysis.sum() ** 2 / np.square(analysis).sum())
    intervals = {}
    for column, category in enumerate(categories):
        radius = 1.96 * math.sqrt(max(float(variance[column]), 0.0))
        intervals[category] = {
            "mean": round(float(projected[column]), 6),
            "lower": round(max(0.0, float(projected[column]) - radius), 6),
            "upper": round(min(1.0, float(projected[column]) + radius), 6),
        }
    return {
        "schema_version": "campus-pulse-forum-greg-categorical-v1",
        "budget": len(selected),
        "anchor_observations_excluded": True,
        "all_selected_have_positive_inclusion_probability": True,
        "surrogate_share": {
            category: round(float(base[column]), 6)
            for column, category in enumerate(categories)
        },
        "greg_share": {
            category: round(float(projected[column]), 6)
            for column, category in enumerate(categories)
        },
        "intervals_95": intervals,
        "kish_ess": round(kish, 6),
        "unselected_are_missing_not_zero": True,
        "sample_identities_public": False,
    }


def beta_dirichlet_forum_pooling(
    *,
    design_weights: Sequence[float],
    speaking_probabilities: Sequence[float],
    conditional_stance_probabilities: Sequence[Mapping[str, float]],
    observed_actions: Mapping[int, str],
    observed_stances: Mapping[int, str],
    inclusion_probabilities: Mapping[int, float],
    stance_categories: Sequence[str] = ("support", "oppose", "uncertain"),
    prior_strength: float = 8.0,
) -> dict[str, Any]:
    """Partially pool speaking and speaker-conditional stance from PPS turns.

    The posterior uses the full-population low-cost probabilities only as a
    finite prior mean.  The effective amount of LLM evidence is the Kish ESS
    of the design-adjusted PPS weights, so repeated large weights cannot be
    mistaken for independent semantic observations.
    """

    weights = np.asarray(design_weights, dtype=np.float64)
    speaking = np.asarray(speaking_probabilities, dtype=np.float64)
    if (
        weights.ndim != 1
        or speaking.shape != weights.shape
        or len(weights) != len(conditional_stance_probabilities)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
        or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9)
        or np.any(~np.isfinite(speaking))
        or np.any((speaking < 0) | (speaking > 1))
        or not math.isfinite(float(prior_strength))
        or float(prior_strength) <= 0
    ):
        raise ForumSamplingError("partial-pooling population inputs are invalid")
    categories = tuple(str(value) for value in stance_categories)
    if not categories or len(set(categories)) != len(categories):
        raise ForumSamplingError("partial-pooling stance categories are invalid")
    stance_matrix = _validate_probabilities(
        conditional_stance_probabilities, categories
    )
    selected = set(observed_actions)
    if selected != set(observed_stances) or selected != set(inclusion_probabilities):
        raise ForumSamplingError("partial-pooling PPS observations differ")
    if not selected:
        raise ForumSamplingError("partial-pooling PPS observations cannot be empty")

    analysis_weights: list[float] = []
    speaker_rows: list[tuple[int, float]] = []
    observed_public: list[float] = []
    for index in sorted(selected):
        if not 0 <= index < len(weights) or observed_stances[index] not in categories:
            raise ForumSamplingError("partial-pooling observation is invalid")
        pi = inclusion_probabilities[index]
        if (
            isinstance(pi, bool)
            or not isinstance(pi, (int, float))
            or not math.isfinite(float(pi))
            or not 0 < float(pi) <= 1
        ):
            raise ForumSamplingError("partial-pooling inclusion probability is invalid")
        factor = float(weights[index]) / float(pi)
        analysis_weights.append(factor)
        is_public = 1.0 if observed_actions[index] in PUBLIC_FORUM_ACTIONS else 0.0
        observed_public.append(is_public)
        if is_public:
            speaker_rows.append((index, factor))

    analysis = np.asarray(analysis_weights, dtype=np.float64)
    normalized = analysis / float(analysis.sum())
    public_vector = np.asarray(observed_public, dtype=np.float64)
    observed_speaking = float(normalized @ public_vector)
    kish = float(analysis.sum() ** 2 / np.square(analysis).sum())
    prior_speaking = float(weights @ speaking)
    alpha = prior_speaking * prior_strength + observed_speaking * kish
    beta = (1.0 - prior_speaking) * prior_strength + (1.0 - observed_speaking) * kish
    speaking_mean = alpha / (alpha + beta)
    speaking_variance = alpha * beta / (
        (alpha + beta) ** 2 * (alpha + beta + 1.0)
    )
    speaking_radius = 1.96 * math.sqrt(max(speaking_variance, 0.0))

    prior_stance = weights @ stance_matrix
    if float(prior_stance.sum()) <= 0:
        prior_stance = np.full(len(categories), 1.0 / len(categories))
    else:
        prior_stance /= float(prior_stance.sum())
    speaker_weights = np.asarray([row[1] for row in speaker_rows], dtype=np.float64)
    if len(speaker_rows):
        speaker_weights /= float(speaker_weights.sum())
        observed_counts = np.zeros(len(categories), dtype=np.float64)
        category_index = {category: index for index, category in enumerate(categories)}
        for normalized_weight, (index, _factor) in zip(speaker_weights, speaker_rows):
            observed_counts[category_index[observed_stances[index]]] += normalized_weight
        raw_speaker_weights = np.asarray(
            [row[1] for row in speaker_rows], dtype=np.float64
        )
        speaker_kish = float(
            raw_speaker_weights.sum() ** 2 / np.square(raw_speaker_weights).sum()
        )
    else:
        observed_counts = prior_stance.copy()
        speaker_kish = 0.0
    concentration = prior_stance * prior_strength + observed_counts * speaker_kish
    stance_mean = concentration / float(concentration.sum())
    stance_intervals = {}
    concentration_total = float(concentration.sum())
    for column, category in enumerate(categories):
        mean = float(stance_mean[column])
        variance = mean * (1.0 - mean) / (concentration_total + 1.0)
        radius = 1.96 * math.sqrt(max(variance, 0.0))
        stance_intervals[category] = {
            "mean": round(mean, 6),
            "lower": round(max(0.0, mean - radius), 6),
            "upper": round(min(1.0, mean + radius), 6),
        }
    return {
        "schema_version": "campus-pulse-forum-beta-dirichlet-pooling-v1",
        "pps_budget": len(selected),
        "kish_ess": round(kish, 6),
        "speaker_kish_ess": round(speaker_kish, 6),
        "speaking": {
            "mean": round(speaking_mean, 6),
            "lower": round(max(0.0, speaking_mean - speaking_radius), 6),
            "upper": round(min(1.0, speaking_mean + speaking_radius), 6),
        },
        "speaker_conditional_stance": stance_intervals,
        "unselected_are_missing_not_zero": True,
        "sample_identities_public": False,
    }


_WRITEBACK_FIELDS = {
    "observation_only": (),
    "stance_only": (SUPPORT, OPPOSITION),
    "full_state": (
        ATTENTION, CONCERN, TRUST, SATISFACTION, SUPPORT, OPPOSITION,
        ACTIVITY, VOICE,
    ),
}
_FIELD_INDEX = {
    "attention": ATTENTION,
    "concern": CONCERN,
    "trust": TRUST,
    "satisfaction": SATISFACTION,
    "support": SUPPORT,
    "opposition": OPPOSITION,
    "activity": ACTIVITY,
    "voice": VOICE,
}


def _bounded_mean_shift(
    values: np.ndarray, weights: np.ndarray, target: float
) -> np.ndarray:
    lower, upper = -1.0, 1.0
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        mean = float(np.sum(weights * np.clip(values + midpoint, 0.0, 1.0)))
        if mean < target:
            lower = midpoint
        else:
            upper = midpoint
    return np.clip(values + (lower + upper) / 2.0, 0.0, 1.0)


def apply_forum_posterior_writeback(
    *,
    state: BranchState,
    mode: str,
    posterior_targets: Mapping[str, float],
    scenario_id: str,
    branch: str,
    tick: int,
    innovation_id: str | None = None,
) -> dict[str, Any]:
    """Apply one bounded particle-level innovation exactly once."""

    if mode not in _WRITEBACK_FIELDS:
        raise ForumSamplingError("Forum writeback mode is invalid")
    allowed_indices = set(_WRITEBACK_FIELDS[mode])
    normalized_targets = {}
    for field, value in posterior_targets.items():
        if (
            field not in _FIELD_INDEX
            or _FIELD_INDEX[field] not in allowed_indices
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ForumSamplingError("Forum writeback target is invalid for its mode")
        normalized_targets[field] = float(value)
    if mode != "observation_only" and not normalized_targets:
        raise ForumSamplingError("Forum writeback has no eligible posterior targets")
    innovation = innovation_id or canonical_hash(
        [
            "forum-posterior-writeback-v1", scenario_id, branch, tick, mode,
            sorted(normalized_targets.items()),
        ]
    )
    if innovation in state.ensemble.innovations:
        return {
            "innovation_id": innovation,
            "mode": mode,
            "applied": False,
            "reason": "already_committed",
        }
    state.ensemble.innovations.add(innovation)
    before_weights = state.ensemble.weights.copy()
    achieved = {}
    for field, target in normalized_targets.items():
        index = _FIELD_INDEX[field]
        values = state.ensemble.values[:, :, index]
        shifted = _bounded_mean_shift(values, state.ensemble.weights, target)
        state.ensemble.values[:, :, index] = shifted
        achieved[field] = float(np.sum(state.ensemble.weights * shifted))
    if not np.array_equal(before_weights, state.ensemble.weights):
        raise ForumSamplingError("Forum writeback changed particle weights")
    state.ensemble.assert_conserved()
    return {
        "innovation_id": innovation,
        "mode": mode,
        "applied": True,
        "targets": {key: round(value, 6) for key, value in normalized_targets.items()},
        "achieved": {key: round(value, 6) for key, value in achieved.items()},
        "particle_weights_unchanged": True,
    }


def select_formal_writeback_mode(
    evidence: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the frozen stance/full-state promotion gates without tuning."""

    required_modes = {"observation_only", "stance_only", "full_state"}
    required_metrics = {
        "level_mae", "first_difference_mae", "seed_variance_ratio",
        "coverage_95", "descriptive_degradation",
    }
    if set(evidence) != required_modes or any(
        set(metrics) != required_metrics for metrics in evidence.values()
    ):
        raise ForumSamplingError("writeback gate evidence is incomplete")
    for metrics in evidence.values():
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in metrics.values()
        ):
            raise ForumSamplingError("writeback gate evidence is invalid")
    baseline = evidence["observation_only"]

    def qualifies(candidate_name: str, reference: Mapping[str, float]) -> bool:
        candidate = evidence[candidate_name]
        improvements = (
            float(reference["level_mae"]) - float(candidate["level_mae"]),
            float(reference["first_difference_mae"])
            - float(candidate["first_difference_mae"]),
        )
        return bool(
            candidate["level_mae"] <= reference["level_mae"] + 0.005
            and candidate["first_difference_mae"]
            <= reference["first_difference_mae"] + 0.005
            and max(improvements) >= 0.005 - 1e-12
            and candidate["seed_variance_ratio"] <= 1.10
            and 0.90 <= candidate["coverage_95"] <= 0.99
            and candidate["descriptive_degradation"] <= 0.02
        )

    stance_passed = qualifies("stance_only", baseline)
    full_passed = stance_passed and qualifies(
        "full_state", evidence["stance_only"]
    )
    selected = "full_state" if full_passed else "stance_only" if stance_passed else "observation_only"
    return {
        "schema_version": "campus-pulse-forum-writeback-gate-v1",
        "selected_mode": selected,
        "stance_only_passed": stance_passed,
        "full_state_passed": full_passed,
        "thresholds_frozen": True,
    }
