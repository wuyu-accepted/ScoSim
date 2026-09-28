"""Deterministic ForumTwin product-quality and holdout acceptance gates.

The evaluators consume only aggregate review counts and synthetic lineage
operations.  They do not accept message text, source records, or microscopic
agent identities, and they never infer or manufacture a favorable result.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .mass_population import canonical_hash


HERO_MECHANISM_GATE_SCHEMA = "campus-pulse-forum-hero-mechanism-gate-v1"
HUMAN_REVIEW_GATE_SCHEMA = "campus-pulse-forum-human-review-gate-v1"
ABLATION_GATE_SCHEMA = "campus-pulse-forum-ablation-gate-v1"
ECOLOGY_HOLDOUT_GATE_SCHEMA = "campus-pulse-forum-ecology-holdout-gate-v1"
PRODUCT_ACCEPTANCE_SCHEMA = "campus-pulse-forum-product-acceptance-v1"

REQUIRED_ABLATIONS = (
    "profile_shuffled",
    "memory_disabled",
    "visible_message_text_removed",
    "bridge_disabled",
)
ABLATION_METRICS = (
    "persona_consistency",
    "direct_reply_continuity",
    "semantic_diversity",
)
MAIN_COMMENT_QUANTILES = ("p50", "p75", "p90", "p95", "p99")

LINEAGE_OPERATIONS = frozenset(
    {
        "claim_created",
        "misunderstanding",
        "propagation",
        "challenge",
        "evidence_entered",
        "correction_accept",
        "correction_reject",
        "help_request",
        "service_queue",
        "service_receipt",
        "feedback",
        "governance_message",
        "governance_reference",
        "governance_challenge",
        "governance_acknowledgement",
        "verified",
        "contested",
        "expired",
    }
)
TERMINAL_LINEAGE_OPERATIONS = frozenset(
    {"correction_accept", "correction_reject", "feedback", "verified", "contested", "expired"}
)

_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "agent_id",
        "user_id",
        "student_id",
        "source_id",
        "source_key",
        "post_code",
        "comment_id",
        "message_id",
        "author_id",
        "sample_identity",
        "raw_text",
        "visible_text",
        "text",
        "content",
        "message",
        "synthetic_message",
        "root_text",
        "comment_text",
        "prompt",
        "provider_body",
        "path",
        "profile_evidence",
    }
)


class ForumQualityError(ValueError):
    """Raised when quality evidence is unsafe, incomplete, or malformed."""


def _assert_aggregate_only(value: Any, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_INPUT_KEYS:
                raise ForumQualityError(f"{path}.{key} is prohibited microscopic or textual evidence")
            _assert_aggregate_only(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_aggregate_only(child, path=f"{path}[{index}]")


def _hash_release(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = canonical_hash(result)
    return result


def _bounded_rate(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForumQualityError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ForumQualityError(f"{field} must be finite and in [0, 1]")
    return number


def _count_metric(value: Any, field: str, *, violations: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ({"violations", "total"} if violations else {"passed", "total"}):
        key = "violations" if violations else "passed"
        raise ForumQualityError(f"{field} must contain exactly {key} and total")
    key = "violations" if violations else "passed"
    if (
        isinstance(value[key], bool)
        or isinstance(value["total"], bool)
        or not isinstance(value[key], int)
        or not isinstance(value["total"], int)
    ):
        raise ForumQualityError(f"{field} counts must be integers")
    numerator = value[key]
    total = value["total"]
    if total <= 0 or not 0 <= numerator <= total:
        raise ForumQualityError(f"{field} counts are invalid")
    rate = numerator / total
    return {key: numerator, "total": total, "rate": rate}


def _contains_subsequence(values: Sequence[str], expected: Sequence[str]) -> bool:
    position = 0
    for value in values:
        if position < len(expected) and value == expected[position]:
            position += 1
    return position == len(expected)


def evaluate_hero_mechanism_gate_v1(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate visible hero mechanisms from operation-only claim lineages."""

    _assert_aggregate_only(evidence)
    if set(evidence) != {"scenario_id", "branch_pair", "claim_lineages", "review_evidence_sha256"}:
        raise ForumQualityError("hero evidence fields are incomplete or unsupported")
    lineages = evidence.get("claim_lineages")
    if not isinstance(lineages, list):
        raise ForumQualityError("claim_lineages must be a list")
    review_sha = str(evidence.get("review_evidence_sha256") or "")
    if len(review_sha) != 64 or any(character not in "0123456789abcdef" for character in review_sha):
        raise ForumQualityError("review_evidence_sha256 must be a lowercase SHA-256")
    normalized = []
    complete_count = 0
    correction_chain = False
    help_chain = False
    governance_response = False
    for lineage_index, lineage in enumerate(lineages):
        if not isinstance(lineage, Mapping) or set(lineage) != {"lineage_fingerprint", "events"}:
            raise ForumQualityError("each claim lineage must contain fingerprint and events only")
        fingerprint = str(lineage.get("lineage_fingerprint") or "")
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ForumQualityError("lineage_fingerprint must be a lowercase SHA-256")
        events = lineage.get("events")
        if not isinstance(events, list) or not events:
            raise ForumQualityError("lineage events must be a nonempty list")
        operations = []
        prior_order = -1
        prior_tick = -1
        for event in events:
            if not isinstance(event, Mapping) or set(event) != {"order", "tick", "operation"}:
                raise ForumQualityError("lineage events must contain order, tick and operation only")
            if (
                isinstance(event["order"], bool)
                or isinstance(event["tick"], bool)
                or not isinstance(event["order"], int)
                or not isinstance(event["tick"], int)
            ):
                raise ForumQualityError("lineage order and tick must be integers")
            order = event["order"]
            tick = event["tick"]
            operation = str(event["operation"])
            if order <= prior_order or tick < prior_tick or tick < 0 or operation not in LINEAGE_OPERATIONS:
                raise ForumQualityError(f"lineage {lineage_index} has invalid order, tick, or operation")
            prior_order = order
            prior_tick = tick
            operations.append(operation)
        complete = (
            operations[0] == "claim_created"
            and len(operations) >= 3
            and any(operation in TERMINAL_LINEAGE_OPERATIONS for operation in operations[2:])
        )
        complete_count += complete
        correction_prefix = _contains_subsequence(
            operations,
            ("misunderstanding", "propagation", "challenge", "evidence_entered"),
        )
        if correction_prefix:
            correction_chain = correction_chain or any(
                "correction_accept" in operations[index + 1 :]
                and "correction_reject" in operations[index + 1 :]
                for index, operation in enumerate(operations)
                if operation == "evidence_entered"
            )
        help_chain = help_chain or _contains_subsequence(
            operations,
            ("help_request", "service_queue", "service_receipt", "feedback"),
        )
        if "governance_message" in operations:
            position = operations.index("governance_message")
            governance_response = governance_response or any(
                operation
                in {"governance_reference", "governance_challenge", "governance_acknowledgement"}
                for operation in operations[position + 1 :]
            )
        normalized.append(
            {
                "lineage_fingerprint": fingerprint,
                "event_count": len(operations),
                "complete": complete,
                "operation_counts": dict(
                    sorted({operation: operations.count(operation) for operation in set(operations)}.items())
                ),
            }
        )
    checks = {
        "at_least_three_complete_claim_lineages": complete_count >= 3,
        "misunderstanding_to_evidence_with_accept_and_reject": correction_chain,
        "help_queue_receipt_feedback_chain": help_chain,
        "governance_message_referenced_challenged_or_acknowledged": governance_response,
    }
    body = {
        "schema_version": HERO_MECHANISM_GATE_SCHEMA,
        "scenario_id": str(evidence["scenario_id"]),
        "branch_pair": str(evidence["branch_pair"]),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "complete_claim_lineage_count": complete_count,
        "lineages": normalized,
        "review_evidence_sha256": review_sha,
        "input_boundary": {
            "message_text_used": False,
            "microscopic_identity_used": False,
            "source_records_used": False,
        },
    }
    return _hash_release(body, "gate_sha256")


def evaluate_human_review_quality_gate_v1(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen blind-review and safety thresholds from aggregate counts."""

    _assert_aggregate_only(evidence)
    required = {
        "reviewer_count",
        "blind_scenario_identification",
        "natural_d_mechanism_agreement",
        "persona_consistency",
        "temporal_visibility",
        "direct_reply_continuity",
        "unsupported_real_fact",
        "templated_or_repeated_expression",
        "high_risk_gate",
        "review_evidence_sha256",
    }
    if set(evidence) != required:
        raise ForumQualityError("human-review evidence fields are incomplete or unsupported")
    if isinstance(evidence["reviewer_count"], bool) or not isinstance(evidence["reviewer_count"], int):
        raise ForumQualityError("reviewer_count must be an integer")
    reviewer_count = evidence["reviewer_count"]
    metrics = {
        "blind_scenario_identification": _count_metric(evidence["blind_scenario_identification"], "blind_scenario_identification"),
        "natural_d_mechanism_agreement": _count_metric(evidence["natural_d_mechanism_agreement"], "natural_d_mechanism_agreement"),
        "persona_consistency": _count_metric(evidence["persona_consistency"], "persona_consistency"),
        "temporal_visibility": _count_metric(evidence["temporal_visibility"], "temporal_visibility"),
        "direct_reply_continuity": _count_metric(evidence["direct_reply_continuity"], "direct_reply_continuity"),
        "unsupported_real_fact": _count_metric(evidence["unsupported_real_fact"], "unsupported_real_fact", violations=True),
        "templated_or_repeated_expression": _count_metric(evidence["templated_or_repeated_expression"], "templated_or_repeated_expression", violations=True),
        "high_risk_gate": _count_metric(evidence["high_risk_gate"], "high_risk_gate"),
    }
    thresholds = {
        "minimum_reviewer_count": 3,
        "blind_scenario_identification": 0.90,
        "natural_d_mechanism_agreement": 0.80,
        "persona_consistency": 0.90,
        "temporal_visibility": 1.00,
        "direct_reply_continuity": 0.85,
        "maximum_unsupported_real_fact": 0.02,
        "maximum_templated_or_repeated_expression": 0.05,
        "high_risk_gate": 1.00,
    }
    checks = {
        "reviewer_count": reviewer_count >= 3,
        "blind_scenario_identification": metrics["blind_scenario_identification"]["rate"] >= 0.90,
        "natural_d_mechanism_agreement": metrics["natural_d_mechanism_agreement"]["rate"] >= 0.80,
        "persona_consistency": metrics["persona_consistency"]["rate"] >= 0.90,
        "temporal_visibility": metrics["temporal_visibility"]["rate"] == 1.0,
        "direct_reply_continuity": metrics["direct_reply_continuity"]["rate"] >= 0.85,
        "unsupported_real_fact": metrics["unsupported_real_fact"]["rate"] <= 0.02,
        "templated_or_repeated_expression": metrics["templated_or_repeated_expression"]["rate"] <= 0.05,
        "high_risk_gate": metrics["high_risk_gate"]["rate"] == 1.0,
    }
    body = {
        "schema_version": HUMAN_REVIEW_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "reviewer_count": reviewer_count,
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "review_evidence_sha256": str(evidence["review_evidence_sha256"]),
        "aggregate_counts_only": True,
    }
    if len(body["review_evidence_sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in body["review_evidence_sha256"]
    ):
        raise ForumQualityError("review_evidence_sha256 must be a lowercase SHA-256")
    return _hash_release(body, "gate_sha256")


def evaluate_ablation_gate_v1(
    *,
    full_system: Mapping[str, Any],
    ablations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require full LLM system to beat each key ablation by 10% on 2/3 metrics."""

    _assert_aggregate_only(full_system)
    _assert_aggregate_only(ablations)
    if set(full_system) != set(ABLATION_METRICS):
        raise ForumQualityError("full-system ablation metrics are incomplete")
    if set(ablations) != set(REQUIRED_ABLATIONS):
        raise ForumQualityError("exactly four frozen ablations are required")
    full = {metric: _bounded_rate(full_system[metric], f"full_system.{metric}") for metric in ABLATION_METRICS}
    comparisons = {}
    for name in REQUIRED_ABLATIONS:
        values = ablations[name]
        if set(values) != set(ABLATION_METRICS):
            raise ForumQualityError(f"{name} metrics are incomplete")
        rows = {}
        passed_count = 0
        for metric in ABLATION_METRICS:
            baseline = _bounded_rate(values[metric], f"{name}.{metric}")
            improvement = (full[metric] - baseline) / max(baseline, 1e-12)
            passed = improvement >= 0.10
            passed_count += passed
            rows[metric] = {
                "full_system": full[metric],
                "ablation": baseline,
                "relative_improvement": improvement,
                "improves_by_at_least_10_percent": passed,
            }
        comparisons[name] = {
            "metrics": rows,
            "passing_metric_count": passed_count,
            "passed": passed_count >= 2,
        }
    body = {
        "schema_version": ABLATION_GATE_SCHEMA,
        "status": "pass" if all(item["passed"] for item in comparisons.values()) else "fail",
        "panel": {"agent_count": 200, "start_tick": 3, "end_tick_inclusive": 10},
        "threshold": {"minimum_relative_improvement": 0.10, "minimum_metrics_per_ablation": 2},
        "full_system": full,
        "comparisons": comparisons,
        "all_variants_require_real_llm_or_exact_reviewed_trace": True,
        "emulator_substitution_allowed": False,
    }
    return _hash_release(body, "gate_sha256")


def _distribution(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ForumQualityError(f"{field} must be a nonempty distribution")
    try:
        values = {str(key): float(raw) for key, raw in value.items()}
    except (TypeError, ValueError) as exc:
        raise ForumQualityError(f"{field} distribution is invalid") from exc
    if (
        any(not math.isfinite(number) or number < 0 for number in values.values())
        or sum(values.values()) <= 0
    ):
        raise ForumQualityError(f"{field} distribution is invalid")
    total = sum(values.values())
    return {key: number / total for key, number in sorted(values.items())}


def _js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    result = 0.0
    for values in (left, right):
        for key in keys:
            probability = values.get(key, 0.0)
            middle = (left.get(key, 0.0) + right.get(key, 0.0)) / 2
            if probability > 0:
                result += 0.5 * probability * math.log2(probability / middle)
    return result


def _entropy(distribution: Mapping[str, float]) -> float:
    return -sum(value * math.log2(value) for value in distribution.values() if value > 0)


def _relative_error(candidate: float, reference: float) -> tuple[float | None, bool]:
    if reference == 0:
        return (0.0, True) if candidate == 0 else (None, False)
    error = abs(candidate - reference) / abs(reference)
    return error, error <= 0.20


def evaluate_ecology_holdout_gate_v1(
    *,
    observed_holdout: Mapping[str, Any],
    simulated_forum: Mapping[str, Any],
    methodology: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare aggregate thread ecology inside the frozen stable collection regime."""

    _assert_aggregate_only(observed_holdout)
    _assert_aggregate_only(simulated_forum)
    _assert_aggregate_only(methodology)
    expected_aggregate_fields = {
        "thread_count",
        "comment_probability",
        "comment_quantiles",
        "reply_function_distribution",
        "topic_distribution",
        "disagreement_share",
        "correction_share",
    }
    if set(observed_holdout) != expected_aggregate_fields or set(simulated_forum) != expected_aggregate_fields:
        raise ForumQualityError("ecology aggregate fields are incomplete or unsupported")
    expected_methodology = {
        "train_start_inclusive",
        "train_end_exclusive",
        "holdout_start_inclusive",
        "holdout_end_exclusive",
        "known_gap_dates_excluded",
        "single_collection_regime",
    }
    if set(methodology) != expected_methodology:
        raise ForumQualityError("ecology methodology fields are incomplete")
    frozen_windows = {
        "train_start_inclusive": "2026-05-01",
        "train_end_exclusive": "2026-06-16",
        "holdout_start_inclusive": "2026-06-16",
        "holdout_end_exclusive": "2026-07-16",
    }
    methodology_checks = {
        **{key: str(methodology[key]) == value for key, value in frozen_windows.items()},
        "known_gap_dates_excluded": methodology["known_gap_dates_excluded"] is True,
        "single_collection_regime": methodology["single_collection_regime"] is True,
    }
    if (
        isinstance(observed_holdout["thread_count"], bool)
        or isinstance(simulated_forum["thread_count"], bool)
        or not isinstance(observed_holdout["thread_count"], int)
        or not isinstance(simulated_forum["thread_count"], int)
        or observed_holdout["thread_count"] <= 0
        or simulated_forum["thread_count"] <= 0
    ):
        raise ForumQualityError("ecology thread counts must be positive")
    observed_quantiles = observed_holdout["comment_quantiles"]
    simulated_quantiles = simulated_forum["comment_quantiles"]
    if not isinstance(observed_quantiles, Mapping) or not isinstance(simulated_quantiles, Mapping):
        raise ForumQualityError("comment_quantiles must be objects")
    if set(observed_quantiles) != set(MAIN_COMMENT_QUANTILES) or set(simulated_quantiles) != set(MAIN_COMMENT_QUANTILES):
        raise ForumQualityError("comment_quantiles must contain p50/p75/p90/p95/p99")
    quantile_results = {}
    quantile_checks = []
    for quantile in MAIN_COMMENT_QUANTILES:
        observed = float(observed_quantiles[quantile])
        simulated = float(simulated_quantiles[quantile])
        if not math.isfinite(observed) or not math.isfinite(simulated) or min(observed, simulated) < 0:
            raise ForumQualityError("comment quantiles must be finite and nonnegative")
        error, passed = _relative_error(simulated, observed)
        quantile_checks.append(passed)
        quantile_results[quantile] = {
            "observed": observed,
            "simulated": simulated,
            "relative_error": error,
            "passed": passed,
        }
    observed_reply = _distribution(observed_holdout["reply_function_distribution"], "observed reply functions")
    simulated_reply = _distribution(simulated_forum["reply_function_distribution"], "simulated reply functions")
    reply_js = _js_divergence(observed_reply, simulated_reply)
    observed_topics = _distribution(observed_holdout["topic_distribution"], "observed topics")
    simulated_topics = _distribution(simulated_forum["topic_distribution"], "simulated topics")
    observed_entropy = _entropy(observed_topics)
    simulated_entropy = _entropy(simulated_topics)
    if observed_entropy == 0:
        entropy_error = 0.0 if simulated_entropy == 0 else None
        entropy_passed = simulated_entropy == 0
    else:
        entropy_error = abs(simulated_entropy - observed_entropy) / observed_entropy
        entropy_passed = entropy_error <= 0.15
    comment_probability_error = abs(
        _bounded_rate(simulated_forum["comment_probability"], "simulated comment_probability")
        - _bounded_rate(observed_holdout["comment_probability"], "observed comment_probability")
    )
    disagreement_error = abs(
        _bounded_rate(simulated_forum["disagreement_share"], "simulated disagreement_share")
        - _bounded_rate(observed_holdout["disagreement_share"], "observed disagreement_share")
    )
    correction_error = abs(
        _bounded_rate(simulated_forum["correction_share"], "simulated correction_share")
        - _bounded_rate(observed_holdout["correction_share"], "observed correction_share")
    )
    checks = {
        "frozen_stable_holdout_methodology": all(methodology_checks.values()),
        "all_main_comment_quantiles_within_20_percent": all(quantile_checks),
        "reply_function_js_divergence_at_most_0_15": reply_js <= 0.15,
        "topic_entropy_relative_error_at_most_0_15": entropy_passed,
    }
    body = {
        "schema_version": ECOLOGY_HOLDOUT_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "methodology": dict(methodology),
        "methodology_checks": methodology_checks,
        "comment_quantiles": quantile_results,
        "reply_function_js_divergence": reply_js,
        "topic_entropy": {
            "observed": observed_entropy,
            "simulated": simulated_entropy,
            "relative_error": entropy_error,
        },
        "descriptive_not_release_gates": {
            "comment_probability_absolute_error": comment_probability_error,
            "disagreement_share_absolute_error": disagreement_error,
            "correction_share_absolute_error": correction_error,
        },
        "non_claims": {
            "reply_latency_validated": False,
            "real_user_network_validated": False,
            "causal_policy_effect_validated": False,
            "real_identity_or_longitudinal_user_used": False,
        },
        "aggregate_only": True,
    }
    return _hash_release(body, "gate_sha256")


def compose_forum_twin_product_acceptance_v1(
    *,
    hero_mechanism: Mapping[str, Any],
    human_review: Mapping[str, Any],
    ablation: Mapping[str, Any],
    ecology_holdout: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose precomputed immutable gates without changing any verdict."""

    expected = (
        (hero_mechanism, HERO_MECHANISM_GATE_SCHEMA),
        (human_review, HUMAN_REVIEW_GATE_SCHEMA),
        (ablation, ABLATION_GATE_SCHEMA),
        (ecology_holdout, ECOLOGY_HOLDOUT_GATE_SCHEMA),
    )
    for value, schema in expected:
        _assert_aggregate_only(value)
        if value.get("schema_version") != schema or value.get("status") not in {"pass", "fail"}:
            raise ForumQualityError("product acceptance received an incompatible gate")
        hash_field = "gate_sha256"
        if value.get(hash_field) != canonical_hash({key: item for key, item in value.items() if key != hash_field}):
            raise ForumQualityError("product acceptance gate hash mismatch")
    gates = {
        "hero_mechanism": hero_mechanism["status"],
        "human_review": human_review["status"],
        "ablation": ablation["status"],
        "ecology_holdout": ecology_holdout["status"],
    }
    body = {
        "schema_version": PRODUCT_ACCEPTANCE_SCHEMA,
        "status": "pass" if set(gates.values()) == {"pass"} else "fail",
        "gates": gates,
        "gate_sha256s": {
            "hero_mechanism": hero_mechanism["gate_sha256"],
            "human_review": human_review["gate_sha256"],
            "ablation": ablation["gate_sha256"],
            "ecology_holdout": ecology_holdout["gate_sha256"],
        },
        "verdict_composition_only": True,
        "provider_calls": 0,
        "result_manufacturing_allowed": False,
    }
    return _hash_release(body, "acceptance_sha256")
