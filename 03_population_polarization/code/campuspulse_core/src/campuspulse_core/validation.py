"""Temporal holdout validation for the observable layer of CampusPulse."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .calibration import CATEGORY_PATTERNS, DEFAULT_GAPS, _parse_date
from .config import NEED_TYPES, ScenarioConfig
from .generator import sample_natural_reply_count


NEED_TO_PROXY_CATEGORY = {
    "academic_information": "academic_and_exam",
    "emotional_support": "emotional_support",
    "safety_report": "safety_and_rights",
    "campus_service": "campus_services",
}

CDF_THRESHOLDS = (0, 1, 2, 4, 8, 16, 32)
GRADE_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "insufficient_data": 3}

CLAIM_EVIDENCE_MATRIX = [
    {
        "claim_id": "conditional_event_arrival_shape",
        "claim": "Conditional within-event normalized forum post arrival shape",
        "status": "partially_supported_not_scored",
        "reason": "Post timestamps are usable, but repeated-event shape has not yet been scored.",
        "next_evidence": "Backtest peak-aligned normalized proxy-share shapes across separated events.",
    },
    {
        "claim_id": "event_occurrence_and_absolute_volume",
        "claim": "Event occurrence and absolute forum post volume",
        "status": "not_supported",
        "reason": "Conditioning on observed peaks cannot test when an event will occur or its absolute workload.",
        "next_evidence": "Declare an operational time unit and prospectively test occurrence and count forecasts.",
    },
    {
        "claim_id": "thread_comment_probability",
        "claim": "Probability that a thread receives any comment",
        "status": "temporally_tested",
        "reason": "Final linked-comment presence is observable and is tested on a later holdout window.",
        "next_evidence": "Add human-reviewed need labels to replace overlapping keyword proxies.",
    },
    {
        "claim_id": "thread_comment_count_distribution",
        "claim": "Cumulative thread comment-count distribution",
        "status": "temporally_tested",
        "reason": "Final linked-comment counts are observable; binned CDF and positive-count mean are checked.",
        "next_evidence": "Add unique responder counts if the source system can expose them safely.",
    },
    {
        "claim_id": "urgency_and_risk",
        "claim": "Urgency mix and high-risk identification",
        "status": "not_supported",
        "reason": "No human-reviewed urgency or risk labels exist in the supplied data.",
        "next_evidence": "Create a double-annotated, adjudicated urgency sample.",
    },
    {
        "claim_id": "response_speed_and_diffusion",
        "claim": "Real response speed or information diffusion",
        "status": "not_supported",
        "reason": "Linked comment real_date inherits the parent post time and cannot measure response latency.",
        "next_evidence": "Collect genuine comment creation timestamps and reply-to structure.",
    },
    {
        "claim_id": "intervention_effects",
        "claim": "FAQ, peer, routing, and capacity intervention effects",
        "status": "not_supported",
        "reason": "The supplied forum data contain no intervention assignment or operational outcome logs.",
        "next_evidence": "Run a staged pilot with intervention logs and pre-registered operational outcomes.",
    },
    {
        "claim_id": "resolution_cost_and_capacity",
        "claim": "Institutional resolution, cost, and service capacity",
        "status": "not_supported",
        "reason": "No service ticket completion, staffing, or cost records are linked to forum posts.",
        "next_evidence": "Obtain de-identified service ticket and staffing aggregates.",
    },
    {
        "claim_id": "demographic_fairness",
        "claim": "Demographic fairness or population impact",
        "status": "not_supported",
        "reason": "No persistent identity or demographic attributes exist, by design.",
        "next_evidence": "Use consented aggregate survey or administrative statistics with privacy review.",
    },
]


def _file_identity(path: str | Path) -> dict[str, int | str]:
    """Identify an input snapshot without persisting its path or row contents."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": source.stat().st_size}


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _cdf(values: list[int]) -> dict[str, float]:
    total = len(values)
    if not total:
        return {str(threshold): 0.0 for threshold in CDF_THRESHOLDS}
    counts = Counter(values)
    return {
        str(threshold): sum(count for value, count in counts.items() if value <= threshold) / total
        for threshold in CDF_THRESHOLDS
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_counts(values: list[int]) -> dict:
    positives = [value for value in values if value > 0]
    return {
        "proxy_posts": len(values),
        "responded_posts": len(positives),
        "response_rate": len(positives) / len(values) if values else 0.0,
        "response_rate_wilson_95": _wilson_interval(len(positives), len(values)),
        "mean_comments_all": sum(values) / len(values) if values else 0.0,
        "mean_comments_if_responded": sum(positives) / len(positives) if positives else 0.0,
        "comment_quantiles": {
            "p50": _quantile(values, 0.50),
            "p75": _quantile(values, 0.75),
            "p90": _quantile(values, 0.90),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
        },
        "comment_count_cdf": _cdf(values),
    }


def compare_distributions(reference: dict, target: dict, minimum_posts: int) -> dict:
    if min(reference["proxy_posts"], target["proxy_posts"]) < minimum_posts:
        return {
            "grade": "insufficient_data",
            "response_rate_absolute_error_pp": None,
            "positive_mean_relative_error": None,
            "comment_cdf_mae": None,
        }
    response_error = abs(reference["response_rate"] - target["response_rate"])
    target_positive_mean = target["mean_comments_if_responded"]
    positive_mean_error = (
        abs(reference["mean_comments_if_responded"] - target_positive_mean)
        / target_positive_mean
        if target_positive_mean
        else 0.0
    )
    cdf_error = sum(
        abs(reference["comment_count_cdf"][str(threshold)] - target["comment_count_cdf"][str(threshold)])
        for threshold in CDF_THRESHOLDS
    ) / len(CDF_THRESHOLDS)
    if response_error <= 0.05 and positive_mean_error <= 0.20 and cdf_error <= 0.05:
        grade = "strong"
    elif response_error <= 0.10 and positive_mean_error <= 0.40 and cdf_error <= 0.10:
        grade = "moderate"
    else:
        grade = "weak"
    return {
        "grade": grade,
        "response_rate_absolute_error_pp": response_error * 100,
        "positive_mean_relative_error": positive_mean_error,
        "comment_cdf_mae": cdf_error,
        "rubric": {
            "strong": "<=5 pp response error, <=20% positive-mean error, and <=0.05 CDF MAE",
            "moderate": "<=10 pp response error, <=40% positive-mean error, and <=0.10 CDF MAE",
            "weak": "exceeds at least one moderate threshold",
        },
    }


def _read_temporal_counts(
    posts_path: str | Path,
    comments_path: str | Path,
    train_start: datetime,
    split_at: datetime,
    end_at: datetime,
    platform_marker: str,
) -> tuple[dict[str, dict[str, list[int]]], dict]:
    post_index: dict[str, tuple[str, tuple[str, ...]]] = {}
    excluded_gap_posts = 0
    with Path(posts_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("post_code", "")
            date = _parse_date(row.get("real_date", ""))
            if platform_marker not in code or date is None or not train_start <= date < end_at:
                continue
            if date.date().isoformat() in DEFAULT_GAPS:
                excluded_gap_posts += 1
                continue
            segment = "train" if date < split_at else "holdout"
            content = row.get("content", "")
            needs = tuple(
                need_type
                for need_type, category in NEED_TO_PROXY_CATEGORY.items()
                if CATEGORY_PATTERNS[category].search(content)
            )
            if needs:
                post_index[code] = (segment, needs)

    comments_per_post = Counter()
    with Path(comments_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("post_code", "")
            if code in post_index:
                comments_per_post[code] += 1

    values = {
        "train": {need_type: [] for need_type in sorted(NEED_TYPES)},
        "holdout": {need_type: [] for need_type in sorted(NEED_TYPES)},
    }
    matched_posts = Counter()
    linked_comments = Counter()
    for code, (segment, needs) in post_index.items():
        count = comments_per_post[code]
        for need_type in needs:
            values[segment][need_type].append(count)
            matched_posts[segment] += 1
            linked_comments[segment] += count
    audit = {
        "unique_matched_posts": len(post_index),
        "proxy_memberships": dict(matched_posts),
        "linked_comments_across_proxy_memberships": dict(linked_comments),
        "excluded_gap_posts": excluded_gap_posts,
    }
    return values, audit


def _model_counts(config: ScenarioConfig, need_type: str, samples: int, seed: int) -> list[int]:
    seed_material = f"{seed}|{config.fingerprint()}|{need_type}".encode("utf-8")
    stable_seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    rng = random.Random(stable_seed)
    probability = config.get("natural_response_probability")[need_type]
    mean = config.get("mean_replies_if_responded")[need_type]
    dispersion = config.get("reply_dispersion")
    return [
        sample_natural_reply_count(rng, probability, mean, dispersion)[1]
        for _ in range(samples)
    ]


def _mixture(components: dict[str, dict], weights: dict[str, float]) -> dict:
    response_rate = sum(weights[need] * components[need]["response_rate"] for need in weights)
    positive_numerator = sum(
        weights[need]
        * components[need]["response_rate"]
        * components[need]["mean_comments_if_responded"]
        for need in weights
    )
    cdf = {
        str(threshold): sum(
            weights[need] * components[need]["comment_count_cdf"][str(threshold)]
            for need in weights
        )
        for threshold in CDF_THRESHOLDS
    }
    return {
        "proxy_posts": sum(components[need]["proxy_posts"] for need in weights),
        "responded_posts": 0,
        "response_rate": response_rate,
        "response_rate_wilson_95": [0.0, 0.0],
        "mean_comments_all": positive_numerator,
        "mean_comments_if_responded": positive_numerator / response_rate if response_rate else 0.0,
        "comment_quantiles": {},
        "comment_count_cdf": cdf,
    }


def _worst_grade(grades: Iterable[str]) -> str:
    return max(grades, key=lambda grade: GRADE_ORDER[grade])


def evaluate_forum_data(
    posts_path: str | Path,
    comments_path: str | Path,
    configs: Iterable[ScenarioConfig],
    train_start: str = "2026-05-01",
    split: str = "2026-06-16",
    end: str = "2026-07-16",
    platform_marker: str = "NP",
    simulated_samples_per_need: int = 50000,
    seed: int = 20260721,
    minimum_posts: int = 100,
    evidence_start: str = "2025-12-01",
    evidence_end: str = "2026-07-21",
    event_window_radius_days: int = 3,
    max_focus_events: int = 5,
    minimum_event_posts: int = 40,
    minimum_event_separation_days: int = 14,
    minimum_peak_posts: int = 5,
    minimum_peak_lift: float = 1.25,
) -> dict:
    """Evaluate only observable forum behavior; never persist text or identifiers."""

    config_list = list(configs)
    if not config_list:
        raise ValueError("at least one scenario config is required")
    if simulated_samples_per_need <= 0:
        raise ValueError("simulated_samples_per_need must be positive")
    train_start_dt = datetime.fromisoformat(train_start)
    split_dt = datetime.fromisoformat(split)
    end_dt = datetime.fromisoformat(end)
    if not train_start_dt < split_dt < end_dt:
        raise ValueError("expected train_start < split < end")

    observed, audit = _read_temporal_counts(
        posts_path,
        comments_path,
        train_start_dt,
        split_dt,
        end_dt,
        platform_marker,
    )
    empirical = {
        segment: {
            need_type: summarize_counts(observed[segment][need_type])
            for need_type in sorted(NEED_TYPES)
        }
        for segment in ("train", "holdout")
    }

    model_cache: dict[tuple, dict[str, dict]] = {}
    need_results = {}
    scenario_results = {}
    for config in config_list:
        cache_key = (
            json.dumps(config.get("natural_response_probability"), sort_keys=True),
            json.dumps(config.get("mean_replies_if_responded"), sort_keys=True),
            float(config.get("reply_dispersion")),
        )
        if cache_key not in model_cache:
            model_cache[cache_key] = {
                need_type: summarize_counts(
                    _model_counts(config, need_type, simulated_samples_per_need, seed)
                )
                for need_type in sorted(NEED_TYPES)
            }
        model = model_cache[cache_key]
        per_need = {}
        for need_type in sorted(NEED_TYPES):
            per_need[need_type] = {
                "proxy_category": NEED_TO_PROXY_CATEGORY[need_type],
                "train": empirical["train"][need_type],
                "holdout": empirical["holdout"][need_type],
                "calibrated_model": model[need_type],
                "train_persistence_to_holdout": compare_distributions(
                    empirical["train"][need_type], empirical["holdout"][need_type], minimum_posts
                ),
                "calibrated_generator_to_holdout": compare_distributions(
                    model[need_type], empirical["holdout"][need_type], minimum_posts
                ),
            }
        need_results[config.scenario_id] = per_need

        weights = config.get("need_mix")
        train_mix = _mixture(empirical["train"], weights)
        holdout_mix = _mixture(empirical["holdout"], weights)
        model_mix = _mixture(model, weights)
        temporal_stability = compare_distributions(train_mix, holdout_mix, minimum_posts)
        generator_fidelity = compare_distributions(model_mix, holdout_mix, minimum_posts)
        scenario_results[config.scenario_id] = {
            "title": config.title,
            "scenario_version": config.version,
            "need_mix_is_operator_or_assumption_input": True,
            "temporal_stability": temporal_stability,
            "observable_generator_holdout_fidelity": generator_fidelity,
            "interpretation": (
                "Temporal stability compares earlier empirical behavior with the later window. "
                "Generator fidelity evaluates the train-window-calibrated hurdle/count mechanism "
                "against observations that were not used to set its response probability or mean."
            ),
        }

    # Imported lazily to avoid a module cycle: the focus evaluator reuses the
    # declared distribution summaries and grading rubric above.
    from .focus_evidence import build_focus_evidence

    focus_evidence = build_focus_evidence(
        posts_path,
        comments_path,
        audit_start=evidence_start,
        audit_end=evidence_end,
        event_start=evidence_start,
        event_end=end,
        platform_marker=platform_marker,
        event_window_radius_days=event_window_radius_days,
        max_events=max_focus_events,
        minimum_event_posts=minimum_event_posts,
        minimum_event_separation_days=minimum_event_separation_days,
        minimum_peak_posts=minimum_peak_posts,
        minimum_peak_lift=minimum_peak_lift,
    )

    temporal_grades = [
        result["temporal_stability"]["grade"]
        for result in scenario_results.values()
    ]
    generator_grades = [
        result["observable_generator_holdout_fidelity"]["grade"]
        for result in scenario_results.values()
    ]
    claim_evidence_matrix = [dict(item) for item in CLAIM_EVIDENCE_MATRIX]
    focus_backtests = focus_evidence["scenario_backtests"]
    repeated_windows_scored = all(
        focus_backtests[scenario_id]["overall_grade"] != "insufficient_data"
        for scenario_id in ("exam_pressure", "course_selection")
    )
    if repeated_windows_scored:
        arrival_shape_claim = next(
            item
            for item in claim_evidence_matrix
            if item["claim_id"] == "conditional_event_arrival_shape"
        )
        arrival_shape_claim.update(
            {
                "status": "repeated_window_tested",
                "reason": (
                    "Daily proxy-share arrival shape is tested with deterministic separated "
                    "events and leave-one-event-out folds for both focus scenarios."
                ),
                "next_evidence": (
                    "Validate keyword event definitions and map synthetic decision rounds to "
                    "an operational service cadence before forecasting absolute workload."
                ),
            }
        )
    supported_statuses = {"temporally_tested", "repeated_window_tested"}
    supported_claims = sum(
        item["status"] in supported_statuses for item in claim_evidence_matrix
    )
    result = {
        "validation_schema_version": "2.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "design": "chronological holdout; no random row split",
            "train_window": {"start_inclusive": train_start, "end_exclusive": split},
            "holdout_window": {"start_inclusive": split, "end_exclusive": end},
            "platform_marker": platform_marker,
            "excluded_known_gap_dates": sorted(DEFAULT_GAPS),
            "proxy_overlap": "Posts may belong to more than one keyword-proxy need type.",
            "simulated_samples_per_need": simulated_samples_per_need,
            "seed": seed,
            "minimum_posts_for_grade": minimum_posts,
            "data_quality_regime": (
                "Validation starts on 2026-05-01, after the 2026-04-03/04 crawl discontinuity, "
                "and ends before the documented 2026-07-16/17 synchronization interruption."
            ),
            "observable_parameter_calibration": (
                "Bundled scenario response probabilities and positive reply means use the train window only."
            ),
            "focus_evidence": focus_evidence["methodology"],
        },
        "privacy": "Aggregate output only; source text and direct identifiers are not persisted.",
        "input_evidence": {
            "posts_csv": _file_identity(posts_path),
            "comments_csv": _file_identity(comments_path),
            "note": "File-level fingerprints identify the evaluated snapshot; no source path is persisted.",
        },
        "data_audit": audit,
        "claim_evidence_matrix": claim_evidence_matrix,
        "empirical_claim_coverage": {
            "temporally_tested_claim_families": supported_claims,
            "total_claim_families": len(CLAIM_EVIDENCE_MATRIX),
            "warning": "Coverage is a scope count, not a model-accuracy score.",
        },
        "scenario_results": scenario_results,
        "need_type_details_by_scenario": need_results,
        "focus_evidence": focus_evidence,
        "overall_conclusion": {
            "temporal_stability": _worst_grade(temporal_grades),
            "observable_generator_holdout_fidelity": _worst_grade(generator_grades),
            "intervention_effect_validity": "not_empirically_validated",
            "focus_event_backtest": {
                scenario_id: focus_backtests[scenario_id]["overall_grade"]
                for scenario_id in ("exam_pressure", "course_selection")
            },
            "appropriate_use": "comparative operational stress testing and sensitivity analysis",
            "inappropriate_use": "predicting real policy effect sizes, response times, or student outcomes",
        },
        "limitations": [
            "Keyword proxy categories overlap and are not human-reviewed ground truth.",
            "The validation window is restricted to one stable post-discontinuity collection regime.",
            "Final comment counts measure cumulative attention, not timeliness, unique responders, or solution quality.",
            "Policy-effect outputs remain assumption-driven until intervention and service outcome logs exist.",
            "Event peaks are discovered from observed keyword-proxy activity; backtests are conditional on an event and do not forecast event occurrence or absolute workload.",
            "Annotation completeness does not establish label validity; sentiment, tags, analysis, and enriched text are not used for current scoring.",
        ],
    }
    return result


def write_validation_json(result: dict, output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return str(target)
