"""Aggregate-only field audit and repeated-event backtesting for CampusPulse.

The module reads the two authorized forum exports, but never returns or writes
source text, row identifiers, hashes, or source paths.  Event windows are
keyword proxies rather than human-reviewed labels, and their backtests cover
observable arrival shape and cumulative comment attention only.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median

from .calibration import DEFAULT_GAPS, _parse_date
from .validation import summarize_counts


FOCUS_DEFINITIONS = {
    "exam_pressure": {
        "title": "考试压力",
        "proxy_version": "1.0.0",
        "pattern": re.compile(
            r"考试|期末|期中|复习|备考|挂科|补考|缓考|四六级|六级|四级"
        ),
    },
    "course_selection": {
        "title": "选课治理",
        "proxy_version": "1.0.0",
        "pattern": re.compile(
            r"选课|抢课|退课|补选|改选|选不上|课程容量|课容量|选课系统|选课志愿|课程冲突"
        ),
    },
}

POST_FIELD_POLICIES = {
    "id": (
        "not_used",
        "Source row identifier; excluded from calibration and every persisted artifact.",
    ),
    "post_code": (
        "linkage_only",
        "Used in memory to link final comment counts, then discarded.",
    ),
    "real_date": (
        "direct_observable",
        "Used for post-arrival dates and event windows after crawl-gap exclusion.",
    ),
    "content": (
        "keyword_proxy_only",
        "Scanned in memory for declared scenario keywords; never persisted or treated as ground truth.",
    ),
    "sentiment": (
        "exploratory_not_scored",
        "Completeness can be audited, but the field is not human-reviewed outcome evidence.",
    ),
    "tags": (
        "exploratory_not_scored",
        "Potential feature only after label-definition and stability review.",
    ),
    "analysis": (
        "exploratory_not_scored",
        "Model-produced or free-form analysis is not accepted as validation ground truth.",
    ),
    "enriched_content": (
        "not_used",
        "Derived text is privacy-sensitive and unnecessary for aggregate observable validation.",
    ),
    "post_hash": (
        "not_used",
        "Row-level hash is excluded from all evidence artifacts.",
    ),
    "created_at": (
        "not_used",
        "Database ingestion metadata is not interpreted as post occurrence time.",
    ),
    "tags_text": (
        "exploratory_not_scored",
        "Denormalized tags require label review before analytical use.",
    ),
    "comments_missing": (
        "quality_flag_only",
        "May diagnose collection completeness but is not a student or governance outcome.",
    ),
    "missing_comment_count": (
        "quality_flag_only",
        "May diagnose collection completeness but is not used to impute outcomes.",
    ),
}

COMMENT_FIELD_POLICIES = {
    "id": (
        "not_used",
        "Source row identifier; excluded from every persisted artifact.",
    ),
    "comment_id": (
        "not_used",
        "Direct comment identifier is unnecessary for aggregate validation.",
    ),
    "post_code": (
        "linkage_only",
        "Used in memory to count comments per eligible post, then discarded.",
    ),
    "real_date": (
        "not_latency_usable",
        "The export inherits the parent post date and cannot identify response latency.",
    ),
    "floor": (
        "descriptive_only",
        "Thread position is available but does not identify unique responders or solution quality.",
    ),
    "content": (
        "not_used",
        "Comment text is not required for cumulative linked-comment counts.",
    ),
    "sentiment": (
        "exploratory_not_scored",
        "Completeness can be audited, but the field is not a reviewed governance outcome.",
    ),
    "tags": (
        "exploratory_not_scored",
        "Potential feature only after label-definition and stability review.",
    ),
    "analysis": (
        "exploratory_not_scored",
        "Model-produced or free-form analysis is not accepted as validation ground truth.",
    ),
    "comment_hash": (
        "not_used",
        "Row-level hash is excluded from all evidence artifacts.",
    ),
    "created_at": (
        "not_used",
        "Database ingestion metadata is not interpreted as response time.",
    ),
}

STRUCTURE_AUDIT_FIELDS = {
    "sentiment",
    "tags",
    "analysis",
    "enriched_content",
    "tags_text",
}

GRADE_ORDER = {
    "strong": 0,
    "moderate": 1,
    "weak": 2,
    "insufficient_data": 3,
}

REQUIRED_POST_FIELDS = frozenset({"post_code", "real_date", "content"})
REQUIRED_COMMENT_FIELDS = frozenset({"post_code", "real_date"})


def _empty_field_stats(policies: dict[str, tuple[str, str]]) -> dict[str, Counter]:
    return {field: Counter() for field in policies}


def _structured_kind(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "empty"
    if stripped[:1] not in "[{":
        return "scalar_text"
    try:
        parsed = json.loads(stripped)
        return "json_object" if isinstance(parsed, dict) else "json_array"
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                return "python_literal_object"
            if isinstance(parsed, (list, tuple)):
                return "python_literal_array"
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            pass
    return "invalid_structured_text"


def _update_field_stats(
    stats: dict[str, Counter],
    row: dict[str, str],
    present_fields: set[str],
    month: str | None,
) -> None:
    for field, counter in stats.items():
        if field not in present_fields:
            continue
        counter["rows"] += 1
        if month:
            counter[f"month_rows:{month}"] += 1
        value = row.get(field, "") or ""
        normalized = value.strip().lower()
        if normalized in {"", "null", "none", "nan", "{}", "[]"}:
            continue
        counter["nonempty"] += 1
        if month:
            counter[f"month_nonempty:{month}"] += 1
        counter["characters"] += len(value)
        counter["max_characters"] = max(counter["max_characters"], len(value))
        if field in STRUCTURE_AUDIT_FIELDS:
            counter[f"format:{_structured_kind(value)}"] += 1


def _finalize_field_audit(
    stats: dict[str, Counter],
    policies: dict[str, tuple[str, str]],
    present_fields: set[str],
    dataset_kind: str,
) -> list[dict]:
    result = []
    for field, (model_use, reason) in policies.items():
        counter = stats[field]
        rows = counter["rows"]
        nonempty = counter["nonempty"]
        present = field in present_fields
        parseable_source_dates = sum(
            value for key, value in counter.items() if key.startswith("month_rows:")
        )
        formats = {
            key.removeprefix("format:"): value
            for key, value in sorted(counter.items())
            if key.startswith("format:")
        }
        monthly_rates = {}
        for key, month_rows in sorted(counter.items()):
            if not key.startswith("month_rows:") or not month_rows:
                continue
            month = key.removeprefix("month_rows:")
            monthly_rates[month] = counter[f"month_nonempty:{month}"] / month_rows
        if not present:
            field_gate = "excluded_missing"
        elif not rows or not nonempty:
            field_gate = "excluded_empty"
        elif field == "real_date" and not parseable_source_dates:
            field_gate = "excluded_unparseable"
        elif model_use == "direct_observable":
            field_gate = "pass"
        elif model_use in {"linkage_only", "quality_flag_only", "descriptive_only"}:
            field_gate = "supporting_only"
        elif model_use == "keyword_proxy_only":
            field_gate = "proxy_only"
        elif model_use == "exploratory_not_scored":
            eligible_month_rates = [
                rate
                for month, rate in monthly_rates.items()
                if counter[f"month_rows:{month}"] >= 100
            ]
            if (
                dataset_kind == "comments"
                and eligible_month_rates
                and min(eligible_month_rates) < 0.20
                and max(eligible_month_rates) > 0.40
            ):
                field_gate = "reject_temporal_missingness"
            else:
                field_gate = "auxiliary_only"
        else:
            field_gate = "excluded"
        result.append(
            {
                "field": field,
                "present": present,
                "rows_audited": rows,
                "nonempty_count": nonempty,
                "nonempty_rate": nonempty / rows if rows else 0.0,
                "parseable_date_count": (
                    parseable_source_dates if field == "real_date" else None
                ),
                "parseable_date_rate": (
                    parseable_source_dates / rows
                    if field == "real_date" and rows
                    else None
                ),
                "mean_characters_if_nonempty": (
                    counter["characters"] / nonempty if nonempty else 0.0
                ),
                "max_characters": counter["max_characters"],
                "format_counts": formats,
                "monthly_nonempty_rate": monthly_rates,
                "model_use": model_use,
                "field_gate": field_gate,
                "reason": reason,
            }
        )
    return result


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
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _date_range(start: date, end_exclusive: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end_exclusive - start).days)]


def _discover_peaks(
    daily_counts: Counter,
    daily_total_posts: Counter,
    start: date,
    end_exclusive: date,
    *,
    max_events: int,
    minimum_separation_days: int,
    minimum_peak_posts: int,
    minimum_peak_lift: float,
    local_radius_days: int = 3,
) -> tuple[list[date], dict]:
    valid_dates = [
        day
        for day in _date_range(start, end_exclusive)
        if day.isoformat() not in DEFAULT_GAPS
    ]
    counts = [daily_counts[day] for day in valid_dates]
    shares = [
        daily_counts[day] / daily_total_posts[day]
        if daily_total_posts[day]
        else 0.0
        for day in valid_dates
    ]
    descriptive_share_p75 = _quantile(shares, 0.75)
    lift_by_day = {}
    for day in valid_dates:
        baseline_dates = [
            day - timedelta(days=offset)
            for offset in range(4, 18)
            if start <= day - timedelta(days=offset) < end_exclusive
            and (day - timedelta(days=offset)).isoformat() not in DEFAULT_GAPS
        ]
        baseline_shares = [
            daily_counts[baseline_day] / daily_total_posts[baseline_day]
            if daily_total_posts[baseline_day]
            else 0.0
            for baseline_day in baseline_dates
        ]
        baseline = median(baseline_shares) if len(baseline_shares) >= 5 else 0.0
        current_share = (
            daily_counts[day] / daily_total_posts[day]
            if daily_total_posts[day]
            else 0.0
        )
        lift_by_day[day] = current_share / baseline if baseline > 0 else 0.0
    candidates = []
    for day in valid_dates:
        value = daily_counts[day]
        share = value / daily_total_posts[day] if daily_total_posts[day] else 0.0
        lift = lift_by_day[day]
        if value < minimum_peak_posts or lift < minimum_peak_lift:
            continue
        neighbors = [
            lift_by_day.get(day + timedelta(days=offset), 0.0)
            for offset in range(-local_radius_days, local_radius_days + 1)
            if start <= day + timedelta(days=offset) < end_exclusive
            and (day + timedelta(days=offset)).isoformat() not in DEFAULT_GAPS
        ]
        if lift == max(neighbors):
            candidates.append(day)

    selected = []
    for day in sorted(
        candidates,
        key=lambda item: (
            -lift_by_day[item],
            -daily_counts[item],
            item,
        ),
    ):
        if all(abs((day - chosen).days) >= minimum_separation_days for chosen in selected):
            selected.append(day)
        if len(selected) == max_events:
            break
    selected.sort()
    return selected, {
        "valid_days": len(valid_dates),
        "daily_count_median": median(counts) if counts else 0.0,
        "daily_count_p75": _quantile(counts, 0.75),
        "daily_proxy_share_median": median(shares) if shares else 0.0,
        "daily_proxy_share_p75_descriptive": descriptive_share_p75,
        "selection_rule": (
            "Local maxima must meet the declared absolute proxy-post and prior-baseline "
            "lift minima; the proxy-share p75 above is descriptive and is not a filter."
        ),
        "minimum_peak_posts": minimum_peak_posts,
        "minimum_peak_lift_vs_prior_baseline": minimum_peak_lift,
        "selected_peak_lifts": {
            day.isoformat(): lift_by_day[day] for day in selected
        },
        "candidate_local_peaks": len(candidates),
    }


def _event_record(
    peak: date,
    daily_post_codes: dict[date, list[str]],
    daily_total_posts: Counter,
    comments_per_post: Counter,
    *,
    radius_days: int,
    scope_start: date,
    scope_end_exclusive: date,
) -> tuple[dict, list[int]]:
    values: list[int] = []
    daily = []
    for offset in range(-radius_days, radius_days + 1):
        day = peak + timedelta(days=offset)
        observed = (
            scope_start <= day < scope_end_exclusive
            and day.isoformat() not in DEFAULT_GAPS
        )
        codes = daily_post_codes.get(day, []) if observed else []
        if observed:
            values.extend(comments_per_post[code] for code in codes)
        total_posts = daily_total_posts[day] if observed else None
        daily.append(
            {
                "offset_days": offset,
                "observed": observed,
                "proxy_posts": len(codes) if observed else None,
                "all_posts": total_posts,
                "proxy_share": (
                    len(codes) / total_posts
                    if observed and total_posts
                    else (0.0 if observed else None)
                ),
            }
        )
    summary = summarize_counts(values)
    return (
        {
            "peak_date": peak.isoformat(),
            "window_start": (peak - timedelta(days=radius_days)).isoformat(),
            "window_end_inclusive": (peak + timedelta(days=radius_days)).isoformat(),
            "observed_days": sum(item["observed"] for item in daily),
            "excluded_gap_offsets": [
                item["offset_days"] for item in daily if not item["observed"]
            ],
            "daily_proxy_posts": daily,
            "proxy_posts": summary["proxy_posts"],
            "natural_response_summary_unscored": summary,
        },
        values,
    )


def _normalized_profile(event: dict) -> dict[int, float]:
    observed = {
        item["offset_days"]: item["proxy_share"]
        for item in event["daily_proxy_posts"]
        if item["observed"]
    }
    total = sum(observed.values())
    if not total:
        return {offset: 0.0 for offset in observed}
    return {offset: count / total for offset, count in observed.items()}


def _arrival_profile_comparison(training_events: list[dict], target: dict) -> dict:
    target_profile = _normalized_profile(target)
    training_profiles = [_normalized_profile(event) for event in training_events]
    reference = {}
    for offset in target_profile:
        available = [profile[offset] for profile in training_profiles if offset in profile]
        if available:
            reference[offset] = sum(available) / len(available)
    common_offsets = sorted(set(target_profile) & set(reference))
    if not common_offsets:
        return {"grade": "insufficient_data", "normalized_profile_mae": None}
    target_total = sum(target_profile[offset] for offset in common_offsets)
    reference_total = sum(reference[offset] for offset in common_offsets)
    if not target_total or not reference_total:
        return {"grade": "insufficient_data", "normalized_profile_mae": None}
    error = sum(
        abs(
            target_profile[offset] / target_total
            - reference[offset] / reference_total
        )
        for offset in common_offsets
    ) / len(common_offsets)
    if error <= 0.05:
        grade = "strong"
    elif error <= 0.10:
        grade = "moderate"
    else:
        grade = "weak"
    return {
        "grade": grade,
        "normalized_profile_mae": error,
        "compared_offsets": common_offsets,
        "rubric": {
            "strong": "normalized seven-day proxy-share profile MAE <= 0.05",
            "moderate": "normalized seven-day proxy-share profile MAE <= 0.10",
            "weak": "normalized seven-day proxy-share profile MAE > 0.10",
        },
    }


def _worst_grade(grades: list[str]) -> str:
    return max(grades, key=lambda grade: GRADE_ORDER[grade])


def _backtest_events(events_with_values: list[tuple[dict, list[int]]]) -> dict:
    events = [event for event, _ in events_with_values]
    if len(events) < 3:
        return {
            "events": events,
            "folds": [],
            "event_count": len(events),
            "overall_grade": "insufficient_data",
            "stable_fold_fraction": 0.0,
            "grade_counts": {},
            "mean_normalized_arrival_profile_mae": None,
            "interpretation": "At least three non-overlapping proxy event windows are required.",
            "evidence_scope": "retrospective_event_shape_repeatability_only",
        }

    folds = []
    for holdout_index, (holdout_event, holdout_values) in enumerate(events_with_values):
        training_pairs = [
            pair for index, pair in enumerate(events_with_values) if index != holdout_index
        ]
        training_events = [event for event, _ in training_pairs]
        arrival_comparison = _arrival_profile_comparison(training_events, holdout_event)
        fold_grade = arrival_comparison["grade"]
        training_totals = [event["proxy_posts"] for event in training_events]
        expected_volume = median(training_totals) if training_totals else 0.0
        target_volume = holdout_event["proxy_posts"]
        folds.append(
            {
                "holdout_peak_date": holdout_event["peak_date"],
                "training_event_count": len(training_events),
                "training_proxy_posts": sum(
                    event["proxy_posts"] for event in training_events
                ),
                "holdout_proxy_posts": len(holdout_values),
                "fold_grade": fold_grade,
                "arrival_profile": arrival_comparison,
                "unscored_volume_relative_error": (
                    abs(expected_volume - target_volume) / target_volume
                    if target_volume
                    else None
                ),
            }
        )

    grade_counts = Counter(fold["fold_grade"] for fold in folds)
    stable_folds = grade_counts["strong"] + grade_counts["moderate"]
    profile_errors = [
        fold["arrival_profile"]["normalized_profile_mae"]
        for fold in folds
        if fold["arrival_profile"]["normalized_profile_mae"] is not None
    ]
    return {
        "events": events,
        "folds": folds,
        "event_count": len(events),
        "overall_grade": _worst_grade([fold["fold_grade"] for fold in folds]),
        "stable_fold_fraction": stable_folds / len(folds),
        "grade_counts": dict(sorted(grade_counts.items())),
        "mean_normalized_arrival_profile_mae": (
            sum(profile_errors) / len(profile_errors) if profile_errors else None
        ),
        "interpretation": (
            "Each fold pools all other discovered windows and holds out one complete event. "
            "The conservative grade is the weakest normalized proxy-share arrival-shape fold. "
            "Event occurrence, absolute volume, comment behavior, and policy effects are not scored here."
        ),
        "evidence_scope": "retrospective_event_shape_repeatability_only",
    }


def build_focus_evidence(
    posts_path: str | Path,
    comments_path: str | Path,
    *,
    audit_start: str = "2025-12-01",
    audit_end: str = "2026-07-21",
    event_start: str = "2026-05-01",
    event_end: str = "2026-07-16",
    platform_marker: str = "NP",
    event_window_radius_days: int = 3,
    max_events: int = 5,
    minimum_event_posts: int = 40,
    minimum_event_separation_days: int = 14,
    minimum_peak_posts: int = 5,
    minimum_peak_lift: float = 1.25,
) -> dict:
    """Audit fields and backtest repeated focus events without persisting row data."""

    audit_start_dt = datetime.fromisoformat(audit_start)
    audit_end_dt = datetime.fromisoformat(audit_end)
    event_start_date = datetime.fromisoformat(event_start).date()
    event_end_date = datetime.fromisoformat(event_end).date()
    if not audit_start_dt < audit_end_dt:
        raise ValueError("expected audit_start < audit_end")
    if not audit_start_dt.date() <= event_start_date < event_end_date <= audit_end_dt.date():
        raise ValueError("event window search must be inside the audit range")
    if event_window_radius_days < 1:
        raise ValueError("event_window_radius_days must be positive")
    if max_events < 3:
        raise ValueError("max_events must be at least three")
    if (
        minimum_event_posts <= 0
        or minimum_event_separation_days <= 0
        or minimum_peak_posts <= 0
        or minimum_peak_lift <= 0
    ):
        raise ValueError("event minima must be positive")

    post_stats = _empty_field_stats(POST_FIELD_POLICIES)
    comment_stats = _empty_field_stats(COMMENT_FIELD_POLICIES)
    eligible_post_codes: set[str] = set()
    all_post_dates: dict[str, datetime] = {}
    source_post_rows = 0
    source_comment_rows = 0
    linked_source_comments = 0
    inherited_comment_dates = 0
    post_codes_seen: set[str] = set()
    post_hashes_seen: set[str] = set()
    comment_ids_seen: set[str] = set()
    comment_pairs_seen: set[tuple[str, str]] = set()
    comment_hashes_seen: set[str] = set()
    duplicate_counts = Counter()
    daily_total_posts = Counter()
    daily_post_codes: dict[str, dict[date, list[str]]] = {
        scenario_id: defaultdict(list) for scenario_id in FOCUS_DEFINITIONS
    }
    daily_counts: dict[str, Counter] = {
        scenario_id: Counter() for scenario_id in FOCUS_DEFINITIONS
    }

    with Path(posts_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        post_fields = set(reader.fieldnames or [])
        missing_post_fields = sorted(REQUIRED_POST_FIELDS - post_fields)
        if missing_post_fields:
            raise ValueError(
                "posts CSV missing required fields: " + ", ".join(missing_post_fields)
            )
        for row in reader:
            source_post_rows += 1
            code = row.get("post_code", "")
            timestamp = _parse_date(row.get("real_date", ""))
            _update_field_stats(
                post_stats,
                row,
                post_fields,
                timestamp.strftime("%Y-%m") if timestamp else None,
            )
            if code:
                if code in post_codes_seen:
                    duplicate_counts["post_code"] += 1
                else:
                    post_codes_seen.add(code)
                if timestamp:
                    all_post_dates[code] = timestamp
            post_hash = row.get("post_hash", "")
            if post_hash:
                if post_hash in post_hashes_seen:
                    duplicate_counts["post_hash"] += 1
                else:
                    post_hashes_seen.add(post_hash)
            if (
                platform_marker not in code
                or timestamp is None
                or not audit_start_dt <= timestamp < audit_end_dt
                or timestamp.date().isoformat() in DEFAULT_GAPS
            ):
                continue
            eligible_post_codes.add(code)
            if not event_start_date <= timestamp.date() < event_end_date:
                continue
            daily_total_posts[timestamp.date()] += 1
            content = row.get("content", "") or ""
            for scenario_id, definition in FOCUS_DEFINITIONS.items():
                if definition["pattern"].search(content):
                    daily_post_codes[scenario_id][timestamp.date()].append(code)
                    daily_counts[scenario_id][timestamp.date()] += 1

    comments_per_post = Counter()
    with Path(comments_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        comment_fields = set(reader.fieldnames or [])
        missing_comment_fields = sorted(REQUIRED_COMMENT_FIELDS - comment_fields)
        if missing_comment_fields:
            raise ValueError(
                "comments CSV missing required fields: "
                + ", ".join(missing_comment_fields)
            )
        for row in reader:
            source_comment_rows += 1
            code = row.get("post_code", "")
            timestamp = _parse_date(row.get("real_date", ""))
            _update_field_stats(
                comment_stats,
                row,
                comment_fields,
                timestamp.strftime("%Y-%m") if timestamp else None,
            )
            comment_id = row.get("comment_id", "")
            if comment_id:
                if comment_id in comment_ids_seen:
                    duplicate_counts["comment_id"] += 1
                else:
                    comment_ids_seen.add(comment_id)
                pair = (code, comment_id)
                if pair in comment_pairs_seen:
                    duplicate_counts["post_code_comment_id"] += 1
                else:
                    comment_pairs_seen.add(pair)
            comment_hash = row.get("comment_hash", "")
            if comment_hash:
                if comment_hash in comment_hashes_seen:
                    duplicate_counts["comment_hash"] += 1
                else:
                    comment_hashes_seen.add(comment_hash)
            parent_timestamp = all_post_dates.get(code)
            if parent_timestamp is not None:
                linked_source_comments += 1
                if timestamp == parent_timestamp:
                    inherited_comment_dates += 1
            if code not in eligible_post_codes:
                continue
            comments_per_post[code] += 1

    scenario_backtests = {}
    discovery_audit = {}
    for scenario_id, definition in FOCUS_DEFINITIONS.items():
        peaks, peak_audit = _discover_peaks(
            daily_counts[scenario_id],
            daily_total_posts,
            event_start_date,
            event_end_date,
            max_events=max_events,
            minimum_separation_days=minimum_event_separation_days,
            minimum_peak_posts=minimum_peak_posts,
            minimum_peak_lift=minimum_peak_lift,
        )
        records = [
            _event_record(
                peak,
                daily_post_codes[scenario_id],
                daily_total_posts,
                comments_per_post,
                radius_days=event_window_radius_days,
                scope_start=event_start_date,
                scope_end_exclusive=event_end_date,
            )
            for peak in peaks
        ]
        records = [
            pair
            for pair in records
            if pair[0]["proxy_posts"] >= minimum_event_posts
            and pair[0]["observed_days"] >= 2 * event_window_radius_days
        ]
        scenario_backtests[scenario_id] = {
            "title": definition["title"],
            "proxy_version": definition["proxy_version"],
            "proxy_definition": "Declared scenario keyword pattern; overlapping and not human-reviewed.",
            **_backtest_events(records),
        }
        discovery_audit[scenario_id] = {
            **peak_audit,
            "selected_peaks_before_window_minimum": len(peaks),
            "retained_events": len(records),
        }

    post_field_audit = _finalize_field_audit(
        post_stats, POST_FIELD_POLICIES, post_fields, "posts"
    )
    comment_field_audit = _finalize_field_audit(
        comment_stats, COMMENT_FIELD_POLICIES, comment_fields, "comments"
    )
    gate_counts = Counter(
        item["field_gate"] for item in post_field_audit + comment_field_audit
    )

    return {
        "schema_version": "1.1.0",
        "methodology": {
            "field_audit_scope": "entire_source_files",
            "eligible_scoring_window": {
                "start_inclusive": audit_start,
                "end_exclusive": audit_end,
            },
            "event_search_window": {
                "start_inclusive": event_start,
                "end_exclusive": event_end,
            },
            "event_design": "deterministic separated local peaks with leave-one-event-out backtesting",
            "event_window_radius_days": event_window_radius_days,
            "maximum_events_per_focus": max_events,
            "minimum_event_posts": minimum_event_posts,
            "minimum_event_separation_days": minimum_event_separation_days,
            "minimum_peak_posts": minimum_peak_posts,
            "minimum_peak_lift_vs_prior_baseline": minimum_peak_lift,
            "excluded_known_gap_dates": sorted(DEFAULT_GAPS),
            "selection_note": (
                "Peak discovery conditions on observed proxy activity. The backtest evaluates "
                "within-event shape and natural comment attention, not event occurrence or volume forecasting."
            ),
        },
        "privacy": (
            "Aggregate schema completeness and event summaries only; no text, row identifier, "
            "source hash, or source path is returned."
        ),
        "field_audit": {
            "posts": {
                "source_rows": source_post_rows,
                "eligible_rows": len(eligible_post_codes),
                "fields": post_field_audit,
            },
            "comments": {
                "source_rows": source_comment_rows,
                "eligible_rows": sum(comments_per_post.values()),
                "linked_to_supplied_post_rows": linked_source_comments,
                "linked_to_supplied_post_rate": (
                    linked_source_comments / source_comment_rows
                    if source_comment_rows
                    else 0.0
                ),
                "linked_comment_date_equals_parent_rate": (
                    inherited_comment_dates / linked_source_comments
                    if linked_source_comments
                    else 0.0
                ),
                "fields": comment_field_audit,
            },
            "gate_counts": dict(sorted(gate_counts.items())),
            "identifier_integrity": {
                "duplicate_post_code_rows": duplicate_counts["post_code"],
                "duplicate_post_hash_rows": duplicate_counts["post_hash"],
                "duplicate_comment_id_rows": duplicate_counts["comment_id"],
                "duplicate_post_code_comment_id_rows": duplicate_counts[
                    "post_code_comment_id"
                ],
                "duplicate_comment_hash_rows": duplicate_counts["comment_hash"],
                "decision": (
                    "Comments are counted by source rows after post linkage; comment_id is not "
                    "used as a global deduplication key."
                ),
            },
            "decision": (
                "Only post dates, in-memory post-comment linkage, scenario keyword proxies, "
                "and final linked-comment counts enter current scoring. Annotation fields remain "
                "exploratory because completeness does not establish label validity."
            ),
        },
        "event_discovery_audit": discovery_audit,
        "scenario_backtests": scenario_backtests,
    }
