"""Offline, aggregate-only replay for two externally anchored campus shocks.

The replay uses the authorized post and comment exports to build four-stage
benchmarks.  It never writes source text or row identifiers.  Real timestamps
only define historical bins; a simulation run advances through synthetic
stages immediately and does not wait for wall-clock time.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .calibration import DEFAULT_GAPS, _file_identity, _parse_date


HERE = Path(__file__).resolve().parent
DEFAULT_DEFINITIONS = HERE / "shock_events.json"
BENCHMARK_SCHEMA_VERSION = "1.0.0"
REPLAY_SCHEMA_VERSION = "1.0.0"
EXPECTED_PHASE_IDS = ("baseline", "burst", "spread", "decay")
ALLOWED_SHOCK_ORIGINS = {"governance_action", "exogenous_event"}
GRADE_THRESHOLDS = {
    "strong": {
        "primary_stage_scale_wape": 0.25,
        "topic_incidence_composition_tvd": 0.20,
        "primary_peak_stage_distance": 0,
        "primary_tail_share_error": 0.15,
    },
    "moderate": {
        "primary_stage_scale_wape": 0.40,
        "topic_incidence_composition_tvd": 0.30,
        "primary_peak_stage_distance": 1,
        "primary_tail_share_error": 0.25,
    },
}
GRADE_RANK = {"strong": 0, "moderate": 1, "weak": 2, "insufficient_data": 3}
AGGREGATE_PRIVACY_K = 5
FIRST_POST_DELAY_BUCKET_MINUTES = 15


class ShockReplayError(ValueError):
    """Raised when definitions or aggregate replay inputs fail closed."""


def _dates_touched(start: datetime, end: datetime) -> set[str]:
    """Return local calendar dates touched by the half-open interval [start, end)."""

    if start >= end:
        raise ShockReplayError("event window must have start < end")
    cursor = start.date()
    last = (end - timedelta(microseconds=1)).date()
    touched = set()
    while cursor <= last:
        touched.add(cursor.isoformat())
        cursor += timedelta(days=1)
    return touched


def _known_gaps_in_window(start: datetime, end: datetime) -> list[str]:
    return sorted(_dates_touched(start, end) & set(DEFAULT_GAPS))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
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


def _check_probability_distribution(values: dict[str, Any], path: str) -> None:
    if not values:
        raise ShockReplayError(f"{path} must be a non-empty mapping")
    weights = []
    for role_id, role in values.items():
        if not isinstance(role, dict):
            raise ShockReplayError(f"{path}.{role_id} must be an object")
        weight = role.get("weight")
        multiplier = role.get("response_multiplier")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ShockReplayError(f"{path}.{role_id}.weight must be non-negative")
        if not isinstance(multiplier, (int, float)) or multiplier <= 0:
            raise ShockReplayError(
                f"{path}.{role_id}.response_multiplier must be positive"
            )
        weights.append(float(weight))
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ShockReplayError(f"{path} weights must sum to 1")


def load_shock_definitions(path: str | Path = DEFAULT_DEFINITIONS) -> dict[str, Any]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0.0":
        raise ShockReplayError("unsupported shock definition schema_version")
    role_models = raw.get("role_models")
    events = raw.get("events")
    if not isinstance(role_models, dict) or not isinstance(events, list) or not events:
        raise ShockReplayError("definitions require role_models and a non-empty events list")
    for shock_type, role_model in role_models.items():
        if role_model.get("source_type") != "assumption":
            raise ShockReplayError(
                f"role_models.{shock_type}.source_type must remain assumption"
            )
        _check_probability_distribution(
            role_model.get("roles", {}), f"role_models.{shock_type}.roles"
        )

    seen_scenarios: set[str] = set()
    for event_index, event in enumerate(events):
        prefix = f"events[{event_index}]"
        scenario_id = event.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ShockReplayError(f"{prefix}.scenario_id is required")
        if scenario_id in seen_scenarios:
            raise ShockReplayError(f"duplicate scenario_id: {scenario_id}")
        seen_scenarios.add(scenario_id)
        shock_type = event.get("shock_type")
        if shock_type not in role_models:
            raise ShockReplayError(f"{prefix}.shock_type has no role model")
        shock_origin = event.get("shock_origin")
        if shock_origin not in ALLOWED_SHOCK_ORIGINS:
            raise ShockReplayError(
                f"{prefix}.shock_origin must be governance_action or exogenous_event"
            )
        if not isinstance(event.get("shock_origin_definition"), str) or not event[
            "shock_origin_definition"
        ]:
            raise ShockReplayError(f"{prefix}.shock_origin_definition is required")
        if event.get("anchor_timezone") != "Asia/Shanghai":
            raise ShockReplayError(f"{prefix}.anchor_timezone must be Asia/Shanghai")
        if event.get("anchor_resolution") not in {"day", "minute", "second"}:
            raise ShockReplayError(f"{prefix}.anchor_resolution is unsupported")
        if not isinstance(event.get("comment_collection_regime"), str) or not event[
            "comment_collection_regime"
        ]:
            raise ShockReplayError(f"{prefix}.comment_collection_regime is required")
        if not isinstance(event.get("topic_rules_version"), str) or not event[
            "topic_rules_version"
        ]:
            raise ShockReplayError(f"{prefix}.topic_rules_version is required")
        try:
            anchor = datetime.fromisoformat(event["anchor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShockReplayError(f"{prefix}.anchor must be ISO-8601") from exc
        if anchor.utcoffset() is not None:
            raise ShockReplayError(
                f"{prefix}.anchor must be a naive Asia/Shanghai wall time"
            )
        anchor_source = event.get("anchor_source")
        if not isinstance(anchor_source, dict):
            raise ShockReplayError(f"{prefix}.anchor_source is required")
        for key in ("publisher", "url", "evidence"):
            if not isinstance(anchor_source.get(key), str) or not anchor_source[key]:
                raise ShockReplayError(f"{prefix}.anchor_source.{key} is required")

        phases = event.get("phases")
        if not isinstance(phases, list) or len(phases) != len(EXPECTED_PHASE_IDS):
            raise ShockReplayError(f"{prefix}.phases must contain exactly four stages")
        phase_ids: set[str] = set()
        prior_end: int | float | None = None
        for phase_index, phase in enumerate(phases):
            phase_path = f"{prefix}.phases[{phase_index}]"
            phase_id = phase.get("phase_id")
            start = phase.get("start_offset_minutes")
            end = phase.get("end_offset_minutes")
            if not isinstance(phase_id, str) or not phase_id or phase_id in phase_ids:
                raise ShockReplayError(f"{phase_path}.phase_id must be unique")
            phase_ids.add(phase_id)
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ShockReplayError(f"{phase_path} offsets must be numeric")
            if start >= end:
                raise ShockReplayError(f"{phase_path} must have start < end")
            if prior_end is not None and start != prior_end:
                raise ShockReplayError(f"{prefix}.phases must be ordered and contiguous")
            prior_end = end
        if tuple(phase["phase_id"] for phase in phases) != EXPECTED_PHASE_IDS:
            raise ShockReplayError(
                f"{prefix}.phases must be baseline, burst, spread, decay"
            )
        window_start = anchor + timedelta(minutes=phases[0]["start_offset_minutes"])
        window_end = anchor + timedelta(minutes=phases[-1]["end_offset_minutes"])
        intersecting_gaps = _known_gaps_in_window(window_start, window_end)
        if intersecting_gaps:
            raise ShockReplayError(
                f"{prefix} window intersects known crawl gap(s): "
                + ", ".join(intersecting_gaps)
            )

        topics = event.get("topics")
        if not isinstance(topics, list) or not topics:
            raise ShockReplayError(f"{prefix}.topics must be non-empty")
        prior_topics: set[str] = set()
        for topic_index, topic in enumerate(topics):
            topic_path = f"{prefix}.topics[{topic_index}]"
            topic_id = topic.get("topic_id")
            if not isinstance(topic_id, str) or not topic_id or topic_id in prior_topics:
                raise ShockReplayError(f"{topic_path}.topic_id must be unique")
            terms = topic.get("include_any", [])
            regexes = topic.get("include_regexes", [])
            if not isinstance(terms, list) or not isinstance(regexes, list):
                raise ShockReplayError(f"{topic_path} include rules must be lists")
            if not terms and not regexes:
                raise ShockReplayError(f"{topic_path} requires an include rule")
            if any(not isinstance(term, str) or not term for term in terms):
                raise ShockReplayError(f"{topic_path}.include_any contains an invalid term")
            for pattern in regexes:
                try:
                    re.compile(pattern)
                except (TypeError, re.error) as exc:
                    raise ShockReplayError(
                        f"{topic_path}.include_regexes contains invalid regex"
                    ) from exc
            dependencies = []
            if topic.get("requires_topic") is not None:
                dependencies.append(topic["requires_topic"])
            excluded_topics = topic.get("exclude_topics", [])
            if not isinstance(excluded_topics, list):
                raise ShockReplayError(f"{topic_path}.exclude_topics must be a list")
            if topic.get("requires_topic") is not None and excluded_topics:
                raise ShockReplayError(
                    f"{topic_path} cannot combine requires_topic and exclude_topics in v1"
                )
            if len(excluded_topics) > 1:
                raise ShockReplayError(
                    f"{topic_path} supports at most one exclude_topic in v1"
                )
            dependencies.extend(excluded_topics)
            if any(dependency not in prior_topics for dependency in dependencies):
                raise ShockReplayError(
                    f"{topic_path} dependencies must reference earlier topics"
                )
            prior_topics.add(topic_id)
        if event.get("primary_topic_id") not in prior_topics:
            raise ShockReplayError(f"{prefix}.primary_topic_id is unknown")
        identification = event.get("identification", {})
        for key in (
            "evidence_stage",
            "identification_strength",
            "estimand",
            "claim_boundary",
        ):
            if not isinstance(identification.get(key), str) or not identification[key]:
                raise ShockReplayError(f"{prefix}.identification.{key} is required")
        if identification.get("causal_status") != "not_identified":
            raise ShockReplayError(
                f"{prefix}.identification.causal_status must be not_identified"
            )
    if len(events) != 2 or {event["shock_origin"] for event in events} != (
        ALLOWED_SHOCK_ORIGINS
    ):
        raise ShockReplayError(
            "definitions must contain one governance_action and one exogenous_event"
        )
    if len({event["comment_collection_regime"] for event in events}) != len(events):
        raise ShockReplayError(
            "events on opposite sides of the crawl break require distinct comment regimes"
        )
    raw["definition_fingerprint"] = _canonical_hash(
        {key: value for key, value in raw.items() if key != "definition_fingerprint"}
    )
    return raw


def _compile_topics(event: dict[str, Any]) -> list[dict[str, Any]]:
    compiled = []
    for topic in event["topics"]:
        item = dict(topic)
        item["compiled_regexes"] = [
            re.compile(pattern) for pattern in topic.get("include_regexes", [])
        ]
        compiled.append(item)
    return compiled


def _topic_flags(text: str, topics: list[dict[str, Any]]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for topic in topics:
        matched = any(term in text for term in topic.get("include_any", [])) or any(
            pattern.search(text) for pattern in topic["compiled_regexes"]
        )
        required = topic.get("requires_topic")
        if required and not flags[required]:
            matched = False
        if any(flags[topic_id] for topic_id in topic.get("exclude_topics", [])):
            matched = False
        flags[topic["topic_id"]] = matched
    return flags


def _phase_for_timestamp(
    timestamp: datetime, anchor: datetime, phases: list[dict[str, Any]]
) -> dict[str, Any] | None:
    offset_minutes = (timestamp - anchor).total_seconds() / 60
    for phase in phases:
        if phase["start_offset_minutes"] <= offset_minutes < phase["end_offset_minutes"]:
            return phase
    return None


def _comment_summary(post_codes: set[str], comments_per_post: Counter) -> dict[str, Any]:
    counts = [comments_per_post[post_code] for post_code in post_codes]
    posts = len(counts)
    posts_with_comments = sum(value > 0 for value in counts)
    total_comments = sum(counts)
    suppressed = posts < AGGREGATE_PRIVACY_K
    return {
        "matched_posts": posts,
        "privacy_minimum_matched_posts": AGGREGATE_PRIVACY_K,
        "suppressed": suppressed,
        "posts_with_linked_comments": None if suppressed else posts_with_comments,
        "nonzero_linked_comment_rate": (
            None if suppressed else posts_with_comments / posts
        ),
        "total_linked_comments": None if suppressed else total_comments,
        "mean_linked_comments_per_post": (
            None if suppressed else total_comments / posts
        ),
        "evidence_use": "descriptive_final_attention_only_not_scored",
    }


def _observed_share_shift(
    phases: list[dict[str, Any]],
    primary_topic_id: str,
    anchor_resolution: str,
) -> dict[str, Any]:
    """Describe the baseline-to-burst discontinuity without claiming a policy effect."""

    by_id = {phase["phase_id"]: phase for phase in phases}
    baseline = by_id["baseline"]
    burst = by_id["burst"]
    baseline_posts = baseline["topics"][primary_topic_id]["posts"]
    burst_posts = burst["topics"][primary_topic_id]["posts"]
    baseline_exposure = baseline["all_posts_exposure"]
    burst_exposure = burst["all_posts_exposure"]
    if not baseline_exposure or not burst_exposure:
        return {
            "status": "insufficient_data",
            "evidence_use": "descriptive_attention_shift_not_policy_effect",
        }
    baseline_share = baseline_posts / baseline_exposure
    burst_share = burst_posts / burst_exposure
    share_multiplier = burst_share / baseline_share if baseline_share else None
    approximate_interval = None
    if (
        baseline_posts > 0
        and burst_posts > 0
        and baseline_posts < baseline_exposure
        and burst_posts < burst_exposure
    ):
        log_standard_error = math.sqrt(
            1 / burst_posts
            - 1 / burst_exposure
            + 1 / baseline_posts
            - 1 / baseline_exposure
        )
        log_multiplier = math.log(share_multiplier)
        approximate_interval = {
            "lower": math.exp(log_multiplier - 1.959963984540054 * log_standard_error),
            "upper": math.exp(log_multiplier + 1.959963984540054 * log_standard_error),
            "method": "large_sample_log_risk_ratio_approximation",
        }
    return {
        "status": "estimable",
        "baseline_primary_posts": baseline_posts,
        "baseline_all_posts": baseline_exposure,
        "baseline_primary_share": baseline_share,
        "burst_primary_posts": burst_posts,
        "burst_all_posts": burst_exposure,
        "burst_primary_share": burst_share,
        "burst_minus_baseline_share": burst_share - baseline_share,
        "burst_to_baseline_share_multiplier": share_multiplier,
        "naive_independent_post_95pct_multiplier_interval": approximate_interval,
        "interval_warning": (
            "Exploratory only: the approximation treats posts as independent and is "
            "not a formal causal confidence interval under time clustering."
        ),
        "evidence_use": (
            "descriptive_event_day_association_not_post_event_effect"
            if anchor_resolution == "day"
            else "descriptive_post_anchor_discontinuity_not_policy_effect"
        ),
        "temporal_interpretation": (
            "爆发窗是事件所在自然日，不能整体解释为事件发生之后。"
            if anchor_resolution == "day"
            else "爆发窗从外部可核验的公开发布时间开始。"
        ),
        "counterfactual_limit": (
            "外部锚点固定了事件时点，但没有未处理对照组，因而不能识别"
            "福利效应或治理干预效果。"
        ),
    }


def build_shock_benchmark(
    posts_path: str | Path,
    comments_path: str | Path,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
) -> dict[str, Any]:
    """Build privacy-preserving event-stage aggregates from both authorized CSVs."""

    definitions = load_shock_definitions(definitions_path)
    runtimes: dict[str, dict[str, Any]] = {}
    event_window_gap_dates: set[str] = set()
    for event in definitions["events"]:
        phases = event["phases"]
        anchor = datetime.fromisoformat(event["anchor"])
        window_start = anchor + timedelta(
            minutes=phases[0]["start_offset_minutes"]
        )
        window_end = anchor + timedelta(
            minutes=phases[-1]["end_offset_minutes"]
        )
        window_gaps = _known_gaps_in_window(window_start, window_end)
        event_window_gap_dates.update(window_gaps)
        if window_gaps:
            raise ShockReplayError(
                f"{event['scenario_id']} window intersects known crawl gap(s): "
                + ", ".join(window_gaps)
            )
        runtimes[event["scenario_id"]] = {
            "event": event,
            "anchor": anchor,
            "topics": _compile_topics(event),
            "window_start": window_start,
            "window_end": window_end,
            "phase_total": Counter(),
            "phase_topic_counts": defaultdict(Counter),
            "phase_topic_codes": defaultdict(lambda: defaultdict(set)),
            "primary_after_anchor": [],
        }

    rows_scanned = 0
    valid_post_rows = 0
    invalid_post_dates = 0
    empty_post_codes = 0
    duplicate_post_code_rows = 0
    posts_in_event_windows = 0
    seen_post_codes: set[str] = set()
    relevant_post_codes: set[str] = set()
    post_source = Path(posts_path)
    with post_source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"post_code", "real_date", "content"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ShockReplayError(f"posts CSV missing required fields: {', '.join(missing)}")
        for row in reader:
            rows_scanned += 1
            timestamp = _parse_date(row.get("real_date", ""))
            if timestamp is None:
                invalid_post_dates += 1
                continue
            post_code = (row.get("post_code") or "").strip()
            if not post_code:
                empty_post_codes += 1
                continue
            if post_code in seen_post_codes:
                duplicate_post_code_rows += 1
                continue
            seen_post_codes.add(post_code)
            valid_post_rows += 1
            text = row.get("content", "") or ""
            matched_any_window = False
            for runtime in runtimes.values():
                if not runtime["window_start"] <= timestamp < runtime["window_end"]:
                    continue
                matched_any_window = True
                phase = _phase_for_timestamp(
                    timestamp, runtime["anchor"], runtime["event"]["phases"]
                )
                if phase is None:
                    continue
                phase_id = phase["phase_id"]
                runtime["phase_total"][phase_id] += 1
                flags = _topic_flags(text, runtime["topics"])
                for topic_id, matched in flags.items():
                    if not matched:
                        continue
                    runtime["phase_topic_counts"][phase_id][topic_id] += 1
                    runtime["phase_topic_codes"][phase_id][topic_id].add(post_code)
                    relevant_post_codes.add(post_code)
                primary_topic_id = runtime["event"]["primary_topic_id"]
                if timestamp >= runtime["anchor"] and flags[primary_topic_id]:
                    runtime["primary_after_anchor"].append(timestamp)
            if matched_any_window:
                posts_in_event_windows += 1

    comments_per_post: Counter = Counter()
    comment_rows_scanned = 0
    linked_comment_rows = 0
    comment_source = Path(comments_path)
    with comment_source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "post_code" not in set(reader.fieldnames or []):
            raise ShockReplayError("comments CSV missing required fields: post_code")
        for row in reader:
            comment_rows_scanned += 1
            post_code = (row.get("post_code") or "").strip()
            if post_code in relevant_post_codes:
                comments_per_post[post_code] += 1
                linked_comment_rows += 1

    event_results = []
    for event in definitions["events"]:
        runtime = runtimes[event["scenario_id"]]
        phases = []
        for phase in event["phases"]:
            phase_id = phase["phase_id"]
            total_posts = runtime["phase_total"][phase_id]
            topic_results = {}
            for topic in event["topics"]:
                topic_id = topic["topic_id"]
                count = runtime["phase_topic_counts"][phase_id][topic_id]
                topic_results[topic_id] = {
                    "label": topic["label"],
                    "posts": count,
                    "share_of_all_posts": count / total_posts if total_posts else 0.0,
                    "final_comment_attention": _comment_summary(
                        runtime["phase_topic_codes"][phase_id][topic_id],
                        comments_per_post,
                    ),
                }
            phases.append(
                {
                    "phase_id": phase_id,
                    "label": phase["label"],
                    "start_offset_minutes": phase["start_offset_minutes"],
                    "end_offset_minutes": phase["end_offset_minutes"],
                    "duration_minutes": phase["end_offset_minutes"]
                    - phase["start_offset_minutes"],
                    "all_posts_exposure": total_posts,
                    "topics": topic_results,
                }
            )
        first_response_delay_bucket = None
        first_hour_primary_posts = None
        if event["anchor_resolution"] != "day" and runtime["primary_after_anchor"]:
            first_response_minutes = (
                min(runtime["primary_after_anchor"]) - runtime["anchor"]
            ).total_seconds() / 60
            bucket_start = (
                math.floor(first_response_minutes / FIRST_POST_DELAY_BUCKET_MINUTES)
                * FIRST_POST_DELAY_BUCKET_MINUTES
            )
            first_response_delay_bucket = {
                "lower_inclusive": bucket_start,
                "upper_exclusive": bucket_start + FIRST_POST_DELAY_BUCKET_MINUTES,
                "unit": "minutes",
            }
            first_hour_count = sum(
                runtime["anchor"]
                <= timestamp
                < runtime["anchor"] + timedelta(minutes=60)
                for timestamp in runtime["primary_after_anchor"]
            )
            first_hour_primary_posts = (
                first_hour_count if first_hour_count >= AGGREGATE_PRIVACY_K else None
            )
        event_results.append(
            {
                "scenario_id": event["scenario_id"],
                "title": event["title"],
                "shock_type": event["shock_type"],
                "shock_origin": event["shock_origin"],
                "shock_origin_definition": event["shock_origin_definition"],
                "anchor": event["anchor"],
                "anchor_timezone": event["anchor_timezone"],
                "anchor_resolution": event["anchor_resolution"],
                "comment_collection_regime": event["comment_collection_regime"],
                "topic_rules_version": event["topic_rules_version"],
                "anchor_source": event["anchor_source"],
                "identification": event["identification"],
                "primary_topic_id": event["primary_topic_id"],
                "first_related_post_delay_bucket": first_response_delay_bucket,
                "first_hour_primary_posts": first_hour_primary_posts,
                "first_hour_primary_posts_suppressed": (
                    first_response_delay_bucket is not None
                    and first_hour_primary_posts is None
                ),
                "observed_share_shift": _observed_share_shift(
                    phases,
                    event["primary_topic_id"],
                    event["anchor_resolution"],
                ),
                "phases": phases,
            }
        )

    posts_identity = _file_identity(post_source)
    comments_identity = _file_identity(comment_source)
    engine_identity = _file_identity(Path(__file__))
    data_snapshot_payload = {
        "posts": posts_identity,
        "comments": comments_identity,
    }
    dataset_snapshot_id = _canonical_hash(data_snapshot_payload)
    analysis_snapshot_payload = {
        "dataset_snapshot_id": dataset_snapshot_id,
        "definitions": definitions["definition_fingerprint"],
        "engine": engine_identity,
    }
    benchmark = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_id": dataset_snapshot_id,
        "analysis_snapshot_id": _canonical_hash(analysis_snapshot_payload),
        "definition_fingerprint": definitions["definition_fingerprint"],
        "engine_evidence": {
            **engine_identity,
            "note": "执行引擎源文件指纹；不持久化源文件路径。",
        },
        "input_evidence": {
            "posts_csv": posts_identity,
            "comments_csv": comments_identity,
            "note": "文件指纹绑定聚合基准；不持久化原始输入路径。",
        },
        "data_audit": {
            "post_rows_scanned": rows_scanned,
            "valid_unique_post_rows": valid_post_rows,
            "invalid_post_dates": invalid_post_dates,
            "empty_post_codes": empty_post_codes,
            "duplicate_post_code_rows_excluded": duplicate_post_code_rows,
            "posts_in_any_declared_event_window": posts_in_event_windows,
            "comment_rows_scanned": comment_rows_scanned,
            "linked_comment_rows_to_declared_topics": linked_comment_rows,
            "known_gap_dates": sorted(DEFAULT_GAPS),
            "event_window_known_gap_dates": sorted(event_window_gap_dates),
            "event_windows_intersect_known_gaps": bool(event_window_gap_dates),
        },
        "methodology": {
            "design": "外部锚定的四阶段离线历史复演",
            "simulation_time": "合成阶段索引；不等待墙上时间，也不接入实时数据",
            "post_outcome": "各阶段预声明议题代理帖占全部有效帖的份额",
            "comment_use": "评论仅作最终关注度聚合；不用于响应时延或传播时间",
            "comment_regime_boundary": (
                "两个事件分别标注各自采集制度；不跨3月与6月事件比较评论关注度。"
            ),
            "privacy": "仅输出聚合结果；不含原文、帖子编号、评论编号或源路径",
        },
        "events": event_results,
    }
    benchmark["benchmark_fingerprint"] = _canonical_hash(
        {
            key: value
            for key, value in benchmark.items()
            if key not in {"generated_at_utc", "benchmark_fingerprint"}
        }
    )
    _assert_aggregate_only_output(benchmark, "benchmark")
    return benchmark


def _binomial(rng: random.Random, trials: int, probability: float) -> int:
    probability = min(1.0, max(0.0, probability))
    return sum(rng.random() < probability for _ in range(trials))


def _allocate_roles(
    rng: random.Random, total: int, roles: dict[str, dict[str, float]]
) -> dict[str, int]:
    remaining = total
    remaining_weight = 1.0
    allocation: dict[str, int] = {}
    # Canonical ordering makes the seeded draw stream independent of JSON key order.
    role_items = sorted(roles.items())
    for role_id, role in role_items[:-1]:
        conditional_probability = role["weight"] / remaining_weight
        assigned = _binomial(rng, remaining, conditional_probability)
        allocation[role_id] = assigned
        remaining -= assigned
        remaining_weight -= role["weight"]
    allocation[role_items[-1][0]] = remaining
    return allocation


def _simulate_topic_count(
    rng: random.Random,
    role_allocation: dict[str, int],
    roles: dict[str, dict[str, float]],
    target_rate: float,
) -> tuple[int, dict[str, int]]:
    total_trials = sum(role_allocation.values())
    weighted_multiplier = (
        sum(
            role_allocation[role_id] * roles[role_id]["response_multiplier"]
            for role_id in role_allocation
        )
        / total_trials
        if total_trials
        else 0.0
    )
    base_rate = target_rate / weighted_multiplier if weighted_multiplier else 0.0
    by_role = {}
    for role_id, trials in role_allocation.items():
        probability = min(1.0, base_rate * roles[role_id]["response_multiplier"])
        by_role[role_id] = _binomial(rng, trials, probability)
    return sum(by_role.values()), by_role


def _phase_lookup(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {phase["phase_id"]: phase for phase in event["phases"]}


def _fidelity_grade(metrics: dict[str, Any]) -> str:
    """Apply the versioned, pre-fixed engineering fit thresholds."""

    metric_names = tuple(GRADE_THRESHOLDS["strong"])
    values = [metrics.get(name) for name in metric_names]
    if any(
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return "insufficient_data"
    for grade in ("strong", "moderate"):
        if all(
            float(metrics[name]) <= GRADE_THRESHOLDS[grade][name]
            for name in metric_names
        ):
            return grade
    return "weak"


def _metric_grades(metrics: dict[str, Any]) -> dict[str, str]:
    grades = {}
    for metric_name in GRADE_THRESHOLDS["strong"]:
        value = metrics.get(metric_name)
        if (
            value is None
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            grades[metric_name] = "insufficient_data"
        elif value <= GRADE_THRESHOLDS["strong"][metric_name]:
            grades[metric_name] = "strong"
        elif value <= GRADE_THRESHOLDS["moderate"][metric_name]:
            grades[metric_name] = "moderate"
        else:
            grades[metric_name] = "weak"
    return grades


def _weakest_grade(grades: list[str]) -> str:
    if not grades:
        return "insufficient_data"
    return max(grades, key=lambda grade: GRADE_RANK[grade])


SENSITIVE_OUTPUT_KEYS = {
    "content",
    "text",
    "post_code",
    "comment_code",
    "comment_id",
    "author",
    "nickname",
    "contact",
    "input_path",
    "source_path",
}


def _assert_aggregate_only_output(
    value: Any,
    label: str,
    *,
    forbidden_strings: tuple[str, ...] = (),
) -> None:
    """Fail closed on raw-row fields, paths, and non-finite numbers."""

    def visit(item: Any, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in SENSITIVE_OUTPUT_KEYS:
                    raise ShockReplayError(
                        f"{label} contains sensitive field at {location}.{key}"
                    )
                visit(child, f"{location}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise ShockReplayError(f"{label} contains a non-finite value at {location}")
        if isinstance(item, str):
            if re.match(r"^[A-Za-z]:[\\/]", item) or item.startswith("/"):
                raise ShockReplayError(
                    f"{label} contains an absolute source path at {location}"
                )
            for forbidden in forbidden_strings:
                if forbidden and forbidden in item:
                    raise ShockReplayError(
                        f"{label} contains forbidden source material at {location}"
                    )

    visit(value, "$")


def _assert_safe_report(
    document: str, *, forbidden_strings: tuple[str, ...] = ()
) -> None:
    if re.search(r"(?<![A-Za-z])(nan|infinity)(?![A-Za-z])", document, re.I):
        raise ShockReplayError("HTML report contains a non-finite value")
    for forbidden in forbidden_strings:
        if forbidden and forbidden in document:
            raise ShockReplayError("HTML report contains forbidden source material")


def run_shock_replay(
    benchmark: dict[str, Any],
    definitions: dict[str, Any],
    *,
    repetitions: int = 100,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Replay both shocks through one role-mixture stage engine."""

    if benchmark.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ShockReplayError("unsupported benchmark schema version")
    if benchmark.get("definition_fingerprint") != definitions.get(
        "definition_fingerprint"
    ):
        raise ShockReplayError("benchmark and definitions fingerprints do not match")
    if not isinstance(repetitions, int) or repetitions < 50:
        raise ShockReplayError("repetitions must be an integer of at least 50")
    if benchmark.get("data_audit", {}).get("event_windows_intersect_known_gaps"):
        raise ShockReplayError("benchmark event windows intersect known crawl gaps")
    expected_benchmark_fingerprint = _canonical_hash(
        {
            key: value
            for key, value in benchmark.items()
            if key not in {"generated_at_utc", "benchmark_fingerprint"}
        }
    )
    if benchmark.get("benchmark_fingerprint") != expected_benchmark_fingerprint:
        raise ShockReplayError("benchmark aggregate fingerprint does not match")
    current_engine_identity = _file_identity(Path(__file__))
    recorded_engine = benchmark.get("engine_evidence", {})
    if any(
        recorded_engine.get(key) != current_engine_identity[key]
        for key in ("sha256", "bytes")
    ):
        raise ShockReplayError("benchmark and replay engine fingerprints do not match")

    definitions_by_id = {
        event["scenario_id"]: event for event in definitions["events"]
    }
    benchmark_ids = {event.get("scenario_id") for event in benchmark.get("events", [])}
    if benchmark_ids != set(definitions_by_id):
        raise ShockReplayError("benchmark and definitions scenario sets do not match")
    benchmark_by_id = {
        event["scenario_id"]: event for event in benchmark["events"]
    }
    for scenario_id, definition in definitions_by_id.items():
        event = benchmark_by_id[scenario_id]
        if event.get("primary_topic_id") != definition["primary_topic_id"]:
            raise ShockReplayError(f"{scenario_id} primary topic does not match")
        phase_ids = [phase.get("phase_id") for phase in event.get("phases", [])]
        if phase_ids != [phase["phase_id"] for phase in definition["phases"]]:
            raise ShockReplayError(f"{scenario_id} phase schema does not match")
        expected_topics = {
            topic["topic_id"] for topic in definition["topics"]
        }
        if any(
            set(phase.get("topics", {})) != expected_topics
            for phase in event["phases"]
        ):
            raise ShockReplayError(f"{scenario_id} topic schema does not match")
    predictions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    role_predictions: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    replicate_summaries = []
    structural_constraint_violations = 0

    for repetition in range(repetitions):
        repetition_seed = seed + repetition * 1009
        rng = random.Random(repetition_seed)
        per_event_mae = {}
        for definition in definitions["events"]:
            event = benchmark_by_id[definition["scenario_id"]]
            roles = definitions["role_models"][definition["shock_type"]]["roles"]
            absolute_errors = []
            phases_by_id = _phase_lookup(event)
            for phase_definition in definition["phases"]:
                phase = phases_by_id[phase_definition["phase_id"]]
                exposure = int(phase["all_posts_exposure"])
                role_allocation = _allocate_roles(rng, exposure, roles)
                simulated_topic_role_counts: dict[str, dict[str, int]] = {}
                for topic_definition in definition["topics"]:
                    topic_id = topic_definition["topic_id"]
                    observed = phase["topics"][topic_id]
                    required_topic = topic_definition.get("requires_topic")
                    excluded_topics = topic_definition.get("exclude_topics", [])
                    if required_topic:
                        eligible_allocation = dict(
                            simulated_topic_role_counts[required_topic]
                        )
                        observed_eligible = phase["topics"][required_topic]["posts"]
                    elif excluded_topics:
                        excluded_topic = excluded_topics[0]
                        eligible_allocation = {
                            role_id: max(
                                0,
                                role_allocation[role_id]
                                - simulated_topic_role_counts[excluded_topic][role_id],
                            )
                            for role_id in role_allocation
                        }
                        observed_eligible = (
                            exposure - phase["topics"][excluded_topic]["posts"]
                        )
                    else:
                        eligible_allocation = role_allocation
                        observed_eligible = exposure
                    if observed["posts"] > observed_eligible:
                        raise ShockReplayError(
                            f"{event['scenario_id']} {phase['phase_id']} {topic_id} "
                            "violates its declared topic constraint"
                        )
                    # Jeffreys smoothing prevents a single zero cell from becoming
                    # a structurally impossible action in repeated simulation.
                    target_rate = (observed["posts"] + 0.5) / (
                        observed_eligible + 1.0
                    )
                    count, by_role = _simulate_topic_count(
                        rng, eligible_allocation, roles, target_rate
                    )
                    simulated_topic_role_counts[topic_id] = by_role
                    if required_topic and count > sum(
                        simulated_topic_role_counts[required_topic].values()
                    ):
                        structural_constraint_violations += 1
                    if excluded_topics and count + sum(
                        simulated_topic_role_counts[excluded_topics[0]].values()
                    ) > exposure:
                        structural_constraint_violations += 1
                    key = (event["scenario_id"], phase["phase_id"], topic_id)
                    predictions[key].append(count)
                    for role_id, role_count in by_role.items():
                        role_predictions[(*key, role_id)].append(role_count)
                    simulated_rate = count / exposure if exposure else 0.0
                    absolute_errors.append(
                        abs(simulated_rate - observed["share_of_all_posts"])
                    )
            per_event_mae[event["scenario_id"]] = (
                sum(absolute_errors) / len(absolute_errors)
                if absolute_errors
                else 0.0
            )
        replicate_summaries.append(
            {
                "repetition": repetition,
                "seed": repetition_seed,
                "topic_rate_mae_by_event": per_event_mae,
            }
        )

    if structural_constraint_violations:
        raise ShockReplayError("internal topic constraints were violated")

    event_results = []
    for definition in definitions["events"]:
        event = benchmark_by_id[definition["scenario_id"]]
        phase_results = []
        all_cell_errors = []
        primary_errors = []
        covered_cells = 0
        total_cells = 0
        observed_primary_rates = {}
        predicted_primary_rates = {}
        observed_primary_counts: dict[str, float] = {}
        predicted_primary_counts: dict[str, float] = {}
        topic_observed_totals: Counter = Counter()
        topic_predicted_totals: Counter = Counter()
        phases_by_id = _phase_lookup(event)
        for phase_definition in definition["phases"]:
            phase = phases_by_id[phase_definition["phase_id"]]
            exposure = int(phase["all_posts_exposure"])
            topic_results = {}
            for topic_definition in definition["topics"]:
                topic_id = topic_definition["topic_id"]
                observed = phase["topics"][topic_id]
                key = (event["scenario_id"], phase["phase_id"], topic_id)
                values = predictions[key]
                p05 = _quantile(values, 0.05)
                p50 = _quantile(values, 0.50)
                p95 = _quantile(values, 0.95)
                median_rate = p50 / exposure if exposure else 0.0
                error = abs(median_rate - observed["share_of_all_posts"])
                all_cell_errors.append(error)
                total_cells += 1
                covered = p05 <= observed["posts"] <= p95
                covered_cells += int(covered)
                topic_observed_totals[topic_id] += observed["posts"]
                topic_predicted_totals[topic_id] += p50
                if topic_id == event["primary_topic_id"]:
                    primary_errors.append(error)
                    observed_primary_rates[phase["phase_id"]] = observed[
                        "share_of_all_posts"
                    ]
                    predicted_primary_rates[phase["phase_id"]] = median_rate
                    observed_primary_counts[phase["phase_id"]] = float(
                        observed["posts"]
                    )
                    predicted_primary_counts[phase["phase_id"]] = p50
                by_role = {}
                for role_id in sorted(
                    definitions["role_models"][definition["shock_type"]]["roles"]
                ):
                    role_values = role_predictions[(*key, role_id)]
                    by_role[role_id] = {
                        "p05": _quantile(role_values, 0.05),
                        "p50": _quantile(role_values, 0.50),
                        "p95": _quantile(role_values, 0.95),
                    }
                topic_results[topic_id] = {
                    "label": observed["label"],
                    "simulation_constraint": (
                        {
                            "kind": "conditional_on_topic",
                            "topic_id": topic_definition["requires_topic"],
                        }
                        if topic_definition.get("requires_topic")
                        else {
                            "kind": "excludes_topic",
                            "topic_id": topic_definition["exclude_topics"][0],
                        }
                        if topic_definition.get("exclude_topics")
                        else {"kind": "marginal"}
                    ),
                    "observed_posts": observed["posts"],
                    "observed_share": observed["share_of_all_posts"],
                    "simulated_posts": {"p05": p05, "p50": p50, "p95": p95},
                    "simulated_median_share": median_rate,
                    "absolute_share_error": error,
                    "observed_count_covered_by_p05_p95": covered,
                    "synthetic_role_breakdown": by_role,
                }
            phase_results.append(
                {
                    "phase_id": phase["phase_id"],
                    "label": phase["label"],
                    "all_posts_exposure_held_fixed": exposure,
                    "topics": topic_results,
                }
            )

        observed_primary_total = sum(observed_primary_counts.values())
        predicted_primary_total = sum(predicted_primary_counts.values())
        primary_wape = None
        if observed_primary_total:
            primary_wape = sum(
                abs(
                    predicted_primary_counts[phase_id]
                    - observed_primary_counts[phase_id]
                )
                for phase_id in EXPECTED_PHASE_IDS
            ) / observed_primary_total

        topic_incidence_composition_tvd = None
        observed_topic_total = sum(topic_observed_totals.values())
        predicted_topic_total = sum(topic_predicted_totals.values())
        if observed_topic_total and predicted_topic_total:
            topic_incidence_composition_tvd = 0.5 * sum(
                abs(
                    topic_observed_totals[topic_id] / observed_topic_total
                    - topic_predicted_totals[topic_id] / predicted_topic_total
                )
                for topic_id in topic_observed_totals
            )

        observed_peak_phases: list[str] = []
        predicted_peak_phases: list[str] = []
        peak_stage_distance = None
        tail_share_error = None
        if observed_primary_total and predicted_primary_total:
            observed_peak_rate = max(observed_primary_rates.values())
            predicted_peak_rate = max(predicted_primary_rates.values())
            observed_peak_phases = [
                phase_id
                for phase_id in EXPECTED_PHASE_IDS
                if math.isclose(
                    observed_primary_rates[phase_id],
                    observed_peak_rate,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ]
            predicted_peak_phases = [
                phase_id
                for phase_id in EXPECTED_PHASE_IDS
                if math.isclose(
                    predicted_primary_rates[phase_id],
                    predicted_peak_rate,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ]
            peak_stage_distance = min(
                abs(
                    EXPECTED_PHASE_IDS.index(observed_phase)
                    - EXPECTED_PHASE_IDS.index(predicted_phase)
                )
                for observed_phase in observed_peak_phases
                for predicted_phase in predicted_peak_phases
            )
            observed_tail_share = (
                observed_primary_counts["decay"] / observed_primary_total
            )
            predicted_tail_share = (
                predicted_primary_counts["decay"] / predicted_primary_total
            )
            tail_share_error = abs(predicted_tail_share - observed_tail_share)

        fixed_fit_metrics = {
            "primary_stage_scale_wape": primary_wape,
            "topic_incidence_composition_tvd": topic_incidence_composition_tvd,
            "primary_peak_stage_distance": peak_stage_distance,
            "primary_tail_share_error": tail_share_error,
        }
        event_grade = _fidelity_grade(fixed_fit_metrics)
        event_results.append(
            {
                "scenario_id": event["scenario_id"],
                "title": event["title"],
                "shock_type": event["shock_type"],
                "shock_origin": event["shock_origin"],
                "comment_collection_regime": event[
                    "comment_collection_regime"
                ],
                "identification": event["identification"],
                "role_model": {
                    "source_type": "assumption",
                    "roles": definitions["role_models"][event["shock_type"]][
                        "roles"
                    ],
                },
                "phases": phase_results,
                "fidelity": {
                    "topic_rate_mae": sum(all_cell_errors) / len(all_cell_errors),
                    "primary_topic_rate_mae": sum(primary_errors) / len(primary_errors),
                    "observed_primary_peak_phases": observed_peak_phases,
                    "simulated_primary_peak_phases": predicted_peak_phases,
                    "primary_peak_phase_match": peak_stage_distance == 0,
                    "observed_count_interval_coverage_90": covered_cells / total_cells,
                    **fixed_fit_metrics,
                    "reconstruction_metric_grades": _metric_grades(
                        fixed_fit_metrics
                    ),
                    "reconstruction_fit_grade": event_grade,
                    "reconstruction_grade_rule": "weakest_versioned_metric",
                    "grade_scope": "in_sample_monte_carlo_reconstruction_only",
                    "external_validation_status": "not_evaluated",
                    "causal_status": "not_identified",
                    "role_mechanism_validation_status": "not_evaluated",
                    "evidence_stage": "retrospective_in_sample_replay",
                    "interpretation": (
                        "在观察到的全部帖子暴露量下进行样本内历史重构；不是样本外"
                        "预测，也不是任何治理干预效果的估计。"
                    ),
                },
            }
        )

    overall_maes = [event["fidelity"]["topic_rate_mae"] for event in event_results]
    primary_maes = [
        event["fidelity"]["primary_topic_rate_mae"] for event in event_results
    ]
    event_grades = [
        event["fidelity"]["reconstruction_fit_grade"]
        for event in event_results
    ]
    replay = {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_id": benchmark["dataset_snapshot_id"],
        "analysis_snapshot_id": benchmark["analysis_snapshot_id"],
        "benchmark_fingerprint": benchmark["benchmark_fingerprint"],
        "definition_fingerprint": benchmark["definition_fingerprint"],
        "seed": seed,
        "repetitions": repetitions,
        "engine": {
            "name": "offline_discrete_role_mixture_replay",
            "simulation_clock": "four synthetic stages per event",
            "wall_clock_dependency": False,
            "live_data_dependency": False,
            "llm_dependency": False,
            "calibration": "empirical phase topic rates with Jeffreys smoothing",
            "evidence_stage": "retrospective_in_sample_replay",
            "exposure_control": "observed all-post count held fixed per stage",
            "topic_constraints": (
                "requires_topic is sampled conditionally within the simulated parent; "
                "exclude_topics is sampled only from the remaining role exposure"
            ),
            "structural_constraint_violations": structural_constraint_violations,
        },
        "fidelity_methodology": {
            "primary_stage_scale_wape": (
                "sum over stages of abs(simulated p50 primary count - observed "
                "primary count), divided by total observed primary count"
            ),
            "topic_incidence_composition_tvd": (
                "one half of the L1 distance between observed and simulated-p50 "
                "distributions of declared topic-hit counts aggregated over stages; "
                "topic hits may overlap and are not population shares"
            ),
            "primary_peak_stage_distance": (
                "minimum index distance between any observed and simulated-p50 peak "
                "stage, where peaks use primary-topic share of all stage posts and "
                "all exact ties are retained"
            ),
            "primary_tail_share_error": (
                "absolute difference in decay-stage share of the four-stage primary "
                "topic count total"
            ),
            "simulation_quantiles": (
                "linear interpolation at (n-1)*p over seeded replicate counts"
            ),
            "interval_interpretation": (
                "conditional Monte Carlo quantiles under observed exposure and "
                "same-sample empirical rates; not prediction intervals or external coverage"
            ),
        },
        "events": event_results,
        "overall_fidelity": {
            "mean_topic_rate_mae": sum(overall_maes) / len(overall_maes),
            "mean_primary_topic_rate_mae": sum(primary_maes) / len(primary_maes),
            "all_primary_peak_phases_match": all(
                event["fidelity"]["primary_peak_phase_match"]
                for event in event_results
            ),
            "weakest_reconstruction_fit_grade": _weakest_grade(event_grades),
            "event_reconstruction_fit_grades": {
                event["scenario_id"]: event["fidelity"][
                    "reconstruction_fit_grade"
                ]
                for event in event_results
            },
            "grade_thresholds": GRADE_THRESHOLDS,
            "evidence_stage": "retrospective_in_sample_replay",
            "external_validation_status": "not_evaluated",
            "causal_status": "not_identified",
            "role_mechanism_validation_status": "not_evaluated",
            "policy_effect_status": "not_empirically_validated",
        },
        "replicate_summaries": replicate_summaries,
        "claim_boundary": {
            "supported": (
                "同一个离线阶段引擎在经验校准后，可复演两个外部锚定冲击的"
                "聚合需求曲线。"
            ),
            "not_supported": (
                "实时预测、个体行为、持续社交网络，以及FAQ、道歉、分流、容量"
                "或其他干预的因果效果。"
            ),
        },
    }
    _assert_aggregate_only_output(replay, "replay")
    return replay


def write_json(value: dict[str, Any], output_path: str | Path) -> str:
    _assert_aggregate_only_output(value, "JSON output")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return str(target)


def _render_shock_replay_report(
    benchmark: dict[str, Any], replay: dict[str, Any]
) -> str:
    """Render one self-contained aggregate-only HTML report in memory."""

    cards = []
    benchmark_by_id = {event["scenario_id"]: event for event in benchmark["events"]}
    origin_labels = {
        "governance_action": "治理行动信息冲击",
        "exogenous_event": "外生事件冲击",
    }
    regime_labels = {
        "pre_2026_04_crawl_break": "2026年4月采集断点前",
        "post_2026_04_crawl_break": "2026年4月采集断点后",
    }
    resolution_labels = {"day": "日级", "minute": "分钟级", "second": "秒级"}
    for event in replay["events"]:
        source_event = benchmark_by_id[event["scenario_id"]]
        primary_topic_id = source_event["primary_topic_id"]
        rows = []
        for phase in event["phases"]:
            topic = phase["topics"][primary_topic_id]
            rows.append(
                "<tr>"
                f"<td>{html.escape(phase['label'])}</td>"
                f"<td>{phase['all_posts_exposure_held_fixed']}</td>"
                f"<td>{topic['observed_posts']} ({topic['observed_share'] * 100:.2f}%)</td>"
                f"<td>{topic['simulated_posts']['p50']:.1f} "
                f"[{topic['simulated_posts']['p05']:.1f}, {topic['simulated_posts']['p95']:.1f}]</td>"
                f"<td>{topic['absolute_share_error'] * 100:.2f} pp</td>"
                "</tr>"
            )
        delay_bucket = source_event["first_related_post_delay_bucket"]
        response_line = (
            "首条命中代理规则的帖子出现在外部锚点后"
            f"{delay_bucket['lower_inclusive']}—{delay_bucket['upper_exclusive']}分钟区间内。"
            if delay_bucket is not None
            else "公开来源仅支持日级锚点，报告不伪造小时级起点。"
        )
        if source_event["first_hour_primary_posts"] is not None:
            response_line += (
                f" 锚点后首小时共有 {source_event['first_hour_primary_posts']} 条"
                "主议题代理帖。"
            )
        share_shift = source_event["observed_share_shift"]
        if share_shift["status"] == "estimable":
            multiplier = share_shift["burst_to_baseline_share_multiplier"]
            burst_label = (
                "事件日" if source_event["anchor_resolution"] == "day" else "爆发期"
            )
            shift_line = (
                f"论坛主议题份额由基线 {share_shift['baseline_primary_share'] * 100:.2f}% "
                f"升至{burst_label} {share_shift['burst_primary_share'] * 100:.2f}%"
                f"，增加 {share_shift['burst_minus_baseline_share'] * 100:.2f} 个百分点"
            )
            if multiplier is not None:
                shift_line += f"，约为 {multiplier:.1f} 倍"
            interval = share_shift[
                "naive_independent_post_95pct_multiplier_interval"
            ]
            if interval is not None:
                shift_line += (
                    "（把帖子朴素视为独立时的探索性95%近似区间 "
                    f"{interval['lower']:.1f}—{interval['upper']:.1f} 倍，"
                    "不作正式推断）"
                )
            if source_event["anchor_resolution"] == "day":
                shift_line += "；这只是事件日关联，全天不能解释为事件发生之后。"
            else:
                shift_line += "；这是外部定时锚点下的描述性突变。"
            shift_line += (
                f" 基线仅有 {share_shift['baseline_primary_posts']} 条代理帖，"
                "倍数估计不稳定，且不代表治理干预效果。"
            )
        else:
            shift_line = "基线或爆发期暴露量不足，无法估计份额突变。"
        fidelity = event["fidelity"]
        format_ratio = lambda value: (
            "数据不足" if value is None else f"{value * 100:.1f}%"
        )
        anchor_source = source_event["anchor_source"]
        anchor_source_link = (
            f'<a href="{html.escape(anchor_source["url"], quote=True)}" '
            'target="_blank" rel="noreferrer">'
            f'{html.escape(anchor_source["publisher"])}</a>'
        )
        cards.append(
            f"""
            <section class="card">
              <div class="eyebrow">{html.escape(origin_labels.get(event['shock_origin'], event['shock_origin']))}</div>
              <h2>{html.escape(event['title'])}</h2>
              <p class="boundary">冲击口径：{html.escape(source_event['shock_origin_definition'])}</p>
              <p class="boundary">外部锚点：<code>{html.escape(source_event['anchor'])}</code>（{html.escape(source_event['anchor_timezone'])}，{html.escape(resolution_labels.get(source_event['anchor_resolution'], source_event['anchor_resolution']))}）；来源：{anchor_source_link}；议题规则版本：<code>{html.escape(source_event['topic_rules_version'])}</code>。</p>
              <p>{html.escape(response_line)}</p>
              <p>{html.escape(shift_line)}</p>
              <table>
                <thead><tr><th>阶段</th><th>全部帖子暴露量</th><th>观测主议题代理</th><th>模拟 p50 [p05, p95]</th><th>占比误差</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
              <div class="metrics">
                <span>样本内重构等级：{html.escape(fidelity['reconstruction_fit_grade'])}</span>
                <span>阶段规模 WAPE：{format_ratio(fidelity['primary_stage_scale_wape'])}</span>
                <span>议题命中次数结构 TVD：{format_ratio(fidelity['topic_incidence_composition_tvd'])}</span>
                <span>峰值阶段距离：{fidelity['primary_peak_stage_distance'] if fidelity['primary_peak_stage_distance'] is not None else '数据不足'}</span>
                <span>尾部份额误差：{format_ratio(fidelity['primary_tail_share_error'])}</span>
              </div>
              <p class="boundary">评论采集制度：{html.escape(regime_labels.get(event['comment_collection_regime'], event['comment_collection_regime']))}；评论仅作事件内最终关注度描述，不跨事件比较。</p>
              <p class="boundary">识别边界：{html.escape(event['identification']['claim_boundary'])}</p>
            </section>
            """
        )
    audit_items = [
        ("帖子表 SHA-256", benchmark["input_evidence"]["posts_csv"]["sha256"]),
        ("评论表 SHA-256", benchmark["input_evidence"]["comments_csv"]["sha256"]),
        ("事件定义指纹", benchmark["definition_fingerprint"]),
        ("聚合 benchmark 指纹", benchmark["benchmark_fingerprint"]),
        ("执行引擎指纹", benchmark["engine_evidence"]["sha256"]),
        ("纯数据快照", benchmark["dataset_snapshot_id"]),
        ("数据＋定义＋引擎分析快照", benchmark["analysis_snapshot_id"]),
    ]
    audit_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td><code>{html.escape(value)}</code></td></tr>"
        for label, value in audit_items
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CampusPulse 双冲击离线复演</title>
<style>
:root{{--ink:#152238;--muted:#617089;--line:#dfe6ef;--panel:#fff;--bg:#f4f7fb;--accent:#f26b38;--blue:#2f6fed}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:auto;padding:40px 20px 64px}} h1{{font-size:34px;margin:0 0 8px}} h2{{font-size:22px;margin:4px 0 8px}}
.lead{{color:var(--muted);max-width:850px}} .notice{{background:#fff4ed;border-left:4px solid var(--accent);padding:14px 16px;margin:24px 0;border-radius:8px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 8px 25px rgba(30,55,90,.05);overflow:auto}}
.eyebrow{{color:var(--blue);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em}}
table{{width:100%;border-collapse:collapse;min-width:760px;margin-top:14px}} th,td{{text-align:left;border-bottom:1px solid var(--line);padding:10px 8px}} th{{color:var(--muted);font-size:13px}}
.metrics{{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}} .metrics span{{background:#edf3ff;border-radius:999px;padding:6px 10px}}
.boundary{{color:var(--muted);font-size:13px}} code{{background:#eef2f7;padding:2px 6px;border-radius:5px;overflow-wrap:anywhere}} a{{color:var(--blue)}}
</style></head><body><main>
<div class="eyebrow">historical replay · aggregate only</div>
<h1>双冲击离线历史复演</h1>
<p class="lead">真实时间戳只负责切分历史阶段；程序用四个合成步骤立即完成复演，不接实时流、不等待真实时间，也不调用大模型。</p>
<div class="notice"><strong>证据状态：</strong><code>retrospective_in_sample_replay</code>。这是样本内蒙特卡洛重构拟合，不是模型外部验证或样本外预测，也不验证角色机制，不识别 FAQ、回应、道歉、分流、容量、成本或福利效果。双事件样本内重构采用较弱等级 <strong>{html.escape(replay['overall_fidelity']['weakest_reconstruction_fit_grade'])}</strong>；数据快照 <code>{html.escape(replay['dataset_snapshot_id'][:16])}</code>，分析快照 <code>{html.escape(replay['analysis_snapshot_id'][:16])}</code>。</div>
{''.join(cards)}
<section class="card"><h2>可复核证据链</h2><p class="boundary">两张输入表、事件定义、聚合基准与执行引擎分别固定；这里只展示指纹，不展示原始路径、正文或逐条标识。</p><table><tbody>{audit_rows}</tbody></table></section>
<section class="card"><h2>共同引擎边界</h2><p>{html.escape(replay['claim_boundary']['supported'])}</p><p class="boundary">不支持：{html.escape(replay['claim_boundary']['not_supported'])}</p></section>
</main></body></html>"""
    return document


def generate_shock_replay_report(
    benchmark: dict[str, Any], replay: dict[str, Any], output_path: str | Path
) -> str:
    """Generate one self-contained aggregate-only HTML report."""

    document = _render_shock_replay_report(benchmark, replay)
    _assert_safe_report(document)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8", newline="\n")
    return str(target)


def run_shock_replay_pipeline(
    posts_path: str | Path,
    comments_path: str | Path,
    output_dir: str | Path,
    *,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    repetitions: int = 100,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Build the benchmark, run replay, and emit JSON plus standalone HTML."""

    definitions = load_shock_definitions(definitions_path)
    benchmark = build_shock_benchmark(posts_path, comments_path, definitions_path)
    replay = run_shock_replay(
        benchmark, definitions, repetitions=repetitions, seed=seed
    )
    forbidden_paths = tuple(
        str(Path(path).resolve())
        for path in (posts_path, comments_path, definitions_path)
    )
    _assert_aggregate_only_output(
        benchmark, "benchmark", forbidden_strings=forbidden_paths
    )
    _assert_aggregate_only_output(
        replay, "replay", forbidden_strings=forbidden_paths
    )
    report_document = _render_shock_replay_report(benchmark, replay)
    _assert_safe_report(report_document, forbidden_strings=forbidden_paths)

    # Validation above is intentionally completed before the first output write.
    root = Path(output_dir)
    benchmark_path = write_json(benchmark, root / "shock_benchmark.json")
    replay_path = write_json(replay, root / "shock_replay.json")
    report_target = root / "shock_replay.html"
    report_target.write_text(report_document, encoding="utf-8", newline="\n")
    report_path = str(report_target)
    return {
        "benchmark": benchmark,
        "replay": replay,
        "paths": {
            "benchmark_json": benchmark_path,
            "replay_json": replay_path,
            "report_html": report_path,
        },
    }
