"""Privacy-preserving streaming calibration for the forum CSV exports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CATEGORY_PATTERNS = {
    "academic_and_exam": re.compile(r"考试|期末|成绩|挂科|保研|考研|论文|课程|选课|作业|学习|答辩|毕设|毕业"),
    "career_development": re.compile(r"实习|秋招|春招|就业|求职|工作|offer|面试|考公|职业|简历", re.I),
    "emotional_support": re.compile(r"抑郁|焦虑|崩溃|难受|孤独|失眠|压力|想死|自杀|自残|轻生|活不下去|心理咨询|情绪"),
    "relationships": re.compile(r"室友|恋爱|分手|前任|表白|朋友|社交|吵架|冷战|对象|喜欢的人"),
    "campus_services": re.compile(r"食堂|宿舍|图书馆|校园网|空调|暖气|校医院|教务|澡堂|快递|设施|噪音|洗衣机"),
    "safety_and_rights": re.compile(r"诈骗|被骗|骚扰|偷拍|丢失|偷|盗|泄露|隐私|性侵|安全|打架|暴力|举报"),
    "rumor_or_verification": re.compile(r"听说|据说|真的吗|真的假的|求证|谣言|网传|消息可靠吗|有人知道吗"),
}

DEFAULT_GAPS = {
    "2026-03-04",
    "2026-04-03",
    "2026-04-04",
    "2026-04-20",
    "2026-06-25",
}


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _file_identity(path: str | Path) -> dict[str, int | str]:
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


def _top_counts(counter: Counter, limit: int = 5) -> list[dict[str, int | str]]:
    return [
        {"date": key, "posts": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def calibrate_csvs(
    posts_path: str | Path,
    comments_path: str | Path,
    output_path: str | Path,
    start: str = "2025-12-01",
    end: str = "2026-07-21",
    platform_marker: str = "NP",
) -> dict:
    """Produce aggregate evidence only; never persist source text or identifiers."""

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    post_categories: dict[str, tuple[str, ...]] = {}
    hourly = Counter()
    monthly = Counter()
    daily = Counter()
    category_daily: dict[str, Counter] = defaultdict(Counter)
    category_posts = Counter()
    excluded_gap_posts = 0
    valid_posts = 0

    with Path(posts_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("post_code", "")
            date = _parse_date(row.get("real_date", ""))
            if platform_marker not in code or date is None or not start_dt <= date < end_dt:
                continue
            if date.date().isoformat() in DEFAULT_GAPS:
                excluded_gap_posts += 1
                continue
            content = row.get("content", "")
            categories = tuple(
                category for category, pattern in CATEGORY_PATTERNS.items() if pattern.search(content)
            )
            post_categories[code] = categories
            valid_posts += 1
            hourly[str(date.hour)] += 1
            monthly[date.strftime("%Y-%m")] += 1
            date_key = date.date().isoformat()
            daily[date_key] += 1
            for category in categories:
                category_daily[category][date_key] += 1
            category_posts.update(categories)

    comments_per_post = Counter()
    linked_comments = 0
    with Path(comments_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("post_code", "")
            if code in post_categories:
                comments_per_post[code] += 1
                linked_comments += 1

    category_comment_counts = Counter()
    category_with_comment = Counter()
    category_positive_comments = Counter()
    for code, categories in post_categories.items():
        count = comments_per_post[code]
        for category in categories:
            category_comment_counts[category] += count
            if count > 0:
                category_with_comment[category] += 1
                category_positive_comments[category] += count

    category_metrics = {}
    for category in sorted(CATEGORY_PATTERNS):
        posts = category_posts[category]
        comments = category_comment_counts[category]
        with_comment = category_with_comment[category]
        category_metrics[category] = {
            "posts": posts,
            "linked_comments": comments,
            "with_comment": with_comment,
            "zero_comment_rate": (posts - with_comment) / posts if posts else 0.0,
            "mean_comments_per_post": comments / posts if posts else 0.0,
            "response_probability_proxy": with_comment / posts if posts else 0.0,
            "mean_comments_if_responded": (
                category_positive_comments[category] / with_comment
                if with_comment
                else 0.0
            ),
            "peak_dates": _top_counts(category_daily[category], limit=3),
        }

    comment_counts = [comments_per_post[code] for code in post_categories]
    positive_counts = [count for count in comment_counts if count > 0]

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_evidence": {
            "posts_csv": _file_identity(posts_path),
            "comments_csv": _file_identity(comments_path),
            "note": "File fingerprints identify the calibrated snapshot; input paths are not persisted.",
        },
        "scope": {
            "platform_marker": platform_marker,
            "start_inclusive": start,
            "end_exclusive": end,
            "excluded_known_gap_dates": sorted(DEFAULT_GAPS),
        },
        "privacy": "Aggregate output only; no post text, codes, comment IDs, hashes, or contacts are persisted.",
        "valid_posts": valid_posts,
        "linked_comments": linked_comments,
        "excluded_gap_posts": excluded_gap_posts,
        "engagement": {
            "zero_comment_posts": valid_posts - len(positive_counts),
            "zero_comment_rate": (
                (valid_posts - len(positive_counts)) / valid_posts
                if valid_posts
                else 0.0
            ),
            "mean_comments_all_posts": (
                linked_comments / valid_posts if valid_posts else 0.0
            ),
            "mean_comments_if_responded": (
                sum(positive_counts) / len(positive_counts)
                if positive_counts
                else 0.0
            ),
            "comment_count_quantiles": {
                "p25": _quantile(comment_counts, 0.25),
                "p50": _quantile(comment_counts, 0.50),
                "p75": _quantile(comment_counts, 0.75),
                "p90": _quantile(comment_counts, 0.90),
                "p95": _quantile(comment_counts, 0.95),
                "p99": _quantile(comment_counts, 0.99),
                "max": max(comment_counts, default=0),
            },
        },
        "monthly_posts": dict(sorted(monthly.items())),
        "hourly_posts": dict(sorted(hourly.items(), key=lambda item: int(item[0]))),
        "peak_dates": _top_counts(daily, limit=10),
        "category_metrics": category_metrics,
        "collection_regime": {
            "observed_break": "2026-04-04",
            "validation_train_window": "2026-05-01/2026-06-16",
            "validation_holdout_window": "2026-06-16/2026-07-16",
            "interpretation": "Do not mix pre-break and post-break response probabilities or treat the discontinuity as ordinary user behavior.",
        },
        "limitations": [
            "Categories are overlapping keyword proxies, not human-reviewed ground truth.",
            "Comment timestamps are not used because comment real_date is inherited from the post.",
            "Cumulative comments measure final thread attention, not response speed or unique responders.",
        ],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return result
