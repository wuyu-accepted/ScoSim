"""Held-out, auditable forum-expression rendering for accepted agent decisions.

This module deliberately keeps behavioural decisions and surface realisation
separate.  Raw forum text is streamed only to build aggregate style profiles and
ephemeral, redacted exemplars.  Shock-window text, linked replies, prompts, local
paths, and raw exemplars are never persisted in the release artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .llm_agents import (
    CONTACT_PATTERNS,
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMAgentError,
    PHASE_IDS,
    TokenBudget,
    VectorEngineClient,
    extract_json_object,
)
from .shock_replay import DEFAULT_DEFINITIONS, load_shock_definitions


PUBLIC_ACTIONS = {"comment", "post", "seek_help", "report_risk"}
EVENT_TERM_PATTERN = re.compile(
    r"通州|双校区|两校区|新校区|潞城|通运门|通勤|专线|校车|"
    r"张小龙|粉笔|混吃等死|活该.{0,4}找不到工作"
)
URL_PATTERN = re.compile(r"(?:https?://|www\.|xhslink\.|doubao\.)", re.I)
LONG_DIGIT_PATTERN = re.compile(r"(?<!\d)\d{7,}(?!\d)")
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]",
    flags=re.UNICODE,
)
REPETITION_PATTERN = re.compile(r"([!！?？。…哈啊呜哭])\1{1,}")
FORMAL_MARKER_PATTERN = re.compile(
    r"想确认|是否|能否|请|建议|渠道|安排|信息|目前|需要|明确|公开|"
    r"入口|节点|负责人|时限"
)
SLANG_MARKER_PATTERN = re.compile(
    r"咋|啊啊|扣[12]|老登|小登|破防|调理|难绷|串子|tm|fw|爆改|"
    r"啥|蹲|笑死|无语|补药|寄了|狠狠"
)
ALLOWED_REPLY_FUNCTIONS = {
    "agree",
    "challenge",
    "clarify",
    "ask_followup",
    "add_information",
    "share_experience",
    "emotional_support",
    "joke_or_meme",
    "signal_only",
    "not_a_reply",
}
ALLOWED_REGISTERS = {
    "fragment",
    "colloquial",
    "mixed",
    "standard",
}
SCENARIO_EXPRESSION_CONTEXT = {
    "tongzhou_governance_information_shock": (
        "治理信息刚出现时，学生通常围绕课程、住宿、通勤和组织运行提出具体问题；"
        "允许口语化担忧、反问和轻量玩笑，但不要把每个人都写成制度设计者。"
    ),
    "lecture_external_incident_shock": (
        "校外起源的争议正在论坛扩散；担忧或批评角色可以使用短促情绪句、反问、"
        "轻度讽刺和接话，不必强行保持分析报告式平衡，但不得新增辱骂原话或现场事实。"
    ),
}


class ForumExpressionError(ValueError):
    """Raised when the expression-layer evidence or output contract fails."""


@dataclass(frozen=True)
class HoldoutWindow:
    scenario_id: str
    start: datetime
    end: datetime

    def contains(self, value: datetime | None) -> bool:
        return value is not None and self.start <= value < self.end

    def public_view(self) -> dict[str, str]:
        return {
            "scenario_id": self.scenario_id,
            "start": self.start.isoformat(timespec="seconds"),
            "end": self.end.isoformat(timespec="seconds"),
        }


@dataclass
class _Aggregate:
    lengths: list[int] = field(default_factory=list)
    emoji: int = 0
    repetition: int = 0
    formal_marker: int = 0
    slang_marker: int = 0
    question: int = 0

    def add(self, text: str) -> None:
        self.lengths.append(len(text))
        self.emoji += bool(EMOJI_PATTERN.search(text))
        self.repetition += bool(REPETITION_PATTERN.search(text))
        self.formal_marker += bool(FORMAL_MARKER_PATTERN.search(text))
        self.slang_marker += bool(SLANG_MARKER_PATTERN.search(text))
        self.question += bool(re.search(r"[?？]", text))

    def public_view(self) -> dict[str, Any]:
        lengths = sorted(self.lengths)
        count = len(lengths)
        if not count:
            return {"count": 0}

        def quantile(fraction: float) -> int:
            index = min(count - 1, int(math.floor((count - 1) * fraction)))
            return lengths[index]

        return {
            "count": count,
            "length_chars": {
                "p25": quantile(0.25),
                "median": quantile(0.5),
                "p75": quantile(0.75),
            },
            "short_share_le_12": round(sum(x <= 12 for x in lengths) / count, 4),
            "emoji_share": round(self.emoji / count, 4),
            "repetition_share": round(self.repetition / count, 4),
            "formal_marker_share": round(self.formal_marker / count, 4),
            "slang_marker_share": round(self.slang_marker / count, 4),
            "question_share": round(self.question / count, 4),
        }


@dataclass
class StyleCorpus:
    profiles: dict[str, dict[str, Any]]
    exclusion_counts: dict[str, int]
    holdout_windows: list[HoldoutWindow]
    _reservoirs: dict[str, dict[str, list[str]]] = field(repr=False)

    def examples(self, source_kind: str, *, seed: int, limit: int = 10) -> list[str]:
        buckets = self._reservoirs.get(source_kind, {})
        rng = random.Random(seed)
        keys = sorted(buckets)
        rng.shuffle(keys)
        values: list[str] = []
        per_bucket = {key: list(buckets[key]) for key in keys}
        for items in per_bucket.values():
            rng.shuffle(items)
        while len(values) < limit and any(per_bucket.values()):
            for key in keys:
                if per_bucket[key] and len(values) < limit:
                    values.append(per_bucket[key].pop())
        return values


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace(" ", "T").replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    except ValueError:
        return None


def build_holdout_windows(definitions_path: str | Path = DEFAULT_DEFINITIONS) -> list[HoldoutWindow]:
    definitions = load_shock_definitions(definitions_path)
    windows: list[HoldoutWindow] = []
    for event in definitions["events"]:
        anchor = _parse_datetime(event["anchor"])
        if anchor is None:
            raise ForumExpressionError(f"invalid event anchor for {event['scenario_id']}")
        start_offset = min(phase["start_offset_minutes"] for phase in event["phases"])
        end_offset = max(phase["end_offset_minutes"] for phase in event["phases"])
        windows.append(
            HoldoutWindow(
                scenario_id=event["scenario_id"],
                start=anchor + timedelta(minutes=start_offset),
                end=anchor + timedelta(minutes=end_offset),
            )
        )
    return windows


def _inside_holdout(value: datetime | None, windows: Iterable[HoldoutWindow]) -> bool:
    return any(window.contains(value) for window in windows)


def _normalize_ephemeral_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not 2 <= len(text) <= max_chars:
        return None
    if URL_PATTERN.search(text) or LONG_DIGIT_PATTERN.search(text):
        return None
    if any(pattern.search(text) for pattern in CONTACT_PATTERNS):
        return None
    if EVENT_TERM_PATTERN.search(text):
        return None
    return text


def _style_bucket(text: str) -> str:
    length = len(text)
    length_bucket = "very_short" if length <= 12 else "short" if length <= 30 else "medium"
    form = (
        "emoji"
        if EMOJI_PATTERN.search(text)
        else "slang"
        if SLANG_MARKER_PATTERN.search(text)
        else "repetition"
        if REPETITION_PATTERN.search(text)
        else "plain"
    )
    return f"{length_bucket}:{form}"


def _reservoir_add(
    reservoirs: dict[str, dict[str, list[str]]],
    seen: Counter[tuple[str, str]],
    *,
    kind: str,
    text: str,
    rng: random.Random,
    capacity: int,
) -> None:
    bucket = _style_bucket(text)
    key = (kind, bucket)
    seen[key] += 1
    values = reservoirs[kind][bucket]
    if len(values) < capacity:
        values.append(text)
        return
    index = rng.randrange(seen[key])
    if index < capacity:
        values[index] = text


def build_style_corpus(
    *,
    posts_path: str | Path,
    comments_path: str | Path,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    seed: int = 20260722,
    reservoir_per_bucket: int = 24,
) -> StyleCorpus:
    """Stream raw CSVs into aggregate profiles and ephemeral held-out exemplars."""
    if reservoir_per_bucket < 1:
        raise ForumExpressionError("reservoir_per_bucket must be positive")
    windows = build_holdout_windows(definitions_path)
    rng = random.Random(seed)
    aggregates = {"post": _Aggregate(), "reply": _Aggregate()}
    reservoirs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen: Counter[tuple[str, str]] = Counter()
    exclusions: Counter[str] = Counter()
    holdout_post_codes: set[str] = set()

    with Path(posts_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value_date = _parse_datetime(row.get("real_date"))
            if _inside_holdout(value_date, windows):
                if row.get("post_code"):
                    holdout_post_codes.add(str(row["post_code"]))
                exclusions["post_in_holdout_window"] += 1
                continue
            text = _normalize_ephemeral_text(row.get("content"), max_chars=120)
            if text is None:
                exclusions["post_unsafe_or_out_of_scope"] += 1
                continue
            aggregates["post"].add(text)
            _reservoir_add(
                reservoirs,
                seen,
                kind="post",
                text=text,
                rng=rng,
                capacity=reservoir_per_bucket,
            )

    with Path(comments_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            post_code = str(row.get("post_code") or "")
            value_date = _parse_datetime(row.get("real_date"))
            if post_code in holdout_post_codes or _inside_holdout(value_date, windows):
                exclusions["reply_linked_to_or_inside_holdout"] += 1
                continue
            text = _normalize_ephemeral_text(row.get("content"), max_chars=80)
            if text is None:
                exclusions["reply_unsafe_or_out_of_scope"] += 1
                continue
            aggregates["reply"].add(text)
            _reservoir_add(
                reservoirs,
                seen,
                kind="reply",
                text=text,
                rng=rng,
                capacity=reservoir_per_bucket,
            )

    return StyleCorpus(
        profiles={key: value.public_view() for key, value in aggregates.items()},
        exclusion_counts=dict(sorted(exclusions.items())),
        holdout_windows=windows,
        _reservoirs={
            kind: {bucket: list(values) for bucket, values in buckets.items()}
            for kind, buckets in reservoirs.items()
        },
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ForumExpressionError(f"JSON root must be an object: {path}")
    return value


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _persona_views(persona_release_path: str | Path) -> dict[str, dict[str, Any]]:
    release = _load_json(persona_release_path)
    views: dict[str, dict[str, Any]] = {}
    for profile in release.get("profiles", []):
        variants = {
            variant["variant_id"]: {
                "label": variant.get("label"),
                "language_style": variant.get("language_style"),
                "reaction_logic": variant.get("reaction_logic"),
            }
            for variant in profile.get("variants", [])
        }
        views[profile["archetype_id"]] = {
            "label": profile.get("label"),
            "language_tendencies": profile.get("language_tendencies", []),
            "do_not_simulate": profile.get("do_not_simulate", []),
            "variants": variants,
        }
    return views


def _phase_index(phase_id: str) -> int:
    try:
        return PHASE_IDS.index(phase_id)
    except ValueError as exc:
        raise ForumExpressionError(f"invalid phase_id: {phase_id}") from exc


def _find_parent(phases: list[dict[str, Any]], phase_id: str, decision: dict[str, Any]) -> str:
    current = _phase_index(phase_id)
    candidates: list[dict[str, Any]] = []
    for phase in phases[:current]:
        candidates.extend(
            item
            for item in phase.get("agent_decisions", [])
            if item.get("synthetic_message")
            and item.get("synthetic_agent_id") != decision.get("synthetic_agent_id")
        )
    same_topic = [item for item in candidates if item.get("topic_id") == decision.get("topic_id")]
    pool = same_topic or candidates
    if not pool:
        return ""
    pool.sort(
        key=lambda item: (
            _phase_index(item.get("phase_id", "baseline"))
            if item.get("phase_id") in PHASE_IDS
            else -1,
            float(item.get("urgency", 0.0)),
        ),
        reverse=True,
    )
    return str(pool[0]["synthetic_message"])


def select_expression_cases(
    simulation_artifact: dict[str, Any], *, per_scenario: int = 8
) -> list[dict[str, Any]]:
    """Select distinct archetypes and a post/reply mix from an accepted run."""
    if per_scenario < 2:
        raise ForumExpressionError("per_scenario must be at least 2")
    runs = simulation_artifact.get("policy_runs", [])
    if not runs:
        raise ForumExpressionError("simulation artifact has no policy_runs")
    phases = runs[0].get("phases", [])
    phase_map = {phase.get("phase_id"): phase for phase in phases}
    desired: list[tuple[str, str]] = [
        ("burst", "post"),
        ("burst", "post"),
        ("spread", "reply"),
        ("spread", "reply"),
        ("spread", "post"),
        ("decay", "reply"),
        ("decay", "reply"),
        ("decay", "post"),
    ]
    selected: list[dict[str, Any]] = []
    seen_archetypes: set[str] = set()

    def candidates(phase_id: str, target: str) -> list[dict[str, Any]]:
        phase = phase_map.get(phase_id, {})
        values = []
        for decision in phase.get("agent_decisions", []):
            action = decision.get("action")
            if action not in PUBLIC_ACTIONS or not decision.get("synthetic_message"):
                continue
            if decision.get("topic_id") == "off_topic_daily_life":
                continue
            actual_target = "reply" if action == "comment" else "post"
            if actual_target != target:
                continue
            item = dict(decision)
            item["phase_id"] = phase_id
            item["phase_label"] = phase.get("label")
            item["target_type"] = target
            item["parent_message"] = (
                _find_parent(phases, phase_id, item) if target == "reply" else ""
            )
            values.append(item)
        return sorted(
            values,
            key=lambda item: (
                -float(item.get("population_weight", 0.0)),
                -float(item.get("urgency", 0.0)),
                str(item.get("archetype_id")),
            ),
        )

    for phase_id, target in desired[:per_scenario]:
        pool = candidates(phase_id, target)
        choice = next(
            (item for item in pool if item.get("archetype_id") not in seen_archetypes),
            None,
        )
        if choice is None:
            continue
        selected.append(choice)
        seen_archetypes.add(str(choice.get("archetype_id")))

    if len(selected) < per_scenario:
        fallback: list[dict[str, Any]] = []
        for phase_id in ("burst", "spread", "decay"):
            fallback.extend(candidates(phase_id, "post"))
            fallback.extend(candidates(phase_id, "reply"))
        for item in fallback:
            if len(selected) >= per_scenario:
                break
            if item.get("archetype_id") in seen_archetypes:
                continue
            selected.append(item)
            seen_archetypes.add(str(item.get("archetype_id")))
    if len(selected) != per_scenario:
        raise ForumExpressionError(
            f"could select only {len(selected)} distinct expression cases"
        )
    return selected


def _longest_common_substring(left: str, right: str) -> int:
    left = re.sub(r"\s+", "", left)
    right = re.sub(r"\s+", "", right)
    previous = [0] * (len(right) + 1)
    longest = 0
    for char_left in left:
        current = [0]
        for index, char_right in enumerate(right, start=1):
            value = previous[index - 1] + 1 if char_left == char_right else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


def validate_render_candidates(
    raw: dict[str, Any], *, exemplars: list[str], target_type: str
) -> list[dict[str, Any]]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 4:
        raise ForumExpressionError("renderer must return 2 to 4 candidates")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").strip().split())
        if not 1 <= len(text) <= 100 or text in seen:
            continue
        if URL_PATTERN.search(text) or LONG_DIGIT_PATTERN.search(text):
            continue
        if any(pattern.search(text) for pattern in CONTACT_PATTERNS):
            continue
        if any(_longest_common_substring(text, example) >= 12 for example in exemplars):
            continue
        new_claims = item.get("new_factual_claims", [])
        if not isinstance(new_claims, list) or new_claims:
            continue
        reply_function = str(item.get("reply_function") or "")
        if reply_function not in ALLOWED_REPLY_FUNCTIONS:
            continue
        if target_type == "reply" and reply_function == "not_a_reply":
            continue
        if target_type == "post" and reply_function != "not_a_reply":
            continue
        register = str(item.get("style_register") or "")
        if register not in ALLOWED_REGISTERS:
            continue
        validated.append(
            {
                "text": text,
                "reply_function": reply_function,
                "style_register": register,
                "semantic_summary": str(item.get("semantic_summary") or "")[:160],
                "new_factual_claims": [],
            }
        )
        seen.add(text)
    if len(validated) < 2:
        raise ForumExpressionError("fewer than two safe render candidates remained")
    return validated


def validate_judgment(raw: dict[str, Any], candidate_count: int, target_type: str) -> dict[str, Any]:
    index = raw.get("selected_candidate_index")
    if not isinstance(index, int) or not 0 <= index < candidate_count:
        raise ForumExpressionError("judge selected an invalid candidate")
    required_bools = (
        "semantic_preservation",
        "persona_fit",
        "forum_naturalness",
        "no_new_factual_claim",
    )
    values: dict[str, Any] = {}
    for key in required_bools:
        if not isinstance(raw.get(key), bool):
            raise ForumExpressionError(f"judge field {key} must be boolean")
        values[key] = raw[key]
    parent_relevance = raw.get("parent_relevance")
    if not isinstance(parent_relevance, bool):
        raise ForumExpressionError("judge field parent_relevance must be boolean")
    values["parent_relevance"] = parent_relevance
    reasons = raw.get("rejection_reasons", [])
    if not isinstance(reasons, list):
        raise ForumExpressionError("rejection_reasons must be a list")
    values.update(
        {
            "selected_candidate_index": index,
            "rejection_reasons": [str(value)[:180] for value in reasons[:5]],
            "unresolved_risk": str(raw.get("unresolved_risk") or "")[:240],
        }
    )
    assessments = raw.get("candidate_assessments")
    validated_assessments: list[dict[str, Any]] = []
    if isinstance(assessments, list) and len(assessments) == candidate_count:
        for expected_index, assessment in enumerate(assessments):
            if not isinstance(assessment, dict) or assessment.get("index") != expected_index:
                raise ForumExpressionError("candidate assessments must be ordered by index")
            flags: dict[str, Any] = {"index": expected_index}
            for key in (
                "semantic_preservation",
                "persona_fit",
                "forum_naturalness",
                "no_new_factual_claim",
                "parent_relevance",
            ):
                if not isinstance(assessment.get(key), bool):
                    raise ForumExpressionError(
                        f"candidate assessment field {key} must be boolean"
                    )
                flags[key] = assessment[key]
            validated_assessments.append(flags)
    else:
        validated_assessments = [
            {
                "index": candidate_index,
                "semantic_preservation": values["semantic_preservation"]
                if candidate_index == index
                else False,
                "persona_fit": values["persona_fit"] if candidate_index == index else False,
                "forum_naturalness": values["forum_naturalness"]
                if candidate_index == index
                else False,
                "no_new_factual_claim": values["no_new_factual_claim"]
                if candidate_index == index
                else False,
                "parent_relevance": values["parent_relevance"]
                if candidate_index == index
                else False,
            }
            for candidate_index in range(candidate_count)
        ]
    values["candidate_assessments"] = validated_assessments
    values["accepted"] = bool(
        values["semantic_preservation"]
        and values["persona_fit"]
        and values["forum_naturalness"]
        and values["no_new_factual_claim"]
        and (target_type != "reply" or values["parent_relevance"])
        and not values["unresolved_risk"]
    )
    return values


def _apply_length_calibrated_selection(
    judgment: dict[str, Any],
    candidates: list[dict[str, Any]],
    style_profile: dict[str, Any],
    target_type: str,
) -> dict[str, Any]:
    """Choose among judge-approved candidates using the held-out length profile."""
    model_index = int(judgment["selected_candidate_index"])
    eligible: list[int] = []
    for assessment in judgment.get("candidate_assessments", []):
        if not all(
            assessment.get(key)
            for key in (
                "semantic_preservation",
                "persona_fit",
                "forum_naturalness",
                "no_new_factual_claim",
            )
        ):
            continue
        if target_type == "reply" and not assessment.get("parent_relevance"):
            continue
        eligible.append(int(assessment["index"]))
    lengths = style_profile.get("length_chars", {})
    p25 = int(lengths.get("p25", 1))
    median = int(lengths.get("median", p25))
    p75 = int(lengths.get("p75", 100))
    in_range = [
        index for index in eligible if p25 <= len(candidates[index]["text"]) <= p75
    ]
    chosen = (
        min(in_range, key=lambda index: abs(len(candidates[index]["text"]) - median))
        if in_range
        else model_index
    )
    result = dict(judgment)
    result["model_selected_candidate_index"] = model_index
    result["selected_candidate_index"] = chosen
    result["length_calibrated_selection"] = chosen != model_index
    result["held_out_length_target"] = {"p25": p25, "median": median, "p75": p75}
    return result


def _render_prompt(
    *,
    scenario_id: str,
    case: dict[str, Any],
    persona: dict[str, Any],
    style_profile: dict[str, Any],
    exemplars: list[str],
) -> str:
    variant = persona.get("variants", {}).get(case.get("persona_variant_id"), {})
    payload = {
        "task": "将已冻结的行为底稿渲染成校园匿名论坛表达；不得改变行为决策",
        "scenario_id": scenario_id,
        "phase": {"id": case["phase_id"], "label": case.get("phase_label")},
        "target_type": case["target_type"],
        "scenario_expression_context": SCENARIO_EXPRESSION_CONTEXT.get(scenario_id),
        "parent_message": case.get("parent_message") or None,
        "behavior_ledger": {
            "action": case.get("action"),
            "topic_id": case.get("topic_id"),
            "stance": case.get("stance"),
            "urgency": case.get("urgency"),
            "rationale_tag": case.get("rationale_tag"),
            "canonical_message": case.get("synthetic_message"),
        },
        "persona_expression_evidence": {
            "archetype": persona.get("label"),
            "variant": variant.get("label"),
            "language_tendencies": persona.get("language_tendencies", []),
            "variant_language_style": variant.get("language_style"),
            "reaction_logic": variant.get("reaction_logic"),
            "prohibitions": persona.get("do_not_simulate", [])[:5],
        },
        "held_out_style_profile": style_profile,
        "ephemeral_non_event_style_exemplars": exemplars,
        "hard_rules": [
            "原始行为底稿、议题、立场和核心诉求不能改变",
            "不得补充公告细则、现场经过、处理结果、身份、地点或其他新事实",
            "不得逐句仿写例句，连续复用例句不得达到12个字符",
            "可以省略语法成分、缩短句子、使用少量自然表情或校园网络口语，但不能每条都强行玩梗",
            "若target_type为reply，必须直接承接parent_message，选择一种真实回复功能",
            "若target_type为post，reply_function必须为not_a_reply",
            "每个候选只表达一个中心，不写政策报告，不解释生成过程",
            (
                "三条回复候选必须依次覆盖：不超过12字、13至23字、24至40字；"
                "短候选可以只保留最核心的接话功能，但不得改变立场"
                if case["target_type"] == "reply"
                else "三条主帖候选必须依次覆盖：不超过20字、21至35字、36至55字"
            ),
        ],
        "output_schema": {
            "candidates": [
                {
                    "text": "1至100字论坛表达",
                    "reply_function": sorted(ALLOWED_REPLY_FUNCTIONS),
                    "style_register": sorted(ALLOWED_REGISTERS),
                    "semantic_summary": "该候选保留的核心意思",
                    "new_factual_claims": [],
                }
            ],
            "candidate_count": 3,
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _judge_prompt(
    *,
    case: dict[str, Any],
    persona: dict[str, Any],
    style_profile: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    variant = persona.get("variants", {}).get(case.get("persona_variant_id"), {})
    payload = {
        "task": "从候选中选择既不扭曲底稿、又最像校园论坛自然表达的一条",
        "target_type": case["target_type"],
        "parent_message": case.get("parent_message") or None,
        "behavior_ledger": {
            "action": case.get("action"),
            "topic_id": case.get("topic_id"),
            "stance": case.get("stance"),
            "urgency": case.get("urgency"),
            "canonical_message": case.get("synthetic_message"),
        },
        "persona_style": {
            "archetype": persona.get("label"),
            "variant": variant.get("label"),
            "language_tendencies": persona.get("language_tendencies", []),
            "variant_language_style": variant.get("language_style"),
        },
        "aggregate_style_profile": style_profile,
        "candidates": candidates,
        "review_rules": [
            "语义保真优先于口语化",
            "不得接受新增事件事实或把疑问写成断言的候选",
            "回复必须针对父帖而不是一条独立小作文",
            (
                "回复的语义保真按父帖与回复合读判断：保留核心立场和接话功能即可，"
                "可以省略底稿中的次要背景、并列举例和流程细项；不要仅因短句未复述全部底稿就判false"
                if case["target_type"] == "reply"
                else "主帖必须独立可理解，并保留底稿的核心问题或诉求；不能依赖不存在的父帖补语义"
            ),
            "自然不等于堆叠表情和黑话，应符合该人物语言倾向",
            "语义都合格时，优先选择落在真实文本长度四分位区间内的候选",
            "不要因为长候选复述了更多底稿词语就默认选择长候选；短回复只要保留核心接话功能即可",
            "若候选仍像行政公文，forum_naturalness应为false",
        ],
        "output_schema": {
            "selected_candidate_index": "0-based integer",
            "semantic_preservation": "boolean",
            "persona_fit": "boolean",
            "forum_naturalness": "boolean",
            "no_new_factual_claim": "boolean",
            "parent_relevance": "boolean; post可填true",
            "rejection_reasons": [],
            "unresolved_risk": "empty string when none",
            "candidate_assessments": [
                {
                    "index": "0-based integer; every candidate exactly once in order",
                    "semantic_preservation": "boolean",
                    "persona_fit": "boolean",
                    "forum_naturalness": "boolean",
                    "no_new_factual_claim": "boolean",
                    "parent_relevance": "boolean; post可填true",
                }
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _complete_json(
    *,
    client: VectorEngineClient,
    budget: TokenBudget,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], dict[str, int], str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    total_usage = Counter()
    returned_model = client.model
    for repair in range(2):
        response, _, _ = client._request(messages, budget)  # bounded shared provider client
        usage = response.get("usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[key] += int(usage.get(key) or 0)
        returned_model = str(response.get("model") or returned_model)
        try:
            return (
                extract_json_object(response["choices"][0]["message"]["content"]),
                dict(total_usage),
                returned_model,
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, LLMAgentError) as exc:
            last_error = exc
            messages = [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n上次输出不是有效JSON对象：{str(exc)[:160]}。"
                        "请只输出修正后的JSON对象。"
                    ),
                },
            ]
    raise ForumExpressionError(f"provider JSON contract failed: {last_error}")


def _render_one_case(
    *,
    client: VectorEngineClient,
    budget: TokenBudget,
    scenario_id: str,
    case: dict[str, Any],
    persona: dict[str, Any],
    style_profile: dict[str, Any],
    exemplars: list[str],
) -> dict[str, Any]:
    case_id = hashlib.sha256(
        f"{scenario_id}|{case.get('synthetic_agent_id')}|{case['phase_id']}".encode()
    ).hexdigest()[:16]
    base_result = {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "phase_id": case["phase_id"],
        "phase_label": case.get("phase_label"),
        "target_type": case["target_type"],
        "synthetic_agent_id": case.get("synthetic_agent_id"),
        "archetype_id": case.get("archetype_id"),
        "archetype_label": case.get("archetype_label"),
        "persona_variant_id": case.get("persona_variant_id"),
        "persona_variant_label": case.get("persona_variant_label"),
        "behavior_ledger": {
            "action": case.get("action"),
            "topic_id": case.get("topic_id"),
            "stance": case.get("stance"),
            "urgency": case.get("urgency"),
            "rationale_tag": case.get("rationale_tag"),
            "canonical_message": case.get("synthetic_message"),
            "semantic_preservation_scope": (
                "parent_conditioned_core_stance_and_reply_function; secondary details may be omitted"
                if case["target_type"] == "reply"
                else "self_contained_core_intent"
            ),
        },
        "parent_synthetic_message": case.get("parent_message") or None,
    }
    renderer_system = (
        "你是匿名校园论坛表达渲染器。行为底稿已经冻结；你的工作只是改变表面表达。"
        "不得新增事实、身份、治理承诺或现场细节。只输出JSON。"
    )
    render_prompt = _render_prompt(
        scenario_id=scenario_id,
        case=case,
        persona=persona,
        style_profile=style_profile,
        exemplars=exemplars,
    )
    render_usage: Counter[str] = Counter()
    returned_model = client.model
    candidates: list[dict[str, Any]] = []
    render_error = ""
    for contract_attempt in range(2):
        prompt = render_prompt
        if contract_attempt:
            prompt += (
                f"\n\n上轮候选契约失败：{render_error[:180]}。"
                "请重新生成三条，new_factual_claims必须为空，且不要复用参照文本。"
            )
        try:
            raw_candidates, attempt_usage, returned_model = _complete_json(
                client=client,
                budget=budget,
                system_prompt=renderer_system,
                user_prompt=prompt,
            )
            render_usage.update(attempt_usage)
            candidates = validate_render_candidates(
                raw_candidates, exemplars=exemplars, target_type=case["target_type"]
            )
            break
        except (ForumExpressionError, LLMAgentError) as exc:
            render_error = str(exc)
    if not candidates:
        return {
            **base_result,
            "render_candidates": [],
            "judge": {
                "accepted": False,
                "rejection_reasons": ["renderer_contract_failed_after_retry"],
                "unresolved_risk": render_error[:240],
            },
            "render_status": "fallback_to_canonical",
            "forum_text": case["synthetic_message"],
            "selected_reply_function": None,
            "provider": {
                "render_model": returned_model,
                "judge_model": None,
                "usage": {
                    key: int(render_usage.get(key, 0))
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            },
        }

    judge_usage: Counter[str] = Counter()
    judge_model = client.model
    judgment: dict[str, Any] | None = None
    judge_error = ""
    judge_prompt = _judge_prompt(
        case=case,
        persona=persona,
        style_profile=style_profile,
        candidates=candidates,
    )
    for contract_attempt in range(2):
        prompt = judge_prompt
        if contract_attempt:
            prompt += (
                f"\n\n上轮审查契约失败：{judge_error[:180]}。"
                "请严格按schema重新输出，索引必须指向现有候选。"
            )
        try:
            raw_judgment, attempt_usage, judge_model = _complete_json(
                client=client,
                budget=budget,
                system_prompt=(
                    "你是严格的模拟文本审稿人。先检查语义是否与行为底稿等价，再检查论坛自然度；"
                    "不能因为更口语就放过事实新增或语义漂移。只输出JSON。"
                ),
                user_prompt=prompt,
            )
            judge_usage.update(attempt_usage)
            judgment = validate_judgment(
                raw_judgment, len(candidates), case["target_type"]
            )
            judgment = _apply_length_calibrated_selection(
                judgment, candidates, style_profile, case["target_type"]
            )
            break
        except (ForumExpressionError, LLMAgentError) as exc:
            judge_error = str(exc)
    if judgment is None:
        judgment = {
            "accepted": False,
            "rejection_reasons": ["judge_contract_failed_after_retry"],
            "unresolved_risk": judge_error[:240],
            "selected_candidate_index": 0,
        }
    chosen = candidates[judgment["selected_candidate_index"]]
    forum_text = chosen["text"] if judgment["accepted"] else case["synthetic_message"]
    return {
        **base_result,
        "render_candidates": candidates,
        "judge": judgment,
        "render_status": "accepted" if judgment["accepted"] else "fallback_to_canonical",
        "forum_text": forum_text,
        "selected_reply_function": chosen["reply_function"],
        "provider": {
            "render_model": returned_model,
            "judge_model": judge_model,
            "usage": {
                key: int(render_usage.get(key, 0)) + int(judge_usage.get(key, 0))
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        },
    }


def run_forum_expression_pilot(
    *,
    posts_path: str | Path,
    comments_path: str | Path,
    simulation_artifact_paths: list[str | Path],
    persona_release_path: str | Path,
    output_path: str | Path,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    per_scenario: int = 8,
    seed: int = 20260722,
    max_concurrency: int = 8,
    max_total_tokens: int = 180000,
    max_output_tokens: int = 700,
    timeout_seconds: float = 120.0,
    allow_provider_text: bool = False,
) -> dict[str, Any]:
    """Render a small held-out two-shock pilot without changing agent decisions."""
    if not allow_provider_text:
        raise ForumExpressionError(
            "raw style exemplars may be sent only with allow_provider_text=True"
        )
    if not 1 <= max_concurrency <= 32:
        raise ForumExpressionError("max_concurrency must be between 1 and 32")
    if not simulation_artifact_paths:
        raise ForumExpressionError("at least one simulation artifact is required")

    style = build_style_corpus(
        posts_path=posts_path,
        comments_path=comments_path,
        definitions_path=definitions_path,
        seed=seed,
    )
    personas = _persona_views(persona_release_path)
    client = VectorEngineClient.from_environment(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        max_network_retries=1,
        max_format_repairs=1,
    )
    budget = TokenBudget(max_total_tokens)
    work: list[tuple[str, dict[str, Any], dict[str, Any], list[str]]] = []
    input_fingerprints: list[dict[str, Any]] = []
    for artifact_path in simulation_artifact_paths:
        artifact = _load_json(artifact_path)
        scenario_id = str(artifact.get("scenario_id") or "")
        if not scenario_id:
            raise ForumExpressionError("simulation artifact has no scenario_id")
        input_fingerprints.append(
            {"scenario_id": scenario_id, **_file_fingerprint(artifact_path)}
        )
        for index, case in enumerate(
            select_expression_cases(artifact, per_scenario=per_scenario)
        ):
            archetype_id = str(case.get("archetype_id") or "")
            if archetype_id not in personas:
                raise ForumExpressionError(f"persona missing for {archetype_id}")
            kind = case["target_type"]
            examples = style.examples(
                kind,
                seed=seed + index * 101 + int(hashlib.sha256(scenario_id.encode()).hexdigest()[:6], 16),
                limit=10,
            )
            if len(examples) < 6:
                raise ForumExpressionError(f"insufficient held-out {kind} exemplars")
            work.append((scenario_id, case, personas[archetype_id], examples))

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                _render_one_case,
                client=client,
                budget=budget,
                scenario_id=scenario_id,
                case=case,
                persona=persona,
                style_profile=style.profiles[case["target_type"]],
                exemplars=examples,
            ): (scenario_id, case)
            for scenario_id, case, persona, examples in work
        }
        for future in as_completed(futures):
            scenario_id, case = futures[future]
            try:
                result = future.result()
                with lock:
                    results.append(result)
            except Exception as exc:  # preserve all other completed pilot cases
                with lock:
                    errors.append(
                        f"{scenario_id}:{case.get('synthetic_agent_id')}:{case.get('phase_id')}:{exc}"
                    )
    if errors:
        raise ForumExpressionError("; ".join(errors[:5]))

    results.sort(key=lambda item: (item["scenario_id"], _phase_index(item["phase_id"]), item["case_id"]))
    usage = {
        key: sum(item["provider"]["usage"][key] for item in results)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    accepted = sum(item["render_status"] == "accepted" for item in results)
    artifact = {
        "schema_version": "forum-expression-pilot-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "method": {
            "architecture": "frozen_behavior_ledger_then_surface_renderer_then_independent_semantic_judge",
            "renderer_prompt_version": "1.2.0",
            "behavior_decisions_modified": False,
            "automatic_metrics_are_diagnostic_only": True,
            "direct_blind_reading_required_for_fidelity_verdict": True,
            "reply_scope": "local_root_or_prior-discussion response; no claim of a complete observed reply graph",
        },
        "source_boundaries": {
            "raw_paths_persisted": False,
            "raw_forum_text_persisted": False,
            "provider_prompts_persisted": False,
            "shock_window_text_sent_to_provider": False,
            "replies_linked_to_holdout_posts_sent_to_provider": False,
            "direct_event_terms_allowed_in_style_exemplars": False,
            "contact_or_url_text_allowed_in_style_exemplars": False,
            "ephemeral_style_exemplars_per_case": 10,
            "holdout_windows": [window.public_view() for window in style.holdout_windows],
            "exclusion_counts": style.exclusion_counts,
        },
        "input_evidence": {
            "simulation_artifacts": input_fingerprints,
            "persona_release": _file_fingerprint(persona_release_path),
            "definitions": _file_fingerprint(definitions_path),
        },
        "held_out_style_profiles": style.profiles,
        "provider": {
            "requested_model": model,
            "base_url": base_url,
            "endpoint": "/chat/completions",
            "api_key_or_value_persisted": False,
            "usage": usage,
        },
        "summary": {
            "case_count": len(results),
            "accepted_render_count": accepted,
            "fallback_count": len(results) - accepted,
            "scenario_count": len({item["scenario_id"] for item in results}),
            "distinct_archetype_count": len({item["archetype_id"] for item in results}),
            "post_case_count": sum(item["target_type"] == "post" for item in results),
            "reply_case_count": sum(item["target_type"] == "reply" for item in results),
        },
        "cases": results,
        "validity": {
            "mechanism_fidelity_unchanged_by_renderer": True,
            "surface_fidelity_verdict": "pending_direct_blind_reading",
            "historical_shock_text_is_held_out": True,
            "no_causal_policy_effect_claim": True,
            "no_complete_reply_cascade_claim": True,
        },
    }
    rendered = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
    if EVENT_TERM_PATTERN.search(
        json.dumps(
            artifact["held_out_style_profiles"], ensure_ascii=False, sort_keys=True
        )
    ):
        raise ForumExpressionError("raw event term leaked into aggregate style profile")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    return artifact
