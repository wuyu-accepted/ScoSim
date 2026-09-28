"""Data-bounded lecture thread ecology and parent-conditioned reply pilot.

The module adds a second behavioural layer on top of accepted agent decisions:
an explicit reply function and a local thread state. Raw forum pairs are used
ephemerally for aggregate function coding. Only aggregate counts, synthetic
messages, hashes, and provider usage are persisted.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forum_expression import (
    ALLOWED_REGISTERS,
    CONTACT_PATTERNS,
    EMOJI_PATTERN,
    EVENT_TERM_PATTERN,
    LONG_DIGIT_PATTERN,
    URL_PATTERN,
    ForumExpressionError,
    StyleCorpus,
    _complete_json,
    _file_fingerprint,
    _inside_holdout,
    _load_json,
    _longest_common_substring,
    _parse_datetime,
    _persona_views,
    build_holdout_windows,
    build_style_corpus,
)
from .llm_agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMAgentError,
    TokenBudget,
    VectorEngineClient,
)
from .shock_replay import DEFAULT_DEFINITIONS, load_shock_definitions


LECTURE_SCENARIO_ID = "lecture_external_incident_shock"
KEY_FUNCTIONS = (
    "evidence_request",
    "evidence_relay",
    "irony_or_meme",
    "amplification_or_pile_on",
)
THREAD_FUNCTIONS = {
    *KEY_FUNCTIONS,
    "challenge_or_correction",
    "emotional_support",
    "advice_or_action",
    "clarification",
    "personal_experience",
    "signal_only",
}
FUNCTION_LABELS = {
    "evidence_request": "求证原文、来源或完整语境",
    "evidence_relay": "转述证据状态并明确已核验/未核验边界",
    "irony_or_meme": "用轻度反讽或接梗回应观点",
    "amplification_or_pile_on": "附和、重复或推高已有立场可见度",
    "challenge_or_correction": "反驳、纠正或要求收窄断言",
    "emotional_support": "安慰、共情或承接情绪",
    "advice_or_action": "给建议、下一步或行动方案",
    "clarification": "澄清自己/他人的意思但不新增证据",
    "personal_experience": "分享与讨论直接相关的个人体验",
    "signal_only": "短促态度或纯信号式回复",
}
CAREER_PATTERN = re.compile(
    r"考公|公务员|考编|就业|择业|职业规划|求职|找工作|秋招|实习|offer|工作|职业",
    re.I,
)
FACT_QUERY_PATTERN = re.compile(r"原文|原话|来源|截图|链接|完整|语境|核实|说明|反馈|入口|转述")
CRITICAL_PATTERN = re.compile(r"别|不该|高低|贬低|争议|打分|圣经|贵贱")
CLARIFICATION_PATTERN = re.compile(r"具体|指的是|意思是|也就是|所以|哪个部门|由谁|哪一项|确认由")
IRONY_PATTERN = re.compile(
    r"是吧|圣经|排座次|三六九等|秒变|合着|原来|好家伙|耐热|说明书|判分|😅|🙃|哈哈"
)


class LectureThreadError(ValueError):
    """Raised when the lecture thread pilot violates its evidence contract."""


@dataclass(frozen=True)
class EphemeralPair:
    parent_text: str
    reply_text: str
    split: str


@dataclass
class PairSample:
    reference: list[EphemeralPair]
    lecture_holdout: list[EphemeralPair]
    counts: dict[str, int]


def _safe_pair_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not 2 <= len(text) <= max_chars:
        return None
    if URL_PATTERN.search(text) or LONG_DIGIT_PATTERN.search(text):
        return None
    if any(pattern.search(text) for pattern in CONTACT_PATTERNS):
        return None
    return text


def _reservoir_pair(
    values: list[EphemeralPair],
    item: EphemeralPair,
    *,
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(values) < capacity:
        values.append(item)
        return
    index = rng.randrange(seen)
    if index < capacity:
        values[index] = item


def build_thread_pair_sample(
    *,
    posts_path: str | Path,
    comments_path: str | Path,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    sample_per_split: int = 48,
    max_replies_per_thread: int = 4,
    seed: int = 20260722,
) -> PairSample:
    """Sample comparable career replies and a strictly held-out lecture split."""
    if not 8 <= sample_per_split <= 200:
        raise LectureThreadError("sample_per_split must be between 8 and 200")
    if not 1 <= max_replies_per_thread <= 10:
        raise LectureThreadError("max_replies_per_thread must be between 1 and 10")
    windows = build_holdout_windows(definitions_path)
    lecture_window = next(
        (window for window in windows if window.scenario_id == LECTURE_SCENARIO_ID),
        None,
    )
    if lecture_window is None:
        raise LectureThreadError("lecture holdout window is missing")

    # Only posts that can enter one of the two comparison splits are retained.
    parents: dict[str, tuple[str, str]] = {}
    counts: Counter[str] = Counter()
    with Path(posts_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            post_code = str(row.get("post_code") or "")
            value_date = _parse_datetime(row.get("real_date"))
            text = _safe_pair_text(row.get("content"), max_chars=180)
            if not post_code or value_date is None or text is None:
                counts["post_unsafe_or_missing"] += 1
                continue
            if lecture_window.contains(value_date) and (
                EVENT_TERM_PATTERN.search(text) or CAREER_PATTERN.search(text)
            ):
                parents[post_code] = ("lecture_holdout", text)
                counts["lecture_holdout_parent"] += 1
            elif not _inside_holdout(value_date, windows) and CAREER_PATTERN.search(text):
                parents[post_code] = ("reference", text)
                counts["reference_parent"] += 1

    rngs = {
        "reference": random.Random(seed + 17),
        "lecture_holdout": random.Random(seed + 29),
    }
    samples: dict[str, list[EphemeralPair]] = {
        "reference": [],
        "lecture_holdout": [],
    }
    eligible_seen: Counter[str] = Counter()
    per_thread: Counter[str] = Counter()
    with Path(comments_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            post_code = str(row.get("post_code") or "")
            parent = parents.get(post_code)
            if parent is None:
                continue
            split, parent_text = parent
            reply_text = _safe_pair_text(row.get("content"), max_chars=100)
            if reply_text is None:
                counts[f"{split}_reply_unsafe"] += 1
                continue
            if per_thread[post_code] >= max_replies_per_thread:
                counts[f"{split}_reply_above_thread_cap"] += 1
                continue
            per_thread[post_code] += 1
            eligible_seen[split] += 1
            _reservoir_pair(
                samples[split],
                EphemeralPair(parent_text=parent_text, reply_text=reply_text, split=split),
                seen=eligible_seen[split],
                capacity=sample_per_split,
                rng=rngs[split],
            )
    for split in samples:
        counts[f"{split}_eligible_after_cap"] = eligible_seen[split]
        counts[f"{split}_sampled"] = len(samples[split])
        if len(samples[split]) < sample_per_split:
            raise LectureThreadError(
                f"only {len(samples[split])} usable pairs for {split}"
            )
    return PairSample(
        reference=samples["reference"],
        lecture_holdout=samples["lecture_holdout"],
        counts=dict(sorted(counts.items())),
    )


def validate_pair_labels(raw: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    labels = raw.get("labels")
    if not isinstance(labels, list) or len(labels) != expected_count:
        raise LectureThreadError("classifier must label every pair exactly once")
    validated: list[dict[str, Any]] = []
    for expected_index, item in enumerate(labels):
        if not isinstance(item, dict) or item.get("index") != expected_index:
            raise LectureThreadError("classifier labels must be ordered by local index")
        primary = str(item.get("primary_function") or "")
        if primary not in THREAD_FUNCTIONS:
            raise LectureThreadError(f"unsupported reply function: {primary}")
        flags: dict[str, bool] = {}
        for key in KEY_FUNCTIONS:
            value = item.get(key)
            if not isinstance(value, bool):
                raise LectureThreadError(f"classifier flag {key} must be boolean")
            flags[key] = value
        confidence = str(item.get("confidence") or "")
        if confidence not in {"high", "medium", "low"}:
            raise LectureThreadError("classifier confidence is invalid")
        validated.append(
            {"index": expected_index, "primary_function": primary, **flags, "confidence": confidence}
        )
    return validated


def _classification_prompt(pairs: list[EphemeralPair]) -> str:
    return json.dumps(
        {
            "task": "判定匿名校园论坛中每条回复正在执行的互动功能",
            "label_definitions": FUNCTION_LABELS,
            "rules": [
                "按回复与父帖合读后的主要功能分类，不按关键词机械分类",
                "evidence_relay必须实际转述信息或证据状态；仅索要来源是evidence_request",
                "amplification_or_pile_on包括短促附和、重复立场、推高情绪或可见度",
                "irony_or_meme需要真实反讽、玩梗或接梗，不是普通反对",
                "四个布尔字段可多选，primary_function只能一个",
                "不要复述原文，不输出解释",
            ],
            "pairs": [
                {"index": index, "parent": pair.parent_text, "reply": pair.reply_text}
                for index, pair in enumerate(pairs)
            ],
            "output_schema": {
                "labels": [
                    {
                        "index": "0-based integer, ordered",
                        "primary_function": sorted(THREAD_FUNCTIONS),
                        **{key: "boolean" for key in KEY_FUNCTIONS},
                        "confidence": ["high", "medium", "low"],
                    }
                ]
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _classify_batch(
    *, client: VectorEngineClient, budget: TokenBudget, pairs: list[EphemeralPair]
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    prompt = _classification_prompt(pairs)
    usage: Counter[str] = Counter()
    returned_model = client.model
    last_error = ""
    for attempt in range(2):
        current = prompt
        if attempt:
            current += (
                f"\n\n上轮标签契约失败：{last_error[:160]}。"
                "请逐条按index重做，只输出JSON。"
            )
        raw, attempt_usage, returned_model = _complete_json(
            client=client,
            budget=budget,
            system_prompt=(
                "你是论坛互动功能编码员。只判断回复在对话中做了什么，不推断用户身份。"
                "不得在输出中复述论坛文本。只输出JSON。"
            ),
            user_prompt=current,
        )
        usage.update(attempt_usage)
        try:
            return validate_pair_labels(raw, len(pairs)), dict(usage), returned_model
        except LectureThreadError as exc:
            last_error = str(exc)
    raise LectureThreadError(f"pair classification failed: {last_error}")


def classify_pair_sample(
    *,
    client: VectorEngineClient,
    budget: TokenBudget,
    sample: PairSample,
    batch_size: int = 8,
    max_concurrency: int = 8,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int], str]:
    """Return aggregate public evidence plus ephemeral labels for audit/planning."""
    if not 4 <= batch_size <= 16:
        raise LectureThreadError("batch_size must be between 4 and 16")
    tasks: list[tuple[str, int, list[EphemeralPair]]] = []
    for split, values in (
        ("reference", sample.reference),
        ("lecture_holdout", sample.lecture_holdout),
    ):
        for offset in range(0, len(values), batch_size):
            tasks.append((split, offset, values[offset : offset + batch_size]))
    collected: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    usage: Counter[str] = Counter()
    models: Counter[str] = Counter()
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_classify_batch, client=client, budget=budget, pairs=batch): (
                split,
                offset,
            )
            for split, offset, batch in tasks
        }
        for future in as_completed(futures):
            split, offset = futures[future]
            labels, batch_usage, returned_model = future.result()
            with lock:
                for label in labels:
                    collected[split][offset + label["index"]] = label
                usage.update(batch_usage)
                models[returned_model] += 1
    ordered = {
        split: [values[index] for index in sorted(values)]
        for split, values in collected.items()
    }
    public: dict[str, Any] = {}
    for split, labels in ordered.items():
        primary = Counter(item["primary_function"] for item in labels)
        count = len(labels)
        public[split] = {
            "sample_count": count,
            "primary_counts": {key: primary.get(key, 0) for key in sorted(THREAD_FUNCTIONS)},
            "primary_shares": {
                key: round(primary.get(key, 0) / count, 4)
                for key in sorted(THREAD_FUNCTIONS)
            },
            "multi_label_key_function_rates": {
                key: round(sum(item[key] for item in labels) / count, 4)
                for key in KEY_FUNCTIONS
            },
            "confidence_counts": dict(sorted(Counter(item["confidence"] for item in labels).items())),
        }
    return public, ordered, dict(usage), (models.most_common(1)[0][0] if models else client.model)


def _load_lecture_roots(expression_artifact_path: str | Path) -> list[dict[str, Any]]:
    artifact = _load_json(expression_artifact_path)
    candidates = [
        item
        for item in artifact.get("cases", [])
        if item.get("scenario_id") == LECTURE_SCENARIO_ID
        and item.get("target_type") == "post"
        and item.get("render_status") == "accepted"
        and item.get("forum_text")
        and item.get("behavior_ledger", {}).get("topic_id") != "off_topic_daily_life"
    ]
    direct = [item for item in candidates if item["behavior_ledger"].get("topic_id") == "direct_event"]
    career = [item for item in candidates if item["behavior_ledger"].get("topic_id") == "career_spillover"]
    career.sort(
        key=lambda item: (
            bool(re.search(r"争议|讲座|刷到|舆情", str(item.get("forum_text") or ""))),
            float(item.get("behavior_ledger", {}).get("urgency") or 0),
        ),
        reverse=True,
    )
    roots = (direct[:2] + career[:1]) if len(direct) >= 2 and career else candidates[:3]
    if len(roots) < 3:
        raise LectureThreadError("expression artifact has fewer than three accepted lecture roots")
    return [
        {
            "thread_id": f"lecture-thread-{index + 1}",
            "root_text": item["forum_text"],
            "topic_id": item["behavior_ledger"].get("topic_id"),
            "stance": item["behavior_ledger"].get("stance"),
            "root_case_id": item.get("case_id"),
        }
        for index, item in enumerate(roots)
    ]


def _lecture_reply_decisions(simulation_artifact_path: str | Path) -> list[dict[str, Any]]:
    artifact = _load_json(simulation_artifact_path)
    if artifact.get("scenario_id") != LECTURE_SCENARIO_ID:
        raise LectureThreadError("thread pilot requires a lecture simulation artifact")
    runs = artifact.get("policy_runs", [])
    if not runs:
        raise LectureThreadError("simulation artifact has no policy runs")
    values: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(runs[0].get("phases", [])):
        for decision in phase.get("agent_decisions", []):
            if (
                decision.get("action") == "comment"
                and decision.get("synthetic_message")
                and decision.get("topic_id") in {"direct_event", "career_spillover"}
            ):
                values.append(
                    {
                        **decision,
                        "phase_id": phase.get("phase_id"),
                        "phase_index": phase_index,
                    }
                )
    if len(values) < 12:
        raise LectureThreadError("fewer than twelve relevant lecture reply decisions")
    return values


def validate_thread_routing(
    raw: dict[str, Any],
    *,
    decisions: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    reply_count: int,
) -> list[dict[str, Any]]:
    routes = raw.get("routes")
    if not isinstance(routes, list) or len(routes) != reply_count:
        raise LectureThreadError("router must select exactly reply_count decisions")
    allowed_threads = {root["thread_id"] for root in roots}
    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for item in routes:
        if not isinstance(item, dict):
            raise LectureThreadError("router route must be an object")
        index = item.get("decision_index")
        if not isinstance(index, int) or not 0 <= index < len(decisions):
            raise LectureThreadError("router decision index is invalid")
        if index in seen_indices:
            raise LectureThreadError("router selected a decision more than once")
        thread_id = item.get("thread_id")
        if thread_id not in allowed_threads:
            raise LectureThreadError("router thread id is invalid")
        if item.get("root_fit") is not True:
            raise LectureThreadError("router may select only decisions that fit their root")
        confidence = str(item.get("confidence") or "")
        if confidence not in {"high", "medium", "low"}:
            raise LectureThreadError("router confidence is invalid")
        seen_indices.add(index)
        selected.append(
            {
                **decisions[index],
                "routed_thread_id": thread_id,
                "router_confidence": confidence,
                "router_reason_tag": str(item.get("reason_tag") or "")[:48],
            }
        )
    if {item["routed_thread_id"] for item in selected} != allowed_threads:
        raise LectureThreadError("router must keep all three local threads represented")
    return selected


def _thread_routing_prompt(
    *, decisions: list[dict[str, Any]], roots: list[dict[str, Any]], reply_count: int
) -> str:
    return json.dumps(
        {
            "task": f"从候选合成行为中选择{reply_count}条，并分配到能自然承接的根帖线程",
            "roots": [
                {
                    "thread_id": root["thread_id"],
                    "topic_id": root.get("topic_id"),
                    "stance": root.get("stance"),
                    "root_text": root["root_text"],
                }
                for root in roots
            ],
            "decisions": [
                {
                    "decision_index": index,
                    "phase_id": decision.get("phase_id"),
                    "topic_id": decision.get("topic_id"),
                    "stance": decision.get("stance"),
                    "archetype": decision.get("archetype_label"),
                    "persona_variant": decision.get("persona_variant_label"),
                    "canonical_message": decision.get("synthetic_message"),
                }
                for index, decision in enumerate(decisions)
            ],
            "rules": [
                "同一个topic_id不等于语义可接入；必须判断该行为能否作为根帖下的自然回复或后续讨论",
                "若候选行为讨论的是根帖从未提出的合作暂停、关系设限或其他新支线，不得仅凭主题相近强行接入",
                "有多余候选时，排除无法自然接入任何根帖的行为，并兼顾阶段、人物与立场差异",
                "每条只能选择一次；三个根帖都必须至少分配一条",
                "root_fit必须为true；若做不到，重新选择其他候选，不能把false项目放进结果",
                "reason_tag只写抽象的衔接理由，不复述帖子文本",
            ],
            "output_schema": {
                "routes": [
                    {
                        "decision_index": "0-based integer, unique",
                        "thread_id": "one of: "
                        + "|".join(root["thread_id"] for root in roots),
                        "root_fit": True,
                        "confidence": "one of: high|medium|low",
                        "reason_tag": "不超过20字的抽象衔接理由",
                    }
                ]
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def route_thread_decisions(
    *,
    client: VectorEngineClient,
    budget: TokenBudget,
    decisions: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    reply_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    prompt = _thread_routing_prompt(
        decisions=decisions, roots=roots, reply_count=reply_count
    )
    usage: Counter[str] = Counter()
    returned_model = client.model
    last_error = ""
    for attempt in range(3):
        current = prompt
        if attempt:
            current += (
                f"\n\n上轮路由契约失败：{last_error[:180]}。"
                "请重新选择并逐条检查根帖语义衔接，不能只看topic_id。"
            )
        raw, attempt_usage, returned_model = _complete_json(
            client=client,
            budget=budget,
            system_prompt=(
                "你是独立的论坛线程语义路由员。你只选择和路由已经生成的合成行为，"
                "不改写文本，不为凑人数把偏题行为接入线程。只输出JSON。"
            ),
            user_prompt=current,
        )
        usage.update(attempt_usage)
        try:
            selected = validate_thread_routing(
                raw,
                decisions=decisions,
                roots=roots,
                reply_count=reply_count,
            )
            return selected, dict(usage), returned_model
        except LectureThreadError as exc:
            last_error = str(exc)
    raise LectureThreadError(f"thread routing failed: {last_error}")


def _function_score(case: dict[str, Any], function: str) -> float:
    message = str(case.get("synthetic_message") or "")
    archetype = str(case.get("archetype_label") or "")
    stance = str(case.get("stance") or "")
    topic = str(case.get("topic_id") or "")
    score = float(case.get("urgency") or 0)
    if function == "evidence_request":
        score += 4 * bool(FACT_QUERY_PATTERN.search(message)) + 2 * (topic == "direct_event")
        score += 2 * ("询问" in archetype or "投诉" in archetype)
    elif function == "evidence_relay":
        score += 5 * bool(re.search(r"转述|完整|原文|表述|事实|信息", message))
        score += 2 * (topic == "direct_event") + 1 * ("规范" in archetype)
    elif function == "irony_or_meme":
        score += 5 * bool(CRITICAL_PATTERN.search(message)) + 2 * ("接梗" in str(case.get("persona_variant_label")))
    elif function == "amplification_or_pile_on":
        score += 4 * (stance in {"critical", "concerned"}) + 2 * bool(re.search(r"同感|堵|不该|影响", message))
    elif function == "emotional_support":
        score += 4 * ("情绪" in archetype) + 2 * bool(re.search(r"堵|影响|压力|同感", message))
    elif function == "challenge_or_correction":
        score += 3 * (stance == "critical") + 2 * bool(CRITICAL_PATTERN.search(message))
    elif function == "clarification":
        score += 4 * bool(CLARIFICATION_PATTERN.search(message))
    elif function == "advice_or_action":
        score += 2 * bool(re.search(r"请|能否|如果|怎样", message))
    elif function == "personal_experience":
        score += 2 * bool(re.search(r"我|自己|两天了", message))
    return score


def plan_thread_cases(
    *,
    decisions: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    reference_primary_shares: dict[str, float],
    reply_count: int = 12,
    seed: int = 20260722,
) -> list[dict[str, Any]]:
    """Plan a coverage pilot from reference data only, never held-out labels."""
    if not 12 <= reply_count <= 16:
        raise LectureThreadError("reply_count must be between 12 and 16")
    pool = list(decisions)
    rng = random.Random(seed)
    target_specs = [(function, "minimum_coverage_probe") for function in KEY_FUNCTIONS]
    population = sorted(THREAD_FUNCTIONS)
    weights = [max(0.001, float(reference_primary_shares.get(key, 0))) for key in population]
    target_specs.extend(
        (function, "reference_weighted_draw")
        for function in rng.choices(
            population, weights=weights, k=reply_count - len(target_specs)
        )
    )
    # Key functions are deliberate minimum-coverage probes, not frequency estimates.
    assignments: list[tuple[dict[str, Any], str, str]] = []
    compatibility_threshold = {
        "advice_or_action": 3.0,
        "clarification": 2.0,
        "emotional_support": 2.0,
        "personal_experience": 1.8,
    }
    for function, source in target_specs:
        compatible = max(pool, key=lambda case: (_function_score(case, function), rng.random()))
        if (
            source == "reference_weighted_draw"
            and _function_score(compatible, function) < compatibility_threshold.get(function, 0.0)
        ):
            replacement = max(
                THREAD_FUNCTIONS,
                key=lambda candidate: (
                    _function_score(compatible, candidate)
                    + 0.35 * float(reference_primary_shares.get(candidate, 0))
                ),
            )
            if replacement != function:
                function = replacement
                source = "persona_compatibility_override_of_reference_draw"
        pool.remove(compatible)
        assignments.append((compatible, function, source))
    assignments.sort(key=lambda value: (value[0]["phase_index"], -float(value[0].get("urgency") or 0)))

    planned: list[dict[str, Any]] = []
    per_thread_turn: Counter[str] = Counter()
    roots_by_id = {root["thread_id"]: root for root in roots}
    for index, (decision, function, assignment_source) in enumerate(assignments):
        routed_thread_id = decision.get("routed_thread_id")
        if routed_thread_id in roots_by_id:
            root = roots_by_id[routed_thread_id]
        else:
            matching = [root for root in roots if root["topic_id"] == decision.get("topic_id")]
            root = min(matching or roots, key=lambda value: per_thread_turn[value["thread_id"]])
        per_thread_turn[root["thread_id"]] += 1
        planned.append(
            {
                **decision,
                "case_id": hashlib.sha256(
                    f"{decision.get('synthetic_agent_id')}|{decision.get('phase_id')}|{function}".encode()
                ).hexdigest()[:16],
                "turn_index": index,
                "thread_id": root["thread_id"],
                "root_text": root["root_text"],
                "root_case_id": root["root_case_id"],
                "target_function": function,
                "function_assignment_source": assignment_source,
            }
        )
    return planned


ALLOCATION_BOUNDARIES = {
    "clear",
    "request_vs_clarification",
    "amplification_vs_support",
    "other",
}


def validate_function_allocation(
    raw: dict[str, Any], cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    assignments = raw.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(cases):
        raise LectureThreadError("allocator must return every case exactly once")
    validated: list[dict[str, Any]] = []
    for expected_index, item in enumerate(assignments):
        if not isinstance(item, dict) or item.get("index") != expected_index:
            raise LectureThreadError("allocator assignments must be ordered by index")
        assigned = str(item.get("assigned_function") or "")
        if assigned not in THREAD_FUNCTIONS:
            raise LectureThreadError(f"allocator returned unsupported function: {assigned}")
        if item.get("fit_to_canonical") is not True:
            raise LectureThreadError("allocator must choose a function that fits the canonical ledger")
        boundary = str(item.get("boundary") or "")
        if boundary not in ALLOCATION_BOUNDARIES:
            raise LectureThreadError("allocator boundary label is invalid")
        confidence = str(item.get("confidence") or "")
        if confidence not in {"high", "medium", "low"}:
            raise LectureThreadError("allocator confidence is invalid")
        parent_anchor = str(item.get("parent_anchor") or "")
        if parent_anchor not in {"root", "latest_reply"}:
            raise LectureThreadError("allocator parent anchor is invalid")
        validated.append(
            {
                "index": expected_index,
                "assigned_function": assigned,
                "fit_to_canonical": True,
                "boundary": boundary,
                "confidence": confidence,
                "reason_tag": str(item.get("reason_tag") or "")[:48],
                "parent_anchor": parent_anchor,
            }
        )
    return validated


def _function_allocation_prompt(cases: list[dict[str, Any]]) -> str:
    previous_by_thread: dict[str, str] = {}
    prompt_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        thread_id = str(case["thread_id"])
        prompt_cases.append(
            {
                "index": index,
                "root": case["root_text"],
                "previous_canonical_in_thread": previous_by_thread.get(thread_id),
                "topic_id": case.get("topic_id"),
                "stance": case.get("stance"),
                "archetype": case.get("archetype_label"),
                "persona_variant": case.get("persona_variant_label"),
                "canonical_message": case.get("synthetic_message"),
                "proposed_function_from_reference_prior": case["target_function"],
                "proposal_source": case["function_assignment_source"],
            }
        )
        previous_by_thread[thread_id] = str(case.get("synthetic_message") or "")
    return json.dumps(
        {
            "task": "为每条合成行为底稿确定最自然的主要回复功能",
            "function_definitions": FUNCTION_LABELS,
            "cases": prompt_cases,
            "rules": [
                "按行为底稿真正执行的主要动作分类；真实样本比例只是先验，不能强迫不匹配的人物执行某功能",
                "纯索要原文、来源、公开说明或核验状态属于evidence_request",
                "clarification必须澄清具体所指、概念边界、受理主体或已有说法的含义，不能只是索要来源",
                "amplification_or_pile_on必须附和、重复或推高父帖已有立场的可见度",
                "直接安慰对方、劝其别自责或别怀疑自己属于emotional_support，不属于放大",
                "第一人称讲自己受到的影响属于personal_experience；给下一步做法属于advice_or_action",
                "evidence_relay必须转述信息状态并保留已核验/未核验边界",
                "四个重点功能应优先放到确实匹配的案例上；若本种子没有自然匹配，允许缺席，绝不能为覆盖率硬贴标签",
                "同时选择parent_anchor：canonical_message若直接延续previous_canonical_in_thread则用latest_reply；若切换回根帖的另一子问题或与上一条不直接相接则用root",
                "不能因为同属一个thread或topic就默认latest_reply；parent_anchor必须保证直接父子回复自然",
                "每条assigned_function都必须与canonical_message相符，fit_to_canonical必须为true",
                "reason_tag只写抽象动作标签，不复述整句底稿",
            ],
            "output_schema": {
                "assignments": [
                    {
                        "index": "0-based integer, ordered",
                        "assigned_function": "one of: " + "|".join(sorted(THREAD_FUNCTIONS)),
                        "fit_to_canonical": True,
                        "boundary": "one of: " + "|".join(sorted(ALLOCATION_BOUNDARIES)),
                        "confidence": "one of: high|medium|low",
                        "reason_tag": "不超过20字的抽象理由",
                        "parent_anchor": "one of: root|latest_reply",
                    }
                ]
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def allocate_thread_functions(
    *,
    client: VectorEngineClient,
    budget: TokenBudget,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    prompt = _function_allocation_prompt(cases)
    usage: Counter[str] = Counter()
    returned_model = client.model
    last_error = ""
    for attempt in range(3):
        current = prompt
        if attempt:
            current += (
                f"\n\n上轮分配契约失败：{last_error[:180]}。"
                "请重新检查边界并逐条按index输出；不要为凑重点功能覆盖而牺牲底稿匹配。"
            )
        raw, attempt_usage, returned_model = _complete_json(
            client=client,
            budget=budget,
            system_prompt=(
                "你是独立的对话行为功能编码员。你只分配互动功能，不生成回复文本。"
                "必须优先服从行为底稿语义，不能为凑比例硬贴标签。只输出JSON。"
            ),
            user_prompt=current,
        )
        usage.update(attempt_usage)
        try:
            assignments = validate_function_allocation(raw, cases)
            revised: list[dict[str, Any]] = []
            for case, assignment in zip(cases, assignments, strict=True):
                proposed = case["target_function"]
                proposed_source = case["function_assignment_source"]
                revised.append(
                    {
                        **case,
                        "proposed_function": proposed,
                        "proposed_function_source": proposed_source,
                        "target_function": assignment["assigned_function"],
                        "function_assignment_source": "independent_llm_boundary_allocator",
                        "allocator_boundary": assignment["boundary"],
                        "allocator_confidence": assignment["confidence"],
                        "allocator_reason_tag": assignment["reason_tag"],
                        "parent_anchor": assignment["parent_anchor"],
                    }
                )
            return revised, dict(usage), returned_model
        except LectureThreadError as exc:
            last_error = str(exc)
    raise LectureThreadError(f"function allocation failed: {last_error}")


def validate_thread_candidates(
    raw: dict[str, Any], *, target_function: str, exemplars: list[str]
) -> list[dict[str, Any]]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 4:
        raise LectureThreadError("thread renderer must return 2 to 4 candidates")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    for item in candidates:
        if not isinstance(item, dict):
            rejected["not_object"] += 1
            continue
        text = " ".join(str(item.get("text") or "").strip().split())
        if not 1 <= len(text) <= 40 or text in seen:
            rejected["text_length_or_duplicate"] += 1
            continue
        if target_function == "signal_only" and len(text) > 12:
            rejected["signal_too_long"] += 1
            continue
        if target_function == "irony_or_meme" and not IRONY_PATTERN.search(text):
            rejected["irony_structure_missing"] += 1
            continue
        interaction = item.get("interaction_function")
        if isinstance(interaction, list) and len(interaction) == 1:
            interaction = interaction[0]
        if interaction != target_function:
            rejected["interaction_function"] += 1
            continue
        register = item.get("style_register")
        if isinstance(register, list) and len(register) == 1:
            register = register[0]
        if not isinstance(register, str) or register not in ALLOWED_REGISTERS:
            rejected["style_register"] += 1
            continue
        new_claims = item.get("new_factual_claims")
        if not isinstance(new_claims, list) or new_claims:
            rejected["new_factual_claims"] += 1
            continue
        if URL_PATTERN.search(text) or LONG_DIGIT_PATTERN.search(text):
            rejected["url_or_digits"] += 1
            continue
        if any(pattern.search(text) for pattern in CONTACT_PATTERNS):
            rejected["contact"] += 1
            continue
        if any(_longest_common_substring(text, example) >= 12 for example in exemplars):
            rejected["copy_guard"] += 1
            continue
        evidence_status_value = item.get("evidence_status")
        if isinstance(evidence_status_value, list) and len(evidence_status_value) == 1:
            evidence_status_value = evidence_status_value[0]
        evidence_status = str(evidence_status_value or "not_applicable")
        if target_function == "evidence_relay" and evidence_status != "unverified_relay":
            rejected["evidence_status"] += 1
            continue
        if target_function != "evidence_relay" and evidence_status not in {
            "not_applicable",
            "unverified_relay",
        }:
            rejected["evidence_status"] += 1
            continue
        validated.append(
            {
                "text": text,
                "interaction_function": target_function,
                "style_register": register,
                "evidence_status": evidence_status,
                "new_factual_claims": [],
            }
        )
        seen.add(text)
    if len(validated) < 2:
        reason = ",".join(f"{key}={value}" for key, value in sorted(rejected.items()))
        raise LectureThreadError(
            f"fewer than two safe thread candidates remained ({reason or 'unknown'})"
        )
    return validated


def validate_thread_judgment(raw: dict[str, Any], candidate_count: int) -> dict[str, Any]:
    index = raw.get("selected_candidate_index")
    if not isinstance(index, int) or not 0 <= index < candidate_count:
        raise LectureThreadError("thread judge selected invalid candidate")
    keys = (
        "function_fidelity",
        "stance_preservation",
        "parent_relevance",
        "thread_coherence",
        "no_new_factual_claim",
        "forum_naturalness",
    )
    values: dict[str, bool] = {}
    for key in keys:
        if not isinstance(raw.get(key), bool):
            raise LectureThreadError(f"thread judge field {key} must be boolean")
        values[key] = raw[key]
    assessments = raw.get("candidate_assessments")
    approved: list[int] = []
    if isinstance(assessments, list) and len(assessments) == candidate_count:
        for expected, item in enumerate(assessments):
            if not isinstance(item, dict) or item.get("index") != expected:
                raise LectureThreadError("thread assessments must be ordered")
            if all(item.get(key) is True for key in keys):
                approved.append(expected)
    accepted = all(values.values())
    return {
        **values,
        "selected_candidate_index": index,
        "accepted": accepted,
        "approved_candidate_indices": approved,
        "rejection_reasons": [str(value)[:160] for value in raw.get("rejection_reasons", [])[:5]],
        "unresolved_risk": str(raw.get("unresolved_risk") or "")[:200],
    }


def _thread_render_prompt(
    *,
    case: dict[str, Any],
    persona: dict[str, Any],
    parent_text: str,
    recent_replies: list[str],
    style_profile: dict[str, Any],
    exemplars: list[str],
) -> str:
    variant = persona.get("variants", {}).get(case.get("persona_variant_id"), {})
    return json.dumps(
        {
            "task": "在冻结的高层行为意图之上，生成指定互动功能的校园论坛回复",
            "thread": {
                "root": case["root_text"],
                "direct_parent": parent_text,
                "recent_replies_in_order": recent_replies[-4:],
            },
            "behavior_ledger": {
                "topic_id": case.get("topic_id"),
                "stance": case.get("stance"),
                "urgency": case.get("urgency"),
                "canonical_message": case.get("synthetic_message"),
                "assigned_interaction_function": case["target_function"],
                "function_definition": FUNCTION_LABELS[case["target_function"]],
            },
            "persona": {
                "archetype": persona.get("label"),
                "variant": variant.get("label"),
                "language_tendencies": persona.get("language_tendencies", []),
                "variant_language_style": variant.get("language_style"),
            },
            "held_out_aggregate_style": style_profile,
            "non_shock_ephemeral_style_examples": exemplars,
            "rules": [
                "必须直接回应direct_parent，并与recent_replies形成自然连续对话",
                "保留底稿的主题和立场，但不要把底稿压缩成行政小作文",
                "不得新增讲座原话、现场细节、机构回应、时间、人数、链接或核验结论",
                "evidence_request只能求证；evidence_relay只能转述线程内已有信息并标unverified_relay",
                "evidence_relay应使用‘我看到/我这边/目前线程里’标明信息视角，不得把未核验状态写成客观结论",
                "irony_or_meme只能讽刺观点，不能辱骂、围攻个人或群体",
                "amplification_or_pile_on只能放大已有立场，不得制造新指控",
                "clarification要澄清具体所指、概念或受理主体；纯粹索要原文或来源应是evidence_request",
                "signal_only必须是不超过10字的纯态度或占位信号，不展开新主张",
                "若recent_replies刚使用过某个表情，下一条不要重复该表情",
                "不得照抄例句，连续复用例句少于12字",
                "生成三条长度有差异的候选，回复优先短、口语、单中心",
            ],
            "output_schema": {
                "candidates": [
                    {
                        "text": "1至40字",
                        "interaction_function": case["target_function"],
                        "style_register": "one of: fragment|colloquial|mixed|standard",
                        "evidence_status": (
                            "unverified_relay"
                            if case["target_function"] == "evidence_relay"
                            else "not_applicable"
                        ),
                        "new_factual_claims": [],
                    }
                ]
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _thread_judge_prompt(
    *,
    case: dict[str, Any],
    parent_text: str,
    recent_replies: list[str],
    candidates: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "task": "独立审查线程回复候选",
            "thread": {
                "root": case["root_text"],
                "direct_parent": parent_text,
                "recent_replies_in_order": recent_replies[-4:],
            },
            "ledger": {
                "canonical_message": case.get("synthetic_message"),
                "stance": case.get("stance"),
                "target_function": case["target_function"],
                "function_definition": FUNCTION_LABELS[case["target_function"]],
            },
            "candidates": candidates,
            "rules": [
                "逐条检查功能、立场、父帖承接、线程连贯、无新增事实和论坛自然度",
                "短回复不必复述底稿全部次要细节，但不能改变核心立场",
                "evidence_relay只能标注未核验的转述状态，不能伪造核验",
                "若evidence_relay把‘未看到’写成‘客观不存在’，no_new_factual_claim必须为false",
                "反讽和附和不得变成人身攻击或群体围攻",
                "clarification若只是索要来源而没有澄清对象，function_fidelity必须为false",
                "signal_only超过12字或展开具体论点，function_fidelity必须为false",
                "只选全部检查通过的候选；没有则accepted字段自然由布尔项体现",
            ],
            "output_schema": {
                "selected_candidate_index": "0-based integer",
                "function_fidelity": "boolean",
                "stance_preservation": "boolean",
                "parent_relevance": "boolean",
                "thread_coherence": "boolean",
                "no_new_factual_claim": "boolean",
                "forum_naturalness": "boolean",
                "rejection_reasons": [],
                "unresolved_risk": "",
                "candidate_assessments": [
                    {
                        "index": "0-based ordered",
                        "function_fidelity": "boolean",
                        "stance_preservation": "boolean",
                        "parent_relevance": "boolean",
                        "thread_coherence": "boolean",
                        "no_new_factual_claim": "boolean",
                        "forum_naturalness": "boolean",
                    }
                ],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _choose_parent(case: dict[str, Any], recent: list[dict[str, Any]]) -> str:
    anchor = case.get("parent_anchor")
    if not recent or anchor == "root":
        return case["root_text"]
    if anchor == "latest_reply":
        return recent[-1]["forum_text"]
    if case["target_function"] in {"evidence_request", "evidence_relay"}:
        return case["root_text"]
    return recent[-1]["forum_text"]


def _render_thread(
    *,
    client: VectorEngineClient,
    budget: TokenBudget,
    cases: list[dict[str, Any]],
    personas: dict[str, dict[str, Any]],
    style: StyleCorpus,
    seed: int,
) -> list[dict[str, Any]]:
    recent: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for position, case in enumerate(cases):
        parent = _choose_parent(case, recent)
        recent_texts = [item["forum_text"] for item in recent[-4:]]
        exemplars = style.examples("reply", seed=seed + position * 101, limit=10)
        persona = personas.get(str(case.get("archetype_id") or ""))
        if persona is None:
            raise LectureThreadError(f"persona missing for {case.get('archetype_id')}")
        usage: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []
        error = ""
        returned_model = client.model
        render_prompt = _thread_render_prompt(
            case=case,
            persona=persona,
            parent_text=parent,
            recent_replies=recent_texts,
            style_profile=style.profiles["reply"],
            exemplars=exemplars,
        )
        for attempt in range(2):
            prompt = render_prompt + (f"\n\n上轮失败：{error[:150]}。请重做。" if attempt else "")
            raw, attempt_usage, returned_model = _complete_json(
                client=client,
                budget=budget,
                system_prompt=(
                    "你是匿名校园论坛的线程回复渲染器。互动功能已经冻结；"
                    "不得新增事件事实、原话或治理承诺。只输出JSON。"
                ),
                user_prompt=prompt,
            )
            usage.update(attempt_usage)
            try:
                candidates = validate_thread_candidates(
                    raw, target_function=case["target_function"], exemplars=exemplars
                )
                if recent_texts and EMOJI_PATTERN.search(recent_texts[-1]):
                    without_emoji = [
                        item for item in candidates if not EMOJI_PATTERN.search(item["text"])
                    ]
                    if len(without_emoji) < 2:
                        raise LectureThreadError(
                            "fewer than two candidates remained after consecutive emoji guard"
                        )
                    candidates = without_emoji
                break
            except LectureThreadError as exc:
                error = str(exc)
        judgment: dict[str, Any] | None = None
        judge_model = client.model
        if candidates:
            judge_prompt = _thread_judge_prompt(
                case=case,
                parent_text=parent,
                recent_replies=recent_texts,
                candidates=candidates,
            )
            for attempt in range(2):
                prompt = judge_prompt + (f"\n\n上轮失败：{error[:150]}。请重做。" if attempt else "")
                raw, attempt_usage, judge_model = _complete_json(
                    client=client,
                    budget=budget,
                    system_prompt=(
                        "你是独立的线程语义审稿人。不得因口语化而放过功能错误、"
                        "事实新增或父帖不相关。只输出JSON。"
                    ),
                    user_prompt=prompt,
                )
                usage.update(attempt_usage)
                try:
                    judgment = validate_thread_judgment(raw, len(candidates))
                    break
                except LectureThreadError as exc:
                    error = str(exc)
        if judgment is None:
            judgment = {
                "accepted": False,
                "selected_candidate_index": 0,
                "rejection_reasons": ["thread_contract_failed_after_retry"],
                "unresolved_risk": error[:200],
            }
        if candidates and judgment["accepted"]:
            approved = judgment.get("approved_candidate_indices") or [judgment["selected_candidate_index"]]
            # Among fully approved replies, keep the one closest to the observed reply median.
            median = int(style.profiles["reply"]["length_chars"]["median"])
            selected_index = min(approved, key=lambda index: abs(len(candidates[index]["text"]) - median))
            judgment["selected_candidate_index"] = selected_index
            judgment["length_calibrated_selection"] = selected_index != raw.get("selected_candidate_index")
            forum_text = candidates[selected_index]["text"]
            status = "accepted"
        else:
            forum_text = str(case.get("synthetic_message") or "")
            status = "fallback_to_canonical"
        result = {
            "case_id": case["case_id"],
            "thread_id": case["thread_id"],
            "turn_index": case["turn_index"],
            "phase_id": case.get("phase_id"),
            "synthetic_agent_id": case.get("synthetic_agent_id"),
            "archetype_id": case.get("archetype_id"),
            "archetype_label": case.get("archetype_label"),
            "persona_variant_id": case.get("persona_variant_id"),
            "persona_variant_label": case.get("persona_variant_label"),
            "root_synthetic_message": case["root_text"],
            "parent_synthetic_message": parent,
            "recent_reply_count_seen": len(recent_texts),
            "behavior_ledger": {
                "topic_id": case.get("topic_id"),
                "stance": case.get("stance"),
                "urgency": case.get("urgency"),
                "canonical_message": case.get("synthetic_message"),
                "assigned_interaction_function": case["target_function"],
                "function_assignment_source": case["function_assignment_source"],
                "proposed_interaction_function": case.get("proposed_function"),
                "proposed_function_source": case.get("proposed_function_source"),
                "allocator_boundary": case.get("allocator_boundary"),
                "allocator_confidence": case.get("allocator_confidence"),
                "allocator_reason_tag": case.get("allocator_reason_tag"),
                "parent_anchor": case.get("parent_anchor"),
            },
            "render_candidates": candidates,
            "judge": judgment,
            "render_status": status,
            "forum_text": forum_text,
            "provider": {
                "render_model": returned_model,
                "judge_model": judge_model,
                "usage": {key: int(usage[key]) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            },
        }
        recent.append(result)
        results.append(result)
    return results


def _distribution(values: list[str]) -> dict[str, float]:
    counts = Counter(values)
    total = max(1, len(values))
    return {key: round(counts.get(key, 0) / total, 4) for key in sorted(THREAD_FUNCTIONS)}


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    middle = {key: (left.get(key, 0) + right.get(key, 0)) / 2 for key in keys}

    def kl(one: dict[str, float]) -> float:
        return sum(
            value * math.log2(value / middle[key])
            for key, value in one.items()
            if value > 0 and middle[key] > 0
        )

    return round((kl(left) + kl(right)) / 2, 4)


def run_lecture_thread_ecology_pilot(
    *,
    posts_path: str | Path,
    comments_path: str | Path,
    simulation_artifact_path: str | Path,
    expression_artifact_path: str | Path,
    persona_release_path: str | Path,
    output_path: str | Path,
    observed_evidence_path: str | Path | None = None,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    sample_per_split: int = 48,
    batch_size: int = 8,
    reply_count: int = 12,
    seed: int = 20260722,
    max_concurrency: int = 8,
    max_total_tokens: int = 180000,
    max_output_tokens: int = 800,
    timeout_seconds: float = 120.0,
    allow_provider_text: bool = False,
) -> dict[str, Any]:
    """Run aggregate coding plus a three-thread synthetic lecture pilot."""
    if not allow_provider_text:
        raise LectureThreadError(
            "raw parent/reply pairs may be sent only with allow_provider_text=True"
        )
    if not 1 <= max_concurrency <= 32:
        raise LectureThreadError("max_concurrency must be between 1 and 32")
    style = build_style_corpus(
        posts_path=posts_path,
        comments_path=comments_path,
        definitions_path=definitions_path,
        seed=seed + 1,
    )
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
    reused_coding_identity: dict[str, Any] | None = None
    if observed_evidence_path is not None:
        prior = _load_json(observed_evidence_path)
        if (
            prior.get("schema_version") != "lecture-thread-ecology-pilot-v1"
            or prior.get("status") != "complete"
        ):
            raise LectureThreadError("reused observed evidence artifact is incompatible")
        coding = prior.get("observed_function_evidence")
        observed_sampling = prior.get("observed_pair_sampling")
        if not isinstance(coding, dict) or not all(
            split in coding for split in ("reference", "lecture_holdout")
        ):
            raise LectureThreadError("reused artifact has no complete observed coding")
        if not isinstance(observed_sampling, dict):
            raise LectureThreadError("reused artifact has no sampling evidence")
        coding_usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        coding_model = str(prior.get("provider", {}).get("coding_returned_model") or model)
        reused_coding_identity = _file_fingerprint(observed_evidence_path)
    else:
        sample = build_thread_pair_sample(
            posts_path=posts_path,
            comments_path=comments_path,
            definitions_path=definitions_path,
            sample_per_split=sample_per_split,
            seed=seed,
        )
        coding, _ephemeral_labels, coding_usage, coding_model = classify_pair_sample(
            client=client,
            budget=budget,
            sample=sample,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        observed_sampling = {
            "sample_per_split": sample_per_split,
            "counts": sample.counts,
        }
    # Planning sees reference aggregates only. The held-out labels remain evaluation-only.
    roots = _load_lecture_roots(expression_artifact_path)
    decisions = _lecture_reply_decisions(simulation_artifact_path)
    decisions, routing_usage, routing_model = route_thread_decisions(
        client=client,
        budget=budget,
        decisions=decisions,
        roots=roots,
        reply_count=reply_count,
    )
    planned = plan_thread_cases(
        decisions=decisions,
        roots=roots,
        reference_primary_shares=coding["reference"]["primary_shares"],
        reply_count=reply_count,
        seed=seed,
    )
    planned, allocation_usage, allocation_model = allocate_thread_functions(
        client=client,
        budget=budget,
        cases=planned,
    )
    simulation_identity = _file_fingerprint(simulation_artifact_path)
    simulation_artifact = _load_json(simulation_artifact_path)
    simulation_seed = simulation_artifact.get("execution", {}).get(
        "seed_used_for_initial_population_only"
    )
    for case in planned:
        case["case_id"] = hashlib.sha256(
            (
                f"{simulation_identity['sha256'][:16]}|{case.get('synthetic_agent_id')}|"
                f"{case.get('phase_id')}|{case.get('synthetic_message')}|{case['target_function']}"
            ).encode("utf-8")
        ).hexdigest()[:16]
    personas = _persona_views(persona_release_path)
    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in planned:
        by_thread[case["thread_id"]].append(case)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(by_thread), max_concurrency)) as executor:
        futures = {
            executor.submit(
                _render_thread,
                client=client,
                budget=budget,
                cases=sorted(cases, key=lambda item: item["turn_index"]),
                personas=personas,
                style=style,
                seed=seed + index * 1009,
            ): thread_id
            for index, (thread_id, cases) in enumerate(sorted(by_thread.items()))
        }
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: item["turn_index"])
    synthetic_distribution = _distribution(
        [item["behavior_ledger"]["assigned_interaction_function"] for item in results]
    )
    generation_usage = {
        key: sum(item["provider"]["usage"][key] for item in results)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    total_usage = {
        key: (
            int(coding_usage.get(key, 0))
            + int(routing_usage.get(key, 0))
            + int(allocation_usage.get(key, 0))
            + int(generation_usage.get(key, 0))
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    roots_public = [
        {
            "thread_id": root["thread_id"],
            "topic_id": root["topic_id"],
            "stance": root["stance"],
            "root_synthetic_message": root["root_text"],
        }
        for root in roots
    ]
    artifact = {
        "schema_version": "lecture-thread-ecology-pilot-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "method": {
            "architecture": "frozen_high_level_decision_then_semantic_thread_routing_then_data_grounded_reply_function_then_parent_conditioned_expression_then_independent_judge",
            "function_coder_version": "1.0.0",
            "thread_router_version": "1.0.0",
            "function_allocator_version": "1.0.0",
            "thread_renderer_version": "1.1.0",
            "reply_scope": "three_local_synthetic_threads_not_complete_forum_diffusion_graph",
            "coverage_probe": True,
            "coverage_probe_is_frequency_estimate": False,
            "direct_reading_required": True,
            "automatic_distribution_diagnostics_are_verdicts": False,
        },
        "source_boundaries": {
            "raw_paths_persisted": False,
            "raw_forum_text_persisted": False,
            "raw_ids_persisted": False,
            "provider_prompts_persisted": False,
            "reference_pairs_used_for_function_planning": True,
            "lecture_holdout_pairs_used_for_generation": False,
            "lecture_holdout_pairs_used_for_evaluation_only": True,
            "observed_function_coding_reused_without_raw_text": observed_evidence_path is not None,
            "contact_url_or_long_digit_text_sent": False,
            "max_replies_per_observed_thread_before_sampling": 4,
        },
        "input_evidence": {
            "simulation_artifact": simulation_identity,
            "expression_artifact": _file_fingerprint(expression_artifact_path),
            "persona_release": _file_fingerprint(persona_release_path),
            "definitions": _file_fingerprint(definitions_path),
            "reused_observed_coding_artifact": reused_coding_identity,
            "raw_csvs_fingerprinted_or_released": False,
        },
        "observed_pair_sampling": observed_sampling,
        "observed_function_evidence": coding,
        "synthetic_threads": roots_public,
        "provider": {
            "requested_model": model,
            "coding_returned_model": coding_model,
            "routing_returned_model": routing_model,
            "allocation_returned_model": allocation_model,
            "base_url": base_url,
            "api_key_or_value_persisted": False,
            "usage": total_usage,
        },
        "summary": {
            "reply_case_count": len(results),
            "simulation_population_seed": simulation_seed,
            "thread_count": len(roots_public),
            "router_confidence_counts": dict(
                sorted(Counter(case.get("router_confidence") or "unknown" for case in planned).items())
            ),
            "router_thread_counts": dict(
                sorted(Counter(case["thread_id"] for case in planned).items())
            ),
            "accepted_count": sum(item["render_status"] == "accepted" for item in results),
            "fallback_count": sum(item["render_status"] != "accepted" for item in results),
            "allocator_override_count": sum(
                item["behavior_ledger"]["proposed_interaction_function"]
                != item["behavior_ledger"]["assigned_interaction_function"]
                for item in results
            ),
            "allocator_boundary_counts": dict(
                sorted(
                    Counter(
                        item["behavior_ledger"].get("allocator_boundary") or "unknown"
                        for item in results
                    ).items()
                )
            ),
            "allocator_parent_anchor_counts": dict(
                sorted(
                    Counter(
                        item["behavior_ledger"].get("parent_anchor") or "unknown"
                        for item in results
                    ).items()
                )
            ),
            "key_function_case_counts": {
                key: sum(
                    item["behavior_ledger"]["assigned_interaction_function"] == key
                    for item in results
                )
                for key in KEY_FUNCTIONS
            },
            "synthetic_primary_shares": synthetic_distribution,
            "reference_vs_synthetic_js_divergence_diagnostic": _js_divergence(
                coding["reference"]["primary_shares"], synthetic_distribution
            ),
            "holdout_vs_synthetic_js_divergence_diagnostic": _js_divergence(
                coding["lecture_holdout"]["primary_shares"], synthetic_distribution
            ),
        },
        "cases": results,
        "validity": {
            "high_level_agent_topic_stance_frozen": True,
            "reply_function_is_new_second_stage_behavior": True,
            "historical_holdout_not_fit": True,
            "frequency_calibration_status": "not_calibrated_small_coverage_probe",
            "surface_fidelity_status": "pending_direct_reading",
            "causal_claim": "none",
            "full_diffusion_graph_claim": False,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return artifact
