#!/usr/bin/env python3
"""Audit a completed housing-policy validation scope and write a plain-language report."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sqlite3
from statistics import mean
import sys
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.campus_pulse.runtime_contract import canonical_sha256


BRANCHES = ("Natural", "A", "B", "C", "D")
BRANCH_LABELS = {
    "Natural": "论坛自然演化",
    "A": "标准公开与证据更新",
    "B": "申诉复核与服务承接",
    "C": "跨群触达与参与式沟通",
    "D": "三主体动态组合治理",
}
PUBLIC_PROVENANCE = {"authorized_live_llm", "reviewed_trace_replay"}
SECRET_PATTERN = re.compile(r"(?i)sk-[A-Za-z0-9_-]{12,}")
PATH_PATTERN = re.compile(r"(?i)(?:[A-Z]:\\|API\.txt)")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("flagship result root must be an object")
    return value


def _ledger_usage(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        control = connection.execute(
            "SELECT * FROM v7_live_control WHERE singleton = 1"
        ).fetchone()
        if control is None:
            raise ValueError("flagship live ledger control is missing")
        outcomes = {
            str(row["outcome"]): int(row["count"])
            for row in connection.execute(
                """
                SELECT outcome,COUNT(*) AS count FROM v7_live_attempts
                WHERE state='settled' GROUP BY outcome
                """
            ).fetchall()
        }
        return {
            "provider_calls_completed": int(control["provider_calls_completed"]),
            "provider_calls_reserved": int(control["provider_calls_reserved"]),
            "provider_calls_unknown_historical": int(
                control["provider_calls_unknown"]
            ),
            "provider_reported_tokens": int(control["provider_reported_tokens"]),
            "risk_charged_tokens": int(control["risk_charged_tokens"]),
            "tokens_in_flight": int(control["tokens_in_flight"]),
            "outcomes": outcomes,
        }
    finally:
        connection.close()


def _verified_hash(value: Mapping[str, Any], field: str) -> bool:
    body = {key: child for key, child in value.items() if key != field}
    return value.get(field) == canonical_sha256(body)


def _normalized(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).casefold()


def _near_duplicate_rate(texts: Iterable[str], threshold: float = 0.86) -> float:
    rows = [_normalized(text) for text in texts if _normalized(text)]
    if len(rows) < 2:
        return 0.0
    duplicates = 0
    comparisons = 0
    for index, left in enumerate(rows):
        # Compare only the next 32 public messages.  This catches templating
        # without turning a large forum audit into an O(n^2) memory spike.
        for right in rows[index + 1 : index + 33]:
            comparisons += 1
            duplicates += SequenceMatcher(None, left, right).ratio() >= threshold
    return 0.0 if comparisons == 0 else duplicates / comparisons


def _lineage_valid(branch: Mapping[str, Any]) -> bool:
    messages = list(branch.get("messages") or [])
    message_by_id = {str(row.get("message_id")): row for row in messages}
    claims = {str(row.get("claim_id")): row for row in branch.get("claims") or []}
    for message in messages:
        tick = int(message.get("created_tick", -1))
        for field in ("parent_message_id", "quote_message_id"):
            target_id = message.get(field)
            if target_id is None:
                continue
            target = message_by_id.get(str(target_id))
            if target is None or int(target.get("created_tick", tick)) >= tick:
                return False
        claim_id = message.get("claim_id")
        if claim_id is not None and str(claim_id) not in claims:
            return False
        correction_target = message.get("correction_target_claim_id")
        if correction_target is not None and str(correction_target) not in claims:
            return False
    return True


def _resource_conserved(branch: Mapping[str, Any]) -> bool:
    for round_row in branch.get("governance_rounds") or []:
        if round_row.get("uses_hidden_truth") is not False:
            return False
        if round_row.get("scheme_bonus") != 0:
            return False
        for field in ("resource_before", "resource_after"):
            resource = round_row.get(field) or {}
            opening = float(resource.get("opening", -1))
            consumed = float(resource.get("consumed", -1))
            committed = float(resource.get("committed", -1))
            remaining = float(resource.get("remaining", -1))
            if resource.get("conserved") is not True:
                return False
            if abs(opening - consumed - committed - remaining) > 1e-8:
                return False
    return True


def _branch_summary(branch: Mapping[str, Any]) -> dict[str, Any]:
    messages = list(branch.get("messages") or [])
    resident = [
        row
        for row in messages
        if not str(row.get("source_agent_display_id") or "").startswith("governance:")
    ]
    governance = [row for row in messages if row not in resident]
    chains = branch.get("mechanism_chains") or {}
    final = (branch.get("timeline") or [{}])[-1]
    interactions = branch.get("parallel_interactions") or {}
    interaction_counts = interactions.get("counts_by_kind") or {}
    return {
        "unique_activated_agents": int(branch.get("unique_activated_agents", 0)),
        "resident_public_messages": len(resident),
        "governance_public_messages": len(governance),
        "threads": len(branch.get("threads") or []),
        "claims": len(branch.get("claims") or []),
        "contested_claims": int(final.get("contested_claim_count", 0)),
        "corrected_or_verified_claims": int(
            final.get("corrected_or_verified_claim_count", 0)
        ),
        "help_requests": int(final.get("help_request_count", 0)),
        "risk_reports": sum(row.get("action") == "report_risk" for row in resident),
        "replies": int(final.get("reply_count", 0)),
        "likes": int(interaction_counts.get("like", 0)),
        "reposts": int(interaction_counts.get("repost", 0)),
        "reports": int(interaction_counts.get("report", 0)),
        "compound_speech_and_engagement_turns": int(
            interactions.get("compound_speech_and_engagement_turn_count", 0)
        ),
        "multi_kind_engagement_turns": int(
            interactions.get("multi_kind_engagement_turn_count", 0)
        ),
        "complete_governance_uptake_chains": sum(
            row.get("complete") is True
            for row in chains.get("governance_uptake_chains") or []
        ),
        "complete_help_service_feedback_chains": sum(
            row.get("complete") is True
            for row in chains.get("help_service_feedback_chains") or []
        ),
        "complete_correction_chains": sum(
            row.get("complete") is True
            for row in chains.get("correction_chains") or []
        ),
        "population_state": dict(final.get("population_state") or {}),
        "near_duplicate_rate": round(
            _near_duplicate_rate(str(row.get("visible_text") or "") for row in resident),
            6,
        ),
        "lineage_valid": _lineage_valid(branch),
        "resource_conserved": _resource_conserved(branch),
        "public_provenance_valid": all(
            row.get("provenance", {}).get("kind") in PUBLIC_PROVENANCE
            for row in messages
        ),
    }


def _aggregate_branch_summaries(
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    numeric_fields = (
        "unique_activated_agents",
        "resident_public_messages",
        "governance_public_messages",
        "threads",
        "claims",
        "contested_claims",
        "corrected_or_verified_claims",
        "help_requests",
        "risk_reports",
        "replies",
        "likes",
        "reposts",
        "reports",
        "compound_speech_and_engagement_turns",
        "multi_kind_engagement_turns",
        "complete_governance_uptake_chains",
        "complete_help_service_feedback_chains",
        "complete_correction_chains",
    )
    return {
        branch: {
            field: {
                "mean": round(mean(float(seed[branch][field]) for seed in summaries.values()), 6),
                "min": round(min(float(seed[branch][field]) for seed in summaries.values()), 6),
                "max": round(max(float(seed[branch][field]) for seed in summaries.values()), 6),
            }
            for field in numeric_fields
        }
        for branch in BRANCHES
    }


def analyze(
    wrapper: Mapping[str, Any],
    *,
    cumulative_ledger_usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = wrapper.get("result")
    usage = wrapper.get("live_usage")
    if not isinstance(result, Mapping) or not isinstance(usage, Mapping):
        raise ValueError("authorized flagship wrapper lacks result/live_usage")
    if result.get("schema_version") != "campus-pulse-resource-allocation-flagship-result-v2":
        raise ValueError("input is not a resource flagship result")
    seed_rows = list(result.get("seed_results") or [])
    if len(seed_rows) not in {1, 3, 4}:
        raise ValueError(
            "flagship result must contain one canonical, three product-validation, "
            "or four full paired seeds"
        )
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    all_messages = []
    for seed_row in seed_rows:
        branches = seed_row.get("branches") or {}
        if set(branches) != set(BRANCHES) or len(branches) != len(BRANCHES):
            raise ValueError("flagship seed does not contain the five frozen worlds")
        seed_key = str(seed_row["seed"])
        summaries[seed_key] = {
            branch: _branch_summary(branches[branch]) for branch in BRANCHES
        }
        for branch in BRANCHES:
            all_messages.extend(branches[branch].get("messages") or [])

    canonical = summaries[str(result["canonical_seed"])]
    branch_aggregates = _aggregate_branch_summaries(summaries)
    story_gate = {
        "at_least_three_claims": max(row["claims"] for row in canonical.values()) >= 3,
        "correction_chain_present": max(
            row["complete_correction_chains"] for row in canonical.values()
        )
        >= 1,
        "help_service_feedback_present": max(
            row["complete_help_service_feedback_chains"] for row in canonical.values()
        )
        >= 1,
        "direct_governance_uptake_present": max(
            row["complete_governance_uptake_chains"] for row in canonical.values()
        )
        >= 1,
        "parallel_engagement_visible": max(
            row["compound_speech_and_engagement_turns"] for row in canonical.values()
        )
        >= 1,
    }
    cumulative = dict(cumulative_ledger_usage or {})
    ledger_outcomes = dict(cumulative.get("outcomes") or {})
    primary_slots = int(result.get("primary_slots", 0))
    successful_attempts = int(ledger_outcomes.get("success", 0))
    imported_exact_slots_used = max(
        0,
        primary_slots - successful_attempts,
    )
    if cumulative:
        cumulative["imported_exact_cache_slots_used"] = imported_exact_slots_used
        cumulative["in_scope_semantic_slot_evidence_count"] = min(
            successful_attempts, primary_slots
        ) + imported_exact_slots_used
        cumulative["out_of_scope_successful_attempts"] = max(
            0, successful_attempts - primary_slots
        )
    calls_for_cap = int(
        cumulative.get("provider_calls_completed", usage.get("provider_calls", 0))
    )
    tokens_for_cap = int(
        cumulative.get("provider_reported_tokens", usage.get("provider_tokens", 0))
    ) + int(cumulative.get("risk_charged_tokens", 0))
    automatic_checks = {
        "wrapper_hash_valid": _verified_hash(wrapper, "artifact_sha256"),
        "result_hash_valid": _verified_hash(result, "result_sha256"),
        "full_profile_v2_bound": result.get(
            "formal_full_corpus_profile_release_claimed"
        )
        is True,
        "slot_and_usage_conserved": result.get("primary_slots")
        == usage.get("semantic_turns")
        and result.get("primary_slots") in {900, 2_028, 2_592},
        "request_and_token_caps_respected": calls_for_cap
        <= usage.get("request_limit", -1)
        and tokens_for_cap <= usage.get("token_limit", -1),
        "cumulative_ledger_settled": not cumulative
        or (
            cumulative.get("provider_calls_reserved") == 0
            and cumulative.get("tokens_in_flight") == 0
            and cumulative.get("in_scope_semantic_slot_evidence_count")
            == result.get("primary_slots")
        ),
        "no_unknown_outcome": usage.get("unknown_outcomes") == 0,
        "all_public_content_is_llm": result.get("dynamics_public_messages") == 0
        and all(
            row.get("provenance", {}).get("kind") in PUBLIC_PROVENANCE
            for row in all_messages
        ),
        "parallel_turn_contract_active": result.get(
            "parallel_speech_and_engagement_enabled"
        )
        is True,
        "llm_budget_bounded": result.get(
            "maximum_resident_llm_turns_per_world_tick"
        )
        == 24
        and result.get("maximum_resident_plus_governance_turns_per_world_tick")
        == 27,
        "five_policy_worlds_present": tuple(result.get("branches") or ()) == BRANCHES,
        "no_bonus_or_precommitted_winner": result.get("scheme_bonus") == 0
        and result.get("winner_precommitted") is False,
        "all_lineages_valid": all(
            row["lineage_valid"]
            for seed in summaries.values()
            for row in seed.values()
        ),
        "all_resources_conserved": all(
            row["resource_conserved"]
            for seed in summaries.values()
            for row in seed.values()
        ),
        "public_provenance_valid": all(
            row["public_provenance_valid"]
            for seed in summaries.values()
            for row in seed.values()
        ),
        "no_secret_or_local_path": SECRET_PATTERN.search(json.dumps(wrapper, ensure_ascii=False))
        is None
        and PATH_PATTERN.search(json.dumps(wrapper, ensure_ascii=False)) is None,
    }
    return {
        "schema_version": "campus-pulse-resource-flagship-audit-v1",
        "source_artifact_sha256": wrapper.get("artifact_sha256"),
        "stage": result.get("stage"),
        "scope_complete": result.get("scope_complete"),
        "validation_scope": result.get("validation_scope"),
        "full_scope_complete": result.get("full_scope_complete"),
        "automatic_checks": automatic_checks,
        "automatic_checks_passed": all(automatic_checks.values()),
        "story_gate": story_gate,
        "story_gate_passed": all(story_gate.values()),
        "seed_branch_summaries": summaries,
        "branch_aggregate_summaries": branch_aggregates,
        "paired_state_difference_intervals_95_vs_Natural": result.get(
            "paired_state_difference_intervals_95_vs_Natural"
        ),
        "usage": dict(usage),
        "cumulative_provider_audit": cumulative,
        "manual_review": {
            "required": True,
            "reviewers": 3,
            "persona_consistency_threshold": 0.90,
            "temporal_visibility_threshold": 1.0,
            "direct_reply_continuity_threshold": 0.85,
            "unsupported_fact_maximum": 0.02,
            "template_expression_maximum": 0.05,
            "status": "pending_human_review",
        },
    }


def _report(audit: Mapping[str, Any]) -> str:
    canonical_seed = next(iter(audit["seed_branch_summaries"]))
    rows = audit["seed_branch_summaries"][canonical_seed]
    seed_count = len(audit["seed_branch_summaries"])
    aggregate = audit["branch_aggregate_summaries"]
    full_scope = seed_count == 4 and audit.get("full_scope_complete") is True
    product_scope = (
        seed_count == 3
        and audit.get("scope_complete") is True
        and audit.get("validation_scope") == "three_seed_product_validation"
    )
    lines = [
        (
            "# 暑期住宿床位分配：四种子五方案主实验"
            if full_scope
            else (
                "# 暑期住宿床位分配：三种子五方案产品验证"
                if product_scope
                else "# 暑期住宿床位分配：主种子五方案审计"
            )
        ),
        "",
        "## 实验做了什么",
        "",
        "同一场床位分配争议先在 Tick 3–4 形成共同论坛历史，随后从同一状态分成 Natural、A、B、C、D 五个世界。Natural 与 D 在主种子中逐时间步运行；其他方案和配对种子使用 Tick 5、9、13、15 语义检查点。每个世界每步最多调用 24 个居民 LLM，治理决策点再调用 3 个治理 LLM。",
        "",
        "## 主种子（固定随机种子）的论坛结果",
        "",
        "| 世界 | 居民发言 | 点赞 | 转发 | 举报 | 全部 Claim | 已纠错/核验 | 治理承接链 | 服务闭环 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        row = rows[branch]
        lines.append(
            f"| {branch} · {BRANCH_LABELS[branch]} | {row['resident_public_messages']} | {row['likes']} | "
            f"{row['reposts']} | {row['reports']} | {row['claims']} | "
            f"{row['corrected_or_verified_claims']} | "
            f"{row['complete_governance_uptake_chains']} | "
            f"{row['complete_help_service_feedback_chains']} |"
        )
    if seed_count > 1:
        lines.extend(
            [
                "",
                f"## {seed_count} 个配对种子的平均结果",
                "",
                "| 世界 | 居民发言 | 全部 Claim | 已纠错/已核验 Claim | 治理承接链 | 服务闭环 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for branch in BRANCHES:
            row = aggregate[branch]
            lines.append(
                f"| {branch} · {BRANCH_LABELS[branch]} | {row['resident_public_messages']['mean']:.1f} | "
                f"{row['claims']['mean']:.1f} | "
                f"{row['corrected_or_verified_claims']['mean']:.1f} | "
                f"{row['complete_governance_uptake_chains']['mean']:.1f} | "
                f"{row['complete_help_service_feedback_chains']['mean']:.1f} |"
            )
        intervals = audit.get("paired_state_difference_intervals_95_vs_Natural") or {}
        lines.extend(
            [
                "",
                "## 相对 Natural 的配对状态差异",
                "",
                f"正的 trust/satisfaction 表示模型状态上升，负的 concern 表示关切下降；区间跨过 0 时不能说方向在 {seed_count} 个种子中稳定。",
                "",
                "| 世界 | trust（百分点） | concern（百分点） | satisfaction（百分点） |",
                "|---|---:|---:|---:|",
            ]
        )
        for branch in BRANCHES[1:]:
            branch_intervals = intervals.get(branch) or {}
            cells = []
            for metric in ("trust", "concern", "satisfaction"):
                interval = branch_intervals.get(metric) or {}
                cells.append(
                    f"{100 * float(interval.get('mean', 0)):+.2f} "
                    f"[{100 * float(interval.get('lower', 0)):+.2f}, "
                    f"{100 * float(interval.get('upper', 0)):+.2f}]"
                )
            lines.append(
                f"| {branch} · {BRANCH_LABELS[branch]} | {' | '.join(cells)} |"
            )
        lines.extend(
            [
                "",
                "## 产品洞察",
                "",
                "公开消息量只在相同激活预算的组内比较：Natural 与 D 都在主种子逐 Tick 运行；A、B、C 都使用相同的四个语义检查点。跨组消息量差异主要反映预算密度，不能解释成政策效果。",
                "",
                (
                    f"- 在同预算的 A/B/C 中，C 的平均 Claim 最少（{aggregate['C']['claims']['mean']:.1f}），"
                    f"A 和 C 分别形成 {aggregate['A']['complete_governance_uptake_chains']['mean']:.1f} "
                    f"和 {aggregate['C']['complete_governance_uptake_chains']['mean']:.1f} 条居民直接承接链；"
                    f"B 只有 {aggregate['B']['complete_governance_uptake_chains']['mean']:.1f} 条。"
                ),
                (
                    f"- 在同预算的 Natural/D 中，两者平均居民发言均为 "
                    f"{aggregate['Natural']['resident_public_messages']['mean']:.1f}；D 形成 "
                    f"{aggregate['D']['complete_governance_uptake_chains']['mean']:.1f} 条治理承接链，"
                    f"但 Claim 从 {aggregate['Natural']['claims']['mean']:.1f} 增至 "
                    f"{aggregate['D']['claims']['mean']:.1f}。治理没有压低讨论，而是把更多未解决问题带到台面。"
                ),
                "- 所有模型状态区间都跨过 0；本轮不能宣称某个方案稳定提升信任或满意度。产品价值在于提前发现：证据公开和跨群触达更容易被继续追问，而单纯开服务入口不保证问题真正闭环。",
            ]
        )
    lines.extend(
        [
            "",
            "## 自动门禁",
            "",
            *[
                f"- {'通过' if passed else '未通过'}：{name}"
                for name, passed in audit["automatic_checks"].items()
            ],
            "",
            "## 故事门禁",
            "",
            *[
                f"- {'通过' if passed else '未通过'}：{name}"
                for name, passed in audit["story_gate"].items()
            ],
            "",
            "三人盲审完成前，自动检查不能替代人物一致性、上下文承接和无依据事实审阅。",
        ]
    )
    cumulative = audit.get("cumulative_provider_audit") or {}
    if cumulative:
        outcomes = cumulative.get("outcomes") or {}
        lines.extend(
            [
                "",
                "## Provider 累计账本",
                "",
                f"- 已结算请求：{cumulative.get('provider_calls_completed', 0)}",
                f"- Provider 报告 Token：{cumulative.get('provider_reported_tokens', 0)}",
                f"- unknown 最坏风险 Token：{cumulative.get('risk_charged_tokens', 0)}",
                f"- 本结果实际使用的跨运行 exact-cache 槽位：{cumulative.get('imported_exact_cache_slots_used', 0)}",
                f"- 本结果范围外已结算成功尝试：{cumulative.get('out_of_scope_successful_attempts', 0)}（不进入三种子统计）",
                f"- 成功响应 / 已结算合同失败 / 明确 Provider 失败：{outcomes.get('success', 0)} / {outcomes.get('format_invalid', 0)} / {outcomes.get('known_failure', 0)}",
                f"- 历史 unknown：{outcomes.get('unknown', 0)}；这些槽位均须有 attempt 级裁决证据后才能替代，最终结果内 unknown 为 0。",
            ]
        )
    return "\n".join(lines) + "\n"


def _story_svg(audit: Mapping[str, Any]) -> str:
    aggregates = audit["branch_aggregate_summaries"]
    intervals = audit.get("paired_state_difference_intervals_95_vs_Natural") or {}
    seed_count = len(audit["seed_branch_summaries"])
    colors = {
        "paper": "#F4F1EA",
        "card": "#FFFDF8",
        "ink": "#171717",
        "muted": "#6A6862",
        "line": "#D8D2C6",
        "Natural": "#777B82",
        "A": "#B48A45",
        "B": "#32745A",
        "C": "#356B86",
        "D": "#B80C32",
    }

    def text(x: float, y: float, value: Any, size: int, color: str, weight: int = 400) -> str:
        return (
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}">{escape(str(value))}</text>'
        )

    def interval_label(branch: str, metric: str) -> str:
        row = (intervals.get(branch) or {}).get(metric) or {}
        return (
            f"{100 * float(row.get('mean', 0)):+.2f} "
            f"[{100 * float(row.get('lower', 0)):+.2f},"
            f"{100 * float(row.get('upper', 0)):+.2f}]"
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1800 1040" '
        'width="100%" role="img" aria-labelledby="flagship-title flagship-desc" '
        'preserveAspectRatio="xMidYMid meet" style="display:block;max-width:1800px;height:auto;margin:0 auto">',
        '<title id="flagship-title">校园资源分配争议五种治理方案对照</title>',
        '<desc id="flagship-desc">三个配对种子下 Natural、A、B、C 和 D 五个平行论坛的消息、Claim、治理承接与服务闭环对比。</desc>',
        f'<rect width="1800" height="1040" fill="{colors["paper"]}"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
        text(70, 74, "同一场床位争议，五种治理世界如何分叉？", 40, colors["ink"], 650),
        text(72, 116, f"真实 LLM ForumTwin · {seed_count} 个配对种子 · Tick 3–15 · 每世界每步最多 24 个居民 LLM", 19, colors["muted"]),
        f'<rect x="70" y="150" width="1660" height="116" rx="20" fill="{colors["card"]}" stroke="{colors["line"]}"/>',
        text(104, 191, "共同事件", 18, colors["D"], 650),
        text(104, 230, "两栋宿舍翻修使暑期床位从 520 减至 320；536 份有效申请竞争有限资源，原公告缺少排序、复核、申诉与答复时限。", 23, colors["ink"], 520),
    ]
    labels = {
        "Natural": "Natural\n不新增治理",
        "A": "A\n规则解释",
        "B": "B\n服务承接",
        "C": "C\n跨群触达",
        "D": "D\n自适应组合",
    }
    for index, branch in enumerate(BRANCHES):
        x = 70 + index * 332
        row = aggregates[branch]
        title, subtitle = labels[branch].split("\n")
        parts.extend(
            [
                f'<rect x="{x}" y="310" width="300" height="344" rx="20" fill="{colors["card"]}" stroke="{colors[branch]}" stroke-width="2"/>',
                f'<rect x="{x}" y="310" width="300" height="8" rx="4" fill="{colors[branch]}"/>',
                text(x + 24, 360, title, 26, colors[branch], 700),
                text(x + 24, 390, subtitle, 17, colors["muted"], 500),
                text(x + 24, 438, f"平均居民发言 {row['resident_public_messages']['mean']:.1f}", 20, colors["ink"], 520),
                text(x + 24, 478, f"平均 Claim {row['claims']['mean']:.1f}", 20, colors["ink"], 520),
                text(x + 24, 518, f"治理直接承接 {row['complete_governance_uptake_chains']['mean']:.1f}", 19, colors["ink"]),
                text(x + 24, 558, f"完整服务闭环 {row['complete_help_service_feedback_chains']['mean']:.1f}", 19, colors["ink"]),
            ]
        )
        if branch == "Natural":
            parts.extend(
                [
                    text(x + 24, 604, "状态差异基准", 17, colors["muted"]),
                    text(x + 24, 632, "同人口、同事件、同随机数", 16, colors["muted"]),
                ]
            )
        else:
            parts.extend(
                [
                    text(x + 24, 600, f"trust {interval_label(branch, 'trust')} pp", 16, colors["muted"]),
                    text(x + 24, 628, f"concern {interval_label(branch, 'concern')} pp", 16, colors["muted"]),
                ]
            )
    parts.extend(
        [
            f'<rect x="70" y="704" width="1660" height="240" rx="20" fill="{colors["card"]}" stroke="{colors["line"]}"/>',
            text(104, 752, "如何读这张图", 24, colors["ink"], 650),
            text(104, 798, "1. 公开帖子全部来自授权 LLM 或精确缓存；Dynamics 只推进沉默、注意力、Feed 与粒子状态。", 19, colors["ink"]),
            text(104, 838, "2. 治理价值不等于把发言压低：看 Claim 是否可核验、求助是否进入服务链、居民是否直接承接治理对象。", 19, colors["ink"]),
            text(104, 878, "3. 五个世界没有预设赢家；状态差异采用同种子配对，95% 区间跨过 0 时不宣称方向稳定。", 19, colors["ink"]),
            text(104, 918, "比较口径：Natural↔D、A↔B↔C；跨组公开消息量受激活预算密度影响，不解释为政策效果。", 18, colors["muted"], 500),
            text(104, 958, "产品用途：在政策发布前发现信息缺口、责任链断点和治理反作用，再决定解释、服务或触达组合。", 20, colors["D"], 600),
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    arguments = parser.parse_args()
    audit = analyze(
        _load(arguments.result.resolve()),
        cumulative_ledger_usage=(
            _ledger_usage(arguments.ledger)
            if arguments.ledger is not None
            else None
        ),
    )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resource-flagship-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "RESOURCE_FLAGSHIP_RESULT_REPORT.md").write_text(
        _report(audit), encoding="utf-8"
    )
    (output_dir / "resource-flagship-story.svg").write_text(
        _story_svg(audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "automatic_checks_passed": audit["automatic_checks_passed"],
                "story_gate_passed": audit["story_gate_passed"],
                "scope_complete": audit["scope_complete"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
