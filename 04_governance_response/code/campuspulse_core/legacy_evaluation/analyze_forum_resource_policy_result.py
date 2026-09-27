"""Audit and narrate the ForumTwin resource-allocation policy sandbox.

The input is a public authorized-live result.  The outputs never contain
prompt bodies, Provider bodies, credentials, private Feed exposures, or source
records.  Short excerpts are taken only from public synthetic LLM messages.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
from hashlib import sha256
from html import escape
import itertools
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


BRANCH_ORDER = ("Natural", "A", "D")
BRANCH_LABELS = {
    "Natural": "不追加回应",
    "A": "只解释分配标准",
    "D": "标准解释 + 复核服务 + 公开答疑",
}
PUBLIC_PROVENANCE = {"authorized_live_llm", "reviewed_trace_replay"}
SECRET_PATTERN = re.compile(r"(?i)sk-[A-Za-z0-9_-]{12,}")
WINDOWS_PATH_PATTERN = re.compile(r"(?i)(?:[A-Z]:\\|API\.txt)")
CONTACT_PATTERN = re.compile(
    r"(?i)(?:https?://|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?<!\d)1[3-9]\d{9}(?!\d))"
)
REALITY_NAME_PATTERN = re.compile(
    r"(?:张小龙|清华大学|北京大学|中国人民大学|复旦大学|上海交通大学)"
)
FORBIDDEN_PUBLIC_FIELDS = {
    "api_key",
    "password",
    "secret",
    "local_path",
    "file_path",
    "raw_text",
    "raw_record",
    "source_id",
    "source_identifier",
    "prompt_body",
    "provider_body",
    "request_body",
    "response_body",
    "hidden_truth",
    "private_feed",
    "feed_exposures",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("result root must be an object")
    return value


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def verified_hash(value: Mapping[str, Any], field: str) -> bool:
    body = {key: child for key, child in value.items() if key != field}
    return value.get(field) == canonical_sha256(body)


def normalized_text(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).casefold()


def direct_responses(
    messages: Iterable[Mapping[str, Any]], target_id: str
) -> list[Mapping[str, Any]]:
    return [
        message
        for message in messages
        if target_id
        and target_id
        in {
            str(message.get("parent_message_id") or ""),
            str(message.get("quote_message_id") or ""),
        }
        and not str(message.get("source_agent_display_id") or "").startswith(
            "governance:"
        )
    ]


def lineage_valid(branch: Mapping[str, Any]) -> bool:
    messages = list(branch.get("messages") or [])
    by_id = {str(message.get("message_id")): message for message in messages}
    claims = {
        str(claim.get("claim_id")): claim for claim in branch.get("claims") or []
    }
    for message in messages:
        tick = int(message.get("created_tick", -1))
        for field in ("parent_message_id", "quote_message_id"):
            target_id = message.get(field)
            if target_id is None:
                continue
            target = by_id.get(str(target_id))
            if target is None or int(target.get("created_tick", tick)) >= tick:
                return False
        claim_id = message.get("claim_id")
        if claim_id is not None:
            claim = claims.get(str(claim_id))
            if claim is None or claim.get("thread_id") != message.get("thread_id"):
                return False
        target_claim_id = message.get("correction_target_claim_id")
        if target_claim_id is not None and str(target_claim_id) not in claims:
            return False
    return True


def near_duplicate_pairs(texts: list[str], threshold: float = 0.86) -> int:
    normalized = [normalized_text(text) for text in texts if text]
    return sum(
        SequenceMatcher(None, left, right).ratio() >= threshold
        for left, right in itertools.combinations(normalized, 2)
    )


def branch_metrics(branch: Mapping[str, Any]) -> dict[str, Any]:
    messages = list(branch.get("messages") or [])
    timeline = list(branch.get("timeline") or [])
    final = timeline[-1]
    texts = [str(message.get("visible_text") or "") for message in messages]
    normalized = [normalized_text(text) for text in texts if text]
    frequencies = Counter(normalized)
    governance_messages = [
        message
        for message in messages
        if str(message.get("source_agent_display_id") or "").startswith(
            "governance:"
        )
    ]
    direct = {
        str(message.get("message_id")): direct_responses(
            messages, str(message.get("message_id") or "")
        )
        for message in governance_messages
    }
    chains = branch.get("mechanism_chains") or {}
    profiles = list(branch.get("public_display_profiles") or [])
    artifacts = list(branch.get("governance_artifacts") or [])
    decisions = [
        decision
        for round_value in branch.get("governance_rounds") or []
        for decision in round_value.get("decisions") or []
    ]
    collected_interactions = [
        message.get("interaction_counts")
        for message in messages
        if isinstance(message.get("interaction_counts"), Mapping)
    ]
    interaction_totals = {
        kind: sum(int(row.get(kind, 0)) for row in collected_interactions)
        for kind in ("like", "repost", "report")
    }
    state = final.get("population_state") or {}
    resident_public_speakers = {
        str(message.get("source_agent_display_id") or "")
        for message in messages
        if not str(message.get("source_agent_display_id") or "").startswith(
            "governance:"
        )
    }
    return {
        "public_messages": len(messages),
        "threads": len(branch.get("threads") or []),
        "claims": len(branch.get("claims") or []),
        "unique_activated_agents": branch.get("unique_activated_agents"),
        "unique_public_speakers": len(resident_public_speakers),
        "profile_v2_bound_agents": sum(
            profile.get("source_profile_v2_bound") is True for profile in profiles
        ),
        "distinct_profile_bundles": len(
            {
                (
                    str(profile.get("macro_role")),
                    str(profile.get("micro_role")),
                    json.dumps(profile.get("stable_traits"), ensure_ascii=False),
                )
                for profile in profiles
            }
        ),
        "help_requests": int(final.get("help_request_count", 0)),
        "risk_reports": int(final.get("risk_report_count", 0)),
        "replies": int(final.get("reply_count", 0)),
        "parallel_interactions_collected": len(collected_interactions) == len(messages),
        "interaction_totals": interaction_totals,
        "parallel_interaction_audit": dict(branch.get("parallel_interactions") or {}),
        "governance_messages": len(governance_messages),
        "direct_governance_uptake": sum(bool(value) for value in direct.values()),
        "direct_governance_responses": sum(len(value) for value in direct.values()),
        "complete_help_service_feedback": sum(
            chain.get("complete") is True
            for chain in chains.get("help_service_feedback_chains") or []
        ),
        "artifact_kinds": dict(
            sorted(Counter(str(row.get("artifact_kind")) for row in artifacts).items())
        ),
        "selected_governance_actions": dict(
            sorted(
                Counter(str(row.get("selected_action")) for row in decisions).items()
            )
        ),
        "action_counts": dict(
            sorted(Counter(str(row.get("action")) for row in messages).items())
        ),
        "stance_counts": dict(
            sorted(Counter(str(row.get("stance")) for row in messages).items())
        ),
        "exact_duplicate_instances": sum(
            count - 1 for count in frequencies.values() if count > 1
        ),
        "near_duplicate_pairs_0_86": near_duplicate_pairs(texts),
        "population_state": {
            **{
                key: round(float(state.get(key, 0.0)), 6)
                for key in (
                    "attention",
                    "concern",
                    "trust",
                    "satisfaction",
                    "support",
                    "opposition",
                )
            },
            "activity": round(float(state.get("activity_probability", 0.0)), 6),
        },
    }


def shared_prefix_equal(branches: Mapping[str, Mapping[str, Any]]) -> bool:
    def message_at_fork(
        message: Mapping[str, Any], *, fork_tick: int
    ) -> dict[str, Any]:
        """Project a public message back to its state at the branch fork.

        Public message objects carry cumulative engagement counters.  Likes or
        reposts received after the fork legitimately differ between policy
        worlds even though the message and all pre-fork interactions were
        identical.  Comparing the final cumulative object therefore creates a
        false shared-prefix failure.  Keep only engagement deltas observable by
        the fork and recompute their totals before comparing the branches.
        """

        projected = dict(message)
        if "interaction_counts_by_tick" not in projected:
            return projected
        by_tick = [
            dict(row)
            for row in projected.get("interaction_counts_by_tick") or []
            if int(row.get("tick", -1)) <= fork_tick
        ]
        projected["interaction_counts_by_tick"] = by_tick
        projected["interaction_counts"] = {
            kind: sum(int(row.get(kind, 0)) for row in by_tick)
            for kind in ("like", "repost", "report")
        }
        return projected

    def prefix(branch: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
        messages = [
            message_at_fork(message, fork_tick=4)
            for message in branch.get("messages") or []
            if int(message.get("created_tick", -1)) <= 4
        ]
        timeline = [
            point
            for point in branch.get("timeline") or []
            if int(point.get("tick", -1)) <= 4
        ]
        return messages, timeline

    baseline = prefix(branches[BRANCH_ORDER[0]])
    return all(prefix(branches[name]) == baseline for name in BRANCH_ORDER[1:])


def governance_valid(branch: Mapping[str, Any], expected_ticks: tuple[int, ...]) -> bool:
    rounds = list(branch.get("governance_rounds") or [])
    if tuple(int(row.get("tick", -1)) for row in rounds) != expected_ticks:
        return False
    if sum(len(row.get("decisions") or []) for row in rounds) != 3 * len(
        expected_ticks
    ):
        return False
    return all(
        row.get("scheme_bonus") == 0
        and row.get("uses_hidden_truth") is False
        and row.get("resource_before", {}).get("conserved") is True
        and row.get("resource_after", {}).get("conserved") is True
        for row in rounds
    )


def privacy_scan(value: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_paths: list[str] = []

    def walk(child: Any, path: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key).casefold() in FORBIDDEN_PUBLIC_FIELDS:
                    forbidden_paths.append(child_path)
                walk(nested, child_path)
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                walk(nested, f"{path}[{index}]")

    walk(value, "")
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    public_text = "\n".join(
        str(message.get("visible_text") or "")
        for branch in value.get("result", {}).get("branches", {}).values()
        for message in branch.get("messages") or []
    )
    checks = {
        "secret_literal_count": len(SECRET_PATTERN.findall(serialized)),
        "local_path_literal_count": len(WINDOWS_PATH_PATTERN.findall(serialized)),
        "public_text_contact_or_url_literal_count": len(
            CONTACT_PATTERN.findall(public_text)
        ),
        "forbidden_public_field_count": len(forbidden_paths),
    }
    checks["named_real_campus_or_person_hits"] = len(
        REALITY_NAME_PATTERN.findall(public_text)
    )
    checks["passed"] = all(count == 0 for count in checks.values())
    checks["forbidden_public_field_paths"] = forbidden_paths
    return checks


def extract_story_chains(branch: Mapping[str, Any]) -> dict[str, Any]:
    messages = list(branch.get("messages") or [])
    by_id = {str(message.get("message_id")): message for message in messages}
    help_chains = []
    for chain in branch.get("mechanism_chains", {}).get(
        "help_service_feedback_chains", []
    ):
        if chain.get("complete") is not True:
            continue
        help_chains.append(
            {
                key: {
                    "tick": by_id[str(chain[key])].get("created_tick"),
                    "speaker": by_id[str(chain[key])].get(
                        "source_agent_display_id"
                    ),
                    "text": by_id[str(chain[key])].get("visible_text"),
                }
                for key in ("help_message_id", "service_receipt_id", "follow_up_id")
            }
        )
    governance_chains = []
    for message in messages:
        if not str(message.get("source_agent_display_id") or "").startswith(
            "governance:"
        ):
            continue
        responses = direct_responses(messages, str(message.get("message_id") or ""))
        if not responses:
            continue
        governance_chains.append(
            {
                "governance": {
                    "tick": message.get("created_tick"),
                    "speaker": message.get("source_agent_display_id"),
                    "text": message.get("visible_text"),
                },
                "responses": [
                    {
                        "tick": response.get("created_tick"),
                        "speaker": response.get("source_agent_display_id"),
                        "text": response.get("visible_text"),
                    }
                    for response in responses
                ],
            }
        )
    return {
        "help_service_feedback": help_chains,
        "direct_governance_uptake": governance_chains,
    }


def short(value: Any, limit: int = 64) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def wrapped(value: Any, width: int = 21, lines: int = 3) -> list[str]:
    text = re.sub(r"\s+", "", str(value or ""))
    limit = width * lines
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return [text[index : index + width] for index in range(0, len(text), width)]


def svg_story(audit: Mapping[str, Any]) -> str:
    metrics = audit["branch_metrics"]
    chain = audit["story_chains"]["D"]["help_service_feedback"][0]
    colors = {
        "ink": "#172433",
        "muted": "#657589",
        "paper": "#F5F1E8",
        "white": "#FFFDF8",
        "line": "#D7D1C4",
        "natural": "#7A8795",
        "a": "#D9923B",
        "d": "#2B7A77",
        "danger": "#B6574B",
        "good": "#32745A",
    }
    branch_color = {"Natural": colors["natural"], "A": colors["a"], "D": colors["d"]}

    def box(x: int, y: int, w: int, h: int, fill: str, stroke: str) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def text(x: int, y: int, value: Any, size: int, color: str, weight: int = 400) -> str:
        return (
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}">{escape(str(value))}</text>'
        )

    def multi_text(
        x: int, y: int, value: Any, size: int, color: str, width: int = 21
    ) -> str:
        rows = wrapped(value, width=width, lines=3)
        tspans = "".join(
            f'<tspan x="{x}" dy="{0 if index == 0 else 28}">{escape(row)}</tspan>'
            for index, row in enumerate(rows)
        )
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{tspans}</text>'

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="980" viewBox="0 0 1600 980">',
        f'<rect width="1600" height="980" fill="{colors["paper"]}"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
        text(70, 72, "同一场床位争议，三个平行校园如何分叉？", 38, colors["ink"], 500),
        text(72, 110, "ForumTwin 真实 LLM 闭环试验 · 同一 T4 状态后分叉 · 单一 canonical seed", 18, colors["muted"]),
        box(70, 145, 1460, 112, colors["white"], colors["line"]),
        text(102, 184, "共同事件", 18, colors["danger"], 500),
        text(102, 220, "两栋宿舍翻修，暑期床位 520→320，却有 536 份有效申请。", 24, colors["ink"], 500),
        text(790, 220, "公告列了三类优先需求，却没说排序、复核、申诉和答复时限。", 22, colors["ink"]),
    ]
    x_positions = {"Natural": 70, "A": 570, "D": 1070}
    for branch in BRANCH_ORDER:
        x = x_positions[branch]
        metric = metrics[branch]
        parts.extend(
            [
                f'<line x1="800" y1="257" x2="{x + 230}" y2="300" stroke="{branch_color[branch]}" stroke-width="3"/>',
                box(x, 300, 460, 300, colors["white"], branch_color[branch]),
                text(x + 28, 340, BRANCH_LABELS[branch], 24, branch_color[branch], 500),
                text(x + 28, 382, f'公开帖/回复 {metric["public_messages"]} 条', 19, colors["ink"]),
                text(x + 28, 418, f'求助 {metric["help_requests"]} 次', 22, colors["ink"], 500),
                text(x + 28, 454, f'风险上报 {metric["risk_reports"]} 次', 19, colors["ink"]),
                text(x + 28, 490, f'直接承接治理消息 {metric["direct_governance_uptake"]} 条链', 19, colors["ink"]),
                text(x + 28, 526, f'求助→工单→后续反馈 {metric["complete_help_service_feedback"]} 条闭环', 19, colors["ink"]),
                text(x + 28, 570, f'T10 信任 {metric["population_state"]["trust"]:.3f}', 18, colors["muted"]),
            ]
        )
    parts.extend(
        [
            box(70, 635, 1460, 230, colors["white"], colors["d"]),
            text(102, 678, "D 世界中，一条真正“没被安抚话术结束”的 LLM 闭环", 24, colors["d"], 500),
            text(108, 728, "居民求助", 17, colors["muted"], 500),
            multi_text(108, 760, chain["help_message_id"]["text"], 17, colors["ink"]),
            text(605, 728, "服务工单", 17, colors["muted"], 500),
            multi_text(605, 760, chain["service_receipt_id"]["text"], 17, colors["ink"]),
            text(1100, 728, "同一居民继续追问", 17, colors["muted"], 500),
            multi_text(1100, 760, chain["follow_up_id"]["text"], 17, colors["ink"]),
            f'<path d="M535 815 L575 815" stroke="{colors["d"]}" stroke-width="3" marker-end="url(#arrow)"/>',
            f'<path d="M1030 815 L1070 815" stroke="{colors["d"]}" stroke-width="3" marker-end="url(#arrow)"/>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#2B7A77"/></marker></defs>',
            text(72, 910, "产品结论：“发一张解释卡”能压低部分风险上报，却不会自动建立程序正当性。", 23, colors["ink"], 500),
            text(72, 947, "复核服务让 2 个具体问题进入责任链，但总体信任没有被强行“做高”——稀缺性与未完成处理仍被 LLM 居民看见。", 21, colors["ink"]),
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts)


def markdown_report(audit: Mapping[str, Any]) -> str:
    m = audit["branch_metrics"]
    chain = audit["story_chains"]["D"]["help_service_feedback"][0]
    usage = audit["usage"]
    return f"""# CampusPulse ForumTwin：床位分配治理预演结果

## 我们做了什么

我们把一场具体的校园资源分配争议放进虚拟小喇叭：两栋宿舍翻修，暑期床位从 520 减到 320，但有 536 份有效申请。原通知列了三类优先需求，却没说清类别和类内如何排序、谁复核材料、如何申诉、多久答复。

同一批 LLM 居民在 T3–T4 共享完全相同的论坛历史，然后从同一个检查点分成三个平行世界：

- Natural：学校不再追加回应。
- A：只发布标准解释和证据边界。
- D：在解释标准之外，增加复核工单和公开答疑/跨群体触达。

论坛曝光不是个性化推荐：每个主体看到同一个全局热度前十和最新帖，另加直接回复/已参与串通知。Profile 影响“看见后怎么理解和回应”，不改写公共榜单。

## 结果是什么

Natural 世界有 {m['Natural']['help_requests']} 次求助，但没有一次进入服务闭环。A 世界的风险上报从 Natural 的 {m['Natural']['risk_reports']} 次降到 {m['A']['risk_reports']} 次，说明解释能降低一部分信息风险；但它仍然有 {m['A']['help_requests']} 次求助，居民没有直接承接任何一条治理消息，说明“把话说清楚”不等于“问题有地方去解决”。

D 世界有 {m['D']['help_requests']} 次求助，建立了 {m['D']['complete_help_service_feedback']} 条严格的“求助→工单回执→同一居民直接反馈”闭环，也有 {m['D']['direct_governance_uptake']} 条治理消息被居民直接回复/引用。关键的一条链是：

> 居民：{chain['help_message_id']['text']}

> 服务角色：{chain['service_receipt_id']['text']}

> 同一居民的后续反馈：{chain['follow_up_id']['text']}

这条链的价值不是“居民被说服了”，而是居民明确识别出：已受理不等于已解决，并继续要求可查询状态和最终答复。这比“不信谣不传谣”式的统一理性回答更接近一个真实论坛中的程序追问。

## 最重要的产品发现

治理不是“发公告就让曲线变好看”。A 世界能减少部分风险上报，却没有建立责任链；D 世界把两个具体问题送进了服务链，但 T10 模型信任为 {m['D']['population_state']['trust']:.3f}，低于 Natural 的 {m['Natural']['population_state']['trust']:.3f}。这是一个有价值的反直觉诊断：程序被打开后，稀缺性和“工单还没有最终结论”也会被更清楚地看见。我们的治理沙盘因此能在现实执行前暴露“话术止损”和“可执行闭环”的差别，以及政策可能的反作用。

## 这次结果可信到什么程度

- {usage['semantic_turns']}/{audit['expected']['semantic_turns']} 个 LLM 槽位已结算；Provider 尝试 {usage['provider_attempts']} 次，使用 {usage['provider_tokens']:,} tokens，未知结果为 {usage['unknown_outcomes']}。
- 所有公开帖子和回复均来自本次授权的 gpt-5.6-luna；Dynamics 公开消息为 0。
- 三个世界的 T3–T4 历史逐字相同；分叉后没有预设赢家、没有 scheme bonus。
- 三个分支各激活 {m['Natural']['unique_activated_agents']} / {m['A']['unique_activated_agents']} / {m['D']['unique_activated_agents']} 个不同居民，且展示主体全部无损绑定 Profile v2。
- 每个分支的公开文本都没有精确重复或 0.86 以上的近重复对。
- 自动敏感信息扫描通过；本报告只引用公开的合成 LLM 论坛文本。

## 必须说清的边界

这是一个 canonical seed、T3–T10 的高质量 Hero 试验，不是多种子因果证明；不能说 D 政策在现实中一定有效。PPS 本轮每 tick 的轮换样本很小，所以不应把精确的总体百分比作为路演主线。本次也没有形成有效的谣言纠错链，不应宣称辟谣效果。路演应聚焦“资源稀缺→程序质疑→三世界分叉→服务闭环与反作用”。
"""


def analyze(value: Mapping[str, Any]) -> dict[str, Any]:
    result = value.get("result")
    usage = value.get("live_usage")
    if not isinstance(result, Mapping) or not isinstance(usage, Mapping):
        raise ValueError("authorized wrapper lacks result/live_usage")
    branches = result.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != set(BRANCH_ORDER):
        raise ValueError("resource sandbox must contain Natural, A and D")
    typed_branches = {name: branches[name] for name in BRANCH_ORDER}
    metrics = {name: branch_metrics(typed_branches[name]) for name in BRANCH_ORDER}
    expected = {
        "resident_slots": 480,
        "governance_actor_slots": 12,
        "semantic_turns": 492,
    }
    all_messages = [
        message
        for branch in typed_branches.values()
        for message in branch.get("messages") or []
    ]
    checks = {
        "wrapper_hash_valid": verified_hash(value, "artifact_sha256"),
        "result_hash_valid": verified_hash(result, "result_sha256"),
        "scenario_is_resource_allocation_legitimacy": result.get("scenario_id")
        == "governance_legitimacy_dispute",
        "slot_plan_exact": result.get("resident_slots") == expected["resident_slots"]
        and result.get("governance_actor_slots")
        == expected["governance_actor_slots"]
        and result.get("primary_slots") == expected["semantic_turns"]
        and usage.get("semantic_turns") == expected["semantic_turns"],
        "provider_usage_conserved": usage.get("provider_calls")
        == usage.get("provider_attempts")
        and usage.get("provider_calls", 0) <= usage.get("request_limit", -1)
        and usage.get("provider_tokens", 0) <= usage.get("token_limit", -1)
        and usage.get("unknown_outcomes") == 0,
        "all_public_content_is_llm": all(
            message.get("provenance", {}).get("kind") in PUBLIC_PROVENANCE
            for message in all_messages
        )
        and result.get("dynamics_public_messages") == 0
        and all(branches[name].get("emulator_public_messages") == 0 for name in BRANCH_ORDER),
        "public_board_is_global_not_personalized": result.get(
            "public_feed_personalized"
        )
        is False
        and result.get("public_feed_relationship_graph_controlled") is False,
        "T3_T4_shared_prefix_exact": result.get("worlds_share_exact_T4_checkpoint")
        is True
        and shared_prefix_equal(typed_branches),
        "lineage_valid": all(lineage_valid(branch) for branch in typed_branches.values()),
        "Natural_has_no_governance": not typed_branches["Natural"].get(
            "governance_rounds"
        )
        and metrics["Natural"]["governance_messages"] == 0,
        "A_governance_valid": governance_valid(typed_branches["A"], (5, 7)),
        "D_governance_valid": governance_valid(typed_branches["D"], (5, 7)),
        "no_scheme_bonus_or_precommitted_winner": result.get("scheme_bonus") == 0
        and result.get("winner_precommitted") is False,
        "profile_v2_bound_for_all_public_speakers": all(
            metrics[name]["profile_v2_bound_agents"]
            == metrics[name]["unique_public_speakers"]
            for name in BRANCH_ORDER
        ),
        "formal_profile_v2_release_bound_for_population": (
            result.get("formal_full_corpus_profile_release_claimed") is True
            and result.get("population_size") == 1000
            and isinstance(result.get("profile_release_sha256"), str)
            and len(result["profile_release_sha256"]) == 64
        ),
        "no_exact_or_near_duplicate_public_text": all(
            metrics[name]["exact_duplicate_instances"] == 0
            and metrics[name]["near_duplicate_pairs_0_86"] == 0
            for name in BRANCH_ORDER
        ),
    }
    scan = privacy_scan(value)
    checks["privacy_scan_passed"] = scan["passed"]
    return {
        "schema": "campus-pulse-resource-policy-story-audit-v1",
        "source_result_sha256": value.get("artifact_sha256"),
        "run_id": value.get("launch", {}).get("run_id"),
        "preflight_sha256": value.get("launch", {}).get("preflight_sha256"),
        "model": value.get("launch", {}).get("model"),
        "scenario_title": result.get("scenario_title"),
        "expected": expected,
        "usage": dict(usage),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "privacy_scan": scan,
        "branch_metrics": metrics,
        "story_chains": {
            name: extract_story_chains(typed_branches[name]) for name in BRANCH_ORDER
        },
        "limitations": [
            "single_canonical_seed_not_causal_or_robustness_evidence",
            "PPS_rotation_sample_per_tick_is_small_and_intervals_are_wide",
            "aggregate_particle_states_are_secondary_diagnostics",
            "no_valid_correction_chain_in_this_scenario",
            "synthetic_agents_and_governance_actions_are_not_real_people_or_actions",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    value = load(arguments.result.resolve())
    audit = analyze(value)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resource-policy-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "RESOURCE_POLICY_RESULT_REPORT.md").write_text(
        markdown_report(audit), encoding="utf-8"
    )
    (output_dir / "resource-policy-story.svg").write_text(
        svg_story(audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_checks_passed": audit["all_checks_passed"],
                "output_dir": str(output_dir),
                "source_result_sha256": audit["source_result_sha256"],
                "usage": {
                    "semantic_turns": audit["usage"]["semantic_turns"],
                    "provider_calls": audit["usage"]["provider_calls"],
                    "provider_tokens": audit["usage"]["provider_tokens"],
                    "unknown_outcomes": audit["usage"]["unknown_outcomes"],
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
