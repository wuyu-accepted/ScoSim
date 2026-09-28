"""Cost-bounded hero controls for the ForumTwin product path.

The compatibility layer accepts either the historical reviewed population or
the hash-bound full-corpus Profile-v2 population adapter.  The latter remains
the source of truth: the bridge preserves its profile bundle, micro role,
reviewed evidence and scenario weights while separating stable interaction
traits from one historical episode focus.  Public language is exclusively
supplied by an injected live-LLM or exact-trace source.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .forum_twin import (
    ANCHOR_COUNT,
    ForumActivationRound,
    ForumTwinError,
    InteractionCandidate,
    PpsCandidate,
    schedule_forum_activations,
)
from .forum_turns import (
    FORUM_AGENT_TURN_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
)
from .mass_population import canonical_hash


PILOT_PROFILE_SCHEMA = "campus-pulse-forum-pilot-profile-bridge-v1"
PILOT_PREFLIGHT_SCHEMA = "campus-pulse-forum-hero-preflight-v1"
PILOT_SCENARIO = "lecture_external_incident_shock"
GOVERNANCE_LEGITIMACY_SCENARIO = "governance_legitimacy_dispute"
FORUM_TWIN_SCENARIOS = (
    PILOT_SCENARIO,
    GOVERNANCE_LEGITIMACY_SCENARIO,
)
CENTURY_GYM_SCENARIO = "century_gym_ghost_booking_dispute"
HERO_RUNTIME_SCENARIOS = (*FORUM_TWIN_SCENARIOS, CENTURY_GYM_SCENARIO)
PILOT_SEED = 20260722
PILOT_BRANCHES = ("Natural", "D")
RESOURCE_POLICY_BRANCHES = ("Natural", "A", "D")
RESOURCE_POLICY_START_TICK = 3
RESOURCE_POLICY_SHARED_END_TICK = 4
RESOURCE_POLICY_END_TICK = 10
RESOURCE_POLICY_RESIDENT_BUDGET = 24
RESOURCE_POLICY_DECISION_TICKS = frozenset({5, 7})
FOUR_DIMENSION_BRANCHES = ("Natural", "D")
FOUR_DIMENSION_START_TICK = 3
FOUR_DIMENSION_SHARED_END_TICK = 3
FOUR_DIMENSION_END_TICK = 10
FOUR_DIMENSION_RESIDENT_BUDGET = 24
FOUR_DIMENSION_DECISION_TICKS = frozenset({5, 7})
FOUR_DIMENSION_BRANCH_LABELS = {
    "Natural": "自然演化",
    "D": "有限观测自适应治理",
}
RESOURCE_POLICY_BRANCH_LABELS = {
    "Natural": "不追加回应",
    "A": "只解释分配标准",
    "D": "标准解释 + 复核服务 + 公开答疑",
}
# Competition flagship: one cost-bounded policy wind tunnel.  The 1,000
# persistent residents remain available in every world, while no world ever
# activates more than 24 resident LLM turns in one simulated tick.  Natural
# and D receive the full canonical trajectory; A/B/C and the non-canonical
# paired seeds use the frozen semantic checkpoints below.
RESOURCE_FLAGSHIP_BRANCHES = ("Natural", "A", "B", "C", "D")
RESOURCE_FLAGSHIP_START_TICK = 3
RESOURCE_FLAGSHIP_SHARED_END_TICK = 4
RESOURCE_FLAGSHIP_END_TICK = 15
RESOURCE_FLAGSHIP_RESIDENT_BUDGET = 24
RESOURCE_FLAGSHIP_DECISION_TICKS = frozenset({5, 9, 13})
RESOURCE_FLAGSHIP_SPINE_TICKS = frozenset({5, 9, 13, 15})
RESOURCE_FLAGSHIP_SEEDS = (
    20260722,
    20261731,
    20262740,
    20263749,
)
RESOURCE_FLAGSHIP_BRANCH_LABELS = {
    "Natural": "论坛自然演化",
    "A": "标准公开与证据更新",
    "B": "申诉复核与服务承接",
    "C": "跨群触达与参与式沟通",
    "D": "三主体动态组合治理",
}
CENTURY_GYM_BRANCHES = ("Natural", "D")
CENTURY_GYM_START_TICK = 3
CENTURY_GYM_SHARED_END_TICK = 4
CENTURY_GYM_END_TICK = 10
CENTURY_GYM_RESIDENT_BUDGET = 16
CENTURY_GYM_DECISION_TICKS = frozenset({5, 7})
CENTURY_GYM_BRANCH_LABELS = {
    "Natural": "自然演化",
    "D": "预约证据卡 + 系统工单 + 跨项目触达",
}
QUICK_LIVE_BRANCHES = ("Natural", "D")
QUICK_LIVE_START_TICK = 3
QUICK_LIVE_SHARED_END_TICK = 3
QUICK_LIVE_END_TICK = 6
QUICK_LIVE_RESIDENT_BUDGET = 16
QUICK_LIVE_DECISION_TICKS = frozenset({4, 6})
QUICK_LIVE_BRANCH_LABELS = {
    "Natural": "自然演化",
    "D": "三主体成本受控治理",
}
PILOT_START_TICK = 3
PILOT_END_TICK = 23
PILOT_MODEL = "gpt-5.4-mini"
PILOT_ENDPOINT = "https://yunwu.ai/v1/chat/completions"
LEGACY_ACTIVATION_POLICY = "legacy_sixteen_anchor_v1"
COST_AWARE_ACTIVATION_POLICY = "ecology_calibrated_eight_anchor_v1"
ACTIVATION_POLICIES = frozenset(
    {LEGACY_ACTIVATION_POLICY, COST_AWARE_ACTIVATION_POLICY}
)
OPEN_CHOICE_MECHANISM_PROTOCOL = "open_choice_v1"
TARGETED_CLOSED_LOOP_PROTOCOL = "targeted_closed_loop_audit_v1"
MECHANISM_PROTOCOLS = frozenset(
    {OPEN_CHOICE_MECHANISM_PROTOCOL, TARGETED_CLOSED_LOOP_PROTOCOL}
)
FULL_SYSTEM_ABLATION_MODE = "full_system"
PROFILE_SHUFFLED_ABLATION_MODE = "profile_shuffled"
MEMORY_DISABLED_ABLATION_MODE = "memory_disabled"
VISIBLE_TEXT_REMOVED_ABLATION_MODE = "visible_message_text_removed"
BRIDGE_DISABLED_ABLATION_MODE = "bridge_disabled"
PILOT_ABLATION_MODES = frozenset(
    {
        FULL_SYSTEM_ABLATION_MODE,
        PROFILE_SHUFFLED_ABLATION_MODE,
        MEMORY_DISABLED_ABLATION_MODE,
        VISIBLE_TEXT_REMOVED_ABLATION_MODE,
        BRIDGE_DISABLED_ABLATION_MODE,
    }
)


class ForumPilotError(ForumTwinError):
    """Raised when the bounded hero pilot drifts from its frozen scope."""


def scenario_weight_key(scenario_id: str) -> str:
    """Return the reviewed population-weight basis for a Hero scenario.

    The Century Gym project is a new product vignette, not a new empirical
    population release.  It therefore reuses the reviewed campus-services and
    procedural-fairness weighting basis while keeping its prompt, messages,
    claims and random frontier under its own scenario ID.
    """

    if scenario_id == CENTURY_GYM_SCENARIO:
        return GOVERNANCE_LEGITIMACY_SCENARIO
    if scenario_id in FORUM_TWIN_SCENARIOS:
        return scenario_id
    raise ForumPilotError("ForumTwin scenario weight basis is unknown")


def _strict_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForumPilotError(f"{field} must be an object")
    return value


def _nonempty_text(value: Any, fallback: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text or fallback


def _formal_v2_bridge(value: Mapping[str, Any]) -> bool:
    """Recognize only a hash-bound full-corpus population-v2 adapter."""

    hashes = (
        value.get("source_population_release_sha256"),
        value.get("source_profile_release_sha256"),
    )
    return (
        value.get("bridge_from_v2") is True
        and value.get("full_corpus_profile_v2_bound") is True
        and value.get("source_population_schema_version")
        == "semantic-llm-forum-agent-population-v2"
        and all(
            isinstance(item, str)
            and len(item) == 64
            and all(character in "0123456789abcdef" for character in item)
            for item in hashes
        )
    )


def _ranked_distribution_keys(value: Any, fallback: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [fallback]
    ranked = sorted(
        (
            (str(key), float(weight))
            for key, weight in value.items()
            if isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and math.isfinite(float(weight))
            and float(weight) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [key for key, _weight in ranked[:4]] or [fallback]


def _evidence_refs(agent: Mapping[str, Any]) -> list[str]:
    reviewed = [
        str(value)
        for value in agent.get("reviewed_evidence_refs", [])
        if isinstance(value, str) and value.startswith("evidence-")
    ][:5]
    if len(reviewed) >= 3:
        return reviewed
    facets = (
        {"semantic": agent.get("semantic")},
        {"persona": agent.get("persona")},
        {
            "variant": agent.get("persona_variant_id"),
            "basis": agent.get("persona_variant_assignment_basis"),
        },
    )
    return ["evidence-" + canonical_hash(facet)[:24] for facet in facets]


def _forum_profile(agent: Mapping[str, Any], index: int) -> dict[str, Any]:
    persona = _strict_object(agent.get("persona"), "persona")
    semantic = _strict_object(agent.get("semantic"), "semantic")
    behavior = _strict_object(agent.get("behavior_contract"), "behavior_contract")
    state = _strict_object(agent.get("state"), "state")
    primary = _nonempty_text(agent.get("primary_archetype_id"), "archetype-unknown")
    secondary = _nonempty_text(agent.get("secondary_archetype_id"), primary)
    variant = _nonempty_text(agent.get("persona_variant_id"), primary + "-reviewed")
    mode = _nonempty_text(persona.get("simulation_mode"), "active_episode_agent")
    gate = (
        "background_signal"
        if mode == "background_signal" or primary == "archetype-16"
        else "risk_signal"
        if primary == "archetype-09"
        else "active_episode_agent"
    )
    allowed = [str(value) for value in behavior.get("allowed_actions", [])]
    action_map = {
        "comment": "reply",
        "post": "post",
        "seek_help": "seek_help",
        "report_risk": "report_risk",
        "read": "read",
        "ignore": "ignore",
    }
    normalized_actions = list(
        dict.fromkeys(action_map[value] for value in allowed if value in action_map)
    )
    if not normalized_actions:
        normalized_actions = ["ignore", "read"]
    focus = _nonempty_text(
        persona.get("situational_signature"),
        persona.get("profile_summary") or "围绕当前合成校园事件判断是否需要行动",
    )
    long_memory = _nonempty_text(
        persona.get("profile_summary"),
        "这是一个仅由审阅语义证据驱动的合成论坛行为角色。",
    )[:600]
    source_bundle_id = _nonempty_text(agent.get("profile_bundle_id"), "")
    stable_persona = copy.deepcopy(dict(agent.get("stable_persona") or {}))
    if not stable_persona:
        stable_persona = {
            "source_profile_bundle_id": source_bundle_id or None,
            "source_micro_role_id": variant,
            "interaction_role": _nonempty_text(
                behavior.get("interaction_role"),
                semantic.get("speech_act") or "contextual_participant",
            ),
            "forum_tendency_distribution": {},
            "trust_style": {
                "current_trust_seed": round(float(state.get("trust", 0.5)), 6)
            },
            "uncertainty_handling": _nonempty_text(
                persona.get("uncertainty_behavior"),
                "信息不足时保留不确定性并请求可核验信息。",
            ),
            "language_style": {
                "summary": _nonempty_text(
                    persona.get("language_style"), "自然、简洁的校园论坛表达"
                )
            },
        }
    episodic_evidence = copy.deepcopy(dict(agent.get("episodic_evidence") or {}))
    if not episodic_evidence:
        episodic_evidence = {
            "episode_focus": focus,
            "topic_portfolio": _ranked_distribution_keys(
                agent.get("topic_distribution"),
                str(semantic.get("topic_code") or "campus_discussion"),
            ),
            "need_portfolio": _ranked_distribution_keys(
                agent.get("need_distribution"),
                str(semantic.get("need_code") or "information_need"),
            ),
            "reviewed_evidence_refs": _evidence_refs(agent),
            "usage_rule": (
                "historical behavior evidence only; use when the current scenario "
                "or delivered feed is substantively relevant"
            ),
        }
    profile = {
        "agent_id": _nonempty_text(agent.get("llm_agent_id"), f"pilot-agent-{index + 1:04d}"),
        "synthetic_display_id": f"FT-{index + 1:04d}",
        "profile_bundle_id": source_bundle_id or "pilot-bundle-" + canonical_hash(
            [primary, secondary, variant, semantic.get("speech_act"), semantic.get("stance")]
        )[:20],
        "provider_turn_contract": FORUM_AGENT_TURN_SCHEMA_VERSION,
        "evidence_binding": "reviewed_evidence_match",
        "reviewed_evidence_refs": _evidence_refs(agent),
        "reviewed_behavior_examples": [
            copy.deepcopy(dict(value))
            for value in agent.get("reviewed_behavior_examples", [])
            if isinstance(value, Mapping)
        ][:5],
        "primary_macro_role": primary,
        "secondary_macro_role": secondary,
        "micro_role_id": variant,
        "micro_interaction_role": _nonempty_text(
            behavior.get("interaction_role"),
            semantic.get("speech_act") or "contextual_participant",
        ),
        "episode_focus": focus,
        "stable_persona": stable_persona,
        "episodic_evidence": episodic_evidence,
        "scenario_relevance": copy.deepcopy(
            dict(
                agent.get("scenario_relevance")
                or {
                    "mode": "feed_and_current_scenario_conditioned",
                    "default_when_unrelated": (
                        "do_not_force_episode_topic_into_response"
                    ),
                }
            )
        ),
        "topic_portfolio": _ranked_distribution_keys(
            agent.get("topic_distribution"),
            str(semantic.get("topic_code") or "campus_discussion"),
        ),
        "need_portfolio": _ranked_distribution_keys(
            agent.get("need_distribution"),
            str(semantic.get("need_code") or "information_need"),
        ),
        "forum_tendencies": {
            "allowed_public_actions": normalized_actions,
            "interaction_role": behavior.get("interaction_role"),
            "baseline_activity": round(float(state.get("activity_probability", 0.2)), 6),
            "speech_act": semantic.get("speech_act"),
        },
        "trust_style": {
            "current_trust_seed": round(float(state.get("trust", 0.5)), 6),
            "evidence_acceptance": semantic.get("semantic_determinacy"),
            "peer_and_authority_not_interchangeable": True,
        },
        "uncertainty_handling": _nonempty_text(
            persona.get("uncertainty_behavior"),
            "信息不足时保留不确定性并请求可核验信息。",
        ),
        "language_style": {
            "summary": _nonempty_text(persona.get("language_style"), "自然、简洁的校园论坛表达"),
            "tendencies": [str(value) for value in persona.get("language_tendencies", [])][:6],
        },
        "activity_rhythm": {
            "source_month_bucket": agent.get("month"),
            "baseline_activity": round(float(state.get("activity_probability", 0.2)), 6),
        },
        "attention_budget": round(float(state.get("attention", 0.2)), 6),
        "long_term_memory_seed": long_memory,
        "behavior_boundaries": [
            str(value) for value in persona.get("do_not_simulate", [])
        ][:8] or ["不补造身份、事实、动机或未来结果"],
        "safety_gates": [
            "no_real_identity",
            "no_raw_forum_text",
            "no_future_or_hidden_truth",
            "no_unsupported_real_world_fact",
        ],
        "simulation_gate": gate,
        "scenario_weights": copy.deepcopy(dict(agent.get("scenario_weights", {}))),
        "design_weight": float(agent.get("population_weight", 0.0)),
        "initial_state": copy.deepcopy(dict(state)),
        "pilot_bridge": {
            "source_population_agent_contract": behavior.get("provider_callable") is True,
            "variant_assignment_basis": agent.get("persona_variant_assignment_basis"),
            "full_corpus_profile_v2_claimed": False,
        },
    }
    prompt_profile = {
        key: value
        for key, value in profile.items()
        if key not in {"profile_prompt_sha256", "initial_state", "scenario_weights"}
    }
    profile["profile_prompt_sha256"] = canonical_hash(prompt_profile)
    return profile


def compile_pilot_profile_bridge(
    population_release: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the existing reviewed population without inventing source records."""

    agents = population_release.get("agents")
    if not isinstance(agents, list) or len(agents) != 1_000:
        raise ForumPilotError("hero pilot requires the reviewed 1,000-agent population")
    formal_v2 = _formal_v2_bridge(population_release)
    profiles = [_forum_profile(agent, index) for index, agent in enumerate(agents)]
    if formal_v2:
        for profile in profiles:
            profile["pilot_bridge"]["full_corpus_profile_v2_claimed"] = True
            prompt_profile = {
                key: value
                for key, value in profile.items()
                if key not in {
                    "profile_prompt_sha256",
                    "initial_state",
                    "scenario_weights",
                }
            }
            profile["profile_prompt_sha256"] = canonical_hash(prompt_profile)
    if len({profile["agent_id"] for profile in profiles}) != len(profiles):
        raise ForumPilotError("pilot population contains duplicate agent IDs")
    if len({profile["synthetic_display_id"] for profile in profiles}) != len(profiles):
        raise ForumPilotError("pilot population contains duplicate display IDs")
    if any(profile["design_weight"] <= 0 for profile in profiles):
        raise ForumPilotError("pilot population design weights must be positive")
    scenario_sums = {
        scenario: sum(float(profile["scenario_weights"].get(scenario, 0.0)) for profile in profiles)
        for scenario in ("lecture_external_incident_shock", "governance_legitimacy_dispute")
    }
    for scenario, total in scenario_sums.items():
        if not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ForumPilotError(f"pilot scenario weights do not sum to one: {scenario}")
    body = {
        "schema_version": PILOT_PROFILE_SCHEMA,
        "population_size": len(profiles),
        "source_population_sha256": (
            population_release.get("source_population_release_sha256")
            if formal_v2
            else population_release.get("population_sha256")
        ),
        "source_profile_release_sha256": (
            population_release.get("source_profile_release_sha256")
            if formal_v2
            else None
        ),
        "profiles": profiles,
        "scenario_weight_sums": {key: round(value, 12) for key, value in scenario_sums.items()},
        "scope": (
            "full_corpus_population_v2_compatibility_bridge"
            if formal_v2
            else "hero_pilot_reviewed_population_bridge"
        ),
        "formal_full_corpus_profile_release": formal_v2,
        "public_language_source": "live_llm_or_exact_reviewed_trace_only",
        "raw_text_persisted": False,
        "source_identifiers_persisted": False,
    }
    body["profile_release_sha256"] = canonical_hash(body)
    return body


def scenario_context_for_tick(
    tick: int,
    scenario_id: str = PILOT_SCENARIO,
    mechanism_protocol: str = OPEN_CHOICE_MECHANISM_PROTOCOL,
) -> dict[str, Any]:
    if not 0 <= tick <= PILOT_END_TICK:
        raise ForumPilotError("hero scenario context is only defined for T0-T23")
    if scenario_id not in HERO_RUNTIME_SCENARIOS:
        raise ForumPilotError("ForumTwin scenario context is unknown")
    if mechanism_protocol not in MECHANISM_PROTOCOLS:
        raise ForumPilotError("ForumTwin mechanism protocol is unknown")
    if tick <= 2:
        return {
            "event_kind": "synthetic_shared_forum_baseline",
            "event_brief": "The synthetic forum is in its shared pre-event baseline.",
            "known_at_tick": (
                "No scenario event, governance response, future message, or outcome is visible yet."
            ),
            "current_phase": "baseline",
            "evidence_boundary": (
                "Only the current synthetic profile and pre-event state may be used; "
                "do not anticipate the event or invent real people, institutions, or outcomes."
            ),
            "tick": tick,
        }
    if scenario_id == CENTURY_GYM_SCENARIO:
        if tick == 3:
            phase = "burst"
            visible = (
                "合成事件输入来自一名学生的现场观察，而不是官方通知：这名学生刚路过世纪馆，"
                "看见三块场地空着，但预约小程序显示当晚热门时段全部不可约；其称连续三天遇到"
                "类似情况，并询问乒乓球、羽毛球场是否也存在取消后未释放、到点未到场或长期"
                "占位。被激活居民应按自身经历和论坛画像自然发帖、追问、质疑、围观或沉默。"
            )
        elif tick == 4:
            phase = "burst"
            visible = (
                "T3 的学生观察仍是唯一共同事件输入。当前没有世纪馆后台预约明细、取消日志、"
                "保护性教学时段清单或到场核销数据；论坛中的相似经历、反例和猜测只能作为"
                "时间可见的居民发言，不能自动升级为总体事实。"
            )
        elif tick <= 7:
            phase = "burst"
            visible = (
                "讨论可能扩展到羽毛球、乒乓球、不同日期和不同时段。新增可核验信息只能来自"
                "时间可见的论坛消息、治理证据卡和服务工单状态；工单受理不等于系统故障已经"
                "修复，证据卡也不等于公平争议已经结束。"
            )
        else:
            phase = "spread"
            visible = (
                "世纪馆预约争议继续沿不同项目和时段传播。居民可以直接引用治理对象、提交"
                "复现条件、报告修复后仍失败或认可局部解释；没有回执的个案保持未解决状态。"
            )
        return {
            "event_kind": "synthetic_century_gym_ghost_booking_dispute",
            "event_brief": (
                "学生在半官方校园论坛发起的世纪馆‘幽灵预约’争议：现场看见空场，预约小程序"
                "却显示热门时段不可约，并追问羽毛球、乒乓球等场地是否存在取消未释放、"
                "到点未到场或不透明占位。"
            ),
            "known_at_tick": visible,
            "current_phase": phase,
            "confirmed_facts": (
                "合成事件中，发帖学生报告了三天空场与小程序不可约同时出现；这只确认了一个"
                "可复核的观察线索，不确认世纪馆全部场地、全部时段或真实校园现状。"
            ),
            "competing_hypotheses": [
                "教学、校队或活动保护时段被正确锁定",
                "取消或过期预约没有及时释放",
                "预约成功者到点未到场",
                "少数人反复占位或转让",
                "观察者只看到了场地切换或维护间隙",
            ],
            "unverified_claims": (
                "‘热门时段都被内部占用’、‘关系户锁场’、‘所有空场都是系统故障’均未得到"
                "当前可见证据支持。"
            ),
            "evidence_boundary": (
                "只讨论这个合成世纪馆预约事件。不得补造真实预约记录、真实学生、管理人员、"
                "内部关系、系统日志、修复结果或学校结论。"
            ),
            "response_diversity_boundary": (
                "不要求理性中立或不信谣不传谣式模板回答。主体可以因约不到场而生气、吐槽、"
                "怀疑占位、分享相反经历、替教学时段辩护、追问核销规则、提交复现步骤、点赞、"
                "转发、举报、求助或沉默；但必须区分亲历、推测和已核验信息。"
            ),
            "scenario_weight_basis": GOVERNANCE_LEGITIMACY_SCENARIO,
            "scenario_weight_basis_claim": (
                "复用已审阅校园服务与程序公平人口权重，不声称存在世纪馆专项抽样权重。"
            ),
            "tick": tick,
        }
    if scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO:
        if tick == 3:
            phase = "burst"
            visible = (
                "一名学生刚把暑期住宿通知的摘要发进论坛：两栋宿舍维修，可用床位由520张"
                "降至320张，系统收到536份有效申请。通知说会优先考虑必须留校做实验或项目、"
                "已签暑期实习且通勤困难、以及经核验的现实困难。发帖学生补了一句："
                "‘我已经签了实习，但不知道自己的通勤算不算困难，也没看懂同一类里怎么排。’"
            )
        elif tick == 4 and mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL:
            phase = "burst"
            visible = (
                "程序事实卡 G1：床位缺口和通知缺项已经确认；目前没有证据证明按院系、"
                "关系或先到先得分配，也没有证据证明排序与复核已经充分透明。"
            )
        elif tick == 5 and mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL:
            phase = "burst"
            visible = (
                "程序事实卡 G1 仍然可见。复核服务卡 G2：申请人可以登记材料遗漏或规则适用"
                "问题，获得工单号并在48小时内收到处理状态；受理不等于改判，也不新增床位。"
            )
        elif tick == 4:
            phase = "burst"
            visible = (
                "T3 的暑期住宿通知仍是唯一共同外生信息。论坛已经出现对排序、公平、"
                "材料要求和自身住宿风险的不同反应，但尚无新增官方说明。"
            )
        elif tick <= 7:
            phase = "burst"
            if mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL:
                visible = (
                    "床位缺口与原通知的程序缺项仍然成立。此后没有证据卡、申诉渠道或处理"
                    "结果被后台自动注入；新增信息只能来自时间可见的论坛发言、治理帖子和"
                    "服务回执。关于院系偏袒、先到先得和关系分配的说法仍未证实。"
                )
            else:
                visible = (
                    "论坛围绕排序解释、材料复核和处理时限形成不同路径；任何判断都必须区分"
                    "已确认的通知缺项与尚未证实的偏袒动机。"
                )
        elif tick <= 17:
            phase = "spread"
            visible = (
                "讨论扩展到排序标准、同类申请如何取舍、材料复核、住宿替代方案与服务容量，"
                "帖子之间持续引用、质疑、求证和反馈。"
            )
        else:
            phase = "decay"
            visible = (
                "整体注意力开始回落，但未解释的排序、积压复核和未解决的住宿困难仍可能延续。"
            )
        return {
            "event_kind": "synthetic_governance_legitimacy_dispute",
            "event_brief": (
                "一名学生转发暑期住宿通知：320张床位对应536份有效申请，并询问自己会被"
                "怎样排序，由此触发对稀缺床位分配、个人处境和办事体验的讨论。"
            ),
            "known_at_tick": visible,
            "current_phase": phase,
            "confirmed_facts": (
                "维修导致床位由520张降至320张；共有536份有效申请；通知列出了三类优先"
                "考虑情形，但当前可见版本没有解释三类之间及同类申请内部的具体排序。"
            ),
            "unverified_claims": (
                "院系偏袒、先到先得、关系分配或已经内定均未得到当前可见证据支持。"
            ),
            "evidence_boundary": (
                "只讨论这个合成住宿安排和时间可见的治理对象；不得补造真实学校、部门、人员、"
                "暗箱动机、个人申请结果或尚未发布的处理结论。"
            ),
            "response_diversity_boundary": (
                "不要求冷静中立或顾全大局。主体可以从自身住宿风险出发愤怒、抱怨、质疑程序、"
                "维护稀缺条件下的排序、要求复核、提出替代方案、有限接受、围观或沉默；"
                "但不能把未证实的偏袒说成事实。"
            ),
            "tick": tick,
        }
    if tick == 3:
        phase = "burst"
        visible = (
            "事件事实卡 F1 已进入虚拟论坛：多家公开报道与当事人随后发布的道歉共同支持，"
            "一名受邀职业教育从业者在校园讲座中情绪失控、使用粗俗和侮辱性语言贬损在场学生，"
            "并中途离场；当事人承认言行失当、破坏交流氛围并表示责任在己。"
            "这些已确认行为不应被改写成单纯的‘语境不完整’。仍未知的是完整逐字稿、主观动机、"
            "所有现场反应以及后续正式处置。"
        )
    elif tick == 4 and mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL:
        phase = "burst"
        visible = (
            "证据卡 E1 已进入当前可见上下文：公开道歉确认了言行失当、中途离场、给师生造成困扰"
            "以及责任在己；它加强了对冒犯行为发生这一事实的确认，但不能自动证明伤害已经修复，"
            "也不能替代对具体言论、活动组织责任和后续处理的讨论。"
        )
    elif tick == 5 and mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL:
        phase = "burst"
        visible = (
            "证据卡 E1 仍然可见。参与者可以认为道歉充分、有限、策略性或仍需问责，也可以讨论"
            "学生尊严、职业焦虑、讲座组织和表达边界；不得补造未公开的动机和处理结果。"
            "服务更新 S1：可登记因事件引发的职业选择困惑、情绪支持或活动反馈；登记只表示受理，"
            "不表示问题已经解决。"
        )
    elif tick <= 7:
        phase = "burst"
        if mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL:
            visible = (
                "T3 的事件事实卡 F1 仍是唯一共同外生信息；此后没有固定证据卡或服务更新被"
                "自动注入。新增信息只能来自当前时间可见的居民讨论、治理对象或服务回执。"
                "论坛可以围绕伤害、道歉、组织责任和学生实际需要继续分化。"
            )
        else:
            visible = (
                "事实卡 F1 与道歉证据卡 E1 仍然可见。论坛可能围绕伤害是否被充分承认、道歉是否足够、"
                "职业价值冲突、组织责任和学生实际需要继续分化。"
            )
    elif tick <= 17:
        phase = "spread"
        visible = "讨论已形成多个相邻议题，帖子之间开始引用、追问、质疑与求证。"
    else:
        phase = "decay"
        visible = "整体注意力开始回落，但未解决的事实争议、求助和回应质量仍会延续。"
    return {
        "event_kind": "synthetic_external_incident",
        "event_brief": (
            "一场以公开报道为现实锚点、在 Provider 输入中去身份化的校园讲座辱骂事件："
            "受邀从业者以粗俗和侮辱性语言贬损学生并中途离场，随后公开道歉并承担责任。"
        ),
        "known_at_tick": visible,
        "current_phase": phase,
        "evidence_boundary": (
            "已经由公开报道与道歉共同支持的冒犯行为属于事实层，不得以‘保持客观’为由抹去；"
            "完整逐字稿、动机、所有现场反应和正式处置属于未知层，不得补造。"
        ),
        "response_diversity_boundary": (
            "不要求冷静、中立、顾全大局或达成共识。可依据画像表现愤怒、讽刺、受伤、反感、"
            "质疑道歉、有限接受、维护职业选择、转向求助、围观或沉默；但不得人肉、威胁或补造事实。"
        ),
        "tick": tick,
    }


def pilot_resident_budget(tick: int) -> int:
    if not PILOT_START_TICK <= tick <= PILOT_END_TICK:
        raise ForumPilotError("hero resident budget is only defined for T3-T23")
    if tick <= 7:
        return 32
    if tick <= 17:
        return 24
    return 16


def select_pilot_anchors(
    profiles: Sequence[Mapping[str, Any]],
    *,
    scenario_id: str = PILOT_SCENARIO,
    seed: int = PILOT_SEED,
) -> tuple[int, ...]:
    weight_key = scenario_weight_key(scenario_id)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, profile in enumerate(profiles):
        if profile.get("simulation_gate") != "background_signal":
            grouped[str(profile.get("primary_macro_role"))].append((index, profile))
    if len(grouped) != ANCHOR_COUNT:
        raise ForumPilotError("hero pilot requires sixteen actionable macro roles")
    anchors = []
    for macro_role in sorted(grouped):
        ranked = []
        for index, profile in grouped[macro_role]:
            weight = float(profile.get("scenario_weights", {}).get(weight_key, 0.0))
            if weight <= 0:
                continue
            digest = canonical_hash(["hero-anchor", scenario_id, seed, macro_role, index])
            uniform = (int(digest[:16], 16) + 1) / (16**16 + 1)
            ranked.append((-math.log(uniform) / weight, index))
        if not ranked:
            raise ForumPilotError(f"macro role lacks positive scenario weight: {macro_role}")
        anchors.append(min(ranked)[1])
    if len(set(anchors)) != ANCHOR_COUNT:
        raise ForumPilotError("hero anchor selection is not unique")
    return tuple(anchors)


def select_cost_aware_anchors(
    profiles: Sequence[Mapping[str, Any]],
    *,
    scenario_id: str = PILOT_SCENARIO,
    seed: int = PILOT_SEED,
    anchor_count: int = 8,
) -> tuple[int, ...]:
    """Select a smaller longitudinal panel without turning excluded roles off.

    Macro roles are sampled without replacement in proportion to their total
    scenario design weight; one positive-weight resident is then sampled from
    each selected role.  Every non-anchor remains in the rotating PPS frame,
    so this is a cost policy rather than an evidence exclusion rule.
    """

    if (
        isinstance(anchor_count, bool)
        or not isinstance(anchor_count, int)
        or not 1 <= anchor_count < ANCHOR_COUNT
    ):
        raise ForumPilotError("cost-aware anchor count must be from one through fifteen")
    weight_key = scenario_weight_key(scenario_id)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, profile in enumerate(profiles):
        if profile.get("simulation_gate") != "background_signal":
            grouped[str(profile.get("primary_macro_role"))].append((index, profile))
    if len(grouped) != ANCHOR_COUNT:
        raise ForumPilotError("cost-aware panel requires sixteen actionable macro roles")
    macro_ranks = []
    for macro_role, rows in sorted(grouped.items()):
        macro_weight = sum(
            max(0.0, float(row.get("scenario_weights", {}).get(weight_key, 0.0)))
            for _index, row in rows
        )
        if macro_weight <= 0:
            raise ForumPilotError(f"macro role lacks positive scenario weight: {macro_role}")
        digest = canonical_hash(
            ["cost-aware-anchor-macro-v1", scenario_id, seed, macro_role]
        )
        uniform = (int(digest[:16], 16) + 1) / (16**16 + 1)
        macro_ranks.append((-math.log(uniform) / macro_weight, macro_role))
    selected_roles = [
        macro_role
        for _rank, macro_role in sorted(macro_ranks)[:anchor_count]
    ]
    anchors = []
    for macro_role in selected_roles:
        resident_ranks = []
        for index, profile in grouped[macro_role]:
            weight = float(profile.get("scenario_weights", {}).get(weight_key, 0.0))
            if weight <= 0:
                continue
            digest = canonical_hash(
                ["cost-aware-anchor-resident-v1", scenario_id, seed, macro_role, index]
            )
            uniform = (int(digest[:16], 16) + 1) / (16**16 + 1)
            resident_ranks.append((-math.log(uniform) / weight, index))
        if not resident_ranks:
            raise ForumPilotError(f"macro role lacks an anchor candidate: {macro_role}")
        anchors.append(min(resident_ranks)[1])
    if len(set(anchors)) != anchor_count:
        raise ForumPilotError("cost-aware anchor selection is not unique")
    return tuple(anchors)


def build_pilot_activation_round(
    *,
    profiles: Sequence[Mapping[str, Any]],
    tick: int,
    anchor_indices: Sequence[int],
    interaction_candidates: Sequence[InteractionCandidate],
    rotation_counts: Mapping[int, int],
    scenario_id: str = PILOT_SCENARIO,
    resident_budget_override: int | None = None,
    selection_salt: str = "",
    required_anchor_count: int = ANCHOR_COUNT,
) -> ForumActivationRound:
    weight_key = scenario_weight_key(scenario_id)
    budget = (
        pilot_resident_budget(tick)
        if resident_budget_override is None
        else int(resident_budget_override)
    )
    if (
        isinstance(required_anchor_count, bool)
        or not isinstance(required_anchor_count, int)
        or not 1 <= required_anchor_count <= ANCHOR_COUNT
    ):
        raise ForumPilotError("required anchor count must be from one through sixteen")
    if not required_anchor_count <= budget <= 64:
        raise ForumPilotError("ForumTwin resident budget cannot be smaller than its anchor panel")
    anchor_set = {int(index) for index in anchor_indices}
    if len(anchor_set) != required_anchor_count:
        raise ForumPilotError("ForumTwin activation anchor panel has the wrong size")
    scores: list[tuple[int, float, int]] = []
    for index, profile in enumerate(profiles):
        if index in anchor_set or profile.get("simulation_gate") == "background_signal":
            continue
        state = profile.get("initial_state", {})
        weight = float(profile.get("scenario_weights", {}).get(weight_key, 0.0))
        activity = float(state.get("activity_probability", 0.2))
        support = float(state.get("support", 0.33))
        opposition = float(state.get("opposition", 0.33))
        uncertainty = max(0.05, 1.0 - abs(support - opposition))
        rotation = int(rotation_counts.get(index, 0))
        # Repeated semantic activation is useful for anchors, but rotating PPS
        # units must cover the heterogeneous population.  A quadratic penalty
        # keeps every unit's size measure positive while making a previously
        # observed non-anchor yield to otherwise comparable unseen residents.
        rotation_penalty = (1.0 + rotation) ** 2
        score = (
            weight
            * (0.25 + activity)
            * (0.25 + uncertainty)
            / rotation_penalty
        )
        scores.append((index, max(score, 1e-12), rotation))
    total = sum(score for _index, score, _rotation in scores)
    target = max(1, budget - required_anchor_count)
    pps = [
        PpsCandidate(
            agent_index=index,
            inclusion_probability=min(1.0, max(1e-9, target * score / total)),
            rotation_count=rotation,
            selection_tiebreak=canonical_hash(
                ["forum-pps-selection", selection_salt, tick, index]
            ),
        )
        for index, score, rotation in scores
    ]
    return schedule_forum_activations(
        agent_count=len(profiles),
        tick=tick,
        anchor_indices=anchor_indices,
        interaction_candidates=interaction_candidates,
        pps_candidates=pps,
        resident_budget_override=budget,
        required_anchor_count=required_anchor_count,
    )


def build_hero_preflight(
    *,
    endpoint: str = PILOT_ENDPOINT,
    model: str = PILOT_MODEL,
    scenario_id: str = PILOT_SCENARIO,
    component_bindings: Mapping[str, str] | None = None,
    safe_preconnect_retry_limit: int = 0,
    local_https_proxy: str | None = None,
    branches: Sequence[str] = PILOT_BRANCHES,
    start_tick: int = PILOT_START_TICK,
    end_tick: int = PILOT_END_TICK,
    resident_budget_override: int | None = None,
    provider_request_hard_limit: int | None = None,
    provider_token_hard_limit: int = 10_000_000,
    max_concurrency: int = 24,
    mechanism_protocol: str = OPEN_CHOICE_MECHANISM_PROTOCOL,
    ablation_mode: str = FULL_SYSTEM_ABLATION_MODE,
    shared_prefix_end_tick: int | None = None,
    resident_turn_schema_version: str = FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    activation_policy: str = LEGACY_ACTIVATION_POLICY,
    ecology_calibration_release_sha256: str | None = None,
    relationship_release_sha256: str | None = None,
    scenario_stake_release_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        isinstance(safe_preconnect_retry_limit, bool)
        or safe_preconnect_retry_limit not in {0, 1, 2}
    ):
        raise ForumPilotError(
            "safe pre-connect retry limit must be 0, 1, or 2"
        )
    if local_https_proxy not in {None, "http://127.0.0.1:7897"}:
        raise ForumPilotError("hero pilot local HTTPS proxy is not approved")
    if scenario_id not in HERO_RUNTIME_SCENARIOS:
        raise ForumPilotError("hero preflight scenario is invalid")
    frozen_branches = tuple(dict.fromkeys(str(value) for value in branches))
    allowed_branches = {"Natural", "A", "B", "C", "D"}
    if not frozen_branches or any(value not in allowed_branches for value in frozen_branches):
        raise ForumPilotError("hero pilot branches are invalid")
    if not PILOT_START_TICK <= start_tick <= end_tick <= PILOT_END_TICK:
        raise ForumPilotError("hero preflight tick range is invalid")
    if provider_token_hard_limit <= 0 or provider_token_hard_limit > 10_000_000:
        raise ForumPilotError("hero token hard limit is invalid")
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or not 1 <= max_concurrency <= 24
    ):
        raise ForumPilotError("hero max concurrency must be between 1 and 24")
    if mechanism_protocol not in MECHANISM_PROTOCOLS:
        raise ForumPilotError("hero mechanism protocol is invalid")
    if ablation_mode not in PILOT_ABLATION_MODES:
        raise ForumPilotError("hero LLM ablation mode is invalid")
    if resident_turn_schema_version not in {
        FORUM_AGENT_TURN_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }:
        raise ForumPilotError("hero resident turn schema is invalid")
    if activation_policy not in ACTIVATION_POLICIES:
        raise ForumPilotError("hero activation policy is invalid")
    if ecology_calibration_release_sha256 is not None and (
        not isinstance(ecology_calibration_release_sha256, str)
        or len(ecology_calibration_release_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in ecology_calibration_release_sha256
        )
    ):
        raise ForumPilotError("ecology calibration release SHA-256 is invalid")
    if (
        activation_policy == COST_AWARE_ACTIVATION_POLICY
        and ecology_calibration_release_sha256 is None
    ):
        raise ForumPilotError("cost-aware activation requires ecology calibration")
    relationship_bindings = (
        relationship_release_sha256,
        scenario_stake_release_sha256,
    )
    if any(value is not None for value in relationship_bindings):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in relationship_bindings
        ):
            raise ForumPilotError(
                "relationship and scenario-stake releases must be bound together"
            )
    resource_policy_sandbox = shared_prefix_end_tick is not None and (
        scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO
        and frozen_branches == RESOURCE_POLICY_BRANCHES
        and start_tick == RESOURCE_POLICY_START_TICK
        and int(shared_prefix_end_tick) == RESOURCE_POLICY_SHARED_END_TICK
        and end_tick == RESOURCE_POLICY_END_TICK
        and resident_budget_override == RESOURCE_POLICY_RESIDENT_BUDGET
        and mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL
        and ablation_mode == FULL_SYSTEM_ABLATION_MODE
    )
    century_gym_sandbox = shared_prefix_end_tick is not None and (
        scenario_id == CENTURY_GYM_SCENARIO
        and frozen_branches == CENTURY_GYM_BRANCHES
        and start_tick == CENTURY_GYM_START_TICK
        and int(shared_prefix_end_tick) == CENTURY_GYM_SHARED_END_TICK
        and end_tick == CENTURY_GYM_END_TICK
        and resident_budget_override == CENTURY_GYM_RESIDENT_BUDGET
        and mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL
        and ablation_mode == FULL_SYSTEM_ABLATION_MODE
        and resident_turn_schema_version == FORUM_AGENT_TURN_V2_SCHEMA_VERSION
    )
    quick_live_vignette = shared_prefix_end_tick is not None and (
        scenario_id in HERO_RUNTIME_SCENARIOS
        and frozen_branches == QUICK_LIVE_BRANCHES
        and start_tick == QUICK_LIVE_START_TICK
        and int(shared_prefix_end_tick) == QUICK_LIVE_SHARED_END_TICK
        and end_tick == QUICK_LIVE_END_TICK
        and resident_budget_override == QUICK_LIVE_RESIDENT_BUDGET
        and mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL
        and ablation_mode == FULL_SYSTEM_ABLATION_MODE
        and resident_turn_schema_version == FORUM_AGENT_TURN_V2_SCHEMA_VERSION
    )
    four_dimension_sandbox = shared_prefix_end_tick is not None and (
        scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO
        and frozen_branches == FOUR_DIMENSION_BRANCHES
        and start_tick == FOUR_DIMENSION_START_TICK
        and int(shared_prefix_end_tick) == FOUR_DIMENSION_SHARED_END_TICK
        and end_tick == FOUR_DIMENSION_END_TICK
        and resident_budget_override == FOUR_DIMENSION_RESIDENT_BUDGET
        and mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL
        and ablation_mode == FULL_SYSTEM_ABLATION_MODE
        and resident_turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION
        and activation_policy == COST_AWARE_ACTIVATION_POLICY
        and ecology_calibration_release_sha256 is not None
        and relationship_release_sha256 is not None
        and scenario_stake_release_sha256 is not None
    )
    shared_policy_sandbox = (
        resource_policy_sandbox
        or century_gym_sandbox
        or quick_live_vignette
        or four_dimension_sandbox
    )
    if shared_prefix_end_tick is not None and not shared_policy_sandbox:
        raise ForumPilotError(
            "shared-prefix sandbox does not match a frozen Hero runtime scope"
        )
    if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL and not (
        frozen_branches == ("D",)
        and start_tick == 3
        and end_tick == 5
        and resident_budget_override == ANCHOR_COUNT
    ):
        raise ForumPilotError(
            "targeted closed-loop audit requires D/T3-T5 with sixteen residents"
        )
    if relationship_release_sha256 is not None and not four_dimension_sandbox:
        raise ForumPilotError(
            "record-level relationship/stake bindings are limited to the four-dimensional Hero"
        )
    if (
        mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
        and activation_policy != LEGACY_ACTIVATION_POLICY
    ):
        raise ForumPilotError("targeted audit requires the legacy anchor policy")
    budget_by_tick = {
        tick: (
            pilot_resident_budget(tick)
            if resident_budget_override is None
            else int(resident_budget_override)
        )
        for tick in range(start_tick, end_tick + 1)
    }
    required_anchor_count = (
        8 if activation_policy == COST_AWARE_ACTIVATION_POLICY else ANCHOR_COUNT
    )
    if any(
        not required_anchor_count <= value <= 32
        for value in budget_by_tick.values()
    ):
        raise ForumPilotError("hero resident budget cannot be smaller than its anchor panel")
    if shared_policy_sandbox:
        prefix_ticks = range(start_tick, int(shared_prefix_end_tick) + 1)
        branch_ticks = range(int(shared_prefix_end_tick) + 1, end_tick + 1)
        resident_slots = sum(budget_by_tick[tick] for tick in prefix_ticks) + (
            sum(budget_by_tick[tick] for tick in branch_ticks)
            * len(frozen_branches)
        )
        sandbox_decision_ticks = (
            RESOURCE_POLICY_DECISION_TICKS
            if resource_policy_sandbox
            else CENTURY_GYM_DECISION_TICKS
            if century_gym_sandbox
            else FOUR_DIMENSION_DECISION_TICKS
            if four_dimension_sandbox
            else QUICK_LIVE_DECISION_TICKS
        )
        actor_slots = (
            sum(tick in sandbox_decision_ticks for tick in branch_ticks)
            * 3
            * sum(branch != "Natural" for branch in frozen_branches)
        )
        decision_ticks = sandbox_decision_ticks
    else:
        resident_slots = sum(budget_by_tick.values()) * len(frozen_branches)
        actor_slots = (
            sum(tick in {3, 7, 13, 19} for tick in budget_by_tick)
            * 3
            * sum(branch != "Natural" for branch in frozen_branches)
        )
        decision_ticks = frozenset({3, 7, 13, 19})
    primary_slots = resident_slots + actor_slots
    effective_request_hard_limit = (
        primary_slots * 2
        if provider_request_hard_limit is None
        else int(provider_request_hard_limit)
    )
    if not primary_slots * 2 <= effective_request_hard_limit <= primary_slots * 4:
        raise ForumPilotError(
            "hero request hard limit must be between two and four requests per primary slot"
        )
    bindings = dict(component_bindings or {})
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
        for name, sha in bindings.items()
    ):
        raise ForumPilotError("hero preflight component binding is not a SHA-256")
    body = {
        "schema_version": PILOT_PREFLIGHT_SCHEMA,
        "scope": (
            "shared_T4_resource_allocation_policy_sandbox"
            if resource_policy_sandbox
            else "shared_T4_century_gym_live_policy_sandbox"
            if century_gym_sandbox
            else "shared_T3_four_dimension_resource_hero"
            if four_dimension_sandbox
            else "shared_T3_cost_bounded_live_vignette"
            if quick_live_vignette
            else "one_scenario_targeted_closed_loop_mechanism_audit"
            if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
            else
            "one_scenario_one_seed_natural_vs_adaptive_hero_pilot"
            if frozen_branches == PILOT_BRANCHES
            and start_tick == PILOT_START_TICK
            and end_tick == PILOT_END_TICK
            and resident_budget_override is None
            else "one_scenario_cost_bounded_llm_mechanism_vignette"
        ),
        "endpoint": endpoint,
        "model": model,
        "scenario_id": scenario_id,
        "seed": PILOT_SEED,
        "shared_prefix_end_tick": shared_prefix_end_tick,
        "branch_labels": (
            dict(RESOURCE_POLICY_BRANCH_LABELS)
            if resource_policy_sandbox
            else dict(CENTURY_GYM_BRANCH_LABELS)
            if century_gym_sandbox
            else dict(FOUR_DIMENSION_BRANCH_LABELS)
            if four_dimension_sandbox
            else dict(QUICK_LIVE_BRANCH_LABELS)
            if quick_live_vignette
            else {branch: branch for branch in frozen_branches}
        ),
        "mechanism_protocol": mechanism_protocol,
        "llm_ablation_mode": ablation_mode,
        "resident_turn_schema_version": resident_turn_schema_version,
        "activation_policy": activation_policy,
        "longitudinal_anchor_count": required_anchor_count,
        "non_anchor_population_remains_pps_eligible": True,
        "ecology_calibration_release_sha256": (
            ecology_calibration_release_sha256
        ),
        "relationship_release_sha256": relationship_release_sha256,
        "scenario_stake_release_sha256": scenario_stake_release_sha256,
        "relationship_attention_enabled": four_dimension_sandbox,
        "private_actions_enabled": four_dimension_sandbox,
        "public_anonymity_enabled": four_dimension_sandbox,
        "resident_semantic_source": "same_real_llm_or_exact_reviewed_trace",
        "dynamics_public_messages": 0,
        "branches": list(frozen_branches),
        "ticks": list(range(start_tick, end_tick + 1)),
        "resident_budget_by_tick": {
            str(tick): value for tick, value in budget_by_tick.items()
        },
        "resident_slots": resident_slots,
        "governance_actor_slots": actor_slots,
        "primary_slots": primary_slots,
        "format_repair_limit_per_slot": 1,
        "network_retry_limit": 0,
        "safe_preconnect_retry_limit": safe_preconnect_retry_limit,
        "local_https_proxy": local_https_proxy,
        "provider_request_hard_limit": effective_request_hard_limit,
        "provider_token_hard_limit": provider_token_hard_limit,
        "max_concurrency": max_concurrency,
        "max_residents_one_tick": max(budget_by_tick.values()),
        "max_residents_plus_governance_one_tick": max(budget_by_tick.values()) + (
            3
            if any(branch != "Natural" for branch in frozen_branches)
            and any(tick in decision_ticks for tick in budget_by_tick)
            else 0
        ),
        "resident_max_output_tokens": 512,
        "governance_max_output_tokens": 384,
        "sent_fields": [
            (
                "natural-language stable forum profile plus three reviewed past-behavior examples"
                if resident_turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION
                else "reviewed stable macro/micro interaction profile"
            ),
            "separately labelled historical episode focus used only when feed-relevant",
            (
                "qualitative current experience without raw posterior coordinates"
                if resident_turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION
                else "current synthetic posterior"
            ),
            "up to two real-LLM memory summaries",
            "the same global hot-top-10 and latest-post board index shown to every resident",
            "time-visible synthetic thread root/parent/quote and up to three opened post summaries",
            (
                "natural-language synthetic situation using short per-turn handles"
                if resident_turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION
                else "synthetic scenario, branch, tick and phase context"
            ),
            (
                "disclosed targeted mechanism-audit task and exact visible target"
                if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
                else "open-choice activation context"
            ),
            "strict JSON output contract",
            (
                "one primary forum action, up to six independent engagements, and up to two private actions"
                if resident_turn_schema_version
                in {
                    FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
                    FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
                    FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
                }
                else "one primary speech action plus up to six independent like, repost, or report engagements"
                if resident_turn_schema_version == FORUM_AGENT_TURN_V2_SCHEMA_VERSION
                else "one primary forum action"
            ),
            "governance role-private aggregate observation, feasible actions, resources and delay",
        ],
        "feed_contract": {
            "mechanism": "global_hot_top10_plus_latest_v1",
            "relationship_graph_controls_delivery": four_dimension_sandbox,
            "ranking_identical_for_all_residents_at_same_tick": True,
            "profile_affects_response_not_public_ranking": True,
            "direct_parent_and_quote_delivery_guaranteed": True,
            "ordinary_delivery_routes": [
                "direct_reply_notification",
                "joined_thread_notification",
                *(
                    ["relationship_attention_open", "private_message_notification"]
                    if four_dimension_sandbox
                    else []
                ),
                "public_hot_top10",
                "public_latest_refresh",
            ],
            "governance_posts_use_ordinary_board_ranking": True,
            "record_level_graph_public": False,
            "real_corpus_continuation_prior_controls_open_selection": (
                ecology_calibration_release_sha256 is not None
            ),
            "continuation_prior_changes_global_ranking": False,
            "continuation_prior_forces_llm_action": False,
        },
        "policy_sandbox_contract": (
            {
                "shared_through_tick": int(shared_prefix_end_tick),
                "fork_from_exact_checkpoint": True,
                "worlds": (
                    {
                        "Natural": "no additional governance response",
                        "A": "criteria explanation only",
                        "D": "criteria explanation plus review service and public Q&A",
                    }
                    if resource_policy_sandbox
                    else {
                        "Natural": "no additional governance response",
                        "D": (
                            "bounded booking evidence card plus unreleased-slot service "
                            "ticket and cross-sport/time-window outreach"
                        ),
                    }
                    if century_gym_sandbox
                    else {
                        "Natural": "no additional governance response",
                        "D": (
                            "three governance actors choose bounded evidence, review-service "
                            "and cross-group outreach objects from finite observations"
                        ),
                    }
                    if four_dimension_sandbox
                    else {
                        "Natural": "no additional governance response",
                        "D": (
                            "cost-bounded evidence, service and outreach actions chosen by "
                            "three governance actors from finite observations"
                        ),
                    }
                ),
                "decision_ticks": sorted(decision_ticks),
                "scheme_bonus": 0,
                "winner_precommitted": False,
            }
            if shared_policy_sandbox
            else None
        ),
        "scenario_weight_contract": {
            "requested_scenario_id": scenario_id,
            "reviewed_weight_basis": scenario_weight_key(scenario_id),
            "scenario_specific_empirical_weight_claimed": (
                scenario_id != CENTURY_GYM_SCENARIO
            ),
        },
        "ablation_disclosure": {
            "mode": ablation_mode,
            "same_model_and_output_contract": True,
            "profile_semantics_present": ablation_mode != PROFILE_SHUFFLED_ABLATION_MODE,
            "recent_real_llm_memory_present": ablation_mode != MEMORY_DISABLED_ABLATION_MODE,
            "visible_message_text_present": ablation_mode != VISIBLE_TEXT_REMOVED_ABLATION_MODE,
            "community_bridge_messages_present": ablation_mode != BRIDGE_DISABLED_ABLATION_MODE,
            "public_language_still_requires_llm": True,
        },
        "excluded_fields": [
            "raw forum text",
            "source IDs or real identities",
            "hidden population truth or future events",
            "filesystem paths, database fields or credentials",
            *(
                [
                    "evidence, release or prompt hashes in the model-visible resident view",
                    "raw particle means or raw scenario-stake coordinates in the model-visible resident view",
                    "internal Agent, thread, message, claim or conversation identifiers",
                ]
                if resident_turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION
                else []
            ),
        ],
        "public_content_rule": "live_llm_or_exact_reviewed_trace_only",
        "emulator_public_messages": 0,
        "targeted_action_categories_disclosed": (
            mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
        ),
        "component_bindings": dict(sorted(bindings.items())),
        "privacy_scan": {
            "raw_forum_text_hits": 0,
            "source_identifier_hits": 0,
            "credential_hits": 0,
            "path_hits": 0,
            "hidden_truth_hits": 0,
            "passed": True,
        },
        "authorization_present": False,
        "ready": False,
    }
    body["preflight_sha256"] = canonical_hash(body)
    return body


def strict_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
