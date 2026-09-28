"""Mechanism-explicit governance arena built on the reviewed episode population.

This module is additive to the audited MassSim v1 release.  It corrects the
v1 baseline-drift problem and adds two genuinely different intervention
mechanisms, explicit competing-information diffusion, and a bounded
multi-actor best-response game.  Everything remains deterministic,
zero-Provider, aggregate-only, and model-conditional.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .mass_engine import TIMELINE_PHASES, TIMELINE_TICKS
from .mass_population import canonical_hash


ARENA_SCHEMA_VERSION = "campus-pulse-governance-arena-result-v1"
PROFILE_PRIOR_SCHEMA_VERSION = (
    "campus-pulse-reviewed-profile-response-priors-v1"
)
DIFFUSION_SCHEMA_VERSION = "campus-pulse-competing-information-diffusion-v1"
GAME_SCHEMA_VERSION = "campus-pulse-bounded-sequential-best-response-v1"

SCHEME_IDS = (
    "natural",
    "rapid_transparency",
    "participatory_service",
)
DECISION_TICKS = (3, 7, 11)
DEFAULT_SEEDS = (
    20260730,
    20260731,
    20260732,
    20260733,
    20260734,
    20260735,
    20260736,
    20260737,
)

SCENARIOS = {
    "lecture_external_incident_shock": {
        "label": "外部事件冲击",
        "shock_kind": "external_uncertain_incident",
        "attention_force": 0.50,
        "concern_force": 0.44,
        "trust_force": 0.035,
        "rumor_seed_fraction": 0.075,
        "verified_seed_fraction": 0.012,
        "rumor_virality": 1.18,
        "service_pressure": 0.62,
        "affinity_weights": {
            "fact_need": 0.30,
            "service_need": 0.22,
            "voice_need": 0.08,
            "risk_sensitivity": 0.30,
            "peer_orientation": 0.10,
        },
    },
    "tongzhou_governance_information_shock": {
        "label": "治理信息冲击",
        "shock_kind": "governance_legitimacy_dispute",
        "attention_force": 0.43,
        "concern_force": 0.35,
        "trust_force": 0.15,
        "rumor_seed_fraction": 0.052,
        "verified_seed_fraction": 0.018,
        "rumor_virality": 1.02,
        "service_pressure": 0.42,
        "affinity_weights": {
            "fact_need": 0.27,
            "service_need": 0.15,
            "voice_need": 0.31,
            "risk_sensitivity": 0.12,
            "peer_orientation": 0.15,
        },
    },
}

SCHEMES = {
    "natural": {
        "label": "自然演化",
        "mechanism": "无新增治理动作的配对反事实",
        "color": "#7f8d9d",
    },
    "rapid_transparency": {
        "label": "方案 A｜快速透明响应",
        "mechanism": (
            "用可核验时间线、证据更新和跨群体事实中继压低信息不确定性；"
            "优势是快，风险是单向发布可能留下参与感缺口。"
        ),
        "color": "#51e2d0",
    },
    "participatory_service": {
        "label": "方案 B｜参与式服务协同",
        "mechanism": (
            "用诉求收集、服务扩容和群体对话提升承接能力与程序正当性；"
            "优势是包容和持续，风险是早期澄清速度较慢、资源成本更高。"
        ),
        "color": "#ffbf69",
    },
}

POLICY_OBJECTIVES = {
    "lecture_external_incident_shock": {
        "label": "外部不确定事件：优先遏制错误信息并兼顾峰值风险",
        "weights": {
            "rumor_auc_reduction_ratio": 0.38,
            "concern_auc_reduction_ratio": 0.18,
            "end_trust_gain_ratio": 0.15,
            "end_satisfaction_gain_ratio": 0.05,
            "end_voice_gap_reduction_ratio": 0.04,
            "end_polarization_reduction_ratio": 0.05,
            "resource_cost_ratio": -0.15,
        },
    },
    "tongzhou_governance_information_shock": {
        "label": "治理正当性争议：优先修复信任、参与感与服务承接",
        "weights": {
            "rumor_auc_reduction_ratio": 0.12,
            "concern_auc_reduction_ratio": 0.18,
            "end_trust_gain_ratio": 0.18,
            "end_satisfaction_gain_ratio": 0.15,
            "end_voice_gap_reduction_ratio": 0.22,
            "end_polarization_reduction_ratio": 0.08,
            "resource_cost_ratio": -0.07,
        },
    },
}

TRAIT_TERMS = {
    "fact_need": (
        "核实",
        "明确",
        "事实",
        "规则",
        "说明",
        "来源",
        "口径",
        "确认",
        "解释",
        "时间",
    ),
    "service_need": (
        "支持",
        "帮助",
        "办理",
        "渠道",
        "服务",
        "资源",
        "步骤",
        "方案",
        "求助",
    ),
    "voice_need": (
        "参与",
        "反馈",
        "协商",
        "沟通",
        "诉求",
        "回应",
        "讨论",
        "表达",
    ),
    "risk_sensitivity": (
        "风险",
        "安全",
        "紧急",
        "担忧",
        "后果",
        "影响",
        "危机",
        "压力",
    ),
    "peer_orientation": (
        "他人",
        "经验",
        "同学",
        "群体",
        "讨论",
        "分享",
        "交流",
        "知情者",
    ),
    "verification_need": (
        "可核",
        "正式",
        "官方",
        "来源",
        "确认",
        "核实",
        "证据",
    ),
}

RISK_SCORE = {
    "none": 0.08,
    "low": 0.25,
    "medium": 0.52,
    "high": 0.82,
    "uncertain": 0.42,
}

STANCE_ANCHOR = {
    "supportive": (0.72, 0.10),
    "neutral": (0.28, 0.20),
    "concerned": (0.20, 0.60),
    "critical": (0.10, 0.78),
    "ambivalent": (0.38, 0.42),
}

ACTIVITY_ANCHOR = {
    "risk_report": 0.62,
    "complaint": 0.58,
    "help_request": 0.52,
    "discussion": 0.49,
    "information_share": 0.47,
    "experience_share": 0.43,
    "question": 0.40,
    "emotional_expression": 0.45,
    "other": 0.28,
}

BOUNDED_METRICS = (
    "attention",
    "concern",
    "trust",
    "satisfaction",
    "activity_probability",
    "rumor_belief",
    "verified_belief",
    "rumor_reach",
    "verified_reach",
    "polarization",
    "voice_gap",
    "service_overload",
    "severe_risk_share",
)


class GovernanceArenaError(ValueError):
    """Raised when the governance arena violates its frozen mechanisms."""


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    bounded = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(bounded.values())
    if total <= 0:
        return {key: 1.0 / len(bounded) for key in bounded}
    return {key: value / total for key, value in bounded.items()}


def _hash_uniform(*parts: Any) -> float:
    return int(canonical_hash(list(parts))[:13], 16) / float(16**13)


def _counter_uniform(*counters: int) -> float:
    """Fast deterministic counter-based variate for the inner simulation loop.

    This uses SplitMix64-style integer mixing, not a cryptographic generator.
    Counter tuples are frozen and contain no identities.  Omitting the policy
    branch from those tuples supplies paired common random numbers across
    interventions without millions of JSON/SHA serializations.
    """

    mask = (1 << 64) - 1
    value = 0x6A09E667F3BCC909
    for counter in counters:
        value = (
            value
            + (int(counter) & mask)
            + 0x9E3779B97F4A7C15
        ) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return ((value & mask) >> 11) / float(1 << 53)


def _weighted_mean(
    agents: Sequence[Mapping[str, Any]],
    values: Iterable[float],
) -> float:
    return sum(
        float(agent["population_weight"]) * float(value)
        for agent, value in zip(agents, values)
    )


def _phase_intensity(tick: int) -> float:
    return (
        0.00,
        0.00,
        0.72,
        0.92,
        1.00,
        0.94,
        0.84,
        0.76,
        0.68,
        0.60,
        0.53,
        0.46,
        0.35,
        0.25,
        0.17,
        0.10,
    )[tick]


def _flatten_profile_basis(profile: Mapping[str, Any]) -> str:
    fields = (
        profile.get("profile_summary"),
        profile.get("observable_core_tendencies"),
        profile.get("response_expectations"),
        profile.get("trigger_conditions"),
        profile.get("language_tendencies"),
    )
    parts = []
    for value in fields:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return "\n".join(parts)


def compile_profile_response_priors(
    persona_release: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile auditable response priors from reviewed profile fields only."""

    profiles = persona_release.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 17:
        raise GovernanceArenaError(
            "governance arena requires the reviewed 17-profile release"
        )
    priors = []
    for profile in profiles:
        archetype_id = str(profile.get("archetype_id") or "")
        direct_review = profile.get("direct_review")
        if (
            not archetype_id
            or not isinstance(direct_review, Mapping)
            or direct_review.get("simulation_mode")
            not in {
                "active_episode_agent",
                "governance_risk_signal",
                "background_signal_only",
            }
        ):
            raise GovernanceArenaError(
                "persona profile has no reviewed simulation gate"
            )
        basis = _flatten_profile_basis(profile)
        traits = {}
        evidence_counts = {}
        for trait, terms in TRAIT_TERMS.items():
            count = sum(basis.count(term) for term in terms)
            evidence_counts[trait] = count
            # A bounded transform avoids claiming that keyword frequency is a
            # measured psychological scale.  It is only a response prior.
            traits[trait] = round(
                _clip(0.18 + 0.19 * math.log1p(count), 0.18, 0.92),
                6,
            )
        priors.append(
            {
                "archetype_id": archetype_id,
                "traits": traits,
                "eligible_for_population_agent": bool(
                    direct_review.get("eligible_for_population_agent")
                ),
                "simulation_mode": direct_review["simulation_mode"],
                "term_evidence_counts": evidence_counts,
                "profile_evidence_fingerprint": profile.get(
                    "evidence_fingerprint"
                ),
                "source_fields": [
                    "profile_summary",
                    "observable_core_tendencies",
                    "response_expectations",
                    "trigger_conditions",
                    "language_tendencies",
                ],
                "excluded_from_scoring": [
                    "do_not_simulate",
                    "evidence_limitations",
                    "variant evidence_record_ids",
                ],
            }
        )
    body = {
        "schema_version": PROFILE_PRIOR_SCHEMA_VERSION,
        "source_persona_release_fingerprint": persona_release.get(
            "run_fingerprint"
        ),
        "profile_count": len(priors),
        "profiles": sorted(priors, key=lambda item: item["archetype_id"]),
        "claim_boundary": (
            "Response priors are scenario parameters grounded in reviewed "
            "profile wording; they are not measured stable traits."
        ),
    }
    body["profile_prior_sha256"] = canonical_hash(body)
    return body


def _agent_traits(
    agent: Mapping[str, Any],
    prior_by_archetype: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    prior = prior_by_archetype[str(agent["primary_archetype_id"])]
    traits = dict(prior["traits"])
    traits["_active_gate"] = float(
        prior["simulation_mode"] != "background_signal_only"
    )
    traits["_risk_signal_gate"] = float(
        prior["simulation_mode"] == "governance_risk_signal"
    )
    semantic = agent["semantic"]
    speech_act = str(semantic.get("speech_act"))
    risk = RISK_SCORE.get(str(semantic.get("risk_level")), 0.35)
    if speech_act == "help_request":
        traits["service_need"] = _clip(traits["service_need"] + 0.18)
    if speech_act in {"question", "information_share"}:
        traits["fact_need"] = _clip(traits["fact_need"] + 0.14)
        traits["verification_need"] = _clip(
            traits["verification_need"] + 0.12
        )
    if speech_act in {"complaint", "discussion", "emotional_expression"}:
        traits["voice_need"] = _clip(traits["voice_need"] + 0.14)
    if speech_act == "risk_report":
        traits["risk_sensitivity"] = _clip(
            traits["risk_sensitivity"] + 0.20
        )
    traits["risk_sensitivity"] = _clip(
        0.68 * traits["risk_sensitivity"] + 0.32 * risk
    )
    return traits


def _scenario_affinity(
    traits: Mapping[str, float],
    scenario: Mapping[str, Any],
) -> float:
    return _clip(
        sum(
            float(scenario["affinity_weights"][key])
            * float(traits[key])
            for key in scenario["affinity_weights"]
        )
    )


def _initial_state(
    agent: Mapping[str, Any],
    traits: Mapping[str, float],
) -> dict[str, Any]:
    semantic = agent["semantic"]
    risk = RISK_SCORE.get(str(semantic.get("risk_level")), 0.35)
    support, opposition = STANCE_ANCHOR.get(
        str(semantic.get("stance")), STANCE_ANCHOR["neutral"]
    )
    base_attention = _clip(0.13 + 0.18 * risk + 0.08 * traits["fact_need"])
    base_concern = _clip(
        0.10 + 0.25 * risk + 0.11 * opposition
    )
    base_trust = _clip(
        0.50
        + 0.035
        * (
            _hash_uniform(
                "baseline-trust", agent["micro_agent_id"]
            )
            - 0.5
        ),
        0.42,
        0.58,
    )
    base_satisfaction = _clip(
        0.58 - 0.10 * risk - 0.06 * traits["service_need"],
        0.38,
        0.62,
    )
    base_activity = _clip(
        ACTIVITY_ANCHOR.get(str(semantic.get("speech_act")), 0.35)
        * (0.68 + 0.28 * base_attention),
        0.16,
        0.62,
    )
    if traits["_active_gate"] == 0.0:
        base_activity = 0.035
    return {
        "attention": base_attention,
        "concern": base_concern,
        "trust": base_trust,
        "satisfaction": base_satisfaction,
        "support": support,
        "opposition": opposition,
        "activity_probability": base_activity,
        "rumor_belief": 0.015,
        "verified_belief": _clip(
            0.08 + 0.10 * traits["verification_need"]
        ),
        "ever_rumor_exposed": False,
        "ever_verified_exposed": False,
        "voice_satisfaction": 0.46,
        "service_overload": 0.04,
        "_base_attention": base_attention,
        "_base_concern": base_concern,
        "_base_trust": base_trust,
        "_base_satisfaction": base_satisfaction,
        "_base_activity": base_activity,
        "_base_support": support,
        "_base_opposition": opposition,
        "_active_gate": traits["_active_gate"],
        "_risk_signal_gate": traits["_risk_signal_gate"],
    }


def _graph_adjacency(
    graph: Mapping[str, Any],
    population_size: int,
) -> list[tuple[tuple[int, str], ...]]:
    adjacency: list[list[tuple[int, str]]] = [
        [] for _ in range(population_size)
    ]
    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise GovernanceArenaError("synthetic exposure graph has no edges")
    for edge in edges:
        source = int(edge["source_index"])
        target = int(edge["target_index"])
        edge_type = str(edge["edge_type"])
        if (
            not 0 <= source < population_size
            or not 0 <= target < population_size
            or source == target
            or edge_type
            not in {"semantic_neighbor", "cross_archetype_bridge"}
        ):
            raise GovernanceArenaError("synthetic exposure edge is invalid")
        adjacency[source].append((target, edge_type))
    if any(len(neighbors) != 12 for neighbors in adjacency):
        raise GovernanceArenaError(
            "governance arena requires the frozen 8+4 exposure graph"
        )
    return [tuple(neighbors) for neighbors in adjacency]


AUTHORITY_STRATEGIES = {
    "wait_and_monitor": {
        "cost": 0,
        "verified_injection": 0.00,
        "rumor_correction": 0.00,
        "trust_boost": 0.00,
        "voice_boost": 0.00,
        "service_relief": 0.00,
        "verified_diffusion_boost": 0.00,
        "polarization_relief": 0.00,
    },
    "minimal_notice": {
        "cost": 5,
        "verified_injection": 0.06,
        "rumor_correction": 0.04,
        "trust_boost": 0.012,
        "voice_boost": 0.00,
        "service_relief": 0.00,
        "verified_diffusion_boost": 0.05,
        "polarization_relief": 0.00,
    },
    "full_evidence_disclosure": {
        "cost": 10,
        "verified_injection": 0.18,
        "rumor_correction": 0.15,
        "trust_boost": 0.035,
        "voice_boost": 0.01,
        "service_relief": 0.00,
        "verified_diffusion_boost": 0.18,
        "polarization_relief": 0.015,
    },
    "open_deliberation": {
        "cost": 10,
        "verified_injection": 0.08,
        "rumor_correction": 0.07,
        "trust_boost": 0.025,
        "voice_boost": 0.16,
        "service_relief": 0.015,
        "verified_diffusion_boost": 0.08,
        "polarization_relief": 0.06,
    },
}

SERVICE_STRATEGIES = {
    "hold_capacity": {
        "cost": 0,
        "verified_injection": 0.00,
        "rumor_correction": 0.00,
        "trust_boost": 0.00,
        "voice_boost": 0.00,
        "service_relief": 0.00,
        "verified_diffusion_boost": 0.00,
        "polarization_relief": 0.00,
    },
    "targeted_triage": {
        "cost": 6,
        "verified_injection": 0.02,
        "rumor_correction": 0.00,
        "trust_boost": 0.012,
        "voice_boost": 0.025,
        "service_relief": 0.13,
        "verified_diffusion_boost": 0.00,
        "polarization_relief": 0.01,
    },
    "surge_support_capacity": {
        "cost": 10,
        "verified_injection": 0.02,
        "rumor_correction": 0.00,
        "trust_boost": 0.018,
        "voice_boost": 0.06,
        "service_relief": 0.25,
        "verified_diffusion_boost": 0.00,
        "polarization_relief": 0.025,
    },
}

BRIDGE_STRATEGIES = {
    "passive_relay": {
        "cost": 0,
        "verified_injection": 0.00,
        "rumor_correction": 0.00,
        "trust_boost": 0.00,
        "voice_boost": 0.00,
        "service_relief": 0.00,
        "verified_diffusion_boost": 0.00,
        "polarization_relief": 0.00,
    },
    "fact_check_relay": {
        "cost": 6,
        "verified_injection": 0.035,
        "rumor_correction": 0.10,
        "trust_boost": 0.012,
        "voice_boost": 0.01,
        "service_relief": 0.00,
        "verified_diffusion_boost": 0.25,
        "polarization_relief": 0.025,
    },
    "facilitated_group_dialogue": {
        "cost": 8,
        "verified_injection": 0.02,
        "rumor_correction": 0.06,
        "trust_boost": 0.018,
        "voice_boost": 0.13,
        "service_relief": 0.03,
        "verified_diffusion_boost": 0.10,
        "polarization_relief": 0.09,
    },
}


def _game_metrics(
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    traits_by_agent: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    aggregate = _aggregate_population(
        population,
        states,
        traits_by_agent=traits_by_agent,
    )
    return {
        "concern": aggregate["concern"],
        "trust": aggregate["trust"],
        "rumor": aggregate["rumor_belief"],
        "voice_gap": aggregate["voice_gap"],
        "service_overload": aggregate["service_overload"],
        "polarization": aggregate["polarization"],
    }


def _strategy_utility(
    *,
    actor: str,
    strategy_id: str,
    effects: Mapping[str, float],
    metrics: Mapping[str, float],
    scheme_id: str,
) -> float:
    public_value = (
        effects["rumor_correction"] * (1.35 + 1.8 * metrics["rumor"])
        + effects["verified_injection"] * (0.75 + metrics["concern"])
        + effects["trust_boost"] * (1.2 + 1.5 * (1.0 - metrics["trust"]))
        + effects["voice_boost"] * (0.9 + 1.8 * metrics["voice_gap"])
        + effects["service_relief"]
        * (0.8 + 1.7 * metrics["service_overload"])
        + effects["polarization_relief"]
        * (0.7 + 1.4 * metrics["polarization"])
    )
    scheme_bonus = 0.0
    if scheme_id == "rapid_transparency":
        if strategy_id in {
            "full_evidence_disclosure",
            "targeted_triage",
            "fact_check_relay",
        }:
            scheme_bonus = 0.30
        elif strategy_id in {
            "open_deliberation",
            "surge_support_capacity",
            "facilitated_group_dialogue",
        }:
            scheme_bonus = 0.03
    elif scheme_id == "participatory_service":
        if strategy_id in {
            "open_deliberation",
            "surge_support_capacity",
            "facilitated_group_dialogue",
        }:
            scheme_bonus = 0.32
        elif strategy_id in {
            "full_evidence_disclosure",
            "targeted_triage",
            "fact_check_relay",
        }:
            scheme_bonus = 0.04
    actor_bonus = {
        "governance_authority": (
            0.25 * effects["trust_boost"]
            + 0.18 * effects["verified_injection"]
        ),
        "service_operator": 0.32 * effects["service_relief"],
        "community_bridge": (
            0.25 * effects["voice_boost"]
            + 0.22 * effects["polarization_relief"]
        ),
    }[actor]
    return round(
        public_value
        + scheme_bonus
        + actor_bonus
        - 0.013 * float(effects["cost"]),
        6,
    )


def _select_actor_strategy(
    *,
    actor: str,
    strategies: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, float],
    scheme_id: str,
    remaining_resource: int,
) -> dict[str, Any]:
    candidates = []
    for strategy_id, effects in strategies.items():
        if int(effects["cost"]) > remaining_resource:
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "utility": _strategy_utility(
                    actor=actor,
                    strategy_id=strategy_id,
                    effects=effects,
                    metrics=metrics,
                    scheme_id=scheme_id,
                ),
                "resource_cost": int(effects["cost"]),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["utility"],
            -item["resource_cost"],
            item["strategy_id"],
        ),
    )
    return {
        "actor": actor,
        "selected_strategy": selected["strategy_id"],
        "selected_utility": selected["utility"],
        "resource_cost": selected["resource_cost"],
        "candidate_utilities": sorted(
            candidates, key=lambda item: item["strategy_id"]
        ),
    }


def _merge_effects(
    target: dict[str, float],
    effects: Mapping[str, float],
) -> None:
    for key in target:
        if key == "verified_diffusion_boost":
            target[key] += float(effects[key])
        else:
            target[key] += float(effects[key])


def _play_governance_round(
    *,
    scheme_id: str,
    tick: int,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    traits_by_agent: Sequence[Mapping[str, float]],
    remaining_resource: int,
) -> tuple[dict[str, Any], dict[str, float], int]:
    metrics = _game_metrics(population, states, traits_by_agent)
    if scheme_id == "natural":
        return (
            {
                "schema_version": GAME_SCHEMA_VERSION,
                "tick": tick,
                "scheme_id": scheme_id,
                "decisions": [],
                "resource_before": remaining_resource,
                "resource_consumed": 0,
                "resource_after": remaining_resource,
                "status": "no_new_governance_action",
                "pre_decision_metrics": metrics,
                "equilibrium_claimed": False,
            },
            {
                "verified_injection": 0.0,
                "rumor_correction": 0.0,
                "trust_boost": 0.0,
                "voice_boost": 0.0,
                "service_relief": 0.0,
                "verified_diffusion_boost": 0.0,
                "polarization_relief": 0.0,
            },
            remaining_resource,
        )
    decisions = []
    combined = {
        "verified_injection": 0.0,
        "rumor_correction": 0.0,
        "trust_boost": 0.0,
        "voice_boost": 0.0,
        "service_relief": 0.0,
        "verified_diffusion_boost": 0.0,
        "polarization_relief": 0.0,
    }
    before = remaining_resource
    for actor, strategies in (
        ("governance_authority", AUTHORITY_STRATEGIES),
        ("service_operator", SERVICE_STRATEGIES),
        ("community_bridge", BRIDGE_STRATEGIES),
    ):
        decision = _select_actor_strategy(
            actor=actor,
            strategies=strategies,
            metrics=metrics,
            scheme_id=scheme_id,
            remaining_resource=remaining_resource,
        )
        decisions.append(decision)
        effects = strategies[decision["selected_strategy"]]
        remaining_resource -= decision["resource_cost"]
        _merge_effects(combined, effects)
    return (
        {
            "schema_version": GAME_SCHEMA_VERSION,
            "tick": tick,
            "scheme_id": scheme_id,
            "decisions": decisions,
            "resource_before": before,
            "resource_consumed": before - remaining_resource,
            "resource_after": remaining_resource,
            "status": "bounded_sequential_best_response",
            "pre_decision_metrics": metrics,
            "equilibrium_claimed": False,
        },
        combined,
        remaining_resource,
    )


def _decay_effects(effects: Mapping[str, float]) -> dict[str, float]:
    return {
        key: round(float(value) * (0.66 if key != "service_relief" else 0.78), 8)
        for key, value in effects.items()
    }


def _stance_probabilities(state: Mapping[str, Any]) -> dict[str, float]:
    if float(state.get("_active_gate", 1.0)) == 0.0:
        return {
            "supportive": 0.02,
            "neutral": 0.92,
            "concerned": 0.04,
            "critical": 0.02,
        }
    return _normalize(
        {
            "supportive": (
                0.06
                + 0.62 * float(state["support"])
                + 0.18 * float(state["trust"])
            ),
            "neutral": (
                0.12
                + 0.42
                * (
                    1.0
                    - max(
                        float(state["support"]),
                        float(state["opposition"]),
                    )
                )
            ),
            "concerned": 0.06 + 0.64 * float(state["concern"]),
            "critical": (
                0.05
                + 0.66 * float(state["opposition"])
                + 0.14 * (1.0 - float(state["trust"]))
            ),
        }
    )


def _action_probabilities(
    state: Mapping[str, Any],
    traits: Mapping[str, float],
) -> dict[str, float]:
    if traits.get("_active_gate", 1.0) == 0.0:
        return {
            "ignore": 0.76,
            "read": 0.24,
            "comment": 0.0,
            "post": 0.0,
            "seek_help": 0.0,
            "report_risk": 0.0,
        }
    if traits.get("_risk_signal_gate", 0.0) == 1.0:
        risk_mass = _clip(
            0.08
            + 0.52
            * traits["risk_sensitivity"]
            * float(state["concern"])
        )
        return {
            "ignore": round((1.0 - risk_mass) * 0.30, 8),
            "read": round((1.0 - risk_mass) * 0.70, 8),
            "comment": 0.0,
            "post": 0.0,
            "seek_help": 0.0,
            "report_risk": round(risk_mass, 8),
        }
    public_mass = _clip(
        float(state["activity_probability"])
        * (
            0.42
            + 0.38 * float(state["attention"])
            + 0.20 * float(state["concern"])
        )
    )
    risk = traits["risk_sensitivity"]
    return _normalize(
        {
            "ignore": (1.0 - public_mass) * 0.31,
            "read": (1.0 - public_mass) * 0.69,
            "comment": public_mass * (0.25 + 0.12 * traits["voice_need"]),
            "post": public_mass * (0.20 + 0.12 * traits["peer_orientation"]),
            "seek_help": public_mass
            * (0.12 + 0.26 * traits["service_need"]),
            # Corrected from v1: reporting is monotonic in risk sensitivity.
            "report_risk": public_mass * (0.04 + 0.25 * risk),
        }
    )


def _aggregate_population(
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    *,
    traits_by_agent: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    agents = population["agents"]
    if traits_by_agent is None:
        traits_by_agent = [
            {
                "fact_need": 0.5,
                "service_need": 0.5,
                "voice_need": 0.5,
                "risk_sensitivity": 0.5,
                "peer_orientation": 0.5,
                "verification_need": 0.5,
            }
            for _ in agents
        ]
    means = {
        field: _weighted_mean(
            agents, (float(state[field]) for state in states)
        )
        for field in (
            "attention",
            "concern",
            "trust",
            "satisfaction",
            "activity_probability",
            "rumor_belief",
            "verified_belief",
            "service_overload",
        )
    }
    means["rumor_reach"] = _weighted_mean(
        agents,
        (float(bool(state["ever_rumor_exposed"])) for state in states),
    )
    means["verified_reach"] = _weighted_mean(
        agents,
        (float(bool(state["ever_verified_exposed"])) for state in states),
    )
    voice_gap_values = [
        max(0.0, traits["voice_need"] - float(state["voice_satisfaction"]))
        for traits, state in zip(traits_by_agent, states)
    ]
    means["voice_gap"] = _weighted_mean(agents, voice_gap_values)
    leans = [
        float(state["support"]) - float(state["opposition"])
        for state in states
    ]
    mean_lean = _weighted_mean(agents, leans)
    means["polarization"] = _clip(
        _weighted_mean(
            agents, ((lean - mean_lean) ** 2 for lean in leans)
        )
        * 2.4
    )
    means["severe_risk_share"] = _weighted_mean(
        agents,
        (
            float(
                float(state["concern"]) >= 0.68
                and float(state["activity_probability"]) >= 0.48
            )
            for state in states
        ),
    )
    stance = defaultdict(float)
    action = defaultdict(float)
    for agent, state, traits in zip(agents, states, traits_by_agent):
        weight = float(agent["population_weight"])
        for key, value in _stance_probabilities(state).items():
            stance[key] += weight * value
        for key, value in _action_probabilities(state, traits).items():
            action[key] += weight * value
    group_rows = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, agent in enumerate(agents):
        grouped[str(agent["primary_archetype_id"])].append(index)
    for archetype_id, members in grouped.items():
        mass = sum(float(agents[index]["population_weight"]) for index in members)
        group_rows.append(
            {
                "archetype_id": archetype_id,
                "population_weight": mass,
                "concern": sum(
                    float(agents[index]["population_weight"])
                    * float(states[index]["concern"])
                    for index in members
                )
                / mass,
                "trust": sum(
                    float(agents[index]["population_weight"])
                    * float(states[index]["trust"])
                    for index in members
                )
                / mass,
                "voice_gap": sum(
                    float(agents[index]["population_weight"])
                    * voice_gap_values[index]
                    for index in members
                )
                / mass,
            }
        )
    return {
        **{key: round(value, 6) for key, value in means.items()},
        "stance_shares": {
            key: round(value, 6) for key, value in sorted(stance.items())
        },
        "action_shares": {
            key: round(value, 6) for key, value in sorted(action.items())
        },
        "group_gap": {
            "concern_range": round(
                max(row["concern"] for row in group_rows)
                - min(row["concern"] for row in group_rows),
                6,
            ),
            "trust_range": round(
                max(row["trust"] for row in group_rows)
                - min(row["trust"] for row in group_rows),
                6,
            ),
            "voice_gap_range": round(
                max(row["voice_gap"] for row in group_rows)
                - min(row["voice_gap"] for row in group_rows),
                6,
            ),
            "highest_concern_groups": [
                {
                    key: (
                        round(value, 6)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in row.items()
                }
                for row in sorted(
                    group_rows,
                    key=lambda row: (-row["concern"], row["archetype_id"]),
                )[:3]
            ],
        },
        # Internal seed-level rows are consumed by _aggregate_branch_runs and
        # removed from the public result.  Keeping all 17 groups here avoids
        # accidentally treating one seed's top-three ordering as the mean.
        "_archetype_rows": [
            {
                key: (
                    round(value, 6)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
            for row in sorted(
                group_rows, key=lambda row: row["archetype_id"]
            )
        ],
    }


def _advance_baseline(
    *,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    adjacency: Sequence[Sequence[tuple[int, str]]],
    seed: int,
    tick: int,
) -> list[dict[str, Any]]:
    updated = []
    for index, state in enumerate(states):
        neighbor_concern = sum(
            float(states[target]["concern"])
            for target, _ in adjacency[index]
        ) / len(adjacency[index])
        item = copy.deepcopy(dict(state))
        for field in (
            "attention",
            "concern",
            "trust",
            "satisfaction",
            "activity_probability",
        ):
            base = float(state[f"_base_{field.split('_probability')[0]}"])
            neighbor_term = (
                0.018 * (neighbor_concern - float(state["concern"]))
                if field in {"attention", "concern"}
                else 0.0
            )
            field_counter = {
                "attention": 1,
                "concern": 2,
                "trust": 3,
                "satisfaction": 4,
                "activity_probability": 5,
            }[field]
            noise = 0.004 * (
                _counter_uniform(seed, 0, tick, index, field_counter) - 0.5
            )
            item[field] = round(
                _clip(
                    float(state[field])
                    + 0.18 * (base - float(state[field]))
                    + neighbor_term
                    + noise,
                    0.02,
                    0.98,
                ),
                6,
            )
        updated.append(item)
    return updated


def _advance_shock_tick(
    *,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    traits_by_agent: Sequence[Mapping[str, float]],
    affinities: Sequence[float],
    adjacency: Sequence[Sequence[tuple[int, str]]],
    scenario_id: str,
    scheme_id: str,
    seed: int,
    tick: int,
    active_effects: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    scenario = SCENARIOS[scenario_id]
    scenario_counter = (
        1 if scenario_id == "lecture_external_incident_shock" else 2
    )
    intensity = _phase_intensity(tick)
    transmissions = {
        "rumor_semantic": 0,
        "rumor_bridge": 0,
        "verified_semantic": 0,
        "verified_bridge": 0,
        "direct_rumor_injections": 0,
        "direct_verified_injections": 0,
    }
    updated = []
    for index, (agent, state, traits, affinity) in enumerate(
        zip(
            population["agents"],
            states,
            traits_by_agent,
            affinities,
        )
    ):
        rumor_hit = False
        verified_hit = False
        rumor_neighbor = 0.0
        verified_neighbor = 0.0
        attention_neighbor = 0.0
        for target, edge_type in adjacency[index]:
            neighbor = states[target]
            bridge = edge_type == "cross_archetype_bridge"
            rumor_neighbor += float(neighbor["rumor_belief"])
            verified_neighbor += float(neighbor["verified_belief"])
            attention_neighbor += float(neighbor["attention"])
            rumor_probability = (
                (0.115 if bridge else 0.068)
                * float(neighbor["activity_probability"])
                * float(neighbor["rumor_belief"])
                * float(scenario["rumor_virality"])
                * (1.0 - 0.42 * float(state["verified_belief"]))
            )
            verified_probability = (
                (0.15 if bridge else 0.075)
                * float(neighbor["activity_probability"])
                * float(neighbor["verified_belief"])
                * (
                    1.0
                    + float(active_effects["verified_diffusion_boost"])
                )
            )
            if _counter_uniform(
                seed,
                scenario_counter,
                tick,
                index,
                target,
                1,
            ) < rumor_probability:
                rumor_hit = True
                transmissions[
                    "rumor_bridge" if bridge else "rumor_semantic"
                ] += 1
            if _counter_uniform(
                seed,
                scenario_counter,
                tick,
                index,
                target,
                2,
            ) < verified_probability:
                verified_hit = True
                transmissions[
                    "verified_bridge" if bridge else "verified_semantic"
                ] += 1
        rumor_neighbor /= len(adjacency[index])
        verified_neighbor /= len(adjacency[index])
        attention_neighbor /= len(adjacency[index])

        direct_rumor = (
            tick == 2
            and _counter_uniform(
                seed, scenario_counter, tick, index, 3
            )
            < float(scenario["rumor_seed_fraction"]) * (0.55 + affinity)
        )
        direct_verified = (
            tick == 2
            and _counter_uniform(
                seed, scenario_counter, tick, index, 4
            )
            < float(scenario["verified_seed_fraction"])
            * (0.55 + traits["verification_need"])
        )
        intervention_verified = (
            _counter_uniform(
                seed, scenario_counter, tick, index, 5
            )
            < float(active_effects["verified_injection"])
            * (0.60 + 0.40 * traits["fact_need"])
        )
        if direct_rumor:
            transmissions["direct_rumor_injections"] += 1
        if direct_verified or intervention_verified:
            transmissions["direct_verified_injections"] += 1
        rumor_exposed = rumor_hit or direct_rumor
        verified_exposed = (
            verified_hit or direct_verified or intervention_verified
        )
        rumor = _clip(
            float(state["rumor_belief"])
            + (0.17 if rumor_exposed else 0.0)
            + 0.10 * rumor_neighbor
            - (
                0.19
                + 0.22 * traits["verification_need"]
                + float(active_effects["rumor_correction"])
            )
            * float(state["rumor_belief"])
            * (
                float(state["verified_belief"])
                + (0.45 if verified_exposed else 0.0)
            ),
            0.006,
            0.94,
        )
        verified = _clip(
            float(state["verified_belief"])
            + (0.16 if verified_exposed else 0.0)
            + 0.085 * verified_neighbor
            + 0.045
            * float(active_effects["verified_diffusion_boost"])
            * traits["fact_need"]
            - 0.055 * float(state["verified_belief"]),
            0.02,
            0.96,
        )
        reactance = (
            0.055 * traits["voice_need"]
            if (
                scheme_id == "rapid_transparency"
                and scenario["shock_kind"]
                == "governance_legitimacy_dispute"
                and float(active_effects["voice_boost"]) < 0.05
            )
            else 0.0
        )
        voice_target = _clip(
            0.46
            - 0.18 * intensity * affinity * traits["voice_need"]
            + float(active_effects["voice_boost"])
            - reactance,
            0.08,
            0.88,
        )
        service_target = _clip(
            intensity
            * float(scenario["service_pressure"])
            * traits["service_need"]
            * (0.58 + 0.42 * float(state["activity_probability"]))
            - float(active_effects["service_relief"])
            * traits["service_need"],
            0.02,
            0.92,
        )
        attention_target = _clip(
            float(state["_base_attention"])
            + float(scenario["attention_force"]) * intensity * affinity
            + 0.19 * rumor
            + 0.08 * verified
            + 0.07 * attention_neighbor,
            0.03,
            0.92,
        )
        concern_target = _clip(
            float(state["_base_concern"])
            + float(scenario["concern_force"]) * intensity * affinity
            + 0.29 * rumor
            - 0.17 * verified
            + 0.18 * service_target
            - float(active_effects["service_relief"])
            * traits["service_need"],
            0.035,
            0.92,
        )
        trust_target = _clip(
            float(state["_base_trust"])
            - float(scenario["trust_force"]) * intensity * affinity
            - 0.19 * rumor
            + 0.22 * verified * traits["fact_need"]
            + 0.15 * voice_target * traits["voice_need"]
            + float(active_effects["trust_boost"])
            - reactance,
            0.12,
            0.84,
        )
        satisfaction_target = _clip(
            float(state["_base_satisfaction"])
            - 0.30 * concern_target
            - 0.22 * service_target
            + 0.12 * voice_target
            + 0.16
            * float(active_effects["service_relief"])
            * traits["service_need"],
            0.10,
            0.78,
        )
        opposition_target = _clip(
            float(state["_base_opposition"])
            + 0.18 * (1.0 - trust_target) * traits["voice_need"]
            + 0.13 * rumor
            - 0.10 * voice_target
            + reactance
            - float(active_effects["polarization_relief"])
            * traits["voice_need"],
            0.04,
            0.90,
        )
        support_target = _clip(
            float(state["_base_support"])
            + 0.11 * trust_target
            + 0.07 * verified
            - 0.10 * concern_target
            + 0.08 * voice_target
            - 0.06 * rumor,
            0.05,
            0.88,
        )
        activity_target = _clip(
            float(state["_base_activity"])
            + 0.25 * attention_target
            + 0.16 * concern_target
            + 0.11 * max(0.0, traits["voice_need"] - voice_target)
            - 0.12 * verified * (1.0 - concern_target),
            0.10,
            0.88,
        )
        noise = (
            _counter_uniform(
                seed, scenario_counter, tick, index, 6
            )
            - 0.5
        )
        item = copy.deepcopy(dict(state))
        adjustments = {
            "attention": (attention_target, 0.38),
            "concern": (concern_target, 0.36),
            "trust": (trust_target, 0.28),
            "satisfaction": (satisfaction_target, 0.27),
            "support": (support_target, 0.25),
            "opposition": (opposition_target, 0.25),
            "activity_probability": (activity_target, 0.34),
            "voice_satisfaction": (voice_target, 0.31),
            "service_overload": (service_target, 0.40),
        }
        for field, (target, rate) in adjustments.items():
            item[field] = round(
                _clip(
                    float(state[field])
                    + rate * (target - float(state[field]))
                    + 0.006 * noise,
                    0.015,
                    0.95,
                ),
                6,
            )
        item["rumor_belief"] = round(rumor, 6)
        item["verified_belief"] = round(verified, 6)
        item["ever_rumor_exposed"] = bool(
            state["ever_rumor_exposed"] or rumor_exposed
        )
        item["ever_verified_exposed"] = bool(
            state["ever_verified_exposed"] or verified_exposed
        )
        updated.append(item)
    return updated, transmissions


def _run_seed_scenario(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    prior_by_archetype: Mapping[str, Mapping[str, Any]],
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_id]
    agents = population["agents"]
    traits_by_agent = [
        _agent_traits(agent, prior_by_archetype) for agent in agents
    ]
    affinities = [
        _scenario_affinity(traits, scenario) for traits in traits_by_agent
    ]
    adjacency = _graph_adjacency(graph, len(agents))
    baseline_states = [
        _initial_state(agent, traits)
        for agent, traits in zip(agents, traits_by_agent)
    ]
    baseline_timeline = []
    for tick in (0, 1):
        baseline_states = _advance_baseline(
            population=population,
            states=baseline_states,
            adjacency=adjacency,
            seed=seed,
            tick=tick,
        )
        baseline_timeline.append(
            {
                "tick": tick,
                "phase": TIMELINE_TICKS[tick],
                **_aggregate_population(
                    population,
                    baseline_states,
                    traits_by_agent=traits_by_agent,
                ),
                "diffusion": {
                    "rumor_transmissions": 0,
                    "verified_transmissions": 0,
                    "bridge_share": 0.0,
                },
                "governance_round": None,
            }
        )
    baseline_sha = canonical_hash(baseline_states)
    branches = {}
    for scheme_id in SCHEME_IDS:
        states = copy.deepcopy(baseline_states)
        timeline = copy.deepcopy(baseline_timeline)
        remaining_resource = 100
        active_effects = {
            "verified_injection": 0.0,
            "rumor_correction": 0.0,
            "trust_boost": 0.0,
            "voice_boost": 0.0,
            "service_relief": 0.0,
            "verified_diffusion_boost": 0.0,
            "polarization_relief": 0.0,
        }
        game_rounds = []
        for tick in range(2, 16):
            governance_round = None
            if tick in DECISION_TICKS:
                governance_round, new_effects, remaining_resource = (
                    _play_governance_round(
                        scheme_id=scheme_id,
                        tick=tick,
                        population=population,
                        states=states,
                        traits_by_agent=traits_by_agent,
                        remaining_resource=remaining_resource,
                    )
                )
                for key, value in new_effects.items():
                    active_effects[key] += value
                game_rounds.append(governance_round)
            states, transmissions = _advance_shock_tick(
                population=population,
                states=states,
                traits_by_agent=traits_by_agent,
                affinities=affinities,
                adjacency=adjacency,
                scenario_id=scenario_id,
                scheme_id=scheme_id,
                seed=seed,
                tick=tick,
                active_effects=active_effects,
            )
            rumor_total = (
                transmissions["rumor_semantic"]
                + transmissions["rumor_bridge"]
                + transmissions["direct_rumor_injections"]
            )
            verified_total = (
                transmissions["verified_semantic"]
                + transmissions["verified_bridge"]
                + transmissions["direct_verified_injections"]
            )
            bridge_total = (
                transmissions["rumor_bridge"]
                + transmissions["verified_bridge"]
            )
            all_transmissions = max(rumor_total + verified_total, 1)
            timeline.append(
                {
                    "tick": tick,
                    "phase": TIMELINE_TICKS[tick],
                    **_aggregate_population(
                        population,
                        states,
                        traits_by_agent=traits_by_agent,
                    ),
                    "diffusion": {
                        "rumor_transmissions": rumor_total,
                        "verified_transmissions": verified_total,
                        "bridge_share": round(
                            bridge_total / all_transmissions, 6
                        ),
                    },
                    "governance_round": governance_round,
                }
            )
            active_effects = _decay_effects(active_effects)
        branches[scheme_id] = {
            "scheme_id": scheme_id,
            "timeline": timeline,
            "game_rounds": game_rounds,
            "resource": {
                "opening": 100,
                "consumed": 100 - remaining_resource,
                "remaining": remaining_resource,
                "conserved": (
                    100 == 100 - remaining_resource + remaining_resource
                ),
            },
            "shared_baseline_state_sha256": baseline_sha,
        }
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "branches": branches,
        "shared_baseline_state_sha256": baseline_sha,
        "provider_calls": 0,
    }


def _mean_and_interval(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    if len(values) > 1:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        radius = 2.365 * standard_error
    else:
        radius = 0.0
    return {
        "mean": round(mean, 6),
        "lower": round(_clip(mean - radius), 6),
        "upper": round(_clip(mean + radius), 6),
    }


def _aggregate_branch_runs(
    runs: Sequence[Mapping[str, Any]],
    scheme_id: str,
) -> dict[str, Any]:
    timeline = []
    for tick in range(16):
        rows = [run["branches"][scheme_id]["timeline"][tick] for run in runs]
        metrics = {
            metric: _mean_and_interval(
                [float(row[metric]) for row in rows]
            )
            for metric in BOUNDED_METRICS
        }
        stance = {
            category: round(
                statistics.fmean(
                    row["stance_shares"][category] for row in rows
                ),
                6,
            )
            for category in rows[0]["stance_shares"]
        }
        action = {
            category: round(
                statistics.fmean(
                    row["action_shares"][category] for row in rows
                ),
                6,
            )
            for category in rows[0]["action_shares"]
        }
        group_gap = {
            key: round(
                statistics.fmean(
                    row["group_gap"][key] for row in rows
                ),
                6,
            )
            for key in (
                "concern_range",
                "trust_range",
                "voice_gap_range",
            )
        }
        seed_group_rows = [
            {
                str(group["archetype_id"]): group
                for group in row["_archetype_rows"]
            }
            for row in rows
        ]
        archetype_ids = sorted(seed_group_rows[0])
        if any(set(seed_rows) != set(archetype_ids) for seed_rows in seed_group_rows):
            raise GovernanceArenaError(
                "seed-level archetype groups are inconsistent"
            )
        averaged_groups = []
        for archetype_id in archetype_ids:
            averaged_groups.append(
                {
                    "archetype_id": archetype_id,
                    "population_weight": round(
                        statistics.fmean(
                            seed_rows[archetype_id]["population_weight"]
                            for seed_rows in seed_group_rows
                        ),
                        6,
                    ),
                    "concern": round(
                        statistics.fmean(
                            seed_rows[archetype_id]["concern"]
                            for seed_rows in seed_group_rows
                        ),
                        6,
                    ),
                    "trust": round(
                        statistics.fmean(
                            seed_rows[archetype_id]["trust"]
                            for seed_rows in seed_group_rows
                        ),
                        6,
                    ),
                    "voice_gap": round(
                        statistics.fmean(
                            seed_rows[archetype_id]["voice_gap"]
                            for seed_rows in seed_group_rows
                        ),
                        6,
                    ),
                }
            )
        top_groups = sorted(
            averaged_groups,
            key=lambda row: (-row["concern"], row["archetype_id"]),
        )[:3]
        diffusion = {
            key: round(
                statistics.fmean(row["diffusion"][key] for row in rows),
                6,
            )
            for key in (
                "rumor_transmissions",
                "verified_transmissions",
                "bridge_share",
            )
        }
        timeline.append(
            {
                "tick": tick,
                "phase": TIMELINE_TICKS[tick],
                "metrics": metrics,
                "stance_shares": stance,
                "action_shares": action,
                "group_gap": {
                    **group_gap,
                    "highest_concern_groups": top_groups,
                },
                "diffusion": diffusion,
            }
        )
    resources = [
        run["branches"][scheme_id]["resource"] for run in runs
    ]
    game_rounds = []
    for tick in DECISION_TICKS:
        seed_rounds = [
            next(
                item
                for item in run["branches"][scheme_id]["game_rounds"]
                if item["tick"] == tick
            )
            for run in runs
        ]
        first = seed_rounds[0]
        game_rounds.append(
            {
                **{
                    key: copy.deepcopy(value)
                    for key, value in first.items()
                    if key not in {"pre_decision_metrics"}
                },
                "pre_decision_metrics": {
                    key: round(
                        statistics.fmean(
                            row["pre_decision_metrics"][key]
                            for row in seed_rounds
                        ),
                        6,
                    )
                    for key in first["pre_decision_metrics"]
                },
                "strategy_stable_across_seeds": all(
                    [
                        item["selected_strategy"]
                        for item in row["decisions"]
                    ]
                    == [
                        item["selected_strategy"]
                        for item in first["decisions"]
                    ]
                    for row in seed_rounds
                ),
            }
        )
    return {
        "scheme_id": scheme_id,
        "timeline": timeline,
        "game_rounds": game_rounds,
        "resource": {
            "opening": resources[0]["opening"],
            "consumed": resources[0]["consumed"],
            "remaining": resources[0]["remaining"],
            "conserved": all(row["conserved"] for row in resources),
        },
    }


def _auc(timeline: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = [float(item["metrics"][metric]["mean"]) for item in timeline]
    return sum(
        (values[index - 1] + values[index]) / 2.0
        for index in range(1, len(values))
    )


def _policy_comparison(
    branches: Mapping[str, Mapping[str, Any]],
    scenario_id: str,
) -> dict[str, Any]:
    natural = branches["natural"]["timeline"]
    rows = []
    for scheme_id in SCHEME_IDS:
        timeline = branches[scheme_id]["timeline"]
        concern_auc = _auc(timeline, "concern")
        rumor_auc = _auc(timeline, "rumor_belief")
        row = {
            "scheme_id": scheme_id,
            "label": SCHEMES[scheme_id]["label"],
            "resource_consumed": branches[scheme_id]["resource"]["consumed"],
            "peak_concern": round(
                max(item["metrics"]["concern"]["mean"] for item in timeline),
                6,
            ),
            "concern_auc": round(concern_auc, 6),
            "rumor_auc": round(rumor_auc, 6),
            "end_trust": timeline[-1]["metrics"]["trust"]["mean"],
            "end_satisfaction": timeline[-1]["metrics"]["satisfaction"][
                "mean"
            ],
            "end_voice_gap": timeline[-1]["metrics"]["voice_gap"]["mean"],
            "end_polarization": timeline[-1]["metrics"]["polarization"][
                "mean"
            ],
            "worst_group_concern_at_peak": round(
                max(
                    item["group_gap"]["highest_concern_groups"][0]["concern"]
                    for item in timeline
                ),
                6,
            ),
            "provider_calls": 0,
        }
        if scheme_id == "natural":
            row["vs_natural"] = {
                "concern_auc_reduction": 0.0,
                "rumor_auc_reduction": 0.0,
                "end_trust_gain": 0.0,
            }
        else:
            row["vs_natural"] = {
                "concern_auc_reduction": round(
                    _auc(natural, "concern") - concern_auc, 6
                ),
                "rumor_auc_reduction": round(
                    _auc(natural, "rumor_belief") - rumor_auc, 6
                ),
                "end_trust_gain": round(
                    timeline[-1]["metrics"]["trust"]["mean"]
                    - natural[-1]["metrics"]["trust"]["mean"],
                    6,
                ),
            }
        rows.append(row)
    by_id = {row["scheme_id"]: row for row in rows}
    objective = POLICY_OBJECTIVES[scenario_id]
    natural_row = by_id["natural"]
    evaluated = []
    for scheme_id in SCHEME_IDS[1:]:
        row = by_id[scheme_id]
        components = {
            "rumor_auc_reduction_ratio": (
                natural_row["rumor_auc"] - row["rumor_auc"]
            )
            / max(natural_row["rumor_auc"], 1e-12),
            "concern_auc_reduction_ratio": (
                natural_row["concern_auc"] - row["concern_auc"]
            )
            / max(natural_row["concern_auc"], 1e-12),
            "end_trust_gain_ratio": (
                row["end_trust"] - natural_row["end_trust"]
            )
            / max(1.0 - natural_row["end_trust"], 1e-12),
            "end_satisfaction_gain_ratio": (
                row["end_satisfaction"]
                - natural_row["end_satisfaction"]
            )
            / max(1.0 - natural_row["end_satisfaction"], 1e-12),
            "end_voice_gap_reduction_ratio": (
                natural_row["end_voice_gap"] - row["end_voice_gap"]
            )
            / max(natural_row["end_voice_gap"], 1e-12),
            "end_polarization_reduction_ratio": (
                natural_row["end_polarization"]
                - row["end_polarization"]
            )
            / max(natural_row["end_polarization"], 1e-12),
            "resource_cost_ratio": row["resource_consumed"] / 100.0,
        }
        score = sum(
            float(objective["weights"][key]) * value
            for key, value in components.items()
        )
        evaluation = {
            "scheme_id": scheme_id,
            "objective_score": round(score, 6),
            "normalized_components": {
                key: round(value, 6)
                for key, value in components.items()
            },
        }
        row["objective_evaluation"] = copy.deepcopy(evaluation)
        evaluated.append(evaluation)
    recommendation = max(
        evaluated,
        key=lambda item: (
            item["objective_score"],
            item["scheme_id"],
        ),
    )["scheme_id"]
    alternative = (
        "participatory_service"
        if recommendation == "rapid_transparency"
        else "rapid_transparency"
    )
    return {
        "rows": rows,
        "objective": {
            "label": objective["label"],
            "weights": copy.deepcopy(objective["weights"]),
            "score_rule": (
                "normalized benefit components × frozen weights; "
                "resource_cost_ratio has a negative weight"
            ),
            "candidate_evaluations": evaluated,
        },
        "conditional_recommendation": {
            "recommended_scheme_id": recommendation,
            "reason": (
                f"按冻结目标函数“{objective['label']}”计算，"
                f"{SCHEMES[recommendation]['label']} 的模型条件得分更高。"
            ),
            "tradeoff": (
                f"{SCHEMES[alternative]['label']} 仍可在其他指标上更优；"
                "不存在脱离治理目标权重的统一最优方案。"
            ),
            "recommended_metrics": by_id[recommendation],
        },
    }


def _profile_fidelity_audit(
    population: Mapping[str, Any],
    prior_by_archetype: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    matched_weight = 0.0
    active_weight = 0.0
    background_weight = 0.0
    risk_signal_weight = 0.0
    risk_action = defaultdict(list)
    for agent in population["agents"]:
        traits = _agent_traits(agent, prior_by_archetype)
        state = _initial_state(agent, traits)
        state["_active_gate"] = traits["_active_gate"]
        if traits["_active_gate"] == 0.0:
            background_weight += float(agent["population_weight"])
            continue
        active_weight += float(agent["population_weight"])
        if traits["_risk_signal_gate"] == 1.0:
            risk_signal_weight += float(agent["population_weight"])
        stance = _stance_probabilities(state)
        modal = max(stance, key=stance.get)
        reviewed = str(agent["semantic"].get("stance"))
        acceptable = {
            "supportive": {"supportive"},
            "neutral": {"neutral"},
            "concerned": {"concerned", "critical"},
            "critical": {"critical", "concerned"},
            "ambivalent": {"neutral", "concerned"},
        }.get(reviewed, {"neutral"})
        if modal in acceptable:
            matched_weight += float(agent["population_weight"])
        actions = _action_probabilities(state, traits)
        risk_action[str(agent["semantic"].get("risk_level"))].append(
            actions["report_risk"]
        )
    mean_report = {
        key: statistics.fmean(values)
        for key, values in risk_action.items()
        if values
    }
    monotonic = (
        mean_report.get("high", 0.0)
        > mean_report.get("medium", 0.0)
        > mean_report.get("low", 0.0)
        > mean_report.get("none", 0.0)
    )
    return {
        "reviewed_stance_modal_alignment_weight": round(
            matched_weight / max(active_weight, 1e-12), 6
        ),
        "active_episode_population_weight": round(active_weight, 6),
        "background_signal_population_weight": round(
            background_weight, 6
        ),
        "governance_risk_signal_population_weight": round(
            risk_signal_weight, 6
        ),
        "report_risk_probability_by_reviewed_risk": {
            key: round(value, 6)
            for key, value in sorted(mean_report.items())
        },
        "risk_reporting_is_monotonic": monotonic,
        "invented_demographics": False,
        "stable_personality_claimed": False,
        "profile_variant_identity_inferred": False,
        "assessment": (
            "Baseline opinion anchors preserve reviewed semantic stance; "
            "counterfactual event opinions remain model-conditional."
        ),
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != ARENA_SCHEMA_VERSION
        or result.get("population_identity", {}).get("population_size")
        != 1000
        or result.get("population_identity", {}).get("particle_count")
        != 10_000
        or tuple(result.get("scheme_ids", ())) != SCHEME_IDS
        or result.get("usage", {}).get("provider_calls") != 0
        or result.get("usage", {}).get("real_governance_actions") != 0
    ):
        raise GovernanceArenaError("governance arena identity is invalid")
    if (
        result["profile_audit"]["reviewed_stance_modal_alignment_weight"]
        < 0.78
        or not result["profile_audit"]["risk_reporting_is_monotonic"]
    ):
        raise GovernanceArenaError("profile/opinion fidelity gate failed")
    for scenario in result["scenarios"].values():
        baseline_rows = scenario["branches"]["natural"]["timeline"][:2]
        for metric in ("attention", "concern", "trust", "satisfaction"):
            drift = abs(
                baseline_rows[1]["metrics"][metric]["mean"]
                - baseline_rows[0]["metrics"][metric]["mean"]
            )
            if drift > 0.02:
                raise GovernanceArenaError(
                    "shared baseline is not approximately stationary"
                )
        for branch in scenario["branches"].values():
            if not branch["resource"]["conserved"]:
                raise GovernanceArenaError("resource conservation failed")
            if len(branch["timeline"]) != 16:
                raise GovernanceArenaError("timeline is incomplete")
            for item in branch["timeline"]:
                for metric in (
                    "attention",
                    "concern",
                    "trust",
                    "satisfaction",
                ):
                    value = item["metrics"][metric]["mean"]
                    if not 0.01 < value < 0.99:
                        raise GovernanceArenaError(
                            "continuous state hit an implausible boundary"
                        )
        comparison = {
            row["scheme_id"]: row
            for row in scenario["policy_comparison"]["rows"]
        }
        if (
            comparison["rapid_transparency"]["vs_natural"][
                "rumor_auc_reduction"
            ]
            <= 0
            or comparison["participatory_service"]["vs_natural"][
                "concern_auc_reduction"
            ]
            <= 0
        ):
            raise GovernanceArenaError(
                "intervention mechanisms did not beat natural evolution"
            )
    forbidden = {
        "agents",
        "states",
        "edges",
        "selected",
        "micro_agent_id",
        "source_id",
        "raw_text",
        "redacted_text",
        "evidence_record_ids",
        "payload",
        "api_key",
        "password",
        "credential",
        "local_path",
        "file_path",
    }

    def scan(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                str(key).lower() in forbidden or scan(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(scan(child) for child in value)
        return False

    if scan(result):
        raise GovernanceArenaError(
            "public governance arena contains record-level material"
        )


def build_governance_arena_result(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    persona_release: Mapping[str, Any],
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run two shocks × three branches with paired common random numbers."""

    if (
        population.get("population_size") != 1000
        or graph.get("node_count") != 1000
        or graph.get("edge_count") != 12_000
        or graph.get("population_sha256")
        != population.get("population_sha256")
    ):
        raise GovernanceArenaError(
            "governance arena release identities are inconsistent"
        )
    profile_priors = compile_profile_response_priors(persona_release)
    prior_by_archetype = {
        item["archetype_id"]: item
        for item in profile_priors["profiles"]
    }
    profile_audit = _profile_fidelity_audit(
        population, prior_by_archetype
    )
    scenarios = {}
    for scenario_id in SCENARIOS:
        seed_runs = [
            _run_seed_scenario(
                population=population,
                graph=graph,
                prior_by_archetype=prior_by_archetype,
                scenario_id=scenario_id,
                seed=int(seed),
            )
            for seed in seeds
        ]
        branches = {
            scheme_id: _aggregate_branch_runs(seed_runs, scheme_id)
            for scheme_id in SCHEME_IDS
        }
        scenarios[scenario_id] = {
            "scenario": copy.deepcopy(SCENARIOS[scenario_id]),
            "branches": branches,
            "policy_comparison": _policy_comparison(
                branches, scenario_id
            ),
            "paired_seed_count": len(seeds),
            "shared_baseline_sha256_by_seed": [
                run["shared_baseline_state_sha256"] for run in seed_runs
            ],
        }
    body = {
        "schema_version": ARENA_SCHEMA_VERSION,
        "result_kind": (
            "zero_provider_model_conditional_governance_mechanism_simulation"
        ),
        "population_identity": {
            "population_sha256": population["population_sha256"],
            "graph_sha256": graph["graph_sha256"],
            "profile_prior_sha256": profile_priors[
                "profile_prior_sha256"
            ],
            "population_size": 1000,
            "particle_count": 10_000,
            "unit": "reviewed_forum_episode_micro_agent",
            "graph_status": "fully_synthetic_exposure_graph",
            "active_episode_population_weight": profile_audit[
                "active_episode_population_weight"
            ],
            "background_signal_population_weight": profile_audit[
                "background_signal_population_weight"
            ],
            "governance_risk_signal_population_weight": profile_audit[
                "governance_risk_signal_population_weight"
            ],
        },
        "timeline_contract": {
            "tick_count": 16,
            "phases": [
                {"phase": phase, "tick_count": count}
                for phase, count in TIMELINE_PHASES
            ],
            "shared_baseline_ticks": 2,
            "decision_ticks": list(DECISION_TICKS),
        },
        "scheme_ids": list(SCHEME_IDS),
        "schemes": copy.deepcopy(SCHEMES),
        "diffusion_contract": {
            "schema_version": DIFFUSION_SCHEMA_VERSION,
            "competing_information": [
                "uncertain_or_rumor_information",
                "verified_information",
            ],
            "edge_types": [
                "semantic_neighbor",
                "cross_archetype_bridge",
            ],
            "common_random_numbers_across_schemes": True,
            "real_social_graph_claimed": False,
        },
        "game_contract": {
            "schema_version": GAME_SCHEMA_VERSION,
            "actors": [
                "governance_authority",
                "service_operator",
                "community_bridge",
                "episode_population",
            ],
            "solution_concept": "bounded_sequential_best_response",
            "nash_equilibrium_claimed": False,
            "resource_opening_per_branch": 100,
        },
        "profile_audit": profile_audit,
        "scenarios": scenarios,
        "uncertainty": {
            "seed_count": len(seeds),
            "seeds": [int(seed) for seed in seeds],
            "interval": (
                "paired-seed mean 95% t interval; it is not population "
                "sampling uncertainty or a causal confidence interval"
            ),
        },
        "usage": {
            "provider_calls": 0,
            "real_governance_actions": 0,
            "low_cost_agent_tick_updates": (
                1000
                * (
                    2
                    + (16 - 2) * len(SCHEME_IDS)
                )
                * len(SCENARIOS)
                * len(seeds)
            ),
        },
        "representative_synthetic_statements": [
            {
                "mechanism": "fact_verification",
                "text": "先给出可核验时间线，再说明哪些结论仍待确认。",
                "review_status": "synthetic_reviewed_demo_text",
            },
            {
                "mechanism": "service_capacity",
                "text": "公开说明有帮助，但支持渠道是否接得住同样重要。",
                "review_status": "synthetic_reviewed_demo_text",
            },
            {
                "mechanism": "procedural_voice",
                "text": "如果方案会影响我们，希望能看到反馈如何进入最终决定。",
                "review_status": "synthetic_reviewed_demo_text",
            },
        ],
        "privacy": {
            "micro_agent_states_exported": False,
            "sample_identities_exported": False,
            "graph_edges_exported": False,
            "raw_text_exported": False,
            "source_identifiers_exported": False,
            "persona_evidence_record_ids_exported": False,
        },
        "non_claims": [
            "Episode micro-agents are not identified students.",
            "Profile response priors are not measured stable personality traits.",
            "The exposure graph is synthetic, not an observed social network.",
            "Diffusion rates are stress-test parameters, not measured transmission rates.",
            "Policy differences are model-conditional and do not identify causal effects.",
            "No real governance action or Provider call occurred.",
        ],
    }
    _validate_result(body)
    body["result_sha256"] = canonical_hash(body)
    profile_audit_document = {
        **profile_priors,
        "population_sha256": population["population_sha256"],
        "profile_fidelity": profile_audit,
    }
    profile_audit_document["audit_sha256"] = canonical_hash(
        profile_audit_document
    )
    return body, profile_audit_document


def validate_governance_arena_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    if result.get("result_sha256") != canonical_hash(body):
        raise GovernanceArenaError("governance arena hash does not match")
    _validate_result(body)
    return copy.deepcopy(dict(result))
