"""Unified budgeted-LLM primitives for CampusPulse YuLan-Scale v1.

This module composes the reviewed LLM-agent contract with the existing
Governance Arena particle and message kernel.  It deliberately does not
rename deterministic fixtures as LLM output:

* every resident is LLM-callable and owns a persistent semantic memory;
* sixteen longitudinal anchors are observed as individuals;
* thirty-two rotating PPS residents support population inference;
* exact trace replay accepts only prompt-fingerprint matches; and
* emulator-only execution is explicitly non-publishable.

The implementation is an independent CampusPulse design.  CAPER is used only
as a high-level methodological reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
import random
import re
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import numpy as np

from .governance_arena_v2 import (
    ACTION_CATALOG,
    ACTION_CATEGORIES,
    ACTIVITY,
    BRANCH_ACTIONS,
    CONCERN,
    EVIDENCE_ACCEPTANCE,
    MessageObject,
    NON_EVENT_TOPIC_ID,
    OPPOSITION,
    PUBLIC_ACTIONS,
    RUMOR_BELIEF,
    SATISFACTION,
    STATE_FIELDS,
    STANCE_CATEGORIES,
    SUPPORT,
    TIMELINE,
    TRUST,
    VERIFIED_BELIEF,
    BranchState,
    GovernanceArenaV2Error,
    _action_utility,
    _clip,
    _hash_uniform,
    _greg_categorical,
    _greg_numeric,
    _kish_ess,
    _normalize,
    _parent_features,
    _partial_pooling_v2,
    _prune_dominated_actions,
    _sampling_strata,
    _seed_from,
    _systematic_resample_parent,
    _softmax_probabilities,
    fixed_size_inclusion_probabilities,
    pivotal_sample,
)
from .llm_agents import (
    AgentDecision,
    RenderedAgentTurn,
    extract_json_object,
    validate_rendered_turn,
)
from .mass_population import canonical_hash


POPULATION_SCHEMA_VERSION = "semantic-llm-agent-population-v1"
RESIDENT_TURN_SCHEMA_VERSION = "campus-pulse-budgeted-llm-agent-turn-v1"
ACTOR_TURN_SCHEMA_VERSION = "campus-pulse-governance-actor-turn-v1"
ACTIVATION_ROUND_SCHEMA_VERSION = "campus-pulse-llm-activation-round-v1"
CHECKPOINT_SCHEMA_VERSION = "campus-pulse-llm-population-checkpoint-v3"

EXECUTION_PROVENANCES = (
    "emulator_only_development",
    "reviewed_trace_replay",
    "authorized_live_llm",
)
PUBLISHABLE_PROVENANCES = {
    "reviewed_trace_replay",
    "authorized_live_llm",
}

LONGITUDINAL_ANCHOR_COUNT = 16
PPS_SAMPLE_BUDGET = 32
RESIDENT_LLM_BUDGET = LONGITUDINAL_ANCHOR_COUNT + PPS_SAMPLE_BUDGET
RESIDENT_LLM_SLOTS = 12_288
GOVERNANCE_ACTOR_LLM_SLOTS = 768
TOTAL_LLM_SLOTS = RESIDENT_LLM_SLOTS + GOVERNANCE_ACTOR_LLM_SLOTS

# Resident turns use scenario-level topic identifiers because those are the
# stable semantic contract exposed to the LLM.  The frozen emulator was trained
# on the reviewed benchmark's narrower primary-topic vocabulary.  Keep the
# semantic observation intact for messages and memory, but translate it at the
# estimator boundary so GREG compares like with like.
EMULATOR_TOPIC_OBSERVATION_MAP = {
    "lecture_external_incident_shock": "direct_event",
    "governance_legitimacy_dispute": "tongzhou",
    "tongzhou_governance_information_shock": "tongzhou",
}

ACTOR_IDS = (
    "governance_authority",
    "service_operator",
    "community_bridge",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTACT_RE = re.compile(
    r"(?:1[3-9]\d{9})|(?:\b\d{5,}\b)|(?:https?://)|(?:www\.)|(?:微信|QQ)",
    re.IGNORECASE,
)


class YuLanScaleError(ValueError):
    """Raised when a YuLan-Scale identity or semantic contract is invalid."""


def emulator_topic_observation_id(
    topic_id: str,
    *,
    available_categories: Sequence[str] | None = None,
) -> str:
    """Map an LLM-facing scenario topic into the frozen emulator vocabulary."""

    value = str(topic_id)
    mapped = EMULATOR_TOPIC_OBSERVATION_MAP.get(value, value)
    if available_categories is None:
        return mapped
    available = {str(category) for category in available_categories}
    if mapped in available:
        return mapped
    if value in available:
        return value
    return mapped


@dataclass(frozen=True)
class TurnEnvelope:
    """One validated resident or governance-actor semantic turn."""

    slot_kind: str
    prompt_sha256: str
    response_sha256: str
    provenance: str
    content: str
    returned_model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    provider_calls: int = 0
    cache_hit: bool = False
    provider_attempts: int = 0
    format_repairs: int = 0
    provider_billed_tokens: int = 0
    attempt_records: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RenderedGovernanceActorTurn:
    """Strict prompt carrier for one limited-observation governance actor."""

    actor_id: str
    scenario_id: str
    branch: str
    tick: int
    feasible_actions: tuple[str, ...]
    system_prompt: str
    actor_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class GovernanceActorDecision:
    """Validated LLM preference distribution before constrained QRE."""

    actor_id: str
    action_probabilities: dict[str, float]
    preferred_action: str
    confidence: float
    rationale_tags: tuple[str, ...]
    risk_level: str
    uncertainty: float
    public_message: str


class SemanticTurnSource(Protocol):
    """Transport-neutral source for exact resident and actor turns."""

    provenance: str

    def acquire(
        self,
        *,
        slot_kind: str,
        prompt_sha256: str,
        rendered: Any | None = None,
        validator: Any | None = None,
    ) -> TurnEnvelope: ...


class ReviewedTraceReplaySource:
    """Fail-closed replay keyed by slot kind and exact prompt fingerprint."""

    provenance = "reviewed_trace_replay"

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        expected_release_sha256: str | None = None,
    ) -> None:
        if expected_release_sha256 is not None and (
            not isinstance(expected_release_sha256, str)
            or _SHA256_RE.fullmatch(expected_release_sha256) is None
        ):
            raise YuLanScaleError("trace release SHA-256 is invalid")
        by_key: dict[tuple[str, str], TurnEnvelope] = {}
        for raw in entries:
            slot_kind = str(raw.get("slot_kind") or "")
            prompt_sha = str(
                raw.get("prompt_sha256")
                or raw.get("prompt_fingerprint")
                or ""
            )
            response_sha = str(
                raw.get("response_sha256")
                or raw.get("response_fingerprint")
                or ""
            )
            trace_origin = raw.get("provenance")
            is_reviewed_release_entry = (
                isinstance(raw.get("validated_decision"), Mapping)
                and raw.get("review_status") == "approved"
                and isinstance(raw.get("source_artifact_sha256"), str)
            )
            if (
                slot_kind not in {"resident", "governance_actor"}
                or _SHA256_RE.fullmatch(prompt_sha) is None
                or _SHA256_RE.fullmatch(response_sha) is None
                or (
                    trace_origin
                    not in {"authorized_live_llm", "reviewed_trace_replay"}
                    and not is_reviewed_release_entry
                )
            ):
                raise YuLanScaleError("trace entry identity is invalid")
            # Formal trace releases persist the validated structured decision,
            # never the Provider response body.  ``content`` remains accepted
            # for the in-memory transport adapter, while release entries are
            # reconstructed from ``validated_decision``.
            content = raw.get("content")
            if content is None:
                validated = raw.get("validated_decision")
                if not isinstance(validated, Mapping):
                    raise YuLanScaleError(
                        "trace entry lacks a validated structured decision"
                    )
                validated = {
                    key: value
                    for key, value in validated.items()
                    if key != "schema_version"
                }
                content = _strict_json(validated)
            if not isinstance(content, str) or not content:
                raise YuLanScaleError("trace entry lacks validated content")
            usage = raw.get("provider_usage")
            if usage is not None and not isinstance(usage, Mapping):
                raise YuLanScaleError("trace entry Provider usage is invalid")
            envelope = TurnEnvelope(
                slot_kind=slot_kind,
                prompt_sha256=prompt_sha,
                response_sha256=response_sha,
                provenance="reviewed_trace_replay",
                content=content,
                returned_model=str(
                    raw.get("returned_model")
                    or raw.get("model_name")
                    or "unknown"
                ),
                prompt_tokens=int(
                    (
                        usage.get("input_tokens", 0)
                        if isinstance(usage, Mapping)
                        else raw.get("prompt_tokens", 0)
                    )
                ),
                completion_tokens=int(
                    (
                        usage.get("output_tokens", 0)
                        if isinstance(usage, Mapping)
                        else raw.get("completion_tokens", 0)
                    )
                ),
                total_tokens=int(
                    (
                        int(usage.get("input_tokens", 0))
                        + int(usage.get("output_tokens", 0))
                        if isinstance(usage, Mapping)
                        else raw.get("total_tokens", 0)
                    )
                ),
                provider_calls=0,
                cache_hit=True,
            )
            key = (slot_kind, prompt_sha)
            if key in by_key and by_key[key] != envelope:
                raise YuLanScaleError("trace release contains a conflicting prompt")
            by_key[key] = envelope
        self._entries = by_key
        self.expected_release_sha256 = expected_release_sha256

    def acquire(
        self,
        *,
        slot_kind: str,
        prompt_sha256: str,
        rendered: Any | None = None,
        validator: Any | None = None,
    ) -> TurnEnvelope:
        del rendered, validator
        try:
            return self._entries[(slot_kind, prompt_sha256)]
        except KeyError as exc:
            raise YuLanScaleError(
                "exact reviewed trace is missing for the rendered prompt"
            ) from exc


def _strict_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scenario_weight(
    agent: Mapping[str, Any],
    scenario_id: str,
) -> float:
    weights = agent.get("scenario_weights")
    if isinstance(weights, Mapping) and scenario_id in weights:
        return float(weights[scenario_id])
    return float(agent.get("population_weight", 0.0))


def scenario_population_view(
    population: Mapping[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    """Return a v2-kernel compatible view with scenario-specific weights."""

    agents = population.get("agents")
    if (
        population.get("schema_version") != POPULATION_SCHEMA_VERSION
        or not isinstance(agents, list)
        or not agents
    ):
        raise YuLanScaleError("unified LLM-agent population is invalid")
    view_agents = []
    raw_weights = []
    for agent in agents:
        weight = _scenario_weight(agent, scenario_id)
        if not math.isfinite(weight) or weight <= 0.0:
            raise YuLanScaleError("scenario agent weights must be positive")
        copied = copy.deepcopy(agent)
        copied["population_weight"] = weight
        view_agents.append(copied)
        raw_weights.append(weight)
    denominator = sum(raw_weights)
    if denominator <= 0.0:
        raise YuLanScaleError("scenario population has no weight")
    for agent, weight in zip(view_agents, raw_weights):
        agent["population_weight"] = weight / denominator
    body = {
        "schema_version": "semantic-population-v2",
        "population_kind": "scenario_view_of_persistent_llm_agents",
        "population_size": len(view_agents),
        "scenario_id": scenario_id,
        "source_population_sha256": population.get("population_sha256"),
        "agents": view_agents,
    }
    body["population_sha256"] = canonical_hash(body)
    return body


def _allocate_covered_budget(
    capacities: Sequence[int],
    budget: int,
) -> list[int]:
    nonempty = [index for index, value in enumerate(capacities) if value > 0]
    if budget < len(nonempty) or budget > sum(capacities):
        raise YuLanScaleError("PPS budget cannot cover the non-empty strata")
    allocation = [0 for _ in capacities]
    for index in nonempty:
        allocation[index] = 1
    remaining = budget - len(nonempty)
    while remaining:
        candidates = [
            index
            for index in nonempty
            if allocation[index] < capacities[index]
        ]
        if not candidates:
            raise YuLanScaleError("PPS allocation exhausted stratum capacity")
        selected = max(
            candidates,
            key=lambda index: (
                capacities[index] / max(allocation[index], 1),
                capacities[index],
                -index,
            ),
        )
        allocation[selected] += 1
        remaining -= 1
    return allocation


def draw_rotating_pps_sample_v1(
    *,
    population: Mapping[str, Any],
    state: BranchState,
    features: Mapping[str, np.ndarray],
    anchor_indices: Sequence[int],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    budget: int = PPS_SAMPLE_BUDGET,
) -> dict[str, Any]:
    """Draw the 32-unit inference sample while excluding fixed anchors."""

    count = state.ensemble.parent_count
    anchors = tuple(sorted(set(int(index) for index in anchor_indices)))
    if (
        len(anchors) != LONGITUDINAL_ANCHOR_COUNT
        or any(index < 0 or index >= count for index in anchors)
    ):
        raise YuLanScaleError("the longitudinal anchor panel is invalid")
    if budget != PPS_SAMPLE_BUDGET or count - len(anchors) < budget:
        raise YuLanScaleError("the rotating PPS budget is invalid")

    parent = state.ensemble.parent_means()
    variances = state.ensemble.parent_variances()
    labels, median_activity = _sampling_strata(
        features=features,
        parent_state=parent,
    )
    anchor_set = set(anchors)
    strata = sorted(
        {
            str(labels[index])
            for index in range(count)
            if index not in anchor_set
        }
    )
    members = {
        label: [
            index
            for index in range(count)
            if index not in anchor_set and str(labels[index]) == label
        ]
        for label in strata
    }
    allocations = _allocate_covered_budget(
        [len(members[label]) for label in strata],
        budget,
    )
    inclusion = np.zeros(count, dtype=np.float64)
    selected: list[int] = []
    allocation_by_stratum: dict[str, int] = {}
    for position, label in enumerate(strata):
        local_members = members[label]
        local_budget = allocations[position]
        allocation_by_stratum[label] = local_budget
        scores = []
        for index in local_members:
            uncertainty = (
                float(variances[index, SUPPORT])
                + float(variances[index, OPPOSITION])
                + 0.5
                * (
                    1.0
                    - abs(
                        float(parent[index, SUPPORT])
                        - float(parent[index, OPPOSITION])
                    )
                )
            )
            rotation_penalty = 1.0 / (
                1.0 + 0.45 * float(state.rotation_counts[index])
            )
            scores.append(
                max(float(state.ensemble.parent_weights[index]), 1e-12)
                * max(float(parent[index, ACTIVITY]), 0.03)
                * (
                    0.55
                    + uncertainty
                    + 0.22 * float(features["relevance"][index])
                )
                * rotation_penalty
            )
        local_pi = fixed_size_inclusion_probabilities(scores, local_budget)
        local_selection = pivotal_sample(
            local_pi,
            random.Random(
                _seed_from(
                    seed,
                    scenario_id,
                    branch,
                    tick,
                    label,
                    "yulan-scale-pps",
                )
            ),
        )
        for local_index, parent_index in enumerate(local_members):
            inclusion[parent_index] = float(local_pi[local_index])
        selected.extend(
            local_members[local_index] for local_index in local_selection
        )
    selected = tuple(sorted(selected))
    if (
        len(selected) != budget
        or set(selected) & anchor_set
        or any(inclusion[index] <= 0.0 for index in selected)
        or any(
            inclusion[index] <= 0.0
            for index in range(count)
            if index not in anchor_set
        )
    ):
        raise YuLanScaleError("rotating PPS sample violated its frozen design")
    return {
        "schema_version": ACTIVATION_ROUND_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "anchor_indices": anchors,
        "anchor_count": len(anchors),
        "selected": selected,
        "pps_count": len(selected),
        "resident_llm_budget": len(anchors) + len(selected),
        "inclusion_probability": inclusion,
        "strata": labels,
        "median_activity": median_activity,
        "allocation_by_stratum": allocation_by_stratum,
        "anchor_identity_sha256": canonical_hash(
            [
                population["agents"][index].get(
                    "llm_agent_id",
                    population["agents"][index].get("micro_agent_id"),
                )
                for index in anchors
            ]
        ),
        "pps_identity_sha256": canonical_hash(
            [
                population["agents"][index].get(
                    "llm_agent_id",
                    population["agents"][index].get("micro_agent_id"),
                )
                for index in selected
            ]
        ),
    }


def _visible_message_view(
    messages: Sequence[MessageObject | Mapping[str, Any]],
    *,
    tick: int,
) -> list[dict[str, Any]]:
    visible = []
    for raw in messages:
        if isinstance(raw, MessageObject):
            item = {
                "claim_id": raw.claim_id,
                "source_role": raw.source_role,
                "topic": raw.topic,
                "evidence_status": raw.evidence_status,
                "emotion": round(float(raw.emotion), 4),
                "actionability": round(float(raw.actionability), 4),
                "distortion": round(float(raw.distortion), 4),
                "created_tick": int(raw.created_tick),
                "ttl": int(raw.ttl),
                "provenance": raw.provenance,
            }
        elif isinstance(raw, Mapping):
            item = {
                "claim_id": str(raw.get("claim_id") or ""),
                "source_role": str(raw.get("source_role") or "synthetic"),
                "topic": str(raw.get("topic") or NON_EVENT_TOPIC_ID),
                "evidence_status": str(
                    raw.get("evidence_status") or "uncertain"
                ),
                "emotion": round(_clip(raw.get("emotion", 0.0)), 4),
                "actionability": round(
                    _clip(raw.get("actionability", 0.0)),
                    4,
                ),
                "distortion": round(_clip(raw.get("distortion", 0.0)), 4),
                "created_tick": int(raw.get("created_tick", -1)),
                "ttl": int(raw.get("ttl", 0)),
                "provenance": str(
                    raw.get("provenance") or "amortized_llm_policy"
                ),
            }
        else:
            raise YuLanScaleError("resident visible message is invalid")
        if (
            item["created_tick"] <= tick
            and tick - item["created_tick"] <= item["ttl"]
        ):
            visible.append(item)
    return sorted(
        visible,
        key=lambda item: (
            -item["created_tick"],
            item["claim_id"],
            item["source_role"],
        ),
    )[:3]


def render_resident_llm_turn_v1(
    *,
    agent: Mapping[str, Any],
    posterior_state: Mapping[str, float],
    visible_messages: Sequence[MessageObject | Mapping[str, Any]],
    llm_memory: Sequence[Mapping[str, Any]],
    amortized_history: Sequence[Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
) -> RenderedAgentTurn:
    """Render one persistent resident LLM agent without hidden truth."""

    if not 0 <= tick < len(TIMELINE):
        raise YuLanScaleError("resident LLM tick is invalid")
    persona = agent.get("persona")
    if not isinstance(persona, Mapping):
        raise YuLanScaleError("resident LLM agent lacks a reviewed persona")
    gate = str(agent.get("simulation_gate") or "active_episode_agent")
    allowed_actions = list(ACTION_CATEGORIES)
    if str(agent.get("primary_archetype_id")) == "archetype-16":
        allowed_actions = ["ignore", "read"]
    elif str(agent.get("primary_archetype_id")) == "archetype-09":
        allowed_actions = ["ignore", "read", "report_risk"]
    posterior = {
        field: round(_clip(posterior_state.get(field, 0.0)), 6)
        for field in STATE_FIELDS
    }
    prompt = {
        "schema_version": RESIDENT_TURN_SCHEMA_VERSION,
        "simulation_target": (
            "one persistent synthetic LLM agent grounded by a reviewed "
            "forum-behaviour episode; not an identified student"
        ),
        "agent_contract": {
            "llm_agent_id": str(agent.get("llm_agent_id") or ""),
            "primary_archetype_id": str(
                agent.get("primary_archetype_id") or ""
            ),
            "persona_variant_id": str(
                agent.get("persona_variant_id") or ""
            ),
            "simulation_gate": gate,
            "persona": {
                key: persona.get(key)
                for key in (
                    "profile_summary",
                    "observable_core_tendencies",
                    "trigger_conditions",
                    "language_tendencies",
                    "do_not_simulate",
                    "situational_signature",
                    "reaction_logic",
                    "language_style",
                    "uncertainty_behavior",
                    "simulation_instruction",
                    "assigned_episode_focus",
                )
            },
        },
        "temporal_contract": {
            "current_tick": tick,
            "phase": TIMELINE[tick],
            "future_information_forbidden": True,
        },
        "scenario_id": scenario_id,
        "branch": branch,
        "posterior_state": posterior,
        "last_real_llm_memory": list(llm_memory)[-2:],
        "amortized_state_history": list(amortized_history)[-2:],
        "visible_messages": _visible_message_view(
            visible_messages,
            tick=tick,
        ),
        "output_contract": {
            "action": {"allowed": allowed_actions},
            "topic_id": {
                "allowed": [scenario_id, NON_EVENT_TOPIC_ID],
            },
            "stance": {"allowed": list(STANCE_CATEGORIES)},
            "urgency": "finite number from 0 to 1",
            "trust_delta": "finite number from -0.25 to 0.25",
            "satisfied": "boolean",
            "rationale_tag": "controlled existing rationale tag",
            "message": (
                "empty for ignore/read; otherwise synthetic text of at "
                "most 80 Chinese characters"
            ),
        },
    }
    agent_prompt = _strict_json(prompt)
    system_prompt = (
        "You are one persistent synthetic CampusPulse LLM social agent. "
        "Follow the reviewed persona, your own memory and only temporally "
        "visible messages. Return exactly one JSON object."
    )
    topic_ids = tuple(sorted((scenario_id, NON_EVENT_TOPIC_ID)))
    activation = {
        "event_visible": TIMELINE[tick] != "baseline",
        "public_action_allowed": True,
        "temporal_cutoff_tick": tick,
        "provider_call_authorized": True,
    }
    fingerprint = canonical_hash(
        {
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
            "topic_ids": list(topic_ids),
            "simulation_mode": "persistent_budgeted_llm_agent",
            "prior_messages": [
                str(item.get("message") or "")
                for item in list(llm_memory)[-2:]
                if item.get("message")
            ],
        }
    )
    return RenderedAgentTurn(
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        topic_ids=topic_ids,
        simulation_mode="persistent_budgeted_llm_agent",
        prior_messages=tuple(
            str(item.get("message") or "")
            for item in list(llm_memory)[-2:]
            if item.get("message")
        ),
        activation=activation,
        prompt_sha256=fingerprint,
    )


def validate_resident_llm_turn_v1(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedAgentTurn,
    agent: Mapping[str, Any],
) -> AgentDecision:
    """Validate exact LLM/replay content and enforce population gates."""

    if envelope.slot_kind != "resident":
        raise YuLanScaleError("resident slot received a non-resident trace")
    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise YuLanScaleError("resident trace prompt fingerprint drifted")
    if envelope.provenance not in PUBLISHABLE_PROVENANCES:
        raise YuLanScaleError("resident LLM turn is not publishable evidence")
    decision = validate_rendered_turn(envelope.content, rendered)
    archetype = str(agent.get("primary_archetype_id") or "")
    if archetype == "archetype-16" and decision.action not in {
        "ignore",
        "read",
    }:
        raise YuLanScaleError("background signal emitted an autonomous action")
    if archetype == "archetype-09" and decision.action not in {
        "ignore",
        "read",
        "report_risk",
    }:
        raise YuLanScaleError("risk signal emitted an inadmissible action")
    return decision


def _resident_semantic(
    decision: AgentDecision,
    *,
    parent_state: np.ndarray,
    provenance: str,
    prompt_sha256: str,
    response_sha256: str,
) -> dict[str, Any]:
    return {
        "action": decision.action,
        "stance": decision.stance,
        "topic_id": decision.topic_id,
        "urgency": round(float(decision.urgency), 6),
        "trust": round(
            _clip(float(parent_state[TRUST]) + decision.trust_delta),
            6,
        ),
        "satisfaction": float(decision.satisfied),
        "activity": float(decision.action in PUBLIC_ACTIONS),
        "message": decision.message,
        "trust_delta": round(float(decision.trust_delta), 6),
        "rationale_tag": decision.rationale_tag,
        "prompt_fingerprint": prompt_sha256,
        "response_fingerprint": response_sha256,
        "provenance": provenance,
    }


def apply_individual_llm_updates_v1(
    *,
    population: Mapping[str, Any],
    state: BranchState,
    decisions: Mapping[int, AgentDecision],
    envelopes: Mapping[int, TurnEnvelope],
    llm_memories: dict[int, list[dict[str, Any]]],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
) -> dict[str, Any]:
    """Update only activated agents and replace their current emissions."""

    if set(decisions) != set(envelopes):
        raise YuLanScaleError("resident decisions and envelopes do not align")
    if len(decisions) != RESIDENT_LLM_BUDGET:
        raise YuLanScaleError("resident LLM round must contain exactly 48 turns")
    parent_before = state.ensemble.parent_means()
    semantics: dict[int, dict[str, Any]] = {}
    applied = 0
    resampling_before = state.ensemble.resampling_count
    for index in sorted(decisions):
        if not 0 <= index < state.ensemble.parent_count:
            raise YuLanScaleError("resident LLM decision index is invalid")
        decision = decisions[index]
        envelope = envelopes[index]
        if envelope.provenance not in PUBLISHABLE_PROVENANCES:
            raise YuLanScaleError("individual LLM update lacks real trace provenance")
        innovation_id = canonical_hash(
            [
                "individual-llm-update-v1",
                scenario_id,
                branch,
                tick,
                seed,
                index,
                envelope.prompt_sha256,
                envelope.response_sha256,
            ]
        )
        if innovation_id in state.ensemble.innovations:
            continue
        state.ensemble.innovations.add(innovation_id)
        # Keep the pre-innovation particle state stable while reconciling all
        # output fields.  A NumPy view here would make later assignments read
        # already-mutated values and introduce an undocumented field-order
        # dependence.
        particles = state.ensemble.values[index].copy()
        if decision.stance == "supportive":
            likelihood = 0.12 + 0.88 * particles[:, SUPPORT]
        elif decision.stance in {"critical", "concerned"}:
            likelihood = 0.12 + 0.88 * np.maximum(
                particles[:, OPPOSITION],
                particles[:, CONCERN],
            )
        else:
            likelihood = 0.20 + 0.80 * (
                1.0
                - np.abs(
                    particles[:, SUPPORT] - particles[:, OPPOSITION]
                )
            )
        activity_observed = float(decision.action in PUBLIC_ACTIONS)
        likelihood *= (
            0.20
            + 0.80
            * (
                particles[:, ACTIVITY]
                if activity_observed
                else 1.0 - particles[:, ACTIVITY]
            )
        )
        state.ensemble.weights[index] *= np.maximum(likelihood, 1e-8)
        mass = float(state.ensemble.weights[index].sum())
        if mass <= 1e-15:
            state.ensemble.weights[index] = (
                state.ensemble.parent_weights[index]
                / state.ensemble.values.shape[1]
            )
        else:
            state.ensemble.weights[index] *= (
                state.ensemble.parent_weights[index] / mass
            )

        # Bounded state reconciliation applies to this persistent agent only.
        state.ensemble.values[index, :, CONCERN] = np.clip(
            0.72 * particles[:, CONCERN] + 0.28 * decision.urgency,
            0.0,
            1.0,
        )
        state.ensemble.values[index, :, TRUST] = np.clip(
            particles[:, TRUST] + decision.trust_delta,
            0.0,
            1.0,
        )
        state.ensemble.values[index, :, SATISFACTION] = np.clip(
            0.78 * particles[:, SATISFACTION]
            + 0.22 * float(decision.satisfied),
            0.0,
            1.0,
        )
        state.ensemble.values[index, :, ACTIVITY] = np.clip(
            0.70 * particles[:, ACTIVITY] + 0.30 * activity_observed,
            0.0,
            1.0,
        )
        if decision.stance == "supportive":
            state.ensemble.values[index, :, SUPPORT] = np.clip(
                particles[:, SUPPORT] + 0.08,
                0.0,
                1.0,
            )
        elif decision.stance in {"critical", "concerned"}:
            state.ensemble.values[index, :, OPPOSITION] = np.clip(
                particles[:, OPPOSITION] + 0.08,
                0.0,
                1.0,
            )
        if state.ensemble.parent_ess()[index] < 5.0:
            _systematic_resample_parent(
                state.ensemble,
                index,
                seed=seed,
                innovation_id=innovation_id,
            )

        semantic = _resident_semantic(
            decision,
            parent_state=parent_before[index],
            provenance=envelope.provenance,
            prompt_sha256=envelope.prompt_sha256,
            response_sha256=envelope.response_sha256,
        )
        semantics[index] = semantic
        memory = llm_memories.setdefault(index, [])
        memory.append(
            {
                "tick": tick,
                "phase": TIMELINE[tick],
                "branch": branch,
                "action": decision.action,
                "topic_id": decision.topic_id,
                "stance": decision.stance,
                "urgency": round(float(decision.urgency), 6),
                "satisfied": bool(decision.satisfied),
                "message": decision.message,
                "provenance": envelope.provenance,
                "innovation_id": innovation_id,
            }
        )
        if len(memory) > 4:
            del memory[:-4]
        _replace_agent_emission_v1(
            population=population,
            state=state,
            index=index,
            semantic=semantic,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            seed=seed,
        )
        state.rotation_counts[index] += 1
        applied += 1
    state.ensemble.assert_conserved()
    return {
        "semantics": semantics,
        "summary": {
            "applied_count": applied,
            "resampling_count": (
                state.ensemble.resampling_count - resampling_before
            ),
        },
    }


def _replace_agent_emission_v1(
    *,
    population: Mapping[str, Any],
    state: BranchState,
    index: int,
    semantic: Mapping[str, Any],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
) -> None:
    previous_position = int(state.ledger.last_by_parent[index])
    previous = (
        state.ledger.messages[previous_position]
        if previous_position >= 0
        and state.ledger.messages[previous_position].created_tick == tick
        else None
    )
    is_public = str(semantic["action"]) in PUBLIC_ACTIONS
    if not is_public and previous is not None:
        state.ledger.messages[previous_position] = MessageObject(
            **{
                **previous.__dict__,
                # The v2 propagation kernel already treats ``suppressed`` as
                # inactive.  Provenance records why it was suppressed without
                # adding a second inactive-action spelling downstream.
                "action": "suppressed",
                "provenance": "superseded_by_exact_llm_turn",
            }
        )
        prior_position = next(
            (
                position
                for position in range(previous_position - 1, -1, -1)
                if state.ledger.messages[position].source_parent_index == index
                and state.ledger.messages[position].action
                not in {"suppressed", "superseded"}
            ),
            -1,
        )
        state.ledger.last_by_parent[index] = prior_position
        state.ledger.last_tick_by_parent[index] = (
            state.ledger.messages[prior_position].created_tick
            if prior_position >= 0
            else -100
        )
        return
    if not is_public:
        return
    parent_state = state.ensemble.parent_means()[index]
    agent = population["agents"][index]
    replacement = MessageObject(
        message_id=canonical_hash(
            [
                scenario_id,
                seed,
                branch,
                tick,
                index,
                semantic["response_fingerprint"],
                "llm-message",
            ]
        ),
        parent_message_id=(
            previous.parent_message_id if previous is not None else None
        ),
        claim_id=f"{scenario_id}:primary-claim",
        source_role="persistent_llm_resident",
        source_parent_index=index,
        source_group=str(agent["primary_archetype_id"]),
        target_group=str(agent["primary_archetype_id"]),
        topic=str(semantic["topic_id"]),
        stance_support=float(parent_state[SUPPORT]),
        stance_opposition=float(parent_state[OPPOSITION]),
        evidence_status=(
            "verified"
            if parent_state[VERIFIED_BELIEF]
            >= parent_state[RUMOR_BELIEF]
            else "unverified"
        ),
        emotion=float(semantic["urgency"]),
        actionability=float(parent_state[ACTIVITY]),
        distortion=float(
            max(
                0.0,
                parent_state[RUMOR_BELIEF]
                - parent_state[VERIFIED_BELIEF],
            )
        ),
        created_tick=tick,
        ttl=3,
        action=str(semantic["action"]),
        provenance=str(semantic["provenance"]),
        bridge=bool(previous.bridge) if previous is not None else False,
        depth=int(previous.depth) if previous is not None else 0,
    )
    if previous is None:
        state.ledger.add(replacement)
    else:
        state.ledger.messages[previous_position] = replacement


def estimate_population_from_llm_v1(
    *,
    state: BranchState,
    pps_sample: Mapping[str, Any],
    pps_semantics: Mapping[int, Mapping[str, Any]],
    emulator_predictions: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the population from only the 32 rotating LLM observations."""

    selected = tuple(int(index) for index in pps_sample["selected"])
    if len(selected) != PPS_SAMPLE_BUDGET or set(selected) != set(pps_semantics):
        raise YuLanScaleError("PPS observations do not match the 32-unit sample")
    actions = emulator_predictions.get("action")
    stances = emulator_predictions.get("stance")
    topics = emulator_predictions.get("topic_id")
    numeric = emulator_predictions.get("numeric")
    if not all(isinstance(item, Mapping) for item in (actions, stances, topics)):
        raise YuLanScaleError("emulator categorical predictions are incomplete")
    if not isinstance(numeric, Mapping):
        raise YuLanScaleError("emulator numeric predictions are incomplete")
    weights = state.ensemble.parent_weights
    corrected_action = _greg_categorical(
        population_weights=weights,
        prediction=actions,
        observations={
            index: str(pps_semantics[index]["action"])
            for index in selected
        },
        sample=pps_sample,
    )
    corrected_stance = _greg_categorical(
        population_weights=weights,
        prediction=stances,
        observations={
            index: str(pps_semantics[index]["stance"])
            for index in selected
        },
        sample=pps_sample,
    )
    corrected_topic = _greg_categorical(
        population_weights=weights,
        prediction=topics,
        observations={
            index: emulator_topic_observation_id(
                str(pps_semantics[index]["topic_id"]),
                available_categories=tuple(str(key) for key in topics),
            )
            for index in selected
        },
        sample=pps_sample,
    )
    corrected_numeric = {
        name: _greg_numeric(
            population_weights=weights,
            prediction=np.asarray(prediction, dtype=np.float64),
            observations={
                index: float(pps_semantics[index][name])
                for index in selected
            },
            sample=pps_sample,
        )
        for name, prediction in numeric.items()
        if name in {"urgency", "trust", "satisfaction", "activity"}
    }
    pooling_semantics = {
        index: {
            **pps_semantics[index],
            "activity": float(pps_semantics[index]["activity"]),
        }
        for index in selected
    }
    pooling = _partial_pooling_v2(
        sample=pps_sample,
        parent_weights=weights,
        semantics=pooling_semantics,
    )
    kish = _kish_ess(pps_sample, weights)
    pi_values = [
        float(pps_sample["inclusion_probability"][index])
        for index in selected
    ]
    public = {
        "schema_version": "campus-pulse-llm-population-posterior-v1",
        "budget": PPS_SAMPLE_BUDGET,
        "anchor_observations_excluded_from_estimator": True,
        "topic_observation_mapping": copy.deepcopy(
            EMULATOR_TOPIC_OBSERVATION_MAP
        ),
        "allocation_by_stratum": copy.deepcopy(
            pps_sample["allocation_by_stratum"]
        ),
        "inclusion_probability_range": [
            round(min(pi_values), 8),
            round(max(pi_values), 8),
        ],
        "kish_ess": round(kish, 6),
        "corrected": {
            "action": {
                key: round(value, 6)
                for key, value in corrected_action.items()
            },
            "stance": {
                key: round(value, 6)
                for key, value in corrected_stance.items()
            },
            "topic_id": {
                key: round(value, 6)
                for key, value in corrected_topic.items()
            },
            "numeric": {
                key: round(value, 6)
                for key, value in corrected_numeric.items()
            },
        },
        "partial_pooling": pooling,
    }
    private = {
        **copy.deepcopy(public),
        "pps_identity_sha256": pps_sample["pps_identity_sha256"],
        "selected_indices": list(selected),
        "selected_inclusion_probability": {
            str(index): round(
                float(pps_sample["inclusion_probability"][index]),
                12,
            )
            for index in selected
        },
        "llm_decisions": {
            str(index): copy.deepcopy(dict(pps_semantics[index]))
            for index in selected
        },
    }
    return public, private


def render_governance_actor_turn_v1(
    *,
    actor_id: str,
    scenario_id: str,
    branch: str,
    tick: int,
    private_observation: Mapping[str, float],
    feasible_actions: Sequence[str],
    resource: Mapping[str, float],
) -> RenderedGovernanceActorTurn:
    """Render an actor-specific prompt that cannot contain hidden truth."""

    if (
        actor_id not in ACTOR_IDS
        or branch not in {"A", "B", "C", "D"}
        or tick not in {3, 7, 13, 19}
    ):
        raise YuLanScaleError("governance actor turn identity is invalid")
    if any("truth" in str(key).lower() for key in private_observation):
        raise YuLanScaleError("governance actor observation leaked truth")
    allowed = tuple(
        action_id
        for action_id in _prune_dominated_actions(feasible_actions)
        if ACTION_CATALOG.get(action_id, {}).get("actor") == actor_id
    )
    # A/B/C constrain the catalogue by policy mechanism.  Some roles
    # legitimately have no non-noop action in one of those branches, but the
    # frozen trace budget still reserves all three actor slots.  Such a role
    # receives an explicit noop-only contract instead of a fabricated action.
    prompt = {
        "schema_version": ACTOR_TURN_SCHEMA_VERSION,
        "actor_id": actor_id,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "phase": TIMELINE[tick],
        "private_observation": {
            str(key): round(float(value), 6)
            for key, value in sorted(private_observation.items())
        },
        "resource": {
            key: round(float(resource.get(key, 0.0)), 6)
            for key in ("opening", "consumed", "committed", "remaining")
        },
        "feasible_actions": [
            {
                "action_id": action_id,
                "cost": ACTION_CATALOG[action_id]["cost"],
                "delay": ACTION_CATALOG[action_id]["delay"],
                "reliability": ACTION_CATALOG[action_id]["reliability"],
            }
            for action_id in allowed
        ],
        "output_contract": {
            "action_probabilities": (
                "object containing exactly noop and every feasible action; "
                "finite non-negative values summing to one"
            ),
            "preferred_action": "one key from action_probabilities",
            "confidence": "finite number from 0 to 1",
            "rationale_tags": "one to three controlled short tags",
            "risk_level": "low, medium, high, or uncertain",
            "uncertainty": "finite number from 0 to 1",
            "public_message": (
                "empty or a synthetic governance message of at most "
                "100 Chinese characters"
            ),
        },
    }
    actor_prompt = _strict_json(prompt)
    system_prompt = (
        "You are a bounded synthetic governance actor. Use only your private "
        "observation and the feasible action list. Do not assume hidden "
        "population truth. Return exactly one JSON object."
    )
    return RenderedGovernanceActorTurn(
        actor_id=actor_id,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        feasible_actions=allowed,
        system_prompt=system_prompt,
        actor_prompt=actor_prompt,
        prompt_sha256=canonical_hash(
            {
                "system_prompt": system_prompt,
                "actor_prompt": actor_prompt,
                "actor_id": actor_id,
            }
        ),
    )


def validate_governance_actor_turn_v1(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedGovernanceActorTurn,
) -> GovernanceActorDecision:
    """Validate one governance LLM preference distribution."""

    if envelope.slot_kind != "governance_actor":
        raise YuLanScaleError("actor slot received a resident trace")
    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise YuLanScaleError("actor trace prompt fingerprint drifted")
    if envelope.provenance not in PUBLISHABLE_PROVENANCES:
        raise YuLanScaleError("actor turn is not publishable LLM evidence")
    raw = extract_json_object(envelope.content)
    expected = {
        "action_probabilities",
        "preferred_action",
        "confidence",
        "rationale_tags",
        "risk_level",
        "uncertainty",
        "public_message",
    }
    if set(raw) != expected:
        raise YuLanScaleError("actor output keys do not match the contract")
    probabilities = raw["action_probabilities"]
    allowed = {"noop", *rendered.feasible_actions}
    if not isinstance(probabilities, Mapping) or set(probabilities) != allowed:
        raise YuLanScaleError("actor action probabilities are incomplete")
    normalized: dict[str, float] = {}
    for key in sorted(allowed):
        value = probabilities[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise YuLanScaleError("actor probability is invalid")
        normalized[key] = float(value)
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise YuLanScaleError("actor probabilities do not sum to one")
    preferred = str(raw["preferred_action"])
    if preferred not in allowed:
        raise YuLanScaleError("actor preferred action is infeasible")
    tags = raw["rationale_tags"]
    if (
        not isinstance(tags, list)
        or not 1 <= len(tags) <= 3
        or any(
            not isinstance(tag, str)
            or not 1 <= len(tag) <= 40
            for tag in tags
        )
    ):
        raise YuLanScaleError("actor rationale tags are invalid")
    risk = str(raw["risk_level"])
    if risk not in {"low", "medium", "high", "uncertain"}:
        raise YuLanScaleError("actor risk level is invalid")
    confidence = raw["confidence"]
    uncertainty = raw["uncertainty"]
    for value, name in (
        (confidence, "confidence"),
        (uncertainty, "uncertainty"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise YuLanScaleError(f"actor {name} is invalid")
    message = raw["public_message"]
    if (
        not isinstance(message, str)
        or len(message) > 100
        or _CONTACT_RE.search(message)
    ):
        raise YuLanScaleError("actor public message is unsafe")
    return GovernanceActorDecision(
        actor_id=rendered.actor_id,
        action_probabilities=normalized,
        preferred_action=preferred,
        confidence=float(confidence),
        rationale_tags=tuple(tags),
        risk_level=risk,
        uncertainty=float(uncertainty),
        public_message=" ".join(message.strip().split()),
    )


def _zscore(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    numbers = [float(value) for value in values.values()]
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(value) - mean) / scale for key, value in values.items()}


def play_llm_governance_round_v1(
    *,
    state: BranchState,
    actor_decisions: Mapping[str, GovernanceActorDecision],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    temperature: float = 0.20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fuse LLM preferences with rollout and coordination under hard bounds."""

    if branch not in {"A", "B", "C", "D"}:
        raise YuLanScaleError("Natural cannot execute governance LLM actors")
    if set(actor_decisions) != set(ACTOR_IDS):
        raise YuLanScaleError("governance LLM round lacks actor decisions")
    observations = {
        actor: dict(state.actor_beliefs[actor]) for actor in ACTOR_IDS
    }
    if any("truth" in key.lower() for obs in observations.values() for key in obs):
        raise YuLanScaleError("actor observation leaked hidden truth")
    remaining = (
        state.resource_opening
        - state.resource_consumed
        - state.resource_committed
    )
    coordination: dict[str, float] = {}
    final_probabilities: dict[str, dict[str, float]] = {}
    final_utilities: dict[str, dict[str, float]] = {}
    selections = {actor: "noop" for actor in ACTOR_IDS}
    iterations = []
    for iteration in range(3):
        next_coordination: dict[str, float] = {}
        iteration_rows = []
        for actor in ACTOR_IDS:
            decision = actor_decisions[actor]
            permitted = set(
                _prune_dominated_actions(
                    [
                        action_id
                        for action_id in BRANCH_ACTIONS[branch]
                        if ACTION_CATALOG[action_id]["actor"] == actor
                        and float(ACTION_CATALOG[action_id]["cost"])
                        <= remaining + 1e-9
                    ]
                )
            )
            if not set(decision.action_probabilities).issubset(
                {"noop", *permitted}
            ):
                raise YuLanScaleError("LLM actor proposed an infeasible action")
            candidates = tuple(sorted(decision.action_probabilities))
            rollout = {
                action_id: _action_utility(
                    actor=actor,
                    action_id=action_id,
                    observation=observations[actor],
                    coordination=coordination,
                )
                for action_id in candidates
            }
            rollout_z = _zscore(rollout)
            coordination_values = {}
            for action_id in candidates:
                if action_id == "noop":
                    coordination_values[action_id] = 0.0
                    continue
                effects = ACTION_CATALOG[action_id]["effects"]
                coordination_values[action_id] = sum(
                    min(float(value), float(coordination.get(key, 0.0)))
                    for key, value in effects.items()
                )
            coordination_z = _zscore(coordination_values)
            utilities = {
                action_id: (
                    math.log(
                        max(
                            float(
                                decision.action_probabilities[action_id]
                            ),
                            1e-6,
                        )
                    )
                    + 0.35 * rollout_z[action_id]
                    + 0.20 * coordination_z[action_id]
                )
                for action_id in candidates
            }
            probabilities = _softmax_probabilities(utilities, temperature)
            uniform = _hash_uniform(
                seed,
                scenario_id,
                branch,
                tick,
                iteration,
                actor,
                "llm-qre",
            )
            cumulative = 0.0
            selection = candidates[-1]
            for action_id in candidates:
                cumulative += probabilities[action_id]
                if uniform <= cumulative + 1e-12:
                    selection = action_id
                    break
            selections[actor] = selection
            final_probabilities[actor] = probabilities
            final_utilities[actor] = utilities
            if selection != "noop":
                for key, value in ACTION_CATALOG[selection]["effects"].items():
                    next_coordination[key] = (
                        next_coordination.get(key, 0.0) + float(value)
                    )
            iteration_rows.append(
                {
                    "actor": actor,
                    "selected_action": selection,
                    "candidate_utilities": {
                        key: round(value, 6)
                        for key, value in utilities.items()
                    },
                }
            )
        coordination = {
            key: 0.45 * float(coordination.get(key, 0.0))
            + 0.55 * float(next_coordination.get(key, 0.0))
            for key in set(coordination) | set(next_coordination)
        }
        iterations.append(iteration_rows)

    rows = []
    for actor in ACTOR_IDS:
        selection = selections[actor]
        scheduled = False
        cost = 0.0
        delay = 0
        remaining = (
            state.resource_opening
            - state.resource_consumed
            - state.resource_committed
        )
        if selection != "noop":
            definition = ACTION_CATALOG[selection]
            cost = float(definition["cost"])
            if cost <= remaining + 1e-9:
                delay = int(definition["delay"])
                state.resource_committed += cost
                state.pending_actions.append(
                    {
                        "action_id": selection,
                        "actor": actor,
                        "selected_tick": tick,
                        "due_tick": tick + delay,
                        "cost": cost,
                        "effects": dict(definition["effects"]),
                        "provenance": "budgeted_llm_actor_qre_v1",
                    }
                )
                scheduled = True
            else:
                selection = "noop"
                cost = 0.0
        probabilities = final_probabilities[actor]
        rows.append(
            {
                "actor": actor,
                "llm_preferred_action": actor_decisions[
                    actor
                ].preferred_action,
                "selected_action": selection,
                "selection_probability": round(
                    float(probabilities.get(selection, 1.0)),
                    6,
                ),
                "candidate_probabilities": {
                    key: round(value, 6)
                    for key, value in probabilities.items()
                },
                "candidate_utilities": {
                    key: round(value, 6)
                    for key, value in final_utilities[actor].items()
                },
                "confidence": round(actor_decisions[actor].confidence, 6),
                "uncertainty": round(actor_decisions[actor].uncertainty, 6),
                "risk_level": actor_decisions[actor].risk_level,
                "rationale_tags": list(
                    actor_decisions[actor].rationale_tags
                ),
                "scheduled": scheduled,
                "execution_delay": delay,
                "resource_units": cost,
                "public_message_present": bool(
                    actor_decisions[actor].public_message
                ),
            }
        )
    remaining = (
        state.resource_opening
        - state.resource_consumed
        - state.resource_committed
    )
    if remaining < -1e-8:
        raise YuLanScaleError("LLM governance round overspent resources")
    public = {
        "schema_version": "campus-pulse-llm-qre-game-v1",
        "tick": tick,
        "temperature": temperature,
        "response_iterations": 3,
        "lookahead_ticks": 4,
        "utility_contract": {
            "llm_log_probability_weight": 1.0,
            "normalized_rollout_weight": 0.35,
            "coordination_weight": 0.20,
        },
        "decisions": rows,
        "resource_after_decision": {
            "opening": round(state.resource_opening, 6),
            "consumed": round(state.resource_consumed, 6),
            "committed": round(state.resource_committed, 6),
            "remaining": round(remaining, 6),
            "conserved": math.isclose(
                state.resource_opening,
                state.resource_consumed
                + state.resource_committed
                + remaining,
                abs_tol=1e-8,
            ),
        },
    }
    private = {
        **copy.deepcopy(public),
        "private_observations": observations,
        "llm_actor_decisions": {
            actor: {
                "action_probabilities": dict(
                    actor_decisions[actor].action_probabilities
                ),
                "preferred_action": actor_decisions[
                    actor
                ].preferred_action,
                "public_message": actor_decisions[actor].public_message,
            }
            for actor in ACTOR_IDS
        },
        "iterations": iterations,
    }
    state.game_rounds.append(copy.deepcopy(public))
    return public, private


def assert_formal_trace_budget(
    *,
    resident_slots: int,
    actor_slots: int,
    primary_slots: int,
    provider_request_limit: int,
    token_limit: int,
) -> None:
    """Validate the frozen full-trace budget before authorization."""

    if (
        resident_slots != RESIDENT_LLM_SLOTS
        or actor_slots != GOVERNANCE_ACTOR_LLM_SLOTS
        or primary_slots != TOTAL_LLM_SLOTS
        or provider_request_limit != 2 * TOTAL_LLM_SLOTS
        or token_limit != 40_000_000
    ):
        raise YuLanScaleError("formal LLM trace budget drifted")


def build_provider_preflight_v1(
    *,
    plan_sha256: str,
    population_sha256: str,
    model_release_sha256: str,
    provider_profile_sha256: str,
    model: str,
    endpoint: str,
    authorization_required: bool = True,
    allow_insecure_localhost_for_testing: bool = False,
) -> dict[str, Any]:
    """Build the body-only, secret-free preflight required before live use."""

    for value, field in (
        (plan_sha256, "plan_sha256"),
        (population_sha256, "population_sha256"),
        (model_release_sha256, "model_release_sha256"),
        (provider_profile_sha256, "provider_profile_sha256"),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise YuLanScaleError(f"{field} is invalid")
    if not model or not endpoint:
        raise YuLanScaleError("provider preflight lacks model or endpoint")
    parsed_endpoint = urlsplit(endpoint)
    hostname = (parsed_endpoint.hostname or "").lower()
    insecure_localhost = (
        parsed_endpoint.scheme == "http"
        and allow_insecure_localhost_for_testing is True
        and hostname in {"localhost", "127.0.0.1", "::1"}
    )
    if (
        parsed_endpoint.scheme != "https"
        and not insecure_localhost
        or not parsed_endpoint.netloc
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    ):
        raise YuLanScaleError(
            "formal Provider endpoints require HTTPS; insecure HTTP is "
            "allowed only for an explicitly opted-in localhost test"
        )
    if authorization_required is not True:
        raise YuLanScaleError(
            "live execution always requires a new single-use authorization"
        )
    body = {
        "schema_version": "campus-pulse-llm-provider-preflight-v1",
        "plan_sha256": plan_sha256,
        "population_sha256": population_sha256,
        "model_release_sha256": model_release_sha256,
        "provider_profile_sha256": provider_profile_sha256,
        "model": model,
        "endpoint": endpoint,
        "transport_security": (
            "explicit_insecure_localhost_test_only"
            if insecure_localhost
            else "https_required"
        ),
        "resident_semantic_slots": RESIDENT_LLM_SLOTS,
        "governance_actor_slots": GOVERNANCE_ACTOR_LLM_SLOTS,
        "primary_semantic_slots": TOTAL_LLM_SLOTS,
        "provider_request_limit": TOTAL_LLM_SLOTS * 2,
        "provider_token_limit": 40_000_000,
        "network_retry_limit": 0,
        "format_repair_limit": 1,
        "resident_max_output_tokens": 512,
        "actor_max_output_tokens": 384,
        "max_concurrency": 32,
        "authorization_required": True,
        "authorization_present": False,
        "authorization_validity": (
            "one explicit single-use authorization issued after this "
            "exact preflight; expires when the run starts, is cancelled, "
            "or any bound hash changes"
        ),
        "cancellation_method": (
            "cancel the run through the existing runtime cancel route; "
            "unstarted slots stop immediately, while an unknown Provider "
            "outcome pauses for manual reconciliation"
        ),
        "sent_fields": [
            "reviewed persona and variant",
            "bounded posterior state",
            "two prior semantic memories",
            "three temporally visible synthetic messages",
            "scenario, branch, phase, and strict output contract",
        ],
        "fields_forbidden": [
            "raw forum text",
            "source identifiers",
            "hidden population truth",
            "future events",
            "credentials",
            "historical target shares",
        ],
        "privacy_scan_status": "required_before_authorization",
        "provider_calls_executed": 0,
    }
    body["preflight_sha256"] = canonical_hash(body)
    return body


def publication_eligibility(
    *,
    execution_provenance: str,
    completed_resident_slots: int,
    completed_actor_slots: int,
    fixture_slots: int,
    prompt_fingerprint_mismatches: int,
) -> dict[str, Any]:
    """Return a fail-closed publication gate; never relabel development output."""

    if execution_provenance not in EXECUTION_PROVENANCES:
        raise YuLanScaleError("unknown YuLan-Scale execution provenance")
    complete = (
        completed_resident_slots == RESIDENT_LLM_SLOTS
        and completed_actor_slots == GOVERNANCE_ACTOR_LLM_SLOTS
    )
    eligible = (
        execution_provenance in PUBLISHABLE_PROVENANCES
        and complete
        and fixture_slots == 0
        and prompt_fingerprint_mismatches == 0
    )
    reasons = []
    if execution_provenance not in PUBLISHABLE_PROVENANCES:
        reasons.append("development_or_non_llm_provenance")
    if not complete:
        reasons.append("formal_trace_slots_incomplete")
    if fixture_slots:
        reasons.append("fixture_slots_present")
    if prompt_fingerprint_mismatches:
        reasons.append("prompt_fingerprint_mismatch")
    return {
        "publication_eligible": eligible,
        "execution_provenance": execution_provenance,
        "required_resident_slots": RESIDENT_LLM_SLOTS,
        "required_actor_slots": GOVERNANCE_ACTOR_LLM_SLOTS,
        "completed_resident_slots": int(completed_resident_slots),
        "completed_actor_slots": int(completed_actor_slots),
        "fixture_slots": int(fixture_slots),
        "prompt_fingerprint_mismatches": int(
            prompt_fingerprint_mismatches
        ),
        "reasons": reasons,
    }
