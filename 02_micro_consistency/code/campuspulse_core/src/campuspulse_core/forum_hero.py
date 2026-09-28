"""End-to-end Natural/D ForumTwin hero experiment.

This is the product experiment runner, not a second simulation kernel.  It
coordinates the existing strict resident turns, ForumLedger, activation
scheduler and governance QRE.  Background updates remain silent; every public
message must come from the injected formal semantic source.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from .forum_actor_execution import execute_forum_actor_round_v1
from .forum_execution import (
    execute_resident_activation_round_v1,
    execute_resident_activation_round_v2,
    execute_resident_activation_round_v3,
    execute_resident_activation_round_v4,
    execute_resident_activation_round_v5,
)
from .forum_turns import (
    FORUM_AGENT_TURN_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    V5_FULL_NATURAL_PROFILE,
    V5_MODEL_VIEW_CONDITIONS,
)
from .forum_relationships import (
    RelationshipUpdate,
    SyntheticRelationship,
    apply_relationship_update,
    relationship_attention_probability,
    select_relation_attention_opens,
)
from .forum_ecology_calibration import (
    continuation_probability,
    validate_forum_ecology_calibration,
)
from .forum_risk import RISK_SIGNAL_NAMES, build_governance_risk_result_v1
from .forum_pilot import (
    BRIDGE_DISABLED_ABLATION_MODE,
    ACTIVATION_POLICIES,
    COST_AWARE_ACTIVATION_POLICY,
    FULL_SYSTEM_ABLATION_MODE,
    FORUM_TWIN_SCENARIOS,
    HERO_RUNTIME_SCENARIOS,
    MEMORY_DISABLED_ABLATION_MODE,
    LEGACY_ACTIVATION_POLICY,
    MECHANISM_PROTOCOLS,
    OPEN_CHOICE_MECHANISM_PROTOCOL,
    PILOT_ABLATION_MODES,
    PILOT_BRANCHES,
    PILOT_END_TICK,
    PILOT_MODEL,
    PILOT_SCENARIO,
    PILOT_SEED,
    PILOT_START_TICK,
    RESOURCE_POLICY_BRANCH_LABELS,
    RESOURCE_POLICY_BRANCHES,
    RESOURCE_POLICY_DECISION_TICKS,
    RESOURCE_POLICY_END_TICK,
    RESOURCE_POLICY_RESIDENT_BUDGET,
    RESOURCE_POLICY_SHARED_END_TICK,
    RESOURCE_POLICY_START_TICK,
    FOUR_DIMENSION_BRANCHES,
    FOUR_DIMENSION_BRANCH_LABELS,
    FOUR_DIMENSION_DECISION_TICKS,
    FOUR_DIMENSION_END_TICK,
    FOUR_DIMENSION_RESIDENT_BUDGET,
    FOUR_DIMENSION_SHARED_END_TICK,
    FOUR_DIMENSION_START_TICK,
    RESOURCE_FLAGSHIP_BRANCH_LABELS,
    RESOURCE_FLAGSHIP_BRANCHES,
    RESOURCE_FLAGSHIP_DECISION_TICKS,
    RESOURCE_FLAGSHIP_END_TICK,
    RESOURCE_FLAGSHIP_RESIDENT_BUDGET,
    RESOURCE_FLAGSHIP_SEEDS,
    RESOURCE_FLAGSHIP_SHARED_END_TICK,
    RESOURCE_FLAGSHIP_SPINE_TICKS,
    RESOURCE_FLAGSHIP_START_TICK,
    GOVERNANCE_LEGITIMACY_SCENARIO,
    CENTURY_GYM_BRANCH_LABELS,
    CENTURY_GYM_BRANCHES,
    CENTURY_GYM_DECISION_TICKS,
    CENTURY_GYM_END_TICK,
    CENTURY_GYM_RESIDENT_BUDGET,
    CENTURY_GYM_SCENARIO,
    CENTURY_GYM_SHARED_END_TICK,
    CENTURY_GYM_START_TICK,
    QUICK_LIVE_BRANCH_LABELS,
    QUICK_LIVE_BRANCHES,
    QUICK_LIVE_DECISION_TICKS,
    QUICK_LIVE_END_TICK,
    QUICK_LIVE_RESIDENT_BUDGET,
    QUICK_LIVE_SHARED_END_TICK,
    QUICK_LIVE_START_TICK,
    TARGETED_CLOSED_LOOP_PROTOCOL,
    PROFILE_SHUFFLED_ABLATION_MODE,
    VISIBLE_TEXT_REMOVED_ABLATION_MODE,
    build_pilot_activation_round,
    compile_pilot_profile_bridge,
    pilot_resident_budget,
    scenario_context_for_tick,
    scenario_weight_key,
    select_cost_aware_anchors,
    select_pilot_anchors,
)
from .forum_sampling import (
    PUBLIC_FORUM_ACTIONS,
    apply_forum_posterior_writeback,
    audit_scheduler_recall,
    beta_dirichlet_forum_pooling,
    generalized_difference_categorical,
)
from .forum_twin import (
    AgentMemory,
    FeedExposure,
    ForumLedger,
    ForumMessage,
    INTERACTION_KINDS,
    INTERACTION_REASON_PRIORITY,
    InteractionCandidate,
)
from .governance_arena_v2 import (
    BRANCH_ACTIONS,
    MessageObject as ParticleMessageObject,
    STATE_FIELDS as PARTICLE_STATE_FIELDS,
    _parent_features,
    advance_particle_tick,
    assimilate_forum_llm_updates_v2,
    compile_message_exposure_graph_v2,
    initial_branch_state,
)
from .mass_population import build_synthetic_exposure_graph, canonical_hash
from .yulan_scale import ACTOR_IDS, SemanticTurnSource, scenario_population_view


HERO_RESULT_SCHEMA = "campus-pulse-forum-hero-result-v1"
RESOURCE_POLICY_RESULT_SCHEMA = "campus-pulse-resource-allocation-sandbox-result-v1"
RESOURCE_FLAGSHIP_RESULT_SCHEMA = (
    "campus-pulse-resource-allocation-flagship-result-v2"
)
CENTURY_GYM_RESULT_SCHEMA = "campus-pulse-century-gym-sandbox-result-v1"
QUICK_LIVE_RESULT_SCHEMA = "campus-pulse-cost-bounded-forum-vignette-result-v1"
FORUM_V4_RESOURCE_HERO_RESULT_SCHEMA = (
    "campus-pulse-forum-v4-resource-closed-loop-result-v1"
)
FOUR_DIMENSION_HERO_RESULT_SCHEMA = (
    "campus-pulse-four-dimension-resource-hero-result-v1"
)
PUBLIC_TICK_SCHEMA = "campus-pulse-forum-public-tick-v1"
HERO_ARTIFACT_SCHEMA = "campus-pulse-forum-hero-artifacts-v1"
EVENT_INITIATOR_COUNT = 4
DECISION_TICKS = frozenset({3, 7, 13, 19})
PUBLIC_HOT_LIST_SIZE = 10
PUBLIC_LATEST_LIST_SIZE = 10
PUBLIC_BOARD_BROWSE_POOL_SIZE = 12
# Frozen full-corpus snapshot counts.  Their ratio controls only whether a
# non-notified resident opens the independent-post composer or an existing
# thread.  It never forces the resident to publish and never changes ranking.
SOURCE_FORUM_POST_ROWS = 132_917
SOURCE_FORUM_COMMENT_ROWS = 454_053
INDEPENDENT_ROOT_OPPORTUNITY_RATE = SOURCE_FORUM_POST_ROWS / (
    SOURCE_FORUM_POST_ROWS + SOURCE_FORUM_COMMENT_ROWS
)
GOVERNANCE_UPTAKE_MAP_SCHEMA_VERSION = "campus-pulse-governance-uptake-map-v1"
PUBLIC_HOT_WINDOW_TICKS = 3
STATE_KEYS = (
    "attention",
    "concern",
    "trust",
    "satisfaction",
    "support",
    "opposition",
    "activity_probability",
)
FORUM_BRANCHES = ("Natural", "A", "B", "C", "D")
SHARED_BASELINE_BRANCH = "shared_baseline"
FORUM_TWIN_FORMAL_SEEDS = (
    20260722,
    20261731,
    20262740,
    20263749,
    20264758,
    20265767,
    20266776,
    20267785,
)


class ForumHeroError(ValueError):
    """Raised when the hero experiment violates public-content or pairing rules."""


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _scenario_design_weight(
    profile: Mapping[str, Any], scenario_id: str
) -> float:
    return float(
        profile.get("scenario_weights", {}).get(
            scenario_weight_key(scenario_id), 0.0
        )
    )


def _pooled_stance(value: str) -> str:
    return value if value in {"support", "oppose"} else "uncertain"


def _forum_population_recovery(
    *,
    profiles: Sequence[Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, Any]],
    activation: Any,
    resident_result: Mapping[str, Any],
    interaction_candidates: Sequence[InteractionCandidate],
    scenario_id: str,
) -> dict[str, Any] | None:
    """Recover public-action/stance composition using PPS turns only."""

    if not activation.pps_indices:
        return None
    raw_weights = [
        _scenario_design_weight(profile, scenario_id)
        for profile in profiles
    ]
    weight_total = sum(raw_weights)
    if weight_total <= 0 or any(value <= 0 for value in raw_weights):
        raise ForumHeroError("ForumTwin estimator requires positive population weights")
    design_weights = [value / weight_total for value in raw_weights]
    speaking_probabilities: list[float] = []
    stance_probabilities: list[dict[str, float]] = []
    for index in range(len(profiles)):
        state = posterior[index]
        attention = _clamp(float(state.get("attention", 0.0)))
        activity = _clamp(float(state.get("activity_probability", 0.0)))
        speaking_probabilities.append(_clamp(0.02 + 0.68 * activity + 0.20 * attention))
        support = _clamp(float(state.get("support", 0.0)))
        oppose = _clamp(float(state.get("opposition", 0.0)))
        uncertain = max(0.05, 1.0 - min(1.0, support + oppose))
        total = support + oppose + uncertain
        stance_probabilities.append(
            {
                "support": support / total,
                "oppose": oppose / total,
                "uncertain": uncertain / total,
            }
        )
    slots = {
        str(row["slot_id"]): int(row["agent_index"])
        for row in resident_result["slot_design"]["slots"]
        if row["selection_role"] == "pps"
    }
    attempts = {
        str(row["slot_id"]): row
        for row in resident_result["attempts"]
        if row["selection_role"] == "pps"
    }
    if set(slots) != set(attempts):
        raise ForumHeroError("PPS turn slots and validated observations drifted")
    inclusion = {
        int(index): float(probability)
        for index, probability in zip(
            activation.pps_indices, activation.pps_inclusion_probabilities
        )
    }
    observed_actions: dict[int, str] = {}
    observed_stances: dict[int, str] = {}
    for slot_id, index in slots.items():
        decision = attempts[slot_id]["validated_decision"]
        observed_actions[index] = str(decision["action"])
        observed_stances[index] = _pooled_stance(str(decision["stance"]))
    if set(observed_actions) != set(inclusion):
        raise ForumHeroError("PPS inclusion design and LLM observations drifted")
    action_categories = ("public", "silent")
    action_surrogate = [
        {"public": value, "silent": 1.0 - value}
        for value in speaking_probabilities
    ]
    observed_public = {
        index: ("public" if action in PUBLIC_FORUM_ACTIONS else "silent")
        for index, action in observed_actions.items()
    }
    return {
        "schema_version": "campus-pulse-forum-population-recovery-v1",
        "action_greg": generalized_difference_categorical(
            design_weights=design_weights,
            surrogate_probabilities=action_surrogate,
            observed_categories=observed_public,
            inclusion_probabilities=inclusion,
            categories=action_categories,
            anchor_indices=activation.anchor_indices,
        ),
        "stance_greg": generalized_difference_categorical(
            design_weights=design_weights,
            surrogate_probabilities=stance_probabilities,
            observed_categories=observed_stances,
            inclusion_probabilities=inclusion,
            categories=("support", "oppose", "uncertain"),
            anchor_indices=activation.anchor_indices,
        ),
        "partial_pooling": beta_dirichlet_forum_pooling(
            design_weights=design_weights,
            speaking_probabilities=speaking_probabilities,
            conditional_stance_probabilities=stance_probabilities,
            observed_actions=observed_actions,
            observed_stances=observed_stances,
            inclusion_probabilities=inclusion,
        ),
        "scheduler_recall": audit_scheduler_recall(
            pps_observed_actions=observed_actions,
            scheduled_interaction_indices=(
                candidate.agent_index for candidate in interaction_candidates
            ),
        ),
        "pps_identities_public": False,
    }


def _display_to_index(profiles: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        str(profile["synthetic_display_id"]): index
        for index, profile in enumerate(profiles)
    }
    if len(result) != len(profiles):
        raise ForumHeroError("hero profiles contain duplicate public display IDs")
    return result


_SHUFFLED_PROFILE_FIELDS = (
    "profile_bundle_id",
    "primary_macro_role",
    "secondary_macro_role",
    "micro_role_id",
    "micro_interaction_role",
    "episode_focus",
    "reviewed_behavior_examples",
    "stable_persona",
    "episodic_evidence",
    "scenario_relevance",
    "topic_portfolio",
    "need_portfolio",
    "trust_style",
    "uncertainty_handling",
    "language_style",
    "activity_rhythm",
    "attention_budget",
    "long_term_memory_seed",
    "reviewed_evidence_refs",
    "evidence_binding",
)


def _profile_shuffled_prompt_population(
    profiles: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically reassign semantic profiles without changing subjects.

    Identity, state, weights, safety gates and action permissions stay attached
    to the original synthetic subject.  Only the semantic persona supplied to
    the LLM is reassigned to a different compatible profile.
    """

    pools: dict[str, list[int]] = defaultdict(list)
    for index, profile in enumerate(profiles):
        pools[str(profile.get("simulation_gate") or "active_episode_agent")].append(index)
    shuffled: list[dict[str, Any]] = []
    for index, original in enumerate(profiles):
        gate = str(original.get("simulation_gate") or "active_episode_agent")
        candidates = [candidate for candidate in pools[gate] if candidate != index]
        if not candidates:
            raise ForumHeroError("profile-shuffled ablation lacks a compatible donor")
        candidates.sort(
            key=lambda candidate: canonical_hash(
                ["forum-profile-shuffle", seed, index, candidate]
            )
        )
        donor = profiles[candidates[0]]
        prompt_profile = copy.deepcopy(dict(original))
        for field in _SHUFFLED_PROFILE_FIELDS:
            prompt_profile[field] = copy.deepcopy(donor[field])
        original_tendencies = dict(original.get("forum_tendencies", {}))
        donor_tendencies = copy.deepcopy(dict(donor.get("forum_tendencies", {})))
        donor_tendencies["allowed_public_actions"] = list(
            original_tendencies.get("allowed_public_actions", ())
        )
        prompt_profile["forum_tendencies"] = donor_tendencies
        hash_view = {
            key: value
            for key, value in prompt_profile.items()
            if key not in {"profile_prompt_sha256", "initial_state", "scenario_weights"}
        }
        prompt_profile["profile_prompt_sha256"] = canonical_hash(hash_view)
        shuffled.append(prompt_profile)
    return shuffled


def _initial_memories(
    profiles: Sequence[Mapping[str, Any]],
    initial_memory_summaries: Mapping[str, str] | None,
) -> dict[int, AgentMemory]:
    summaries = dict(initial_memory_summaries or {})
    memories = {}
    for index, profile in enumerate(profiles):
        agent_id = str(profile["agent_id"])
        summary = str(
            summaries.get(agent_id)
            or profile.get("long_term_memory_seed")
            or ""
        ).strip()[:600]
        memories[index] = AgentMemory(
            agent_id=agent_id,
            rolling_long_term_summary=summary,
        )
    return memories


def _relationship_runtime_state(
    release: Mapping[str, Any] | None,
    *,
    profiles: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], SyntheticRelationship]:
    if release is None:
        return {}
    agent_ids = {str(profile["agent_id"]) for profile in profiles}
    rows: dict[tuple[str, str], SyntheticRelationship] = {}
    for raw in release.get("edges", ()):
        row = dict(raw)
        row.pop("synthetic", None)
        row["feature_basis"] = tuple(row.get("feature_basis", ()))
        relationship = SyntheticRelationship(**row)
        if (
            relationship.source_agent_id not in agent_ids
            or relationship.target_agent_id not in agent_ids
        ):
            raise ForumHeroError("relationship release references an unknown Agent")
        rows[(relationship.source_agent_id, relationship.target_agent_id)] = relationship
    counts = Counter(source for source, _target in rows)
    if (
        len(agent_ids) != 1_000
        or len(rows) != 8_000
        or set(counts) != agent_ids
        or any(value != 8 for value in counts.values())
    ):
        raise ForumHeroError("relationship release does not provide the frozen 2/4/2 degree")
    return rows


def _scenario_stake_runtime_state(
    release: Mapping[str, Any] | None,
    *,
    scenario_id: str,
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if release is None:
        return {}
    rows = {
        str(row["agent_id"]): row
        for row in release.get("positions", ())
        if row.get("scenario_id") == scenario_id
    }
    expected = {str(profile["agent_id"]) for profile in profiles}
    if set(rows) != expected:
        raise ForumHeroError("scenario stake release is incomplete")
    return rows


def _scenario_stake_prompt_summary(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep all frozen stake values without copying release audit prose into prompts."""

    fields = row.get("fields")
    if not isinstance(fields, Mapping):
        raise ForumHeroError("scenario stake fields are invalid")
    values: dict[str, Any] = {}
    for name, raw in sorted(fields.items()):
        if not isinstance(name, str) or not isinstance(raw, Mapping) or "value" not in raw:
            raise ForumHeroError("scenario stake field is invalid")
        values[name] = raw["value"]
    return {
        "basis": "frozen_synthetic_scenario_position",
        "values": values,
    }


def _relationship_prompt_rows(
    agent_id: str,
    relationships: Mapping[tuple[str, str], SyntheticRelationship],
    *,
    profiles_by_agent_id: Mapping[str, Mapping[str, Any]],
    scenario_stakes: Mapping[str, Mapping[str, Any]],
    public_board_active: bool,
) -> tuple[dict[str, Any], ...]:
    relation_labels = {
        "close_friend": "关系较强的好友",
        "familiar_peer": "熟悉的同伴",
        "cross_group_acquaintance": "不同群体的熟人",
        "co_participant": "曾共同参与讨论的人",
    }
    rows = []
    for (_source, _target), relationship in sorted(relationships.items()):
        if relationship.source_agent_id != agent_id:
            continue
        contact = profiles_by_agent_id.get(relationship.target_agent_id, {})
        stable = contact.get("stable_persona")
        stable = stable if isinstance(stable, Mapping) else {}
        language_style = stable.get("language_style")
        language_style = language_style if isinstance(language_style, Mapping) else {}
        reply_function = str(language_style.get("reply_function") or "参与讨论")
        if len(reply_function) > 24:
            reply_function = "参与讨论"
        stake = scenario_stakes.get(relationship.target_agent_id, {})
        stake_fields = stake.get("fields") if isinstance(stake, Mapping) else {}
        stake_fields = stake_fields if isinstance(stake_fields, Mapping) else {}
        direct_row = stake_fields.get("directly_affected")
        directly_affected = bool(
            isinstance(direct_row, Mapping) and direct_row.get("value") is True
        )
        shared_parts = []
        if public_board_active:
            shared_parts.append("你们都能看到当前全局热榜和最新帖")
        shared_parts.append(
            f"对方是{relation_labels.get(relationship.relation_type, '合成联系人')}"
        )
        shared_parts.append(f"常以{reply_function}方式互动")
        shared_parts.append(
            "对方当前也直接受影响" if directly_affected else "对方当前是间接受影响者"
        )
        rows.append(
            {
                "contact_agent_id": relationship.target_agent_id,
                "relation_type": relationship.relation_type,
                "strength": relationship.strength,
                "trust": relationship.trust,
                "recent_shared_context": "；".join(shared_parts)[:120],
            }
        )
    return tuple(rows)


def _visible_private_conversation_ids(
    ledger: ForumLedger,
    *,
    agent_id: str,
    scenario_id: str,
    branch: str,
    tick: int,
) -> tuple[str, ...]:
    values = []
    for conversation in ledger.private_conversations.values():
        if (
            conversation.scenario_id != scenario_id
            or conversation.branch not in {branch, SHARED_BASELINE_BRANCH}
        ):
            continue
        active = (
            conversation.status == "active"
            and agent_id in conversation.member_agent_ids
        )
        pending = (
            conversation.status == "pending"
            and agent_id in conversation.pending_invite_agent_ids
            and tick <= conversation.invitation_expires_tick
        )
        if active or pending:
            values.append(conversation.conversation_id)
    return tuple(sorted(values))


def _private_interaction_candidates(
    *,
    ledger: ForumLedger,
    profiles: Sequence[Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, float]],
    scenario_id: str,
    branch: str,
    tick: int,
    limit: int,
    excluded_indices: Sequence[int] = (),
) -> tuple[InteractionCandidate, ...]:
    if limit <= 0:
        return ()
    index_by_agent = {
        str(profile["agent_id"]): index for index, profile in enumerate(profiles)
    }
    excluded = {int(index) for index in excluded_indices}
    ranked: dict[int, InteractionCandidate] = {}
    for conversation in ledger.private_conversations.values():
        if (
            conversation.scenario_id != scenario_id
            or conversation.branch not in {branch, SHARED_BASELINE_BRANCH}
        ):
            continue
        recipient_ids = set(conversation.pending_invite_agent_ids)
        recent = [
            message
            for message in ledger.private_messages.values()
            if message.conversation_id == conversation.conversation_id
            and message.created_tick == tick - 1
            and not message.superseded
        ]
        if recent:
            recipient_ids.update(conversation.member_agent_ids)
            recipient_ids.difference_update(message.source_agent_id for message in recent)
        for agent_id in recipient_ids:
            index = index_by_agent.get(agent_id)
            if index is None or index in excluded:
                continue
            attention = float(posterior.get(index, {}).get("attention", 0.0))
            candidate = InteractionCandidate(
                agent_index=index,
                reason="active_claim_participant",
                attention_score=min(1.0, 0.55 + 0.40 * attention),
                enqueued_tick=tick,
                sequence=int(canonical_hash(["private-queue", agent_id, tick])[:8], 16),
            )
            ranked.setdefault(index, candidate)
    return tuple(
        sorted(
            ranked.values(),
            key=lambda item: (
                INTERACTION_REASON_PRIORITY[item.reason],
                -item.attention_score,
                item.sequence,
                item.agent_index,
            ),
        )[:limit]
    )


def _private_initiation_candidates(
    *,
    ledger: ForumLedger,
    profiles: Sequence[Mapping[str, Any]],
    relationships: Mapping[tuple[str, str], SyntheticRelationship],
    scenario_stakes: Mapping[str, Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, float]],
    scenario_id: str,
    branch: str,
    tick: int,
    limit: int,
    excluded_indices: Sequence[int] = (),
    rotation_counts: Mapping[int, int] | None = None,
) -> tuple[InteractionCandidate, ...]:
    """Select likely *first* private-channel users without forcing speech.

    Private-reply scheduling cannot bootstrap a channel that has no messages.
    This selector therefore reserves a bounded opportunity for residents who
    have a concrete public stimulus, a trusted synthetic contact and a
    profile/stake-based reason to tell, ask, vent or coordinate.  The LLM still
    decides whether to return zero, one or two private actions.
    """

    if limit <= 0 or not any(
        message.created_tick < tick and not message.superseded
        for message in ledger.messages.values()
    ):
        return ()

    excluded = {int(index) for index in excluded_indices}
    prior_activations = rotation_counts or {}
    relationships_by_source: dict[str, list[SyntheticRelationship]] = defaultdict(list)
    for relationship in relationships.values():
        relationships_by_source[relationship.source_agent_id].append(relationship)

    recent_public_speakers = {
        message.source_agent_display_id
        for message in ledger.messages.values()
        if not message.superseded and message.created_tick == tick - 1
    }
    ranked: list[tuple[float, int, int, InteractionCandidate]] = []
    for index, profile in enumerate(profiles):
        if (
            index in excluded
            or profile.get("simulation_gate") != "active_episode_agent"
        ):
            continue
        agent_id = str(profile["agent_id"])
        contacts = relationships_by_source.get(agent_id, ())
        if not contacts:
            continue
        stable = profile.get("stable_persona")
        stable = stable if isinstance(stable, Mapping) else {}
        distribution = stable.get("forum_tendency_distribution")
        distribution = distribution if isinstance(distribution, Mapping) else {}

        def tendency(*names: str) -> float:
            return sum(
                max(0.0, float(distribution.get(name, 0.0)))
                for name in names
                if isinstance(distribution.get(name, 0.0), (int, float))
                and not isinstance(distribution.get(name, 0.0), bool)
            )

        private_drive = min(
            1.0,
            max(
                tendency("question", "discussion"),
                tendency("emotional_vent", "complaint", "emotional_expression", "other"),
                tendency("request", "help_request"),
                tendency("information_share"),
            ),
        )
        state = posterior.get(index, {})
        attention = max(0.0, min(1.0, float(state.get("attention", 0.0))))
        concern = max(0.0, min(1.0, float(state.get("concern", 0.0))))
        baseline_activity = max(
            0.0,
            min(
                1.0,
                float(profile.get("forum_tendencies", {}).get("baseline_activity", 0.0)),
            ),
        )
        stake = scenario_stakes.get(agent_id, {})
        fields = stake.get("fields") if isinstance(stake, Mapping) else {}
        fields = fields if isinstance(fields, Mapping) else {}

        def stake_value(name: str) -> float:
            row = fields.get(name)
            value = row.get("value", 0.0) if isinstance(row, Mapping) else 0.0
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0.0, min(1.0, float(value)))
            return 0.0

        direct_row = fields.get("directly_affected")
        directly_affected = bool(
            isinstance(direct_row, Mapping) and direct_row.get("value") is True
        )
        closest = max(
            contacts,
            key=lambda row: (
                row.relation_type == "close_friend",
                row.trust,
                row.strength,
                row.target_agent_id,
            ),
        )
        stable_jitter = (
            int(canonical_hash(["private-init", scenario_id, branch, tick, agent_id])[:8], 16)
            / 0xFFFFFFFF
        )
        score = (
            0.20 * private_drive
            + 0.17 * attention
            + 0.16 * concern
            + 0.08 * baseline_activity
            + 0.10 * closest.trust
            + 0.07 * closest.strength
            + (0.09 if directly_affected else 0.0)
            + 0.05 * stake_value("time_urgency")
            + 0.03 * stake_value("appeal_cost")
            + 0.03 * stake_value("service_channel_dependency")
            + 0.05 * stable_jitter
            - 0.06 * min(4, max(0, int(prior_activations.get(index, 0))))
            + (
                0.04
                if str(profile.get("synthetic_display_id") or "")
                in recent_public_speakers
                else 0.0
            )
        )
        if score < 0.30:
            continue
        sequence = int(
            canonical_hash(["private-init-order", scenario_id, branch, tick, agent_id])[:8],
            16,
        )
        candidate = InteractionCandidate(
            agent_index=index,
            reason="active_claim_participant",
            attention_score=min(1.0, score),
            enqueued_tick=tick,
            sequence=sequence,
        )
        ranked.append((-score, sequence, index, candidate))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(item[-1] for item in ranked[:limit])


def _apply_forum_relationship_updates(
    *,
    relationships: MutableMapping[tuple[str, str], SyntheticRelationship],
    attempts: Sequence[Mapping[str, Any]],
    ledger: ForumLedger,
    profiles: Sequence[Mapping[str, Any]],
    tick: int,
    applied_update_ids: set[str],
) -> int:
    display_to_agent = {
        str(profile["synthetic_display_id"]): str(profile["agent_id"])
        for profile in profiles
    }
    applied = 0
    for attempt in attempts:
        source_id = str(attempt["agent_id"])
        decision = dict(attempt["validated_decision"])
        targets: set[str] = set()
        for field in ("target_message_id", "quote_message_id"):
            message_id = decision.get(field)
            message = ledger.messages.get(str(message_id)) if message_id else None
            if message is not None:
                target = display_to_agent.get(message.source_agent_display_id)
                if target:
                    targets.add(target)
        for action in decision.get("private_actions", ()):
            targets.update(str(value) for value in action.get("target_agent_ids", ()))
            conversation_id = action.get("conversation_id")
            conversation = (
                ledger.private_conversations.get(str(conversation_id))
                if conversation_id
                else None
            )
            if conversation is not None:
                targets.update(
                    value
                    for value in conversation.member_agent_ids
                    if value != source_id
                )
        for target_id in sorted(targets):
            relationship = relationships.get((source_id, target_id))
            if relationship is None:
                continue
            update_id = "relationship-update-" + canonical_hash(
                [source_id, target_id, tick, attempt["response_sha256"]]
            )[:24]
            if update_id in applied_update_ids:
                continue
            update = RelationshipUpdate(
                update_id=update_id,
                relationship_id=relationship.relationship_id,
                tick=tick,
                strength_delta=0.015,
                trust_delta=0.005,
                interaction_increment=1,
                reason="validated_public_or_private_interaction",
            )
            relationships[(source_id, target_id)] = apply_relationship_update(
                relationship, update
            )
            applied_update_ids.add(update_id)
            applied += 1
    return applied


def _relationship_attention_exposures(
    *,
    selected_indices: Sequence[int],
    profiles: Sequence[Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, float]],
    memories: Mapping[int, AgentMemory],
    ledger: ForumLedger,
    relationships: Mapping[tuple[str, str], SyntheticRelationship],
    existing: Mapping[int, Sequence[FeedExposure]],
    tick: int,
    seed: int,
) -> dict[int, tuple[FeedExposure, ...]]:
    if not ledger.messages:
        return {int(index): tuple(existing.get(int(index), ())) for index in selected_indices}
    _board, candidates, routes = build_global_public_board(ledger, tick)
    author_by_display = {
        str(profile["synthetic_display_id"]): str(profile["agent_id"])
        for profile in profiles
    }
    message_by_id = {message.message_id: message for message in candidates}
    ranked = [{"message_id": message.message_id} for message in candidates]
    result: dict[int, tuple[FeedExposure, ...]] = {}
    for raw_index in selected_indices:
        index = int(raw_index)
        profile = profiles[index]
        agent_id = str(profile["agent_id"])
        memory = memories[index]
        probabilities: dict[str, float] = {}
        for message in candidates:
            author_id = author_by_display.get(message.source_agent_display_id)
            relationship = relationships.get((agent_id, str(author_id)))
            probabilities[message.message_id] = relationship_attention_probability(
                profile=profile,
                # Weighted float32 parent means can exceed one by a few ulps
                # after checkpoint restore even though every particle is
                # clipped.  The relationship contract is a probability and
                # therefore receives the explicitly bounded semantic value.
                current_need_match=_clamp(
                    float(posterior[index].get("concern", 0.0))
                ),
                relation_strength=relationship.strength if relationship else 0.0,
                relation_trust=relationship.trust if relationship else 0.0,
                participated_before=message.thread_id in memory.participated_thread_ids,
                direct_notification=message.thread_id in memory.participated_thread_ids,
            )
        opened_ids = select_relation_attention_opens(
            agent_id=agent_id,
            tick=tick,
            globally_ranked_items=ranked,
            open_probabilities=probabilities,
            seed=seed,
        )
        rows = list(existing.get(index, ()))
        present = {row.message_id for row in rows}
        for message_id in opened_ids:
            if message_id in present or len(rows) >= 6:
                continue
            message = message_by_id.get(message_id)
            if message is None:
                continue
            source, rank = routes.get(message_id, ("global_public_board", 0.5))
            author_id = author_by_display.get(message.source_agent_display_id)
            if (agent_id, str(author_id)) in relationships:
                source = "relationship_attention_from_global_board"
            rows.append(
                _exposure(
                    recipient_agent_id=agent_id,
                    message=message,
                    tick=tick,
                    rank=rank,
                    source=source,
                )
            )
            present.add(message_id)
        result[index] = tuple(rows[:6])
    return result


def _initial_posterior(profiles: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    for index, profile in enumerate(profiles):
        state = profile.get("initial_state", {})
        values[index] = {
            key: _clamp(float(state.get(key, 0.5 if key in {"trust", "satisfaction"} else 0.2)))
            for key in STATE_KEYS
        }
    return values


def _particle_posterior(branch_state: Any) -> dict[int, dict[str, float]]:
    """Derive the ForumTwin parent posterior from the actual SMC ensemble."""

    means = branch_state.ensemble.parent_means()
    field_index = {
        name: index for index, name in enumerate(PARTICLE_STATE_FIELDS)
    }
    return {
        index: {
            "attention": float(row[field_index["attention"]]),
            "concern": float(row[field_index["concern"]]),
            "trust": float(row[field_index["trust"]]),
            "satisfaction": float(row[field_index["satisfaction"]]),
            "support": float(row[field_index["support"]]),
            "opposition": float(row[field_index["opposition"]]),
            "activity_probability": float(row[field_index["activity"]]),
        }
        for index, row in enumerate(means)
    }


def _particle_uncertainty(branch_state: Any) -> dict[str, Any]:
    ess = branch_state.ensemble.parent_ess()
    return {
        "particle_count": branch_state.ensemble.particle_count,
        "particles_per_agent": int(branch_state.ensemble.values.shape[1]),
        "minimum_parent_ess": round(float(ess.min()), 6),
        "mean_parent_ess": round(float(ess.mean()), 6),
        "resampling_count": int(branch_state.ensemble.resampling_count),
        "particle_weight_sum": round(
            float(branch_state.ensemble.weights.sum()), 12
        ),
    }


def _forum_particle_updates(
    *,
    attempts: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    index_by_agent = {
        str(profile["agent_id"]): index
        for index, profile in enumerate(profiles)
    }
    updates: dict[int, dict[str, Any]] = {}
    for attempt in attempts:
        index = index_by_agent[str(attempt["agent_id"])]
        updates[index] = {
            "prompt_sha256": str(attempt["prompt_sha256"]),
            "response_sha256": str(attempt["response_sha256"]),
            "decision": copy.deepcopy(dict(attempt["validated_decision"])),
        }
    return updates


def _sync_forum_messages_to_particle_ledger(
    *,
    forum_ledger: ForumLedger,
    branch_state: Any,
    profiles: Sequence[Mapping[str, Any]],
    tick: int,
) -> int:
    """Expose validated LLM messages to the private particle diffusion layer."""

    display_index = _display_to_index(profiles)
    known_ids = {message.message_id for message in branch_state.ledger.messages}
    particle_by_id = {
        message.message_id: message for message in branch_state.ledger.messages
    }
    emotion_values = {
        "calm": 0.15,
        "neutral": 0.25,
        "concerned": 0.62,
        "anxious": 0.78,
        "angry": 0.90,
        "hopeful": 0.45,
        "supportive": 0.40,
        "skeptical": 0.58,
    }
    actionability_values = {
        "post": 0.52,
        "reply": 0.48,
        "quote": 0.58,
        "repost": 0.42,
        "seek_verify": 0.72,
        "correct": 0.86,
        "seek_help": 0.88,
        "report_risk": 0.92,
        "governance_message": 0.90,
    }
    added = 0
    current_messages = sorted(
        (
            message
            for message in forum_ledger.messages.values()
            if message.created_tick == tick and not message.superseded
        ),
        key=lambda message: message.message_id,
    )
    for message in current_messages:
        if message.message_id in known_ids:
            continue
        source_index = display_index.get(message.source_agent_display_id)
        # Governance objects affect residents only after ForumTwin Feed
        # delivery and a real LLM turn.  Treating them as particle-ledger
        # broadcasts would grant every hidden parent the action instantly and
        # bypass the limited-observation forum mechanism.
        if source_index is None and message.source_agent_display_id.startswith(
            "governance:"
        ):
            continue
        source_group = (
            str(profiles[source_index]["primary_macro_role"])
            if source_index is not None
            else message.source_agent_display_id
        )
        parent_id = message.parent_message_id or message.quote_message_id
        parent_particle = particle_by_id.get(str(parent_id)) if parent_id else None
        target_group = (
            parent_particle.source_group if parent_particle is not None else "forum-wide"
        )
        stance_support = (
            message.confidence
            if message.stance in {"support", "supportive"}
            else 0.5 * (1.0 - message.confidence)
        )
        stance_opposition = (
            message.confidence
            if message.stance in {"oppose", "critical", "concerned"}
            else 0.5 * (1.0 - message.confidence)
        )
        distortion = {
            "verified": 0.02,
            "service_receipt": 0.03,
            "corrected": 0.05,
            "contested": 0.42,
            "unverified": 0.55,
        }.get(message.evidence_status, 0.30)
        particle_message = ParticleMessageObject(
            message_id=message.message_id,
            parent_message_id=parent_id,
            claim_id=str(message.claim_id or f"forum:{message.message_id}"),
            source_role=(
                "synthetic_resident" if source_index is not None else source_group
            ),
            source_parent_index=source_index,
            source_group=source_group,
            target_group=target_group,
            topic=message.topic,
            stance_support=_clamp(stance_support),
            stance_opposition=_clamp(stance_opposition),
            evidence_status=message.evidence_status,
            emotion=emotion_values.get(message.emotion, 0.45),
            actionability=actionability_values.get(message.action, 0.50),
            distortion=distortion,
            created_tick=message.created_tick,
            ttl=message.ttl,
            action=message.action,
            provenance=f"forum:{message.provenance.kind}",
            bridge=(
                target_group != "forum-wide" and target_group != source_group
            ),
            depth=(parent_particle.depth + 1 if parent_particle is not None else 0),
        )
        branch_state.ledger.add(particle_message)
        known_ids.add(message.message_id)
        particle_by_id[message.message_id] = particle_message
        added += 1
    return added


def _settle_forum_governance_actions(branch_state: Any, tick: int) -> int:
    """Settle resource commitments without applying global policy bonuses."""

    due = [
        item for item in branch_state.pending_actions
        if int(item.get("due_tick", 10_000)) <= tick
    ]
    branch_state.pending_actions = [
        item for item in branch_state.pending_actions
        if int(item.get("due_tick", 10_000)) > tick
    ]
    for item in due:
        cost = float(item["cost"])
        branch_state.resource_committed -= cost
        branch_state.resource_consumed += cost
    if branch_state.resource_committed < -1e-8:
        raise ForumHeroError("ForumTwin governance commitment became negative")
    return len(due)


def _silent_background_step(
    posterior: MutableMapping[int, dict[str, float]],
    *,
    ledger: ForumLedger,
    tick: int,
) -> None:
    """Apply only non-linguistic decay/exposure pressure between LLM turns."""

    active_messages = sum(message.active_at(tick) for message in ledger.messages.values())
    exposure_pressure = min(0.08, active_messages / 4_000.0)
    for state in posterior.values():
        state["attention"] = _clamp(0.94 * state["attention"] + exposure_pressure)
        state["concern"] = _clamp(0.97 * state["concern"] + 0.45 * exposure_pressure)
        state["activity_probability"] = _clamp(
            0.96 * state["activity_probability"] + 0.30 * exposure_pressure
        )
        state["trust"] = _clamp(0.995 * state["trust"] + 0.0025)
        state["satisfaction"] = _clamp(0.995 * state["satisfaction"] + 0.0025)


def _responded_target_ids(
    ledger: ForumLedger,
    display_id: str,
) -> frozenset[str]:
    """Return visible messages this resident has already answered or quoted.

    A joined thread remains readable, but the same parent/quote must not be
    promoted as a fresh focal notification on every later tick.  This models
    an acknowledged notification without inventing a private read receipt.
    """

    return frozenset(
        str(target_id)
        for message in ledger.messages.values()
        if not message.superseded
        and message.source_agent_display_id == display_id
        for target_id in (message.parent_message_id, message.quote_message_id)
        if target_id is not None
    )


def _claim_participant_display_ids(
    ledger: ForumLedger, claim_id: str | None
) -> frozenset[str]:
    """Return residents whose public messages created, supported or challenged a claim."""

    claim = ledger.claims.get(str(claim_id)) if claim_id is not None else None
    if claim is None:
        return frozenset()
    message_ids = {
        claim.created_by_message_id,
        *claim.supporting_message_ids,
        *claim.challenging_message_ids,
    }
    return frozenset(
        message.source_agent_display_id
        for message_id in message_ids
        if (message := ledger.messages.get(message_id)) is not None
        and not message.superseded
        and not message.source_agent_display_id.startswith("governance:")
    )


def _interaction_candidates(
    *,
    ledger: ForumLedger,
    profiles: Sequence[Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, Any]],
    tick: int,
) -> list[InteractionCandidate]:
    display_index = _display_to_index(profiles)
    responded_by_display = {
        str(profile["synthetic_display_id"]): _responded_target_ids(
            ledger, str(profile["synthetic_display_id"])
        )
        for profile in profiles
    }
    best: dict[int, InteractionCandidate] = {}

    def offer(index: int, reason: str, attention: float, sequence: int) -> None:
        candidate = InteractionCandidate(
            agent_index=index,
            reason=reason,
            attention_score=_clamp(attention),
            enqueued_tick=max(PILOT_START_TICK, tick - 1),
            sequence=sequence,
        )
        previous = best.get(index)
        if previous is None or (
            INTERACTION_REASON_PRIORITY[candidate.reason],
            -candidate.attention_score,
            candidate.sequence,
        ) < (
            INTERACTION_REASON_PRIORITY[previous.reason],
            -previous.attention_score,
            previous.sequence,
        ):
            best[index] = candidate

    recent = sorted(
        (
            message
            for message in ledger.messages.values()
            if not message.superseded and message.created_tick == tick - 1
        ),
        key=lambda item: item.message_id,
    )
    for sequence, message in enumerate(recent):
        linked = None
        reason = "active_claim_participant"
        if message.parent_message_id in ledger.messages:
            linked = ledger.messages[message.parent_message_id]
            reason = "direct_reply"
        elif message.quote_message_id in ledger.messages:
            linked = ledger.messages[message.quote_message_id]
            reason = "quote"
        if linked is not None and linked.source_agent_display_id in display_index:
            index = display_index[linked.source_agent_display_id]
            if message.message_id not in responded_by_display[
                linked.source_agent_display_id
            ]:
                offer(index, reason, 0.95, sequence)
        author_index = display_index.get(message.source_agent_display_id)
        if author_index is not None and message.action in {"seek_help", "report_risk"}:
            offer(
                author_index,
                "help_request" if message.action == "seek_help" else "risk_report",
                0.90,
                sequence,
            )
        if author_index is not None and message.claim_id is not None:
            offer(author_index, "active_claim_participant", 0.75, sequence)
        artifact = message.governance_artifact
        correction_participants = (
            _claim_participant_display_ids(ledger, artifact.target_claim_id)
            if artifact is not None and artifact.artifact_kind == "evidence_card"
            else frozenset()
        )
        for participant in sorted(correction_participants):
            if (
                participant in display_index
                and message.message_id not in responded_by_display[participant]
            ):
                offer(display_index[participant], "correction", 0.98, sequence)
        if message.source_agent_display_id.startswith("governance:"):
            thread_participants = {
                other.source_agent_display_id
                for other in ledger.messages.values()
                if other.thread_id == message.thread_id
            }
            for participant in sorted(thread_participants):
                if (
                    participant in display_index
                    and message.message_id not in responded_by_display[participant]
                ):
                    offer(display_index[participant], "service_receipt", 0.92, sequence)

    # The queue also contains non-speaking candidates inferred from private
    # attention/activity.  This is deliberately non-linguistic: it decides who
    # deserves an LLM turn but never manufactures what the resident would say.
    # Keeping these candidates (including overflow) is what makes PPS
    # scheduler-recall auditable instead of defining success after observing a
    # response.
    base_sequence = len(recent) + 1
    for index, profile in enumerate(profiles):
        if profile.get("simulation_gate") == "background_signal":
            continue
        state = posterior.get(index, {})
        attention = _clamp(float(state.get("attention", 0.0)))
        activity = _clamp(float(state.get("activity_probability", 0.0)))
        voice = _clamp(float(state.get("voice", activity)))
        priority_score = 0.45 * attention + 0.35 * activity + 0.20 * voice
        public_actions = (
            set(
                profile.get("forum_tendencies", {}).get(
                    "allowed_public_actions", ()
                )
            )
            & PUBLIC_FORUM_ACTIONS
        )
        if not public_actions:
            continue
        micro_role = str(profile.get("micro_interaction_role") or "")
        speech_act = str(
            profile.get("forum_tendencies", {}).get("speech_act") or ""
        )
        if (
            any(
                token in micro_role.lower()
                for token in (
                    "bridge", "correction", "evidence", "verification",
                    "桥接", "纠错", "证据", "核实",
                )
            )
            or speech_act in {"information_share", "question"}
        ) and priority_score >= 0.12:
            offer(index, "novel_bridge_exposure", priority_score, base_sequence + index)
        elif priority_score >= 0.15:
            offer(index, "high_attention_silent", priority_score, base_sequence + index)
    return sorted(best.values(), key=lambda item: (item.sequence, item.agent_index))


def _merge_interaction_candidates(
    backlog: Sequence[InteractionCandidate],
    fresh: Sequence[InteractionCandidate],
) -> list[InteractionCandidate]:
    """Merge a persistent scheduler queue without manufacturing public speech."""

    by_agent: dict[int, InteractionCandidate] = {}
    for candidate in (*backlog, *fresh):
        if not isinstance(candidate, InteractionCandidate):
            raise ForumHeroError("interaction backlog contains an invalid candidate")
        previous = by_agent.get(candidate.agent_index)
        if previous is None:
            by_agent[candidate.agent_index] = candidate
            continue
        preferred = min(
            (previous, candidate),
            key=lambda item: (
                INTERACTION_REASON_PRIORITY[item.reason],
                -item.attention_score,
                item.sequence,
            ),
        )
        by_agent[candidate.agent_index] = InteractionCandidate(
            agent_index=candidate.agent_index,
            reason=preferred.reason,
            attention_score=max(previous.attention_score, candidate.attention_score),
            enqueued_tick=min(previous.enqueued_tick, candidate.enqueued_tick),
            sequence=min(previous.sequence, candidate.sequence),
        )
    return sorted(
        by_agent.values(),
        key=lambda item: (
            item.enqueued_tick,
            INTERACTION_REASON_PRIORITY[item.reason],
            item.sequence,
            item.agent_index,
        ),
    )


def _latest_direct_message(
    *,
    ledger: ForumLedger,
    display_id: str,
    tick: int,
    bridge_enabled: bool = True,
) -> ForumMessage | None:
    candidates = []
    responded_targets = _responded_target_ids(ledger, display_id)
    for message in ledger.messages.values():
        if (
            message.created_tick >= tick
            or message.superseded
            or not message.active_at(tick)
            or message.message_id in responded_targets
        ):
            continue
        if (
            not bridge_enabled
            and message.source_agent_display_id == "governance:community_bridge"
        ):
            continue
        linked_ids = (message.parent_message_id, message.quote_message_id)
        if any(
            linked_id in ledger.messages
            and ledger.messages[str(linked_id)].source_agent_display_id == display_id
            for linked_id in linked_ids
            if linked_id is not None
        ):
            candidates.append(message)
            continue
        # A governance object posted into an existing discussion is a direct
        # thread notification for residents who already spoke there.  This
        # changes neither the public hot/latest order nor the resident's
        # freedom to read or ignore it; it only prevents a later ordinary
        # reply in the same tick from hiding the governance object from the
        # very participants it addresses.
        if message.source_agent_display_id.startswith("governance:") and any(
            other.thread_id == message.thread_id
            and other.created_tick < message.created_tick
            and other.source_agent_display_id == display_id
            and not other.superseded
            for other in ledger.messages.values()
        ):
            candidates.append(message)
            continue
        artifact = message.governance_artifact
        if (
            artifact is not None
            and artifact.artifact_kind == "evidence_card"
            and display_id
            in _claim_participant_display_ids(ledger, artifact.target_claim_id)
        ):
            candidates.append(message)
    return max(
        candidates,
        key=lambda item: (
            item.created_tick,
            item.source_agent_display_id.startswith("governance:"),
            item.message_id,
        ),
        default=None,
    )


def _public_board_snapshot(
    ledger: ForumLedger,
    tick: int,
    ecology_calibration_release: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[ForumMessage, ...], dict[str, tuple[str, float]]]:
    """Build one global XiaoLaba-style board for every resident at a tick.

    The board has no profile, relationship, or agent argument by design.  Hot
    ranking is a transparent synthetic proxy for the product experiment, not a
    claim about the platform's proprietary formula: recent public activity,
    recent distinct speakers, total replies, and recency.  Latest is strictly
    ordered by root-post time.
    """

    visible_by_thread: dict[str, list[ForumMessage]] = defaultdict(list)
    for message in ledger.messages.values():
        if (
            message.created_tick < tick
            and not message.superseded
            and message.active_at(tick)
            and message.thread_id in ledger.threads
        ):
            visible_by_thread[message.thread_id].append(message)
    hot_rows: list[
        tuple[float, int, str, ForumMessage, ForumMessage, int, int, int, int]
    ] = []
    latest_rows: list[tuple[int, str, ForumMessage]] = []
    window_start = max(0, tick - PUBLIC_HOT_WINDOW_TICKS)
    for thread_id, messages in visible_by_thread.items():
        thread = ledger.threads[thread_id]
        root = ledger.messages.get(thread.root_message_id)
        if (
            root is None
            or root.superseded
            or root.created_tick >= tick
            or not root.active_at(tick)
        ):
            continue
        ordered = sorted(messages, key=lambda row: (row.created_tick, row.message_id))
        latest = ordered[-1]
        recent = [row for row in ordered if row.created_tick >= window_start]
        recent_speakers = len(
            {row.source_agent_display_id for row in recent}
        )
        total_replies = sum(row.message_id != root.message_id for row in ordered)
        positive_interactions = Counter()
        for row in ordered:
            positive_interactions.update(
                ledger.interaction_counts(row.message_id, through_tick=tick - 1)
            )
        like_count = int(positive_interactions["like"])
        repost_count = int(positive_interactions["repost"])
        age = max(0, tick - latest.created_tick)
        hot_score = (
            float(len(recent))
            + 0.5 * float(recent_speakers)
            + 0.20 * math.log1p(total_replies)
            + 0.30 * math.log1p(like_count)
            + 0.55 * math.log1p(repost_count)
            + 1.0 / (1.0 + float(age))
        )
        hot_rows.append(
            (
                hot_score,
                latest.created_tick,
                thread_id,
                latest,
                root,
                len(recent),
                recent_speakers,
                like_count,
                repost_count,
            )
        )
        latest_rows.append((root.created_tick, thread_id, root))
    hot_rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    latest_rows.sort(key=lambda row: (-row[0], row[1]))
    hot_rows = hot_rows[:PUBLIC_HOT_LIST_SIZE]
    latest_rows = latest_rows[:PUBLIC_LATEST_LIST_SIZE]
    routes: dict[str, tuple[str, float]] = {}
    hot_candidates: list[ForumMessage] = []
    latest_candidates: list[ForumMessage] = []
    hot_public = []
    for rank, row in enumerate(hot_rows, start=1):
        (
            score,
            _last_tick,
            _thread_id,
            latest,
            root,
            recent_count,
            speaker_count,
            like_count,
            repost_count,
        ) = row
        hot_public.append(
            [
                rank,
                root.visible_text[:28],
                recent_count,
                speaker_count,
                like_count,
                repost_count,
                latest.created_tick,
            ]
        )
        routes[latest.message_id] = (
            "public_hot_top10",
            round(0.90 - 0.02 * (rank - 1), 6),
        )
        hot_candidates.append(latest)
    latest_public = []
    for rank, (created_tick, _thread_id, root) in enumerate(latest_rows, start=1):
        latest_public.append([rank, root.visible_text[:28], created_tick])
        routes.setdefault(
            root.message_id,
            ("public_latest_refresh", round(0.68 - 0.015 * (rank - 1), 6)),
        )
        latest_candidates.append(root)
    # The two public lists remain globally ranked and identical for everyone.
    # For the internal browsing pool we interleave their ranks before removing
    # duplicates.  Previously the first eight items were always hot-list rows,
    # which made the latest-post surface effectively unreachable whenever the
    # hot list was full and created an artificial rich-get-richer cascade.
    candidates: list[ForumMessage] = []
    seen: set[str] = set()
    for rank_index in range(max(len(hot_candidates), len(latest_candidates))):
        for source in (hot_candidates, latest_candidates):
            if rank_index >= len(source):
                continue
            message = source[rank_index]
            if message.message_id in seen:
                continue
            candidates.append(message)
            seen.add(message.message_id)
    board = {
        "mechanism": "global_hot_top10_plus_latest_v1",
        "ranking_identical_for_all_residents": True,
        "personalized_ranking": False,
        "relationship_graph_used": False,
        "hot_proxy": (
            "recent public activity + recent distinct speakers + total replies + "
            "LLM likes + LLM reposts + recency; reports never raise public heat"
        ),
        "hot_columns": [
            "rank",
            "root_preview",
            "recent_activity",
            "recent_speakers",
            "likes",
            "reposts",
            "last_active_tick",
        ],
        "hot_top10": hot_public,
        "latest_columns": ["rank", "root_preview", "created_tick"],
        "latest_posts": latest_public,
        "browse_pool": "rank_interleaved_hot_latest_deduplicated_v1",
        "browse_pool_size": min(PUBLIC_BOARD_BROWSE_POOL_SIZE, len(candidates)),
        "tick": tick,
    }
    if ecology_calibration_release is not None:
        validate_forum_ecology_calibration(ecology_calibration_release)
        board["resident_open_selection"] = (
            "real-corpus continuation prior over this shared ranking"
        )
        board["ecology_calibration_release_sha256"] = str(
            ecology_calibration_release["calibration_release_sha256"]
        )
        board["public_ranking_changed_by_calibration"] = False
    return board, tuple(candidates), routes


def build_global_public_board(
    ledger: ForumLedger,
    tick: int,
) -> tuple[dict[str, Any], tuple[ForumMessage, ...], dict[str, tuple[str, float]]]:
    """Public entry point for the globally identical hot/latest board."""

    return _public_board_snapshot(ledger, tick)


def _feed_delivery_route(
    *,
    message: ForumMessage,
    recipient_index: int,
    profiles: Sequence[Mapping[str, Any]],
    memory: AgentMemory,
    tick: int,
    exposure_graph: Any | None,
    bridge_enabled: bool,
    current_thread_id: str | None = None,
    public_board_routes: Mapping[str, tuple[str, float]] | None = None,
) -> tuple[str, float] | None:
    """Return a public-board or notification route, never a graph route."""

    if message.created_tick >= tick or message.superseded or not message.active_at(tick):
        return None
    display_id = str(profiles[recipient_index]["synthetic_display_id"])
    if message.source_agent_display_id == display_id:
        return None
    participated = set(memory.participated_thread_ids)
    if message.thread_id == current_thread_id or message.thread_id in participated:
        return ("joined_thread_notification", 0.96)
    if (
        not bridge_enabled
        and message.source_agent_display_id == "governance:community_bridge"
    ):
        return None
    if public_board_routes is None:
        return ("public_latest_refresh", 0.50)
    return public_board_routes.get(message.message_id)


def _relevant_message(
    *,
    ledger: ForumLedger,
    profile: Mapping[str, Any],
    memory: AgentMemory,
    tick: int,
    bridge_enabled: bool = True,
    recipient_index: int | None = None,
    profiles: Sequence[Mapping[str, Any]] | None = None,
    exposure_graph: Any | None = None,
    public_board_candidates: Sequence[ForumMessage] | None = None,
    ecology_calibration_release: Mapping[str, Any] | None = None,
) -> ForumMessage | None:
    """Choose which globally ranked board item this resident opens.

    The candidate order is global.  A deterministic per-resident offset models
    browsing/click variation without changing or personalizing the ranking.
    """

    display_id = str(profile["synthetic_display_id"])
    responded_targets = _responded_target_ids(ledger, display_id)
    source = public_board_candidates
    if source is None:
        _board, source, _routes = _public_board_snapshot(
            ledger, tick, ecology_calibration_release
        )
    candidates = []
    for message in source:
        if (
            message.created_tick >= tick
            or message.superseded
            or message.source_agent_display_id == display_id
            or message.message_id in responded_targets
        ):
            continue
        if (
            not bridge_enabled
            and message.source_agent_display_id == "governance:community_bridge"
        ):
            continue
        candidates.append(message)
    if not candidates:
        return None
    pool = candidates[: min(PUBLIC_BOARD_BROWSE_POOL_SIZE, len(candidates))]
    if ecology_calibration_release is not None:
        return _ecology_ordered_board_candidates(
            ledger=ledger,
            candidates=pool,
            display_id=display_id,
            tick=tick,
            profiles=profiles,
            ecology_calibration_release=ecology_calibration_release,
        )[0]
    offset = int(
        canonical_hash(["global-board-click", display_id, tick, len(pool)])[:12], 16
    ) % len(pool)
    return pool[offset]


def _ecology_ordered_board_candidates(
    *,
    ledger: ForumLedger,
    candidates: Sequence[ForumMessage],
    display_id: str,
    tick: int,
    profiles: Sequence[Mapping[str, Any]] | None,
    ecology_calibration_release: Mapping[str, Any],
) -> list[ForumMessage]:
    """Order shared-board items by a deterministic, soft continuation prior.

    This is an opening/click model below the globally identical XiaoLaba hot
    and latest lists.  Exponential-race sampling avoids a deterministic quota:
    every candidate keeps positive probability, while root-only and moderately
    developed threads receive the continuation opportunity observed in the
    real aggregate ecology.  Large threads retain an exploration floor.
    """

    validate_forum_ecology_calibration(ecology_calibration_release)
    macro_by_display = {
        str(profile.get("synthetic_display_id") or ""): str(
            profile.get("primary_macro_role") or ""
        )
        for profile in profiles or ()
    }
    release_hash = str(ecology_calibration_release["calibration_release_sha256"])
    ranked: list[tuple[float, int, str, ForumMessage]] = []
    for ordinal, message in enumerate(candidates):
        thread = ledger.threads[message.thread_id]
        root = ledger.messages.get(thread.root_message_id)
        source_macro_role = (
            macro_by_display.get(root.source_agent_display_id) if root else None
        )
        continuation = continuation_probability(
            ecology_calibration_release,
            current_reply_count=int(thread.reply_count),
            source_macro_role=source_macro_role,
        )
        # Preserve a mild preference for higher positions in the *existing*
        # global list while letting empirically plausible continuation compete.
        rank_discount = 1.0 / (1.0 + 0.08 * float(ordinal))
        weight = max(1e-9, continuation * rank_discount)
        digest = canonical_hash(
            [
                "forum-ecology-open-v2",
                release_hash,
                display_id,
                tick,
                message.message_id,
            ]
        )
        uniform = (int(digest[:16], 16) + 1) / (16**16 + 1)
        ranked.append((-math.log(uniform) / weight, ordinal, message.message_id, message))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row[3] for row in ranked]


def _independent_root_opportunity(
    *,
    display_id: str,
    tick: int,
    ecology_calibration_release: Mapping[str, Any],
) -> bool:
    """Choose a post-composer visit from the frozen corpus activity mix.

    This is an interface opportunity, not a semantic assignment.  ``True``
    exposes ``post/read/ignore`` without a preselected parent; the LLM may
    still remain silent.  The release hash and resident/tick identity make the
    choice deterministic and identical across paired policy branches.
    """

    validate_forum_ecology_calibration(ecology_calibration_release)
    digest = canonical_hash(
        [
            "forum-independent-root-opportunity-v1",
            ecology_calibration_release["calibration_release_sha256"],
            display_id,
            tick,
        ]
    )
    uniform = (int(digest[:16], 16) + 0.5) / 16**16
    return uniform < INDEPENDENT_ROOT_OPPORTUNITY_RATE


def _exposure(
    *,
    recipient_agent_id: str,
    message: ForumMessage,
    tick: int,
    rank: float,
    source: str,
) -> FeedExposure:
    return FeedExposure(
        exposure_id="hero-exposure-" + canonical_hash(
            [recipient_agent_id, message.message_id, tick, source]
        )[:28],
        recipient_agent_id=recipient_agent_id,
        message_id=message.message_id,
        delivered_tick=tick,
        exposure_source=source,
        rank_score=rank,
    )


def make_forum_feed_exposure(
    *,
    recipient_agent_id: str,
    message: ForumMessage,
    tick: int,
    rank: float,
    source: str,
) -> FeedExposure:
    """Build one private delivery edge without changing the global ranking."""

    return _exposure(
        recipient_agent_id=recipient_agent_id,
        message=message,
        tick=tick,
        rank=rank,
        source=source,
    )


def _select_event_initiators(
    *,
    selected_indices: Sequence[int],
    profiles: Sequence[Mapping[str, Any]],
    ledger: ForumLedger,
    tick: int,
    scenario_id: str = PILOT_SCENARIO,
) -> frozenset[int]:
    """Pick a small, heterogeneous public-action queue when the event arrives.

    This is an activation decision, not generated speech: selected residents
    still choose the public action and all visible text through the LLM turn.
    The paired branches use the same deterministic rule.
    """

    if tick != PILOT_START_TICK or ledger.messages:
        return frozenset()
    ranked = []
    for raw_index in selected_indices:
        index = int(raw_index)
        profile = profiles[index]
        tendencies = profile.get("forum_tendencies", {})
        public_actions = set(tendencies.get("allowed_public_actions", ())) & {
            "post",
            "seek_help",
            "report_risk",
        }
        if not public_actions:
            continue
        weight = _scenario_design_weight(profile, scenario_id)
        activity = float(
            tendencies.get(
                "baseline_activity",
                profile.get("initial_state", {}).get(
                    "activity_probability", 0.0
                ),
            )
        )
        ranked.append(
            (
                -(max(weight, 0.0) * (0.25 + max(activity, 0.0))),
                str(profile.get("primary_macro_role") or ""),
                index,
            )
        )
    initiators = []
    used_roles = set()
    for _negative_score, macro_role, index in sorted(ranked):
        if macro_role in used_roles:
            continue
        initiators.append(index)
        used_roles.add(macro_role)
        if len(initiators) == EVENT_INITIATOR_COUNT:
            break
    if len(initiators) < EVENT_INITIATOR_COUNT:
        for _negative_score, _macro_role, index in sorted(ranked):
            if index not in initiators:
                initiators.append(index)
            if len(initiators) == EVENT_INITIATOR_COUNT:
                break
    if len(initiators) != EVENT_INITIATOR_COUNT:
        raise ForumHeroError("T3 lacks four eligible event initiators")
    return frozenset(initiators)


def _targeted_capability_panel(
    *,
    profiles: Sequence[Mapping[str, Any]],
    anchor_indices: Sequence[int],
    scenario_id: str,
) -> tuple[int, ...]:
    """Keep sixteen slots while covering the disclosed T3 capabilities.

    The old reviewed pilot happened to place every required capability in its
    sixteen anchors.  The full-corpus population-v2 does not guarantee that.
    This deterministic panel swaps only the missing capability holders into
    the bounded mechanism audit; it never widens an agent's action contract.
    """

    panel = [int(index) for index in anchor_indices]
    if len(panel) != 16 or len(set(panel)) != 16:
        raise ForumHeroError("targeted capability panel requires sixteen anchors")
    protected: set[int] = set()
    required = ("seek_help", "report_risk", "post", "post")

    def allows(index: int, action: str) -> bool:
        profile = profiles[index]
        allowed = set(
            profile.get("forum_tendencies", {}).get(
                "allowed_public_actions", ()
            )
        )
        return action in allowed and (
            action == "report_risk"
            or profile.get("simulation_gate") == "active_episode_agent"
        )

    def rank(index: int) -> tuple[float, str, str, int]:
        profile = profiles[index]
        weight = _scenario_design_weight(profile, scenario_id)
        activity = float(
            profile.get("forum_tendencies", {}).get(
                "baseline_activity", 0.0
            )
        )
        return (
            -(max(weight, 0.0) * (0.25 + max(activity, 0.0))),
            str(profile.get("primary_macro_role") or ""),
            str(profile.get("agent_id") or ""),
            index,
        )

    for action in required:
        existing = sorted(
            (
                index
                for index in panel
                if index not in protected and allows(index, action)
            ),
            key=rank,
        )
        if existing:
            protected.add(existing[0])
            continue
        candidates = sorted(
            (
                index
                for index in range(len(profiles))
                if index not in panel and allows(index, action)
            ),
            key=rank,
        )
        if not candidates:
            raise ForumHeroError(
                f"full population lacks targeted capability: {action}"
            )
        victims = [index for index in reversed(panel) if index not in protected]
        if not victims:
            raise ForumHeroError("targeted capability panel cannot replace an anchor")
        victim = victims[0]
        position = panel.index(victim)
        panel[position] = candidates[0]
        protected.add(candidates[0])
    if len(panel) != 16 or len(set(panel)) != 16:
        raise ForumHeroError("targeted capability panel is not unique")
    return tuple(panel)


def _targeted_mechanism_assignments(
    *,
    selected_indices: Sequence[int],
    profiles: Sequence[Mapping[str, Any]],
    ledger: ForumLedger,
    tick: int,
    event_initiators: frozenset[int],
) -> dict[int, dict[str, Any]]:
    """Freeze disclosed action categories for a bounded mechanism stress test.

    The scheduler fixes only the mechanism role and exact time-visible target.
    All visible language, reasoning, confidence and state updates remain LLM
    outputs.  This protocol is never presented as an open-choice population
    estimate and is kept separate from the paired r9 hero result.
    """

    def actionable(indices: Sequence[int]) -> list[int]:
        return sorted(
            (
                int(index)
                for index in indices
                if profiles[int(index)].get("simulation_gate")
                == "active_episode_agent"
            ),
            key=lambda index: (
                str(profiles[index].get("primary_macro_role") or ""),
                str(profiles[index].get("agent_id") or ""),
                index,
            ),
        )

    assignments: dict[int, dict[str, Any]] = {}
    if tick == 3:
        # Archetype-09 is intentionally a risk-only signal agent.  It may
        # initiate ``report_risk`` but must never be widened into an ordinary
        # episode agent.  Keep all four frozen initiators here and match each
        # disclosed task against the profile's existing action contract below.
        initiators = sorted(
            (int(index) for index in selected_indices),
            key=lambda index: (
                profiles[index].get("simulation_gate") != "risk_signal",
                str(profiles[index].get("primary_macro_role") or ""),
                str(profiles[index].get("agent_id") or ""),
                index,
            ),
        )
        tasks = (
            (
                "help_request",
                "seek_help",
                "create",
                "提出一个与自身画像和当前职业选择困惑一致的具体求助；不得补造个人身份或现实处理结果。",
            ),
            (
                "risk_signal",
                "report_risk",
                "create",
                "报告一个由当前片段化转述造成的具体信息风险，并明确哪些内容尚未核验。",
            ),
            (
                "initial_claim_a",
                "post",
                "create",
                "依据自身画像提出一个可被后续 E1 核验或收窄的具体解释，不得写成既定事实。",
            ),
            (
                "initial_claim_b",
                "post",
                "create",
                "从不同需求或利益角度提出第二个可争议主张，避免复述其他首帖。",
            ),
        )
        remaining = list(initiators)
        matched_tasks: list[tuple[int, tuple[str, str, str, str]]] = []
        for task in tasks:
            action = task[1]
            match = next(
                (
                    index
                    for index in remaining
                    if action
                    in set(
                        profiles[index]
                        .get("forum_tendencies", {})
                        .get("allowed_public_actions", ())
                    )
                    and (
                        action == "report_risk"
                        or profiles[index].get("simulation_gate")
                        == "active_episode_agent"
                    )
                ),
                None,
            )
            if match is None:
                raise ForumHeroError(
                    f"targeted mechanism audit cannot assign required action: {action}"
                )
            remaining.remove(match)
            matched_tasks.append((match, task))
        for index, (task_id, action, operation, brief) in matched_tasks:
            assignments[index] = {
                "task_id": task_id,
                "focal_message_id": None,
                "allowed_actions": [action],
                "brief": brief,
                "required_output": {
                    "action": action,
                    "claim_operation": operation,
                },
            }
        return assignments

    if tick == 4:
        reserved: set[int] = set()
        display_index = _display_to_index(profiles)
        help_messages = sorted(
            (
                message
                for message in ledger.messages.values()
                if not message.superseded and message.action == "seek_help"
            ),
            key=lambda message: (message.created_tick, message.message_id),
        )
        if help_messages:
            help_message = help_messages[0]
            service = next(
                (
                    message
                    for message in sorted(
                        ledger.messages.values(),
                        key=lambda item: (item.created_tick, item.message_id),
                    )
                    if not message.superseded
                    and message.thread_id == help_message.thread_id
                    and message.created_tick >= help_message.created_tick
                    and message.source_agent_display_id
                    == "governance:service_operator"
                ),
                None,
            )
            author_index = display_index.get(help_message.source_agent_display_id)
            if service is not None and author_index in selected_indices:
                reserved.add(int(author_index))
                assignments[int(author_index)] = {
                    "task_id": "help_feedback",
                    "focal_message_id": service.message_id,
                    "allowed_actions": ["reply"],
                    "brief": (
                        "直接回复服务主体的可见回执：说明回执解决了什么、仍缺什么，"
                        "不得把受理表述成已经解决。"
                    ),
                    "required_output": {
                        "action": "reply",
                        "claim_operation": "none",
                        "target_thread_id": service.thread_id,
                        "target_message_id": service.message_id,
                    },
                }
        target_claims = sorted(
            (
                claim
                for claim in ledger.claims.values()
                if not claim.expired and claim.first_seen_tick == 3
            ),
            key=lambda claim: (claim.first_seen_tick, claim.claim_id),
        )
        candidates = [
            index
            for index in actionable(selected_indices)
            if index not in reserved
        ]
        if target_claims and candidates:
            claim = target_claims[0]
            source = ledger.messages[claim.created_by_message_id]
            correction_index = next(
                (
                    index
                    for index in candidates
                    if profiles[index]["synthetic_display_id"]
                    != source.source_agent_display_id
                ),
                candidates[0],
            )
            assignments[correction_index] = {
                "task_id": "bounded_correction",
                "focal_message_id": source.message_id,
                "allowed_actions": ["correct"],
                "brief": (
                    "E1 只确认片段缺少前后语境，因此该可见主张超出了证据边界。"
                    "针对原 claim 做最小幅度纠正：撤回或收窄无依据推断，但不要断言相反结论。"
                ),
                "evidence_packet": {
                    "evidence_id": "E1-targeted-boundary",
                    "target_claim_id": claim.claim_id,
                    "confirmed": "片段缺少足以支持该扩展推断的前后语境",
                    "unresolved": "讲者动机、逐字原话及现实处理结果",
                },
                "required_output": {
                    "action": "correct",
                    "claim_operation": "correct",
                    "target_thread_id": claim.thread_id,
                    "target_claim_id": claim.claim_id,
                    "correction_target_claim_id": claim.claim_id,
                },
            }
        return assignments

    if tick == 5:
        corrections = sorted(
            (
                message
                for message in ledger.messages.values()
                if not message.superseded
                and message.action == "correct"
                and message.correction_target_claim_id is not None
            ),
            key=lambda message: (message.created_tick, message.message_id),
        )
        candidates = actionable(selected_indices)
        if corrections and len(candidates) >= 2:
            correction = corrections[0]
            candidates = [
                index
                for index in candidates
                if profiles[index]["synthetic_display_id"]
                != correction.source_agent_display_id
            ]
            if len(candidates) >= 2:
                roles = (
                    (
                        candidates[0],
                        "correction_accept",
                        "support",
                        "support",
                        "明确说明你接受纠正的哪一部分，以及它如何改变了你的判断。",
                    ),
                    (
                        candidates[1],
                        "correction_reject",
                        "challenge",
                        "oppose",
                        "提出有边界的保留或拒绝：只说明 E1 仍不能解决的具体问题，不得补造反证。",
                    ),
                )
                for index, task_id, operation, stance, brief in roles:
                    assignments[index] = {
                        "task_id": task_id,
                        "focal_message_id": correction.message_id,
                        "allowed_actions": ["reply"],
                        "brief": brief,
                        "required_output": {
                            "action": "reply",
                            "claim_operation": operation,
                            "stance": stance,
                            "target_thread_id": correction.thread_id,
                            "target_message_id": correction.message_id,
                            "target_claim_id": str(
                                correction.correction_target_claim_id
                            ),
                        },
                    }
        return assignments
    return assignments


def _turn_inputs(
    *,
    selected_indices: Sequence[int],
    profiles: Sequence[Mapping[str, Any]],
    memories: Mapping[int, AgentMemory],
    ledger: ForumLedger,
    tick: int,
    scenario_id: str = PILOT_SCENARIO,
    mechanism_protocol: str = OPEN_CHOICE_MECHANISM_PROTOCOL,
    bridge_enabled: bool = True,
    forum_context_mode: str = "full_text",
    exposure_graph: Any | None = None,
    ecology_calibration_release: Mapping[str, Any] | None = None,
    opening_public_action_mode: str = "event_initiators_only",
) -> tuple[dict[int, list[FeedExposure]], dict[int, dict[str, Any]]]:
    if mechanism_protocol not in MECHANISM_PROTOCOLS:
        raise ForumHeroError("turn inputs received an invalid mechanism protocol")
    if forum_context_mode not in {"full_text", "metadata_only"}:
        raise ForumHeroError("turn inputs received an invalid forum context mode")
    if opening_public_action_mode not in {
        "event_initiators_only",
        "diagnostic_open_choice",
    }:
        raise ForumHeroError("opening public-action mode is invalid")
    exposure_rows: dict[int, list[FeedExposure]] = {}
    context_rows: dict[int, dict[str, Any]] = {}
    public_board, board_candidates, public_board_routes = _public_board_snapshot(
        ledger, tick, ecology_calibration_release
    )
    event_initiators = _select_event_initiators(
        selected_indices=selected_indices,
        profiles=profiles,
        ledger=ledger,
        tick=tick,
        scenario_id=scenario_id,
    )
    targeted_assignments = (
        _targeted_mechanism_assignments(
            selected_indices=selected_indices,
            profiles=profiles,
            ledger=ledger,
            tick=tick,
            event_initiators=event_initiators,
        )
        if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
        else {}
    )
    for index in selected_indices:
        profile = profiles[index]
        agent_id = str(profile["agent_id"])
        display_id = str(profile["synthetic_display_id"])
        responded_targets = _responded_target_ids(ledger, display_id)
        assignment = targeted_assignments.get(int(index))
        independent_root_opportunity = False
        assigned_message_id = (
            assignment.get("focal_message_id") if assignment is not None else None
        )
        if assigned_message_id is not None:
            focal = ledger.messages.get(str(assigned_message_id))
            if focal is None or focal.created_tick >= tick or focal.superseded:
                raise ForumHeroError(
                    "targeted mechanism focal message is not time-visible"
                )
        else:
            service_receipts = [
                message
                for message in ledger.messages.values()
                if not message.superseded
                and message.created_tick < tick
                and message.message_id not in responded_targets
                and message.source_agent_display_id == "governance:service_operator"
                and any(
                    own.thread_id == message.thread_id
                    for own in ledger.messages.values()
                    if not own.superseded
                    and own.source_agent_display_id == display_id
                    and own.action == "seek_help"
                )
            ]
            service_focal = max(
                service_receipts,
                key=lambda message: (message.created_tick, message.message_id),
                default=None,
            )
            participated_threads = set(memories[index].participated_thread_ids)
            joined_thread_focal = max(
                (
                    message
                    for message in ledger.messages.values()
                    if not message.superseded
                    and message.created_tick < tick
                    and message.message_id not in responded_targets
                    and message.thread_id in participated_threads
                    and message.source_agent_display_id != display_id
                ),
                key=lambda message: (message.created_tick, message.message_id),
                default=None,
            )
            direct = _latest_direct_message(
                ledger=ledger,
                display_id=display_id,
                tick=tick,
                bridge_enabled=bridge_enabled,
            )
            required_notification = service_focal or direct or joined_thread_focal
            if required_notification is not None:
                focal = required_notification
            elif (
                ecology_calibration_release is not None
                and tick > PILOT_START_TICK
                and _independent_root_opportunity(
                    display_id=display_id,
                    tick=tick,
                    ecology_calibration_release=ecology_calibration_release,
                )
            ):
                # The resident opened the independent composer rather than a
                # discussion.  This does not force speech; read/ignore remain
                # available and no synthetic text is created by the scheduler.
                focal = None
                independent_root_opportunity = True
            else:
                focal = _relevant_message(
                    ledger=ledger,
                    profile=profile,
                    memory=memories[index],
                    tick=tick,
                    bridge_enabled=bridge_enabled,
                    recipient_index=int(index),
                    profiles=profiles,
                    exposure_graph=exposure_graph,
                    public_board_candidates=board_candidates,
                    ecology_calibration_release=ecology_calibration_release,
                )
        current_thread_id = focal.thread_id if focal is not None else None
        parent_message_id = focal.message_id if focal is not None else None
        required: list[ForumMessage] = []
        if focal is not None:
            thread = ledger.threads[focal.thread_id]
            required.append(ledger.messages[thread.root_message_id])
            if focal.message_id != thread.root_message_id:
                required.append(focal)
        seen = {message.message_id for message in required}
        extras = []
        extra_routes: dict[str, tuple[str, float]] = {}
        extra_candidates = (
            _ecology_ordered_board_candidates(
                ledger=ledger,
                candidates=board_candidates,
                display_id=display_id,
                tick=tick,
                profiles=profiles,
                ecology_calibration_release=ecology_calibration_release,
            )
            if ecology_calibration_release is not None
            else list(board_candidates)
        )
        for message in extra_candidates:
            if (
                message.message_id in seen
                or message.message_id in responded_targets
            ):
                continue
            if (
                not bridge_enabled
                and message.source_agent_display_id == "governance:community_bridge"
            ):
                continue
            route = _feed_delivery_route(
                message=message,
                recipient_index=int(index),
                profiles=profiles,
                memory=memories[index],
                tick=tick,
                exposure_graph=exposure_graph,
                bridge_enabled=bridge_enabled,
                current_thread_id=current_thread_id,
                public_board_routes=public_board_routes,
            )
            if route is not None:
                extras.append(message)
                extra_routes[message.message_id] = route
            if len(extras) == 3:
                break
        exposures = [
            _exposure(
                recipient_agent_id=agent_id,
                message=message,
                tick=tick,
                rank=1.0 - ordinal * 0.02,
                source="direct_context",
            )
            for ordinal, message in enumerate(required)
        ]
        exposures.extend(
            _exposure(
                recipient_agent_id=agent_id,
                message=message,
                tick=tick,
                rank=extra_routes[message.message_id][1] - ordinal * 0.01,
                source=extra_routes[message.message_id][0],
            )
            for ordinal, message in enumerate(extras)
        )
        visible_claim = any(
            message.claim_id in ledger.claims and not ledger.claims[str(message.claim_id)].expired
            for message in required + extras
            if message.claim_id is not None
        )
        opening_without_threads = tick == PILOT_START_TICK and not ledger.messages
        actions = (
            ["ignore", "read"]
            if opening_without_threads
            else ["ignore", "read", "post", "seek_help", "report_risk"]
        )
        if tick < 3:
            actions = ["ignore", "read"]
        scenario_context = scenario_context_for_tick(
            tick,
            scenario_id,
            mechanism_protocol,
        )
        if independent_root_opportunity:
            scenario_context = {
                **scenario_context,
                "forum_entry_mode": (
                    "你这次从全局论坛进入独立发帖入口，没有预选回复对象。你可以根据自己的"
                    "当前处境发一个新帖，也可以退出入口继续阅读；不要为了填满版面重复已有帖子。"
                ),
            }
        if (
            mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL
            and focal is not None
            and focal.source_agent_display_id.startswith("governance:")
        ):
            scenario_context = {
                **scenario_context,
                "governance_followup": (
                    "This time-visible governance post is an action with limits, not proof that "
                    "the problem is solved. You may accept, challenge, question or ignore it; "
                    "any public response must directly target the visible message."
                ),
            }
            artifact = focal.governance_artifact
            if (
                artifact is not None
                and artifact.artifact_kind == "evidence_card"
                and artifact.target_claim_id is not None
            ):
                scenario_context = {
                    **scenario_context,
                    "claim_correction_followup": (
                        "This evidence card narrows the linked claim. If you speak, reply to or "
                        "quote it and specify what you accept, reject or still need evidenced."
                    ),
                }
        if (
            mechanism_protocol == OPEN_CHOICE_MECHANISM_PROTOCOL
            and
            focal is not None
            and focal.source_agent_display_id == "governance:service_operator"
            and any(
                own.thread_id == focal.thread_id
                for own in ledger.messages.values()
                if not own.superseded
                and own.source_agent_display_id == display_id
                and own.action == "seek_help"
            )
        ):
            scenario_context = {
                **scenario_context,
                "service_receipt_followup": (
                    "You authored the help request and received this service receipt. State what "
                    "it acknowledges and what remains; a public action must reply directly to it."
                ),
            }
        if index in event_initiators:
            profile_public_actions = set(
                profile.get("forum_tendencies", {}).get(
                    "allowed_public_actions", ()
                )
            )
            actions = ["ignore", "read"] + [
                action
                for action in ("post", "seek_help", "report_risk")
                if action in profile_public_actions
            ]
            scenario_context = {
                **scenario_context,
                "activation_directive": (
                    "预算调度器只给本主体开放了首轮发帖机会；是否公开表达仍由本主体依据画像、"
                    "利益位置和当前注意力决定。没有独特内容时应选择阅读或忽略，不要复述通知。"
                ),
            }
        elif opening_without_threads and opening_public_action_mode == "diagnostic_open_choice":
            profile_public_actions = set(
                profile.get("forum_tendencies", {}).get(
                    "allowed_public_actions", ()
                )
            )
            actions = ["ignore", "read"] + [
                action
                for action in ("post", "seek_help", "report_risk")
                if action in profile_public_actions
            ]
            scenario_context = {
                **scenario_context,
                "activation_directive": (
                    "你和其他面板成员同时看到了上面的学生发帖摘要。是否另开帖子、求助、"
                    "报告风险或继续浏览，完全按自己的经历与处境决定；不要复述整份通知。"
                ),
            }
        if focal is not None:
            actions.extend(["reply", "quote", "repost"])
            if visible_claim:
                actions.extend(["verify", "correct"])
        if assignment is not None:
            actions = list(assignment["allowed_actions"])
            # The disclosed targeted audit already carries a long exact task
            # contract.  Keep the open-choice diversity guidance out of that
            # bounded stress-test payload so the frozen 1,000-character
            # scenario envelope remains fail-closed.
            scenario_context = {
                key: value
                for key, value in scenario_context.items()
                if key != "response_diversity_boundary"
            }
            mechanism_audit = {
                "protocol": TARGETED_CLOSED_LOOP_PROTOCOL,
                "task_id": assignment["task_id"],
                "disclosure": (
                    "动作类别与可见目标由机制压力测试预先指定；公开文字、理由、"
                    "置信度和状态变化仍由当前 LLM 主体生成。"
                ),
                "task_brief": assignment["brief"],
                "required_output": dict(assignment["required_output"]),
            }
            if assignment.get("evidence_packet") is not None:
                mechanism_audit["evidence_packet"] = copy.deepcopy(
                    assignment["evidence_packet"]
                )
            scenario_context = {
                **scenario_context,
                "mechanism_audit": mechanism_audit,
            }
        exposure_rows[index] = exposures
        context_rows[index] = {
            "current_thread_id": current_thread_id,
            "direct_parent_message_id": parent_message_id,
            "quote_message_id": None,
            "allowed_actions": list(dict.fromkeys(actions)),
            "require_reviewed_evidence": True,
            "scenario_context": scenario_context,
            "public_board": public_board,
            "forum_context_mode": forum_context_mode,
        }
    return exposure_rows, context_rows


def _integrate_llm_state(
    *,
    attempts: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    posterior: MutableMapping[int, dict[str, float]],
) -> None:
    index_by_agent = {str(profile["agent_id"]): index for index, profile in enumerate(profiles)}
    for attempt in attempts:
        index = index_by_agent[str(attempt["agent_id"])]
        decision = attempt["validated_decision"]
        state = posterior[index]
        state["trust"] = _clamp(state["trust"] + float(decision["trust_delta"]))
        state["satisfaction"] = _clamp(
            state["satisfaction"] + float(decision["satisfaction_delta"])
        )
        urgency = float(decision["urgency"])
        state["attention"] = _clamp(max(state["attention"], urgency))
        state["concern"] = _clamp(0.60 * state["concern"] + 0.40 * urgency)
        public = decision["action"] not in {"ignore", "read"}
        state["activity_probability"] = _clamp(
            0.65 * state["activity_probability"] + (0.35 if public else 0.08)
        )
        stance = str(decision["stance"])
        if stance == "support":
            state["support"] = _clamp(state["support"] + 0.10 * float(decision["confidence"]))
            state["opposition"] = _clamp(state["opposition"] - 0.05)
        elif stance == "oppose":
            state["opposition"] = _clamp(
                state["opposition"] + 0.10 * float(decision["confidence"])
            )
            state["support"] = _clamp(state["support"] - 0.05)


def _weighted_state(
    *,
    profiles: Sequence[Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, float]],
    scenario_id: str,
) -> dict[str, float]:
    totals = {key: 0.0 for key in STATE_KEYS}
    weight_sum = 0.0
    for index, profile in enumerate(profiles):
        weight = _scenario_design_weight(profile, scenario_id)
        weight_sum += weight
        for key in STATE_KEYS:
            totals[key] += weight * float(posterior[index][key])
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-8):
        raise ForumHeroError("hero state weights are not conserved")
    return {key: round(value, 6) for key, value in totals.items()}


def _stake_scalar(row: Mapping[str, Any], name: str) -> float:
    fields = row.get("fields")
    fields = fields if isinstance(fields, Mapping) else {}
    value_row = fields.get(name)
    value = value_row.get("value") if isinstance(value_row, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return _clamp(float(value))


def _stake_group(row: Mapping[str, Any]) -> str:
    """Map a synthetic scenario position to one mutually exclusive public group.

    The grouping is deliberately about the current governance issue rather
    than demographic inference.  It is used only for aggregate comparison;
    record-level membership is never exported.
    """

    fields = row.get("fields")
    fields = fields if isinstance(fields, Mapping) else {}
    direct = fields.get("directly_affected")
    if isinstance(direct, Mapping) and direct.get("value") is True:
        return "directly_affected"
    if max(
        _stake_scalar(row, "service_channel_dependency"),
        _stake_scalar(row, "resource_need_intensity"),
    ) >= 0.65:
        return "service_dependent"
    if _stake_scalar(row, "substitute_resource_availability") >= 0.65:
        return "alternatives_available"
    return "bystander_or_mixed"


def _stake_conditioned_particle_features(
    *,
    base_features: Mapping[str, np.ndarray],
    profiles: Sequence[Mapping[str, Any]],
    scenario_stakes: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Bind synthetic issue positions to the silent particle transition.

    The mapping is intentionally branch-agnostic: it has no Natural/D input,
    policy bonus, target curve or observed outcome.  It only lets residents
    with different resource needs, time pressure and procedural capacity react
    differently to the same exogenous event.  Governance can then help only
    through an actually activated evidence/service/bridge effect.
    """

    result = {key: np.asarray(value).copy() for key, value in base_features.items()}
    if not scenario_stakes:
        return result
    if len(profiles) != len(result.get("relevance", ())):
        raise ForumHeroError("scenario-stake feature population size drifted")
    expected_ids = {str(profile["agent_id"]) for profile in profiles}
    if set(scenario_stakes) != expected_ids:
        raise ForumHeroError("scenario-stake features require every parent Agent")

    def column(name: str) -> np.ndarray:
        return np.asarray(
            [
                _stake_scalar(scenario_stakes[str(profile["agent_id"])], name)
                for profile in profiles
            ],
            dtype=np.float32,
        )

    directly_affected = np.asarray(
        [
            1.0
            if (
                isinstance(
                    scenario_stakes[str(profile["agent_id"])].get("fields"),
                    Mapping,
                )
                and isinstance(
                    scenario_stakes[str(profile["agent_id"])]["fields"].get(
                        "directly_affected"
                    ),
                    Mapping,
                )
                and scenario_stakes[str(profile["agent_id"])]["fields"][
                    "directly_affected"
                ].get("value")
                is True
            )
            else 0.0
            for profile in profiles
        ],
        dtype=np.float32,
    )
    resource_need = column("resource_need_intensity")
    time_urgency = column("time_urgency")
    service_dependency = column("service_channel_dependency")
    information_access = column("information_access_capacity")
    appeal_cost = column("appeal_cost")
    alternatives = column("substitute_resource_availability")
    usage_frequency = column("usage_frequency")
    bridge_responsibility = column("bridge_organizing_care_responsibility")

    result.update(
        {
            "stake_exposure": np.clip(
                0.30 * directly_affected
                + 0.22 * resource_need
                + 0.16 * time_urgency
                + 0.12 * service_dependency
                + 0.10 * usage_frequency
                + 0.10 * appeal_cost
                - 0.16 * alternatives,
                0.0,
                1.0,
            ).astype(np.float32),
            "service_pressure": np.clip(
                0.45 * service_dependency
                + 0.30 * resource_need
                + 0.25 * time_urgency,
                0.0,
                1.0,
            ).astype(np.float32),
            "appeal_friction": appeal_cost.astype(np.float32),
            "alternative_buffer": alternatives.astype(np.float32),
            "voice_capacity": np.clip(
                0.35 + 0.35 * information_access + 0.30 * bridge_responsibility,
                0.0,
                1.0,
            ).astype(np.float32),
            "verification_reach": np.clip(
                0.40 + 0.60 * information_access,
                0.0,
                1.0,
            ).astype(np.float32),
            "service_reach": np.clip(
                0.45 + 0.55 * service_dependency,
                0.0,
                1.0,
            ).astype(np.float32),
            "bridge_reach": np.clip(
                0.35
                + 0.35 * bridge_responsibility
                + 0.30 * (1.0 - information_access),
                0.0,
                1.0,
            ).astype(np.float32),
        }
    )
    return result


def _population_heterogeneity_frame(
    *,
    profiles: Sequence[Mapping[str, Any]],
    posterior: Mapping[int, Mapping[str, float]],
    scenario_stakes: Mapping[str, Mapping[str, Any]],
    branch_state: Any,
    scenario_id: str,
    branch: str,
    tick: int,
) -> dict[str, Any]:
    """Return privacy-safe heterogeneity over the 1,000 parent Agents.

    Parent design weights define the population.  The ten particles attached
    to each parent contribute only one within-Agent uncertainty statistic and
    never become additional people or stake-group observations.
    """

    weights = [_scenario_design_weight(profile, scenario_id) for profile in profiles]
    if any(value <= 0.0 for value in weights) or not math.isclose(
        sum(weights), 1.0, abs_tol=1e-8
    ):
        raise ForumHeroError("heterogeneity frame requires conserved positive weights")

    weighted_parent_state_std: dict[str, float] = {}
    for key in STATE_KEYS:
        mean = sum(
            weights[index] * float(posterior[index][key])
            for index in range(len(profiles))
        )
        variance = sum(
            weights[index] * (float(posterior[index][key]) - mean) ** 2
            for index in range(len(profiles))
        )
        weighted_parent_state_std[key] = round(math.sqrt(max(0.0, variance)), 6)

    stance_distribution = {"support": 0.0, "oppose": 0.0, "uncertain": 0.0}
    for index, weight in enumerate(weights):
        support = _clamp(float(posterior[index]["support"]))
        oppose = _clamp(float(posterior[index]["opposition"]))
        uncertain = max(0.05, 1.0 - min(1.0, support + oppose))
        total = support + oppose + uncertain
        stance_distribution["support"] += weight * support / total
        stance_distribution["oppose"] += weight * oppose / total
        stance_distribution["uncertain"] += weight * uncertain / total
    stance_distribution = {
        key: round(value, 6) for key, value in stance_distribution.items()
    }
    stance_total = sum(stance_distribution.values())
    stance_distribution["uncertain"] = round(
        stance_distribution["uncertain"] + (1.0 - stance_total), 6
    )

    group_members: dict[str, list[int]] = defaultdict(list)
    for index, profile in enumerate(profiles):
        agent_id = str(profile["agent_id"])
        group = (
            _stake_group(scenario_stakes[agent_id])
            if agent_id in scenario_stakes
            else "stake_release_unavailable"
        )
        group_members[group].append(index)
    group_rows: dict[str, dict[str, Any]] = {}
    eligible_group_means: dict[str, Mapping[str, float]] = {}
    for group, indices in sorted(group_members.items()):
        group_weight = sum(weights[index] for index in indices)
        means = {
            key: round(
                sum(
                    weights[index] * float(posterior[index][key])
                    for index in indices
                )
                / group_weight,
                6,
            )
            for key in STATE_KEYS
        }
        group_rows[group] = {
            "parent_agent_count": len(indices),
            "population_weight": round(group_weight, 6),
            "state_means": means,
        }
        if len(indices) >= 20 and group_weight >= 0.01:
            eligible_group_means[group] = means

    maximum_gap = 0.0
    maximum_gap_state = None
    maximum_gap_groups: list[str] = []
    group_names = sorted(eligible_group_means)
    for left_position, left in enumerate(group_names):
        for right in group_names[left_position + 1 :]:
            for key in STATE_KEYS:
                gap = abs(
                    float(eligible_group_means[left][key])
                    - float(eligible_group_means[right][key])
                )
                if gap > maximum_gap:
                    maximum_gap = gap
                    maximum_gap_state = key
                    maximum_gap_groups = [left, right]

    parent_variances = branch_state.ensemble.parent_variances()
    parent_uncertainty = parent_variances.mean(axis=1) ** 0.5
    mean_within_agent_particle_std = sum(
        weights[index] * float(parent_uncertainty[index])
        for index in range(len(profiles))
    )
    return {
        "branch": branch,
        "tick": tick,
        "parent_agent_count": len(profiles),
        "evidence_subject_count": len(profiles),
        "particle_count": int(branch_state.ensemble.particle_count),
        "particle_evidence_subject_count": 0,
        "stance_distribution": stance_distribution,
        "weighted_parent_state_std": weighted_parent_state_std,
        "stake_groups": group_rows,
        "maximum_between_stake_group_gap": round(maximum_gap, 6),
        "maximum_gap_state": maximum_gap_state,
        "maximum_gap_groups": maximum_gap_groups,
        "mean_within_agent_particle_std": round(
            mean_within_agent_particle_std, 6
        ),
        "record_level_agent_ids_public": False,
        "record_level_particle_rows_public": False,
    }


def _ledger_metrics(ledger: ForumLedger, tick: int) -> dict[str, Any]:
    messages = [message for message in ledger.messages.values() if not message.superseded]
    current = [message for message in messages if message.created_tick == tick]
    actions = Counter(message.action for message in messages)
    current_actions = Counter(message.action for message in current)
    stances = Counter(message.stance for message in messages)
    interactions = list(ledger.interactions.values())
    current_interactions = [
        interaction
        for interaction in interactions
        if interaction.created_tick == tick
    ]
    interaction_counts = Counter(interaction.kind for interaction in interactions)
    current_interaction_counts = Counter(
        interaction.kind for interaction in current_interactions
    )
    governance_evidence_corrections = sum(
        message.governance_artifact is not None
        and message.governance_artifact.artifact_kind == "evidence_card"
        and message.governance_artifact.target_claim_id is not None
        for message in messages
    )
    return {
        "tick": tick,
        "thread_count": len(ledger.threads),
        "message_count": len(messages),
        "messages_this_tick": len(current),
        "claim_count": len(ledger.claims),
        "contested_claim_count": sum(
            claim.status == "contested" for claim in ledger.claims.values()
        ),
        "corrected_or_verified_claim_count": sum(
            claim.status in {"corrected", "verified"} for claim in ledger.claims.values()
        ),
        "help_request_count": actions["seek_help"],
        "risk_report_count": actions["report_risk"],
        "correction_count": actions["correct"] + governance_evidence_corrections,
        "resident_correction_count": actions["correct"],
        "governance_evidence_correction_count": governance_evidence_corrections,
        "governance_message_count": actions["governance_message"],
        "reply_count": actions["reply"],
        "this_tick_actions": dict(sorted(current_actions.items())),
        "interaction_count": len(interactions),
        "interactions_this_tick": len(current_interactions),
        "interaction_counts": dict(sorted(interaction_counts.items())),
        "this_tick_interactions": dict(sorted(current_interaction_counts.items())),
        "speaker_stance_distribution": dict(sorted(stances.items())),
        "unique_public_speakers": len({message.source_agent_display_id for message in messages}),
    }


def _governance_visible_posterior(
    population_recovery: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Project the current PPS estimate into the actor-visible state space."""

    if not isinstance(population_recovery, Mapping):
        return {}
    action = population_recovery.get("action_greg")
    stance = population_recovery.get("stance_greg")
    action_share = action.get("greg_share") if isinstance(action, Mapping) else None
    stance_share = stance.get("greg_share") if isinstance(stance, Mapping) else None
    if not isinstance(action_share, Mapping) or not isinstance(stance_share, Mapping):
        return {}
    required = {
        "public": action_share.get("public"),
        "support": stance_share.get("support"),
        "opposition": stance_share.get("oppose"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in required.values()
    ):
        return {}
    return {
        "activity_probability": _clamp(float(required["public"])),
        "support": _clamp(float(required["support"])),
        "opposition": _clamp(float(required["opposition"])),
    }


def _actor_views(
    ledger: ForumLedger,
    tick: int,
    *,
    scenario_id: str = PILOT_SCENARIO,
    branch: str = "D",
    mechanism_protocol: str = OPEN_CHOICE_MECHANISM_PROTOCOL,
    population_recovery: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = _ledger_metrics(ledger, tick)
    message_count = max(1, int(metrics["message_count"]))
    claim_count = max(1, int(metrics["claim_count"]))
    private_messages = [
        message
        for message in ledger.private_messages.values()
        if message.created_tick <= tick and not message.superseded
    ]
    private_spillovers = sum(
        message.private_hearsay_origin
        for message in ledger.messages.values()
        if message.created_tick <= tick and not message.superseded
    )
    private_aggregate = {
        "private_message_count": len(private_messages),
        "active_group_count": sum(
            conversation.conversation_type == "group"
            and conversation.status == "active"
            for conversation in ledger.private_conversations.values()
        ),
        "private_to_public_spillover_count": private_spillovers,
        "contains_private_text": False,
        "contains_relationship_edges": False,
    }
    observations = {
        "governance_authority": {
            "discussion_volume": min(1.0, message_count / 120.0),
            "risk_reports": min(1.0, metrics["risk_report_count"] / 12.0),
            "pps_uncertainty": min(1.0, metrics["contested_claim_count"] / claim_count),
            "claim_contested_share": min(1.0, metrics["contested_claim_count"] / claim_count),
        },
        "service_operator": {
            "help_queue": min(1.0, metrics["help_request_count"] / 12.0),
            "sla_pressure": min(1.0, metrics["help_request_count"] / 16.0),
            "capacity_remaining": max(0.0, 1.0 - metrics["help_request_count"] / 24.0),
            "service_strain": min(1.0, metrics["help_request_count"] / 20.0),
        },
        "community_bridge": {
            "local_discussion": min(1.0, message_count / 100.0),
            "cross_group_reach": min(1.0, metrics["unique_public_speakers"] / 64.0),
            "correction_delay": min(1.0, metrics["contested_claim_count"] / claim_count),
            "participation_gap": max(0.0, 1.0 - metrics["unique_public_speakers"] / 64.0),
        },
    }
    visible_posterior = _governance_visible_posterior(population_recovery)
    if visible_posterior:
        action = population_recovery["action_greg"]
        stance = population_recovery["stance_greg"]
        observations["governance_authority"] = {
            **observations["governance_authority"],
            "pps_public_action_share": visible_posterior[
                "activity_probability"
            ],
            "pps_support_share": visible_posterior["support"],
            "pps_oppose_share": visible_posterior["opposition"],
            "pps_uncertain_share": _clamp(
                float(stance["greg_share"]["uncertain"])
            ),
            "pps_kish_ess": round(
                min(
                    float(action.get("kish_ess", 0.0)),
                    float(stance.get("kish_ess", 0.0)),
                ),
                6,
            ),
        }
    scenario = scenario_context_for_tick(
        tick, scenario_id, mechanism_protocol
    )
    policy_world: dict[str, Any] | None = None
    if scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO and branch == "A":
        policy_world = {
            "world": "criteria_explanation_only",
            "required_scope": (
                "Publish only a bounded criteria card: the three stated need categories, "
                "how category and same-category ordering are applied, required evidence, "
                "and what remains unresolved. Do not create a review service, promise an "
                "outcome, add beds, or claim the dispute is solved."
            ),
        }
    elif scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO and branch == "D":
        policy_world = {
            "world": "criteria_plus_review_service_and_public_QA",
            "required_scope": (
                "Coordinate a sustained response: a bounded criteria card; a review ticket "
                "with responsible role, status and 48-hour response window; and public Q&A "
                "or cross-group outreach. Acceptance is not guaranteed and no extra beds "
                "or individual outcome may be invented."
            ),
        }
    elif scenario_id == CENTURY_GYM_SCENARIO and branch == "D":
        policy_world = {
            "world": "booking_evidence_ticket_and_cross_sport_outreach",
            "required_scope": (
                "Use visible posts only. Coordinate: (1) an evidence card separating "
                "protected slots, released holds, no-shows and unknowns; (2) a reproducible-"
                "case ticket with role, status and response window; and (3) cross-sport, "
                "venue and time outreach. A ticket is not a fix. Never invent logs, "
                "punishment, capacity, repair or relationship allocation."
            ),
        }
    contexts = {
        actor: {
            "visible_message_count": metrics["message_count"],
            "visible_thread_count": metrics["thread_count"],
            "visible_claim_count": metrics["claim_count"],
            "latest_visible_tick": tick,
            "correction_count": metrics["correction_count"],
            "help_request_count": metrics["help_request_count"],
            "scenario_event_brief": scenario["event_brief"],
            "current_known_boundary": scenario["known_at_tick"],
            "policy_world": copy.deepcopy(policy_world),
            "privacy_safe_private_channel": dict(private_aggregate),
        }
        for actor in ACTOR_IDS
    }
    targets = []
    for claim in sorted(
        ledger.claims.values(), key=lambda item: (-item.last_seen_tick, item.claim_id)
    )[:6]:
        targets.append(
            {
                "target_id": claim.claim_id,
                "target_kind": "claim",
                "linked_thread_id": claim.thread_id,
                "summary": claim.claim_summary[:160],
                "last_visible_tick": min(tick, claim.last_seen_tick),
            }
        )
    for thread in sorted(
        ledger.threads.values(),
        key=lambda item: (-(item.last_active_tick or item.created_tick), item.thread_id),
    )[: max(0, 12 - len(targets))]:
        root = ledger.messages[thread.root_message_id]
        targets.append(
            {
                "target_id": thread.thread_id,
                "target_kind": "thread",
                "linked_thread_id": thread.thread_id,
                "summary": root.visible_text[:160],
                "last_visible_tick": min(tick, thread.last_active_tick or thread.created_tick),
            }
        )
    catalogs = {actor: copy.deepcopy(targets) for actor in ACTOR_IDS}
    help_messages = sorted(
        (
            message
            for message in ledger.messages.values()
            if not message.superseded and message.action == "seek_help"
        ),
        key=lambda message: (-message.created_tick, message.message_id),
    )
    help_targets = []
    seen_help_threads: set[str] = set()
    for help_message in help_messages:
        if help_message.thread_id in seen_help_threads:
            continue
        seen_help_threads.add(help_message.thread_id)
        thread = ledger.threads[help_message.thread_id]
        help_targets.append(
            {
                "target_id": thread.thread_id,
                "target_kind": "thread",
                "linked_thread_id": thread.thread_id,
                "summary": help_message.visible_text[:160],
                "last_visible_tick": min(
                    tick, thread.last_active_tick or thread.created_tick
                ),
            }
        )
        if len(help_targets) == 6:
            break
    if help_targets:
        generic_targets = [
            row
            for row in catalogs["service_operator"]
            if row["target_id"] not in seen_help_threads
        ]
        catalogs["service_operator"] = help_targets
        contexts["service_operator"] = {
            **contexts["service_operator"],
            "pending_help_threads": len(help_targets),
            "service_contract": (
                "Registration is acknowledgement only. Target one visible help thread; "
                "state receipt scope, response window and responsible role; never claim "
                "resolution."
            ),
        }
    if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL and tick == 3:
        targeted_help_messages = sorted(
            (
                message
                for message in ledger.messages.values()
                if not message.superseded and message.action == "seek_help"
            ),
            key=lambda message: (message.created_tick, message.message_id),
        )
        if targeted_help_messages:
            help_message = targeted_help_messages[0]
            help_thread = ledger.threads[help_message.thread_id]
            catalogs["service_operator"] = [
                {
                    "target_id": help_thread.thread_id,
                    "target_kind": "thread",
                    "linked_thread_id": help_thread.thread_id,
                    "summary": help_message.visible_text[:160],
                    "last_visible_tick": min(
                        tick,
                        help_thread.last_active_tick or help_thread.created_tick,
                    ),
                }
            ]
            contexts["service_operator"] = {
                **contexts["service_operator"],
                "mechanism_audit": (
                    "This stress test contains one visible help request. If service action "
                    "is warranted, target that thread and state only queue/receipt limits."
                ),
            }
    return observations, contexts, catalogs


def _mechanism_chains(ledger: ForumLedger) -> dict[str, list[dict[str, Any]]]:
    correction_chains = []
    help_chains = []
    governance_uptake = []
    by_thread: dict[str, list[ForumMessage]] = defaultdict(list)
    for message in ledger.messages.values():
        if not message.superseded:
            by_thread[message.thread_id].append(message)
    for messages in by_thread.values():
        messages.sort(key=lambda item: (item.created_tick, item.message_id))
    for message in ledger.messages.values():
        if message.superseded:
            continue
        governance_correction_target = (
            message.governance_artifact.target_claim_id
            if message.governance_artifact is not None
            and message.governance_artifact.artifact_kind == "evidence_card"
            else None
        )
        correction_target = (
            message.correction_target_claim_id or governance_correction_target
        )
        if correction_target is not None:
            claim = ledger.claims.get(correction_target)
            if claim is not None:
                later = [
                    row for row in by_thread[message.thread_id]
                    if row.created_tick > message.created_tick
                    and (
                        row.parent_message_id == message.message_id
                        or row.quote_message_id == message.message_id
                    )
                ]
                accepted = [
                    row
                    for row in later
                    if row.claim_operation == "support" or row.stance == "support"
                ]
                rejected = [
                    row
                    for row in later
                    if row.claim_operation == "challenge" or row.stance == "oppose"
                ]
                correction_chains.append(
                    {
                        "claim_id": claim.claim_id,
                        "correction_source": (
                            "governance_evidence_card"
                            if governance_correction_target is not None
                            else "resident_llm"
                        ),
                        "claim_created_tick": claim.first_seen_tick,
                        "correction_message_id": message.message_id,
                        "correction_tick": message.created_tick,
                        "acceptance_response_ids": [
                            row.message_id for row in accepted
                        ],
                        "rejection_response_ids": [
                            row.message_id for row in rejected
                        ],
                        "later_response_ids": [
                            row.message_id for row in later[:5]
                        ],
                        "complete": bool(accepted) and bool(rejected),
                    }
                )
        if message.action == "seek_help":
            later = [
                row for row in by_thread[message.thread_id]
                if row.created_tick > message.created_tick
                or (
                    row.created_tick == message.created_tick
                    and row.source_agent_display_id.startswith("governance:")
                )
            ]
            service = next(
                (
                    row for row in later
                    if row.source_agent_display_id == "governance:service_operator"
                ),
                None,
            )
            feedback = next(
                (
                    row for row in later
                    if service is not None
                    and row.created_tick > service.created_tick
                    and row.source_agent_display_id
                    == message.source_agent_display_id
                    and (
                        row.parent_message_id == service.message_id
                        or row.quote_message_id == service.message_id
                    )
                ),
                None,
            )
            help_chains.append(
                {
                    "help_message_id": message.message_id,
                    "service_receipt_id": service.message_id if service else None,
                    "follow_up_id": feedback.message_id if feedback else None,
                    "complete": service is not None and feedback is not None,
                }
            )
        if message.source_agent_display_id.startswith("governance:"):
            later = [
                row for row in by_thread[message.thread_id]
                if row.created_tick > message.created_tick
                and not row.source_agent_display_id.startswith("governance:")
                and (
                    row.parent_message_id == message.message_id
                    or row.quote_message_id == message.message_id
                )
            ]
            governance_uptake.append(
                {
                    "governance_message_id": message.message_id,
                    "resident_response_ids": [row.message_id for row in later[:5]],
                    "complete": bool(later),
                }
            )
    return {
        "correction_chains": correction_chains,
        "help_service_feedback_chains": help_chains,
        "governance_uptake_chains": governance_uptake,
    }


def _governance_uptake_map(ledger: ForumLedger) -> dict[str, Any]:
    """Explain how each public governance object was received in the forum.

    Only time-ordered public messages and anonymous interaction totals are used.
    Same-thread activity is reported separately from strict direct uptake so a
    later, unrelated comment cannot be presented as a governance response.
    """

    resident_messages_by_thread: dict[str, list[ForumMessage]] = defaultdict(list)
    for message in ledger.messages.values():
        if (
            not message.superseded
            and not message.source_agent_display_id.startswith("governance:")
        ):
            resident_messages_by_thread[message.thread_id].append(message)
    for messages in resident_messages_by_thread.values():
        messages.sort(key=lambda item: (item.created_tick, item.message_id))

    entries: list[dict[str, Any]] = []
    all_latencies: list[int] = []
    artifact_totals: Counter[str] = Counter()
    artifact_with_uptake: Counter[str] = Counter()
    direct_stances: Counter[str] = Counter()
    direct_operations: Counter[str] = Counter()
    for message in sorted(
        ledger.messages.values(), key=lambda item: (item.created_tick, item.message_id)
    ):
        if message.superseded or not message.source_agent_display_id.startswith(
            "governance:"
        ):
            continue
        later = [
            row
            for row in resident_messages_by_thread.get(message.thread_id, [])
            if row.created_tick > message.created_tick
        ]
        direct = [
            row
            for row in later
            if row.parent_message_id == message.message_id
            or row.quote_message_id == message.message_id
        ]
        direct_ids = {row.message_id for row in direct}
        same_thread_other = [row for row in later if row.message_id not in direct_ids]
        stance_counts = Counter(row.stance for row in direct)
        operation_counts = Counter(row.claim_operation for row in direct)
        direct_stances.update(stance_counts)
        direct_operations.update(operation_counts)
        latencies = [row.created_tick - message.created_tick for row in direct]
        all_latencies.extend(latencies)
        artifact = message.governance_artifact
        artifact_kind = artifact.artifact_kind if artifact is not None else "message_only"
        artifact_totals[artifact_kind] += 1
        if direct:
            artifact_with_uptake[artifact_kind] += 1
        support_count = int(stance_counts["support"] + operation_counts["support"])
        challenge_count = int(stance_counts["oppose"] + operation_counts["challenge"])
        if support_count and challenge_count:
            uptake_status = "mixed_direct_uptake"
        elif challenge_count:
            uptake_status = "direct_challenge"
        elif support_count:
            uptake_status = "direct_support"
        elif direct:
            uptake_status = "direct_response_without_clear_alignment"
        else:
            uptake_status = "no_direct_uptake"
        entries.append(
            {
                "governance_message_id": message.message_id,
                "thread_id": message.thread_id,
                "role": message.source_agent_display_id.removeprefix("governance:"),
                "created_tick": message.created_tick,
                "visible_text": message.visible_text,
                "artifact": artifact.public_view() if artifact is not None else None,
                "uptake_status": uptake_status,
                "direct_response_count": len(direct),
                "direct_response_latency_ticks": latencies,
                "direct_response_stances": dict(sorted(stance_counts.items())),
                "direct_response_claim_operations": dict(
                    sorted(operation_counts.items())
                ),
                "direct_response_examples": [
                    {
                        "message_id": row.message_id,
                        "created_tick": row.created_tick,
                        "link_kind": (
                            "reply"
                            if row.parent_message_id == message.message_id
                            else "quote"
                        ),
                        "stance": row.stance,
                        "claim_operation": row.claim_operation,
                        "visible_text": row.visible_text,
                    }
                    for row in direct[:3]
                ],
                "same_thread_later_message_count_excluding_direct": len(
                    same_thread_other
                ),
                "governance_message_interactions": ledger.interaction_counts(
                    message.message_id
                ),
                "strict_uptake_rule": "later_direct_reply_or_quote_only",
            }
        )

    ordered_latencies = sorted(all_latencies)
    median_latency = None
    if ordered_latencies:
        middle = len(ordered_latencies) // 2
        if len(ordered_latencies) % 2:
            median_latency = float(ordered_latencies[middle])
        else:
            median_latency = round(
                (ordered_latencies[middle - 1] + ordered_latencies[middle]) / 2.0,
                6,
            )
    message_count = len(entries)
    uptake_count = sum(row["direct_response_count"] > 0 for row in entries)
    return {
        "schema_version": GOVERNANCE_UPTAKE_MAP_SCHEMA_VERSION,
        "summary": {
            "governance_message_count": message_count,
            "messages_with_direct_uptake": uptake_count,
            "direct_uptake_rate": round(
                uptake_count / message_count if message_count else 0.0, 6
            ),
            "direct_resident_response_count": len(all_latencies),
            "median_direct_response_latency_ticks": median_latency,
            "direct_response_stances": dict(sorted(direct_stances.items())),
            "direct_response_claim_operations": dict(
                sorted(direct_operations.items())
            ),
            "artifact_message_counts": dict(sorted(artifact_totals.items())),
            "artifact_messages_with_direct_uptake": dict(
                sorted(artifact_with_uptake.items())
            ),
            "unanswered_governance_message_count": message_count - uptake_count,
        },
        "entries": entries,
        "decision_use": (
            "Inspect unanswered, challenged and slowly answered governance objects before "
            "choosing whether to clarify evidence, close a service loop or change outreach."
        ),
        "causal_boundary": (
            "This map proves public temporal linkage and direct forum uptake only; it does "
            "not by itself identify a causal policy effect on population state."
        ),
        "hidden_truth_used": False,
    }


def _public_branch_result(
    *,
    branch: str,
    ledger: ForumLedger,
    timeline: Sequence[Mapping[str, Any]],
    activated_indices: set[int],
    profiles: Sequence[Mapping[str, Any]],
    usage: Mapping[str, int],
    actor_rounds: Sequence[Mapping[str, Any]],
    feed_delivery_sources: Mapping[str, int],
    feed_role_matrix: Mapping[str, int],
    resident_turn_schema_version: str,
) -> dict[str, Any]:
    parallel_interactions = resident_turn_schema_version != FORUM_AGENT_TURN_SCHEMA_VERSION
    private_turn = resident_turn_schema_version in {
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }
    public_messages = []
    for message in sorted(
        ledger.messages.values(), key=lambda item: (item.created_tick, item.message_id)
    ):
        if message.superseded:
            continue
        public_message = message.public_view()
        if parallel_interactions:
            public_message["interaction_counts"] = ledger.interaction_counts(
                message.message_id
            )
            public_message["interaction_counts_by_tick"] = (
                ledger.interaction_counts_by_tick(message.message_id)
            )
        public_messages.append(public_message)
    public_threads = [
        {
            "thread_id": thread.thread_id,
            "board": thread.board,
            "root_message_id": thread.root_message_id,
            "scenario_id": thread.scenario_id,
            "branch": thread.branch,
            "created_tick": thread.created_tick,
            "status": thread.status,
            "topic": thread.topic,
            "need": thread.need,
            "claim_ids": list(thread.claim_ids),
            "participant_count": thread.participant_count,
            "reply_count": thread.reply_count,
            "last_active_tick": thread.last_active_tick,
            "provenance_kind": thread.provenance_kind,
        }
        for thread in sorted(ledger.threads.values(), key=lambda item: item.thread_id)
    ]
    public_claims = [
        {
            "claim_id": claim.claim_id,
            "thread_id": claim.thread_id,
            "claim_summary": claim.claim_summary,
            "status": claim.status,
            "first_seen_tick": claim.first_seen_tick,
            "last_seen_tick": claim.last_seen_tick,
            "parent_claim_id": claim.parent_claim_id,
            "correction_target_claim_id": claim.correction_target_claim_id,
        }
        for claim in sorted(ledger.claims.values(), key=lambda item: item.claim_id)
    ]
    public_governance_artifacts = [
        message.governance_artifact.public_view()
        for message in sorted(
            ledger.messages.values(), key=lambda item: item.message_id
        )
        if not message.superseded and message.governance_artifact is not None
    ]
    public_speaker_ids = {
        message.source_agent_display_id
        for message in ledger.messages.values()
        if not message.superseded
        and not message.source_agent_display_id.startswith("governance:")
    }
    public_display_profiles = []
    for profile in profiles:
        display_id = str(profile["synthetic_display_id"])
        if display_id not in public_speaker_ids:
            continue
        stable = dict(profile.get("stable_persona") or {})
        episode = dict(profile.get("episodic_evidence") or {})
        language = dict(profile.get("language_style") or {})
        public_display_profiles.append(
            {
                "display_id": display_id,
                "macro_role": str(profile.get("primary_macro_role") or "unknown"),
                "micro_role": str(
                    profile.get("micro_interaction_role") or "contextual_participant"
                ),
                "stable_traits": [
                    str(stable.get("interaction_role") or "contextual_participant"),
                    str(stable.get("uncertainty_handling") or "context-dependent"),
                    str(language.get("summary") or language.get("language_form") or "natural"),
                ],
                "historical_episode_focus": str(
                    episode.get("episode_focus") or profile.get("episode_focus") or ""
                )[:180],
                "historical_episode_usage": str(
                    episode.get("usage_rule")
                    or "historical evidence only; not a permanent biography"
                ),
                "topic_portfolio": list(profile.get("topic_portfolio") or [])[:4],
                "need_portfolio": list(profile.get("need_portfolio") or [])[:4],
                "language_style": language,
                "source_profile_v2_bound": bool(
                    profile.get("pilot_bridge", {}).get(
                        "full_corpus_profile_v2_claimed"
                    )
                ),
            }
        )
    mechanism_chains = _mechanism_chains(ledger)
    governance_uptake_map = _governance_uptake_map(ledger)
    result = {
        "branch": branch,
        "timeline": list(timeline),
        "threads": public_threads,
        "messages": public_messages,
        "claims": public_claims,
        "governance_artifacts": public_governance_artifacts,
        "public_display_profiles": public_display_profiles,
        "mechanism_chains": mechanism_chains,
        "governance_uptake_map": governance_uptake_map,
        "unique_activated_agents": len(activated_indices),
        "activated_macro_roles": sorted(
            {str(profiles[index]["primary_macro_role"]) for index in activated_indices}
        ),
        "usage": dict(usage),
        "governance_rounds": list(actor_rounds),
        "feed_recommendation": {
            "mechanism": "global_hot_top10_plus_latest_v1",
            "graph_controlled": False,
            "ranking_identical_for_all_residents_at_same_tick": True,
            "profile_affects_response_not_public_ranking": True,
            "governance_posts_use_ordinary_board_ranking": True,
            "delivery_sources": dict(sorted(feed_delivery_sources.items())),
            "macro_role_matrix": [
                {
                    "source_to_recipient": key,
                    "exposure_count": int(value),
                }
                for key, value in sorted(feed_role_matrix.items())
            ],
            "record_level_edges_public": False,
        },
        "emulator_public_messages": 0,
    }
    if parallel_interactions:
        interactions_by_kind = Counter(
            interaction.kind for interaction in ledger.interactions.values()
        )
        engagements_by_turn: dict[tuple[str, int], set[str]] = defaultdict(set)
        for interaction in ledger.interactions.values():
            engagements_by_turn[
                (interaction.source_agent_display_id, interaction.created_tick)
            ].add(interaction.kind)
        speech_turns = {
            (message.source_agent_display_id, message.created_tick)
            for message in ledger.messages.values()
            if not message.superseded
            and not message.source_agent_display_id.startswith("governance:")
        }
        result["parallel_interactions"] = {
            "schema_version": resident_turn_schema_version,
            "speech_messages_per_agent_tick": 1,
            "independent_engagements_per_turn": 6,
            "kinds": ["like", "repost", "report"],
            "public_interaction_count": len(ledger.interactions),
            "counts_by_kind": {
                kind: int(interactions_by_kind[kind])
                for kind in sorted(INTERACTION_KINDS)
            },
            "engaged_turn_count": len(engagements_by_turn),
            "compound_speech_and_engagement_turn_count": sum(
                turn in speech_turns for turn in engagements_by_turn
            ),
            "multi_kind_engagement_turn_count": sum(
                len(kinds) > 1 for kinds in engagements_by_turn.values()
            ),
            "temporal_aggregation": "anonymous_message_deltas_by_tick",
            "record_level_interaction_identities_public": False,
        }
    if private_turn:
        result["private_channel"] = {
            "schema_version": resident_turn_schema_version,
            "conversation_count": len(ledger.private_conversations),
            "direct_conversation_count": sum(
                value.conversation_type == "direct"
                for value in ledger.private_conversations.values()
            ),
            "group_count": sum(
                value.conversation_type == "group"
                for value in ledger.private_conversations.values()
            ),
            "private_message_count": len(ledger.private_messages),
            "private_reaction_count": len(ledger.private_reactions),
            "private_claim_count": len(ledger.private_claims),
            "private_to_public_spillover_count": sum(
                message.private_hearsay_origin
                for message in ledger.messages.values()
                if not message.superseded
            ),
            "contains_private_text": False,
            "contains_private_agent_ids": False,
            "contains_relationship_edges": False,
        }
        result["public_anonymity"] = {
            "enabled": resident_turn_schema_version
            in {
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
            },
            "anonymous_public_message_count": sum(
                getattr(message, "author_visibility", "named") == "anonymous"
                for message in ledger.messages.values()
                if not message.superseded
            ),
            "anonymous_is_same_forum_author_face": True,
        }
    return result


def run_forum_hero_experiment(
    *,
    population_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    scenario_id: str = PILOT_SCENARIO,
    seed: int = PILOT_SEED,
    start_tick: int = PILOT_START_TICK,
    end_tick: int = PILOT_END_TICK,
    branches: Sequence[str] = PILOT_BRANCHES,
    resident_budget_by_tick: Mapping[int, int] | None = None,
    reviewed_trace_release_sha256: str | None = None,
    initial_memory_summaries: Mapping[str, str] | None = None,
    mechanism_protocol: str = OPEN_CHOICE_MECHANISM_PROTOCOL,
    ablation_mode: str = FULL_SYSTEM_ABLATION_MODE,
    initial_snapshot: Mapping[str, Any] | None = None,
    snapshot_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    tick_snapshot_callback: Callable[[str, int, Mapping[str, Any]], None] | None = None,
    public_tick_callback: Callable[[str, int, Mapping[str, Any]], None] | None = None,
    interaction_activation_enabled: bool = True,
    activation_selection_salt: str | None = None,
    governance_decision_ticks: Sequence[int] | None = None,
    governance_action_overrides: Mapping[str, Sequence[str]] | None = None,
    resident_turn_schema_version: str = FORUM_AGENT_TURN_SCHEMA_VERSION,
    require_scenario_entry_post: bool = False,
    relationship_release: Mapping[str, Any] | None = None,
    scenario_stake_release: Mapping[str, Any] | None = None,
    relationship_attention_enabled: bool = False,
    private_actions_enabled: bool = False,
    public_anonymity_enabled: bool = False,
    ecology_calibration_release: Mapping[str, Any] | None = None,
    activation_policy: str = LEGACY_ACTIVATION_POLICY,
    resident_model_view_condition: str = V5_FULL_NATURAL_PROFILE,
    anchor_indices_override: Sequence[int] | None = None,
    opening_public_action_mode: str = "event_initiators_only",
) -> dict[str, Any]:
    """Run the paired forum experiment with no internally generated speech."""

    if scenario_id not in HERO_RUNTIME_SCENARIOS:
        raise ForumHeroError("hero experiment scenario is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ForumHeroError("hero experiment seed is invalid")
    if not 0 <= start_tick <= end_tick <= PILOT_END_TICK:
        raise ForumHeroError("hero experiment tick range is invalid")
    if resident_source is None:
        raise ForumHeroError("hero experiment requires a formal resident semantic source")
    if mechanism_protocol not in MECHANISM_PROTOCOLS:
        raise ForumHeroError("hero experiment mechanism protocol is invalid")
    if ablation_mode not in PILOT_ABLATION_MODES:
        raise ForumHeroError("hero experiment LLM ablation mode is invalid")
    frozen_branches = tuple(dict.fromkeys(str(value) for value in branches))
    valid_branches = (*FORUM_BRANCHES, SHARED_BASELINE_BRANCH)
    if not frozen_branches or any(value not in valid_branches for value in frozen_branches):
        raise ForumHeroError("hero experiment branches are invalid")
    shared_baseline_round = (
        frozen_branches == (SHARED_BASELINE_BRANCH,)
        and end_tick == 2
        and start_tick in {0, 2}
    )
    shared_resource_prefix = (
        frozen_branches == (SHARED_BASELINE_BRANCH,)
        and scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO
        and start_tick == RESOURCE_POLICY_START_TICK
        and end_tick == RESOURCE_POLICY_SHARED_END_TICK
    )
    shared_century_gym_prefix = (
        frozen_branches == (SHARED_BASELINE_BRANCH,)
        and scenario_id == CENTURY_GYM_SCENARIO
        and start_tick == CENTURY_GYM_START_TICK
        and end_tick == CENTURY_GYM_SHARED_END_TICK
    )
    shared_quick_live_prefix = (
        frozen_branches == (SHARED_BASELINE_BRANCH,)
        and scenario_id in HERO_RUNTIME_SCENARIOS
        and start_tick == QUICK_LIVE_START_TICK
        and end_tick == QUICK_LIVE_SHARED_END_TICK
        and resident_budget_by_tick is not None
        and dict(resident_budget_by_tick)
        == {QUICK_LIVE_START_TICK: QUICK_LIVE_RESIDENT_BUDGET}
    )
    shared_four_dimension_prefix = (
        frozen_branches == (SHARED_BASELINE_BRANCH,)
        and scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO
        and start_tick == FOUR_DIMENSION_START_TICK
        and end_tick == FOUR_DIMENSION_SHARED_END_TICK
        and resident_budget_by_tick is not None
        and dict(resident_budget_by_tick)
        == {
            FOUR_DIMENSION_START_TICK: FOUR_DIMENSION_RESIDENT_BUDGET
        }
        and resident_turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION
        and activation_policy == COST_AWARE_ACTIVATION_POLICY
    )
    if SHARED_BASELINE_BRANCH in frozen_branches and not (
        shared_baseline_round
        or shared_resource_prefix
        or shared_century_gym_prefix
        or shared_quick_live_prefix
        or shared_four_dimension_prefix
    ):
        raise ForumHeroError("shared baseline must be one isolated T0-T2 or T2 round")
    if not isinstance(interaction_activation_enabled, bool):
        raise ForumHeroError("interaction activation flag is invalid")
    if not isinstance(require_scenario_entry_post, bool):
        raise ForumHeroError("scenario entry post flag is invalid")
    if require_scenario_entry_post and not (
        shared_resource_prefix
        or shared_quick_live_prefix
        or shared_four_dimension_prefix
    ):
        raise ForumHeroError(
            "scenario entry post is limited to a frozen shared-prefix run"
        )
    if activation_selection_salt is not None and (
        not isinstance(activation_selection_salt, str)
        or len(activation_selection_salt) != 64
        or any(
            character not in "0123456789abcdef"
            for character in activation_selection_salt
        )
    ):
        raise ForumHeroError("activation selection salt is invalid")
    if resident_turn_schema_version not in {
        FORUM_AGENT_TURN_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }:
        raise ForumHeroError("resident turn schema is invalid")
    if resident_model_view_condition not in V5_MODEL_VIEW_CONDITIONS:
        raise ForumHeroError("resident V5 model-view condition is invalid")
    if opening_public_action_mode not in {
        "event_initiators_only",
        "diagnostic_open_choice",
    }:
        raise ForumHeroError("opening public-action mode is invalid")
    if (
        resident_turn_schema_version != FORUM_AGENT_TURN_V5_SCHEMA_VERSION
        and resident_model_view_condition != V5_FULL_NATURAL_PROFILE
    ):
        raise ForumHeroError("resident model-view control requires turn V5")
    private_turn = resident_turn_schema_version in {
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }
    if any(
        not isinstance(value, bool)
        for value in (
            relationship_attention_enabled,
            private_actions_enabled,
            public_anonymity_enabled,
        )
    ):
        raise ForumHeroError("ForumTwin v4 channel flags must be boolean")
    if (
        (relationship_attention_enabled or private_actions_enabled)
        and (relationship_release is None or scenario_stake_release is None)
    ):
        raise ForumHeroError("relationship/private execution requires frozen releases")
    if private_actions_enabled and not private_turn:
        raise ForumHeroError("private actions require resident turn v3, v4 or v5")
    if public_anonymity_enabled and resident_turn_schema_version not in {
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }:
        raise ForumHeroError("public anonymity requires resident turn v4 or v5")
    if activation_policy not in ACTIVATION_POLICIES:
        raise ForumHeroError("ForumTwin activation policy is invalid")
    if (
        activation_policy == COST_AWARE_ACTIVATION_POLICY
        and ecology_calibration_release is None
    ):
        raise ForumHeroError("cost-aware activation requires an ecology calibration")
    ecology_calibration_sha256 = None
    if ecology_calibration_release is not None:
        validate_forum_ecology_calibration(ecology_calibration_release)
        ecology_calibration_sha256 = str(
            ecology_calibration_release["calibration_release_sha256"]
        )
    if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL and not (
        scenario_id == PILOT_SCENARIO
        and frozen_branches == ("D",)
        and start_tick == 3
        and end_tick == 5
    ):
        raise ForumHeroError(
            "targeted closed-loop audit requires the D/T3-T5 vignette"
        )
    if (
        mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
        and activation_policy != LEGACY_ACTIVATION_POLICY
    ):
        raise ForumHeroError("targeted closed-loop audit requires the legacy anchor panel")
    decision_ticks = (
        DECISION_TICKS
        if governance_decision_ticks is None
        else frozenset(int(value) for value in governance_decision_ticks)
    )
    if any(not 0 <= tick <= PILOT_END_TICK for tick in decision_ticks):
        raise ForumHeroError("governance decision tick is invalid")
    action_overrides = {
        str(branch): tuple(str(action) for action in actions)
        for branch, actions in dict(governance_action_overrides or {}).items()
    }
    if any(
        branch not in {"A", "B", "C", "D"}
        or not set(actions).issubset(BRANCH_ACTIONS[branch])
        for branch, actions in action_overrides.items()
    ):
        raise ForumHeroError("governance action override exceeds its branch catalogue")
    budgets = {
        tick: (
            pilot_resident_budget(tick)
            if resident_budget_by_tick is None
            else int(resident_budget_by_tick[tick])
        )
        for tick in range(start_tick, end_tick + 1)
    }
    required_anchor_count = (
        8 if activation_policy == COST_AWARE_ACTIVATION_POLICY else 16
    )
    if any(
        value != 0 and not required_anchor_count <= value <= 64
        for value in budgets.values()
    ):
        raise ForumHeroError(
            "hero resident budget must be zero or fit the selected anchor panel"
        )
    if governance_source is None:
        governance_source = resident_source
    profile_release = compile_pilot_profile_bridge(population_release)
    profiles = profile_release["profiles"]
    common_relationships = _relationship_runtime_state(
        relationship_release,
        profiles=profiles,
    )
    scenario_stakes = _scenario_stake_runtime_state(
        scenario_stake_release,
        scenario_id=scenario_id,
        profiles=profiles,
    )
    prompt_profiles = (
        _profile_shuffled_prompt_population(profiles, seed=seed)
        if ablation_mode == PROFILE_SHUFFLED_ABLATION_MODE
        else profiles
    )
    prompt_profiles_by_agent_id = {
        str(profile["agent_id"]): profile for profile in prompt_profiles
    }
    if anchor_indices_override is None:
        anchors = (
            select_cost_aware_anchors(
                profiles,
                scenario_id=scenario_id,
                seed=seed,
                anchor_count=required_anchor_count,
            )
            if activation_policy == COST_AWARE_ACTIVATION_POLICY
            else select_pilot_anchors(profiles, scenario_id=scenario_id, seed=seed)
        )
    else:
        anchors = tuple(int(value) for value in anchor_indices_override)
        if (
            len(anchors) != required_anchor_count
            or len(set(anchors)) != required_anchor_count
            or any(not 0 <= value < len(profiles) for value in anchors)
            or any(
                profiles[value].get("simulation_gate") == "background_signal"
                for value in anchors
            )
        ):
            raise ForumHeroError("resident anchor override is invalid")
    if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL:
        anchors = _targeted_capability_panel(
            profiles=profiles,
            anchor_indices=anchors,
            scenario_id=scenario_id,
        )
    population_view = scenario_population_view(
        population_release,
        scenario_id=scenario_weight_key(scenario_id),
    )
    # The 8+4 graph is private state-diffusion machinery only.  Public forum
    # exposure is built independently as one global hot/latest board.
    particle_graph_source = build_synthetic_exposure_graph(
        population_view,
        profile_aware=True,
    )
    particle_graph = compile_message_exposure_graph_v2(
        population_view, particle_graph_source
    )
    particle_features = _stake_conditioned_particle_features(
        base_features=_parent_features(population_view, scenario_id),
        profiles=profiles,
        scenario_stakes=scenario_stakes,
    )
    common_state = initial_branch_state(population_view, scenario_id=scenario_id, seed=seed)
    common_memories = _initial_memories(profiles, initial_memory_summaries)
    common_ledger = ForumLedger()
    common_rotation_counts: dict[int, int] = {}
    common_interaction_backlog: list[InteractionCandidate] = []
    common_activated_indices: set[int] = set()
    common_applied_innovations: set[str] = set()
    common_relationship_update_ids: set[str] = set()
    common_risk_signal_values = {name: 0.0 for name in RISK_SIGNAL_NAMES}
    if initial_snapshot is not None:
        legacy_snapshot_keys = {
            "scenario_id",
            "seed",
            "tick",
            "profile_release_sha256",
            "branch_state",
            "ledger",
            "memories",
            "rotation_counts",
            "activated_indices",
            "applied_innovations",
        }
        current_snapshot_keys = legacy_snapshot_keys | {"interaction_backlog"}
        risk_snapshot_keys = current_snapshot_keys | {"risk_signal_values"}
        v4_snapshot_keys = risk_snapshot_keys | {
            "relationships",
            "relationship_update_ids",
        }
        allowed_snapshot_keys = {
            frozenset(legacy_snapshot_keys),
            frozenset(current_snapshot_keys),
            frozenset(risk_snapshot_keys),
            frozenset(v4_snapshot_keys),
        }
        allowed_snapshot_keys.update(
            frozenset(set(keys) | {"ecology_calibration_release_sha256"})
            for keys in tuple(allowed_snapshot_keys)
        )
        if frozenset(initial_snapshot) not in allowed_snapshot_keys:
            raise ForumHeroError("initial ForumTwin snapshot contract drifted")
        if (
            initial_snapshot["scenario_id"] != scenario_id
            or int(initial_snapshot["seed"]) != seed
            or int(initial_snapshot["tick"]) != start_tick - 1
            or initial_snapshot["profile_release_sha256"]
            != profile_release["profile_release_sha256"]
            or not isinstance(initial_snapshot["ledger"], ForumLedger)
        ):
            raise ForumHeroError("initial ForumTwin snapshot identity drifted")
        snapshot_calibration_sha256 = initial_snapshot.get(
            "ecology_calibration_release_sha256"
        )
        if snapshot_calibration_sha256 != ecology_calibration_sha256:
            raise ForumHeroError("initial ForumTwin ecology calibration drifted")
        try:
            common_state = initial_snapshot["branch_state"].clone()
        except AttributeError as exc:
            raise ForumHeroError("initial ForumTwin particle state is invalid") from exc
        common_ledger = copy.deepcopy(initial_snapshot["ledger"])
        common_memories = copy.deepcopy(dict(initial_snapshot["memories"]))
        common_rotation_counts = {
            int(index): int(count)
            for index, count in dict(initial_snapshot["rotation_counts"]).items()
        }
        common_interaction_backlog = list(
            initial_snapshot.get("interaction_backlog", ())
        )
        if any(
            not isinstance(candidate, InteractionCandidate)
            for candidate in common_interaction_backlog
        ):
            raise ForumHeroError("initial ForumTwin interaction backlog is invalid")
        common_activated_indices = {
            int(index) for index in initial_snapshot["activated_indices"]
        }
        common_applied_innovations = {
            str(value) for value in initial_snapshot["applied_innovations"]
        }
        if "risk_signal_values" in initial_snapshot:
            common_risk_signal_values = {
                str(name): float(value)
                for name, value in dict(initial_snapshot["risk_signal_values"]).items()
            }
            if set(common_risk_signal_values) != set(RISK_SIGNAL_NAMES):
                raise ForumHeroError("initial ForumTwin risk frontier drifted")
        if "relationships" in initial_snapshot:
            common_relationships = copy.deepcopy(
                dict(initial_snapshot["relationships"])
            )
            common_relationship_update_ids = {
                str(value)
                for value in initial_snapshot.get("relationship_update_ids", ())
            }
        common_ledger.assert_consistent()
        common_state.ensemble.assert_conserved()
    common_posterior = _particle_posterior(common_state)
    branch_results: dict[str, Any] = {}
    branch_activated_indices: dict[str, set[int]] = {}
    artifact_log: list[dict[str, Any]] = []

    for branch in frozen_branches:
        ledger = copy.deepcopy(common_ledger)
        memories = copy.deepcopy(common_memories)
        posterior = copy.deepcopy(common_posterior)
        branch_state = common_state.clone()
        rotation_counts: dict[int, int] = defaultdict(int, common_rotation_counts)
        interaction_backlog = copy.deepcopy(common_interaction_backlog)
        relationships = copy.deepcopy(common_relationships)
        relationship_update_ids = set(common_relationship_update_ids)
        activated_indices: set[int] = set(common_activated_indices)
        timeline = []
        for prior_tick in range(0, start_tick):
            if initial_snapshot is not None and prior_tick == start_tick - 1:
                prior_metrics = _ledger_metrics(ledger, prior_tick)
            else:
                prior_metrics = {
                    "tick": prior_tick,
                    "thread_count": 0,
                    "message_count": 0,
                    "messages_this_tick": 0,
                    "claim_count": 0,
                }
            timeline.append(
                {
                    **prior_metrics,
                    "phase": "baseline",
                    "shared_baseline": True,
                    "population_state": _weighted_state(
                        profiles=profiles,
                        posterior=posterior,
                        scenario_id=scenario_id,
                    ),
                }
            )
        usage = Counter()
        actor_rounds = []
        applied_innovations: set[str] = set(common_applied_innovations)
        population_recovery_rounds: list[dict[str, Any]] = []
        scheduler_recall_totals = Counter()
        feed_delivery_sources = Counter()
        feed_role_matrix = Counter()
        risk_signal_values = dict(common_risk_signal_values)

        def current_snapshot(completed_tick: int) -> dict[str, Any]:
            return {
                "scenario_id": scenario_id,
                "seed": seed,
                "tick": completed_tick,
                "profile_release_sha256": profile_release[
                    "profile_release_sha256"
                ],
                "branch_state": branch_state.clone(),
                "ledger": copy.deepcopy(ledger),
                "memories": copy.deepcopy(memories),
                "rotation_counts": dict(rotation_counts),
                "interaction_backlog": copy.deepcopy(interaction_backlog),
                "activated_indices": set(activated_indices),
                "applied_innovations": set(applied_innovations),
                "risk_signal_values": dict(risk_signal_values),
                **(
                    {
                        "relationships": copy.deepcopy(relationships),
                        "relationship_update_ids": set(relationship_update_ids),
                    }
                    if private_turn or relationship_attention_enabled
                    else {}
                ),
                **(
                    {
                        "ecology_calibration_release_sha256": (
                            ecology_calibration_sha256
                        )
                    }
                    if ecology_calibration_sha256 is not None
                    else {}
                ),
            }

        for tick in range(start_tick, end_tick + 1):
            settled_governance_actions = _settle_forum_governance_actions(
                branch_state, tick
            )
            governance_artifact_updates = ledger.advance_governance_artifacts(
                tick
            )
            expired_private_invitations = (
                ledger.expire_private_invitations(tick) if private_turn else 0
            )
            particle_branch = (
                "natural"
                if branch in {"Natural", SHARED_BASELINE_BRANCH}
                or (
                    scenario_id == GOVERNANCE_LEGITIMACY_SCENARIO
                    and tick <= RESOURCE_POLICY_SHARED_END_TICK + 1
                )
                or (
                    scenario_id == CENTURY_GYM_SCENARIO
                    and tick <= CENTURY_GYM_SHARED_END_TICK + 1
                )
                else branch
            )
            particle_diffusion = advance_particle_tick(
                population=population_view,
                graph=particle_graph,
                state=branch_state,
                features=particle_features,
                scenario_id=scenario_id,
                branch=particle_branch,
                seed=seed,
                tick=tick,
                emit_messages=False,
            )
            posterior = _particle_posterior(branch_state)
            fresh_interactions = _interaction_candidates(
                ledger=ledger,
                profiles=profiles,
                posterior=posterior,
                tick=tick,
            )
            interactions = (
                _merge_interaction_candidates(
                    interaction_backlog, fresh_interactions
                )
                if interaction_activation_enabled
                else []
            )
            private_candidates: tuple[InteractionCandidate, ...] = ()
            private_reply_candidates: tuple[InteractionCandidate, ...] = ()
            private_initiation_candidates: tuple[InteractionCandidate, ...] = ()
            if private_actions_enabled and budgets[tick] > required_anchor_count:
                private_limit = max(
                    0, int((budgets[tick] - required_anchor_count) * 0.40)
                )
                private_reply_candidates = _private_interaction_candidates(
                    ledger=ledger,
                    profiles=profiles,
                    posterior=posterior,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    limit=private_limit,
                    excluded_indices=anchors,
                )
                reply_indices = {
                    candidate.agent_index for candidate in private_reply_candidates
                }
                private_initiation_candidates = tuple(
                    candidate
                    for candidate in _private_initiation_candidates(
                        ledger=ledger,
                        profiles=profiles,
                        relationships=relationships,
                        scenario_stakes=scenario_stakes,
                        posterior=posterior,
                        scenario_id=scenario_id,
                        branch=branch,
                        tick=tick,
                        limit=max(0, private_limit - len(reply_indices)),
                        excluded_indices=anchors,
                        rotation_counts=rotation_counts,
                    )
                    if candidate.agent_index not in reply_indices
                )
                private_candidates = (
                    private_reply_candidates + private_initiation_candidates
                )[:private_limit]
                public_indices = {candidate.agent_index for candidate in interactions}
                interactions.extend(
                    candidate
                    for candidate in private_candidates
                    if candidate.agent_index not in public_indices
                )
            activation = None
            selected: tuple[int, ...] = ()
            population_recovery = None
            particle_update: Mapping[str, Any] = {"applied_count": 0}
            relationship_updates_applied = 0
            if budgets[tick] > 0:
                activation = build_pilot_activation_round(
                    profiles=profiles,
                    tick=tick,
                    anchor_indices=anchors,
                    interaction_candidates=(
                        interactions if interaction_activation_enabled else ()
                    ),
                    rotation_counts=rotation_counts,
                    scenario_id=scenario_id,
                    resident_budget_override=budgets[tick],
                    selection_salt=(
                        activation_selection_salt
                        or canonical_hash(
                            ["forum-hero-activation", scenario_id, seed, branch]
                        )
                    ),
                    required_anchor_count=required_anchor_count,
                )
                selected = activation.resident_indices
                exposures, contexts = _turn_inputs(
                    selected_indices=selected,
                    profiles=profiles,
                    memories=memories,
                    ledger=ledger,
                    tick=tick,
                    scenario_id=scenario_id,
                    mechanism_protocol=mechanism_protocol,
                    bridge_enabled=ablation_mode != BRIDGE_DISABLED_ABLATION_MODE,
                    forum_context_mode=(
                        "metadata_only"
                        if ablation_mode == VISIBLE_TEXT_REMOVED_ABLATION_MODE
                        else "full_text"
                    ),
                    exposure_graph=particle_graph,
                    ecology_calibration_release=ecology_calibration_release,
                    opening_public_action_mode=opening_public_action_mode,
                )
                if relationship_attention_enabled:
                    exposures = _relationship_attention_exposures(
                        selected_indices=selected,
                        profiles=profiles,
                        posterior=posterior,
                        memories=memories,
                        ledger=ledger,
                        relationships=relationships,
                        existing=exposures,
                        tick=tick,
                        seed=seed,
                    )
                if private_turn:
                    for index in selected:
                        agent_id = str(profiles[index]["agent_id"])
                        context = copy.deepcopy(contexts[index])
                        context["relationship_context"] = list(
                            _relationship_prompt_rows(
                                agent_id,
                                relationships,
                                profiles_by_agent_id=prompt_profiles_by_agent_id,
                                scenario_stakes=scenario_stakes,
                                public_board_active=bool(ledger.messages),
                            )
                        )
                        context["private_conversation_ids"] = list(
                            _visible_private_conversation_ids(
                                ledger,
                                agent_id=agent_id,
                                scenario_id=scenario_id,
                                branch=branch,
                                tick=tick,
                            )
                        )
                        context["private_actions_enabled"] = private_actions_enabled
                        context["public_anonymity_enabled"] = (
                            public_anonymity_enabled
                        )
                        if scenario_stakes:
                            context["scenario_context"] = {
                                **dict(context.get("scenario_context") or {}),
                                "current_synthetic_stake": (
                                    _scenario_stake_prompt_summary(
                                        scenario_stakes[agent_id]
                                    )
                                ),
                            }
                        contexts[index] = context
                if require_scenario_entry_post and tick == QUICK_LIVE_START_TICK:
                    if ledger.messages:
                        raise ForumHeroError(
                            "scenario entry post requires an empty shared forum"
                        )
                    entry_candidates = []
                    for raw_index in selected:
                        index = int(raw_index)
                        profile = profiles[index]
                        tendencies = profile.get("forum_tendencies", {})
                        if (
                            profile.get("simulation_gate") == "active_episode_agent"
                            and "post"
                            in set(tendencies.get("allowed_public_actions", ()))
                        ):
                            activity = float(
                                tendencies.get(
                                    "baseline_activity",
                                    profile.get("initial_state", {}).get(
                                        "activity_probability", 0.0
                                    ),
                                )
                            )
                            entry_candidates.append(
                                (
                                    -(
                                        _scenario_design_weight(profile, scenario_id)
                                        * (0.25 + max(activity, 0.0))
                                    ),
                                    str(profile.get("primary_macro_role") or ""),
                                    str(profile.get("agent_id") or ""),
                                    index,
                                )
                            )
                    if not entry_candidates:
                        raise ForumHeroError(
                            "quick-live shared round lacks an eligible scenario observer"
                        )
                    entry_index = min(entry_candidates)[-1]
                    entry_context = copy.deepcopy(contexts[entry_index])
                    entry_context["current_thread_id"] = None
                    entry_context["direct_parent_message_id"] = None
                    entry_context["quote_message_id"] = None
                    entry_context["allowed_actions"] = ["post"]
                    entry_context["scenario_context"] = {
                        **dict(entry_context.get("scenario_context") or {}),
                        "scenario_entry_role": (
                            "You are the time-visible synthetic observer who opens this "
                            "forum event. Write one profile-consistent root post. Separate "
                            "what you observed from competing explanations; do not invent "
                            "a cause, an official conclusion or a real-world identity."
                        ),
                    }
                    contexts[entry_index] = entry_context
                display_to_profile = {
                    str(row["synthetic_display_id"]): row for row in profiles
                }
                for recipient_index, exposure_rows in exposures.items():
                    recipient_role = str(
                        profiles[recipient_index].get("primary_macro_role")
                        or "unknown"
                    )
                    for exposure in exposure_rows:
                        feed_delivery_sources[exposure.exposure_source] += 1
                        message = ledger.messages.get(exposure.message_id)
                        if message is None:
                            continue
                        if message.source_agent_display_id.startswith("governance:"):
                            source_role = message.source_agent_display_id
                        else:
                            source_profile = display_to_profile.get(
                                message.source_agent_display_id, {}
                            )
                            source_role = str(
                                source_profile.get("primary_macro_role") or "unknown"
                            )
                        feed_role_matrix[f"{source_role}->{recipient_role}"] += 1
                prompt_memories = None
                if ablation_mode == MEMORY_DISABLED_ABLATION_MODE:
                    prompt_memories = {
                        index: AgentMemory(agent_id=str(profiles[index]["agent_id"]))
                        for index in selected
                    }
                round_artifacts: list[tuple[str, dict[str, Any]]] = []
                resident_executor = {
                    FORUM_AGENT_TURN_SCHEMA_VERSION: execute_resident_activation_round_v1,
                    FORUM_AGENT_TURN_V2_SCHEMA_VERSION: execute_resident_activation_round_v2,
                    FORUM_AGENT_TURN_V3_SCHEMA_VERSION: execute_resident_activation_round_v3,
                    FORUM_AGENT_TURN_V4_SCHEMA_VERSION: execute_resident_activation_round_v4,
                    FORUM_AGENT_TURN_V5_SCHEMA_VERSION: execute_resident_activation_round_v5,
                }[resident_turn_schema_version]
                resident_result = resident_executor(
                    activation_round=activation,
                    population=profiles,
                    posterior_by_index=posterior,
                    memories_by_index=memories,
                    ledger=ledger,
                    exposures_by_index=exposures,
                    context_by_index=contexts,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    expected_model=expected_model,
                    source=resident_source,
                    slot_design_callback=lambda row: round_artifacts.append(
                        ("resident_slot_design", copy.deepcopy(dict(row)))
                    ),
                    attempt_callback=lambda row: round_artifacts.append(
                        ("resident_attempt_received", copy.deepcopy(dict(row)))
                    ),
                    applied_innovation_ids=applied_innovations,
                    reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                    planned_resident_budget=budgets[tick],
                    prompt_population=prompt_profiles,
                    prompt_memories_by_index=prompt_memories,
                    resident_model_view_condition=(
                        resident_model_view_condition
                    ),
                )
                relationship_updates_applied = (
                    _apply_forum_relationship_updates(
                        relationships=relationships,
                        attempts=resident_result["attempts"],
                        ledger=ledger,
                        profiles=profiles,
                        tick=tick,
                        applied_update_ids=relationship_update_ids,
                    )
                    if private_turn or relationship_attention_enabled
                    else 0
                )
                particle_update = assimilate_forum_llm_updates_v2(
                    state=branch_state,
                    updates=_forum_particle_updates(
                        attempts=resident_result["attempts"],
                        profiles=profiles,
                    ),
                    scenario_id=scenario_id,
                    branch=particle_branch,
                    tick=tick,
                    seed=seed,
                )
                posterior = _particle_posterior(branch_state)
                population_recovery = _forum_population_recovery(
                    profiles=profiles,
                    posterior=posterior,
                    activation=activation,
                    resident_result=resident_result,
                    interaction_candidates=interactions,
                    scenario_id=scenario_id,
                )
                selected_set = set(selected)
                interaction_backlog = [
                    candidate
                    for candidate in activation.overflow
                    if candidate.agent_index not in selected_set
                ]
            elif not interaction_activation_enabled:
                interaction_backlog = []
            else:
                interaction_backlog = interactions
            if population_recovery is not None:
                recall_row = population_recovery["scheduler_recall"]
                scheduler_recall_totals["observed"] += int(
                    recall_row["pps_observed_public_actions"]
                )
                scheduler_recall_totals["recalled"] += int(
                    recall_row["scheduler_recalled_public_actions"]
                )
                writeback = apply_forum_posterior_writeback(
                    state=branch_state,
                    mode="observation_only",
                    posterior_targets={},
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    innovation_id="forum-observation-writeback-"
                    + canonical_hash(
                        [scenario_id, seed, branch, tick, activation.content_hash]
                    )[:32],
                )
                population_recovery = {
                    **population_recovery,
                    "tick": tick,
                    "formal_writeback": writeback,
                }
                population_recovery_rounds.append(population_recovery)
            if budgets[tick] > 0:
                for index in selected:
                    if index not in anchors:
                        rotation_counts[index] += 1
                activated_indices.update(selected)
                usage.update(resident_result["usage"])
                artifact_log.append(
                    {
                        "branch": branch,
                        "tick": tick,
                        "kind": "resident_round",
                        "content_sha256": resident_result["content_sha256"],
                        "artifact_count": len(round_artifacts),
                    }
                )

            if branch not in {"Natural", SHARED_BASELINE_BRANCH} and tick in decision_ticks:
                observations, forum_contexts, target_catalogs = _actor_views(
                    ledger,
                    tick,
                    scenario_id=scenario_id,
                    branch=branch,
                    mechanism_protocol=mechanism_protocol,
                    population_recovery=population_recovery,
                )
                actor_artifacts: list[tuple[str, dict[str, Any]]] = []
                actor_result = execute_forum_actor_round_v1(
                    state=branch_state,
                    ledger=ledger,
                    source=governance_source,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    seed=seed,
                    private_observations=observations,
                    forum_contexts=forum_contexts,
                    target_catalogs=target_catalogs,
                    feasible_actions=action_overrides.get(
                        branch, BRANCH_ACTIONS[branch]
                    ),
                    reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                    artifact_callback=lambda kind, row: actor_artifacts.append(
                        (kind, copy.deepcopy(dict(row)))
                    ),
                )
                actor_round = actor_result["governance_round"]
                actor_rounds.append(
                    {
                        "tick": tick,
                        "decisions": actor_round["qre_public"]["decisions"],
                        "resource_before": actor_round["resource_before"],
                        "resource_after": actor_round["resource_after"],
                        "message_ids": actor_round["message_ids"],
                        "governance_artifacts": actor_round["qre_public"].get(
                            "governance_artifacts", []
                        ),
                        "uses_hidden_truth": False,
                        "scheme_bonus": 0,
                    }
                )
                for attempt in actor_result["attempts"]:
                    actor_usage = attempt["usage"]
                    usage["provider_calls"] += int(actor_usage["provider_calls"])
                    usage["provider_attempts"] += int(actor_usage["provider_attempts"])
                    usage["total_tokens"] += int(actor_usage["total_tokens"])
                artifact_log.append(
                    {
                        "branch": branch,
                        "tick": tick,
                        "kind": "governance_round",
                        "content_sha256": actor_result["execution_sha256"],
                        "artifact_count": len(actor_artifacts),
                    }
                )

            particle_messages_added = _sync_forum_messages_to_particle_ledger(
                forum_ledger=ledger,
                branch_state=branch_state,
                profiles=profiles,
                tick=tick,
            )
            metrics = _ledger_metrics(ledger, tick)
            population_state = _weighted_state(
                profiles=profiles,
                posterior=posterior,
                scenario_id=scenario_id,
            )
            risk_result = None
            if private_turn or relationship_attention_enabled:
                recipient_counts = Counter()
                for key, count in feed_role_matrix.items():
                    _source, _separator, recipient = key.partition("->")
                    recipient_counts[recipient] += int(count)
                maximum_recipient = max(recipient_counts.values(), default=1)
                exposure_rates = {
                    key: value / maximum_recipient
                    for key, value in recipient_counts.items()
                }
                correction_forward_count = 0
                for private_message in ledger.private_messages.values():
                    source_id = private_message.public_source_message_id
                    public_message = ledger.messages.get(source_id) if source_id else None
                    if public_message is not None and public_message.evidence_status in {
                        "corrected",
                        "verified",
                    }:
                        correction_forward_count += 1
                private_message_count = max(1, len(ledger.private_messages))
                correction_rates = {
                    "all_private_conversations": correction_forward_count
                    / private_message_count
                }
                service_receipts = sum(
                    message.governance_artifact is not None
                    and message.governance_artifact.artifact_kind == "service_ticket"
                    for message in ledger.messages.values()
                )
                macro_by_display = {
                    str(profile["synthetic_display_id"]): str(
                        profile.get("primary_macro_role") or "unknown"
                    )
                    for profile in profiles
                }
                risk_result = build_governance_risk_result_v1(
                    ledger=ledger,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    macro_by_display_id=macro_by_display,
                    cross_group_exposure_rates=exposure_rates,
                    correction_private_reach_rates=correction_rates,
                    help_queue={
                        "backlog": max(
                            0,
                            metrics["help_request_count"] - service_receipts,
                        ),
                        "capacity": max(1, service_receipts),
                        "sla_breaches": max(
                            0,
                            metrics["help_request_count"] - service_receipts,
                        ),
                        "help_requests": metrics["help_request_count"],
                    },
                    population_public_state=population_state,
                    governance_visible_state=_governance_visible_posterior(
                        population_recovery
                    ),
                    previous_signal_values=risk_signal_values,
                )
                risk_signal_values = {
                    row["signal_name"]: float(row["value"])
                    for row in risk_result["signals"]
                }
            timeline.append(
                {
                    **metrics,
                    "phase": scenario_context_for_tick(
                        tick,
                        scenario_id,
                        mechanism_protocol,
                    )["current_phase"],
                    "resident_llm_activations": len(selected),
                    "anchor_activations": (
                        len(activation.anchor_indices) if activation is not None else 0
                    ),
                    "interaction_activations": (
                        len(activation.interaction_indices) if activation is not None else 0
                    ),
                    "pps_activations": (
                        len(activation.pps_indices) if activation is not None else 0
                    ),
                    "governance_llm_activations": (
                        3
                        if branch not in {"Natural", SHARED_BASELINE_BRANCH}
                        and tick in decision_ticks
                        else 0
                    ),
                    "particle_diffusion_reads": int(particle_diffusion["reads"]),
                    "settled_governance_actions": settled_governance_actions,
                    "governance_artifact_status_updates": governance_artifact_updates,
                    "expired_private_invitations": expired_private_invitations,
                    "private_interaction_candidates": len(private_candidates),
                    "private_reply_candidates": len(private_reply_candidates),
                    "private_initiation_candidates": len(
                        private_initiation_candidates
                    ),
                    "relationship_updates_applied": relationship_updates_applied,
                    "private_conversation_count": len(ledger.private_conversations),
                    "private_message_count": len(ledger.private_messages),
                    "private_group_count": sum(
                        conversation.conversation_type == "group"
                        for conversation in ledger.private_conversations.values()
                    ),
                    "anonymous_public_message_count": sum(
                        getattr(message, "author_visibility", "named") == "anonymous"
                        for message in ledger.messages.values()
                        if not message.superseded
                    ),
                    "governance_risk": risk_result,
                    "particle_messages_ingested": particle_messages_added,
                    "particle_llm_updates": int(particle_update["applied_count"]),
                    "particle_uncertainty": _particle_uncertainty(branch_state),
                    "pps_population_recovery": population_recovery,
                    "population_state": population_state,
                    "population_heterogeneity": _population_heterogeneity_frame(
                        profiles=profiles,
                        posterior=posterior,
                        scenario_stakes=scenario_stakes,
                        branch_state=branch_state,
                        scenario_id=scenario_id,
                        branch=branch,
                        tick=tick,
                    ),
                }
            )
            ledger.assert_consistent()
            if public_tick_callback is not None:
                public_branch = _public_branch_result(
                    branch=branch,
                    ledger=ledger,
                    timeline=timeline,
                    activated_indices=activated_indices,
                    profiles=profiles,
                    usage=usage,
                    actor_rounds=actor_rounds,
                    feed_delivery_sources=feed_delivery_sources,
                    feed_role_matrix=feed_role_matrix,
                    resident_turn_schema_version=resident_turn_schema_version,
                )
                public_projection = {
                    "schema_version": PUBLIC_TICK_SCHEMA,
                    "scenario_id": scenario_id,
                    "branch": branch,
                    "seed": seed,
                    "tick": tick,
                    "tick_committed": True,
                    "public_content_provenance_rule": (
                        "authorized_live_llm_or_exact_reviewed_trace"
                    ),
                    "dynamics_public_messages": 0,
                    "public_branch": public_branch,
                }
                public_projection["content_sha256"] = canonical_hash(
                    public_projection
                )
                public_tick_callback(branch, tick, public_projection)
            if tick_snapshot_callback is not None:
                tick_snapshot_callback(branch, tick, current_snapshot(tick))

        branch_results[branch] = _public_branch_result(
            branch=branch,
            ledger=ledger,
            timeline=timeline,
            activated_indices=activated_indices,
            profiles=profiles,
            usage=usage,
            actor_rounds=actor_rounds,
            feed_delivery_sources=feed_delivery_sources,
            feed_role_matrix=feed_role_matrix,
            resident_turn_schema_version=resident_turn_schema_version,
        )
        observed_public_actions = int(scheduler_recall_totals["observed"])
        recalled_public_actions = int(scheduler_recall_totals["recalled"])
        aggregate_recall = (
            1.0
            if observed_public_actions == 0
            else recalled_public_actions / observed_public_actions
        )
        branch_results[branch]["population_recovery"] = {
            "round_count": len(population_recovery_rounds),
            "rounds": population_recovery_rounds,
            "formal_writeback_mode": "observation_only",
            "scheduler_recall": {
                "pps_observed_public_actions": observed_public_actions,
                "scheduler_recalled_public_actions": recalled_public_actions,
                "scheduler_false_negative_count": (
                    observed_public_actions - recalled_public_actions
                ),
                "public_action_recall": round(aggregate_recall, 6),
                "gate_threshold": 0.90,
                "gate_passed": aggregate_recall >= 0.90,
                "identities_public": False,
            },
        }
        if snapshot_callback is not None:
            snapshot_callback(branch, current_snapshot(end_tick))
        branch_activated_indices[branch] = set(activated_indices)

    paired_difference = None
    if {"Natural", "D"}.issubset(branch_results):
        natural_final = branch_results["Natural"]["timeline"][-1]
        governed_final = branch_results["D"]["timeline"][-1]
        paired_difference = {
            key: governed_final[key] - natural_final[key]
            for key in (
                "thread_count",
                "message_count",
                "claim_count",
                "contested_claim_count",
                "corrected_or_verified_claim_count",
                "help_request_count",
                "correction_count",
                "reply_count",
            )
        }
    body = {
        "schema_version": HERO_RESULT_SCHEMA,
        "scenario_id": scenario_id,
        "seed": seed,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "mechanism_protocol": mechanism_protocol,
        "llm_ablation_mode": ablation_mode,
        "scenario_entry_post_required": require_scenario_entry_post,
        "resident_semantic_source": "same_real_llm_or_exact_reviewed_trace",
        "dynamics_public_messages": 0,
        "targeted_action_categories_disclosed": (
            mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
        ),
        "branches_executed": list(frozen_branches),
        "resident_budget_by_tick": {str(key): value for key, value in budgets.items()},
        "activation_policy": activation_policy,
        "longitudinal_anchor_count": required_anchor_count,
        "non_anchor_population_remains_pps_eligible": True,
        "population_size": 1_000,
        "particle_count": int(common_state.ensemble.particle_count),
        "particles_per_agent": int(common_state.ensemble.values.shape[1]),
        "particle_state_drives_parent_posterior": True,
        "low_cost_particle_tick_updates": (
            int(common_state.ensemble.particle_count)
            * len(frozen_branches)
            * len(budgets)
        ),
        "particle_graph_sha256": particle_graph.graph_sha256,
        "private_particle_graph_selection_method": particle_graph_source.get(
            "selection_method"
        ),
        "feed_graph_selection_method": "not_used_for_public_feed",
        "forum_feed_graph_controlled": False,
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "ecology_calibration_release_sha256": ecology_calibration_sha256,
        "ecology_calibrated_open_selection": (
            ecology_calibration_sha256 is not None
        ),
        "ecology_calibration_changes_public_ranking": False,
        "resident_turn_schema_version": resident_turn_schema_version,
        "relationship_attention_enabled": relationship_attention_enabled,
        "private_actions_enabled": private_actions_enabled,
        "public_anonymity_enabled": public_anonymity_enabled,
        "relationship_release_sha256": (
            relationship_release.get("relationship_release_sha256")
            if relationship_release is not None
            else None
        ),
        "scenario_stake_release_sha256": (
            scenario_stake_release.get("stake_release_sha256")
            if scenario_stake_release is not None
            else None
        ),
        "relationship_update_count": sum(
            int(point.get("relationship_updates_applied", 0))
            for branch in branch_results.values()
            for point in branch["timeline"]
        ),
        "profile_release_sha256": profile_release["profile_release_sha256"],
        "anchor_count": len(anchors),
        "anchor_macro_role_coverage": len(
            {str(profiles[index].get("primary_macro_role")) for index in anchors}
        ),
        "targeted_capability_panel_used": (
            mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
        ),
        "branches": branch_results,
        "unique_activated_agents_across_branches": len(
            set().union(*branch_activated_indices.values())
            if branch_activated_indices
            else set()
        ),
        "activation_coverage_gate": {
            "applicable": bool(
                start_tick == 3
                and end_tick == 23
                and {"Natural", "D"}.issubset(frozen_branches)
            ),
            "required_unique_agents": 500,
            "observed_unique_agents": len(
                set().union(*branch_activated_indices.values())
                if branch_activated_indices
                else set()
            ),
            "passed": (
                len(set().union(*branch_activated_indices.values())) >= 500
                if start_tick == 3
                and end_tick == 23
                and {"Natural", "D"}.issubset(frozen_branches)
                else None
            ),
        },
        "paired_final_difference_D_minus_Natural": paired_difference,
        "public_content_provenance_rule": "authorized_live_llm_or_exact_reviewed_trace",
        "emulator_public_messages": 0,
        "real_governance_actions": 0,
        "governance_effect_contract": (
            "resource-constrained LLM action object -> ordinary global board post -> "
            "resident LLM read/response; no hidden reach boost and no scheme bonus"
        ),
        "formal_full_corpus_profile_release_claimed": bool(
            profile_release.get("formal_full_corpus_profile_release")
        ),
        "non_claims": [
            "The 1,000 agents are synthetic forum behaviour agents, not students.",
            "The paired branches are model-conditioned parallel forums, not causal policy estimates.",
            "The hot ranking is a disclosed synthetic proxy, not the platform's proprietary formula.",
            "The private particle graph does not rank or deliver public forum posts.",
            *(
                [
                    "The full-corpus population-v2 is consumed through a hash-bound compatibility adapter; the adapter is not a new source population."
                ]
                if profile_release.get("formal_full_corpus_profile_release")
                else [
                    "This hero pilot reuses the reviewed population bridge and is not the final full-corpus profile release."
                ]
            ),
            *(
                [
                    "The closed-loop mechanism audit preassigns action categories and exact visible targets; only semantic content and reactions are LLM-generated."
                ]
                if mechanism_protocol == TARGETED_CLOSED_LOOP_PROTOCOL
                else []
            ),
        ],
        "artifact_log": {
            "schema_version": HERO_ARTIFACT_SCHEMA,
            "items": artifact_log,
            "prompt_or_response_bodies_persisted": False,
        },
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def _summed_usage(*rows: Mapping[str, Any]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[str(key)] += int(value)
    return dict(sorted(totals.items()))


def run_forum_v4_resource_closed_loop_hero(
    *,
    population_release: Mapping[str, Any],
    relationship_release: Mapping[str, Any],
    scenario_stake_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    seed: int = PILOT_SEED,
    reviewed_trace_release_sha256: str | None = None,
    checkpoint_callback: Callable[[str, int, Mapping[str, Any]], None]
    | None = None,
    public_tick_callback: Callable[[str, int, Mapping[str, Any]], None]
    | None = None,
) -> dict[str, Any]:
    """Run the cost-bounded v4 public/private resource-allocation Hero.

    Sixteen anchors open one shared Tick-3 world. Natural and D clone that
    exact complete state and each run 24 residents per Tick through Tick 10.
    D receives three governance actors after resident turns at Ticks 4 and 7.
    Public speech, private speech, parallel reactions and optional public
    anonymity all come from the injected semantic source; Dynamics generates
    no language.
    """

    if resident_source is None:
        raise ForumHeroError("ForumTwin v4 resource Hero requires a resident source")
    if governance_source is None:
        governance_source = resident_source
    shared_snapshot: dict[str, Any] = {}

    def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
        shared_snapshot.update(copy.deepcopy(dict(snapshot)))

    shared = run_forum_hero_experiment(
        population_release=population_release,
        relationship_release=relationship_release,
        scenario_stake_release=scenario_stake_release,
        resident_source=resident_source,
        governance_source=governance_source,
        expected_model=expected_model,
        scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
        seed=seed,
        start_tick=3,
        end_tick=3,
        branches=(SHARED_BASELINE_BRANCH,),
        resident_budget_by_tick={3: 16},
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
        snapshot_callback=capture_shared,
        tick_snapshot_callback=checkpoint_callback,
        public_tick_callback=public_tick_callback,
        governance_decision_ticks=(),
        resident_turn_schema_version=FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        require_scenario_entry_post=True,
        relationship_attention_enabled=True,
        private_actions_enabled=True,
        public_anonymity_enabled=True,
    )
    if int(shared_snapshot.get("tick", -1)) != 3:
        raise ForumHeroError("ForumTwin v4 resource Hero did not commit shared T3")

    continuation_budgets = {tick: 24 for tick in range(4, 11)}
    decision_ticks = frozenset({4, 7})
    action_overrides = {
        "D": (
            "verified_update",
            "evidence_dashboard",
            "listening_session",
            "support_channel",
            "queue_status",
            "peer_feedback",
            "targeted_prebunk",
            "bridge_outreach",
        )
    }
    continuations: dict[str, Any] = {}
    for branch in ("Natural", "D"):
        continuations[branch] = run_forum_hero_experiment(
            population_release=population_release,
            relationship_release=relationship_release,
            scenario_stake_release=scenario_stake_release,
            resident_source=resident_source,
            governance_source=governance_source,
            expected_model=expected_model,
            scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
            seed=seed,
            start_tick=4,
            end_tick=10,
            branches=(branch,),
            resident_budget_by_tick=continuation_budgets,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            initial_snapshot=shared_snapshot,
            activation_selection_salt=canonical_hash(
                ["forum-v4-resource-hero-paired-sample", seed]
            ),
            governance_decision_ticks=decision_ticks,
            governance_action_overrides=action_overrides,
            tick_snapshot_callback=checkpoint_callback,
            public_tick_callback=public_tick_callback,
            resident_turn_schema_version=FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
            relationship_attention_enabled=True,
            private_actions_enabled=True,
            public_anonymity_enabled=True,
        )

    shared_public = shared["branches"][SHARED_BASELINE_BRANCH]
    branches: dict[str, Any] = {}
    usage_rows: list[Mapping[str, Any]] = [shared_public["usage"]]
    for branch in ("Natural", "D"):
        branch_public = copy.deepcopy(continuations[branch]["branches"][branch])
        branch_public["timeline"] = (
            copy.deepcopy(shared_public["timeline"])
            + branch_public["timeline"][4:]
        )
        branch_public["world_label"] = (
            "自然演化" if branch == "Natural" else "有限观测自适应治理"
        )
        branch_public["shared_prefix_result_sha256"] = shared["result_sha256"]
        branch_public["usage"] = _summed_usage(
            shared_public["usage"], branch_public["usage"]
        )
        branches[branch] = branch_public
        usage_rows.append(continuations[branch]["branches"][branch]["usage"])

    natural_final = branches["Natural"]["timeline"][-1]
    governed_final = branches["D"]["timeline"][-1]
    count_keys = (
        "thread_count",
        "message_count",
        "claim_count",
        "contested_claim_count",
        "corrected_or_verified_claim_count",
        "help_request_count",
        "correction_count",
        "reply_count",
        "interaction_count",
        "private_conversation_count",
        "private_message_count",
        "private_group_count",
        "anonymous_public_message_count",
    )
    state_difference = {
        key: round(
            float(governed_final["population_state"][key])
            - float(natural_final["population_state"][key]),
            6,
        )
        for key in STATE_KEYS
    }
    natural_risk = {
        row["signal_name"]: float(row["value"])
        for row in natural_final["governance_risk"]["signals"]
    }
    governed_risk = {
        row["signal_name"]: float(row["value"])
        for row in governed_final["governance_risk"]["signals"]
    }
    body = {
        "schema_version": FORUM_V4_RESOURCE_HERO_RESULT_SCHEMA,
        "scenario_id": GOVERNANCE_LEGITIMACY_SCENARIO,
        "scenario_title": "暑期住宿床位分配正当性争议",
        "seed": seed,
        "start_tick": 3,
        "shared_through_tick": 3,
        "fork_tick": 4,
        "end_tick": 10,
        "branches": branches,
        "worlds_share_exact_T3_checkpoint": True,
        "resident_turn_schema_version": FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        "resident_slots": 352,
        "governance_actor_slots": 6,
        "primary_slots": 358,
        "population_size": 1_000,
        "particle_count": 10_000,
        "activation_design": {
            "shared_T3_anchors": 16,
            "branch_tick_anchors": 16,
            "branch_tick_interaction_or_PPS": 8,
            "private_origin_quota_of_non_anchor_slots": 0.40,
        },
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "public_feed_personalized": False,
        "relationship_attention_enabled": True,
        "private_actions_enabled": True,
        "dynamic_groups_enabled": True,
        "public_anonymity_enabled": True,
        "dynamics_public_or_private_messages": 0,
        "public_content_provenance_rule": (
            "authorized_live_llm_or_exact_reviewed_trace"
        ),
        "decision_ticks": sorted(decision_ticks),
        "policy_action_sets": {"Natural": [], "D": list(action_overrides["D"])},
        "scheme_bonus": 0,
        "winner_precommitted": False,
        "paired_final_count_difference_D_minus_Natural": {
            key: int(governed_final.get(key, 0)) - int(natural_final.get(key, 0))
            for key in count_keys
        },
        "paired_final_state_difference_D_minus_Natural": state_difference,
        "paired_final_risk_difference_D_minus_Natural": {
            key: round(governed_risk[key] - natural_risk[key], 6)
            for key in RISK_SIGNAL_NAMES
        },
        "usage": _summed_usage(*usage_rows),
        "low_cost_particle_tick_updates": (
            shared["low_cost_particle_tick_updates"]
            + sum(
                value["low_cost_particle_tick_updates"]
                for value in continuations.values()
            )
        ),
        "profile_release_sha256": shared["profile_release_sha256"],
        "relationship_release_sha256": shared["relationship_release_sha256"],
        "scenario_stake_release_sha256": shared["scenario_stake_release_sha256"],
        "formal_full_corpus_profile_release_claimed": shared[
            "formal_full_corpus_profile_release_claimed"
        ],
        "risk_threshold_source": (
            natural_final["governance_risk"]["threshold_source"]
        ),
        "non_claims": [
            "The 1,000 Agents and all relationships are synthetic.",
            "Natural versus D is a paired model rehearsal, not a causal estimate.",
            "Private text and record-level relationship edges are excluded from this public result.",
            "No governance world is guaranteed to improve the outcome.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def run_forum_four_dimension_resource_hero(
    *,
    population_release: Mapping[str, Any],
    relationship_release: Mapping[str, Any],
    scenario_stake_release: Mapping[str, Any],
    ecology_calibration_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    seed: int = PILOT_SEED,
    reviewed_trace_release_sha256: str | None = None,
    checkpoint_callback: Callable[[str, int, Mapping[str, Any]], None]
    | None = None,
    public_tick_callback: Callable[[str, int, Mapping[str, Any]], None]
    | None = None,
) -> dict[str, Any]:
    """Run one same-source experiment that yields all four fidelity dimensions.

    The shared Tick-3 world uses the natural V5 resident view and an
    ecology-calibrated 8-anchor/16-rotation panel.  Natural and D then clone
    the exact committed world and advance through Tick 10 with the same
    population, relationship graph, stakes and exogenous random frontier.
    Only D may add governance objects, at Ticks 5 and 7.

    This is an orchestration wrapper around :func:`run_forum_hero_experiment`;
    it does not introduce another simulation kernel or any target-result
    reward.  All public and private language still comes from the supplied
    formal semantic source.
    """

    if resident_source is None:
        raise ForumHeroError("four-dimensional Hero requires a resident source")
    if governance_source is None:
        governance_source = resident_source
    validate_forum_ecology_calibration(ecology_calibration_release)
    shared_snapshot: dict[str, Any] = {}

    def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
        shared_snapshot.update(copy.deepcopy(dict(snapshot)))

    shared = run_forum_hero_experiment(
        population_release=population_release,
        relationship_release=relationship_release,
        scenario_stake_release=scenario_stake_release,
        ecology_calibration_release=ecology_calibration_release,
        resident_source=resident_source,
        governance_source=governance_source,
        expected_model=expected_model,
        scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
        seed=seed,
        start_tick=FOUR_DIMENSION_START_TICK,
        end_tick=FOUR_DIMENSION_SHARED_END_TICK,
        branches=(SHARED_BASELINE_BRANCH,),
        resident_budget_by_tick={
            FOUR_DIMENSION_START_TICK: FOUR_DIMENSION_RESIDENT_BUDGET
        },
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
        snapshot_callback=capture_shared,
        tick_snapshot_callback=checkpoint_callback,
        public_tick_callback=public_tick_callback,
        governance_decision_ticks=(),
        resident_turn_schema_version=FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
        require_scenario_entry_post=True,
        relationship_attention_enabled=True,
        private_actions_enabled=True,
        public_anonymity_enabled=True,
        activation_policy=COST_AWARE_ACTIVATION_POLICY,
    )
    if int(shared_snapshot.get("tick", -1)) != FOUR_DIMENSION_SHARED_END_TICK:
        raise ForumHeroError("four-dimensional Hero did not commit shared Tick 3")

    continuation_budgets = {
        tick: FOUR_DIMENSION_RESIDENT_BUDGET
        for tick in range(
            FOUR_DIMENSION_SHARED_END_TICK + 1,
            FOUR_DIMENSION_END_TICK + 1,
        )
    }
    action_overrides = {
        "D": (
            "verified_update",
            "evidence_dashboard",
            "listening_session",
            "support_channel",
            "queue_status",
            "peer_feedback",
            "targeted_prebunk",
            "bridge_outreach",
        )
    }
    continuations: dict[str, Any] = {}
    for branch in FOUR_DIMENSION_BRANCHES:
        continuations[branch] = run_forum_hero_experiment(
            population_release=population_release,
            relationship_release=relationship_release,
            scenario_stake_release=scenario_stake_release,
            ecology_calibration_release=ecology_calibration_release,
            resident_source=resident_source,
            governance_source=governance_source,
            expected_model=expected_model,
            scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
            seed=seed,
            start_tick=FOUR_DIMENSION_SHARED_END_TICK + 1,
            end_tick=FOUR_DIMENSION_END_TICK,
            branches=(branch,),
            resident_budget_by_tick=continuation_budgets,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            initial_snapshot=shared_snapshot,
            activation_selection_salt=canonical_hash(
                ["four-dimension-resource-hero-paired-sample-v1", seed]
            ),
            governance_decision_ticks=FOUR_DIMENSION_DECISION_TICKS,
            governance_action_overrides=action_overrides,
            tick_snapshot_callback=checkpoint_callback,
            public_tick_callback=public_tick_callback,
            resident_turn_schema_version=FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
            relationship_attention_enabled=True,
            private_actions_enabled=True,
            public_anonymity_enabled=True,
            activation_policy=COST_AWARE_ACTIVATION_POLICY,
        )

    shared_public = shared["branches"][SHARED_BASELINE_BRANCH]
    branches: dict[str, Any] = {}
    usage_rows: list[Mapping[str, Any]] = [shared_public["usage"]]
    for branch in FOUR_DIMENSION_BRANCHES:
        branch_public = copy.deepcopy(continuations[branch]["branches"][branch])
        branch_public["timeline"] = (
            copy.deepcopy(shared_public["timeline"])
            + branch_public["timeline"][FOUR_DIMENSION_SHARED_END_TICK + 1 :]
        )
        branch_public["world_label"] = FOUR_DIMENSION_BRANCH_LABELS[branch]
        branch_public["shared_prefix_result_sha256"] = shared["result_sha256"]
        branch_public["usage"] = _summed_usage(
            shared_public["usage"], branch_public["usage"]
        )
        branches[branch] = branch_public
        usage_rows.append(continuations[branch]["branches"][branch]["usage"])

    natural_final = branches["Natural"]["timeline"][-1]
    governed_final = branches["D"]["timeline"][-1]
    count_keys = (
        "thread_count",
        "message_count",
        "claim_count",
        "contested_claim_count",
        "corrected_or_verified_claim_count",
        "help_request_count",
        "risk_report_count",
        "correction_count",
        "reply_count",
        "interaction_count",
        "private_conversation_count",
        "private_message_count",
        "private_group_count",
        "anonymous_public_message_count",
    )
    state_difference = {
        key: round(
            float(governed_final["population_state"][key])
            - float(natural_final["population_state"][key]),
            6,
        )
        for key in STATE_KEYS
    }
    natural_risk = {
        row["signal_name"]: float(row["value"])
        for row in natural_final["governance_risk"]["signals"]
    }
    governed_risk = {
        row["signal_name"]: float(row["value"])
        for row in governed_final["governance_risk"]["signals"]
    }
    resident_slots = FOUR_DIMENSION_RESIDENT_BUDGET + (
        len(FOUR_DIMENSION_BRANCHES)
        * (FOUR_DIMENSION_END_TICK - FOUR_DIMENSION_SHARED_END_TICK)
        * FOUR_DIMENSION_RESIDENT_BUDGET
    )
    actor_slots = (
        len(FOUR_DIMENSION_DECISION_TICKS)
        * len(ACTOR_IDS)
        * sum(branch != "Natural" for branch in FOUR_DIMENSION_BRANCHES)
    )
    body = {
        "schema_version": FOUR_DIMENSION_HERO_RESULT_SCHEMA,
        "scenario_id": GOVERNANCE_LEGITIMACY_SCENARIO,
        "scenario_title": "暑期住宿床位分配正当性争议",
        "seed": seed,
        "start_tick": FOUR_DIMENSION_START_TICK,
        "shared_through_tick": FOUR_DIMENSION_SHARED_END_TICK,
        "fork_tick": FOUR_DIMENSION_SHARED_END_TICK + 1,
        "end_tick": FOUR_DIMENSION_END_TICK,
        "branches": branches,
        "worlds_share_exact_T3_checkpoint": True,
        "resident_turn_schema_version": FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
        "model_visible_resident_context": (
            "natural_profile_history_situation_and_visible_forum_context"
        ),
        "model_visible_internal_hashes": False,
        "model_visible_raw_particle_or_stake_values": False,
        "resident_slots": resident_slots,
        "governance_actor_slots": actor_slots,
        "primary_slots": resident_slots + actor_slots,
        "population_size": 1_000,
        "particle_count": 10_000,
        "activation_design": {
            "shared_T3_residents": FOUR_DIMENSION_RESIDENT_BUDGET,
            "branch_tick_residents": FOUR_DIMENSION_RESIDENT_BUDGET,
            "longitudinal_anchors": 8,
            "interaction_or_rotation_PPS_slots": 16,
            "private_origin_quota_of_non_anchor_slots": 0.40,
        },
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "public_feed_personalized": False,
        "relationship_attention_changes_opening_not_public_ranking": True,
        "relationship_attention_enabled": True,
        "private_actions_enabled": True,
        "dynamic_groups_enabled": True,
        "public_anonymity_enabled": True,
        "ecology_calibration_release_sha256": ecology_calibration_release[
            "calibration_release_sha256"
        ],
        "dynamics_public_or_private_messages": 0,
        "public_content_provenance_rule": (
            "authorized_live_llm_or_exact_reviewed_trace"
        ),
        "decision_ticks": sorted(FOUR_DIMENSION_DECISION_TICKS),
        "policy_action_sets": {"Natural": [], "D": list(action_overrides["D"])},
        "scheme_bonus": 0,
        "winner_precommitted": False,
        "paired_final_count_difference_D_minus_Natural": {
            key: int(governed_final.get(key, 0)) - int(natural_final.get(key, 0))
            for key in count_keys
        },
        "paired_final_state_difference_D_minus_Natural": state_difference,
        "paired_final_risk_difference_D_minus_Natural": {
            key: round(governed_risk[key] - natural_risk[key], 6)
            for key in RISK_SIGNAL_NAMES
        },
        "four_dimension_evidence_contract": {
            "forum_ecology": "thread continuation and direct-reply structure",
            "persona_consistency": "independent review of V5 language and next-tick state",
            "population_heterogeneity": "1,000 weighted parent Agents, not particles",
            "governance_response": "strict direct uptake and traceable risk/service mechanisms",
            "all_dimensions_share_one_run": True,
        },
        "usage": _summed_usage(*usage_rows),
        "low_cost_particle_tick_updates": (
            shared["low_cost_particle_tick_updates"]
            + sum(
                value["low_cost_particle_tick_updates"]
                for value in continuations.values()
            )
        ),
        "profile_release_sha256": shared["profile_release_sha256"],
        "relationship_release_sha256": shared["relationship_release_sha256"],
        "scenario_stake_release_sha256": shared[
            "scenario_stake_release_sha256"
        ],
        "formal_full_corpus_profile_release_claimed": shared[
            "formal_full_corpus_profile_release_claimed"
        ],
        "risk_threshold_source": natural_final["governance_risk"][
            "threshold_source"
        ],
        "non_claims": [
            "The 1,000 Agents and all relationships are synthetic.",
            "Natural versus D is a paired model rehearsal, not a causal estimate.",
            "Particles express within-Agent uncertainty and are not people.",
            "No governance world is guaranteed to improve the outcome.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def run_resource_allocation_policy_sandbox(
    *,
    population_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    seed: int = PILOT_SEED,
    reviewed_trace_release_sha256: str | None = None,
    resident_turn_schema_version: str = FORUM_AGENT_TURN_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Run one shared T3-T4 forum and fork three policy worlds at T5.

    This wrapper deliberately reuses the audited Hero kernel.  The exact T4
    particle, memory, message, sampling-rotation, and RNG frontier is cloned;
    the three worlds therefore do not pay for or regenerate their common
    public reaction.  All post-fork visible language still comes from the
    injected LLM/trace source.
    """

    if resident_source is None:
        raise ForumHeroError("resource policy sandbox requires a resident source")
    if governance_source is None:
        governance_source = resident_source
    shared_snapshot: dict[str, Any] = {}

    def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
        shared_snapshot.update(copy.deepcopy(dict(snapshot)))

    shared_budgets = {
        tick: RESOURCE_POLICY_RESIDENT_BUDGET
        for tick in range(
            RESOURCE_POLICY_START_TICK,
            RESOURCE_POLICY_SHARED_END_TICK + 1,
        )
    }
    shared = run_forum_hero_experiment(
        population_release=population_release,
        resident_source=resident_source,
        governance_source=governance_source,
        expected_model=expected_model,
        scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
        seed=seed,
        start_tick=RESOURCE_POLICY_START_TICK,
        end_tick=RESOURCE_POLICY_SHARED_END_TICK,
        branches=(SHARED_BASELINE_BRANCH,),
        resident_budget_by_tick=shared_budgets,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
        snapshot_callback=capture_shared,
        governance_decision_ticks=(),
        resident_turn_schema_version=resident_turn_schema_version,
    )
    if int(shared_snapshot.get("tick", -1)) != RESOURCE_POLICY_SHARED_END_TICK:
        raise ForumHeroError("resource policy sandbox did not commit its shared T4 checkpoint")

    continuation_budgets = {
        tick: RESOURCE_POLICY_RESIDENT_BUDGET
        for tick in range(
            RESOURCE_POLICY_SHARED_END_TICK + 1,
            RESOURCE_POLICY_END_TICK + 1,
        )
    }
    action_overrides = {
        "A": ("verified_update",),
        "D": (
            "verified_update",
            "evidence_dashboard",
            "listening_session",
            "support_channel",
            "queue_status",
            "peer_feedback",
            "bridge_outreach",
        ),
    }
    continuations: dict[str, Any] = {}
    for branch in RESOURCE_POLICY_BRANCHES:
        continuations[branch] = run_forum_hero_experiment(
            population_release=population_release,
            resident_source=resident_source,
            governance_source=governance_source,
            expected_model=expected_model,
            scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
            seed=seed,
            start_tick=RESOURCE_POLICY_SHARED_END_TICK + 1,
            end_tick=RESOURCE_POLICY_END_TICK,
            branches=(branch,),
            resident_budget_by_tick=continuation_budgets,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            initial_snapshot=shared_snapshot,
            governance_decision_ticks=RESOURCE_POLICY_DECISION_TICKS,
            governance_action_overrides=action_overrides,
            resident_turn_schema_version=resident_turn_schema_version,
        )

    shared_public = shared["branches"][SHARED_BASELINE_BRANCH]
    branches: dict[str, Any] = {}
    total_usage_rows: list[Mapping[str, Any]] = [shared_public["usage"]]
    for branch in RESOURCE_POLICY_BRANCHES:
        continuation_public = copy.deepcopy(continuations[branch]["branches"][branch])
        continuation_public["timeline"] = (
            copy.deepcopy(shared_public["timeline"])
            + continuation_public["timeline"][RESOURCE_POLICY_SHARED_END_TICK + 1 :]
        )
        continuation_public["world_label"] = RESOURCE_POLICY_BRANCH_LABELS[branch]
        continuation_public["shared_prefix_result_sha256"] = shared["result_sha256"]
        continuation_public["usage"] = _summed_usage(
            shared_public["usage"], continuation_public["usage"]
        )
        branches[branch] = continuation_public
        total_usage_rows.append(
            continuations[branch]["branches"][branch]["usage"]
        )

    metric_keys = (
        "thread_count",
        "message_count",
        "claim_count",
        "contested_claim_count",
        "corrected_or_verified_claim_count",
        "help_request_count",
        "correction_count",
        "reply_count",
    )
    natural_final = branches["Natural"]["timeline"][-1]
    paired_differences = {
        branch: {
            key: branches[branch]["timeline"][-1][key] - natural_final[key]
            for key in metric_keys
        }
        for branch in ("A", "D")
    }
    body = {
        "schema_version": RESOURCE_POLICY_RESULT_SCHEMA,
        "scenario_id": GOVERNANCE_LEGITIMACY_SCENARIO,
        "scenario_title": "暑期住宿床位分配正当性争议",
        "seed": seed,
        "start_tick": RESOURCE_POLICY_START_TICK,
        "shared_through_tick": RESOURCE_POLICY_SHARED_END_TICK,
        "fork_tick": RESOURCE_POLICY_SHARED_END_TICK + 1,
        "end_tick": RESOURCE_POLICY_END_TICK,
        "world_labels": dict(RESOURCE_POLICY_BRANCH_LABELS),
        "worlds_share_exact_T4_checkpoint": True,
        "shared_prefix_result_sha256": shared["result_sha256"],
        "resident_budget_per_tick": RESOURCE_POLICY_RESIDENT_BUDGET,
        "governance_decision_ticks": sorted(RESOURCE_POLICY_DECISION_TICKS),
        "resident_slots": 480,
        "governance_actor_slots": 12,
        "primary_slots": 492,
        "population_size": 1_000,
        "particle_count": 10_000,
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "public_feed_personalized": False,
        "public_feed_relationship_graph_controlled": False,
        "private_particle_graph_retained": True,
        "dynamics_public_messages": 0,
        "public_content_provenance_rule": "authorized_live_llm_or_exact_reviewed_trace",
        "policy_action_sets": {
            "Natural": [],
            "A": list(action_overrides["A"]),
            "D": list(action_overrides["D"]),
        },
        "scheme_bonus": 0,
        "winner_precommitted": False,
        "branches": branches,
        "paired_final_differences_vs_Natural": paired_differences,
        "usage": _summed_usage(*total_usage_rows),
        "low_cost_particle_tick_updates": (
            shared["low_cost_particle_tick_updates"]
            + sum(
                result["low_cost_particle_tick_updates"]
                for result in continuations.values()
            )
        ),
        "profile_release_sha256": shared["profile_release_sha256"],
        "formal_full_corpus_profile_release_claimed": shared[
            "formal_full_corpus_profile_release_claimed"
        ],
        "non_claims": [
            "This is a synthetic campus policy rehearsal, not a real allocation outcome.",
            "The three worlds are model-conditioned comparisons, not causal estimates.",
            "The hot ranking is a disclosed proxy and the private particle graph never ranks posts.",
            "No policy world is guaranteed to improve the forum response.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def run_resource_allocation_flagship_suite(
    *,
    population_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    reviewed_trace_release_sha256: str | None = None,
    stage: str = "canonical",
    checkpoint_callback: Callable[[int, str, int, Mapping[str, Any]], None]
    | None = None,
) -> dict[str, Any]:
    """Run the four-seed, five-world competition flagship experiment.

    The function deliberately composes :func:`run_forum_hero_experiment`
    instead of introducing another simulation kernel.  Every seed pays once
    for the common T3-T4 forum.  The canonical seed then runs Natural and D on
    every tick through T15, while A/B/C use four semantic checkpoints.  The
    other three paired seeds use the same four-checkpoint semantic spine for
    all five policy worlds.  A canonical-only stage is an exact prefix of the
    full run, so the authorized semantic ledger can resume without repeating
    a settled Provider call.
    """

    if stage not in {"canonical", "full"}:
        raise ForumHeroError("resource flagship stage must be canonical or full")
    if resident_source is None:
        raise ForumHeroError("resource flagship requires a resident source")
    if governance_source is None:
        governance_source = resident_source

    frozen_seeds = (
        RESOURCE_FLAGSHIP_SEEDS[:1]
        if stage == "canonical"
        else RESOURCE_FLAGSHIP_SEEDS
    )
    action_overrides = {
        "A": ("verified_update", "evidence_dashboard", "queue_status"),
        "B": ("listening_session", "support_channel", "peer_feedback"),
        "C": ("targeted_prebunk", "bridge_outreach", "queue_status"),
        "D": tuple(BRANCH_ACTIONS["D"]),
    }
    shared_budgets = {
        tick: RESOURCE_FLAGSHIP_RESIDENT_BUDGET
        for tick in range(
            RESOURCE_FLAGSHIP_START_TICK,
            RESOURCE_FLAGSHIP_SHARED_END_TICK + 1,
        )
    }
    full_budgets = {
        tick: RESOURCE_FLAGSHIP_RESIDENT_BUDGET
        for tick in range(
            RESOURCE_FLAGSHIP_SHARED_END_TICK + 1,
            RESOURCE_FLAGSHIP_END_TICK + 1,
        )
    }
    spine_budgets = {
        tick: (
            RESOURCE_FLAGSHIP_RESIDENT_BUDGET
            if tick in RESOURCE_FLAGSHIP_SPINE_TICKS
            else 0
        )
        for tick in range(
            RESOURCE_FLAGSHIP_SHARED_END_TICK + 1,
            RESOURCE_FLAGSHIP_END_TICK + 1,
        )
    }

    seed_results: list[dict[str, Any]] = []
    total_semantic_slots = 0
    total_particle_updates = 0
    formal_profile_claims: list[bool] = []
    profile_release_sha256: str | None = None
    interval_values: dict[str, dict[str, list[float]]] = {
        branch: {key: [] for key in STATE_KEYS}
        for branch in RESOURCE_FLAGSHIP_BRANCHES
    }
    paired_values: dict[str, dict[str, list[float]]] = {
        branch: {key: [] for key in STATE_KEYS}
        for branch in RESOURCE_FLAGSHIP_BRANCHES
        if branch != "Natural"
    }
    metric_keys = (
        "thread_count",
        "message_count",
        "claim_count",
        "contested_claim_count",
        "corrected_or_verified_claim_count",
        "help_request_count",
        "correction_count",
        "reply_count",
        "interaction_count",
    )

    for seed_position, seed in enumerate(frozen_seeds):
        shared_snapshot: dict[str, Any] = {}

        def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
            shared_snapshot.update(copy.deepcopy(dict(snapshot)))

        shared = run_forum_hero_experiment(
            population_release=population_release,
            resident_source=resident_source,
            governance_source=governance_source,
            expected_model=expected_model,
            scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
            seed=seed,
            start_tick=RESOURCE_FLAGSHIP_START_TICK,
            end_tick=RESOURCE_FLAGSHIP_SHARED_END_TICK,
            branches=(SHARED_BASELINE_BRANCH,),
            resident_budget_by_tick=shared_budgets,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            snapshot_callback=capture_shared,
            activation_selection_salt=canonical_hash(
                ["resource-flagship-shared-sample", seed]
            ),
            tick_snapshot_callback=(
                (
                    lambda branch, tick, snapshot, frozen_seed=seed: checkpoint_callback(
                        frozen_seed, branch, tick, snapshot
                    )
                )
                if checkpoint_callback is not None
                else None
            ),
            governance_decision_ticks=(),
            resident_turn_schema_version=FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        )
        if int(shared_snapshot.get("tick", -1)) != RESOURCE_FLAGSHIP_SHARED_END_TICK:
            raise ForumHeroError(
                "resource flagship did not commit its shared T4 checkpoint"
            )

        branch_parts: dict[str, dict[str, Any]] = {}
        canonical_seed = seed_position == 0
        for branch in RESOURCE_FLAGSHIP_BRANCHES:
            budgets = (
                full_budgets
                if canonical_seed and branch in {"Natural", "D"}
                else spine_budgets
            )
            branch_parts[branch] = run_forum_hero_experiment(
                population_release=population_release,
                resident_source=resident_source,
                governance_source=governance_source,
                expected_model=expected_model,
                scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
                seed=seed,
                start_tick=RESOURCE_FLAGSHIP_SHARED_END_TICK + 1,
                end_tick=RESOURCE_FLAGSHIP_END_TICK,
                branches=(branch,),
                resident_budget_by_tick=budgets,
                reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
                initial_snapshot=shared_snapshot,
                interaction_activation_enabled=(
                    canonical_seed and branch in {"Natural", "D"}
                ),
                activation_selection_salt=canonical_hash(
                    ["resource-flagship-paired-sample", seed]
                ),
                tick_snapshot_callback=(
                    (
                        lambda branch_id, tick, snapshot, frozen_seed=seed: checkpoint_callback(
                            frozen_seed, branch_id, tick, snapshot
                        )
                    )
                    if checkpoint_callback is not None
                    else None
                ),
                governance_decision_ticks=RESOURCE_FLAGSHIP_DECISION_TICKS,
                governance_action_overrides=action_overrides,
                resident_turn_schema_version=FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
            )

        shared_public = shared["branches"][SHARED_BASELINE_BRANCH]
        branches: dict[str, Any] = {}
        total_usage_rows: list[Mapping[str, Any]] = [shared_public["usage"]]
        seed_semantic_slots = _semantic_turn_count(shared)
        seed_particle_updates = int(shared["low_cost_particle_tick_updates"])
        for branch in RESOURCE_FLAGSHIP_BRANCHES:
            part = branch_parts[branch]
            branch_public = copy.deepcopy(part["branches"][branch])
            branch_public["timeline"] = (
                copy.deepcopy(shared_public["timeline"])
                + branch_public["timeline"][RESOURCE_FLAGSHIP_SHARED_END_TICK + 1 :]
            )
            branch_public["world_label"] = RESOURCE_FLAGSHIP_BRANCH_LABELS[branch]
            branch_public["shared_prefix_result_sha256"] = shared["result_sha256"]
            branch_public["semantic_schedule"] = (
                "every_tick"
                if canonical_seed and branch in {"Natural", "D"}
                else "T5_T9_T13_T15_spine"
            )
            branch_public["usage"] = _summed_usage(
                shared_public["usage"], branch_public["usage"]
            )
            branches[branch] = branch_public
            total_usage_rows.append(part["branches"][branch]["usage"])
            seed_semantic_slots += _semantic_turn_count(part)
            seed_particle_updates += int(part["low_cost_particle_tick_updates"])

        natural_final = branches["Natural"]["timeline"][-1]
        paired_metrics = {
            branch: {
                key: branches[branch]["timeline"][-1].get(key, 0)
                - natural_final.get(key, 0)
                for key in metric_keys
            }
            for branch in RESOURCE_FLAGSHIP_BRANCHES
            if branch != "Natural"
        }
        for branch in RESOURCE_FLAGSHIP_BRANCHES:
            final_state = branches[branch]["timeline"][-1]["population_state"]
            for key in STATE_KEYS:
                value = float(final_state[key])
                interval_values[branch][key].append(value)
                if branch != "Natural":
                    paired_values[branch][key].append(
                        value - float(natural_final["population_state"][key])
                    )

        seed_profile_sha = str(shared["profile_release_sha256"])
        if profile_release_sha256 is None:
            profile_release_sha256 = seed_profile_sha
        elif seed_profile_sha != profile_release_sha256:
            raise ForumHeroError("resource flagship profile release drifted across seeds")
        formal_profile_claims.append(
            bool(shared["formal_full_corpus_profile_release_claimed"])
        )
        seed_results.append(
            {
                "seed": seed,
                "canonical_full_forum_seed": canonical_seed,
                "shared_prefix_result_sha256": shared["result_sha256"],
                "semantic_slots": seed_semantic_slots,
                "particle_updates": seed_particle_updates,
                "branches": branches,
                "paired_final_metric_differences_vs_Natural": paired_metrics,
                "usage": _summed_usage(*total_usage_rows),
                "seed_result_sha256": canonical_hash(
                    {
                        "seed": seed,
                        "branch_result_sha256s": {
                            branch: canonical_hash(branches[branch])
                            for branch in RESOURCE_FLAGSHIP_BRANCHES
                        },
                    }
                ),
            }
        )
        total_semantic_slots += seed_semantic_slots
        total_particle_updates += seed_particle_updates

    expected_slots = 900 if stage == "canonical" else 2_592
    if total_semantic_slots != expected_slots:
        raise ForumHeroError(
            "resource flagship semantic slot count drifted: "
            f"{total_semantic_slots} != {expected_slots}"
        )
    body = {
        "schema_version": RESOURCE_FLAGSHIP_RESULT_SCHEMA,
        "scenario_id": GOVERNANCE_LEGITIMACY_SCENARIO,
        "scenario_title": "暑期住宿床位分配正当性争议",
        "stage": stage,
        "scope_complete": stage == "full",
        "seeds": list(frozen_seeds),
        "canonical_seed": RESOURCE_FLAGSHIP_SEEDS[0],
        "branches": list(RESOURCE_FLAGSHIP_BRANCHES),
        "branch_labels": dict(RESOURCE_FLAGSHIP_BRANCH_LABELS),
        "start_tick": RESOURCE_FLAGSHIP_START_TICK,
        "shared_through_tick": RESOURCE_FLAGSHIP_SHARED_END_TICK,
        "fork_tick": RESOURCE_FLAGSHIP_SHARED_END_TICK + 1,
        "end_tick": RESOURCE_FLAGSHIP_END_TICK,
        "decision_ticks": sorted(RESOURCE_FLAGSHIP_DECISION_TICKS),
        "semantic_spine_ticks": sorted(RESOURCE_FLAGSHIP_SPINE_TICKS),
        "resident_budget_per_active_world_tick": RESOURCE_FLAGSHIP_RESIDENT_BUDGET,
        "maximum_resident_llm_turns_per_world_tick": 24,
        "maximum_resident_plus_governance_turns_per_world_tick": 27,
        "resident_slots": 864 if stage == "canonical" else 2_448,
        "governance_actor_slots": 36 if stage == "canonical" else 144,
        "primary_slots": total_semantic_slots,
        "full_scope_primary_slots": 2_592,
        "population_size": 1_000,
        "particle_count": 10_000,
        "low_cost_particle_tick_updates": total_particle_updates,
        "resident_turn_schema_version": FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        "parallel_speech_and_engagement_enabled": True,
        "activation_design": {
            "anchors_per_round": 16,
            "rotating_or_interaction_slots_per_round": 8,
            "paired_semantic_spines_share_selection_salt": True,
            "interaction_priority_applies_only_to_canonical_Natural_and_D": True,
            "record_level_activation_identities_public": False,
        },
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "public_feed_personalized": False,
        "relationship_contract": (
            "reply, quote, co-participation, notification, trust and attention; "
            "the private particle graph never re-ranks the public board"
        ),
        "public_content_provenance_rule": (
            "authorized_live_llm_or_exact_reviewed_trace"
        ),
        "dynamics_public_messages": 0,
        "policy_action_sets": {
            "Natural": [],
            **{branch: list(actions) for branch, actions in action_overrides.items()},
        },
        "scheme_bonus": 0,
        "winner_precommitted": False,
        "seed_results": seed_results,
        "branch_final_state_intervals_95": {
            branch: {
                key: _mean_interval(values)
                for key, values in metrics.items()
            }
            for branch, metrics in interval_values.items()
        },
        "paired_state_difference_intervals_95_vs_Natural": {
            branch: {
                key: _mean_interval(values)
                for key, values in metrics.items()
            }
            for branch, metrics in paired_values.items()
        },
        "profile_release_sha256": profile_release_sha256,
        "formal_full_corpus_profile_release_claimed": bool(
            formal_profile_claims and all(formal_profile_claims)
        ),
        "publication_eligible": False,
        "publication_blocker": (
            "full_live_scope_and_human_quality_review_pending"
        ),
        "non_claims": [
            "The 1,000 agents are synthetic forum-behaviour agents, not students.",
            "The paired policy worlds are model-conditioned rehearsals, not causal estimates.",
            "The four-seed intervals describe simulation sensitivity, not school-wide opinion.",
            "The experiment does not precommit a winning governance scheme.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def run_century_gym_policy_sandbox(
    *,
    population_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    seed: int = PILOT_SEED,
    reviewed_trace_release_sha256: str | None = None,
    resident_turn_schema_version: str = FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    public_tick_callback: Callable[[str, int, Mapping[str, Any]], None]
    | None = None,
) -> dict[str, Any]:
    """Run the live Century Gym Natural/D rehearsal from one exact T4 fork.

    T3-T4 are paid for once and committed as a shared student-origin forum.
    Natural and D then clone the complete particle, memory, forum, sampling and
    RNG frontier.  The D world may create only bounded evidence cards, service
    tickets and cross-sport/time-window outreach through the existing three
    governance LLM actors.  No outcome direction is hard-coded.
    """

    if resident_source is None:
        raise ForumHeroError("Century Gym sandbox requires a resident source")
    if governance_source is None:
        governance_source = resident_source
    if resident_turn_schema_version != FORUM_AGENT_TURN_V2_SCHEMA_VERSION:
        raise ForumHeroError(
            "Century Gym requires the parallel speech-and-engagement turn contract"
        )

    shared_snapshot: dict[str, Any] = {}

    def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
        shared_snapshot.update(copy.deepcopy(dict(snapshot)))

    shared_budgets = {
        tick: CENTURY_GYM_RESIDENT_BUDGET
        for tick in range(
            CENTURY_GYM_START_TICK,
            CENTURY_GYM_SHARED_END_TICK + 1,
        )
    }
    shared = run_forum_hero_experiment(
        population_release=population_release,
        resident_source=resident_source,
        governance_source=governance_source,
        expected_model=expected_model,
        scenario_id=CENTURY_GYM_SCENARIO,
        seed=seed,
        start_tick=CENTURY_GYM_START_TICK,
        end_tick=CENTURY_GYM_SHARED_END_TICK,
        branches=(SHARED_BASELINE_BRANCH,),
        resident_budget_by_tick=shared_budgets,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
        snapshot_callback=capture_shared,
        public_tick_callback=public_tick_callback,
        governance_decision_ticks=(),
        resident_turn_schema_version=resident_turn_schema_version,
    )
    if int(shared_snapshot.get("tick", -1)) != CENTURY_GYM_SHARED_END_TICK:
        raise ForumHeroError(
            "Century Gym sandbox did not commit its shared T4 checkpoint"
        )

    continuation_budgets = {
        tick: CENTURY_GYM_RESIDENT_BUDGET
        for tick in range(
            CENTURY_GYM_SHARED_END_TICK + 1,
            CENTURY_GYM_END_TICK + 1,
        )
    }
    action_overrides = {
        "D": (
            "verified_update",
            "evidence_dashboard",
            "support_channel",
            "queue_status",
            "peer_feedback",
            "bridge_outreach",
        ),
    }
    continuations: dict[str, Any] = {}
    for branch in CENTURY_GYM_BRANCHES:
        continuations[branch] = run_forum_hero_experiment(
            population_release=population_release,
            resident_source=resident_source,
            governance_source=governance_source,
            expected_model=expected_model,
            scenario_id=CENTURY_GYM_SCENARIO,
            seed=seed,
            start_tick=CENTURY_GYM_SHARED_END_TICK + 1,
            end_tick=CENTURY_GYM_END_TICK,
            branches=(branch,),
            resident_budget_by_tick=continuation_budgets,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            initial_snapshot=shared_snapshot,
            governance_decision_ticks=CENTURY_GYM_DECISION_TICKS,
            governance_action_overrides=action_overrides,
            public_tick_callback=public_tick_callback,
            resident_turn_schema_version=resident_turn_schema_version,
        )

    shared_public = shared["branches"][SHARED_BASELINE_BRANCH]
    branches: dict[str, Any] = {}
    total_usage_rows: list[Mapping[str, Any]] = [shared_public["usage"]]
    for branch in CENTURY_GYM_BRANCHES:
        continuation_public = copy.deepcopy(
            continuations[branch]["branches"][branch]
        )
        continuation_public["timeline"] = (
            copy.deepcopy(shared_public["timeline"])
            + continuation_public["timeline"][CENTURY_GYM_SHARED_END_TICK + 1 :]
        )
        continuation_public["world_label"] = CENTURY_GYM_BRANCH_LABELS[branch]
        continuation_public["shared_prefix_result_sha256"] = shared[
            "result_sha256"
        ]
        continuation_public["usage"] = _summed_usage(
            shared_public["usage"], continuation_public["usage"]
        )
        branches[branch] = continuation_public
        total_usage_rows.append(
            continuations[branch]["branches"][branch]["usage"]
        )

    metric_keys = (
        "thread_count",
        "message_count",
        "claim_count",
        "contested_claim_count",
        "corrected_or_verified_claim_count",
        "help_request_count",
        "correction_count",
        "reply_count",
        "interaction_count",
    )
    natural_final = branches["Natural"]["timeline"][-1]
    governed_final = branches["D"]["timeline"][-1]
    body = {
        "schema_version": CENTURY_GYM_RESULT_SCHEMA,
        "scenario_id": CENTURY_GYM_SCENARIO,
        "scenario_title": "世纪馆场地‘幽灵预约’争议",
        "scenario_entry": (
            "学生在论坛报告世纪馆空场与小程序不可约同时出现，并询问羽毛球、"
            "乒乓球等项目是否存在取消未释放、到点未到场或不透明占位。"
        ),
        "seed": seed,
        "start_tick": CENTURY_GYM_START_TICK,
        "shared_through_tick": CENTURY_GYM_SHARED_END_TICK,
        "fork_tick": CENTURY_GYM_SHARED_END_TICK + 1,
        "end_tick": CENTURY_GYM_END_TICK,
        "world_labels": dict(CENTURY_GYM_BRANCH_LABELS),
        "worlds_share_exact_T4_checkpoint": True,
        "shared_prefix_result_sha256": shared["result_sha256"],
        "resident_budget_per_tick": CENTURY_GYM_RESIDENT_BUDGET,
        "governance_decision_ticks": sorted(CENTURY_GYM_DECISION_TICKS),
        "resident_slots": 224,
        "governance_actor_slots": 6,
        "primary_slots": 230,
        "population_size": 1_000,
        "particle_count": 10_000,
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "public_feed_personalized": False,
        "parallel_engagements_enabled": True,
        "dynamics_public_messages": 0,
        "public_content_provenance_rule": (
            "authorized_live_llm_or_exact_reviewed_trace"
        ),
        "scenario_weight_basis": scenario_weight_key(CENTURY_GYM_SCENARIO),
        "scenario_specific_empirical_weight_claimed": False,
        "policy_action_sets": {
            "Natural": [],
            "D": list(action_overrides["D"]),
        },
        "policy_objects": [
            "bounded_booking_evidence_card",
            "unreleased_slot_service_ticket",
            "cross_sport_time_window_outreach",
        ],
        "scheme_bonus": 0,
        "winner_precommitted": False,
        "branches": branches,
        "paired_final_difference_D_minus_Natural": {
            key: governed_final[key] - natural_final[key]
            for key in metric_keys
        },
        "usage": _summed_usage(*total_usage_rows),
        "low_cost_particle_tick_updates": (
            shared["low_cost_particle_tick_updates"]
            + sum(
                result["low_cost_particle_tick_updates"]
                for result in continuations.values()
            )
        ),
        "profile_release_sha256": shared["profile_release_sha256"],
        "formal_full_corpus_profile_release_claimed": shared[
            "formal_full_corpus_profile_release_claimed"
        ],
        "non_claims": [
            "The Century Gym event and every forum identity are synthetic.",
            "The population weights reuse a reviewed campus-services proxy; they are not Century Gym survey weights.",
            "Natural versus D is a paired model rehearsal, not a causal estimate or a real repair outcome.",
            "A service ticket may remain pending; acceptance never implies that a defect was fixed.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def run_cost_bounded_forum_vignette(
    *,
    population_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    scenario_id: str = PILOT_SCENARIO,
    seed: int = PILOT_SEED,
    reviewed_trace_release_sha256: str | None = None,
    public_tick_callback: Callable[[str, int, Mapping[str, Any]], None]
    | None = None,
) -> dict[str, Any]:
    """Run a 118-slot, shared-baseline forum vignette for live product use.

    The mode is deliberately small enough for an operator-facing rehearsal:
    sixteen residents establish one shared Tick-3 forum, then Natural and D
    clone the exact checkpoint for Ticks 4--6.  D receives two three-actor
    decisions.  It reuses the formal ForumTwin kernel, V2 resident contract,
    checkpoints and public-tick callback; no public text is generated locally.
    """

    if scenario_id not in HERO_RUNTIME_SCENARIOS:
        raise ForumHeroError("quick live vignette scenario is invalid")
    if resident_source is None:
        raise ForumHeroError("quick live vignette requires a resident source")
    if governance_source is None:
        governance_source = resident_source
    shared_snapshot: dict[str, Any] = {}

    def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
        shared_snapshot.update(copy.deepcopy(dict(snapshot)))

    shared = run_forum_hero_experiment(
        population_release=population_release,
        resident_source=resident_source,
        governance_source=governance_source,
        expected_model=expected_model,
        scenario_id=scenario_id,
        seed=seed,
        start_tick=QUICK_LIVE_START_TICK,
        end_tick=QUICK_LIVE_SHARED_END_TICK,
        branches=(SHARED_BASELINE_BRANCH,),
        resident_budget_by_tick={
            QUICK_LIVE_START_TICK: QUICK_LIVE_RESIDENT_BUDGET
        },
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
        snapshot_callback=capture_shared,
        public_tick_callback=public_tick_callback,
        governance_decision_ticks=(),
        resident_turn_schema_version=FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        require_scenario_entry_post=True,
    )
    if int(shared_snapshot.get("tick", -1)) != QUICK_LIVE_SHARED_END_TICK:
        raise ForumHeroError("quick live vignette did not commit its shared checkpoint")

    continuation_budgets = {
        tick: QUICK_LIVE_RESIDENT_BUDGET
        for tick in range(QUICK_LIVE_SHARED_END_TICK + 1, QUICK_LIVE_END_TICK + 1)
    }
    action_overrides = {
        "D": (
            "verified_update",
            "evidence_dashboard",
            "listening_session",
            "support_channel",
            "queue_status",
            "peer_feedback",
            "targeted_prebunk",
            "bridge_outreach",
        )
    }
    continuations: dict[str, Any] = {}
    for branch in QUICK_LIVE_BRANCHES:
        continuations[branch] = run_forum_hero_experiment(
            population_release=population_release,
            resident_source=resident_source,
            governance_source=governance_source,
            expected_model=expected_model,
            scenario_id=scenario_id,
            seed=seed,
            start_tick=QUICK_LIVE_SHARED_END_TICK + 1,
            end_tick=QUICK_LIVE_END_TICK,
            branches=(branch,),
            resident_budget_by_tick=continuation_budgets,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            initial_snapshot=shared_snapshot,
            governance_decision_ticks=QUICK_LIVE_DECISION_TICKS,
            governance_action_overrides=action_overrides,
            public_tick_callback=public_tick_callback,
            resident_turn_schema_version=FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        )

    shared_public = shared["branches"][SHARED_BASELINE_BRANCH]
    branches: dict[str, Any] = {}
    usage_rows: list[Mapping[str, Any]] = [shared_public["usage"]]
    for branch in QUICK_LIVE_BRANCHES:
        continuation_public = copy.deepcopy(continuations[branch]["branches"][branch])
        continuation_public["timeline"] = (
            copy.deepcopy(shared_public["timeline"])
            + continuation_public["timeline"][QUICK_LIVE_SHARED_END_TICK + 1 :]
        )
        continuation_public["world_label"] = QUICK_LIVE_BRANCH_LABELS[branch]
        continuation_public["shared_prefix_result_sha256"] = shared["result_sha256"]
        continuation_public["usage"] = _summed_usage(
            shared_public["usage"], continuation_public["usage"]
        )
        branches[branch] = continuation_public
        usage_rows.append(continuations[branch]["branches"][branch]["usage"])

    metric_keys = (
        "thread_count",
        "message_count",
        "claim_count",
        "contested_claim_count",
        "corrected_or_verified_claim_count",
        "help_request_count",
        "correction_count",
        "reply_count",
        "interaction_count",
    )
    natural_final = branches["Natural"]["timeline"][-1]
    governed_final = branches["D"]["timeline"][-1]
    resident_slots = QUICK_LIVE_RESIDENT_BUDGET + (
        (QUICK_LIVE_END_TICK - QUICK_LIVE_SHARED_END_TICK)
        * len(QUICK_LIVE_BRANCHES)
        * QUICK_LIVE_RESIDENT_BUDGET
    )
    actor_slots = len(QUICK_LIVE_DECISION_TICKS) * len(ACTOR_IDS)
    titles = {
        PILOT_SCENARIO: "讲座辱骂学生事件",
        GOVERNANCE_LEGITIMACY_SCENARIO: "暑期住宿床位分配正当性争议",
        CENTURY_GYM_SCENARIO: "世纪馆场地‘幽灵预约’争议",
    }
    scenario_context = scenario_context_for_tick(
        QUICK_LIVE_START_TICK, scenario_id
    )
    body = {
        "schema_version": QUICK_LIVE_RESULT_SCHEMA,
        "scenario_id": scenario_id,
        "scenario_title": titles[scenario_id],
        "scenario_entry": str(scenario_context.get("event_brief") or ""),
        "seed": seed,
        "start_tick": QUICK_LIVE_START_TICK,
        "shared_through_tick": QUICK_LIVE_SHARED_END_TICK,
        "fork_tick": QUICK_LIVE_SHARED_END_TICK + 1,
        "end_tick": QUICK_LIVE_END_TICK,
        "world_labels": dict(QUICK_LIVE_BRANCH_LABELS),
        "worlds_share_exact_checkpoint": True,
        "scenario_entry_post_required": True,
        "shared_prefix_result_sha256": shared["result_sha256"],
        "resident_budget_per_tick": QUICK_LIVE_RESIDENT_BUDGET,
        "governance_decision_ticks": sorted(QUICK_LIVE_DECISION_TICKS),
        "resident_slots": resident_slots,
        "governance_actor_slots": actor_slots,
        "primary_slots": resident_slots + actor_slots,
        "population_size": 1_000,
        "particle_count": 10_000,
        "public_feed_mechanism": "global_hot_top10_plus_latest_v1",
        "public_feed_personalized": False,
        "parallel_engagements_enabled": True,
        "dynamics_public_messages": 0,
        "public_content_provenance_rule": (
            "authorized_live_llm_or_exact_reviewed_trace"
        ),
        "policy_action_sets": {"Natural": [], "D": list(action_overrides["D"])},
        "scheme_bonus": 0,
        "winner_precommitted": False,
        "branches": branches,
        "paired_final_difference_D_minus_Natural": {
            key: governed_final[key] - natural_final[key] for key in metric_keys
        },
        "usage": _summed_usage(*usage_rows),
        "low_cost_particle_tick_updates": (
            shared["low_cost_particle_tick_updates"]
            + sum(
                result["low_cost_particle_tick_updates"]
                for result in continuations.values()
            )
        ),
        "profile_release_sha256": shared["profile_release_sha256"],
        "formal_full_corpus_profile_release_claimed": shared[
            "formal_full_corpus_profile_release_claimed"
        ],
        "product_scope": "cost_bounded_live_vignette",
        "non_claims": [
            "This is a short synthetic policy rehearsal, not a real-world outcome.",
            "The paired worlds share one checkpoint but do not identify causal effects.",
            "A six-tick vignette is for mechanism inspection, not final policy ranking.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    return body


def _semantic_turn_count(result: Mapping[str, Any]) -> int:
    resident = sum(int(value) for value in result["resident_budget_by_tick"].values())
    resident *= len(result["branches_executed"])
    actors = sum(
        int(point.get("governance_llm_activations", 0))
        for branch in result["branches"].values()
        for point in branch["timeline"]
    )
    return resident + actors


def _mean_interval(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ForumHeroError("paired suite interval cannot be empty")
    mean = sum(float(value) for value in values) / len(values)
    variance = (
        sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    radius = 1.96 * math.sqrt(variance / len(values))
    return {
        "mean": round(mean, 6),
        "lower": round(mean - radius, 6),
        "upper": round(mean + radius, 6),
    }


def run_forum_twin_paired_suite(
    *,
    population_release: Mapping[str, Any],
    resident_source: SemanticTurnSource,
    governance_source: SemanticTurnSource | None = None,
    expected_model: str = PILOT_MODEL,
    scenarios: Sequence[str] = FORUM_TWIN_SCENARIOS,
    seeds: Sequence[int] = FORUM_TWIN_FORMAL_SEEDS,
    reviewed_trace_release_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute the frozen 16,544-slot, two-scenario paired ForumTwin suite.

    The first seed receives full tick-by-tick Natural/D resident activation;
    A/B/C and the remaining seven seeds use the semantic spine while all five
    branches still advance the same 10,000-particle background every tick.
    """

    frozen_scenarios = tuple(str(value) for value in scenarios)
    frozen_seeds = tuple(int(value) for value in seeds)
    if frozen_scenarios != tuple(FORUM_TWIN_SCENARIOS):
        raise ForumHeroError("paired suite requires both frozen scenarios")
    if len(frozen_seeds) != 8 or len(set(frozen_seeds)) != 8:
        raise ForumHeroError("paired suite requires eight distinct seeds")
    if governance_source is None:
        governance_source = resident_source
    total_semantic_turns = 0
    total_particle_updates = 0
    scenario_outputs: dict[str, Any] = {}

    for scenario_id in frozen_scenarios:
        seed_summaries: list[dict[str, Any]] = []
        canonical_results: dict[str, Any] | None = None
        final_states: dict[str, dict[str, list[float]]] = {
            branch: {key: [] for key in STATE_KEYS} for branch in FORUM_BRANCHES
        }
        paired_differences: dict[str, dict[str, list[float]]] = {
            branch: {key: [] for key in STATE_KEYS}
            for branch in FORUM_BRANCHES
            if branch != "Natural"
        }
        for seed_ordinal, seed in enumerate(frozen_seeds):
            snapshot_box: dict[str, Mapping[str, Any]] = {}
            baseline_start = 0
            baseline_budgets = (
                {0: 24, 1: 24, 2: 48}
                if seed_ordinal == 0
                else {0: 0, 1: 0, 2: 48}
            )
            baseline = run_forum_hero_experiment(
                population_release=population_release,
                resident_source=resident_source,
                governance_source=governance_source,
                expected_model=expected_model,
                scenario_id=scenario_id,
                seed=seed,
                start_tick=baseline_start,
                end_tick=2,
                branches=(SHARED_BASELINE_BRANCH,),
                resident_budget_by_tick=baseline_budgets,
                reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                snapshot_callback=lambda _branch, value, box=snapshot_box: box.update(
                    {"shared": value}
                ),
                interaction_activation_enabled=False,
            )
            total_semantic_turns += _semantic_turn_count(baseline)
            total_particle_updates += int(baseline["low_cost_particle_tick_updates"])
            shared_snapshot = snapshot_box.get("shared")
            if shared_snapshot is None:
                raise ForumHeroError("paired suite did not capture shared T2 state")

            spine_budgets = {
                tick: 48 if tick in {7, 13, 19} else 0
                for tick in range(3, 24)
            }
            run_parts: list[Mapping[str, Any]] = []
            if seed_ordinal == 0:
                hero = run_forum_hero_experiment(
                    population_release=population_release,
                    resident_source=resident_source,
                    governance_source=governance_source,
                    expected_model=expected_model,
                    scenario_id=scenario_id,
                    seed=seed,
                    start_tick=3,
                    end_tick=23,
                    branches=("Natural", "D"),
                    resident_budget_by_tick={
                        tick: (
                            64 if tick <= 7 else 48 if tick <= 17 else 32
                        )
                        for tick in range(3, 24)
                    },
                    reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                    initial_snapshot=shared_snapshot,
                )
                restricted = run_forum_hero_experiment(
                    population_release=population_release,
                    resident_source=resident_source,
                    governance_source=governance_source,
                    expected_model=expected_model,
                    scenario_id=scenario_id,
                    seed=seed,
                    start_tick=3,
                    end_tick=23,
                    branches=("A", "B", "C"),
                    resident_budget_by_tick=spine_budgets,
                    reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                    initial_snapshot=shared_snapshot,
                    interaction_activation_enabled=False,
                )
                run_parts.extend((hero, restricted))
                canonical_results = {
                    "shared_baseline": baseline,
                    "natural_and_adaptive": hero,
                    "restricted_policies": restricted,
                }
            else:
                spine = run_forum_hero_experiment(
                    population_release=population_release,
                    resident_source=resident_source,
                    governance_source=governance_source,
                    expected_model=expected_model,
                    scenario_id=scenario_id,
                    seed=seed,
                    start_tick=3,
                    end_tick=23,
                    branches=FORUM_BRANCHES,
                    resident_budget_by_tick=spine_budgets,
                    reviewed_trace_release_sha256=reviewed_trace_release_sha256,
                    initial_snapshot=shared_snapshot,
                    interaction_activation_enabled=False,
                )
                run_parts.append(spine)

            branch_rows: dict[str, Any] = {}
            for part in run_parts:
                total_semantic_turns += _semantic_turn_count(part)
                total_particle_updates += int(part["low_cost_particle_tick_updates"])
                for branch, branch_result in part["branches"].items():
                    branch_rows[branch] = branch_result
            if set(branch_rows) != set(FORUM_BRANCHES):
                raise ForumHeroError("paired suite did not complete all five branches")
            natural_state = branch_rows["Natural"]["timeline"][-1]["population_state"]
            for branch in FORUM_BRANCHES:
                state = branch_rows[branch]["timeline"][-1]["population_state"]
                for key in STATE_KEYS:
                    final_states[branch][key].append(float(state[key]))
                    if branch != "Natural":
                        paired_differences[branch][key].append(
                            float(state[key]) - float(natural_state[key])
                        )
            seed_summaries.append(
                {
                    "seed": seed,
                    "branch_result_sha256": {
                        branch: canonical_hash(branch_rows[branch])
                        for branch in FORUM_BRANCHES
                    },
                    "final_population_state": {
                        branch: branch_rows[branch]["timeline"][-1]["population_state"]
                        for branch in FORUM_BRANCHES
                    },
                    "governance_round_count": {
                        branch: len(branch_rows[branch]["governance_rounds"])
                        for branch in FORUM_BRANCHES
                    },
                }
            )

        scenario_outputs[scenario_id] = {
            "seed_count": len(frozen_seeds),
            "seed_summaries": seed_summaries,
            "branch_intervals_95": {
                branch: {
                    key: _mean_interval(values)
                    for key, values in metrics.items()
                }
                for branch, metrics in final_states.items()
            },
            "paired_difference_intervals_95": {
                branch: {
                    key: _mean_interval(values)
                    for key, values in metrics.items()
                }
                for branch, metrics in paired_differences.items()
            },
            "canonical_results": canonical_results,
        }

    if total_semantic_turns != 16_544:
        raise ForumHeroError(
            f"formal paired suite semantic slot count drifted: {total_semantic_turns}"
        )
    if total_particle_updates != 17_280_000:
        raise ForumHeroError(
            f"formal paired suite particle update count drifted: {total_particle_updates}"
        )
    body = {
        "schema_version": "campus-pulse-forum-twin-paired-suite-v1",
        "scenarios": list(frozen_scenarios),
        "seeds": list(frozen_seeds),
        "branches": list(FORUM_BRANCHES),
        "semantic_slots": total_semantic_turns,
        "particle_updates": total_particle_updates,
        "population_size": 1_000,
        "particle_count": 10_000,
        "maximum_resident_turns_per_tick": 64,
        "maximum_total_llm_turns_per_tick": 67,
        "dynamics_public_messages": 0,
        "scenarios_results": scenario_outputs,
        "publication_eligible": False,
        "formal_full_corpus_profile_release_claimed": bool(
            population_release.get("bridge_from_v2") is True
            and population_release.get("full_corpus_profile_v2_bound") is True
            and population_release.get("source_population_schema_version")
            == "semantic-llm-forum-agent-population-v2"
        ),
        "publication_blocker": (
            "reviewed_forum_trace_and_human_quality_gates_incomplete"
        ),
    }
    body["result_sha256"] = canonical_hash(body)
    return body
