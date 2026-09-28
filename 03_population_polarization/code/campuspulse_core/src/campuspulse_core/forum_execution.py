"""Zero-network execution glue for one ForumTwin resident activation round.

The function in this module is deliberately below Store/worker concerns.  It
renders every selected slot, persists the design through a caller callback,
acquires an injected live/exact-replay envelope, persists response metadata,
strictly validates and transactionally materializes the resulting forum batch.
Committed attempts can be resumed without another acquire or a second semantic
integration.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import re
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .forum_turns import (
    FORUM_AGENT_TURN_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    V5_FULL_NATURAL_PROFILE,
    V5_MODEL_VIEW_CONDITIONS,
    OUTPUT_FIELDS,
    ForumTurnError,
    RenderedForumAgentTurn,
    RenderedForumAgentTurnV2,
    RenderedForumAgentTurnV3,
    RenderedForumAgentTurnV4,
    RenderedForumAgentTurnV5,
    ValidatedForumAgentTurnV5,
    materialize_forum_agent_turn_v1,
    materialize_forum_agent_turn_v2,
    materialize_forum_agent_turn_v3,
    materialize_forum_agent_turn_v4,
    materialize_forum_agent_turn_v5,
    render_forum_agent_turn_v1,
    render_forum_agent_turn_v2,
    render_forum_agent_turn_v3,
    render_forum_agent_turn_v4,
    render_forum_agent_turn_v5,
    validate_forum_agent_turn_v1,
    validate_forum_agent_turn_v2,
    validate_forum_agent_turn_v3,
    validate_forum_agent_turn_v4,
    validate_forum_agent_turn_v5,
)
from .forum_twin import (
    PUBLIC_PROVENANCE_KINDS,
    AgentMemory,
    FeedExposure,
    ForumActivationRound,
    ForumClaim,
    ForumInteraction,
    ForumLedger,
    ForumMessage,
    ForumThread,
    ForumTwinError,
    resident_turn_budget,
)
from .mass_population import canonical_hash
from .yulan_scale import SemanticTurnSource, TurnEnvelope


ROUND_EXECUTION_SCHEMA = "campus-pulse-forum-resident-round-execution-v1"
ROUND_EXECUTION_V2_SCHEMA = "campus-pulse-forum-resident-round-execution-v2"
ROUND_EXECUTION_V3_SCHEMA = "campus-pulse-forum-resident-round-execution-v3"
ROUND_EXECUTION_V4_SCHEMA = "campus-pulse-forum-resident-round-execution-v4"
ROUND_EXECUTION_V5_SCHEMA = "campus-pulse-forum-resident-round-execution-v5"
SLOT_DESIGN_SCHEMA = "campus-pulse-forum-resident-slot-design-v1"
SLOT_DESIGN_V2_SCHEMA = "campus-pulse-forum-resident-slot-design-v2"
SLOT_DESIGN_V3_SCHEMA = "campus-pulse-forum-resident-slot-design-v3"
SLOT_DESIGN_V4_SCHEMA = "campus-pulse-forum-resident-slot-design-v4"
SLOT_DESIGN_V5_SCHEMA = "campus-pulse-forum-resident-slot-design-v5"
ATTEMPT_SCHEMA = "campus-pulse-forum-resident-attempt-v1"
ATTEMPT_V2_SCHEMA = "campus-pulse-forum-resident-attempt-v2"
ATTEMPT_V3_SCHEMA = "campus-pulse-forum-resident-attempt-v3"
ATTEMPT_V4_SCHEMA = "campus-pulse-forum-resident-attempt-v4"
ATTEMPT_V5_SCHEMA = "campus-pulse-forum-resident-attempt-v5"
THREAD_BATCH_SCHEMA = "campus-pulse-forum-thread-batch-v1"
THREAD_BATCH_V2_SCHEMA = "campus-pulse-forum-thread-batch-v2"
THREAD_BATCH_V3_SCHEMA = "campus-pulse-forum-thread-batch-v3"
THREAD_BATCH_V4_SCHEMA = "campus-pulse-forum-thread-batch-v4"
THREAD_BATCH_V5_SCHEMA = "campus-pulse-forum-thread-batch-v5"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ForumExecutionError(ForumTwinError):
    """Raised when a resident activation round cannot be executed safely."""


def _selection_role(round_: ForumActivationRound, index: int) -> str:
    if index in round_.anchor_indices:
        return "anchor"
    if index in round_.interaction_indices:
        return "interaction"
    if index in round_.pps_indices:
        return "pps"
    raise ForumExecutionError("resident index is not part of the activation round")


def _slot_id(
    *,
    scenario_id: str,
    branch: str,
    tick: int,
    agent_id: str,
    selection_role: str,
) -> str:
    return "forum-resident-slot-" + canonical_hash(
        [scenario_id, branch, tick, agent_id, selection_role]
    )[:32]


def _strict_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ForumExecutionError("resident execution payload is not strict JSON") from exc


_SUPPORTED_TURN_SCHEMAS = {
    FORUM_AGENT_TURN_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
}


def _schema_for_turn(turn_schema_version: str, values: Mapping[str, str]) -> str:
    try:
        return values[turn_schema_version]
    except KeyError as exc:
        raise ForumExecutionError("resident turn schema is unsupported") from exc


def _turn_has_parallel_engagements(turn_schema_version: str) -> bool:
    return turn_schema_version != FORUM_AGENT_TURN_SCHEMA_VERSION


def _turn_has_private_channel(turn_schema_version: str) -> bool:
    return turn_schema_version in {
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }


def _decision_artifact(validated: Any) -> dict[str, Any]:
    # V1/V2 recovery must persist exactly the public decision contract.  The
    # shared decision dataclass also carries newer private-channel metadata;
    # leaking those optional fields into a legacy artifact makes a validated
    # checkpoint impossible to replay through the strict V1/V2 validator.
    if isinstance(validated, ValidatedForumAgentTurnV5):
        public_identity_mode = validated.base_validated.public_identity_mode
        validated = validated.base_validated.base_validated
        schema = FORUM_AGENT_TURN_V5_SCHEMA_VERSION
    elif hasattr(validated, "base_validated"):
        public_identity_mode = validated.public_identity_mode
        validated = validated.base_validated
        schema = FORUM_AGENT_TURN_V4_SCHEMA_VERSION
    elif hasattr(validated, "private_actions"):
        public_identity_mode = None
        schema = FORUM_AGENT_TURN_V3_SCHEMA_VERSION
    elif hasattr(validated, "interactions"):
        public_identity_mode = None
        schema = FORUM_AGENT_TURN_V2_SCHEMA_VERSION
    else:
        public_identity_mode = None
        schema = FORUM_AGENT_TURN_SCHEMA_VERSION
    raw_decision = asdict(validated.decision)
    value = {
        key: child
        for key, child in raw_decision.items()
        if key in OUTPUT_FIELDS
        or (
            schema in {
                FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
            }
            and key == "public_private_origin_claim_id"
        )
    }
    if hasattr(validated, "interactions"):
        value["engagements"] = [asdict(item) for item in validated.interactions]
    if hasattr(validated, "private_actions"):
        value["private_actions"] = [
            {
                **asdict(item),
                "target_agent_ids": list(item.target_agent_ids),
                "topic_tags": list(item.topic_tags),
            }
            for item in validated.private_actions
        ]
    if public_identity_mode is not None:
        value["public_identity_mode"] = public_identity_mode
    return {"schema_version": schema, **value}


def _attempt_received_payload(
    *,
    slot: Mapping[str, Any],
    envelope: TurnEnvelope,
) -> dict[str, Any]:
    if envelope.provenance not in PUBLIC_PROVENANCE_KINDS:
        raise ForumExecutionError("resident source returned non-formal provenance")
    turn_schema = str(
        slot.get("turn_schema_version") or FORUM_AGENT_TURN_SCHEMA_VERSION
    )
    return {
        "schema_version": _schema_for_turn(
            turn_schema,
            {
                FORUM_AGENT_TURN_SCHEMA_VERSION: "campus-pulse-forum-attempt-received-v1",
                FORUM_AGENT_TURN_V2_SCHEMA_VERSION: "campus-pulse-forum-attempt-received-v2",
                FORUM_AGENT_TURN_V3_SCHEMA_VERSION: "campus-pulse-forum-attempt-received-v3",
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION: "campus-pulse-forum-attempt-received-v4",
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION: "campus-pulse-forum-attempt-received-v5",
            },
        ),
        "slot_id": slot["slot_id"],
        "slot_kind": "resident",
        "prompt_sha256": envelope.prompt_sha256,
        "response_sha256": envelope.response_sha256,
        "provenance": envelope.provenance,
        "returned_model": envelope.returned_model,
        "token_usage": {
            "input_tokens": int(envelope.prompt_tokens),
            "output_tokens": int(envelope.completion_tokens),
            "total_tokens": int(envelope.total_tokens),
        },
        "provider_calls": int(envelope.provider_calls),
        "provider_attempts": int(envelope.provider_attempts),
        "cache_hit": bool(envelope.cache_hit),
        "response_body_persisted": False,
    }


def _committed_envelope(
    record: Mapping[str, Any],
    *,
    slot: Mapping[str, Any],
    rendered: (
        RenderedForumAgentTurn
        | RenderedForumAgentTurnV2
        | RenderedForumAgentTurnV3
        | RenderedForumAgentTurnV4
        | RenderedForumAgentTurnV5
        | None
    ),
) -> TurnEnvelope:
    required = {
        "schema_version",
        "slot_id",
        "slot_kind",
        "agent_id",
        "selection_role",
        "prompt_sha256",
        "response_sha256",
        "provenance",
        "returned_model",
        "token_usage",
        "provider_calls",
        "provider_attempts",
        "cache_hit",
        "validated_decision",
        "validation_status",
        "integration_status",
        "innovation_id",
        "materialization",
        "trace_release_sha256",
        "response_body_persisted",
    }
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise ForumExecutionError("committed resident attempt is incomplete")
    turn_schema = str(
        slot.get("turn_schema_version") or FORUM_AGENT_TURN_SCHEMA_VERSION
    )
    expected_attempt_schema = _schema_for_turn(
        turn_schema,
        {
            FORUM_AGENT_TURN_SCHEMA_VERSION: ATTEMPT_SCHEMA,
            FORUM_AGENT_TURN_V2_SCHEMA_VERSION: ATTEMPT_V2_SCHEMA,
            FORUM_AGENT_TURN_V3_SCHEMA_VERSION: ATTEMPT_V3_SCHEMA,
            FORUM_AGENT_TURN_V4_SCHEMA_VERSION: ATTEMPT_V4_SCHEMA,
            FORUM_AGENT_TURN_V5_SCHEMA_VERSION: ATTEMPT_V5_SCHEMA,
        },
    )
    expected_decision_schema = turn_schema
    if (
        record.get("schema_version") != expected_attempt_schema
        or record.get("slot_id") != slot["slot_id"]
        or record.get("slot_kind") != "resident"
        or record.get("agent_id") != slot["agent_id"]
        or record.get("selection_role") != slot["selection_role"]
        or record.get("validation_status") != "passed"
        or record.get("integration_status") not in {"validated", "materialized"}
        or record.get("response_body_persisted") is not False
    ):
        raise ForumExecutionError("committed resident attempt identity/status drifted")
    prompt_sha = str(record.get("prompt_sha256") or "")
    response_sha = str(record.get("response_sha256") or "")
    if _SHA256_RE.fullmatch(prompt_sha) is None or _SHA256_RE.fullmatch(response_sha) is None:
        raise ForumExecutionError("committed resident attempt fingerprint is invalid")
    if rendered is not None and prompt_sha != rendered.prompt_sha256:
        raise ForumExecutionError("committed resident prompt no longer matches its renderer")
    provenance = str(record.get("provenance") or "")
    if provenance not in PUBLIC_PROVENANCE_KINDS:
        raise ForumExecutionError("committed resident attempt is not formal LLM evidence")
    decision = record.get("validated_decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("schema_version") != expected_decision_schema
    ):
        raise ForumExecutionError("committed resident decision schema drifted")
    usage = record.get("token_usage")
    if not isinstance(usage, Mapping):
        raise ForumExecutionError("committed resident token usage is invalid")
    try:
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
        total_tokens = int(usage["total_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ForumExecutionError("committed resident token usage is incomplete") from exc
    if min(input_tokens, output_tokens, total_tokens) < 0 or total_tokens != input_tokens + output_tokens:
        raise ForumExecutionError("committed resident token usage is not conserved")
    content = _strict_json(
        {key: value for key, value in decision.items() if key != "schema_version"}
    )
    return TurnEnvelope(
        slot_kind="resident",
        prompt_sha256=prompt_sha,
        response_sha256=response_sha,
        provenance=provenance,
        content=content,
        returned_model=str(record.get("returned_model") or ""),
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        provider_calls=int(record.get("provider_calls", 0)),
        cache_hit=bool(record.get("cache_hit", False)),
        provider_attempts=int(record.get("provider_attempts", 0)),
        provider_billed_tokens=(
            total_tokens if int(record.get("provider_calls", 0)) > 0 else 0
        ),
    )


def _thread_view(thread: ForumThread) -> dict[str, Any]:
    return {
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


def _claim_view(claim: ForumClaim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "thread_id": claim.thread_id,
        "claim_summary": claim.claim_summary,
        "status": claim.status,
        "first_seen_tick": claim.first_seen_tick,
        "last_seen_tick": claim.last_seen_tick,
        "created_by_message_id": claim.created_by_message_id,
        "parent_claim_id": claim.parent_claim_id,
        "supporting_message_ids": list(claim.supporting_message_ids),
        "challenging_message_ids": list(claim.challenging_message_ids),
        "correction_target_claim_id": claim.correction_target_claim_id,
        "evidence_reference": claim.evidence_reference,
    }


def _verify_committed_materialization(
    *,
    record: Mapping[str, Any],
    envelope: TurnEnvelope,
    ledger: ForumLedger,
    memory: AgentMemory,
    tick: int,
) -> None:
    materialization = record.get("materialization")
    if not isinstance(materialization, Mapping):
        raise ForumExecutionError("materialized committed attempt lacks its identity")
    interaction_ids = materialization.get("interaction_ids", [])
    if not isinstance(interaction_ids, list) or any(
        not isinstance(value, str) for value in interaction_ids
    ):
        raise ForumExecutionError("committed resident interactions are invalid")
    if materialization.get("interaction_count", len(interaction_ids)) != len(
        interaction_ids
    ):
        raise ForumExecutionError("committed resident interaction count drifted")
    for interaction_id in interaction_ids:
        interaction = ledger.interactions.get(interaction_id)
        if (
            interaction is None
            or interaction.created_tick != tick
            or interaction.provenance.response_sha256 != envelope.response_sha256
        ):
            raise ForumExecutionError(
                "committed resident interaction is missing from restored ledger"
            )
    message_id = materialization.get("message_id")
    if message_id is not None:
        message = ledger.messages.get(str(message_id))
        if (
            message is None
            or message.created_tick != tick
            or message.provenance.response_sha256 != envelope.response_sha256
        ):
            raise ForumExecutionError("committed public message is missing from restored ledger")
    elif not interaction_ids and materialization.get("silent") is not True:
        raise ForumExecutionError("committed silent attempt identity is invalid")
    decision = record.get("validated_decision")
    memory_expected = isinstance(decision, Mapping) and bool(
        str(decision.get("memory_update") or "").strip()
    )
    if memory_expected and not any(
        episode.tick == tick
        and episode.provenance.response_sha256 == envelope.response_sha256
        and episode.message_id == message_id
        for episode in memory.recent_real_llm_episodes
    ):
        raise ForumExecutionError("committed resident memory is missing from restored state")


def execute_resident_activation_round_v1(
    *,
    activation_round: ForumActivationRound,
    population: Sequence[Mapping[str, Any]],
    posterior_by_index: Mapping[int, Mapping[str, Any]],
    memories_by_index: MutableMapping[int, AgentMemory],
    ledger: ForumLedger,
    exposures_by_index: Mapping[int, Sequence[FeedExposure]],
    context_by_index: Mapping[int, Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
    expected_model: str,
    source: SemanticTurnSource,
    slot_design_callback: Callable[[Mapping[str, Any]], None] | None,
    attempt_callback: Callable[[Mapping[str, Any]], None] | None,
    committed_attempts: Mapping[str, Mapping[str, Any]] | None = None,
    applied_innovation_ids: set[str] | None = None,
    reviewed_trace_release_sha256: str | None = None,
    round_design_committed: bool = False,
    planned_resident_budget: int | None = None,
    prompt_population: Sequence[Mapping[str, Any]] | None = None,
    prompt_memories_by_index: Mapping[int, AgentMemory] | None = None,
    turn_schema_version: str = FORUM_AGENT_TURN_SCHEMA_VERSION,
    resident_model_view_condition: str = V5_FULL_NATURAL_PROFILE,
) -> dict[str, Any]:
    """Execute one complete resident round with callbacks and exact recovery."""

    if turn_schema_version not in _SUPPORTED_TURN_SCHEMAS:
        raise ForumExecutionError("resident turn schema is unsupported")
    if resident_model_view_condition not in V5_MODEL_VIEW_CONDITIONS:
        raise ForumExecutionError("resident V5 model-view condition is invalid")
    if (
        turn_schema_version != FORUM_AGENT_TURN_V5_SCHEMA_VERSION
        and resident_model_view_condition != V5_FULL_NATURAL_PROFILE
    ):
        raise ForumExecutionError("resident model-view control requires turn V5")
    use_parallel_interactions = _turn_has_parallel_engagements(turn_schema_version)
    use_private_channel = _turn_has_private_channel(turn_schema_version)
    use_public_anonymity = turn_schema_version in {
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    }

    expected_budget = (
        resident_turn_budget(tick)
        if planned_resident_budget is None
        else planned_resident_budget
    )
    if (
        isinstance(expected_budget, bool)
        or not isinstance(expected_budget, int)
        or not 1 <= expected_budget <= 64
        or expected_budget
        > max(resident_turn_budget(tick), 48 if tick in {2, 7, 13, 19} else 0)
    ):
        raise ForumExecutionError("planned resident budget is outside the tick bound")
    if tick != activation_round.tick or activation_round.resident_budget != expected_budget:
        raise ForumExecutionError("activation round tick/budget drifted")
    indices = activation_round.resident_indices
    if (
        activation_round.resident_count > 64
        or activation_round.resident_count != activation_round.resident_budget
        or len(set(indices)) != len(indices)
    ):
        raise ForumExecutionError("resident activation exceeds 64 or repeats a subject")
    if any(index < 0 or index >= len(population) for index in indices):
        raise ForumExecutionError("resident activation index falls outside the population")
    prompt_profiles = population if prompt_population is None else prompt_population
    if len(prompt_profiles) != len(population):
        raise ForumExecutionError("resident prompt population size drifted")
    committed = dict(committed_attempts or {})
    committed_innovations = applied_innovation_ids if applied_innovation_ids is not None else set()
    innovations = set(committed_innovations)
    slot_rows: list[dict[str, Any]] = []
    rendered_by_slot: dict[
        str,
        RenderedForumAgentTurn
        | RenderedForumAgentTurnV2
        | RenderedForumAgentTurnV3
        | RenderedForumAgentTurnV4
        | RenderedForumAgentTurnV5,
    ] = {}
    for index in indices:
        agent = population[index]
        prompt_agent = prompt_profiles[index]
        if (
            str(prompt_agent.get("agent_id") or "") != str(agent.get("agent_id") or "")
            or str(prompt_agent.get("synthetic_display_id") or "")
            != str(agent.get("synthetic_display_id") or "")
        ):
            raise ForumExecutionError("resident prompt profile changed subject identity")
        agent_id = str(agent.get("agent_id") or "")
        role = _selection_role(activation_round, index)
        slot_id = _slot_id(
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            agent_id=agent_id,
            selection_role=role,
        )
        existing = committed.get(slot_id)
        rendered: (
            RenderedForumAgentTurn
            | RenderedForumAgentTurnV2
            | RenderedForumAgentTurnV3
            | RenderedForumAgentTurnV4
            | RenderedForumAgentTurnV5
            | None
        ) = None
        if existing is None or existing.get("integration_status") != "materialized":
            context = dict(context_by_index.get(index, {}))
            renderer = {
                FORUM_AGENT_TURN_SCHEMA_VERSION: render_forum_agent_turn_v1,
                FORUM_AGENT_TURN_V2_SCHEMA_VERSION: render_forum_agent_turn_v2,
                FORUM_AGENT_TURN_V3_SCHEMA_VERSION: render_forum_agent_turn_v3,
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION: render_forum_agent_turn_v4,
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION: render_forum_agent_turn_v5,
            }[turn_schema_version]
            render_kwargs = dict(
                agent=prompt_agent,
                posterior_state=posterior_by_index.get(index, {}),
                memory=(
                    prompt_memories_by_index[index]
                    if prompt_memories_by_index is not None
                    else memories_by_index[index]
                ),
                ledger=ledger,
                exposures=exposures_by_index.get(index, ()),
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
                expected_model=expected_model,
                scenario_context=context.get("scenario_context"),
                public_board=context.get("public_board"),
                current_thread_id=context.get("current_thread_id"),
                direct_parent_message_id=context.get("direct_parent_message_id"),
                quote_message_id=context.get("quote_message_id"),
                allowed_actions=context.get("allowed_actions"),
                require_reviewed_evidence=bool(
                    context.get("require_reviewed_evidence", True)
                ),
                forum_context_mode=str(
                    context.get("forum_context_mode", "full_text")
                ),
            )
            if use_private_channel:
                render_kwargs.update(
                    relationship_context=context.get("relationship_context", ()),
                    private_conversation_ids=context.get(
                        "private_conversation_ids", ()
                    ),
                    private_actions_enabled=bool(
                        context.get("private_actions_enabled", True)
                    ),
                )
            if use_public_anonymity:
                render_kwargs["public_anonymity_enabled"] = bool(
                    context.get("public_anonymity_enabled", True)
                )
            if turn_schema_version == FORUM_AGENT_TURN_V5_SCHEMA_VERSION:
                render_kwargs["model_view_condition"] = (
                    resident_model_view_condition
                )
            rendered = renderer(**render_kwargs)
            prompt_sha = rendered.prompt_sha256
            rendered_by_slot[slot_id] = rendered
        else:
            prompt_sha = str(existing.get("prompt_sha256") or "")
            if _SHA256_RE.fullmatch(prompt_sha) is None:
                raise ForumExecutionError("committed materialized slot lacks prompt identity")
        slot_rows.append(
            {
                "schema_version": _schema_for_turn(
                    turn_schema_version,
                    {
                        FORUM_AGENT_TURN_SCHEMA_VERSION: SLOT_DESIGN_SCHEMA,
                        FORUM_AGENT_TURN_V2_SCHEMA_VERSION: SLOT_DESIGN_V2_SCHEMA,
                        FORUM_AGENT_TURN_V3_SCHEMA_VERSION: SLOT_DESIGN_V3_SCHEMA,
                        FORUM_AGENT_TURN_V4_SCHEMA_VERSION: SLOT_DESIGN_V4_SCHEMA,
                        FORUM_AGENT_TURN_V5_SCHEMA_VERSION: SLOT_DESIGN_V5_SCHEMA,
                    },
                ),
                "slot_id": slot_id,
                "slot_kind": "resident",
                "scenario_id": scenario_id,
                "branch": branch,
                "tick": tick,
                "agent_index": index,
                "agent_id": agent_id,
                "selection_role": role,
                "prompt_sha256": prompt_sha,
                **(
                    {"turn_schema_version": turn_schema_version}
                    if use_parallel_interactions
                    else {}
                ),
                "expected_model": expected_model,
                "prompt_body_persisted": False,
            }
        )
    if set(committed) - {slot["slot_id"] for slot in slot_rows}:
        raise ForumExecutionError("committed attempts contain slots outside this round")
    design_payload = {
        "schema_version": _schema_for_turn(
            turn_schema_version,
            {
                FORUM_AGENT_TURN_SCHEMA_VERSION: "campus-pulse-forum-resident-round-design-v1",
                FORUM_AGENT_TURN_V2_SCHEMA_VERSION: "campus-pulse-forum-resident-round-design-v2",
                FORUM_AGENT_TURN_V3_SCHEMA_VERSION: "campus-pulse-forum-resident-round-design-v3",
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION: "campus-pulse-forum-resident-round-design-v4",
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION: "campus-pulse-forum-resident-round-design-v5",
            },
        ),
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "resident_budget": activation_round.resident_budget,
        "activation_round_sha256": activation_round.content_hash,
        "slots": slot_rows,
    }
    design_payload["content_sha256"] = canonical_hash(design_payload)
    if not round_design_committed:
        if slot_design_callback is None:
            raise ForumExecutionError("new resident round requires a durable slot-design callback")
        slot_design_callback(copy.deepcopy(design_payload))
    before_thread_snapshots = {
        thread_id: canonical_hash(_thread_view(thread))
        for thread_id, thread in ledger.threads.items()
    }
    before_messages = set(ledger.messages)
    before_interactions = set(ledger.interactions)
    before_private_conversations = set(ledger.private_conversations)
    before_private_messages = set(ledger.private_messages)
    before_private_reactions = set(ledger.private_reactions)
    before_claim_snapshots = {
        claim_id: canonical_hash(_claim_view(claim))
        for claim_id, claim in ledger.claims.items()
    }
    working_ledger = copy.deepcopy(ledger)
    working_memories = copy.deepcopy(dict(memories_by_index))
    attempt_rows: list[dict[str, Any]] = []
    materialized_by_slot: dict[str, dict[str, Any]] = {}
    validator = {
        FORUM_AGENT_TURN_SCHEMA_VERSION: validate_forum_agent_turn_v1,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION: validate_forum_agent_turn_v2,
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION: validate_forum_agent_turn_v3,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION: validate_forum_agent_turn_v4,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION: validate_forum_agent_turn_v5,
    }[turn_schema_version]
    materializer = {
        FORUM_AGENT_TURN_SCHEMA_VERSION: materialize_forum_agent_turn_v1,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION: materialize_forum_agent_turn_v2,
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION: materialize_forum_agent_turn_v3,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION: materialize_forum_agent_turn_v4,
        FORUM_AGENT_TURN_V5_SCHEMA_VERSION: materialize_forum_agent_turn_v5,
    }[turn_schema_version]
    acquired_slots = 0
    restored_slots = 0

    def validate_for_acquisition(candidate: TurnEnvelope, *, rendered: Any) -> Any:
        validated_turn = validator(
            candidate,
            rendered=rendered,
            ledger=ledger,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        )
        # The authorized exact-cache layer receives the complete V2+ turn,
        # including public engagements, private actions and optional anonymity.
        # V1 keeps its historical compact decision for byte-compatible replay.
        return validated_turn if use_parallel_interactions else validated_turn.decision

    pending_slots = [
        slot for slot in slot_rows if slot["slot_id"] not in committed
    ]
    acquired_by_slot: dict[str, TurnEnvelope] = {}

    def acquire_new_slot(slot: Mapping[str, Any]) -> TurnEnvelope:
        slot_id = str(slot["slot_id"])
        rendered = rendered_by_slot.get(slot_id)
        if rendered is None:
            raise ForumExecutionError("new resident slot lacks its rendered prompt")
        return source.acquire(
            slot_kind="resident",
            prompt_sha256=rendered.prompt_sha256,
            rendered=rendered,
            validator=lambda candidate, rendered=rendered: (
                validate_for_acquisition(candidate, rendered=rendered)
            ),
        )

    source_concurrency = getattr(source, "max_concurrency", 1)
    if isinstance(source_concurrency, bool) or not isinstance(source_concurrency, int):
        source_concurrency = 1
    worker_count = min(max(1, source_concurrency), len(pending_slots))
    if worker_count == 1:
        for slot in pending_slots:
            acquired_by_slot[str(slot["slot_id"])] = acquire_new_slot(slot)
    elif pending_slots:
        # Prompts and the validation ledger are frozen before acquisition.
        # Provider/cache I/O may therefore run concurrently, while response
        # persistence callbacks and state materialization remain in the exact
        # ordinal slot order below.  The context manager waits for every
        # already-started request to settle before propagating any failure;
        # no in-flight request is abandoned or blindly retried.
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="forum-resident-acquire",
        ) as executor:
            futures = {
                str(slot["slot_id"]): executor.submit(acquire_new_slot, slot)
                for slot in pending_slots
            }
            for slot in pending_slots:
                slot_id = str(slot["slot_id"])
                acquired_by_slot[slot_id] = futures[slot_id].result()

    for slot in slot_rows:
        slot_id = slot["slot_id"]
        index = int(slot["agent_index"])
        rendered = rendered_by_slot.get(slot_id)
        prior = committed.get(slot_id)
        if prior is not None:
            envelope = _committed_envelope(prior, slot=slot, rendered=rendered)
            restored_slots += 1
        else:
            envelope = acquired_by_slot[slot_id]
            acquired_slots += 1
            if attempt_callback is None:
                raise ForumExecutionError("new resident acquisition requires an attempt callback")
            attempt_callback(
                copy.deepcopy(_attempt_received_payload(slot=slot, envelope=envelope))
            )
        innovation_id = "forum-resident-innovation-" + canonical_hash(
            [slot_id, envelope.response_sha256]
        )[:32]
        if prior is not None and prior.get("innovation_id") != innovation_id:
            raise ForumExecutionError("committed resident innovation identity drifted")
        if prior is not None and prior.get("integration_status") == "materialized":
            if innovation_id not in innovations:
                raise ForumExecutionError("materialized committed attempt lacks applied innovation")
            _verify_committed_materialization(
                record=prior,
                envelope=envelope,
                ledger=working_ledger,
                memory=working_memories[index],
                tick=tick,
            )
            attempt_rows.append(copy.deepcopy(dict(prior)))
            materialized_by_slot[slot_id] = copy.deepcopy(
                dict(prior["materialization"])
            )
            continue
        if innovation_id in innovations:
            raise ForumExecutionError("resident semantic innovation would be integrated twice")
        if rendered is None:
            raise ForumExecutionError("validated committed slot requires its original renderer")
        trace_release = (
            prior.get("trace_release_sha256")
            if prior is not None
            else reviewed_trace_release_sha256
        )
        validated = validator(
            envelope,
            rendered=rendered,
            ledger=working_ledger,
            reviewed_trace_release_sha256=(
                str(trace_release) if trace_release is not None else None
            ),
        )
        materialized = materializer(
            validated=validated,
            ledger=working_ledger,
            memory=working_memories[index],
        )
        innovations.add(innovation_id)
        materialization = {
            "action": materialized.action,
            "message_id": materialized.message_id,
            "thread_id": materialized.thread_id,
            "claim_id": materialized.claim_id,
            "silent": materialized.silent,
            "content_hash": materialized.content_hash,
        }
        if hasattr(materialized, "interaction_ids"):
            materialization["interaction_ids"] = list(materialized.interaction_ids)
            materialization["interaction_count"] = len(materialized.interaction_ids)
        if hasattr(materialized, "private_artifact_ids"):
            materialization["private_artifact_ids"] = list(
                materialized.private_artifact_ids
            )
            materialization["private_action_count"] = int(
                materialized.private_action_count
            )
        if hasattr(materialized, "public_identity_mode"):
            materialization["public_identity_mode"] = (
                materialized.public_identity_mode
            )
            materialization["public_pseudonym_id"] = (
                materialized.public_pseudonym_id
            )
        attempt = {
            "schema_version": _schema_for_turn(
                turn_schema_version,
                {
                    FORUM_AGENT_TURN_SCHEMA_VERSION: ATTEMPT_SCHEMA,
                    FORUM_AGENT_TURN_V2_SCHEMA_VERSION: ATTEMPT_V2_SCHEMA,
                    FORUM_AGENT_TURN_V3_SCHEMA_VERSION: ATTEMPT_V3_SCHEMA,
                    FORUM_AGENT_TURN_V4_SCHEMA_VERSION: ATTEMPT_V4_SCHEMA,
                    FORUM_AGENT_TURN_V5_SCHEMA_VERSION: ATTEMPT_V5_SCHEMA,
                },
            ),
            "slot_id": slot_id,
            "slot_kind": "resident",
            "agent_id": slot["agent_id"],
            "selection_role": slot["selection_role"],
            "prompt_sha256": envelope.prompt_sha256,
            "response_sha256": envelope.response_sha256,
            "provenance": envelope.provenance,
            "returned_model": envelope.returned_model,
            "token_usage": {
                "input_tokens": int(envelope.prompt_tokens),
                "output_tokens": int(envelope.completion_tokens),
                "total_tokens": int(envelope.total_tokens),
            },
            "provider_calls": int(envelope.provider_calls),
            "provider_attempts": int(envelope.provider_attempts),
            "cache_hit": bool(envelope.cache_hit),
            "validated_decision": _decision_artifact(validated),
            "validation_status": "passed",
            "integration_status": "materialized",
            "innovation_id": innovation_id,
            "materialization": materialization,
            "trace_release_sha256": (
                str(trace_release) if trace_release is not None else None
            ),
            "response_body_persisted": False,
        }
        attempt_rows.append(attempt)
        materialized_by_slot[slot_id] = materialization
    # Commit the in-memory transaction only after every slot validated.
    ledger.threads = working_ledger.threads
    ledger.messages = working_ledger.messages
    ledger.interactions = working_ledger.interactions
    ledger.claims = working_ledger.claims
    if use_private_channel:
        ledger.private_conversations = working_ledger.private_conversations
        ledger.private_membership_events = working_ledger.private_membership_events
        ledger.private_messages = working_ledger.private_messages
        ledger.private_reactions = working_ledger.private_reactions
        ledger.private_claims = working_ledger.private_claims
    for index in indices:
        memories_by_index[index] = working_memories[index]
    if applied_innovation_ids is not None:
        applied_innovation_ids.clear()
        applied_innovation_ids.update(innovations)
    changed_threads = [
        thread
        for thread_id, thread in sorted(ledger.threads.items())
        if before_thread_snapshots.get(thread_id) != canonical_hash(_thread_view(thread))
    ]
    new_messages = [
        ledger.messages[message_id]
        for message_id in sorted(set(ledger.messages) - before_messages)
    ]
    new_interactions = [
        ledger.interactions[interaction_id]
        for interaction_id in sorted(set(ledger.interactions) - before_interactions)
    ]
    changed_claims = [
        claim
        for claim_id, claim in sorted(ledger.claims.items())
        if before_claim_snapshots.get(claim_id) != canonical_hash(_claim_view(claim))
    ]
    if any(message.provenance.kind not in PUBLIC_PROVENANCE_KINDS for message in new_messages):
        raise ForumExecutionError("resident round produced non-formal public content")
    if any(
        interaction.provenance.kind not in PUBLIC_PROVENANCE_KINDS
        for interaction in new_interactions
    ):
        raise ForumExecutionError("resident round produced non-formal public interactions")
    thread_batch = {
        "schema_version": _schema_for_turn(
            turn_schema_version,
            {
                FORUM_AGENT_TURN_SCHEMA_VERSION: THREAD_BATCH_SCHEMA,
                FORUM_AGENT_TURN_V2_SCHEMA_VERSION: THREAD_BATCH_V2_SCHEMA,
                FORUM_AGENT_TURN_V3_SCHEMA_VERSION: THREAD_BATCH_V3_SCHEMA,
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION: THREAD_BATCH_V4_SCHEMA,
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION: THREAD_BATCH_V5_SCHEMA,
            },
        ),
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "threads": [_thread_view(value) for value in changed_threads],
        "messages": [value.public_view() for value in new_messages],
        "claims": [_claim_view(value) for value in changed_claims],
        "public_message_count": len(new_messages),
        "emulator_public_message_count": 0,
    }
    if use_parallel_interactions:
        thread_batch["interactions"] = [
            value.public_view() for value in new_interactions
        ]
        thread_batch["public_interaction_count"] = len(new_interactions)
    if use_private_channel:
        thread_batch["private_runtime_counts"] = {
            "conversations": len(ledger.private_conversations),
            "messages": len(ledger.private_messages),
            "reactions": len(ledger.private_reactions),
            "claims": len(ledger.private_claims),
        }
        thread_batch["contains_private_text"] = False
    thread_batch["content_sha256"] = canonical_hash(thread_batch)
    memory_updates = [
        {
            "agent_id": slot["agent_id"],
            "agent_index": slot["agent_index"],
            "recent_real_llm_episode_count": len(
                memories_by_index[int(slot["agent_index"])].recent_real_llm_episodes
            ),
            "latest_tick": tick,
            "latest_response_sha256": attempt["response_sha256"],
        }
        for slot, attempt in zip(slot_rows, attempt_rows)
    ]
    result = {
        "schema_version": _schema_for_turn(
            turn_schema_version,
            {
                FORUM_AGENT_TURN_SCHEMA_VERSION: ROUND_EXECUTION_SCHEMA,
                FORUM_AGENT_TURN_V2_SCHEMA_VERSION: ROUND_EXECUTION_V2_SCHEMA,
                FORUM_AGENT_TURN_V3_SCHEMA_VERSION: ROUND_EXECUTION_V3_SCHEMA,
                FORUM_AGENT_TURN_V4_SCHEMA_VERSION: ROUND_EXECUTION_V4_SCHEMA,
                FORUM_AGENT_TURN_V5_SCHEMA_VERSION: ROUND_EXECUTION_V5_SCHEMA,
            },
        ),
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "resident_count": len(slot_rows),
        "acquired_slot_count": acquired_slots,
        "restored_slot_count": restored_slots,
        "slot_design": design_payload,
        "turn_slots": [
            {
                **slot,
                "status": "completed",
                "response_sha256": attempt["response_sha256"],
                "provenance": attempt["provenance"],
                "innovation_id": attempt["innovation_id"],
                "materialization": materialized_by_slot[slot["slot_id"]],
            }
            for slot, attempt in zip(slot_rows, attempt_rows)
        ],
        "attempts": attempt_rows,
        "thread_batch": thread_batch,
        "memory_updates": memory_updates,
        "applied_innovation_ids": sorted(innovations),
        "usage": {
            "provider_calls": sum(int(row["provider_calls"]) for row in attempt_rows),
            "provider_attempts": sum(int(row["provider_attempts"]) for row in attempt_rows),
            "incremental_provider_calls": sum(
                int(row["provider_calls"])
                for row in attempt_rows
                if row["slot_id"] not in committed
            ),
            "incremental_provider_attempts": sum(
                int(row["provider_attempts"])
                for row in attempt_rows
                if row["slot_id"] not in committed
            ),
            "total_tokens": sum(int(row["token_usage"]["total_tokens"]) for row in attempt_rows),
            "public_llm_messages": len(new_messages),
            "emulator_public_messages": 0,
        },
    }
    if use_parallel_interactions:
        result["usage"]["public_llm_interactions"] = len(new_interactions)
    if use_private_channel:
        result["usage"]["private_llm_messages"] = len(
            set(ledger.private_messages) - before_private_messages
        )
        result["usage"]["private_llm_conversations"] = len(
            set(ledger.private_conversations) - before_private_conversations
        )
        result["usage"]["private_llm_reactions"] = len(
            set(ledger.private_reactions) - before_private_reactions
        )
    result["content_sha256"] = canonical_hash(result)
    return result


def execute_resident_activation_round_v2(**kwargs: Any) -> dict[str, Any]:
    """Execute the parallel-interaction contract without changing V1 replay."""

    if "turn_schema_version" in kwargs:
        raise ForumExecutionError(
            "V2 execution wrapper owns the resident turn schema version"
        )
    return execute_resident_activation_round_v1(
        **kwargs,
        turn_schema_version=FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    )


def execute_resident_activation_round_v3(**kwargs: Any) -> dict[str, Any]:
    """Execute public engagements plus private/group actions atomically."""

    if "turn_schema_version" in kwargs:
        raise ForumExecutionError(
            "V3 execution wrapper owns the resident turn schema version"
        )
    return execute_resident_activation_round_v1(
        **kwargs,
        turn_schema_version=FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    )


def execute_resident_activation_round_v4(**kwargs: Any) -> dict[str, Any]:
    """Execute V3 channels plus the ordinary-forum anonymity choice."""

    if "turn_schema_version" in kwargs:
        raise ForumExecutionError(
            "V4 execution wrapper owns the resident turn schema version"
        )
    return execute_resident_activation_round_v1(
        **kwargs,
        turn_schema_version=FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    )


def execute_resident_activation_round_v5(**kwargs: Any) -> dict[str, Any]:
    """Execute the natural resident view through the unchanged V4 ledger rules."""

    if "turn_schema_version" in kwargs:
        raise ForumExecutionError(
            "V5 execution wrapper owns the resident turn schema version"
        )
    return execute_resident_activation_round_v1(
        **kwargs,
        turn_schema_version=FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
    )
