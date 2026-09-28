"""Transport-neutral execution glue for one ForumTwin governance actor round.

The module performs no networking and owns no persistence.  It emits artifacts
that can be mapped to v12 slot, attempt, governance-round and thread-batch
tables while enforcing the persistence order around an injected semantic turn
source.  Committed attempts are replayed from validated structures without
calling the source again; a committed round is verified and returned without
reapplying QRE, resources or forum messages.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .forum_governance import (
    ForumGovernanceDecision,
    RenderedForumGovernanceTurn,
    play_forum_governance_round_v2,
    render_forum_governance_actor_turn_v2,
    validate_forum_governance_actor_turn_v2,
)
from .forum_twin import (
    ForumLedger,
    ForumMessage,
    ForumThread,
    GovernanceArtifact,
    PublicProvenance,
)
from .governance_arena_v2 import BranchState
from .mass_population import canonical_hash
from .yulan_scale import (
    ACTOR_IDS,
    PUBLISHABLE_PROVENANCES,
    SemanticTurnSource,
    TurnEnvelope,
    YuLanScaleError,
)


ACTOR_SLOT_DESIGN_SCHEMA = "campus-pulse-forum-actor-slot-design-v1"
ACTOR_ATTEMPT_SCHEMA = "campus-pulse-forum-actor-attempt-v1"
ACTOR_ROUND_SCHEMA = "campus-pulse-forum-actor-execution-round-v1"
GOVERNANCE_THREAD_BATCH_SCHEMA = "campus-pulse-forum-governance-thread-batch-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ArtifactCallback = Callable[[str, Mapping[str, Any]], None]


class ForumActorExecutionError(YuLanScaleError):
    """Raised when an actor round cannot be recovered or committed exactly."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _resource_view(state: BranchState) -> dict[str, Any]:
    remaining = (
        float(state.resource_opening)
        - float(state.resource_consumed)
        - float(state.resource_committed)
    )
    conserved = math.isclose(
        float(state.resource_opening),
        float(state.resource_consumed) + float(state.resource_committed) + remaining,
        abs_tol=1e-8,
    )
    if remaining < -1e-8 or not conserved:
        raise ForumActorExecutionError("governance resource ledger is not conserved")
    return {
        "opening": round(float(state.resource_opening), 6),
        "consumed": round(float(state.resource_consumed), 6),
        "committed": round(float(state.resource_committed), 6),
        "remaining": round(remaining, 6),
        "conserved": conserved,
    }


def _decision_payload(decision: ForumGovernanceDecision) -> dict[str, Any]:
    return {
        "action_probabilities": dict(sorted(decision.action_probabilities.items())),
        "preferred_action": decision.preferred_action,
        "confidence": decision.confidence,
        "rationale_tags": list(decision.rationale_tags),
        "risk_level": decision.risk_level,
        "uncertainty": decision.uncertainty,
        "public_message": decision.public_message,
        "target_claim_id": decision.target_claim_id,
        "target_thread_id": decision.target_thread_id,
        "evidence_or_service_reference": decision.evidence_or_service_reference,
    }


def _slot_design(
    *,
    rendered: RenderedForumGovernanceTurn,
    private_observation: Mapping[str, float],
    forum_context: Mapping[str, Any],
    target_catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    slot_id = "forum-actor-slot-" + canonical_hash(
        [
            rendered.scenario_id,
            rendered.branch,
            rendered.tick,
            rendered.actor_id,
            rendered.prompt_sha256,
        ]
    )[:24]
    body = {
        "schema_version": ACTOR_SLOT_DESIGN_SCHEMA,
        "slot_id": slot_id,
        "idempotency_key": canonical_hash(["actor-slot", slot_id]),
        "slot_kind": "governance_actor",
        "actor_id": rendered.actor_id,
        "scenario_id": rendered.scenario_id,
        "branch": rendered.branch,
        "tick": rendered.tick,
        "prompt_sha256": rendered.prompt_sha256,
        "private_observation_sha256": canonical_hash(private_observation),
        "forum_context_sha256": canonical_hash(forum_context),
        "target_catalog_sha256": canonical_hash(list(target_catalog)),
        "feasible_action_ids": list(rendered.feasible_actions),
        "prompt_body_persisted": False,
        "private_observation_persisted": False,
    }
    return {**body, "slot_design_sha256": canonical_hash(body)}


def _usage(envelope: TurnEnvelope) -> dict[str, Any]:
    values = {
        "prompt_tokens": int(envelope.prompt_tokens),
        "completion_tokens": int(envelope.completion_tokens),
        "total_tokens": int(envelope.total_tokens),
        "provider_calls": int(envelope.provider_calls),
        "provider_attempts": int(envelope.provider_attempts),
        "format_repairs": int(envelope.format_repairs),
        "provider_billed_tokens": int(envelope.provider_billed_tokens),
        "cache_hit": bool(envelope.cache_hit),
    }
    if any(value < 0 for key, value in values.items() if key != "cache_hit"):
        raise ForumActorExecutionError("actor attempt usage cannot be negative")
    return values


def _acquired_attempt_artifact(
    *,
    slot: Mapping[str, Any],
    envelope: TurnEnvelope,
) -> dict[str, Any]:
    if (
        envelope.slot_kind != "governance_actor"
        or envelope.prompt_sha256 != slot["prompt_sha256"]
        or _SHA256_RE.fullmatch(envelope.response_sha256) is None
        or envelope.provenance not in PUBLISHABLE_PROVENANCES
        or not str(envelope.returned_model).strip()
    ):
        raise ForumActorExecutionError("acquired actor response identity is invalid")
    body = {
        "schema_version": ACTOR_ATTEMPT_SCHEMA,
        "attempt_id": "forum-actor-attempt-" + canonical_hash(
            [slot["slot_id"], envelope.response_sha256]
        )[:24],
        "slot_id": slot["slot_id"],
        "idempotency_key": canonical_hash(
            ["actor-attempt", slot["idempotency_key"], envelope.response_sha256]
        ),
        "actor_id": slot["actor_id"],
        "prompt_sha256": envelope.prompt_sha256,
        "response_sha256": envelope.response_sha256,
        "provenance": envelope.provenance,
        "returned_model": envelope.returned_model,
        "usage": _usage(envelope),
        "outcome": "response_acquired",
        "provider_body_persisted": False,
    }
    return {**body, "attempt_sha256": canonical_hash(body)}


def _committed_attempt_artifact(
    *,
    acquired: Mapping[str, Any],
    decision: ForumGovernanceDecision,
    reviewed_trace_release_sha256: str | None,
) -> dict[str, Any]:
    body = {
        **{key: value for key, value in acquired.items() if key != "attempt_sha256"},
        "outcome": "committed",
        "validated_decision": _decision_payload(decision),
        "validated_decision_sha256": canonical_hash(_decision_payload(decision)),
        "trace_release_sha256": (
            reviewed_trace_release_sha256
            if decision.provenance == "reviewed_trace_replay"
            else None
        ),
    }
    return {**body, "attempt_sha256": canonical_hash(body)}


def _validate_slot_design(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if dict(stored) != dict(expected):
        raise ForumActorExecutionError("committed actor slot does not match the rendered design")


def _restore_committed_attempt(
    *,
    artifact: Mapping[str, Any],
    slot: Mapping[str, Any],
    rendered: RenderedForumGovernanceTurn,
    reviewed_trace_release_sha256: str | None,
) -> ForumGovernanceDecision:
    expected_fields = {
        "schema_version",
        "attempt_id",
        "slot_id",
        "idempotency_key",
        "actor_id",
        "prompt_sha256",
        "response_sha256",
        "provenance",
        "returned_model",
        "usage",
        "outcome",
        "provider_body_persisted",
        "validated_decision",
        "validated_decision_sha256",
        "trace_release_sha256",
        "attempt_sha256",
    }
    if set(artifact) != expected_fields:
        raise ForumActorExecutionError("committed actor attempt fields are incomplete")
    body = {key: value for key, value in artifact.items() if key != "attempt_sha256"}
    if artifact.get("attempt_sha256") != canonical_hash(body):
        raise ForumActorExecutionError("committed actor attempt hash mismatch")
    if (
        artifact.get("schema_version") != ACTOR_ATTEMPT_SCHEMA
        or artifact.get("outcome") != "committed"
        or artifact.get("slot_id") != slot["slot_id"]
        or artifact.get("actor_id") != rendered.actor_id
        or artifact.get("prompt_sha256") != rendered.prompt_sha256
        or artifact.get("provider_body_persisted") is not False
        or artifact.get("provenance") not in PUBLISHABLE_PROVENANCES
        or _SHA256_RE.fullmatch(str(artifact.get("response_sha256") or "")) is None
    ):
        raise ForumActorExecutionError("committed actor attempt identity drifted")
    decision_payload = artifact.get("validated_decision")
    if not isinstance(decision_payload, Mapping) or artifact.get(
        "validated_decision_sha256"
    ) != canonical_hash(decision_payload):
        raise ForumActorExecutionError("committed actor validated decision hash mismatch")
    usage = artifact.get("usage")
    expected_usage_fields = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "provider_calls",
        "provider_attempts",
        "format_repairs",
        "provider_billed_tokens",
        "cache_hit",
    }
    if not isinstance(usage, Mapping) or set(usage) != expected_usage_fields:
        raise ForumActorExecutionError("committed actor attempt usage is invalid")
    if artifact["provenance"] == "reviewed_trace_replay":
        trace_sha = artifact.get("trace_release_sha256")
        if not isinstance(trace_sha, str) or _SHA256_RE.fullmatch(trace_sha) is None:
            raise ForumActorExecutionError("replayed actor attempt lacks a trace release")
        if trace_sha != reviewed_trace_release_sha256:
            raise ForumActorExecutionError(
                "replayed actor attempt belongs to a different trace release"
            )
    elif artifact.get("trace_release_sha256") is not None:
        raise ForumActorExecutionError("live actor attempt cannot claim a trace release")
    envelope = TurnEnvelope(
        slot_kind="governance_actor",
        prompt_sha256=str(artifact["prompt_sha256"]),
        response_sha256=str(artifact["response_sha256"]),
        provenance=str(artifact["provenance"]),
        content=_canonical_json(decision_payload),
        returned_model=str(artifact["returned_model"]),
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        total_tokens=int(usage["total_tokens"]),
        provider_calls=int(usage["provider_calls"]),
        cache_hit=bool(usage["cache_hit"]),
        provider_attempts=int(usage["provider_attempts"]),
        format_repairs=int(usage["format_repairs"]),
        provider_billed_tokens=int(usage["provider_billed_tokens"]),
    )
    if _usage(envelope) != dict(usage):
        raise ForumActorExecutionError("committed actor attempt usage no longer validates")
    decision = validate_forum_governance_actor_turn_v2(envelope, rendered=rendered)
    if _decision_payload(decision) != dict(decision_payload):
        raise ForumActorExecutionError("committed actor decision no longer validates identically")
    return decision


def _public_provenance(
    attempt: Mapping[str, Any],
) -> PublicProvenance:
    return PublicProvenance(
        kind=str(attempt["provenance"]),
        prompt_sha256=str(attempt["prompt_sha256"]),
        response_sha256=str(attempt["response_sha256"]),
        model=str(attempt["returned_model"]),
        trace_release_sha256=(
            str(attempt["trace_release_sha256"])
            if attempt.get("trace_release_sha256") is not None
            else None
        ),
    )


def _materialize_governance_messages(
    *,
    ledger: ForumLedger,
    governance_messages: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
    round_id: str,
) -> tuple[list[str], list[str]]:
    message_ids: list[str] = []
    new_thread_ids: list[str] = []
    for row in governance_messages:
        actor = str(row.get("source_role") or "")
        if actor not in ACTOR_IDS or actor not in attempts:
            raise ForumActorExecutionError("governance message actor is invalid")
        target_claim_id = row.get("target_claim_id")
        target_thread_id = row.get("target_thread_id")
        if target_claim_id is not None:
            claim = ledger.claims.get(str(target_claim_id))
            if claim is None:
                raise ForumActorExecutionError("governance message targets an unknown claim")
            if target_thread_id is None:
                target_thread_id = claim.thread_id
            elif claim.thread_id != target_thread_id:
                raise ForumActorExecutionError("governance claim and thread targets disagree")
        message_id = "forum-governance-message-" + canonical_hash(
            [round_id, actor, attempts[actor]["response_sha256"]]
        )[:24]
        provenance = _public_provenance(attempts[actor])
        artifact_row = row.get("governance_artifact")
        if not isinstance(artifact_row, Mapping):
            raise ForumActorExecutionError(
                "scheduled governance message lacks its persistent action object"
            )
        artifact = GovernanceArtifact(
            artifact_id=str(artifact_row["artifact_id"]),
            artifact_kind=str(artifact_row["artifact_kind"]),
            action_id=str(artifact_row["action_id"]),
            responsible_role=str(artifact_row["responsible_role"]),
            status=str(artifact_row["status"]),
            created_tick=int(artifact_row["created_tick"]),
            due_tick=int(artifact_row["due_tick"]),
            expires_tick=int(artifact_row["expires_tick"]),
            target_thread_id=(
                str(artifact_row["target_thread_id"])
                if artifact_row.get("target_thread_id") is not None
                else None
            ),
            target_claim_id=(
                str(artifact_row["target_claim_id"])
                if artifact_row.get("target_claim_id") is not None
                else None
            ),
            public_scope=str(artifact_row["public_scope"]),
            confirmed_scope=tuple(str(value) for value in artifact_row["confirmed_scope"]),
            unresolved_scope=tuple(str(value) for value in artifact_row["unresolved_scope"]),
        )
        if target_thread_id is None:
            thread_id = "forum-governance-thread-" + canonical_hash(
                [round_id, actor, message_id]
            )[:24]
        else:
            thread_id = str(target_thread_id)
            thread = ledger.threads.get(thread_id)
            if (
                thread is None
                or thread.scenario_id != scenario_id
                or thread.branch not in {branch, "shared_baseline"}
                or thread.status != "open"
            ):
                raise ForumActorExecutionError("governance target thread is unavailable")
        message = ForumMessage(
            message_id=message_id,
            thread_id=thread_id,
            source_agent_display_id=f"governance:{actor}",
            action="governance_message",
            visible_text=str(row["visible_text"]),
            topic="governance",
            stance="neutral",
            emotion="calm",
            evidence_status=(
                "verified" if artifact.artifact_kind == "evidence_card" else "uncertain"
            ),
            confidence=float(_decision_from_attempt(attempts[actor])["confidence"]),
            created_tick=tick,
            ttl=8,
            provenance=provenance,
            claim_id=(str(target_claim_id) if target_claim_id is not None else None),
            claim_operation="none",
            governance_artifact=artifact,
        )
        if target_thread_id is None:
            ledger.add_thread(
                ForumThread(
                    thread_id=thread_id,
                    board="synthetic-campus-governance",
                    root_message_id=message_id,
                    scenario_id=scenario_id,
                    branch=branch,
                    created_tick=tick,
                    status="open",
                    topic="governance",
                    need="public_governance_update",
                    participant_count=1,
                    reply_count=0,
                    last_active_tick=tick,
                    provenance_kind=provenance.kind,
                ),
                message,
            )
            new_thread_ids.append(thread_id)
        else:
            ledger.add_message(message)
            if (
                artifact.artifact_kind == "evidence_card"
                and artifact.target_claim_id is not None
            ):
                target_claim = ledger.claims[str(artifact.target_claim_id)]
                ledger.advance_claim(
                    replace(
                        target_claim,
                        status="corrected",
                        last_seen_tick=tick,
                        challenging_message_ids=(
                            *target_claim.challenging_message_ids,
                            message.message_id,
                        ),
                        evidence_reference=artifact.artifact_id,
                    )
                )
        message_ids.append(message_id)
    return message_ids, new_thread_ids


def _decision_from_attempt(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    value = attempt.get("validated_decision")
    if not isinstance(value, Mapping):
        raise ForumActorExecutionError("attempt lacks a validated decision")
    return value


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


def _thread_batch(
    *,
    ledger: ForumLedger,
    round_id: str,
    message_ids: Sequence[str],
    new_thread_ids: Sequence[str],
) -> dict[str, Any]:
    body = {
        "schema_version": GOVERNANCE_THREAD_BATCH_SCHEMA,
        "batch_id": "forum-governance-batch-" + canonical_hash(
            [round_id, list(message_ids)]
        )[:24],
        "round_id": round_id,
        "threads": [_thread_view(ledger.threads[thread_id]) for thread_id in new_thread_ids],
        "messages": [ledger.messages[message_id].public_view() for message_id in message_ids],
        "message_ids": list(message_ids),
        "new_thread_ids": list(new_thread_ids),
        "public_content_provenance_only": True,
    }
    return {**body, "thread_batch_sha256": canonical_hash(body)}


def _validate_committed_round(
    *,
    committed_round: Mapping[str, Any],
    expected_round_id: str,
    attempts: Mapping[str, Mapping[str, Any]],
    state: BranchState,
    ledger: ForumLedger,
) -> dict[str, Any]:
    body = {key: value for key, value in committed_round.items() if key != "round_sha256"}
    if committed_round.get("round_sha256") != canonical_hash(body):
        raise ForumActorExecutionError("committed governance round hash mismatch")
    if (
        committed_round.get("schema_version") != ACTOR_ROUND_SCHEMA
        or committed_round.get("round_id") != expected_round_id
        or committed_round.get("attempt_sha256s")
        != {actor: attempts[actor]["attempt_sha256"] for actor in ACTOR_IDS}
    ):
        raise ForumActorExecutionError("committed governance round identity drifted")
    if _resource_view(state) != committed_round.get("resource_after"):
        raise ForumActorExecutionError("restored state does not contain the committed resource result")
    game_hash = committed_round.get("state_game_round_sha256")
    if not any(canonical_hash(row) == game_hash for row in state.game_rounds):
        raise ForumActorExecutionError("restored state lacks the committed QRE game round")
    message_ids = committed_round.get("message_ids")
    new_thread_ids = committed_round.get("new_thread_ids")
    if not isinstance(message_ids, list) or not isinstance(new_thread_ids, list):
        raise ForumActorExecutionError("committed governance round message frontier is invalid")
    batch = _thread_batch(
        ledger=ledger,
        round_id=expected_round_id,
        message_ids=message_ids,
        new_thread_ids=new_thread_ids,
    )
    if batch["thread_batch_sha256"] != committed_round.get("thread_batch_sha256"):
        raise ForumActorExecutionError("restored ForumLedger does not match the committed thread batch")
    return batch


def execute_forum_actor_round_v1(
    *,
    state: BranchState,
    ledger: ForumLedger,
    source: SemanticTurnSource | None,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    private_observations: Mapping[str, Mapping[str, float]],
    forum_contexts: Mapping[str, Mapping[str, Any]],
    target_catalogs: Mapping[str, Sequence[Mapping[str, Any]]],
    feasible_actions: Sequence[str],
    reviewed_trace_release_sha256: str | None = None,
    committed_attempts: Sequence[Mapping[str, Any]] = (),
    committed_round: Mapping[str, Any] | None = None,
    artifact_callback: ArtifactCallback | None = None,
) -> dict[str, Any]:
    """Execute or exactly resume one three-actor ForumTwin governance round."""

    if branch.lower() == "natural":
        raise ForumActorExecutionError("Natural cannot create governance actor slots")
    if branch not in {"A", "B", "C", "D"}:
        raise ForumActorExecutionError("governance actor branch is invalid")
    for values, label in (
        (private_observations, "private observations"),
        (forum_contexts, "forum contexts"),
        (target_catalogs, "target catalogs"),
    ):
        if set(values) != set(ACTOR_IDS):
            raise ForumActorExecutionError(f"{label} must cover all three actors")
    current_resource = _resource_view(state)
    if committed_round is None:
        resource_before = current_resource
    else:
        stored_resource_before = committed_round.get("resource_before")
        if not isinstance(stored_resource_before, Mapping):
            raise ForumActorExecutionError(
                "committed governance round lacks its pre-decision resource view"
            )
        resource_before = dict(stored_resource_before)
    rendered_by_actor: dict[str, RenderedForumGovernanceTurn] = {}
    slots = []
    event_sequence = []
    for actor in ACTOR_IDS:
        rendered = render_forum_governance_actor_turn_v2(
            actor_id=actor,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            private_observation=private_observations[actor],
            forum_context=forum_contexts[actor],
            target_catalog=target_catalogs[actor],
            feasible_actions=feasible_actions,
            resource=resource_before,
        )
        rendered_by_actor[actor] = rendered
        slots.append(
            _slot_design(
                rendered=rendered,
                private_observation=private_observations[actor],
                forum_context=forum_contexts[actor],
                target_catalog=target_catalogs[actor],
            )
        )
    slot_by_actor = {str(slot["actor_id"]): slot for slot in slots}
    for slot in slots:
        event_sequence.append(f"slot_design:{slot['actor_id']}")
        if artifact_callback is not None:
            artifact_callback("slot_design", slot)

    prior_attempts: dict[str, Mapping[str, Any]] = {}
    for attempt in committed_attempts:
        actor = str(attempt.get("actor_id") or "")
        if actor not in ACTOR_IDS or actor in prior_attempts:
            raise ForumActorExecutionError("committed attempts contain an unknown or duplicate actor")
        prior_attempts[actor] = attempt
    if committed_round is not None and set(prior_attempts) != set(ACTOR_IDS):
        raise ForumActorExecutionError("committed round recovery requires all three committed attempts")

    decisions: dict[str, ForumGovernanceDecision] = {}
    attempts: dict[str, Mapping[str, Any]] = {}
    for actor in ACTOR_IDS:
        slot = slot_by_actor[actor]
        rendered = rendered_by_actor[actor]
        _validate_slot_design(slot, slot)
        if actor in prior_attempts:
            decision = _restore_committed_attempt(
                artifact=prior_attempts[actor],
                slot=slot,
                rendered=rendered,
                reviewed_trace_release_sha256=reviewed_trace_release_sha256,
            )
            decisions[actor] = decision
            attempts[actor] = prior_attempts[actor]
            event_sequence.append(f"attempt_reused:{actor}")
            continue
        if source is None:
            raise ForumActorExecutionError("uncommitted actor slot requires a semantic turn source")
        envelope = source.acquire(
            slot_kind="governance_actor",
            prompt_sha256=rendered.prompt_sha256,
            rendered=rendered,
            validator=lambda candidate, rendered=rendered: (
                validate_forum_governance_actor_turn_v2(
                    candidate,
                    rendered=rendered,
                )
            ),
        )
        if envelope.provenance != getattr(source, "provenance", None):
            raise ForumActorExecutionError(
                "semantic source and actor response provenance disagree"
            )
        acquired = _acquired_attempt_artifact(slot=slot, envelope=envelope)
        event_sequence.append(f"attempt_acquired:{actor}")
        if artifact_callback is not None:
            artifact_callback("attempt_acquired", acquired)
        decision = validate_forum_governance_actor_turn_v2(
            envelope,
            rendered=rendered,
        )
        if envelope.provenance == "reviewed_trace_replay":
            if (
                reviewed_trace_release_sha256 is None
                or _SHA256_RE.fullmatch(reviewed_trace_release_sha256) is None
            ):
                raise ForumActorExecutionError("replay actor round must bind a reviewed trace release")
        committed = _committed_attempt_artifact(
            acquired=acquired,
            decision=decision,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        )
        decisions[actor] = decision
        attempts[actor] = committed
        event_sequence.append(f"attempt_committed:{actor}")
        if artifact_callback is not None:
            artifact_callback("attempt_committed", committed)

    # A governance round is a semantic identity, not a transport-usage
    # identity.  ``attempt_sha256`` deliberately seals provider/cache usage,
    # so it changes when an exact live response is replayed from the durable
    # cache.  Basing public message IDs on that hash made an otherwise exact
    # checkpoint resume produce different message/thread IDs.  The attempt ID
    # already binds the immutable slot and response fingerprints and therefore
    # remains identical across live acquisition and exact-cache replay.
    round_id = "forum-governance-round-" + canonical_hash(
        [
            scenario_id,
            branch,
            tick,
            seed,
            [slots[index]["slot_design_sha256"] for index in range(len(slots))],
            [attempts[actor]["attempt_id"] for actor in ACTOR_IDS],
        ]
    )[:24]
    if committed_round is not None:
        batch = _validate_committed_round(
            committed_round=committed_round,
            expected_round_id=round_id,
            attempts=attempts,
            state=state,
            ledger=ledger,
        )
        event_sequence.append("committed_round_reused")
        result_body = {
            "slot_designs": slots,
            "attempts": [attempts[actor] for actor in ACTOR_IDS],
            "governance_round": dict(committed_round),
            "thread_batch": batch,
            "recovered_committed_attempt_count": len(prior_attempts),
            "recovered_committed_round": True,
            "source_acquire_count": 0,
            "event_sequence": event_sequence,
        }
        return {**result_body, "execution_sha256": canonical_hash(result_body)}

    working_state = state.clone()
    working_ledger = copy.deepcopy(ledger)
    qre_public, qre_private = play_forum_governance_round_v2(
        state=working_state,
        actor_decisions=decisions,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
    )
    event_sequence.append("qre_resource_solved")
    message_ids, new_thread_ids = _materialize_governance_messages(
        ledger=working_ledger,
        governance_messages=qre_public["governance_messages"],
        attempts=attempts,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        round_id=round_id,
    )
    working_ledger.assert_consistent()
    event_sequence.append("forum_ledger_materialized")
    batch = _thread_batch(
        ledger=working_ledger,
        round_id=round_id,
        message_ids=message_ids,
        new_thread_ids=new_thread_ids,
    )
    state.resource_opening = working_state.resource_opening
    state.resource_consumed = working_state.resource_consumed
    state.resource_committed = working_state.resource_committed
    state.pending_actions = copy.deepcopy(working_state.pending_actions)
    state.game_rounds = copy.deepcopy(working_state.game_rounds)
    ledger.threads = copy.deepcopy(working_ledger.threads)
    ledger.messages = copy.deepcopy(working_ledger.messages)
    ledger.claims = copy.deepcopy(working_ledger.claims)
    resource_after = _resource_view(state)
    round_body = {
        "schema_version": ACTOR_ROUND_SCHEMA,
        "round_id": round_id,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "seed": seed,
        "slot_design_sha256s": {
            actor: slot_by_actor[actor]["slot_design_sha256"] for actor in ACTOR_IDS
        },
        "attempt_sha256s": {
            actor: attempts[actor]["attempt_sha256"] for actor in ACTOR_IDS
        },
        "qre_public": qre_public,
        "qre_private_sha256": canonical_hash(qre_private),
        "state_game_round_sha256": canonical_hash(state.game_rounds[-1]),
        "resource_before": resource_before,
        "resource_after": resource_after,
        "message_ids": message_ids,
        "message_sha256s": {
            message_id: canonical_hash(ledger.messages[message_id].public_view())
            for message_id in message_ids
        },
        "new_thread_ids": new_thread_ids,
        "thread_batch_sha256": batch["thread_batch_sha256"],
        "uses_hidden_truth": False,
        "scheme_bonus": 0,
    }
    governance_round = {**round_body, "round_sha256": canonical_hash(round_body)}
    if artifact_callback is not None:
        artifact_callback("governance_round", governance_round)
        artifact_callback("thread_batch", batch)
    result_body = {
        "slot_designs": slots,
        "attempts": [attempts[actor] for actor in ACTOR_IDS],
        "governance_round": governance_round,
        "thread_batch": batch,
        "recovered_committed_attempt_count": len(prior_attempts),
        "recovered_committed_round": False,
        "source_acquire_count": len(ACTOR_IDS) - len(prior_attempts),
        "event_sequence": event_sequence,
    }
    return {**result_body, "execution_sha256": canonical_hash(result_body)}
