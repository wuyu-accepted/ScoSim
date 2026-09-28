"""Pure ForumTwin domain contracts for the LLM-first parallel forum.

The module intentionally owns no Provider, database, HTTP, particle-update or
release-packaging code.  It is the narrow boundary that prevents the silent
background Dynamics layer from becoming a language generator: a public forum
message is accepted only when it is bound to an authorized live LLM response
or an immutable reviewed trace replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .governance_arena_v2 import DECISION_TICKS, TIMELINE
from .mass_population import canonical_hash


THREAD_SCHEMA_VERSION = "campus-pulse-forum-thread-v1"
MESSAGE_SCHEMA_VERSION = "campus-pulse-forum-message-v1"
MESSAGE_V2_SCHEMA_VERSION = "campus-pulse-forum-message-v2"
INTERACTION_SCHEMA_VERSION = "campus-pulse-forum-interaction-v1"
CLAIM_SCHEMA_VERSION = "campus-pulse-forum-claim-v1"
EXPOSURE_SCHEMA_VERSION = "campus-pulse-feed-exposure-v1"
MEMORY_SCHEMA_VERSION = "campus-pulse-forum-agent-memory-v1"
TURN_SCHEMA_VERSION = "campus-pulse-forum-agent-turn-v1"
GOVERNANCE_ARTIFACT_SCHEMA_VERSION = "campus-pulse-governance-artifact-v1"
PRIVATE_CONVERSATION_SCHEMA_VERSION = "campus-pulse-private-conversation-v1"
PRIVATE_MESSAGE_SCHEMA_VERSION = "campus-pulse-private-message-v1"
PRIVATE_CLAIM_SCHEMA_VERSION = "campus-pulse-private-claim-v1"
PRIVATE_REACTION_SCHEMA_VERSION = "campus-pulse-private-reaction-v1"
PRIVATE_MEMBERSHIP_EVENT_SCHEMA_VERSION = (
    "campus-pulse-private-membership-event-v1"
)

PUBLIC_PROVENANCE_KINDS = frozenset(
    {"authorized_live_llm", "reviewed_trace_replay"}
)
PUBLIC_ACTIONS = frozenset(
    {
        "post",
        "reply",
        "quote",
        "repost",
        "verify",
        "correct",
        "seek_help",
        "report_risk",
        "governance_message",
        "service_receipt",
        "governance_evidence",
    }
)
INTERACTION_KINDS = frozenset({"like", "repost", "report"})
REPORT_REASON_TAGS = frozenset(
    {"harassment", "misinformation", "privacy", "safety", "spam", "other"}
)
ROOT_ACTIONS = frozenset(
    {"post", "seek_help", "report_risk", "governance_message"}
)
CLAIM_REQUIRED_ACTIONS = frozenset(
    {"verify", "correct", "service_receipt", "governance_evidence"}
)
CLAIM_OPERATIONS = frozenset(
    {"none", "create", "support", "challenge", "verify", "correct"}
)
STANCE_VALUES = frozenset({"support", "oppose", "mixed", "uncertain", "neutral"})
CLAIM_STATUSES = frozenset(
    {"unverified", "contested", "corrected", "verified", "expired"}
)
EVIDENCE_STATUSES = frozenset(
    {"unverified", "uncertain", "contested", "corrected", "verified"}
)
MODERATION_STATUSES = frozenset({"accepted", "superseded"})
READ_STATES = frozenset({"unread", "read", "ignored", "deferred"})
PRIVATE_CONVERSATION_TYPES = frozenset({"direct", "group"})
PRIVATE_CONVERSATION_STATUSES = frozenset(
    {"pending", "active", "expired", "dissolved"}
)
PRIVATE_MEMBERSHIP_ACTIONS = frozenset(
    {"invite", "accept", "decline", "leave", "dissolve"}
)
PRIVATE_MESSAGE_ACTIONS = frozenset(
    {
        "send_private",
        "reply_private",
        "forward_public_to_private",
        "forward_private_to_private",
    }
)
PRIVATE_CLAIM_STATUSES = frozenset(
    {"hearsay", "speculation", "contested", "corrected", "verified", "expired"}
)
PRIVATE_EVIDENCE_STATUSES = EVIDENCE_STATUSES | frozenset(
    {"hearsay", "speculation"}
)
PRIVATE_REACTION_KINDS = frozenset(
    {"agree", "question", "laugh", "concern", "acknowledge"}
)

MAX_FEED_ITEMS = 6
MAX_FORUM_CONTEXT_CHARS = 1_200
MAX_PRIVATE_CONTEXT_CHARS = 600
MAX_DYNAMIC_CONTEXT_CHARS = 1_800
MAX_RECENT_REAL_LLM_EPISODES = 4
MAX_RECENT_PRIVATE_LLM_EPISODES = 4
ANCHOR_COUNT = 16
STANDARD_PPS_BUDGET = 32
MAX_RESIDENT_TURNS = 64
MAX_GOVERNANCE_TURNS = 3
MAX_TOTAL_TURNS = 67

INTERACTION_CAP_BY_TICK = tuple(
    0 if tick <= 2 else 24 if tick <= 7 else 16 if tick <= 17 else 8
    for tick in range(len(TIMELINE))
)
INTERACTION_REASON_PRIORITY = {
    "direct_reply": 0,
    "quote": 0,
    "correction": 0,
    "service_receipt": 0,
    "risk_report": 1,
    "help_request": 1,
    "active_claim_participant": 2,
    "novel_bridge_exposure": 3,
    "high_attention_silent": 4,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/(?:home|Users|srv|var|tmp)/|\.\.[\\/]|\\\\)"
)
_CREDENTIAL_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|api[_ -]?key|password|secret)", re.I
)
_FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "source_id",
        "source_ids",
        "post_code",
        "comment_id",
        "raw_text",
        "original_text",
        "original_body",
        "provider_body",
        "api_key",
        "password",
        "credential",
        "filesystem_path",
        "database_path",
        "hidden_truth",
        "population_truth",
        "future_messages",
        "future_events",
        "target_share",
        "target_result",
    }
)
_DYNAMICS_ALLOWED_KEYS = frozenset(
    {
        "silent_agent_updates",
        "read_states",
        "attention",
        "forgetting",
        "activity_probabilities",
        "feed_deliveries",
        "ttl_expirations",
        "service_queue",
        "resources",
        "execution_delays",
        "particles",
        "aggregate_statistics",
        "rng_counters",
    }
)
_DYNAMICS_FORBIDDEN_KEYS = frozenset(
    {
        "visible_text",
        "message",
        "messages",
        "forum_message",
        "forum_messages",
        "claim_summary",
        "claim_text",
        "representative_speech",
        "synthetic_message",
        "private_message",
        "private_messages",
        "private_claim",
        "private_claims",
        "provenance",
    }
)


class ForumTwinError(ValueError):
    """Raised when a ForumTwin domain invariant is violated."""


def _require_nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ForumTwinError(f"{label} must be non-empty")
    return text


def _require_tick(value: int, label: str = "tick") -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(TIMELINE):
        raise ForumTwinError(f"{label} must be a ForumTwin tick")
    return value


def _require_probability(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ForumTwinError(f"{label} must be between zero and one")
    return number


def _validated_visible_text(action: str, text: str) -> str:
    normalized = " ".join(_require_nonempty(text, "visible_text").split())
    limit = 180 if action in ROOT_ACTIONS else 120
    if len(normalized) > limit:
        raise ForumTwinError(
            f"{action} visible_text exceeds the {limit}-character limit"
        )
    if _PATH_RE.search(normalized) or _CREDENTIAL_RE.search(normalized):
        raise ForumTwinError("visible_text contains a path or credential marker")
    return normalized


def _validated_private_text(text: str) -> str:
    normalized = " ".join(_require_nonempty(text, "private_text").split())
    if len(normalized) > 180:
        raise ForumTwinError("private_text exceeds the 180-character limit")
    if _PATH_RE.search(normalized) or _CREDENTIAL_RE.search(normalized):
        raise ForumTwinError("private_text contains a path or credential marker")
    return normalized


@dataclass(frozen=True)
class PublicProvenance:
    """Immutable semantic provenance for a public message or memory episode."""

    kind: str
    prompt_sha256: str
    response_sha256: str
    model: str
    trace_release_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in PUBLIC_PROVENANCE_KINDS:
            raise ForumTwinError("public content must originate from live LLM or reviewed replay")
        if _SHA256_RE.fullmatch(self.prompt_sha256) is None:
            raise ForumTwinError("public provenance prompt SHA-256 is invalid")
        if _SHA256_RE.fullmatch(self.response_sha256) is None:
            raise ForumTwinError("public provenance response SHA-256 is invalid")
        _require_nonempty(self.model, "provenance model")
        if self.kind == "reviewed_trace_replay":
            if not self.trace_release_sha256 or _SHA256_RE.fullmatch(self.trace_release_sha256) is None:
                raise ForumTwinError("reviewed replay must bind an immutable trace release")
        elif self.trace_release_sha256 is not None:
            raise ForumTwinError("live LLM provenance cannot claim a trace release")

    def public_view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "model": self.model,
            "trace_release_sha256": self.trace_release_sha256,
        }


@dataclass(frozen=True)
class ForumThread:
    thread_id: str
    board: str
    root_message_id: str
    scenario_id: str
    branch: str
    created_tick: int
    status: str
    topic: str
    need: str
    claim_ids: tuple[str, ...] = ()
    participant_count: int = 0
    reply_count: int = 0
    last_active_tick: int | None = None
    provenance_kind: str = "authorized_live_llm"

    def __post_init__(self) -> None:
        for value, label in (
            (self.thread_id, "thread_id"),
            (self.board, "board"),
            (self.root_message_id, "root_message_id"),
            (self.scenario_id, "scenario_id"),
            (self.branch, "branch"),
            (self.topic, "topic"),
            (self.need, "need"),
        ):
            _require_nonempty(value, label)
        _require_tick(self.created_tick, "thread created_tick")
        if self.status not in {"open", "closed", "archived"}:
            raise ForumTwinError("thread status is invalid")
        if self.provenance_kind not in PUBLIC_PROVENANCE_KINDS:
            raise ForumTwinError("thread provenance is not publishable")
        if self.participant_count < 0 or self.reply_count < 0:
            raise ForumTwinError("thread counters cannot be negative")
        if self.last_active_tick is not None:
            _require_tick(self.last_active_tick, "thread last_active_tick")
            if self.last_active_tick < self.created_tick:
                raise ForumTwinError("thread activity predates its creation")
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise ForumTwinError("thread claim IDs must be unique")


@dataclass(frozen=True)
class GovernanceArtifact:
    """A persistent, bounded governance object carried by an LLM message."""

    artifact_id: str
    artifact_kind: str
    action_id: str
    responsible_role: str
    status: str
    created_tick: int
    due_tick: int
    expires_tick: int
    target_thread_id: str | None
    target_claim_id: str | None
    public_scope: str
    confirmed_scope: tuple[str, ...]
    unresolved_scope: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.artifact_id, "governance artifact_id"),
            (self.action_id, "governance action_id"),
            (self.responsible_role, "governance responsible_role"),
            (self.public_scope, "governance public_scope"),
        ):
            _require_nonempty(value, label)
        if self.artifact_kind not in {
            "evidence_card",
            "service_ticket",
            "cross_group_outreach",
        }:
            raise ForumTwinError("governance artifact kind is invalid")
        if self.status not in {"scheduled", "active", "expired"}:
            raise ForumTwinError("governance artifact status is invalid")
        _require_tick(self.created_tick, "artifact created_tick")
        _require_tick(self.due_tick, "artifact due_tick")
        _require_tick(self.expires_tick, "artifact expires_tick")
        if not self.created_tick <= self.due_tick <= self.expires_tick:
            raise ForumTwinError("governance artifact lifecycle is invalid")
        if not self.confirmed_scope or not self.unresolved_scope:
            raise ForumTwinError("governance artifact must state confirmed and unresolved scope")
        for value in (*self.confirmed_scope, *self.unresolved_scope):
            if not isinstance(value, str) or not value.strip() or len(value) > 160:
                raise ForumTwinError("governance artifact scope item is invalid")
            if _PATH_RE.search(value) or _CREDENTIAL_RE.search(value):
                raise ForumTwinError("governance artifact contains a sensitive marker")

    def public_view(self) -> dict[str, Any]:
        return {
            "schema": GOVERNANCE_ARTIFACT_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "action_id": self.action_id,
            "responsible_role": self.responsible_role,
            "status": self.status,
            "created_tick": self.created_tick,
            "due_tick": self.due_tick,
            "expires_tick": self.expires_tick,
            "target_thread_id": self.target_thread_id,
            "target_claim_id": self.target_claim_id,
            "public_scope": self.public_scope,
            "confirmed_scope": list(self.confirmed_scope),
            "unresolved_scope": list(self.unresolved_scope),
        }


@dataclass(frozen=True)
class ForumMessage:
    message_id: str
    thread_id: str
    source_agent_display_id: str
    action: str
    visible_text: str
    topic: str
    stance: str
    emotion: str
    evidence_status: str
    confidence: float
    created_tick: int
    ttl: int
    provenance: PublicProvenance
    parent_message_id: str | None = None
    quote_message_id: str | None = None
    claim_id: str | None = None
    claim_operation: str = "none"
    correction_target_claim_id: str | None = None
    moderation_status: str = "accepted"
    governance_artifact: GovernanceArtifact | None = None
    private_hearsay_origin: bool = False
    private_origin_claim_id: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.message_id, "message_id"),
            (self.thread_id, "thread_id"),
            (self.source_agent_display_id, "source_agent_display_id"),
            (self.topic, "message topic"),
            (self.stance, "message stance"),
            (self.emotion, "message emotion"),
        ):
            _require_nonempty(value, label)
        if self.action not in PUBLIC_ACTIONS:
            raise ForumTwinError("ForumMessage action is not public or supported")
        if not isinstance(self.provenance, PublicProvenance):
            raise ForumTwinError("ForumMessage provenance object is invalid")
        object.__setattr__(self, "visible_text", _validated_visible_text(self.action, self.visible_text))
        _require_tick(self.created_tick, "message created_tick")
        if not isinstance(self.ttl, int) or isinstance(self.ttl, bool) or self.ttl < 1:
            raise ForumTwinError("message TTL must be a positive integer")
        if self.evidence_status not in EVIDENCE_STATUSES:
            raise ForumTwinError("message evidence status is invalid")
        if self.stance not in STANCE_VALUES:
            raise ForumTwinError("message stance is invalid")
        if self.claim_operation not in CLAIM_OPERATIONS:
            raise ForumTwinError("message claim operation is invalid")
        _require_probability(self.confidence, "message confidence")
        if self.moderation_status not in MODERATION_STATUSES:
            raise ForumTwinError("message moderation status is invalid")
        if self.action == "reply" and self.parent_message_id is None:
            raise ForumTwinError("reply must target a parent message")
        if self.action == "quote" and self.quote_message_id is None:
            raise ForumTwinError("quote must target a quoted message")
        if self.action in CLAIM_REQUIRED_ACTIONS and self.claim_id is None:
            raise ForumTwinError(f"{self.action} must cite a claim")
        if self.claim_operation == "create" and self.claim_id is None:
            raise ForumTwinError("claim creation must declare the new claim ID")
        if self.action == "correct" and self.correction_target_claim_id is None:
            raise ForumTwinError("correction must identify its target claim")
        if self.correction_target_claim_id is not None and self.action != "correct":
            raise ForumTwinError("only a correction may set correction_target_claim_id")
        if self.parent_message_id == self.message_id or self.quote_message_id == self.message_id:
            raise ForumTwinError("message cannot refer to itself")
        if self.governance_artifact is not None and (
            self.action != "governance_message"
            or not self.source_agent_display_id.startswith("governance:")
            or self.governance_artifact.created_tick != self.created_tick
        ):
            raise ForumTwinError("governance artifact is not bound to its governance message")
        if self.private_hearsay_origin:
            if self.private_origin_claim_id is None:
                raise ForumTwinError("private-origin public message must bind a private claim")
            if self.claim_operation != "create" or self.claim_id is None:
                raise ForumTwinError("private hearsay must create a new public claim")
        elif self.private_origin_claim_id is not None:
            raise ForumTwinError("private origin claim requires private_hearsay_origin")

    @property
    def superseded(self) -> bool:
        return self.moderation_status == "superseded"

    def active_at(self, tick: int) -> bool:
        _require_tick(tick)
        return not self.superseded and self.created_tick <= tick <= self.created_tick + self.ttl

    def public_view(self) -> dict[str, Any]:
        return {
            "schema": MESSAGE_SCHEMA_VERSION,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "parent_message_id": self.parent_message_id,
            "quote_message_id": self.quote_message_id,
            "source_agent_display_id": self.source_agent_display_id,
            "action": self.action,
            "visible_text": self.visible_text,
            "topic": self.topic,
            "stance": self.stance,
            "emotion": self.emotion,
            "evidence_status": self.evidence_status,
            "confidence": self.confidence,
            "claim_id": self.claim_id,
            "claim_operation": self.claim_operation,
            "correction_target_claim_id": self.correction_target_claim_id,
            "created_tick": self.created_tick,
            "ttl": self.ttl,
            "moderation_status": self.moderation_status,
            "private_hearsay_origin": self.private_hearsay_origin,
            "governance_artifact": (
                self.governance_artifact.public_view()
                if self.governance_artifact is not None
                else None
            ),
            "provenance": self.provenance.public_view(),
        }


@dataclass(frozen=True)
class ForumMessageV2(ForumMessage):
    """A public forum message with an optional 小喇叭-style anonymous face."""

    author_visibility: str = "named"
    anonymous_pseudonym_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.author_visibility not in {"named", "anonymous"}:
            raise ForumTwinError("forum author visibility is invalid")
        if self.author_visibility == "anonymous":
            _require_nonempty(
                self.anonymous_pseudonym_id,
                "anonymous pseudonym id",
            )
            if self.source_agent_display_id.startswith("governance:"):
                raise ForumTwinError("governance messages cannot hide their role")
        elif self.anonymous_pseudonym_id is not None:
            raise ForumTwinError("named forum message cannot carry a pseudonym")

    def public_view(self) -> dict[str, Any]:
        value = super().public_view()
        value["schema"] = MESSAGE_V2_SCHEMA_VERSION
        value["author_visibility"] = self.author_visibility
        value["source_agent_display_id"] = (
            str(self.anonymous_pseudonym_id)
            if self.author_visibility == "anonymous"
            else self.source_agent_display_id
        )
        return value


@dataclass(frozen=True)
class ForumInteraction:
    """One non-speech interaction emitted by an activated LLM resident.

    A resident turn may still create at most one public message, but it may
    independently like, repost or report several time-visible messages in the
    same tick.  Keeping these events separate prevents a reply from erasing a
    simultaneous like/report and prevents low-cost Dynamics from inventing
    visible engagement.
    """

    interaction_id: str
    scenario_id: str
    branch: str
    thread_id: str
    target_message_id: str
    source_agent_display_id: str
    kind: str
    created_tick: int
    provenance: PublicProvenance
    report_reason_tag: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.interaction_id, "interaction_id"),
            (self.scenario_id, "interaction scenario_id"),
            (self.branch, "interaction branch"),
            (self.thread_id, "interaction thread_id"),
            (self.target_message_id, "interaction target_message_id"),
            (self.source_agent_display_id, "interaction source_agent_display_id"),
        ):
            _require_nonempty(value, label)
        if self.kind not in INTERACTION_KINDS:
            raise ForumTwinError("forum interaction kind is invalid")
        _require_tick(self.created_tick, "interaction created_tick")
        if not isinstance(self.provenance, PublicProvenance):
            raise ForumTwinError("forum interaction provenance is invalid")
        if self.kind == "report":
            if self.report_reason_tag not in REPORT_REASON_TAGS:
                raise ForumTwinError("forum report reason tag is invalid")
        elif self.report_reason_tag is not None:
            raise ForumTwinError("only report interactions may carry a reason tag")

    def public_view(self) -> dict[str, Any]:
        return {
            "schema": INTERACTION_SCHEMA_VERSION,
            "interaction_id": self.interaction_id,
            "scenario_id": self.scenario_id,
            "branch": self.branch,
            "thread_id": self.thread_id,
            "target_message_id": self.target_message_id,
            "source_agent_display_id": self.source_agent_display_id,
            "kind": self.kind,
            "created_tick": self.created_tick,
            "report_reason_tag": self.report_reason_tag,
            "provenance": self.provenance.public_view(),
        }


@dataclass(frozen=True)
class ForumClaim:
    claim_id: str
    thread_id: str
    claim_summary: str
    status: str
    first_seen_tick: int
    last_seen_tick: int
    created_by_message_id: str
    parent_claim_id: str | None = None
    supporting_message_ids: tuple[str, ...] = ()
    challenging_message_ids: tuple[str, ...] = ()
    correction_target_claim_id: str | None = None
    evidence_reference: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.claim_id, "claim_id"),
            (self.thread_id, "claim thread_id"),
            (self.claim_summary, "claim_summary"),
            (self.created_by_message_id, "created_by_message_id"),
        ):
            _require_nonempty(value, label)
        if len(self.claim_summary) > 160:
            raise ForumTwinError("claim summary exceeds 160 characters")
        if _PATH_RE.search(self.claim_summary) or _CREDENTIAL_RE.search(self.claim_summary):
            raise ForumTwinError("claim summary contains a path or credential marker")
        if self.status not in CLAIM_STATUSES:
            raise ForumTwinError("claim status is invalid")
        _require_tick(self.first_seen_tick, "claim first_seen_tick")
        _require_tick(self.last_seen_tick, "claim last_seen_tick")
        if self.last_seen_tick < self.first_seen_tick:
            raise ForumTwinError("claim last_seen_tick predates first_seen_tick")
        if self.parent_claim_id == self.claim_id or self.correction_target_claim_id == self.claim_id:
            raise ForumTwinError("claim cannot refer to itself")
        if len(set(self.supporting_message_ids)) != len(self.supporting_message_ids):
            raise ForumTwinError("claim supporting messages must be unique")
        if len(set(self.challenging_message_ids)) != len(self.challenging_message_ids):
            raise ForumTwinError("claim challenging messages must be unique")

    @property
    def expired(self) -> bool:
        return self.status == "expired"


@dataclass(frozen=True)
class PrivateConversation:
    """Private synthetic conversation; never part of the public forum export."""

    conversation_id: str
    scenario_id: str
    branch: str
    conversation_type: str
    creator_agent_id: str
    member_agent_ids: tuple[str, ...]
    pending_invite_agent_ids: tuple[str, ...]
    left_agent_ids: tuple[str, ...]
    created_tick: int
    status: str
    invitation_expires_tick: int
    topic_tags: tuple[str, ...]
    provenance: PublicProvenance

    def __post_init__(self) -> None:
        for value, label in (
            (self.conversation_id, "private conversation_id"),
            (self.scenario_id, "private scenario_id"),
            (self.branch, "private branch"),
            (self.creator_agent_id, "private creator_agent_id"),
        ):
            _require_nonempty(value, label)
        _require_tick(self.created_tick, "private conversation created_tick")
        if self.conversation_type not in PRIVATE_CONVERSATION_TYPES:
            raise ForumTwinError("private conversation type is invalid")
        if self.status not in PRIVATE_CONVERSATION_STATUSES:
            raise ForumTwinError("private conversation status is invalid")
        if not isinstance(self.provenance, PublicProvenance):
            raise ForumTwinError("private conversation provenance is invalid")
        members = tuple(self.member_agent_ids)
        pending = tuple(self.pending_invite_agent_ids)
        left = tuple(self.left_agent_ids)
        if (
            len(set(members)) != len(members)
            or len(set(pending)) != len(pending)
            or len(set(left)) != len(left)
            or set(members) & set(pending)
            or set(members) & set(left)
            or set(pending) & set(left)
        ):
            raise ForumTwinError("private conversation membership sets overlap or repeat")
        if self.creator_agent_id not in members and self.status != "dissolved":
            raise ForumTwinError("private conversation creator must remain a member")
        if not isinstance(self.invitation_expires_tick, int) or isinstance(
            self.invitation_expires_tick, bool
        ):
            raise ForumTwinError("private invitation expiry is invalid")
        if not self.created_tick <= self.invitation_expires_tick <= len(TIMELINE) + 2:
            raise ForumTwinError("private invitation expiry is outside its bounded horizon")
        if not 1 <= len(self.topic_tags) <= 4 or len(set(self.topic_tags)) != len(
            self.topic_tags
        ):
            raise ForumTwinError("private conversation requires 1-4 unique topic tags")
        for tag in self.topic_tags:
            _require_nonempty(tag, "private topic tag")
            if len(tag) > 60:
                raise ForumTwinError("private topic tag exceeds 60 characters")
        if self.conversation_type == "direct":
            if len(members) != 2 or pending or self.status != "active":
                raise ForumTwinError("direct conversation requires exactly two active members")
        else:
            if len(members) > 12 or len(pending) > 11:
                raise ForumTwinError("private group exceeds its twelve-member ceiling")
            if self.status == "pending" and len(members) >= 3:
                raise ForumTwinError("three accepted group members must activate the group")
            if self.status == "active" and len(members) < 3:
                raise ForumTwinError("active private group requires at least three members")

    def private_view(self) -> dict[str, Any]:
        return {
            "schema": PRIVATE_CONVERSATION_SCHEMA_VERSION,
            "conversation_id": self.conversation_id,
            "scenario_id": self.scenario_id,
            "branch": self.branch,
            "conversation_type": self.conversation_type,
            "creator_agent_id": self.creator_agent_id,
            "member_agent_ids": list(self.member_agent_ids),
            "pending_invite_agent_ids": list(self.pending_invite_agent_ids),
            "left_agent_ids": list(self.left_agent_ids),
            "created_tick": self.created_tick,
            "status": self.status,
            "invitation_expires_tick": self.invitation_expires_tick,
            "topic_tags": list(self.topic_tags),
            "provenance": self.provenance.public_view(),
        }


@dataclass(frozen=True)
class PrivateMembershipEvent:
    event_id: str
    conversation_id: str
    actor_agent_id: str
    subject_agent_id: str
    action: str
    created_tick: int
    provenance: PublicProvenance

    def __post_init__(self) -> None:
        for value, label in (
            (self.event_id, "private membership event_id"),
            (self.conversation_id, "private membership conversation_id"),
            (self.actor_agent_id, "private membership actor_agent_id"),
            (self.subject_agent_id, "private membership subject_agent_id"),
        ):
            _require_nonempty(value, label)
        if self.action not in PRIVATE_MEMBERSHIP_ACTIONS:
            raise ForumTwinError("private membership action is invalid")
        _require_tick(self.created_tick, "private membership event tick")
        if not isinstance(self.provenance, PublicProvenance):
            raise ForumTwinError("private membership provenance is invalid")


@dataclass(frozen=True)
class PrivateMessage:
    message_id: str
    conversation_id: str
    source_agent_id: str
    action: str
    visible_text: str
    stance: str
    emotion: str
    evidence_status: str
    created_tick: int
    provenance: PublicProvenance
    parent_message_id: str | None = None
    quote_message_id: str | None = None
    public_source_message_id: str | None = None
    forwarded_private_source_message_id: str | None = None
    private_claim_id: str | None = None
    claim_operation: str = "none"
    correction_target_claim_id: str | None = None
    moderation_status: str = "accepted"

    def __post_init__(self) -> None:
        for value, label in (
            (self.message_id, "private message_id"),
            (self.conversation_id, "private message conversation_id"),
            (self.source_agent_id, "private message source_agent_id"),
            (self.stance, "private message stance"),
            (self.emotion, "private message emotion"),
        ):
            _require_nonempty(value, label)
        if self.action not in PRIVATE_MESSAGE_ACTIONS:
            raise ForumTwinError("private message action is invalid")
        object.__setattr__(self, "visible_text", _validated_private_text(self.visible_text))
        if self.stance not in STANCE_VALUES:
            raise ForumTwinError("private message stance is invalid")
        if self.evidence_status not in PRIVATE_EVIDENCE_STATUSES:
            raise ForumTwinError("private message evidence status is invalid")
        if self.claim_operation not in CLAIM_OPERATIONS:
            raise ForumTwinError("private message claim operation is invalid")
        if self.moderation_status not in MODERATION_STATUSES:
            raise ForumTwinError("private message moderation status is invalid")
        _require_tick(self.created_tick, "private message created_tick")
        if not isinstance(self.provenance, PublicProvenance):
            raise ForumTwinError("private message provenance is invalid")
        if self.action == "reply_private" and self.parent_message_id is None:
            raise ForumTwinError("private reply requires a parent private message")
        if (
            self.action == "forward_public_to_private"
            and self.public_source_message_id is None
        ):
            raise ForumTwinError("public-to-private forward requires its public source")
        if (
            self.action == "forward_private_to_private"
            and self.forwarded_private_source_message_id is None
        ):
            raise ForumTwinError("private forward requires its private source")
        if (
            self.action != "forward_private_to_private"
            and self.forwarded_private_source_message_id is not None
        ):
            raise ForumTwinError("only private forwarding may bind a cross-conversation source")
        if self.claim_operation == "create" and self.private_claim_id is None:
            raise ForumTwinError("private claim creation requires a new private claim ID")
        if self.claim_operation != "none" and self.private_claim_id is None:
            raise ForumTwinError("private claim operation requires a private claim")
        if self.correction_target_claim_id is not None and self.claim_operation != "correct":
            raise ForumTwinError("only private correction may name a correction target")
        if self.claim_operation == "correct" and self.correction_target_claim_id is None:
            raise ForumTwinError("private correction requires its target")
        if self.message_id in {self.parent_message_id, self.quote_message_id}:
            raise ForumTwinError("private message cannot refer to itself")

    @property
    def superseded(self) -> bool:
        return self.moderation_status == "superseded"

    def private_view(self) -> dict[str, Any]:
        return {
            "schema": PRIVATE_MESSAGE_SCHEMA_VERSION,
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "source_agent_id": self.source_agent_id,
            "action": self.action,
            "visible_text": self.visible_text,
            "stance": self.stance,
            "emotion": self.emotion,
            "evidence_status": self.evidence_status,
            "created_tick": self.created_tick,
            "parent_message_id": self.parent_message_id,
            "quote_message_id": self.quote_message_id,
            "public_source_message_id": self.public_source_message_id,
            "forwarded_private_source_message_id": self.forwarded_private_source_message_id,
            "private_claim_id": self.private_claim_id,
            "claim_operation": self.claim_operation,
            "correction_target_claim_id": self.correction_target_claim_id,
            "moderation_status": self.moderation_status,
            "provenance": self.provenance.public_view(),
        }


@dataclass(frozen=True)
class PrivateReaction:
    reaction_id: str
    conversation_id: str
    target_message_id: str
    source_agent_id: str
    kind: str
    created_tick: int
    provenance: PublicProvenance

    def __post_init__(self) -> None:
        for value, label in (
            (self.reaction_id, "private reaction_id"),
            (self.conversation_id, "private reaction conversation_id"),
            (self.target_message_id, "private reaction target_message_id"),
            (self.source_agent_id, "private reaction source_agent_id"),
        ):
            _require_nonempty(value, label)
        if self.kind not in PRIVATE_REACTION_KINDS:
            raise ForumTwinError("private reaction kind is invalid")
        _require_tick(self.created_tick, "private reaction tick")
        if not isinstance(self.provenance, PublicProvenance):
            raise ForumTwinError("private reaction provenance is invalid")


@dataclass(frozen=True)
class PrivateClaim:
    claim_id: str
    conversation_id: str
    claim_summary: str
    status: str
    first_seen_tick: int
    last_seen_tick: int
    created_by_message_id: str
    origin_kind: str
    origin_public_message_id: str | None = None
    origin_private_message_id: str | None = None
    parent_claim_id: str | None = None
    supporting_message_ids: tuple[str, ...] = ()
    challenging_message_ids: tuple[str, ...] = ()
    correction_target_claim_id: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.claim_id, "private claim_id"),
            (self.conversation_id, "private claim conversation_id"),
            (self.claim_summary, "private claim_summary"),
            (self.created_by_message_id, "private claim creator"),
        ):
            _require_nonempty(value, label)
        if len(self.claim_summary) > 160:
            raise ForumTwinError("private claim summary exceeds 160 characters")
        if _PATH_RE.search(self.claim_summary) or _CREDENTIAL_RE.search(
            self.claim_summary
        ):
            raise ForumTwinError("private claim summary contains a sensitive marker")
        if self.status not in PRIVATE_CLAIM_STATUSES:
            raise ForumTwinError("private claim status is invalid")
        if self.origin_kind not in {"public", "private"}:
            raise ForumTwinError("private claim origin kind is invalid")
        if self.origin_kind == "public":
            if self.origin_public_message_id is None or self.origin_private_message_id is not None:
                raise ForumTwinError("public-origin private claim must bind one public source")
        elif self.origin_private_message_id is None or self.origin_public_message_id is not None:
            raise ForumTwinError("private-origin claim must bind one private source")
        _require_tick(self.first_seen_tick, "private claim first_seen_tick")
        _require_tick(self.last_seen_tick, "private claim last_seen_tick")
        if self.last_seen_tick < self.first_seen_tick:
            raise ForumTwinError("private claim chronology is invalid")
        if self.claim_id in {self.parent_claim_id, self.correction_target_claim_id}:
            raise ForumTwinError("private claim cannot refer to itself")
        if len(set(self.supporting_message_ids)) != len(self.supporting_message_ids):
            raise ForumTwinError("private claim support IDs must be unique")
        if len(set(self.challenging_message_ids)) != len(self.challenging_message_ids):
            raise ForumTwinError("private claim challenge IDs must be unique")


@dataclass(frozen=True)
class FeedExposure:
    """A private delivery edge.  Never serialize this into public results."""

    exposure_id: str
    recipient_agent_id: str
    message_id: str
    delivered_tick: int
    exposure_source: str
    rank_score: float
    read_state: str = "unread"

    def __post_init__(self) -> None:
        for value, label in (
            (self.exposure_id, "exposure_id"),
            (self.recipient_agent_id, "recipient_agent_id"),
            (self.message_id, "exposure message_id"),
            (self.exposure_source, "exposure_source"),
        ):
            _require_nonempty(value, label)
        _require_tick(self.delivered_tick, "exposure delivered_tick")
        if not isinstance(self.rank_score, (int, float)):
            raise ForumTwinError("exposure rank_score must be numeric")
        if not math.isfinite(float(self.rank_score)):
            raise ForumTwinError("exposure rank_score must be finite")
        if self.read_state not in READ_STATES:
            raise ForumTwinError("exposure read_state is invalid")

    def private_view(self) -> dict[str, Any]:
        return {
            "schema": EXPOSURE_SCHEMA_VERSION,
            "exposure_id": self.exposure_id,
            "recipient_agent_id": self.recipient_agent_id,
            "message_id": self.message_id,
            "delivered_tick": self.delivered_tick,
            "exposure_source": self.exposure_source,
            "rank_score": float(self.rank_score),
            "read_state": self.read_state,
        }


@dataclass(frozen=True)
class RealLlmMemoryEpisode:
    tick: int
    thread_id: str
    action: str
    semantic_summary: str
    message_id: str | None
    provenance: PublicProvenance

    def __post_init__(self) -> None:
        _require_tick(self.tick, "memory tick")
        _require_nonempty(self.thread_id, "memory thread_id")
        _require_nonempty(self.action, "memory action")
        if self.action not in PUBLIC_ACTIONS | {"read", "ignore"}:
            raise ForumTwinError("memory action is invalid")
        if self.action in PUBLIC_ACTIONS and self.message_id is None:
            raise ForumTwinError("public real-LLM memory must bind its ForumMessage")
        if self.action in {"read", "ignore"} and self.message_id is not None:
            raise ForumTwinError("silent real-LLM memory cannot bind public speech")
        summary = _require_nonempty(self.semantic_summary, "memory semantic_summary")
        if len(summary) > 240:
            raise ForumTwinError("memory semantic summary exceeds 240 characters")
        if _PATH_RE.search(summary) or _CREDENTIAL_RE.search(summary):
            raise ForumTwinError("memory contains a path or credential marker")


@dataclass(frozen=True)
class PrivateLlmMemoryEpisode:
    tick: int
    conversation_id: str
    relation_type: str
    action: str
    semantic_summary: str
    message_id: str | None
    provenance: PublicProvenance

    def __post_init__(self) -> None:
        _require_tick(self.tick, "private memory tick")
        _require_nonempty(self.conversation_id, "private memory conversation_id")
        if self.relation_type not in {"close_friend", "familiar_peer", "cross_group", "group"}:
            raise ForumTwinError("private memory relation type is invalid")
        if self.action not in PRIVATE_MESSAGE_ACTIONS | PRIVATE_MEMBERSHIP_ACTIONS | {
            "private_reaction"
        }:
            raise ForumTwinError("private memory action is invalid")
        summary = _require_nonempty(
            self.semantic_summary, "private memory semantic_summary"
        )
        if len(summary) > 200:
            raise ForumTwinError("private memory summary exceeds 200 characters")
        if _PATH_RE.search(summary) or _CREDENTIAL_RE.search(summary):
            raise ForumTwinError("private memory contains a sensitive marker")


@dataclass
class AgentMemory:
    agent_id: str
    recent_real_llm_episodes: list[RealLlmMemoryEpisode] = field(default_factory=list)
    rolling_long_term_summary: str = ""
    participated_thread_ids: set[str] = field(default_factory=set)
    unread_reply_ids: set[str] = field(default_factory=set)
    accepted_correction_claim_ids: set[str] = field(default_factory=set)
    rejected_correction_claim_ids: set[str] = field(default_factory=set)
    background_state_history: list[dict[str, float]] = field(default_factory=list)
    recent_private_llm_episodes: list[PrivateLlmMemoryEpisode] = field(
        default_factory=list
    )
    private_inbox_message_ids: set[str] = field(default_factory=set)
    pending_private_invitation_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        _require_nonempty(self.agent_id, "memory agent_id")
        if len(self.recent_real_llm_episodes) > MAX_RECENT_REAL_LLM_EPISODES:
            raise ForumTwinError("AgentMemory contains more than four real LLM episodes")
        if len(self.recent_private_llm_episodes) > MAX_RECENT_PRIVATE_LLM_EPISODES:
            raise ForumTwinError("AgentMemory contains more than four private LLM episodes")
        if len(self.rolling_long_term_summary) > 600:
            raise ForumTwinError("long-term memory summary exceeds 600 characters")
        if _PATH_RE.search(self.rolling_long_term_summary) or _CREDENTIAL_RE.search(
            self.rolling_long_term_summary
        ):
            raise ForumTwinError("long-term memory contains a path or credential marker")

    def append_real_episode(self, episode: RealLlmMemoryEpisode) -> None:
        if self.recent_real_llm_episodes and episode.tick < self.recent_real_llm_episodes[-1].tick:
            raise ForumTwinError("real LLM memory cannot travel backwards in time")
        self.recent_real_llm_episodes.append(episode)
        del self.recent_real_llm_episodes[:-MAX_RECENT_REAL_LLM_EPISODES]
        self.participated_thread_ids.add(episode.thread_id)

    def append_background_state(self, state: Mapping[str, Any]) -> None:
        forbidden = _DYNAMICS_FORBIDDEN_KEYS & {str(key) for key in state}
        if forbidden:
            raise ForumTwinError("background state cannot contain remembered speech")
        clean: dict[str, float] = {}
        for key, value in state.items():
            if not isinstance(value, (int, float)):
                raise ForumTwinError("background state values must be numeric")
            clean[str(key)] = float(value)
        self.background_state_history.append(clean)
        del self.background_state_history[:-8]

    def append_private_episode(self, episode: PrivateLlmMemoryEpisode) -> None:
        if (
            self.recent_private_llm_episodes
            and episode.tick < self.recent_private_llm_episodes[-1].tick
        ):
            raise ForumTwinError("private LLM memory cannot travel backwards in time")
        self.recent_private_llm_episodes.append(episode)
        del self.recent_private_llm_episodes[:-MAX_RECENT_PRIVATE_LLM_EPISODES]

class ForumLedger:
    """Deterministic in-memory forum graph with fail-closed lineage checks."""

    def __init__(self) -> None:
        self.threads: dict[str, ForumThread] = {}
        self.messages: dict[str, ForumMessage] = {}
        self.interactions: dict[str, ForumInteraction] = {}
        self.claims: dict[str, ForumClaim] = {}
        self.private_conversations: dict[str, PrivateConversation] = {}
        self.private_membership_events: dict[str, PrivateMembershipEvent] = {}
        self.private_messages: dict[str, PrivateMessage] = {}
        self.private_reactions: dict[str, PrivateReaction] = {}
        self.private_claims: dict[str, PrivateClaim] = {}

    def add_thread(self, thread: ForumThread, root: ForumMessage) -> None:
        if thread.thread_id in self.threads or root.message_id in self.messages:
            raise ForumTwinError("thread or root message already exists")
        if root.thread_id != thread.thread_id or root.message_id != thread.root_message_id:
            raise ForumTwinError("thread root identity does not align")
        if root.action not in ROOT_ACTIONS or root.parent_message_id or root.quote_message_id:
            raise ForumTwinError("thread root must be an unparented root action")
        if root.created_tick != thread.created_tick:
            raise ForumTwinError("thread and root must be created at the same tick")
        if root.provenance.kind != thread.provenance_kind:
            raise ForumTwinError("thread and root provenance do not align")
        if root.private_hearsay_origin:
            private_claim = self.private_claims.get(str(root.private_origin_claim_id))
            if private_claim is None or private_claim.first_seen_tick >= root.created_tick:
                raise ForumTwinError("private-origin thread lost temporal private lineage")
        self._assert_one_public_action_per_tick(root)
        self.threads[thread.thread_id] = thread
        self.messages[root.message_id] = root

    def add_message(self, message: ForumMessage) -> None:
        if message.message_id in self.messages:
            raise ForumTwinError("message already exists")
        thread = self.threads.get(message.thread_id)
        if thread is None:
            raise ForumTwinError("message thread does not exist")
        if thread.status != "open":
            raise ForumTwinError("message cannot enter a closed thread")
        if message.action == "post":
            raise ForumTwinError("a thread cannot receive a second root message")
        self._assert_one_public_action_per_tick(message)
        for linked_id, label in (
            (message.parent_message_id, "parent"),
            (message.quote_message_id, "quote"),
        ):
            if linked_id is None:
                continue
            linked = self.messages.get(linked_id)
            if linked is None or linked.thread_id != message.thread_id:
                raise ForumTwinError(f"message {label} must exist in the same thread")
            if linked.created_tick >= message.created_tick:
                raise ForumTwinError(f"message {label} lineage must advance in time")
        if message.claim_id is not None:
            claim = self.claims.get(message.claim_id)
            if claim is None and message.claim_operation == "create":
                claim = None
            elif claim is None or claim.thread_id != message.thread_id:
                raise ForumTwinError("message claim must exist in the same thread")
            if claim is not None and claim.first_seen_tick > message.created_tick:
                raise ForumTwinError("message cannot see a future claim")
        if message.correction_target_claim_id is not None:
            target = self.claims.get(message.correction_target_claim_id)
            if target is None or target.thread_id != message.thread_id:
                raise ForumTwinError("correction target must exist in the same thread")
            if target.first_seen_tick >= message.created_tick:
                raise ForumTwinError("correction must follow its target claim")
        if message.private_hearsay_origin:
            private_claim = self.private_claims.get(str(message.private_origin_claim_id))
            if private_claim is None:
                raise ForumTwinError("private-origin public message lost its private lineage")
            if private_claim.first_seen_tick >= message.created_tick:
                raise ForumTwinError("private-to-public spillover must advance in time")
            if message.claim_id == private_claim.claim_id:
                raise ForumTwinError("private spillover must create a new public claim ID")
        self.messages[message.message_id] = message
        participant_ids = {
            item.source_agent_display_id
            for item in self.messages.values()
            if item.thread_id == message.thread_id and not item.superseded
        }
        replies = sum(
            item.thread_id == message.thread_id and item.message_id != thread.root_message_id
            for item in self.messages.values()
            if not item.superseded
        )
        self.threads[message.thread_id] = replace(
            thread,
            participant_count=len(participant_ids),
            reply_count=replies,
            last_active_tick=message.created_tick,
        )

    def _assert_one_public_action_per_tick(self, message: ForumMessage) -> None:
        """Keep one speech emission while allowing separate interactions."""

        if any(
            existing.source_agent_display_id == message.source_agent_display_id
            and existing.created_tick == message.created_tick
            and not existing.superseded
            for existing in self.messages.values()
        ):
            raise ForumTwinError("one agent may emit at most one public message per tick")

    def add_interaction(self, interaction: ForumInteraction) -> None:
        """Commit a like/repost/report without consuming the speech slot."""

        if interaction.interaction_id in self.interactions:
            raise ForumTwinError("forum interaction already exists")
        target = self.messages.get(interaction.target_message_id)
        thread = self.threads.get(interaction.thread_id)
        if target is None or target.superseded:
            raise ForumTwinError("forum interaction target is unavailable")
        if thread is None or target.thread_id != interaction.thread_id:
            raise ForumTwinError("forum interaction crossed thread boundaries")
        if (
            thread.scenario_id != interaction.scenario_id
            or thread.branch not in {interaction.branch, "shared_baseline"}
        ):
            raise ForumTwinError("forum interaction crossed scenario or branch")
        if target.created_tick > interaction.created_tick:
            raise ForumTwinError("forum interaction targeted a future message")
        if interaction.provenance.kind not in PUBLIC_PROVENANCE_KINDS:
            raise ForumTwinError("non-LLM provenance cannot create public interaction")
        if any(
            existing.source_agent_display_id == interaction.source_agent_display_id
            and existing.created_tick == interaction.created_tick
            and existing.kind == interaction.kind
            and existing.target_message_id == interaction.target_message_id
            for existing in self.interactions.values()
        ):
            raise ForumTwinError("duplicate interaction kind/target in one resident turn")
        self.interactions[interaction.interaction_id] = interaction

    def interaction_counts(
        self,
        message_id: str,
        *,
        through_tick: int | None = None,
    ) -> dict[str, int]:
        if message_id not in self.messages:
            raise ForumTwinError("cannot count interactions for an unknown message")
        if through_tick is not None:
            _require_tick(through_tick, "interaction count tick")
        counts = {kind: 0 for kind in sorted(INTERACTION_KINDS)}
        for interaction in self.interactions.values():
            if interaction.target_message_id != message_id:
                continue
            if through_tick is not None and interaction.created_tick > through_tick:
                continue
            counts[interaction.kind] += 1
        return counts

    def interaction_counts_by_tick(self, message_id: str) -> list[dict[str, int]]:
        """Return anonymous per-tick deltas for replay-safe public animation.

        Record-level interaction identities remain private.  Publishing only
        the three counters by tick lets the UI reconstruct the board exactly
        as it looked at T3, T4, ... without leaking future engagement into an
        earlier frame.
        """

        if message_id not in self.messages:
            raise ForumTwinError("cannot count interactions for an unknown message")
        by_tick: dict[int, dict[str, int]] = {}
        for interaction in self.interactions.values():
            if interaction.target_message_id != message_id:
                continue
            counts = by_tick.setdefault(
                interaction.created_tick,
                {kind: 0 for kind in sorted(INTERACTION_KINDS)},
            )
            counts[interaction.kind] += 1
        return [
            {"tick": tick, **by_tick[tick]}
            for tick in sorted(by_tick)
        ]

    def add_private_conversation(
        self,
        conversation: PrivateConversation,
        *,
        allowed_contact_ids: Iterable[str] | None = None,
    ) -> None:
        """Create one direct chat or pending group from a closed synthetic contact set."""

        if conversation.conversation_id in self.private_conversations:
            raise ForumTwinError("private conversation already exists")
        invited = set(conversation.member_agent_ids) | set(
            conversation.pending_invite_agent_ids
        )
        invited.discard(conversation.creator_agent_id)
        if allowed_contact_ids is not None and not invited.issubset(
            {str(value) for value in allowed_contact_ids}
        ):
            raise ForumTwinError("private conversation invited an agent outside the closed contact set")
        if conversation.conversation_type == "group":
            created_count = sum(
                value.conversation_type == "group"
                and value.creator_agent_id == conversation.creator_agent_id
                and value.scenario_id == conversation.scenario_id
                for value in self.private_conversations.values()
            )
            if created_count >= 1:
                raise ForumTwinError("one agent may create at most one private group per scenario")
        for member in conversation.member_agent_ids:
            active_group_count = sum(
                value.conversation_type == "group"
                and value.status == "active"
                and member in value.member_agent_ids
                for value in self.private_conversations.values()
            )
            if conversation.conversation_type == "group" and conversation.status == "active":
                active_group_count += 1
            if active_group_count > 3:
                raise ForumTwinError("one agent cannot join more than three active groups")
        self.private_conversations[conversation.conversation_id] = conversation

    def apply_private_membership_event(
        self,
        event: PrivateMembershipEvent,
        *,
        allowed_contact_ids: Iterable[str] | None = None,
    ) -> None:
        if event.event_id in self.private_membership_events:
            raise ForumTwinError("private membership event already exists")
        conversation = self.private_conversations.get(event.conversation_id)
        if conversation is None or conversation.conversation_type != "group":
            raise ForumTwinError("private membership event requires an existing group")
        if event.created_tick <= conversation.created_tick:
            raise ForumTwinError("private membership event must follow group creation")
        if conversation.status in {"expired", "dissolved"}:
            raise ForumTwinError("closed private group cannot accept membership events")
        members = list(conversation.member_agent_ids)
        pending = list(conversation.pending_invite_agent_ids)
        left = list(conversation.left_agent_ids)
        if event.action == "invite":
            if event.actor_agent_id not in members:
                raise ForumTwinError("only a current group member may invite")
            if event.subject_agent_id in set(members) | set(pending) | set(left):
                raise ForumTwinError("private group invite target is already resolved")
            if allowed_contact_ids is not None and event.subject_agent_id not in {
                str(value) for value in allowed_contact_ids
            }:
                raise ForumTwinError("private group invite target is outside the closed contact set")
            if len(members) + len(pending) >= 12:
                raise ForumTwinError("private group invite exceeds twelve members")
            pending.append(event.subject_agent_id)
        elif event.action in {"accept", "decline"}:
            if event.actor_agent_id != event.subject_agent_id:
                raise ForumTwinError("invitee must accept or decline their own invitation")
            if event.subject_agent_id not in pending:
                raise ForumTwinError("private invitation is not pending")
            if event.created_tick > conversation.invitation_expires_tick:
                raise ForumTwinError("private invitation has expired")
            pending.remove(event.subject_agent_id)
            if event.action == "accept":
                active_group_count = sum(
                    value.conversation_type == "group"
                    and value.status == "active"
                    and event.subject_agent_id in value.member_agent_ids
                    for value in self.private_conversations.values()
                )
                if active_group_count >= 3:
                    raise ForumTwinError("invitee already belongs to three active groups")
                members.append(event.subject_agent_id)
            else:
                left.append(event.subject_agent_id)
        elif event.action == "leave":
            if event.actor_agent_id != event.subject_agent_id or event.subject_agent_id not in members:
                raise ForumTwinError("only a current member may leave their private group")
            if event.subject_agent_id == conversation.creator_agent_id:
                raise ForumTwinError("group creator must dissolve rather than leave")
            members.remove(event.subject_agent_id)
            left.append(event.subject_agent_id)
        else:
            if (
                event.action != "dissolve"
                or event.actor_agent_id != conversation.creator_agent_id
                or event.subject_agent_id != conversation.creator_agent_id
            ):
                raise ForumTwinError("only the group creator may dissolve a group")
        if event.action == "dissolve":
            status = "dissolved"
            left = list(dict.fromkeys(left + members + pending))
            members = []
            pending = []
        elif len(members) >= 3:
            status = "active"
        elif event.created_tick > conversation.invitation_expires_tick or not pending:
            status = "expired"
        else:
            status = "pending"
        updated = replace(
            conversation,
            member_agent_ids=tuple(members),
            pending_invite_agent_ids=tuple(pending),
            left_agent_ids=tuple(dict.fromkeys(left)),
            status=status,
        )
        self.private_membership_events[event.event_id] = event
        self.private_conversations[event.conversation_id] = updated

    def expire_private_invitations(self, tick: int) -> int:
        _require_tick(tick, "private invitation sweep tick")
        changed = 0
        for conversation_id, conversation in list(self.private_conversations.items()):
            if (
                conversation.conversation_type == "group"
                and conversation.status == "pending"
                and tick > conversation.invitation_expires_tick
            ):
                self.private_conversations[conversation_id] = replace(
                    conversation,
                    status="expired",
                    left_agent_ids=tuple(
                        dict.fromkeys(
                            conversation.left_agent_ids
                            + conversation.pending_invite_agent_ids
                        )
                    ),
                    pending_invite_agent_ids=(),
                )
                changed += 1
        return changed

    def add_private_message(self, message: PrivateMessage) -> None:
        if message.message_id in self.private_messages:
            raise ForumTwinError("private message already exists")
        conversation = self.private_conversations.get(message.conversation_id)
        if conversation is None or conversation.status != "active":
            raise ForumTwinError("private message requires an active conversation")
        if message.source_agent_id not in conversation.member_agent_ids:
            raise ForumTwinError("private message source is not a current member")
        if message.created_tick < conversation.created_tick:
            raise ForumTwinError("private message predates its conversation")
        if sum(
            existing.source_agent_id == message.source_agent_id
            and existing.created_tick == message.created_tick
            and not existing.superseded
            for existing in self.private_messages.values()
        ) >= 2:
            raise ForumTwinError("one agent may create at most two private messages per tick")
        for linked_id, label in (
            (message.parent_message_id, "parent"),
            (message.quote_message_id, "quote"),
        ):
            if linked_id is None:
                continue
            linked = self.private_messages.get(linked_id)
            if linked is None or linked.conversation_id != message.conversation_id:
                raise ForumTwinError(f"private message {label} must exist in the same conversation")
            if linked.created_tick >= message.created_tick:
                raise ForumTwinError(f"private message {label} lineage must advance in time")
        if message.public_source_message_id is not None:
            public = self.messages.get(message.public_source_message_id)
            if public is None or public.superseded:
                raise ForumTwinError("private forward public source is unavailable")
            if public.created_tick >= message.created_tick:
                raise ForumTwinError("public-to-private forward must advance in time")
        if message.forwarded_private_source_message_id is not None:
            private_source = self.private_messages.get(
                message.forwarded_private_source_message_id
            )
            source_conversation = (
                self.private_conversations.get(private_source.conversation_id)
                if private_source is not None
                else None
            )
            if (
                private_source is None
                or private_source.superseded
                or private_source.created_tick >= message.created_tick
                or source_conversation is None
                or message.source_agent_id not in source_conversation.member_agent_ids
            ):
                raise ForumTwinError("private-to-private forward source is unavailable")
        if message.private_claim_id is not None:
            private_claim = self.private_claims.get(message.private_claim_id)
            if private_claim is None and message.claim_operation == "create":
                private_claim = None
            elif private_claim is None or private_claim.conversation_id != message.conversation_id:
                raise ForumTwinError("private message claim is unavailable in this conversation")
            if private_claim is not None and private_claim.first_seen_tick > message.created_tick:
                raise ForumTwinError("private message sees a future private claim")
        if message.correction_target_claim_id is not None:
            target = self.private_claims.get(message.correction_target_claim_id)
            if target is None or target.conversation_id != message.conversation_id:
                raise ForumTwinError("private correction target is unavailable")
            if target.first_seen_tick >= message.created_tick:
                raise ForumTwinError("private correction must follow its target")
        self.private_messages[message.message_id] = message

    def add_private_reaction(self, reaction: PrivateReaction) -> None:
        if reaction.reaction_id in self.private_reactions:
            raise ForumTwinError("private reaction already exists")
        conversation = self.private_conversations.get(reaction.conversation_id)
        target = self.private_messages.get(reaction.target_message_id)
        if (
            conversation is None
            or conversation.status != "active"
            or target is None
            or target.superseded
            or target.conversation_id != reaction.conversation_id
            or reaction.source_agent_id not in conversation.member_agent_ids
            or target.created_tick >= reaction.created_tick
        ):
            raise ForumTwinError("private reaction target or membership is invalid")
        if any(
            existing.source_agent_id == reaction.source_agent_id
            and existing.target_message_id == reaction.target_message_id
            and existing.kind == reaction.kind
            and existing.created_tick == reaction.created_tick
            for existing in self.private_reactions.values()
        ):
            raise ForumTwinError("duplicate private reaction")
        self.private_reactions[reaction.reaction_id] = reaction

    def add_private_claim(self, claim: PrivateClaim) -> None:
        if claim.claim_id in self.private_claims:
            raise ForumTwinError("private claim already exists")
        conversation = self.private_conversations.get(claim.conversation_id)
        creator = self.private_messages.get(claim.created_by_message_id)
        if (
            conversation is None
            or creator is None
            or creator.conversation_id != claim.conversation_id
            or creator.created_tick != claim.first_seen_tick
        ):
            raise ForumTwinError("private claim creator or conversation is unresolved")
        if claim.origin_kind == "public":
            origin = self.messages.get(str(claim.origin_public_message_id))
            if origin is None or origin.created_tick >= claim.first_seen_tick:
                raise ForumTwinError("private claim public origin is unresolved or non-temporal")
        else:
            origin = self.private_messages.get(str(claim.origin_private_message_id))
            if origin is None or origin.created_tick > claim.first_seen_tick:
                raise ForumTwinError("private claim origin is unresolved or from the future")
        for linked_id, label in (
            (claim.parent_claim_id, "parent"),
            (claim.correction_target_claim_id, "correction target"),
        ):
            if linked_id is None:
                continue
            linked = self.private_claims.get(linked_id)
            if linked is None or linked.conversation_id != claim.conversation_id:
                raise ForumTwinError(f"private claim {label} is unresolved")
            if linked.first_seen_tick >= claim.first_seen_tick:
                raise ForumTwinError(f"private claim {label} must precede its child")
        for message_id in claim.supporting_message_ids + claim.challenging_message_ids:
            linked_message = self.private_messages.get(message_id)
            if (
                linked_message is None
                or linked_message.conversation_id != claim.conversation_id
                or linked_message.created_tick > claim.last_seen_tick
            ):
                raise ForumTwinError("private claim evidence is unresolved")
        self.private_claims[claim.claim_id] = claim

    def advance_private_claim(self, claim: PrivateClaim) -> None:
        previous = self.private_claims.get(claim.claim_id)
        if previous is None:
            raise ForumTwinError("cannot advance an unknown private claim")
        stable = (
            "conversation_id",
            "claim_summary",
            "first_seen_tick",
            "created_by_message_id",
            "origin_kind",
            "origin_public_message_id",
            "origin_private_message_id",
            "parent_claim_id",
            "correction_target_claim_id",
        )
        if any(getattr(previous, key) != getattr(claim, key) for key in stable):
            raise ForumTwinError("private claim immutable identity changed")
        if claim.last_seen_tick < previous.last_seen_tick:
            raise ForumTwinError("private claim advancement moved backwards")
        if previous.status == "expired" and claim.status != "expired":
            raise ForumTwinError("expired private claim cannot reactivate")
        if not set(previous.supporting_message_ids).issubset(claim.supporting_message_ids):
            raise ForumTwinError("private claim advancement removed support")
        if not set(previous.challenging_message_ids).issubset(claim.challenging_message_ids):
            raise ForumTwinError("private claim advancement removed challenges")
        self.private_claims[claim.claim_id] = claim

    def build_private_context(
        self,
        *,
        recipient_agent_id: str,
        current_tick: int,
        conversation_ids: Sequence[str],
    ) -> tuple[dict[str, Any], ...]:
        """Return at most four member-visible messages under the 600-char budget."""

        _require_tick(current_tick, "private context tick")
        rows: list[PrivateMessage] = []
        allowed_conversations: set[str] = set()
        for conversation_id in conversation_ids:
            conversation = self.private_conversations.get(str(conversation_id))
            if (
                conversation is None
                or recipient_agent_id not in conversation.member_agent_ids
                or conversation.status != "active"
            ):
                raise ForumTwinError("private context crosses conversation membership")
            allowed_conversations.add(conversation.conversation_id)
        for message in self.private_messages.values():
            if (
                message.conversation_id in allowed_conversations
                and message.created_tick < current_tick
                and not message.superseded
            ):
                rows.append(message)
        rows.sort(key=lambda item: (-item.created_tick, item.message_id))
        context: list[dict[str, Any]] = []
        used_chars = 0
        for message in rows:
            if used_chars + len(message.visible_text) > MAX_PRIVATE_CONTEXT_CHARS:
                continue
            context.append(message.private_view())
            used_chars += len(message.visible_text)
            if len(context) == 4:
                break
        return tuple(context)

    def add_claim(self, claim: ForumClaim) -> None:
        if claim.claim_id in self.claims:
            raise ForumTwinError("claim already exists")
        thread = self.threads.get(claim.thread_id)
        creator = self.messages.get(claim.created_by_message_id)
        if thread is None or creator is None or creator.thread_id != claim.thread_id:
            raise ForumTwinError("claim creator and thread must exist")
        if creator.created_tick != claim.first_seen_tick:
            raise ForumTwinError("claim first_seen_tick must match its creating message")
        if creator.claim_id not in {None, claim.claim_id}:
            raise ForumTwinError("claim creator points at a different claim")
        if claim.parent_claim_id is not None:
            parent = self.claims.get(claim.parent_claim_id)
            if parent is None or parent.thread_id != claim.thread_id:
                raise ForumTwinError("parent claim must exist in the same thread")
            if parent.first_seen_tick >= claim.first_seen_tick:
                raise ForumTwinError("claim lineage must advance in time")
        if claim.correction_target_claim_id is not None:
            target = self.claims.get(claim.correction_target_claim_id)
            if target is None or target.thread_id != claim.thread_id:
                raise ForumTwinError("claim correction target must exist in the same thread")
            if target.first_seen_tick >= claim.first_seen_tick:
                raise ForumTwinError("claim correction must follow the target")
        referenced = claim.supporting_message_ids + claim.challenging_message_ids
        for message_id in referenced:
            message = self.messages.get(message_id)
            if message is None or message.thread_id != claim.thread_id:
                raise ForumTwinError("claim references an unknown or cross-thread message")
            if message.created_tick > claim.last_seen_tick:
                raise ForumTwinError("claim last_seen_tick predates a referenced message")
        self.claims[claim.claim_id] = claim
        self.threads[claim.thread_id] = replace(
            thread,
            claim_ids=thread.claim_ids + (claim.claim_id,),
        )

    def supersede_message(self, message_id: str) -> None:
        message = self.messages.get(message_id)
        if message is None:
            raise ForumTwinError("cannot supersede an unknown message")
        if message.message_id == self.threads[message.thread_id].root_message_id:
            raise ForumTwinError("a committed thread root cannot be superseded")
        self.messages[message_id] = replace(message, moderation_status="superseded")

    def advance_governance_artifacts(self, tick: int) -> int:
        """Advance persistent action objects without creating public language."""

        _require_tick(tick, "artifact advance tick")
        changed = 0
        for message_id, message in list(self.messages.items()):
            artifact = message.governance_artifact
            if artifact is None:
                continue
            status = (
                "expired"
                if tick > artifact.expires_tick
                else "active"
                if tick >= artifact.due_tick
                else "scheduled"
            )
            if status != artifact.status:
                self.messages[message_id] = replace(
                    message,
                    governance_artifact=replace(artifact, status=status),
                )
                changed += 1
        return changed

    def advance_claim(self, claim: ForumClaim) -> None:
        """Apply an immutable later snapshot of one evolving claim."""

        previous = self.claims.get(claim.claim_id)
        if previous is None:
            raise ForumTwinError("cannot advance an unknown claim")
        stable_fields = (
            "thread_id",
            "claim_summary",
            "first_seen_tick",
            "created_by_message_id",
            "parent_claim_id",
            "correction_target_claim_id",
        )
        if any(getattr(previous, name) != getattr(claim, name) for name in stable_fields):
            raise ForumTwinError("claim immutable identity changed during advancement")
        if claim.last_seen_tick < previous.last_seen_tick:
            raise ForumTwinError("claim advancement moved backwards in time")
        if previous.expired and claim.status != "expired":
            raise ForumTwinError("expired claim cannot be reactivated")
        if not set(previous.supporting_message_ids).issubset(claim.supporting_message_ids):
            raise ForumTwinError("claim advancement removed supporting evidence")
        if not set(previous.challenging_message_ids).issubset(claim.challenging_message_ids):
            raise ForumTwinError("claim advancement removed challenges")
        for message_id in claim.supporting_message_ids + claim.challenging_message_ids:
            message = self.messages.get(message_id)
            if message is None or message.thread_id != claim.thread_id:
                raise ForumTwinError("claim advancement references an unknown message")
            if message.created_tick > claim.last_seen_tick:
                raise ForumTwinError("claim advancement predates referenced evidence")
        self.claims[claim.claim_id] = claim

    def assert_consistent(self) -> None:
        for thread in self.threads.values():
            root = self.messages.get(thread.root_message_id)
            if root is None or root.thread_id != thread.thread_id:
                raise ForumTwinError("forum ledger has a missing thread root")
        for message in self.messages.values():
            if message.message_id == self.threads[message.thread_id].root_message_id:
                continue
            for linked_id in (message.parent_message_id, message.quote_message_id):
                if linked_id is not None:
                    linked = self.messages.get(linked_id)
                    if linked is None or linked.created_tick >= message.created_tick:
                        raise ForumTwinError("forum ledger message lineage is inconsistent")
        for interaction in self.interactions.values():
            target = self.messages.get(interaction.target_message_id)
            thread = self.threads.get(interaction.thread_id)
            if (
                target is None
                or target.superseded
                or thread is None
                or target.thread_id != interaction.thread_id
                or target.created_tick > interaction.created_tick
                or thread.scenario_id != interaction.scenario_id
                or thread.branch not in {interaction.branch, "shared_baseline"}
            ):
                raise ForumTwinError("forum ledger interaction is inconsistent")
        for claim in self.claims.values():
            if claim.parent_claim_id is not None:
                parent = self.claims.get(claim.parent_claim_id)
                if parent is None or parent.first_seen_tick >= claim.first_seen_tick:
                    raise ForumTwinError("forum ledger claim lineage is inconsistent")
        for conversation in self.private_conversations.values():
            if conversation.status == "active" and not conversation.member_agent_ids:
                raise ForumTwinError("forum ledger has an empty active private conversation")
        for event in self.private_membership_events.values():
            if event.conversation_id not in self.private_conversations:
                raise ForumTwinError("forum ledger private membership lineage is inconsistent")
        for message in self.private_messages.values():
            conversation = self.private_conversations.get(message.conversation_id)
            if conversation is None:
                raise ForumTwinError("forum ledger private message lost its conversation")
            for linked_id in (message.parent_message_id, message.quote_message_id):
                if linked_id is not None:
                    linked = self.private_messages.get(linked_id)
                    if (
                        linked is None
                        or linked.conversation_id != message.conversation_id
                        or linked.created_tick >= message.created_tick
                    ):
                        raise ForumTwinError("forum ledger private message lineage is inconsistent")
            if message.public_source_message_id is not None:
                public = self.messages.get(message.public_source_message_id)
                if public is None or public.created_tick >= message.created_tick:
                    raise ForumTwinError("forum ledger public-to-private lineage is inconsistent")
            if message.forwarded_private_source_message_id is not None:
                source = self.private_messages.get(
                    message.forwarded_private_source_message_id
                )
                if source is None or source.created_tick >= message.created_tick:
                    raise ForumTwinError("forum ledger private-to-private lineage is inconsistent")
        for reaction in self.private_reactions.values():
            target = self.private_messages.get(reaction.target_message_id)
            if (
                target is None
                or target.conversation_id != reaction.conversation_id
                or target.created_tick >= reaction.created_tick
            ):
                raise ForumTwinError("forum ledger private reaction is inconsistent")
        for claim in self.private_claims.values():
            creator = self.private_messages.get(claim.created_by_message_id)
            if creator is None or creator.conversation_id != claim.conversation_id:
                raise ForumTwinError("forum ledger private claim creator is inconsistent")
        for message in self.messages.values():
            if message.private_hearsay_origin:
                origin = self.private_claims.get(str(message.private_origin_claim_id))
                if origin is None or origin.first_seen_tick >= message.created_tick:
                    raise ForumTwinError("forum ledger private-to-public lineage is inconsistent")


def build_feed_context(
    *,
    ledger: ForumLedger,
    exposures: Sequence[FeedExposure],
    recipient_agent_id: str,
    current_tick: int,
    current_thread_id: str | None = None,
    direct_parent_message_id: str | None = None,
    quote_message_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Pack an auditable, time-visible feed under both frozen limits."""

    _require_tick(current_tick, "feed current_tick")
    _require_nonempty(recipient_agent_id, "feed recipient_agent_id")
    ranked = sorted(
        exposures,
        key=lambda item: (-float(item.rank_score), item.delivered_tick, item.exposure_id),
    )
    context: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    used_chars = 0
    for exposure in ranked:
        if exposure.recipient_agent_id != recipient_agent_id:
            raise ForumTwinError("feed exposure recipient mismatch")
        if exposure.delivered_tick > current_tick:
            raise ForumTwinError("feed contains a future exposure")
        message = ledger.messages.get(exposure.message_id)
        if message is None:
            raise ForumTwinError("feed exposure refers to an unknown message")
        if message.created_tick > exposure.delivered_tick:
            raise ForumTwinError("feed delivered a message before it existed")
        if message.message_id in seen_messages:
            continue
        thread = ledger.threads[message.thread_id]
        is_current_root = (
            message.message_id == thread.root_message_id
            and message.thread_id == current_thread_id
        )
        required_context = (
            is_current_root
            or message.message_id == direct_parent_message_id
            or message.message_id == quote_message_id
        )
        if message.superseded or (
            not required_context and not message.active_at(current_tick)
        ):
            continue
        if message.claim_id is not None:
            claim = ledger.claims.get(message.claim_id)
            if claim is None:
                raise ForumTwinError("feed message refers to an unknown claim")
            if claim.expired and not required_context:
                continue
        text = message.visible_text if required_context else _message_summary(message)
        if used_chars + len(text) > MAX_FORUM_CONTEXT_CHARS:
            continue
        context.append(
            {
                "message_id": message.message_id,
                "thread_id": message.thread_id,
                "relation": (
                    "root"
                    if is_current_root
                    else "direct_parent"
                    if message.message_id == direct_parent_message_id
                    else "quote"
                    if message.message_id == quote_message_id
                    else "feed_summary"
                ),
                "visible_text": text,
                "topic": message.topic,
                "stance": message.stance,
                "evidence_status": message.evidence_status,
                "claim_id": message.claim_id,
                "created_tick": message.created_tick,
                "provenance_kind": message.provenance.kind,
                "exposure_source": exposure.exposure_source,
                "governance_artifact": (
                    message.governance_artifact.public_view()
                    if message.governance_artifact is not None
                    else None
                ),
            }
        )
        used_chars += len(text)
        seen_messages.add(message.message_id)
        if len(context) == MAX_FEED_ITEMS:
            break
    if len(context) > MAX_FEED_ITEMS or used_chars > MAX_FORUM_CONTEXT_CHARS:
        raise ForumTwinError("feed packer violated its frozen limits")
    if current_thread_id is not None and any(
        item["relation"] in {"root", "direct_parent", "quote"}
        and item["thread_id"] != current_thread_id
        for item in context
    ):
        raise ForumTwinError("direct forum context crossed thread boundaries")
    return tuple(context)


def _message_summary(message: ForumMessage) -> str:
    text = message.visible_text
    if len(text) > 72:
        text = text[:71] + "…"
    return f"[{message.action}/{message.evidence_status}] {text}"


@dataclass(frozen=True)
class InteractionCandidate:
    agent_index: int
    reason: str
    attention_score: float
    enqueued_tick: int
    sequence: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.agent_index, int) or isinstance(self.agent_index, bool) or self.agent_index < 0:
            raise ForumTwinError("interaction candidate agent_index is invalid")
        if self.reason not in INTERACTION_REASON_PRIORITY:
            raise ForumTwinError("interaction candidate reason is invalid")
        _require_probability(self.attention_score, "interaction attention_score")
        _require_tick(self.enqueued_tick, "interaction enqueued_tick")
        if self.sequence < 0:
            raise ForumTwinError("interaction sequence cannot be negative")


@dataclass(frozen=True)
class PpsCandidate:
    agent_index: int
    inclusion_probability: float
    rotation_count: int = 0
    selection_tiebreak: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.agent_index, int) or isinstance(self.agent_index, bool) or self.agent_index < 0:
            raise ForumTwinError("PPS candidate agent_index is invalid")
        if not 0.0 < float(self.inclusion_probability) <= 1.0:
            raise ForumTwinError("PPS inclusion probability must be positive")
        if self.rotation_count < 0:
            raise ForumTwinError("PPS rotation_count cannot be negative")
        if not isinstance(self.selection_tiebreak, str):
            raise ForumTwinError("PPS selection_tiebreak must be a string")


@dataclass(frozen=True)
class ForumActivationRound:
    tick: int
    resident_budget: int
    anchor_indices: tuple[int, ...]
    interaction_indices: tuple[int, ...]
    pps_indices: tuple[int, ...]
    pps_inclusion_probabilities: tuple[float, ...]
    overflow: tuple[InteractionCandidate, ...]

    @property
    def resident_indices(self) -> tuple[int, ...]:
        return self.anchor_indices + self.interaction_indices + self.pps_indices

    @property
    def resident_count(self) -> int:
        return len(self.resident_indices)

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "tick": self.tick,
                "resident_budget": self.resident_budget,
                "anchors": self.anchor_indices,
                "interactions": self.interaction_indices,
                "pps": self.pps_indices,
                "pps_pi": self.pps_inclusion_probabilities,
                "overflow": [candidate.__dict__ for candidate in self.overflow],
            }
        )


def resident_turn_budget(tick: int) -> int:
    _require_tick(tick)
    if tick <= 1:
        return 24
    if tick == 2:
        return 48
    if tick <= 7:
        return 64
    if tick <= 17:
        return 48
    return 32


def governance_turn_budget(*, branch: str, tick: int) -> int:
    _require_tick(tick)
    return 0 if branch.lower() == "natural" or tick not in DECISION_TICKS else 3


def total_turn_budget(*, branch: str, tick: int) -> int:
    value = resident_turn_budget(tick) + governance_turn_budget(branch=branch, tick=tick)
    if value > MAX_TOTAL_TURNS:
        raise ForumTwinError("resident plus governance budget exceeds 67")
    return value


def schedule_forum_activations(
    *,
    agent_count: int,
    tick: int,
    anchor_indices: Sequence[int],
    interaction_candidates: Sequence[InteractionCandidate],
    pps_candidates: Sequence[PpsCandidate],
    resident_budget_override: int | None = None,
    required_anchor_count: int = ANCHOR_COUNT,
) -> ForumActivationRound:
    """Schedule anchors, bounded interactions and PPS fill without speech synthesis."""

    maximum_budget = max(
        resident_turn_budget(tick),
        48 if tick in {2, 7, 13, 19} else 0,
    )
    budget = (
        maximum_budget
        if resident_budget_override is None
        else resident_budget_override
    )
    if (
        isinstance(required_anchor_count, bool)
        or not isinstance(required_anchor_count, int)
        or not 1 <= required_anchor_count <= ANCHOR_COUNT
    ):
        raise ForumTwinError("required anchor count must be from one through sixteen")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or not required_anchor_count <= budget <= maximum_budget
    ):
        raise ForumTwinError("activation budget override is outside the tick bound")
    anchors = tuple(int(index) for index in anchor_indices)
    if (
        len(anchors) != required_anchor_count
        or len(set(anchors)) != required_anchor_count
    ):
        raise ForumTwinError("activation round has the wrong number of distinct anchors")
    if agent_count < budget or any(not 0 <= index < agent_count for index in anchors):
        raise ForumTwinError("activation population or anchor identity is invalid")
    anchor_set = set(anchors)
    interaction_cap = min(
        INTERACTION_CAP_BY_TICK[tick],
        (budget - required_anchor_count) // 2,
    )
    rotation_by_index = {
        candidate.agent_index: int(candidate.rotation_count)
        for candidate in pps_candidates
    }
    ranked_interactions = sorted(
        interaction_candidates,
        key=lambda item: (
            INTERACTION_REASON_PRIORITY[item.reason],
            rotation_by_index.get(item.agent_index, 0),
            -float(item.attention_score),
            item.enqueued_tick,
            item.sequence,
            item.agent_index,
        ),
    )
    selected_interactions: list[InteractionCandidate] = []
    overflow: list[InteractionCandidate] = []
    seen = set(anchor_set)
    for candidate in ranked_interactions:
        if not 0 <= candidate.agent_index < agent_count:
            raise ForumTwinError("interaction candidate falls outside the population")
        if candidate.agent_index in seen:
            continue
        if len(selected_interactions) < interaction_cap:
            selected_interactions.append(candidate)
            seen.add(candidate.agent_index)
        else:
            overflow.append(candidate)
    pps_needed = budget - required_anchor_count - len(selected_interactions)
    eligible_pps: list[PpsCandidate] = []
    for candidate in pps_candidates:
        if not 0 <= candidate.agent_index < agent_count:
            raise ForumTwinError("PPS candidate falls outside the population")
        if candidate.agent_index not in seen:
            eligible_pps.append(candidate)
    if len({item.agent_index for item in eligible_pps}) != len(eligible_pps):
        raise ForumTwinError("PPS candidates repeat a population identity")

    # Calibrate the supplied positive size measures to exact first-order
    # inclusion probabilities summing to the number of slots left after
    # anchors and interaction-priority activations.  Certainty units are peeled
    # off iteratively so every remaining eligible unit retains positive pi.
    calibrated: list[PpsCandidate] = []
    if pps_needed:
        if len(eligible_pps) < pps_needed:
            raise ForumTwinError("positive-probability PPS candidates cannot fill the resident budget")
        probabilities = [0.0] * len(eligible_pps)
        active = set(range(len(eligible_pps)))
        remaining = float(pps_needed)
        while active:
            raw_total = sum(
                float(eligible_pps[index].inclusion_probability)
                for index in active
            )
            if raw_total <= 0:
                raise ForumTwinError("PPS size measures have zero mass")
            scale = remaining / raw_total
            certainty = [
                index
                for index in active
                if float(eligible_pps[index].inclusion_probability) * scale >= 1.0 - 1e-12
            ]
            if not certainty:
                for index in active:
                    probabilities[index] = (
                        float(eligible_pps[index].inclusion_probability) * scale
                    )
                break
            for index in certainty:
                probabilities[index] = 1.0
                active.remove(index)
                remaining -= 1.0
            if remaining < -1e-9:
                raise ForumTwinError("PPS certainty calibration exceeded its budget")
            if remaining <= 1e-12:
                if active:
                    raise ForumTwinError("PPS calibration removed positive inclusion probability")
                break
        if not math.isclose(sum(probabilities), float(pps_needed), abs_tol=1e-8):
            raise ForumTwinError("PPS inclusion probabilities do not sum to the fixed budget")
        calibrated = [
            replace(candidate, inclusion_probability=float(probabilities[index]))
            for index, candidate in enumerate(eligible_pps)
        ]

    # Ordered pivotal sampling converts the calibrated pi vector into an exact
    # fixed-size sample.  The hash is a frozen random variate, not an ordering
    # bonus; changing scenario/seed/branch salt changes the paired sample while
    # preserving every unit's positive first-order inclusion probability.
    working = [float(item.inclusion_probability) for item in calibrated]
    step = 0
    while True:
        fractional = [
            index for index, value in enumerate(working) if 1e-12 < value < 1.0 - 1e-12
        ]
        if len(fractional) < 2:
            break
        left, right = fractional[0], fractional[1]
        a, b = working[left], working[right]
        salt = canonical_hash(
            [
                "forum-pivotal-pps-v1",
                calibrated[left].selection_tiebreak,
                calibrated[right].selection_tiebreak,
                calibrated[left].agent_index,
                calibrated[right].agent_index,
                step,
            ]
        )
        uniform = (int(salt[:16], 16) + 1) / (16**16 + 1)
        pair_sum = a + b
        if pair_sum < 1.0:
            probability_left = a / pair_sum
            if uniform <= probability_left:
                working[left], working[right] = pair_sum, 0.0
            else:
                working[left], working[right] = 0.0, pair_sum
        else:
            denominator = 2.0 - pair_sum
            probability_left_one = (1.0 - b) / denominator
            if uniform <= probability_left_one:
                working[left], working[right] = 1.0, pair_sum - 1.0
            else:
                working[left], working[right] = pair_sum - 1.0, 1.0
        step += 1
    selected_pps = [
        candidate
        for candidate, final_probability in zip(calibrated, working)
        if final_probability >= 1.0 - 1e-9
    ]
    selected_pps.sort(key=lambda item: item.agent_index)
    seen.update(item.agent_index for item in selected_pps)
    if len(selected_pps) != pps_needed:
        raise ForumTwinError("positive-probability PPS candidates cannot fill the resident budget")
    result = ForumActivationRound(
        tick=tick,
        resident_budget=budget,
        anchor_indices=anchors,
        interaction_indices=tuple(item.agent_index for item in selected_interactions),
        pps_indices=tuple(item.agent_index for item in selected_pps),
        pps_inclusion_probabilities=tuple(float(item.inclusion_probability) for item in selected_pps),
        overflow=tuple(overflow),
    )
    if result.resident_count != budget or len(set(result.resident_indices)) != budget:
        raise ForumTwinError("activation round violated its exact resident budget")
    return result


def render_forum_turn_payload(
    *,
    agent_profile: Mapping[str, Any],
    posterior_state: Mapping[str, Any],
    memory: AgentMemory,
    feed_context: Sequence[Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
    allowed_actions: Sequence[str],
    current_thread_id: str | None = None,
) -> dict[str, Any]:
    """Render the JSON payload only; a transport layer may later serialize it."""

    _require_tick(tick, "forum turn tick")
    if len(feed_context) > MAX_FEED_ITEMS:
        raise ForumTwinError("forum turn contains more than six Feed items")
    if any(int(item.get("created_tick", -1)) > tick for item in feed_context):
        raise ForumTwinError("forum turn contains a future message")
    context_chars = sum(len(str(item.get("visible_text") or "")) for item in feed_context)
    if context_chars > MAX_FORUM_CONTEXT_CHARS:
        raise ForumTwinError("forum turn context exceeds 1,200 characters")
    actions = tuple(dict.fromkeys(str(action) for action in allowed_actions))
    if not actions or any(action not in PUBLIC_ACTIONS | {"ignore", "read"} for action in actions):
        raise ForumTwinError("forum turn allowed_actions is invalid")
    recent_memory = memory.recent_real_llm_episodes[-2:]
    if any(item.tick > tick for item in recent_memory):
        raise ForumTwinError("forum turn memory comes from the future")
    payload = {
        "schema": TURN_SCHEMA_VERSION,
        "simulation_target": "one synthetic forum LLM agent; not an identified student",
        "profile": dict(agent_profile),
        "posterior_state": dict(posterior_state),
        "real_llm_memory": [
            {
                "tick": item.tick,
                "thread_id": item.thread_id,
                "action": item.action,
                "semantic_summary": item.semantic_summary,
                "message_id": item.message_id,
                "provenance_kind": item.provenance.kind,
            }
            for item in recent_memory
        ],
        "background_state_summary": (
            dict(memory.background_state_history[-1])
            if memory.background_state_history
            else {}
        ),
        "forum_context": [dict(item) for item in feed_context],
        "scenario_id": _require_nonempty(scenario_id, "scenario_id"),
        "branch": _require_nonempty(branch, "branch"),
        "tick": tick,
        "phase": TIMELINE[tick],
        "current_thread_id": current_thread_id,
        "allowed_actions": list(actions),
        "safety": {
            "raw_corpus_forbidden": True,
            "real_identity_forbidden": True,
            "future_information_forbidden": True,
            "hidden_population_truth_forbidden": True,
            "new_real_world_facts_forbidden": True,
        },
        "output_contract": {
            "action": "one allowed action",
            "target_thread_id": "existing thread or empty for a new root",
            "target_message_id": "time-visible message or null",
            "quote_message_id": "time-visible message or null",
            "topic": "controlled topic",
            "subtopic": "short synthetic label",
            "stance": "support, oppose, mixed, uncertain or neutral",
            "urgency": "number from 0 to 1",
            "emotion": "controlled label",
            "confidence": "number from 0 to 1",
            "evidence_status": "controlled status",
            "claim_operation": "none, create, support, challenge, verify or correct",
            "trust_delta": "number from -0.25 to 0.25",
            "satisfaction_delta": "number from -0.25 to 0.25",
            "memory_update": "synthetic semantic summary",
            "visible_text": "root <=180 chars; other public actions <=120; empty for read/ignore",
        },
    }
    validate_prompt_payload(payload, current_tick=tick)
    return {
        "payload": payload,
        "prompt_sha256": canonical_hash(payload),
    }


def validate_prompt_payload(payload: Mapping[str, Any], *, current_tick: int) -> None:
    """Reject sensitive/internal fields and all temporal leakage recursively."""

    _require_tick(current_tick, "prompt current_tick")

    def forbidden_key(normalized: str) -> bool:
        # Positive safety declarations describe what is forbidden; they do not
        # carry the forbidden payload itself.
        if normalized.endswith("_forbidden"):
            return False
        if normalized in _FORBIDDEN_PROMPT_KEYS:
            return True
        return (
            normalized.endswith("source_id")
            or normalized.endswith("source_ids")
            or normalized.startswith("raw_")
            or normalized.startswith("original_")
            or normalized.endswith("_path")
            or normalized.startswith("hidden_truth")
            or normalized.startswith("future_")
        )

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.strip().lower()
                if forbidden_key(normalized):
                    raise ForumTwinError(f"prompt contains forbidden field {'.'.join(path + (key,))}")
                if normalized in {"created_tick", "tick"} and isinstance(child, int) and child > current_tick:
                    raise ForumTwinError("prompt contains future information")
                walk(child, path + (key,))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))
        elif isinstance(value, str):
            if _PATH_RE.search(value) or _CREDENTIAL_RE.search(value):
                raise ForumTwinError("prompt contains a path or credential marker")

    walk(payload, ())


def validate_background_dynamics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce that Dynamics performs only silent, nonlinguistic work."""

    if not isinstance(payload, Mapping):
        raise ForumTwinError("background Dynamics payload must be an object")
    unexpected = {str(key) for key in payload} - _DYNAMICS_ALLOWED_KEYS
    if unexpected:
        raise ForumTwinError(
            "background Dynamics contains unsupported/public fields: "
            + ", ".join(sorted(unexpected))
        )

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            keys = {str(key).lower() for key in value}
            if keys & _DYNAMICS_FORBIDDEN_KEYS:
                raise ForumTwinError("background Dynamics attempted to create language content")
            if value.get("action") in PUBLIC_ACTIONS:
                raise ForumTwinError("background Dynamics attempted a public action")
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif isinstance(value, str) and (
            _PATH_RE.search(value) or _CREDENTIAL_RE.search(value)
        ):
            raise ForumTwinError("background Dynamics contains a sensitive string")

    walk(payload)
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def validate_public_thread_batch(
    *,
    threads: Iterable[ForumThread],
    messages: Iterable[ForumMessage],
    claims: Iterable[ForumClaim],
) -> str:
    """Validate a complete public batch and return its canonical identity."""

    thread_values = list(threads)
    message_values = list(messages)
    claim_values = list(claims)
    if len({item.thread_id for item in thread_values}) != len(thread_values):
        raise ForumTwinError("public batch contains duplicate thread IDs")
    if len({item.message_id for item in message_values}) != len(message_values):
        raise ForumTwinError("public batch contains duplicate message IDs")
    if len({item.claim_id for item in claim_values}) != len(claim_values):
        raise ForumTwinError("public batch contains duplicate claim IDs")
    thread_by_id = {item.thread_id: item for item in thread_values}
    message_by_id = {item.message_id: item for item in message_values}
    claim_by_id = {item.claim_id: item for item in claim_values}
    root_ids = {item.root_message_id for item in thread_values}
    if len(root_ids) != len(thread_values):
        raise ForumTwinError("public batch reuses a root across threads")
    action_keys: set[tuple[str, str, str, int]] = set()
    for thread in thread_values:
        root = message_by_id.get(thread.root_message_id)
        if (
            root is None
            or root.thread_id != thread.thread_id
            or root.action not in ROOT_ACTIONS
            or root.parent_message_id is not None
            or root.quote_message_id is not None
            or root.created_tick != thread.created_tick
            or root.provenance.kind != thread.provenance_kind
        ):
            raise ForumTwinError("public thread root contract is invalid")
    for message in message_values:
        thread = thread_by_id.get(message.thread_id)
        if thread is None:
            raise ForumTwinError("public message refers to an unknown thread")
        action_key = (
            thread.scenario_id,
            thread.branch,
            message.source_agent_display_id,
            message.created_tick,
        )
        if action_key in action_keys:
            raise ForumTwinError("public batch has multiple agent actions in one tick")
        action_keys.add(action_key)
        is_root = message.message_id == thread.root_message_id
        if not is_root and message.action == "post":
            raise ForumTwinError("public batch has a second post root")
        for linked_id, label in (
            (message.parent_message_id, "parent"),
            (message.quote_message_id, "quote"),
        ):
            if linked_id is None:
                continue
            linked = message_by_id.get(linked_id)
            if linked is None or linked.thread_id != message.thread_id:
                raise ForumTwinError(f"public message {label} is unresolved")
            if linked.created_tick >= message.created_tick:
                raise ForumTwinError(f"public message {label} lineage is cyclic or non-temporal")
        if message.claim_id is not None:
            linked_claim = claim_by_id.get(message.claim_id)
            if linked_claim is None or linked_claim.thread_id != message.thread_id:
                raise ForumTwinError("public message claim is unresolved")
            if linked_claim.first_seen_tick > message.created_tick:
                raise ForumTwinError("public message sees a future claim")
            if message.claim_operation == "create" and (
                linked_claim.created_by_message_id != message.message_id
                or linked_claim.first_seen_tick != message.created_tick
            ):
                raise ForumTwinError("public claim creation does not bind its creator")
        if message.correction_target_claim_id is not None:
            target = claim_by_id.get(message.correction_target_claim_id)
            if target is None or target.thread_id != message.thread_id:
                raise ForumTwinError("public correction target is unresolved")
            if target.first_seen_tick >= message.created_tick:
                raise ForumTwinError("public correction does not follow its target")
    for claim in claim_values:
        creator = message_by_id.get(claim.created_by_message_id)
        if (
            creator is None
            or creator.thread_id != claim.thread_id
            or creator.created_tick != claim.first_seen_tick
        ):
            raise ForumTwinError("public claim creator is unresolved")
        for linked_id, label in (
            (claim.parent_claim_id, "parent"),
            (claim.correction_target_claim_id, "correction target"),
        ):
            if linked_id is None:
                continue
            linked = claim_by_id.get(linked_id)
            if linked is None or linked.thread_id != claim.thread_id:
                raise ForumTwinError(f"public claim {label} is unresolved")
            if linked.first_seen_tick >= claim.first_seen_tick:
                raise ForumTwinError(f"public claim {label} lineage is cyclic or non-temporal")
        for message_id in claim.supporting_message_ids + claim.challenging_message_ids:
            linked_message = message_by_id.get(message_id)
            if linked_message is None or linked_message.thread_id != claim.thread_id:
                raise ForumTwinError("public claim evidence is unresolved")
            if linked_message.created_tick > claim.last_seen_tick:
                raise ForumTwinError("public claim evidence is newer than last_seen_tick")
    return canonical_hash(
        {
            "threads": [thread.__dict__ for thread in sorted(thread_values, key=lambda item: item.thread_id)],
            "messages": [message.public_view() for message in sorted(message_values, key=lambda item: item.message_id)],
            "claims": [claim.__dict__ for claim in sorted(claim_values, key=lambda item: item.claim_id)],
        }
    )
