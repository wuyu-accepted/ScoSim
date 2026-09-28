"""Social Visibility Harness contracts for ForumTwin.

The harness keeps four delivery channels plus the public author's visibility
choice explicit instead of flattening all social behaviour into one feed.  It owns no Provider transport.  Its
responsibilities are deterministic run binding, activation priority and
privacy-preserving risk summaries that can be replayed and audited.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .forum_twin import ForumLedger, ForumTwinError
from .mass_population import canonical_hash


RUN_PLAN_V10_SCHEMA_VERSION = "campus-pulse-run-plan-v10"
RESULT_V8_SCHEMA_VERSION = "campus-pulse-live-aggregate-result-v8"
CHECKPOINT_V6_SCHEMA_VERSION = "campus-pulse-forum-checkpoint-v6"
VISIBILITY_HARNESS_SCHEMA_VERSION = "campus-pulse-social-visibility-harness-v1"
ACTIVATION_ALGORITHM_VERSION = "visibility-priority-rotation-pps-v3"

CHANNELS = (
    "public_forum",
    "friend_chat",
    "dynamic_group",
    "service_desk",
)
VISIBILITY_RISK_SIGNALS = (
    "anonymous_claim_velocity",
    "private_to_public_spillover",
    "visible_hidden_opinion_divergence",
    "correction_channel_penetration",
    "service_conversion_gap",
)


def _probability(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ForumTwinError(f"{label} must be a probability")
    return float(value)


@dataclass(frozen=True)
class ActivationCandidate:
    agent_id: str
    stratum: str
    design_weight: float
    speaking_probability: float
    particle_uncertainty: float
    ticks_since_activation: int
    is_anchor: bool = False
    direct_reply_or_correction: bool = False
    service_receipt_or_risk_help: bool = False
    private_inbox_pending: bool = False
    active_claim_participant: bool = False
    novel_cross_group_exposure: bool = False

    def __post_init__(self) -> None:
        if not self.agent_id or not self.stratum:
            raise ForumTwinError("activation candidate identity is incomplete")
        if float(self.design_weight) <= 0.0:
            raise ForumTwinError("activation candidate design weight must be positive")
        _probability(self.speaking_probability, "speaking_probability")
        _probability(self.particle_uncertainty, "particle_uncertainty")
        if self.ticks_since_activation < 0:
            raise ForumTwinError("ticks_since_activation cannot be negative")

    def priority_tier(self) -> int:
        if self.is_anchor:
            return 0
        if self.direct_reply_or_correction:
            return 1
        if self.service_receipt_or_risk_help:
            return 2
        if self.private_inbox_pending:
            return 3
        if self.active_claim_participant:
            return 4
        if self.novel_cross_group_exposure:
            return 5
        return 6

    def pps_score(self) -> float:
        rotation = min(2.5, 1.0 + self.ticks_since_activation / 8.0)
        return (
            float(self.design_weight)
            * max(0.02, float(self.speaking_probability))
            * max(0.05, float(self.particle_uncertainty))
            * rotation
        )


def select_visibility_activations_v3(
    candidates: Sequence[ActivationCandidate],
    *,
    budget: int,
    private_origin_share_max: float = 0.40,
) -> tuple[str, ...]:
    """Deterministically select anchors, urgent interactions and rotating PPS.

    The production sampler records positive first-order inclusion
    probabilities.  This scheduler is the deterministic ordering surface used
    by the UI, fixture and full-population planner; it never writes language.
    """

    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ForumTwinError("activation budget must be a positive integer")
    private_origin_share_max = _probability(
        private_origin_share_max, "private_origin_share_max"
    )
    unique = {candidate.agent_id: candidate for candidate in candidates}
    if len(unique) != len(candidates) or budget > len(candidates):
        raise ForumTwinError("activation candidates repeat or cannot fill budget")
    ordered = sorted(
        candidates,
        key=lambda value: (
            value.priority_tier(),
            -value.pps_score(),
            canonical_hash([ACTIVATION_ALGORITHM_VERSION, value.agent_id]),
        ),
    )
    selected: list[ActivationCandidate] = []
    private_cap = int(math.floor(budget * private_origin_share_max))
    private_selected = 0
    for candidate in ordered:
        private_only = (
            candidate.private_inbox_pending
            and not candidate.direct_reply_or_correction
            and not candidate.service_receipt_or_risk_help
        )
        if private_only and private_selected >= private_cap and candidate.priority_tier() > 3:
            continue
        selected.append(candidate)
        private_selected += int(private_only)
        if len(selected) == budget:
            break
    if len(selected) != budget:
        raise ForumTwinError("activation scheduler could not fill the frozen budget")
    return tuple(candidate.agent_id for candidate in selected)


def build_forum_twin_run_plan_v10(
    *,
    parent_plan_v9: Mapping[str, Any],
    profile_override_release_sha256: str,
    channel_configuration: Mapping[str, bool],
    public_anonymity_enabled: bool = True,
    relationship_attention_enabled: bool = True,
) -> dict[str, Any]:
    """Bind the immutable v9 simulation plan to the visibility extension."""

    if parent_plan_v9.get("schema_version") != "campus-pulse-run-plan-v9":
        raise ForumTwinError("v10 requires one canonical v9 parent plan")
    parent_sha = str(parent_plan_v9.get("plan_sha256") or "")
    parent_body = {
        key: value for key, value in parent_plan_v9.items() if key != "plan_sha256"
    }
    if len(parent_sha) != 64 or canonical_hash(parent_body) != parent_sha:
        raise ForumTwinError("v9 parent plan hash drifted")
    if len(profile_override_release_sha256) != 64:
        raise ForumTwinError("profile override release hash is invalid")
    if set(channel_configuration) != set(CHANNELS) or any(
        not isinstance(value, bool) for value in channel_configuration.values()
    ):
        raise ForumTwinError("visibility channel configuration is incomplete")
    if channel_configuration["dynamic_group"] and not channel_configuration["friend_chat"]:
        raise ForumTwinError("dynamic groups require the private relationship channel")
    if not isinstance(public_anonymity_enabled, bool):
        raise ForumTwinError("public_anonymity_enabled must be boolean")
    if not isinstance(relationship_attention_enabled, bool):
        raise ForumTwinError("relationship_attention_enabled must be boolean")
    body = {
        "schema_version": RUN_PLAN_V10_SCHEMA_VERSION,
        "execution_mode": "llm_forum_twin_visibility",
        "parent_plan_v9_sha256": parent_sha,
        "profile_override_release_sha256": profile_override_release_sha256,
        "visibility_harness": {
            "schema_version": VISIBILITY_HARNESS_SCHEMA_VERSION,
            "activation_algorithm": ACTIVATION_ALGORITHM_VERSION,
            "channels": {key: channel_configuration[key] for key in CHANNELS},
            "public_ranking": "global_top_ten_plus_latest",
            "relationship_attention_enabled": relationship_attention_enabled,
            "public_anonymity_enabled": public_anonymity_enabled,
            "public_anonymity_semantics": "same_forum_optional_author_face",
            "anonymous_author_visible_to_governance": False,
            "private_text_visible_to_governance": False,
            "same_tick_two_phase_commit": True,
            "risk_signals": list(VISIBILITY_RISK_SIGNALS),
        },
        "resident_turn_schema_version": "campus-pulse-forum-agent-turn-v4",
        "checkpoint_schema_version": CHECKPOINT_V6_SCHEMA_VERSION,
        "result_schema_version": RESULT_V8_SCHEMA_VERSION,
    }
    return {**body, "plan_sha256": canonical_hash(body)}


def validate_forum_twin_run_plan_v10(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical visibility sidecar without changing its v9 parent.

    The parent plan is stored next to this sidecar in the workbench snapshot.
    Only its immutable SHA is duplicated here, so old v9 plans remain readable
    while new projects bind the resident-v4 visibility contract.
    """

    if not isinstance(plan, Mapping):
        raise ForumTwinError("run-plan v10 must be an object")
    expected_keys = {
        "schema_version",
        "execution_mode",
        "parent_plan_v9_sha256",
        "profile_override_release_sha256",
        "visibility_harness",
        "resident_turn_schema_version",
        "checkpoint_schema_version",
        "result_schema_version",
        "plan_sha256",
    }
    if set(plan) != expected_keys:
        raise ForumTwinError("run-plan v10 fields drifted")
    if (
        plan.get("schema_version") != RUN_PLAN_V10_SCHEMA_VERSION
        or plan.get("execution_mode") != "llm_forum_twin_visibility"
        or plan.get("resident_turn_schema_version")
        != "campus-pulse-forum-agent-turn-v4"
        or plan.get("checkpoint_schema_version") != CHECKPOINT_V6_SCHEMA_VERSION
        or plan.get("result_schema_version") != RESULT_V8_SCHEMA_VERSION
    ):
        raise ForumTwinError("run-plan v10 identity is invalid")
    for name in ("parent_plan_v9_sha256", "profile_override_release_sha256"):
        value = plan.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ForumTwinError(f"{name} is invalid")
    harness = plan.get("visibility_harness")
    expected_harness_keys = {
        "schema_version",
        "activation_algorithm",
        "channels",
        "public_ranking",
        "relationship_attention_enabled",
        "public_anonymity_enabled",
        "public_anonymity_semantics",
        "anonymous_author_visible_to_governance",
        "private_text_visible_to_governance",
        "same_tick_two_phase_commit",
        "risk_signals",
    }
    if not isinstance(harness, Mapping) or set(harness) != expected_harness_keys:
        raise ForumTwinError("visibility harness fields drifted")
    channels = harness.get("channels")
    if (
        not isinstance(channels, Mapping)
        or set(channels) != set(CHANNELS)
        or any(not isinstance(value, bool) for value in channels.values())
        or channels.get("public_forum") is not True
        or (channels.get("dynamic_group") and not channels.get("friend_chat"))
    ):
        raise ForumTwinError("visibility channel configuration is invalid")
    if any(
        not isinstance(harness.get(name), bool)
        for name in (
            "relationship_attention_enabled",
            "public_anonymity_enabled",
            "anonymous_author_visible_to_governance",
            "private_text_visible_to_governance",
            "same_tick_two_phase_commit",
        )
    ):
        raise ForumTwinError("visibility harness boolean fields are invalid")
    if (
        harness.get("schema_version") != VISIBILITY_HARNESS_SCHEMA_VERSION
        or harness.get("activation_algorithm") != ACTIVATION_ALGORITHM_VERSION
        or harness.get("public_ranking") != "global_top_ten_plus_latest"
        or harness.get("public_anonymity_semantics")
        != "same_forum_optional_author_face"
        or harness.get("anonymous_author_visible_to_governance") is not False
        or harness.get("private_text_visible_to_governance") is not False
        or harness.get("same_tick_two_phase_commit") is not True
        or harness.get("risk_signals") != list(VISIBILITY_RISK_SIGNALS)
    ):
        raise ForumTwinError("visibility harness semantics drifted")
    expected_sha = canonical_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected_sha:
        raise ForumTwinError("run-plan v10 hash drifted")
    return dict(plan)


def build_visibility_risk_result_v1(
    *,
    ledger: ForumLedger,
    scenario_id: str,
    branch: str,
    tick: int,
    previous_public_peak_tick: int | None = None,
    service_ticket_count: int = 0,
    service_receipt_count: int = 0,
) -> dict[str, Any]:
    """Create a privacy-safe five-signal visibility snapshot."""

    if not isinstance(tick, int) or isinstance(tick, bool) or not 0 <= tick <= 23:
        raise ForumTwinError("visibility risk tick is invalid")
    if min(service_ticket_count, service_receipt_count) < 0:
        raise ForumTwinError("service counters cannot be negative")
    public_messages = []
    for value in ledger.messages.values():
        thread = ledger.threads.get(value.thread_id)
        if (
            thread is not None
            and thread.scenario_id == scenario_id
            and thread.branch in {branch, "shared_baseline"}
            and not value.superseded
            and value.created_tick <= tick
        ):
            public_messages.append(value)
    private_messages = [
        value
        for value in ledger.private_messages.values()
        if (
            not value.superseded
            and value.created_tick <= tick
            and (conversation := ledger.private_conversations.get(value.conversation_id))
            is not None
            and conversation.scenario_id == scenario_id
            and conversation.branch in {branch, "shared_baseline"}
        )
    ]
    anonymous_messages = [
        value
        for value in public_messages
        if getattr(value, "author_visibility", "named") == "anonymous"
    ]
    named_messages = [
        value
        for value in public_messages
        if getattr(value, "author_visibility", "named") == "named"
    ]
    private_to_public = sum(
        bool(value.private_hearsay_origin) for value in public_messages
    )
    speculative_private = sum(
        value.evidence_status in {"hearsay", "speculation", "unverified"}
        for value in private_messages
    )
    anonymous_uncertain = sum(
        value.evidence_status in {"unverified", "uncertain", "contested"}
        for value in anonymous_messages
    )
    named_uncertain = sum(
        value.evidence_status in {"unverified", "uncertain", "contested"}
        for value in named_messages
    )
    named_ratio = named_uncertain / max(1, len(named_messages))
    hidden_ratio = (speculative_private + anonymous_uncertain) / max(
        1, len(private_messages) + len(anonymous_messages)
    )
    correction_count = sum(
        value.evidence_status in {"corrected", "verified"}
        for value in private_messages + public_messages
    )
    correction_denominator = max(
        1, speculative_private + anonymous_uncertain + correction_count
    )
    signal_values = {
        "anonymous_claim_velocity": round(
            anonymous_uncertain / max(1, tick + 1),
            6,
        ),
        "private_to_public_spillover": round(
            private_to_public / max(1, len(public_messages)), 6
        ),
        "visible_hidden_opinion_divergence": round(
            abs(hidden_ratio - named_ratio), 6
        ),
        "correction_channel_penetration": round(
            correction_count / correction_denominator, 6
        ),
        "service_conversion_gap": round(
            max(0.0, 1.0 - service_receipt_count / max(1, service_ticket_count)),
            6,
        ),
    }
    body = {
        "schema_version": "campus-pulse-governance-risk-result-v2",
        "visibility_harness_schema_version": VISIBILITY_HARNESS_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "signals": [
            {"signal": name, "value": signal_values[name]}
            for name in VISIBILITY_RISK_SIGNALS
        ],
        "channel_counts": {
            "public_messages": len(public_messages),
            "named_public_messages": len(named_messages),
            "anonymous_public_messages": len(anonymous_messages),
            "private_messages": len(private_messages),
            "first_anonymous_public_tick": min(
                (value.created_tick for value in anonymous_messages),
                default=None,
            ),
            "previous_public_peak_tick": previous_public_peak_tick,
            "service_tickets": service_ticket_count,
            "service_receipts": service_receipt_count,
        },
        "privacy": {
            "contains_private_text": False,
            "contains_private_agent_ids": False,
            "contains_anonymous_author_ids": False,
            "contains_relationship_edges": False,
        },
    }
    return {**body, "content_sha256": canonical_hash(body)}


__all__ = [
    "ACTIVATION_ALGORITHM_VERSION",
    "ActivationCandidate",
    "CHANNELS",
    "CHECKPOINT_V6_SCHEMA_VERSION",
    "RESULT_V8_SCHEMA_VERSION",
    "RUN_PLAN_V10_SCHEMA_VERSION",
    "VISIBILITY_HARNESS_SCHEMA_VERSION",
    "VISIBILITY_RISK_SIGNALS",
    "build_forum_twin_run_plan_v10",
    "build_visibility_risk_result_v1",
    "select_visibility_activations_v3",
    "validate_forum_twin_run_plan_v10",
]
