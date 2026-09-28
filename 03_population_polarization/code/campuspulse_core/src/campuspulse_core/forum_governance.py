"""ForumTwin governance actor-v2 contracts built on the audited QRE kernel.

This module reuses YuLan-Scale's action catalogue, dominance pruning, delayed
rollout and resource ledger.  It adds only the ForumTwin-specific observation,
claim/thread targeting and public governance-message contracts.  No policy or
branch receives an outcome bonus.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .governance_arena_v2 import (
    ACTION_CATALOG,
    BRANCH_ACTIONS,
    BranchState,
    _prune_dominated_actions,
)
from .yulan_scale import (
    ACTOR_IDS,
    PUBLISHABLE_PROVENANCES,
    GovernanceActorDecision,
    TurnEnvelope,
    YuLanScaleError,
    play_llm_governance_round_v1,
)


ACTOR_TURN_V2_SCHEMA = "campus-pulse-governance-actor-turn-v2"
FORUM_GAME_V2_SCHEMA = "campus-pulse-forum-governance-qre-v2"
ACTOR_TURN_V2_KEYS = frozenset(
    {
        "action_probabilities",
        "preferred_action",
        "confidence",
        "rationale_tags",
        "risk_level",
        "uncertainty",
        "public_message",
        "target_claim_id",
        "target_thread_id",
        "evidence_or_service_reference",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CONTACT = re.compile(
    r"(?:1[3-9]\d{9})|(?:\b\d{5,}\b)|(?:https?://)|(?:www\.)|(?:微信|QQ)",
    re.IGNORECASE,
)
_FORBIDDEN_CONTEXT_KEYS = (
    "truth",
    "future",
    "source_id",
    "raw_text",
    "raw_prompt",
    "provider_body",
    "api_key",
    "password",
    "secret",
    "private_observation",
)
_OBSERVATION_PREFIXES = {
    "governance_authority": (
        "discussion_", "risk_", "pps_", "claim_", "uncertainty_",
    ),
    "service_operator": (
        "help_", "sla_", "capacity_", "backlog_", "service_",
    ),
    "community_bridge": (
        "local_", "cross_group_", "correction_", "participation_",
    ),
}
_ARTIFACT_KIND_BY_ACTION = {
    "verified_update": "evidence_card",
    "evidence_dashboard": "evidence_card",
    "targeted_prebunk": "evidence_card",
    "queue_status": "service_ticket",
    "support_channel": "service_ticket",
    "listening_session": "cross_group_outreach",
    "peer_feedback": "cross_group_outreach",
    "bridge_outreach": "cross_group_outreach",
}


@dataclass(frozen=True)
class RenderedForumGovernanceTurn:
    actor_id: str
    scenario_id: str
    branch: str
    tick: int
    feasible_actions: tuple[str, ...]
    target_ids: tuple[str, ...]
    claim_target_ids: tuple[str, ...]
    thread_target_ids: tuple[str, ...]
    service_target_ids: tuple[str, ...]
    claim_thread_pairs: tuple[tuple[str, str], ...]
    service_target_required: bool
    system_prompt: str
    actor_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class ForumGovernanceDecision:
    actor_id: str
    action_probabilities: dict[str, float]
    preferred_action: str
    confidence: float
    rationale_tags: tuple[str, ...]
    risk_level: str
    uncertainty: float
    public_message: str
    target_claim_id: str | None
    target_thread_id: str | None
    evidence_or_service_reference: str | None
    provenance: str


def _persistent_artifact(
    *,
    row: Mapping[str, Any],
    decision: ForumGovernanceDecision,
    scenario_id: str,
    branch: str,
    tick: int,
) -> dict[str, Any] | None:
    action_id = str(row.get("selected_action") or "noop")
    if not row.get("scheduled") or action_id == "noop":
        return None
    artifact_kind = _ARTIFACT_KIND_BY_ACTION[action_id]
    due_tick = tick + int(row["execution_delay"])
    expires_tick = (
        23
        if artifact_kind == "evidence_card"
        else min(23, due_tick + (8 if artifact_kind == "service_ticket" else 5))
    )
    target = (
        decision.target_claim_id
        or decision.target_thread_id
        or "bounded-board-audience"
    )
    if artifact_kind == "evidence_card":
        confirmed = (
            "卡片只发布当前可见证据支持的事实与责任边界",
            "发布行为不等于争议、伤害或问责已经结束",
        )
        unresolved = (
            "未被当前证据支持的动机、完整逐字稿和后续处置",
            "居民是否接受、质疑或忽略该证据卡",
        )
    elif artifact_kind == "service_ticket":
        confirmed = (
            "工单已登记并绑定合成服务责任角色与处理窗口",
            "受理回执只证明进入队列，不证明问题已经解决",
        )
        unresolved = (
            "具体处理结果和居民后续反馈",
            "超出当前服务范围的现实事项",
        )
    else:
        confirmed = (
            "跨群触达对象已排期并绑定当前可见目标",
            "触达只表示内容被投递，不表示居民认同",
        )
        unresolved = (
            "不同群体的阅读、回应、拒绝和二次传播",
            "未阅读当前公共热榜或最新帖的现实人群",
        )
    artifact_body = {
        "artifact_kind": artifact_kind,
        "action_id": action_id,
        "responsible_role": decision.actor_id,
        "status": "scheduled",
        "created_tick": tick,
        "due_tick": due_tick,
        "expires_tick": expires_tick,
        "target_thread_id": decision.target_thread_id,
        "target_claim_id": decision.target_claim_id,
        "public_scope": f"{artifact_kind}:{target}",
        "confirmed_scope": list(confirmed),
        "unresolved_scope": list(unresolved),
    }
    artifact_id = "governance-artifact-" + hashlib.sha256(
        _canonical(
            [scenario_id, branch, tick, decision.actor_id, artifact_body]
        ).encode("utf-8")
    ).hexdigest()[:24]
    return {"artifact_id": artifact_id, **artifact_body}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scan_context(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or any(
                forbidden in key.lower() for forbidden in _FORBIDDEN_CONTEXT_KEYS
            ):
                raise YuLanScaleError(
                    f"Forum governance context contains a forbidden field at {path}"
                )
            _scan_context(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_context(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise YuLanScaleError("Forum governance context is non-finite")


def _safe_optional_id(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise YuLanScaleError(f"{field} is not a safe synthetic identifier")
    return value


def render_forum_governance_actor_turn_v2(
    *,
    actor_id: str,
    scenario_id: str,
    branch: str,
    tick: int,
    private_observation: Mapping[str, float],
    forum_context: Mapping[str, Any],
    target_catalog: Sequence[Mapping[str, Any]],
    feasible_actions: Sequence[str],
    resource: Mapping[str, float],
) -> RenderedForumGovernanceTurn:
    """Render an actor-specific forum prompt with no hidden population truth."""

    if (
        actor_id not in ACTOR_IDS
        or branch not in {"A", "B", "C", "D"}
        # T4/T6 are the two decision points of the 118-slot live vignette and
        # T9 is the middle point of the competition flagship.  Callers still
        # schedule rounds explicitly; accepting a tick here creates no call.
        or tick not in {3, 4, 5, 6, 7, 9, 13, 19}
        or not isinstance(private_observation, Mapping)
        or not isinstance(forum_context, Mapping)
        or not isinstance(target_catalog, Sequence)
        or isinstance(target_catalog, (str, bytes))
        or len(target_catalog) > 12
    ):
        raise YuLanScaleError("Forum governance actor identity is invalid")
    prefixes = _OBSERVATION_PREFIXES[actor_id]
    normalized_observation: dict[str, float] = {}
    for key, value in private_observation.items():
        if (
            not isinstance(key, str)
            or not key.startswith(prefixes)
            or "truth" in key.lower()
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise YuLanScaleError("Forum actor received an observation outside its role")
        normalized_observation[key] = round(float(value), 6)
    _scan_context(forum_context)
    context_json = _canonical(forum_context)
    if len(context_json) > 1_200:
        raise YuLanScaleError("Forum governance context exceeds 1,200 characters")

    targets = []
    target_ids = []
    claim_thread_pairs = []
    for item in target_catalog:
        if not isinstance(item, Mapping) or set(item) != {
            "target_id", "target_kind", "linked_thread_id", "summary",
            "last_visible_tick",
        }:
            raise YuLanScaleError("Forum governance target is invalid")
        target_id = _safe_optional_id(item["target_id"], "target_id")
        linked_thread_id = _safe_optional_id(
            item["linked_thread_id"], "linked_thread_id"
        )
        if (
            target_id is None
            or item["target_kind"] not in {"claim", "thread", "service_case"}
            or not isinstance(item["summary"], str)
            or not 1 <= len(item["summary"]) <= 160
            or isinstance(item["last_visible_tick"], bool)
            or not isinstance(item["last_visible_tick"], int)
            or not 0 <= item["last_visible_tick"] <= tick
        ):
            raise YuLanScaleError("Forum governance target is not time-visible")
        if item["target_kind"] == "claim":
            if linked_thread_id is None:
                raise YuLanScaleError("Forum claim target lacks its linked thread")
            claim_thread_pairs.append((target_id, linked_thread_id))
        elif item["target_kind"] == "thread":
            if linked_thread_id != target_id:
                raise YuLanScaleError("Forum thread target link is inconsistent")
        elif linked_thread_id is not None:
            raise YuLanScaleError("Service target cannot link a forum thread")
        targets.append(dict(item))
        target_ids.append(target_id)

    opening = float(resource.get("opening", 0.0))
    consumed = float(resource.get("consumed", 0.0))
    committed = float(resource.get("committed", 0.0))
    remaining = float(resource.get("remaining", 0.0))
    if not all(math.isfinite(value) and value >= 0 for value in (
        opening, consumed, committed, remaining,
    )) or not math.isclose(opening, consumed + committed + remaining, abs_tol=1e-8):
        raise YuLanScaleError("Forum governance resource view is not conserved")
    permitted_branch_actions = set(BRANCH_ACTIONS[branch])
    allowed = tuple(
        action_id
        for action_id in _prune_dominated_actions(feasible_actions)
        if action_id in permitted_branch_actions
        and ACTION_CATALOG.get(action_id, {}).get("actor") == actor_id
        and float(ACTION_CATALOG[action_id]["cost"]) <= remaining + 1e-9
    )

    prompt = {
        "schema_version": ACTOR_TURN_V2_SCHEMA,
        "actor_id": actor_id,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "private_observation": normalized_observation,
        "visible_forum_context": dict(forum_context),
        "target_catalog": targets,
        "closed_set_target_values": {
            "claim_ids": [
                item["target_id"]
                for item in targets
                if item["target_kind"] == "claim"
            ],
            "thread_ids": [
                item["target_id"]
                for item in targets
                if item["target_kind"] == "thread"
            ],
            "service_case_ids": [
                item["target_id"]
                for item in targets
                if item["target_kind"] == "service_case"
            ],
        },
        "targeting_modes": {
            "none": {
                "target_claim_id": None,
                "target_thread_id": None,
                "rule": "use for noop or an untargeted board-wide notice",
            },
            "claim": {
                "target_claim_id": "copy exactly one closed_set_target_values.claim_ids token",
                "target_thread_id": None,
                "rule": "the runtime resolves the claim's linked thread; never also set target_thread_id",
            },
            "thread": {
                "target_claim_id": None,
                "target_thread_id": "copy exactly one closed_set_target_values.thread_ids token",
                "rule": "use for a thread-level response or outreach",
            },
            "service_help_thread": {
                "target_claim_id": None,
                "target_thread_id": "copy exactly one visible help-thread token",
                "evidence_or_service_reference": "copy that same help-thread token",
                "rule": "required for a non-noop service action when pending_help_threads is positive",
            },
        },
        "resource": {
            "opening": round(opening, 6),
            "consumed": round(consumed, 6),
            "committed": round(committed, 6),
            "remaining": round(remaining, 6),
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
            "action_probabilities": "exactly noop plus every feasible action; sums to 1",
            "preferred_action": "one probability key",
            "confidence": "0..1",
            "rationale_tags": "one to three short tags",
            "risk_level": "low|medium|high|uncertain",
            "uncertainty": "0..1",
            "public_message": "empty or at most 120 Chinese characters",
            "target_claim_id": (
                "JSON null, or copy one token exactly from "
                "closed_set_target_values.claim_ids; when non-null, "
                "target_thread_id must be JSON null"
            ),
            "target_thread_id": (
                "JSON null, or copy one token exactly from "
                "closed_set_target_values.thread_ids; when non-null, "
                "target_claim_id must be JSON null"
            ),
            "evidence_or_service_reference": (
                "JSON null, or copy one token exactly from any list in "
                "closed_set_target_values; labels such as F1/E1/G1/S1, action names, "
                "descriptions and newly invented IDs are forbidden"
            ),
        },
        "required_top_level_keys": sorted(ACTOR_TURN_V2_KEYS),
        "additional_top_level_keys_forbidden": True,
        "validation_rules": {
            "single_target_level_rule": (
                "target_claim_id and target_thread_id are mutually exclusive. "
                "Choose one targeting_modes entry exactly. When a claim is selected, "
                "the runtime derives its linked thread from target_catalog."
            ),
            "closed_set_target_rule": (
                "Target fields are closed-set machine references, not prose. Copy a token "
                "character-for-character from closed_set_target_values or use JSON null. "
                "If preferred_action is noop, use null for all three target fields. "
                "For a non-service action, evidence_or_service_reference may safely be null."
            ),
            "actionable_public_message_rule": (
                "when publishing a non-noop governance message, state what is known "
                "and unknown plus one concrete next step. For service actions, name "
                "the receipt meaning, the bounded response window and the responsible "
                "synthetic service role; never claim the case is already resolved. "
                "For authority actions, name the synthetic notice or appeal location, "
                "review window and review responsibility without inventing a real office."
            ),
        },
    }
    actor_prompt = _canonical(prompt)
    system_prompt = (
        "You are one bounded synthetic campus governance actor. Use only your "
        "role-specific observation, visible synthetic forum context and feasible "
        "actions. Never infer hidden population truth or future events. Return one JSON "
        "object containing exactly required_top_level_keys and no other keys. Target fields "
        "are closed-set identifiers: copy only supplied tokens or return JSON null; never "
        "invent an evidence label, action name, description, or placeholder."
    )
    prompt_sha = hashlib.sha256(
        _canonical(
            {
                "system_prompt": system_prompt,
                "actor_prompt": actor_prompt,
                "actor_id": actor_id,
            }
        ).encode("utf-8")
    ).hexdigest()
    return RenderedForumGovernanceTurn(
        actor_id=actor_id,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        feasible_actions=allowed,
        target_ids=tuple(target_ids),
        claim_target_ids=tuple(
            item["target_id"]
            for item in targets
            if item["target_kind"] == "claim"
        ),
        thread_target_ids=tuple(
            item["target_id"]
            for item in targets
            if item["target_kind"] == "thread"
        ),
        service_target_ids=tuple(
            item["target_id"]
            for item in targets
            if item["target_kind"] == "service_case"
        ),
        claim_thread_pairs=tuple(claim_thread_pairs),
        service_target_required=(
            actor_id == "service_operator"
            and int(forum_context.get("pending_help_threads", 0)) > 0
        ),
        system_prompt=system_prompt,
        actor_prompt=actor_prompt,
        prompt_sha256=prompt_sha,
    )


def validate_forum_governance_actor_turn_v2(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumGovernanceTurn,
) -> ForumGovernanceDecision:
    if (
        envelope.slot_kind != "governance_actor"
        or envelope.prompt_sha256 != rendered.prompt_sha256
        or envelope.provenance not in PUBLISHABLE_PROVENANCES
    ):
        raise YuLanScaleError("Forum governance turn provenance or prompt drifted")
    try:
        raw = json.loads(envelope.content)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise YuLanScaleError("Forum governance turn is not valid JSON") from exc
    expected = ACTOR_TURN_V2_KEYS
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise YuLanScaleError("Forum governance output keys do not match v2")
    allowed = {"noop", *rendered.feasible_actions}
    probabilities = raw["action_probabilities"]
    if not isinstance(probabilities, Mapping) or set(probabilities) != allowed:
        raise YuLanScaleError("Forum governance probabilities are incomplete")
    normalized = {}
    for action_id in sorted(allowed):
        value = probabilities[action_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise YuLanScaleError("Forum governance probability is invalid")
        normalized[action_id] = float(value)
    if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-6):
        raise YuLanScaleError("Forum governance probabilities do not sum to one")
    preferred = raw["preferred_action"]
    if preferred not in allowed:
        raise YuLanScaleError("Forum governance preferred action is infeasible")
    tags = raw["rationale_tags"]
    if (
        not isinstance(tags, list)
        or not 1 <= len(tags) <= 3
        or any(not isinstance(tag, str) or not 1 <= len(tag) <= 40 for tag in tags)
    ):
        raise YuLanScaleError("Forum governance rationale tags are invalid")
    if raw["risk_level"] not in {"low", "medium", "high", "uncertain"}:
        raise YuLanScaleError("Forum governance risk level is invalid")
    for field in ("confidence", "uncertainty"):
        value = raw[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise YuLanScaleError(f"Forum governance {field} is invalid")
    public_message = raw["public_message"]
    if (
        not isinstance(public_message, str)
        or len(public_message) > 120
        or _CONTACT.search(public_message)
    ):
        raise YuLanScaleError("Forum governance public message is unsafe")
    targets = {
        field: _safe_optional_id(raw[field], field)
        for field in (
            "target_claim_id", "target_thread_id",
            "evidence_or_service_reference",
        )
    }
    target_domains = {
        "target_claim_id": set(rendered.claim_target_ids),
        "target_thread_id": set(rendered.thread_target_ids),
        "evidence_or_service_reference": set(rendered.target_ids),
    }
    for field, value in targets.items():
        if value is not None and value not in target_domains[field]:
            raise YuLanScaleError(
                f"Forum governance {field} targets invisible context"
            )
    claim_id = targets["target_claim_id"]
    thread_id = targets["target_thread_id"]
    if claim_id is not None and thread_id is not None:
        raise YuLanScaleError(
            "Forum governance claim and thread targets are mutually exclusive"
        )
    if rendered.service_target_required and preferred != "noop":
        if (
            thread_id is None
            or thread_id not in rendered.target_ids
            or targets["evidence_or_service_reference"] != thread_id
            or not public_message.strip()
        ):
            raise YuLanScaleError(
                "Forum service action must target and acknowledge one visible help thread"
            )
    return ForumGovernanceDecision(
        actor_id=rendered.actor_id,
        action_probabilities=normalized,
        preferred_action=str(preferred),
        confidence=float(raw["confidence"]),
        rationale_tags=tuple(tags),
        risk_level=str(raw["risk_level"]),
        uncertainty=float(raw["uncertainty"]),
        public_message=" ".join(public_message.strip().split()),
        target_claim_id=targets["target_claim_id"],
        target_thread_id=targets["target_thread_id"],
        evidence_or_service_reference=targets["evidence_or_service_reference"],
        provenance=envelope.provenance,
    )


def play_forum_governance_round_v2(
    *,
    state: BranchState,
    actor_decisions: Mapping[str, ForumGovernanceDecision],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the audited QRE/resource solver and emit only LLM-backed messages."""

    if branch == "natural":
        raise YuLanScaleError("Natural cannot create ForumTwin governance turns")
    if set(actor_decisions) != set(ACTOR_IDS):
        raise YuLanScaleError("Forum governance round lacks three actors")
    if any(
        decision.provenance not in PUBLISHABLE_PROVENANCES
        for decision in actor_decisions.values()
    ):
        raise YuLanScaleError("Forum governance message lacks formal LLM provenance")
    base = {
        actor: GovernanceActorDecision(
            actor_id=decision.actor_id,
            action_probabilities=dict(decision.action_probabilities),
            preferred_action=decision.preferred_action,
            confidence=decision.confidence,
            rationale_tags=decision.rationale_tags,
            risk_level=decision.risk_level,
            uncertainty=decision.uncertainty,
            public_message=decision.public_message,
        )
        for actor, decision in actor_decisions.items()
    }
    public_v1, private_v1 = play_llm_governance_round_v1(
        state=state,
        actor_decisions=base,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
        temperature=0.20,
    )
    governance_messages = []
    governance_artifacts = []
    decisions = []
    for row in public_v1["decisions"]:
        actor = row["actor"]
        decision = actor_decisions[actor]
        artifact = _persistent_artifact(
            row=row,
            decision=decision,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
        )
        enriched = {
            **row,
            "target_claim_id": decision.target_claim_id,
            "target_thread_id": decision.target_thread_id,
            "evidence_or_service_reference": decision.evidence_or_service_reference,
            "llm_provenance": decision.provenance,
            "governance_artifact": artifact,
        }
        decisions.append(enriched)
        if artifact is not None:
            governance_artifacts.append(artifact)
        if row["scheduled"] and decision.public_message:
            governance_messages.append(
                {
                    "action": "governance_message",
                    "source_role": actor,
                    "visible_text": decision.public_message,
                    "target_claim_id": decision.target_claim_id,
                    "target_thread_id": decision.target_thread_id,
                    "evidence_or_service_reference": (
                        decision.evidence_or_service_reference
                    ),
                    "scenario_id": scenario_id,
                    "branch": branch,
                    "created_tick": tick,
                    "provenance": decision.provenance,
                    "governance_artifact": artifact,
                }
            )
    public = {
        **public_v1,
        "schema_version": FORUM_GAME_V2_SCHEMA,
        "decisions": decisions,
        "governance_messages": governance_messages,
        "governance_artifacts": governance_artifacts,
        "scheme_bonus": 0,
        "uses_hidden_truth": False,
    }
    private = {
        **private_v1,
        "schema_version": FORUM_GAME_V2_SCHEMA,
        "forum_targets": {
            actor: {
                "claim": decision.target_claim_id,
                "thread": decision.target_thread_id,
                "reference": decision.evidence_or_service_reference,
            }
            for actor, decision in actor_decisions.items()
        },
    }
    if not public["resource_after_decision"]["conserved"]:
        raise YuLanScaleError("Forum governance resource ledger is not conserved")
    return public, private
