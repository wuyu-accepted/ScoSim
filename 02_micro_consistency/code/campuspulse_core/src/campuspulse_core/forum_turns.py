"""Strict resident-turn renderer, validator and ForumLedger materializer.

The transport boundary remains the existing :class:`TurnEnvelope`.  This
module never calls a Provider: it renders a bounded prompt, validates one exact
JSON response, and materializes public content only for authorized live LLM or
exact reviewed-trace provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from typing import Any, Mapping, Sequence

from .forum_twin import (
    CLAIM_OPERATIONS,
    EVIDENCE_STATUSES,
    INTERACTION_KINDS,
    MAX_FORUM_CONTEXT_CHARS,
    PUBLIC_PROVENANCE_KINDS,
    REPORT_REASON_TAGS,
    STANCE_VALUES,
    TURN_SCHEMA_VERSION,
    AgentMemory,
    FeedExposure,
    ForumClaim,
    ForumInteraction,
    ForumLedger,
    ForumMessage,
    ForumMessageV2,
    ForumThread,
    ForumTwinError,
    PRIVATE_CLAIM_STATUSES,
    PRIVATE_EVIDENCE_STATUSES,
    PRIVATE_MEMBERSHIP_ACTIONS,
    PRIVATE_MESSAGE_ACTIONS,
    PRIVATE_REACTION_KINDS,
    PrivateClaim,
    PrivateConversation,
    PrivateLlmMemoryEpisode,
    PrivateMembershipEvent,
    PrivateMessage,
    PrivateReaction,
    PublicProvenance,
    RealLlmMemoryEpisode,
    build_feed_context,
    render_forum_turn_payload,
    validate_prompt_payload,
)
from .governance_arena_v2 import TIMELINE
from .mass_population import canonical_hash
from .yulan_scale import TurnEnvelope


FORUM_AGENT_TURN_SCHEMA_VERSION = "campus-pulse-forum-agent-turn-v1"
FORUM_AGENT_TURN_V2_SCHEMA_VERSION = "campus-pulse-forum-agent-turn-v2"
FORUM_AGENT_TURN_V3_SCHEMA_VERSION = "campus-pulse-forum-agent-turn-v3"
FORUM_AGENT_TURN_V4_SCHEMA_VERSION = "campus-pulse-forum-agent-turn-v4"
FORUM_AGENT_TURN_V5_SCHEMA_VERSION = "campus-pulse-forum-agent-turn-v5"
V5_FULL_NATURAL_PROFILE = "full_natural_profile"
V5_CONTEXT_ABLATION = "context_ablation"
V5_CONTEXT_ONLY = "context_only"
V5_CONTEXT_PLUS_PROFILE = "context_plus_profile"
V5_MODEL_VIEW_CONDITIONS = frozenset(
    {
        V5_FULL_NATURAL_PROFILE,
        V5_CONTEXT_ABLATION,
        V5_CONTEXT_ONLY,
        V5_CONTEXT_PLUS_PROFILE,
    }
)
FORUM_PRIVATE_SHADOW_TURN_SCHEMA_VERSION = (
    "campus-pulse-forum-private-shadow-turn-v5"
)
RESIDENT_SLOT_KIND = "resident"
MAX_SCENARIO_CONTEXT_CHARS = 1_600
MAX_INTERACTIONS_PER_TURN = 6
MAX_PRIVATE_ACTIONS_PER_TURN = 2
RESIDENT_ACTIONS = frozenset(
    {
        "ignore",
        "read",
        "post",
        "reply",
        "quote",
        "repost",
        "verify",
        "correct",
        "seek_help",
        "report_risk",
    }
)
PUBLIC_RESIDENT_ACTIONS = RESIDENT_ACTIONS - {"ignore", "read"}
NEW_THREAD_ACTIONS = frozenset({"post", "seek_help", "report_risk"})
EMOTION_VALUES = frozenset(
    {
        "neutral",
        "calm",
        "curious",
        "concerned",
        "skeptical",
        "frustrated",
        "supportive",
        "urgent",
    }
)
OUTPUT_FIELDS = frozenset(
    {
        "action",
        "target_thread_id",
        "target_message_id",
        "quote_message_id",
        "target_claim_id",
        "topic",
        "subtopic",
        "stance",
        "urgency",
        "emotion",
        "confidence",
        "evidence_status",
        "claim_operation",
        "claim_summary",
        "correction_target_claim_id",
        "trust_delta",
        "satisfaction_delta",
        "memory_update",
        "visible_text",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_MECHANISM_REQUIREMENT_FIELDS = frozenset(
    {
        "action",
        "claim_operation",
        "stance",
        "target_thread_id",
        "target_message_id",
        "target_claim_id",
        "correction_target_claim_id",
    }
)
_UNSAFE_TEXT_RE = re.compile(
    r"(?:https?://|www\.|(?<!\d)1[3-9]\d{9}(?!\d)|"
    r"[A-Za-z]:[\\/]|/(?:home|Users|srv|var|tmp)/|"
    r"sk-[A-Za-z0-9_-]{12,}|api[_ -]?key|password|secret)",
    re.I,
)


class ForumTurnError(ForumTwinError):
    """Raised when a ForumTwin resident turn violates its frozen contract."""


@dataclass(frozen=True)
class RenderedForumAgentTurn:
    agent_id: str
    source_agent_display_id: str
    scenario_id: str
    branch: str
    tick: int
    current_thread_id: str | None
    direct_parent_message_id: str | None
    quote_message_id: str | None
    allowed_actions: tuple[str, ...]
    allowed_topics: tuple[str, ...]
    visible_message_ids: tuple[str, ...]
    visible_claim_ids: tuple[str, ...]
    feed_context: tuple[dict[str, Any], ...]
    public_board: dict[str, Any]
    expected_model: str
    system_prompt: str
    agent_prompt: str
    prompt_sha256: str
    mechanism_requirements: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ForumAgentDecision:
    action: str
    target_thread_id: str | None
    target_message_id: str | None
    quote_message_id: str | None
    target_claim_id: str | None
    topic: str
    subtopic: str
    stance: str
    urgency: float
    emotion: str
    confidence: float
    evidence_status: str
    claim_operation: str
    claim_summary: str
    correction_target_claim_id: str | None
    trust_delta: float
    satisfaction_delta: float
    memory_update: str
    visible_text: str
    private_origin_claim_id: str | None = None


@dataclass(frozen=True)
class ValidatedForumAgentTurn:
    rendered: RenderedForumAgentTurn
    decision: ForumAgentDecision
    provenance: PublicProvenance


@dataclass(frozen=True)
class MaterializedForumTurn:
    action: str
    message_id: str | None
    thread_id: str | None
    claim_id: str | None
    memory_episode_count: int
    silent: bool
    content_hash: str


@dataclass(frozen=True)
class ForumInteractionDecision:
    kind: str
    target_message_id: str
    report_reason_tag: str | None = None


@dataclass(frozen=True)
class RenderedForumAgentTurnV2:
    """V2 prompt carrier: one speech move plus parallel engagements."""

    base_rendered: RenderedForumAgentTurn
    system_prompt: str
    agent_prompt: str
    prompt_sha256: str

    @property
    def agent_id(self) -> str:
        return self.base_rendered.agent_id

    @property
    def source_agent_display_id(self) -> str:
        return self.base_rendered.source_agent_display_id

    @property
    def scenario_id(self) -> str:
        return self.base_rendered.scenario_id

    @property
    def branch(self) -> str:
        return self.base_rendered.branch

    @property
    def tick(self) -> int:
        return self.base_rendered.tick

    @property
    def expected_model(self) -> str:
        return self.base_rendered.expected_model

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_actions

    @property
    def allowed_topics(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_topics

    @property
    def visible_message_ids(self) -> tuple[str, ...]:
        return self.base_rendered.visible_message_ids

    @property
    def visible_claim_ids(self) -> tuple[str, ...]:
        return self.base_rendered.visible_claim_ids

    @property
    def mechanism_requirements(self) -> tuple[tuple[str, str], ...]:
        return self.base_rendered.mechanism_requirements


@dataclass(frozen=True)
class ValidatedForumAgentTurnV2:
    rendered: RenderedForumAgentTurnV2
    decision: ForumAgentDecision
    interactions: tuple[ForumInteractionDecision, ...]
    provenance: PublicProvenance


@dataclass(frozen=True)
class MaterializedForumTurnV2:
    action: str
    message_id: str | None
    thread_id: str | None
    claim_id: str | None
    interaction_ids: tuple[str, ...]
    interaction_count: int
    memory_episode_count: int
    silent: bool
    content_hash: str


def _nonempty(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ForumTurnError(f"{field} must be text")
    text = " ".join(value.strip().split())
    if not text or len(text) > maximum:
        raise ForumTurnError(f"{field} must contain 1-{maximum} characters")
    if _UNSAFE_TEXT_RE.search(text):
        raise ForumTurnError(f"{field} contains a contact, URL, path or credential")
    return text


def _optional_id(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ForumTurnError(f"{field} must be null or a bounded identifier")
    text = value.strip()
    # Machine identifiers are already constrained to a deliberately small
    # alphabet.  Do not apply the natural-language PII scanner here: a
    # cryptographic digest can legitimately contain an eleven-digit run that
    # merely resembles a phone number.  The strict identifier grammar still
    # rejects whitespace, URLs, paths, credentials and arbitrary source text.
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise ForumTurnError(f"{field} must be null or a bounded identifier")
    return text


def _bounded_number(value: Any, field: str, low: float, high: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not low <= float(value) <= high
    ):
        raise ForumTurnError(f"{field} must be a finite number from {low} to {high}")
    return round(float(value), 6)


def _profile_view(
    agent: Mapping[str, Any],
    *,
    require_reviewed_evidence: bool,
    tick: int,
) -> tuple[str, str, dict[str, Any]]:
    if agent.get("provider_turn_contract") != FORUM_AGENT_TURN_SCHEMA_VERSION:
        raise ForumTurnError("agent is not bound to the ForumTwin turn contract")
    agent_id = _nonempty(agent.get("agent_id"), "agent_id", maximum=100)
    display_id = _nonempty(
        agent.get("synthetic_display_id"),
        "synthetic_display_id",
        maximum=40,
    )
    profile_sha = str(agent.get("profile_prompt_sha256") or "")
    if _SHA256_RE.fullmatch(profile_sha) is None:
        raise ForumTurnError("agent profile prompt SHA-256 is invalid")
    binding = str(agent.get("evidence_binding") or "")
    if binding not in {"reviewed_evidence_match", "reviewed_profile_fallback"}:
        raise ForumTurnError("agent evidence binding is invalid")
    if require_reviewed_evidence and binding != "reviewed_evidence_match":
        raise ForumTurnError("hero/formal activation cannot use a fallback profile")
    references = agent.get("reviewed_evidence_refs")
    if (
        not isinstance(references, list)
        or not 3 <= len(references) <= 5
        or any(not isinstance(value, str) or not value.startswith("evidence-") for value in references)
    ):
        raise ForumTurnError("agent profile must bind 3-5 reviewed semantic references")
    required = (
        "primary_macro_role",
        "secondary_macro_role",
        "micro_role_id",
        "micro_interaction_role",
        "episode_focus",
        "topic_portfolio",
        "need_portfolio",
        "forum_tendencies",
        "trust_style",
        "uncertainty_handling",
        "language_style",
        "activity_rhythm",
        "attention_budget",
        "long_term_memory_seed",
        "behavior_boundaries",
        "safety_gates",
    )
    if any(key not in agent for key in required):
        raise ForumTurnError("agent profile bundle is incomplete")
    if not isinstance(agent.get("topic_portfolio"), list) or not agent["topic_portfolio"]:
        raise ForumTurnError("agent topic portfolio is empty")
    if not isinstance(agent.get("need_portfolio"), list) or not agent["need_portfolio"]:
        raise ForumTurnError("agent need portfolio is empty")
    for field in ("forum_tendencies", "trust_style", "language_style"):
        if not isinstance(agent.get(field), Mapping):
            raise ForumTurnError(f"agent {field} must be an object")
    for field in ("behavior_boundaries", "safety_gates"):
        if not isinstance(agent.get(field), list) or not agent[field]:
            raise ForumTurnError(f"agent {field} must be a non-empty list")
    profile = {
        "profile_bundle_id": agent.get("profile_bundle_id"),
        "profile_prompt_sha256": profile_sha,
        "primary_macro_role": agent["primary_macro_role"],
        "secondary_macro_role": agent["secondary_macro_role"],
        "micro_role_id": agent["micro_role_id"],
        "micro_interaction_role": agent["micro_interaction_role"],
        "episode_focus": agent["episode_focus"],
        "reviewed_behavior_examples": [
            dict(value)
            for value in agent.get("reviewed_behavior_examples", [])
            if isinstance(value, Mapping)
        ][:5],
        "stable_persona": dict(
            agent.get("stable_persona")
            or {
                "interaction_role": agent["micro_interaction_role"],
                "uncertainty_handling": agent["uncertainty_handling"],
                "language_style": dict(agent["language_style"]),
            }
        ),
        "episodic_evidence": dict(
            agent.get("episodic_evidence")
            or {
                "episode_focus": agent["episode_focus"],
                "topic_portfolio": list(agent["topic_portfolio"]),
                "need_portfolio": list(agent["need_portfolio"]),
                "reviewed_evidence_refs": list(references),
                "usage_rule": (
                    "historical behavior evidence only; use only when the current "
                    "scenario or delivered feed is substantively relevant"
                ),
            }
        ),
        "scenario_relevance": dict(
            agent.get("scenario_relevance")
            or {
                "mode": "feed_and_current_scenario_conditioned",
                "default_when_unrelated": "do_not_force_episode_topic_into_response",
            }
        ),
        "topic_portfolio": list(agent["topic_portfolio"]),
        "need_portfolio": list(agent["need_portfolio"]),
        "forum_tendencies": dict(agent["forum_tendencies"]),
        "trust_style": dict(agent["trust_style"]),
        "uncertainty_handling": agent["uncertainty_handling"],
        "language_style": dict(agent["language_style"]),
        "activity_rhythm": agent["activity_rhythm"],
        "attention_budget": agent["attention_budget"],
        "long_term_memory_seed": agent["long_term_memory_seed"],
        "behavior_boundaries": list(agent["behavior_boundaries"]),
        "safety_gates": list(agent["safety_gates"]),
        "reviewed_evidence_refs": list(references),
        "evidence_binding": binding,
    }
    validate_prompt_payload(profile, current_tick=tick)
    return agent_id, display_id, profile


def _allowed_actions_for_agent(
    agent: Mapping[str, Any],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    actions = tuple(dict.fromkeys(requested or sorted(RESIDENT_ACTIONS)))
    if not actions or any(action not in RESIDENT_ACTIONS for action in actions):
        raise ForumTurnError("resident allowed-actions catalogue is invalid")
    gate = str(agent.get("simulation_gate") or "active_episode_agent")
    macro = str(agent.get("primary_macro_role") or "")
    if gate == "background_signal" or macro == "archetype-16":
        admissible = {"ignore", "read"}
    elif gate == "risk_signal" or macro == "archetype-09":
        admissible = {"ignore", "read", "report_risk"}
    else:
        admissible = set(RESIDENT_ACTIONS)
    result = tuple(action for action in actions if action in admissible)
    if not result:
        raise ForumTurnError("profile gate removed every resident action")
    return result


def _profile_interaction_move(profile: Mapping[str, Any]) -> dict[str, str]:
    """Translate reviewed reply-function evidence into one bounded social move.

    The move is a language-realisation bias, not a forced action.  It keeps the
    full-corpus profile visible to the LLM without turning one historical
    episode into a biography or making every resident sound like an official
    information desk.
    """

    language = profile.get("language_style") or {}
    tendencies = profile.get("forum_tendencies") or {}
    raw = str(
        language.get("reply_function")
        or tendencies.get("speech_act")
        or "other"
    ).strip().lower()
    aliases = {
        "ask_followup": "question",
        "information_request": "question",
        "emotional_expression": "emotional_vent",
        "risk_report": "complaint",
        "contextual_participant": "other",
    }
    function = aliases.get(raw, raw)
    moves: dict[str, tuple[str, ...]] = {
        "question": (
            "只追问一个真正影响判断的缺失信息；若同一问题已在可见消息中出现，就阅读或换一个新问题。",
            "针对父消息中的一个具体断言追问依据，不把回复写成‘入口、时限、负责方’三项清单。",
        ),
        "emotional_vent": (
            "先表达此事带来的真实情绪或被冒犯感，再给一句判断；不要自动转成程序建议。",
            "用自然的论坛口吻表达不满、受伤、讽刺或共情，不必为了显得理性而主动调和分歧。",
        ),
        "help_request": (
            "当当前事件确实触发需要时，优先用 seek_help 说明一项具体支持需要和希望得到的回应。",
            "把个人困惑或压力说具体；不要把求助伪装成面向所有人的治理方案。",
        ),
        "information_share": (
            "只补充当前可见材料能够支持的一条信息或经验，并明确它不能证明什么。",
            "提供一个有边界的信息点；没有新信息时选择阅读，不复述事实卡。",
        ),
        "complaint": (
            "明确指出被破坏的边界及其影响，不必用‘双方都有道理’稀释已经确认的冒犯。",
            "聚焦一个具体不满或后果，允许直接、尖锐但不威胁、不人肉、不补造事实。",
        ),
        "request": (
            "提出一个清楚、可回应的诉求，不同时罗列多个官样化流程要求。",
            "说清最想得到的一项回应及原因；不要替其他人假定共同诉求。",
        ),
        "discussion": (
            "推进一个反例、分歧或新角度，而不是复述父消息后表示赞同。",
            "回应父消息的一处具体逻辑，可保留不同意见，不以达成共识为目标。",
        ),
        "experience_share_or_ask": (
            "只分享不含真实身份细节的合成经验感受，或据此提出一个相关问题。",
            "把经验和当前帖子连接起来，但不得把历史 episode 当成本次真实经历。",
        ),
        "correction": (
            "只有可见证据足以收窄某条 claim 时才纠正，并直接指向该 claim；否则选择求证或阅读。",
            "说明原主张哪一部分超出证据、应怎样最小幅度收窄，不断言相反的未知结论。",
        ),
        "other": (
            "没有与画像和当前 Feed 都相关的新贡献时，选择 read 或 ignore。",
            "若发言，只完成一个自然的社会动作；不要模仿公告、总结报告或万能理性旁观者。",
        ),
    }
    options = moves.get(function, moves["other"])
    offset = int(str(profile["profile_prompt_sha256"])[:12], 16) % len(options)
    return {
        "reviewed_reply_function": raw or "other",
        "primary_social_move": options[offset],
        "status": "profile_bias_not_forced_action",
    }


def _selected_context(
    *,
    ledger: ForumLedger,
    exposures: Sequence[FeedExposure],
    recipient_agent_id: str,
    tick: int,
    current_thread_id: str | None,
    direct_parent_message_id: str | None,
    quote_message_id: str | None,
) -> tuple[dict[str, Any], ...]:
    if any(exposure.recipient_agent_id != recipient_agent_id for exposure in exposures):
        raise ForumTurnError("resident Feed contains another agent's private exposure")
    required: list[str] = []
    if current_thread_id is not None:
        thread = ledger.threads.get(current_thread_id)
        if thread is None or thread.created_tick > tick:
            raise ForumTurnError("resident current thread is missing or from the future")
        required.append(thread.root_message_id)
        for message_id, label in (
            (direct_parent_message_id, "direct parent"),
            (quote_message_id, "quote"),
        ):
            if message_id is None:
                continue
            message = ledger.messages.get(message_id)
            if (
                message is None
                or message.thread_id != current_thread_id
                or message.created_tick > tick
            ):
                raise ForumTurnError(f"resident {label} is not time-visible in the thread")
            required.append(message_id)
    elif direct_parent_message_id is not None or quote_message_id is not None:
        raise ForumTurnError("parent/quote context requires a current thread")
    required = list(dict.fromkeys(required))
    eligible = [exposure for exposure in exposures if exposure.delivered_tick <= tick]
    by_message: dict[str, FeedExposure] = {}
    for exposure in sorted(
        eligible,
        key=lambda item: (-float(item.rank_score), item.delivered_tick, item.exposure_id),
    ):
        by_message.setdefault(exposure.message_id, exposure)
    if any(message_id not in by_message for message_id in required):
        raise ForumTurnError("root/parent/quote was not delivered to this resident")
    extra = [
        exposure
        for message_id, exposure in by_message.items()
        if message_id not in set(required)
    ]
    extra.sort(
        key=lambda item: (-float(item.rank_score), item.delivered_tick, item.exposure_id)
    )
    selected = [by_message[message_id] for message_id in required] + extra[:3]
    context = build_feed_context(
        ledger=ledger,
        exposures=selected,
        recipient_agent_id=recipient_agent_id,
        current_tick=tick,
        current_thread_id=current_thread_id,
        direct_parent_message_id=direct_parent_message_id,
        quote_message_id=quote_message_id,
    )
    by_id = {str(item["message_id"]): dict(item) for item in context}
    if any(message_id not in by_id for message_id in required):
        raise ForumTurnError("required root/parent/quote context was filtered")
    ordered = [by_id[message_id] for message_id in required]
    ordered.extend(
        item for item in context if str(item["message_id"]) not in set(required)
    )
    if sum(item["relation"] == "feed_summary" for item in ordered) > 3:
        raise ForumTurnError("resident turn contains more than three additional summaries")
    if sum(len(str(item.get("visible_text") or "")) for item in ordered) > MAX_FORUM_CONTEXT_CHARS:
        raise ForumTurnError("resident turn exceeds its forum text budget")
    return tuple(ordered)


def render_forum_agent_turn_v1(
    *,
    agent: Mapping[str, Any],
    posterior_state: Mapping[str, Any],
    memory: AgentMemory,
    ledger: ForumLedger,
    exposures: Sequence[FeedExposure],
    scenario_id: str,
    branch: str,
    tick: int,
    expected_model: str,
    scenario_context: Mapping[str, Any] | None = None,
    public_board: Mapping[str, Any] | None = None,
    current_thread_id: str | None = None,
    direct_parent_message_id: str | None = None,
    quote_message_id: str | None = None,
    allowed_actions: Sequence[str] | None = None,
    require_reviewed_evidence: bool = True,
    forum_context_mode: str = "full_text",
) -> RenderedForumAgentTurn:
    """Render one bounded ForumTwin resident prompt without Provider access."""

    if not isinstance(tick, int) or isinstance(tick, bool) or not 0 <= tick < len(TIMELINE):
        raise ForumTurnError("resident turn tick is invalid")
    agent_id, display_id, profile = _profile_view(
        agent,
        require_reviewed_evidence=require_reviewed_evidence,
        tick=tick,
    )
    if memory.agent_id != agent_id:
        raise ForumTurnError("resident memory belongs to another agent")
    model = _nonempty(expected_model, "expected_model", maximum=120)
    actions = _allowed_actions_for_agent(agent, allowed_actions)
    context = _selected_context(
        ledger=ledger,
        exposures=exposures,
        recipient_agent_id=agent_id,
        tick=tick,
        current_thread_id=current_thread_id,
        direct_parent_message_id=direct_parent_message_id,
        quote_message_id=quote_message_id,
    )
    if forum_context_mode not in {"full_text", "metadata_only"}:
        raise ForumTurnError("resident forum context mode is invalid")
    if forum_context_mode == "metadata_only":
        context = tuple(
            {
                **item,
                "visible_text": "",
                "text_available": False,
            }
            for item in context
        )
    has_visible_messages = any(item.get("message_id") for item in context)
    has_visible_claims = any(item.get("claim_id") for item in context)
    context_admissible_actions = {
        "ignore",
        "read",
        "post",
        "seek_help",
        "report_risk",
    }
    if has_visible_messages:
        context_admissible_actions.update({"reply", "quote", "repost"})
    if has_visible_claims:
        context_admissible_actions.update({"verify", "correct"})
    actions = tuple(
        action for action in actions if action in context_admissible_actions
    )
    if not actions:
        raise ForumTurnError("time-visible context removed every resident action")
    thread_root_topic: str | None = None
    if current_thread_id is not None:
        current_thread = ledger.threads[current_thread_id]
        if (
            current_thread.scenario_id != scenario_id
            or current_thread.branch not in {branch, "shared_baseline"}
        ):
            raise ForumTurnError("resident current thread belongs to another scenario/branch")
        root_message = ledger.messages.get(current_thread.root_message_id)
        if root_message is None:
            raise ForumTurnError("resident current thread root message is missing")
        thread_root_topic = root_message.topic
    topics = tuple(
        dict.fromkeys(
            [str(value) for value in profile["topic_portfolio"]]
            + [scenario_id, "off_topic_daily_life"]
            + ([thread_root_topic] if thread_root_topic is not None else [])
        )
    )
    base = render_forum_turn_payload(
        agent_profile=profile,
        posterior_state=posterior_state,
        memory=memory,
        feed_context=context,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        allowed_actions=actions,
        current_thread_id=current_thread_id,
    )["payload"]
    base["schema"] = FORUM_AGENT_TURN_SCHEMA_VERSION
    base["forum_context_mode"] = forum_context_mode
    public_board_view: dict[str, Any] = {}
    if public_board is not None:
        if not isinstance(public_board, Mapping):
            raise ForumTurnError("public_board must be an object")
        public_board_view = dict(public_board)
        validate_prompt_payload(public_board_view, current_tick=tick)
        try:
            board_chars = len(
                json.dumps(
                    public_board_view,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ForumTurnError("public_board must be strict JSON") from exc
        if board_chars > 2_600:
            raise ForumTurnError("public_board exceeds 2,600 characters")
        if public_board_view.get("ranking_identical_for_all_residents") is not True:
            raise ForumTurnError("public_board must disclose global identical ranking")
        base["public_board"] = public_board_view
    mechanism_requirements: dict[str, str] = {}
    if scenario_context is not None:
        if not isinstance(scenario_context, Mapping):
            raise ForumTurnError("scenario_context must be an object")
        scenario_view = dict(scenario_context)
        validate_prompt_payload(scenario_view, current_tick=tick)
        try:
            scenario_chars = len(
                json.dumps(
                    scenario_view,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ForumTurnError("scenario_context must be strict JSON") from exc
        if scenario_chars > MAX_SCENARIO_CONTEXT_CHARS:
            raise ForumTurnError("scenario_context exceeds 1,600 characters")
        audit = scenario_view.get("mechanism_audit")
        if audit is not None:
            if not isinstance(audit, Mapping):
                raise ForumTurnError("mechanism_audit must be an object")
            requirements = audit.get("required_output")
            if not isinstance(requirements, Mapping) or not requirements:
                raise ForumTurnError(
                    "mechanism_audit must disclose nonempty required_output"
                )
            if set(requirements) - _MECHANISM_REQUIREMENT_FIELDS:
                raise ForumTurnError("mechanism_audit required_output is invalid")
            if any(
                not isinstance(value, str) or not value.strip()
                for value in requirements.values()
            ):
                raise ForumTurnError(
                    "mechanism_audit required_output values must be bounded text"
                )
            mechanism_requirements = {
                str(key): str(value).strip() for key, value in requirements.items()
            }
        base["scenario_context"] = scenario_view
    base["agent_contract"] = {
        "agent_id": agent_id,
        "synthetic_display_id": display_id,
        "profile_prompt_sha256": profile["profile_prompt_sha256"],
        "evidence_binding": profile["evidence_binding"],
    }
    base["participation_brief"] = {
        "micro_role": profile["micro_interaction_role"],
        "stable_interaction_traits": dict(profile["stable_persona"]),
        "historical_episode_evidence": dict(profile["episodic_evidence"]),
        "episode_relevance_rule": (
            "The historical episode is evidence about interaction tendencies, not a "
            "permanent biography or current agenda. Mention its topic only when a delivered "
            "message or the current event makes it directly relevant; otherwise respond from "
            "the stable interaction traits and current situation."
        ),
        "need_focus": list(profile["need_portfolio"])[:2],
        "language_style": dict(profile["language_style"]),
        "interaction_move": _profile_interaction_move(profile),
        "single_move_rule": (
            "One public message should perform one main social move. Do not combine an "
            "emotion, an evidence audit, a service checklist and a governance proposal "
            "into the same templated reply. Read or ignore is valid when no distinct move exists."
        ),
        "forum_register_rule": (
            "Realise the supplied language style in natural campus-forum Chinese. A colloquial "
            "profile must not sound like an official report. Repeated checklist phrasing such as "
            "'入口、时限、负责方' is appropriate only when this profile is asking a genuinely "
            "new service or procedure question and the same checklist is not already visible."
        ),
        "novel_contribution_rule": (
            "If you speak, first address one specific point in the visible parent/root, "
            "then add one profile-grounded contribution that is not already present: a "
            "concrete question, impact, uncertainty, counterpoint, evidence boundary, "
            "help need or bridge summary. If you have nothing new, choose read or ignore."
        ),
        "anti_template_rule": (
            "Do not merely restate another message and do not open with generic agreement "
            "such as '同意', '先核对', '看完整语境' or '不信谣不传谣'. Do not perform "
            "neutrality when the scenario explicitly marks harmful conduct as confirmed. "
            "Natural anger, sarcasm, hurt, moral condemnation, defence, reservation, "
            "selective acceptance, disengagement and silence are all allowed when consistent "
            "with the supplied profile and visible evidence."
        ),
        "confirmed_harm_rule": (
            "Separate confirmed conduct from disputed interpretation. A confirmed insult or "
            "harm must not be downgraded to 'everything is uncertain'; uncertainty applies "
            "only to fields that scenario_context explicitly marks unknown. Civility is not "
            "a substitute for viewpoint realism, and the agent is not required to reconcile."
        ),
        "memory_continuity_rule": (
            "When real_llm_memory is nonempty, explicitly continue, revise or reject one "
            "of your own prior judgements where relevant; do not repeat it as if it were "
            "new. When it is empty, do not invent a prior personal position."
        ),
        "public_board_rule": (
            "public_board is the same global hot-top-10 and latest-post index shown to every "
            "resident at this tick. It is not personalized by profile or relationship graph. "
            "The opened root/parent/feed_context contains the full text you actually read; "
            "respond to that visible item or choose read/ignore."
        ),
        "bridge_relevance_rule": (
            "Legacy cross_thread_bridge compatibility label: there is no relationship-graph recommendation. "
            "A community-bridge governance post is an ordinary public-board item; use it only "
            "when its opened text is concretely relevant, otherwise choose read or ignore."
        ),
    }
    base["allowed_topics"] = list(topics)
    visible_thread_ids = sorted(
        {str(item["thread_id"]) for item in context if item.get("thread_id")}
    )
    visible_message_ids = sorted(
        {str(item["message_id"]) for item in context if item.get("message_id")}
    )
    visible_claim_ids = sorted(
        {
            str(item["claim_id"])
            for item in context
            if item.get("claim_id")
        }
    )
    visible_claim_thread_pairs = sorted(
        {
            (str(item["claim_id"]), str(item["thread_id"]))
            for item in context
            if item.get("claim_id") and item.get("thread_id")
        }
    )
    base["output_contract"] = {
        "strict_json_only": True,
        "required_fields": sorted(OUTPUT_FIELDS),
        "actions": list(actions),
        "topics": list(topics),
        "stances": sorted(STANCE_VALUES),
        "emotions": sorted(EMOTION_VALUES),
        "evidence_statuses": sorted(EVIDENCE_STATUSES),
        "claim_operations": sorted(CLAIM_OPERATIONS),
        "numeric_fields": {
            "urgency": {
                "json_type": "number_not_string",
                "minimum": 0.0,
                "maximum": 1.0,
                "example": 0.5,
            },
            "confidence": {
                "json_type": "number_not_string",
                "minimum": 0.0,
                "maximum": 1.0,
                "example": 0.7,
            },
            "trust_delta": {
                "json_type": "number_not_string",
                "minimum": -0.25,
                "maximum": 0.25,
                "example": 0.0,
            },
            "satisfaction_delta": {
                "json_type": "number_not_string",
                "minimum": -0.25,
                "maximum": 0.25,
                "example": 0.0,
            },
        },
        "numeric_rule": (
            "the four numeric_fields must be bare JSON numbers inside their "
            "inclusive ranges; labels such as low/medium/high, quoted numbers, "
            "null and booleans are forbidden"
        ),
        "available_target_ids": {
            "thread_ids": visible_thread_ids,
            "message_ids": visible_message_ids,
            "claim_ids": visible_claim_ids,
        },
        "available_claim_thread_pairs": [
            {"claim_id": claim_id, "thread_id": thread_id}
            for claim_id, thread_id in visible_claim_thread_pairs
        ],
        "target_rule": (
            "each target ID must be null or copied exactly from available_target_ids; "
            "when no matching visible target exists use null; never invent placeholders "
            "such as new, thread-1, unknown or N/A. Whenever target_claim_id is not null, "
            "target_thread_id must come from the same object in available_claim_thread_pairs"
        ),
        "required_thread_topic": thread_root_topic,
        "thread_topic_rule": (
            "when replying, quoting, reposting, verifying, correcting, seeking help "
            "or reporting risk inside an existing thread, copy required_thread_topic "
            "exactly. The entire visible_text must respond to the root message or the "
            "direct parent within that same substantive issue. Use the profile only to "
            "interpret that issue; never append an unrelated personal question, travel, "
            "food, course, shopping, relationship or logistics topic as an aside. If the "
            "profile has nothing relevant to add, choose read or ignore"
            if thread_root_topic is not None
            else "new roots may choose any allowed topic"
        ),
        "new_root_rule": (
            "for a new root post, target_thread_id, target_message_id, "
            "quote_message_id, target_claim_id and correction_target_claim_id must all be null"
        ),
        "action_templates": {
            "ignore_or_read": (
                "all target IDs null; claim_operation none; claim_summary and visible_text empty"
            ),
            "post": (
                "all target IDs null; use claim_operation create with a nonempty claim_summary, "
                "or none with an empty claim_summary"
            ),
            "reply": (
                "copy one visible thread_id and one message_id as target_thread_id and "
                "target_message_id"
            ),
            "quote_or_repost": (
                "copy one visible thread_id and message_id as target_thread_id and "
                "quote_message_id"
            ),
            "verify": (
                "copy one exact claim_id/thread_id pair from available_claim_thread_pairs; "
                "claim_operation must be challenge"
            ),
            "correct": (
                "copy one exact claim_id/thread_id pair from available_claim_thread_pairs, "
                "then copy the same claim_id into target_claim_id and "
                "correction_target_claim_id; claim_operation correct; evidence_status "
                "corrected or verified"
            ),
            "seek_help_or_report_risk": (
                "either create a new root with null thread/message targets, or copy a "
                "visible thread_id and parent message_id"
            ),
        },
        "silent_rule": "read and ignore require empty visible_text",
        "memory_rule": (
            "public actions require a nonempty memory_update of at most 240 characters; "
            "read/ignore may use an empty memory_update when nothing should be retained"
        ),
        "text_rule": "new root <=180 Chinese chars; all other public text <=120; read/ignore empty",
        "fact_rule": "do not add real-world facts absent from the synthetic context",
        "evidence_update_rule": (
            "A synthetic evidence/service update in scenario_context may be used only at "
            "its stated strength. Distinguish confirmed limits from unresolved claims; "
            "never expand a bounded update into a verdict."
        ),
        "mechanism_audit_requirement": mechanism_requirements or None,
    }
    validate_prompt_payload(base, current_tick=tick)
    agent_prompt = json.dumps(
        base,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = (
        "You are one persistent synthetic CampusPulse forum LLM agent, not a real student. "
        "Use only the supplied profile, posterior, real-LLM memory and time-visible synthetic "
        "forum context. Preserve this agent's micro-role and language style; public speech "
        "must add a distinct, context-responsive contribution rather than a generic paraphrase. "
        "Return exactly one JSON object matching the contract; never invent real identities, "
        "private records, future events or unsupported real-world facts."
    )
    prompt_sha = canonical_hash(
        {
            "renderer": FORUM_AGENT_TURN_SCHEMA_VERSION,
            "expected_model": model,
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
        }
    )
    visible_claims = []
    for item in context:
        claim_id = item.get("claim_id")
        if claim_id and claim_id in ledger.claims and not ledger.claims[claim_id].expired:
            visible_claims.append(str(claim_id))
    return RenderedForumAgentTurn(
        agent_id=agent_id,
        source_agent_display_id=display_id,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        current_thread_id=current_thread_id,
        direct_parent_message_id=direct_parent_message_id,
        quote_message_id=quote_message_id,
        allowed_actions=actions,
        allowed_topics=topics,
        visible_message_ids=tuple(str(item["message_id"]) for item in context),
        visible_claim_ids=tuple(dict.fromkeys(visible_claims)),
        feed_context=tuple(dict(item) for item in context),
        public_board=public_board_view,
        mechanism_requirements=tuple(sorted(mechanism_requirements.items())),
        expected_model=model,
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        prompt_sha256=prompt_sha,
    )


def _strict_response_object(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ForumTurnError("resident response content must be text")
    text = content.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ForumTurnError("resident response must contain only one JSON object")
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ForumTurnError("resident response contains a duplicate JSON key")
            value[key] = child
        return value
    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except ForumTurnError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForumTurnError("resident response is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != OUTPUT_FIELDS:
        raise ForumTurnError("resident response fields do not exactly match the contract")
    return value


def validate_forum_agent_turn_v1(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumAgentTurn,
    ledger: ForumLedger,
    reviewed_trace_release_sha256: str | None = None,
    public_text_limits: tuple[int, int] = (180, 120),
) -> ValidatedForumAgentTurn:
    """Validate exact JSON, targets, claim operations and public provenance."""

    if envelope.slot_kind != RESIDENT_SLOT_KIND:
        raise ForumTurnError("forum resident received a non-resident envelope")
    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise ForumTurnError("forum resident prompt fingerprint drifted")
    if envelope.provenance not in PUBLIC_PROVENANCE_KINDS:
        raise ForumTurnError("emulator/fixture content cannot become a public forum turn")
    if envelope.returned_model != rendered.expected_model:
        raise ForumTurnError("forum resident returned model does not match the run plan")
    if _SHA256_RE.fullmatch(str(envelope.response_sha256)) is None:
        raise ForumTurnError("forum resident response SHA-256 is invalid")
    if envelope.provenance == "reviewed_trace_replay":
        if (
            reviewed_trace_release_sha256 is None
            or _SHA256_RE.fullmatch(reviewed_trace_release_sha256) is None
        ):
            raise ForumTurnError("exact replay must bind a reviewed trace release")
        if envelope.provider_calls or envelope.provider_attempts:
            raise ForumTurnError("reviewed trace replay cannot consume Provider calls")
    elif reviewed_trace_release_sha256 is not None:
        raise ForumTurnError("live LLM response cannot claim a trace release")
    raw = _strict_response_object(envelope.content)
    action = str(raw["action"])
    if action not in rendered.allowed_actions:
        raise ForumTurnError("resident action is not allowed by its profile gate")
    target_thread_id = _optional_id(raw["target_thread_id"], "target_thread_id")
    target_message_id = _optional_id(raw["target_message_id"], "target_message_id")
    quote_message_id = _optional_id(raw["quote_message_id"], "quote_message_id")
    target_claim_id = _optional_id(raw["target_claim_id"], "target_claim_id")
    correction_target = _optional_id(
        raw["correction_target_claim_id"],
        "correction_target_claim_id",
    )
    topic = _nonempty(raw["topic"], "topic", maximum=100)
    if topic not in rendered.allowed_topics:
        raise ForumTurnError("resident topic is outside its profile/scenario portfolio")
    subtopic = _nonempty(raw["subtopic"], "subtopic", maximum=80)
    stance = str(raw["stance"])
    emotion = str(raw["emotion"])
    evidence_status = str(raw["evidence_status"])
    claim_operation = str(raw["claim_operation"])
    if stance not in STANCE_VALUES:
        raise ForumTurnError("resident stance is invalid")
    if emotion not in EMOTION_VALUES:
        raise ForumTurnError("resident emotion is invalid")
    if evidence_status not in EVIDENCE_STATUSES:
        raise ForumTurnError("resident evidence status is invalid")
    if claim_operation not in CLAIM_OPERATIONS:
        raise ForumTurnError("resident claim operation is invalid")
    claim_summary_raw = raw["claim_summary"]
    if not isinstance(claim_summary_raw, str):
        raise ForumTurnError("claim_summary must be text")
    claim_summary = " ".join(claim_summary_raw.strip().split())
    if not isinstance(raw["memory_update"], str):
        raise ForumTurnError("memory_update must be text")
    memory_update = " ".join(raw["memory_update"].strip().split())
    if len(memory_update) > 240:
        raise ForumTurnError("memory_update exceeds 240 characters")
    if memory_update and _UNSAFE_TEXT_RE.search(memory_update):
        raise ForumTurnError(
            "memory_update contains a contact, URL, path or credential"
        )
    if not isinstance(raw["visible_text"], str):
        raise ForumTurnError("visible_text must be text")
    visible_text = " ".join(raw["visible_text"].strip().split())
    if visible_text and _UNSAFE_TEXT_RE.search(visible_text):
        raise ForumTurnError("visible_text contains a contact, URL, path or credential")
    visible_messages = set(rendered.visible_message_ids)
    visible_threads = {
        ledger.messages[message_id].thread_id
        for message_id in visible_messages
        if message_id in ledger.messages
    }
    if target_thread_id is not None and target_thread_id not in visible_threads:
        raise ForumTurnError("resident target thread was not time-visible")
    for message_id, label in (
        (target_message_id, "target message"),
        (quote_message_id, "quote message"),
    ):
        if message_id is not None and message_id not in visible_messages:
            raise ForumTurnError(f"resident {label} was not time-visible")
        if message_id is not None and target_thread_id is not None:
            if ledger.messages[message_id].thread_id != target_thread_id:
                raise ForumTurnError(f"resident {label} belongs to another thread")
    if action == "post" and any(
        value is not None
        for value in (target_thread_id, target_message_id, quote_message_id)
    ):
        raise ForumTurnError("post must create a new root without targets")
    if action == "reply" and (
        target_thread_id is None or target_message_id is None
    ):
        raise ForumTurnError("reply requires a visible thread and parent message")
    if action in {"quote", "repost"} and (
        target_thread_id is None or quote_message_id is None
    ):
        raise ForumTurnError(f"{action} requires a visible quoted message")
    if action in {"seek_help", "report_risk"}:
        if target_thread_id is None and any(
            value is not None for value in (target_message_id, quote_message_id)
        ):
            raise ForumTurnError(f"new-root {action} cannot target a message")
        if target_thread_id is not None and target_message_id is None:
            raise ForumTurnError(f"in-thread {action} requires a visible parent")
    if action in {"ignore", "read"}:
        if visible_text:
            raise ForumTurnError("ignore/read cannot emit public text")
        if claim_operation != "none" or any(
            value is not None
            for value in (target_claim_id, correction_target)
        ) or claim_summary:
            raise ForumTurnError("ignore/read cannot mutate a claim")
    else:
        if not visible_text:
            raise ForumTurnError("public resident action requires visible_text")
        if not memory_update:
            raise ForumTurnError("public resident action requires memory_update")
        new_root = target_thread_id is None and action in NEW_THREAD_ACTIONS
        if len(public_text_limits) != 2 or any(type(v) is not int or not 1 <= v <= 8000 for v in public_text_limits):
            raise ForumTurnError("Invalid public text limits")
        text_limit = public_text_limits[0] if new_root else public_text_limits[1]
        if len(visible_text) > text_limit:
            raise ForumTurnError(f"resident visible_text exceeds {text_limit} characters")
        if target_thread_id is not None:
            target_thread = ledger.threads[target_thread_id]
            root_message = ledger.messages.get(target_thread.root_message_id)
            if root_message is None or topic != root_message.topic:
                raise ForumTurnError(
                    "resident public reply topic must match the thread root topic"
                )
    visible_claims = set(rendered.visible_claim_ids)
    if claim_operation == "none":
        if target_claim_id is not None or correction_target is not None or claim_summary:
            raise ForumTurnError("claim_operation none cannot carry claim fields")
    elif claim_operation == "create":
        if target_claim_id is not None or correction_target is not None:
            raise ForumTurnError("new claim cannot target an existing claim")
        claim_summary = _nonempty(claim_summary, "claim_summary", maximum=160)
    else:
        if target_claim_id not in visible_claims:
            target_thread = target_thread_id or rendered.current_thread_id
            unique_thread_claims = sorted(
                claim_id
                for claim_id in visible_claims
                if claim_id in ledger.claims
                and ledger.claims[claim_id].thread_id == target_thread
                and not ledger.claims[claim_id].expired
            )
            if len(unique_thread_claims) == 1:
                target_claim_id = unique_thread_claims[0]
                if action == "correct" or claim_operation == "correct":
                    correction_target = target_claim_id
        if target_claim_id not in visible_claims:
            raise ForumTurnError("claim operation target is not visible or has expired")
        # Existing claim text is immutable ledger state. Some otherwise valid
        # LLM turns redundantly repeat that visible summary; discarding the
        # duplicate is deterministic and cannot introduce or rewrite facts.
        claim_summary = ""
        target_claim = ledger.claims.get(str(target_claim_id))
        if (
            target_thread_id is None
            or target_claim is None
            or target_claim.thread_id != target_thread_id
        ):
            raise ForumTurnError("existing claim operation must stay in its visible thread")
    if action == "verify" and claim_operation != "challenge":
        raise ForumTurnError("verify request must challenge a visible claim for evidence")
    if claim_operation == "verify" and evidence_status != "verified":
        raise ForumTurnError("verified claim operation requires verified evidence status")
    if action == "correct" or claim_operation == "correct":
        if action != "correct" or claim_operation != "correct":
            raise ForumTurnError("correction action and claim operation must align")
        if correction_target != target_claim_id or correction_target not in visible_claims:
            raise ForumTurnError("correction must target one visible non-expired claim")
        if evidence_status not in {"corrected", "verified"}:
            raise ForumTurnError("correction requires corrected or verified evidence")
    elif correction_target is not None:
        raise ForumTurnError("only correction may carry correction_target_claim_id")
    required = dict(rendered.mechanism_requirements)
    actual = {
        "action": action,
        "claim_operation": claim_operation,
        "stance": stance,
        "target_thread_id": target_thread_id,
        "target_message_id": target_message_id,
        "target_claim_id": target_claim_id,
        "correction_target_claim_id": correction_target,
    }
    for field, expected in required.items():
        if actual[field] != expected:
            raise ForumTurnError(
                f"mechanism audit required {field}={expected}"
            )
    decision = ForumAgentDecision(
        action=action,
        target_thread_id=target_thread_id,
        target_message_id=target_message_id,
        quote_message_id=quote_message_id,
        target_claim_id=target_claim_id,
        topic=topic,
        subtopic=subtopic,
        stance=stance,
        urgency=_bounded_number(raw["urgency"], "urgency", 0.0, 1.0),
        emotion=emotion,
        confidence=_bounded_number(raw["confidence"], "confidence", 0.0, 1.0),
        evidence_status=evidence_status,
        claim_operation=claim_operation,
        claim_summary=claim_summary,
        correction_target_claim_id=correction_target,
        trust_delta=_bounded_number(raw["trust_delta"], "trust_delta", -0.25, 0.25),
        satisfaction_delta=_bounded_number(
            raw["satisfaction_delta"],
            "satisfaction_delta",
            -0.25,
            0.25,
        ),
        memory_update=memory_update,
        visible_text=visible_text,
    )
    provenance = PublicProvenance(
        kind=envelope.provenance,
        prompt_sha256=envelope.prompt_sha256,
        response_sha256=envelope.response_sha256,
        model=envelope.returned_model,
        trace_release_sha256=reviewed_trace_release_sha256,
    )
    return ValidatedForumAgentTurn(
        rendered=rendered,
        decision=decision,
        provenance=provenance,
    )


def _message_identity(validated: ValidatedForumAgentTurn) -> str:
    return "forum-message-" + canonical_hash(
        [
            validated.rendered.scenario_id,
            validated.rendered.branch,
            validated.rendered.tick,
            validated.rendered.agent_id,
            validated.provenance.response_sha256,
        ]
    )[:24]


def materialize_forum_agent_turn_v1(
    *,
    validated: ValidatedForumAgentTurn,
    ledger: ForumLedger,
    memory: AgentMemory,
) -> MaterializedForumTurn:
    """Commit one validated semantic turn; Dynamics/fixture cannot call through."""

    rendered = validated.rendered
    decision = validated.decision
    if validated.provenance.kind not in PUBLIC_PROVENANCE_KINDS:
        raise ForumTurnError("non-LLM provenance cannot materialize a forum turn")
    if memory.agent_id != rendered.agent_id:
        raise ForumTurnError("forum materialization received another agent's memory")
    message_id: str | None = None
    thread_id = decision.target_thread_id
    claim_id = decision.target_claim_id
    if decision.action not in {"ignore", "read"}:
        message_id = _message_identity(validated)
        if thread_id is None:
            if decision.action not in NEW_THREAD_ACTIONS:
                raise ForumTurnError("only root-capable resident actions can create a thread")
            thread_id = "forum-thread-" + canonical_hash(
                [
                    rendered.scenario_id,
                    rendered.branch,
                    rendered.tick,
                    rendered.agent_id,
                    message_id,
                ]
            )[:24]
        else:
            thread = ledger.threads.get(thread_id)
            if (
                thread is None
                or thread.scenario_id != rendered.scenario_id
                or thread.branch not in {rendered.branch, "shared_baseline"}
                or thread.status != "open"
            ):
                raise ForumTurnError("target thread changed after the resident prompt")
        if decision.claim_operation == "create":
            claim_id = "forum-claim-" + canonical_hash(
                [thread_id, message_id, decision.claim_summary]
            )[:24]
        elif decision.claim_operation != "none":
            current_claim = ledger.claims.get(str(claim_id))
            if (
                current_claim is None
                or current_claim.expired
                or current_claim.thread_id != thread_id
            ):
                raise ForumTurnError("target claim changed after the resident prompt")
        correction_target = (
            decision.correction_target_claim_id
            if decision.claim_operation == "correct"
            else None
        )
        message = ForumMessage(
            message_id=message_id,
            thread_id=thread_id,
            source_agent_display_id=rendered.source_agent_display_id,
            action=decision.action,
            visible_text=decision.visible_text,
            topic=decision.topic,
            stance=decision.stance,
            emotion=decision.emotion,
            evidence_status=decision.evidence_status,
            confidence=decision.confidence,
            created_tick=rendered.tick,
            ttl=12 if decision.target_thread_id is None else 8,
            provenance=validated.provenance,
            parent_message_id=decision.target_message_id,
            quote_message_id=decision.quote_message_id,
            claim_id=claim_id,
            claim_operation=decision.claim_operation,
            correction_target_claim_id=correction_target,
            private_hearsay_origin=decision.private_origin_claim_id is not None,
            private_origin_claim_id=decision.private_origin_claim_id,
        )
        if decision.target_thread_id is None:
            ledger.add_thread(
                ForumThread(
                    thread_id=thread_id,
                    board="synthetic-campus-forum",
                    root_message_id=message_id,
                    scenario_id=rendered.scenario_id,
                    branch=rendered.branch,
                    created_tick=rendered.tick,
                    status="open",
                    topic=decision.topic,
                    need=decision.subtopic,
                    participant_count=1,
                    reply_count=0,
                    last_active_tick=rendered.tick,
                    provenance_kind=validated.provenance.kind,
                ),
                message,
            )
        else:
            ledger.add_message(message)
        if decision.claim_operation == "create":
            ledger.add_claim(
                ForumClaim(
                    claim_id=claim_id,
                    thread_id=thread_id,
                    claim_summary=decision.claim_summary,
                    status=(
                        "verified"
                        if decision.evidence_status == "verified"
                        else "contested"
                        if decision.evidence_status == "contested"
                        else "unverified"
                    ),
                    first_seen_tick=rendered.tick,
                    last_seen_tick=rendered.tick,
                    created_by_message_id=message_id,
                    supporting_message_ids=(message_id,),
                )
            )
        elif decision.claim_operation != "none":
            current = ledger.claims[str(claim_id)]
            support = list(current.supporting_message_ids)
            challenge = list(current.challenging_message_ids)
            if decision.claim_operation in {"support", "verify", "correct"}:
                support.append(message_id)
            if decision.claim_operation in {"challenge", "correct"}:
                challenge.append(message_id)
            status = {
                "support": current.status,
                "challenge": "contested",
                "verify": "verified",
                "correct": "corrected",
            }[decision.claim_operation]
            ledger.advance_claim(
                replace(
                    current,
                    status=status,
                    last_seen_tick=rendered.tick,
                    supporting_message_ids=tuple(dict.fromkeys(support)),
                    challenging_message_ids=tuple(dict.fromkeys(challenge)),
                )
            )
    memory_thread_id = thread_id
    if memory_thread_id is None and decision.target_message_id in ledger.messages:
        memory_thread_id = ledger.messages[str(decision.target_message_id)].thread_id
    if decision.memory_update:
        memory.append_real_episode(
            RealLlmMemoryEpisode(
                tick=rendered.tick,
                thread_id=memory_thread_id or "forum-feed",
                action=decision.action,
                semantic_summary=decision.memory_update,
                message_id=message_id,
                provenance=validated.provenance,
            )
        )
    identity = {
        "schema": FORUM_AGENT_TURN_SCHEMA_VERSION,
        "prompt_sha256": rendered.prompt_sha256,
        "response_sha256": validated.provenance.response_sha256,
        "agent_id": rendered.agent_id,
        "action": decision.action,
        "message_id": message_id,
        "thread_id": thread_id,
        "claim_id": claim_id,
        "memory_episode_count": len(memory.recent_real_llm_episodes),
    }
    return MaterializedForumTurn(
        action=decision.action,
        message_id=message_id,
        thread_id=thread_id,
        claim_id=claim_id,
        memory_episode_count=len(memory.recent_real_llm_episodes),
        silent=message_id is None,
        content_hash=canonical_hash(identity),
    )


def render_forum_agent_turn_v2(
    *,
    agent: Mapping[str, Any],
    posterior_state: Mapping[str, Any],
    memory: AgentMemory,
    ledger: ForumLedger,
    exposures: Sequence[FeedExposure],
    scenario_id: str,
    branch: str,
    tick: int,
    expected_model: str,
    scenario_context: Mapping[str, Any] | None = None,
    public_board: Mapping[str, Any] | None = None,
    current_thread_id: str | None = None,
    direct_parent_message_id: str | None = None,
    quote_message_id: str | None = None,
    allowed_actions: Sequence[str] | None = None,
    require_reviewed_evidence: bool = True,
    forum_context_mode: str = "full_text",
) -> RenderedForumAgentTurnV2:
    """Render the additive parallel-interaction resident contract.

    Existing profile releases bound to the V1 speech contract remain usable:
    V2 changes the per-turn output shape, not the reviewed persona evidence.
    The resulting prompt fingerprint is always V2-specific.
    """

    profile_contract = str(agent.get("provider_turn_contract") or "")
    if profile_contract not in {
        FORUM_AGENT_TURN_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
    }:
        raise ForumTurnError("agent is not bound to a compatible ForumTwin turn contract")
    compatible_agent = dict(agent)
    compatible_agent["provider_turn_contract"] = FORUM_AGENT_TURN_SCHEMA_VERSION
    base_rendered = render_forum_agent_turn_v1(
        agent=compatible_agent,
        posterior_state=posterior_state,
        memory=memory,
        ledger=ledger,
        exposures=exposures,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        expected_model=expected_model,
        scenario_context=scenario_context,
        public_board=public_board,
        current_thread_id=current_thread_id,
        direct_parent_message_id=direct_parent_message_id,
        quote_message_id=quote_message_id,
        allowed_actions=allowed_actions,
        require_reviewed_evidence=require_reviewed_evidence,
        forum_context_mode=forum_context_mode,
    )
    payload = json.loads(base_rendered.agent_prompt)
    payload["schema"] = FORUM_AGENT_TURN_V2_SCHEMA_VERSION
    payload["agent_contract"]["provider_turn_contract"] = (
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION
    )
    payload["agent_contract"]["profile_contract_source"] = profile_contract
    payload["participation_brief"]["parallel_interaction_rule"] = (
        "The primary action controls at most one public message. Independently, engagements "
        "may like, repost or report several visible messages in the same tick. A reply may "
        "coexist with a like/repost/report; do not suppress a genuine engagement merely "
        "because you also commented. Use an empty list when none is warranted."
    )
    payload["participation_brief"]["interaction_decision_guide"] = {
        "evaluate_independently": (
            "Evaluate every time-visible message you actually read independently from the "
            "primary speech action. Multiple different engagement kinds may be selected, "
            "but there is no target count and an empty array is valid."
        ),
        "like": (
            "Like only when this resident genuinely endorses, appreciates or finds that "
            "specific message useful; disagreement or mere exposure is not a like."
        ),
        "repost": (
            "Repost only when this resident wants that specific message to reach the wider "
            "public board, for example because it is useful, urgent, identity-expressive or "
            "worth contesting publicly. Repost does not imply agreement."
        ),
        "report": (
            "Report only for a concrete visible safety, privacy, harassment, fraud, spam or "
            "misinformation concern and select the matching reason tag. Do not report a "
            "message merely because the resident disagrees with it."
        ),
        "persona_conditioning": (
            "Use stable_interaction_traits, current posterior and the exact visible text to "
            "decide; do not satisfy a quota or infer an invisible violation."
        ),
    }
    contract = payload["output_contract"]
    contract["required_fields"] = sorted(OUTPUT_FIELDS | {"engagements"})
    contract["engagement_contract"] = {
        "json_type": "array",
        "maximum_items": MAX_INTERACTIONS_PER_TURN,
        "item_exact_fields": ["kind", "target_message_id", "report_reason_tag"],
        "kinds": sorted(INTERACTION_KINDS),
        "target_rule": (
            "copy target_message_id exactly from available_target_ids.message_ids; "
            "a target may receive different kinds in the same turn, but the same "
            "kind/target pair may appear only once"
        ),
        "report_reason_tags": sorted(REPORT_REASON_TAGS),
        "report_rule": (
            "report requires one report_reason_tag; like/repost require null. "
            "A silent repost is an engagement; quote/repost as the primary action "
            "is reserved for a visible public message."
        ),
        "independence_rule": (
            "engagements do not consume or replace the primary speech action"
        ),
    }
    agent_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = (
        base_rendered.system_prompt
        + " Return one primary action and an independent engagements array; likes, silent "
        "reposts and moderation reports may coexist with a reply in the same turn."
    )
    prompt_sha = canonical_hash(
        {
            "renderer": FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
            "expected_model": expected_model,
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
        }
    )
    return RenderedForumAgentTurnV2(
        base_rendered=base_rendered,
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        prompt_sha256=prompt_sha,
    )


def _strict_response_object_v2(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ForumTurnError("resident V2 response content must be text")
    text = content.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ForumTurnError("resident V2 response must contain only one JSON object")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ForumTurnError("resident V2 response contains a duplicate JSON key")
            value[key] = child
        return value

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except ForumTurnError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForumTurnError("resident V2 response is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != OUTPUT_FIELDS | {"engagements"}:
        raise ForumTurnError("resident V2 response fields do not exactly match the contract")
    return value


def validate_forum_agent_turn_v2(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumAgentTurnV2,
    ledger: ForumLedger,
    reviewed_trace_release_sha256: str | None = None,
) -> ValidatedForumAgentTurnV2:
    """Validate one speech action and zero-to-six simultaneous engagements."""

    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise ForumTurnError("forum resident V2 prompt fingerprint drifted")
    raw = _strict_response_object_v2(envelope.content)
    engagement_rows = raw.pop("engagements")
    if not isinstance(engagement_rows, list) or len(engagement_rows) > MAX_INTERACTIONS_PER_TURN:
        raise ForumTurnError("resident engagements must be an array of at most six items")
    base_envelope = replace(
        envelope,
        prompt_sha256=rendered.base_rendered.prompt_sha256,
        content=json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    base_validated = validate_forum_agent_turn_v1(
        base_envelope,
        rendered=rendered.base_rendered,
        ledger=ledger,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
    )
    provenance = replace(
        base_validated.provenance,
        prompt_sha256=rendered.prompt_sha256,
    )
    visible_ids = set(rendered.base_rendered.visible_message_ids)
    interactions: list[ForumInteractionDecision] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(engagement_rows):
        if not isinstance(value, dict) or set(value) != {
            "kind",
            "target_message_id",
            "report_reason_tag",
        }:
            raise ForumTurnError(f"engagements[{index}] does not match its exact contract")
        kind = str(value["kind"])
        if kind not in INTERACTION_KINDS:
            raise ForumTurnError(f"engagements[{index}].kind is invalid")
        target_message_id = _optional_id(
            value["target_message_id"],
            f"engagements[{index}].target_message_id",
        )
        if target_message_id is None or target_message_id not in visible_ids:
            raise ForumTurnError(f"engagements[{index}] target was not time-visible")
        target = ledger.messages.get(target_message_id)
        if target is None or target.superseded or target.created_tick > rendered.tick:
            raise ForumTurnError(f"engagements[{index}] target is unavailable")
        target_thread = ledger.threads.get(target.thread_id)
        if (
            target_thread is None
            or target_thread.scenario_id != rendered.scenario_id
            or target_thread.branch not in {rendered.branch, "shared_baseline"}
        ):
            raise ForumTurnError(f"engagements[{index}] crossed scenario or branch")
        reason = value["report_reason_tag"]
        if kind == "report":
            if reason not in REPORT_REASON_TAGS:
                raise ForumTurnError(f"engagements[{index}] report reason is invalid")
            reason_value: str | None = str(reason)
        else:
            if reason is not None:
                raise ForumTurnError(
                    f"engagements[{index}] non-report reason must be null"
                )
            reason_value = None
        key = (kind, target_message_id)
        if key in seen:
            raise ForumTurnError("resident response duplicates an engagement kind/target")
        seen.add(key)
        if (
            kind == "repost"
            and base_validated.decision.action == "repost"
            and base_validated.decision.quote_message_id == target_message_id
        ):
            raise ForumTurnError("public repost and silent repost cannot double-count one target")
        interactions.append(
            ForumInteractionDecision(
                kind=kind,
                target_message_id=target_message_id,
                report_reason_tag=reason_value,
            )
        )
    return ValidatedForumAgentTurnV2(
        rendered=rendered,
        decision=base_validated.decision,
        interactions=tuple(interactions),
        provenance=provenance,
    )


def materialize_forum_agent_turn_v2(
    *,
    validated: ValidatedForumAgentTurnV2,
    ledger: ForumLedger,
    memory: AgentMemory,
) -> MaterializedForumTurnV2:
    """Commit one optional message plus independent like/repost/report events."""

    base_validated = ValidatedForumAgentTurn(
        rendered=validated.rendered.base_rendered,
        decision=validated.decision,
        provenance=validated.provenance,
    )
    base = materialize_forum_agent_turn_v1(
        validated=base_validated,
        ledger=ledger,
        memory=memory,
    )
    interaction_ids: list[str] = []
    for decision in validated.interactions:
        target = ledger.messages[decision.target_message_id]
        interaction_id = "forum-interaction-" + canonical_hash(
            [
                validated.rendered.scenario_id,
                validated.rendered.branch,
                validated.rendered.tick,
                validated.rendered.agent_id,
                decision.kind,
                decision.target_message_id,
                validated.provenance.response_sha256,
            ]
        )[:24]
        ledger.add_interaction(
            ForumInteraction(
                interaction_id=interaction_id,
                scenario_id=validated.rendered.scenario_id,
                branch=validated.rendered.branch,
                thread_id=target.thread_id,
                target_message_id=decision.target_message_id,
                source_agent_display_id=validated.rendered.source_agent_display_id,
                kind=decision.kind,
                created_tick=validated.rendered.tick,
                report_reason_tag=decision.report_reason_tag,
                provenance=validated.provenance,
            )
        )
        interaction_ids.append(interaction_id)
    identity = {
        "schema": FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        "prompt_sha256": validated.rendered.prompt_sha256,
        "response_sha256": validated.provenance.response_sha256,
        "agent_id": validated.rendered.agent_id,
        "message_content_hash": base.content_hash,
        "interaction_ids": interaction_ids,
        "memory_episode_count": len(memory.recent_real_llm_episodes),
    }
    return MaterializedForumTurnV2(
        action=base.action,
        message_id=base.message_id,
        thread_id=base.thread_id,
        claim_id=base.claim_id,
        interaction_ids=tuple(interaction_ids),
        interaction_count=len(interaction_ids),
        memory_episode_count=base.memory_episode_count,
        silent=base.message_id is None and not interaction_ids,
        content_hash=canonical_hash(identity),
    )


_PRIVATE_ACTION_FIELDS = frozenset(
    {
        "kind",
        "conversation_id",
        "target_agent_ids",
        "target_private_message_id",
        "public_source_message_id",
        "visible_text",
        "stance",
        "emotion",
        "evidence_status",
        "claim_operation",
        "target_private_claim_id",
        "claim_summary",
        "correction_target_private_claim_id",
        "topic_tags",
        "reaction_kind",
    }
)
_PRIVATE_MEMBERSHIP_TURN_TO_EVENT = {
    "invite_member": "invite",
    "accept_invite": "accept",
    "decline_invite": "decline",
    "leave_group": "leave",
    "dissolve_group": "dissolve",
}
_PRIVATE_TURN_ACTIONS = PRIVATE_MESSAGE_ACTIONS | set(
    _PRIVATE_MEMBERSHIP_TURN_TO_EVENT
) | {
    "create_group",
    "private_reaction",
}
_PRIVATE_SHADOW_OUTPUT_FIELDS = frozenset(
    {
        "public_observation",
        "private_actions",
    }
)
_PRIVATE_SHADOW_ACTION_FIELDS = frozenset(
    {
        "kind",
        "target_agent_ids",
        "conversation_id",
        "text",
        "stance",
        "emotion",
        "epistemic_status",
    }
)
_PRIVATE_SHADOW_ACTION_KINDS = frozenset(
    {
        "direct_message",
        "conversation_message",
        "share_public",
        "create_group",
        "accept_invite",
        "decline_invite",
        "leave_group",
    }
)


@dataclass(frozen=True)
class PrivateActionDecision:
    kind: str
    conversation_id: str | None
    target_agent_ids: tuple[str, ...]
    target_private_message_id: str | None
    public_source_message_id: str | None
    visible_text: str
    stance: str
    emotion: str
    evidence_status: str
    claim_operation: str
    target_private_claim_id: str | None
    claim_summary: str
    correction_target_private_claim_id: str | None
    topic_tags: tuple[str, ...]
    reaction_kind: str | None


@dataclass(frozen=True)
class RenderedForumAgentTurnV3:
    base_rendered: RenderedForumAgentTurnV2
    allowed_private_contact_ids: tuple[str, ...]
    allowed_private_conversation_ids: tuple[str, ...]
    visible_private_message_ids: tuple[str, ...]
    visible_private_claim_ids: tuple[str, ...]
    system_prompt: str
    agent_prompt: str
    prompt_sha256: str
    private_actions_enabled: bool = True
    allowed_private_member_conversation_ids: tuple[str, ...] = ()
    allowed_pending_invitation_ids: tuple[str, ...] = ()

    @property
    def agent_id(self) -> str:
        return self.base_rendered.agent_id

    @property
    def source_agent_display_id(self) -> str:
        return self.base_rendered.source_agent_display_id

    @property
    def scenario_id(self) -> str:
        return self.base_rendered.scenario_id

    @property
    def branch(self) -> str:
        return self.base_rendered.branch

    @property
    def tick(self) -> int:
        return self.base_rendered.tick

    @property
    def expected_model(self) -> str:
        return self.base_rendered.expected_model

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_actions

    @property
    def allowed_topics(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_topics

    @property
    def visible_message_ids(self) -> tuple[str, ...]:
        return self.base_rendered.visible_message_ids

    @property
    def visible_claim_ids(self) -> tuple[str, ...]:
        return self.base_rendered.visible_claim_ids


@dataclass(frozen=True)
class ValidatedForumAgentTurnV3:
    rendered: RenderedForumAgentTurnV3
    decision: ForumAgentDecision
    interactions: tuple[ForumInteractionDecision, ...]
    private_actions: tuple[PrivateActionDecision, ...]
    provenance: PublicProvenance


@dataclass(frozen=True)
class MaterializedForumTurnV3:
    action: str
    message_id: str | None
    thread_id: str | None
    claim_id: str | None
    interaction_ids: tuple[str, ...]
    private_artifact_ids: tuple[str, ...]
    private_action_count: int
    memory_episode_count: int
    private_memory_episode_count: int
    silent: bool
    content_hash: str


def _relationship_prompt_context(
    relationship_context: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    if len(relationship_context) > 16:
        raise ForumTurnError("resident relationship context exceeds sixteen closed contacts")
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, raw in enumerate(relationship_context):
        if not isinstance(raw, Mapping):
            raise ForumTurnError(f"relationship_context[{index}] must be an object")
        contact_id = _optional_id(raw.get("contact_agent_id"), "contact_agent_id")
        if contact_id is None or contact_id in ids:
            raise ForumTurnError("relationship context repeats or omits a contact")
        relation_type = str(raw.get("relation_type") or "")
        if relation_type not in {
            "close_friend",
            "familiar_peer",
            "cross_group_acquaintance",
            "co_participant",
        }:
            raise ForumTurnError("relationship context relation type is invalid")
        strength = _bounded_number(raw.get("strength"), "relationship strength", 0.0, 1.0)
        trust = _bounded_number(raw.get("trust"), "relationship trust", 0.0, 1.0)
        shared = str(raw.get("recent_shared_context") or "").strip()
        if len(shared) > 120 or (shared and _UNSAFE_TEXT_RE.search(shared)):
            raise ForumTurnError("relationship shared context is unsafe or too long")
        rows.append(
            {
                "contact_agent_id": contact_id,
                "relation_type": relation_type,
                "strength": strength,
                "trust": trust,
                "recent_shared_context": shared,
            }
        )
        ids.append(contact_id)
    return tuple(rows), tuple(ids)


def _private_channel_choice_brief(
    *,
    agent: Mapping[str, Any],
    posterior_state: Mapping[str, Any],
    relationships: Sequence[Mapping[str, Any]],
    scenario_context: Mapping[str, Any] | None,
    has_public_context: bool,
    has_private_context: bool,
) -> dict[str, Any]:
    """Translate the frozen profile into an explicit, non-coercive channel choice.

    The full Profile-v2 tendency distribution is already part of the resident
    prompt, but a model should not have to infer a private-channel affordance
    from an opaque contact ID.  This compact brief surfaces *why* this resident
    might naturally tell, ask, vent or coordinate with a real synthetic
    contact.  It never requires a private action and never supplies language.
    """

    stable = agent.get("stable_persona")
    stable = stable if isinstance(stable, Mapping) else {}
    raw_tendencies = stable.get("forum_tendency_distribution")
    if not isinstance(raw_tendencies, Mapping):
        # Population-v2 stores the reviewed distribution at the profile root;
        # bridged Hero profiles additionally expose it under stable_persona.
        # Falling back here keeps the private-channel affordance bound to the
        # same reviewed profile instead of silently treating every resident as
        # the generic question/discussion persona.
        raw_tendencies = agent.get("forum_tendencies")
    raw_tendencies = raw_tendencies if isinstance(raw_tendencies, Mapping) else {}

    def tendency(*names: str) -> float:
        values = []
        for name in names:
            value = raw_tendencies.get(name, 0.0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(max(0.0, float(value)))
        return sum(values)

    intent_scores = {
        "ask_or_compare_interpretations": tendency("question", "discussion"),
        "vent_or_gossip_with_trusted_friend": tendency(
            "emotional_vent", "complaint", "emotional_expression", "other"
        ),
        "share_what_was_seen": tendency("information_share"),
        "seek_help_or_coordinate": tendency("request", "help_request"),
    }
    likely_intents = [
        name
        for name, score in sorted(
            intent_scores.items(), key=lambda item: (-item[1], item[0])
        )
        if score > 0
    ][:3]
    if not likely_intents:
        language_style = stable.get("language_style")
        if not isinstance(language_style, Mapping):
            language_style = agent.get("language_style")
        language_style = language_style if isinstance(language_style, Mapping) else {}
        reply_function = str(language_style.get("reply_function") or "")
        likely_intents = [
            {
                "question": "ask_or_compare_interpretations",
                "request": "seek_help_or_coordinate",
                "help_request": "seek_help_or_coordinate",
                "information_share": "share_what_was_seen",
                "emotional_vent": "vent_or_gossip_with_trusted_friend",
                "complaint": "vent_or_gossip_with_trusted_friend",
            }.get(reply_function, "ask_or_compare_interpretations")
        ]

    stake_values: Mapping[str, Any] = {}
    if isinstance(scenario_context, Mapping):
        current_stake = scenario_context.get("current_synthetic_stake")
        if isinstance(current_stake, Mapping) and isinstance(
            current_stake.get("values"), Mapping
        ):
            stake_values = current_stake["values"]

    def numeric_stake(name: str) -> float:
        value = stake_values.get(name, 0.0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return 0.0

    cues: list[str] = []
    if has_public_context:
        cues.append("time_visible_public_discussion_exists")
    if has_private_context:
        cues.append("private_reply_or_invitation_waiting")
    if stake_values.get("directly_affected") is True:
        cues.append("resident_is_directly_affected")
    if numeric_stake("time_urgency") >= 0.55:
        cues.append("time_pressure_is_high")
    if numeric_stake("appeal_cost") >= 0.60:
        cues.append("public_or_formal_appeal_is_costly")
    if numeric_stake("service_channel_dependency") >= 0.45:
        cues.append("coordination_or_service_help_may_be_needed")
    attention = posterior_state.get("attention", 0.0)
    concern = posterior_state.get("concern", 0.0)
    if isinstance(attention, (int, float)) and float(attention) >= 0.55:
        cues.append("attention_is_high")
    if isinstance(concern, (int, float)) and float(concern) >= 0.55:
        cues.append("concern_is_high")

    relation_priority = {
        "close_friend": 0,
        "familiar_peer": 1,
        "cross_group_acquaintance": 2,
        "co_participant": 3,
    }
    recommended = sorted(
        relationships,
        key=lambda row: (
            relation_priority.get(str(row.get("relation_type")), 9),
            -float(row.get("trust", 0.0)),
            -float(row.get("strength", 0.0)),
            str(row.get("contact_agent_id") or ""),
        ),
    )[:4]
    coordination_score = min(
        1.0,
        intent_scores["seek_help_or_coordinate"]
        + 0.35 * intent_scores["share_what_was_seen"]
        + 0.20 * intent_scores["ask_or_compare_interpretations"],
    )
    group_reasons: list[str] = []
    if len(relationships) >= 2:
        group_reasons.append("at_least_two_closed_contacts_available")
    if coordination_score >= 0.18:
        group_reasons.append("profile_supports_multi_person_coordination")
    if stake_values.get("directly_affected") is True:
        group_reasons.append("resident_is_directly_affected")
    if numeric_stake("time_urgency") >= 0.55:
        group_reasons.append("time_sensitive_comparison_or_coordination")
    if numeric_stake("service_channel_dependency") >= 0.45:
        group_reasons.append("shared_service_navigation_may_help")
    group_candidate = bool(
        has_public_context
        and len(relationships) >= 2
        and (
            coordination_score >= 0.18
            or (
                stake_values.get("directly_affected") is True
                and (
                    numeric_stake("time_urgency") >= 0.55
                    or numeric_stake("service_channel_dependency") >= 0.45
                )
            )
        )
    )

    return {
        "derived_from": [
            "stable_profile_forum_tendencies",
            "current_synthetic_stake",
            "posterior_attention_and_concern",
            "closed_synthetic_relationships",
        ],
        "likely_private_intents": likely_intents,
        "current_cues": cues,
        "recommended_first_contact_ids": [
            str(row["contact_agent_id"]) for row in recommended
        ],
        "group_candidacy": {
            "is_candidate": group_candidate,
            "coordination_score": round(coordination_score, 4),
            "reason_codes": group_reasons,
            "candidate_contact_ids": [
                str(row["contact_agent_id"]) for row in recommended[:3]
            ],
            "decision_rule": (
                "When the same concrete question, evidence comparison, service navigation or "
                "coordination need concerns several known contacts, compare one small group "
                "against repeating separate direct messages. Create a group only when it is "
                "plausible for this resident; invitations still require later acceptance."
            ),
        },
        "decision_rule": (
            "Make a separate private-channel decision after the public-action decision. "
            "Use a listed contact only when this resident would naturally tell, ask, vent, "
            "gossip, verify or coordinate with that person now. An empty private_actions array "
            "is valid when none of those motives fits, but it must not be the automatic safe "
            "default merely because public speech was already chosen."
        ),
        "group_emergence_rule": (
            "Ordinary uncertainty or gossip should usually begin one-to-one. Create a group "
            "only when this profile has a concrete coordination need involving at least two "
            "listed contacts; group invitations still require later acceptance."
        ),
    }


def render_forum_agent_turn_v3(
    *,
    agent: Mapping[str, Any],
    posterior_state: Mapping[str, Any],
    memory: AgentMemory,
    ledger: ForumLedger,
    exposures: Sequence[FeedExposure],
    relationship_context: Sequence[Mapping[str, Any]],
    private_conversation_ids: Sequence[str],
    scenario_id: str,
    branch: str,
    tick: int,
    expected_model: str,
    scenario_context: Mapping[str, Any] | None = None,
    public_board: Mapping[str, Any] | None = None,
    current_thread_id: str | None = None,
    direct_parent_message_id: str | None = None,
    quote_message_id: str | None = None,
    allowed_actions: Sequence[str] | None = None,
    require_reviewed_evidence: bool = True,
    forum_context_mode: str = "full_text",
    private_actions_enabled: bool = True,
) -> RenderedForumAgentTurnV3:
    """Render public speech, parallel engagements and up to two private actions."""

    profile_contract = str(agent.get("provider_turn_contract") or "")
    if profile_contract not in {
        FORUM_AGENT_TURN_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    }:
        raise ForumTurnError("agent is not bound to a compatible ForumTwin v3 contract")
    compatible_agent = dict(agent)
    compatible_agent["provider_turn_contract"] = FORUM_AGENT_TURN_V2_SCHEMA_VERSION
    base_rendered = render_forum_agent_turn_v2(
        agent=compatible_agent,
        posterior_state=posterior_state,
        memory=memory,
        ledger=ledger,
        exposures=exposures,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        expected_model=expected_model,
        scenario_context=scenario_context,
        public_board=public_board,
        current_thread_id=current_thread_id,
        direct_parent_message_id=direct_parent_message_id,
        quote_message_id=quote_message_id,
        allowed_actions=allowed_actions,
        require_reviewed_evidence=require_reviewed_evidence,
        forum_context_mode=forum_context_mode,
    )
    relationships, contact_ids = _relationship_prompt_context(relationship_context)
    if not isinstance(private_actions_enabled, bool):
        raise ForumTurnError("private_actions_enabled must be boolean")
    conversation_ids = tuple(
        dict.fromkeys(
            value
            for value in (
                _optional_id(item, "private_conversation_id")
                for item in private_conversation_ids
            )
            if value is not None
        )
    )
    if not private_actions_enabled and conversation_ids:
        raise ForumTurnError("private-chat-disabled turn cannot expose conversations")
    conversation_rows: list[dict[str, Any]] = []
    member_conversation_ids: list[str] = []
    pending_invitation_ids: list[str] = []
    for conversation_id in conversation_ids:
        conversation = ledger.private_conversations.get(conversation_id)
        is_member = bool(
            conversation is not None
            and conversation.status == "active"
            and base_rendered.agent_id in conversation.member_agent_ids
        )
        is_pending_invitee = bool(
            conversation is not None
            and conversation.status == "pending"
            and base_rendered.agent_id in conversation.pending_invite_agent_ids
            and tick <= conversation.invitation_expires_tick
        )
        if (
            conversation is None
            or (not is_member and not is_pending_invitee)
            or conversation.scenario_id != scenario_id
            or conversation.branch not in {branch, "shared_baseline"}
        ):
            raise ForumTurnError("resident private conversation is unavailable")
        if is_member:
            member_conversation_ids.append(conversation_id)
        else:
            pending_invitation_ids.append(conversation_id)
        conversation_rows.append(
            {
                "conversation_id": conversation.conversation_id,
                "conversation_type": conversation.conversation_type,
                "topic_tags": list(conversation.topic_tags),
                "member_agent_ids": list(conversation.member_agent_ids),
                "pending_invite_agent_ids": list(
                    conversation.pending_invite_agent_ids
                ),
                "creator_agent_id": conversation.creator_agent_id,
                "status": conversation.status,
                "created_tick": conversation.created_tick,
                "recipient_access": (
                    "active_member" if is_member else "pending_invitation"
                ),
            }
        )
    private_rows = ledger.build_private_context(
        recipient_agent_id=base_rendered.agent_id,
        current_tick=tick,
        conversation_ids=member_conversation_ids,
    ) if member_conversation_ids else ()
    private_context = tuple(
        {
            "message_id": row["message_id"],
            "conversation_id": row["conversation_id"],
            "source_agent_id": row["source_agent_id"],
            "action": row["action"],
            "visible_text": row["visible_text"],
            "stance": row["stance"],
            "emotion": row["emotion"],
            "evidence_status": row["evidence_status"],
            "created_tick": row["created_tick"],
            "public_source_message_id": row["public_source_message_id"],
            "private_claim_id": row["private_claim_id"],
            "claim_operation": row["claim_operation"],
        }
        for row in private_rows
    )
    public_context_chars = sum(
        len(str(item.get("visible_text") or ""))
        for item in base_rendered.base_rendered.feed_context
    )
    private_context_chars = sum(
        len(str(item.get("visible_text") or "")) for item in private_context
    )
    if private_context_chars > 600 or public_context_chars + private_context_chars > 1_800:
        raise ForumTurnError("resident v3 dynamic context exceeds its 1,800-character limit")
    payload = json.loads(base_rendered.agent_prompt)
    payload["schema"] = FORUM_AGENT_TURN_V3_SCHEMA_VERSION
    payload["agent_contract"]["provider_turn_contract"] = (
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION
    )
    payload["agent_contract"]["profile_contract_source"] = profile_contract
    payload["relationship_context"] = list(relationships)
    payload["private_channel"] = {
        "conversations": conversation_rows,
        "recent_time_visible_messages": list(private_context),
        "channel_choice_brief": _private_channel_choice_brief(
            agent=agent,
            posterior_state=posterior_state,
            relationships=relationships,
            scenario_context=scenario_context,
            has_public_context=bool(base_rendered.base_rendered.feed_context),
            has_private_context=bool(private_context or conversation_rows),
        ),
        "style_and_epistemic_contract": {
            "relationship_specific": (
                "Treat close friends, familiar peers and cross-group contacts differently. "
                "A close-friend chat may be colloquial, emotional, incomplete, joking, "
                "complaining or gossip-oriented when that fits this profile and state."
            ),
            "no_forced_neutrality": (
                "Do not automatically become neutral, repeat a safety slogan or claim to see "
                "the whole picture. The resident may misunderstand, speculate, agree, doubt, "
                "reject a correction or keep asking."
            ),
            "epistemic_labels": (
                "Mark anything learned from another person as hearsay and the resident's own "
                "guess as speculation. Never invent a real identity, official conclusion or "
                "unseen real-world fact."
            ),
            "channel_difference": (
                "Public expression carries forum consequences; private expression may be less "
                "polished. Both remain conditioned on this stable profile, relationship and "
                "current synthetic stake."
            ),
        },
        "limits": {
            "maximum_private_actions": (
                MAX_PRIVATE_ACTIONS_PER_TURN if private_actions_enabled else 0
            ),
            "maximum_recent_private_messages": 4,
            "private_context_characters": 600,
            "total_dynamic_context_characters": 1_800,
        },
    }
    payload["participation_brief"]["channel_independence_rule"] = (
        (
            "The public primary action, public engagements and private_actions are independent. "
            "You may stay publicly silent while chatting privately, or reply publicly and then "
            "continue with a friend. Decide from the explicit channel_choice_brief and this "
            "resident's profile; do not suppress a natural private action merely because a "
            "public action was also selected, and do not chat mechanically when no motive fits."
        )
        if private_actions_enabled
        else (
            "Private chat is disabled in this controlled condition. Return private_actions as "
            "an empty array; public speech and engagements remain available."
        )
    )
    payload["output_contract"]["required_fields"] = sorted(
        OUTPUT_FIELDS
        | {"engagements", "private_actions", "public_private_origin_claim_id"}
    )
    payload["output_contract"]["public_private_origin_claim_id"] = (
        "null unless a time-visible private claim is deliberately retold publicly; then copy "
        "that private claim ID and create a distinct new public claim"
    )
    payload["output_contract"]["private_action_contract"] = {
        "json_type": "array",
        "maximum_items": MAX_PRIVATE_ACTIONS_PER_TURN if private_actions_enabled else 0,
        "item_exact_fields": sorted(_PRIVATE_ACTION_FIELDS),
        "kinds": sorted(_PRIVATE_TURN_ACTIONS),
        "message_rule": (
            "Private language is permitted only for send/reply/forward actions and is limited "
            "to 180 Chinese characters. Other actions use an empty visible_text."
        ),
        "group_rule": (
            "create_group invites 2-7 IDs copied from relationship_context; invite_member, "
            "accept, decline, leave and dissolve must target a listed conversation."
        ),
        "private_claim_rule": (
            "hearsay and speculation are explicit evidence_status values; correction targets "
            "one time-visible private claim."
        ),
        "common_kind_rules": {
            "new_direct_message": (
                "Use kind=send_private, conversation_id=null, exactly one target_agent_id from "
                "relationship_context, no source-message IDs, no topic_tags and reaction_kind=null."
            ),
            "existing_conversation_message": (
                "Use one listed conversation_id and target_agent_ids=[]; use reply_private only "
                "with a time-visible target_private_message_id."
            ),
            "new_group": (
                "Use kind=create_group, conversation_id=null, 2-7 closed contact IDs, 1-4 "
                "topic_tags, empty visible_text, neutral stance/emotion, uncertain evidence, "
                "claim_operation=none and all message/claim/reaction IDs null."
            ),
        },
    }
    validate_prompt_payload(payload, current_tick=tick)
    agent_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = base_rendered.system_prompt + (
        " Return one public primary action, independent public engagements and at most two "
        "private actions. Make the private-channel choice independently instead of defaulting "
        "to an empty array after public speech. Private chat may be informal and relationship-"
        "specific; label hearsay/speculation and never invent unseen facts or identities."
        if private_actions_enabled
        else " Private chat is disabled for this controlled run; private_actions must be []."
    )
    prompt_sha = canonical_hash(
        {
            "renderer": FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
            "expected_model": expected_model,
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
        }
    )
    return RenderedForumAgentTurnV3(
        base_rendered=base_rendered,
        allowed_private_contact_ids=contact_ids,
        allowed_private_conversation_ids=conversation_ids,
        visible_private_message_ids=tuple(
            str(row["message_id"]) for row in private_context
        ),
        visible_private_claim_ids=tuple(
            dict.fromkeys(
                str(row["private_claim_id"])
                for row in private_context
                if row.get("private_claim_id")
            )
        ),
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        prompt_sha256=prompt_sha,
        private_actions_enabled=private_actions_enabled,
        allowed_private_member_conversation_ids=tuple(member_conversation_ids),
        allowed_pending_invitation_ids=tuple(pending_invitation_ids),
    )


def render_forum_private_shadow_turn_v2(
    *,
    private_shadow_focus: str = "organic_private_channel",
    group_candidate: bool = False,
    **kwargs: Any,
) -> RenderedForumAgentTurnV3:
    """Render a low-cost, conditioned private-channel supplement.

    The public forum timeline is supplied as frozen exogenous context.  The
    resident may read or ignore it and may produce up to two private/group
    actions, but cannot create a second public message or engagement.  This is
    useful for enriching an already sealed public Hero trace; it is explicitly
    not a closed-loop estimate of private-chat effects on that frozen trace.
    """

    if private_shadow_focus not in {
        "organic_private_channel",
        "group_lifecycle_observation",
    }:
        raise ForumTurnError("private shadow focus is invalid")
    if not isinstance(group_candidate, bool):
        raise ForumTurnError("private shadow group_candidate must be boolean")
    supplied_actions = kwargs.pop("allowed_actions", None)
    if supplied_actions not in (None, ("read", "ignore"), ["read", "ignore"]):
        raise ForumTurnError("private shadow renderer owns its public action set")
    rendered = render_forum_agent_turn_v3(
        **kwargs,
        allowed_actions=("read", "ignore"),
        private_actions_enabled=True,
    )
    payload = json.loads(rendered.agent_prompt)
    has_created_group = any(
        conversation.conversation_type == "group"
        and conversation.creator_agent_id == rendered.agent_id
        and conversation.scenario_id == rendered.scenario_id
        for conversation in kwargs["ledger"].private_conversations.values()
    )
    group_creation_allowed = bool(
        not has_created_group
        and (
            private_shadow_focus != "group_lifecycle_observation"
            or group_candidate
        )
    )
    pending_invitation_decision_required = bool(
        rendered.allowed_pending_invitation_ids
    )
    available_kinds = set(_PRIVATE_SHADOW_ACTION_KINDS)
    if pending_invitation_decision_required:
        available_kinds = {"accept_invite", "decline_invite"}
    elif not group_creation_allowed:
        available_kinds.discard("create_group")
    private_action_contract = {
        "json_type": "array",
        "minimum_items": 1 if pending_invitation_decision_required else 0,
        "maximum_items": 1
        if pending_invitation_decision_required
        else MAX_PRIVATE_ACTIONS_PER_TURN,
        "item_allowed_fields": sorted(_PRIVATE_SHADOW_ACTION_FIELDS),
        "item_required_fields": ["kind"],
        "kinds": sorted(available_kinds),
        "controlled_values": {
            "stance": sorted(STANCE_VALUES),
            "emotion": sorted(EMOTION_VALUES),
            "epistemic_status": sorted(PRIVATE_EVIDENCE_STATUSES),
        },
        "empty_defaults": {
            "target_agent_ids": [],
            "conversation_id": None,
            "text": "",
            "stance": "neutral",
            "emotion": "neutral",
            "epistemic_status": "uncertain",
        },
        "kind_rules": {
            "direct_message": (
                "Write to one listed relationship contact. Supply one target_agent_id and "
                "natural private text; conversation_id may be omitted."
            ),
            "conversation_message": (
                "Continue one listed active private conversation. Copy its conversation_id "
                "and provide natural private text."
            ),
            "share_public": (
                "Tell one listed contact or active conversation about a time-visible public "
                "post. The runtime binds the currently visible public source; provide text."
            ),
            "create_group": (
                "Invite 2-7 listed contacts for a concrete coordination need. Text is empty; "
                "later acceptance is required before the group becomes active."
            ),
            "accept_invite": "Accept one listed pending conversation invitation.",
            "decline_invite": "Decline one listed pending conversation invitation.",
            "leave_group": "Leave one listed active group conversation.",
        },
        "runtime_mapping_boundary": (
            "The runtime copies only IDs from the closed relationship, conversation and "
            "time-visible message sets, supplies empty non-applicable V3 fields, and then "
            "applies the unchanged V3 visibility, membership, evidence and safety validator."
        ),
        "create_group_eligibility": {
            "allowed_now": group_creation_allowed,
            "reason": (
                "selected_data_driven_candidate_without_an_existing_group"
                if group_creation_allowed
                and private_shadow_focus == "group_lifecycle_observation"
                else "organic_mode_without_an_existing_group"
                if group_creation_allowed
                else "not_selected_or_group_already_created"
            ),
        },
        "pending_invitation_decision": {
            "required_now": pending_invitation_decision_required,
            "allowed_kinds_when_required": ["accept_invite", "decline_invite"],
            "rule": (
                "A resident activated with a pending invitation must independently accept or "
                "decline it in this turn; the runtime never chooses on the resident's behalf."
            ),
        },
    }
    payload["schema"] = FORUM_PRIVATE_SHADOW_TURN_SCHEMA_VERSION
    payload["experiment_contract"] = {
        "mode": "conditioned_private_shadow_replay",
        "private_shadow_focus": private_shadow_focus,
        "group_candidate_selected_from_frozen_profile_and_relationships": group_candidate,
        "group_creation_allowed_now": group_creation_allowed,
        "frozen_public_timeline": True,
        "public_primary_action_must_be": ["read", "ignore"],
        "public_engagements_must_be_empty": True,
        "private_actions_remain_parallel_with_the_frozen_public_observation": True,
        "causal_claim_boundary": (
            "Private outputs may demonstrate public-to-private interpretation, group formation "
            "and correction reach. They do not retroactively change or explain the frozen "
            "public trajectory."
        ),
    }
    payload["participation_brief"]["channel_independence_rule"] = (
        "This is a conditioned private-channel replay. Observe the frozen public board, encode "
        "that observation only as public_observation=read or ignore, and independently "
        "decide whether zero to two private/group actions fit this profile, relationship and "
        "time-visible context. Use channel_choice_brief as concrete motivation: a public post can "
        "naturally be forwarded to a friend, discussed more frankly in private, or used to organise "
        "a small group. An empty private_actions array is valid when no motive fits, but it is not "
        "the automatic safe default after choosing read or ignore."
    )
    if private_shadow_focus == "group_lifecycle_observation":
        group_brief = payload["private_channel"]["channel_choice_brief"][
            "group_candidacy"
        ]
        profile_affordance_candidate = bool(group_brief["is_candidate"])
        group_brief["profile_affordance_candidate"] = profile_affordance_candidate
        group_brief["is_candidate"] = bool(
            group_candidate and profile_affordance_candidate
        )
        payload["private_channel"]["group_lifecycle_observation"] = {
            "candidate": bool(group_brief["is_candidate"]),
            "group_creation_allowed_now": group_creation_allowed,
            "candidate_contact_ids": list(group_brief["candidate_contact_ids"]),
            "instruction": (
                (
                    "This resident is one of the frozen creator candidates. Actively compare "
                    "create_group with duplicating the same direct message when the listed "
                    "multi-person need is present; do not avoid a plausible group merely because "
                    "one-to-one chat is safer."
                    if group_creation_allowed
                    else "This resident is not allowed to create_group now; do not output that "
                    "kind. They may independently accept or decline a listed invitation, speak "
                    "inside an active group, use a permitted direct/private action, or do nothing."
                )
                + " If an invitation is pending, decide accept_invite or decline_invite before "
                "starting an unrelated chat. If an active group exists, use "
                "conversation_message when the current topic still matters. No action is "
                "required when the profile, relationship or current need does not support it."
            ),
        }
    # The Provider decides only the two semantic objects that can affect this
    # conditioned supplement.  Public speech, public engagements, claims and
    # state deltas are frozen, so asking a model to reproduce their 26 empty
    # fields is both costly and a source of avoidable contract failures.
    payload["output_contract"] = {
        "required_fields": sorted(_PRIVATE_SHADOW_OUTPUT_FIELDS),
        "exact_field_set": True,
        "public_observation": {
            "allowed": ["read", "ignore"],
            "rule": (
                "Choose read when at least one time-visible public item was actually "
                "considered; otherwise choose ignore. This never creates public content."
            ),
        },
        "private_actions": private_action_contract,
        "available_target_ids": {
            "private_contacts": list(rendered.allowed_private_contact_ids),
            "private_conversations": list(
                rendered.allowed_private_conversation_ids
            ),
            "private_messages": list(rendered.visible_private_message_ids),
            "private_claims": list(rendered.visible_private_claim_ids),
            "public_messages": list(rendered.visible_message_ids),
        },
        "forbidden_extra_fields": (
            "Do not return action, engagements, public text, explanation, schema, "
            "memory_update or any field other than the two required fields."
        ),
    }
    validate_prompt_payload(payload, current_tick=rendered.tick)
    agent_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = (
        "You are one persistent synthetic resident in a conditioned campus-forum replay. "
        "The public timeline is frozen: you cannot post, reply, react or change a public claim. "
        "Use the supplied stable profile, current stake, relationships, public board and private "
        "inbox to decide whether the resident read the board and whether zero to two private or "
        "small-group actions naturally occur now. Close-friend language may be colloquial, "
        "emotional, incomplete, joking, complaining or gossip-oriented when the profile fits; "
        "do not default to neutral safety slogans. Mark hearsay and speculation, never invent an "
        "unseen fact or real identity, and return exactly one JSON object with only "
        "public_observation and private_actions."
    )
    if private_shadow_focus == "group_lifecycle_observation":
        system_prompt += (
            (
                " This resident is a frozen group-creator candidate. Treat one small group as a "
                "real option when the same concrete issue concerns two or more listed contacts; "
                "do not create it without that motive."
                if group_creation_allowed
                else " This resident cannot create_group in this turn. Never output that kind; "
                "use only the available kinds listed in output_contract.private_actions.kinds."
            )
            + " Invitees must independently accept or decline later, and members may speak only "
            "after activation."
        )
    prompt_sha = canonical_hash(
        {
            "renderer": FORUM_PRIVATE_SHADOW_TURN_SCHEMA_VERSION,
            "parent_renderer": FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
            "expected_model": rendered.expected_model,
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
        }
    )
    return replace(
        rendered,
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        prompt_sha256=prompt_sha,
    )


def _strict_private_shadow_response_object(content: Any) -> dict[str, Any]:
    """Parse the deliberately small Provider contract without accepting extras."""

    if not isinstance(content, str):
        raise ForumTurnError("private shadow response content must be text")
    text = content.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ForumTurnError(
            "private shadow response must contain only one JSON object"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ForumTurnError(
                    "private shadow response contains a duplicate JSON key"
                )
            value[key] = child
        return value

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except ForumTurnError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForumTurnError(
            "private shadow response is not valid JSON"
        ) from exc
    if not isinstance(value, dict) or set(value) != _PRIVATE_SHADOW_OUTPUT_FIELDS:
        raise ForumTurnError(
            "private shadow response fields do not exactly match the compact contract"
        )
    if value["public_observation"] not in {"read", "ignore"}:
        raise ForumTurnError(
            "private shadow public_observation must be read or ignore"
        )
    if not isinstance(value["private_actions"], list):
        raise ForumTurnError("private shadow private_actions must be an array")
    if len(value["private_actions"]) > MAX_PRIVATE_ACTIONS_PER_TURN:
        raise ForumTurnError("private shadow private_actions exceed two items")
    return value


def _normalize_private_shadow_action(
    value: Any,
    *,
    index: int,
    rendered: RenderedForumAgentTurnV3,
    ledger: ForumLedger,
) -> dict[str, Any]:
    """Map one product-level private intent into the exact internal V3 shape.

    The LLM owns channel choice and language. The adapter owns only closed-set
    ID resolution and empty bookkeeping fields; it never invents message text,
    evidence, a relationship, a conversation or a claim.
    """

    if not isinstance(value, Mapping):
        raise ForumTurnError(f"private_actions[{index}] must be an object")
    unknown = set(value) - _PRIVATE_SHADOW_ACTION_FIELDS
    prompt_payload = json.loads(rendered.agent_prompt)
    allowed_compact_kinds = set(
        prompt_payload.get("output_contract", {})
        .get("private_actions", {})
        .get("kinds", ())
    )
    kind_aliases = {
        "send_private": "direct_message",
        "reply_private": "conversation_message",
        "forward_public_to_private": "share_public",
        "group_message": "conversation_message",
    }
    raw_kind = str(value.get("kind") or "").strip()
    kind = kind_aliases.get(raw_kind, raw_kind)
    if (
        unknown
        or kind not in _PRIVATE_SHADOW_ACTION_KINDS
        or kind not in allowed_compact_kinds
    ):
        raise ForumTurnError(
            f"private_actions[{index}] contains unknown fields or an invalid kind"
        )

    raw_targets = value.get("target_agent_ids", [])
    if raw_targets is None:
        raw_targets = []
    if not isinstance(raw_targets, list):
        raise ForumTurnError(
            f"private_actions[{index}].target_agent_ids must be an array"
        )
    contacts = tuple(rendered.allowed_private_contact_ids)
    supplied_targets = tuple(
        str(item).strip()
        for item in raw_targets
        if isinstance(item, str) and str(item).strip() in contacts
    )
    recommended = tuple(
        str(item)
        for item in (
            prompt_payload.get("private_channel", {})
            .get("channel_choice_brief", {})
            .get("recommended_first_contact_ids", [])
        )
        if str(item) in contacts
    )
    contact_order = tuple(dict.fromkeys((*supplied_targets, *recommended, *contacts)))

    raw_conversation = value.get("conversation_id")
    conversation_id = (
        str(raw_conversation).strip()
        if isinstance(raw_conversation, str) and str(raw_conversation).strip()
        else None
    )
    member_conversations = tuple(rendered.allowed_private_member_conversation_ids)
    pending_invitations = tuple(rendered.allowed_pending_invitation_ids)
    if conversation_id not in set(rendered.allowed_private_conversation_ids):
        conversation_id = None
    member_groups = tuple(
        identifier
        for identifier in member_conversations
        if ledger.private_conversations[identifier].conversation_type == "group"
    )

    text_raw = value.get("text", "")
    if text_raw is None:
        text_raw = ""
    if not isinstance(text_raw, str):
        raise ForumTurnError(f"private_actions[{index}].text must be text")
    text = " ".join(text_raw.strip().split())

    controlled_aliases = {
        "stance": {
            "supportive": "support",
            "skeptical": "uncertain",
            "concerned": "uncertain",
            "支持": "support",
            "赞同": "support",
            "反对": "oppose",
            "怀疑": "uncertain",
            "不确定": "uncertain",
            "中立": "neutral",
        },
        "emotion": {
            "worried": "concerned",
            "anxious": "concerned",
            "angry": "frustrated",
            "doubtful": "skeptical",
            "担心": "concerned",
            "生气": "frustrated",
            "好奇": "curious",
            "平静": "calm",
        },
        "epistemic_status": {
            "unknown": "uncertain",
            "unconfirmed": "unverified",
            "rumor": "hearsay",
            "猜测": "speculation",
            "转述": "hearsay",
            "未核实": "unverified",
            "不确定": "uncertain",
        },
    }

    def controlled(field: str, default: str, allowed: frozenset[str]) -> str:
        raw = value.get(field, default)
        text_value = str(raw or default).strip()
        resolved = controlled_aliases.get(field, {}).get(
            text_value.lower(), text_value
        )
        return resolved if resolved in allowed else default

    stance = controlled("stance", "neutral", STANCE_VALUES)
    emotion = controlled("emotion", "neutral", EMOTION_VALUES)
    evidence_status = controlled(
        "epistemic_status", "uncertain", PRIVATE_EVIDENCE_STATUSES
    )
    internal: dict[str, Any] = {
        "kind": "send_private",
        "conversation_id": None,
        "target_agent_ids": [],
        "target_private_message_id": None,
        "public_source_message_id": None,
        "visible_text": text,
        "stance": stance,
        "emotion": emotion,
        "evidence_status": evidence_status,
        "claim_operation": "none",
        "target_private_claim_id": None,
        "claim_summary": "",
        "correction_target_private_claim_id": None,
        "topic_tags": [],
        "reaction_kind": None,
    }
    if kind == "direct_message":
        if not text or not contact_order:
            raise ForumTurnError("direct private intent requires text and a closed contact")
        internal["target_agent_ids"] = [contact_order[0]]
    elif kind in {"conversation_message", "share_public"}:
        selected_conversation = (
            conversation_id if conversation_id in member_conversations else None
        )
        if selected_conversation is None and kind == "conversation_message":
            selected_conversation = (
                member_groups[0]
                if member_groups
                else member_conversations[0]
                if member_conversations
                else None
            )
        if not text:
            raise ForumTurnError("private conversation intent requires text")
        if selected_conversation is not None:
            internal["conversation_id"] = selected_conversation
        elif contact_order:
            internal["target_agent_ids"] = [contact_order[0]]
        else:
            raise ForumTurnError("private conversation intent has no closed recipient")
        if kind == "share_public":
            if not rendered.visible_message_ids:
                raise ForumTurnError("share_public has no time-visible public source")
            internal["kind"] = "forward_public_to_private"
            internal["public_source_message_id"] = rendered.visible_message_ids[0]
    elif kind == "create_group":
        group_targets = list(
            (supplied_targets if len(supplied_targets) >= 2 else contact_order[:2])[:7]
        )
        if len(group_targets) < 2:
            raise ForumTurnError("create_group requires two closed contacts")
        public_turn = rendered.base_rendered.base_rendered
        group_topic = {
            "governance_legitimacy_dispute": "暑期住宿床位分配",
            "lecture_external_incident_shock": "校园讲座事件讨论",
            "century_gym_ghost_booking_dispute": "世纪馆场地预约",
        }.get(rendered.scenario_id, rendered.scenario_id)
        if public_turn.current_thread_id in ledger.threads:
            group_topic = ledger.threads[public_turn.current_thread_id].topic
        else:
            for visible_item in public_turn.feed_context:
                visible_topic = str(visible_item.get("topic") or "").strip()
                if visible_topic:
                    group_topic = visible_topic
                    break
        internal.update(
            {
                "kind": "create_group",
                "target_agent_ids": group_targets,
                "visible_text": "",
                "stance": "neutral",
                "emotion": "neutral",
                "evidence_status": "uncertain",
                "topic_tags": [group_topic],
            }
        )
    elif kind in {"accept_invite", "decline_invite"}:
        selected_invitation = (
            conversation_id if conversation_id in pending_invitations else None
        )
        if selected_invitation is None and pending_invitations:
            selected_invitation = pending_invitations[0]
        if selected_invitation is None:
            raise ForumTurnError(f"{kind} has no pending invitation")
        internal.update(
            {
                "kind": kind,
                "conversation_id": selected_invitation,
                "visible_text": "",
                "stance": "neutral",
                "emotion": "neutral",
                "evidence_status": "uncertain",
            }
        )
    else:
        selected_group = (
            conversation_id if conversation_id in member_groups else None
        )
        if selected_group is None and member_groups:
            selected_group = member_groups[0]
        if selected_group is None:
            raise ForumTurnError("leave_group has no active group")
        internal.update(
            {
                "kind": "leave_group",
                "conversation_id": selected_group,
                "visible_text": "",
                "stance": "neutral",
                "emotion": "neutral",
                "evidence_status": "uncertain",
            }
        )
    return internal


def validate_forum_private_shadow_turn_v3(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumAgentTurnV3,
    ledger: ForumLedger,
    reviewed_trace_release_sha256: str | None = None,
) -> ValidatedForumAgentTurnV3:
    """Validate compact private semantics through the unchanged V3 materializer.

    Provider-authored private actions remain subject to every normal V3 target,
    membership, visibility, evidence and safety check.  Only the public-channel
    no-op fields are filled deterministically because this experiment freezes
    that channel by construction.
    """

    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise ForumTurnError("private shadow prompt fingerprint drifted")
    # The authorized-live repository caches the fully validated V3 decision,
    # not the smaller Provider-facing intent object. Accept that one exact
    # internal shape on recovery so a settled slot is never called again.
    # Every semantic field is still revalidated against the current frozen
    # ledger before it can be integrated.
    try:
        _strict_response_object_v3(envelope.content)
    except ForumTurnError:
        pass
    else:
        cached = validate_forum_agent_turn_v3(
            envelope,
            rendered=rendered,
            ledger=ledger,
            reviewed_trace_release_sha256=reviewed_trace_release_sha256,
        )
        if cached.decision.action not in {"read", "ignore"} or cached.interactions:
            raise ForumTurnError("private shadow cache mutated the public channel")
        return cached
    compact = _strict_private_shadow_response_object(envelope.content)
    if rendered.allowed_pending_invitation_ids:
        actions = compact["private_actions"]
        if (
            len(actions) != 1
            or not isinstance(actions[0], Mapping)
            or actions[0].get("kind") not in {"accept_invite", "decline_invite"}
        ):
            raise ForumTurnError(
                "pending private invitation requires one accept or decline decision"
            )
    allowed_topics = tuple(rendered.allowed_topics)
    if not allowed_topics:
        raise ForumTurnError("private shadow resident has no allowed topic")
    expanded = {
        "action": compact["public_observation"],
        "target_thread_id": None,
        "target_message_id": None,
        "quote_message_id": None,
        "target_claim_id": None,
        "topic": allowed_topics[0],
        "subtopic": "私聊观察",
        "stance": "uncertain",
        "urgency": 0.0,
        "emotion": "neutral",
        "confidence": 0.5,
        "evidence_status": "uncertain",
        "claim_operation": "none",
        "claim_summary": "",
        "correction_target_claim_id": None,
        "trust_delta": 0.0,
        "satisfaction_delta": 0.0,
        "memory_update": "",
        "visible_text": "",
        "engagements": [],
        "private_actions": [
            _normalize_private_shadow_action(
                value,
                index=index,
                rendered=rendered,
                ledger=ledger,
            )
            for index, value in enumerate(compact["private_actions"])
        ],
        "public_private_origin_claim_id": None,
    }
    expanded_envelope = replace(
        envelope,
        content=json.dumps(
            expanded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    validated = validate_forum_agent_turn_v3(
        expanded_envelope,
        rendered=rendered,
        ledger=ledger,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
    )
    if validated.decision.action not in {"read", "ignore"} or validated.interactions:
        raise ForumTurnError("private shadow compact expansion mutated the public channel")
    return validated


def render_forum_private_shadow_turn_v1(
    **kwargs: Any,
) -> RenderedForumAgentTurnV3:
    """Compatibility alias for callers written before the v2 prompt release."""

    return render_forum_private_shadow_turn_v2(**kwargs)


def _strict_response_object_v3(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ForumTurnError("resident V3 response content must be text")
    text = content.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ForumTurnError("resident V3 response must contain only one JSON object")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ForumTurnError("resident V3 response contains a duplicate JSON key")
            value[key] = child
        return value

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except ForumTurnError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForumTurnError("resident V3 response is not valid JSON") from exc
    expected = OUTPUT_FIELDS | {
        "engagements",
        "private_actions",
        "public_private_origin_claim_id",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ForumTurnError("resident V3 response fields do not exactly match the contract")
    return value


def _empty_private_fields(raw: Mapping[str, Any], *, allow_topics: bool = False) -> None:
    empty_fields = (
        "target_private_message_id",
        "public_source_message_id",
        "visible_text",
        "target_private_claim_id",
        "claim_summary",
        "correction_target_private_claim_id",
        "reaction_kind",
    )
    if any(raw[field] not in {None, ""} for field in empty_fields):
        raise ForumTurnError("non-message private action carries message-only fields")
    if raw["claim_operation"] != "none":
        raise ForumTurnError("non-message private action cannot mutate a claim")
    if not allow_topics and raw["topic_tags"] != []:
        raise ForumTurnError("private action carries unexpected topic tags")


def validate_forum_agent_turn_v3(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumAgentTurnV3,
    ledger: ForumLedger,
    reviewed_trace_release_sha256: str | None = None,
) -> ValidatedForumAgentTurnV3:
    """Validate one public move, six engagements and at most two private actions."""

    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise ForumTurnError("forum resident V3 prompt fingerprint drifted")
    raw = _strict_response_object_v3(envelope.content)
    private_rows = raw.pop("private_actions")
    private_origin = _optional_id(
        raw.pop("public_private_origin_claim_id"),
        "public_private_origin_claim_id",
    )
    base_envelope = replace(
        envelope,
        prompt_sha256=rendered.base_rendered.prompt_sha256,
        content=json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    base = validate_forum_agent_turn_v2(
        base_envelope,
        rendered=rendered.base_rendered,
        ledger=ledger,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
    )
    provenance = replace(base.provenance, prompt_sha256=rendered.prompt_sha256)
    if private_origin is not None:
        if private_origin not in rendered.visible_private_claim_ids:
            raise ForumTurnError("public retelling references a non-visible private claim")
        if base.decision.claim_operation != "create" or base.decision.action in {"ignore", "read"}:
            raise ForumTurnError("private-to-public retelling must create a new public claim")
    decision = replace(base.decision, private_origin_claim_id=private_origin)
    private_limit = MAX_PRIVATE_ACTIONS_PER_TURN if rendered.private_actions_enabled else 0
    if not isinstance(private_rows, list) or len(private_rows) > private_limit:
        raise ForumTurnError("private_actions must contain at most two items")
    contacts = set(rendered.allowed_private_contact_ids)
    member_conversations = set(rendered.allowed_private_member_conversation_ids)
    pending_invitations = set(rendered.allowed_pending_invitation_ids)
    visible_private_messages = set(rendered.visible_private_message_ids)
    visible_private_claims = set(rendered.visible_private_claim_ids)
    visible_public_messages = set(rendered.base_rendered.visible_message_ids)
    private_actions: list[PrivateActionDecision] = []
    for index, value in enumerate(private_rows):
        if not isinstance(value, dict) or set(value) != _PRIVATE_ACTION_FIELDS:
            raise ForumTurnError(f"private_actions[{index}] does not match its exact contract")
        kind = str(value["kind"])
        if kind not in _PRIVATE_TURN_ACTIONS:
            raise ForumTurnError(f"private_actions[{index}].kind is invalid")
        conversation_id = _optional_id(
            value["conversation_id"], f"private_actions[{index}].conversation_id"
        )
        raw_targets = value["target_agent_ids"]
        if not isinstance(raw_targets, list):
            raise ForumTurnError(f"private_actions[{index}].target_agent_ids must be an array")
        target_ids = tuple(
            _optional_id(item, f"private_actions[{index}].target_agent_ids")
            for item in raw_targets
        )
        if any(item is None for item in target_ids) or len(set(target_ids)) != len(target_ids):
            raise ForumTurnError("private action target IDs are null or repeated")
        target_ids = tuple(str(item) for item in target_ids)
        if not set(target_ids).issubset(contacts):
            raise ForumTurnError("private action targets an agent outside the closed contact set")
        target_private_message_id = _optional_id(
            value["target_private_message_id"],
            f"private_actions[{index}].target_private_message_id",
        )
        public_source_message_id = _optional_id(
            value["public_source_message_id"],
            f"private_actions[{index}].public_source_message_id",
        )
        target_private_claim_id = _optional_id(
            value["target_private_claim_id"],
            f"private_actions[{index}].target_private_claim_id",
        )
        correction_target = _optional_id(
            value["correction_target_private_claim_id"],
            f"private_actions[{index}].correction_target_private_claim_id",
        )
        if not isinstance(value["topic_tags"], list):
            raise ForumTurnError("private action topic_tags must be an array")
        topic_tags = tuple(
            _nonempty(item, "private topic tag", maximum=60)
            for item in value["topic_tags"]
        )
        if len(set(topic_tags)) != len(topic_tags) or len(topic_tags) > 4:
            raise ForumTurnError("private action topic tags repeat or exceed four")
        visible_text_raw = value["visible_text"]
        if not isinstance(visible_text_raw, str):
            raise ForumTurnError("private visible_text must be text")
        visible_text = " ".join(visible_text_raw.strip().split())
        if visible_text and (len(visible_text) > 180 or _UNSAFE_TEXT_RE.search(visible_text)):
            raise ForumTurnError("private visible_text is unsafe or exceeds 180 characters")
        stance = str(value["stance"])
        emotion = str(value["emotion"])
        evidence_status = str(value["evidence_status"])
        claim_operation = str(value["claim_operation"])
        claim_summary_raw = value["claim_summary"]
        if not isinstance(claim_summary_raw, str):
            raise ForumTurnError("private claim_summary must be text")
        claim_summary = " ".join(claim_summary_raw.strip().split())
        reaction_kind = value["reaction_kind"]
        message_action = kind in PRIVATE_MESSAGE_ACTIONS
        if message_action:
            if not visible_text:
                raise ForumTurnError("private message action requires visible_text")
            if stance not in STANCE_VALUES or emotion not in EMOTION_VALUES:
                raise ForumTurnError("private message stance or emotion is invalid")
            if evidence_status not in PRIVATE_EVIDENCE_STATUSES:
                raise ForumTurnError("private message evidence status is invalid")
            if claim_operation not in CLAIM_OPERATIONS:
                raise ForumTurnError("private message claim operation is invalid")
            if conversation_id is None:
                if kind not in {"send_private", "forward_public_to_private"} or len(target_ids) != 1:
                    raise ForumTurnError(
                        "new direct chat requires send/private-forward to one contact"
                    )
            elif conversation_id not in member_conversations or target_ids:
                raise ForumTurnError("existing private conversation cannot carry target_agent_ids")
            if kind == "reply_private":
                if target_private_message_id not in visible_private_messages:
                    raise ForumTurnError("private reply target was not time-visible")
            elif kind == "forward_public_to_private":
                if public_source_message_id not in visible_public_messages:
                    raise ForumTurnError("public-to-private source was not time-visible")
            elif kind == "forward_private_to_private":
                if target_private_message_id not in visible_private_messages:
                    raise ForumTurnError("private forward source was not time-visible")
            elif target_private_message_id is not None or public_source_message_id is not None:
                raise ForumTurnError("send_private carries an unexpected source message")
            if claim_operation == "none":
                if target_private_claim_id is not None or correction_target is not None or claim_summary:
                    raise ForumTurnError("private claim_operation none carries claim fields")
            elif claim_operation == "create":
                if target_private_claim_id is not None or correction_target is not None:
                    raise ForumTurnError("new private claim cannot target another private claim")
                claim_summary = _nonempty(
                    claim_summary, "private claim_summary", maximum=160
                )
            else:
                if target_private_claim_id not in visible_private_claims:
                    raise ForumTurnError("private claim target was not time-visible")
                claim_summary = ""
                if claim_operation == "correct":
                    if correction_target != target_private_claim_id:
                        raise ForumTurnError("private correction target must match its claim")
                    if evidence_status not in {"corrected", "verified"}:
                        raise ForumTurnError("private correction requires corrected evidence")
                elif correction_target is not None:
                    raise ForumTurnError("only private correction may name a correction target")
            if reaction_kind is not None:
                raise ForumTurnError("private message action cannot carry reaction_kind")
            if topic_tags:
                raise ForumTurnError("private message action cannot redefine conversation topics")
        elif kind == "create_group":
            if conversation_id is not None or not 2 <= len(target_ids) <= 7 or not 1 <= len(topic_tags) <= 4:
                raise ForumTurnError("create_group requires 2-7 contacts and 1-4 topic tags")
            _empty_private_fields(value, allow_topics=True)
            stance = emotion = "neutral"
            evidence_status = "uncertain"
            claim_operation = "none"
        elif kind == "invite_member":
            if conversation_id not in member_conversations or len(target_ids) != 1:
                raise ForumTurnError("invite requires one contact and one visible conversation")
            _empty_private_fields(value)
            stance = emotion = "neutral"
            evidence_status = "uncertain"
            claim_operation = "none"
        elif kind in {"accept_invite", "decline_invite"}:
            if conversation_id not in pending_invitations or target_ids:
                raise ForumTurnError(f"{kind} requires one pending invitation and no targets")
            _empty_private_fields(value)
            stance = emotion = "neutral"
            evidence_status = "uncertain"
            claim_operation = "none"
        elif kind in {"leave_group", "dissolve_group"}:
            if conversation_id not in member_conversations or target_ids:
                raise ForumTurnError(f"{kind} requires one listed conversation and no targets")
            _empty_private_fields(value)
            stance = emotion = "neutral"
            evidence_status = "uncertain"
            claim_operation = "none"
        else:
            if (
                kind != "private_reaction"
                or conversation_id not in member_conversations
                or target_ids
                or target_private_message_id not in visible_private_messages
                or reaction_kind not in PRIVATE_REACTION_KINDS
            ):
                raise ForumTurnError("private reaction contract is invalid")
            if any(
                value[field] not in {None, ""}
                for field in (
                    "public_source_message_id",
                    "visible_text",
                    "target_private_claim_id",
                    "claim_summary",
                    "correction_target_private_claim_id",
                )
            ) or value["topic_tags"] != [] or claim_operation != "none":
                raise ForumTurnError("private reaction carries message-only fields")
            stance = emotion = "neutral"
            evidence_status = "uncertain"
            claim_operation = "none"
        private_actions.append(
            PrivateActionDecision(
                kind=kind,
                conversation_id=conversation_id,
                target_agent_ids=target_ids,
                target_private_message_id=target_private_message_id,
                public_source_message_id=public_source_message_id,
                visible_text=visible_text,
                stance=stance,
                emotion=emotion,
                evidence_status=evidence_status,
                claim_operation=claim_operation,
                target_private_claim_id=target_private_claim_id,
                claim_summary=claim_summary,
                correction_target_private_claim_id=correction_target,
                topic_tags=topic_tags,
                reaction_kind=(str(reaction_kind) if reaction_kind is not None else None),
            )
        )
    return ValidatedForumAgentTurnV3(
        rendered=rendered,
        decision=decision,
        interactions=base.interactions,
        private_actions=tuple(private_actions),
        provenance=provenance,
    )


def _private_conversation_for_action(
    *,
    action: PrivateActionDecision,
    validated: ValidatedForumAgentTurnV3,
    ledger: ForumLedger,
) -> str:
    if action.conversation_id is not None:
        return action.conversation_id
    contact_id = action.target_agent_ids[0]
    existing = sorted(
        conversation.conversation_id
        for conversation in ledger.private_conversations.values()
        if conversation.conversation_type == "direct"
        and conversation.status == "active"
        and set(conversation.member_agent_ids)
        == {validated.rendered.agent_id, contact_id}
        and conversation.scenario_id == validated.rendered.scenario_id
        and conversation.branch in {validated.rendered.branch, "shared_baseline"}
    )
    if existing:
        return existing[0]
    conversation_id = "private-conversation-" + canonical_hash(
        [
            validated.rendered.scenario_id,
            validated.rendered.branch,
            "direct",
            *sorted((validated.rendered.agent_id, contact_id)),
        ]
    )[:24]
    ledger.add_private_conversation(
        PrivateConversation(
            conversation_id=conversation_id,
            scenario_id=validated.rendered.scenario_id,
            branch=validated.rendered.branch,
            conversation_type="direct",
            creator_agent_id=validated.rendered.agent_id,
            member_agent_ids=(validated.rendered.agent_id, contact_id),
            pending_invite_agent_ids=(),
            left_agent_ids=(),
            created_tick=validated.rendered.tick,
            status="active",
            invitation_expires_tick=min(26, validated.rendered.tick + 3),
            topic_tags=(validated.decision.topic,),
            provenance=validated.provenance,
        ),
        allowed_contact_ids=validated.rendered.allowed_private_contact_ids,
    )
    return conversation_id


def materialize_forum_agent_turn_v3(
    *,
    validated: ValidatedForumAgentTurnV3,
    ledger: ForumLedger,
    memory: AgentMemory,
) -> MaterializedForumTurnV3:
    """Commit one validated multi-channel turn after all channel checks pass."""

    base = materialize_forum_agent_turn_v2(
        validated=ValidatedForumAgentTurnV2(
            rendered=validated.rendered.base_rendered,
            decision=validated.decision,
            interactions=validated.interactions,
            provenance=validated.provenance,
        ),
        ledger=ledger,
        memory=memory,
    )
    artifact_ids: list[str] = []
    for ordinal, action in enumerate(validated.private_actions):
        action_identity = [
            validated.rendered.scenario_id,
            validated.rendered.branch,
            validated.rendered.tick,
            validated.rendered.agent_id,
            ordinal,
            action.kind,
            validated.provenance.response_sha256,
        ]
        if action.kind == "create_group":
            conversation_id = "private-group-" + canonical_hash(action_identity)[:24]
            ledger.add_private_conversation(
                PrivateConversation(
                    conversation_id=conversation_id,
                    scenario_id=validated.rendered.scenario_id,
                    branch=validated.rendered.branch,
                    conversation_type="group",
                    creator_agent_id=validated.rendered.agent_id,
                    member_agent_ids=(validated.rendered.agent_id,),
                    pending_invite_agent_ids=action.target_agent_ids,
                    left_agent_ids=(),
                    created_tick=validated.rendered.tick,
                    status="pending",
                    invitation_expires_tick=min(26, validated.rendered.tick + 3),
                    topic_tags=action.topic_tags,
                    provenance=validated.provenance,
                ),
                allowed_contact_ids=validated.rendered.allowed_private_contact_ids,
            )
            artifact_ids.append(conversation_id)
            memory.pending_private_invitation_ids.update(
                f"{conversation_id}:{agent_id}" for agent_id in action.target_agent_ids
            )
            continue
        if action.kind in _PRIVATE_MEMBERSHIP_TURN_TO_EVENT:
            conversation_id = str(action.conversation_id)
            subjects = (
                action.target_agent_ids
                if action.kind == "invite_member"
                else (validated.rendered.agent_id,)
            )
            for subject in subjects:
                event_id = "private-membership-" + canonical_hash(
                    action_identity + [subject]
                )[:24]
                ledger.apply_private_membership_event(
                    PrivateMembershipEvent(
                        event_id=event_id,
                        conversation_id=conversation_id,
                        actor_agent_id=validated.rendered.agent_id,
                        subject_agent_id=subject,
                        action=_PRIVATE_MEMBERSHIP_TURN_TO_EVENT[action.kind],
                        created_tick=validated.rendered.tick,
                        provenance=validated.provenance,
                    ),
                    allowed_contact_ids=validated.rendered.allowed_private_contact_ids,
                )
                artifact_ids.append(event_id)
            continue
        if action.kind == "private_reaction":
            reaction_id = "private-reaction-" + canonical_hash(action_identity)[:24]
            ledger.add_private_reaction(
                PrivateReaction(
                    reaction_id=reaction_id,
                    conversation_id=str(action.conversation_id),
                    target_message_id=str(action.target_private_message_id),
                    source_agent_id=validated.rendered.agent_id,
                    kind=str(action.reaction_kind),
                    created_tick=validated.rendered.tick,
                    provenance=validated.provenance,
                )
            )
            artifact_ids.append(reaction_id)
            memory.append_private_episode(
                PrivateLlmMemoryEpisode(
                    tick=validated.rendered.tick,
                    conversation_id=str(action.conversation_id),
                    relation_type="group",
                    action="private_reaction",
                    semantic_summary="对一条时间可见的合成私聊作出反应",
                    message_id=None,
                    provenance=validated.provenance,
                )
            )
            continue
        conversation_id = _private_conversation_for_action(
            action=action,
            validated=validated,
            ledger=ledger,
        )
        message_id = "private-message-" + canonical_hash(action_identity)[:24]
        claim_id = action.target_private_claim_id
        if action.claim_operation == "create":
            claim_id = "private-claim-" + canonical_hash(
                [conversation_id, message_id, action.claim_summary]
            )[:24]
        private_message = PrivateMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            source_agent_id=validated.rendered.agent_id,
            action=action.kind,
            visible_text=action.visible_text,
            stance=action.stance,
            emotion=action.emotion,
            evidence_status=action.evidence_status,
            created_tick=validated.rendered.tick,
            provenance=validated.provenance,
            parent_message_id=(
                action.target_private_message_id
                if action.kind == "reply_private"
                else None
            ),
            public_source_message_id=action.public_source_message_id,
            forwarded_private_source_message_id=(
                action.target_private_message_id
                if action.kind == "forward_private_to_private"
                else None
            ),
            private_claim_id=claim_id,
            claim_operation=action.claim_operation,
            correction_target_claim_id=action.correction_target_private_claim_id,
        )
        ledger.add_private_message(private_message)
        if action.claim_operation == "create":
            origin_kind = (
                "public"
                if action.public_source_message_id is not None
                else "private"
            )
            ledger.add_private_claim(
                PrivateClaim(
                    claim_id=str(claim_id),
                    conversation_id=conversation_id,
                    claim_summary=action.claim_summary,
                    status=(
                        action.evidence_status
                        if action.evidence_status in PRIVATE_CLAIM_STATUSES
                        else "verified"
                        if action.evidence_status == "verified"
                        else "hearsay"
                    ),
                    first_seen_tick=validated.rendered.tick,
                    last_seen_tick=validated.rendered.tick,
                    created_by_message_id=message_id,
                    origin_kind=origin_kind,
                    origin_public_message_id=action.public_source_message_id,
                    origin_private_message_id=(
                        None if origin_kind == "public" else message_id
                    ),
                    supporting_message_ids=(message_id,),
                )
            )
        elif action.claim_operation != "none":
            current = ledger.private_claims[str(claim_id)]
            support = list(current.supporting_message_ids)
            challenge = list(current.challenging_message_ids)
            if action.claim_operation in {"support", "verify", "correct"}:
                support.append(message_id)
            if action.claim_operation in {"challenge", "correct"}:
                challenge.append(message_id)
            status = {
                "support": current.status,
                "challenge": "contested",
                "verify": "verified",
                "correct": "corrected",
            }[action.claim_operation]
            ledger.advance_private_claim(
                replace(
                    current,
                    status=status,
                    last_seen_tick=validated.rendered.tick,
                    supporting_message_ids=tuple(dict.fromkeys(support)),
                    challenging_message_ids=tuple(dict.fromkeys(challenge)),
                )
            )
        artifact_ids.append(message_id)
        conversation = ledger.private_conversations[conversation_id]
        relation_type = (
            "group" if conversation.conversation_type == "group" else "close_friend"
        )
        memory.append_private_episode(
            PrivateLlmMemoryEpisode(
                tick=validated.rendered.tick,
                conversation_id=conversation_id,
                relation_type=relation_type,
                action=action.kind,
                semantic_summary=(
                    action.claim_summary
                    or f"在合成私聊中执行 {action.kind}，证据状态为 {action.evidence_status}"
                ),
                message_id=message_id,
                provenance=validated.provenance,
            )
        )
    identity = {
        "schema": FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        "prompt_sha256": validated.rendered.prompt_sha256,
        "response_sha256": validated.provenance.response_sha256,
        "agent_id": validated.rendered.agent_id,
        "base_content_hash": base.content_hash,
        "private_artifact_ids": artifact_ids,
        "private_memory_episode_count": len(memory.recent_private_llm_episodes),
    }
    return MaterializedForumTurnV3(
        action=base.action,
        message_id=base.message_id,
        thread_id=base.thread_id,
        claim_id=base.claim_id,
        interaction_ids=base.interaction_ids,
        private_artifact_ids=tuple(artifact_ids),
        private_action_count=len(validated.private_actions),
        memory_episode_count=base.memory_episode_count,
        private_memory_episode_count=len(memory.recent_private_llm_episodes),
        silent=(
            base.message_id is None
            and not base.interaction_ids
            and not artifact_ids
        ),
        content_hash=canonical_hash(identity),
    )


@dataclass(frozen=True)
class RenderedForumAgentTurnV4:
    """Resident turn with 小喇叭-style anonymity on public speech."""

    base_rendered: RenderedForumAgentTurnV3
    system_prompt: str
    agent_prompt: str
    prompt_sha256: str
    public_anonymity_enabled: bool = True

    @property
    def agent_id(self) -> str:
        return self.base_rendered.agent_id

    @property
    def source_agent_display_id(self) -> str:
        return self.base_rendered.source_agent_display_id

    @property
    def scenario_id(self) -> str:
        return self.base_rendered.scenario_id

    @property
    def branch(self) -> str:
        return self.base_rendered.branch

    @property
    def tick(self) -> int:
        return self.base_rendered.tick

    @property
    def expected_model(self) -> str:
        return self.base_rendered.expected_model

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_actions

    @property
    def allowed_topics(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_topics


@dataclass(frozen=True)
class ValidatedForumAgentTurnV4:
    rendered: RenderedForumAgentTurnV4
    base_validated: ValidatedForumAgentTurnV3
    public_identity_mode: str
    provenance: PublicProvenance


@dataclass(frozen=True)
class MaterializedForumTurnV4:
    action: str
    message_id: str | None
    thread_id: str | None
    claim_id: str | None
    interaction_ids: tuple[str, ...]
    private_artifact_ids: tuple[str, ...]
    public_identity_mode: str
    public_pseudonym_id: str | None
    private_action_count: int
    memory_episode_count: int
    private_memory_episode_count: int
    silent: bool
    content_hash: str


def render_forum_agent_turn_v4(
    *,
    public_anonymity_enabled: bool = True,
    **kwargs: Any,
) -> RenderedForumAgentTurnV4:
    """Render one normal forum turn whose public author face is selectable."""

    if not isinstance(public_anonymity_enabled, bool):
        raise ForumTurnError("public_anonymity_enabled must be boolean")
    supplied_agent = kwargs.get("agent")
    if not isinstance(supplied_agent, Mapping):
        raise ForumTurnError("resident v4 requires one Agent mapping")
    profile_contract = str(supplied_agent.get("provider_turn_contract") or "")
    if profile_contract not in {
        FORUM_AGENT_TURN_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V2_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    }:
        raise ForumTurnError("agent is not bound to a compatible ForumTwin v4 contract")
    compatible_agent = dict(supplied_agent)
    compatible_agent["provider_turn_contract"] = FORUM_AGENT_TURN_V3_SCHEMA_VERSION
    kwargs["agent"] = compatible_agent
    rendered = render_forum_agent_turn_v3(**kwargs)
    payload = json.loads(rendered.agent_prompt)
    payload["schema"] = FORUM_AGENT_TURN_V4_SCHEMA_VERSION
    payload["agent_contract"]["provider_turn_contract"] = (
        FORUM_AGENT_TURN_V4_SCHEMA_VERSION
    )
    payload["agent_contract"]["profile_contract_source"] = profile_contract
    payload["public_anonymity"] = {
        "enabled": public_anonymity_enabled,
        "platform_rule": (
            "This campus forum allows an ordinary public post or comment to be shown under "
            "a synthetic anonymous pseudonym. It is not a separate treehole channel and uses "
            "the same board, ranking, reply, quote, claim and moderation rules."
        ),
        "decision_guidance": (
            "Choose anonymous when this resident wants to say something sensitive, severe or "
            "vulnerable, worries acquaintances may recognise them, or would speak more candidly "
            "without a visible profile. Choose named when accountability, reputation or an "
            "ongoing public identity matters. Follow the profile and current relationship cost; "
            "do not make everyone anonymous."
        ),
        "epistemic_boundary": (
            "Anonymity changes author visibility, not truth. Mark hearsay/speculation and never "
            "invent a real identity, official conclusion or unseen fact."
        ),
        "governance_visibility": (
            "Governance sees the same public text and aggregates but not the hidden Agent id."
        ),
    }
    required_fields = set(payload["output_contract"]["required_fields"])
    required_fields.add("public_identity_mode")
    payload["output_contract"]["required_fields"] = sorted(required_fields)
    payload["output_contract"]["public_identity_mode"] = {
        "allowed": ["named", "anonymous"] if public_anonymity_enabled else ["named"],
        "rule": "ignore/read must be named; a public speech action may choose either mode",
    }
    validate_prompt_payload(payload, current_tick=rendered.tick)
    agent_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = rendered.system_prompt + (
        " For the ordinary public speech action, choose public_identity_mode named or anonymous. "
        "Use anonymity when the profile plausibly fears recognition or is expressing something "
        "sensitive; anonymity never permits invented facts or identities."
        if public_anonymity_enabled
        else " Public anonymity is disabled; public_identity_mode must be named."
    )
    prompt_sha = canonical_hash(
        {
            "renderer": FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
            "parent_renderer": FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
            "expected_model": rendered.expected_model,
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
        }
    )
    return RenderedForumAgentTurnV4(
        base_rendered=rendered,
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        prompt_sha256=prompt_sha,
        public_anonymity_enabled=public_anonymity_enabled,
    )


def _strict_response_object_v4(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ForumTurnError("resident V4 response content must be text")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ForumTurnError("resident V4 response contains a duplicate JSON key")
            value[key] = child
        return value

    text = content.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ForumTurnError("resident V4 response must contain only one JSON object")
    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except ForumTurnError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForumTurnError("resident V4 response is not valid JSON") from exc
    expected = OUTPUT_FIELDS | {
        "engagements",
        "private_actions",
        "public_private_origin_claim_id",
        "public_identity_mode",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ForumTurnError("resident V4 response fields do not exactly match the contract")
    return value


def validate_forum_agent_turn_v4(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumAgentTurnV4,
    ledger: ForumLedger,
    reviewed_trace_release_sha256: str | None = None,
) -> ValidatedForumAgentTurnV4:
    """Validate an exact v4 object without weakening the v3 contract."""

    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise ForumTurnError("forum resident V4 prompt fingerprint drifted")
    raw = _strict_response_object_v4(envelope.content)
    public_identity_mode = str(raw.pop("public_identity_mode"))
    if public_identity_mode not in {"named", "anonymous"}:
        raise ForumTurnError("public_identity_mode must be named or anonymous")
    if not rendered.public_anonymity_enabled and public_identity_mode != "named":
        raise ForumTurnError("public anonymity is disabled for this run")
    base = validate_forum_agent_turn_v3(
        replace(
            envelope,
            prompt_sha256=rendered.base_rendered.prompt_sha256,
            content=json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        ),
        rendered=rendered.base_rendered,
        ledger=ledger,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
    )
    provenance = replace(base.provenance, prompt_sha256=rendered.prompt_sha256)
    if base.decision.action in {"ignore", "read"} and public_identity_mode != "named":
        raise ForumTurnError("silent/read action cannot claim an anonymous author face")
    return ValidatedForumAgentTurnV4(
        rendered=rendered,
        base_validated=replace(base, provenance=provenance),
        public_identity_mode=public_identity_mode,
        provenance=provenance,
    )


def materialize_forum_agent_turn_v4(
    *,
    validated: ValidatedForumAgentTurnV4,
    ledger: ForumLedger,
    memory: AgentMemory,
) -> MaterializedForumTurnV4:
    """Commit one v3 turn, then replace only its public author projection."""

    base = materialize_forum_agent_turn_v3(
        validated=validated.base_validated,
        ledger=ledger,
        memory=memory,
    )
    pseudonym_id: str | None = None
    if base.message_id is not None:
        message = ledger.messages[base.message_id]
        if validated.public_identity_mode == "anonymous":
            pseudonym_id = "ANON-" + canonical_hash(
                [
                    validated.rendered.scenario_id,
                    validated.rendered.branch,
                    message.thread_id,
                    validated.rendered.agent_id,
                    "forum-anonymous-face-v1",
                ]
            )[:8].upper()
        ledger.messages[base.message_id] = ForumMessageV2(
            **message.__dict__,
            author_visibility=validated.public_identity_mode,
            anonymous_pseudonym_id=pseudonym_id,
        )
    elif validated.public_identity_mode != "named":
        raise ForumTurnError("silent materialization cannot be anonymous")
    identity = {
        "schema": FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        "prompt_sha256": validated.rendered.prompt_sha256,
        "response_sha256": validated.provenance.response_sha256,
        "agent_id": validated.rendered.agent_id,
        "base_content_hash": base.content_hash,
        "public_identity_mode": validated.public_identity_mode,
        "public_pseudonym_id": pseudonym_id,
    }
    return MaterializedForumTurnV4(
        action=base.action,
        message_id=base.message_id,
        thread_id=base.thread_id,
        claim_id=base.claim_id,
        interaction_ids=base.interaction_ids,
        private_artifact_ids=base.private_artifact_ids,
        public_identity_mode=validated.public_identity_mode,
        public_pseudonym_id=pseudonym_id,
        private_action_count=base.private_action_count,
        memory_episode_count=base.memory_episode_count,
        private_memory_episode_count=base.private_memory_episode_count,
        silent=base.silent,
        content_hash=canonical_hash(identity),
    )


@dataclass(frozen=True)
class RenderedForumAgentTurnV5:
    """V4 mechanics exposed through a human-readable, alias-only model view.

    Internal release hashes, evidence fingerprints, raw particle means and raw
    synthetic-stake numbers stay in the runtime object.  They are deliberately
    absent from the Provider-facing prompt.  Closed-set forum identifiers are
    represented by short per-turn handles and translated back before V4
    validation, so the existing ledger and recovery contracts remain usable.
    """

    base_rendered: RenderedForumAgentTurnV4
    system_prompt: str
    agent_prompt: str
    prompt_sha256: str
    alias_to_internal: Mapping[str, str]
    model_view_condition: str = V5_FULL_NATURAL_PROFILE

    @property
    def agent_id(self) -> str:
        return self.base_rendered.agent_id

    @property
    def source_agent_display_id(self) -> str:
        return self.base_rendered.source_agent_display_id

    @property
    def scenario_id(self) -> str:
        return self.base_rendered.scenario_id

    @property
    def branch(self) -> str:
        return self.base_rendered.branch

    @property
    def tick(self) -> int:
        return self.base_rendered.tick

    @property
    def expected_model(self) -> str:
        return self.base_rendered.expected_model

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_actions

    @property
    def allowed_topics(self) -> tuple[str, ...]:
        return self.base_rendered.allowed_topics

    @property
    def public_anonymity_enabled(self) -> bool:
        """Expose the unchanged V4 channel flag to transport/test adapters."""

        return self.base_rendered.public_anonymity_enabled

    @property
    def allowed_private_contact_ids(self) -> tuple[str, ...]:
        """Expose the unchanged closed contact set without leaking it in prompts."""

        return self.base_rendered.base_rendered.allowed_private_contact_ids


@dataclass(frozen=True)
class ValidatedForumAgentTurnV5:
    rendered: RenderedForumAgentTurnV5
    base_validated: ValidatedForumAgentTurnV4
    provenance: PublicProvenance


def _qualitative_band(value: Any, *, low: str, middle: str, high: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return middle
    number = float(value)
    if number < 0.34:
        return low
    if number < 0.67:
        return middle
    return high


def _natural_profile_context(profile: Mapping[str, Any]) -> dict[str, str]:
    stable = profile.get("stable_persona")
    stable = stable if isinstance(stable, Mapping) else {}
    episodic = profile.get("episodic_evidence")
    episodic = episodic if isinstance(episodic, Mapping) else {}
    language = profile.get("language_style")
    language = language if isinstance(language, Mapping) else {}
    stable_language = stable.get("language_style")
    stable_language = stable_language if isinstance(stable_language, Mapping) else {}
    tendencies = stable.get("forum_tendency_distribution")
    tendencies = tendencies if isinstance(tendencies, Mapping) else {}

    tendency_names = {
        "question": "追问关键信息",
        "request": "直接提出请求",
        "information_share": "补充自己掌握的信息",
        "help_request": "说明具体困难并求助",
        "emotional_expression": "表达当下感受",
        "emotional_vent": "表达当下感受",
        "complaint": "指出具体不满",
        "discussion": "回应观点并继续讨论",
        "other": "视情况参与或继续浏览",
    }
    ranked = sorted(
        (
            (str(key), float(value))
            for key, value in tendencies.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        key=lambda row: (-row[1], row[0]),
    )
    usual_moves = [tendency_names.get(key, "参与相关讨论") for key, _ in ranked[:2]]
    if not usual_moves:
        speech_act = str((profile.get("forum_tendencies") or {}).get("speech_act") or "")
        usual_moves = [tendency_names.get(speech_act, "视内容决定是否参与")]

    role = str(profile.get("micro_interaction_role") or "普通论坛参与者")
    role = role.split("_")[0].replace("-", " ").strip() or "普通论坛参与者"
    episode_focus = str(
        episodic.get("episode_focus") or profile.get("episode_focus") or ""
    ).strip()
    topics = [str(value) for value in episodic.get("topic_portfolio") or profile.get("topic_portfolio") or ()]
    needs = [str(value) for value in episodic.get("need_portfolio") or profile.get("need_portfolio") or ()]
    form = str(
        stable_language.get("language_form")
        or language.get("language_form")
        or language.get("summary")
        or "自然论坛表达"
    )
    density = str(
        stable_language.get("information_density")
        or language.get("information_density")
        or "适中"
    )
    uncertainty = str(
        stable.get("uncertainty_handling")
        or profile.get("uncertainty_handling")
        or "根据眼前信息形成判断"
    )
    behavior_examples = profile.get("reviewed_behavior_examples")
    behavior_examples = (
        behavior_examples
        if isinstance(behavior_examples, Sequence)
        and not isinstance(behavior_examples, (str, bytes))
        else ()
    )
    history_parts = []
    for example in behavior_examples[:3]:
        if not isinstance(example, Mapping):
            continue
        situation = str(example.get("past_situation") or "").strip()
        participation = str(example.get("usual_participation") or "").strip()
        communication = str(example.get("communication_style") or "").strip()
        if not situation or not participation:
            continue
        sentence = f"过去遇到“{situation}”时，{participation}"
        if communication:
            sentence += f"；{communication}"
        history_parts.append(sentence)
    if not history_parts and episode_focus:
        history_parts.append(f"过去曾围绕“{episode_focus}”参与论坛")
    if not behavior_examples and topics:
        history_parts.append("常见关注包括" + "、".join(topics[:3]))
    if not behavior_examples and needs:
        history_parts.append("常见诉求包括" + "、".join(needs[:2]))
    if not history_parts:
        history_parts.append("历史上会根据帖子与自身需要决定是否参与")
    interaction_move = _profile_interaction_move(profile)
    form = {
        "colloquial": "日常口语",
        "concise": "简短直接",
        "direct": "直接表达",
        "formal": "较完整、克制的表达",
        "narrative": "带具体经过的叙述",
    }.get(form.lower(), form)
    density = {
        "low": "偏低，通常只说最在意的一点",
        "medium": "适中，会给出足够让人接话的背景",
        "high": "较高，常补充较完整的来龙去脉",
    }.get(density.lower(), density)
    primary_move = str(interaction_move["primary_social_move"])
    identity = (
        f"这个人平时在校园论坛里更常以“{role}”的方式参与："
        f"{usual_moves[0]}。这描述的是长期参与习惯，不代表遇到每个话题都必须发言。"
    )
    return {
        "forum_identity": identity,
        "reviewed_history": "；".join(history_parts) + "。这些是过去的参与经历，用来理解此人的反应习惯。",
        "usual_participation": "通常会" + "，也会".join(dict.fromkeys(usual_moves)) + "。",
        "typical_thread_entry": primary_move,
        "communication_style": f"平时更接近{form}；信息密度{density}。",
        "handling_uncertainty": uncertainty.rstrip("。") + "。",
        "evidence_basis": "该人物由多条已审阅论坛经历综合形成。",
    }


def _natural_current_experience(posterior: Mapping[str, Any]) -> str:
    attention = _qualitative_band(
        posterior.get("attention"), low="目前只是顺手看到", middle="目前有所关注", high="目前高度关注"
    )
    concern = _qualitative_band(
        posterior.get("concern"), low="尚未感到明显压力", middle="对此有些担忧", high="此事带来较强担忧"
    )
    trust = _qualitative_band(
        posterior.get("trust"), low="对现有说法较不信任", middle="对现有说法仍有保留", high="目前较信任已看到的信息"
    )
    satisfaction = _qualitative_band(
        posterior.get("satisfaction"), low="当前体验较差", middle="当前体验一般", high="当前体验较好"
    )
    # Activity is deliberately not translated into an instruction such as
    # "more likely to browse".  That phrase was shared by most of the Stage-A
    # panel and unintentionally suppressed otherwise valid persona expression.
    # The scheduler may use activity privately; the resident still chooses its
    # action from the visible forum, its history and its concrete situation.
    return "；".join((attention, concern, trust, satisfaction)) + "。"


def _natural_scenario_position(
    stake: Mapping[str, Any], *, scenario_id: str
) -> str:
    values = stake.get("values")
    values = values if isinstance(values, Mapping) else {}
    direct = values.get("directly_affected") is True
    urgency = float(values.get("time_urgency", 0.5) or 0.5)
    substitute = float(
        values.get("substitute_resource_availability", 0.5) or 0.5
    )
    material = float(
        values.get("eligibility_material_completeness", 0.5) or 0.5
    )
    appeal_cost = float(values.get("appeal_cost", 0.5) or 0.5)
    bridge = float(
        values.get("bridge_organizing_care_responsibility", 0.0) or 0.0
    )

    if scenario_id == "century_gym_ghost_booking_dispute":
        direct_openings = (
            "你经常使用世纪馆，热门时段约不上会直接打乱已约好的运动安排",
            "你最近在尝试预约世纪馆，但还能考虑其他时段或场地",
        )
        direct_opening = direct_openings[0] if urgency >= 0.67 or substitute < 0.34 else direct_openings[1]
        bridge_opening = "你自己不一定要抢这个时段，但有熟人正在对照预约记录"
        bystander_opening = "你这次没有预约世纪馆，是在全局热榜或最新帖里看到讨论"
        material_sentences = (
            "你手里有比较完整的预约截图或到场记录",
            "你手里的预约截图或到场记录不齐，还不能说清空场是怎样造成的",
            "你手里有一些预约与到场信息，但还需要和别人的记录对照",
        )
        nondirect_material = "你没有亲自预约的截图或到场记录"
    elif scenario_id == "lecture_external_incident_shock":
        direct_opening = "你关注这场讲座事件，当事言论和后续处理会影响你是否继续参与相关活动"
        bridge_opening = "你不是当事人，但身边有人看过讲座片段并在讨论它"
        bystander_opening = "你没有亲历这场讲座，只看到论坛上时间可见的片段和转述"
        material_sentences = (
            "你看到的视频、录音或文字片段相对完整",
            "你只看到很短的片段，不会把没看到的部分当成事实",
            "你看到了一部分现场材料，足以谈可见言行，但不足以推断未知细节",
        )
        nondirect_material = "你没有亲历现场，不会把论坛转述写成亲眼所见"
    else:
        direct_opening = (
            "你本人正在申请暑期床位，暑期安排已经定下；如果没分到，临时找住处会很被动"
            if urgency >= 0.67 and substitute < 0.34
            else "你本人正在申请暑期床位，确认结果会影响实习、实验或通勤安排"
            if urgency >= 0.67
            else "你本人正在申请暑期床位，眼下几乎没有省事的替代住处"
            if substitute < 0.34
            else "你本人正在申请暑期床位，但还在比较校内申请与其他住处"
        )
        bridge_opening = "你本人不申请床位，但身边有人正在准备申请，你有时会替同伴整理信息"
        bystander_opening = "你本人这次不申请床位，只是在全局热榜或最新帖里看到这件事"
        material_sentences = (
            "你觉得自己能拿出的与申请有关的证明材料比较齐",
            "你手头与申请有关的证明材料并不齐，担心来回确认会错过时间",
            "你手头有一些与申请有关的证明材料，但不确定是否足以说明问题",
        )
        nondirect_material = "你不需要准备这次床位申请的办事材料"

    opening = direct_opening if direct else bridge_opening if bridge >= 0.67 else bystander_opening
    if direct:
        if material >= 0.67:
            material_sentence = material_sentences[0]
        elif material < 0.34:
            material_sentence = material_sentences[1]
        else:
            material_sentence = material_sentences[2]
        if appeal_cost >= 0.67:
            followup = "反复找入口、补充信息或追问进度会占掉你不少精力"
        elif appeal_cost < 0.34:
            followup = "真有疑问时，你愿意直接去问清楚或提交复核"
        else:
            followup = "你会先判断继续追问是否值得花时间"
    elif bridge >= 0.67:
        material_sentence = nondirect_material
        followup = "如果参与，你更可能整理已知信息、转达一个人的困难或连接不同圈子"
    else:
        material_sentence = nondirect_material
        followup = "是否继续参与，取决于它是否符合你平时的论坛兴趣和参与方式"
    return f"{opening}；{material_sentence}；{followup}。"


_PROMPT_ID_KEY_KINDS = {
    "agent_id": "contact",
    "source_agent_id": "contact",
    "contact_agent_id": "contact",
    "creator_agent_id": "contact",
    "target_agent_id": "contact",
    "target_agent_ids": "contact",
    "contact_agent_ids": "contact",
    "member_agent_ids": "contact",
    "pending_invite_agent_ids": "contact",
    "thread_id": "thread",
    "thread_ids": "thread",
    "target_thread_id": "thread",
    "current_thread_id": "thread",
    "linked_thread_id": "thread",
    "message_id": "post",
    "message_ids": "post",
    "target_message_id": "post",
    "quote_message_id": "post",
    "public_source_message_id": "post",
    "claim_id": "claim",
    "claim_ids": "claim",
    "target_claim_id": "claim",
    "correction_target_claim_id": "claim",
    "conversation_id": "chat",
    "conversation_ids": "chat",
    "private_conversation_id": "chat",
    "target_private_message_id": "private-post",
    "private_claim_id": "private-claim",
    "target_private_claim_id": "private-claim",
    "correction_target_private_claim_id": "private-claim",
}


def _prompt_aliases(payload: Mapping[str, Any], *, self_agent_id: str) -> dict[str, str]:
    values: dict[str, set[str]] = {}

    def visit(value: Any, key: str | None = None) -> None:
        kind = _PROMPT_ID_KEY_KINDS.get(str(key or ""))
        if kind and isinstance(value, str) and value and value != self_agent_id:
            values.setdefault(kind, set()).add(value)
        elif kind and isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, str) and child and child != self_agent_id:
                    values.setdefault(kind, set()).add(child)
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)

    visit(payload)
    aliases: dict[str, str] = {self_agent_id: "self"}
    for kind in sorted(values):
        for index, internal in enumerate(sorted(values[kind]), start=1):
            aliases.setdefault(internal, f"{kind}-{index:02d}")
    return aliases


def _replace_prompt_tokens(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _replace_prompt_tokens(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_prompt_tokens(child, replacements) for child in value]
    if isinstance(value, tuple):
        return [_replace_prompt_tokens(child, replacements) for child in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _naturalize_v4_prompt(
    payload: Mapping[str, Any],
    *,
    self_agent_id: str,
    model_view_condition: str = V5_FULL_NATURAL_PROFILE,
) -> tuple[dict[str, Any], dict[str, str]]:
    if model_view_condition not in V5_MODEL_VIEW_CONDITIONS:
        raise ForumTurnError("resident V5 model-view condition is invalid")
    view = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    profile = view.pop("profile", {})
    posterior = view.pop("posterior_state", {})
    view.pop("background_state_summary", None)
    view.pop("agent_contract", None)
    participation = view.pop("participation_brief", None)
    participation = participation if isinstance(participation, Mapping) else {}
    view["schema"] = FORUM_AGENT_TURN_V5_SCHEMA_VERSION
    view["model_visible_view"] = "campus-pulse-natural-resident-view-v1"
    view["resident_context"] = _natural_profile_context(
        profile if isinstance(profile, Mapping) else {}
    )
    view["current_experience"] = _natural_current_experience(
        posterior if isinstance(posterior, Mapping) else {}
    )

    scenario = view.get("scenario_context")
    if isinstance(scenario, dict):
        stake = scenario.pop("current_synthetic_stake", None)
        scenario.pop("response_diversity_boundary", None)
        if isinstance(stake, Mapping):
            scenario["your_current_situation"] = _natural_scenario_position(
                stake,
                scenario_id=str(view.get("scenario_id") or ""),
            )

    # V4 already carried a sound parallel-engagement contract, but the first
    # V5 naturalisation discarded it together with the internal participation
    # brief.  Surface the same affordances in ordinary Chinese so that a
    # resident may comment and also like/repost/report without being nudged to
    # satisfy a quota.
    visible_opportunities = []
    forum_context = view.get("forum_context")
    if isinstance(forum_context, list):
        for row in forum_context:
            if not isinstance(row, Mapping) or not row.get("message_id"):
                continue
            evidence_status = str(row.get("evidence_status") or "")
            relation = str(row.get("relation") or "")
            governance_artifact = row.get("governance_artifact")
            if isinstance(governance_artifact, Mapping):
                reason = (
                    "这是一条治理证据或服务回执；可以回应其具体内容，也可以转发给可能需要的人，"
                    "是否赞同仍由你自己判断"
                )
            elif evidence_status in {"unverified", "uncertain", "contested"}:
                reason = (
                    "这条内容仍有不确定处；可以追问、公开转发以征求更多信息，或先不操作；"
                    "转发不等于赞同"
                )
            elif relation in {"direct_parent", "quote"}:
                reason = "这条内容直接指向你正在看的讨论；可以具体回应，也可以只点赞、转发或不操作"
            else:
                reason = "如果它确实有用、值得更多人看到或需要公开质疑，可以分别点赞、转发或回应"
            visible_opportunities.append(
                {"message_id": row.get("message_id"), "natural_opportunity": reason}
            )
    view["forum_action_context"] = {
        "parallel_choices": (
            "公开主动作与点赞、转发、举报彼此独立：回复后仍可点赞或转发，也可以公开沉默但进行互动。"
        ),
        "like": "只在你确实认同、欣赏或觉得这条消息有用时点赞。",
        "repost": "只在你确实希望更多人看到时转发；转发可以是求证或公开质疑，不等于赞同。",
        "report": (
            "只有看见具体的骚扰、隐私、欺诈、垃圾信息或无依据有害断言时才举报；不同意本身不是举报理由。"
        ),
        "visible_message_opportunities": visible_opportunities,
        "decision_boundary": "没有数量要求；逐条按这个人的习惯和此刻处境决定，空列表也完全合理。",
    }

    # This is a bounded evaluation control, not a second resident model.  The
    # same world, state, action contract and safety boundary remain visible;
    # only the stable/history/person-specific inputs under test are removed.
    # Deliberately do not name the condition in the Provider-facing payload,
    # because doing so would cue the model about the expected comparison.
    if model_view_condition in {V5_CONTEXT_ABLATION, V5_CONTEXT_ONLY}:
        view["resident_context"] = {
            "forum_identity": "你正在以一名普通校园论坛居民的身份浏览。",
            "reviewed_history": "当前只需要处理眼前这条校园事件。",
            "usual_participation": "根据当前看见的内容决定是否参与。",
            "typical_thread_entry": "如果确有新内容就完成一个自然动作，否则继续阅读。",
            "communication_style": "使用自然、简洁的校园论坛中文。",
            "handling_uncertainty": "只依据此刻可见的信息形成判断。",
            "evidence_basis": "当前论坛上下文。",
        }
        if isinstance(scenario, dict):
            scenario["your_current_situation"] = (
                "这件事可能与你的校园生活有关；请只根据当前可见内容决定是否参与。"
            )

    # The paired joint-constraint experiment needs to distinguish the
    # contribution of the previous committed particle state from the stable
    # Profile/stake contribution.  Keep the exact same forum world and action
    # contract in all four views; only replace the model-visible state wording
    # with a neutral sentence when state is the held-out factor.  Raw particle
    # values remain private in every condition.
    if model_view_condition in {V5_CONTEXT_ONLY, V5_CONTEXT_PLUS_PROFILE}:
        view["current_experience"] = (
            "你刚刚打开这条讨论；此处不提供额外的情绪、信任或关注程度线索，"
            "请根据眼前可见内容决定是否参与。"
        )

    relationships = view.get("relationship_context")
    if isinstance(relationships, list):
        simplified = []
        labels = {
            "close_friend": "关系较近的好友",
            "familiar_peer": "熟悉的同伴",
            "cross_group_acquaintance": "不同圈子的熟人",
            "co_participant": "曾共同参与讨论的人",
        }
        for row in relationships:
            if not isinstance(row, Mapping):
                continue
            simplified.append(
                {
                    "contact_agent_id": row.get("contact_agent_id"),
                    "relationship": labels.get(
                        str(row.get("relation_type")), "认识的论坛联系人"
                    ),
                    "recent_shared_context": str(row.get("recent_shared_context") or ""),
                }
            )
        view["relationship_context"] = simplified

    private_channel = view.get("private_channel")
    if isinstance(private_channel, dict):
        channel_choice = private_channel.pop("channel_choice_brief", None)
        channel_choice = channel_choice if isinstance(channel_choice, Mapping) else {}
        private_channel.pop("style_and_epistemic_contract", None)
        private_channel["conversation_setting"] = (
            "请像使用普通校园论坛一样分别决定公开操作与私下交流。好友间可以简短、口语化，"
            "也可以只转述自己刚看到的内容；群聊邀请取决于是否真的需要多人一起确认或协调。"
        )
        intent_labels = {
            "ask_or_compare_interpretations": "找认识的人问一句，或比较彼此看到的信息",
            "vent_or_gossip_with_trusted_friend": "和信任的人吐槽、吃瓜或表达情绪",
            "share_what_was_seen": "把刚看到的公开内容转给认识的人",
            "seek_help_or_coordinate": "私下求助、一起核对或协调下一步",
        }
        cue_labels = {
            "time_visible_public_discussion_exists": "眼前已有可转述的公开讨论",
            "private_reply_or_invitation_waiting": "有私聊回复或群邀请等待处理",
            "resident_is_directly_affected": "这件事会直接影响自己",
            "time_pressure_is_high": "眼下有时间压力",
            "public_or_formal_appeal_is_costly": "公开或正式追问的成本较高",
            "coordination_or_service_help_may_be_needed": "可能需要一起核对或寻求服务帮助",
            "attention_is_high": "此刻确实在关注这件事",
            "concern_is_high": "此刻对此比较担心",
        }
        group_labels = {
            "at_least_two_closed_contacts_available": "至少有两名当前可以联系的人",
            "profile_supports_multi_person_coordination": "这个人的既往参与方式包含多人协调",
            "resident_is_directly_affected": "这件事会直接影响自己",
            "time_sensitive_comparison_or_coordination": "需要较快比较信息或协调",
            "shared_service_navigation_may_help": "多人一起确认服务流程可能更省事",
        }
        group = channel_choice.get("group_candidacy")
        group = group if isinstance(group, Mapping) else {}
        likely_intents = [
            intent_labels.get(str(value), str(value))
            for value in channel_choice.get("likely_private_intents", [])
            if isinstance(value, str)
        ]
        if model_view_condition in {V5_CONTEXT_ABLATION, V5_CONTEXT_ONLY}:
            likely_intents = ["只根据眼前内容判断是否有必要联系认识的人"]
        private_channel["current_private_opportunity"] = {
            "possible_motives": likely_intents,
            "current_reasons": [
                cue_labels.get(str(value), str(value))
                for value in channel_choice.get("current_cues", [])
                if isinstance(value, str)
            ],
            "contact_agent_ids": [
                str(value)
                for value in channel_choice.get("recommended_first_contact_ids", [])
                if isinstance(value, str)
            ],
            "small_group": {
                "plausible_now": bool(group.get("is_candidate")),
                "contact_agent_ids": [
                    str(value)
                    for value in group.get("candidate_contact_ids", [])
                    if isinstance(value, str)
                ],
                "reasons": [
                    group_labels.get(str(value), str(value))
                    for value in group.get("reason_codes", [])
                    if isinstance(value, str)
                ],
            },
            "decision_boundary": (
                "这些只是此刻真实存在的交流机会，不是任务指标。可以公开发言并同时私聊，"
                "也可以只私聊、只阅读或什么都不做；建群只在确实需要多人一起核对或协调时发生。"
            ),
        }
        recent = private_channel.get("recent_time_visible_messages")
        if isinstance(recent, list):
            private_channel["recent_time_visible_messages"] = [
                {
                    key: row.get(key)
                    for key in (
                        "message_id",
                        "conversation_id",
                        "source_agent_id",
                        "action",
                        "visible_text",
                        "created_tick",
                        "public_source_message_id",
                    )
                    if row.get(key) is not None
                }
                for row in recent
                if isinstance(row, Mapping)
            ]

    anonymity = view.get("public_anonymity")
    if isinstance(anonymity, dict):
        view["public_anonymity"] = {
            "enabled": bool(anonymity.get("enabled")),
            "forum_setting": (
                "同一个公开论坛允许普通帖子和评论使用稳定的合成昵称，也允许匿名显示。"
                "敏感经历或担心被熟人认出时可以匿名；希望持续公开交流时可以使用昵称。"
            ),
        }

    internal_to_alias = _prompt_aliases(view, self_agent_id=self_agent_id)
    view = _replace_prompt_tokens(view, internal_to_alias)

    def scrub_remaining_hashes(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): scrub_remaining_hashes(child)
                for key, child in value.items()
                if "sha256" not in str(key).lower()
                and "fingerprint" not in str(key).lower()
                and str(key) not in {"reviewed_evidence_refs", "profile_bundle_id", "micro_role_id"}
            }
        if isinstance(value, list):
            return [scrub_remaining_hashes(child) for child in value]
        if isinstance(value, str) and _SHA256_RE.fullmatch(value):
            return "internal-reference-omitted"
        return value

    return scrub_remaining_hashes(view), {
        alias: internal for internal, alias in internal_to_alias.items()
    }


def render_forum_agent_turn_v5(**kwargs: Any) -> RenderedForumAgentTurnV5:
    """Render a natural model view while retaining the proven V4 ledger contract."""

    model_view_condition = str(
        kwargs.pop("model_view_condition", V5_FULL_NATURAL_PROFILE)
    )
    if model_view_condition not in V5_MODEL_VIEW_CONDITIONS:
        raise ForumTurnError("resident V5 model-view condition is invalid")

    supplied_agent = kwargs.get("agent")
    if not isinstance(supplied_agent, Mapping):
        raise ForumTurnError("resident v5 requires one Agent mapping")
    compatible_agent = dict(supplied_agent)
    if compatible_agent.get("provider_turn_contract") == FORUM_AGENT_TURN_V5_SCHEMA_VERSION:
        compatible_agent["provider_turn_contract"] = FORUM_AGENT_TURN_V4_SCHEMA_VERSION
    kwargs["agent"] = compatible_agent
    base_rendered = render_forum_agent_turn_v4(**kwargs)
    base_payload = json.loads(base_rendered.agent_prompt)
    payload, alias_to_internal = _naturalize_v4_prompt(
        base_payload,
        self_agent_id=base_rendered.agent_id,
        model_view_condition=model_view_condition,
    )
    validate_prompt_payload(payload, current_tick=base_rendered.tick)
    agent_prompt = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = (
        "你正在继续一名合成校园论坛居民此刻的使用过程。请根据人物的论坛经历、当前处境、"
        "刚刚看到的公开帖子与私下消息，完成此刻最自然的一组论坛操作。公开发言、点赞转发和"
        "私下交流可以同时发生，也可以继续浏览。只做这个人此刻会做的事，不要替整个平台总结"
        "争议；可以偏向自己的处境、带情绪、只追问一个细节，也可以沉默。所有文字直接使用自然"
        "的校园论坛中文，所有判断以当前可见的合成信息为范围。请按操作回执合同返回一个 JSON 对象。"
    )
    prompt_sha = canonical_hash(
        {
            "renderer": FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
            "parent_renderer": FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
            "expected_model": base_rendered.expected_model,
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
        }
    )
    return RenderedForumAgentTurnV5(
        base_rendered=base_rendered,
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        prompt_sha256=prompt_sha,
        alias_to_internal=alias_to_internal,
        model_view_condition=model_view_condition,
    )


def validate_forum_agent_turn_v5(
    envelope: TurnEnvelope,
    *,
    rendered: RenderedForumAgentTurnV5,
    ledger: ForumLedger,
    reviewed_trace_release_sha256: str | None = None,
) -> ValidatedForumAgentTurnV5:
    if envelope.prompt_sha256 != rendered.prompt_sha256:
        raise ForumTurnError("forum resident V5 prompt fingerprint drifted")
    # V5 changes only the model-visible identifiers and context.  Its output
    # shape remains the strict V4 action receipt, including duplicate-key
    # rejection; using the same parser here prevents the alias bridge from
    # accidentally weakening the proven ledger contract.
    raw = _strict_response_object_v4(envelope.content)
    restored = _replace_prompt_tokens(raw, rendered.alias_to_internal)
    # A read/ignore turn can still carry parallel likes, reports or private
    # actions, but it cannot create public speech or mutate the public claim
    # graph.  Some otherwise usable Provider responses repeat the visible post
    # target or leave the bookkeeping-only topic fields blank.  V5 owns the
    # human-facing adapter, so normalize only those silent public fields before
    # handing the receipt to the unchanged V4/V3/V1 ledger validators.  This
    # never invents text, a claim, an interaction or a private action.
    if restored.get("action") in {"read", "ignore"}:
        topic = str(restored.get("topic") or "").strip()
        if topic not in rendered.allowed_topics:
            if not rendered.allowed_topics:
                raise ForumTurnError("silent resident turn has no allowed topic")
            topic = rendered.allowed_topics[0]
        restored.update(
            {
                "target_thread_id": None,
                "target_message_id": None,
                "quote_message_id": None,
                "target_claim_id": None,
                "topic": topic,
                "subtopic": str(restored.get("subtopic") or "background_read").strip()
                or "background_read",
                "claim_operation": "none",
                "claim_summary": "",
                "correction_target_claim_id": None,
                "visible_text": "",
                "public_private_origin_claim_id": None,
                "public_identity_mode": "named",
            }
        )
    base = validate_forum_agent_turn_v4(
        replace(
            envelope,
            prompt_sha256=rendered.base_rendered.prompt_sha256,
            content=json.dumps(
                restored,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        ),
        rendered=rendered.base_rendered,
        ledger=ledger,
        reviewed_trace_release_sha256=reviewed_trace_release_sha256,
    )
    provenance = replace(base.provenance, prompt_sha256=rendered.prompt_sha256)
    return ValidatedForumAgentTurnV5(
        rendered=rendered,
        base_validated=replace(base, provenance=provenance),
        provenance=provenance,
    )


def materialize_forum_agent_turn_v5(
    *,
    validated: ValidatedForumAgentTurnV5,
    ledger: ForumLedger,
    memory: AgentMemory,
) -> MaterializedForumTurnV4:
    return materialize_forum_agent_turn_v4(
        validated=validated.base_validated,
        ledger=ledger,
        memory=memory,
    )
