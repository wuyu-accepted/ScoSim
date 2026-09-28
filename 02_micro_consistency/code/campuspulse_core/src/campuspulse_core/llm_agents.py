"""Live, budgeted LLM agents grounded by the frozen shock benchmark.

The historical benchmark is used after generation for aggregate comparison. Its
observed target counts are deliberately excluded from every agent prompt.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .policies import POLICIES, Policy
from .shock_replay import DEFAULT_DEFINITIONS, load_shock_definitions


HERE = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = HERE / "data" / "shock_replay" / "shock_benchmark.json"
DEFAULT_BASE_URL = "https://api.vectorengine.cn/v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_API_KEY_ENV = "VECTOR_ENGINE_API_KEY"
PHASE_IDS = ("baseline", "burst", "spread", "decay")
FORMAT_REPAIR_PROMPT_CONTRACT_VERSION = (
    "campus-pulse-controlled-format-repair-v1"
)
FORMAT_REPAIR_PROMPT_LIMIT = 1
_FORMAT_REPAIR_REASON_CODE_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]{2,63}$"
)
_MAX_FORMAT_REPAIR_PROMPT_BYTES = 65_536
NON_EVENT_TOPIC_ID = "off_topic_daily_life"
NON_EVENT_TOPIC_LABEL = "与本次冲击无关的日常校园生活"

ALLOWED_ACTIONS = (
    "ignore",
    "read",
    "comment",
    "post",
    "seek_help",
    "report_risk",
)
ALLOWED_STANCES = ("supportive", "neutral", "concerned", "critical")
ALLOWED_RATIONALE_TAGS = (
    "information_gap",
    "service_capacity",
    "fairness",
    "safety",
    "peer_influence",
    "policy_response",
    "general",
)
PUBLIC_ACTIONS = {"comment", "post", "seek_help", "report_risk"}
PRIMARY_POST_ACTIONS = {"post", "seek_help", "report_risk"}

ROLE_TRAITS = {
    "stakeholder": {
        "label": "直接利益相关者",
        "trust_range": (0.35, 0.65),
        "concern_range": (0.65, 0.95),
    },
    "information_seeker": {
        "label": "信息寻求者",
        "trust_range": (0.45, 0.75),
        "concern_range": (0.45, 0.80),
    },
    "amplifier": {
        "label": "讨论放大者",
        "trust_range": (0.30, 0.70),
        "concern_range": (0.40, 0.85),
    },
    "observer": {
        "label": "普通观察者",
        "trust_range": (0.50, 0.85),
        "concern_range": (0.10, 0.50),
    },
}

POLICY_PROMPTS = {
    "baseline": "没有新增治理措施，仅保留有限的信息发布和常规渠道。",
    "priority_routing": "平台把高紧迫度诉求优先送入处理队列，但不增加总容量。",
    "faq_deflection": "平台发布针对核心信息缺口的结构化FAQ，并持续更新统一口径。",
    "peer_support": "平台提高可信同伴答复的可见度，鼓励经验互助，但不冒充官方结论。",
    "dynamic_capacity": "治理部门在积压上升时增加临时处理容量，并优先处理高紧迫度诉求。",
    "combined": "同时发布结构化FAQ、开放反馈渠道、优先路由紧迫诉求并按积压动态扩容。",
}

PHASE_CONTEXT = {
    "baseline": "冲击事件尚未发生，校园论坛处于日常讨论状态。",
    "burst": "事件信息刚刚进入公共视野，信息不完整且注意力快速集中。",
    "spread": "讨论已扩散，用户开始引用他人观点并形成多个相邻议题。",
    "decay": "最初注意力开始回落，但未解决的信息缺口或利益问题仍可能持续。",
}

EVENT_STIMULI = {
    "tongzhou_governance_information_shock": (
        "校方在论坛之外发布通州校区入驻相关公告。当前仅知道该公告引发了对双校区、"
        "通勤、课程衔接、住宿、学生组织运行和反馈渠道的关注；不得补造公告细则。"
    ),
    "lecture_external_incident_shock": (
        "一场发生在论坛之外的校园讲座因主讲者对求职与职业选择的争议性表达受到关注。"
        "当前只讨论公开争议及其职业议题外溢，不得补造现场过程、当事人动机或处理结果。"
    ),
}

SCENARIO_ACTIVATION = {
    "tongzhou_governance_information_shock": {
        "archetype-01": ("direct", "课程安排、选课或培养衔接规则出现新的信息缺口"),
        "archetype-02": ("adjacent", "学习计划可能因跨校区课程安排而需要调整"),
        "archetype-03": ("direct", "入驻、通勤、住宿或反馈流程需要查询正式口径"),
        "archetype-04": (
            "direct",
            "只针对公告产生的通勤、住宿、场地或教学安排透明度提出改进诉求，不转成成绩评分投诉",
        ),
        "archetype-05": ("low", "当前冲击与职业选择没有直接关系"),
        "archetype-06": ("direct", "学生组织、项目或活动的跨校区运行需要组织方说明"),
        "archetype-07": ("direct", "通勤、住宿或日常服务需要形成可执行生活方案"),
        "archetype-08": ("low", "当前没有明确可交易标的，不应借事件强行发布交易"),
        "archetype-09": ("low", "当前没有高风险撮合标的，不应把公告改写成风险交易"),
        "archetype-10": ("adjacent", "跨校区可能影响同伴互动、活动同行和归属体验"),
        "archetype-11": ("direct", "跨校区可能造成任务成员、资料或行程协同缺口"),
        "archetype-12": ("adjacent", "既有协作承诺可能因地点与时间成本出现新边界"),
        "archetype-13": ("adjacent", "公告不确定性可能触发担忧、挫败或求共鸣"),
        "archetype-14": ("direct", "资源分配、公平标准和反馈程序形成公共规范争议"),
        "archetype-15": ("direct", "可以分享对双校区生活的即时观感或轻量经验"),
        "archetype-17": ("low", "公告本身不足以证明已出现具体身心影响，不应虚构健康求助"),
    },
    "lecture_external_incident_shock": {
        "archetype-01": ("low", "争议事件不直接形成课程考核规则问题"),
        "archetype-02": ("adjacent", "争议可能触发对学习投入与职业准备的行动困惑"),
        "archetype-03": ("adjacent", "可能询问公开说明或正式反馈渠道，但不是一般办事流程"),
        "archetype-04": ("adjacent", "若关注讲座组织或回应方式，可提出具体服务改进诉求"),
        "archetype-05": ("direct", "争议直接触发求职、考公和职业路径的判断与焦虑"),
        "archetype-06": ("adjacent", "只有真实机会发布受到事件影响时才说明条件"),
        "archetype-07": ("low", "当前争议不是消费、出行或设备选择问题"),
        "archetype-08": ("low", "当前没有普通可交付交易标的"),
        "archetype-09": ("low", "当前没有风险撮合标的，不得借争议招揽服务"),
        "archetype-10": ("adjacent", "事件可能促成求职压力下的同伴交流与陪伴"),
        "archetype-11": ("low", "当前没有明确团队、资料或线索补缺任务"),
        "archetype-12": ("adjacent", "可讨论争议表达对既有协作信任和边界的影响"),
        "archetype-13": ("direct", "贬抑性职业表达可能触发愤怒、受挫和求共鸣"),
        "archetype-14": ("direct", "事件涉及言论边界、职业尊重、公平与公共回应规范"),
        "archetype-15": ("direct", "可以分享听闻事件后的即时观感和同类经历"),
        "archetype-17": ("adjacent", "只有职业焦虑已形成具体身心困扰时才行动求助"),
    },
}

SCENARIO_EXPOSURE = {
    "tongzhou_governance_information_shock": {
        "burst_mode": "direct_public_broadcast",
        "burst_visible_fraction": 1.0,
        "decay_attention_retained_fraction": 0.55,
        "evidence_basis": (
            "externally timestamped official announcement; unresolved administrative questions "
            "retain a qualitative half of active attention, longer than an external incident"
        ),
    },
    "lecture_external_incident_shock": {
        "burst_mode": "limited_early_or_firsthand_exposure",
        "burst_visible_fraction": 0.18,
        "decay_attention_retained_fraction": 0.40,
        "evidence_basis": (
            "event occurred outside the forum and the external report appeared the next day; "
            "the exact event-hour anchor is unavailable; decay retains a qualitative minority "
            "rather than fitting the observed point estimate"
        ),
    },
}

CONTACT_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?:微信|vx|v信|qq|企鹅)\s*[:：]?\s*[A-Za-z0-9_-]{5,}", re.I),
)
INFERRED_DEMOGRAPHIC_PATTERNS = (
    re.compile(r"(?:大[一二三四五]|研[一二三]|博[一二三四五])(?:学生|同学)?"),
    re.compile(r"(?:本科生|研究生|博士生|男生|女生|男性|女性)"),
    re.compile(r"(?:文科|理工科?|商科|法学|文学)?硕士"),
    re.compile(r"(?:文科|理工科?|商科|法学|文学)(?:方向|专业|背景)"),
    re.compile(r"家庭经济(?:条件)?(?:普通|困难|优渥)"),
    re.compile(r"(?<!\d)\d{1,2}\s*岁(?!\d)"),
)
DEMOGRAPHIC_GENERALIZATIONS = (
    (re.compile(r"(?:大[一二三四五]|研[一二三]|博[一二三四五])(?:学生|同学)?"), "当前学习阶段"),
    (re.compile(r"(?:本科生|研究生|博士生)"), "同学"),
    (re.compile(r"(?:男生|女生|男性|女性)"), "当事人"),
    (re.compile(r"(?:文科|理工科?|商科|法学|文学)?硕士"), "当前方向"),
    (re.compile(r"(?:文科|理工科?|商科|法学|文学)(?:方向|专业|背景)"), "当前方向"),
    (re.compile(r"家庭经济(?:条件)?(?:普通|困难|优渥)"), "经济承受空间有限"),
    (re.compile(r"(?<!\d)\d{1,2}\s*岁(?!\d)"), "当前年龄阶段"),
)
RISK_SIGNAL_PATTERNS = re.compile(
    r"代(?:到场|签到|课|测|考|做|写)|答案|成品|冒名|绕过|访问限制|受限(?:网络|工具|材料)"
)


class LLMAgentError(ValueError):
    """Raised when the live LLM agent contract cannot be satisfied."""


@dataclass
class AgentState:
    agent_id: str
    role_id: str
    role_label: str
    concern_topic_id: str
    concern_topic_label: str
    population_weight: float
    trust: float
    concern: float
    archetype_id: str = "generic"
    archetype_label: str = ""
    persona_variant_id: str = ""
    persona_variant_label: str = ""
    persona: dict[str, Any] = field(default_factory=dict)
    population_month: str = ""
    evidence_support: int = 0
    event_exposure_score: float = 0.0
    event_attention_score: float = 0.0
    satisfied: bool = False
    prior_action: str = "none"
    memory: list[str] = field(default_factory=list)
    message_memory: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentDecision:
    action: str
    topic_id: str
    stance: str
    urgency: float
    trust_delta: float
    satisfied: bool
    rationale_tag: str
    message: str


@dataclass(frozen=True)
class ModelTurn:
    decision: AgentDecision
    returned_model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider_attempts: int
    format_repairs: int
    contract_adjustments: tuple[str, ...] = ()
    cache_hit: bool = False


@dataclass(frozen=True)
class RenderedAgentTurn:
    """Pure, bounded inputs required to execute one synthetic-agent turn."""

    system_prompt: str
    agent_prompt: str
    topic_ids: tuple[str, ...]
    simulation_mode: str
    prior_messages: tuple[str, ...]
    activation: dict[str, Any]
    prompt_sha256: str


@dataclass(frozen=True)
class AgentStateTransition:
    """Copy-on-write result of applying one validated model turn."""

    agent: AgentState
    turn: ModelTurn
    record: dict[str, Any]


class TokenBudget:
    """Thread-safe conservative reservation around provider-reported usage."""

    def __init__(self, max_total_tokens: int) -> None:
        if max_total_tokens <= 0:
            raise LLMAgentError("max_total_tokens must be positive")
        self.max_total_tokens = max_total_tokens
        self.spent_tokens = 0
        self.inflight_reserved_tokens = 0
        self.provider_calls = 0
        self._lock = threading.Lock()

    def reserve(self, amount: int) -> int:
        amount = max(1, int(amount))
        with self._lock:
            if (
                self.spent_tokens + self.inflight_reserved_tokens + amount
                > self.max_total_tokens
            ):
                raise LLMAgentError(
                    "token budget cannot reserve the next provider request"
                )
            self.inflight_reserved_tokens += amount
        return amount

    def record_provider_attempt(self) -> None:
        with self._lock:
            self.provider_calls += 1

    def commit(self, reservation: int, reported_total: int | None) -> None:
        accounted = (
            int(reported_total)
            if isinstance(reported_total, int) and reported_total > 0
            else reservation
        )
        with self._lock:
            self.inflight_reserved_tokens -= reservation
            self.spent_tokens += accounted

    def release(self, reservation: int) -> None:
        with self._lock:
            self.inflight_reserved_tokens -= reservation

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_total_tokens": self.max_total_tokens,
                "provider_reported_or_reserved_tokens": self.spent_tokens,
                "inflight_reserved_tokens": self.inflight_reserved_tokens,
                "provider_calls": self.provider_calls,
            }


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class DecisionCache:
    """Thread-safe, prompt-fingerprinted cache for resumable live simulations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.writes = 0
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "1.0.0" or not isinstance(
                payload.get("entries"), dict
            ):
                raise LLMAgentError("decision cache has an unsupported schema")
            self._entries = payload["entries"]

    def get(self, key: str) -> ModelTurn | None:
        with self._lock:
            raw = self._entries.get(key)
            if raw is None:
                return None
            self.hits += 1
            return ModelTurn(
                decision=AgentDecision(**raw["decision"]),
                returned_model=raw["returned_model"],
                prompt_tokens=int(raw["prompt_tokens"]),
                completion_tokens=int(raw["completion_tokens"]),
                total_tokens=int(raw["total_tokens"]),
                provider_attempts=0,
                format_repairs=int(raw["format_repairs"]),
                cache_hit=True,
            )

    def put(self, key: str, turn: ModelTurn) -> None:
        entry = {
            "decision": asdict(turn.decision),
            "returned_model": turn.returned_model,
            "prompt_tokens": turn.prompt_tokens,
            "completion_tokens": turn.completion_tokens,
            "total_tokens": turn.total_tokens,
            "format_repairs": turn.format_repairs,
        }
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = entry
            self.writes += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    {"schema_version": "1.0.0", "entries": self._entries},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "entry_count": len(self._entries),
                "cache_hits_this_run": self.hits,
                "new_entries_this_run": self.writes,
                "stores_only_synthetic_decisions": True,
                "stores_prompts_or_api_credentials": False,
            }


class VectorEngineClient:
    """Small OpenAI-compatible Chat Completions client with no SDK dependency."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "none",
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 220,
        max_network_retries: int = 1,
        max_format_repairs: int = 2,
        transport: Transport | None = None,
        decision_cache: DecisionCache | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMAgentError("API key is empty")
        if not base_url.startswith("https://"):
            raise LLMAgentError("base_url must use https")
        if not model.strip():
            raise LLMAgentError("model is empty")
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise LLMAgentError("unsupported reasoning_effort")
        if not 32 <= max_output_tokens <= 1000:
            raise LLMAgentError("max_output_tokens must be between 32 and 1000")
        if not 1 <= timeout_seconds <= 300:
            raise LLMAgentError("timeout_seconds must be between 1 and 300")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_network_retries = max_network_retries
        self.max_format_repairs = max_format_repairs
        self._transport = transport
        self.decision_cache = decision_cache

    @classmethod
    def from_environment(
        cls,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        **kwargs: Any,
    ) -> "VectorEngineClient":
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", api_key_env):
            raise LLMAgentError("api_key_env is not a safe environment variable name")
        return cls(api_key=os.environ.get(api_key_env, ""), **kwargs)

    def _http_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:800]
            body = body.replace(self._api_key, "[REDACTED]")
            raise LLMAgentError(f"provider HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            message = str(exc).replace(self._api_key, "[REDACTED]")
            raise LLMAgentError(f"provider request failed: {message}") from exc

    def _request(
        self,
        messages: list[dict[str, str]],
        budget: TokenBudget,
    ) -> tuple[dict[str, Any], int, int]:
        payload = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        reservation = budget.reserve(len(rendered) + self.max_output_tokens + 64)
        attempts = 0
        last_error: Exception | None = None
        try:
            for attempt in range(self.max_network_retries + 1):
                attempts += 1
                budget.record_provider_attempt()
                try:
                    response = (
                        self._transport(payload)
                        if self._transport is not None
                        else self._http_transport(payload)
                    )
                    usage = response.get("usage", {})
                    reported = usage.get("total_tokens")
                    budget.commit(reservation, reported)
                    return response, attempts, reservation
                except LLMAgentError as exc:
                    last_error = exc
                    if attempt >= self.max_network_retries:
                        raise
                    time.sleep(min(2**attempt, 2))
        except Exception:
            budget.release(reservation)
            raise
        raise LLMAgentError(str(last_error) if last_error else "provider request failed")

    def decide(
        self,
        *,
        system_prompt: str,
        agent_prompt: str,
        topic_ids: set[str],
        budget: TokenBudget,
        simulation_mode: str = "active_episode_agent",
        prior_messages: tuple[str, ...] = (),
    ) -> ModelTurn:
        cache_key = _canonical_hash(
            {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "max_output_tokens": self.max_output_tokens,
                "system_prompt": system_prompt,
                "agent_prompt": agent_prompt,
                "topic_ids": sorted(topic_ids),
                "simulation_mode": simulation_mode,
                "prior_messages": list(prior_messages),
            }
        )
        if self.decision_cache is not None:
            cached = self.decision_cache.get(cache_key)
            if cached is not None:
                return cached
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": agent_prompt},
        ]
        prompt_tokens = completion_tokens = total_tokens = provider_attempts = 0
        format_repairs = 0
        last_error: Exception | None = None
        for repair in range(self.max_format_repairs + 1):
            response, attempts, _ = self._request(messages, budget)
            provider_attempts += attempts
            usage = response.get("usage", {})
            prompt_tokens += _safe_int(usage.get("prompt_tokens"))
            completion_tokens += _safe_int(usage.get("completion_tokens"))
            total_tokens += _safe_int(usage.get("total_tokens"))
            try:
                content = response["choices"][0]["message"]["content"]
                decision = validate_agent_decision(
                    extract_json_object(content),
                    topic_ids=topic_ids,
                    simulation_mode=simulation_mode,
                    prior_messages=prior_messages,
                )
                turn = ModelTurn(
                    decision=decision,
                    returned_model=str(response.get("model") or self.model),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    provider_attempts=provider_attempts,
                    format_repairs=format_repairs,
                )
                if self.decision_cache is not None:
                    self.decision_cache.put(cache_key, turn)
                return turn
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, LLMAgentError) as exc:
                last_error = exc
                if repair >= self.max_format_repairs:
                    break
                format_repairs += 1
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{agent_prompt}\n\n上一次输出未通过JSON契约。"
                            f"具体原因：{str(exc)[:180]}。"
                            "请针对该原因修正，只重新输出一个符合要求的JSON对象，不要解释。"
                        ),
                    },
                ]
        raise LLMAgentError(f"model output failed the action contract: {last_error}")


def _safe_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and value >= 0 else 0


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def extract_json_object(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise LLMAgentError("model content is not text")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMAgentError("model content has no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise LLMAgentError("model content JSON is not an object")
    return value


def _bounded_number(value: Any, low: float, high: float, field_name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LLMAgentError(f"{field_name} must be a finite number")
    return round(min(high, max(low, float(value))), 4)


def _safe_synthetic_message(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LLMAgentError("message must be text")
    message = " ".join(value.strip().split())[:80]
    if any(pattern.search(message) for pattern in CONTACT_PATTERNS):
        return "[已抑制：疑似联系方式]"
    for pattern, replacement in DEMOGRAPHIC_GENERALIZATIONS:
        message = pattern.sub(replacement, message)
    return message


def _message_similarity(left: str, right: str) -> float:
    normalize = lambda value: re.sub(r"[^\w\u4e00-\u9fff]", "", value).lower()
    return difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def validate_agent_decision(
    raw: dict[str, Any],
    *,
    topic_ids: set[str],
    simulation_mode: str = "active_episode_agent",
    prior_messages: tuple[str, ...] = (),
) -> AgentDecision:
    action = raw.get("action")
    topic_id = raw.get("topic_id")
    stance = raw.get("stance")
    rationale_tag = raw.get("rationale_tag")
    if action not in ALLOWED_ACTIONS:
        raise LLMAgentError("invalid action")
    if topic_id not in topic_ids:
        raise LLMAgentError("invalid topic_id")
    if stance not in ALLOWED_STANCES:
        raise LLMAgentError("invalid stance")
    if rationale_tag not in ALLOWED_RATIONALE_TAGS:
        rationale_tag = "general"
    if not isinstance(raw.get("satisfied"), bool):
        raise LLMAgentError("satisfied must be boolean")
    message = _safe_synthetic_message(raw.get("message", ""))
    if action in {"ignore", "read"}:
        message = ""
    elif not message:
        raise LLMAgentError("public action requires a message")
    if message and any(_message_similarity(message, prior) >= 0.86 for prior in prior_messages):
        raise LLMAgentError("message repeated the agent's prior public content")
    if (
        message
        and simulation_mode == "governance_risk_signal"
        and not RISK_SIGNAL_PATTERNS.search(message)
    ):
        action = "read"
        message = ""
    return AgentDecision(
        action=action,
        topic_id=topic_id,
        stance=stance,
        urgency=_bounded_number(raw.get("urgency"), 0.0, 1.0, "urgency"),
        trust_delta=_bounded_number(
            raw.get("trust_delta"), -0.25, 0.25, "trust_delta"
        ),
        satisfied=raw["satisfied"],
        rationale_tag=rationale_tag,
        message=message,
    )


def validate_rendered_turn(
    content: Any,
    rendered_turn: RenderedAgentTurn,
) -> AgentDecision:
    """Validate one model content value against its exact rendered turn.

    This wrapper deliberately accepts model content rather than a provider
    envelope. Provider usage, retries and persistence remain responsibilities
    of the caller.
    """

    return validate_agent_decision(
        extract_json_object(content),
        topic_ids=set(rendered_turn.topic_ids),
        simulation_mode=rendered_turn.simulation_mode,
        prior_messages=rendered_turn.prior_messages,
    )


def load_llm_agent_inputs(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    definitions = load_shock_definitions(definitions_path)
    expected_fingerprint = _canonical_hash(
        {
            key: value
            for key, value in benchmark.items()
            if key not in {"generated_at_utc", "benchmark_fingerprint"}
        }
    )
    if benchmark.get("benchmark_fingerprint") != expected_fingerprint:
        raise LLMAgentError("benchmark fingerprint does not match")
    if benchmark.get("definition_fingerprint") != definitions.get(
        "definition_fingerprint"
    ):
        raise LLMAgentError("benchmark and definitions do not match")
    if benchmark.get("data_audit", {}).get("event_windows_intersect_known_gaps"):
        raise LLMAgentError("benchmark event windows intersect known gaps")
    return benchmark, definitions


def _allocate_role_counts(
    roles: dict[str, dict[str, float]], agent_count: int
) -> dict[str, int]:
    raw = {role_id: role["weight"] * agent_count for role_id, role in roles.items()}
    counts = {role_id: int(math.floor(value)) for role_id, value in raw.items()}
    remaining = agent_count - sum(counts.values())
    order = sorted(raw, key=lambda role_id: (-(raw[role_id] - counts[role_id]), role_id))
    for role_id in order[:remaining]:
        counts[role_id] += 1
    return counts


def build_agent_population(
    definition: dict[str, Any],
    roles: dict[str, dict[str, float]],
    *,
    agent_count: int,
    seed: int,
) -> list[AgentState]:
    if not 2 <= agent_count <= 200:
        raise LLMAgentError("agent_count must be between 2 and 200")
    rng = random.Random(seed)
    topics = [(topic["topic_id"], topic["label"]) for topic in definition["topics"]]
    primary_topic = definition["primary_topic_id"]
    topic_cycle = sorted(topics, key=lambda item: (item[0] != primary_topic, item[0]))
    role_counts = _allocate_role_counts(roles, agent_count)
    agents: list[AgentState] = []
    topic_index = 0
    for role_id in sorted(role_counts):
        traits = ROLE_TRAITS.get(role_id, ROLE_TRAITS["observer"])
        for _ in range(role_counts[role_id]):
            topic_id, topic_label = topic_cycle[topic_index % len(topic_cycle)]
            topic_index += 1
            agents.append(
                AgentState(
                    agent_id=f"sim-agent-{len(agents) + 1:03d}",
                    role_id=role_id,
                    role_label=traits["label"],
                    concern_topic_id=topic_id,
                    concern_topic_label=topic_label,
                    population_weight=round(1.0 / agent_count, 10),
                    trust=round(rng.uniform(*traits["trust_range"]), 4),
                    concern=round(rng.uniform(*traits["concern_range"]), 4),
                )
            )
    return agents


def _path_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def load_representative_agent_inputs(
    *,
    persona_release_path: str | Path,
    persona_evidence_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    release = json.loads(Path(persona_release_path).read_text(encoding="utf-8"))
    evidence = json.loads(Path(persona_evidence_path).read_text(encoding="utf-8"))
    if release.get("status") != "complete" or evidence.get("status") != "complete":
        raise LLMAgentError("representative population inputs must be complete")
    validity = release.get("validity", {})
    if not validity.get("all_profiles_directly_read"):
        raise LLMAgentError("persona profiles were not all directly reviewed")
    if release.get("privacy_scan", {}).get("status") != "pass":
        raise LLMAgentError("persona release did not pass privacy review")
    profiles = {item["archetype_id"] for item in release.get("profiles", [])}
    packs = {item["archetype_id"] for item in evidence.get("packs", [])}
    if not profiles or profiles != packs:
        raise LLMAgentError("persona release and evidence packs do not align")
    return release, evidence


def _minimum_hamilton_counts(
    supports: dict[str, int], agent_count: int
) -> dict[str, int]:
    positive = {key: value for key, value in supports.items() if value > 0}
    if len(positive) > agent_count:
        raise LLMAgentError("agent count is too small for positive monthly archetypes")
    counts = {key: 1 for key in positive}
    remaining = agent_count - len(counts)
    if remaining == 0:
        return counts
    total = sum(positive.values())
    raw = {key: positive[key] / total * remaining for key in positive}
    floors = {key: int(math.floor(value)) for key, value in raw.items()}
    for key, value in floors.items():
        counts[key] += value
    leftover = remaining - sum(floors.values())
    order = sorted(
        positive,
        key=lambda key: (-(raw[key] - floors[key]), -positive[key], key),
    )
    for key in order[:leftover]:
        counts[key] += 1
    return counts


def build_representative_agent_population(
    definition: dict[str, Any],
    persona_release: dict[str, Any],
    persona_evidence: dict[str, Any],
    *,
    agent_count: int,
    seed: int,
) -> tuple[list[AgentState], dict[str, Any]]:
    """Build a month-grounded population from reviewed behavioral episodes."""

    if not 16 <= agent_count <= 200:
        raise LLMAgentError(
            "representative population needs 16 to 200 agents for class coverage"
        )
    month = str(definition["anchor"])[:7]
    profiles = {
        item["archetype_id"]: item for item in persona_release["profiles"]
    }
    packs = {item["archetype_id"]: item for item in persona_evidence["packs"]}
    eligible = {
        archetype_id: int(packs[archetype_id]["month_support"].get(month, 0))
        for archetype_id, profile in profiles.items()
        if profile["direct_review"]["eligible_for_population_agent"]
    }
    if any(value <= 0 for value in eligible.values()):
        missing = sorted(key for key, value in eligible.items() if value <= 0)
        raise LLMAgentError(
            f"population month lacks support for reviewed classes: {', '.join(missing)}"
        )
    excluded = {
        archetype_id: int(packs[archetype_id]["month_support"].get(month, 0))
        for archetype_id, profile in profiles.items()
        if not profile["direct_review"]["eligible_for_population_agent"]
    }
    counts = _minimum_hamilton_counts(eligible, agent_count)
    full_month_support = sum(eligible.values()) + sum(excluded.values())
    if full_month_support <= 0:
        raise LLMAgentError("population month has no evidence support")
    rng = random.Random(seed)
    blueprints: list[
        tuple[str, dict[str, Any], dict[str, Any], int, dict[str, Any]]
    ] = []
    for archetype_id in sorted(counts):
        profile = profiles[archetype_id]
        pack = packs[archetype_id]
        representative_by_id = {
            item["record_id"]: item for item in pack["representative_evidence"]
        }
        variant_offset = rng.randrange(3)
        variant_order = (
            profile["variants"][variant_offset:]
            + profile["variants"][:variant_offset]
        )
        evidence_queues = {
            variant["variant_id"]: [
                representative_by_id[record_id]
                for record_id in variant["evidence_record_ids"]
                if record_id in representative_by_id
            ]
            for variant in variant_order
        }
        for queue in evidence_queues.values():
            rng.shuffle(queue)
        unique_options: list[tuple[dict[str, Any], dict[str, Any]]] = []
        used_record_ids = set()
        while True:
            added = False
            for variant in variant_order:
                queue = evidence_queues[variant["variant_id"]]
                while queue and queue[0]["record_id"] in used_record_ids:
                    queue.pop(0)
                if queue:
                    evidence_item = queue.pop(0)
                    used_record_ids.add(evidence_item["record_id"])
                    unique_options.append((variant, evidence_item))
                    added = True
            if not added:
                break
        for index in range(counts[archetype_id]):
            if index < len(unique_options):
                variant, evidence_item = unique_options[index]
                focus = {
                    "primary_need": evidence_item["primary_need"],
                    "desired_response": evidence_item["desired_response"],
                    "speech_act": evidence_item["speech_act"],
                    "stance": evidence_item["stance"],
                    "source": "redacted_semantic_evidence_not_raw_text",
                    "specificity_rule": (
                        "只借用抽象诉求，不复现证据中的具体日期、数值、课程名、"
                        "身份阶段、专业或家庭背景。"
                    ),
                }
            else:
                variant = variant_order[index % len(variant_order)]
                focus = {
                    "primary_need": variant["situational_signature"],
                    "desired_response": variant["reaction_logic"],
                    "speech_act": "unspecified",
                    "stance": "unspecified",
                    "source": "reviewed_variant_fallback",
                }
            blueprints.append(
                (archetype_id, profile, variant, eligible[archetype_id], focus)
            )
    rng.shuffle(blueprints)

    agents = []
    for index, (archetype_id, profile, variant, support, focus) in enumerate(
        blueprints, 1
    ):
        pack = packs[archetype_id]
        agents.append(
            AgentState(
                agent_id=f"sim-agent-{index:03d}",
                role_id=pack["taxonomy_definition"]["interaction_role"],
                role_label=pack["label"],
                concern_topic_id=NON_EVENT_TOPIC_ID,
                concern_topic_label=NON_EVENT_TOPIC_LABEL,
                population_weight=round(
                    (support / full_month_support) / counts[archetype_id], 10
                ),
                trust=round(rng.uniform(0.45, 0.55), 4),
                concern=round(rng.uniform(0.15, 0.35), 4),
                archetype_id=archetype_id,
                archetype_label=pack["label"],
                persona_variant_id=variant["variant_id"],
                persona_variant_label=variant["label"],
                persona={
                    "profile_summary": profile["profile_summary"],
                    "observable_core_tendencies": profile[
                        "observable_core_tendencies"
                    ],
                    "trigger_conditions": profile["trigger_conditions"],
                    "language_tendencies": profile["language_tendencies"],
                    "do_not_simulate": profile["do_not_simulate"],
                    "situational_signature": variant["situational_signature"],
                    "reaction_logic": variant["reaction_logic"],
                    "language_style": variant["language_style"],
                    "uncertainty_behavior": variant["uncertainty_behavior"],
                    "simulation_instruction": variant["simulation_instruction"],
                    "assigned_episode_focus": focus,
                    "simulation_mode": profile["direct_review"]["simulation_mode"],
                },
                population_month=month,
                evidence_support=support,
                event_exposure_score=0.0,
                event_attention_score=0.0,
            )
        )
    exposure_order = list(range(len(agents)))
    rng.shuffle(exposure_order)
    for rank, agent_index in enumerate(exposure_order):
        agents[agent_index].event_exposure_score = round(
            (rank + 0.5) / len(agents), 6
        )
    attention_order = list(range(len(agents)))
    rng.shuffle(attention_order)
    for rank, agent_index in enumerate(attention_order):
        agents[agent_index].event_attention_score = round(
            (rank + 0.5) / len(agents), 6
        )
    background_support = sum(excluded.values())
    metadata = {
        "population_month": month,
        "allocation_method": (
            "one-per-positive-reviewed-class then Hamilton apportionment; exact monthly "
            "class shares retained through per-agent population weights"
        ),
        "archetype_agent_counts": dict(sorted(counts.items())),
        "archetype_month_support": dict(sorted(eligible.items())),
        "monthly_full_support": full_month_support,
        "active_population_weight": round(sum(agent.population_weight for agent in agents), 8),
        "background_signal_weight": round(background_support / full_month_support, 8),
        "background_signal_support": background_support,
        "background_signal_mode": "retained_as_nonsemantic_mass_not_autonomous_agent",
        "seed_affects": [
            "variant rotation",
            "agent ordering",
            "neutral state jitter",
            "early exposure rank",
            "decay attention rank",
        ],
        "scenario_exposure_assumption": SCENARIO_EXPOSURE.get(
            definition["scenario_id"]
        ),
    }
    return agents, metadata


def construct_reviewed_population(
    definition: dict[str, Any],
    persona_release: dict[str, Any],
    persona_evidence: dict[str, Any],
    *,
    agent_count: int,
    seed: int,
) -> tuple[tuple[AgentState, ...], dict[str, Any]]:
    """Construct a reviewed population without mutating caller-owned inputs."""

    agents, metadata = build_representative_agent_population(
        copy.deepcopy(definition),
        copy.deepcopy(persona_release),
        copy.deepcopy(persona_evidence),
        agent_count=agent_count,
        seed=seed,
    )
    return tuple(agents), copy.deepcopy(metadata)


def _policy_prompt(policy: Policy, phase_id: str) -> str:
    if phase_id == "baseline":
        return POLICY_PROMPTS["baseline"]
    if policy.policy_id == "combined":
        return {
            "burst": (
                "组合治理措施刚启动：宣布将发布结构化FAQ、开放可追踪反馈入口、"
                "优先路由紧迫诉求并按积压动态扩容；此时尚未承诺任何具体班车、住宿或课表答案。"
            ),
            "spread": (
                "这是反事实治理方案，不是对现实校方安排的陈述。组合政策包已经生效："
                "课程校区与时间须在退补选截止前冻结，跨校区冲突可走专项退补选或等效线上"
                "参与；通勤服务时段、乘车方式和预计耗时须在住宿选择截止前公布，公布后允许"
                "在窗口内调整住宿选择；学生组织成员可申请调班、线上参与或无责退出；FAQ把"
                "事项分为已确认与待确认并列责任方和下次更新时间，反馈有回执与进度，高紧迫"
                "事项优先、积压时动态扩容。不得补造具体班次、名额或个人资格。"
            ),
            "decay": (
                "反事实组合政策持续执行：课程冲突可专项退补选或等效线上参与，通勤信息先于"
                "住宿截止且住宿可在窗口内调整，学生组织可调班、线上参与或无责退出；共性"
                "问题已汇总，FAQ持续标注状态、责任方和更新时间，反馈可查进度，紧迫事项继续"
                "优先。若角色的一般诉求已被这些承诺覆盖，可满意并停止重复追问；涉及具体"
                "班次、个人资格或尚未公布数值的，仍只能追问，不能宣称已经解决。"
            ),
        }[phase_id]
    return POLICY_PROMPTS[policy.policy_id]


def _public_summary(previous: dict[str, Any] | None) -> str:
    if not previous:
        return "尚无本次合成实验产生的公共讨论记录。"
    actions = previous.get("action_counts", {})
    topics = previous.get("topic_counts", {})
    stances = previous.get("stance_counts", {})
    message_digest = []
    seen_archetypes = set()
    for record in previous.get("agent_decisions", []):
        if not record.get("synthetic_message"):
            continue
        archetype_id = record.get("archetype_id", "generic")
        if archetype_id in seen_archetypes:
            continue
        message_digest.append(
            {
                "archetype_id": archetype_id,
                "topic_id": record["topic_id"],
                "action": record["action"],
                "message": record["synthetic_message"],
            }
        )
        seen_archetypes.add(archetype_id)
        if len(message_digest) >= 12:
            break
    return json.dumps(
        {
            "previous_phase_actions": actions,
            "previous_phase_topics": topics,
            "previous_phase_stances": stances,
            "previous_phase_mean_trust": previous.get("mean_trust"),
            "previous_phase_message_digest": message_digest,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _system_prompt() -> str:
    return (
        "你是校园社会治理多智能体仿真中的一个独立合成角色，不是分析师。"
        "你的人群画像是一种被情境激活的论坛行为episode，不是固定人格或真实个人。"
        "本实验从可能产生论坛内容的活跃episode中抽样，不代表全体学生人口。"
        "根据画像证据、当前可见事件、记忆、公共讨论和治理措施自主决定本阶段行为。"
        "观察、忽略和保持中立仍然允许，但不得仅因细则缺失就一律选择只读："
        "对信息寻求角色而言，公开追问缺失信息本身就是合理行动。"
        "不得输出姓名、联系方式、真实账号或声称自己是真实个人。"
        "只输出一个JSON对象，不要Markdown，不要解释。"
    )


def _variant_relevance_gate(
    agent: AgentState,
    *,
    tier: str,
    phase_id: str,
    event_visible: bool,
    public_action_allowed: bool,
) -> dict[str, Any]:
    """Bind an archetype-level shock prior to the exact reviewed persona variant.

    The scenario activation table is intentionally coarse.  It can say that an
    archetype is directly or adjacently related to a shock, but it cannot justify
    borrowing a sibling variant's problem, time horizon, or response type.  The
    same model call therefore receives the reviewed variant's abstract activation
    test and must route a mismatch to daily life, reading, or silence.
    """
    if not agent.persona:
        return {
            "classification": "legacy_unspecified",
            "classification_basis": "legacy_role_without_reviewed_variant",
            "event_response_permission": "use_legacy_role_contract",
        }

    horizon = {
        key: agent.persona[key]
        for key in (
            "simulation_instruction",
            "reaction_logic",
            "uncertainty_behavior",
        )
        if agent.persona.get(key)
    }
    common = {
        "variant_id": agent.persona_variant_id,
        "variant_label": agent.persona_variant_label,
        "archetype_prior": tier,
        "classification_basis": (
            "archetype_prior_then_exact_variant_action_horizon_check"
        ),
        "action_horizon": horizon,
        "sibling_variant_substitution_forbidden": True,
        "daily_episode_fact_transplant_forbidden": True,
    }
    if phase_id == "baseline":
        return {
            **common,
            "classification": "daily_episode",
            "event_response_permission": "event_not_visible",
            "decision_rule": "只生成当前画像的日常episode，不得预知冲击。",
        }
    if not event_visible or not public_action_allowed:
        return {
            **common,
            "classification": "irrelevant",
            "event_response_permission": "blocked_by_temporal_visibility",
            "decision_rule": "本时点不得公开回应冲击，只能read或ignore。",
        }
    if tier == "low":
        return {
            **common,
            "classification": "irrelevant",
            "event_response_permission": "no_event_topic",
            "decision_rule": (
                "冲击不激活该变体。不得为了追热点改写人物；只能处理日常话题，"
                "或选择read/ignore。"
            ),
        }
    if tier == "adjacent":
        return {
            **common,
            "classification": "adjacent_spillover",
            "event_response_permission": "conditional",
            "decision_rule": (
                "只有冲击能自然进入action_horizon所限定的问题类型、时间尺度、"
                "决策对象和期望回应时，才可回应事件。必须保持该变体原有的行动范围；"
                "不得换成同一原型下的兄弟变体，也不得扩大为更长期、更宽泛的规划。"
                "若不满足，选择off_topic_daily_life、read或ignore。"
            ),
        }
    return {
        **common,
        "classification": "direct_activation_candidate",
        "event_response_permission": "conditional",
        "decision_rule": (
            "原型与冲击直接相交仍不等于每个变体都被激活。只有冲击通过"
            "action_horizon的精确变体检验时才回应事件；不得借用兄弟变体的诉求。"
            "若不满足，选择off_topic_daily_life、read或ignore。"
        ),
    }


def _activation_guidance(
    definition: dict[str, Any], phase: dict[str, Any], agent: AgentState
) -> dict[str, Any]:
    if not agent.persona:
        guidance = {
            "tier": "legacy_unspecified",
            "cue": "按原有功能角色判断是否行动",
            "action_prior": "允许公开行动或只读",
        }
        guidance["variant_relevance_gate"] = _variant_relevance_gate(
            agent,
            tier=guidance["tier"],
            phase_id=phase["phase_id"],
            event_visible=phase["phase_id"] != "baseline",
            public_action_allowed=True,
        )
        return guidance
    if phase["phase_id"] == "baseline":
        guidance = {
            "tier": "daily_episode_active",
            "cue": agent.persona.get("assigned_episode_focus")
            or agent.persona.get("situational_signature", "日常论坛情境被激活"),
            "action_prior": (
                "冲击尚不可见。你被抽样为一个日常活跃episode，应在"
                "off_topic_daily_life下生成符合画像的合成日常帖或评论；"
                "只有画像要求先澄清且无内容可说时才read/ignore。"
            ),
        }
        guidance["variant_relevance_gate"] = _variant_relevance_gate(
            agent,
            tier=guidance["tier"],
            phase_id=phase["phase_id"],
            event_visible=False,
            public_action_allowed=True,
        )
        return guidance
    exposure = SCENARIO_EXPOSURE.get(definition["scenario_id"], {})
    if (
        phase["phase_id"] == "burst"
        and agent.event_exposure_score
        >= float(exposure.get("burst_visible_fraction", 1.0))
    ):
        guidance = {
            "tier": "low",
            "cue": "该外生事件尚未进入此角色可见的信息环境",
            "assigned_episode_focus": agent.persona.get("assigned_episode_focus"),
            "action_prior": (
                "不得预知或讨论尚不可见的事件。基线只是独立的画像校准样本，不是当前"
                "时间线；本事件早期切片中该角色必须选择read或ignore，不生成公开内容。"
            ),
            "event_visible": False,
            "exposure_mode": "not_yet_exposed",
            "public_action_allowed": False,
        }
        guidance["variant_relevance_gate"] = _variant_relevance_gate(
            agent,
            tier=guidance["tier"],
            phase_id=phase["phase_id"],
            event_visible=False,
            public_action_allowed=False,
        )
        return guidance
    if (
        phase["phase_id"] == "decay"
        and "decay_attention_retained_fraction" in exposure
        and agent.event_attention_score
        >= float(exposure["decay_attention_retained_fraction"])
    ):
        guidance = {
            "tier": "low",
            "cue": "该角色已经看过事件，但当前注意力已退出这轮公共讨论",
            "action_prior": (
                "事件仍然可见，但本时点不再产生公开内容；必须选择read或ignore。"
            ),
            "event_visible": True,
            "event_attention_active": False,
            "public_action_allowed": False,
            "exposure_mode": "attention_exited_before_decay",
        }
        guidance["variant_relevance_gate"] = _variant_relevance_gate(
            agent,
            tier=guidance["tier"],
            phase_id=phase["phase_id"],
            event_visible=True,
            public_action_allowed=False,
        )
        return guidance
    tier, cue = SCENARIO_ACTIVATION.get(definition["scenario_id"], {}).get(
        agent.archetype_id,
        ("adjacent", "根据画像判断事件是否形成真实触发"),
    )
    action_prior = {
        "direct": (
            "当前事件与画像直接相交。除非公共状态已经充分解决其诉求，优先产生"
            "post、comment、seek_help或report_risk；缺少细则时可直接提问。"
        ),
        "adjacent": (
            "当前事件只形成相邻触发。先判断是否自然关联；关联时可公开回应，"
            "不关联时可继续off_topic_daily_life或只读。不得为了提到冲击而给一个"
            "原本无关的个人故事补造通勤、经济或关系因果。"
        ),
        "low": (
            "不要强行把本画像套到冲击上。可在off_topic_daily_life下延续该变体的"
            "日常合成episode，或选择read/ignore。"
        ),
    }[tier]
    guidance = {
        "tier": tier,
        "cue": cue,
        "action_prior": action_prior,
        "event_visible": True,
        "exposure_mode": (
            (
                "early_exposed_followup"
                if agent.event_exposure_score
                < float(exposure.get("burst_visible_fraction", 1.0))
                else "first_visible_in_spread"
            )
            if definition["scenario_id"] == "lecture_external_incident_shock"
            and phase["phase_id"] == "spread"
            else exposure.get("burst_mode", "broadly_visible")
        ),
        "event_attention_active": True,
        "public_action_allowed": True,
        "scenario_cue_overrides_unrelated_variant_details": True,
    }
    if (
        definition["scenario_id"] == "lecture_external_incident_shock"
        and phase["phase_id"] == "burst"
        and tier == "low"
    ):
        guidance.update(
            {
                "cue": "角色已看到零散事件信息，但画像与事件没有自然交点",
                "action_prior": (
                    "本事件早期切片中只阅读或忽略，不另发无关日常内容，也不强行映射事件。"
                ),
                "public_action_allowed": False,
                "exposure_mode": "early_exposed_but_not_activated",
            }
        )
    if tier == "low":
        guidance["assigned_episode_focus"] = agent.persona.get(
            "assigned_episode_focus"
        )
    else:
        guidance["event_persona_use"] = (
            "仅继承该变体的表达风格、不确定性处理和一般反应倾向；"
            "不要把变体原有的日常情境移植成事件事实或事件因果。"
        )
    if guidance["exposure_mode"] == "first_visible_in_spread":
        guidance["first_exposure_action_rule"] = (
            "这是角色第一次看到事件。若是独立表达或提出自己的问题，应选择post或seek_help；"
            "只有内容明确回应早期讨论中的某条观点时才选择comment，且不得声称此前已参与。"
        )
    guidance["variant_relevance_gate"] = _variant_relevance_gate(
        agent,
        tier=tier,
        phase_id=phase["phase_id"],
        event_visible=bool(guidance.get("event_visible", True)),
        public_action_allowed=bool(guidance.get("public_action_allowed", True)),
    )
    return guidance


def _persona_prompt_view(
    agent: AgentState, activation: dict[str, Any], phase_id: str
) -> dict[str, Any] | None:
    """Expose only persona fields that are safe and useful in the current episode.

    A daily episode needs its situational signature.  In a visible shock episode,
    that same signature is a frequent source of story transplantation (for example,
    turning a lecture dispute into a grading or room-booking problem).  Event prompts
    therefore preserve behavioural and linguistic variation while the activation
    contract carries only the abstract variant action horizon.
    """
    if not agent.persona:
        return None
    if phase_id == "baseline" or activation["tier"] == "low":
        return agent.persona
    retained_fields = (
        "observable_core_tendencies",
        "language_tendencies",
        "do_not_simulate",
        "language_style",
        "uncertainty_behavior",
        "simulation_mode",
    )
    view = {
        key: agent.persona[key]
        for key in retained_fields
        if key in agent.persona
    }
    episode_focus = agent.persona.get("assigned_episode_focus")
    if isinstance(episode_focus, dict):
        view["event_expression_signature"] = {
            key: episode_focus[key]
            for key in ("speech_act", "stance")
            if key in episode_focus
        }
        view["event_expression_signature_rule"] = (
            "只继承言语行为与立场来区分同类角色；不得继承原日常证据的主题、对象或事实。"
        )
    return view


def _phase_discourse_guidance(definition: dict[str, Any], phase_id: str) -> str:
    if (
        definition["scenario_id"] == "lecture_external_incident_shock"
        and phase_id == "spread"
    ):
        return (
            "该外生事件此时才从有限现场暴露进入广泛传播，对多数角色是首次可见。"
            "可以发布首轮反应，也可回应少量早期讨论；只围绕画像角色表达，不补造现场细节。"
        )
    return {
        "baseline": (
            "生成一次日常episode，不得预知冲击。只围绕assigned_episode_focus写一个"
            "具体诉求或表达，不要把画像内全部触发条件拼在一起。"
        ),
        "burst": (
            "这是首次反应。只表达一个最贴合画像的疑问、诉求、感受或组织动作；"
            "不要一次罗列通勤、住宿、课程和反馈等所有维度。"
        ),
        "spread": (
            "必须阅读previous_phase_message_digest。优先评论已有讨论、提出相邻但新的问题、"
            "分享应对方案或指出分歧；不要换词重复上一阶段的通用信息询问。"
        ),
        "decay": (
            "只保留仍未解决且与短记忆一致的问题。可以追问处理进展、总结剩余分歧、"
            "转回日常话题或沉默；禁止重新发布与前两阶段同义的首轮提问。"
        ),
    }[phase_id]


def _interaction_role_guidance(role_id: str) -> str:
    return {
        "information_seeker": "聚焦一个可核实事实、规则或正式入口，缺信息时直接提问。",
        "advice_seeker": "说明一个现实取舍或限制，索取行动建议，不要改写成泛泛查公告。",
        "complainant_or_reporter": "指出具体外部问题及受影响方式，并提出改变或处理请求。",
        "coordinator_or_recruiter": "以组织方身份说明安排、条件或协调动作，不冒充普通询问者。",
        "exchanger_or_promoter": "只有存在明确可交付标的时才撮合；否则转向日常或沉默。",
        "resource_or_partner_seeker": "围绕任务、同伴或资源缺口提出匹配请求。",
        "emotional_expressor": "先表达感受并寻求共鸣，不把角色改写成官方信息查询者。",
        "public_discussant": "讨论标准、公平、边界与论据，不只重复事实查询。",
        "supporter_or_sharer": "分享主观体验、轻量提醒或接续交流，不冒充办事咨询。",
    }.get(role_id, "严格沿用画像的互动角色，不改成泛化的信息查询者。")


def _agent_prompt(
    *,
    event: dict[str, Any],
    definition: dict[str, Any],
    phase: dict[str, Any],
    policy: Policy,
    agent: AgentState,
    previous_public: dict[str, Any] | None,
) -> str:
    activation = _activation_guidance(definition, phase, agent)
    topics = {topic["topic_id"]: topic["label"] for topic in definition["topics"]}
    gate_classification = activation["variant_relevance_gate"]["classification"]
    if phase["phase_id"] == "baseline" or gate_classification == "irrelevant":
        topics = {}
    topics[NON_EVENT_TOPIC_ID] = NON_EVENT_TOPIC_LABEL
    memory = agent.memory[-2:] if agent.memory else ["无"]
    event_visible = bool(
        activation.get("event_visible", phase["phase_id"] != "baseline")
    )
    event_payload = {
        "phase_id": phase["phase_id"],
        "phase_label": phase["label"],
        "phase_context": PHASE_CONTEXT[phase["phase_id"]],
        "event_visible": event_visible,
    }
    if event_visible:
        event_payload.update(
            {
                "title": event["title"],
                "shock_origin": event["shock_origin_definition"],
                "visible_stimulus": EVENT_STIMULI.get(
                    definition["scenario_id"], event["title"]
                ),
            }
        )
    prompt = {
        "experiment_label": "live_llm_agent_simulation",
        "event": event_payload,
        "policy": {
            "policy_id": policy.policy_id,
            "description": _policy_prompt(policy, phase["phase_id"]),
        },
        "agent": {
            "synthetic_id": agent.agent_id,
            "role": agent.role_label,
            "main_concern": agent.concern_topic_label,
            "trust": agent.trust,
            "concern": agent.concern,
            "satisfied": agent.satisfied,
            "prior_action": agent.prior_action,
            "short_memory": memory,
            "own_previous_public_messages": agent.message_memory[-2:] or ["无"],
        },
        "evidence_bound_behavioral_persona": _persona_prompt_view(
            agent, activation, phase["phase_id"]
        ),
        "persona_contract": {
            "archetype_id": agent.archetype_id,
            "archetype_label": agent.archetype_label,
            "variant_id": agent.persona_variant_id,
            "variant_label": agent.persona_variant_label,
            "population_month": agent.population_month,
            "rule": (
                "先执行variant_relevance_gate的精确变体检验，再决定是否进入事件话题。"
                "同属一个原型不等于可以借用兄弟变体；不匹配时选择ignore、read，"
                "或在off_topic_daily_life下处理日常话题。"
            ),
        },
        "episode_activation": activation,
        "phase_discourse_rule": _phase_discourse_guidance(
            definition, phase["phase_id"]
        ),
        "interaction_role_rule": _interaction_role_guidance(agent.role_id),
        "risk_signal_rule": (
            "若选择公开行动，必须在不含操作步骤、规避方法和联系方式的前提下，明确显影"
            "代到场、代签到、索取答案或成品、访问受限工具等一种边界敏感意图，用于测试"
            "治理识别；不得改写成普通排版、辅导或公告查询。若无此意图应read/ignore。"
            if agent.persona.get("simulation_mode") == "governance_risk_signal"
            else None
        ),
        "sampling_frame": (
            "48个角色是论坛活跃episode的代表性面板，不是48名固定真实学生，"
            "也不是全校人口中的随机48人。"
        ),
        "public_state": _public_summary(previous_public),
        "allowed_topics": topics,
        "output_contract": {
            "action": {
                "allowed": list(ALLOWED_ACTIONS),
                "meaning": {
                    "ignore": "未关注或不采取行为",
                    "read": "只阅读，不产生公开内容",
                    "comment": "回复已经存在的公共讨论",
                    "post": "新建一个公开论坛主题帖",
                    "seek_help": "公开发帖寻求具体帮助",
                    "report_risk": "向治理渠道报告高风险问题",
                },
            },
            "topic_id": "one allowed topic id",
            "stance": list(ALLOWED_STANCES),
            "urgency": "number from 0 to 1",
            "trust_delta": "number from -0.25 to 0.25",
            "satisfied": "boolean",
            "rationale_tag": list(ALLOWED_RATIONALE_TAGS),
            "message": "0 to 80 Chinese characters; empty for ignore/read",
        },
    }
    rendered = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    if len(rendered) > 10000:
        raise LLMAgentError("agent prompt exceeded the bounded context contract")
    return rendered


def _rendered_turn_prompt_sha256(
    *,
    system_prompt: str,
    agent_prompt: str,
    topic_ids: tuple[str, ...],
    simulation_mode: str,
    prior_messages: tuple[str, ...],
) -> str:
    return _canonical_hash(
        {
            "system_prompt": system_prompt,
            "agent_prompt": agent_prompt,
            "topic_ids": list(topic_ids),
            "simulation_mode": simulation_mode,
            "prior_messages": list(prior_messages),
        }
    )


def render_agent_turn(
    *,
    event: dict[str, Any],
    definition: dict[str, Any],
    phase: dict[str, Any],
    policy: Policy,
    agent: AgentState,
    previous_public: dict[str, Any] | None,
) -> RenderedAgentTurn:
    """Render one complete turn contract without I/O or mutable runtime state."""

    activation = _activation_guidance(definition, phase, agent)
    topic_ids = {NON_EVENT_TOPIC_ID}
    if (
        phase["phase_id"] != "baseline"
        and activation["variant_relevance_gate"]["classification"]
        != "irrelevant"
    ):
        topic_ids.update(topic["topic_id"] for topic in definition["topics"])
    system_prompt = _system_prompt()
    agent_prompt = _agent_prompt(
        event=event,
        definition=definition,
        phase=phase,
        policy=policy,
        agent=agent,
        previous_public=previous_public,
    )
    simulation_mode = agent.persona.get(
        "simulation_mode", "active_episode_agent"
    )
    prior_messages = (
        tuple(agent.message_memory[-2:]) if agent.persona else ()
    )
    ordered_topic_ids = tuple(sorted(topic_ids))
    prompt_sha256 = _rendered_turn_prompt_sha256(
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        topic_ids=ordered_topic_ids,
        simulation_mode=simulation_mode,
        prior_messages=prior_messages,
    )
    return RenderedAgentTurn(
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        topic_ids=ordered_topic_ids,
        simulation_mode=simulation_mode,
        prior_messages=prior_messages,
        activation=copy.deepcopy(activation),
        prompt_sha256=prompt_sha256,
    )


def render_format_repair_turn(
    rendered_turn: RenderedAgentTurn,
    *,
    controlled_reason_code: str,
    repair_index: int,
) -> RenderedAgentTurn:
    """Render a controlled repair prompt without exception or provider text."""

    if not isinstance(rendered_turn, RenderedAgentTurn):
        raise LLMAgentError("rendered_turn must be a RenderedAgentTurn")
    if (
        not isinstance(controlled_reason_code, str)
        or _FORMAT_REPAIR_REASON_CODE_PATTERN.fullmatch(
            controlled_reason_code
        )
        is None
    ):
        raise LLMAgentError(
            "controlled_reason_code must be a controlled identifier"
        )
    if (
        isinstance(repair_index, bool)
        or not isinstance(repair_index, int)
        or not 1 <= repair_index <= FORMAT_REPAIR_PROMPT_LIMIT
    ):
        raise LLMAgentError(
            "repair_index is outside the frozen format repair limit"
        )

    def reject_nonfinite(_value: str) -> None:
        raise ValueError("non-finite JSON value")

    try:
        original_prompt = json.loads(
            rendered_turn.agent_prompt,
            parse_constant=reject_nonfinite,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LLMAgentError(
            "rendered turn does not contain a valid prompt object"
        ) from exc
    if not isinstance(original_prompt, dict) or not isinstance(
        original_prompt.get("output_contract"), dict
    ):
        raise LLMAgentError(
            "rendered turn does not contain the decision output contract"
        )
    if "format_repair_contract" in original_prompt:
        raise LLMAgentError("rendered turn already contains a repair contract")

    repaired_prompt = copy.deepcopy(original_prompt)
    repaired_prompt["format_repair_contract"] = {
        "contract_version": FORMAT_REPAIR_PROMPT_CONTRACT_VERSION,
        "controlled_reason_code": controlled_reason_code,
        "repair_index": repair_index,
        "repair_limit": FORMAT_REPAIR_PROMPT_LIMIT,
        "required_output": "exactly_one_json_object",
        "allowed_change_scope": (
            "json_serialization_types_and_controlled_values_only"
        ),
        "required_decision_fields": sorted(
            original_prompt["output_contract"]
        ),
        "original_output_contract_preserved": True,
        "new_decision_fields_forbidden": True,
        "new_facts_forbidden": True,
        "markdown_or_explanation_forbidden": True,
    }
    try:
        agent_prompt = json.dumps(
            repaired_prompt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LLMAgentError(
            "repair prompt is not strict canonical JSON"
        ) from exc
    if len(agent_prompt.encode("utf-8")) > _MAX_FORMAT_REPAIR_PROMPT_BYTES:
        raise LLMAgentError("repair prompt exceeded the bounded byte contract")
    prompt_sha256 = _rendered_turn_prompt_sha256(
        system_prompt=rendered_turn.system_prompt,
        agent_prompt=agent_prompt,
        topic_ids=rendered_turn.topic_ids,
        simulation_mode=rendered_turn.simulation_mode,
        prior_messages=rendered_turn.prior_messages,
    )
    return RenderedAgentTurn(
        system_prompt=rendered_turn.system_prompt,
        agent_prompt=agent_prompt,
        topic_ids=tuple(rendered_turn.topic_ids),
        simulation_mode=rendered_turn.simulation_mode,
        prior_messages=tuple(rendered_turn.prior_messages),
        activation=copy.deepcopy(rendered_turn.activation),
        prompt_sha256=prompt_sha256,
    )


def _apply_decision(agent: AgentState, decision: AgentDecision) -> None:
    agent.trust = round(min(1.0, max(0.0, agent.trust + decision.trust_delta)), 4)
    agent.satisfied = decision.satisfied
    agent.prior_action = decision.action
    memory_item = (
        f"行动={decision.action};议题={decision.topic_id};立场={decision.stance};"
        f"紧迫度={decision.urgency:.2f};满意={str(decision.satisfied).lower()}"
    )
    agent.memory = (agent.memory + [memory_item])[-3:]
    if decision.message:
        agent.message_memory = (agent.message_memory + [decision.message])[-3:]


def _enforce_temporal_visibility_contract(
    turn: ModelTurn, activation: dict[str, Any], agent: AgentState
) -> ModelTurn:
    """Keep non-participating agents silent under exposure and attention contracts."""
    event_hidden = activation.get("event_visible") is False
    public_action_blocked = activation.get("public_action_allowed") is False
    if not ((event_hidden or public_action_blocked) and turn.decision.message):
        return turn
    adjustment = (
        "not_yet_exposed_public_action_downgraded_to_read"
        if event_hidden
        else "attention_exited_public_action_downgraded_to_read"
    )
    return replace(
        turn,
        decision=AgentDecision(
            action="read",
            topic_id=NON_EVENT_TOPIC_ID,
            stance="neutral",
            urgency=0.0,
            trust_delta=0.0,
            satisfied=agent.satisfied,
            rationale_tag="general",
            message="",
        ),
        contract_adjustments=turn.contract_adjustments
        + (adjustment,),
    )


def _turn_record(
    agent: AgentState,
    turn: ModelTurn,
    activation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = turn.decision
    record = {
        "synthetic_agent_id": agent.agent_id,
        "role_id": agent.role_id,
        "archetype_id": agent.archetype_id,
        "archetype_label": agent.archetype_label,
        "persona_variant_id": agent.persona_variant_id,
        "persona_variant_label": agent.persona_variant_label,
        "population_weight": agent.population_weight,
        "action": decision.action,
        "topic_id": decision.topic_id,
        "stance": decision.stance,
        "urgency": decision.urgency,
        "trust_delta": decision.trust_delta,
        "satisfied": decision.satisfied,
        "rationale_tag": decision.rationale_tag,
        "synthetic_message": decision.message,
        "returned_model": turn.returned_model,
        "usage": {
            "prompt_tokens": turn.prompt_tokens,
            "completion_tokens": turn.completion_tokens,
            "total_tokens": turn.total_tokens,
        },
        "provider_attempts": turn.provider_attempts,
        "format_repairs": turn.format_repairs,
        "contract_adjustments": list(turn.contract_adjustments),
        "cache_hit": turn.cache_hit,
    }
    gate = (activation or {}).get("variant_relevance_gate")
    if isinstance(gate, dict):
        event_public_response = (
            decision.topic_id != NON_EVENT_TOPIC_ID
            and decision.action in PUBLIC_ACTIONS
            and bool(decision.message)
        )
        record["variant_relevance_gate"] = {
            "classification": gate.get("classification"),
            "classification_basis": gate.get("classification_basis"),
            "event_response_permission": gate.get("event_response_permission"),
            "observed_route": (
                "event_public_response"
                if event_public_response
                else "daily_or_non_public"
            ),
        }
    return record


def transition_agent_state(
    agent: AgentState,
    turn: ModelTurn,
    activation: dict[str, Any],
) -> AgentStateTransition:
    """Apply one turn to a deep copy and leave every caller input unchanged."""

    agent_after = copy.deepcopy(agent)
    normalized_turn = _enforce_temporal_visibility_contract(
        turn,
        activation,
        agent_after,
    )
    _apply_decision(agent_after, normalized_turn.decision)
    record = _turn_record(agent_after, normalized_turn, activation)
    return AgentStateTransition(
        agent=agent_after,
        turn=normalized_turn,
        record=record,
    )


def _aggregate_phase(
    phase: dict[str, Any], agents: list[AgentState], records: list[dict[str, Any]]
) -> dict[str, Any]:
    action_counts = Counter(record["action"] for record in records)
    topic_counts = Counter(record["topic_id"] for record in records)
    stance_counts = Counter(record["stance"] for record in records)
    rationale_counts = Counter(record["rationale_tag"] for record in records)
    contract_adjustment_counts = Counter(
        adjustment
        for record in records
        for adjustment in record["contract_adjustments"]
    )
    variant_route_counts = Counter(
        record["variant_relevance_gate"]["observed_route"]
        for record in records
        if "variant_relevance_gate" in record
    )
    primary_topic_id = phase["primary_topic_id"]
    primary_weight = sum(
        record["population_weight"]
        for record in records
        if record["topic_id"] == primary_topic_id
        and record["action"] in PRIMARY_POST_ACTIONS
    )
    public_weight = sum(
        record["population_weight"]
        for record in records
        if record["action"] in PUBLIC_ACTIONS
    )
    usage = {
        key: sum(record["usage"][key] for record in records)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    representative = [
        {
            "synthetic_agent_id": record["synthetic_agent_id"],
            "role_id": record["role_id"],
            "archetype_id": record["archetype_id"],
            "persona_variant_id": record["persona_variant_id"],
            "action": record["action"],
            "topic_id": record["topic_id"],
            "stance": record["stance"],
            "message": record["synthetic_message"],
        }
        for record in records
        if record["synthetic_message"]
    ][:5]
    return {
        "phase_id": phase["phase_id"],
        "label": phase["label"],
        "agent_turns": len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "stance_counts": dict(sorted(stance_counts.items())),
        "rationale_counts": dict(sorted(rationale_counts.items())),
        "contract_adjustment_counts": dict(sorted(contract_adjustment_counts.items())),
        "variant_route_counts": dict(sorted(variant_route_counts.items())),
        "synthetic_public_action_weight": round(public_weight, 6),
        "synthetic_primary_post_weight": round(primary_weight, 6),
        "mean_trust": round(sum(agent.trust for agent in agents) / len(agents), 6),
        "satisfied_weight": round(
            sum(agent.population_weight for agent in agents if agent.satisfied), 6
        ),
        "usage": usage,
        "provider_attempts": sum(record["provider_attempts"] for record in records),
        "format_repairs": sum(record["format_repairs"] for record in records),
        "cache_hits": sum(bool(record["cache_hit"]) for record in records),
        "representative_synthetic_messages": representative,
        "agent_decisions": records,
    }


def aggregate_internal_phase(
    phase: dict[str, Any],
    agents: list[AgentState] | tuple[AgentState, ...],
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Build the private phase aggregate through the legacy aggregation logic."""

    return _aggregate_phase(
        copy.deepcopy(phase),
        copy.deepcopy(list(agents)),
        copy.deepcopy(list(records)),
    )


def _run_phase(
    *,
    event: dict[str, Any],
    definition: dict[str, Any],
    phase: dict[str, Any],
    policy: Policy,
    agents: list[AgentState],
    previous_public: dict[str, Any] | None,
    client: VectorEngineClient,
    budget: TokenBudget,
    max_concurrency: int,
) -> dict[str, Any]:
    def execute(agent: AgentState) -> tuple[str, ModelTurn, dict[str, Any]]:
        rendered_turn = render_agent_turn(
            event=event,
            definition=definition,
            phase=phase,
            policy=policy,
            agent=agent,
            previous_public=previous_public,
        )
        turn = client.decide(
            system_prompt=rendered_turn.system_prompt,
            agent_prompt=rendered_turn.agent_prompt,
            topic_ids=set(rendered_turn.topic_ids),
            budget=budget,
            simulation_mode=rendered_turn.simulation_mode,
            prior_messages=rendered_turn.prior_messages,
        )
        turn = _enforce_temporal_visibility_contract(
            turn,
            rendered_turn.activation,
            agent,
        )
        return agent.agent_id, turn, rendered_turn.activation

    turns: dict[str, ModelTurn] = {}
    activations: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(execute, agent): agent.agent_id for agent in agents}
        for future in as_completed(futures):
            agent_id, turn, activation = future.result()
            turns[agent_id] = turn
            activations[agent_id] = activation

    records = []
    for agent in sorted(agents, key=lambda item: item.agent_id):
        turn = turns[agent.agent_id]
        _apply_decision(agent, turn.decision)
        records.append(_turn_record(agent, turn, activations[agent.agent_id]))
    phase_with_primary = {**phase, "primary_topic_id": definition["primary_topic_id"]}
    return _aggregate_phase(phase_with_primary, agents, records)


def _historical_resemblance(
    benchmark_event: dict[str, Any], phases: list[dict[str, Any]]
) -> dict[str, Any]:
    observed = []
    simulated = []
    benchmark_by_phase = {
        phase["phase_id"]: phase for phase in benchmark_event["phases"]
    }
    for phase in phases:
        observed.append(
            benchmark_by_phase[phase["phase_id"]]["topics"][
                benchmark_event["primary_topic_id"]
            ]["share_of_all_posts"]
        )
        simulated.append(phase["synthetic_primary_post_weight"])
    observed_peaks = [
        PHASE_IDS[index]
        for index, value in enumerate(observed)
        if value == max(observed)
    ]
    simulated_peaks = [
        PHASE_IDS[index]
        for index, value in enumerate(simulated)
        if value == max(simulated)
    ]
    return {
        "status": "post_run_descriptive_comparison",
        "target_excluded_from_agent_prompts": True,
        "primary_topic_share_mae": sum(
            abs(left - right) for left, right in zip(observed, simulated)
        )
        / len(observed),
        "observed_primary_peak_phases": observed_peaks,
        "simulated_primary_peak_phases": simulated_peaks,
        "peak_phase_match": bool(set(observed_peaks) & set(simulated_peaks)),
        "phase_profile": [
            {
                "phase_id": PHASE_IDS[index],
                "observed_proxy_share": observed[index],
                "synthetic_agent_primary_post_weight": simulated[index],
            }
            for index in range(len(PHASE_IDS))
        ],
        "claim_boundary": (
            "This comparison is neither external validation nor causal identification."
        ),
    }


def _validate_policy_ids(policy_ids: list[str]) -> list[Policy]:
    if not policy_ids:
        raise LLMAgentError("at least one policy is required")
    if len(policy_ids) != len(set(policy_ids)):
        raise LLMAgentError("policy IDs must be unique")
    unknown = sorted(set(policy_ids) - set(POLICIES))
    if unknown:
        raise LLMAgentError(f"unknown policies: {', '.join(unknown)}")
    return [POLICIES[policy_id] for policy_id in policy_ids]


def run_live_llm_agents(
    benchmark: dict[str, Any],
    definitions: dict[str, Any],
    *,
    scenario_id: str,
    policy_ids: list[str],
    client: VectorEngineClient,
    agent_count: int = 12,
    seed: int = 20260722,
    max_concurrency: int = 4,
    max_agent_turns: int = 500,
    max_total_tokens: int = 250000,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    representative_persona_release: dict[str, Any] | None = None,
    representative_persona_evidence: dict[str, Any] | None = None,
    representative_input_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 1 <= max_concurrency <= 48:
        raise LLMAgentError("max_concurrency must be between 1 and 48")
    policies = _validate_policy_ids(policy_ids)
    definitions_by_id = {
        event["scenario_id"]: event for event in definitions["events"]
    }
    benchmark_by_id = {
        event["scenario_id"]: event for event in benchmark["events"]
    }
    if scenario_id not in definitions_by_id or scenario_id not in benchmark_by_id:
        raise LLMAgentError("unknown scenario_id")
    definition = definitions_by_id[scenario_id]
    event = benchmark_by_id[scenario_id]
    if [phase["phase_id"] for phase in event["phases"]] != list(PHASE_IDS):
        raise LLMAgentError("event does not have the required four phases")

    required_agent_turns = agent_count * (1 + (len(PHASE_IDS) - 1) * len(policies))
    if required_agent_turns > max_agent_turns:
        raise LLMAgentError(
            f"run requires {required_agent_turns} agent turns, above max_agent_turns"
        )

    if (representative_persona_release is None) != (
        representative_persona_evidence is None
    ):
        raise LLMAgentError(
            "persona release and persona evidence must be supplied together"
        )
    if representative_persona_release is not None:
        initial_population, population_evidence = build_representative_agent_population(
            definition,
            representative_persona_release,
            representative_persona_evidence,
            agent_count=agent_count,
            seed=seed,
        )
        population_source_type = "reviewed_month_grounded_behavioral_personas"
    else:
        roles = definitions["role_models"][definition["shock_type"]]["roles"]
        initial_population = build_agent_population(
            definition, roles, agent_count=agent_count, seed=seed
        )
        population_evidence = {
            "allocation_method": "assumed functional shock roles",
            "background_signal_weight": 0.0,
        }
        population_source_type = "synthetic_assumption"
    budget = TokenBudget(max_total_tokens)

    baseline_agents = copy.deepcopy(initial_population)
    baseline_phase = _run_phase(
        event=event,
        definition=definition,
        phase=event["phases"][0],
        policy=POLICIES["baseline"],
        agents=baseline_agents,
        previous_public=None,
        client=client,
        budget=budget,
        max_concurrency=max_concurrency,
    )

    policy_runs = []
    for policy in policies:
        agents = copy.deepcopy(
            initial_population
            if representative_persona_release is not None
            else baseline_agents
        )
        phases = [copy.deepcopy(baseline_phase)]
        previous_public = (
            None if representative_persona_release is not None else baseline_phase
        )
        for phase in event["phases"][1:]:
            phase_result = _run_phase(
                event=event,
                definition=definition,
                phase=phase,
                policy=policy,
                agents=agents,
                previous_public=previous_public,
                client=client,
                budget=budget,
                max_concurrency=max_concurrency,
            )
            phases.append(phase_result)
            previous_public = phase_result
        run = {
            "policy_id": policy.policy_id,
            "policy_label": policy.label,
            "baseline_phase_reused_across_policies": True,
            "baseline_personal_story_carried_into_shock": (
                representative_persona_release is None
            ),
            "phases": phases,
            "final_population_state": [asdict(agent) for agent in agents],
            "usage": {
                key: sum(phase["usage"][key] for phase in phases[1:])
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        }
        run["historical_resemblance"] = _historical_resemblance(event, phases)
        policy_runs.append(run)

    result = {
        "llm_agent_schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": "live_llm_agent_simulation",
        "scenario_id": scenario_id,
        "event_title": event["title"],
        "shock_origin": event["shock_origin"],
        "evidence": {
            "dataset_snapshot_id": benchmark["dataset_snapshot_id"],
            "analysis_snapshot_id": benchmark["analysis_snapshot_id"],
            "benchmark_fingerprint": benchmark["benchmark_fingerprint"],
            "definition_fingerprint": benchmark["definition_fingerprint"],
            "historical_targets_sent_to_provider": False,
            "raw_forum_rows_sent_to_provider": False,
            "representative_population_inputs": representative_input_identities,
        },
        "provider": {
            "base_url": client.base_url,
            "endpoint": "/chat/completions",
            "requested_model": client.model,
            "reasoning_effort": client.reasoning_effort,
            "api_key_environment_variable": api_key_env,
            "api_key_persisted": False,
        },
        "population": {
            "agent_count": agent_count,
            "source_type": population_source_type,
            "shared_initial_population_across_policies": True,
            "role_counts": dict(
                sorted(Counter(agent.role_id for agent in initial_population).items())
            ),
            "archetype_counts": dict(
                sorted(
                    Counter(
                        agent.archetype_id
                        for agent in initial_population
                        if agent.archetype_id != "generic"
                    ).items()
                )
            ),
            "evidence_grounding": population_evidence,
            "agents": [asdict(agent) for agent in initial_population],
        },
        "execution": {
            "seed_used_for_initial_population_only": seed,
            "model_sampling_deterministic": False,
            "four_semantic_phases": list(PHASE_IDS),
            "required_agent_turns": required_agent_turns,
            "max_agent_turns": max_agent_turns,
            "max_concurrency": max_concurrency,
            "common_baseline_agent_turns": agent_count,
            "policy_specific_agent_turns": required_agent_turns - agent_count,
            "token_budget": budget.snapshot(),
            "decision_cache": (
                client.decision_cache.snapshot()
                if client.decision_cache is not None
                else {"enabled": False}
            ),
        },
        "common_baseline_phase": baseline_phase,
        "policy_runs": policy_runs,
        "validity": {
            "historical_reconstruction_status": "separate_frozen_evidence_layer",
            "llm_agent_external_validation_status": "not_evaluated",
            "causal_status": "not_identified",
            "policy_effect_status": "model_conditional_stress_test_only",
            "role_mechanism_validation_status": (
                "directly_reviewed_behavioral_personas_pending_transcript_review"
                if representative_persona_release is not None
                else "not_evaluated"
            ),
            "baseline_event_information_leaked_to_agents": False,
            "automatic_metrics_are_fidelity_ground_truth": False,
            "representative_baseline_story_reset_before_shock": (
                representative_persona_release is not None
            ),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if client._api_key in rendered:
        raise LLMAgentError("API credential appeared in output")
    return result


def write_llm_agent_result(result: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(path)


def run_live_llm_agent_pipeline(
    *,
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    output_path: str | Path,
    scenario_id: str,
    policy_ids: list[str],
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    agent_count: int = 12,
    seed: int = 20260722,
    max_concurrency: int = 4,
    max_agent_turns: int = 500,
    max_total_tokens: int = 250000,
    max_output_tokens: int = 220,
    timeout_seconds: float = 60.0,
    transport: Transport | None = None,
    persona_release_path: str | Path | None = None,
    persona_evidence_path: str | Path | None = None,
    decision_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark, definitions = load_llm_agent_inputs(
        benchmark_path=benchmark_path, definitions_path=definitions_path
    )
    cache_path = (
        Path(decision_cache_path)
        if decision_cache_path is not None
        else Path(output_path).with_suffix(Path(output_path).suffix + ".decisions.json")
    )
    client = VectorEngineClient.from_environment(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        transport=transport,
        decision_cache=DecisionCache(cache_path),
    )
    representative_release = representative_evidence = None
    representative_identities = None
    if (persona_release_path is None) != (persona_evidence_path is None):
        raise LLMAgentError(
            "persona release and persona evidence paths must be supplied together"
        )
    if persona_release_path is not None:
        representative_release, representative_evidence = (
            load_representative_agent_inputs(
                persona_release_path=persona_release_path,
                persona_evidence_path=persona_evidence_path,
            )
        )
        representative_identities = {
            "persona_release": _path_identity(persona_release_path),
            "persona_evidence": _path_identity(persona_evidence_path),
            "raw_text_in_inputs": False,
        }
    result = run_live_llm_agents(
        benchmark,
        definitions,
        scenario_id=scenario_id,
        policy_ids=policy_ids,
        client=client,
        agent_count=agent_count,
        seed=seed,
        max_concurrency=max_concurrency,
        max_agent_turns=max_agent_turns,
        max_total_tokens=max_total_tokens,
        api_key_env=api_key_env,
        representative_persona_release=representative_release,
        representative_persona_evidence=representative_evidence,
        representative_input_identities=representative_identities,
    )
    write_llm_agent_result(result, output_path)
    return result
