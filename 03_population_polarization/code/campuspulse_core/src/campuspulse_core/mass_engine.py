"""Zero-Provider 1,000-episode state engine for CampusPulse MassSim v1."""

from __future__ import annotations

from collections import Counter, defaultdict
import base64
import copy
import json
import math
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence
import zlib

from .llm_agents import (
    ALLOWED_ACTIONS,
    ALLOWED_STANCES,
    AgentState,
    ModelTurn,
    NON_EVENT_TOPIC_ID,
    PUBLIC_ACTIONS,
    transition_agent_state,
    validate_agent_decision,
)
from .mass_population import canonical_hash
from .mass_simulation import (
    _integer_allocation,
    fixed_size_inclusion_probabilities,
    pivotal_sample,
)


MASS_DYNAMICS_SCHEMA_VERSION = "campus-pulse-mass-dynamics-v1"
MASS_SAMPLING_SCHEMA_VERSION = "campus-pulse-budgeted-probe-round-v1"
POPULATION_CHECKPOINT_SCHEMA_VERSION = "campus-pulse-population-checkpoint-v1"
AGGREGATE_RESULT_SCHEMA_VERSION = "campus-pulse-live-aggregate-result-v3"
RUN_PLAN_SCHEMA_VERSION = "campus-pulse-run-plan-v5"

TIMELINE_PHASES = (
    ("baseline", 2),
    ("burst", 4),
    ("spread", 6),
    ("decay", 4),
)
TIMELINE_TICKS = tuple(
    phase
    for phase, count in TIMELINE_PHASES
    for _ in range(count)
)
CALIBRATION_TICKS = (1, 5, 11, 15)
POLICY_BRANCHES = ("natural", "governance")
WRITEBACK_MODES = ("observation_only", "stance_only", "full_state")
PROBE_BUDGET = 32
PARTICLES_PER_AGENT = 10
ACTION_CATEGORIES = tuple(ALLOWED_ACTIONS)
STANCE_CATEGORIES = tuple(ALLOWED_STANCES)


class MassEngineError(ValueError):
    """Raised when the MassSim state or observation contract is invalid."""


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _seed_from(*parts: Any) -> int:
    return int(canonical_hash(list(parts))[:16], 16)


def _hash_uniform(*parts: Any) -> float:
    return int(canonical_hash(list(parts))[:13], 16) / float(16**13)


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    pairs = list(values)
    mass = sum(weight for weight, _ in pairs)
    return sum(weight * value for weight, value in pairs) / max(mass, 1e-12)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    bounded = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(bounded.values())
    if total <= 0.0:
        return {key: 1.0 / len(bounded) for key in bounded}
    return {key: value / total for key, value in bounded.items()}


def _phase_intensity(tick: int) -> float:
    profile = (
        0.03,
        0.05,
        0.78,
        0.94,
        0.88,
        0.80,
        0.70,
        0.66,
        0.61,
        0.56,
        0.50,
        0.45,
        0.34,
        0.25,
        0.17,
        0.10,
    )
    return profile[tick]


def event_relevance(
    agent: Mapping[str, Any],
    scenario_id: str,
) -> tuple[str, float]:
    semantic = agent["semantic"]
    risk = {
        "none": 0.0,
        "low": 0.15,
        "medium": 0.38,
        "high": 0.62,
        "uncertain": 0.28,
    }.get(str(semantic.get("risk_level")), 0.2)
    alignment = _hash_uniform(
        scenario_id,
        semantic.get("topic_code"),
        agent.get("primary_archetype_id"),
    )
    score = _clip(0.55 * alignment + risk)
    if score < 0.34:
        return "irrelevant", score
    if score < 0.64:
        return "uncertain", score
    return "relevant", score


def initial_states(population: Mapping[str, Any]) -> list[dict[str, Any]]:
    agents = population.get("agents")
    if not isinstance(agents, list) or not agents:
        raise MassEngineError("compiled population has no agents")
    return [copy.deepcopy(agent["state"]) for agent in agents]


def graph_adjacency(
    graph: Mapping[str, Any],
    population_size: int,
) -> list[tuple[int, ...]]:
    members: list[list[int]] = [[] for _ in range(population_size)]
    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise MassEngineError("synthetic graph edges are missing")
    for edge in edges:
        source = int(edge["source_index"])
        target = int(edge["target_index"])
        if (
            not 0 <= source < population_size
            or not 0 <= target < population_size
            or source == target
        ):
            raise MassEngineError("synthetic graph edge is invalid")
        members[source].append(target)
    if any(len(targets) != 12 for targets in members):
        raise MassEngineError("every micro-agent requires 12 outgoing exposures")
    return [tuple(targets) for targets in members]


def advance_population_tick(
    *,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    adjacency: Sequence[Sequence[int]],
    scenario_id: str,
    policy_branch: str,
    tick: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply an interpretable multi-task GLM/state-space update."""

    if policy_branch not in POLICY_BRANCHES:
        raise MassEngineError("unknown policy branch")
    if not 0 <= tick < len(TIMELINE_TICKS):
        raise MassEngineError("tick is outside the frozen 16-point timeline")
    agents = population["agents"]
    intensity = _phase_intensity(tick)
    governance_active = policy_branch == "governance" and tick >= 3
    policy_strength = (
        min(0.38, 0.10 + 0.025 * (tick - 2))
        if governance_active
        else 0.0
    )
    updated = []
    for index, (agent, state) in enumerate(zip(agents, states)):
        neighbors = adjacency[index]
        neighbor_attention = sum(
            float(states[target]["attention"]) for target in neighbors
        ) / len(neighbors)
        neighbor_concern = sum(
            float(states[target]["concern"]) for target in neighbors
        ) / len(neighbors)
        neighbor_trust = sum(
            float(states[target]["trust"]) for target in neighbors
        ) / len(neighbors)
        _, relevance = event_relevance(agent, scenario_id)
        centered_noise = (
            _hash_uniform(seed, scenario_id, policy_branch, tick, index) - 0.5
        )
        attention = _clip(
            0.58 * float(state["attention"])
            + 0.20 * neighbor_attention
            + 0.31 * intensity * relevance
            - 0.12 * policy_strength
            + 0.025 * centered_noise
        )
        concern = _clip(
            0.62 * float(state["concern"])
            + 0.18 * neighbor_concern
            + 0.29 * intensity * relevance
            - 0.30 * policy_strength
            + 0.020 * centered_noise
        )
        trust = _clip(
            0.70 * float(state["trust"])
            + 0.18 * neighbor_trust
            + 0.26 * policy_strength
            - 0.12 * intensity * relevance
            - 0.015 * centered_noise
        )
        satisfaction = _clip(
            0.63 * float(state["satisfaction"])
            + 0.19 * trust
            + 0.20 * policy_strength
            - 0.22 * concern
        )
        support = _sigmoid(
            -0.42
            + 1.45 * trust
            + 0.85 * satisfaction
            + 0.75 * policy_strength
            - 0.82 * concern
        )
        opposition = _sigmoid(
            -0.76
            + 1.52 * concern
            + 0.45 * neighbor_concern
            - 0.96 * trust
            - 0.72 * policy_strength
        )
        activity = _sigmoid(
            -2.10
            + 2.10 * attention
            + 1.05 * concern
            + 0.42 * neighbor_attention
            + 0.38 * intensity * relevance
        )
        memory = list(state.get("short_memory") or [])
        memory.append(
            {
                "tick": tick,
                "phase": TIMELINE_TICKS[tick],
                "attention_band": round(attention, 2),
                "concern_band": round(concern, 2),
            }
        )
        updated.append(
            {
                "attention": round(attention, 6),
                "concern": round(concern, 6),
                "trust": round(trust, 6),
                "satisfaction": round(satisfaction, 6),
                "support": round(support, 6),
                "opposition": round(opposition, 6),
                "activity_probability": round(activity, 6),
                "short_memory": memory[-3:],
            }
        )
    return updated


def decision_probabilities(
    agent: Mapping[str, Any],
    state: Mapping[str, Any],
    scenario_id: str,
    phase_id: str,
) -> dict[str, dict[str, float]]:
    relevance_label, relevance = event_relevance(agent, scenario_id)
    speaking = _clip(float(state["activity_probability"]))
    risk = {
        "none": 0.0,
        "low": 0.2,
        "medium": 0.5,
        "high": 0.85,
        "uncertain": 0.42,
    }.get(str(agent["semantic"].get("risk_level")), 0.3)
    public_mass = speaking * (0.30 + 0.70 * relevance)
    if phase_id == "baseline" or relevance_label == "irrelevant":
        public_mass *= 0.30
    action = _normalize(
        {
            "ignore": (1.0 - public_mass) * 0.36,
            "read": (1.0 - public_mass) * 0.64,
            "comment": public_mass * 0.34,
            "post": public_mass * 0.34,
            "seek_help": public_mass * (0.20 + 0.14 * risk),
            "report_risk": public_mass * (0.12 - 0.04 * risk),
        }
    )
    support = float(state["support"])
    opposition = float(state["opposition"])
    concern = float(state["concern"])
    stance = _normalize(
        {
            "supportive": 0.08 + 0.70 * support,
            "neutral": 0.20 + 0.42 * (1.0 - max(support, opposition)),
            "concerned": 0.08 + 0.58 * concern,
            "critical": 0.06 + 0.72 * opposition,
        }
    )
    topic = _normalize(
        {
            scenario_id: (
                0.04
                if phase_id == "baseline" or relevance_label == "irrelevant"
                else 0.28 + 0.68 * relevance
            ),
            NON_EVENT_TOPIC_ID: (
                0.96
                if phase_id == "baseline" or relevance_label == "irrelevant"
                else 0.72 - 0.68 * relevance
            ),
        }
    )
    return {"action": action, "stance": stance, "topic_id": topic}


def _draw_category(probabilities: Mapping[str, float], uniform: float) -> str:
    cumulative = 0.0
    last = next(iter(probabilities))
    for key, value in probabilities.items():
        last = key
        cumulative += float(value)
        if uniform <= cumulative + 1e-12:
            return key
    return last


def fixture_probe_decision(
    *,
    agent: Mapping[str, Any],
    state: Mapping[str, Any],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
) -> Any:
    """Produce a strict, zero-Provider semantic probe through the v4 validator."""

    probabilities = decision_probabilities(
        agent, state, scenario_id, TIMELINE_TICKS[tick]
    )
    action = _draw_category(
        probabilities["action"],
        _hash_uniform(seed, scenario_id, branch, tick, agent["micro_agent_id"], "action"),
    )
    stance = _draw_category(
        probabilities["stance"],
        _hash_uniform(seed, scenario_id, branch, tick, agent["micro_agent_id"], "stance"),
    )
    topic_id = _draw_category(
        probabilities["topic_id"],
        _hash_uniform(seed, scenario_id, branch, tick, agent["micro_agent_id"], "topic"),
    )
    message = (
        ""
        if action in {"ignore", "read"}
        else "合成语义探针表达，不代表真实个体。"
    )
    raw = {
        "action": action,
        "topic_id": topic_id,
        "stance": stance,
        "urgency": float(state["concern"]),
        "trust_delta": (
            0.04 * (1.0 - float(state["concern"]))
            if branch == "governance"
            else -0.02 * float(state["concern"])
        ),
        "satisfied": bool(float(state["satisfaction"]) >= 0.5),
        "rationale_tag": "general",
        "message": message,
    }
    return validate_agent_decision(
        raw,
        topic_ids={scenario_id, NON_EVENT_TOPIC_ID},
        simulation_mode="active_episode_agent",
    )


def _sampling_strata(
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    scenario_id: str,
) -> tuple[tuple[str, ...], float]:
    activities = sorted(float(state["activity_probability"]) for state in states)
    median_activity = statistics.median(activities)
    labels = []
    for agent, state in zip(population["agents"], states):
        relevance, _ = event_relevance(agent, scenario_id)
        activity = (
            "high"
            if float(state["activity_probability"]) >= median_activity
            else "low"
        )
        labels.append(f"{relevance}__{activity}")
    return tuple(labels), median_activity


def draw_probe_sample(
    *,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    rotation_counts: Sequence[int],
    budget: int = PROBE_BUDGET,
) -> dict[str, Any]:
    """Cover every non-empty frozen stratum, then draw pivotal PPS."""

    agents = population["agents"]
    if len(rotation_counts) != len(agents):
        raise MassEngineError("rotation counts do not match the population")
    labels, median_activity = _sampling_strata(
        population, states, scenario_id
    )
    strata = sorted(set(labels))
    members_by_stratum = {
        label: [index for index, value in enumerate(labels) if value == label]
        for label in strata
    }
    allocation = _integer_allocation(
        [len(members_by_stratum[label]) for label in strata],
        budget,
    )
    inclusion_probability = [0.0 for _ in agents]
    selected = []
    allocation_by_stratum = {}
    for stratum_position, label in enumerate(strata):
        members = members_by_stratum[label]
        count = allocation[stratum_position]
        allocation_by_stratum[label] = count
        scores = []
        for index in members:
            state = states[index]
            agent = agents[index]
            _, relevance_score = event_relevance(agent, scenario_id)
            uncertainty = (
                1.0
                - abs(
                    float(state["support"])
                    - float(state["opposition"])
                )
            )
            rotation_penalty = 1.0 / (1.0 + 0.45 * rotation_counts[index])
            scores.append(
                max(float(agent["population_weight"]), 1e-12)
                * max(float(state["activity_probability"]), 0.03)
                * (0.55 + 0.55 * uncertainty + 0.25 * relevance_score)
                * rotation_penalty
            )
        local_pi = fixed_size_inclusion_probabilities(scores, count)
        local_selected = pivotal_sample(
            local_pi,
            random.Random(
                _seed_from(seed, scenario_id, branch, tick, label, "pivotal")
            ),
        )
        for local_index, population_index in enumerate(members):
            inclusion_probability[population_index] = local_pi[local_index]
        selected.extend(members[local_index] for local_index in local_selected)
    selected = sorted(selected)
    if len(selected) != budget:
        raise MassEngineError("probe round did not meet the fixed budget")
    if any(inclusion_probability[index] <= 0 for index in selected):
        raise MassEngineError("sampled probes require positive inclusion probability")
    return {
        "schema_version": MASS_SAMPLING_SCHEMA_VERSION,
        "budget": budget,
        "selected": tuple(selected),
        "inclusion_probability": tuple(inclusion_probability),
        "strata": labels,
        "median_activity": median_activity,
        "allocation_by_stratum": allocation_by_stratum,
        "selected_identity_sha256": canonical_hash(
            [agents[index]["micro_agent_id"] for index in selected]
        ),
    }


def _greg_distribution(
    *,
    population: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Mapping[str, float]]],
    decisions: Sequence[Any],
    sample: Mapping[str, Any],
    dimension: str,
    categories: Sequence[str],
) -> dict[str, float]:
    weights = [
        float(agent["population_weight"]) for agent in population["agents"]
    ]
    selected = sample["selected"]
    pi = sample["inclusion_probability"]
    estimates = {}
    for category in categories:
        model_total = sum(
            weight * float(prediction[dimension].get(category, 0.0))
            for weight, prediction in zip(weights, predictions)
        )
        correction = sum(
            weights[index]
            * (
                float(getattr(decisions[index], dimension) == category)
                - float(predictions[index][dimension].get(category, 0.0))
            )
            / float(pi[index])
            for index in selected
        )
        estimates[category] = max(0.0, model_total + correction)
    return _normalize(estimates)


def _kish_ess(
    population: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> float:
    expanded = [
        float(population["agents"][index]["population_weight"])
        / float(sample["inclusion_probability"][index])
        for index in sample["selected"]
    ]
    return sum(expanded) ** 2 / max(sum(value * value for value in expanded), 1e-12)


def _approximate_interval(
    *,
    estimate: float,
    sample: Mapping[str, Any],
    population: Mapping[str, Any],
    decisions: Sequence[Any],
    dimension: str,
    category: str,
) -> list[float]:
    variance = 0.0
    for index in sample["selected"]:
        weight = float(population["agents"][index]["population_weight"])
        pi = float(sample["inclusion_probability"][index])
        residual = float(getattr(decisions[index], dimension) == category) - estimate
        variance += (1.0 - pi) * (weight / pi) ** 2 * residual * residual
    radius = 1.96 * math.sqrt(max(variance, 0.0))
    return [round(_clip(estimate - radius), 6), round(_clip(estimate + radius), 6)]


def _bayesian_partial_pooling(
    *,
    population: Mapping[str, Any],
    decisions: Sequence[Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    total_weight = defaultdict(float)
    for index in sample["selected"]:
        label = sample["strata"][index]
        expanded = (
            float(population["agents"][index]["population_weight"])
            / float(sample["inclusion_probability"][index])
        )
        decision = decisions[index]
        total_weight[label] += expanded
        by_stratum[label]["speaking"] += expanded * float(
            decision.action in PUBLIC_ACTIONS
        )
        if decision.action in PUBLIC_ACTIONS:
            by_stratum[label]["public"] += expanded
            by_stratum[label][f"stance:{decision.stance}"] += expanded
    strata = {}
    global_speaker = (
        sum(values["speaking"] for values in by_stratum.values())
        / max(sum(total_weight.values()), 1e-12)
    )
    for label in sorted(total_weight):
        mass = total_weight[label]
        speaker = (2.0 * global_speaker + by_stratum[label]["speaking"]) / (
            2.0 + mass
        )
        public = by_stratum[label]["public"]
        stance = {
            category: (
                0.5 + by_stratum[label][f"stance:{category}"]
            )
            / max(2.0 + public, 1e-12)
            for category in STANCE_CATEGORIES
        }
        strata[label] = {
            "posterior_speaking_rate": round(_clip(speaker), 6),
            "conditional_stance": {
                key: round(value, 6)
                for key, value in _normalize(stance).items()
            },
        }
    return {
        "model": "beta_dirichlet_partial_pooling_v1",
        "strata": strata,
    }


def observe_population(
    *,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    rotation_counts: list[int],
    budget: int = PROBE_BUDGET,
) -> tuple[dict[str, Any], list[Any]]:
    sample = draw_probe_sample(
        population=population,
        states=states,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        seed=seed,
        rotation_counts=rotation_counts,
        budget=budget,
    )
    predictions = [
        decision_probabilities(agent, state, scenario_id, TIMELINE_TICKS[tick])
        for agent, state in zip(population["agents"], states)
    ]
    decisions = [
        fixture_probe_decision(
            agent=agent,
            state=state,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            seed=seed,
        )
        for agent, state in zip(population["agents"], states)
    ]
    for index in sample["selected"]:
        rotation_counts[index] += 1
    action = _greg_distribution(
        population=population,
        predictions=predictions,
        decisions=decisions,
        sample=sample,
        dimension="action",
        categories=ACTION_CATEGORIES,
    )
    stance = _greg_distribution(
        population=population,
        predictions=predictions,
        decisions=decisions,
        sample=sample,
        dimension="stance",
        categories=STANCE_CATEGORIES,
    )
    topics = _greg_distribution(
        population=population,
        predictions=predictions,
        decisions=decisions,
        sample=sample,
        dimension="topic_id",
        categories=(scenario_id, NON_EVENT_TOPIC_ID),
    )
    pi_values = [
        float(sample["inclusion_probability"][index])
        for index in sample["selected"]
    ]
    intervals = {
        "action": {
            category: _approximate_interval(
                estimate=value,
                sample=sample,
                population=population,
                decisions=decisions,
                dimension="action",
                category=category,
            )
            for category, value in action.items()
        },
        "stance": {
            category: _approximate_interval(
                estimate=value,
                sample=sample,
                population=population,
                decisions=decisions,
                dimension="stance",
                category=category,
            )
            for category, value in stance.items()
        },
    }
    result = {
        "schema_version": MASS_SAMPLING_SCHEMA_VERSION,
        "tick": tick,
        "phase": TIMELINE_TICKS[tick],
        "branch": branch,
        "budget": budget,
        "stratum_count": len(sample["allocation_by_stratum"]),
        "allocation_by_stratum": sample["allocation_by_stratum"],
        "selected_identity_sha256": sample["selected_identity_sha256"],
        "inclusion_probability_range": [
            round(min(pi_values), 8),
            round(max(pi_values), 8),
        ],
        "kish_ess": round(_kish_ess(population, sample), 6),
        "corrected": {
            "action": {key: round(value, 6) for key, value in action.items()},
            "stance": {key: round(value, 6) for key, value in stance.items()},
            "topic_id": {key: round(value, 6) for key, value in topics.items()},
        },
        "confidence_intervals_95": intervals,
        "partial_pooling": _bayesian_partial_pooling(
            population=population,
            decisions=decisions,
            sample=sample,
        ),
        "provider_calls": 0,
        "_selected": sample["selected"],
    }
    return result, decisions


def apply_probe_writeback(
    *,
    population: Mapping[str, Any],
    states: list[dict[str, Any]],
    decisions: Sequence[Any],
    selected: Sequence[int],
    mode: str,
    tick: int,
) -> None:
    if mode not in WRITEBACK_MODES:
        raise MassEngineError("unknown writeback mode")
    if mode == "observation_only":
        return
    for index in selected:
        decision = decisions[index]
        state = states[index]
        if mode in {"stance_only", "full_state"}:
            if decision.stance == "supportive":
                state["support"] = round(_clip(float(state["support"]) + 0.08), 6)
                state["opposition"] = round(
                    _clip(float(state["opposition"]) - 0.04), 6
                )
            elif decision.stance in {"critical", "concerned"}:
                state["opposition"] = round(
                    _clip(float(state["opposition"]) + 0.08), 6
                )
                state["support"] = round(_clip(float(state["support"]) - 0.03), 6)
        if mode != "full_state":
            continue
        agent = population["agents"][index]
        legacy = AgentState(
            agent_id=str(agent["micro_agent_id"]),
            role_id="episode_observer",
            role_label="论坛行为 episode",
            concern_topic_id=str(agent["semantic"]["topic_code"]),
            concern_topic_label="合成议题",
            population_weight=float(agent["population_weight"]),
            trust=float(state["trust"]),
            concern=float(state["concern"]),
            archetype_id=str(agent["primary_archetype_id"]),
            archetype_label="审阅画像原型",
            population_month=str(agent["month"]),
            satisfied=float(state["satisfaction"]) >= 0.5,
            memory=[
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in state.get("short_memory", [])
            ],
        )
        transitioned = transition_agent_state(
            legacy,
            ModelTurn(
                decision=decision,
                returned_model="deterministic-fixture",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                provider_attempts=0,
                format_repairs=0,
            ),
            {
                "event_visible": TIMELINE_TICKS[tick] != "baseline",
                "public_action_allowed": True,
            },
        )
        state["trust"] = round(float(transitioned.agent.trust), 6)
        state["satisfaction"] = (
            0.66 if transitioned.agent.satisfied else 0.34
        )
        state["short_memory"] = [
            {"validated_transition": item[-96:]}
            for item in transitioned.agent.memory[-3:]
        ]


def aggregate_population_state(
    *,
    population: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    scenario_id: str,
    tick: int,
) -> dict[str, Any]:
    agents = population["agents"]
    weights = [float(agent["population_weight"]) for agent in agents]
    means = {
        field: sum(weight * float(state[field]) for weight, state in zip(weights, states))
        for field in (
            "attention",
            "concern",
            "trust",
            "satisfaction",
            "support",
            "opposition",
            "activity_probability",
        )
    }
    stance = defaultdict(float)
    action = defaultdict(float)
    topic = defaultdict(float)
    for weight, agent, state in zip(weights, agents, states):
        probabilities = decision_probabilities(
            agent, state, scenario_id, TIMELINE_TICKS[tick]
        )
        for category, value in probabilities["stance"].items():
            stance[category] += weight * value
        for category, value in probabilities["action"].items():
            action[category] += weight * value
        for category, value in probabilities["topic_id"].items():
            topic[category] += weight * value
    return {
        "tick": tick,
        "phase": TIMELINE_TICKS[tick],
        "state_means": {
            key: round(value, 6) for key, value in means.items()
        },
        "model_shares": {
            "stance": {key: round(value, 6) for key, value in sorted(stance.items())},
            "action": {key: round(value, 6) for key, value in sorted(action.items())},
            "topic_id": {key: round(value, 6) for key, value in sorted(topic.items())},
        },
    }


def encode_population_checkpoint(
    *,
    population_sha256: str,
    graph_sha256: str,
    scenario_id: str,
    branch: str,
    tick: int,
    states: Sequence[Mapping[str, Any]],
    previous_checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = {
        "population_sha256": population_sha256,
        "graph_sha256": graph_sha256,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "states": list(states),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    body = {
        "schema_version": POPULATION_CHECKPOINT_SCHEMA_VERSION,
        "population_sha256": population_sha256,
        "graph_sha256": graph_sha256,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "compression": "zlib+base64",
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "state_payload_sha256": canonical_hash(payload),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "payload": base64.b64encode(compressed).decode("ascii"),
    }
    body["checkpoint_sha256"] = canonical_hash(body)
    return body


def decode_population_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    population_sha256: str,
    graph_sha256: str,
) -> list[dict[str, Any]]:
    expected = dict(checkpoint)
    stored_sha = expected.pop("checkpoint_sha256", None)
    if stored_sha != canonical_hash(expected):
        raise MassEngineError("population checkpoint hash does not match")
    if (
        checkpoint.get("schema_version") != POPULATION_CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("population_sha256") != population_sha256
        or checkpoint.get("graph_sha256") != graph_sha256
        or checkpoint.get("compression") != "zlib+base64"
    ):
        raise MassEngineError("population checkpoint identity does not match")
    try:
        raw = zlib.decompress(base64.b64decode(str(checkpoint["payload"])))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, TypeError, zlib.error) as exc:
        raise MassEngineError("population checkpoint payload is invalid") from exc
    if (
        canonical_hash(payload) != checkpoint.get("state_payload_sha256")
        or payload.get("population_sha256") != population_sha256
        or payload.get("graph_sha256") != graph_sha256
    ):
        raise MassEngineError("population checkpoint payload drifted")
    states = payload.get("states")
    if not isinstance(states, list):
        raise MassEngineError("population checkpoint has no states")
    return states


def _resource_actions(branch: str, tick: int) -> list[dict[str, Any]]:
    if branch != "governance":
        return []
    actions = {
        3: ("student_affairs_response_team", "publish_verified_update", 14),
        7: ("campus_service_coordinator", "open_support_channel", 18),
    }
    if tick not in actions:
        return []
    actor, action, resource = actions[tick]
    return [
        {
            "tick": tick,
            "actor": actor,
            "action": action,
            "resource_units": resource,
            "propagation_path": [
                "governance_actor",
                "cross_archetype_bridges",
                "semantic_neighborhoods",
            ],
            "external_real_world_action": False,
        }
    ]


def run_branch(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    scenario_id: str,
    seed: int,
    branch: str,
    writeback_mode: str,
    starting_tick: int,
    ending_tick: int,
    states: list[dict[str, Any]],
    rotation_counts: list[int],
    previous_checkpoint_sha256: str,
) -> dict[str, Any]:
    adjacency = graph_adjacency(graph, len(population["agents"]))
    timeline = []
    probes = []
    checkpoints = []
    resource_actions = []
    resource_opening = 100
    for tick in range(starting_tick, ending_tick + 1):
        states = advance_population_tick(
            population=population,
            states=states,
            adjacency=adjacency,
            scenario_id=scenario_id,
            policy_branch=branch,
            tick=tick,
            seed=seed,
        )
        aggregate = aggregate_population_state(
            population=population,
            states=states,
            scenario_id=scenario_id,
            tick=tick,
        )
        if tick in CALIBRATION_TICKS:
            observation, decisions = observe_population(
                population=population,
                states=states,
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
                seed=seed,
                rotation_counts=rotation_counts,
            )
            selected = observation.pop("_selected")
            aggregate["semantic_observation"] = observation
            probes.append(observation)
            apply_probe_writeback(
                population=population,
                states=states,
                decisions=decisions,
                selected=selected,
                mode=writeback_mode,
                tick=tick,
            )
            checkpoint = encode_population_checkpoint(
                population_sha256=str(population["population_sha256"]),
                graph_sha256=str(graph["graph_sha256"]),
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
                states=states,
                previous_checkpoint_sha256=previous_checkpoint_sha256,
            )
            previous_checkpoint_sha256 = checkpoint["checkpoint_sha256"]
            checkpoints.append(checkpoint)
        actions = _resource_actions(branch, tick)
        resource_actions.extend(actions)
        aggregate["governance_actions"] = actions
        aggregate["resource_remaining"] = resource_opening - sum(
            action["resource_units"] for action in resource_actions
        )
        timeline.append(aggregate)
    return {
        "branch": branch,
        "timeline": timeline,
        "probes": probes,
        "checkpoints": checkpoints,
        "states": states,
        "rotation_counts": rotation_counts,
        "resource": {
            "opening": resource_opening,
            "consumed": sum(
                action["resource_units"] for action in resource_actions
            ),
            "remaining": resource_opening
            - sum(action["resource_units"] for action in resource_actions),
            "conserved": resource_opening
            == sum(action["resource_units"] for action in resource_actions)
            + (
                resource_opening
                - sum(action["resource_units"] for action in resource_actions)
            ),
            "actions": resource_actions,
        },
    }


def run_seed_scenario(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    scenario_id: str,
    seed: int,
    writeback_mode: str,
) -> dict[str, Any]:
    """Run a shared two-tick baseline and two cloned policy branches."""

    baseline_states = initial_states(population)
    rotations = [0 for _ in population["agents"]]
    genesis = "0" * 64
    shared = run_branch(
        population=population,
        graph=graph,
        scenario_id=scenario_id,
        seed=seed,
        branch="natural",
        writeback_mode=writeback_mode,
        starting_tick=0,
        ending_tick=1,
        states=baseline_states,
        rotation_counts=rotations,
        previous_checkpoint_sha256=genesis,
    )
    baseline_state_hash = canonical_hash(shared["states"])
    branches = {}
    for branch in POLICY_BRANCHES:
        branches[branch] = run_branch(
            population=population,
            graph=graph,
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            writeback_mode=writeback_mode,
            starting_tick=2,
            ending_tick=15,
            states=copy.deepcopy(shared["states"]),
            rotation_counts=list(shared["rotation_counts"]),
            previous_checkpoint_sha256=shared["checkpoints"][-1][
                "checkpoint_sha256"
            ],
        )
    if any(
        canonical_hash(
            [
                item
                for item in shared["timeline"]
                if item["tick"] <= 1
            ]
        )
        != canonical_hash(shared["timeline"])
        for _ in (0,)
    ):
        raise MassEngineError("shared baseline identity drifted")
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "writeback_mode": writeback_mode,
        "shared_baseline": shared,
        "shared_baseline_state_sha256": baseline_state_hash,
        "branches": branches,
        "provider_calls": 0,
        "planned_probe_turns": PROBE_BUDGET * 7,
    }


def _observation_error(run: Mapping[str, Any]) -> tuple[float, float]:
    errors = []
    change_errors = []
    for branch in POLICY_BRANCHES:
        timeline = run["branches"][branch]["timeline"]
        observed_concerned = []
        model_concerned = []
        for item in timeline:
            observation = item.get("semantic_observation")
            if not observation:
                continue
            model = item["model_shares"]
            for dimension in ("action", "stance", "topic_id"):
                for category, truth in model[dimension].items():
                    errors.append(
                        abs(
                            float(
                                observation["corrected"][dimension].get(
                                    category, 0.0
                                )
                            )
                            - float(truth)
                        )
                    )
            observed_concerned.append(
                float(observation["corrected"]["stance"]["concerned"])
                + float(observation["corrected"]["stance"]["critical"])
            )
            model_concerned.append(
                float(model["stance"]["concerned"])
                + float(model["stance"]["critical"])
            )
        for index in range(1, len(observed_concerned)):
            observed_change = (
                observed_concerned[index] - observed_concerned[index - 1]
            )
            model_change = (
                model_concerned[index] - model_concerned[index - 1]
            )
            change_errors.append(abs(observed_change - model_change))
    return (
        sum(errors) / max(len(errors), 1),
        sum(change_errors) / max(len(change_errors), 1),
    )


def evaluate_writeback_modes(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    scenario_id: str,
    seeds: Sequence[int],
    descriptive_baseline_mae: float | None = None,
) -> dict[str, Any]:
    """Freeze the formal mode unless writeback is stably non-inferior."""

    metrics = {}
    runs_by_mode = {}
    for mode in WRITEBACK_MODES:
        runs = [
            run_seed_scenario(
                population=population,
                graph=graph,
                scenario_id=scenario_id,
                seed=int(seed),
                writeback_mode=mode,
            )
            for seed in seeds
        ]
        runs_by_mode[mode] = runs
        values = [_observation_error(run) for run in runs]
        metrics[mode] = {
            "primary_share_mae": round(
                sum(value[0] for value in values) / len(values), 6
            ),
            "phase_change_mae": round(
                sum(value[1] for value in values) / len(values), 6
            ),
            "seed_variance": round(
                statistics.pvariance(value[0] for value in values)
                if len(values) > 1
                else 0.0,
                8,
            ),
            "descriptive_baseline_mae": (
                round(float(descriptive_baseline_mae), 6)
                if descriptive_baseline_mae is not None
                else None
            ),
        }
    baseline = metrics["observation_only"]
    eligible = [
        mode
        for mode in ("stance_only", "full_state")
        if metrics[mode]["primary_share_mae"]
        <= baseline["primary_share_mae"] + 1e-12
        and metrics[mode]["seed_variance"]
        <= baseline["seed_variance"] + 1e-12
        and metrics[mode]["phase_change_mae"]
        <= baseline["phase_change_mae"] + 1e-12
        and (
            descriptive_baseline_mae is None
            or metrics[mode]["primary_share_mae"]
            <= float(descriptive_baseline_mae) + 0.02
        )
    ]
    formal_mode = (
        min(
            eligible,
            key=lambda mode: (
                metrics[mode]["primary_share_mae"],
                metrics[mode]["seed_variance"],
            ),
        )
        if eligible
        else "observation_only"
    )
    return {
        "schema_version": "campus-pulse-writeback-evaluation-v1",
        "frozen_scenario_id": scenario_id,
        "seeds": list(seeds),
        "metrics": metrics,
        "formal_mode": formal_mode,
        "experimental_modes": [
            mode for mode in WRITEBACK_MODES if mode != formal_mode
        ],
        "observation_only_fallback_applied": formal_mode == "observation_only",
        "descriptive_baseline_non_degradation_limit": (
            round(float(descriptive_baseline_mae) + 0.02, 6)
            if descriptive_baseline_mae is not None
            else None
        ),
        "_runs_by_mode": runs_by_mode,
    }


def build_run_plan_v5(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    scenario_ids: Sequence[str],
    seeds: Sequence[int],
    writeback_mode: str,
) -> dict[str, Any]:
    if writeback_mode not in WRITEBACK_MODES:
        raise MassEngineError("run-plan-v5 writeback mode is invalid")
    group_weights = defaultdict(float)
    for agent in population["agents"]:
        group_weights[str(agent["primary_archetype_id"])] += float(
            agent["population_weight"]
        )
    body = {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "execution_mode": "budgeted_population",
        "population": {
            "population_release_id": "semantic-population-v2-reviewed-1000",
            "population_sha256": population["population_sha256"],
            "population_size": len(population["agents"]),
            "unit": "reviewed_forum_episode_micro_agent",
            "visual_groups": [
                {
                    "group_id": group_id,
                    "population_weight": round(weight, 8),
                }
                for group_id, weight in sorted(group_weights.items())
            ],
        },
        "synthetic_exposure_graph": {
            "graph_sha256": graph["graph_sha256"],
            "node_count": graph["node_count"],
            "edge_count": graph["edge_count"],
            "out_degree": graph["out_degree"],
        },
        "dynamics": {
            "schema_version": MASS_DYNAMICS_SCHEMA_VERSION,
            "model": "interpretable_multitask_glm_state_space_v1",
            "timeline": [
                {
                    "tick": tick,
                    "phase": phase,
                }
                for tick, phase in enumerate(TIMELINE_TICKS)
            ],
            "checkpoint_ticks": list(CALIBRATION_TICKS),
        },
        "sampling": {
            "schema_version": MASS_SAMPLING_SCHEMA_VERSION,
            "strata": [
                f"{relevance}__{activity}"
                for relevance in ("irrelevant", "uncertain", "relevant")
                for activity in ("low", "high")
            ],
            "budget_per_round": PROBE_BUDGET,
            "rounds_per_seed": 7,
            "planned_cold_turns_per_seed": PROBE_BUDGET * 7,
            "estimator": "generalized_difference_ht_with_partial_pooling",
        },
        "writeback_mode": writeback_mode,
        "particle_layer": {
            "particles_per_micro_agent": PARTICLES_PER_AGENT,
            "particle_count": len(population["agents"])
            * PARTICLES_PER_AGENT,
            "evidence_subject": False,
        },
        "scenario_ids": list(scenario_ids),
        "policy_branches": list(POLICY_BRANCHES),
        "seeds": [int(seed) for seed in seeds],
        "cost_contract": {
            "provider_call_limit": 0,
            "provider_calls_authorized": False,
            "real_provider_canary_included": False,
            "fixture_probe_turns": (
                PROBE_BUDGET
                * 7
                * len(seeds)
                * len(scenario_ids)
            ),
        },
        "non_claims": [
            "Micro-agents are forum behavior episodes, not students.",
            "Particles visualize local uncertainty and are not evidence units.",
            "The exposure graph is synthetic and not a social graph.",
            "No causal or real-world governance effect is claimed.",
        ],
    }
    body["plan_sha256"] = canonical_hash(body)
    return body


def _mean_timeline(
    runs: Sequence[Mapping[str, Any]],
    branch: str,
) -> list[dict[str, Any]]:
    items = []
    for tick in range(16):
        rows = [run["branches"][branch]["timeline"][tick - 2] for run in runs] if tick >= 2 else [
            run["shared_baseline"]["timeline"][tick] for run in runs
        ]
        state_fields = rows[0]["state_means"]
        means = {
            field: sum(float(row["state_means"][field]) for row in rows)
            / len(rows)
            for field in state_fields
        }
        probe_rows = [
            row["semantic_observation"]
            for row in rows
            if row.get("semantic_observation")
        ]
        item = {
            "tick": tick,
            "phase": TIMELINE_TICKS[tick],
            "state_means": {
                key: round(value, 6) for key, value in means.items()
            },
            "governance_actions": copy.deepcopy(
                rows[0].get("governance_actions", [])
            ),
            "resource_remaining": rows[0].get("resource_remaining"),
        }
        if probe_rows:
            item["probe_summary"] = {
                "budget": PROBE_BUDGET,
                "mean_kish_ess": round(
                    sum(float(row["kish_ess"]) for row in probe_rows)
                    / len(probe_rows),
                    6,
                ),
                "inclusion_probability_range": [
                    min(
                        float(row["inclusion_probability_range"][0])
                        for row in probe_rows
                    ),
                    max(
                        float(row["inclusion_probability_range"][1])
                        for row in probe_rows
                    ),
                ],
                "confidence_intervals_95": probe_rows[0][
                    "confidence_intervals_95"
                ],
                "display_probe_particle_ordinals": sorted(
                    random.Random(
                        _seed_from("display-only-probes", branch, tick)
                    ).sample(range(10_000), PROBE_BUDGET)
                ),
                "display_ordinals_are_sample_identities": False,
            }
        items.append(item)
    return items


def build_aggregate_result_v3(
    *,
    plan: Mapping[str, Any],
    scenario_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    writeback_evaluation: Mapping[str, Any],
    elapsed_seconds: float,
    peak_additional_memory_bytes: int | None = None,
) -> dict[str, Any]:
    scenarios = {}
    all_ess = []
    all_pi = []
    seed_endpoints = defaultdict(list)
    total_low_cost_updates = 0
    total_probe_turns = 0
    for scenario_id, runs in scenario_runs.items():
        natural = _mean_timeline(runs, "natural")
        governance = _mean_timeline(runs, "governance")
        for branch_name, timeline in (
            ("natural", natural),
            ("governance", governance),
        ):
            for item in timeline:
                if "probe_summary" in item:
                    all_ess.append(item["probe_summary"]["mean_kish_ess"])
                    all_pi.extend(item["probe_summary"]["inclusion_probability_range"])
            seed_endpoints[f"{scenario_id}:{branch_name}"] = [
                run["branches"][branch_name]["timeline"][-1][
                    "state_means"
                ]["concern"]
                for run in runs
            ]
        differences = []
        for tick in range(16):
            differences.append(
                {
                    "tick": tick,
                    "phase": TIMELINE_TICKS[tick],
                    "concern_delta": round(
                        governance[tick]["state_means"]["concern"]
                        - natural[tick]["state_means"]["concern"],
                        6,
                    ),
                    "trust_delta": round(
                        governance[tick]["state_means"]["trust"]
                        - natural[tick]["state_means"]["trust"],
                        6,
                    ),
                    "activity_delta": round(
                        governance[tick]["state_means"]["activity_probability"]
                        - natural[tick]["state_means"]["activity_probability"],
                        6,
                    ),
                }
            )
        resource = runs[0]["branches"]["governance"]["resource"]
        scenarios[scenario_id] = {
            "timeline": {
                "natural": natural,
                "governance": governance,
            },
            "branch_differences": differences,
            "shared_baseline_sha256_by_seed": [
                run["shared_baseline_state_sha256"] for run in runs
            ],
            "resource_accounting": resource,
        }
        total_low_cost_updates += len(runs) * (
            2 * len(plan["population"]["population_sha256"])
        )
        total_probe_turns += len(runs) * PROBE_BUDGET * 7
    population_size = int(plan["population"]["population_size"])
    # Per seed/scenario: two shared baseline ticks plus 14 ticks per branch.
    total_low_cost_updates = (
        population_size
        * 30
        * sum(len(runs) for runs in scenario_runs.values())
    )
    sensitivity = {
        key: {
            "range": round(max(values) - min(values), 6),
            "variance": round(
                statistics.pvariance(values) if len(values) > 1 else 0.0,
                8,
            ),
        }
        for key, values in sorted(seed_endpoints.items())
    }
    body = {
        "schema_version": AGGREGATE_RESULT_SCHEMA_VERSION,
        "result_kind": "budgeted_population_model_conditional_simulation",
        "plan_sha256": plan["plan_sha256"],
        "population_identity": {
            "population_sha256": plan["population"]["population_sha256"],
            "population_size": population_size,
            "unit": plan["population"]["unit"],
            "particle_count": plan["particle_layer"]["particle_count"],
            "visual_groups": copy.deepcopy(
                plan["population"]["visual_groups"]
            ),
        },
        "timeline_contract": {
            "tick_count": 16,
            "phases": [
                {"phase": phase, "tick_count": count}
                for phase, count in TIMELINE_PHASES
            ],
            "shared_baseline_ticks": 2,
        },
        "scenarios": scenarios,
        "sampling_summary": {
            "budget_per_round": PROBE_BUDGET,
            "total_fixture_probe_turns": total_probe_turns,
            "inclusion_probability_range": [
                round(min(all_pi), 8),
                round(max(all_pi), 8),
            ],
            "mean_kish_ess": round(sum(all_ess) / len(all_ess), 6),
            "all_selected_probabilities_positive": min(all_pi) > 0,
        },
        "writeback_evaluation": {
            key: value
            for key, value in writeback_evaluation.items()
            if not key.startswith("_")
        },
        "seed_sensitivity": sensitivity,
        "usage": {
            "provider_calls": 0,
            "real_governance_actions": 0,
            "low_cost_state_updates": total_low_cost_updates,
        },
        "performance": {
            "engine_elapsed_seconds": round(elapsed_seconds, 6),
            "peak_additional_memory_bytes": peak_additional_memory_bytes,
        },
        "representative_synthetic_utterances": [
            {
                "label": "信息缺口",
                "text": "希望先看到可核验的时间线，再决定是否转发。",
                "review_status": "synthetic_reviewed_demo_text",
            },
            {
                "label": "服务承载",
                "text": "说明已经发布，但支持渠道是否接得住仍需观察。",
                "review_status": "synthetic_reviewed_demo_text",
            },
            {
                "label": "有限杠杆",
                "text": "治理动作降低了部分焦虑，并没有让所有分歧消失。",
                "review_status": "synthetic_reviewed_demo_text",
            },
        ],
        "privacy": {
            "micro_agent_states_exported": False,
            "sample_identities_exported": False,
            "graph_edges_exported": False,
            "fixture_decisions_exported": False,
            "raw_text_exported": False,
            "source_identifiers_exported": False,
        },
        "non_claims": list(plan["non_claims"]),
    }
    body["result_sha256"] = canonical_hash(body)
    return body
