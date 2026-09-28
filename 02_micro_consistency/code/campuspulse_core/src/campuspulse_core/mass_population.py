"""Private population compiler and synthetic exposure graph for MassSim v1.

The compiler joins the already-reviewed semantic cards and taxonomy release
with the deterministic sampling frame.  Source rows are used only to recover
the month-by-difficulty design probabilities.  Neither source identifiers nor
forum text are copied into the compiled population.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .calibration import DEFAULT_GAPS, _parse_date
from .semantic_population import (
    DEFAULT_DEFINITIONS,
    _shock_windows,
    heuristic_difficulty,
    load_shock_definitions,
    redact_for_provider,
    sample_forum_posts,
)


SEMANTIC_POPULATION_SCHEMA_VERSION = "semantic-population-v2"
EXPOSURE_GRAPH_SCHEMA_VERSION = "campus-pulse-synthetic-exposure-graph-v1"
POPULATION_SIZE = 1000
SEMANTIC_NEIGHBOR_COUNT = 8
CROSS_ARCHETYPE_BRIDGE_COUNT = 4


class MassPopulationError(ValueError):
    """Raised when a reviewed population cannot be compiled safely."""


def canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MassPopulationError(f"{Path(path).name} must contain an object")
    return payload


def _frame_strata_counts(
    posts_path: str | Path,
    *,
    definitions_path: str | Path,
    start: str,
    end: str,
    platform_marker: str,
) -> Counter[tuple[str, str]]:
    """Enumerate only the design strata using the reviewed sampling rules."""

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    windows = _shock_windows(load_shock_definitions(definitions_path))
    counts: Counter[tuple[str, str]] = Counter()
    with Path(posts_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("post_code", "")
            date = _parse_date(row.get("real_date", ""))
            if (
                platform_marker not in code
                or date is None
                or not start_dt <= date < end_dt
                or date.date().isoformat() in DEFAULT_GAPS
            ):
                continue
            if any(window_start <= date < window_end for _, window_start, window_end in windows):
                continue
            redacted, _ = redact_for_provider(row.get("content", ""))
            counts[(date.strftime("%Y-%m"), heuristic_difficulty(redacted))] += 1
    return counts


def _bounded_probability(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)


def _initial_state(card: Mapping[str, Any]) -> dict[str, Any]:
    risk = {
        "none": 0.05,
        "low": 0.22,
        "medium": 0.48,
        "high": 0.78,
        "uncertain": 0.42,
    }.get(str(card.get("risk_level")), 0.3)
    stance = {
        "supportive": (0.65, 0.10),
        "neutral": (0.30, 0.18),
        "concerned": (0.20, 0.56),
        "critical": (0.10, 0.76),
        "ambivalent": (0.34, 0.38),
    }.get(str(card.get("stance")), (0.30, 0.30))
    confidence = float(card.get("confidence", 0.5))
    uncertainty = 1.0 - confidence
    attention = 0.16 + 0.52 * risk + 0.18 * uncertainty
    concern = 0.14 + 0.62 * risk + 0.12 * stance[1]
    trust = 0.62 - 0.22 * stance[1] + 0.12 * stance[0]
    satisfaction = 0.56 - 0.32 * concern + 0.15 * trust
    activity = (
        0.08
        + 0.42 * attention
        + 0.16 * risk
        + (
            0.13
            if card.get("speech_act")
            in {"complaint", "risk_report", "help_request", "discussion"}
            else 0.0
        )
    )
    return {
        "attention": _bounded_probability(attention),
        "concern": _bounded_probability(concern),
        "trust": _bounded_probability(trust),
        "satisfaction": _bounded_probability(satisfaction),
        "support": _bounded_probability(stance[0]),
        "opposition": _bounded_probability(stance[1]),
        "activity_probability": _bounded_probability(activity),
        "short_memory": [],
    }


def _semantic_codes(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded semantic features without human-readable source content."""

    topic_basis = {
        "literal_topic": str(card.get("literal_topic") or ""),
        "affected_object": str(card.get("affected_object") or ""),
    }
    need_basis = {
        "implicit_need": str(card.get("implicit_need") or ""),
        "desired_response": str(card.get("desired_response") or ""),
    }
    return {
        "language_form": str(card.get("language_form") or "standard"),
        "semantic_determinacy": str(
            card.get("semantic_determinacy") or "partially_ambiguous"
        ),
        "speech_act": str(card.get("speech_act") or "other"),
        "stance": str(card.get("stance") or "neutral"),
        "risk_level": str(card.get("risk_level") or "uncertain"),
        "confidence": round(float(card.get("confidence", 0.5)), 4),
        "context_needed": bool(card.get("context_needed")),
        "topic_code": f"topic-{canonical_hash(topic_basis)[:10]}",
        "need_code": f"need-{canonical_hash(need_basis)[:10]}",
    }


def compile_semantic_population_v2(
    *,
    posts_path: str | Path,
    semantic_path: str | Path,
    taxonomy_path: str | Path,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
) -> dict[str, Any]:
    """Compile the reviewed 1,000 episode cards into a design-weighted population."""

    semantic = _load_json(semantic_path)
    taxonomy = _load_json(taxonomy_path)
    cards = semantic.get("records")
    assignments = taxonomy.get("assignments")
    sampling = semantic.get("sampling")
    if (
        not isinstance(cards, list)
        or len(cards) != POPULATION_SIZE
        or not isinstance(assignments, list)
        or len(assignments) != POPULATION_SIZE
        or not isinstance(sampling, dict)
        or sampling.get("sample_size") != POPULATION_SIZE
    ):
        raise MassPopulationError(
            "MassSim requires the reviewed 1,000-card semantic and taxonomy releases"
        )
    fit_scope = sampling.get("fit_scope")
    if not isinstance(fit_scope, dict):
        raise MassPopulationError("semantic sampling scope is missing")
    sampled_posts, regenerated_audit = sample_forum_posts(
        posts_path,
        definitions_path=definitions_path,
        sample_size=POPULATION_SIZE,
        seed=int(sampling["seed"]),
        start=str(fit_scope["start_inclusive"]),
        end=str(fit_scope["end_exclusive"]),
        platform_marker=str(fit_scope["platform_marker"]),
    )
    if (
        regenerated_audit["sample_manifest_fingerprint"]
        != sampling.get("sample_manifest_fingerprint")
    ):
        raise MassPopulationError(
            "private frame no longer reproduces the reviewed semantic sample"
        )
    card_by_id = {str(card.get("record_id")): card for card in cards}
    assignment_by_id = {
        str(item.get("record_id")): item for item in assignments
    }
    if (
        len(card_by_id) != POPULATION_SIZE
        or len(assignment_by_id) != POPULATION_SIZE
    ):
        raise MassPopulationError("reviewed record identities are not unique")

    frame_counts = _frame_strata_counts(
        posts_path,
        definitions_path=definitions_path,
        start=str(fit_scope["start_inclusive"]),
        end=str(fit_scope["end_exclusive"]),
        platform_marker=str(fit_scope["platform_marker"]),
    )
    sample_counts = Counter(
        (post.month, post.heuristic_difficulty) for post in sampled_posts
    )
    if sum(frame_counts.values()) != sampling.get("eligible_posts"):
        raise MassPopulationError(
            "month-by-difficulty frame counts do not conserve eligible posts"
        )
    raw_design_weights = []
    joined = []
    for post in sampled_posts:
        card = card_by_id.get(post.record_id)
        assignment = assignment_by_id.get(post.record_id)
        if card is None or assignment is None:
            raise MassPopulationError("semantic and taxonomy releases do not align")
        if (
            card.get("input_fingerprint") != post.input_fingerprint
            or assignment.get("input_fingerprint") != post.input_fingerprint
        ):
            raise MassPopulationError("reviewed input fingerprint drifted")
        stratum = (post.month, post.heuristic_difficulty)
        frame_n = frame_counts[stratum]
        sample_n = sample_counts[stratum]
        if frame_n <= 0 or sample_n <= 0 or sample_n > frame_n:
            raise MassPopulationError("invalid sampling stratum capacity")
        raw_weight = frame_n / sample_n
        raw_design_weights.append(raw_weight)
        joined.append((post, card, assignment, stratum))
    denominator = sum(raw_design_weights)
    if denominator <= 0:
        raise MassPopulationError("population design weights have no mass")

    agents = []
    for index, ((post, card, assignment, stratum), raw_weight) in enumerate(
        zip(joined, raw_design_weights),
        start=1,
    ):
        semantic_features = _semantic_codes(card)
        primary_archetype = str(
            assignment.get("primary_archetype_id") or "archetype-01"
        )
        secondary_archetype = str(
            assignment.get("secondary_archetype_id") or primary_archetype
        )
        agents.append(
            {
                "micro_agent_id": f"episode-{index:04d}",
                "stratum_id": f"{stratum[0]}__{stratum[1]}",
                "month": stratum[0],
                "heuristic_difficulty": stratum[1],
                "inclusion_probability": round(
                    sample_counts[stratum] / frame_counts[stratum], 12
                ),
                "population_weight": raw_weight / denominator,
                "primary_archetype_id": primary_archetype,
                "secondary_archetype_id": secondary_archetype,
                "semantic": semantic_features,
                "topic_distribution": {
                    semantic_features["topic_code"]: 0.72,
                    "off_topic_daily_life": 0.28,
                },
                "need_distribution": {
                    semantic_features["need_code"]: 0.78,
                    "other_need": 0.22,
                },
                "state": _initial_state(card),
            }
        )
    weight_sum = sum(float(agent["population_weight"]) for agent in agents)
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-12):
        raise MassPopulationError("normalized population weights do not sum to one")

    strata = []
    for month, difficulty in sorted(frame_counts):
        frame_n = frame_counts[(month, difficulty)]
        sample_n = sample_counts[(month, difficulty)]
        if not sample_n:
            continue
        strata.append(
            {
                "stratum_id": f"{month}__{difficulty}",
                "month": month,
                "heuristic_difficulty": difficulty,
                "frame_population": frame_n,
                "sample_population": sample_n,
                "inclusion_probability": round(sample_n / frame_n, 12),
                "inverse_probability_weight": round(frame_n / sample_n, 8),
            }
        )
    body = {
        "schema_version": SEMANTIC_POPULATION_SCHEMA_VERSION,
        "population_kind": "reviewed_forum_episode_micro_agents",
        "population_size": len(agents),
        "frame_population": sum(frame_counts.values()),
        "sampling_version": sampling.get("sampling_version"),
        "sampling_seed": sampling.get("seed"),
        "sample_manifest_fingerprint": sampling.get(
            "sample_manifest_fingerprint"
        ),
        "semantic_run_fingerprint": semantic.get("run_fingerprint"),
        "taxonomy_release_fingerprint": taxonomy.get(
            "taxonomy_release_fingerprint"
        ),
        "weight_sum": weight_sum,
        "strata": strata,
        "agents": agents,
        "privacy": {
            "raw_text_persisted": False,
            "source_identifiers_persisted": False,
            "user_relationships_persisted": False,
        },
        "non_claims": [
            "Episode micro-agents are not identified students.",
            "Design weights target the reviewed forum episode frame only.",
        ],
    }
    body["population_sha256"] = canonical_hash(body)
    return body


def _ordered_candidates(
    agents: list[Mapping[str, Any]],
    source_index: int,
    candidates: Iterable[int],
    edge_type: str,
    seed: int,
) -> list[int]:
    return sorted(
        set(candidates),
        key=lambda target: (
            canonical_hash(
                [
                    seed,
                    edge_type,
                    agents[source_index]["micro_agent_id"],
                    agents[target]["micro_agent_id"],
                ]
            ),
            target,
        ),
    )


def _positive_distribution(
    agent: Mapping[str, Any],
    field: str,
    *,
    semantic_fallback: str,
) -> dict[str, float]:
    raw = agent.get(field)
    values: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
            ):
                values[str(key)] = float(value)
    if values:
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}
    semantic = agent.get("semantic")
    fallback = (
        str(semantic.get(semantic_fallback) or "")
        if isinstance(semantic, Mapping)
        else ""
    )
    return {fallback or "unknown": 1.0}


def _cosine_similarity(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    keys = set(left) | set(right)
    numerator = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm <= 1e-15 or right_norm <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _stable_persona(agent: Mapping[str, Any]) -> Mapping[str, Any]:
    value = agent.get("stable_persona")
    return value if isinstance(value, Mapping) else {}


def _interaction_function(agent: Mapping[str, Any]) -> str:
    stable = _stable_persona(agent)
    language = stable.get("language_style")
    if isinstance(language, Mapping) and language.get("reply_function"):
        return str(language["reply_function"])
    semantic = agent.get("semantic")
    if isinstance(semantic, Mapping) and semantic.get("speech_act"):
        return str(semantic["speech_act"])
    behavior = agent.get("behavior_contract")
    if isinstance(behavior, Mapping) and behavior.get("interaction_role"):
        return str(behavior["interaction_role"])
    return "contextual_participant"


def _trust_signal(agent: Mapping[str, Any]) -> float:
    stable = _stable_persona(agent)
    trust = stable.get("trust_style")
    values = []
    if isinstance(trust, Mapping):
        values = [
            float(value)
            for value in trust.values()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
    if not values:
        state = agent.get("state")
        if isinstance(state, Mapping) and isinstance(state.get("trust"), (int, float)):
            values = [float(state["trust"])]
    return max(0.0, min(1.0, sum(values) / len(values))) if values else 0.5


def _reply_complement(left: str, right: str) -> float:
    complements = {
        "question": {"information_share", "correction", "discussion", "experience_share_or_ask"},
        "information_request": {"information_share", "correction", "experience_share_or_ask"},
        "help_request": {"information_share", "experience_share_or_ask", "discussion"},
        "risk_advice_request": {"correction", "information_share", "discussion"},
        "emotional_vent": {"discussion", "experience_share_or_ask", "information_share"},
        "complaint": {"correction", "discussion", "information_share"},
        "request": {"information_share", "experience_share_or_ask", "discussion"},
        "correction": {"question", "information_request", "complaint", "discussion"},
        "information_share": {"question", "information_request", "help_request", "request"},
    }
    if right in complements.get(left, set()) or left in complements.get(right, set()):
        return 1.0
    if left == right:
        return 0.35
    return 0.55


def _edge_features(agent: Mapping[str, Any]) -> dict[str, Any]:
    stable = _stable_persona(agent)
    return {
        "topics": _positive_distribution(
            agent, "topic_distribution", semantic_fallback="topic_code"
        ),
        "needs": _positive_distribution(
            agent, "need_distribution", semantic_fallback="need_code"
        ),
        "tendencies": _positive_distribution(
            {"distribution": stable.get("forum_tendency_distribution")},
            "distribution",
            semantic_fallback="speech_act",
        ),
        "function": _interaction_function(agent),
        "macro": str(agent.get("primary_archetype_id") or ""),
        "trust": _trust_signal(agent),
        "rhythm": str(stable.get("activity_rhythm") or ""),
    }


def _feature_affinity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    bridge: bool,
) -> float:
    topics = _cosine_similarity(left["topics"], right["topics"])
    needs = _cosine_similarity(left["needs"], right["needs"])
    tendencies = _cosine_similarity(left["tendencies"], right["tendencies"])
    left_function = str(left["function"])
    right_function = str(right["function"])
    same_function = float(left_function == right_function)
    same_macro = float(left["macro"] == right["macro"])
    trust_closeness = 1.0 - abs(float(left["trust"]) - float(right["trust"]))
    left_rhythm = str(left["rhythm"])
    right_rhythm = str(right["rhythm"])
    rhythm_closeness = float(bool(left_rhythm) and left_rhythm == right_rhythm)
    if bridge:
        score = (
            0.28 * topics
            + 0.17 * needs
            + 0.28 * _reply_complement(left_function, right_function)
            + 0.10 * tendencies
            + 0.09 * trust_closeness
            + 0.05 * rhythm_closeness
            + 0.03 * (1.0 - same_macro)
        )
    else:
        score = (
            0.38 * topics
            + 0.20 * needs
            + 0.12 * tendencies
            + 0.10 * same_function
            + 0.08 * same_macro
            + 0.07 * trust_closeness
            + 0.05 * rhythm_closeness
        )
    return max(0.0, min(1.0, float(score)))


def profile_edge_affinity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    bridge: bool,
) -> float:
    """Score one synthetic exposure edge from deidentified profile features.

    This is a modelling constraint, not an inferred real relationship.  The
    score uses topic/need portfolios, interaction function, evidence/trust
    handling and activity rhythm.  A bridge edge rewards complementary reply
    functions across macro roles; a semantic edge rewards local similarity.
    """

    return _feature_affinity(
        _edge_features(left),
        _edge_features(right),
        bridge=bridge,
    )


def build_synthetic_exposure_graph(
    population: Mapping[str, Any],
    *,
    seed: int = 20260730,
    profile_aware: bool = False,
) -> dict[str, Any]:
    """Build a deterministic directed 8+4 synthetic exposure graph."""

    agents = population.get("agents")
    if not isinstance(agents, list) or len(agents) < 16:
        raise MassPopulationError("exposure graph requires a compiled population")
    by_archetype: dict[str, list[int]] = defaultdict(list)
    by_topic: dict[str, list[int]] = defaultdict(list)
    for index, agent in enumerate(agents):
        by_archetype[str(agent["primary_archetype_id"])].append(index)
        by_topic[str(agent["semantic"]["topic_code"])].append(index)
    all_indices = list(range(len(agents)))
    profile_features = (
        [_edge_features(agent) for agent in agents]
        if profile_aware
        else None
    )

    def affinity(source: int, target: int, *, bridge: bool) -> float:
        if profile_features is None:
            return profile_edge_affinity(
                agents[source], agents[target], bridge=bridge
            )
        return _feature_affinity(
            profile_features[source], profile_features[target], bridge=bridge
        )

    edges = []
    for source, agent in enumerate(agents):
        if profile_aware:
            ordered_same = sorted(
                (index for index in all_indices if index != source),
                key=lambda target: (
                    -affinity(source, target, bridge=False),
                    canonical_hash(
                        [
                            seed,
                            "profile_semantic_neighbor",
                            agent["micro_agent_id"],
                            agents[target]["micro_agent_id"],
                        ]
                    ),
                    target,
                ),
            )
        else:
            same_candidates = (
                by_topic[str(agent["semantic"]["topic_code"])]
                + by_archetype[str(agent["primary_archetype_id"])]
            )
            same_candidates = [
                index for index in same_candidates if index != source
            ]
            ordered_same = _ordered_candidates(
                agents, source, same_candidates, "semantic_neighbor", seed
            )
            if len(ordered_same) < SEMANTIC_NEIGHBOR_COUNT:
                supplement = _ordered_candidates(
                    agents,
                    source,
                    (index for index in all_indices if index != source),
                    "semantic_supplement",
                    seed,
                )
                ordered_same.extend(
                    index for index in supplement if index not in ordered_same
                )
        semantic_targets = ordered_same[:SEMANTIC_NEIGHBOR_COUNT]

        cross_candidates = [
            index
            for index in all_indices
            if index != source
            and agents[index]["primary_archetype_id"]
            != agent["primary_archetype_id"]
            and index not in semantic_targets
        ]
        bridge_targets = (
            sorted(
                cross_candidates,
                key=lambda target: (
                    -affinity(source, target, bridge=True),
                    canonical_hash(
                        [
                            seed,
                            "profile_cross_archetype_bridge",
                            agent["micro_agent_id"],
                            agents[target]["micro_agent_id"],
                        ]
                    ),
                    target,
                ),
            )
            if profile_aware
            else _ordered_candidates(
                agents, source, cross_candidates, "cross_archetype_bridge", seed
            )
        )[:CROSS_ARCHETYPE_BRIDGE_COUNT]
        if len(bridge_targets) != CROSS_ARCHETYPE_BRIDGE_COUNT:
            raise MassPopulationError("population lacks cross-archetype bridges")
        for target in semantic_targets:
            edge = {
                "source_index": source,
                "target_index": target,
                "edge_type": "semantic_neighbor",
            }
            if profile_aware:
                edge["affinity_score"] = round(
                    affinity(source, target, bridge=False), 8
                )
            edges.append(edge)
        for target in bridge_targets:
            edge = {
                "source_index": source,
                "target_index": target,
                "edge_type": "cross_archetype_bridge",
            }
            if profile_aware:
                edge["affinity_score"] = round(
                    affinity(source, target, bridge=True), 8
                )
            edges.append(edge)
    expected = len(agents) * (
        SEMANTIC_NEIGHBOR_COUNT + CROSS_ARCHETYPE_BRIDGE_COUNT
    )
    if len(edges) != expected:
        raise MassPopulationError("exposure graph does not meet the 8+4 contract")
    body = {
        "schema_version": EXPOSURE_GRAPH_SCHEMA_VERSION,
        "graph_kind": "fully_synthetic_directed_exposure_graph",
        "population_sha256": population.get("population_sha256"),
        "seed": seed,
        "node_count": len(agents),
        "edge_count": len(edges),
        "out_degree": SEMANTIC_NEIGHBOR_COUNT
        + CROSS_ARCHETYPE_BRIDGE_COUNT,
        "edge_type_counts": dict(
            sorted(Counter(edge["edge_type"] for edge in edges).items())
        ),
        "selection_method": (
            "profile_semantic_affinity_v2"
            if profile_aware
            else "legacy_topic_macro_hash_v1"
        ),
        "selection_features": (
            [
                "topic_portfolio",
                "need_portfolio",
                "interaction_function",
                "forum_tendencies",
                "trust_or_evidence_handling",
                "activity_rhythm",
                "macro_role",
            ]
            if profile_aware
            else ["top_topic", "macro_role", "deterministic_hash"]
        ),
        "edges": edges,
        "non_claim": (
            "Synthetic exposure constraints do not represent real follows, "
            "friendships, or measured diffusion."
        ),
    }
    body["graph_sha256"] = canonical_hash(body)
    return body
