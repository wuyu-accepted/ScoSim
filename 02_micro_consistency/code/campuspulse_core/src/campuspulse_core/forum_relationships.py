"""ForumTwin v2 synthetic stakes, relationships and relation-aware attention.

The release builders consume only the already reviewed synthetic population.
They never infer or reconstruct a real person's relationship, demographic
attribute or longitudinal identity.  Public ranking remains global; this
module only controls whether one synthetic resident opens a globally ranked
item and which closed-set synthetic contacts may receive a private turn.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping, Sequence

from .forum_data import POPULATION_SCHEMA
from .forum_twin import ForumTwinError
from .mass_population import canonical_hash


RELATIONSHIP_RELEASE_SCHEMA_VERSION = (
    "campus-pulse-synthetic-relationship-release-v1"
)
STAKE_RELEASE_SCHEMA_VERSION = "campus-pulse-scenario-stake-release-v1"
RELATIONSHIP_UPDATE_SCHEMA_VERSION = "campus-pulse-relationship-update-v1"

RELATION_TYPES = frozenset(
    {"close_friend", "familiar_peer", "cross_group_acquaintance"}
)
STAKE_PROVENANCE = frozenset(
    {"forum_derived", "scenario_assumption", "runtime_state"}
)
STAKE_FIELDS = (
    "resource_need_intensity",
    "eligibility_material_completeness",
    "time_urgency",
    "service_channel_dependency",
    "information_access_capacity",
    "appeal_cost",
    "substitute_resource_availability",
    "directly_affected",
    "usage_frequency",
    "historical_service_experience",
    "bridge_organizing_care_responsibility",
)


@dataclass(frozen=True)
class SyntheticRelationship:
    relationship_id: str
    source_agent_id: str
    target_agent_id: str
    relation_type: str
    strength: float
    trust: float
    interaction_count: int = 0
    last_active_tick: int | None = None
    mutual: bool = False
    feature_basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.relationship_id or not self.source_agent_id or not self.target_agent_id:
            raise ForumTwinError("synthetic relationship identity is incomplete")
        if self.source_agent_id == self.target_agent_id:
            raise ForumTwinError("synthetic relationship cannot target itself")
        if self.relation_type not in RELATION_TYPES:
            raise ForumTwinError("synthetic relationship type is invalid")
        for value, label in ((self.strength, "strength"), (self.trust, "trust")):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ForumTwinError(f"synthetic relationship {label} is invalid")
        if self.interaction_count < 0:
            raise ForumTwinError("synthetic relationship interaction count is negative")
        if self.last_active_tick is not None and not 0 <= self.last_active_tick <= 23:
            raise ForumTwinError("synthetic relationship last-active tick is invalid")
        if not self.feature_basis or len(set(self.feature_basis)) != len(self.feature_basis):
            raise ForumTwinError("synthetic relationship feature basis is invalid")

    def private_view(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "relation_type": self.relation_type,
            "strength": round(float(self.strength), 6),
            "trust": round(float(self.trust), 6),
            "interaction_count": self.interaction_count,
            "last_active_tick": self.last_active_tick,
            "mutual": self.mutual,
            "feature_basis": list(self.feature_basis),
            "synthetic": True,
        }


@dataclass(frozen=True)
class RelationshipUpdate:
    update_id: str
    relationship_id: str
    tick: int
    strength_delta: float
    trust_delta: float
    interaction_increment: int
    reason: str

    def __post_init__(self) -> None:
        if not self.update_id or not self.relationship_id or not self.reason:
            raise ForumTwinError("relationship update identity is incomplete")
        if not 0 <= self.tick <= 23:
            raise ForumTwinError("relationship update tick is invalid")
        if not -0.15 <= float(self.strength_delta) <= 0.15:
            raise ForumTwinError("relationship strength delta exceeds its bound")
        if not -0.15 <= float(self.trust_delta) <= 0.15:
            raise ForumTwinError("relationship trust delta exceeds its bound")
        if self.interaction_increment not in {0, 1}:
            raise ForumTwinError("relationship interaction increment is invalid")


def apply_relationship_update(
    relationship: SyntheticRelationship,
    update: RelationshipUpdate,
) -> SyntheticRelationship:
    if update.relationship_id != relationship.relationship_id:
        raise ForumTwinError("relationship update targets another edge")
    if (
        relationship.last_active_tick is not None
        and update.tick < relationship.last_active_tick
    ):
        raise ForumTwinError("relationship update moved backwards in time")
    return replace(
        relationship,
        strength=min(1.0, max(0.0, relationship.strength + update.strength_delta)),
        trust=min(1.0, max(0.0, relationship.trust + update.trust_delta)),
        interaction_count=relationship.interaction_count + update.interaction_increment,
        last_active_tick=update.tick,
    )


def _population_agents(population: Mapping[str, Any]) -> list[dict[str, Any]]:
    if population.get("schema_version") != POPULATION_SCHEMA:
        raise ForumTwinError("ForumTwin v2 requires semantic population v2")
    expected = canonical_hash(
        {key: value for key, value in population.items() if key != "population_sha256"}
    )
    if population.get("population_sha256") != expected:
        raise ForumTwinError("ForumTwin population hash drifted")
    agents = population.get("agents")
    if not isinstance(agents, list) or len(agents) != 1_000:
        raise ForumTwinError("ForumTwin relationship release requires 1,000 agents")
    if len({str(agent.get("agent_id")) for agent in agents}) != len(agents):
        raise ForumTwinError("ForumTwin population repeats an agent identity")
    return [dict(agent) for agent in agents]


def _unit_interval(*parts: Any) -> float:
    return (int(canonical_hash(parts)[:16], 16) + 0.5) / 16**16


def _bounded(value: float, low: float = 0.03, high: float = 0.97) -> float:
    return round(min(high, max(low, float(value))), 6)


def _tendency(agent: Mapping[str, Any], *keys: str) -> float:
    values = agent.get("forum_tendencies")
    if not isinstance(values, Mapping):
        return 0.0
    return sum(float(values.get(key, 0.0)) for key in keys)


def _stake_value(value: Any, provenance: str, rationale_tag: str) -> dict[str, Any]:
    if provenance not in STAKE_PROVENANCE:
        raise ForumTwinError("scenario stake provenance is invalid")
    return {
        "value": value,
        "provenance": provenance,
        "rationale_tag": rationale_tag,
    }


def compile_scenario_stake_release_v1(
    *,
    population: Mapping[str, Any],
    scenario_ids: Sequence[str] | None = None,
    seed: int = 20260815,
) -> dict[str, Any]:
    """Create auditable synthetic interests without demographic inference."""

    agents = _population_agents(population)
    scenarios = tuple(
        dict.fromkeys(
            str(value)
            for value in (
                scenario_ids
                if scenario_ids is not None
                else population.get("scenario_ids", ())
            )
            if str(value)
        )
    )
    if not scenarios:
        raise ForumTwinError("scenario stake release requires a scenario")
    positions: list[dict[str, Any]] = []
    for scenario_id in scenarios:
        for agent in agents:
            agent_id = str(agent["agent_id"])
            attention = float(agent.get("attention_budget", 0.5))
            help_need = _tendency(agent, "help_request", "request")
            information = _tendency(agent, "information_share", "question")
            reviewed_trust = float(
                (agent.get("trust_style") or {}).get(
                    "trusts_reviewed_information", 0.5
                )
            )
            scenario_random = _unit_interval(
                "scenario-stake-v1", seed, scenario_id, agent_id
            )
            direct = scenario_random < _bounded(
                0.28 + 0.30 * help_need + 0.12 * attention,
                0.18,
                0.72,
            )
            fields = {
                "resource_need_intensity": _stake_value(
                    _bounded(0.18 + 0.52 * help_need + 0.18 * attention),
                    "forum_derived",
                    "reviewed_need_and_help_tendencies",
                ),
                "eligibility_material_completeness": _stake_value(
                    _bounded(0.25 + 0.65 * _unit_interval(seed, scenario_id, agent_id, "materials")),
                    "scenario_assumption",
                    "bounded_synthetic_eligibility_condition",
                ),
                "time_urgency": _stake_value(
                    _bounded(0.22 + 0.45 * attention + (0.18 if direct else 0.0)),
                    "scenario_assumption",
                    "scenario_deadline_and_attention_condition",
                ),
                "service_channel_dependency": _stake_value(
                    _bounded(0.15 + 0.60 * help_need),
                    "forum_derived",
                    "reviewed_help_and_request_tendencies",
                ),
                "information_access_capacity": _stake_value(
                    _bounded(0.20 + 0.46 * information + 0.20 * attention),
                    "forum_derived",
                    "reviewed_information_activity",
                ),
                "appeal_cost": _stake_value(
                    _bounded(0.18 + 0.68 * _unit_interval(seed, scenario_id, agent_id, "appeal")),
                    "scenario_assumption",
                    "bounded_synthetic_process_cost",
                ),
                "substitute_resource_availability": _stake_value(
                    _bounded(0.12 + 0.76 * _unit_interval(seed, scenario_id, agent_id, "substitute")),
                    "scenario_assumption",
                    "bounded_synthetic_alternative_supply",
                ),
                "directly_affected": _stake_value(
                    bool(direct),
                    "scenario_assumption",
                    "scenario_specific_synthetic_exposure",
                ),
                "usage_frequency": _stake_value(
                    _bounded(0.18 + 0.54 * attention + 0.16 * help_need),
                    "forum_derived",
                    "reviewed_activity_and_need_intensity",
                ),
                "historical_service_experience": _stake_value(
                    _bounded(0.12 + 0.70 * reviewed_trust),
                    "forum_derived",
                    "reviewed_information_trust_style",
                ),
                "bridge_organizing_care_responsibility": _stake_value(
                    _bounded(
                        0.10
                        + 0.42 * information
                        + 0.18 * _tendency(agent, "emotional_expression")
                    ),
                    "forum_derived",
                    "reviewed_sharing_and_interaction_function",
                ),
            }
            if set(fields) != set(STAKE_FIELDS):
                raise ForumTwinError("scenario stake fields drifted")
            positions.append(
                {
                    "stake_position_id": "stake-"
                    + canonical_hash([scenario_id, agent_id])[:24],
                    "scenario_id": scenario_id,
                    "agent_id": agent_id,
                    "fields": fields,
                }
            )
    body = {
        "schema_version": STAKE_RELEASE_SCHEMA_VERSION,
        "population_sha256": population["population_sha256"],
        "seed": seed,
        "scenario_ids": list(scenarios),
        "positions": positions,
        "field_provenance_contract": {
            "allowed": sorted(STAKE_PROVENANCE),
            "forbidden_inferences": [
                "gender",
                "age",
                "department",
                "real_identity",
                "real_relationship",
            ],
        },
    }
    return {**body, "stake_release_sha256": canonical_hash(body)}


def _tokens(agent: Mapping[str, Any]) -> set[str]:
    return {
        str(value).strip().lower()
        for key in ("topic_portfolio", "need_portfolio")
        for value in agent.get(key, ())
        if str(value).strip()
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _compatibility(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    source_tokens: set[str],
    target_tokens: set[str],
) -> float:
    trust_left = float(
        (source.get("trust_style") or {}).get("trusts_reviewed_information", 0.5)
    )
    trust_right = float(
        (target.get("trust_style") or {}).get("trusts_reviewed_information", 0.5)
    )
    reply_left = str((source.get("language_style") or {}).get("reply_function", ""))
    reply_right = str((target.get("language_style") or {}).get("reply_function", ""))
    return (
        0.45 * _jaccard(source_tokens, target_tokens)
        + 0.18 * (source.get("primary_macro_role") == target.get("primary_macro_role"))
        + 0.12 * (source.get("activity_rhythm") == target.get("activity_rhythm"))
        + 0.13 * (1.0 - abs(trust_left - trust_right))
        + 0.12 * (reply_left != reply_right)
    )


def compile_synthetic_relationship_release_v1(
    *,
    population: Mapping[str, Any],
    stake_release: Mapping[str, Any],
    seed: int = 20260815,
) -> dict[str, Any]:
    """Compile 2 mutual friends, 4 peers and 2 cross-group contacts per agent."""

    agents = _population_agents(population)
    if (
        stake_release.get("schema_version") != STAKE_RELEASE_SCHEMA_VERSION
        or stake_release.get("population_sha256") != population.get("population_sha256")
    ):
        raise ForumTwinError("relationship release stake lineage is invalid")
    expected_stake_hash = canonical_hash(
        {
            key: value
            for key, value in stake_release.items()
            if key != "stake_release_sha256"
        }
    )
    if stake_release.get("stake_release_sha256") != expected_stake_hash:
        raise ForumTwinError("relationship release stake hash drifted")
    by_id = {str(agent["agent_id"]): agent for agent in agents}
    tokens = {agent_id: _tokens(agent) for agent_id, agent in by_id.items()}
    ordered_ids = sorted(
        by_id,
        key=lambda agent_id: canonical_hash(
            ["relationship-cycle-start", seed, agent_id]
        ),
    )

    # A nearest-neighbour Hamiltonian cycle gives every node exactly two
    # reciprocal close friends while still conditioning edges on reviewed
    # topics, needs, interaction style and evidence acceptance.
    cycle = [ordered_ids[0]]
    remaining = set(ordered_ids[1:])
    while remaining:
        source_id = cycle[-1]
        target_id = max(
            remaining,
            key=lambda candidate_id: (
                _compatibility(
                    by_id[source_id],
                    by_id[candidate_id],
                    source_tokens=tokens[source_id],
                    target_tokens=tokens[candidate_id],
                ),
                canonical_hash([seed, source_id, candidate_id]),
            ),
        )
        cycle.append(target_id)
        remaining.remove(target_id)

    close_targets: dict[str, set[str]] = defaultdict(set)
    for index, source_id in enumerate(cycle):
        close_targets[source_id].add(cycle[(index - 1) % len(cycle)])
        close_targets[source_id].add(cycle[(index + 1) % len(cycle)])

    direct_by_agent: dict[str, bool] = {}
    primary_scenario = str(stake_release["scenario_ids"][0])
    for row in stake_release["positions"]:
        if row["scenario_id"] == primary_scenario:
            direct_by_agent[str(row["agent_id"])] = bool(
                row["fields"]["directly_affected"]["value"]
            )

    edges: list[SyntheticRelationship] = []
    for source_id in sorted(by_id):
        source = by_id[source_id]
        close = close_targets[source_id]
        candidate_scores = []
        for target_id, target in by_id.items():
            if target_id == source_id or target_id in close:
                continue
            candidate_scores.append(
                (
                    _compatibility(
                        source,
                        target,
                        source_tokens=tokens[source_id],
                        target_tokens=tokens[target_id],
                    ),
                    canonical_hash([seed, source_id, target_id, "familiar"]),
                    target_id,
                )
            )
        candidate_scores.sort(reverse=True)
        familiar = {value[2] for value in candidate_scores[:4]}
        cross_candidates = [
            value
            for value in candidate_scores
            if value[2] not in familiar
            and by_id[value[2]].get("primary_macro_role")
            != source.get("primary_macro_role")
            and direct_by_agent.get(value[2]) != direct_by_agent.get(source_id)
        ]
        if len(cross_candidates) < 2:
            cross_candidates = [
                value
                for value in candidate_scores
                if value[2] not in familiar
                and by_id[value[2]].get("primary_macro_role")
                != source.get("primary_macro_role")
            ]
        cross = {value[2] for value in cross_candidates[:2]}
        if len(close) != 2 or len(familiar) != 4 or len(cross) != 2:
            raise ForumTwinError("relationship compiler could not satisfy exact degree")
        for relation_type, targets in (
            ("close_friend", close),
            ("familiar_peer", familiar),
            ("cross_group_acquaintance", cross),
        ):
            for target_id in sorted(targets):
                score = _compatibility(
                    source,
                    by_id[target_id],
                    source_tokens=tokens[source_id],
                    target_tokens=tokens[target_id],
                )
                base = {
                    "close_friend": 0.72,
                    "familiar_peer": 0.48,
                    "cross_group_acquaintance": 0.34,
                }[relation_type]
                edges.append(
                    SyntheticRelationship(
                        relationship_id="relationship-"
                        + canonical_hash(
                            [source_id, target_id, relation_type]
                        )[:24],
                        source_agent_id=source_id,
                        target_agent_id=target_id,
                        relation_type=relation_type,
                        strength=_bounded(base + 0.20 * score),
                        trust=_bounded(base - 0.08 + 0.22 * score),
                        mutual=(relation_type == "close_friend"),
                        feature_basis=(
                            "topic_need_compatibility",
                            "interaction_function",
                            "evidence_acceptance",
                            "activity_rhythm",
                            "synthetic_stake_position",
                        ),
                    )
                )
    edge_keys = {
        (edge.source_agent_id, edge.target_agent_id, edge.relation_type)
        for edge in edges
    }
    for edge in edges:
        if edge.relation_type == "close_friend" and (
            edge.target_agent_id,
            edge.source_agent_id,
            "close_friend",
        ) not in edge_keys:
            raise ForumTwinError("close-friend relationship is not reciprocal")
    degree = Counter((edge.source_agent_id, edge.relation_type) for edge in edges)
    for agent_id in by_id:
        if (
            degree[(agent_id, "close_friend")] != 2
            or degree[(agent_id, "familiar_peer")] != 4
            or degree[(agent_id, "cross_group_acquaintance")] != 2
        ):
            raise ForumTwinError("relationship release degree contract drifted")
    matrix: dict[tuple[str, str, str], int] = Counter()
    for edge in edges:
        matrix[
            (
                str(by_id[edge.source_agent_id]["primary_macro_role"]),
                str(by_id[edge.target_agent_id]["primary_macro_role"]),
                edge.relation_type,
            )
        ] += 1
    body = {
        "schema_version": RELATIONSHIP_RELEASE_SCHEMA_VERSION,
        "population_sha256": population["population_sha256"],
        "stake_release_sha256": stake_release["stake_release_sha256"],
        "seed": seed,
        "agent_count": len(agents),
        "edge_count": len(edges),
        "edges": [edge.private_view() for edge in edges],
        "public_aggregate_matrix": [
            {
                "source_macro_role": source,
                "target_macro_role": target,
                "relation_type": relation_type,
                "directed_edge_count": count,
            }
            for (source, target, relation_type), count in sorted(matrix.items())
        ],
        "privacy": {
            "fully_synthetic": True,
            "maps_real_user_relationships": False,
            "record_level_edges_public": False,
        },
    }
    return {**body, "relationship_release_sha256": canonical_hash(body)}


def relationship_attention_probability(
    *,
    profile: Mapping[str, Any],
    current_need_match: float,
    relation_strength: float,
    relation_trust: float,
    participated_before: bool,
    direct_notification: bool,
) -> float:
    """Open probability only; callers must not use it to reorder the public board."""

    for value, label in (
        (current_need_match, "current_need_match"),
        (relation_strength, "relation_strength"),
        (relation_trust, "relation_trust"),
    ):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ForumTwinError(f"relationship attention {label} is invalid")
    attention = float(profile.get("attention_budget", 0.5))
    probability = (
        0.08
        + 0.28 * attention
        + 0.22 * float(current_need_match)
        + 0.16 * float(relation_strength)
        + 0.10 * float(relation_trust)
        + (0.08 if participated_before else 0.0)
        + (0.20 if direct_notification else 0.0)
    )
    return _bounded(probability, 0.01, 0.995)


def select_relation_attention_opens(
    *,
    agent_id: str,
    tick: int,
    globally_ranked_items: Sequence[Mapping[str, Any]],
    open_probabilities: Mapping[str, float],
    seed: int,
) -> tuple[str, ...]:
    """Deterministically decide opens without changing the common rank order."""

    if not 0 <= tick <= 23:
        raise ForumTwinError("relationship attention tick is invalid")
    message_ids = [str(item.get("message_id") or "") for item in globally_ranked_items]
    if not message_ids or any(not value for value in message_ids):
        raise ForumTwinError("relationship attention requires ranked message IDs")
    opened: list[str] = []
    for message_id in message_ids:
        probability = float(open_probabilities.get(message_id, 0.0))
        if not 0 <= probability <= 1:
            raise ForumTwinError("relationship attention probability is invalid")
        if _unit_interval("relationship-attention-v1", seed, agent_id, tick, message_id) <= probability:
            opened.append(message_id)
    return tuple(opened)


def private_activation_quota(
    *,
    non_anchor_slots: int,
    requested_private_activations: int,
) -> int:
    """Private-origin work may consume at most forty percent of non-anchor slots."""

    if non_anchor_slots < 0 or requested_private_activations < 0:
        raise ForumTwinError("private activation quota inputs must be nonnegative")
    return min(requested_private_activations, math.floor(non_anchor_slots * 0.40))


__all__ = [
    "RELATIONSHIP_RELEASE_SCHEMA_VERSION",
    "RELATIONSHIP_UPDATE_SCHEMA_VERSION",
    "STAKE_FIELDS",
    "STAKE_RELEASE_SCHEMA_VERSION",
    "RelationshipUpdate",
    "SyntheticRelationship",
    "apply_relationship_update",
    "compile_scenario_stake_release_v1",
    "compile_synthetic_relationship_release_v1",
    "private_activation_quota",
    "relationship_attention_probability",
    "select_relation_attention_opens",
]
