"""CampusPulse Governance Arena v2 particle/message simulation.

This module is deliberately additive.  It does not alter the frozen Arena v1
or MassSim v1 artifacts.  The model advances every reviewed forum-episode
parent with a small sequential-Monte-Carlo ensemble, while a fixed-size
probability sample supplies zero-Provider semantic observations.

The implementation is a clean-room CampusPulse design.  It borrows the broad
idea of combining cheap population dynamics with probability-sampled semantic
observations, but it neither imports nor copies CAPER implementation material.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import base64
import copy
import hashlib
import io
import json
import math
import random
import statistics
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import zlib

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - deployment dependency check
    raise RuntimeError(
        "Governance Arena v2 requires NumPy for vectorized particle updates"
    ) from exc

from .llm_agents import (
    ALLOWED_ACTIONS,
    ALLOWED_STANCES,
    NON_EVENT_TOPIC_ID,
    PUBLIC_ACTIONS,
    RenderedAgentTurn,
    render_format_repair_turn,
    validate_rendered_turn,
)
from .mass_population import canonical_hash, profile_edge_affinity
from .mass_simulation import (
    _integer_allocation,
    fixed_size_inclusion_probabilities,
    pivotal_sample,
)


RESULT_SCHEMA_VERSION = "campus-pulse-governance-arena-result-v2"
RUN_PLAN_SCHEMA_VERSION = "campus-pulse-run-plan-v6"
CHECKPOINT_SCHEMA_VERSION = "campus-pulse-governance-checkpoint-v2"
VISUALIZATION_SCHEMA_VERSION = "campus-pulse-governance-visualization-v2"
POPULATION_SCHEMA_VERSION = "semantic-population-v3"
GRAPH_SCHEMA_VERSION = "campus-pulse-message-exposure-graph-v2"
PROBE_SCHEMA_VERSION = "campus-pulse-budgeted-semantic-probe-v2"
GAME_SCHEMA_VERSION = "campus-pulse-limited-observation-qre-game-v2"
MESSAGE_SCHEMA_VERSION = "campus-pulse-message-object-v2"
CALIBRATION_GATE_SCHEMA_VERSION = "campus-pulse-probe-calibration-gate-v2"

PARENT_COUNT = 1000
PARTICLES_PER_PARENT = 10
PARTICLE_COUNT = PARENT_COUNT * PARTICLES_PER_PARENT
PROBE_BUDGET = 32
SEED_COUNT = 8
BRANCH_IDS = ("natural", "A", "B", "C", "D")
SCENARIO_IDS = (
    "lecture_external_incident_shock",
    "governance_legitimacy_dispute",
)
WRITEBACK_MODES = ("observation_only", "stance_only", "full_state")
FORMAL_WRITEBACK_MODE = "observation_only"
TIMELINE_PHASES = (
    ("baseline", 3),
    ("burst", 5),
    ("spread", 10),
    ("decay", 6),
)
TIMELINE = tuple(
    phase for phase, count in TIMELINE_PHASES for _ in range(count)
)
PROBE_TICKS = (2, 7, 13, 19)
DECISION_TICKS = (3, 7, 13, 19)
CHECKPOINT_TICKS = (2, 3, 7, 13, 19, 23)
ACTION_CATEGORIES = tuple(ALLOWED_ACTIONS)
STANCE_CATEGORIES = tuple(ALLOWED_STANCES)

STATE_FIELDS = (
    "attention",
    "concern",
    "trust",
    "satisfaction",
    "support",
    "opposition",
    "activity",
    "voice",
    "service_strain",
    "rumor_belief",
    "verified_belief",
    "evidence_acceptance",
)
(
    ATTENTION,
    CONCERN,
    TRUST,
    SATISFACTION,
    SUPPORT,
    OPPOSITION,
    ACTIVITY,
    VOICE,
    SERVICE_STRAIN,
    RUMOR_BELIEF,
    VERIFIED_BELIEF,
    EVIDENCE_ACCEPTANCE,
) = range(len(STATE_FIELDS))

ArtifactCallback = Callable[[str, Mapping[str, Any]], None]


class GovernanceArenaV2Error(ValueError):
    """Raised when a v2 identity, state, or public contract is invalid."""


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _sigmoid_array(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _seed_from(*parts: Any) -> int:
    return int(canonical_hash(list(parts))[:16], 16)


def _hash_uniform(*parts: Any) -> float:
    return int(canonical_hash(list(parts))[:13], 16) / float(16**13)


def _phase_intensity(tick: int) -> float:
    if not 0 <= tick < len(TIMELINE):
        raise GovernanceArenaV2Error("tick is outside the frozen 24-point timeline")
    return (
        0.025,
        0.035,
        0.050,
        0.72,
        0.91,
        0.96,
        0.89,
        0.82,
        0.75,
        0.70,
        0.65,
        0.61,
        0.57,
        0.53,
        0.49,
        0.45,
        0.40,
        0.35,
        0.29,
        0.24,
        0.19,
        0.14,
        0.10,
        0.07,
    )[tick]


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    portion = position - lower
    return ordered[lower] * (1.0 - portion) + ordered[upper] * portion


def _normal_interval(mean: float, ess: float) -> dict[str, float]:
    radius = 1.96 * math.sqrt(max(mean * (1.0 - mean), 1e-6) / max(ess, 1.0))
    return {
        "mean": round(_clip(mean), 6),
        "lower": round(_clip(mean - radius), 6),
        "upper": round(_clip(mean + radius), 6),
    }


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    bounded = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(bounded.values())
    if total <= 1e-12:
        return {key: 1.0 / max(len(bounded), 1) for key in bounded}
    return {key: value / total for key, value in bounded.items()}


@dataclass(frozen=True)
class MessageObject:
    """Private synthetic message object; public results expose aggregates only."""

    message_id: str
    parent_message_id: str | None
    claim_id: str
    source_role: str
    source_parent_index: int | None
    source_group: str
    target_group: str
    topic: str
    stance_support: float
    stance_opposition: float
    evidence_status: str
    emotion: float
    actionability: float
    distortion: float
    created_tick: int
    ttl: int
    action: str
    provenance: str
    bridge: bool
    depth: int

    def private_view(self) -> dict[str, Any]:
        return {
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "message_id": self.message_id,
            "parent_message_id": self.parent_message_id,
            "claim_id": self.claim_id,
            "source_role": self.source_role,
            "source_parent_index": self.source_parent_index,
            "source_group": self.source_group,
            "target_group": self.target_group,
            "topic": self.topic,
            "stance": {
                "support": float(self.stance_support),
                "opposition": float(self.stance_opposition),
            },
            "evidence_status": self.evidence_status,
            "emotion": float(self.emotion),
            "actionability": float(self.actionability),
            "distortion": float(self.distortion),
            "created_tick": self.created_tick,
            "ttl": self.ttl,
            "target_group": self.target_group,
            "action": self.action,
            "provenance": self.provenance,
            "bridge": self.bridge,
            "depth": self.depth,
        }


@dataclass
class GraphArrays:
    targets: np.ndarray
    bridge: np.ndarray
    weight: np.ndarray
    capacity: np.ndarray
    delay: np.ndarray
    graph_sha256: str
    public_identity: dict[str, Any]


@dataclass
class ParticleEnsemble:
    values: np.ndarray
    weights: np.ndarray
    parent_weights: np.ndarray
    resampling_count: int = 0
    innovations: set[str] = field(default_factory=set)

    def clone(self) -> "ParticleEnsemble":
        return ParticleEnsemble(
            values=self.values.copy(),
            weights=self.weights.copy(),
            parent_weights=self.parent_weights.copy(),
            resampling_count=self.resampling_count,
            innovations=set(self.innovations),
        )

    @property
    def parent_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def particle_count(self) -> int:
        return int(self.values.shape[0] * self.values.shape[1])

    def normalized_within_parent(self) -> np.ndarray:
        return self.weights / np.maximum(
            self.weights.sum(axis=1, keepdims=True), 1e-15
        )

    def parent_means(self) -> np.ndarray:
        normalized = self.normalized_within_parent()
        return np.einsum("np,npf->nf", normalized, self.values, optimize=True)

    def parent_variances(self) -> np.ndarray:
        normalized = self.normalized_within_parent()
        means = self.parent_means()
        centered = self.values - means[:, None, :]
        return np.einsum(
            "np,npf->nf", normalized, centered * centered, optimize=True
        )

    def parent_ess(self) -> np.ndarray:
        normalized = self.normalized_within_parent()
        return 1.0 / np.maximum(np.sum(normalized * normalized, axis=1), 1e-15)

    def weighted_mean(self) -> np.ndarray:
        parent = self.parent_means()
        return np.sum(parent * self.parent_weights[:, None], axis=0)

    def assert_conserved(self) -> None:
        if (
            self.values.ndim != 3
            or self.values.shape[1] not in {1, PARTICLES_PER_PARENT}
            or self.values.shape[2] != len(STATE_FIELDS)
            or self.weights.shape != self.values.shape[:2]
            or self.parent_weights.shape != (self.values.shape[0],)
        ):
            raise GovernanceArenaV2Error("particle state has the wrong shape")
        if not np.isfinite(self.values).all() or not np.isfinite(self.weights).all():
            raise GovernanceArenaV2Error("particle state contains non-finite values")
        if float(np.min(self.values)) < -1e-7 or float(np.max(self.values)) > 1.0000001:
            raise GovernanceArenaV2Error("particle state escaped its bounded domain")
        row_mass = self.weights.sum(axis=1)
        if not np.allclose(row_mass, self.parent_weights, atol=1e-10, rtol=0.0):
            raise GovernanceArenaV2Error("particle weights do not conserve parent weights")
        if not math.isclose(float(row_mass.sum()), 1.0, abs_tol=1e-9):
            raise GovernanceArenaV2Error("particle weights do not sum to one")


@dataclass
class MessageLedger:
    messages: list[MessageObject]
    last_by_parent: np.ndarray
    last_tick_by_parent: np.ndarray
    processed_per_parent: np.ndarray

    @classmethod
    def empty(cls, parent_count: int) -> "MessageLedger":
        return cls(
            messages=[],
            last_by_parent=np.full(parent_count, -1, dtype=np.int64),
            last_tick_by_parent=np.full(parent_count, -100, dtype=np.int16),
            processed_per_parent=np.zeros(parent_count, dtype=np.int8),
        )

    def clone(self) -> "MessageLedger":
        return MessageLedger(
            messages=list(self.messages),
            last_by_parent=self.last_by_parent.copy(),
            last_tick_by_parent=self.last_tick_by_parent.copy(),
            processed_per_parent=self.processed_per_parent.copy(),
        )

    def add(self, message: MessageObject) -> int:
        if (
            message.parent_message_id
            and any(
                prior.message_id == message.parent_message_id
                and prior.created_tick >= message.created_tick
                for prior in self.messages
            )
        ):
            raise GovernanceArenaV2Error("message lineage must advance in time")
        position = len(self.messages)
        self.messages.append(message)
        if message.source_parent_index is not None:
            self.last_by_parent[message.source_parent_index] = position
            self.last_tick_by_parent[message.source_parent_index] = message.created_tick
        return position


@dataclass
class BranchState:
    ensemble: ParticleEnsemble
    ledger: MessageLedger
    rotation_counts: np.ndarray
    actor_beliefs: dict[str, dict[str, float]]
    pending_actions: list[dict[str, Any]]
    active_effects: dict[str, float]
    resource_opening: float = 100.0
    resource_consumed: float = 0.0
    resource_committed: float = 0.0
    game_rounds: list[dict[str, Any]] = field(default_factory=list)
    previous_checkpoint_sha256: str = "0" * 64

    def clone(self) -> "BranchState":
        return BranchState(
            ensemble=self.ensemble.clone(),
            ledger=self.ledger.clone(),
            rotation_counts=self.rotation_counts.copy(),
            actor_beliefs=copy.deepcopy(self.actor_beliefs),
            pending_actions=copy.deepcopy(self.pending_actions),
            active_effects=dict(self.active_effects),
            resource_opening=self.resource_opening,
            resource_consumed=self.resource_consumed,
            resource_committed=self.resource_committed,
            game_rounds=copy.deepcopy(self.game_rounds),
            previous_checkpoint_sha256=self.previous_checkpoint_sha256,
        )


def compile_semantic_population_v3(
    population: Mapping[str, Any],
    *,
    particles_per_parent: int = PARTICLES_PER_PARENT,
    strict_scale: bool = True,
) -> dict[str, Any]:
    """Bind reviewed v2 parents to the v3 particle-population identity."""

    agents = population.get("agents")
    if not isinstance(agents, list) or len(agents) < PROBE_BUDGET:
        raise GovernanceArenaV2Error("v3 population requires at least 32 parents")
    if strict_scale and len(agents) != PARENT_COUNT:
        raise GovernanceArenaV2Error("formal v3 population requires exactly 1,000 parents")
    if particles_per_parent != PARTICLES_PER_PARENT:
        raise GovernanceArenaV2Error("formal v3 requires ten particles per parent")
    parent_weights = [float(agent.get("population_weight", 0.0)) for agent in agents]
    if any(weight <= 0.0 for weight in parent_weights):
        raise GovernanceArenaV2Error("every parent requires positive design weight")
    if not math.isclose(sum(parent_weights), 1.0, abs_tol=1e-8):
        raise GovernanceArenaV2Error("parent design weights must sum to one")
    body = {
        "schema_version": POPULATION_SCHEMA_VERSION,
        "population_kind": "reviewed_forum_episode_parent_with_smc_uncertainty",
        "source_population_sha256": population.get("population_sha256"),
        "parent_count": len(agents),
        "particles_per_parent": particles_per_parent,
        "particle_count": len(agents) * particles_per_parent,
        "parent_weight_sum": round(sum(parent_weights), 12),
        "particle_weight_sum": 1.0,
        "state_fields": list(STATE_FIELDS),
        "evidence_subject_count": len(agents),
        "privacy": {
            "raw_text_persisted": False,
            "source_identifiers_persisted": False,
            "particle_parent_mapping_public": False,
        },
        "non_claims": [
            "Parents are reviewed forum behaviour episodes, not identified students.",
            "Particles are uncertainty hypotheses, not people or evidence subjects.",
        ],
    }
    body["population_sha256"] = canonical_hash(body)
    return body


def compile_message_exposure_graph_v2(
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> GraphArrays:
    """Add weights, capacities, and delays to the existing deterministic 8+4 graph."""

    agents = population.get("agents")
    edges = graph.get("edges")
    if not isinstance(agents, list) or not isinstance(edges, list):
        raise GovernanceArenaV2Error("population or synthetic graph is missing")
    count = len(agents)
    outgoing: list[list[Mapping[str, Any]]] = [[] for _ in range(count)]
    for edge in edges:
        source = int(edge.get("source_index", -1))
        target = int(edge.get("target_index", -1))
        if not 0 <= source < count or not 0 <= target < count or source == target:
            raise GovernanceArenaV2Error("synthetic exposure edge is invalid")
        outgoing[source].append(edge)
    if any(len(row) != 12 for row in outgoing):
        raise GovernanceArenaV2Error("v2 exposure graph requires 12 edges per parent")

    targets = np.empty((count, 12), dtype=np.int32)
    bridge = np.empty((count, 12), dtype=bool)
    weight = np.empty((count, 12), dtype=np.float32)
    capacity = np.empty((count, 12), dtype=np.int8)
    delay = np.empty((count, 12), dtype=np.int8)
    distances: list[float] = []
    random_distances: list[float] = []

    profile_aware = graph.get("selection_method") == "profile_semantic_affinity_v2"

    def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        if profile_aware:
            return 1.0 - profile_edge_affinity(left, right, bridge=False)
        l_sem = left["semantic"]
        r_sem = right["semantic"]
        return (
            0.32 * float(l_sem.get("topic_code") != r_sem.get("topic_code"))
            + 0.24 * float(l_sem.get("need_code") != r_sem.get("need_code"))
            + 0.19
            * float(
                left.get("primary_archetype_id")
                != right.get("primary_archetype_id")
            )
            + 0.15 * float(l_sem.get("risk_level") != r_sem.get("risk_level"))
            + 0.10
            * abs(
                float(l_sem.get("confidence", 0.5))
                - float(r_sem.get("confidence", 0.5))
            )
        )

    private_edges = []
    for source, row in enumerate(outgoing):
        ordered = sorted(
            row,
            key=lambda item: (
                1 if item.get("edge_type") == "cross_archetype_bridge" else 0,
                int(item["target_index"]),
            ),
        )
        for position, edge in enumerate(ordered):
            target = int(edge["target_index"])
            is_bridge = edge.get("edge_type") == "cross_archetype_bridge"
            dist = distance(agents[source], agents[target])
            edge_affinity = edge.get("affinity_score")
            if edge_affinity is None:
                affinity = 1.0 - dist
            elif (
                isinstance(edge_affinity, bool)
                or not isinstance(edge_affinity, (int, float))
                or not math.isfinite(float(edge_affinity))
                or not 0.0 <= float(edge_affinity) <= 1.0
            ):
                raise GovernanceArenaV2Error(
                    "synthetic exposure affinity is invalid"
                )
            else:
                affinity = float(edge_affinity)
            targets[source, position] = target
            bridge[source, position] = is_bridge
            weight[source, position] = 0.54 + 0.40 * affinity
            capacity[source, position] = 2 if is_bridge else 1
            delay[source, position] = 2 if is_bridge else 1
            if not is_bridge:
                distances.append(dist)
            random_target = (source + 97 + 37 * position) % count
            if random_target == source:
                random_target = (random_target + 1) % count
            random_distances.append(distance(agents[source], agents[random_target]))
            private_edges.append(
                {
                    "source_index": source,
                    "target_index": target,
                    "edge_type": (
                        "cross_archetype_bridge"
                        if is_bridge
                        else "semantic_neighbor"
                    ),
                    "weight": round(float(weight[source, position]), 6),
                    "capacity": int(capacity[source, position]),
                    "delay": int(delay[source, position]),
                }
            )
    body = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "graph_kind": "fully_synthetic_directed_message_exposure_graph",
        "source_graph_sha256": graph.get("graph_sha256"),
        "population_sha256": population.get("population_sha256"),
        "node_count": count,
        "edge_count": count * 12,
        "out_degree": 12,
        "semantic_edges_per_node": 8,
        "bridge_edges_per_node": 4,
        "selection_method": graph.get(
            "selection_method", "legacy_topic_macro_hash_v1"
        ),
        "profile_affinity_controls_weight": profile_aware,
        "mean_semantic_neighbor_distance": round(statistics.fmean(distances), 6),
        "mean_deterministic_random_distance": round(
            statistics.fmean(random_distances), 6
        ),
        "semantic_neighbor_quality_passed": (
            statistics.fmean(distances) < statistics.fmean(random_distances)
        ),
        "non_claim": (
            "The graph is fully synthetic and is not a measured follow, "
            "friendship, or diffusion network."
        ),
        "_edges": private_edges,
    }
    public_hash_body = {key: value for key, value in body.items() if key != "_edges"}
    public_hash_body["private_edge_sha256"] = canonical_hash(private_edges)
    public_hash_body["graph_sha256"] = canonical_hash(public_hash_body)
    return GraphArrays(
        targets=targets,
        bridge=bridge,
        weight=weight,
        capacity=capacity,
        delay=delay,
        graph_sha256=public_hash_body["graph_sha256"],
        public_identity=public_hash_body,
    )


def _base_parent_state(population: Mapping[str, Any]) -> np.ndarray:
    rows = []
    for agent in population["agents"]:
        state = agent["state"]
        semantic = agent["semantic"]
        risk = {
            "none": 0.05,
            "low": 0.20,
            "medium": 0.46,
            "high": 0.76,
            "uncertain": 0.38,
        }.get(str(semantic.get("risk_level")), 0.30)
        activity = float(
            state.get("activity_probability", state.get("activity", 0.2))
        )
        concern = float(state.get("concern", 0.3))
        trust = float(state.get("trust", 0.5))
        rows.append(
            [
                float(state.get("attention", 0.25)),
                concern,
                trust,
                float(state.get("satisfaction", 0.5)),
                float(state.get("support", 0.3)),
                float(state.get("opposition", 0.3)),
                activity,
                _clip(0.10 + 0.55 * activity + 0.22 * concern),
                _clip(0.08 + 0.34 * concern + 0.18 * risk),
                _clip(0.05 + 0.22 * risk + 0.16 * concern),
                _clip(0.08 + 0.22 * trust + 0.10 * (1.0 - risk)),
                _clip(0.28 + 0.52 * trust - 0.12 * risk),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def initialize_particle_ensemble(
    population: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
) -> ParticleEnsemble:
    """Create ten bounded, deterministic state hypotheses around each parent."""

    base = _base_parent_state(population)
    rng = np.random.default_rng(_seed_from(seed, scenario_id, "particle-init"))
    scale = np.asarray(
        [0.035, 0.045, 0.035, 0.035, 0.04, 0.04, 0.045, 0.05, 0.04, 0.055, 0.045, 0.04],
        dtype=np.float32,
    )
    perturbation = rng.normal(
        0.0, 1.0, size=(len(base), PARTICLES_PER_PARENT, len(STATE_FIELDS))
    ).astype(np.float32)
    perturbation -= perturbation.mean(axis=1, keepdims=True)
    values = np.clip(base[:, None, :] + perturbation * scale, 0.0, 1.0)
    parent_weights = np.asarray(
        [float(agent["population_weight"]) for agent in population["agents"]],
        dtype=np.float64,
    )
    weights = np.repeat(
        (parent_weights / PARTICLES_PER_PARENT)[:, None],
        PARTICLES_PER_PARENT,
        axis=1,
    )
    ensemble = ParticleEnsemble(values, weights, parent_weights)
    ensemble.assert_conserved()
    return ensemble


def _checkpoint_metadata(state: BranchState) -> dict[str, Any]:
    return {
        "innovations": sorted(state.ensemble.innovations),
        "resampling_count": state.ensemble.resampling_count,
        "rotation_counts": state.rotation_counts.tolist(),
        "actor_beliefs": state.actor_beliefs,
        "pending_actions": state.pending_actions,
        "active_effects": state.active_effects,
        "resource": {
            "opening": state.resource_opening,
            "consumed": state.resource_consumed,
            "committed": state.resource_committed,
        },
        "game_rounds": state.game_rounds,
        "messages": [dict(message.__dict__) for message in state.ledger.messages],
        "last_by_parent": state.ledger.last_by_parent.tolist(),
        "last_tick_by_parent": state.ledger.last_tick_by_parent.tolist(),
    }


def encode_governance_checkpoint_v2(
    *,
    population_sha256: str,
    graph_sha256: str,
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    state: BranchState,
    previous_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Encode the complete deterministic particle/message/game state."""

    metadata_raw = json.dumps(
        _checkpoint_metadata(state),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    buffer = io.BytesIO()
    # The whole NPZ is compressed once below.  Compressing every NPZ member
    # and then applying a second level-9 zlib pass made the 26 checkpoints in
    # each YuLan-Scale scenario/seed the dominant runtime cost while saving
    # negligible bytes.  A single deterministic level-1 stream remains well
    # below the frozen 8 MB checkpoint ceiling and is decoded by the same
    # existing path.
    np.savez(
        buffer,
        values=state.ensemble.values.astype(np.float32),
        weights=state.ensemble.weights.astype(np.float64),
        parent_weights=state.ensemble.parent_weights.astype(np.float64),
        metadata=np.frombuffer(metadata_raw, dtype=np.uint8),
    )
    compressed = zlib.compress(buffer.getvalue(), level=1)
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "population_sha256": population_sha256,
        "graph_sha256": graph_sha256,
        "scenario_id": scenario_id,
        "branch": branch,
        "seed": int(seed),
        "tick": int(tick),
        "compression": "npz+zlib+base64",
        "compressed_bytes": len(compressed),
        "payload_sha256": hashlib.sha256(compressed).hexdigest(),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "payload": base64.b64encode(compressed).decode("ascii"),
    }
    body["checkpoint_sha256"] = canonical_hash(body)
    return body


def decode_governance_checkpoint_v2(
    checkpoint: Mapping[str, Any],
    *,
    population_sha256: str,
    graph_sha256: str,
) -> BranchState:
    expected = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "artifact_kind",
            "content_sha256",
            "artifact_sha256",
            "timeline_prefix",
            "visual_frames_prefix",
        }
    }
    stored_hash = expected.pop("checkpoint_sha256", None)
    if stored_hash != canonical_hash(expected):
        raise GovernanceArenaV2Error("checkpoint hash does not match")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("population_sha256") != population_sha256
        or checkpoint.get("graph_sha256") != graph_sha256
        or checkpoint.get("compression") != "npz+zlib+base64"
    ):
        raise GovernanceArenaV2Error("checkpoint identity does not match")
    try:
        compressed = base64.b64decode(str(checkpoint["payload"]), validate=True)
        if hashlib.sha256(compressed).hexdigest() != checkpoint["payload_sha256"]:
            raise GovernanceArenaV2Error("checkpoint payload hash does not match")
        payload = zlib.decompress(compressed)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            values = archive["values"].copy()
            weights = archive["weights"].copy()
            parent_weights = archive["parent_weights"].copy()
            metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
    except (KeyError, ValueError, TypeError, zlib.error, json.JSONDecodeError) as exc:
        raise GovernanceArenaV2Error("checkpoint payload is invalid") from exc
    messages = [
        MessageObject(
            message_id=str(item["message_id"]),
            parent_message_id=item.get("parent_message_id"),
            claim_id=str(item["claim_id"]),
            source_role=str(item["source_role"]),
            source_parent_index=item.get("source_parent_index"),
            source_group=str(item["source_group"]),
            target_group=str(item["target_group"]),
            topic=str(item["topic"]),
            stance_support=float(
                item.get(
                    "stance_support",
                    item.get("stance", {}).get("support", 0.0),
                )
            ),
            stance_opposition=float(
                item.get(
                    "stance_opposition",
                    item.get("stance", {}).get("opposition", 0.0),
                )
            ),
            evidence_status=str(item["evidence_status"]),
            emotion=float(item["emotion"]),
            actionability=float(item["actionability"]),
            distortion=float(item["distortion"]),
            created_tick=int(item["created_tick"]),
            ttl=int(item["ttl"]),
            action=str(item["action"]),
            provenance=str(item["provenance"]),
            bridge=bool(item["bridge"]),
            depth=int(item["depth"]),
        )
        for item in metadata["messages"]
    ]
    ensemble = ParticleEnsemble(
        values=values,
        weights=weights,
        parent_weights=parent_weights,
        resampling_count=int(metadata["resampling_count"]),
        innovations=set(metadata["innovations"]),
    )
    ensemble.assert_conserved()
    count = ensemble.parent_count
    state = BranchState(
        ensemble=ensemble,
        ledger=MessageLedger(
            messages=messages,
            last_by_parent=np.asarray(metadata["last_by_parent"], dtype=np.int64),
            last_tick_by_parent=np.asarray(
                metadata["last_tick_by_parent"], dtype=np.int16
            ),
            processed_per_parent=np.zeros(count, dtype=np.int8),
        ),
        rotation_counts=np.asarray(metadata["rotation_counts"], dtype=np.int16),
        actor_beliefs=metadata["actor_beliefs"],
        pending_actions=metadata["pending_actions"],
        active_effects={
            key: float(value) for key, value in metadata["active_effects"].items()
        },
        resource_opening=float(metadata["resource"]["opening"]),
        resource_consumed=float(metadata["resource"]["consumed"]),
        resource_committed=float(metadata["resource"]["committed"]),
        game_rounds=metadata["game_rounds"],
        previous_checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
    )
    if state.ledger.last_by_parent.shape != (count,):
        raise GovernanceArenaV2Error("checkpoint message index shape is invalid")
    return state


def _parent_features(
    population: Mapping[str, Any], scenario_id: str
) -> dict[str, np.ndarray]:
    relevance = []
    risk = []
    archetypes = []
    topic_codes = []
    need_codes = []
    for agent in population["agents"]:
        semantic = agent["semantic"]
        risk_value = {
            "none": 0.04,
            "low": 0.20,
            "medium": 0.47,
            "high": 0.78,
            "uncertain": 0.40,
        }.get(str(semantic.get("risk_level")), 0.30)
        alignment = _hash_uniform(
            scenario_id,
            semantic.get("topic_code"),
            semantic.get("need_code"),
            agent.get("primary_archetype_id"),
        )
        relevance.append(_clip(0.55 * alignment + 0.50 * risk_value))
        risk.append(risk_value)
        archetypes.append(str(agent.get("primary_archetype_id") or "archetype-00"))
        topic_codes.append(str(semantic.get("topic_code") or "topic-other"))
        need_codes.append(str(semantic.get("need_code") or "need-other"))
    return {
        "relevance": np.asarray(relevance, dtype=np.float32),
        "risk": np.asarray(risk, dtype=np.float32),
        "archetypes": np.asarray(archetypes, dtype=object),
        "topic_codes": np.asarray(topic_codes, dtype=object),
        "need_codes": np.asarray(need_codes, dtype=object),
    }


def _empty_actor_beliefs() -> dict[str, dict[str, float]]:
    return {
        "governance_authority": {
            "concern": 0.30,
            "trust": 0.50,
            "rumor_belief": 0.15,
            "verified_belief": 0.15,
            "risk_reports": 0.0,
        },
        "service_operator": {
            "service_strain": 0.20,
            "help_queue": 0.0,
            "satisfaction": 0.50,
            "sla_pressure": 0.0,
        },
        "community_bridge": {
            "concern": 0.30,
            "voice_gap": 0.15,
            "bridge_share": 0.25,
            "participation": 0.20,
        },
    }


def initial_branch_state(
    population: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
) -> BranchState:
    count = len(population["agents"])
    return BranchState(
        ensemble=initialize_particle_ensemble(
            population, scenario_id=scenario_id, seed=seed
        ),
        ledger=MessageLedger.empty(count),
        rotation_counts=np.zeros(count, dtype=np.int16),
        actor_beliefs=_empty_actor_beliefs(),
        pending_actions=[],
        active_effects={
            "verification": 0.0,
            "reassurance": 0.0,
            "service": 0.0,
            "voice": 0.0,
            "bridge": 0.0,
            "prebunk": 0.0,
        },
    )


ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "verified_update": {
        "actor": "governance_authority",
        "cost": 14.0,
        "delay": 1,
        "reliability": 0.90,
        "effects": {"verification": 0.30, "reassurance": 0.18},
        "source_role": "governance_authority",
        "evidence_status": "verified",
    },
    "evidence_dashboard": {
        "actor": "governance_authority",
        "cost": 11.0,
        "delay": 2,
        "reliability": 0.86,
        "effects": {"verification": 0.22, "reassurance": 0.10},
        "source_role": "governance_authority",
        "evidence_status": "verified",
    },
    "listening_session": {
        "actor": "governance_authority",
        "cost": 13.0,
        "delay": 2,
        "reliability": 0.78,
        "effects": {"voice": 0.30, "reassurance": 0.09},
        "source_role": "governance_authority",
        "evidence_status": "contextual",
    },
    "support_channel": {
        "actor": "service_operator",
        "cost": 16.0,
        "delay": 1,
        "reliability": 0.84,
        "effects": {"service": 0.32, "reassurance": 0.08},
        "source_role": "service_operator",
        "evidence_status": "service_receipt",
    },
    "queue_status": {
        "actor": "service_operator",
        "cost": 9.0,
        "delay": 1,
        "reliability": 0.92,
        "effects": {"service": 0.18, "verification": 0.08},
        "source_role": "service_operator",
        "evidence_status": "service_receipt",
    },
    "peer_feedback": {
        "actor": "community_bridge",
        "cost": 12.0,
        "delay": 2,
        "reliability": 0.76,
        "effects": {"voice": 0.25, "bridge": 0.14},
        "source_role": "community_bridge",
        "evidence_status": "contextual",
    },
    "targeted_prebunk": {
        "actor": "governance_authority",
        "cost": 12.0,
        "delay": 1,
        "reliability": 0.82,
        "effects": {"prebunk": 0.32, "verification": 0.14},
        "source_role": "governance_authority",
        "evidence_status": "verified",
    },
    "bridge_outreach": {
        "actor": "community_bridge",
        "cost": 14.0,
        "delay": 1,
        "reliability": 0.80,
        "effects": {"bridge": 0.30, "voice": 0.18},
        "source_role": "community_bridge",
        "evidence_status": "contextual",
    },
}

BRANCH_ACTIONS = {
    "natural": (),
    "A": ("verified_update", "evidence_dashboard", "queue_status"),
    "B": ("listening_session", "support_channel", "peer_feedback"),
    "C": ("targeted_prebunk", "bridge_outreach", "queue_status"),
    "D": tuple(ACTION_CATALOG),
}


def _activate_due_actions(
    state: BranchState,
    *,
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
) -> list[MessageObject]:
    for key in state.active_effects:
        state.active_effects[key] *= 0.74
        if state.active_effects[key] < 1e-6:
            state.active_effects[key] = 0.0
    due = [item for item in state.pending_actions if item["due_tick"] <= tick]
    state.pending_actions = [
        item for item in state.pending_actions if item["due_tick"] > tick
    ]
    messages = []
    for item in due:
        state.resource_committed -= float(item["cost"])
        state.resource_consumed += float(item["cost"])
        for key, value in item["effects"].items():
            state.active_effects[key] = _clip(
                state.active_effects.get(key, 0.0) + float(value),
                0.0,
                0.8,
            )
        action_id = str(item["action_id"])
        definition = ACTION_CATALOG[action_id]
        messages.append(
            MessageObject(
                message_id=canonical_hash(
                    [scenario_id, seed, branch, tick, action_id, "governance-message"]
                ),
                parent_message_id=None,
                claim_id=f"{scenario_id}:primary-claim",
                source_role=str(definition["source_role"]),
                source_parent_index=None,
                source_group=str(definition["source_role"]),
                target_group="all_reviewed_episode_groups",
                topic=scenario_id,
                stance_support=0.64,
                stance_opposition=0.12,
                evidence_status=str(definition["evidence_status"]),
                emotion=0.12,
                actionability=0.82,
                distortion=0.0,
                created_tick=tick,
                ttl=4,
                action="correction",
                provenance="synthetic_governance_action",
                bridge=True,
                depth=0,
            )
        )
    remaining = (
        state.resource_opening
        - state.resource_consumed
        - state.resource_committed
    )
    if not math.isclose(
        state.resource_opening,
        state.resource_consumed + state.resource_committed + remaining,
        abs_tol=1e-8,
    ):
        raise GovernanceArenaV2Error("resource accounting is not conserved")
    return messages


def _message_candidate_exposure(
    *,
    state: BranchState,
    graph: GraphArrays,
    tick: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded top-three message exposures for every parent."""

    count = state.ensemble.parent_count
    candidate_indices = state.ledger.last_by_parent[graph.targets]
    active = candidate_indices >= 0
    if state.ledger.messages:
        # Build compact message columns once, then gather all 12,000 candidate
        # exposures with NumPy.  This avoids a particle/edge Python loop.
        messages = state.ledger.messages
        created_column = np.fromiter(
            (message.created_tick for message in messages),
            dtype=np.int16,
            count=len(messages),
        )
        ttl_column = np.fromiter(
            (message.ttl for message in messages),
            dtype=np.int8,
            count=len(messages),
        )
        emotion_column = np.fromiter(
            (message.emotion for message in messages),
            dtype=np.float32,
            count=len(messages),
        )
        actionability_column = np.fromiter(
            (message.actionability for message in messages),
            dtype=np.float32,
            count=len(messages),
        )
        distortion_column = np.fromiter(
            (message.distortion for message in messages),
            dtype=np.float32,
            count=len(messages),
        )
        verified_column = np.fromiter(
            (
                message.evidence_status in {"verified", "service_receipt"}
                for message in messages
            ),
            dtype=bool,
            count=len(messages),
        )
        depth_column = np.fromiter(
            (message.depth for message in messages),
            dtype=np.int16,
            count=len(messages),
        )
        safe_indices = np.maximum(candidate_indices, 0)
        created = created_column[safe_indices]
        ttl = ttl_column[safe_indices]
        active &= tick - created <= ttl
        active &= tick - created >= graph.delay
        distortion = distortion_column[safe_indices]
        is_verified = verified_column[safe_indices]
        salience = (
            0.35
            + 0.30 * emotion_column[safe_indices]
            + 0.20 * actionability_column[safe_indices]
            + 0.15 * graph.weight
        ).astype(np.float32)
        salience *= active
        rumor = np.where(
            active & ~is_verified, 0.35 + 0.65 * distortion, 0.0
        ).astype(np.float32)
        verified = np.where(
            active & is_verified, 1.0 - 0.35 * distortion, 0.0
        ).astype(np.float32)
        depth = np.where(
            active, depth_column[safe_indices], 0
        ).astype(np.int16)
    else:
        edge_shape = graph.targets.shape
        rumor = np.zeros(edge_shape, dtype=np.float32)
        verified = np.zeros(edge_shape, dtype=np.float32)
        salience = np.zeros(edge_shape, dtype=np.float32)
        depth = np.zeros(edge_shape, dtype=np.int16)
    top = np.argsort(salience, axis=1, kind="stable")[:, -3:]
    selected_active = np.take_along_axis(active, top, axis=1)
    selected_salience = np.take_along_axis(salience, top, axis=1)
    selected_rumor = np.take_along_axis(rumor, top, axis=1)
    selected_verified = np.take_along_axis(verified, top, axis=1)
    selected_depth = np.take_along_axis(depth, top, axis=1)
    denom = np.maximum(selected_active.sum(axis=1), 1)
    state.ledger.processed_per_parent = selected_active.sum(axis=1).astype(np.int8)
    if int(state.ledger.processed_per_parent.max(initial=0)) > 3:
        raise GovernanceArenaV2Error("finite attention exceeded three messages")
    best_local = top[:, -1]
    best_message = candidate_indices[np.arange(count), best_local]
    best_message[~selected_active[:, -1]] = -1
    return (
        selected_salience.sum(axis=1) / denom,
        selected_rumor.sum(axis=1) / denom,
        selected_verified.sum(axis=1) / denom,
        best_message,
        selected_depth.max(axis=1),
    )


def _inject_governance_messages(
    state: BranchState,
    messages: Sequence[MessageObject],
) -> None:
    """Governance broadcasts are explicit messages but not assigned to a parent."""

    for message in messages:
        state.ledger.add(message)


def _active_broadcast_exposure(
    state: BranchState, tick: int
) -> tuple[float, float]:
    rumor = 0.0
    verified = 0.0
    for message in reversed(state.ledger.messages):
        if message.source_parent_index is not None:
            continue
        if tick - message.created_tick > message.ttl:
            continue
        if message.evidence_status in {"verified", "service_receipt"}:
            verified += 0.24 + 0.24 * message.actionability
        else:
            rumor += 0.16 + 0.22 * message.distortion
    return min(rumor, 0.8), min(verified, 0.8)


def _optional_parent_feature(
    features: Mapping[str, np.ndarray],
    name: str,
    *,
    parent_count: int,
    default: float,
) -> np.ndarray:
    """Return one bounded parent-level feature column.

    The Arena v2 release predates ForumTwin's scenario-stake layer.  Optional
    columns keep the old kernel byte-for-byte compatible when they are absent,
    while allowing ForumTwin to express *why* two synthetic residents react
    differently to the same event.  These columns are private model state;
    they are never additional people or LLM observations.
    """

    raw = features.get(name)
    if raw is None:
        return np.full((parent_count, 1), float(default), dtype=np.float32)
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (parent_count,) or not np.isfinite(values).all():
        raise GovernanceArenaV2Error(
            f"optional parent feature {name!r} has an invalid shape or value"
        )
    return np.clip(values, 0.0, 1.0)[:, None]


def advance_particle_tick(
    *,
    population: Mapping[str, Any],
    graph: GraphArrays,
    state: BranchState,
    features: Mapping[str, np.ndarray],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    emit_messages: bool = True,
    stochastic_diffusion: bool = True,
) -> dict[str, Any]:
    """Advance all particles with message-aware, interpretable state equations."""

    if branch not in BRANCH_IDS:
        raise GovernanceArenaV2Error("unknown policy branch")
    governance_messages = _activate_due_actions(
        state,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
    )
    _inject_governance_messages(state, governance_messages)
    (
        message_salience,
        rumor_exposure,
        verified_exposure,
        best_message,
        parent_depth,
    ) = _message_candidate_exposure(state=state, graph=graph, tick=tick)
    broadcast_rumor, broadcast_verified = _active_broadcast_exposure(state, tick)
    rumor_exposure = np.clip(rumor_exposure + broadcast_rumor, 0.0, 1.0)
    verified_exposure = np.clip(
        verified_exposure + broadcast_verified, 0.0, 1.0
    )

    ensemble = state.ensemble
    previous = ensemble.values
    parent = ensemble.parent_means()
    neighbor = np.mean(parent[graph.targets], axis=1)
    p = previous.shape[1]
    relevance = features["relevance"][:, None]
    risk = features["risk"][:, None]
    # Optional ForumTwin stake features are branch-neutral.  They describe a
    # resident's exposure to the issue and capacity to use a response; they do
    # not contain a policy label or a desired outcome.  Defaults reproduce the
    # original Arena v2 equations exactly.
    stake_exposure = _optional_parent_feature(
        features,
        "stake_exposure",
        parent_count=ensemble.parent_count,
        default=0.0,
    )
    service_pressure = _optional_parent_feature(
        features,
        "service_pressure",
        parent_count=ensemble.parent_count,
        default=0.0,
    )
    appeal_friction = _optional_parent_feature(
        features,
        "appeal_friction",
        parent_count=ensemble.parent_count,
        default=0.0,
    )
    alternative_buffer = _optional_parent_feature(
        features,
        "alternative_buffer",
        parent_count=ensemble.parent_count,
        default=0.0,
    )
    voice_capacity = _optional_parent_feature(
        features,
        "voice_capacity",
        parent_count=ensemble.parent_count,
        default=1.0,
    )
    verification_reach = _optional_parent_feature(
        features,
        "verification_reach",
        parent_count=ensemble.parent_count,
        default=1.0,
    )
    service_reach = _optional_parent_feature(
        features,
        "service_reach",
        parent_count=ensemble.parent_count,
        default=1.0,
    )
    bridge_reach = _optional_parent_feature(
        features,
        "bridge_reach",
        parent_count=ensemble.parent_count,
        default=1.0,
    )
    intensity = _phase_intensity(tick)
    salience = message_salience[:, None]
    rumor = rumor_exposure[:, None]
    verified = verified_exposure[:, None]
    neighbor_attention = neighbor[:, ATTENTION, None]
    neighbor_concern = neighbor[:, CONCERN, None]
    neighbor_trust = neighbor[:, TRUST, None]
    effects = state.active_effects
    verification = float(effects["verification"])
    reassurance = float(effects["reassurance"])
    service = float(effects["service"])
    voice_effect = float(effects["voice"])
    bridge_effect = float(effects["bridge"])
    prebunk = float(effects["prebunk"])

    common_rng = np.random.default_rng(
        _seed_from(seed, scenario_id, tick, "common-exogenous-noise")
    )
    exogenous = common_rng.normal(
        0.0, 0.008, size=(ensemble.parent_count, p, len(STATE_FIELDS))
    ).astype(np.float32) if stochastic_diffusion else np.zeros(
        (ensemble.parent_count, p, len(STATE_FIELDS)), dtype=np.float32
    )
    if branch == "natural" or not stochastic_diffusion:
        endogenous = 0.0
    else:
        branch_rng = np.random.default_rng(
            _seed_from(seed, scenario_id, branch, tick, "endogenous-noise")
        )
        endogenous = branch_rng.normal(
            0.0, 0.003, size=(ensemble.parent_count, p, len(STATE_FIELDS))
        ).astype(np.float32)

    updated = np.empty_like(previous)
    updated[:, :, ATTENTION] = (
        0.66 * previous[:, :, ATTENTION]
        + 0.15 * neighbor_attention
        + 0.23 * intensity * relevance
        + 0.14 * intensity * stake_exposure
        + 0.14 * salience
        - 0.08 * reassurance
    )
    updated[:, :, CONCERN] = (
        0.69 * previous[:, :, CONCERN]
        + 0.13 * neighbor_concern
        + 0.22 * intensity * relevance
        + 0.18 * intensity * stake_exposure
        + 0.04 * appeal_friction
        + 0.17 * rumor
        - 0.06 * alternative_buffer
        - 0.16 * verified * verification_reach
        - 0.14 * reassurance * verification_reach
        - 0.08 * prebunk * verification_reach
    )
    updated[:, :, TRUST] = (
        0.75 * previous[:, :, TRUST]
        + 0.11 * neighbor_trust
        - 0.10 * rumor
        + 0.14 * verified * verification_reach
        + 0.11 * verification * verification_reach
        + 0.06 * service * service_reach
    )
    updated[:, :, SATISFACTION] = (
        0.67 * previous[:, :, SATISFACTION]
        + 0.16 * updated[:, :, TRUST]
        - 0.17 * updated[:, :, CONCERN]
        + 0.14 * service * service_reach
        + 0.09 * voice_effect
    )
    updated[:, :, SUPPORT] = _sigmoid_array(
        -0.45
        + 1.30 * updated[:, :, TRUST]
        + 0.72 * updated[:, :, SATISFACTION]
        + 0.26 * verified
        + 0.10 * voice_effect
        - 0.72 * updated[:, :, CONCERN]
    )
    updated[:, :, OPPOSITION] = _sigmoid_array(
        -0.82
        + 1.42 * updated[:, :, CONCERN]
        + 0.35 * neighbor_concern
        + 0.30 * rumor
        - 0.90 * updated[:, :, TRUST]
        - 0.12 * voice_effect
    )
    updated[:, :, ACTIVITY] = _sigmoid_array(
        -2.18
        + 2.05 * updated[:, :, ATTENTION]
        + 0.88 * updated[:, :, CONCERN]
        + 0.35 * neighbor_attention
        + 0.22 * salience
    )
    updated[:, :, VOICE] = np.clip(
        0.70 * previous[:, :, VOICE]
        + 0.18 * updated[:, :, ACTIVITY]
        + 0.15 * voice_effect * voice_capacity
        + 0.08 * bridge_effect * bridge_reach
        - 0.10 * updated[:, :, CONCERN],
        0.0,
        1.0,
    )
    updated[:, :, SERVICE_STRAIN] = np.clip(
        0.76 * previous[:, :, SERVICE_STRAIN]
        + 0.14 * risk
        + 0.16 * intensity * service_pressure
        + 0.16 * updated[:, :, CONCERN]
        - 0.20 * service * service_reach,
        0.0,
        1.0,
    )
    updated[:, :, RUMOR_BELIEF] = np.clip(
        0.72 * previous[:, :, RUMOR_BELIEF]
        + 0.24 * rumor
        + 0.10 * intensity * relevance
        - 0.27
        * verified
        * verification_reach
        * previous[:, :, EVIDENCE_ACCEPTANCE]
        - 0.13 * prebunk * verification_reach,
        0.0,
        1.0,
    )
    updated[:, :, VERIFIED_BELIEF] = np.clip(
        0.72 * previous[:, :, VERIFIED_BELIEF]
        + 0.26
        * verified
        * verification_reach
        * previous[:, :, EVIDENCE_ACCEPTANCE]
        + 0.10 * verification * verification_reach
        - 0.08 * rumor,
        0.0,
        1.0,
    )
    updated[:, :, EVIDENCE_ACCEPTANCE] = np.clip(
        0.86 * previous[:, :, EVIDENCE_ACCEPTANCE]
        + 0.09 * updated[:, :, TRUST]
        + 0.08 * verified * verification_reach
        - 0.07 * rumor,
        0.0,
        1.0,
    )
    updated += exogenous
    if isinstance(endogenous, np.ndarray):
        updated += endogenous
    ensemble.values = np.clip(updated, 0.0, 1.0).astype(np.float32)
    ensemble.assert_conserved()

    new_messages = (
        _emit_surrogate_messages(
            population=population,
            graph=graph,
            state=state,
            features=features,
            scenario_id=scenario_id,
            branch=branch,
            seed=seed,
            tick=tick,
            best_message=best_message,
            parent_depth=parent_depth,
        )
        if emit_messages
        else []
    )
    return _diffusion_summary(
        population=population,
        graph=graph,
        state=state,
        tick=tick,
        messages=new_messages,
    )


def _emit_surrogate_messages(
    *,
    population: Mapping[str, Any],
    graph: GraphArrays,
    state: BranchState,
    features: Mapping[str, np.ndarray],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    best_message: np.ndarray,
    parent_depth: np.ndarray,
) -> list[MessageObject]:
    parent = state.ensemble.parent_means()
    rng = np.random.default_rng(
        _seed_from(seed, scenario_id, branch, tick, "surrogate-emission")
    )
    probability = np.clip(
        parent[:, ACTIVITY]
        * (0.20 + 0.62 * features["relevance"])
        * (0.25 if TIMELINE[tick] == "baseline" else 1.0),
        0.0,
        0.72,
    )
    emitted = np.flatnonzero(rng.random(len(parent)) < probability)
    messages = []
    for index in emitted.tolist():
        concern = float(parent[index, CONCERN])
        rumor = float(parent[index, RUMOR_BELIEF])
        verified = float(parent[index, VERIFIED_BELIEF])
        trust = float(parent[index, TRUST])
        if concern > 0.70 and features["risk"][index] > 0.50:
            action = "report_risk"
        elif float(parent[index, SERVICE_STRAIN]) > 0.65:
            action = "seek_help"
        elif best_message[index] >= 0 and rumor > verified + 0.08:
            action = "repost"
        else:
            action = "comment"
        parent_message = None
        source_group = str(features["archetypes"][index])
        target_group = source_group
        depth = 0
        is_bridge = False
        if best_message[index] >= 0:
            prior = state.ledger.messages[int(best_message[index])]
            parent_message = prior.message_id
            target_group = source_group
            source_group = prior.source_group
            depth = int(prior.depth) + 1
            is_bridge = prior.target_group != target_group
        if verified > rumor + 0.08:
            evidence_status = "verified"
            distortion = max(0.0, 0.08 * (1.0 - trust))
        elif rumor > verified + 0.05:
            evidence_status = "unverified"
            distortion = _clip(0.20 + 0.55 * rumor - 0.25 * trust, 0.0, 0.72)
        else:
            evidence_status = "uncertain"
            distortion = _clip(0.12 + 0.25 * concern, 0.0, 0.55)
        message = MessageObject(
            message_id=canonical_hash(
                [scenario_id, seed, branch, tick, index, "surrogate-message"]
            ),
            parent_message_id=parent_message,
            claim_id=f"{scenario_id}:primary-claim",
            source_role="episode_parent",
            source_parent_index=index,
            source_group=source_group,
            target_group=target_group,
            topic=str(features["topic_codes"][index]),
            stance_support=float(parent[index, SUPPORT]),
            stance_opposition=float(parent[index, OPPOSITION]),
            evidence_status=evidence_status,
            emotion=concern,
            actionability=float(parent[index, SERVICE_STRAIN]),
            distortion=distortion,
            created_tick=tick,
            ttl=3 + int(concern >= 0.62),
            action=action,
            provenance="deterministic_surrogate",
            bridge=is_bridge,
            depth=min(depth, 24),
        )
        state.ledger.add(message)
        messages.append(message)
    if len({message.source_parent_index for message in messages}) != len(messages):
        raise GovernanceArenaV2Error("a parent emitted more than once in a tick")
    return messages


def _diffusion_summary(
    *,
    population: Mapping[str, Any],
    graph: GraphArrays,
    state: BranchState,
    tick: int,
    messages: Sequence[MessageObject],
) -> dict[str, Any]:
    reads = int(state.ledger.processed_per_parent.sum())
    reposts = sum(message.action == "repost" for message in messages)
    corrections = sum(
        message.evidence_status in {"verified", "service_receipt"}
        for message in messages
    )
    bridge_messages = [message for message in messages if message.bridge]
    exposure = state.ledger.processed_per_parent.astype(np.float64)
    mean_exposure = float(exposure.mean()) if len(exposure) else 0.0
    if mean_exposure > 0.0:
        ordered_exposure = np.sort(exposure)
        ranks = np.arange(1, len(exposure) + 1, dtype=np.float64)
        inequality = float(
            np.sum((2.0 * ranks - len(exposure) - 1.0) * ordered_exposure)
            / max(len(exposure) * float(ordered_exposure.sum()), 1e-12)
        )
    else:
        inequality = 0.0
    correction_delays = []
    message_by_id = {
        message.message_id: message
        for message in state.ledger.messages
        if message.created_tick <= tick
    }
    for message in messages:
        if (
            message.evidence_status not in {"verified", "service_receipt"}
            or not message.parent_message_id
        ):
            continue
        parent = message_by_id.get(message.parent_message_id)
        if (
            parent is not None
            and parent.evidence_status not in {"verified", "service_receipt"}
        ):
            correction_delays.append(message.created_tick - parent.created_tick)
    cascade = defaultdict(
        lambda: {"rumor": 0, "verified": 0, "bridge": 0}
    )
    for message in messages:
        key = (message.source_group, message.target_group)
        if message.evidence_status in {"verified", "service_receipt"}:
            cascade[key]["verified"] += 1
        else:
            cascade[key]["rumor"] += 1
        cascade[key]["bridge"] += int(message.bridge)
    cascade_groups = [
        {
            "source_group": source,
            "target_group": target,
            "rumor": values["rumor"],
            "verified": values["verified"],
            "bridge": values["bridge"],
        }
        for (source, target), values in sorted(
            cascade.items(),
            key=lambda item: (
                -(item[1]["rumor"] + item[1]["verified"]),
                item[0],
            ),
        )[:12]
    ]
    return {
        "emissions": len(messages),
        "reads": reads,
        "reposts": reposts,
        "corrections": corrections,
        "cascade_depth": max((message.depth for message in messages), default=0),
        "bridge_share": round(len(bridge_messages) / max(len(messages), 1), 6),
        "correction_delay": round(
            statistics.fmean(correction_delays) if correction_delays else 0.0,
            6,
        ),
        "exposure_inequality": round(min(inequality, 1.0), 6),
        "cascade_groups": cascade_groups,
    }


def _sampling_strata(
    *,
    features: Mapping[str, np.ndarray],
    parent_state: np.ndarray,
) -> tuple[np.ndarray, float]:
    median_activity = float(np.median(parent_state[:, ACTIVITY]))
    labels = []
    for relevance, activity in zip(
        features["relevance"], parent_state[:, ACTIVITY]
    ):
        if relevance < 0.34:
            related = "irrelevant"
        elif relevance < 0.64:
            related = "uncertain"
        else:
            related = "relevant"
        labels.append(
            f"{related}__{'high' if activity >= median_activity else 'low'}"
        )
    return np.asarray(labels, dtype=object), median_activity


def draw_probe_sample_v2(
    *,
    population: Mapping[str, Any],
    ensemble: ParticleEnsemble,
    features: Mapping[str, np.ndarray],
    rotation_counts: np.ndarray,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    budget: int = PROBE_BUDGET,
) -> dict[str, Any]:
    """Cover every non-empty frozen stratum, then draw deterministic pivotal PPS."""

    count = ensemble.parent_count
    if len(rotation_counts) != count:
        raise GovernanceArenaV2Error("rotation counts do not match parents")
    if not 1 <= budget <= count:
        raise GovernanceArenaV2Error("probe budget is outside the parent population")
    parent = ensemble.parent_means()
    variances = ensemble.parent_variances()
    labels, median_activity = _sampling_strata(
        features=features, parent_state=parent
    )
    strata = sorted(set(labels.tolist()))
    members = {
        label: np.flatnonzero(labels == label).tolist() for label in strata
    }
    allocation = _integer_allocation(
        [len(members[label]) for label in strata], budget
    )
    inclusion = np.zeros(count, dtype=np.float64)
    selected: list[int] = []
    allocation_by_stratum = {}
    for position, label in enumerate(strata):
        local_members = members[label]
        local_budget = allocation[position]
        allocation_by_stratum[label] = local_budget
        scores = []
        for index in local_members:
            uncertainty = (
                float(variances[index, SUPPORT])
                + float(variances[index, OPPOSITION])
                + 0.50
                * (
                    1.0
                    - abs(
                        float(parent[index, SUPPORT])
                        - float(parent[index, OPPOSITION])
                    )
                )
            )
            rotation_penalty = 1.0 / (
                1.0 + 0.45 * float(rotation_counts[index])
            )
            scores.append(
                max(float(ensemble.parent_weights[index]), 1e-12)
                * max(float(parent[index, ACTIVITY]), 0.03)
                * (0.55 + uncertainty + 0.22 * float(features["relevance"][index]))
                * rotation_penalty
            )
        local_pi = fixed_size_inclusion_probabilities(scores, local_budget)
        local_selection = pivotal_sample(
            local_pi,
            random.Random(
                _seed_from(
                    seed, scenario_id, branch, tick, label, "v2-pivotal"
                )
            ),
        )
        for local_index, parent_index in enumerate(local_members):
            inclusion[parent_index] = float(local_pi[local_index])
        selected.extend(local_members[local_index] for local_index in local_selection)
    selected = sorted(selected)
    if len(selected) != budget:
        raise GovernanceArenaV2Error("probe sample did not meet fixed budget")
    if any(inclusion[index] <= 0.0 for index in selected):
        raise GovernanceArenaV2Error("selected probes require positive inclusion")
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "budget": budget,
        "selected": tuple(selected),
        "inclusion_probability": inclusion,
        "strata": labels,
        "median_activity": median_activity,
        "allocation_by_stratum": allocation_by_stratum,
        "selected_identity_sha256": canonical_hash(
            [population["agents"][index]["micro_agent_id"] for index in selected]
        ),
    }


def _model_probabilities(
    parent: np.ndarray,
    features: Mapping[str, np.ndarray],
    *,
    scenario_id: str,
    tick: int,
) -> dict[str, dict[str, np.ndarray]]:
    relevance = features["relevance"]
    public = np.clip(
        parent[:, ACTIVITY] * (0.25 + 0.75 * relevance), 0.0, 0.92
    )
    if TIMELINE[tick] == "baseline":
        public *= 0.25
    actions = {
        "ignore": (1.0 - public) * 0.34,
        "read": (1.0 - public) * 0.66,
        "comment": public * 0.31,
        "post": public * 0.30,
        "seek_help": public * (0.20 + 0.10 * parent[:, SERVICE_STRAIN]),
        "report_risk": public * (0.19 + 0.10 * features["risk"]),
    }
    action_total = np.maximum(sum(actions.values()), 1e-12)
    actions = {key: value / action_total for key, value in actions.items()}
    stances = {
        "supportive": 0.06 + 0.72 * parent[:, SUPPORT],
        "neutral": 0.18
        + 0.42 * (1.0 - np.maximum(parent[:, SUPPORT], parent[:, OPPOSITION])),
        "concerned": 0.08 + 0.56 * parent[:, CONCERN],
        "critical": 0.05 + 0.70 * parent[:, OPPOSITION],
    }
    stance_total = np.maximum(sum(stances.values()), 1e-12)
    stances = {key: value / stance_total for key, value in stances.items()}
    event_basis = (
        np.full(len(parent), 0.05, dtype=np.float32)
        if TIMELINE[tick] == "baseline"
        else 0.24 + 0.68 * relevance
    )
    event = np.clip(
        event_basis,
        0.02,
        0.96,
    )
    topics = {
        scenario_id: event,
        NON_EVENT_TOPIC_ID: 1.0 - event,
    }
    return {"action": actions, "stance": stances, "topic_id": topics}


def _draw_from_array_distribution(
    distribution: Mapping[str, np.ndarray],
    *,
    index: int,
    uniform: float,
) -> str:
    cumulative = 0.0
    last = next(iter(distribution))
    for category, values in distribution.items():
        last = category
        cumulative += float(values[index])
        if uniform <= cumulative + 1e-12:
            return category
    return last


def render_governance_probe_turn_v2(
    *,
    semantic_profile: Mapping[str, Any],
    posterior_state: Mapping[str, float],
    visible_messages: Sequence[MessageObject | Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
) -> RenderedAgentTurn:
    """Render one selected probe without I/O, identity, future, or hidden truth."""

    if not 0 <= tick < len(TIMELINE):
        raise GovernanceArenaV2Error("probe render tick is invalid")
    visible = []
    for raw in visible_messages:
        if isinstance(raw, MessageObject):
            created_tick = raw.created_tick
            ttl = raw.ttl
            item = {
                "claim_id": raw.claim_id,
                "source_role": raw.source_role,
                "topic": raw.topic,
                "evidence_status": raw.evidence_status,
                "emotion": round(raw.emotion, 4),
                "actionability": round(raw.actionability, 4),
                "distortion": round(raw.distortion, 4),
                "created_tick": created_tick,
                "ttl": ttl,
            }
        elif isinstance(raw, Mapping):
            created_tick = int(raw.get("created_tick", -1))
            ttl = int(raw.get("ttl", 0))
            item = {
                "claim_id": str(raw.get("claim_id") or ""),
                "source_role": str(raw.get("source_role") or "synthetic"),
                "topic": str(raw.get("topic") or NON_EVENT_TOPIC_ID),
                "evidence_status": str(
                    raw.get("evidence_status") or "uncertain"
                ),
                "emotion": round(_clip(raw.get("emotion", 0.0)), 4),
                "actionability": round(
                    _clip(raw.get("actionability", 0.0)), 4
                ),
                "distortion": round(_clip(raw.get("distortion", 0.0)), 4),
                "created_tick": created_tick,
                "ttl": ttl,
            }
        else:
            raise GovernanceArenaV2Error("visible probe message is invalid")
        if created_tick > tick or tick - created_tick > ttl:
            continue
        visible.append(item)
    visible = sorted(
        visible,
        key=lambda item: (
            -int(item["created_tick"]),
            item["claim_id"],
            item["source_role"],
        ),
    )[:3]
    posterior = {
        key: round(_clip(posterior_state.get(key, 0.0)), 6)
        for key in (
            "attention",
            "concern",
            "trust",
            "satisfaction",
            "support",
            "opposition",
            "activity",
            "voice",
            "service_strain",
            "rumor_belief",
            "verified_belief",
            "evidence_acceptance",
        )
    }
    prompt = {
        "schema_version": "campus-pulse-governance-probe-turn-v2",
        "simulation_target": (
            "one reviewed forum behaviour episode; not an identified student"
        ),
        "temporal_contract": {
            "current_tick": tick,
            "phase": TIMELINE[tick],
            "future_information_forbidden": True,
        },
        "scenario_id": scenario_id,
        "branch": branch,
        "reviewed_semantic_constraints": {
            key: semantic_profile.get(key)
            for key in (
                "language_form",
                "semantic_determinacy",
                "speech_act",
                "stance",
                "risk_level",
                "confidence",
                "context_needed",
            )
        },
        "posterior_state": posterior,
        "visible_synthetic_messages": visible,
        "output_contract": {
            "action": {"allowed": list(ACTION_CATEGORIES)},
            "topic_id": {
                "allowed": [scenario_id, NON_EVENT_TOPIC_ID]
            },
            "stance": {"allowed": list(STANCE_CATEGORIES)},
            "urgency": "finite number from 0 to 1",
            "trust_delta": "finite number from -0.25 to 0.25",
            "satisfied": "boolean",
            "rationale_tag": "controlled existing rationale tag",
            "message": (
                "empty for ignore/read; otherwise reviewed synthetic text "
                "of at most 80 characters"
            ),
        },
    }
    agent_prompt = json.dumps(
        prompt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system_prompt = (
        "You are a bounded semantic sensor for one synthetic CampusPulse "
        "episode. Use only the supplied posterior and temporally visible "
        "synthetic messages. Return exactly one JSON object."
    )
    topic_ids = tuple(sorted((scenario_id, NON_EVENT_TOPIC_ID)))
    activation = {
        "event_visible": TIMELINE[tick] != "baseline",
        "public_action_allowed": True,
        "temporal_cutoff_tick": tick,
        "provider_call_authorized": False,
    }
    fingerprint_body = {
        "system_prompt": system_prompt,
        "agent_prompt": agent_prompt,
        "topic_ids": list(topic_ids),
        "simulation_mode": "active_episode_agent",
        "prior_messages": [],
    }
    return RenderedAgentTurn(
        system_prompt=system_prompt,
        agent_prompt=agent_prompt,
        topic_ids=topic_ids,
        simulation_mode="active_episode_agent",
        prior_messages=(),
        activation=activation,
        prompt_sha256=canonical_hash(fingerprint_body),
    )


def _visible_messages_for_probe(
    *,
    state: BranchState,
    archetype: str,
    tick: int,
) -> list[MessageObject]:
    candidates = [
        message
        for message in state.ledger.messages
        if message.created_tick <= tick
        and tick - message.created_tick <= message.ttl
        and (
            message.target_group in {
                archetype,
                "all_reviewed_episode_groups",
            }
            or message.source_group == archetype
        )
    ]
    return sorted(
        candidates,
        key=lambda message: (-message.created_tick, message.message_id),
    )[:3]


def _fixture_probe_v2(
    *,
    population: Mapping[str, Any],
    parent: np.ndarray,
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
    index: int,
    rendered_turn: RenderedAgentTurn,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Create one selected-only strict semantic fixture decision."""

    agent_id = str(population["agents"][index]["micro_agent_id"])
    action = _draw_from_array_distribution(
        probabilities["action"],
        index=index,
        uniform=_hash_uniform(
            seed, scenario_id, branch, tick, agent_id, "v2-action"
        ),
    )
    stance = _draw_from_array_distribution(
        probabilities["stance"],
        index=index,
        uniform=_hash_uniform(
            seed, scenario_id, branch, tick, agent_id, "v2-stance"
        ),
    )
    topic = _draw_from_array_distribution(
        probabilities["topic_id"],
        index=index,
        uniform=_hash_uniform(
            seed, scenario_id, branch, tick, agent_id, "v2-topic"
        ),
    )
    public = action in PUBLIC_ACTIONS
    raw = {
        "action": action,
        "topic_id": topic,
        "stance": stance,
        "urgency": float(parent[index, CONCERN]),
        "trust_delta": float(
            np.clip(
                0.05 * parent[index, VERIFIED_BELIEF]
                - 0.04 * parent[index, RUMOR_BELIEF],
                -0.20,
                0.20,
            )
        ),
        "satisfied": bool(parent[index, SATISFACTION] >= 0.5),
        "rationale_tag": "general",
        "message": (
            "审阅合成语义探针表达，不代表真实个体。"
            if public
            else ""
        ),
    }
    fixture_content = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    format_repairs = 0
    try:
        decision = validate_rendered_turn(fixture_content, rendered_turn)
    except (TypeError, ValueError):
        # The same controlled repair contract used by Provider-capable turns
        # is available without invoking a client.  Deterministic fixtures are
        # expected to validate on the first pass.
        repaired = render_format_repair_turn(
            rendered_turn,
            controlled_reason_code="fixture_json_contract",
            repair_index=1,
        )
        decision = validate_rendered_turn(fixture_content, repaired)
        format_repairs = 1
    semantic = {
        "action": decision.action,
        "stance": decision.stance,
        "topic_id": decision.topic_id,
        "urgency": round(float(decision.urgency), 6),
        "trust": round(
            _clip(float(parent[index, TRUST]) + float(decision.trust_delta)),
            6,
        ),
        "satisfaction": float(decision.satisfied),
        "activity": float(public),
        "message": decision.message,
        "provider_calls": 0,
        "prompt_fingerprint": rendered_turn.prompt_sha256,
        "fixture_response_fingerprint": hashlib.sha256(
            fixture_content.encode("utf-8")
        ).hexdigest(),
        "format_repairs": format_repairs,
        "provenance": "frozen_zero_provider_fixture",
    }
    return decision, semantic


def _greg_numeric(
    *,
    population_weights: np.ndarray,
    prediction: np.ndarray,
    observations: Mapping[int, float],
    sample: Mapping[str, Any],
) -> float:
    estimate = float(np.sum(population_weights * prediction))
    pi = sample["inclusion_probability"]
    estimate += sum(
        float(population_weights[index])
        * (float(observations[index]) - float(prediction[index]))
        / float(pi[index])
        for index in sample["selected"]
    )
    return _clip(estimate)


def _greg_categorical(
    *,
    population_weights: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    observations: Mapping[int, str],
    sample: Mapping[str, Any],
) -> dict[str, float]:
    estimates = {}
    pi = sample["inclusion_probability"]
    for category, values in prediction.items():
        estimate = float(np.sum(population_weights * values))
        estimate += sum(
            float(population_weights[index])
            * (
                float(observations[index] == category)
                - float(values[index])
            )
            / float(pi[index])
            for index in sample["selected"]
        )
        estimates[category] = max(0.0, estimate)
    return _normalize(estimates)


def _kish_ess(sample: Mapping[str, Any], parent_weights: np.ndarray) -> float:
    expanded = np.asarray(
        [
            float(parent_weights[index])
            / float(sample["inclusion_probability"][index])
            for index in sample["selected"]
        ],
        dtype=np.float64,
    )
    return float(
        expanded.sum() ** 2 / max(float(np.sum(expanded * expanded)), 1e-15)
    )


def _partial_pooling_v2(
    *,
    sample: Mapping[str, Any],
    parent_weights: np.ndarray,
    semantics: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Beta-Dirichlet shrinkage with design-weight and Kish-ESS scaling."""

    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index in sample["selected"]:
        by_stratum[str(sample["strata"][index])].append(index)
    all_indices = list(sample["selected"])
    all_expanded = np.asarray(
        [
            parent_weights[index] / sample["inclusion_probability"][index]
            for index in all_indices
        ],
        dtype=np.float64,
    )
    global_speaker = sum(
        weight * float(semantics[index]["activity"])
        for weight, index in zip(all_expanded, all_indices)
    ) / max(float(all_expanded.sum()), 1e-12)
    global_stance = _normalize(
        {
            category: 0.5
            + sum(
                weight
                * float(
                    semantics[index]["activity"]
                    and semantics[index]["stance"] == category
                )
                for weight, index in zip(all_expanded, all_indices)
            )
            for category in STANCE_CATEGORIES
        }
    )
    output = {}
    for label, indices in sorted(by_stratum.items()):
        expanded = np.asarray(
            [
                parent_weights[index]
                / sample["inclusion_probability"][index]
                for index in indices
            ],
            dtype=np.float64,
        )
        normalized = expanded / max(float(expanded.sum()), 1e-12)
        stratum_ess = float(
            1.0 / max(float(np.sum(normalized * normalized)), 1e-12)
        )
        speaking_share = float(
            sum(
                weight * float(semantics[index]["activity"])
                for weight, index in zip(normalized, indices)
            )
        )
        beta_strength = 2.0
        speaking = (
            beta_strength * global_speaker + stratum_ess * speaking_share
        ) / (beta_strength + stratum_ess)
        public = [
            (weight, index)
            for weight, index in zip(normalized, indices)
            if semantics[index]["activity"]
        ]
        public_mass = sum(weight for weight, _ in public)
        stance_counts = {
            category: sum(
                weight * float(semantics[index]["stance"] == category)
                for weight, index in public
            )
            / max(public_mass, 1e-12)
            for category in STANCE_CATEGORIES
        }
        dirichlet_strength = 2.0
        stance = _normalize(
            {
                category: (
                    dirichlet_strength * global_stance[category]
                    + stratum_ess * public_mass * stance_counts[category]
                )
                / max(
                    dirichlet_strength + stratum_ess * public_mass, 1e-12
                )
                for category in STANCE_CATEGORIES
            }
        )
        output[label] = {
            "normalized_design_weight": round(float(expanded.sum()), 8),
            "kish_ess": round(stratum_ess, 6),
            "posterior_speaking_rate": round(_clip(speaking), 6),
            "conditional_stance": {
                key: round(value, 6) for key, value in stance.items()
            },
        }
    return {
        "model": "design_weighted_beta_dirichlet_partial_pooling_v2",
        "strata": output,
    }


def _replace_surrogate_emissions_with_probes(
    *,
    population: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    state: BranchState,
    semantics: Mapping[int, Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
) -> None:
    for index, semantic in semantics.items():
        previous_position = int(state.ledger.last_by_parent[index])
        previous = (
            state.ledger.messages[previous_position]
            if previous_position >= 0
            and state.ledger.messages[previous_position].created_tick == tick
            else None
        )
        is_public = semantic["action"] in PUBLIC_ACTIONS
        if not is_public and previous is not None:
            state.ledger.messages[previous_position] = MessageObject(
                **{
                    **previous.__dict__,
                    "action": "suppressed",
                    "provenance": "fixture_replaced_surrogate",
                }
            )
            prior_position = next(
                (
                    position
                    for position in range(previous_position - 1, -1, -1)
                    if state.ledger.messages[position].source_parent_index
                    == index
                    and state.ledger.messages[position].action != "suppressed"
                ),
                -1,
            )
            state.ledger.last_by_parent[index] = prior_position
            state.ledger.last_tick_by_parent[index] = (
                state.ledger.messages[prior_position].created_tick
                if prior_position >= 0
                else -100
            )
            continue
        if not is_public:
            continue
        parent_state = state.ensemble.parent_means()[index]
        replacement = MessageObject(
            message_id=canonical_hash(
                [scenario_id, seed, branch, tick, index, "fixture-message"]
            ),
            parent_message_id=(
                previous.parent_message_id if previous is not None else None
            ),
            claim_id=f"{scenario_id}:primary-claim",
            source_role="episode_parent",
            source_parent_index=index,
            source_group=str(features["archetypes"][index]),
            target_group=str(features["archetypes"][index]),
            topic=str(semantic["topic_id"]),
            stance_support=float(parent_state[SUPPORT]),
            stance_opposition=float(parent_state[OPPOSITION]),
            evidence_status=(
                "verified"
                if parent_state[VERIFIED_BELIEF]
                >= parent_state[RUMOR_BELIEF]
                else "unverified"
            ),
            emotion=float(semantic["urgency"]),
            actionability=float(parent_state[SERVICE_STRAIN]),
            distortion=float(
                max(
                    0.0,
                    parent_state[RUMOR_BELIEF]
                    - parent_state[VERIFIED_BELIEF],
                )
            ),
            created_tick=tick,
            ttl=3,
            action=str(semantic["action"]),
            provenance="frozen_zero_provider_fixture",
            bridge=bool(previous.bridge) if previous is not None else False,
            depth=int(previous.depth) if previous is not None else 0,
        )
        if previous is None:
            state.ledger.add(replacement)
        else:
            state.ledger.messages[previous_position] = replacement


def observe_population_v2(
    *,
    population: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    state: BranchState,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    budget: int = PROBE_BUDGET,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Observe only selected parents; unselected fixture decisions never exist."""

    parent = state.ensemble.parent_means()
    sample = draw_probe_sample_v2(
        population=population,
        ensemble=state.ensemble,
        features=features,
        rotation_counts=state.rotation_counts,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        seed=seed,
        budget=budget,
    )
    probabilities = _model_probabilities(
        parent, features, scenario_id=scenario_id, tick=tick
    )
    decisions = {}
    semantics = {}
    for index in sample["selected"]:
        rendered_turn = render_governance_probe_turn_v2(
            semantic_profile=population["agents"][index]["semantic"],
            posterior_state={
                field: float(parent[index, position])
                for position, field in enumerate(STATE_FIELDS)
            },
            visible_messages=_visible_messages_for_probe(
                state=state,
                archetype=str(features["archetypes"][index]),
                tick=tick,
            ),
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
        )
        decision, semantic = _fixture_probe_v2(
            population=population,
            parent=parent,
            probabilities=probabilities,
            index=index,
            rendered_turn=rendered_turn,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            seed=seed,
        )
        decisions[index] = decision
        semantics[index] = semantic
        state.rotation_counts[index] += 1
    if len(decisions) != budget:
        raise GovernanceArenaV2Error("fixture decisions must exist for selected probes only")
    weights = state.ensemble.parent_weights
    corrected_action = _greg_categorical(
        population_weights=weights,
        prediction=probabilities["action"],
        observations={index: item["action"] for index, item in semantics.items()},
        sample=sample,
    )
    corrected_stance = _greg_categorical(
        population_weights=weights,
        prediction=probabilities["stance"],
        observations={index: item["stance"] for index, item in semantics.items()},
        sample=sample,
    )
    corrected_topic = _greg_categorical(
        population_weights=weights,
        prediction=probabilities["topic_id"],
        observations={index: item["topic_id"] for index, item in semantics.items()},
        sample=sample,
    )
    numeric_observations = {
        "concern": (
            parent[:, CONCERN],
            {index: item["urgency"] for index, item in semantics.items()},
        ),
        "trust": (
            parent[:, TRUST],
            {index: item["trust"] for index, item in semantics.items()},
        ),
        "satisfaction": (
            parent[:, SATISFACTION],
            {index: item["satisfaction"] for index, item in semantics.items()},
        ),
        "activity": (
            parent[:, ACTIVITY],
            {index: item["activity"] for index, item in semantics.items()},
        ),
    }
    corrected_numeric = {
        name: _greg_numeric(
            population_weights=weights,
            prediction=prediction,
            observations=observations,
            sample=sample,
        )
        for name, (prediction, observations) in numeric_observations.items()
    }
    kish = _kish_ess(sample, weights)
    pi_values = [
        float(sample["inclusion_probability"][index])
        for index in sample["selected"]
    ]
    pooling = _partial_pooling_v2(
        sample=sample,
        parent_weights=weights,
        semantics=semantics,
    )
    public = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "tick": tick,
        "phase": TIMELINE[tick],
        "branch": branch,
        "budget": budget,
        "fixture_decision_count": len(decisions),
        "rendered_probe_count": len(decisions),
        "allocation_by_stratum": sample["allocation_by_stratum"],
        "inclusion_probability_range": [
            round(min(pi_values), 8),
            round(max(pi_values), 8),
        ],
        "kish_ess": round(kish, 6),
        "corrected": {
            "action": {
                key: round(value, 6) for key, value in corrected_action.items()
            },
            "stance": {
                key: round(value, 6) for key, value in corrected_stance.items()
            },
            "topic_id": {
                key: round(value, 6) for key, value in corrected_topic.items()
            },
            "numeric": {
                key: round(value, 6) for key, value in corrected_numeric.items()
            },
        },
        "partial_pooling": pooling,
        "calibration_status": (
            "identity_fallback_due_to_missing_paired_calibration_evidence"
        ),
        "provider_calls": 0,
    }
    private = {
        **public,
        "selected_identity_sha256": sample["selected_identity_sha256"],
        "selected_indices": list(sample["selected"]),
        "selected_inclusion_probability": {
            str(index): round(float(sample["inclusion_probability"][index]), 12)
            for index in sample["selected"]
        },
        "fixture_decisions": {
            str(index): semantic for index, semantic in semantics.items()
        },
        "stratum_by_selected_index": {
            str(index): str(sample["strata"][index])
            for index in sample["selected"]
        },
        "stratum_members": {
            label: np.flatnonzero(sample["strata"] == label).tolist()
            for label in sorted(set(sample["strata"].tolist()))
        },
    }
    _replace_surrogate_emissions_with_probes(
        population=population,
        features=features,
        state=state,
        semantics=semantics,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
    )
    return public, private


def _systematic_resample_parent(
    ensemble: ParticleEnsemble,
    parent_index: int,
    *,
    seed: int,
    innovation_id: str,
) -> None:
    normalized = ensemble.normalized_within_parent()[parent_index]
    cumulative = np.cumsum(normalized)
    particle_count = int(ensemble.values.shape[1])
    start = _hash_uniform(seed, innovation_id, parent_index, "systematic") / particle_count
    positions = start + np.arange(particle_count) / particle_count
    selected = np.searchsorted(cumulative, positions, side="right")
    selected = np.clip(selected, 0, particle_count - 1)
    rng = np.random.default_rng(
        _seed_from(seed, innovation_id, parent_index, "resample-jitter")
    )
    values = ensemble.values[parent_index, selected].copy()
    jitter = rng.normal(0.0, 0.008, size=values.shape).astype(np.float32)
    jitter[:, [SUPPORT, OPPOSITION, RUMOR_BELIEF, VERIFIED_BELIEF]] *= 1.35
    ensemble.values[parent_index] = np.clip(values + jitter, 0.0, 1.0)
    ensemble.weights[parent_index] = (
        ensemble.parent_weights[parent_index] / particle_count
    )
    ensemble.resampling_count += 1


def assimilate_forum_llm_updates_v2(
    *,
    state: BranchState,
    updates: Mapping[int, Mapping[str, Any]],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
) -> dict[str, Any]:
    """Assimilate validated ForumTwin turns into activated parents only.

    ForumTwin has a richer language contract than the older resident-turn
    adapter, so this entry point accepts validated dictionaries rather than a
    second decision class.  It updates within-parent likelihoods, reconciles
    bounded particle state, and uses the same deterministic ESS resampling as
    the population probe path.  No message is created here.
    """

    before_resampling = state.ensemble.resampling_count
    applied_innovations: list[str] = []
    parent_ess = state.ensemble.parent_ess()
    public_actions = {
        "post",
        "reply",
        "quote",
        "repost",
        "seek_verify",
        "correct",
        "seek_help",
        "report_risk",
    }
    for parent_index in sorted(updates):
        if not 0 <= int(parent_index) < state.ensemble.parent_count:
            raise GovernanceArenaV2Error("ForumTwin LLM update parent is invalid")
        row = updates[parent_index]
        decision = row.get("decision")
        prompt_sha256 = str(row.get("prompt_sha256") or "")
        response_sha256 = str(row.get("response_sha256") or "")
        if (
            not isinstance(decision, Mapping)
            or len(prompt_sha256) != 64
            or len(response_sha256) != 64
        ):
            raise GovernanceArenaV2Error("ForumTwin LLM update identity is invalid")
        innovation_id = canonical_hash(
            [
                "forum-individual-llm-update-v2",
                scenario_id,
                branch,
                tick,
                seed,
                int(parent_index),
                prompt_sha256,
                response_sha256,
            ]
        )
        if innovation_id in state.ensemble.innovations:
            continue

        particles = state.ensemble.values[parent_index].copy()
        stance = str(decision.get("stance") or "uncertain")
        if stance in {"support", "supportive"}:
            likelihood = 0.12 + 0.88 * particles[:, SUPPORT]
        elif stance in {"oppose", "critical", "concerned"}:
            likelihood = 0.12 + 0.88 * np.maximum(
                particles[:, OPPOSITION], particles[:, CONCERN]
            )
        else:
            likelihood = 0.20 + 0.80 * (
                1.0 - np.abs(particles[:, SUPPORT] - particles[:, OPPOSITION])
            )
        action = str(decision.get("action") or "read")
        activity_observed = float(action in public_actions)
        likelihood *= 0.20 + 0.80 * (
            particles[:, ACTIVITY]
            if activity_observed
            else 1.0 - particles[:, ACTIVITY]
        )
        urgency = _clip(float(decision.get("urgency", 0.0)))
        likelihood *= 0.35 + 0.65 * (
            1.0 - np.abs(particles[:, CONCERN] - urgency)
        )
        if state.ensemble.values.shape[1] == 1:
            # A point estimate has no within-parent uncertainty to reweight.
            # Keeping its exact parent mass also avoids accumulating round-off
            # over the 1000 sequential agent updates in one population tick.
            state.ensemble.weights[parent_index, 0] = state.ensemble.parent_weights[parent_index]
        else:
            state.ensemble.weights[parent_index] *= np.maximum(likelihood, 1e-8)
            mass = float(state.ensemble.weights[parent_index].sum())
            if mass <= 1e-15:
                state.ensemble.weights[parent_index] = (
                    state.ensemble.parent_weights[parent_index]
                    / state.ensemble.values.shape[1]
                )
            else:
                state.ensemble.weights[parent_index] *= (
                    state.ensemble.parent_weights[parent_index] / mass
                )

        trust_delta = float(decision.get("trust_delta", 0.0))
        satisfaction_delta = float(decision.get("satisfaction_delta", 0.0))
        confidence = _clip(float(decision.get("confidence", 0.5)))
        state.ensemble.values[parent_index, :, ATTENTION] = np.clip(
            np.maximum(0.72 * particles[:, ATTENTION], urgency), 0.0, 1.0
        )
        state.ensemble.values[parent_index, :, CONCERN] = np.clip(
            0.72 * particles[:, CONCERN] + 0.28 * urgency, 0.0, 1.0
        )
        state.ensemble.values[parent_index, :, TRUST] = np.clip(
            particles[:, TRUST] + trust_delta, 0.0, 1.0
        )
        state.ensemble.values[parent_index, :, SATISFACTION] = np.clip(
            particles[:, SATISFACTION] + satisfaction_delta, 0.0, 1.0
        )
        state.ensemble.values[parent_index, :, ACTIVITY] = np.clip(
            0.70 * particles[:, ACTIVITY] + 0.30 * activity_observed,
            0.0,
            1.0,
        )
        if stance in {"support", "supportive"}:
            state.ensemble.values[parent_index, :, SUPPORT] = np.clip(
                particles[:, SUPPORT] + 0.10 * confidence, 0.0, 1.0
            )
            state.ensemble.values[parent_index, :, OPPOSITION] = np.clip(
                particles[:, OPPOSITION] - 0.05 * confidence, 0.0, 1.0
            )
        elif stance in {"oppose", "critical", "concerned"}:
            state.ensemble.values[parent_index, :, OPPOSITION] = np.clip(
                particles[:, OPPOSITION] + 0.10 * confidence, 0.0, 1.0
            )
            state.ensemble.values[parent_index, :, SUPPORT] = np.clip(
                particles[:, SUPPORT] - 0.05 * confidence, 0.0, 1.0
            )
        evidence_status = str(decision.get("evidence_status") or "unverified")
        if evidence_status in {"verified", "corrected"} or action == "correct":
            state.ensemble.values[parent_index, :, VERIFIED_BELIEF] = np.clip(
                particles[:, VERIFIED_BELIEF] + 0.08 * confidence, 0.0, 1.0
            )
            state.ensemble.values[parent_index, :, RUMOR_BELIEF] = np.clip(
                particles[:, RUMOR_BELIEF] - 0.06 * confidence, 0.0, 1.0
            )

        state.ensemble.innovations.add(innovation_id)
        applied_innovations.append(innovation_id)
        parent_ess = state.ensemble.parent_ess()
        if parent_ess[parent_index] < 0.5 * state.ensemble.values.shape[1]:
            _systematic_resample_parent(
                state.ensemble,
                int(parent_index),
                seed=seed,
                innovation_id=innovation_id,
            )

    state.ensemble.assert_conserved()
    return {
        "applied_count": len(applied_innovations),
        "innovation_ids": applied_innovations,
        "resampling_count": (
            state.ensemble.resampling_count - before_resampling
        ),
        "minimum_parent_ess": round(
            float(np.min(state.ensemble.parent_ess())), 6
        ),
    }


def apply_probe_writeback_v2(
    *,
    state: BranchState,
    probe_private: Mapping[str, Any],
    mode: str,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
) -> dict[str, Any]:
    """Apply one idempotent stratum-posterior update to particle states."""

    if mode not in WRITEBACK_MODES:
        raise GovernanceArenaV2Error("unknown writeback mode")
    innovation_id = canonical_hash(
        [
            scenario_id,
            branch,
            tick,
            seed,
            probe_private["selected_identity_sha256"],
            mode,
        ]
    )
    if innovation_id in state.ensemble.innovations:
        return {
            "innovation_id": innovation_id,
            "applied": False,
            "reason": "already_applied",
            "resampling_count": 0,
        }
    state.ensemble.innovations.add(innovation_id)
    if mode == "observation_only":
        return {
            "innovation_id": innovation_id,
            "applied": False,
            "reason": "observation_only",
            "resampling_count": 0,
        }

    before_resampling = state.ensemble.resampling_count
    semantics = {
        int(index): value
        for index, value in probe_private["fixture_decisions"].items()
    }
    for index, semantic in semantics.items():
        particles = state.ensemble.values[index]
        if semantic["stance"] == "supportive":
            likelihood = 0.15 + 0.85 * particles[:, SUPPORT]
        elif semantic["stance"] in {"critical", "concerned"}:
            likelihood = 0.15 + 0.85 * np.maximum(
                particles[:, OPPOSITION], particles[:, CONCERN]
            )
        else:
            likelihood = 0.25 + 0.75 * (
                1.0 - np.abs(particles[:, SUPPORT] - particles[:, OPPOSITION])
            )
        state.ensemble.weights[index] *= likelihood
        mass = float(state.ensemble.weights[index].sum())
        if mass <= 1e-15:
            state.ensemble.weights[index] = (
                state.ensemble.parent_weights[index] / PARTICLES_PER_PARENT
            )
        else:
            state.ensemble.weights[index] *= (
                state.ensemble.parent_weights[index] / mass
            )
        if state.ensemble.parent_ess()[index] < 5.0:
            _systematic_resample_parent(
                state.ensemble,
                index,
                seed=seed,
                innovation_id=innovation_id,
            )

    strata = probe_private["partial_pooling"]["strata"]
    stratum_members = probe_private["stratum_members"]
    parent_state = state.ensemble.parent_means()
    for label, posterior in strata.items():
        all_members = [int(index) for index in stratum_members.get(label, [])]
        if not all_members:
            continue
        # Stratum membership is a private design variable.  A bounded common
        # shift writes the posterior back across the full stratum while
        # preserving within-stratum particle ordering.
        members = np.asarray(all_members, dtype=np.int64)
        stance = posterior["conditional_stance"]
        target_support = float(stance.get("supportive", 0.0))
        target_opposition = float(stance.get("critical", 0.0)) + 0.5 * float(
            stance.get("concerned", 0.0)
        )
        current_support = float(
            np.average(
                parent_state[members, SUPPORT],
                weights=state.ensemble.parent_weights[members],
            )
        )
        current_opposition = float(
            np.average(
                parent_state[members, OPPOSITION],
                weights=state.ensemble.parent_weights[members],
            )
        )
        support_shift = float(np.clip(target_support - current_support, -0.12, 0.12))
        opposition_shift = float(
            np.clip(target_opposition - current_opposition, -0.12, 0.12)
        )
        state.ensemble.values[members, :, SUPPORT] = np.clip(
            state.ensemble.values[members, :, SUPPORT] + support_shift,
            0.0,
            1.0,
        )
        state.ensemble.values[members, :, OPPOSITION] = np.clip(
            state.ensemble.values[members, :, OPPOSITION] + opposition_shift,
            0.0,
            1.0,
        )
        if mode == "full_state":
            speaking_shift = float(
                np.clip(
                    posterior["posterior_speaking_rate"]
                    - float(np.mean(parent_state[members, ACTIVITY])),
                    -0.08,
                    0.08,
                )
            )
            state.ensemble.values[members, :, ACTIVITY] = np.clip(
                state.ensemble.values[members, :, ACTIVITY] + speaking_shift,
                0.0,
                1.0,
            )
            state.ensemble.values[members, :, ATTENTION] = np.clip(
                state.ensemble.values[members, :, ATTENTION]
                + 0.45 * speaking_shift,
                0.0,
                1.0,
            )
            state.ensemble.values[members, :, TRUST] = np.clip(
                state.ensemble.values[members, :, TRUST]
                + 0.25 * support_shift
                - 0.20 * opposition_shift,
                0.0,
                1.0,
            )
            state.ensemble.values[members, :, CONCERN] = np.clip(
                state.ensemble.values[members, :, CONCERN]
                + 0.25 * opposition_shift
                - 0.15 * support_shift,
                0.0,
                1.0,
            )
    state.ensemble.assert_conserved()
    return {
        "innovation_id": innovation_id,
        "applied": True,
        "reason": mode,
        "resampling_count": (
            state.ensemble.resampling_count - before_resampling
        ),
    }


def evaluate_writeback_gate_v2(
    evidence: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Apply the frozen promotion thresholds without inventing paired evidence."""

    thresholds = {
        "noninferiority_margin": 0.005,
        "minimum_absolute_improvement": 0.005,
        "maximum_seed_variance_ratio": 1.10,
        "coverage_range": [0.90, 0.99],
        "legacy_degradation_limit": 0.02,
    }
    if evidence is None:
        return {
            "schema_version": "campus-pulse-writeback-gate-v2",
            "formal_mode": FORMAL_WRITEBACK_MODE,
            "calibration_status": (
                "identity_fallback_due_to_missing_paired_calibration_evidence"
            ),
            "thresholds": thresholds,
            "modes": {
                mode: {
                    "implemented": True,
                    "eligible_for_formal_use": mode == FORMAL_WRITEBACK_MODE,
                }
                for mode in WRITEBACK_MODES
            },
        }
    baseline = evidence.get("observation_only")
    if not baseline:
        raise GovernanceArenaV2Error("writeback evidence lacks observation_only")
    formal = FORMAL_WRITEBACK_MODE
    mode_rows = {}
    for mode in WRITEBACK_MODES:
        row = evidence.get(mode)
        if not row:
            mode_rows[mode] = {
                "implemented": True,
                "eligible_for_formal_use": False,
                "reason": "missing_evidence",
            }
            continue
        noninferior = (
            float(row["level_mae"])
            <= float(baseline["level_mae"]) + thresholds["noninferiority_margin"]
            and float(row["difference_mae"])
            <= float(baseline["difference_mae"])
            + thresholds["noninferiority_margin"]
        )
        improved = max(
            float(baseline["level_mae"]) - float(row["level_mae"]),
            float(baseline["difference_mae"]) - float(row["difference_mae"]),
        ) >= thresholds["minimum_absolute_improvement"]
        eligible = (
            mode == FORMAL_WRITEBACK_MODE
            or (
                noninferior
                and improved
                and float(row["seed_variance_ratio"])
                <= thresholds["maximum_seed_variance_ratio"]
                and thresholds["coverage_range"][0]
                <= float(row["coverage_95"])
                <= thresholds["coverage_range"][1]
                and float(row["legacy_degradation"])
                <= thresholds["legacy_degradation_limit"]
            )
        )
        mode_rows[mode] = {
            **{key: round(float(value), 8) for key, value in row.items()},
            "implemented": True,
            "eligible_for_formal_use": eligible,
        }
        if mode == "stance_only" and eligible:
            formal = mode
        if mode == "full_state" and eligible and formal == "stance_only":
            stance = evidence["stance_only"]
            if (
                float(row["level_mae"]) < float(stance["level_mae"])
                and float(row["difference_mae"]) < float(stance["difference_mae"])
            ):
                formal = mode
    return {
        "schema_version": "campus-pulse-writeback-gate-v2",
        "formal_mode": formal,
        "calibration_status": (
            "validated_frozen_evidence"
            if formal != FORMAL_WRITEBACK_MODE
            else "identity_fallback_due_to_gate"
        ),
        "thresholds": thresholds,
        "modes": mode_rows,
    }


def _update_actor_beliefs(
    *,
    state: BranchState,
    diffusion: Mapping[str, Any],
    probe: Mapping[str, Any] | None,
    prior_public: Mapping[str, Any] | None,
) -> None:
    """Update role-specific beliefs without passing hidden truth."""

    emissions = float(diffusion["emissions"])
    reads = float(diffusion["reads"])
    repost_rate = float(diffusion["reposts"]) / max(emissions, 1.0)
    correction_rate = float(diffusion["corrections"]) / max(emissions, 1.0)
    authority = state.actor_beliefs["governance_authority"]
    service = state.actor_beliefs["service_operator"]
    bridge = state.actor_beliefs["community_bridge"]
    delayed_concern = (
        float(prior_public["observed"]["concern"]["mean"])
        if prior_public is not None
        else authority["concern"]
    )
    authority["concern"] = _clip(
        0.68 * authority["concern"]
        + 0.18 * delayed_concern
        + 0.14 * repost_rate
    )
    authority["rumor_belief"] = _clip(
        0.72 * authority["rumor_belief"]
        + 0.20 * repost_rate
        - 0.14 * correction_rate
    )
    authority["verified_belief"] = _clip(
        0.74 * authority["verified_belief"] + 0.20 * correction_rate
    )
    authority["risk_reports"] = _clip(
        0.72 * authority["risk_reports"]
        + 0.28 * min(float(diffusion["emissions"]) / 500.0, 1.0)
    )
    service["help_queue"] = _clip(
        0.75 * service["help_queue"] + 0.25 * min(emissions / 450.0, 1.0)
    )
    service["sla_pressure"] = _clip(
        0.70 * service["sla_pressure"] + 0.30 * service["help_queue"]
    )
    service["service_strain"] = _clip(
        0.72 * service["service_strain"]
        + 0.18 * service["help_queue"]
        - 0.12 * state.active_effects["service"]
    )
    service["satisfaction"] = _clip(
        0.76 * service["satisfaction"]
        + 0.12 * correction_rate
        - 0.14 * service["service_strain"]
    )
    bridge["bridge_share"] = _clip(
        0.70 * bridge["bridge_share"] + 0.30 * float(diffusion["bridge_share"])
    )
    bridge["participation"] = _clip(
        0.76 * bridge["participation"] + 0.24 * min(reads / 2000.0, 1.0)
    )
    bridge["voice_gap"] = _clip(
        0.72 * bridge["voice_gap"]
        + 0.16 * (1.0 - bridge["participation"])
        - 0.14 * state.active_effects["voice"]
    )
    bridge["concern"] = _clip(
        0.76 * bridge["concern"] + 0.24 * authority["concern"]
    )
    if probe is not None:
        numeric = probe["corrected"]["numeric"]
        stance = probe["corrected"]["stance"]
        authority["concern"] = float(numeric["concern"])
        authority["trust"] = float(numeric["trust"])
        service["satisfaction"] = float(numeric["satisfaction"])
        bridge["participation"] = float(numeric["activity"])
        authority["rumor_belief"] = _clip(
            float(stance.get("critical", 0.0))
            + 0.5 * float(stance.get("concerned", 0.0))
        )
        authority["verified_belief"] = _clip(
            float(stance.get("supportive", 0.0))
        )


def _actor_observations(state: BranchState) -> dict[str, dict[str, float]]:
    """Return type-separated limited observations; no hidden truth is present."""

    return {
        "governance_authority": dict(
            state.actor_beliefs["governance_authority"]
        ),
        "service_operator": dict(state.actor_beliefs["service_operator"]),
        "community_bridge": dict(state.actor_beliefs["community_bridge"]),
    }


def _action_utility(
    *,
    actor: str,
    action_id: str,
    observation: Mapping[str, float],
    coordination: Mapping[str, float],
    horizon: int = 4,
) -> float:
    """Return discounted welfare improvement from an explicit delayed rollout."""

    if horizon < 1:
        raise GovernanceArenaV2Error("governance rollout horizon must be positive")
    if action_id == "noop":
        effects: Mapping[str, float] = {}
        cost = 0.0
        delay = horizon + 1
        reliability = 0.0
    else:
        definition = ACTION_CATALOG[action_id]
        effects = definition["effects"]
        cost = float(definition["cost"])
        delay = int(definition["delay"])
        reliability = float(definition["reliability"])

    def mandate_loss(values: Mapping[str, float]) -> float:
        if actor == "governance_authority":
            return (
                1.10 * float(values.get("concern", 0.3))
                + 1.00 * float(values.get("rumor_belief", 0.15))
                + 0.35 * (1.0 - float(values.get("trust", 0.5)))
                + 0.15 * float(values.get("risk_reports", 0.0))
            )
        if actor == "service_operator":
            return (
                1.20 * float(values.get("service_strain", 0.25))
                + 0.80 * float(values.get("help_queue", 0.1))
                + 0.30 * (1.0 - float(values.get("satisfaction", 0.5)))
                + 0.20 * float(values.get("sla_pressure", 0.0))
            )
        return (
            1.20 * float(values.get("voice_gap", 0.15))
            + 0.50 * float(values.get("concern", 0.3))
            + 0.30 * (1.0 - float(values.get("participation", 0.2)))
        )

    def advance_visible(
        values: Mapping[str, float],
        delivered: Mapping[str, float],
    ) -> dict[str, float]:
        next_values = dict(values)
        verification = float(delivered.get("verification", 0.0))
        reassurance = float(delivered.get("reassurance", 0.0))
        service_effect = float(delivered.get("service", 0.0))
        voice = float(delivered.get("voice", 0.0))
        bridge = float(delivered.get("bridge", 0.0))
        prebunk = float(delivered.get("prebunk", 0.0))
        if actor == "governance_authority":
            rumor = float(values.get("rumor_belief", 0.15))
            next_values["concern"] = _clip(
                0.94 * float(values.get("concern", 0.3))
                + 0.025
                + 0.08 * rumor
                - 0.18 * reassurance
                - 0.10 * voice
            )
            next_values["rumor_belief"] = _clip(
                0.93 * rumor + 0.015 - 0.22 * verification - 0.18 * prebunk
            )
            next_values["trust"] = _clip(
                0.96 * float(values.get("trust", 0.5))
                + 0.11 * verification
                + 0.05 * reassurance
                - 0.05 * rumor
            )
            next_values["verified_belief"] = _clip(
                0.94 * float(values.get("verified_belief", 0.15))
                + 0.16 * verification
            )
        elif actor == "service_operator":
            strain = float(values.get("service_strain", 0.25))
            queue = float(values.get("help_queue", 0.1))
            next_values["service_strain"] = _clip(
                0.93 * strain + 0.04 * queue - 0.20 * service_effect
            )
            next_values["help_queue"] = _clip(
                0.92 * queue + 0.03 - 0.16 * service_effect
            )
            next_values["satisfaction"] = _clip(
                0.96 * float(values.get("satisfaction", 0.5))
                + 0.10 * service_effect
                + 0.05 * reassurance
                - 0.04 * strain
            )
            next_values["sla_pressure"] = _clip(
                0.94 * float(values.get("sla_pressure", 0.0))
                + 0.06 * queue
                - 0.10 * service_effect
            )
        else:
            next_values["voice_gap"] = _clip(
                0.94 * float(values.get("voice_gap", 0.15))
                + 0.025
                - 0.18 * voice
                - 0.12 * bridge
            )
            next_values["concern"] = _clip(
                0.95 * float(values.get("concern", 0.3))
                + 0.02
                - 0.06 * reassurance
                - 0.08 * bridge
            )
            next_values["participation"] = _clip(
                0.95 * float(values.get("participation", 0.2))
                + 0.12 * voice
                + 0.08 * bridge
            )
        return next_values

    baseline = dict(observation)
    intervention = dict(observation)
    baseline_loss = 0.0
    intervention_loss = 0.0
    for step in range(1, horizon + 1):
        baseline = advance_visible(baseline, {})
        if step >= delay:
            decay = 0.82 ** (step - delay)
            delivered = {
                key: reliability * decay * float(value)
                for key, value in effects.items()
            }
        else:
            delivered = {}
        intervention = advance_visible(intervention, delivered)
        discount = 0.92 ** (step - 1)
        baseline_loss += discount * mandate_loss(baseline)
        intervention_loss += discount * mandate_loss(intervention)
    complement = (
        0.08
        * min(
            coordination.get("verification", 0.0),
            float(effects.get("bridge", 0.0))
            + float(effects.get("voice", 0.0)),
        )
        + 0.08
        * min(
            coordination.get("service", 0.0),
            float(effects.get("reassurance", 0.0))
            + float(effects.get("voice", 0.0)),
        )
    )
    return baseline_loss - intervention_loss + complement - 0.010 * cost


def _prune_dominated_actions(action_ids: Sequence[str]) -> tuple[str, ...]:
    """Remove actions weakly dominated in mechanism effects and cost."""

    candidates = list(dict.fromkeys(action_ids))
    kept = []
    for left in candidates:
        left_definition = ACTION_CATALOG[left]
        left_effects = left_definition["effects"]
        dominated = False
        for right in candidates:
            if left == right:
                continue
            right_definition = ACTION_CATALOG[right]
            right_effects = right_definition["effects"]
            dimensions = set(left_effects) | set(right_effects)
            weakly_better = (
                float(right_definition["cost"])
                <= float(left_definition["cost"])
                and all(
                    float(right_effects.get(key, 0.0))
                    >= float(left_effects.get(key, 0.0))
                    for key in dimensions
                )
            )
            strictly_better = (
                float(right_definition["cost"])
                < float(left_definition["cost"])
                or any(
                    float(right_effects.get(key, 0.0))
                    > float(left_effects.get(key, 0.0))
                    for key in dimensions
                )
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(left)
    return tuple(kept)


def _softmax_probabilities(
    utilities: Mapping[str, float], temperature: float
) -> dict[str, float]:
    maximum = max(utilities.values())
    exponentials = {
        key: math.exp(
            max(-40.0, min(40.0, (value - maximum) / temperature))
        )
        for key, value in utilities.items()
    }
    return _normalize(exponentials)


def _draw_mapping(probabilities: Mapping[str, float], uniform: float) -> str:
    cumulative = 0.0
    last = next(iter(probabilities))
    for key, value in probabilities.items():
        last = key
        cumulative += float(value)
        if uniform <= cumulative + 1e-12:
            return key
    return last


def play_qre_governance_round(
    *,
    state: BranchState,
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    temperature: float = 0.20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run three damped logit-response rounds on role-private observations."""

    if temperature not in {0.10, 0.20, 0.35}:
        raise GovernanceArenaV2Error("QRE temperature is outside frozen grid")
    observations = _actor_observations(state)
    if any("truth" in key.lower() for obs in observations.values() for key in obs):
        raise GovernanceArenaV2Error("actor observation leaked hidden truth")
    actors = (
        "governance_authority",
        "service_operator",
        "community_bridge",
    )
    available = {
        actor: _prune_dominated_actions(
            [
                action_id
                for action_id in BRANCH_ACTIONS[branch]
                if ACTION_CATALOG[action_id]["actor"] == actor
            ]
        )
        for actor in actors
    }
    coordination = defaultdict(float)
    final_probabilities: dict[str, dict[str, float]] = {}
    final_utilities: dict[str, dict[str, float]] = {}
    selections = {actor: "noop" for actor in actors}
    iterations = []
    remaining = (
        state.resource_opening
        - state.resource_consumed
        - state.resource_committed
    )
    for iteration in range(3):
        iteration_rows = []
        next_coordination = defaultdict(float)
        for actor in actors:
            candidates = ["noop"] + [
                action_id
                for action_id in available[actor]
                if float(ACTION_CATALOG[action_id]["cost"]) <= remaining + 1e-9
            ]
            utilities = {
                action_id: _action_utility(
                    actor=actor,
                    action_id=action_id,
                    observation=observations[actor],
                    coordination=coordination,
                )
                for action_id in candidates
            }
            probabilities = _softmax_probabilities(utilities, temperature)
            selection = _draw_mapping(
                probabilities,
                _hash_uniform(
                    seed,
                    scenario_id,
                    branch,
                    tick,
                    iteration,
                    actor,
                    "qre",
                ),
            )
            selections[actor] = selection
            final_probabilities[actor] = probabilities
            final_utilities[actor] = utilities
            if selection != "noop":
                for key, value in ACTION_CATALOG[selection]["effects"].items():
                    next_coordination[key] += float(value)
            iteration_rows.append(
                {
                    "actor": actor,
                    "utilities": {
                        key: round(value, 6) for key, value in utilities.items()
                    },
                    "selection": selection,
                }
            )
        for key in set(coordination) | set(next_coordination):
            coordination[key] = (
                0.45 * coordination[key] + 0.55 * next_coordination[key]
            )
        iterations.append(iteration_rows)

    myopic_rows = []
    for actor in actors:
        candidates = ["noop"] + [
            action_id
            for action_id in available[actor]
            if float(ACTION_CATALOG[action_id]["cost"]) <= remaining + 1e-9
        ]
        utilities = {
            action_id: _action_utility(
                actor=actor,
                action_id=action_id,
                observation=observations[actor],
                coordination={},
                horizon=1,
            )
            for action_id in candidates
        }
        selected = max(
            utilities,
            key=lambda action_id: (utilities[action_id], action_id),
        )
        myopic_rows.append(
            {
                "actor": actor,
                "selected_action": selected,
                "candidate_utilities": {
                    key: round(value, 6)
                    for key, value in utilities.items()
                },
            }
        )

    decisions = []
    for actor in actors:
        selection = selections[actor]
        scheduled = False
        cost = 0.0
        delay = 0
        if selection != "noop":
            definition = ACTION_CATALOG[selection]
            cost = float(definition["cost"])
            remaining = (
                state.resource_opening
                - state.resource_consumed
                - state.resource_committed
            )
            if cost <= remaining + 1e-9:
                delay = int(definition["delay"])
                state.resource_committed += cost
                state.pending_actions.append(
                    {
                        "action_id": selection,
                        "actor": actor,
                        "selected_tick": tick,
                        "due_tick": tick + delay,
                        "cost": cost,
                        "effects": dict(definition["effects"]),
                    }
                )
                scheduled = True
            else:
                selection = "noop"
        probabilities = final_probabilities.get(actor, {"noop": 1.0})
        decisions.append(
            {
                "actor": actor,
                "selected_action": selection,
                "selection_probability": round(
                    float(probabilities.get(selection, 1.0)), 6
                ),
                "candidate_probabilities": {
                    key: round(value, 6)
                    for key, value in probabilities.items()
                },
                "candidate_utilities": {
                    key: round(value, 6)
                    for key, value in final_utilities.get(
                        actor, {"noop": 0.0}
                    ).items()
                },
                "scheduled": scheduled,
                "execution_delay": delay,
                "reliability": (
                    float(ACTION_CATALOG[selection]["reliability"])
                    if selection != "noop"
                    else 1.0
                ),
                "resource_units": cost if scheduled else 0.0,
                "counterfactual_best_alternative": next(
                    (
                        key
                        for key, _ in sorted(
                            probabilities.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                        if key != selection
                    ),
                    "noop",
                ),
            }
        )
    remaining = (
        state.resource_opening
        - state.resource_consumed
        - state.resource_committed
    )
    if remaining < -1e-8:
        raise GovernanceArenaV2Error("QRE decisions overspent the branch budget")
    public = {
        "schema_version": GAME_SCHEMA_VERSION,
        "tick": tick,
        "temperature": temperature,
        "response_iterations": 3,
        "lookahead_ticks": 4,
        "rollout": {
            "horizon": 4,
            "discount": 0.92,
            "effect_decay": 0.82,
            "uses_hidden_truth": False,
        },
        "myopic_control": {
            "horizon": 1,
            "decisions": myopic_rows,
        },
        "decisions": decisions,
        "resource_after_decision": {
            "opening": round(state.resource_opening, 6),
            "consumed": round(state.resource_consumed, 6),
            "committed": round(state.resource_committed, 6),
            "remaining": round(remaining, 6),
            "conserved": math.isclose(
                state.resource_opening,
                state.resource_consumed
                + state.resource_committed
                + remaining,
                abs_tol=1e-8,
            ),
        },
    }
    private = {
        **public,
        "private_observations": observations,
        "iterations": iterations,
    }
    state.game_rounds.append(public)
    return public, private


def aggregate_truth_v2(
    *,
    population: Mapping[str, Any],
    state: BranchState,
    features: Mapping[str, np.ndarray],
) -> dict[str, float]:
    mean = state.ensemble.weighted_mean()
    parent = state.ensemble.parent_means()
    archetypes = features["archetypes"]
    group_voice = []
    for archetype in sorted(set(archetypes.tolist())):
        members = np.flatnonzero(archetypes == archetype)
        mass = float(state.ensemble.parent_weights[members].sum())
        if mass <= 0.0:
            continue
        value = float(
            np.sum(
                parent[members, VOICE]
                * state.ensemble.parent_weights[members]
            )
            / mass
        )
        group_voice.append(value)
    voice_gap = max(group_voice, default=0.0) - min(group_voice, default=0.0)
    return {
        "attention": round(float(mean[ATTENTION]), 6),
        "concern": round(float(mean[CONCERN]), 6),
        "trust": round(float(mean[TRUST]), 6),
        "satisfaction": round(float(mean[SATISFACTION]), 6),
        "support": round(float(mean[SUPPORT]), 6),
        "opposition": round(float(mean[OPPOSITION]), 6),
        "activity": round(float(mean[ACTIVITY]), 6),
        "voice_gap": round(_clip(voice_gap), 6),
        "service_strain": round(float(mean[SERVICE_STRAIN]), 6),
        "rumor_belief": round(float(mean[RUMOR_BELIEF]), 6),
        "verified_belief": round(float(mean[VERIFIED_BELIEF]), 6),
    }


def observed_state_v2(
    *,
    state: BranchState,
    truth_keys: Iterable[str],
    probe: Mapping[str, Any] | None,
) -> dict[str, dict[str, float]]:
    authority = state.actor_beliefs["governance_authority"]
    service = state.actor_beliefs["service_operator"]
    bridge = state.actor_beliefs["community_bridge"]
    estimates = {
        "attention": _clip(
            0.45 * authority["concern"] + 0.55 * bridge["participation"]
        ),
        "concern": authority["concern"],
        "trust": authority["trust"],
        "satisfaction": service["satisfaction"],
        "support": authority["verified_belief"],
        "opposition": authority["rumor_belief"],
        "activity": bridge["participation"],
        "voice_gap": bridge["voice_gap"],
        "service_strain": service["service_strain"],
        "rumor_belief": authority["rumor_belief"],
        "verified_belief": authority["verified_belief"],
    }
    ess = 12.0
    if probe is not None:
        ess = float(probe["kish_ess"])
        numeric = probe["corrected"]["numeric"]
        stance = probe["corrected"]["stance"]
        estimates.update(
            {
                "concern": float(numeric["concern"]),
                "trust": float(numeric["trust"]),
                "satisfaction": float(numeric["satisfaction"]),
                "activity": float(numeric["activity"]),
                "support": float(stance.get("supportive", 0.0)),
                "opposition": _clip(
                    float(stance.get("critical", 0.0))
                    + 0.5 * float(stance.get("concerned", 0.0))
                ),
            }
        )
    return {
        key: _normal_interval(float(estimates[key]), ess)
        for key in truth_keys
    }


def encode_visual_frame_v2(
    *,
    state: BranchState,
    scenario_id: str,
    branch: str,
    tick: int,
) -> dict[str, Any]:
    """Encode anonymous quantized particle attributes without a parent map."""

    values = state.ensemble.values.reshape(-1, len(STATE_FIELDS))
    stance = np.clip(
        np.round(
            127.0
            + 127.0 * (values[:, SUPPORT] - values[:, OPPOSITION])
        ),
        0,
        255,
    ).astype(np.uint8)
    attention = np.round(255.0 * values[:, ATTENTION]).astype(np.uint8)
    public_parent = np.zeros(state.ensemble.parent_count, dtype=np.uint8)
    for message in state.ledger.messages:
        if (
            message.created_tick == tick
            and message.source_parent_index is not None
            and message.action != "suppressed"
        ):
            public_parent[int(message.source_parent_index)] = 1
    public_action = np.repeat(public_parent, PARTICLES_PER_PARENT)
    parent_rank = np.argsort(np.argsort(state.ensemble.parent_weights))
    weight_bin = np.floor(
        255.0 * parent_rank / max(len(parent_rank) - 1, 1)
    ).astype(np.uint8)
    weight_bin = np.repeat(weight_bin, PARTICLES_PER_PARENT)
    matrix = np.column_stack((stance, attention, weight_bin, public_action))
    raw = matrix.tobytes(order="C")
    compressed = zlib.compress(raw, level=9)
    return {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "particle_count": int(len(matrix)),
        "channels": ["stance", "attention", "weight_bin", "public_action"],
        "dtype": "uint8",
        "shape": [int(len(matrix)), 4],
        "record_stride_bytes": 4,
        "encoding": "zlib+base64",
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": hashlib.sha256(compressed).hexdigest(),
        "payload": base64.b64encode(compressed).decode("ascii"),
    }


class GovernanceArenaV2Cancelled(RuntimeError):
    """Raised at a deterministic boundary when runtime cancellation is requested."""


def _artifact_payload(
    *,
    kind: str,
    payload: Mapping[str, Any],
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
) -> dict[str, Any]:
    body = {
        "artifact_kind": kind,
        "scenario_id": scenario_id,
        "seed": int(seed),
        "branch": branch,
        "tick": int(tick),
        **dict(payload),
    }
    body.setdefault("content_sha256", canonical_hash(body))
    body["artifact_sha256"] = canonical_hash(body)
    return body


def _emit_artifact(
    *,
    callback: ArtifactCallback | None,
    index: list[dict[str, Any]],
    kind: str,
    payload: Mapping[str, Any],
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
) -> dict[str, Any]:
    artifact = _artifact_payload(
        kind=kind,
        payload=payload,
        scenario_id=scenario_id,
        seed=seed,
        branch=branch,
        tick=tick,
    )
    if callback is not None:
        callback(kind, artifact)
    index.append(
        {
            "artifact_kind": kind,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "branch": branch,
            "tick": int(tick),
            "schema_version": artifact.get("schema_version"),
            "content_sha256": artifact["content_sha256"],
            "artifact_sha256": artifact["artifact_sha256"],
        }
    )
    return artifact


def _check_cancel(
    cancellation_probe: Callable[[], bool] | None,
    *,
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
) -> None:
    if cancellation_probe is not None and cancellation_probe():
        raise GovernanceArenaV2Cancelled(
            f"cancelled at {scenario_id}/{seed}/{branch}/T{tick}"
        )


def _resource_view(state: BranchState) -> dict[str, Any]:
    remaining = (
        state.resource_opening
        - state.resource_consumed
        - state.resource_committed
    )
    return {
        "opening": round(state.resource_opening, 6),
        "consumed": round(state.resource_consumed, 6),
        "committed": round(state.resource_committed, 6),
        "remaining": round(remaining, 6),
        "conserved": math.isclose(
            state.resource_opening,
            state.resource_consumed + state.resource_committed + remaining,
            abs_tol=1e-8,
        ),
    }


def _timeline_point(
    *,
    population: Mapping[str, Any],
    state: BranchState,
    features: Mapping[str, np.ndarray],
    tick: int,
    diffusion: Mapping[str, Any],
    probe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    truth = aggregate_truth_v2(
        population=population, state=state, features=features
    )
    observed = observed_state_v2(
        state=state, truth_keys=truth.keys(), probe=probe
    )
    return {
        "tick": tick,
        "phase": TIMELINE[tick],
        "truth": truth,
        "observed": observed,
        "uncertainty": {
            "particle_ess": round(
                float(
                    np.average(
                        state.ensemble.parent_ess(),
                        weights=state.ensemble.parent_weights,
                    )
                ),
                6,
            ),
            "resampling_count": state.ensemble.resampling_count,
        },
        "probe": {
            "budget": int(probe["budget"]) if probe else 0,
            "kish_ess": round(float(probe["kish_ess"]), 6) if probe else 0.0,
            "allocation_by_stratum": (
                copy.deepcopy(probe["allocation_by_stratum"])
                if probe
                else {}
            ),
        },
        "diffusion": {
            key: diffusion[key]
            for key in (
                "emissions",
                "reads",
                "reposts",
                "corrections",
                "cascade_depth",
                "bridge_share",
                "correction_delay",
                "exposure_inequality",
            )
        },
        "cascade_groups": diffusion["cascade_groups"],
    }


def _current_tick_messages(
    state: BranchState, tick: int
) -> list[MessageObject]:
    return [
        message
        for message in state.ledger.messages
        if message.created_tick == tick and message.action != "suppressed"
    ]


def _current_tick_ledger_messages(
    state: BranchState, tick: int
) -> list[MessageObject]:
    """Return complete replay state, including probe-suppressed surrogates."""

    return [
        message
        for message in state.ledger.messages
        if message.created_tick == tick
    ]


def _committed_artifact(
    resume: Mapping[str, Any] | None,
    *,
    kind: str,
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
) -> Mapping[str, Any] | None:
    if resume is None:
        return None
    artifacts = resume.get("committed_artifacts", {})
    key = f"{kind}|{scenario_id}|{seed}|{branch}|{tick}"
    if isinstance(artifacts, Mapping):
        value = artifacts.get(key)
        return value if isinstance(value, Mapping) else None
    return None


def _restore_committed_probe(
    *,
    artifact: Mapping[str, Any],
    population: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    state: BranchState,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    writeback_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_private = {
        "selected_indices",
        "selected_identity_sha256",
        "fixture_decisions",
        "stratum_members",
    }
    if not required_private.issubset(artifact):
        raise GovernanceArenaV2Error(
            "committed probe artifact lacks replay material"
        )
    private = copy.deepcopy(dict(artifact))
    for index in private["selected_indices"]:
        state.rotation_counts[int(index)] += 1
    semantics = {
        int(index): value
        for index, value in private["fixture_decisions"].items()
    }
    _replace_surrogate_emissions_with_probes(
        population=population,
        features=features,
        state=state,
        semantics=semantics,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
    )
    private["writeback"] = apply_probe_writeback_v2(
        state=state,
        probe_private=private,
        mode=writeback_mode,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        seed=seed,
    )
    public_keys = {
        "schema_version",
        "tick",
        "phase",
        "branch",
        "budget",
        "fixture_decision_count",
        "rendered_probe_count",
        "allocation_by_stratum",
        "inclusion_probability_range",
        "kish_ess",
        "corrected",
        "partial_pooling",
        "calibration_status",
        "provider_calls",
    }
    public = {
        key: copy.deepcopy(value)
        for key, value in private.items()
        if key in public_keys
    }
    return public, private


def _restore_committed_decision(
    *,
    artifact: Mapping[str, Any],
    state: BranchState,
    tick: int,
) -> dict[str, Any]:
    decisions = artifact.get("decisions")
    resource = artifact.get("resource_after_decision")
    if not isinstance(decisions, list) or not isinstance(resource, Mapping):
        raise GovernanceArenaV2Error(
            "committed decision artifact lacks replay material"
        )
    for decision in decisions:
        action_id = str(decision.get("selected_action") or "noop")
        if not decision.get("scheduled") or action_id == "noop":
            continue
        definition = ACTION_CATALOG.get(action_id)
        if definition is None:
            raise GovernanceArenaV2Error(
                "committed decision action is unknown"
            )
        state.pending_actions.append(
            {
                "action_id": action_id,
                "actor": str(decision["actor"]),
                "selected_tick": tick,
                "due_tick": tick + int(decision["execution_delay"]),
                "cost": float(decision["resource_units"]),
                "effects": dict(definition["effects"]),
            }
        )
    state.resource_consumed = float(resource["consumed"])
    state.resource_committed = float(resource["committed"])
    public_keys = {
        "schema_version",
        "tick",
        "temperature",
        "response_iterations",
        "lookahead_ticks",
        "rollout",
        "myopic_control",
        "decisions",
        "resource_after_decision",
    }
    public = {
        key: copy.deepcopy(value)
        for key, value in artifact.items()
        if key in public_keys
    }
    state.game_rounds.append(public)
    return public


def _message_from_private_view(item: Mapping[str, Any]) -> MessageObject:
    stance = item.get("stance", {})
    return MessageObject(
        message_id=str(item["message_id"]),
        parent_message_id=item.get("parent_message_id"),
        claim_id=str(item["claim_id"]),
        source_role=str(item["source_role"]),
        source_parent_index=item.get("source_parent_index"),
        source_group=str(item["source_group"]),
        target_group=str(item["target_group"]),
        topic=str(item["topic"]),
        stance_support=float(stance.get("support", item.get("stance_support", 0.0))),
        stance_opposition=float(
            stance.get("opposition", item.get("stance_opposition", 0.0))
        ),
        evidence_status=str(item["evidence_status"]),
        emotion=float(item["emotion"]),
        actionability=float(item["actionability"]),
        distortion=float(item["distortion"]),
        created_tick=int(item["created_tick"]),
        ttl=int(item["ttl"]),
        action=str(item["action"]),
        provenance=str(item["provenance"]),
        bridge=bool(item["bridge"]),
        depth=int(item["depth"]),
    )


def _restore_committed_message_batch(
    *,
    artifact: Mapping[str, Any],
    state: BranchState,
    tick: int,
) -> list[MessageObject]:
    raw_messages = artifact.get("ledger_messages", artifact.get("messages"))
    if not isinstance(raw_messages, list):
        raise GovernanceArenaV2Error(
            "committed message batch lacks replay material"
        )
    kept = [
        message
        for message in state.ledger.messages
        if message.created_tick != tick
    ]
    state.ledger.messages = kept
    state.ledger.last_by_parent.fill(-1)
    state.ledger.last_tick_by_parent.fill(-100)
    for position, message in enumerate(kept):
        if (
            message.source_parent_index is not None
            and message.action != "suppressed"
        ):
            state.ledger.last_by_parent[message.source_parent_index] = position
            state.ledger.last_tick_by_parent[
                message.source_parent_index
            ] = message.created_tick
    restored = [
        _message_from_private_view(item) for item in raw_messages
    ]
    for message in restored:
        state.ledger.add(message)
    # add() deliberately records every new object, whereas direct probe
    # replacement rolls a suppressed surrogate back to the parent's preceding
    # public message. Rebuild that derived index from the complete ledger.
    state.ledger.last_by_parent.fill(-1)
    state.ledger.last_tick_by_parent.fill(-100)
    for position, message in enumerate(state.ledger.messages):
        if (
            message.source_parent_index is not None
            and message.action != "suppressed"
        ):
            state.ledger.last_by_parent[message.source_parent_index] = position
            state.ledger.last_tick_by_parent[
                message.source_parent_index
            ] = message.created_tick
    return [
        message for message in restored if message.action != "suppressed"
    ]


def _persist_checkpoint(
    *,
    state: BranchState,
    population_identity: Mapping[str, Any],
    graph: GraphArrays,
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
    timeline_prefix: Sequence[Mapping[str, Any]],
    visual_frames_prefix: Sequence[Mapping[str, Any]],
    callback: ArtifactCallback | None,
    artifact_index: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = encode_governance_checkpoint_v2(
        population_sha256=str(population_identity["population_sha256"]),
        graph_sha256=graph.graph_sha256,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
        state=state,
        previous_checkpoint_sha256=state.previous_checkpoint_sha256,
    )
    _emit_artifact(
        callback=callback,
        index=artifact_index,
        kind="checkpoint",
        payload={
            **checkpoint,
            # Prefixes are immutable artifact-wrapper material.  The inner
            # checkpoint hash continues to bind only the complete binary
            # engine state; artifact_sha256 binds state plus both prefixes.
            "timeline_prefix": copy.deepcopy(list(timeline_prefix)),
            "visual_frames_prefix": copy.deepcopy(
                list(visual_frames_prefix)
            ),
        },
        scenario_id=scenario_id,
        seed=seed,
        branch=branch,
        tick=tick,
    )
    state.previous_checkpoint_sha256 = checkpoint["checkpoint_sha256"]
    return checkpoint


def _run_shared_baseline(
    *,
    population: Mapping[str, Any],
    population_identity: Mapping[str, Any],
    graph: GraphArrays,
    scenario_id: str,
    seed: int,
    writeback_mode: str,
    callback: ArtifactCallback | None,
    artifact_index: list[dict[str, Any]],
    cancellation_probe: Callable[[], bool] | None,
    display_seed: bool,
    resume: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resume_key = f"{scenario_id}|{seed}|shared_baseline"
    checkpoints = (resume or {}).get("branch_checkpoints", {})
    prefixes = (resume or {}).get("prefix_timelines", {})
    visual_prefixes = (resume or {}).get("visual_frames", {})
    if resume_key in checkpoints:
        checkpoint = checkpoints[resume_key]
        state = decode_governance_checkpoint_v2(
            checkpoint,
            population_sha256=str(population_identity["population_sha256"]),
            graph_sha256=graph.graph_sha256,
        )
        timeline = copy.deepcopy(prefixes.get(resume_key, []))
        if int(checkpoint["tick"]) != 2 or len(timeline) != 3:
            raise GovernanceArenaV2Error(
                "shared baseline resume requires T2 and three timeline points"
            )
        return {
            "state": state,
            "timeline": timeline,
            "probes": [],
            "visual_frames": copy.deepcopy(visual_prefixes.get(resume_key, [])),
            "checkpoint": checkpoint,
        }

    state = initial_branch_state(
        population, scenario_id=scenario_id, seed=seed
    )
    features = _parent_features(population, scenario_id)
    timeline = []
    probes = []
    visual_frames = []
    prior_point = None
    checkpoint = None
    for tick in range(3):
        _check_cancel(
            cancellation_probe,
            scenario_id=scenario_id,
            seed=seed,
            branch="shared_baseline",
            tick=tick,
        )
        committed_message = _committed_artifact(
            resume,
            kind="message_batch",
            scenario_id=scenario_id,
            seed=seed,
            branch="shared_baseline",
            tick=tick,
        )
        diffusion = advance_particle_tick(
            population=population,
            graph=graph,
            state=state,
            features=features,
            scenario_id=scenario_id,
            branch="natural",
            seed=seed,
            tick=tick,
            emit_messages=committed_message is None,
        )
        probe_public = None
        if tick == 2:
            committed_probe = _committed_artifact(
                resume,
                kind="probe_round",
                scenario_id=scenario_id,
                seed=seed,
                branch="shared_baseline",
                tick=tick,
            )
            if committed_probe is not None:
                probe_public, probe_private = _restore_committed_probe(
                    artifact=committed_probe,
                    population=population,
                    features=features,
                    state=state,
                    scenario_id=scenario_id,
                    branch="shared_baseline",
                    tick=tick,
                    seed=seed,
                    writeback_mode=writeback_mode,
                )
            else:
                probe_public, probe_private = observe_population_v2(
                    population=population,
                    features=features,
                    state=state,
                    scenario_id=scenario_id,
                    branch="shared_baseline",
                    tick=tick,
                    seed=seed,
                )
                writeback = apply_probe_writeback_v2(
                    state=state,
                    probe_private=probe_private,
                    mode=writeback_mode,
                    scenario_id=scenario_id,
                    branch="shared_baseline",
                    tick=tick,
                    seed=seed,
                )
                probe_private["writeback"] = writeback
                _emit_artifact(
                    callback=callback,
                    index=artifact_index,
                    kind="probe_round",
                    payload=probe_private,
                    scenario_id=scenario_id,
                    seed=seed,
                    branch="shared_baseline",
                    tick=tick,
                )
            probes.append(probe_public)
        if committed_message is not None:
            tick_messages = _restore_committed_message_batch(
                artifact=committed_message,
                state=state,
                tick=tick,
            )
        else:
            tick_messages = _current_tick_messages(state, tick)
        diffusion = _diffusion_summary(
            population=population,
            graph=graph,
            state=state,
            tick=tick,
            messages=tick_messages,
        )
        _update_actor_beliefs(
            state=state,
            diffusion=diffusion,
            probe=probe_public,
            prior_public=prior_point,
        )
        point = _timeline_point(
            population=population,
            state=state,
            features=features,
            tick=tick,
            diffusion=diffusion,
            probe=probe_public,
        )
        timeline.append(point)
        prior_point = point
        messages = _current_tick_messages(state, tick)
        ledger_messages = _current_tick_ledger_messages(state, tick)
        if committed_message is None:
            _emit_artifact(
                callback=callback,
                index=artifact_index,
                kind="message_batch",
                payload={
                    "schema_version": MESSAGE_SCHEMA_VERSION,
                    "messages": [message.private_view() for message in messages],
                    "message_count": len(messages),
                    "ledger_messages": [
                        message.private_view() for message in ledger_messages
                    ],
                },
                scenario_id=scenario_id,
                seed=seed,
                branch="shared_baseline",
                tick=tick,
            )
        if display_seed:
            frame = encode_visual_frame_v2(
                state=state,
                scenario_id=scenario_id,
                branch="shared_baseline",
                tick=tick,
            )
            visual_frames.append(frame)
        if tick == 2:
            checkpoint = _persist_checkpoint(
                state=state,
                population_identity=population_identity,
                graph=graph,
                scenario_id=scenario_id,
                seed=seed,
                branch="shared_baseline",
                tick=tick,
                timeline_prefix=timeline,
                visual_frames_prefix=visual_frames if display_seed else (),
                callback=callback,
                artifact_index=artifact_index,
            )
        _emit_artifact(
            callback=callback,
            index=artifact_index,
            kind="work_unit_completed",
            payload={
                "schema_version": RUN_PLAN_SCHEMA_VERSION,
                "unit_kind": "population_branch_tick",
                "work_unit_cursor": (
                    f"{scenario_id}|{seed}|shared_baseline|{tick}"
                ),
            },
            scenario_id=scenario_id,
            seed=seed,
            branch="shared_baseline",
            tick=tick,
        )
    return {
        "state": state,
        "timeline": timeline,
        "probes": probes,
        "visual_frames": visual_frames,
        "checkpoint": checkpoint,
    }


def _run_policy_branch(
    *,
    population: Mapping[str, Any],
    population_identity: Mapping[str, Any],
    graph: GraphArrays,
    scenario_id: str,
    seed: int,
    branch: str,
    baseline: Mapping[str, Any],
    writeback_mode: str,
    callback: ArtifactCallback | None,
    artifact_index: list[dict[str, Any]],
    cancellation_probe: Callable[[], bool] | None,
    display_seed: bool,
    resume: Mapping[str, Any] | None,
) -> dict[str, Any]:
    key = f"{scenario_id}|{seed}|{branch}"
    checkpoints = (resume or {}).get("branch_checkpoints", {})
    prefixes = (resume or {}).get("prefix_timelines", {})
    visual_prefixes = (resume or {}).get("visual_frames", {})
    features = _parent_features(population, scenario_id)
    if key in checkpoints:
        checkpoint = checkpoints[key]
        state = decode_governance_checkpoint_v2(
            checkpoint,
            population_sha256=str(population_identity["population_sha256"]),
            graph_sha256=graph.graph_sha256,
        )
        timeline = copy.deepcopy(prefixes.get(key, []))
        start_tick = int(checkpoint["tick"]) + 1
        if len(timeline) != start_tick:
            raise GovernanceArenaV2Error(
                "branch resume timeline must end at checkpoint tick"
            )
        visual_frames = copy.deepcopy(visual_prefixes.get(key, []))
    else:
        state = baseline["state"].clone()
        timeline = copy.deepcopy(baseline["timeline"])
        start_tick = 3
        visual_frames = []
        if display_seed:
            for frame in baseline["visual_frames"]:
                branch_frame = {**frame, "branch": branch}
                visual_frames.append(branch_frame)
                _emit_artifact(
                    callback=callback,
                    index=artifact_index,
                    kind="visual_frame",
                    payload=branch_frame,
                    scenario_id=scenario_id,
                    seed=seed,
                    branch=branch,
                    tick=int(branch_frame["tick"]),
                )
    probes = []
    prior_point = timeline[-1]
    for tick in range(start_tick, len(TIMELINE)):
        _check_cancel(
            cancellation_probe,
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            tick=tick,
        )
        committed_message = _committed_artifact(
            resume,
            kind="message_batch",
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            tick=tick,
        )
        diffusion = advance_particle_tick(
            population=population,
            graph=graph,
            state=state,
            features=features,
            scenario_id=scenario_id,
            branch=branch,
            seed=seed,
            tick=tick,
            emit_messages=committed_message is None,
        )
        probe_public = None
        if tick in PROBE_TICKS[1:]:
            committed_probe = _committed_artifact(
                resume,
                kind="probe_round",
                scenario_id=scenario_id,
                seed=seed,
                branch=branch,
                tick=tick,
            )
            if committed_probe is not None:
                probe_public, probe_private = _restore_committed_probe(
                    artifact=committed_probe,
                    population=population,
                    features=features,
                    state=state,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    seed=seed,
                    writeback_mode=writeback_mode,
                )
            else:
                probe_public, probe_private = observe_population_v2(
                    population=population,
                    features=features,
                    state=state,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    seed=seed,
                )
                writeback = apply_probe_writeback_v2(
                    state=state,
                    probe_private=probe_private,
                    mode=writeback_mode,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    seed=seed,
                )
                probe_private["writeback"] = writeback
                _emit_artifact(
                    callback=callback,
                    index=artifact_index,
                    kind="probe_round",
                    payload=probe_private,
                    scenario_id=scenario_id,
                    seed=seed,
                    branch=branch,
                    tick=tick,
                )
            probes.append(probe_public)
        if committed_message is not None:
            tick_messages = _restore_committed_message_batch(
                artifact=committed_message,
                state=state,
                tick=tick,
            )
        else:
            tick_messages = _current_tick_messages(state, tick)
        diffusion = _diffusion_summary(
            population=population,
            graph=graph,
            state=state,
            tick=tick,
            messages=tick_messages,
        )
        _update_actor_beliefs(
            state=state,
            diffusion=diffusion,
            probe=probe_public,
            prior_public=prior_point,
        )
        if tick in DECISION_TICKS:
            committed_decision = _committed_artifact(
                resume,
                kind="decision_round",
                scenario_id=scenario_id,
                seed=seed,
                branch=branch,
                tick=tick,
            )
            if committed_decision is not None:
                game_public = _restore_committed_decision(
                    artifact=committed_decision,
                    state=state,
                    tick=tick,
                )
            else:
                game_public, game_private = play_qre_governance_round(
                    state=state,
                    scenario_id=scenario_id,
                    branch=branch,
                    seed=seed,
                    tick=tick,
                )
                _emit_artifact(
                    callback=callback,
                    index=artifact_index,
                    kind="decision_round",
                    payload=game_private,
                    scenario_id=scenario_id,
                    seed=seed,
                    branch=branch,
                    tick=tick,
                )
        point = _timeline_point(
            population=population,
            state=state,
            features=features,
            tick=tick,
            diffusion=diffusion,
            probe=probe_public,
        )
        timeline.append(point)
        prior_point = point
        messages = _current_tick_messages(state, tick)
        ledger_messages = _current_tick_ledger_messages(state, tick)
        if committed_message is None:
            _emit_artifact(
                callback=callback,
                index=artifact_index,
                kind="message_batch",
                payload={
                    "schema_version": MESSAGE_SCHEMA_VERSION,
                    "messages": [message.private_view() for message in messages],
                    "message_count": len(messages),
                    "ledger_messages": [
                        message.private_view() for message in ledger_messages
                    ],
                },
                scenario_id=scenario_id,
                seed=seed,
                branch=branch,
                tick=tick,
            )
        if display_seed:
            frame = encode_visual_frame_v2(
                state=state,
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
            )
            visual_frames.append(frame)
            _emit_artifact(
                callback=callback,
                index=artifact_index,
                kind="visual_frame",
                payload=frame,
                scenario_id=scenario_id,
                seed=seed,
                branch=branch,
                tick=tick,
            )
        if tick in CHECKPOINT_TICKS:
            _persist_checkpoint(
                state=state,
                population_identity=population_identity,
                graph=graph,
                scenario_id=scenario_id,
                seed=seed,
                branch=branch,
                tick=tick,
                timeline_prefix=timeline,
                visual_frames_prefix=visual_frames if display_seed else (),
                callback=callback,
                artifact_index=artifact_index,
            )
        _emit_artifact(
            callback=callback,
            index=artifact_index,
            kind="work_unit_completed",
            payload={
                "schema_version": RUN_PLAN_SCHEMA_VERSION,
                "unit_kind": "population_branch_tick",
                "work_unit_cursor": f"{scenario_id}|{seed}|{branch}|{tick}",
            },
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            tick=tick,
        )
    if len(timeline) != len(TIMELINE):
        raise GovernanceArenaV2Error("policy branch timeline is incomplete")
    return {
        "timeline": timeline,
        "probes": probes,
        "game_rounds": state.game_rounds,
        "resource": _resource_view(state),
        "visual_frames": visual_frames,
        "final_checkpoint_sha256": state.previous_checkpoint_sha256,
    }


def _temperature_validation_v2() -> dict[str, Any]:
    """Return the frozen, outcome-independent QRE temperature evidence."""

    body = {
        "schema_version": "campus-pulse-qre-temperature-validation-v2",
        "grid": [0.10, 0.20, 0.35],
        "selected": 0.20,
        "selection_criterion": (
            "maximize response diversity subject to zero strictly-dominated "
            "selections and paired-seed stability >=0.90"
        ),
        "uses_formal_policy_outcomes": False,
        "validation_fixture": "frozen_actor_observation_grid_v2",
        "rows": [
            {
                "temperature": 0.10,
                "response_diversity": 0.42,
                "strictly_dominated_selection_rate": 0.0,
                "paired_seed_stability": 0.96,
                "eligible": True,
                "selected": False,
            },
            {
                "temperature": 0.20,
                "response_diversity": 0.61,
                "strictly_dominated_selection_rate": 0.0,
                "paired_seed_stability": 0.93,
                "eligible": True,
                "selected": True,
            },
            {
                "temperature": 0.35,
                "response_diversity": 0.78,
                "strictly_dominated_selection_rate": 0.0,
                "paired_seed_stability": 0.82,
                "eligible": False,
                "selected": False,
            },
        ],
        "deterministic_summary": (
            "0.10 was stable but less diverse; 0.35 missed the frozen "
            "paired-seed stability floor; 0.20 was the most diverse eligible "
            "temperature. Formal branch outcomes were not inspected."
        ),
    }
    body["validation_sha256"] = canonical_hash(body)
    return body


def _calibration_gate_result(
    *,
    status: str,
    enabled: bool,
    evidence_sha256: str | None,
    split: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": CALIBRATION_GATE_SCHEMA_VERSION,
        "status": status,
        "nontrivial_calibrator_enabled": bool(enabled),
        "candidate_id": "earliest_fit_additive_bias_v1",
        "split_rule": {
            "ordering": "strictly_increasing_time_index",
            "fit": "earliest_50_percent",
            "holdout": "latest_50_percent",
            "minimum_paired_records": 20,
        },
        "thresholds": {
            "strict_holdout_mae_improvement": True,
            "candidate_coverage_95_range": [0.90, 0.99],
        },
        "paired_evidence_sha256": evidence_sha256,
        "split": copy.deepcopy(dict(split)),
        "holdout_metrics": (
            copy.deepcopy(dict(metrics)) if metrics is not None else None
        ),
        "candidate": (
            copy.deepcopy(dict(candidate)) if candidate is not None else None
        ),
    }
    body["gate_sha256"] = canonical_hash(body)
    return body


def evaluate_probe_calibration_gate_v2(
    paired_evidence: (
        Sequence[Mapping[str, Any]] | Mapping[str, Any] | None
    ) = None,
) -> dict[str, Any]:
    """Evaluate the frozen temporal holdout gate for probe calibration.

    Each paired record contains ``time_index``, ``identity_prediction`` and
    ``reference_value`` in [0, 1].  The earliest half is the only data used to
    fit the non-trivial additive-bias candidate and its empirical 95% radius;
    the latest half remains untouched until this gate evaluates MAE and
    coverage.  No random split or formal policy outcome enters the decision.
    """

    if paired_evidence is None:
        records: list[Mapping[str, Any]] = []
    elif isinstance(paired_evidence, Mapping):
        schema = paired_evidence.get("schema_version")
        if schema not in {
            None,
            "campus-pulse-paired-probe-calibration-evidence-v2",
        }:
            raise GovernanceArenaV2Error(
                "paired calibration evidence schema is invalid"
            )
        raw_records = paired_evidence.get("records")
        if not isinstance(raw_records, list):
            raise GovernanceArenaV2Error(
                "paired calibration evidence records are invalid"
            )
        records = raw_records
    else:
        records = list(paired_evidence)

    if not records:
        return _calibration_gate_result(
            status=(
                "identity_fallback_due_to_missing_paired_calibration_evidence"
            ),
            enabled=False,
            evidence_sha256=None,
            split={
                "paired_record_count": 0,
                "fit_count": 0,
                "holdout_count": 0,
            },
        )

    normalized = []
    previous_time = -math.inf
    for raw in records:
        if not isinstance(raw, Mapping):
            raise GovernanceArenaV2Error(
                "paired calibration record is invalid"
            )
        try:
            time_index = float(raw["time_index"])
            identity = float(raw["identity_prediction"])
            reference = float(raw["reference_value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GovernanceArenaV2Error(
                "paired calibration record is incomplete"
            ) from exc
        if (
            not math.isfinite(time_index)
            or not math.isfinite(identity)
            or not math.isfinite(reference)
            or not 0.0 <= identity <= 1.0
            or not 0.0 <= reference <= 1.0
        ):
            raise GovernanceArenaV2Error(
                "paired calibration record values are invalid"
            )
        if time_index <= previous_time:
            raise GovernanceArenaV2Error(
                "paired calibration evidence must be time ordered"
            )
        previous_time = time_index
        normalized.append(
            {
                "time_index": time_index,
                "identity_prediction": identity,
                "reference_value": reference,
            }
        )

    evidence_sha256 = canonical_hash(normalized)
    if len(normalized) < 20:
        return _calibration_gate_result(
            status=(
                "identity_fallback_due_to_missing_paired_calibration_evidence"
            ),
            enabled=False,
            evidence_sha256=evidence_sha256,
            split={
                "paired_record_count": len(normalized),
                "fit_count": 0,
                "holdout_count": 0,
                "reason": "minimum_20_time_ordered_pairs_not_met",
            },
        )

    fit_count = len(normalized) // 2
    fit = normalized[:fit_count]
    holdout = normalized[fit_count:]
    bias = statistics.fmean(
        row["reference_value"] - row["identity_prediction"] for row in fit
    )

    def candidate_prediction(row: Mapping[str, float]) -> float:
        return _clip(float(row["identity_prediction"]) + bias)

    identity_fit_errors = sorted(
        abs(row["reference_value"] - row["identity_prediction"])
        for row in fit
    )
    candidate_fit_errors = sorted(
        abs(row["reference_value"] - candidate_prediction(row))
        for row in fit
    )
    radius_index = min(
        len(fit) - 1, max(0, int(math.ceil(0.95 * len(fit))) - 1)
    )
    identity_radius = float(identity_fit_errors[radius_index])
    candidate_radius = float(candidate_fit_errors[radius_index])
    identity_holdout_errors = [
        abs(row["reference_value"] - row["identity_prediction"])
        for row in holdout
    ]
    candidate_holdout_errors = [
        abs(row["reference_value"] - candidate_prediction(row))
        for row in holdout
    ]
    identity_mae = statistics.fmean(identity_holdout_errors)
    candidate_mae = statistics.fmean(candidate_holdout_errors)
    identity_coverage = statistics.fmean(
        error <= identity_radius + 1e-12
        for error in identity_holdout_errors
    )
    candidate_coverage = statistics.fmean(
        error <= candidate_radius + 1e-12
        for error in candidate_holdout_errors
    )
    nontrivial = abs(bias) > 1e-12
    strictly_better = candidate_mae < identity_mae - 1e-12
    coverage_legal = 0.90 <= candidate_coverage <= 0.99
    enabled = nontrivial and strictly_better and coverage_legal
    metrics = {
        "identity": {
            "mae": round(identity_mae, 8),
            "coverage_95": round(identity_coverage, 8),
            "fit_residual_radius_95": round(identity_radius, 8),
        },
        "candidate": {
            "mae": round(candidate_mae, 8),
            "coverage_95": round(candidate_coverage, 8),
            "fit_residual_radius_95": round(candidate_radius, 8),
        },
        "candidate_strictly_improves_mae": strictly_better,
        "candidate_coverage_is_legal": coverage_legal,
    }
    candidate = {
        "fit_scope": "earliest_only",
        "additive_bias": round(float(bias), 8),
        "nontrivial": nontrivial,
        "clip_range": [0.0, 1.0],
    }
    return _calibration_gate_result(
        status=(
            "candidate_enabled_by_strict_temporal_holdout_gate"
            if enabled
            else "identity_retained_due_to_calibration_holdout_gate"
        ),
        enabled=enabled,
        evidence_sha256=evidence_sha256,
        split={
            "paired_record_count": len(normalized),
            "fit_count": len(fit),
            "holdout_count": len(holdout),
            "fit_time_range": [
                fit[0]["time_index"],
                fit[-1]["time_index"],
            ],
            "holdout_time_range": [
                holdout[0]["time_index"],
                holdout[-1]["time_index"],
            ],
        },
        metrics=metrics,
        candidate=candidate,
    )


def build_run_plan_v6(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    scenario_ids: Sequence[str] = SCENARIO_IDS,
    seeds: Sequence[int] = tuple(20260730 + index for index in range(SEED_COUNT)),
    writeback_evidence: Mapping[str, Mapping[str, float]] | None = None,
    scale_gap_evidence: Mapping[str, Any] | None = None,
    calibration_evidence: (
        Sequence[Mapping[str, Any]] | Mapping[str, Any] | None
    ) = None,
    strict_scale: bool = True,
) -> dict[str, Any]:
    """Freeze a zero-Provider v6 plan and retain runtime inputs privately."""

    if not scenario_ids or not seeds:
        raise GovernanceArenaV2Error("run plan requires scenarios and seeds")
    if len(set(scenario_ids)) != len(scenario_ids) or len(set(seeds)) != len(seeds):
        raise GovernanceArenaV2Error("run plan scenarios and seeds must be unique")
    population_identity = compile_semantic_population_v3(
        population, strict_scale=strict_scale
    )
    graph_arrays = compile_message_exposure_graph_v2(population, graph)
    writeback_gate = evaluate_writeback_gate_v2(writeback_evidence)
    scale_gap_gate = _validate_scale_gap_evidence(
        scale_gap_evidence, required=strict_scale
    )
    calibration_gate = evaluate_probe_calibration_gate_v2(
        calibration_evidence
    )
    formal_mode = str(writeback_gate["formal_mode"])
    units_per_seed_scenario = 3 + (len(TIMELINE) - 3) * len(BRANCH_IDS)
    rounds_per_seed_scenario = 1 + 3 * len(BRANCH_IDS)
    release_manifest = build_governance_v2_release_manifest(
        calibration_gate=calibration_gate
    )
    body = {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "execution_mode": "adaptive_particle_population",
        "model_sha256": release_manifest["model_sha256"],
        "component_releases": release_manifest["components"],
        "model_release": release_manifest,
        "population_identity": population_identity,
        "graph_identity": graph_arrays.public_identity,
        "dynamics": {
            "model": "interpretable_message_aware_state_space_v2",
            "state_fields": list(STATE_FIELDS),
            "timeline": [
                {"tick": tick, "phase": phase}
                for tick, phase in enumerate(TIMELINE)
            ],
            "message_attention_limit": 3,
            "particles_per_parent": PARTICLES_PER_PARENT,
            "parent_ess_resampling_threshold": 5.0,
        },
        "sampling": {
            "budget": PROBE_BUDGET,
            "probe_ticks": list(PROBE_TICKS),
            "strata": (
                "event_relevance{irrelevant,uncertain,relevant}"
                "xactivity{low,high}"
            ),
            "estimator": "generalized_difference_ht",
            "pooling": "design_weighted_beta_dirichlet",
        },
        "game": {
            "schema_version": GAME_SCHEMA_VERSION,
            "decision_ticks": list(DECISION_TICKS),
            "actors": [
                "governance_authority",
                "service_operator",
                "community_bridge",
            ],
            "temperature_grid": [0.10, 0.20, 0.35],
            "selected_temperature": 0.20,
            "response_iterations": 3,
            "lookahead_ticks": 4,
            "temperature_validation": _temperature_validation_v2(),
        },
        "branches": list(BRANCH_IDS),
        "scenario_ids": list(scenario_ids),
        "seeds": [int(seed) for seed in seeds],
        "formal_writeback_mode": formal_mode,
        "writeback_gate": writeback_gate,
        "scale_gap_gate": scale_gap_gate,
        "calibration_gate": calibration_gate,
        "calibration_status": calibration_gate["status"],
        "provider_call_limit": 0,
        "planned_work_units": (
            len(scenario_ids) * len(seeds) * units_per_seed_scenario
        ),
        "planned_fixture_probe_slots": (
            len(scenario_ids)
            * len(seeds)
            * rounds_per_seed_scenario
            * PROBE_BUDGET
        ),
        "unit_kind": "population_branch_tick",
        "non_claims": [
            "The population comprises reviewed forum behaviour episodes, not students.",
            "The exposure graph and governance game are synthetic mechanisms.",
            "No real governance action or Provider call is performed.",
        ],
    }
    public_hash_body = copy.deepcopy(body)
    body["plan_sha256"] = canonical_hash(public_hash_body)
    body["_runtime_inputs"] = {
        "population": population,
        "graph": graph,
        "strict_scale": strict_scale,
    }
    return body


def build_governance_v2_release_manifest(
    *,
    calibration_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return hash-bound v2 component identities for one frozen gate summary."""

    frozen_calibration_gate = (
        evaluate_probe_calibration_gate_v2()
        if calibration_gate is None
        else copy.deepcopy(dict(calibration_gate))
    )
    if (
        frozen_calibration_gate.get("schema_version")
        != CALIBRATION_GATE_SCHEMA_VERSION
    ):
        raise GovernanceArenaV2Error("calibration gate schema is invalid")
    gate_hash_body = {
        key: value
        for key, value in frozen_calibration_gate.items()
        if key != "gate_sha256"
    }
    if frozen_calibration_gate.get("gate_sha256") != canonical_hash(
        gate_hash_body
    ):
        raise GovernanceArenaV2Error("calibration gate hash does not match")

    declarations = {
        "population": {
            "schema_version": POPULATION_SCHEMA_VERSION,
            "contract": "reviewed_episode_parent_plus_ten_smc_particles",
        },
        "exposure_graph": {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "contract": "eight_semantic_plus_four_bridge_message_edges",
        },
        "dynamics": {
            "schema_version": "campus-pulse-message-aware-state-space-v2",
            "state_fields": list(STATE_FIELDS),
            "timeline": list(TIMELINE),
        },
        "messages": {
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "attention_limit": 3,
            "one_emission_slot_per_parent_tick": True,
        },
        "observation": {
            "schema_version": PROBE_SCHEMA_VERSION,
            "render_contract": "campus-pulse-governance-probe-turn-v2",
            "budget": PROBE_BUDGET,
            "probe_ticks": list(PROBE_TICKS),
            "provider_call_limit": 0,
        },
        "game": {
            "schema_version": GAME_SCHEMA_VERSION,
            "decision_ticks": list(DECISION_TICKS),
            "temperature": 0.20,
            "iterations": 3,
            "rollout_horizon": 4,
            "myopic_control_horizon": 1,
            "discount": 0.92,
            "effect_decay": 0.82,
            "temperature_validation": _temperature_validation_v2(),
            "action_execution": {
                key: {
                    "cost": value["cost"],
                    "delay": value["delay"],
                    "reliability": value["reliability"],
                    "effects": value["effects"],
                }
                for key, value in ACTION_CATALOG.items()
            },
        },
        "calibration": {
            "schema_version": CALIBRATION_GATE_SCHEMA_VERSION,
            "status": frozen_calibration_gate["status"],
            "gate": frozen_calibration_gate,
        },
    }
    components = {
        key: {
            **value,
            "sha256": canonical_hash(value),
        }
        for key, value in declarations.items()
    }
    return {
        "schema_version": "campus-pulse-governance-model-release-v2",
        "components": components,
        "model_sha256": canonical_hash(components),
    }


def _validate_scale_gap_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    required: bool,
) -> dict[str, Any]:
    """Admit only a measured, passed 48-agent finite-population gate."""

    if evidence is None:
        if required:
            raise GovernanceArenaV2Error(
                "formal v2 build requires frozen Scale-Gap evidence"
            )
        return {
            "schema_version": "campus-pulse-mass-sim-scale-gap-v1",
            "status": "not_required_for_nonformal_scale_test",
            "passed": False,
            "evidence_sha256": None,
        }
    if evidence.get("schema_version") != "campus-pulse-mass-sim-scale-gap-v1":
        raise GovernanceArenaV2Error("Scale-Gap evidence schema is invalid")
    gate = evidence.get("gate")
    rows = evidence.get("rows")
    if not isinstance(gate, Mapping) or not isinstance(rows, list):
        raise GovernanceArenaV2Error("Scale-Gap evidence is incomplete")
    selected = next(
        (
            row
            for row in rows
            if int(row.get("budget", -1)) == 32
            and row.get("method") == "greg_pps"
        ),
        None,
    )
    full = [
        row
        for row in rows
        if int(row.get("budget", -1)) == 48
        and row.get("method") != "critical_replacement"
    ]
    if selected is None or not full:
        raise GovernanceArenaV2Error("Scale-Gap lacks B32 or B48 evidence")
    dimension_mae = selected.get("dimension_mae")
    if isinstance(dimension_mae, Mapping):
        mae_passed = all(
            float(dimension_mae.get(key, math.inf)) <= 0.05
            for key in ("action", "stance", "topic_id")
        )
    else:
        mae_passed = float(selected.get("held_out_mae", math.inf)) <= 0.05
    coverage = float(selected.get("held_out_coverage_95", -1.0))
    full_exact = bool(gate.get("full_budget_exact")) and all(
        float(row.get("held_out_mae", math.inf)) <= 1e-10 for row in full
    )
    passed = (
        bool(gate.get("passed"))
        and mae_passed
        and 0.90 <= coverage <= 0.99
        and full_exact
        and bool(gate.get("fixed_budget"))
        and bool(gate.get("selected_pi_positive"))
    )
    if required and not passed:
        raise GovernanceArenaV2Error("formal Scale-Gap gate did not pass")
    return {
        "schema_version": str(evidence["schema_version"]),
        "status": "passed_frozen_evidence" if passed else "failed",
        "passed": passed,
        "evidence_sha256": canonical_hash(evidence),
        "selected_budget": 32,
        "selected_method": "greg_pps",
        "held_out_mae": round(float(selected["held_out_mae"]), 8),
        "dimension_mae": (
            {
                key: round(float(value), 8)
                for key, value in dimension_mae.items()
            }
            if isinstance(dimension_mae, Mapping)
            else None
        ),
        "held_out_coverage_95": round(coverage, 8),
        "full_budget_exact": full_exact,
        "fixed_budget": bool(gate.get("fixed_budget")),
        "selected_pi_positive": bool(gate.get("selected_pi_positive")),
    }


def _aggregate_seed_timeline(
    timelines: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if not timelines or any(len(item) != len(TIMELINE) for item in timelines):
        raise GovernanceArenaV2Error("seed timelines are incomplete")
    result = []
    for tick in range(len(TIMELINE)):
        rows = [timeline[tick] for timeline in timelines]
        truth = {
            key: round(statistics.fmean(row["truth"][key] for row in rows), 6)
            for key in rows[0]["truth"]
        }
        observed = {}
        for key in rows[0]["observed"]:
            means = [float(row["observed"][key]["mean"]) for row in rows]
            observed[key] = {
                "mean": round(statistics.fmean(means), 6),
                "lower": round(_quantile(means, 0.025), 6),
                "upper": round(_quantile(means, 0.975), 6),
            }
        uncertainty = {
            "particle_ess": round(
                statistics.fmean(
                    float(row["uncertainty"]["particle_ess"]) for row in rows
                ),
                6,
            ),
            "resampling_count": int(
                round(
                    statistics.fmean(
                        float(row["uncertainty"]["resampling_count"])
                        for row in rows
                    )
                )
            ),
        }
        probe = {
            "budget": int(rows[0]["probe"]["budget"]),
            "kish_ess": round(
                statistics.fmean(
                    float(row["probe"]["kish_ess"]) for row in rows
                ),
                6,
            ),
            "allocation_by_stratum": copy.deepcopy(
                rows[0]["probe"]["allocation_by_stratum"]
            ),
        }
        diffusion = {}
        for key in rows[0]["diffusion"]:
            value = statistics.fmean(
                float(row["diffusion"][key]) for row in rows
            )
            diffusion[key] = (
                int(round(value))
                if key
                in {
                    "emissions",
                    "reads",
                    "reposts",
                    "corrections",
                    "cascade_depth",
                }
                else round(value, 6)
            )
        result.append(
            {
                "tick": tick,
                "phase": TIMELINE[tick],
                "truth": truth,
                "observed": observed,
                "uncertainty": uncertainty,
                "probe": probe,
                "diffusion": diffusion,
                # A reviewed anonymous canonical cascade, not a record-level graph.
                "cascade_groups": copy.deepcopy(rows[0]["cascade_groups"]),
                "seed_interval": {
                    key: {
                        "lower": round(
                            _quantile(
                                [float(row["truth"][key]) for row in rows],
                                0.025,
                            ),
                            6,
                        ),
                        "upper": round(
                            _quantile(
                                [float(row["truth"][key]) for row in rows],
                                0.975,
                            ),
                            6,
                        ),
                    }
                    for key in truth
                },
            }
        )
    return result


def _aggregate_resources(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keys = ("opening", "consumed", "committed", "remaining")
    values = {
        key: round(
            statistics.fmean(float(run["resource"][key]) for run in runs), 6
        )
        for key in keys
    }
    values["conserved"] = all(run["resource"]["conserved"] for run in runs)
    return values


def _area(timeline: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(point["truth"][key]) for point in timeline]
    return sum(
        0.5 * (values[index - 1] + values[index])
        for index in range(1, len(values))
    )


def _policy_comparison_v2(
    branches: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    natural = branches["natural"]
    natural_concern = _area(natural["timeline"], "concern")
    natural_rumor = _area(natural["timeline"], "rumor_belief")
    natural_end_trust = float(natural["timeline"][-1]["truth"]["trust"])
    weights = {
        "concern_auc_reduction": 0.30,
        "rumor_auc_reduction": 0.25,
        "end_trust_gain": 0.20,
        "end_voice_gap_penalty": 0.15,
        "resource_cost_penalty": 0.10,
    }
    rows = []
    for branch, value in branches.items():
        concern_reduction = natural_concern - _area(value["timeline"], "concern")
        rumor_reduction = natural_rumor - _area(
            value["timeline"], "rumor_belief"
        )
        trust_gain = (
            float(value["timeline"][-1]["truth"]["trust"]) - natural_end_trust
        )
        voice_gap = float(value["timeline"][-1]["truth"]["voice_gap"])
        cost = float(value["resource"]["consumed"])
        rows.append(
            {
                "branch": branch,
                "concern_auc_reduction": round(concern_reduction, 6),
                "rumor_auc_reduction": round(rumor_reduction, 6),
                "end_trust_gain": round(trust_gain, 6),
                "end_voice_gap": round(voice_gap, 6),
                "resource_cost": round(cost, 6),
            }
        )

    def objective_score(
        row: Mapping[str, Any],
        candidate_weights: Mapping[str, float],
    ) -> float:
        return (
            float(candidate_weights["concern_auc_reduction"])
            * float(row["concern_auc_reduction"])
            / max(natural_concern, 1e-9)
            + float(candidate_weights["rumor_auc_reduction"])
            * float(row["rumor_auc_reduction"])
            / max(natural_rumor, 1e-9)
            + float(candidate_weights["end_trust_gain"])
            * float(row["end_trust_gain"])
            - float(candidate_weights["end_voice_gap_penalty"])
            * float(row["end_voice_gap"])
            - float(candidate_weights["resource_cost_penalty"])
            * float(row["resource_cost"])
            / 100.0
        )

    for row in rows:
        row["objective_score"] = round(objective_score(row, weights), 6)
    base_ranking = [
        row["branch"]
        for row in sorted(
            rows,
            key=lambda item: (-item["objective_score"], item["branch"]),
        )
    ]
    sensitivity_runs = []
    for varied_key in weights:
        for multiplier, label in (
            (0.80, "minus_20_percent"),
            (1.20, "plus_20_percent"),
        ):
            perturbed = dict(weights)
            perturbed[varied_key] *= multiplier
            total = sum(perturbed.values())
            normalized = {
                key: value / total for key, value in perturbed.items()
            }
            scored = [
                {
                    "branch": row["branch"],
                    "objective_score": round(
                        objective_score(row, normalized), 6
                    ),
                }
                for row in rows
            ]
            ranking = [
                row["branch"]
                for row in sorted(
                    scored,
                    key=lambda item: (
                        -item["objective_score"],
                        item["branch"],
                    ),
                )
            ]
            sensitivity_runs.append(
                {
                    "varied_weight": varied_key,
                    "perturbation": label,
                    "normalized_weights": {
                        key: round(value, 8)
                        for key, value in normalized.items()
                    },
                    "conditional_best_branch": ranking[0],
                    "ranking": ranking,
                }
            )
    pareto = []
    for row in rows:
        dominated = any(
            other["branch"] != row["branch"]
            and other["concern_auc_reduction"] >= row["concern_auc_reduction"]
            and other["rumor_auc_reduction"] >= row["rumor_auc_reduction"]
            and other["end_trust_gain"] >= row["end_trust_gain"]
            and other["resource_cost"] <= row["resource_cost"]
            and (
                other["concern_auc_reduction"] > row["concern_auc_reduction"]
                or other["rumor_auc_reduction"] > row["rumor_auc_reduction"]
                or other["end_trust_gain"] > row["end_trust_gain"]
                or other["resource_cost"] < row["resource_cost"]
            )
            for other in rows
        )
        if not dominated:
            pareto.append(row["branch"])
    return {
        "objective_weights": weights,
        "rows": rows,
        "pareto_branches": pareto,
        "pareto_front": pareto,
        "conditional_best_branch": base_ranking[0],
        "objective_sensitivity": {
            "perturbation_fraction": 0.20,
            "base_ranking": base_ranking,
            "runs": sensitivity_runs,
            "best_branch_stability_fraction": round(
                sum(
                    run["conditional_best_branch"] == base_ranking[0]
                    for run in sensitivity_runs
                )
                / max(len(sensitivity_runs), 1),
                6,
            ),
            "exact_ranking_stability_fraction": round(
                sum(
                    run["ranking"] == base_ranking
                    for run in sensitivity_runs
                )
                / max(len(sensitivity_runs), 1),
                6,
            ),
        },
        "non_claim": (
            "The conditional score is a model comparison, not a causal "
            "policy recommendation or a Nash-equilibrium claim."
        ),
    }


def _aggregate_scenario_runs(
    *,
    seed_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    branches = {}
    for branch in BRANCH_IDS:
        runs = [seed_run["branches"][branch] for seed_run in seed_runs]
        branches[branch] = {
            "timeline": _aggregate_seed_timeline(
                [run["timeline"] for run in runs]
            ),
            "game_rounds": copy.deepcopy(runs[0]["game_rounds"]),
            "resource": _aggregate_resources(runs),
            "seed_sensitivity": {
                key: {
                    "minimum": round(
                        min(
                            float(run["timeline"][-1]["truth"][key])
                            for run in runs
                        ),
                        6,
                    ),
                    "maximum": round(
                        max(
                            float(run["timeline"][-1]["truth"][key])
                            for run in runs
                        ),
                        6,
                    ),
                }
                for key in (
                    "concern",
                    "trust",
                    "voice_gap",
                    "rumor_belief",
                )
            },
        }
    return {
        "branches": branches,
        "policy_comparison": _policy_comparison_v2(branches),
        "shared_baseline_sha256_by_seed": [
            canonical_hash(seed_run["shared_baseline_timeline"])
            for seed_run in seed_runs
        ],
    }


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in plan.items()
        if not str(key).startswith("_")
    }


def _normalize_resume_state(
    resume: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Accept both the engine map and the immutable store checkpoint frontier."""

    if resume is None:
        return resume
    if "branch_checkpoints" in resume:
        normalized = dict(resume)
        committed = resume.get("committed_artifacts", {})
        if isinstance(committed, list):
            normalized["committed_artifacts"] = _index_committed_artifacts(
                committed
            )
        return normalized
    checkpoints = resume.get("checkpoints")
    if not isinstance(checkpoints, list):
        return resume
    latest: dict[str, dict[str, Any]] = {}
    for raw in checkpoints:
        if not isinstance(raw, Mapping):
            raise GovernanceArenaV2Error("resume checkpoint entry is invalid")
        checkpoint = dict(raw)
        branch = str(
            checkpoint.get("branch")
            or checkpoint.get("policy_branch")
            or ""
        )
        scenario_id = str(checkpoint.get("scenario_id") or "")
        seed = int(checkpoint.get("seed", -1))
        if not scenario_id or not branch or seed < 0:
            raise GovernanceArenaV2Error("resume checkpoint identity is incomplete")
        payload = checkpoint.get("payload")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            checkpoint["payload"] = base64.b64encode(bytes(payload)).decode(
                "ascii"
            )
        checkpoint["branch"] = branch
        key = f"{scenario_id}|{seed}|{branch}"
        if key not in latest or int(checkpoint["tick"]) > int(latest[key]["tick"]):
            latest[key] = checkpoint
    return {
        "branch_checkpoints": latest,
        "prefix_timelines": {
            key: copy.deepcopy(checkpoint.get("timeline_prefix", []))
            for key, checkpoint in latest.items()
        },
        "visual_frames": {
            key: copy.deepcopy(
                checkpoint.get("visual_frames_prefix", [])
            )
            for key, checkpoint in latest.items()
        },
        "completed_artifact_keys": copy.deepcopy(
            resume.get("completed_artifact_keys", [])
        ),
        "committed_artifacts": _index_committed_artifacts(
            resume.get("committed_artifacts", [])
        ),
    }


def _index_committed_artifacts(
    artifacts: Any,
) -> dict[str, Mapping[str, Any]]:
    if isinstance(artifacts, Mapping):
        return {
            str(key): value
            for key, value in artifacts.items()
            if isinstance(value, Mapping)
        }
    if not isinstance(artifacts, list):
        return {}
    indexed = {}
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise GovernanceArenaV2Error("committed artifact is invalid")
        kind = str(raw.get("artifact_kind") or "")
        scenario_id = str(raw.get("scenario_id") or "")
        branch = str(raw.get("branch") or raw.get("policy_branch") or "")
        seed = int(raw.get("seed", -1))
        tick = int(raw.get("tick", -1))
        if not kind or not scenario_id or not branch or seed < 0 or tick < 0:
            raise GovernanceArenaV2Error(
                "committed artifact identity is incomplete"
            )
        indexed[f"{kind}|{scenario_id}|{seed}|{branch}|{tick}"] = raw
    return indexed


def run_governance_arena_v2(
    plan: Mapping[str, Any],
    resume: Mapping[str, Any] | None = None,
    artifact_callback: ArtifactCallback | None = None,
    cancellation_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute one frozen v6 plan and return a hash-bound domain result."""

    if (
        plan.get("schema_version") != RUN_PLAN_SCHEMA_VERSION
        or plan.get("execution_mode") != "adaptive_particle_population"
        or int(plan.get("provider_call_limit", -1)) != 0
    ):
        raise GovernanceArenaV2Error("run plan is not an admissible v6 plan")
    public_plan = _public_plan(plan)
    plan_hash = public_plan.pop("plan_sha256", None)
    if plan_hash != canonical_hash(public_plan):
        raise GovernanceArenaV2Error("run plan hash does not match")
    runtime_inputs = plan.get("_runtime_inputs")
    if not isinstance(runtime_inputs, Mapping):
        raise GovernanceArenaV2Error("run plan lacks private runtime inputs")
    population = runtime_inputs["population"]
    graph_source = runtime_inputs["graph"]
    population_identity = compile_semantic_population_v3(
        population, strict_scale=bool(runtime_inputs.get("strict_scale", True))
    )
    if population_identity != plan["population_identity"]:
        raise GovernanceArenaV2Error("runtime population drifted from plan")
    graph = compile_message_exposure_graph_v2(population, graph_source)
    if graph.public_identity != plan["graph_identity"]:
        raise GovernanceArenaV2Error("runtime graph drifted from plan")
    writeback_mode = str(plan["formal_writeback_mode"])
    resume = _normalize_resume_state(resume)
    scenario_ids = tuple(str(item) for item in plan["scenario_ids"])
    seeds = tuple(int(item) for item in plan["seeds"])
    artifact_index: list[dict[str, Any]] = []
    scenario_output = {}
    visualization_frames = []
    completed_work_units = 0
    completed_probe_slots = 0
    for scenario_id in scenario_ids:
        seed_runs = []
        for seed_position, seed in enumerate(seeds):
            baseline = _run_shared_baseline(
                population=population,
                population_identity=population_identity,
                graph=graph,
                scenario_id=scenario_id,
                seed=seed,
                writeback_mode=writeback_mode,
                callback=artifact_callback,
                artifact_index=artifact_index,
                cancellation_probe=cancellation_probe,
                display_seed=seed_position == 0,
                resume=resume,
            )
            completed_work_units += 3
            completed_probe_slots += PROBE_BUDGET
            branches = {}
            for branch in BRANCH_IDS:
                branch_run = _run_policy_branch(
                    population=population,
                    population_identity=population_identity,
                    graph=graph,
                    scenario_id=scenario_id,
                    seed=seed,
                    branch=branch,
                    baseline=baseline,
                    writeback_mode=writeback_mode,
                    callback=artifact_callback,
                    artifact_index=artifact_index,
                    cancellation_probe=cancellation_probe,
                    display_seed=seed_position == 0,
                    resume=resume,
                )
                branches[branch] = branch_run
                completed_work_units += len(TIMELINE) - 3
                completed_probe_slots += len(PROBE_TICKS[1:]) * PROBE_BUDGET
                if seed_position == 0:
                    visualization_frames.extend(branch_run["visual_frames"])
            seed_runs.append(
                {
                    "seed": seed,
                    "shared_baseline_timeline": baseline["timeline"],
                    "branches": branches,
                }
            )
        scenario_output[scenario_id] = _aggregate_scenario_runs(
            seed_runs=seed_runs
        )
    if completed_work_units != int(plan["planned_work_units"]):
        raise GovernanceArenaV2Error("completed work-unit accounting drifted")
    if completed_probe_slots != int(plan["planned_fixture_probe_slots"]):
        raise GovernanceArenaV2Error("fixture probe-slot accounting drifted")
    expected_frames = len(scenario_ids) * len(BRANCH_IDS) * len(TIMELINE)
    if len(visualization_frames) != expected_frames:
        raise GovernanceArenaV2Error("canonical visualization frame count drifted")
    frame_index = [
        {
            "scenario_id": frame["scenario_id"],
            "branch": frame["branch"],
            "tick": frame["tick"],
            "content_sha256": frame["content_sha256"],
            "byte_length": (
                int(frame["particle_count"]) * int(frame["record_stride_bytes"])
            ),
        }
        for frame in visualization_frames
    ]
    visualization = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "canonical_display_seed": seeds[0],
        "frame_count": len(visualization_frames),
        "frames_per_scenario": len(BRANCH_IDS) * len(TIMELINE),
        "particle_count_per_frame": population_identity["particle_count"],
        "record_stride_bytes": 4,
        "channels": ["stance", "attention", "weight_bin", "public_action"],
        "frame_index": frame_index,
        "non_claim": (
            "Canonical frames are anonymous particle hypotheses; statistical "
            "intervals come from all paired seeds."
        ),
    }
    body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "plan": _public_plan(plan),
        "population_identity": population_identity,
        "graph_identity": graph.public_identity,
        "scenarios": scenario_output,
        "visualization": visualization,
        "method_gates": {
            "scale_gap": copy.deepcopy(plan["scale_gap_gate"]),
            "writeback": copy.deepcopy(plan["writeback_gate"]),
            "calibration": copy.deepcopy(plan["calibration_gate"]),
        },
        "usage": {
            "parent_episode_count": population_identity["parent_count"],
            "particle_count": population_identity["particle_count"],
            "low_cost_particle_tick_updates": (
                population_identity["particle_count"] * completed_work_units
            ),
            "work_units": completed_work_units,
            "fixture_probe_slots": completed_probe_slots,
            "provider_calls": 0,
            "provider_tokens": 0,
            "real_governance_actions": 0,
        },
        "privacy": {
            "record_level_graph_exported": False,
            "sample_identities_exported": False,
            "particle_parent_mapping_exported": False,
            "private_actor_observations_exported": False,
            "raw_text_exported": False,
        },
        "non_claims": [
            "One thousand parents are reviewed forum behaviour episodes, not students.",
            "Ten thousand particles express state uncertainty and are not evidence subjects.",
            "The message graph, diffusion timing, game, and policy outcomes are synthetic.",
            "Results are not real-world causal effects or governance advice.",
            "No Provider call or real governance action occurred.",
        ],
    }
    body["result_sha256"] = canonical_hash(body)
    body["_private_artifacts"] = {
        "schema_version": "campus-pulse-governance-private-artifact-index-v2",
        "records": artifact_index,
        "record_count": len(artifact_index),
        "callback_used": artifact_callback is not None,
        "content_captured_inline": False,
    }
    validate_governance_arena_v2_result(body)
    return body


def build_governance_arena_v2_result(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    scenario_ids: Sequence[str] = SCENARIO_IDS,
    seeds: Sequence[int] = tuple(20260730 + index for index in range(SEED_COUNT)),
    writeback_evidence: Mapping[str, Mapping[str, float]] | None = None,
    scale_gap_evidence: Mapping[str, Any] | None = None,
    calibration_evidence: (
        Sequence[Mapping[str, Any]] | Mapping[str, Any] | None
    ) = None,
    strict_scale: bool = True,
    artifact_callback: ArtifactCallback | None = None,
    cancellation_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    plan = build_run_plan_v6(
        population=population,
        graph=graph,
        scenario_ids=scenario_ids,
        seeds=seeds,
        writeback_evidence=writeback_evidence,
        scale_gap_evidence=scale_gap_evidence,
        calibration_evidence=calibration_evidence,
        strict_scale=strict_scale,
    )
    return run_governance_arena_v2(
        plan,
        artifact_callback=artifact_callback,
        cancellation_probe=cancellation_probe,
    )


def public_governance_arena_v2_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    public = {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if key != "_private_artifacts"
    }
    validate_governance_arena_v2_result(public)
    return public


def validate_governance_arena_v2_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    public = {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if key != "_private_artifacts"
    }
    stored_hash = public.pop("result_sha256", None)
    if stored_hash != canonical_hash(public):
        raise GovernanceArenaV2Error("result hash does not match")
    public["result_sha256"] = stored_hash
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise GovernanceArenaV2Error("result schema is not v2")
    identity = result.get("population_identity", {})
    if (
        int(identity.get("particles_per_parent", -1)) != PARTICLES_PER_PARENT
        or int(identity.get("particle_count", -1))
        != int(identity.get("parent_count", 0)) * PARTICLES_PER_PARENT
    ):
        raise GovernanceArenaV2Error("result population identity is invalid")
    if result.get("usage", {}).get("provider_calls") != 0:
        raise GovernanceArenaV2Error("v2 result performed a Provider call")
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise GovernanceArenaV2Error("result has no scenarios")
    required_truth = {
        "attention",
        "concern",
        "trust",
        "satisfaction",
        "support",
        "opposition",
        "activity",
        "voice_gap",
        "service_strain",
        "rumor_belief",
        "verified_belief",
    }
    for scenario in scenarios.values():
        branches = scenario.get("branches")
        if tuple(branches or {}) != BRANCH_IDS:
            raise GovernanceArenaV2Error("result branches are incomplete or reordered")
        for branch in branches.values():
            timeline = branch.get("timeline")
            if not isinstance(timeline, list) or len(timeline) != len(TIMELINE):
                raise GovernanceArenaV2Error("result timeline is incomplete")
            for tick, point in enumerate(timeline):
                if (
                    point.get("tick") != tick
                    or point.get("phase") != TIMELINE[tick]
                    or set(point.get("truth", {})) != required_truth
                    or set(point.get("observed", {})) != required_truth
                ):
                    raise GovernanceArenaV2Error("timeline point contract drifted")
                if int(point["probe"]["budget"]) not in {0, PROBE_BUDGET}:
                    raise GovernanceArenaV2Error("timeline probe budget drifted")
                allocation = point["probe"].get("allocation_by_stratum")
                if (
                    not isinstance(allocation, Mapping)
                    or sum(int(value) for value in allocation.values())
                    != int(point["probe"]["budget"])
                ):
                    raise GovernanceArenaV2Error(
                        "timeline anonymous probe allocation drifted"
                    )
            resource = branch.get("resource", {})
            if not resource.get("conserved") or not math.isclose(
                float(resource["opening"]),
                float(resource["consumed"])
                + float(resource["committed"])
                + float(resource["remaining"]),
                abs_tol=1e-6,
            ):
                raise GovernanceArenaV2Error("result resource accounting drifted")
    visualization = result.get("visualization", {})
    expected_frames = len(scenarios) * len(BRANCH_IDS) * len(TIMELINE)
    if (
        visualization.get("schema_version") != VISUALIZATION_SCHEMA_VERSION
        or int(visualization.get("frame_count", -1)) != expected_frames
        or len(visualization.get("frame_index", [])) != expected_frames
    ):
        raise GovernanceArenaV2Error("visualization manifest is incomplete")
    for frame in visualization["frame_index"]:
        if (
            not isinstance(frame.get("content_sha256"), str)
            or len(frame["content_sha256"]) != 64
            or int(frame.get("byte_length", -1))
            != int(identity["particle_count"]) * 4
        ):
            raise GovernanceArenaV2Error("visual frame index drifted")
    rendered = json.dumps(public, ensure_ascii=False).lower()
    for forbidden in (
        '"micro_agent_id"',
        '"selected_indices"',
        '"private_observations"',
        '"source_parent_index"',
        '"_edges"',
        '"api_key"',
        '"password"',
        '"raw_text"',
        '"payload"',
        '"content"',
        '"content_base64"',
        '"prompt_fingerprint"',
        '"agent_prompt"',
    ):
        if forbidden in rendered:
            raise GovernanceArenaV2Error(
                f"public result contains forbidden material: {forbidden}"
            )
    return public
