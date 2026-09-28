"""Pure-domain YuLan-Scale simulation engine.

The engine composes the existing vectorised particle/message kernel with the
budgeted resident/actor LLM contracts.  It deliberately does not know about
SQLite, workers, HTTP routes, Provider credentials, or release packaging.

Three execution provenances are supported here:

``emulator_only_development``
    Advances the complete population and creates every activation/actor slot
    plan, but acquires no semantic turn.  Such a result is never publishable.

``reviewed_trace_replay``
    Acquires every resident and governance-actor turn through a
    :class:`SemanticTurnSource`.  Exact prompt fingerprints are verified by
    the shared YuLan-Scale turn validators.  A missing or mismatched trace
    fails closed.

``authorized_live_llm``
    Uses the same semantic source protocol, but the source must be explicitly
    injected by the authorized runtime layer.  This pure-domain module still
    owns no credential, HTTP client, authorization row, or retry policy.
"""

from __future__ import annotations

import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence
import zlib

import numpy as np

from .governance_arena_v2 import (
    ACTION_CATALOG,
    ACTIVITY,
    CONCERN,
    BRANCH_ACTIONS,
    BRANCH_IDS,
    CHECKPOINT_TICKS,
    DECISION_TICKS,
    PARTICLES_PER_PARENT,
    PUBLIC_ACTIONS,
    PROBE_TICKS,
    SCENARIO_IDS,
    STATE_FIELDS,
    SUPPORT,
    OPPOSITION,
    TRUST,
    SATISFACTION,
    RUMOR_BELIEF,
    VERIFIED_BELIEF,
    TIMELINE,
    BranchState,
    GraphArrays,
    MessageObject,
    _actor_observations,
    _aggregate_scenario_runs,
    _diffusion_summary,
    _draw_from_array_distribution,
    _hash_uniform,
    _message_candidate_exposure,
    _model_probabilities,
    _parent_features,
    _resource_view,
    _timeline_point,
    _update_actor_beliefs,
    _visible_messages_for_probe,
    advance_particle_tick,
    compile_message_exposure_graph_v2,
    decode_governance_checkpoint_v2,
    encode_governance_checkpoint_v2,
    encode_visual_frame_v2,
    initial_branch_state,
    play_qre_governance_round,
)
from .mass_population import canonical_hash
from .yulan_scale import (
    ACTOR_IDS,
    GOVERNANCE_ACTOR_LLM_SLOTS,
    LONGITUDINAL_ANCHOR_COUNT,
    PPS_SAMPLE_BUDGET,
    RESIDENT_LLM_BUDGET,
    RESIDENT_LLM_SLOTS,
    SemanticTurnSource,
    TurnEnvelope,
    apply_individual_llm_updates_v1,
    draw_rotating_pps_sample_v1,
    emulator_topic_observation_id,
    estimate_population_from_llm_v1,
    play_llm_governance_round_v1,
    render_governance_actor_turn_v1,
    render_resident_llm_turn_v1,
    scenario_population_view,
    validate_governance_actor_turn_v1,
    validate_resident_llm_turn_v1,
)


RESULT_SCHEMA_VERSION = "campus-pulse-llm-governance-result-v3"
CHECKPOINT_SCHEMA_VERSION = "campus-pulse-llm-population-checkpoint-v3"
VISUALIZATION_SCHEMA_VERSION = "campus-pulse-llm-population-visualization-v3"
AGGREGATE_CURSOR_SCHEMA_VERSION = (
    "campus-pulse-llm-population-aggregate-cursor-v1"
)
EXECUTION_MODE = "budgeted_llm_agent_population"
EMULATOR_ONLY_PROVENANCE = "emulator_only_development"
REVIEWED_TRACE_PROVENANCE = "reviewed_trace_replay"
AUTHORIZED_LIVE_PROVENANCE = "authorized_live_llm"
RESIDENT_TURN_SCHEMA_VERSION = "campus-pulse-budgeted-llm-agent-turn-v1"
ACTOR_TURN_SCHEMA_VERSION = "campus-pulse-governance-actor-turn-v1"
SUPPORTED_PROVENANCES = (
    EMULATOR_ONLY_PROVENANCE,
    REVIEWED_TRACE_PROVENANCE,
    AUTHORIZED_LIVE_PROVENANCE,
)
CommittedSemanticTurns = (
    Mapping[tuple[str, str], Mapping[str, Any]]
    | frozenset[tuple[str, str]]
)
FORMAL_SCENARIO_COUNT = 2
FORMAL_SEED_COUNT = 8
FORMAL_PARENT_COUNT = 1000
CANONICAL_SEEDS = (
    20260722,
    20261731,
    20262740,
    20263749,
    20264758,
    20265767,
    20266776,
    20267785,
)
FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED = 3 + (len(TIMELINE) - 3) * len(
    BRANCH_IDS
)
FORMAL_RESIDENT_ROUNDS_PER_SCENARIO_SEED = 1 + 3 * len(BRANCH_IDS)
FORMAL_ACTOR_ROUNDS_PER_SCENARIO_SEED = 4 * 4


class YuLanScaleEngineError(ValueError):
    """Raised when a pure-domain YuLan-Scale run violates its contract."""


class YuLanScaleEngineCancelled(RuntimeError):
    """Raised before a tick when the runtime cancellation probe is set."""


def _resident_decision_artifact(decision: Any) -> dict[str, Any]:
    return {
        "schema_version": RESIDENT_TURN_SCHEMA_VERSION,
        "action": decision.action,
        "topic_id": decision.topic_id,
        "stance": decision.stance,
        "urgency": float(decision.urgency),
        "trust_delta": float(decision.trust_delta),
        "satisfied": bool(decision.satisfied),
        "rationale_tag": decision.rationale_tag,
        "message": decision.message,
    }


def _actor_decision_artifact(decision: Any) -> dict[str, Any]:
    return {
        "schema_version": ACTOR_TURN_SCHEMA_VERSION,
        "action_probabilities": dict(decision.action_probabilities),
        "preferred_action": decision.preferred_action,
        "confidence": float(decision.confidence),
        "rationale_tags": list(decision.rationale_tags),
        "risk_level": decision.risk_level,
        "uncertainty": float(decision.uncertainty),
        "public_message": decision.public_message,
    }


def _committed_envelope(
    *,
    slot_kind: str,
    prompt_sha256: str,
    record: Mapping[str, Any],
    execution_provenance: str,
) -> TurnEnvelope:
    decision = record.get("validated_decision")
    response_sha256 = record.get("response_fingerprint")
    if not isinstance(decision, Mapping) or not isinstance(
        response_sha256, str
    ):
        raise YuLanScaleEngineError(
            "committed semantic turn lacks its validated decision"
        )
    expected_schema = (
        RESIDENT_TURN_SCHEMA_VERSION
        if slot_kind == "resident"
        else ACTOR_TURN_SCHEMA_VERSION
    )
    if decision.get("schema_version") != expected_schema:
        raise YuLanScaleEngineError(
            "committed semantic decision schema drifted"
        )
    content = json.dumps(
        {
            key: value
            for key, value in decision.items()
            if key != "schema_version"
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return TurnEnvelope(
        slot_kind=slot_kind,
        prompt_sha256=prompt_sha256,
        response_sha256=response_sha256,
        provenance=execution_provenance,
        content=content,
        returned_model=str(record.get("returned_model") or "durable-attempt"),
        prompt_tokens=int(record.get("input_tokens", 0)),
        completion_tokens=int(record.get("output_tokens", 0)),
        total_tokens=(
            int(record.get("input_tokens", 0))
            + int(record.get("output_tokens", 0))
        ),
        provider_calls=int(record.get("provider_call_count", 0)),
        cache_hit=record.get("transport_kind") == "cache",
        provider_attempts=int(record.get("provider_attempt_count", 0)),
        format_repairs=int(record.get("format_repair_count", 0)),
        provider_billed_tokens=(
            int(record.get("input_tokens", 0))
            + int(record.get("output_tokens", 0))
            if int(record.get("provider_call_count", 0)) > 0
            else 0
        ),
    )


@dataclass
class YuLanScaleBranchRuntime:
    """Complete in-memory state needed for deterministic branch recovery."""

    state: BranchState
    llm_memories: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    amortized_histories: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    activation_rounds: list[dict[str, Any]] = field(default_factory=list)
    resident_slots: list[dict[str, Any]] = field(default_factory=list)
    actor_slots: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    visual_frames: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "resident_slots_planned": 0,
            "resident_llm_turns_completed": 0,
            "governance_actor_slots_planned": 0,
            "governance_actor_llm_turns_completed": 0,
            "trace_replay_turns": 0,
            "provider_calls": 0,
            "provider_tokens": 0,
            "provider_attempts": 0,
            "cache_hits": 0,
            "format_repairs": 0,
            "fixture_turns": 0,
            "emulator_population_updates": 0,
        }
    )

    def clone(self) -> "YuLanScaleBranchRuntime":
        return YuLanScaleBranchRuntime(
            state=self.state.clone(),
            llm_memories=copy.deepcopy(self.llm_memories),
            amortized_histories=copy.deepcopy(self.amortized_histories),
            activation_rounds=copy.deepcopy(self.activation_rounds),
            resident_slots=copy.deepcopy(self.resident_slots),
            actor_slots=copy.deepcopy(self.actor_slots),
            timeline=copy.deepcopy(self.timeline),
            visual_frames=copy.deepcopy(self.visual_frames),
            usage=dict(self.usage),
        )


def build_yulan_scale_workload_summary(
    *,
    scenario_count: int = FORMAL_SCENARIO_COUNT,
    seed_count: int = FORMAL_SEED_COUNT,
    parent_count: int = FORMAL_PARENT_COUNT,
    particles_per_parent: int = PARTICLES_PER_PARENT,
) -> dict[str, int]:
    """Return the frozen workload arithmetic without executing a simulation."""

    if (
        scenario_count <= 0
        or seed_count <= 0
        or parent_count <= 0
        or particles_per_parent != PARTICLES_PER_PARENT
    ):
        raise YuLanScaleEngineError("YuLan-Scale workload dimensions are invalid")
    population_branch_ticks = (
        scenario_count * seed_count * FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
    )
    resident_rounds = (
        scenario_count
        * seed_count
        * FORMAL_RESIDENT_ROUNDS_PER_SCENARIO_SEED
    )
    actor_rounds = (
        scenario_count * seed_count * FORMAL_ACTOR_ROUNDS_PER_SCENARIO_SEED
    )
    resident_slots = resident_rounds * RESIDENT_LLM_BUDGET
    actor_slots = actor_rounds * len(ACTOR_IDS)
    return {
        "population_branch_ticks": population_branch_ticks,
        "particle_updates": (
            population_branch_ticks * parent_count * particles_per_parent
        ),
        "resident_activation_rounds": resident_rounds,
        "resident_llm_slots": resident_slots,
        "governance_actor_rounds": actor_rounds,
        "governance_actor_llm_slots": actor_slots,
        "total_llm_slots": resident_slots + actor_slots,
    }


def build_yulan_scale_aggregate_cursor(
    *,
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
    scenario_seed_work_unit_offset: int,
) -> dict[str, Any]:
    """Bind one domain checkpoint to the frozen global work-unit frontier."""

    if (
        isinstance(scenario_seed_work_unit_offset, bool)
        or not isinstance(scenario_seed_work_unit_offset, int)
        or scenario_seed_work_unit_offset < 0
        or scenario_seed_work_unit_offset
        % FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
        != 0
    ):
        raise YuLanScaleEngineError("aggregate cursor offset is invalid")
    if branch == "shared_baseline":
        if tick != 2:
            raise YuLanScaleEngineError(
                "shared aggregate cursor must bind baseline T2"
            )
        branch_ordinal = -1
        completed_units = scenario_seed_work_unit_offset + 3
    else:
        if branch not in BRANCH_IDS or tick not in CHECKPOINT_TICKS or tick < 3:
            raise YuLanScaleEngineError("branch aggregate cursor is invalid")
        branch_ordinal = BRANCH_IDS.index(branch)
        completed_units = (
            scenario_seed_work_unit_offset
            + 3
            + branch_ordinal * (len(TIMELINE) - 3)
            + (tick - 2)
        )
    body = {
        "schema_version": AGGREGATE_CURSOR_SCHEMA_VERSION,
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "branch": branch,
        "tick": int(tick),
        "scenario_seed_work_unit_offset": scenario_seed_work_unit_offset,
        "branch_ordinal": branch_ordinal,
        "completed_units": completed_units,
    }
    return {**body, "cursor_sha256": canonical_hash(body)}


def _validate_yulan_scale_aggregate_cursor(
    cursor: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
    branch: str,
    tick: int,
    scenario_seed_work_unit_offset: int,
) -> dict[str, Any]:
    expected = build_yulan_scale_aggregate_cursor(
        scenario_id=scenario_id,
        seed=seed,
        branch=branch,
        tick=tick,
        scenario_seed_work_unit_offset=scenario_seed_work_unit_offset,
    )
    if dict(cursor) != expected:
        raise YuLanScaleEngineError("aggregate cursor identity drifted")
    return expected


def _validated_emulator_release_identity(
    release: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(release, Mapping):
        raise YuLanScaleEngineError(
            "formal YuLan-Scale requires a frozen LLM emulator release"
        )
    if release.get("schema_version") != "llm-behavior-emulator-v1":
        raise YuLanScaleEngineError("LLM emulator release schema drifted")
    gate = release.get("validation_gate")
    passed = bool(
        release.get("formal_eligible") is True
        or (
            isinstance(gate, Mapping)
            and (
                gate.get("passed") is True
                or gate.get("status") in {"passed", "eligible"}
            )
        )
    )
    if not passed:
        raise YuLanScaleEngineError(
            "LLM emulator release has not passed its frozen gate"
        )
    release_sha = str(
        release.get("emulator_sha256")
        or release.get("release_sha256")
        or release.get("model_sha256")
        or ""
    )
    if (
        len(release_sha) != 64
        or any(character not in "0123456789abcdef" for character in release_sha)
    ):
        raise YuLanScaleEngineError("LLM emulator release hash is invalid")
    return {
        "schema_version": "llm-behavior-emulator-v1",
        "emulator_sha256": release_sha,
        "validation_gate_passed": True,
    }


def _select_anchor_indices(
    population: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
) -> tuple[int, ...]:
    """Use the population compiler's public anchor-selection API."""

    # Imported lazily so the population/compiler layer remains independently
    # testable while this engine stays a consumer rather than an owner.
    from .llm_population import select_longitudinal_anchors

    selected = select_longitudinal_anchors(
        population=population,
        scenario_id=scenario_id,
        seed=seed,
    )
    if isinstance(selected, Mapping):
        raw = (
            selected.get("agent_indices")
            or selected.get("anchor_indices")
            or selected.get("selected_indices")
            or selected.get("indices")
        )
    else:
        raw = selected
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise YuLanScaleEngineError("anchor selector returned no index sequence")
    indices = tuple(sorted(int(index) for index in raw))
    if (
        len(indices) != LONGITUDINAL_ANCHOR_COUNT
        or len(set(indices)) != LONGITUDINAL_ANCHOR_COUNT
        or min(indices, default=-1) < 0
        or max(indices, default=-1) >= len(population["agents"])
    ):
        raise YuLanScaleEngineError("anchor selector violated the 16-unit panel")
    return indices


def _state_summary(state: BranchState, index: int) -> dict[str, float]:
    values = state.ensemble.parent_means()[index]
    return {
        field: round(float(values[position]), 6)
        for position, field in enumerate(STATE_FIELDS)
    }


def _public_activation_round(
    sample: Mapping[str, Any],
    *,
    completed: int,
    posterior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected = tuple(int(index) for index in sample["selected"])
    inclusion = sample["inclusion_probability"]
    pi = [float(inclusion[index]) for index in selected]
    body = {
        "schema_version": "campus-pulse-llm-activation-round-public-v1",
        "scenario_id": str(sample["scenario_id"]),
        "branch": str(sample["branch"]),
        "tick": int(sample["tick"]),
        "anchor_count": int(sample["anchor_count"]),
        "pps_count": int(sample["pps_count"]),
        "resident_slots_planned": int(sample["resident_llm_budget"]),
        "resident_llm_turns_completed": int(completed),
        "fixture_turns": 0,
        "allocation_by_stratum": copy.deepcopy(
            sample["allocation_by_stratum"]
        ),
        "selected_inclusion_probability_range": [
            round(min(pi), 8),
            round(max(pi), 8),
        ],
        "kish_ess": (
            round(float(posterior["kish_ess"]), 6)
            if posterior is not None
            else 0.0
        ),
    }
    body["sampling_design_sha256"] = canonical_hash(body)
    return body


def _aggregate_anchor_state(
    *,
    state: BranchState,
    anchor_indices: Sequence[int],
) -> dict[str, float]:
    parent = state.ensemble.parent_means()[list(anchor_indices)]
    return {
        field: round(float(np.mean(parent[:, position])), 6)
        for position, field in enumerate(STATE_FIELDS)
    }


def _categorical_emulator_residuals(
    *,
    indices: Sequence[int],
    semantics: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for family, semantic_key in (
        ("action", "action"),
        ("stance", "stance"),
        ("topic_id", "topic_id"),
    ):
        distribution = predictions.get(family)
        if not isinstance(distribution, Mapping):
            raise YuLanScaleEngineError(
                f"emulator residual lacks {family} predictions"
            )
        categories = sorted(str(key) for key in distribution)
        observed = {category: 0.0 for category in categories}
        predicted = {category: 0.0 for category in categories}
        assigned_probability = []
        for index in indices:
            actual = str(semantics[int(index)][semantic_key])
            if family == "topic_id":
                actual = emulator_topic_observation_id(
                    actual,
                    available_categories=categories,
                )
            if actual not in observed:
                raise YuLanScaleEngineError(
                    "LLM decision escaped emulator categories"
                )
            observed[actual] += 1.0
            for category in categories:
                value = float(
                    np.asarray(distribution[category], dtype=np.float64)[
                        int(index)
                    ]
                )
                predicted[category] += value
                if category == actual:
                    assigned_probability.append(value)
        denominator = max(len(indices), 1)
        observed = {
            key: value / denominator for key, value in observed.items()
        }
        predicted = {
            key: value / denominator for key, value in predicted.items()
        }
        result[family] = {
            "observed_share": {
                key: round(value, 6) for key, value in observed.items()
            },
            "emulator_share": {
                key: round(value, 6) for key, value in predicted.items()
            },
            "share_mae": round(
                sum(
                    abs(observed[key] - predicted[key])
                    for key in categories
                )
                / max(len(categories), 1),
                6,
            ),
            "mean_one_minus_assigned_probability": round(
                float(np.mean([1.0 - value for value in assigned_probability])),
                6,
            ),
        }
        if family == "topic_id":
            result[family]["observation_vocabulary"] = (
                "reviewed_benchmark_primary_topic"
            )
    return result


def _emulator_predictions_from_kernel(
    *,
    state: BranchState,
    features: Mapping[str, np.ndarray],
    scenario_id: str,
    tick: int,
) -> dict[str, Any]:
    """Return model predictions, never decisions or synthetic LLM traces."""

    parent = state.ensemble.parent_means()
    categorical = _model_probabilities(
        parent,
        features,
        scenario_id=scenario_id,
        tick=tick,
    )
    return {
        **categorical,
        "numeric": {
            "urgency": parent[:, 1].copy(),
            "trust": parent[:, 2].copy(),
            "satisfaction": parent[:, 3].copy(),
            "activity": parent[:, ACTIVITY].copy(),
        },
    }


def _resolve_emulator_predictions(
    *,
    predictor: Callable[..., Mapping[str, Any]] | None,
    population: Mapping[str, Any],
    state: BranchState,
    features: Mapping[str, np.ndarray],
    scenario_id: str,
    branch: str,
    tick: int,
) -> dict[str, Any]:
    if predictor is None:
        return _emulator_predictions_from_kernel(
            state=state,
            features=features,
            scenario_id=scenario_id,
            tick=tick,
        )
    value = predictor(
        population=population,
        state=state,
        features=features,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
    )
    if not isinstance(value, Mapping):
        raise YuLanScaleEngineError("LLM behavior emulator returned no predictions")
    return dict(value)


def _prediction_array(
    values: Any,
    *,
    parent_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (parent_count,)
        or not np.isfinite(array).all()
    ):
        raise YuLanScaleEngineError(
            f"LLM behavior emulator {name} prediction is invalid"
        )
    return array


def _apply_emulator_state_correction(
    *,
    state: BranchState,
    predictions: Mapping[str, Any],
) -> None:
    """Let the frozen LLM emulator correct every parent's particle state."""

    count = state.ensemble.parent_count
    numeric = predictions.get("numeric")
    if not isinstance(numeric, Mapping):
        raise YuLanScaleEngineError(
            "LLM behavior emulator lacks continuous predictions"
        )
    absolute_targets = {
        CONCERN: numeric.get("concern", numeric.get("urgency")),
        TRUST: numeric.get("trust"),
        SATISFACTION: numeric.get("satisfaction"),
        ACTIVITY: numeric.get("activity"),
    }
    for field_index, raw in absolute_targets.items():
        if raw is None:
            continue
        target = np.clip(
            _prediction_array(
                raw,
                parent_count=count,
                name=STATE_FIELDS[field_index],
            ),
            0.0,
            1.0,
        )
        state.ensemble.values[:, :, field_index] = np.clip(
            0.92 * state.ensemble.values[:, :, field_index]
            + 0.08 * target[:, None],
            0.0,
            1.0,
        )
    deltas = predictions.get("state_delta")
    if deltas is not None:
        if not isinstance(deltas, Mapping):
            raise YuLanScaleEngineError(
                "LLM behavior emulator state deltas are invalid"
            )
        for field, raw in deltas.items():
            if field not in STATE_FIELDS:
                raise YuLanScaleEngineError(
                    "LLM behavior emulator emitted an unknown state field"
                )
            index = STATE_FIELDS.index(str(field))
            delta = np.clip(
                _prediction_array(
                    raw,
                    parent_count=count,
                    name=f"{field}_delta",
                ),
                -0.25,
                0.25,
            )
            state.ensemble.values[:, :, index] = np.clip(
                state.ensemble.values[:, :, index]
                + 0.12 * delta[:, None],
                0.0,
                1.0,
            )
    stance = predictions.get("stance")
    if isinstance(stance, Mapping):
        supportive = stance.get("supportive")
        critical = stance.get("critical")
        concerned = stance.get("concerned")
        if supportive is not None:
            target = np.clip(
                _prediction_array(
                    supportive,
                    parent_count=count,
                    name="supportive",
                ),
                0.0,
                1.0,
            )
            state.ensemble.values[:, :, SUPPORT] = np.clip(
                0.96 * state.ensemble.values[:, :, SUPPORT]
                + 0.04 * target[:, None],
                0.0,
                1.0,
            )
        if critical is not None:
            target = _prediction_array(
                critical,
                parent_count=count,
                name="critical",
            )
            if concerned is not None:
                target = target + 0.5 * _prediction_array(
                    concerned,
                    parent_count=count,
                    name="concerned",
                )
            target = np.clip(target, 0.0, 1.0)
            state.ensemble.values[:, :, OPPOSITION] = np.clip(
                0.96 * state.ensemble.values[:, :, OPPOSITION]
                + 0.04 * target[:, None],
                0.0,
                1.0,
            )
    state.ensemble.assert_conserved()


def _emit_emulator_structured_messages(
    *,
    population: Mapping[str, Any],
    state: BranchState,
    predictions: Mapping[str, Any],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    best_message: np.ndarray,
    parent_depth: np.ndarray,
) -> list[MessageObject]:
    """Emit at most one text-free semantic packet per persistent agent."""

    actions = predictions.get("action")
    stances = predictions.get("stance")
    topics = predictions.get("topic_id")
    if not all(
        isinstance(value, Mapping)
        for value in (actions, stances, topics)
    ):
        raise YuLanScaleEngineError(
            "LLM behavior emulator categorical predictions are incomplete"
        )
    count = state.ensemble.parent_count
    for name, distribution in (
        ("action", actions),
        ("stance", stances),
        ("topic", topics),
    ):
        for category, raw in distribution.items():
            _prediction_array(
                raw,
                parent_count=count,
                name=f"{name}:{category}",
            )
    parent = state.ensemble.parent_means()
    emitted = []
    for index, agent in enumerate(population["agents"]):
        action = _draw_from_array_distribution(
            actions,
            index=index,
            uniform=_hash_uniform(
                seed,
                scenario_id,
                branch,
                tick,
                index,
                "llm-emulator-action",
            ),
        )
        archetype = str(agent["primary_archetype_id"])
        if archetype == "archetype-16":
            action = "read"
        elif archetype == "archetype-09" and action not in {
            "ignore",
            "read",
            "report_risk",
        }:
            action = "read"
        if action not in PUBLIC_ACTIONS:
            continue
        stance = _draw_from_array_distribution(
            stances,
            index=index,
            uniform=_hash_uniform(
                seed,
                scenario_id,
                branch,
                tick,
                index,
                "llm-emulator-stance",
            ),
        )
        topic = _draw_from_array_distribution(
            topics,
            index=index,
            uniform=_hash_uniform(
                seed,
                scenario_id,
                branch,
                tick,
                index,
                "llm-emulator-topic",
            ),
        )
        parent_position = int(best_message[index])
        previous = (
            state.ledger.messages[parent_position]
            if parent_position >= 0
            else None
        )
        if previous is not None and previous.created_tick >= tick:
            raise YuLanScaleEngineError(
                "emulator message lineage did not advance in time"
            )
        source_group = archetype
        target_group = archetype
        bridge = False
        depth = 0
        if previous is not None:
            source_group = previous.source_group
            target_group = archetype
            bridge = previous.target_group != archetype
            depth = min(int(parent_depth[index]) + 1, 24)
        support = float(parent[index, SUPPORT])
        opposition = float(parent[index, OPPOSITION])
        if stance == "supportive":
            support = max(support, 0.55)
        elif stance in {"critical", "concerned"}:
            opposition = max(opposition, 0.50)
        message = MessageObject(
            message_id=canonical_hash(
                [
                    "llm-behavior-emulator-v1",
                    scenario_id,
                    seed,
                    branch,
                    tick,
                    index,
                    action,
                    topic,
                    stance,
                ]
            ),
            parent_message_id=(
                previous.message_id if previous is not None else None
            ),
            claim_id=f"{scenario_id}:primary-claim",
            source_role="persistent_llm_resident_emulator",
            source_parent_index=index,
            source_group=source_group,
            target_group=target_group,
            topic=str(topic),
            stance_support=support,
            stance_opposition=opposition,
            evidence_status=(
                "verified"
                if parent[index, VERIFIED_BELIEF]
                >= parent[index, RUMOR_BELIEF]
                else "unverified"
            ),
            emotion=float(parent[index, CONCERN]),
            actionability=float(parent[index, ACTIVITY]),
            distortion=float(
                max(
                    0.0,
                    parent[index, RUMOR_BELIEF]
                    - parent[index, VERIFIED_BELIEF],
                )
            ),
            created_tick=tick,
            ttl=3 + int(
                _hash_uniform(
                    seed,
                    scenario_id,
                    branch,
                    tick,
                    index,
                    "llm-emulator-ttl",
                )
                > 0.5
            ),
            action=str(action),
            provenance="llm_behavior_emulator_v1",
            bridge=bridge,
            depth=depth,
        )
        # The parent message was resolved by its ledger position above and its
        # timestamp was already checked.  Appending directly keeps the same
        # lineage invariant while avoiding MessageLedger.add's defensive
        # full-ledger scan for every one of the up to 1,000 emissions.  That
        # scan is quadratic over a 24-tick branch and dominates the formal
        # 17.28M-update workload.
        position = len(state.ledger.messages)
        state.ledger.messages.append(message)
        state.ledger.last_by_parent[index] = position
        state.ledger.last_tick_by_parent[index] = tick
        emitted.append(message)
    return emitted


def _prune_inactive_messages(
    *,
    state: BranchState,
    tick: int,
) -> None:
    """Keep only messages that can still participate in future exposure.

    Message batches are persisted before checkpoints, so a checkpoint needs
    the active attention queue rather than the whole branch history.  Keeping
    expired packets made every later checkpoint repeatedly serialize and
    compress thousands of inert objects, contrary to the v7 checkpoint
    contract's ``active messages`` boundary.
    """

    messages = state.ledger.messages
    kept = [
        message
        for message in messages
        if message.action != "suppressed"
        and message.created_tick + message.ttl > tick
    ]
    if len(kept) == len(messages):
        return
    kept_ids = {message.message_id for message in kept}
    # The immutable message batch already preserves the original lineage.
    # Re-root an active checkpoint packet when its expired parent is omitted,
    # so the compact in-memory ledger remains internally closed and future
    # descendants never reference an unavailable object.
    kept = [
        replace(message, parent_message_id=None)
        if message.parent_message_id not in kept_ids
        else message
        for message in kept
    ]
    state.ledger.messages = kept
    state.ledger.last_by_parent.fill(-1)
    state.ledger.last_tick_by_parent.fill(-100)
    for position, message in enumerate(kept):
        if message.source_parent_index is None:
            continue
        parent = int(message.source_parent_index)
        previous_tick = int(state.ledger.last_tick_by_parent[parent])
        if message.created_tick >= previous_tick:
            state.ledger.last_by_parent[parent] = position
            state.ledger.last_tick_by_parent[parent] = message.created_tick


def _advance_unified_emulator_tick(
    *,
    population: Mapping[str, Any],
    graph: GraphArrays,
    state: BranchState,
    features: Mapping[str, np.ndarray],
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    emulator_predictor: Callable[..., Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance particles, then let the frozen LLM emulator drive semantics."""

    # The v2 surrogate emitter is intentionally disabled.  Its particle and
    # attention equations remain the vectorised state-space substrate.
    advance_particle_tick(
        population=population,
        graph=graph,
        state=state,
        features=features,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
        emit_messages=False,
    )
    predictions = _resolve_emulator_predictions(
        predictor=emulator_predictor,
        population=population,
        state=state,
        features=features,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
    )
    _apply_emulator_state_correction(
        state=state,
        predictions=predictions,
    )
    (
        _,
        _,
        _,
        best_message,
        parent_depth,
    ) = _message_candidate_exposure(
        state=state,
        graph=graph,
        tick=tick,
    )
    messages = _emit_emulator_structured_messages(
        population=population,
        state=state,
        predictions=predictions,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
        best_message=best_message,
        parent_depth=parent_depth,
    )
    diffusion = _diffusion_summary(
        population=population,
        graph=graph,
        state=state,
        tick=tick,
        messages=messages,
    )
    _prune_inactive_messages(state=state, tick=tick)
    return diffusion, predictions


def _v2_probe_view(posterior: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if posterior is None:
        return None
    value = copy.deepcopy(dict(posterior))
    numeric = value["corrected"]["numeric"]
    # The v2 observer names the semantic urgency correction ``concern``.
    # Preserve both names in the unified result instead of changing either
    # frozen component contract.
    numeric.setdefault("concern", numeric.get("urgency", 0.0))
    return value


def _run_resident_activation(
    *,
    runtime: YuLanScaleBranchRuntime,
    population: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    anchor_indices: Sequence[int],
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    execution_provenance: str,
    semantic_source: SemanticTurnSource | None,
    emulator_predictions: Mapping[str, Any],
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    committed_semantic_turns: CommittedSemanticTurns = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample = draw_rotating_pps_sample_v1(
        population=population,
        state=runtime.state,
        features=features,
        anchor_indices=anchor_indices,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
        seed=seed,
    )
    activated = tuple(anchor_indices) + tuple(sample["selected"])
    rendered_by_index = {}
    private_slots = []
    for ordinal, index in enumerate(activated):
        agent = population["agents"][index]
        visible = _visible_messages_for_probe(
            state=runtime.state,
            archetype=str(agent["primary_archetype_id"]),
            tick=tick,
        )
        rendered = render_resident_llm_turn_v1(
            agent=agent,
            posterior_state=_state_summary(runtime.state, index),
            visible_messages=visible,
            llm_memory=runtime.llm_memories.get(index, ()),
            amortized_history=runtime.amortized_histories.get(index, ()),
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
        )
        rendered_by_index[index] = rendered
        private_slots.append(
            {
                "slot_kind": "resident",
                "slot_ordinal": ordinal,
                "parent_index": int(index),
                "primary_archetype_id": str(
                    agent["primary_archetype_id"]
                ),
                "persona_variant_id": str(
                    agent.get("persona_variant_id") or ""
                ),
                "role": (
                    "longitudinal_anchor"
                    if index in set(anchor_indices)
                    else "rotating_pps"
                ),
                "prompt_sha256": rendered.prompt_sha256,
                "status": (
                    "planned_not_acquired"
                    if execution_provenance == EMULATOR_ONLY_PROVENANCE
                    else "planned"
                ),
            }
        )
    selected_pi = np.asarray(
        [
            float(sample["inclusion_probability"][int(index)])
            for index in sample["selected"]
        ],
        dtype=np.float64,
    )
    design_weights = 1.0 / np.clip(selected_pi, 1e-12, 1.0)
    design_kish_ess = float(
        np.square(np.sum(design_weights))
        / np.sum(np.square(design_weights))
    )
    if artifact_callback is not None:
        artifact_callback(
            "activation_round",
            {
                "schema_version": (
                    "campus-pulse-llm-activation-round-private-v1"
                ),
                "scenario_id": scenario_id,
                "seed": int(seed),
                "branch": branch,
                "tick": int(tick),
                "anchor_count": LONGITUDINAL_ANCHOR_COUNT,
                "pps_count": PPS_SAMPLE_BUDGET,
                "anchor_identity_sha256": sample[
                    "anchor_identity_sha256"
                ],
                "pps_identity_sha256": sample["pps_identity_sha256"],
                "selected_inclusion_probability_range": [
                    round(float(np.min(selected_pi)), 8),
                    round(float(np.max(selected_pi)), 8),
                ],
                "kish_ess": round(design_kish_ess, 6),
                "slots": [
                    {
                        "slot_ordinal": int(slot["slot_ordinal"]),
                        "slot_kind": "resident",
                        "parent_index": int(slot["parent_index"]),
                        "primary_archetype_id": slot[
                            "primary_archetype_id"
                        ],
                        "persona_variant_id": slot["persona_variant_id"],
                        "role": slot["role"],
                        "prompt_sha256": slot["prompt_sha256"],
                        "status": slot["status"],
                    }
                    for slot in private_slots
                ],
            },
        )
    runtime.usage["resident_slots_planned"] += RESIDENT_LLM_BUDGET
    runtime.resident_slots.extend(private_slots)
    posterior_public = None
    posterior_private = None
    completed = 0
    semantics: Mapping[int, Mapping[str, Any]] | None = None
    if execution_provenance in {
        REVIEWED_TRACE_PROVENANCE,
        AUTHORIZED_LIVE_PROVENANCE,
    }:
        if semantic_source is None:
            raise YuLanScaleEngineError(
                "reviewed trace replay requires an exact semantic source"
            )
        decisions = {}
        envelopes: dict[int, TurnEnvelope] = {}

        def acquire_resident(index: int) -> TurnEnvelope:
            rendered = rendered_by_index[index]
            committed_key = ("resident", rendered.prompt_sha256)
            committed_record = (
                committed_semantic_turns.get(committed_key)
                if isinstance(committed_semantic_turns, Mapping)
                else None
            )
            if isinstance(committed_record, Mapping):
                envelope = _committed_envelope(
                    slot_kind="resident",
                    prompt_sha256=rendered.prompt_sha256,
                    record=committed_record,
                    execution_provenance=execution_provenance,
                )
            elif committed_key in committed_semantic_turns:
                acquire_committed = getattr(
                    semantic_source, "acquire_committed", None
                )
                if not callable(acquire_committed):
                    raise YuLanScaleEngineError(
                        "committed resident slot cannot be restored without "
                        "a non-acquiring trace resolver"
                    )
                envelope = acquire_committed(
                    slot_kind="resident",
                    prompt_sha256=rendered.prompt_sha256,
                )
            else:
                acquire_kwargs = {
                    "slot_kind": "resident",
                    "prompt_sha256": rendered.prompt_sha256,
                }
                if execution_provenance == AUTHORIZED_LIVE_PROVENANCE:
                    acquire_kwargs.update(
                        {
                            "rendered": rendered,
                            "validator": (
                                lambda candidate, rendered=rendered,
                                agent=population["agents"][index]:
                                validate_resident_llm_turn_v1(
                                    candidate,
                                    rendered=rendered,
                                    agent=agent,
                                )
                            ),
                        }
                    )
                return semantic_source.acquire(**acquire_kwargs)
            return envelope

        live_workers = (
            min(
                len(activated),
                max(1, int(getattr(semantic_source, "max_concurrency", 1))),
            )
            if execution_provenance == AUTHORIZED_LIVE_PROVENANCE
            else 1
        )
        if live_workers > 1:
            pool = ThreadPoolExecutor(max_workers=live_workers)
            future_by_index = {}
            try:
                future_by_index = {
                    int(index): pool.submit(acquire_resident, int(index))
                    for index in activated
                }
                for index in activated:
                    envelopes[int(index)] = future_by_index[int(index)].result()
            except BaseException:
                for future in future_by_index.values():
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        else:
            for index in activated:
                envelope = acquire_resident(int(index))
                # Preserve replay's fail-fast fingerprint behavior.  Live
                # panels validate after all bounded concurrent requests have
                # joined so state application remains deterministic.
                if execution_provenance != AUTHORIZED_LIVE_PROVENANCE:
                    validate_resident_llm_turn_v1(
                        envelope,
                        rendered=rendered_by_index[int(index)],
                        agent=population["agents"][int(index)],
                    )
                envelopes[int(index)] = envelope

        for ordinal, index in enumerate(activated):
            rendered = rendered_by_index[index]
            committed_key = ("resident", rendered.prompt_sha256)
            envelope = envelopes[int(index)]
            decision = validate_resident_llm_turn_v1(
                envelope,
                rendered=rendered,
                agent=population["agents"][index],
            )
            if (
                artifact_callback is not None
                and committed_key not in committed_semantic_turns
            ):
                attempt_records = (
                    envelope.attempt_records
                    if execution_provenance == AUTHORIZED_LIVE_PROVENANCE
                    else (
                        {
                            "attempt_ordinal": 0,
                            "transport_kind": "reviewed_trace",
                            "status": "succeeded",
                            "request_fingerprint": rendered.prompt_sha256,
                            "response_fingerprint": envelope.response_sha256,
                            "input_tokens": int(envelope.prompt_tokens),
                            "output_tokens": int(envelope.completion_tokens),
                            "validated_decision": (
                                _resident_decision_artifact(decision)
                            ),
                        },
                    )
                )
                for attempt_record in attempt_records:
                    artifact_callback(
                        "attempt",
                        {
                        "scenario_id": scenario_id,
                        "seed": int(seed),
                        "branch": branch,
                        "tick": int(tick),
                        "slot_kind": "resident",
                        "slot_ordinal": int(ordinal),
                            **copy.deepcopy(dict(attempt_record)),
                        },
                    )
            decisions[index] = decision
            envelopes[index] = envelope
        updates = apply_individual_llm_updates_v1(
            population=population,
            state=runtime.state,
            decisions=decisions,
            envelopes=envelopes,
            llm_memories=runtime.llm_memories,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            seed=seed,
        )
        semantics = updates["semantics"]
        posterior_public, posterior_private = estimate_population_from_llm_v1(
            state=runtime.state,
            pps_sample=sample,
            pps_semantics={
                int(index): semantics[int(index)]
                for index in sample["selected"]
            },
            emulator_predictions=emulator_predictions,
        )
        completed = RESIDENT_LLM_BUDGET
        runtime.usage["resident_llm_turns_completed"] += completed
        if execution_provenance == REVIEWED_TRACE_PROVENANCE:
            runtime.usage["trace_replay_turns"] += completed
        runtime.usage["provider_calls"] += sum(
            int(envelope.provider_calls) for envelope in envelopes.values()
        )
        runtime.usage["provider_tokens"] += sum(
            int(envelope.provider_billed_tokens)
            for envelope in envelopes.values()
        )
        runtime.usage["provider_attempts"] += sum(
            int(envelope.provider_attempts)
            for envelope in envelopes.values()
        )
        runtime.usage["cache_hits"] += sum(
            int(envelope.cache_hit) for envelope in envelopes.values()
        )
        runtime.usage["format_repairs"] += sum(
            int(envelope.format_repairs) for envelope in envelopes.values()
        )
        for slot in private_slots:
            slot["status"] = (
                "validated_replay"
                if execution_provenance == REVIEWED_TRACE_PROVENANCE
                else "validated_authorized_live"
            )
    else:
        # Planning a sample must still affect rotation, otherwise all future
        # development rounds would repeatedly plan the same PPS panel.
        for index in sample["selected"]:
            runtime.state.rotation_counts[int(index)] += 1

    for index in activated:
        history = runtime.amortized_histories.setdefault(int(index), [])
        history.append(
            {
                "tick": tick,
                "phase": TIMELINE[tick],
                "source": "llm_behavior_emulator_state",
                "state": _state_summary(runtime.state, int(index)),
            }
        )
        if len(history) > 2:
            del history[:-2]

    public = _public_activation_round(
        sample,
        completed=completed,
        posterior=posterior_public,
    )
    public["anchor_aggregate"] = {
        "panel_size": LONGITUDINAL_ANCHOR_COUNT,
        "semantic_source": execution_provenance,
        "model_state": _aggregate_anchor_state(
            state=runtime.state,
            anchor_indices=anchor_indices,
        ),
    }
    if semantics is None:
        public["emulator_residuals"] = {
            "status": "not_available_without_llm_observations",
            "semantic_source": EMULATOR_ONLY_PROVENANCE,
        }
    else:
        public["emulator_residuals"] = {
            "status": "available_from_llm_semantic_turns",
            "semantic_source": execution_provenance,
            "all_activated": _categorical_emulator_residuals(
                indices=activated,
                semantics=semantics,
                predictions=emulator_predictions,
            ),
            "longitudinal_anchors": _categorical_emulator_residuals(
                indices=anchor_indices,
                semantics=semantics,
                predictions=emulator_predictions,
            ),
            "rotating_pps": _categorical_emulator_residuals(
                indices=sample["selected"],
                semantics=semantics,
                predictions=emulator_predictions,
            ),
        }
    public["activation_round_sha256"] = canonical_hash(public)
    private = {
        **copy.deepcopy(public),
        "anchor_indices": list(anchor_indices),
        "pps_indices": list(sample["selected"]),
        "anchor_identity_sha256": sample["anchor_identity_sha256"],
        "pps_identity_sha256": sample["pps_identity_sha256"],
        "slots": private_slots,
        "posterior": posterior_private,
    }
    runtime.activation_rounds.append(private)
    return public, posterior_public


def _run_governance_decision(
    *,
    runtime: YuLanScaleBranchRuntime,
    scenario_id: str,
    branch: str,
    tick: int,
    seed: int,
    execution_provenance: str,
    semantic_source: SemanticTurnSource | None,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    committed_semantic_turns: CommittedSemanticTurns = frozenset(),
) -> dict[str, Any] | None:
    if branch == "natural":
        return None
    observations = _actor_observations(runtime.state)
    remaining = _resource_view(runtime.state)["remaining"]
    rendered_by_actor = {}
    private_slots = []
    for ordinal, actor_id in enumerate(ACTOR_IDS):
        feasible = [
            action_id
            for action_id in BRANCH_ACTIONS[branch]
            if ACTION_CATALOG[action_id]["actor"] == actor_id
            and float(ACTION_CATALOG[action_id]["cost"])
            <= float(remaining) + 1e-9
        ]
        rendered = render_governance_actor_turn_v1(
            actor_id=actor_id,
            scenario_id=scenario_id,
            branch=branch,
            tick=tick,
            private_observation=observations[actor_id],
            feasible_actions=feasible,
            resource=_resource_view(runtime.state),
        )
        rendered_by_actor[actor_id] = rendered
        private_slots.append(
            {
                "slot_kind": "governance_actor",
                "slot_ordinal": ordinal,
                "actor_id": actor_id,
                "prompt_sha256": rendered.prompt_sha256,
                "status": (
                    "planned_not_acquired"
                    if execution_provenance == EMULATOR_ONLY_PROVENANCE
                    else "planned"
                ),
            }
        )
    runtime.actor_slots.extend(private_slots)
    runtime.usage["governance_actor_slots_planned"] += len(ACTOR_IDS)
    if artifact_callback is not None:
        artifact_callback(
            "actor_round",
            {
                "schema_version": (
                    "campus-pulse-governance-actor-round-private-v1"
                ),
                "scenario_id": scenario_id,
                "seed": int(seed),
                "branch": branch,
                "tick": int(tick),
                "actor_count": len(ACTOR_IDS),
                "stage": "slots_persisted_before_trace_acquire",
                "actors": [
                    {"actor_id": slot["actor_id"]}
                    for slot in private_slots
                ],
                "slots": [
                    {
                        "slot_ordinal": int(slot["slot_ordinal"]),
                        "actor_id": slot["actor_id"],
                        "prompt_sha256": slot["prompt_sha256"],
                        "status": slot["status"],
                    }
                    for slot in private_slots
                ],
                "resource": _resource_view(runtime.state),
            },
        )
    if execution_provenance == EMULATOR_ONLY_PROVENANCE:
        public, _ = play_qre_governance_round(
            state=runtime.state,
            scenario_id=scenario_id,
            branch=branch,
            seed=seed,
            tick=tick,
        )
        result = {
            **copy.deepcopy(public),
            "decision_driver": "llm_behavior_emulator_development",
            "governance_actor_llm_turns_completed": 0,
            "fixture_turns": 0,
        }
        return result
    if semantic_source is None:
        raise YuLanScaleEngineError(
            "reviewed trace replay lacks governance actor traces"
        )
    actor_decisions = {}
    envelopes = {}

    def acquire_actor(actor_id: str) -> TurnEnvelope:
        rendered = rendered_by_actor[actor_id]
        committed_key = ("governance_actor", rendered.prompt_sha256)
        committed_record = (
            committed_semantic_turns.get(committed_key)
            if isinstance(committed_semantic_turns, Mapping)
            else None
        )
        if isinstance(committed_record, Mapping):
            envelope = _committed_envelope(
                slot_kind="governance_actor",
                prompt_sha256=rendered.prompt_sha256,
                record=committed_record,
                execution_provenance=execution_provenance,
            )
        elif committed_key in committed_semantic_turns:
            acquire_committed = getattr(
                semantic_source, "acquire_committed", None
            )
            if not callable(acquire_committed):
                raise YuLanScaleEngineError(
                    "committed actor slot cannot be restored without a "
                    "non-acquiring trace resolver"
                )
            envelope = acquire_committed(
                slot_kind="governance_actor",
                prompt_sha256=rendered.prompt_sha256,
            )
        else:
            acquire_kwargs = {
                "slot_kind": "governance_actor",
                "prompt_sha256": rendered.prompt_sha256,
            }
            if execution_provenance == AUTHORIZED_LIVE_PROVENANCE:
                acquire_kwargs.update(
                    {
                        "rendered": rendered,
                        "validator": (
                            lambda candidate, rendered=rendered:
                            validate_governance_actor_turn_v1(
                                candidate,
                                rendered=rendered,
                            )
                        ),
                    }
                )
            return semantic_source.acquire(**acquire_kwargs)
        return envelope

    actor_items = tuple(rendered_by_actor.items())
    live_workers = (
        min(
            len(actor_items),
            max(1, int(getattr(semantic_source, "max_concurrency", 1))),
        )
        if execution_provenance == AUTHORIZED_LIVE_PROVENANCE
        else 1
    )
    if live_workers > 1:
        pool = ThreadPoolExecutor(max_workers=live_workers)
        future_by_actor = {}
        try:
            future_by_actor = {
                actor_id: pool.submit(acquire_actor, actor_id)
                for actor_id, _rendered in actor_items
            }
            for actor_id, _rendered in actor_items:
                envelopes[actor_id] = future_by_actor[actor_id].result()
        except BaseException:
            for future in future_by_actor.values():
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    else:
        for actor_id, _rendered in actor_items:
            envelope = acquire_actor(actor_id)
            if execution_provenance != AUTHORIZED_LIVE_PROVENANCE:
                validate_governance_actor_turn_v1(
                    envelope,
                    rendered=rendered_by_actor[actor_id],
                )
            envelopes[actor_id] = envelope

    for ordinal, (actor_id, rendered) in enumerate(actor_items):
        committed_key = ("governance_actor", rendered.prompt_sha256)
        envelope = envelopes[actor_id]
        decision = validate_governance_actor_turn_v1(
            envelope,
            rendered=rendered,
        )
        if (
            artifact_callback is not None
            and committed_key not in committed_semantic_turns
        ):
            attempt_records = (
                envelope.attempt_records
                if execution_provenance == AUTHORIZED_LIVE_PROVENANCE
                else (
                    {
                        "attempt_ordinal": 0,
                        "transport_kind": "reviewed_trace",
                        "status": "succeeded",
                        "request_fingerprint": rendered.prompt_sha256,
                        "response_fingerprint": envelope.response_sha256,
                        "input_tokens": int(envelope.prompt_tokens),
                        "output_tokens": int(envelope.completion_tokens),
                        "validated_decision": (
                            _actor_decision_artifact(decision)
                        ),
                    },
                )
            )
            for attempt_record in attempt_records:
                artifact_callback(
                    "attempt",
                    {
                    "scenario_id": scenario_id,
                    "seed": int(seed),
                    "branch": branch,
                    "tick": int(tick),
                    "slot_kind": "governance_actor",
                    "slot_ordinal": int(ordinal),
                        **copy.deepcopy(dict(attempt_record)),
                    },
                )
        actor_decisions[actor_id] = decision
        envelopes[actor_id] = envelope
    public, _ = play_llm_governance_round_v1(
        state=runtime.state,
        actor_decisions=actor_decisions,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
    )
    runtime.usage["governance_actor_llm_turns_completed"] += len(ACTOR_IDS)
    if execution_provenance == REVIEWED_TRACE_PROVENANCE:
        runtime.usage["trace_replay_turns"] += len(ACTOR_IDS)
    runtime.usage["provider_calls"] += sum(
        int(envelope.provider_calls) for envelope in envelopes.values()
    )
    runtime.usage["provider_tokens"] += sum(
        int(envelope.provider_billed_tokens)
        for envelope in envelopes.values()
    )
    runtime.usage["provider_attempts"] += sum(
        int(envelope.provider_attempts) for envelope in envelopes.values()
    )
    runtime.usage["cache_hits"] += sum(
        int(envelope.cache_hit) for envelope in envelopes.values()
    )
    runtime.usage["format_repairs"] += sum(
        int(envelope.format_repairs) for envelope in envelopes.values()
    )
    for slot in private_slots:
        slot["status"] = (
            "validated_replay"
            if execution_provenance == REVIEWED_TRACE_PROVENANCE
            else "validated_authorized_live"
        )
    return {
        **copy.deepcopy(public),
        "decision_driver": (
            "reviewed_llm_trace_plus_constrained_qre"
            if execution_provenance == REVIEWED_TRACE_PROVENANCE
            else "authorized_live_llm_plus_constrained_qre"
        ),
        "governance_actor_llm_turns_completed": len(ACTOR_IDS),
        "fixture_turns": 0,
    }


def _emit_message_batch_artifact(
    *,
    runtime: YuLanScaleBranchRuntime,
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ),
) -> None:
    """Persist only aggregate private message accounting before checkpoint."""

    if artifact_callback is None:
        return
    created = [
        message
        for message in runtime.state.ledger.messages
        if int(message.created_tick) == int(tick)
    ]
    artifact_callback(
        "message_batch",
        {
            "schema_version": (
                "campus-pulse-llm-message-batch-private-v1"
            ),
            "scenario_id": scenario_id,
            "seed": int(seed),
            "branch": branch,
            "tick": int(tick),
            "message_count": len(created),
            "superseded_count": sum(
                1 for message in created if message.action == "suppressed"
            ),
            "provenance_counts": {
                provenance: sum(
                    1
                    for message in created
                    if str(message.provenance) == provenance
                )
                for provenance in sorted(
                    {str(message.provenance) for message in created}
                )
            },
        },
    )


def encode_yulan_scale_checkpoint_v3(
    *,
    runtime: YuLanScaleBranchRuntime,
    population_sha256: str,
    graph_sha256: str,
    scenario_id: str,
    branch: str,
    seed: int,
    tick: int,
    previous_checkpoint_sha256: str,
    visual_frame_prefix: Sequence[Mapping[str, Any]] | None = None,
    scenario_seed_work_unit_offset: int | None = None,
) -> dict[str, Any]:
    """Encode a complete, private, deterministic domain checkpoint."""

    kernel = encode_governance_checkpoint_v2(
        population_sha256=population_sha256,
        graph_sha256=graph_sha256,
        scenario_id=scenario_id,
        branch=branch,
        seed=seed,
        tick=tick,
        state=runtime.state,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
    )
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "population_sha256": population_sha256,
        "graph_sha256": graph_sha256,
        "scenario_id": scenario_id,
        "branch": branch,
        "seed": int(seed),
        "tick": int(tick),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "kernel_checkpoint": kernel,
        "llm_memories": copy.deepcopy(runtime.llm_memories),
        "amortized_histories": copy.deepcopy(runtime.amortized_histories),
        "activation_rounds": copy.deepcopy(runtime.activation_rounds),
        "resident_slots": copy.deepcopy(runtime.resident_slots),
        "actor_slots": copy.deepcopy(runtime.actor_slots),
        "timeline": copy.deepcopy(runtime.timeline),
        # Canonical display frames are immutable public artifacts, but their
        # committed prefix is needed for fail-safe branch recovery.  Only the
        # canonical display seed captures frames, keeping the compressed
        # checkpoint below the frozen 8 MB bound.
        "visual_frame_prefix": copy.deepcopy(
            list(
                runtime.visual_frames
                if visual_frame_prefix is None
                else visual_frame_prefix
            )
        ),
        "usage": dict(runtime.usage),
    }
    if scenario_seed_work_unit_offset is not None:
        body["aggregate_cursor"] = build_yulan_scale_aggregate_cursor(
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            tick=tick,
            scenario_seed_work_unit_offset=scenario_seed_work_unit_offset,
        )
    body["checkpoint_sha256"] = canonical_hash(body)
    return body


def decode_yulan_scale_checkpoint_v3(
    checkpoint: Mapping[str, Any],
    *,
    population_sha256: str,
    graph_sha256: str,
) -> YuLanScaleBranchRuntime:
    """Restore an exact branch runtime from a v3 domain checkpoint."""

    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("population_sha256") != population_sha256
        or checkpoint.get("graph_sha256") != graph_sha256
    ):
        raise YuLanScaleEngineError("YuLan-Scale checkpoint identity drifted")
    expected = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_sha256"
    }
    if checkpoint.get("checkpoint_sha256") != canonical_hash(expected):
        raise YuLanScaleEngineError("YuLan-Scale checkpoint hash does not match")
    cursor = checkpoint.get("aggregate_cursor")
    if cursor is not None:
        if not isinstance(cursor, Mapping):
            raise YuLanScaleEngineError("aggregate cursor is invalid")
        _validate_yulan_scale_aggregate_cursor(
            cursor,
            scenario_id=str(checkpoint["scenario_id"]),
            seed=int(checkpoint["seed"]),
            branch=str(checkpoint["branch"]),
            tick=int(checkpoint["tick"]),
            scenario_seed_work_unit_offset=int(
                cursor.get("scenario_seed_work_unit_offset", -1)
            ),
        )
    state = decode_governance_checkpoint_v2(
        checkpoint["kernel_checkpoint"],
        population_sha256=population_sha256,
        graph_sha256=graph_sha256,
    )
    return YuLanScaleBranchRuntime(
        state=state,
        llm_memories={
            int(key): copy.deepcopy(value)
            for key, value in checkpoint["llm_memories"].items()
        },
        amortized_histories={
            int(key): copy.deepcopy(value)
            for key, value in checkpoint["amortized_histories"].items()
        },
        activation_rounds=copy.deepcopy(checkpoint["activation_rounds"]),
        resident_slots=copy.deepcopy(checkpoint["resident_slots"]),
        actor_slots=copy.deepcopy(checkpoint["actor_slots"]),
        timeline=copy.deepcopy(checkpoint["timeline"]),
        visual_frames=copy.deepcopy(
            checkpoint.get("visual_frame_prefix", [])
        ),
        usage={
            str(key): int(value)
            for key, value in checkpoint["usage"].items()
        },
    )


def resume_yulan_scale_branch_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    target_branch: str | None = None,
    execution_provenance: str = EMULATOR_ONLY_PROVENANCE,
    semantic_source: SemanticTurnSource | None = None,
    emulator_predictor: Callable[..., Mapping[str, Any]] | None = None,
    emulator_release: Mapping[str, Any] | None = None,
    strict_scale: bool = True,
    target_tick: int = len(TIMELINE) - 1,
    capture_visual_frames: bool = False,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    committed_semantic_turns: CommittedSemanticTurns = frozenset(),
    progress_callback: (
        Callable[[int, Mapping[str, Any] | None], None] | None
    ) = None,
    completed_work_units_offset: int = 0,
    scenario_seed_work_unit_offset: int = 0,
    cancellation_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Continue one branch from the next tick after a committed checkpoint.

    The function never replays the checkpoint tick.  Therefore committed LLM
    innovations, messages and governance decisions are not reacquired or
    reintegrated.  Earlier timeline/visual artifacts remain the responsibility
    of the runtime artifact store; returned frames cover only resumed ticks.
    """

    agents = population.get("agents")
    if not isinstance(agents, list):
        raise YuLanScaleEngineError("resume population is invalid")
    if strict_scale:
        if len(agents) != FORMAL_PARENT_COUNT or emulator_predictor is None:
            raise YuLanScaleEngineError(
                "formal resume requires 1,000 agents and a frozen emulator"
            )
        _validated_emulator_release_identity(emulator_release)
    if execution_provenance not in SUPPORTED_PROVENANCES:
        raise YuLanScaleEngineError("resume provenance is unsupported")
    if execution_provenance in {
        REVIEWED_TRACE_PROVENANCE,
        AUTHORIZED_LIVE_PROVENANCE,
    } and (
        semantic_source is None
        or getattr(semantic_source, "provenance", None)
        != execution_provenance
    ):
        raise YuLanScaleEngineError(
            "semantic resume requires its exact provenance-bound source"
        )
    if execution_provenance == EMULATOR_ONLY_PROVENANCE and semantic_source:
        raise YuLanScaleEngineError(
            "development resume cannot acquire semantic turns"
        )
    scenario_id = str(checkpoint.get("scenario_id") or "")
    seed = int(checkpoint.get("seed", -1))
    checkpoint_branch = str(checkpoint.get("branch") or "")
    if checkpoint_branch == "shared_baseline":
        if target_branch not in BRANCH_IDS:
            raise YuLanScaleEngineError(
                "shared baseline resume requires a target branch"
            )
        branch = str(target_branch)
    else:
        branch = checkpoint_branch
        if branch not in BRANCH_IDS or (
            target_branch is not None and target_branch != branch
        ):
            raise YuLanScaleEngineError("resume branch identity drifted")
    start_tick = int(checkpoint.get("tick", -1)) + 1
    if (
        start_tick < 3
        or target_tick >= len(TIMELINE)
        or target_tick < start_tick - 1
    ):
        raise YuLanScaleEngineError("resume tick range is invalid")

    scenario_population = scenario_population_view(
        population,
        scenario_id=scenario_id,
    )
    graph_arrays = compile_message_exposure_graph_v2(
        scenario_population,
        graph,
    )
    runtime = decode_yulan_scale_checkpoint_v3(
        checkpoint,
        population_sha256=scenario_population["population_sha256"],
        graph_sha256=graph_arrays.graph_sha256,
    )
    if checkpoint_branch == "shared_baseline" and runtime.visual_frames:
        runtime.visual_frames = [
            {**copy.deepcopy(frame), "branch": branch}
            for frame in runtime.visual_frames
        ]
    # The checkpoint binds the complete canonical frame prefix.  Resumed
    # frames append to this verified prefix instead of regenerating it.
    features = _parent_features(scenario_population, scenario_id)
    if not runtime.activation_rounds:
        raise YuLanScaleEngineError("resume checkpoint lacks the anchor panel")
    anchor_indices = tuple(
        int(index)
        for index in runtime.activation_rounds[0]["anchor_indices"]
    )
    if len(anchor_indices) != LONGITUDINAL_ANCHOR_COUNT:
        raise YuLanScaleEngineError("resume anchor panel drifted")
    last_point = runtime.timeline[-1] if runtime.timeline else None
    previous_checkpoint_sha = str(checkpoint["checkpoint_sha256"])
    emitted_checkpoints = []
    completed = int(completed_work_units_offset)
    for tick in range(start_tick, target_tick + 1):
        if cancellation_probe is not None and cancellation_probe():
            raise YuLanScaleEngineCancelled(
                f"cancelled at {scenario_id}/{seed}/{branch}/T{tick}"
            )
        diffusion, predictions = _advance_unified_emulator_tick(
            population=scenario_population,
            graph=graph_arrays,
            state=runtime.state,
            features=features,
            scenario_id=scenario_id,
            branch=branch,
            seed=seed,
            tick=tick,
            emulator_predictor=emulator_predictor,
        )
        runtime.usage["emulator_population_updates"] += len(
            scenario_population["agents"]
        )
        posterior = None
        activation_public = None
        if tick in PROBE_TICKS:
            activation_public, posterior = _run_resident_activation(
                runtime=runtime,
                population=scenario_population,
                features=features,
                anchor_indices=anchor_indices,
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
                seed=seed,
                execution_provenance=execution_provenance,
                semantic_source=semantic_source,
                emulator_predictions=predictions,
                artifact_callback=artifact_callback,
                committed_semantic_turns=committed_semantic_turns,
            )
        _update_actor_beliefs(
            state=runtime.state,
            diffusion=diffusion,
            probe=_v2_probe_view(posterior),
            prior_public=last_point,
        )
        decision = None
        if tick in DECISION_TICKS:
            decision = _run_governance_decision(
                runtime=runtime,
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
                seed=seed,
                execution_provenance=execution_provenance,
                semantic_source=semantic_source,
                artifact_callback=artifact_callback,
                committed_semantic_turns=committed_semantic_turns,
            )
        point = _timeline_point(
            population=scenario_population,
            state=runtime.state,
            features=features,
            tick=tick,
            diffusion=diffusion,
            probe=_v2_probe_view(posterior),
        )
        point["llm_activation"] = activation_public
        point["governance_decision"] = decision
        runtime.timeline.append(point)
        last_point = point
        if capture_visual_frames:
            _capture_frame(
                runtime,
                scenario_id=scenario_id,
                branch=branch,
                tick=tick,
            )
        _emit_message_batch_artifact(
            runtime=runtime,
            scenario_id=scenario_id,
            branch=branch,
            seed=seed,
            tick=tick,
            artifact_callback=artifact_callback,
        )
        tick_checkpoint = None
        if tick in CHECKPOINT_TICKS:
            tick_checkpoint = encode_yulan_scale_checkpoint_v3(
                runtime=runtime,
                population_sha256=scenario_population["population_sha256"],
                graph_sha256=graph_arrays.graph_sha256,
                scenario_id=scenario_id,
                branch=branch,
                seed=seed,
                tick=tick,
                previous_checkpoint_sha256=previous_checkpoint_sha,
                scenario_seed_work_unit_offset=(
                    scenario_seed_work_unit_offset
                ),
            )
            previous_checkpoint_sha = tick_checkpoint["checkpoint_sha256"]
            runtime.state.previous_checkpoint_sha256 = (
                previous_checkpoint_sha
            )
            emitted_checkpoints.append(tick_checkpoint)
            if checkpoint_callback is not None:
                checkpoint_callback(tick_checkpoint)
        completed += 1
        if progress_callback is not None:
            progress_callback(
                completed,
                _checkpoint_index(tick_checkpoint)
                if tick_checkpoint is not None
                else None,
            )
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "branch": branch,
        "resumed_from_tick": start_tick - 1,
        "completed_through_tick": target_tick,
        "completed_work_units": completed,
        "timeline": copy.deepcopy(runtime.timeline),
        "visual_frames": copy.deepcopy(runtime.visual_frames),
        "resource": _resource_view(runtime.state),
        "checkpoint_index": [
            _checkpoint_index(value) for value in emitted_checkpoints
        ],
        "_runtime": runtime,
        "_checkpoints": emitted_checkpoints,
    }


def _capture_frame(
    runtime: YuLanScaleBranchRuntime,
    *,
    scenario_id: str,
    branch: str,
    tick: int,
) -> None:
    frame = encode_visual_frame_v2(
        state=runtime.state,
        scenario_id=scenario_id,
        branch=branch,
        tick=tick,
    )
    # The kernel matrix is parent-major for efficient generation.  Public
    # assets must not reveal the ten-particle parent grouping, so bind a
    # deterministic display permutation before publishing the frame.
    compressed = base64.b64decode(frame["payload"], validate=True)
    raw = zlib.decompress(compressed)
    matrix = np.frombuffer(raw, dtype=np.uint8).reshape(frame["shape"])
    display_seed = int(
        canonical_hash(
            [
                "yulan-scale-visual-permutation-v1",
                scenario_id,
                branch,
                tick,
                frame["payload_sha256"],
            ]
        )[:16],
        16,
    )
    permutation = np.random.default_rng(display_seed).permutation(len(matrix))
    anonymized_raw = matrix[permutation].tobytes(order="C")
    anonymized_compressed = zlib.compress(anonymized_raw, level=9)
    public_frame = {
        **frame,
        "schema_version": (
            "campus-pulse-llm-population-visualization-frame-v3"
        ),
        "kernel_frame_schema_version": frame["schema_version"],
        "particle_order": "deterministically_shuffled_no_parent_mapping",
        "content_sha256": hashlib.sha256(anonymized_raw).hexdigest(),
        "payload_sha256": hashlib.sha256(
            anonymized_compressed
        ).hexdigest(),
        "payload": base64.b64encode(anonymized_compressed).decode("ascii"),
    }
    runtime.visual_frames.append(public_frame)


def _checkpoint_index(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scenario_id": checkpoint["scenario_id"],
        "branch": checkpoint["branch"],
        "seed": checkpoint["seed"],
        "tick": checkpoint["tick"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "previous_checkpoint_sha256": checkpoint[
            "previous_checkpoint_sha256"
        ],
    }


def _run_scenario_seed(
    *,
    unified_population: Mapping[str, Any],
    graph_source: Mapping[str, Any],
    scenario_id: str,
    seed: int,
    execution_provenance: str,
    semantic_source: SemanticTurnSource | None,
    emulator_predictor: Callable[..., Mapping[str, Any]] | None,
    capture_visual_frames: bool,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ),
    committed_semantic_turns: CommittedSemanticTurns,
    retain_checkpoint_payloads: bool,
    cancellation_probe: Callable[[], bool] | None,
    unit_completed_callback: (
        Callable[[Mapping[str, Any] | None], None] | None
    ),
    scenario_seed_work_unit_offset: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    population = scenario_population_view(
        unified_population,
        scenario_id=scenario_id,
    )
    graph = compile_message_exposure_graph_v2(population, graph_source)
    features = _parent_features(population, scenario_id)
    anchor_indices = _select_anchor_indices(
        unified_population,
        scenario_id=scenario_id,
        seed=seed,
    )
    shared = YuLanScaleBranchRuntime(
        state=initial_branch_state(
            population,
            scenario_id=scenario_id,
            seed=seed,
        )
    )
    checkpoint_records = []
    public_checkpoints = []
    last_point = None
    last_checkpoint_sha = "0" * 64
    shared_activation_public = None
    for tick in range(3):
        if cancellation_probe is not None and cancellation_probe():
            raise YuLanScaleEngineCancelled(
                f"cancelled at {scenario_id}/{seed}/shared_baseline/T{tick}"
            )
        diffusion, predictions = _advance_unified_emulator_tick(
            population=population,
            graph=graph,
            state=shared.state,
            features=features,
            scenario_id=scenario_id,
            branch="natural",
            seed=seed,
            tick=tick,
            emulator_predictor=emulator_predictor,
        )
        shared.usage["emulator_population_updates"] += len(population["agents"])
        posterior = None
        if tick == 2:
            shared_activation_public, posterior = _run_resident_activation(
                runtime=shared,
                population=population,
                features=features,
                anchor_indices=anchor_indices,
                scenario_id=scenario_id,
                branch="shared_baseline",
                tick=tick,
                seed=seed,
                execution_provenance=execution_provenance,
                semantic_source=semantic_source,
                emulator_predictions=predictions,
                artifact_callback=artifact_callback,
                committed_semantic_turns=committed_semantic_turns,
            )
        _update_actor_beliefs(
            state=shared.state,
            diffusion=diffusion,
            probe=_v2_probe_view(posterior),
            prior_public=last_point,
        )
        point = _timeline_point(
            population=population,
            state=shared.state,
            features=features,
            tick=tick,
            diffusion=diffusion,
            probe=_v2_probe_view(posterior),
        )
        point["llm_activation"] = shared_activation_public if tick == 2 else None
        shared.timeline.append(point)
        last_point = point
        if capture_visual_frames:
            _capture_frame(
                shared,
                scenario_id=scenario_id,
                branch="shared_baseline",
                tick=tick,
            )
        _emit_message_batch_artifact(
            runtime=shared,
            scenario_id=scenario_id,
            branch="shared_baseline",
            seed=seed,
            tick=tick,
            artifact_callback=artifact_callback,
        )
        if tick < 2 and unit_completed_callback is not None:
            unit_completed_callback(None)
    checkpoint = encode_yulan_scale_checkpoint_v3(
        runtime=shared,
        population_sha256=population["population_sha256"],
        graph_sha256=graph.graph_sha256,
        scenario_id=scenario_id,
        branch="shared_baseline",
        seed=seed,
        tick=2,
        previous_checkpoint_sha256=last_checkpoint_sha,
        scenario_seed_work_unit_offset=scenario_seed_work_unit_offset,
    )
    last_checkpoint_sha = checkpoint["checkpoint_sha256"]
    shared.state.previous_checkpoint_sha256 = last_checkpoint_sha
    if checkpoint_callback is not None:
        checkpoint_callback(checkpoint)
    if retain_checkpoint_payloads:
        checkpoint_records.append(checkpoint)
    public_checkpoints.append(_checkpoint_index(checkpoint))
    if unit_completed_callback is not None:
        unit_completed_callback(checkpoint)

    branches = {}
    private_branches = {}
    for branch in BRANCH_IDS:
        runtime = shared.clone()
        runtime.visual_frames = []
        branch_previous_checkpoint = last_checkpoint_sha
        last_point = runtime.timeline[-1]
        for tick in range(3, len(TIMELINE)):
            if cancellation_probe is not None and cancellation_probe():
                raise YuLanScaleEngineCancelled(
                    f"cancelled at {scenario_id}/{seed}/{branch}/T{tick}"
                )
            diffusion, predictions = _advance_unified_emulator_tick(
                population=population,
                graph=graph,
                state=runtime.state,
                features=features,
                scenario_id=scenario_id,
                branch=branch,
                seed=seed,
                tick=tick,
                emulator_predictor=emulator_predictor,
            )
            runtime.usage["emulator_population_updates"] += len(
                population["agents"]
            )
            posterior = None
            activation_public = None
            if tick in PROBE_TICKS:
                activation_public, posterior = _run_resident_activation(
                    runtime=runtime,
                    population=population,
                    features=features,
                    anchor_indices=anchor_indices,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    seed=seed,
                    execution_provenance=execution_provenance,
                    semantic_source=semantic_source,
                    emulator_predictions=predictions,
                    artifact_callback=artifact_callback,
                    committed_semantic_turns=committed_semantic_turns,
                )
            _update_actor_beliefs(
                state=runtime.state,
                diffusion=diffusion,
                probe=_v2_probe_view(posterior),
                prior_public=last_point,
            )
            decision = None
            if tick in DECISION_TICKS:
                decision = _run_governance_decision(
                    runtime=runtime,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                    seed=seed,
                    execution_provenance=execution_provenance,
                    semantic_source=semantic_source,
                    artifact_callback=artifact_callback,
                    committed_semantic_turns=committed_semantic_turns,
                )
            point = _timeline_point(
                population=population,
                state=runtime.state,
                features=features,
                tick=tick,
                diffusion=diffusion,
                probe=_v2_probe_view(posterior),
            )
            point["llm_activation"] = activation_public
            point["governance_decision"] = decision
            runtime.timeline.append(point)
            last_point = point
            if capture_visual_frames:
                _capture_frame(
                    runtime,
                    scenario_id=scenario_id,
                    branch=branch,
                    tick=tick,
                )
            _emit_message_batch_artifact(
                runtime=runtime,
                scenario_id=scenario_id,
                branch=branch,
                seed=seed,
                tick=tick,
                artifact_callback=artifact_callback,
            )
            tick_checkpoint = None
            if tick in CHECKPOINT_TICKS:
                checkpoint = encode_yulan_scale_checkpoint_v3(
                    runtime=runtime,
                    population_sha256=population["population_sha256"],
                    graph_sha256=graph.graph_sha256,
                    scenario_id=scenario_id,
                    branch=branch,
                    seed=seed,
                    tick=tick,
                    previous_checkpoint_sha256=branch_previous_checkpoint,
                    visual_frame_prefix=[
                        {
                            **copy.deepcopy(frame),
                            "branch": branch,
                        }
                        for frame in shared.visual_frames
                    ]
                    + copy.deepcopy(runtime.visual_frames),
                    scenario_seed_work_unit_offset=(
                        scenario_seed_work_unit_offset
                    ),
                )
                branch_previous_checkpoint = checkpoint["checkpoint_sha256"]
                runtime.state.previous_checkpoint_sha256 = (
                    branch_previous_checkpoint
                )
                if checkpoint_callback is not None:
                    checkpoint_callback(checkpoint)
                if retain_checkpoint_payloads:
                    checkpoint_records.append(checkpoint)
                public_checkpoints.append(_checkpoint_index(checkpoint))
                tick_checkpoint = checkpoint
            if unit_completed_callback is not None:
                unit_completed_callback(tick_checkpoint)
        branches[branch] = {
            "timeline": copy.deepcopy(runtime.timeline),
            "game_rounds": copy.deepcopy(runtime.state.game_rounds),
            "resource": _resource_view(runtime.state),
            # Each branch is a complete 24-tick visual trajectory.  Baseline
            # particle bytes are shared, but branch metadata is explicit so
            # consumers never need to join a sixth pseudo-branch.
            "visual_frames": [
                {**copy.deepcopy(frame), "branch": branch}
                for frame in shared.visual_frames
            ]
            + copy.deepcopy(runtime.visual_frames),
        }
        private_branches[branch] = {
            "activation_rounds": copy.deepcopy(runtime.activation_rounds),
            "resident_slots": copy.deepcopy(runtime.resident_slots),
            "actor_slots": copy.deepcopy(runtime.actor_slots),
            "usage": dict(runtime.usage),
            "anchor_indices": list(anchor_indices),
            "shared_baseline_sha256": canonical_hash(runtime.timeline[:3]),
        }
    seed_run = {
        "seed": int(seed),
        "shared_baseline_timeline": copy.deepcopy(shared.timeline),
        "branches": branches,
    }
    private = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "population_sha256": population["population_sha256"],
        "graph_sha256": graph.graph_sha256,
        "anchor_indices": list(anchor_indices),
        "anchor_identity_sha256": (
            shared.activation_rounds[0]["anchor_identity_sha256"]
        ),
        "shared_baseline_sha256": canonical_hash(shared.timeline),
        "shared_activation": copy.deepcopy(shared.activation_rounds[0]),
        "shared_visual_frames": copy.deepcopy(shared.visual_frames),
        "shared_usage": dict(shared.usage),
        "branches": private_branches,
        "checkpoints": checkpoint_records,
    }
    public = {
        "seed_run": seed_run,
        "checkpoint_index": public_checkpoints,
    }
    return public, private


def _sum_usage(private_runs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "resident_slots_planned": 0,
        "resident_llm_turns_completed": 0,
        "governance_actor_slots_planned": 0,
        "governance_actor_llm_turns_completed": 0,
        "trace_replay_turns": 0,
        "provider_calls": 0,
        "provider_tokens": 0,
        "provider_attempts": 0,
        "cache_hits": 0,
        "format_repairs": 0,
        "fixture_turns": 0,
        "emulator_population_updates": 0,
    }
    # Shared baseline work is copied into each branch runtime. Count it once,
    # then count only the post-baseline delta for all five branches.
    for run in private_runs:
        shared_usage = {
            key: int(run["shared_usage"].get(key, 0))
            for key in totals
        }
        for key in totals:
            if key != "emulator_population_updates":
                totals[key] += shared_usage[key]
        for branch in BRANCH_IDS:
            usage = run["branches"][branch]["usage"]
            for key in totals:
                if key == "emulator_population_updates":
                    continue
                baseline_value = shared_usage.get(key, 0)
                totals[key] += int(usage.get(key, 0)) - baseline_value
    return totals


def _public_llm_analysis_sections(
    private_runs: Sequence[Mapping[str, Any]],
    *,
    execution_provenance: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchor_trajectories = []
    pps_posteriors = []
    residual_rows = []
    for run in private_runs:
        round_rows = [
            ("shared_baseline", run["shared_activation"])
        ]
        for branch in BRANCH_IDS:
            round_rows.extend(
                (branch, round_)
                for round_ in run["branches"][branch][
                    "activation_rounds"
                ][1:]
            )
        for branch, round_ in round_rows:
            anchor_trajectories.append(
                {
                    "scenario_id": run["scenario_id"],
                    "seed": int(run["seed"]),
                    "branch": branch,
                    "tick": int(round_["tick"]),
                    "phase": TIMELINE[int(round_["tick"])],
                    "panel_size": LONGITUDINAL_ANCHOR_COUNT,
                    "semantic_source": round_["anchor_aggregate"][
                        "semantic_source"
                    ],
                    "aggregate_state": copy.deepcopy(
                        round_["anchor_aggregate"]["model_state"]
                    ),
                }
            )
            residual = round_["emulator_residuals"]
            if residual.get("status") in {
                "available_from_reviewed_llm_trace",
                "available_from_llm_semantic_turns",
            }:
                residual_rows.append(residual["all_activated"])
            posterior = round_.get("posterior")
            if isinstance(posterior, Mapping):
                pps_posteriors.append(
                    {
                        "scenario_id": run["scenario_id"],
                        "seed": int(run["seed"]),
                        "branch": branch,
                        "tick": int(round_["tick"]),
                        "budget": int(posterior["budget"]),
                        "kish_ess": float(posterior["kish_ess"]),
                        "inclusion_probability_range": copy.deepcopy(
                            posterior["inclusion_probability_range"]
                        ),
                        "corrected": copy.deepcopy(posterior["corrected"]),
                        "partial_pooling": copy.deepcopy(
                            posterior["partial_pooling"]
                        ),
                    }
                )
    if not residual_rows:
        residual_summary = {
            "status": "not_available_without_llm_observations",
            "semantic_source": execution_provenance,
            "round_count": 0,
        }
    else:
        families = ("action", "stance", "topic_id")
        residual_summary = {
            "status": "available_from_llm_semantic_turns",
            "semantic_source": execution_provenance,
            "round_count": len(residual_rows),
            "all_activated": {
                family: {
                    "mean_share_mae": round(
                        float(
                            np.mean(
                                [
                                    row[family]["share_mae"]
                                    for row in residual_rows
                                ]
                            )
                        ),
                        6,
                    ),
                    "mean_one_minus_assigned_probability": round(
                        float(
                            np.mean(
                                [
                                    row[family][
                                        "mean_one_minus_assigned_probability"
                                    ]
                                    for row in residual_rows
                                ]
                            )
                        ),
                        6,
                    ),
                }
                for family in families
            },
        }
    return anchor_trajectories, pps_posteriors, residual_summary


def _assemble_yulan_scale_result(
    *,
    population: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    scenario_ids: Sequence[str],
    seeds: Sequence[int],
    execution_provenance: str,
    emulator_identity: Mapping[str, Any],
    scenario_public: Mapping[str, Any],
    all_private_runs: Sequence[Mapping[str, Any]],
    all_checkpoint_indices: Sequence[Mapping[str, Any]],
    visualization_frames: Sequence[Mapping[str, Any]],
    completed_work_units: int,
) -> dict[str, Any]:
    """Assemble the exact public result from complete scenario/seed shards."""

    workload = build_yulan_scale_workload_summary(
        scenario_count=len(scenario_ids),
        seed_count=len(seeds),
        parent_count=len(agents),
    )
    if completed_work_units != workload["population_branch_ticks"]:
        raise YuLanScaleEngineError(
            "completed population-branch tick count drifted"
        )
    usage = _sum_usage(all_private_runs)
    usage["parent_emulator_updates"] = (
        workload["population_branch_ticks"] * len(agents)
    )
    usage["particle_updates"] = workload["particle_updates"]
    usage["emulator_population_updates"] = workload["particle_updates"]
    usage["real_governance_actions"] = 0
    usage["fixture_turns"] = 0
    (
        anchor_trajectories,
        pps_posteriors,
        emulator_residuals,
    ) = _public_llm_analysis_sections(
        all_private_runs,
        execution_provenance=execution_provenance,
    )
    public = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "execution_mode": EXECUTION_MODE,
        "execution_provenance": execution_provenance,
        "publication_eligible": (
            execution_provenance in {
                REVIEWED_TRACE_PROVENANCE,
                AUTHORIZED_LIVE_PROVENANCE,
            }
            and usage["resident_llm_turns_completed"]
            == workload["resident_llm_slots"]
            and usage["governance_actor_llm_turns_completed"]
            == workload["governance_actor_llm_slots"]
            and usage["fixture_turns"] == 0
        ),
        "population_identity": {
            "schema_version": population.get("schema_version"),
            "population_sha256": population.get("population_sha256"),
            "llm_agent_count": len(agents),
            "particles_per_agent": PARTICLES_PER_PARENT,
            "particle_count": len(agents) * PARTICLES_PER_PARENT,
            "evidence_subject_count": len(agents),
        },
        "llm_behavior_emulator": copy.deepcopy(dict(emulator_identity)),
        "workload": workload,
        "usage": usage,
        "scenarios": copy.deepcopy(dict(scenario_public)),
        "anchor_trajectories": anchor_trajectories,
        "pps_posteriors": pps_posteriors,
        "emulator_residuals": emulator_residuals,
        "result_view_contract": {
            "timeline_aggregates": "scenarios.*.branches.*.timeline",
            "simulated_speaker_conditionals": (
                "pps_posteriors.*.partial_pooling"
            ),
            "population_posterior": "pps_posteriors.*.corrected",
            "anchor_aggregate_trajectories": "anchor_trajectories",
            "llm_emulator_residuals": "emulator_residuals",
        },
        "checkpoint_index": copy.deepcopy(list(all_checkpoint_indices)),
        "visualization": {
            "schema_version": VISUALIZATION_SCHEMA_VERSION,
            "canonical_seed": int(seeds[0]),
            "frame_count": len(visualization_frames),
            "frames": copy.deepcopy(list(visualization_frames)),
            "statistics_use_all_paired_seeds": True,
        },
        "result_layers": [
            "all_1000_episode_model_state",
            "simulated_public_speaker_conditional_opinion",
            "rotating_pps_population_posterior",
            "governance_actor_limited_observation",
            "longitudinal_anchor_trajectories",
            "llm_emulator_residuals",
            "paired_policy_branch_differences",
            "separate_particle_pps_and_seed_intervals",
        ],
        "audit": {
            "provider_calls": usage["provider_calls"],
            "fixture_turns": 0,
            "real_governance_actions": 0,
            "sample_identity_public": False,
            "record_level_edges_public": False,
            "free_text_emulator_output_public": False,
        },
        "non_claims": [
            "The 1,000 units are synthetic LLM agents grounded by reviewed forum-behaviour episodes, not identified students.",
            "Particles are state-uncertainty hypotheses, not people or evidence subjects.",
            "The exposure graph, propagation speed, governance actors, and policy effects are model-conditional synthetic mechanisms.",
            "Development emulator runs are not LLM evidence and cannot be published as a formal trace result.",
            "Results are not causal policy effects, campus-wide opinion, or real governance advice.",
        ],
    }
    public["result_sha256"] = canonical_hash(public)
    return {
        **public,
        "_runtime_private": {
            "scenario_seed_runs": copy.deepcopy(list(all_private_runs)),
        },
    }


def run_yulan_scale_engine(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    execution_provenance: str = EMULATOR_ONLY_PROVENANCE,
    semantic_source: SemanticTurnSource | None = None,
    emulator_predictor: Callable[..., Mapping[str, Any]] | None = None,
    emulator_release: Mapping[str, Any] | None = None,
    scenario_ids: Sequence[str] = SCENARIO_IDS,
    seeds: Sequence[int] = CANONICAL_SEEDS,
    strict_scale: bool = True,
    capture_visual_frames: bool = True,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    committed_semantic_turns: CommittedSemanticTurns = frozenset(),
    retain_checkpoint_payloads: bool = False,
    progress_callback: (
        Callable[[int, Mapping[str, Any] | None], None] | None
    ) = None,
    cancellation_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the unified 24-tick domain model and return public plus private data."""

    if execution_provenance not in SUPPORTED_PROVENANCES:
        raise YuLanScaleEngineError(
            "pure-domain engine received an unsupported provenance"
        )
    if execution_provenance == EMULATOR_ONLY_PROVENANCE and semantic_source:
        raise YuLanScaleEngineError(
            "development provenance cannot acquire semantic turns"
        )
    if execution_provenance in {
        REVIEWED_TRACE_PROVENANCE,
        AUTHORIZED_LIVE_PROVENANCE,
    } and (
        semantic_source is None
        or getattr(semantic_source, "provenance", None)
        != execution_provenance
    ):
        raise YuLanScaleEngineError(
            "semantic execution requires its exact provenance-bound source"
        )
    agents = population.get("agents")
    if not isinstance(agents, list) or len(agents) < (
        LONGITUDINAL_ANCHOR_COUNT + PPS_SAMPLE_BUDGET
    ):
        raise YuLanScaleEngineError("unified population is too small")
    if strict_scale and len(agents) != FORMAL_PARENT_COUNT:
        raise YuLanScaleEngineError("formal YuLan-Scale requires 1,000 agents")
    if strict_scale:
        if emulator_predictor is None:
            raise YuLanScaleEngineError(
                "formal YuLan-Scale cannot use the kernel frequency fallback"
            )
        emulator_identity = _validated_emulator_release_identity(
            emulator_release
        )
    else:
        emulator_identity = (
            _validated_emulator_release_identity(emulator_release)
            if emulator_release is not None
            else {
                "schema_version": "development-kernel-probability-fallback",
                "emulator_sha256": None,
                "validation_gate_passed": False,
            }
        )
    if (
        not scenario_ids
        or not seeds
        or len(set(scenario_ids)) != len(scenario_ids)
        or len(set(int(seed) for seed in seeds)) != len(seeds)
    ):
        raise YuLanScaleEngineError("scenario/seed plan is invalid")

    scenario_public = {}
    all_private_runs = []
    all_checkpoint_indices = []
    visualization_frames = []
    completed_work_units = 0

    def _unit_completed(
        checkpoint: Mapping[str, Any] | None,
    ) -> None:
        nonlocal completed_work_units
        completed_work_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_work_units,
                _checkpoint_index(checkpoint)
                if checkpoint is not None
                else None,
            )

    for scenario_position, scenario_id in enumerate(scenario_ids):
        seed_runs = []
        scenario_private = []
        for seed_position, seed in enumerate(seeds):
            capture = capture_visual_frames and seed_position == 0
            public_run, private_run = _run_scenario_seed(
                unified_population=population,
                graph_source=graph,
                scenario_id=str(scenario_id),
                seed=int(seed),
                execution_provenance=execution_provenance,
                semantic_source=semantic_source,
                emulator_predictor=emulator_predictor,
                capture_visual_frames=capture,
                checkpoint_callback=checkpoint_callback,
                artifact_callback=artifact_callback,
                committed_semantic_turns=committed_semantic_turns,
                retain_checkpoint_payloads=retain_checkpoint_payloads,
                cancellation_probe=cancellation_probe,
                unit_completed_callback=_unit_completed,
                scenario_seed_work_unit_offset=(
                    (
                        scenario_position * len(seeds)
                        + seed_position
                    )
                    * FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
                ),
            )
            seed_runs.append(public_run["seed_run"])
            scenario_private.append(private_run)
            all_private_runs.append(private_run)
            all_checkpoint_indices.extend(public_run["checkpoint_index"])
            if capture:
                for branch in BRANCH_IDS:
                    visualization_frames.extend(
                        public_run["seed_run"]["branches"][branch][
                            "visual_frames"
                        ]
                    )
        aggregate = _aggregate_scenario_runs(seed_runs=seed_runs)
        scenario_public[str(scenario_id)] = {
            **aggregate,
            "seed_count": len(seed_runs),
            "activation_summary": {
                "rounds": sum(
                    1 + 3 * len(BRANCH_IDS) for _ in scenario_private
                ),
                "anchors_per_round": LONGITUDINAL_ANCHOR_COUNT,
                "pps_per_round": PPS_SAMPLE_BUDGET,
                "anchor_panel_fixed_across_branches": True,
                "unique_activated_agents_by_seed": [
                    len(
                        {
                            *run["anchor_indices"],
                            *(
                                index
                                for branch in BRANCH_IDS
                                for round_ in run["branches"][branch][
                                    "activation_rounds"
                                ][1:]
                                for index in round_["pps_indices"]
                            ),
                        }
                    )
                    for run in scenario_private
                ],
            },
        }

    return _assemble_yulan_scale_result(
        population=population,
        agents=agents,
        scenario_ids=scenario_ids,
        seeds=seeds,
        execution_provenance=execution_provenance,
        emulator_identity=emulator_identity,
        scenario_public=scenario_public,
        all_private_runs=all_private_runs,
        all_checkpoint_indices=all_checkpoint_indices,
        visualization_frames=visualization_frames,
        completed_work_units=completed_work_units,
    )


def _checkpoint_schedule(
    scenario_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[tuple[str, int, str, int, int]]:
    rows = []
    for scenario_position, scenario_id in enumerate(scenario_ids):
        for seed_position, seed in enumerate(seeds):
            offset = (
                scenario_position * len(seeds) + seed_position
            ) * FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
            rows.append(
                (
                    str(scenario_id),
                    int(seed),
                    "shared_baseline",
                    2,
                    offset,
                )
            )
            for branch in BRANCH_IDS:
                for tick in CHECKPOINT_TICKS:
                    if tick >= 3:
                        rows.append(
                            (
                                str(scenario_id),
                                int(seed),
                                branch,
                                int(tick),
                                offset,
                            )
                        )
    return rows


def _validated_checkpoint_frontier(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    population: Mapping[str, Any],
    graph_source: Mapping[str, Any],
    scenario_ids: Sequence[str],
    seeds: Sequence[int],
    persisted_completed_units: int,
    capture_visual_frames: bool,
) -> tuple[
    dict[tuple[str, int, str, int], dict[str, Any]],
    int,
]:
    if (
        isinstance(persisted_completed_units, bool)
        or not isinstance(persisted_completed_units, int)
        or not 0
        <= persisted_completed_units
        <= len(scenario_ids)
        * len(seeds)
        * FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
    ):
        raise YuLanScaleEngineError("persisted aggregate progress is invalid")
    schedule = _checkpoint_schedule(scenario_ids, seeds)
    expected_keys = {
        (scenario_id, seed, branch, tick)
        for scenario_id, seed, branch, tick, _offset in schedule
    }
    by_key: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for raw in checkpoints:
        if not isinstance(raw, Mapping):
            raise YuLanScaleEngineError("checkpoint frontier row is invalid")
        checkpoint = copy.deepcopy(dict(raw))
        body = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        if checkpoint.get("checkpoint_sha256") != canonical_hash(body):
            raise YuLanScaleEngineError(
                "checkpoint frontier hash does not match"
            )
        key = (
            str(checkpoint.get("scenario_id") or ""),
            int(checkpoint.get("seed", -1)),
            str(checkpoint.get("branch") or ""),
            int(checkpoint.get("tick", -1)),
        )
        if key not in expected_keys:
            raise YuLanScaleEngineError(
                "checkpoint frontier is outside the frozen plan"
            )
        existing = by_key.get(key)
        if existing is not None and existing != checkpoint:
            raise YuLanScaleEngineError(
                "checkpoint frontier contains a conflicting duplicate"
            )
        by_key[key] = checkpoint

    present_in_order = [
        (scenario_id, seed, branch, tick)
        for scenario_id, seed, branch, tick, _offset in schedule
        if (scenario_id, seed, branch, tick) in by_key
    ]
    if present_in_order != [
        (scenario_id, seed, branch, tick)
        for scenario_id, seed, branch, tick, _offset in schedule[
            : len(present_in_order)
        ]
    ]:
        raise YuLanScaleEngineError(
            "checkpoint frontier is not a frozen-order prefix"
        )

    compiled_identity: dict[str, tuple[str, str]] = {}
    max_checkpoint_units = 0
    for scenario_id, seed, branch, tick, offset in schedule[
        : len(present_in_order)
    ]:
        checkpoint = by_key[(scenario_id, seed, branch, tick)]
        if scenario_id not in compiled_identity:
            scenario_population = scenario_population_view(
                population,
                scenario_id=scenario_id,
            )
            graph = compile_message_exposure_graph_v2(
                scenario_population,
                graph_source,
            )
            compiled_identity[scenario_id] = (
                str(scenario_population["population_sha256"]),
                str(graph.graph_sha256),
            )
        population_sha, graph_sha = compiled_identity[scenario_id]
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("population_sha256") != population_sha
            or checkpoint.get("graph_sha256") != graph_sha
        ):
            raise YuLanScaleEngineError(
                "checkpoint frontier component identity drifted"
            )
        cursor = checkpoint.get("aggregate_cursor")
        if not isinstance(cursor, Mapping):
            raise YuLanScaleEngineError(
                "checkpoint frontier lacks its aggregate cursor"
            )
        validated_cursor = _validate_yulan_scale_aggregate_cursor(
            cursor,
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            tick=tick,
            scenario_seed_work_unit_offset=offset,
        )
        max_checkpoint_units = max(
            max_checkpoint_units,
            int(validated_cursor["completed_units"]),
        )
        if branch == "shared_baseline":
            expected_previous = "0" * 64
        elif tick == 3:
            expected_previous = by_key[
                (scenario_id, seed, "shared_baseline", 2)
            ]["checkpoint_sha256"]
        else:
            prior_tick = {7: 3, 13: 7, 19: 13, 23: 19}[tick]
            expected_previous = by_key[
                (scenario_id, seed, branch, prior_tick)
            ]["checkpoint_sha256"]
        if checkpoint.get("previous_checkpoint_sha256") != expected_previous:
            raise YuLanScaleEngineError(
                "checkpoint frontier predecessor drifted"
            )
        frames = checkpoint.get("visual_frame_prefix", [])
        if not isinstance(frames, list):
            raise YuLanScaleEngineError(
                "checkpoint visual frame prefix is invalid"
            )
        if seed != int(seeds[0]) and frames:
            raise YuLanScaleEngineError(
                "non-canonical checkpoint exposed visual frames"
            )
        if frames:
            if not capture_visual_frames:
                raise YuLanScaleEngineError(
                    "checkpoint visual frames violate the resume mode"
                )
            frame_ticks = [frame.get("tick") for frame in frames]
            if frame_ticks != list(range(tick + 1)):
                raise YuLanScaleEngineError(
                    "checkpoint visual prefix is not contiguous"
                )
            expected_branch = (
                "shared_baseline" if branch == "shared_baseline" else branch
            )
            if any(
                not isinstance(frame, Mapping)
                or frame.get("branch") != expected_branch
                for frame in frames
            ):
                raise YuLanScaleEngineError(
                    "checkpoint visual prefix branch drifted"
                )
    for scenario_id, seed, branch, tick, offset in schedule:
        completed = build_yulan_scale_aggregate_cursor(
            scenario_id=scenario_id,
            seed=seed,
            branch=branch,
            tick=tick,
            scenario_seed_work_unit_offset=offset,
        )["completed_units"]
        if completed <= persisted_completed_units and (
            scenario_id,
            seed,
            branch,
            tick,
        ) not in by_key:
            raise YuLanScaleEngineError(
                "runtime progress crossed a missing domain checkpoint"
            )
    return by_key, max(persisted_completed_units, max_checkpoint_units)


def _scenario_seed_shard_from_checkpoints(
    *,
    population: Mapping[str, Any],
    graph_source: Mapping[str, Any],
    scenario_id: str,
    seed: int,
    seed_position: int,
    scenario_seed_work_unit_offset: int,
    checkpoint_map: dict[tuple[str, int, str, int], dict[str, Any]],
    execution_provenance: str,
    semantic_source: SemanticTurnSource | None,
    emulator_predictor: Callable[..., Mapping[str, Any]] | None,
    emulator_release: Mapping[str, Any] | None,
    strict_scale: bool,
    capture_visual_frames: bool,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ),
    committed_semantic_turns: CommittedSemanticTurns,
    retain_checkpoint_payloads: bool,
    progress_callback: (
        Callable[[int, Mapping[str, Any] | None], None] | None
    ),
    cancellation_probe: Callable[[], bool] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario_population = scenario_population_view(
        population,
        scenario_id=scenario_id,
    )
    graph = compile_message_exposure_graph_v2(
        scenario_population,
        graph_source,
    )
    shared_checkpoint = checkpoint_map[
        (scenario_id, seed, "shared_baseline", 2)
    ]
    shared = decode_yulan_scale_checkpoint_v3(
        shared_checkpoint,
        population_sha256=scenario_population["population_sha256"],
        graph_sha256=graph.graph_sha256,
    )
    if not shared.activation_rounds:
        raise YuLanScaleEngineError(
            "shared checkpoint lacks its activation round"
        )
    anchor_indices = tuple(
        int(index)
        for index in shared.activation_rounds[0]["anchor_indices"]
    )
    branches = {}
    private_branches = {}
    checkpoint_records = []

    def persist_checkpoint(value: Mapping[str, Any]) -> None:
        key = (
            str(value["scenario_id"]),
            int(value["seed"]),
            str(value["branch"]),
            int(value["tick"]),
        )
        existing = checkpoint_map.get(key)
        if existing is not None and existing != value:
            raise YuLanScaleEngineError(
                "resumed checkpoint conflicts with the aggregate frontier"
            )
        checkpoint_map[key] = copy.deepcopy(dict(value))
        if checkpoint_callback is not None:
            checkpoint_callback(value)

    for branch_position, branch in enumerate(BRANCH_IDS):
        latest = shared_checkpoint
        for tick in CHECKPOINT_TICKS:
            candidate = checkpoint_map.get(
                (scenario_id, seed, branch, int(tick))
            )
            if candidate is not None:
                latest = candidate
        if latest["branch"] == branch and int(latest["tick"]) == len(TIMELINE) - 1:
            runtime = decode_yulan_scale_checkpoint_v3(
                latest,
                population_sha256=scenario_population["population_sha256"],
                graph_sha256=graph.graph_sha256,
            )
        else:
            prior_tick = int(latest["tick"])
            completed_before = (
                scenario_seed_work_unit_offset
                + 3
                + branch_position * (len(TIMELINE) - 3)
                + max(prior_tick - 2, 0)
            )
            resumed = resume_yulan_scale_branch_from_checkpoint(
                latest,
                population=population,
                graph=graph_source,
                target_branch=(
                    branch
                    if latest["branch"] == "shared_baseline"
                    else None
                ),
                execution_provenance=execution_provenance,
                semantic_source=semantic_source,
                emulator_predictor=emulator_predictor,
                emulator_release=emulator_release,
                strict_scale=strict_scale,
                target_tick=len(TIMELINE) - 1,
                capture_visual_frames=(
                    capture_visual_frames and seed_position == 0
                ),
                checkpoint_callback=persist_checkpoint,
                artifact_callback=artifact_callback,
                committed_semantic_turns=committed_semantic_turns,
                progress_callback=progress_callback,
                completed_work_units_offset=completed_before,
                scenario_seed_work_unit_offset=(
                    scenario_seed_work_unit_offset
                ),
                cancellation_probe=cancellation_probe,
            )
            runtime = resumed["_runtime"]
            if retain_checkpoint_payloads:
                checkpoint_records.extend(resumed["_checkpoints"])
        branches[branch] = {
            "timeline": copy.deepcopy(runtime.timeline),
            "game_rounds": copy.deepcopy(runtime.state.game_rounds),
            "resource": _resource_view(runtime.state),
            "visual_frames": copy.deepcopy(runtime.visual_frames),
        }
        private_branches[branch] = {
            "activation_rounds": copy.deepcopy(runtime.activation_rounds),
            "resident_slots": copy.deepcopy(runtime.resident_slots),
            "actor_slots": copy.deepcopy(runtime.actor_slots),
            "usage": dict(runtime.usage),
            "anchor_indices": list(anchor_indices),
            "shared_baseline_sha256": canonical_hash(runtime.timeline[:3]),
        }
    seed_run = {
        "seed": int(seed),
        "shared_baseline_timeline": copy.deepcopy(shared.timeline),
        "branches": branches,
    }
    private = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "population_sha256": scenario_population["population_sha256"],
        "graph_sha256": graph.graph_sha256,
        "anchor_indices": list(anchor_indices),
        "anchor_identity_sha256": shared.activation_rounds[0][
            "anchor_identity_sha256"
        ],
        "shared_baseline_sha256": canonical_hash(shared.timeline),
        "shared_activation": copy.deepcopy(shared.activation_rounds[0]),
        "shared_visual_frames": copy.deepcopy(shared.visual_frames),
        "shared_usage": dict(shared.usage),
        "branches": private_branches,
        "checkpoints": checkpoint_records,
    }
    return {"seed_run": seed_run}, private


def resume_yulan_scale_engine_from_frontier(
    *,
    population: Mapping[str, Any],
    graph: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    persisted_completed_units: int,
    execution_provenance: str = EMULATOR_ONLY_PROVENANCE,
    semantic_source: SemanticTurnSource | None = None,
    emulator_predictor: Callable[..., Mapping[str, Any]] | None = None,
    emulator_release: Mapping[str, Any] | None = None,
    scenario_ids: Sequence[str] = SCENARIO_IDS,
    seeds: Sequence[int] = CANONICAL_SEEDS,
    strict_scale: bool = True,
    capture_visual_frames: bool = True,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    artifact_callback: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    committed_semantic_turns: CommittedSemanticTurns = frozenset(),
    retain_checkpoint_payloads: bool = False,
    progress_callback: (
        Callable[[int, Mapping[str, Any] | None], None] | None
    ) = None,
    cancellation_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Resume the frozen global schedule and assemble all completed shards."""

    if execution_provenance not in SUPPORTED_PROVENANCES:
        raise YuLanScaleEngineError("resume provenance is unsupported")
    if execution_provenance == EMULATOR_ONLY_PROVENANCE and semantic_source:
        raise YuLanScaleEngineError(
            "development resume cannot acquire semantic turns"
        )
    if execution_provenance in {
        REVIEWED_TRACE_PROVENANCE,
        AUTHORIZED_LIVE_PROVENANCE,
    } and (
        semantic_source is None
        or getattr(semantic_source, "provenance", None)
        != execution_provenance
    ):
        raise YuLanScaleEngineError(
            "semantic resume requires its exact provenance-bound source"
        )
    agents = population.get("agents")
    if not isinstance(agents, list) or len(agents) < (
        LONGITUDINAL_ANCHOR_COUNT + PPS_SAMPLE_BUDGET
    ):
        raise YuLanScaleEngineError("unified population is too small")
    if strict_scale and len(agents) != FORMAL_PARENT_COUNT:
        raise YuLanScaleEngineError("formal YuLan-Scale requires 1,000 agents")
    if strict_scale:
        if emulator_predictor is None:
            raise YuLanScaleEngineError(
                "formal resume requires the frozen emulator"
            )
        emulator_identity = _validated_emulator_release_identity(
            emulator_release
        )
    else:
        emulator_identity = (
            _validated_emulator_release_identity(emulator_release)
            if emulator_release is not None
            else {
                "schema_version": "development-kernel-probability-fallback",
                "emulator_sha256": None,
                "validation_gate_passed": False,
            }
        )
    if (
        not scenario_ids
        or not seeds
        or len(set(scenario_ids)) != len(scenario_ids)
        or len(set(int(seed) for seed in seeds)) != len(seeds)
    ):
        raise YuLanScaleEngineError("scenario/seed resume plan is invalid")

    checkpoint_map, durable_completed = _validated_checkpoint_frontier(
        checkpoints,
        population=population,
        graph_source=graph,
        scenario_ids=scenario_ids,
        seeds=seeds,
        persisted_completed_units=persisted_completed_units,
        capture_visual_frames=capture_visual_frames,
    )
    scenario_public = {}
    all_private_runs = []
    visualization_frames = []

    def report_progress(
        completed: int,
        checkpoint_index: Mapping[str, Any] | None,
    ) -> None:
        if completed > durable_completed and progress_callback is not None:
            progress_callback(completed, checkpoint_index)

    for scenario_position, scenario_id in enumerate(scenario_ids):
        seed_runs = []
        scenario_private = []
        for seed_position, seed in enumerate(seeds):
            offset = (
                scenario_position * len(seeds) + seed_position
            ) * FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
            shared_key = (str(scenario_id), int(seed), "shared_baseline", 2)
            if shared_key not in checkpoint_map:
                absolute_completed = offset

                def unit_completed(
                    value: Mapping[str, Any] | None,
                ) -> None:
                    nonlocal absolute_completed
                    absolute_completed += 1
                    report_progress(
                        absolute_completed,
                        _checkpoint_index(value)
                        if value is not None
                        else None,
                    )

                def persist_fresh_checkpoint(
                    value: Mapping[str, Any],
                ) -> None:
                    key = (
                        str(value["scenario_id"]),
                        int(value["seed"]),
                        str(value["branch"]),
                        int(value["tick"]),
                    )
                    existing = checkpoint_map.get(key)
                    if existing is not None and existing != value:
                        raise YuLanScaleEngineError(
                            "fresh checkpoint conflicts with frontier"
                        )
                    checkpoint_map[key] = copy.deepcopy(dict(value))
                    if checkpoint_callback is not None:
                        checkpoint_callback(value)

                public_run, private_run = _run_scenario_seed(
                    unified_population=population,
                    graph_source=graph,
                    scenario_id=str(scenario_id),
                    seed=int(seed),
                    execution_provenance=execution_provenance,
                    semantic_source=semantic_source,
                    emulator_predictor=emulator_predictor,
                    capture_visual_frames=(
                        capture_visual_frames and seed_position == 0
                    ),
                    checkpoint_callback=persist_fresh_checkpoint,
                    artifact_callback=artifact_callback,
                    committed_semantic_turns=committed_semantic_turns,
                    retain_checkpoint_payloads=retain_checkpoint_payloads,
                    cancellation_probe=cancellation_probe,
                    unit_completed_callback=unit_completed,
                    scenario_seed_work_unit_offset=offset,
                )
            else:
                public_run, private_run = (
                    _scenario_seed_shard_from_checkpoints(
                        population=population,
                        graph_source=graph,
                        scenario_id=str(scenario_id),
                        seed=int(seed),
                        seed_position=seed_position,
                        scenario_seed_work_unit_offset=offset,
                        checkpoint_map=checkpoint_map,
                        execution_provenance=execution_provenance,
                        semantic_source=semantic_source,
                        emulator_predictor=emulator_predictor,
                        emulator_release=emulator_release,
                        strict_scale=strict_scale,
                        capture_visual_frames=capture_visual_frames,
                        checkpoint_callback=checkpoint_callback,
                        artifact_callback=artifact_callback,
                        committed_semantic_turns=committed_semantic_turns,
                        retain_checkpoint_payloads=(
                            retain_checkpoint_payloads
                        ),
                        progress_callback=report_progress,
                        cancellation_probe=cancellation_probe,
                    )
                )
            seed_runs.append(public_run["seed_run"])
            scenario_private.append(private_run)
            all_private_runs.append(private_run)
            if capture_visual_frames and seed_position == 0:
                for branch in BRANCH_IDS:
                    visualization_frames.extend(
                        public_run["seed_run"]["branches"][branch][
                            "visual_frames"
                        ]
                    )
        aggregate = _aggregate_scenario_runs(seed_runs=seed_runs)
        scenario_public[str(scenario_id)] = {
            **aggregate,
            "seed_count": len(seed_runs),
            "activation_summary": {
                "rounds": sum(
                    1 + 3 * len(BRANCH_IDS) for _ in scenario_private
                ),
                "anchors_per_round": LONGITUDINAL_ANCHOR_COUNT,
                "pps_per_round": PPS_SAMPLE_BUDGET,
                "anchor_panel_fixed_across_branches": True,
                "unique_activated_agents_by_seed": [
                    len(
                        {
                            *run["anchor_indices"],
                            *(
                                index
                                for branch in BRANCH_IDS
                                for round_ in run["branches"][branch][
                                    "activation_rounds"
                                ][1:]
                                for index in round_["pps_indices"]
                            ),
                        }
                    )
                    for run in scenario_private
                ],
            },
        }

    schedule = _checkpoint_schedule(scenario_ids, seeds)
    expected_checkpoint_count = len(schedule)
    if len(checkpoint_map) != expected_checkpoint_count:
        raise YuLanScaleEngineError(
            "resumed aggregate checkpoint frontier is incomplete"
        )
    all_checkpoint_indices = [
        _checkpoint_index(
            checkpoint_map[(scenario_id, seed, branch, tick)]
        )
        for scenario_id, seed, branch, tick, _offset in schedule
    ]
    completed_work_units = (
        len(scenario_ids)
        * len(seeds)
        * FORMAL_BRANCH_TICKS_PER_SCENARIO_SEED
    )
    return _assemble_yulan_scale_result(
        population=population,
        agents=agents,
        scenario_ids=scenario_ids,
        seeds=seeds,
        execution_provenance=execution_provenance,
        emulator_identity=emulator_identity,
        scenario_public=scenario_public,
        all_private_runs=all_private_runs,
        all_checkpoint_indices=all_checkpoint_indices,
        visualization_frames=visualization_frames,
        completed_work_units=completed_work_units,
    )


def public_yulan_scale_result_v3(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove private sample, prompt, memory, and checkpoint payload state."""

    return {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if not str(key).startswith("_")
    }
