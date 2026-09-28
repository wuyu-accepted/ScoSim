"""Unified, privacy-safe LLM-agent population and emulator contracts.

This module is deliberately independent from the runtime and database layers.
It turns the reviewed MassSim episode population into *callable LLM-agent
descriptors*, bridges already-produced live-LLM artifacts without retaining
prompts or raw provider bodies, and fits a small deterministic behavioral
emulator for the ticks at which an agent is not sent to a Provider.

The emulator is an amortized executor of an LLM-agent policy.  Its output is
never represented as a Provider response or free-form LLM speech.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .mass_population import canonical_hash
from .mass_simulation import (
    FinitePopulationCell,
    _design_for_cell,
    _true_distribution,
    estimate_cell_distribution,
)


LLM_AGENT_POPULATION_SCHEMA_VERSION = "semantic-llm-agent-population-v1"
ANCHOR_SELECTION_SCHEMA_VERSION = "campus-pulse-llm-anchor-selection-v1"
TRACE_RELEASE_SCHEMA_VERSION = "reviewed-llm-behavior-trace-release-v1"
EMULATOR_RELEASE_SCHEMA_VERSION = "llm-behavior-emulator-v1"
EMULATOR_VALIDATION_SCHEMA_VERSION = "llm-behavior-emulator-validation-v1"
LEGACY_SEMANTIC_CONTRACT_ID = "legacy_llm_decision_contract_v1"
BUDGETED_SEMANTIC_CONTRACT_ID = "budgeted_llm_agent_turn_v1"

LLM_AGENT_POPULATION_SIZE = 1000
LONGITUDINAL_ANCHOR_COUNT = 16
ACTIVE_ARCHETYPES = tuple(
    f"archetype-{index:02d}" for index in range(1, 18) if index != 16
)
BACKGROUND_ARCHETYPE = "archetype-16"
RESTRICTED_ARCHETYPE_ACTIONS = {
    "archetype-09": ("ignore", "read", "report_risk"),
    BACKGROUND_ARCHETYPE: ("ignore", "read"),
}
SCALE_GAP_BUDGETS = (4, 8, 16, 24, 32, 40, 48)

DEFAULT_SCENARIO_MONTHS = {
    "governance_legitimacy_dispute": "2026-03",
    "lecture_external_incident_shock": "2026-06",
}
ACTION_LABELS = (
    "ignore",
    "read",
    "comment",
    "post",
    "seek_help",
    "report_risk",
)
STANCE_LABELS = ("supportive", "neutral", "concerned", "critical")
NUMERIC_TARGETS = (
    "urgency",
    "trust_delta",
    "satisfaction",
    "attention_delta",
    "concern_delta",
    "activity_delta",
    "voice_delta",
)
MODEL_CATEGORICAL_FIELDS = (
    "archetype_id",
    "persona_variant_id",
    "scenario_id",
    "phase_id",
    "policy_id",
    "semantic_contract_id",
)
MODEL_NUMERIC_FIELDS = (
    "population_weight",
    "prior_attention",
    "prior_concern",
    "prior_trust",
    "prior_satisfaction",
    "prior_activity",
    "particle_attention_mean",
    "particle_concern_mean",
    "particle_trust_mean",
    "particle_satisfaction_mean",
    "particle_activity_mean",
    "particle_state_variance",
    "neighbor_public_action_share",
    "service_strain",
    "visible_message_count",
)
MODEL_INTERACTIONS = (
    ("archetype_id", "phase_id"),
    ("scenario_id", "phase_id"),
    ("scenario_id", "policy_id"),
    ("semantic_contract_id", "phase_id"),
)
MODEL_TARGET_FIELDS = {
    "action": "action",
    "stance": "stance",
    "topic": "topic_id",
}
PHASE_ORDER = {
    "baseline": 0,
    "burst": 1,
    "spread": 2,
    "decay": 3,
}
SCALE_GAP_METHODS = (
    "srs",
    "stratified",
    "pps_ht",
    "greg_ht",
    "deterministic_critical_node_replacement",
)
SCALE_GAP_DIMENSIONS = ("action", "stance", "topic_id")


class LLMPopulationError(ValueError):
    """Raised when a release cannot be built without inventing evidence."""


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LLMPopulationError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise LLMPopulationError(f"{name} must be a list")
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalize(values: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    nonnegative = {label: max(0.0, _safe_float(values.get(label))) for label in labels}
    total = sum(nonnegative.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    normalized = {label: value / total for label, value in nonnegative.items()}
    # Bind floating drift to the final label so exact sums remain stable in JSON.
    normalized[labels[-1]] += 1.0 - sum(normalized.values())
    return normalized


def _persona_indexes(
    persona_release: Mapping[str, Any],
    persona_evidence: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    profiles = {
        str(item.get("archetype_id")): item
        for item in _list(persona_release.get("profiles"), "persona profiles")
        if isinstance(item, Mapping)
    }
    packs = {
        str(item.get("archetype_id")): item
        for item in _list(persona_evidence.get("packs"), "persona evidence packs")
        if isinstance(item, Mapping)
    }
    expected = set(f"archetype-{index:02d}" for index in range(1, 18))
    if set(profiles) != expected or set(packs) != expected:
        raise LLMPopulationError("reviewed 17-profile persona inputs do not align")
    if persona_release.get("status") != "complete":
        raise LLMPopulationError("persona release is not complete")
    if persona_evidence.get("status") != "complete":
        raise LLMPopulationError("persona evidence is not complete")
    if persona_release.get("privacy_scan", {}).get("status") != "pass":
        raise LLMPopulationError("persona release privacy scan did not pass")
    return profiles, packs


def _variant_for_episode(
    *,
    profile: Mapping[str, Any],
    semantic: Mapping[str, Any],
    index: int,
    seed: int,
) -> tuple[Mapping[str, Any], str]:
    variants = [
        item for item in _list(profile.get("variants"), "persona variants")
        if isinstance(item, Mapping)
    ]
    if len(variants) != 3:
        raise LLMPopulationError("each reviewed profile must contain three variants")
    # Mass population intentionally removed source record IDs.  Consequently a
    # record-id join would violate the privacy boundary.  Reuse the reviewed
    # deterministic rotation rule and explicitly record it as fallback.
    order_key = canonical_hash(
        [seed, index, semantic.get("topic_code"), semantic.get("need_code")]
    )
    variant = variants[int(order_key[:8], 16) % len(variants)]
    return variant, "reviewed_variant_fallback"


def _scenario_agent_weights(
    agents: Sequence[Mapping[str, Any]],
    packs: Mapping[str, Mapping[str, Any]],
    month: str,
) -> list[float]:
    by_archetype: dict[str, list[int]] = defaultdict(list)
    for index, agent in enumerate(agents):
        by_archetype[str(agent.get("primary_archetype_id"))].append(index)
    support = {
        archetype: max(
            0.0,
            _safe_float(packs[archetype].get("month_support", {}).get(month)),
        )
        for archetype in packs
    }
    # Bounded smoothing gives every existing episode positive probability while
    # preserving the reviewed month structure.
    positive = [value for value in support.values() if value > 0]
    epsilon = min(positive) * 1e-6 if positive else 1e-6
    support = {
        archetype: (value if value > 0 else epsilon)
        for archetype, value in support.items()
        if by_archetype.get(archetype)
    }
    target_total = sum(support.values())
    if target_total <= 0:
        raise LLMPopulationError(f"scenario month {month} has no reviewed support")
    result = [0.0] * len(agents)
    for archetype, indices in by_archetype.items():
        if archetype not in support:
            continue
        within = [
            max(1e-15, _safe_float(agents[index].get("population_weight")))
            for index in indices
        ]
        within_total = sum(within)
        archetype_mass = support[archetype] / target_total
        for index, base in zip(indices, within):
            result[index] = archetype_mass * base / within_total
    result[-1] += 1.0 - sum(result)
    if any(value <= 0 for value in result):
        raise LLMPopulationError("scenario weights must remain strictly positive")
    return result


def compile_semantic_llm_agent_population_v1(
    *,
    mass_population: Mapping[str, Any],
    persona_release: Mapping[str, Any],
    persona_evidence: Mapping[str, Any],
    scenario_months: Mapping[str, str] = DEFAULT_SCENARIO_MONTHS,
    seed: int = 20260731,
) -> dict[str, Any]:
    """Compile exactly 1,000 reviewed episodes into callable LLM-agent descriptors.

    The existing MassSim fields are retained so that particle/message kernels can
    consume the release without a second population implementation.
    """

    population = _object(mass_population, "mass population")
    source_agents = _list(population.get("agents"), "mass population agents")
    if (
        population.get("schema_version") != "semantic-population-v2"
        or len(source_agents) != LLM_AGENT_POPULATION_SIZE
        or population.get("population_size") != LLM_AGENT_POPULATION_SIZE
    ):
        raise LLMPopulationError("LLM population requires semantic-population-v2/1000")
    expected_source_hash = canonical_hash(
        {key: value for key, value in population.items() if key != "population_sha256"}
    )
    if population.get("population_sha256") not in {
        expected_source_hash,
        canonical_hash(population),
    }:
        # Legacy v2 compilers hash the body before inserting population_sha256.
        raise LLMPopulationError("mass population hash does not bind its body")
    profiles, packs = _persona_indexes(persona_release, persona_evidence)
    basis_sha256 = canonical_hash(
        {
            "mass_population_sha256": population.get("population_sha256"),
            "persona_release": persona_release.get("run_fingerprint")
            or persona_release.get("persona_release_fingerprint"),
            "persona_evidence": persona_evidence.get("run_fingerprint"),
            "scenario_months": dict(sorted(scenario_months.items())),
            "seed": int(seed),
        }
    )
    compiled: list[dict[str, Any]] = []
    for index, source in enumerate(source_agents):
        agent = copy.deepcopy(_object(source, f"agent {index}"))
        archetype = str(agent.get("primary_archetype_id"))
        if archetype not in profiles:
            raise LLMPopulationError(f"unknown reviewed archetype: {archetype}")
        profile = profiles[archetype]
        pack = packs[archetype]
        semantic = _object(agent.get("semantic"), "agent semantic")
        variant, assignment_basis = _variant_for_episode(
            profile=profile,
            semantic=semantic,
            index=index,
            seed=seed,
        )
        agent["llm_agent_id"] = f"llm-agent-{basis_sha256[:12]}-{index + 1:04d}"
        agent["persona_variant_id"] = str(variant.get("variant_id"))
        agent["persona_variant_assignment_basis"] = assignment_basis
        agent["persona"] = {
            "profile_summary": profile.get("profile_summary"),
            "observable_core_tendencies": profile.get("observable_core_tendencies", []),
            "trigger_conditions": profile.get("trigger_conditions", []),
            "language_tendencies": profile.get("language_tendencies", []),
            "do_not_simulate": profile.get("do_not_simulate", []),
            "situational_signature": variant.get("situational_signature"),
            "reaction_logic": variant.get("reaction_logic"),
            "language_style": variant.get("language_style"),
            "uncertainty_behavior": variant.get("uncertainty_behavior"),
            "simulation_instruction": variant.get("simulation_instruction"),
            "assigned_episode_focus": {
                "topic_code": semantic.get("topic_code"),
                "need_code": semantic.get("need_code"),
                "speech_act": semantic.get("speech_act"),
                "stance": semantic.get("stance"),
                "source": "reviewed_semantic_codes_not_raw_text",
            },
            "simulation_mode": profile.get("direct_review", {}).get(
                "simulation_mode", "active_episode_agent"
            ),
        }
        allowed = RESTRICTED_ARCHETYPE_ACTIONS.get(archetype, ACTION_LABELS)
        agent["behavior_contract"] = {
            "provider_callable": True,
            "autonomous_public_action": archetype != BACKGROUND_ARCHETYPE,
            "allowed_actions": list(allowed),
            "raw_text_available": False,
            "persistent_real_person_claimed": False,
            "interaction_role": pack.get("taxonomy_definition", {}).get(
                "interaction_role"
            ),
        }
        agent["scenario_weights"] = {}
        compiled.append(agent)
    scenario_weight_sums = {}
    for scenario_id, month in sorted(scenario_months.items()):
        weights = _scenario_agent_weights(compiled, packs, str(month))
        for agent, weight in zip(compiled, weights):
            agent["scenario_weights"][scenario_id] = weight
        scenario_weight_sums[scenario_id] = sum(weights)
    body = {
        "schema_version": LLM_AGENT_POPULATION_SCHEMA_VERSION,
        "population_kind": "reviewed_episode_conditioned_callable_llm_agents",
        "population_size": len(compiled),
        "particle_hypotheses_per_agent": 10,
        "source_population_sha256": population.get("population_sha256"),
        "population_basis_sha256": basis_sha256,
        "scenario_months": dict(sorted(scenario_months.items())),
        "scenario_weight_sums": scenario_weight_sums,
        "agents": compiled,
        "privacy": {
            "raw_text_persisted": False,
            "source_identifiers_persisted": False,
            "provider_bodies_persisted": False,
        },
        "non_claims": [
            "Agents are reviewed forum-behavior episodes, not identified students.",
            "Every agent has a Provider-callable contract; the reviewed background class cannot emit autonomous public action.",
            "An emulator tick is not a Provider response or independent human observation.",
        ],
    }
    body["population_sha256"] = canonical_hash(body)
    return body


def select_longitudinal_anchors(
    population: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
    count: int = LONGITUDINAL_ANCHOR_COUNT,
) -> dict[str, Any]:
    """Select one positive-weight callable agent from every active archetype."""

    agents = _list(population.get("agents"), "population agents")
    if count != len(ACTIVE_ARCHETYPES):
        raise LLMPopulationError("anchor selection is frozen at 16 active archetypes")
    selected: list[int] = []
    for archetype in ACTIVE_ARCHETYPES:
        candidates = [
            index
            for index, agent in enumerate(agents)
            if agent.get("primary_archetype_id") == archetype
            and agent.get("behavior_contract", {}).get("provider_callable")
            and _safe_float(agent.get("scenario_weights", {}).get(scenario_id)) > 0
        ]
        if not candidates:
            raise LLMPopulationError(f"no callable anchor candidate for {archetype}")
        weights = [
            _safe_float(agents[index]["scenario_weights"][scenario_id])
            for index in candidates
        ]
        rng = random.Random(
            int(canonical_hash([seed, scenario_id, archetype])[:16], 16)
        )
        draw = rng.random() * sum(weights)
        cumulative = 0.0
        chosen = candidates[-1]
        for candidate, weight in zip(candidates, weights):
            cumulative += weight
            if draw <= cumulative:
                chosen = candidate
                break
        selected.append(chosen)
    result = {
        "schema_version": ANCHOR_SELECTION_SCHEMA_VERSION,
        "population_sha256": population.get("population_sha256"),
        "scenario_id": scenario_id,
        "seed": int(seed),
        "count": len(selected),
        "agent_indices": selected,
        "agent_ids": [agents[index]["llm_agent_id"] for index in selected],
        "archetype_ids": list(ACTIVE_ARCHETYPES),
        "selection_method": "one_per_active_archetype_within_archetype_pps",
        "greg_ht_population_sample": False,
    }
    result["selection_sha256"] = canonical_hash(result)
    return result


def _iter_artifact_turns(payload: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    scenario_id = str(payload.get("scenario_id") or "")
    seed = payload.get("execution", {}).get("seed_used_for_initial_population_only")
    for run in payload.get("policy_runs", []):
        if not isinstance(run, Mapping):
            continue
        policy_id = str(run.get("policy_id") or "")
        for phase in run.get("phases", []):
            if not isinstance(phase, Mapping):
                continue
            phase_id = str(phase.get("phase_id") or "")
            normalized_policy = "shared_baseline" if phase_id == "baseline" else policy_id
            for record in phase.get("agent_decisions", []):
                if not isinstance(record, Mapping):
                    continue
                identity = (
                    scenario_id,
                    seed,
                    normalized_policy,
                    phase_id,
                    str(record.get("synthetic_agent_id") or ""),
                )
                yield {
                    "identity": identity,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "policy_id": normalized_policy,
                    "phase_id": phase_id,
                    "synthetic_agent_id": identity[-1],
                    "archetype_id": record.get("archetype_id"),
                    "persona_variant_id": record.get("persona_variant_id"),
                    "population_weight": _safe_float(record.get("population_weight")),
                    "action": record.get("action"),
                    "topic_id": record.get("topic_id"),
                    "stance": record.get("stance"),
                    "urgency": _safe_float(record.get("urgency")),
                    "trust_delta": _safe_float(record.get("trust_delta")),
                    "satisfaction": 1.0 if record.get("satisfied") else 0.0,
                    "rationale_tag": record.get("rationale_tag"),
                    "synthetic_message": record.get("synthetic_message") or "",
                    "returned_model": record.get("returned_model"),
                    "usage": {
                        key: int(_safe_float(record.get("usage", {}).get(key)))
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    },
                    "format_repairs": int(_safe_float(record.get("format_repairs"))),
                    "contract_adjustments": list(
                        record.get("contract_adjustments") or []
                    ),
                }


def compile_reviewed_llm_behavior_trace_release_v1(
    *,
    artifact_paths: Sequence[str | Path],
    expected_unique_turns: int = 912,
) -> dict[str, Any]:
    """Bridge live LLM artifacts while retaining no prompt or raw provider body.

    Shared baseline rows present under multiple policies are deduplicated by
    ``(scenario, seed, normalized_policy, phase, synthetic_agent_id)``.
    Completion is asserted only when the supplied artifacts actually prove the
    expected count.
    """

    paths = [Path(path) for path in artifact_paths]
    source_artifacts = []
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    raw_turn_count = 0
    conflicts: list[list[Any]] = []
    for path in paths:
        if not path.is_file():
            raise LLMPopulationError(f"LLM artifact does not exist: {path}")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMPopulationError(f"invalid LLM artifact: {path.name}") from exc
        source_artifacts.append(
            {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
        for turn in _iter_artifact_turns(_object(payload, "LLM artifact")):
            raw_turn_count += 1
            identity = turn.pop("identity")
            fingerprint = canonical_hash(turn)
            existing = deduplicated.get(identity)
            if existing is not None and canonical_hash(existing) != fingerprint:
                conflicts.append(list(identity))
                continue
            deduplicated[identity] = turn
    entries = [
        {
            **deduplicated[identity],
            "semantic_contract_id": LEGACY_SEMANTIC_CONTRACT_ID,
            "turn_identity_sha256": canonical_hash(list(identity)),
            "structured_response_sha256": canonical_hash(deduplicated[identity]),
            # A prompt is not present in the legacy result. Do not invent one.
            "prompt_fingerprint": None,
            "prompt_replay_eligible": False,
        }
        for identity in sorted(deduplicated, key=lambda item: tuple(map(str, item)))
    ]
    complete = (
        bool(paths)
        and not conflicts
        and len(entries) == expected_unique_turns
    )
    body = {
        "schema_version": TRACE_RELEASE_SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
        "raw_turn_count": raw_turn_count,
        "unique_turn_count": len(entries),
        "expected_unique_turn_count": expected_unique_turns,
        "shared_baseline_deduplication": {
            "identity_fields": [
                "scenario_id",
                "execution.seed_used_for_initial_population_only",
                "normalized_policy",
                "phase_id",
                "synthetic_agent_id",
            ],
            "baseline_policy": "shared_baseline",
        },
        "conflict_count": len(conflicts),
        "source_artifacts": source_artifacts,
        "entries": entries,
        "privacy": {
            "prompt_text_persisted": False,
            "provider_raw_body_persisted": False,
            "api_credentials_persisted": False,
            "source_paths_persisted": False,
            "raw_forum_text_persisted": False,
        },
        "validity": {
            "historical_trace_training_eligible": complete,
            "exact_prompt_replay_eligible": False,
            "reason": (
                "complete_reviewed_structured_decision_evidence"
                if complete
                else "expected_unique_turn_evidence_not_proven"
            ),
        },
    }
    body["trace_release_sha256"] = canonical_hash(body)
    return body


def _feature_group(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(
            row.get("archetype_id")
            or row.get("primary_archetype_id")
            or "generic"
        ),
        str(row.get("scenario_id") or "unknown"),
        str(row.get("phase_id") or "unknown"),
    )


def _categorical_input(row: Mapping[str, Any], field: str) -> str:
    if field == "archetype_id":
        return str(
            row.get("archetype_id")
            or row.get("primary_archetype_id")
            or "generic"
        )
    if field == "persona_variant_id":
        return str(row.get(field) or row.get("persona_variant_id") or "unknown")
    return str(row.get(field) or "unknown")


def _numeric_input(row: Mapping[str, Any], field: str) -> float | None:
    candidates: list[Any] = [row.get(field)]
    for container_name in (
        "state",
        "posterior",
        "particle_state",
        "particle_summary",
        "neighbor_aggregate",
        "service_state",
    ):
        container = row.get(container_name)
        if isinstance(container, Mapping):
            candidates.append(container.get(field))
            if field.startswith("prior_"):
                candidates.append(container.get(field[len("prior_") :]))
            if field.startswith("particle_") and field.endswith("_mean"):
                candidates.append(container.get(field[9:-5]))
    for raw in candidates:
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            return float(raw)
    return None


def _build_feature_schema(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    categorical_vocabulary = {
        field: sorted({_categorical_input(row, field) for row in rows})
        for field in MODEL_CATEGORICAL_FIELDS
    }
    interaction_vocabulary: dict[str, list[str]] = {}
    for left, right in MODEL_INTERACTIONS:
        name = f"{left}*{right}"
        interaction_vocabulary[name] = sorted(
            {
                f"{_categorical_input(row, left)}|{_categorical_input(row, right)}"
                for row in rows
            }
        )
    numeric_statistics = {}
    for field in MODEL_NUMERIC_FIELDS:
        values = [
            value
            for row in rows
            if (value := _numeric_input(row, field)) is not None
        ]
        if not values:
            continue
        mean = float(sum(values) / len(values))
        variance = float(sum((value - mean) ** 2 for value in values) / len(values))
        numeric_statistics[field] = {
            "mean": mean,
            "scale": max(math.sqrt(variance), 1e-8),
            "observed_rows": len(values),
        }
    feature_names = ["intercept"]
    for field in MODEL_CATEGORICAL_FIELDS:
        feature_names.extend(
            f"categorical:{field}={value}"
            for value in categorical_vocabulary[field]
        )
    for left, right in MODEL_INTERACTIONS:
        name = f"{left}*{right}"
        feature_names.extend(
            f"interaction:{name}={value}"
            for value in interaction_vocabulary[name]
        )
    feature_names.extend(
        f"numeric:{field}" for field in sorted(numeric_statistics)
    )
    return {
        "schema_version": "llm-behavior-emulator-feature-schema-v1",
        "categorical_vocabulary": categorical_vocabulary,
        "interaction_vocabulary": interaction_vocabulary,
        "numeric_statistics": numeric_statistics,
        "feature_names": feature_names,
        "unknown_category_policy": "zero_effect_with_fitted_intercept",
        "missing_numeric_policy": "training_mean",
    }


def _encode_features(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> np.ndarray:
    feature_names = list(schema["feature_names"])
    positions = {name: index for index, name in enumerate(feature_names)}
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    matrix[:, 0] = 1.0
    categorical_vocabulary = schema["categorical_vocabulary"]
    for row_index, row in enumerate(rows):
        for field in MODEL_CATEGORICAL_FIELDS:
            value = _categorical_input(row, field)
            if value in categorical_vocabulary.get(field, ()):
                matrix[
                    row_index,
                    positions[f"categorical:{field}={value}"],
                ] = 1.0
        for left, right in MODEL_INTERACTIONS:
            name = f"{left}*{right}"
            value = (
                f"{_categorical_input(row, left)}|"
                f"{_categorical_input(row, right)}"
            )
            feature_name = f"interaction:{name}={value}"
            if feature_name in positions:
                matrix[row_index, positions[feature_name]] = 1.0
        for field, statistics in schema["numeric_statistics"].items():
            raw = _numeric_input(row, field)
            if raw is None:
                raw = float(statistics["mean"])
            matrix[row_index, positions[f"numeric:{field}"]] = (
                raw - float(statistics["mean"])
            ) / max(float(statistics["scale"]), 1e-8)
    return matrix


def _row_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [max(_safe_float(row.get("population_weight")), 1e-12) for row in rows],
        dtype=np.float64,
    )
    return values / float(values.sum())


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -700.0, 0.0))
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _fit_regularized_multinomial_glm(
    matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    labels: Sequence[str],
    regularization: float,
    max_iterations: int,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown = sorted(
        {
            str(row.get(field) or "")
            for row in rows
            if str(row.get(field) or "") not in label_to_index
        }
    )
    if unknown:
        raise LLMPopulationError(
            f"{field} contains labels outside the frozen contract: {unknown}"
        )
    target = np.asarray(
        [label_to_index[str(row.get(field) or "")] for row in rows],
        dtype=np.int64,
    )
    weights = _row_weights(rows)
    class_mass = np.bincount(
        target,
        weights=weights,
        minlength=len(labels),
    )
    class_prior = (class_mass + 1e-4) / (
        float(class_mass.sum()) + 1e-4 * len(labels)
    )
    coefficients = np.zeros(
        (matrix.shape[1], len(labels)),
        dtype=np.float64,
    )
    coefficients[0, :] = np.log(class_prior)
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    learning_rate = 0.045
    previous_loss = math.inf
    converged = False
    iterations = 0
    one_hot = np.eye(len(labels), dtype=np.float64)[target]
    penalty_mask = np.ones_like(coefficients)
    penalty_mask[0, :] = 0.0
    for iteration in range(1, max_iterations + 1):
        probability = _softmax(matrix @ coefficients)
        gradient = (
            matrix.T @ ((probability - one_hot) * weights[:, None])
            + regularization * penalty_mask * coefficients
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        coefficients -= (
            learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + 1e-8)
        )
        # Softmax is invariant to a common class coefficient.  Centering makes
        # the serialized release identifiable without changing predictions.
        coefficients -= coefficients.mean(axis=1, keepdims=True)
        probability = _softmax(matrix @ coefficients)
        loss = float(
            -np.sum(
                weights
                * np.log(
                    np.clip(probability[np.arange(len(rows)), target], 1e-12, 1.0)
                )
            )
            + 0.5
            * regularization
            * np.sum((penalty_mask * coefficients) ** 2)
        )
        iterations = iteration
        if (
            math.isfinite(previous_loss)
            and abs(previous_loss - loss)
            <= 1e-8 * max(1.0, abs(previous_loss))
        ):
            converged = True
            break
        previous_loss = loss
    raw_logits = matrix @ coefficients
    temperature_candidates = np.linspace(0.45, 1.75, 53)
    temperature_losses = []
    for temperature in temperature_candidates:
        calibrated = _softmax(raw_logits / float(temperature))
        temperature_losses.append(
            float(
                -np.sum(
                    weights
                    * np.log(
                        np.clip(
                            calibrated[np.arange(len(rows)), target],
                            1e-12,
                            1.0,
                        )
                    )
                )
            )
        )
    temperature_index = int(np.argmin(temperature_losses))
    temperature = float(temperature_candidates[temperature_index])
    return {
        "labels": list(labels),
        "coefficients": coefficients.tolist(),
        "regularization": float(regularization),
        "iterations": iterations,
        "converged": converged,
        "training_objective": loss,
        "temperature": temperature,
        "temperature_selection": {
            "criterion": "training_weighted_log_loss",
            "grid_min": float(temperature_candidates[0]),
            "grid_max": float(temperature_candidates[-1]),
            "grid_size": len(temperature_candidates),
            "selected_log_loss": temperature_losses[temperature_index],
            "validation_data_used": False,
        },
        "training_class_mass": {
            label: float(class_mass[index])
            for index, label in enumerate(labels)
        },
    }


def _fit_regularized_state_transition(
    matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    regularization: float,
) -> dict[str, Any]:
    target = np.asarray(
        [
            [_safe_float(row.get(name)) for name in NUMERIC_TARGETS]
            for row in rows
        ],
        dtype=np.float64,
    )
    weights = _row_weights(rows)
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)[:, None]
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    lhs = weighted_matrix.T @ weighted_matrix + penalty
    rhs = weighted_matrix.T @ weighted_target
    coefficients = np.linalg.pinv(lhs, rcond=1e-10) @ rhs
    residual = target - matrix @ coefficients
    weighted_rmse = np.sqrt(
        np.sum(weights[:, None] * residual * residual, axis=0)
    )
    return {
        "targets": list(NUMERIC_TARGETS),
        "coefficients": coefficients.tolist(),
        "regularization": float(regularization),
        "weighted_training_rmse": {
            target_name: float(weighted_rmse[index])
            for index, target_name in enumerate(NUMERIC_TARGETS)
        },
        "transition_contract": (
            "Current posterior/state features are optional inputs; missing "
            "features take their training mean. Outputs are bounded semantic "
            "state deltas, not Provider responses."
        ),
    }


def fit_llm_behavior_emulator_v1(
    *,
    trace_release: Mapping[str, Any],
    seed: int = 20260731,
    allow_incomplete_for_testing: bool = False,
    label_schema: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Fit a deterministic, regularized multi-task GLM/state transition.

    The categorical heads are genuine multinomial logistic regressions fitted
    by weighted Adam with an L2 penalty.  The continuous head is a weighted
    ridge state-transition model.  All features and coefficients are frozen in
    the release so prediction never consults the training traces.
    """

    entries = [
        _object(row, "trace entry")
        for row in _list(trace_release.get("entries"), "trace entries")
    ]
    if trace_release.get("status") != "complete" and not allow_incomplete_for_testing:
        raise LLMPopulationError("incomplete LLM trace evidence cannot train a release")
    if not entries:
        raise LLMPopulationError("at least one LLM trace entry is required")
    supplied = dict(label_schema or {})
    labels = {
        "action": tuple(supplied.get("action") or ACTION_LABELS),
        "stance": tuple(supplied.get("stance") or STANCE_LABELS),
        "topic": tuple(
            supplied.get("topic")
            or sorted(
                {
                    str(row.get("topic_id") or "off_topic_daily_life")
                    for row in entries
                }
            )
            or ("off_topic_daily_life",)
        ),
    }
    schema = _build_feature_schema(entries)
    matrix = _encode_features(entries, schema)
    categorical_models = {}
    for target_name, field in MODEL_TARGET_FIELDS.items():
        categorical_models[target_name] = _fit_regularized_multinomial_glm(
            matrix,
            entries,
            field=field,
            labels=labels[target_name],
            regularization=0.015,
            max_iterations=140,
        )
    state_transition = _fit_regularized_state_transition(
        matrix,
        entries,
        regularization=0.025,
    )
    global_distributions = {}
    row_weights = _row_weights(entries)
    for target_name, field in MODEL_TARGET_FIELDS.items():
        counts = {
            label: float(
                sum(
                    row_weights[index]
                    for index, row in enumerate(entries)
                    if str(row.get(field) or "") == label
                )
            )
            for label in labels[target_name]
        }
        global_distributions[f"{target_name}_probabilities"] = _normalize(
            counts,
            labels[target_name],
        )
    continuous_global = {
        target: float(
            sum(
                row_weights[index] * _safe_float(row.get(target))
                for index, row in enumerate(entries)
            )
        )
        for target in NUMERIC_TARGETS
    }
    body = {
        "schema_version": EMULATOR_RELEASE_SCHEMA_VERSION,
        "status": "frozen",
        "method": "regularized_multitask_softmax_glm_ridge_state_transition_v1",
        "seed": int(seed),
        "source_trace_release_sha256": trace_release.get("trace_release_sha256"),
        "training_rows": len(entries),
        "labels": {
            "actions": list(labels["action"]),
            "stances": list(labels["stance"]),
            "topics": list(labels["topic"]),
        },
        "feature_schema": schema,
        "categorical_models": categorical_models,
        "state_transition_model": state_transition,
        "global": {
            **global_distributions,
            "continuous_deltas": continuous_global,
        },
        "training_contract": {
            "sample_weight": "reviewed_population_weight_normalized_over_training_rows",
            "categorical_estimator": "weighted_multinomial_logistic_l2",
            "continuous_estimator": "weighted_multivariate_ridge_transition",
            "finite_interactions": [
                f"{left}*{right}" for left, right in MODEL_INTERACTIONS
            ],
            "formal_branch_outcomes_used_for_tuning": False,
        },
        "online_parameter_updates_allowed": False,
        "output_provenance": "llm_behavior_emulator_not_provider_response",
    }
    body["emulator_sha256"] = canonical_hash(body)
    return body


def _offset_probability_distribution(
    probabilities: Mapping[str, Any],
    *,
    labels: Sequence[str],
    logit_offsets: Sequence[float],
) -> dict[str, float]:
    if len(labels) != len(logit_offsets):
        raise LLMPopulationError("probability calibration shape drifted")
    logits = []
    for label, offset in zip(labels, logit_offsets):
        probability = max(_safe_float(probabilities.get(label)), 0.0)
        logits.append(
            -math.inf
            if probability <= 0.0
            else math.log(probability) + float(offset)
        )
    finite = [value for value in logits if math.isfinite(value)]
    if not finite:
        return _normalize({}, labels)
    maximum = max(finite)
    masses = {
        label: (
            math.exp(max(-700.0, value - maximum))
            if math.isfinite(value)
            else 0.0
        )
        for label, value in zip(labels, logits)
    }
    return _normalize(masses, labels)


def _restrict_action_distribution(
    probabilities: Mapping[str, Any],
    *,
    archetype_id: str,
) -> dict[str, float]:
    allowed = RESTRICTED_ARCHETYPE_ACTIONS.get(archetype_id)
    if allowed is None:
        return _normalize(probabilities, tuple(probabilities))
    return _normalize(
        {
            label: _safe_float(probability) if label in allowed else 0.0
            for label, probability in probabilities.items()
        },
        tuple(probabilities),
    )


def calibrate_emulator_probability_intercepts_v1(
    *,
    emulator_release: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    semantic_contract_id: str = BUDGETED_SEMANTIC_CONTRACT_ID,
    dirichlet_alpha: float = 0.1,
) -> dict[str, Any]:
    """Fit contract-domain intercept offsets without using the holdout panel.

    The calibration changes only categorical probability intercepts.  It keeps
    the historical GLM's relative agent/phase variation and moment-matches the
    reviewed longitudinal-anchor panel after a small symmetric Dirichlet
    smoothing prior.  Validation rows must be held out by the caller.
    """

    if not calibration_rows:
        raise LLMPopulationError("probability calibration requires anchor rows")
    if not semantic_contract_id:
        raise LLMPopulationError("semantic contract id is required")
    if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0.0:
        raise LLMPopulationError("Dirichlet calibration alpha must be positive")
    supplied_sha = str(emulator_release.get("emulator_sha256") or "")
    release_body = {
        key: copy.deepcopy(value)
        for key, value in emulator_release.items()
        if key != "emulator_sha256"
    }
    if supplied_sha != canonical_hash(release_body):
        raise LLMPopulationError("emulator release hash is invalid")
    if emulator_release.get("probability_calibrations"):
        raise LLMPopulationError("emulator release is already calibrated")

    rows = []
    for raw in calibration_rows:
        row = copy.deepcopy(dict(raw))
        existing_contract = str(row.get("semantic_contract_id") or "")
        if existing_contract and existing_contract != semantic_contract_id:
            raise LLMPopulationError("calibration row semantic contract drifted")
        row["semantic_contract_id"] = semantic_contract_id
        rows.append(row)
    base_predictions = predict_llm_behavior(emulator_release, rows)
    weights = _row_weights(rows)
    effective_sample_size = float(1.0 / np.square(weights).sum())
    targets = {
        "action": tuple(
            emulator_release["categorical_models"]["action"]["labels"]
        ),
        "stance": tuple(
            emulator_release["categorical_models"]["stance"]["labels"]
        ),
        "topic": tuple(
            emulator_release["categorical_models"]["topic"]["labels"]
        ),
    }
    calibration_targets: dict[str, Any] = {}
    for target_name, labels in targets.items():
        field = MODEL_TARGET_FIELDS[target_name]
        probability_key = f"{target_name}_probabilities"
        observed = {label: 0.0 for label in labels}
        base_mean = {label: 0.0 for label in labels}
        for index, row in enumerate(rows):
            observed_label = str(row.get(field) or "")
            if observed_label not in observed:
                raise LLMPopulationError(
                    f"calibration {target_name} label is outside the frozen schema"
                )
            observed[observed_label] += float(weights[index])
            for label in labels:
                base_mean[label] += float(weights[index]) * _safe_float(
                    base_predictions[index][probability_key].get(label)
                )
        smoothed = _normalize(
            {
                label: observed[label] * effective_sample_size
                + dirichlet_alpha
                for label in labels
            },
            labels,
        )
        offsets = np.zeros(len(labels), dtype=np.float64)
        fitted = dict(base_mean)
        iterations = 0
        for iterations in range(1, 201):
            fitted = {label: 0.0 for label in labels}
            for index, prediction in enumerate(base_predictions):
                adjusted = _offset_probability_distribution(
                    prediction[probability_key],
                    labels=labels,
                    logit_offsets=offsets,
                )
                for label in labels:
                    fitted[label] += float(weights[index]) * adjusted[label]
            residual = np.asarray(
                [
                    math.log(max(smoothed[label], 1e-15))
                    - math.log(max(fitted[label], 1e-15))
                    for label in labels
                ],
                dtype=np.float64,
            )
            if float(np.max(np.abs(residual))) <= 1e-12:
                break
            offsets += residual
            offsets -= float(offsets.mean())
        calibration_targets[target_name] = {
            "labels": list(labels),
            "logit_offsets": [float(value) for value in offsets],
            "observed_anchor_share": observed,
            "smoothed_target_share": smoothed,
            "base_predicted_share": _normalize(base_mean, labels),
            "fitted_share": _normalize(fitted, labels),
            "iterations": iterations,
        }
        # A single current-contract observation per archetype cannot identify
        # current-contract conditional effects.  Continuing to use legacy
        # archetype coefficients created severe extrapolation even when their
        # aggregate anchor moment was exact.  Freeze an intercept-only current
        # contract distribution and keep mandatory archetype action gates.
        global_distribution = dict(smoothed)
        global_fitted = dict(smoothed)
        global_iterations = 0
        if target_name == "action":
            for global_iterations in range(1, 201):
                global_fitted = {label: 0.0 for label in labels}
                for index, row in enumerate(rows):
                    adjusted = _restrict_action_distribution(
                        global_distribution,
                        archetype_id=str(row.get("archetype_id") or ""),
                    )
                    for label in labels:
                        global_fitted[label] += float(weights[index]) * adjusted[label]
                residual = {
                    label: smoothed[label] / max(global_fitted[label], 1e-15)
                    for label in labels
                }
                if max(
                    abs(math.log(max(value, 1e-15)))
                    for value in residual.values()
                ) <= 1e-12:
                    break
                global_distribution = _normalize(
                    {
                        label: global_distribution[label] * residual[label]
                        for label in labels
                    },
                    labels,
                )
        calibration_targets[target_name].update(
            {
                "application_mode": (
                    "contract_global_distribution_with_archetype_action_gates"
                    if target_name == "action"
                    else "contract_global_distribution"
                ),
                "contract_global_distribution": global_distribution,
                "contract_global_fitted_share": global_fitted,
                "contract_global_iterations": global_iterations,
                "current_contract_conditional_effects_identified": False,
            }
        )

    calibration_evidence = {
        "semantic_contract_id": semantic_contract_id,
        "training_role": "longitudinal_anchors_only",
        "holdout_role": "rotating_pps_excluded",
        "row_count": len(rows),
        "effective_sample_size": effective_sample_size,
        "dirichlet_alpha": float(dirichlet_alpha),
        "row_evidence_sha256": canonical_hash(
            [
                {
                    "archetype_id": row.get("archetype_id"),
                    "persona_variant_id": row.get("persona_variant_id"),
                    "scenario_id": row.get("scenario_id"),
                    "phase_id": row.get("phase_id"),
                    "policy_id": row.get("policy_id"),
                    "semantic_contract_id": row.get("semantic_contract_id"),
                    "population_weight": row.get("population_weight"),
                    "action": row.get("action"),
                    "stance": row.get("stance"),
                    "topic_id": row.get("topic_id"),
                }
                for row in rows
            ]
        ),
        "targets": calibration_targets,
    }
    calibrated = copy.deepcopy(release_body)
    calibrated["method"] = (
        str(calibrated.get("method") or "")
        + "+semantic_contract_intercept_calibration_v1"
    )
    calibrated["probability_calibrations"] = {
        "schema_version": "llm-behavior-emulator-probability-calibration-v1",
        "items": {semantic_contract_id: calibration_evidence},
    }
    calibrated["emulator_sha256"] = canonical_hash(calibrated)
    return calibrated


def predict_llm_behavior(
    emulator_release: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Predict a batch of agent behaviors without emitting free-form text."""

    if emulator_release.get("schema_version") != EMULATOR_RELEASE_SCHEMA_VERSION:
        raise LLMPopulationError("unknown emulator release")
    schema = _object(
        emulator_release.get("feature_schema"),
        "emulator feature schema",
    )
    models = _object(
        emulator_release.get("categorical_models"),
        "emulator categorical models",
    )
    transition = _object(
        emulator_release.get("state_transition_model"),
        "emulator state transition model",
    )
    fallback = _object(emulator_release.get("global"), "emulator global")
    matrix = _encode_features(feature_rows, schema)
    probability_by_target = {}
    for target_name in MODEL_TARGET_FIELDS:
        model = _object(models.get(target_name), f"{target_name} model")
        coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
        if coefficients.shape != (
            matrix.shape[1],
            len(model.get("labels") or ()),
        ):
            raise LLMPopulationError(
                f"{target_name} model coefficients do not match its schema"
            )
        temperature = _safe_float(model.get("temperature"), 1.0)
        if temperature <= 0.0:
            raise LLMPopulationError(
                f"{target_name} model temperature must be positive"
            )
        probability_by_target[target_name] = _softmax(
            (matrix @ coefficients) / temperature
        )
    transition_coefficients = np.asarray(
        transition.get("coefficients"),
        dtype=np.float64,
    )
    transition_targets = list(transition.get("targets") or ())
    if transition_coefficients.shape != (
        matrix.shape[1],
        len(transition_targets),
    ):
        raise LLMPopulationError(
            "state transition coefficients do not match feature schema"
        )
    continuous = matrix @ transition_coefficients
    results = []
    for row_index, row in enumerate(feature_rows):
        action_model = models["action"]
        stance_model = models["stance"]
        topic_model = models["topic"]
        action_probabilities = {
            label: float(probability_by_target["action"][row_index, index])
            for index, label in enumerate(action_model["labels"])
        }
        archetype = str(row.get("archetype_id") or "")
        if not archetype:
            archetype = str(row.get("primary_archetype_id") or "")
        action_probabilities = _restrict_action_distribution(
            action_probabilities,
            archetype_id=archetype,
        )
        topic_probabilities = {
            label: float(probability_by_target["topic"][row_index, index])
            for index, label in enumerate(topic_model["labels"])
        }
        stance_probabilities = {
            label: float(probability_by_target["stance"][row_index, index])
            for index, label in enumerate(stance_model["labels"])
        }
        calibration_root = emulator_release.get("probability_calibrations")
        if isinstance(calibration_root, Mapping):
            calibrations = calibration_root.get("items")
            contract_id = str(row.get("semantic_contract_id") or "")
            calibration = (
                calibrations.get(contract_id)
                if isinstance(calibrations, Mapping)
                else None
            )
            if isinstance(calibration, Mapping):
                target_specs = calibration.get("targets")
                if not isinstance(target_specs, Mapping):
                    raise LLMPopulationError(
                        "probability calibration targets are invalid"
                    )
                distributions = {
                    "action": action_probabilities,
                    "topic": topic_probabilities,
                    "stance": stance_probabilities,
                }
                for target_name, distribution in distributions.items():
                    specification = target_specs.get(target_name)
                    if not isinstance(specification, Mapping):
                        raise LLMPopulationError(
                            f"{target_name} probability calibration is absent"
                        )
                    mode = str(specification.get("application_mode") or "")
                    if mode.startswith("contract_global_distribution"):
                        global_distribution = specification.get(
                            "contract_global_distribution"
                        )
                        if not isinstance(global_distribution, Mapping):
                            raise LLMPopulationError(
                                f"{target_name} contract-global distribution is absent"
                            )
                        calibrated_distribution = _normalize(
                            {
                                label: _safe_float(global_distribution.get(label))
                                for label in tuple(specification.get("labels") or ())
                            },
                            tuple(specification.get("labels") or ()),
                        )
                        if target_name == "action":
                            calibrated_distribution = _restrict_action_distribution(
                                calibrated_distribution,
                                archetype_id=archetype,
                            )
                    else:
                        calibrated_distribution = _offset_probability_distribution(
                            distribution,
                            labels=tuple(specification.get("labels") or ()),
                            logit_offsets=tuple(
                                specification.get("logit_offsets") or ()
                            ),
                        )
                    if target_name == "action":
                        action_probabilities = calibrated_distribution
                    elif target_name == "topic":
                        topic_probabilities = calibrated_distribution
                    else:
                        stance_probabilities = calibrated_distribution
        action = max(action_probabilities, key=action_probabilities.get)
        topic = max(topic_probabilities, key=topic_probabilities.get)
        stance = max(stance_probabilities, key=stance_probabilities.get)
        deltas = {
            target: float(
                np.clip(
                    continuous[row_index, index],
                    -1.0 if target.endswith("_delta") else 0.0,
                    1.0,
                )
            )
            for index, target in enumerate(transition_targets)
        }
        for target in NUMERIC_TARGETS:
            deltas.setdefault(
                target,
                _safe_float(fallback.get("continuous_deltas", {}).get(target)),
            )
        results.append(
            {
                "action_probabilities": action_probabilities,
                "topic_probabilities": topic_probabilities,
                "stance_probabilities": stance_probabilities,
                "continuous_deltas": deltas,
                "emission_packet": {
                    "source": "llm_behavior_emulator",
                    "provider_response": False,
                    "free_text": None,
                    "action": action,
                    "topic": topic,
                    "stance": stance,
                    "urgency": _safe_float(deltas.get("urgency")),
                },
            }
        )
    return results


def _validation_label_schema(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    return {
        "action": tuple(ACTION_LABELS),
        "stance": tuple(STANCE_LABELS),
        "topic": tuple(
            sorted(
                {
                    str(row.get("topic_id") or "off_topic_daily_life")
                    for row in entries
                }
            )
        ),
    }


def _fit_archetype_phase_frequency_baseline(
    training: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    weights = _row_weights(training)
    model: dict[str, Any] = {}
    for target_name, field in MODEL_TARGET_FIELDS.items():
        target_labels = tuple(labels[target_name])
        global_counts = Counter()
        group_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        group_mass: Counter[tuple[str, str]] = Counter()
        for index, row in enumerate(training):
            weight = float(weights[index])
            value = str(row.get(field) or "")
            group = (
                _categorical_input(row, "archetype_id"),
                _categorical_input(row, "phase_id"),
            )
            global_counts[value] += weight
            group_counts[group][value] += weight
            group_mass[group] += weight
        global_distribution = _normalize(global_counts, target_labels)
        groups = {}
        # A fixed 5% total-mass empirical-Bayes prior prevents zero probability
        # without turning the baseline into the fitted GLM.
        prior_strength = 0.05
        for group, counts in group_counts.items():
            groups["|".join(group)] = _normalize(
                {
                    label: counts[label]
                    + prior_strength * global_distribution[label]
                    for label in target_labels
                },
                target_labels,
            )
        model[target_name] = {
            "global": global_distribution,
            "groups": groups,
            "labels": list(target_labels),
            "group_fields": ["archetype_id", "phase_id"],
        }
    return model


def _predict_frequency_baseline(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, dict[str, float]]]:
    predictions = []
    for row in rows:
        group = "|".join(
            (
                _categorical_input(row, "archetype_id"),
                _categorical_input(row, "phase_id"),
            )
        )
        item = {}
        for target_name in MODEL_TARGET_FIELDS:
            target = model[target_name]
            item[f"{target_name}_probabilities"] = dict(
                target["groups"].get(group, target["global"])
            )
        predictions.append(item)
    return predictions


def _weighted_share_pair(
    records: Sequence[Mapping[str, Any]],
    *,
    target_name: str,
    labels: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    field = MODEL_TARGET_FIELDS[target_name]
    weights = _row_weights([record["row"] for record in records])
    observed = {label: 0.0 for label in labels}
    predicted = {label: 0.0 for label in labels}
    probability_key = f"{target_name}_probabilities"
    for index, record in enumerate(records):
        observed[str(record["row"].get(field) or "")] += float(weights[index])
        for label in labels:
            predicted[label] += float(weights[index]) * _safe_float(
                record["prediction"][probability_key].get(label)
            )
    predicted = _normalize(predicted, labels)
    return observed, predicted


def _ece(
    records: Sequence[Mapping[str, Any]],
    *,
    target_name: str,
    bins: int = 10,
) -> float:
    field = MODEL_TARGET_FIELDS[target_name]
    probability_key = f"{target_name}_probabilities"
    weights = _row_weights([record["row"] for record in records])
    value = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = []
        for index, record in enumerate(records):
            probabilities = record["prediction"][probability_key]
            predicted_label = max(probabilities, key=probabilities.get)
            confidence = _safe_float(probabilities[predicted_label])
            if lower <= confidence < upper or (
                bin_index == bins - 1 and confidence == 1.0
            ):
                selected.append(
                    (
                        index,
                        confidence,
                        float(
                            predicted_label
                            == str(record["row"].get(field) or "")
                        ),
                    )
                )
        mass = sum(float(weights[index]) for index, _, _ in selected)
        if mass <= 0.0:
            continue
        bin_confidence = sum(
            float(weights[index]) * confidence
            for index, confidence, _ in selected
        ) / mass
        bin_accuracy = sum(
            float(weights[index]) * correct
            for index, _, correct in selected
        ) / mass
        value += mass * abs(bin_confidence - bin_accuracy)
    return float(value)


def _validation_metrics(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not records:
        raise LLMPopulationError("validation split produced no predictions")
    slices: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        row = record["row"]
        slices[
            (
                str(row.get("scenario_id") or "unknown"),
                str(row.get("seed") or "unknown"),
                str(row.get("policy_id") or "unknown"),
                str(row.get("phase_id") or "unknown"),
            )
        ].append(record)
    slice_shares: dict[
        tuple[str, str, str, str],
        dict[str, tuple[dict[str, float], dict[str, float]]],
    ] = {}
    level_errors = {target: [] for target in MODEL_TARGET_FIELDS}
    for key, values in slices.items():
        slice_shares[key] = {}
        for target_name in MODEL_TARGET_FIELDS:
            pair = _weighted_share_pair(
                values,
                target_name=target_name,
                labels=labels[target_name],
            )
            slice_shares[key][target_name] = pair
            observed, predicted = pair
            level_errors[target_name].extend(
                abs(observed[label] - predicted[label])
                for label in labels[target_name]
            )
    share_mae = {
        target: float(sum(errors) / max(1, len(errors)))
        for target, errors in level_errors.items()
    }

    first_difference_errors = {
        target: [] for target in MODEL_TARGET_FIELDS
    }
    scenario_seeds = sorted({key[:2] for key in slice_shares})
    for scenario_id, seed_id in scenario_seeds:
        shared_baseline = slice_shares.get(
            (scenario_id, seed_id, "shared_baseline", "baseline")
        )
        policies = sorted(
            {
                key[2]
                for key in slice_shares
                if key[:2] == (scenario_id, seed_id)
                and key[2] != "shared_baseline"
            }
        )
        for policy_id in policies:
            ordered: list[
                dict[str, tuple[dict[str, float], dict[str, float]]]
            ] = []
            if shared_baseline is not None:
                ordered.append(shared_baseline)
            ordered.extend(
                slice_shares[(scenario_id, seed_id, policy_id, phase_id)]
                for phase_id in sorted(
                    PHASE_ORDER,
                    key=PHASE_ORDER.get,
                )
                if (scenario_id, seed_id, policy_id, phase_id) in slice_shares
            )
            for previous, current in zip(ordered, ordered[1:]):
                for target_name in MODEL_TARGET_FIELDS:
                    previous_observed, previous_predicted = previous[target_name]
                    current_observed, current_predicted = current[target_name]
                    first_difference_errors[target_name].extend(
                        abs(
                            (
                                current_observed[label]
                                - previous_observed[label]
                            )
                            - (
                                current_predicted[label]
                                - previous_predicted[label]
                            )
                        )
                        for label in labels[target_name]
                    )
    first_difference_by_target = {
        target: (
            float(sum(errors) / len(errors))
            if errors
            else None
        )
        for target, errors in first_difference_errors.items()
    }
    first_difference_values = [
        value
        for value in first_difference_by_target.values()
        if value is not None
    ]
    first_difference_mae = (
        max(first_difference_values) if first_difference_values else None
    )

    row_weights = _row_weights([record["row"] for record in records])
    task_losses = {}
    baseline_task_losses = {}
    ece_by_target = {}
    for target_name, field in MODEL_TARGET_FIELDS.items():
        probability_key = f"{target_name}_probabilities"
        loss = 0.0
        baseline_loss = 0.0
        for index, record in enumerate(records):
            true_label = str(record["row"].get(field) or "")
            loss -= float(row_weights[index]) * math.log(
                max(
                    1e-12,
                    _safe_float(
                        record["prediction"][probability_key].get(true_label)
                    ),
                )
            )
            baseline_loss -= float(row_weights[index]) * math.log(
                max(
                    1e-12,
                    _safe_float(
                        record["baseline"][probability_key].get(true_label)
                    ),
                )
            )
        task_losses[target_name] = float(loss)
        baseline_task_losses[target_name] = float(baseline_loss)
        ece_by_target[target_name] = _ece(
            records,
            target_name=target_name,
        )
    weighted_log_loss = sum(task_losses.values()) / len(task_losses)
    frequency_baseline_log_loss = (
        sum(baseline_task_losses.values()) / len(baseline_task_losses)
    )
    improvement = (
        (frequency_baseline_log_loss - weighted_log_loss)
        / frequency_baseline_log_loss
        if frequency_baseline_log_loss > 0.0
        else 0.0
    )
    return {
        "rows": len(records),
        "slice_count": len(slices),
        "action_share_mae": share_mae["action"],
        "stance_share_mae": share_mae["stance"],
        "topic_share_mae": share_mae["topic"],
        "first_difference_mae": first_difference_mae,
        "first_difference_mae_by_target": first_difference_by_target,
        "weighted_log_loss": weighted_log_loss,
        "weighted_log_loss_by_target": task_losses,
        "frequency_baseline_log_loss": frequency_baseline_log_loss,
        "frequency_baseline_log_loss_by_target": baseline_task_losses,
        "relative_log_loss_improvement": improvement,
        "ece": max(ece_by_target.values()),
        "ece_by_target": ece_by_target,
        "share_estimator": "weighted_observed_one_hot_vs_predicted_probability_mass",
    }


def _evaluate_validation_splits(
    entries: Sequence[Mapping[str, Any]],
    splits: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    *,
    labels: Mapping[str, Sequence[str]],
    seed: int,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    reports = {}
    all_records: list[Mapping[str, Any]] = []
    for split_id, training, holdout in splits:
        if not training or not holdout:
            reports[str(split_id)] = {
                "status": "not_evaluable",
                "training_rows": len(training),
                "holdout_rows": len(holdout),
            }
            continue
        training_release = {
            "status": "incomplete",
            "entries": list(training),
            "trace_release_sha256": canonical_hash(training),
        }
        emulator = fit_llm_behavior_emulator_v1(
            trace_release=training_release,
            seed=seed,
            allow_incomplete_for_testing=True,
            label_schema=labels,
        )
        predictions = predict_llm_behavior(emulator, holdout)
        baseline = _fit_archetype_phase_frequency_baseline(training, labels)
        baseline_predictions = _predict_frequency_baseline(baseline, holdout)
        records = [
            {
                "row": row,
                "prediction": prediction,
                "baseline": baseline_prediction,
                "split_id": str(split_id),
            }
            for row, prediction, baseline_prediction in zip(
                holdout,
                predictions,
                baseline_predictions,
            )
        ]
        reports[str(split_id)] = {
            "status": "evaluated",
            "training_rows": len(training),
            "holdout_rows": len(holdout),
            "metrics": _validation_metrics(records, labels),
        }
        all_records.extend(records)
    return reports, all_records


def validate_llm_behavior_emulator_v1(
    *,
    trace_release: Mapping[str, Any],
    seed: int = 20260731,
) -> dict[str, Any]:
    """Run true grouped, leave-one-seed and leave-one-scenario validation."""

    entries = [
        _object(row, "trace entry")
        for row in _list(trace_release.get("entries"), "trace entries")
    ]
    agents = sorted({str(row.get("synthetic_agent_id")) for row in entries})
    if len(agents) < 2:
        raise LLMPopulationError("grouped validation needs at least two synthetic agents")
    labels = _validation_label_schema(entries)
    fold_by_agent = {
        agent_id: int(canonical_hash([seed, agent_id])[:8], 16) % 5
        for agent_id in agents
    }
    grouped_splits = [
        (
            str(fold),
            [
                row
                for row in entries
                if fold_by_agent[str(row.get("synthetic_agent_id"))] != fold
            ],
            [
                row
                for row in entries
                if fold_by_agent[str(row.get("synthetic_agent_id"))] == fold
            ],
        )
        for fold in range(5)
    ]
    grouped_reports, grouped_records = _evaluate_validation_splits(
        entries,
        grouped_splits,
        labels=labels,
        seed=seed,
    )
    metrics = _validation_metrics(grouped_records, labels)

    seeds = sorted({str(row.get("seed")) for row in entries})
    seed_splits = [
        (
            seed_id,
            [row for row in entries if str(row.get("seed")) != seed_id],
            [row for row in entries if str(row.get("seed")) == seed_id],
        )
        for seed_id in seeds
    ]
    seed_reports, _ = _evaluate_validation_splits(
        entries,
        seed_splits,
        labels=labels,
        seed=seed,
    )
    scenarios = sorted({str(row.get("scenario_id")) for row in entries})
    scenario_splits = [
        (
            scenario_id,
            [
                row
                for row in entries
                if str(row.get("scenario_id")) != scenario_id
            ],
            [
                row
                for row in entries
                if str(row.get("scenario_id")) == scenario_id
            ],
        )
        for scenario_id in scenarios
    ]
    scenario_reports, _ = _evaluate_validation_splits(
        entries,
        scenario_splits,
        labels=labels,
        seed=seed,
    )
    seed_reported = (
        len(seed_reports) >= 2
        and all(value.get("status") == "evaluated" for value in seed_reports.values())
    )
    scenario_reported = (
        len(scenario_reports) >= 2
        and all(
            value.get("status") == "evaluated"
            for value in scenario_reports.values()
        )
    )
    gate = {
        "action_share_mae_lte_0_05": metrics["action_share_mae"] <= 0.05,
        "stance_share_mae_lte_0_05": metrics["stance_share_mae"] <= 0.05,
        "topic_share_mae_lte_0_05": metrics["topic_share_mae"] <= 0.05,
        "first_difference_mae_lte_0_05": (
            isinstance(metrics["first_difference_mae"], (int, float))
            and math.isfinite(metrics["first_difference_mae"])
            and metrics["first_difference_mae"] <= 0.05
        ),
        "log_loss_improvement_gte_0_10": (
            metrics["relative_log_loss_improvement"] >= 0.10
        ),
        "ece_lte_0_10": metrics["ece"] <= 0.10,
        "leave_one_seed_reported": seed_reported,
        "leave_one_scenario_reported": scenario_reported,
    }
    passed = all(gate.values())
    body = {
        "schema_version": EMULATOR_VALIDATION_SCHEMA_VERSION,
        "seed": int(seed),
        "group_split": "synthetic_agent_id_five_fold_no_role_overlap",
        "grouped_holdout": grouped_reports,
        "leave_one_seed": seed_reports,
        "leave_one_scenario": scenario_reports,
        "metrics": metrics,
        "gate": gate,
        "passed": passed,
        "publication_eligible": (
            passed and trace_release.get("status") == "complete"
        ),
        "limitations": [
            "Historical traces contain two seeds and two scenarios; leave-one reports are executable but uncertainty remains broad.",
            "First differences compare weighted probability shares across the ordered baseline/burst/spread/decay trajectory.",
            "This validates LLM behavior emulation, not agreement with human opinion.",
        ],
        "fold_rows": {
            key: value.get("holdout_rows", 0)
            for key, value in grouped_reports.items()
        },
    }
    body["validation_sha256"] = canonical_hash(body)
    return body


def evaluate_scale_gap_v1(
    panel_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260731,
    repetitions: int = 200,
) -> dict[str, Any]:
    """Evaluate five finite-population recovery designs on a 48-turn panel.

    Interval radii are calibrated on the first half of deterministic Monte
    Carlo repetitions and evaluated on the untouched second half.  The GREG
    working model is deliberately outcome-free unless callers provide frozen
    ``emulator_<dimension>_probabilities`` on every row.  Therefore this
    function cannot pass by reading the full panel outcomes into its surrogate.
    """

    if len(panel_rows) != 48:
        raise LLMPopulationError("Scale-Gap evaluation requires a 48-row LLM panel")
    if repetitions < 40 or repetitions % 2:
        raise LLMPopulationError(
            "Scale-Gap evaluation needs an even repetition count of at least 40"
        )
    records = tuple(dict(_object(row, "Scale-Gap row")) for row in panel_rows)
    categories = {
        dimension: tuple(
            sorted({str(row.get(dimension) or "") for row in records})
        )
        for dimension in SCALE_GAP_DIMENSIONS
    }
    if any(not values or "" in values for values in categories.values()):
        raise LLMPopulationError(
            "Scale-Gap rows require action, stance and topic_id"
        )
    cell = FinitePopulationCell(
        cell_id=canonical_hash(
            [
                {
                    "synthetic_agent_id": row.get("synthetic_agent_id"),
                    "archetype_id": row.get("archetype_id"),
                    "population_weight": row.get("population_weight"),
                }
                for row in records
            ]
        ),
        records=records,
        categories=categories,
    )
    truth = _true_distribution(cell)

    # Frozen emulator probabilities, if present, are legitimate presampling
    # auxiliaries.  Otherwise a uniform prior makes GREG reduce to a transparent
    # design correction instead of leaking the finite-population outcomes.
    surrogate: dict[str, dict[str, dict[str, float]]] = {}
    external_auxiliary_complete = True
    for dimension in SCALE_GAP_DIMENSIONS:
        target_name = "topic" if dimension == "topic_id" else dimension
        key = f"emulator_{target_name}_probabilities"
        supplied = []
        for row in records:
            probabilities = row.get(key)
            if not isinstance(probabilities, Mapping):
                external_auxiliary_complete = False
                supplied = []
                break
            supplied.append(
                _normalize(probabilities, categories[dimension])
            )
        if supplied:
            global_distribution = {
                category: sum(
                    supplied[index][category]
                    * max(_safe_float(records[index].get("population_weight")), 1e-12)
                    for index in range(len(records))
                )
                for category in categories[dimension]
            }
            global_distribution = _normalize(
                global_distribution,
                categories[dimension],
            )
            by_archetype: dict[str, dict[str, float]] = {}
            for archetype in sorted(
                {str(row.get("archetype_id") or "generic") for row in records}
            ):
                indices = [
                    index
                    for index, row in enumerate(records)
                    if str(row.get("archetype_id") or "generic") == archetype
                ]
                by_archetype[archetype] = _normalize(
                    {
                        category: sum(
                            supplied[index][category]
                            * max(
                                _safe_float(
                                    records[index].get("population_weight")
                                ),
                                1e-12,
                            )
                            for index in indices
                        )
                        for category in categories[dimension]
                    },
                    categories[dimension],
                )
        else:
            global_distribution = {
                category: 1.0 / len(categories[dimension])
                for category in categories[dimension]
            }
            by_archetype = {}
        surrogate[dimension] = {
            "__global__": global_distribution,
            **by_archetype,
        }

    def quantile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    internal_method = {
        "srs": "srs",
        "stratified": "stratified",
        "pps_ht": "pps",
        "greg_ht": "greg_pps",
        "deterministic_critical_node_replacement": "critical_replacement",
    }
    results = []
    for budget in SCALE_GAP_BUDGETS:
        for method in SCALE_GAP_METHODS:
            repeat_count = (
                1
                if method == "deterministic_critical_node_replacement"
                else repetitions
            )
            draws = []
            representative_selected: list[int] = []
            fixed_budget = True
            selected_pi_positive = True
            all_population_pi_positive = True
            for repetition in range(repeat_count):
                draw_seed = int(
                    canonical_hash(
                        [seed, cell.cell_id, budget, method, repetition]
                    )[:16],
                    16,
                )
                design = _design_for_cell(
                    cell,
                    budget,
                    internal_method[method],
                    random.Random(draw_seed),
                )
                if repetition == 0:
                    representative_selected = list(design.selected)
                fixed_budget = fixed_budget and len(design.selected) == budget
                selected_pi_positive = selected_pi_positive and all(
                    design.inclusion_probability[index] > 0.0
                    for index in design.selected
                )
                all_population_pi_positive = all_population_pi_positive and all(
                    probability > 0.0
                    for probability in design.inclusion_probability
                )
                estimate = estimate_cell_distribution(
                    cell,
                    design,
                    surrogate,
                )
                errors = {
                    dimension: [
                        abs(
                            float(estimate[dimension][category])
                            - float(truth[dimension][category])
                        )
                        for category in categories[dimension]
                    ]
                    for dimension in SCALE_GAP_DIMENSIONS
                }
                draws.append({"estimate": estimate, "errors": errors})
            full_budget_exact = all(
                error <= 1e-12
                for draw in draws
                for errors in draw["errors"].values()
                for error in errors
            )
            if repeat_count == 1:
                evaluation_draws = draws
                interval_radius = {
                    dimension: None for dimension in SCALE_GAP_DIMENSIONS
                }
                coverage = {
                    dimension: None for dimension in SCALE_GAP_DIMENSIONS
                }
            else:
                midpoint = repeat_count // 2
                calibration_draws = draws[:midpoint]
                evaluation_draws = draws[midpoint:]
                interval_radius = {
                    dimension: quantile(
                        [
                            error
                            for draw in calibration_draws
                            for error in draw["errors"][dimension]
                        ],
                        0.95,
                    )
                    for dimension in SCALE_GAP_DIMENSIONS
                }
                coverage = {
                    dimension: sum(
                        error <= interval_radius[dimension] + 1e-12
                        for draw in evaluation_draws
                        for error in draw["errors"][dimension]
                    )
                    / max(
                        1,
                        sum(
                            len(draw["errors"][dimension])
                            for draw in evaluation_draws
                        ),
                    )
                    for dimension in SCALE_GAP_DIMENSIONS
                }
            mae = {
                dimension: sum(
                    error
                    for draw in evaluation_draws
                    for error in draw["errors"][dimension]
                )
                / max(
                    1,
                    sum(
                        len(draw["errors"][dimension])
                        for draw in evaluation_draws
                    ),
                )
                for dimension in SCALE_GAP_DIMENSIONS
            }
            comparable_coverage = [
                value for value in coverage.values() if value is not None
            ]
            if method == "deterministic_critical_node_replacement":
                all_population_pi_positive = False
            results.append({
                "budget": budget,
                "method": method,
                "selected_indices": representative_selected,
                "repetitions": repeat_count,
                "fixed_budget": fixed_budget,
                "selected_pi_positive": selected_pi_positive,
                "all_population_pi_positive": all_population_pi_positive,
                "probability_sampling": (
                    method != "deterministic_critical_node_replacement"
                ),
                "action_share_mae": mae["action"],
                "stance_share_mae": mae["stance"],
                "topic_share_mae": mae["topic_id"],
                "aggregate_share_mae": sum(mae.values()) / len(mae),
                "fit_only_interval_radius_95": interval_radius,
                "held_out_coverage_95_by_dimension": coverage,
                "held_out_coverage_95": (
                    sum(comparable_coverage) / len(comparable_coverage)
                    if comparable_coverage
                    else None
                ),
                "full_budget_exact": full_budget_exact,
            })
    selected = next(
        row
        for row in results
        if row["budget"] == 32 and row["method"] == "greg_ht"
    )
    b48_rows = [row for row in results if row["budget"] == 48]
    b48_exact = all(row["full_budget_exact"] for row in b48_rows)
    selected_coverage = selected["held_out_coverage_95"]
    gate = {
        "selected_method": "greg_ht",
        "selected_budget": 32,
        "action_share_mae_lte_0_05": selected["action_share_mae"] <= 0.05,
        "stance_share_mae_lte_0_05": selected["stance_share_mae"] <= 0.05,
        "topic_share_mae_lte_0_05": selected["topic_share_mae"] <= 0.05,
        "coverage_in_0_90_to_0_99": (
            selected_coverage is not None
            and 0.90 <= selected_coverage <= 0.99
        ),
        "fixed_budget": all(row["fixed_budget"] for row in results),
        "selected_pi_positive": selected["selected_pi_positive"],
        "selected_all_population_pi_positive": (
            selected["all_population_pi_positive"]
        ),
        "b48_exact_recovery": b48_exact,
    }
    gate_passed = all(
        value
        for key, value in gate.items()
        if key not in {"selected_method", "selected_budget"}
    )
    body = {
        "schema_version": "mass-sim-scale-gap-v2",
        "budgets": list(SCALE_GAP_BUDGETS),
        "methods": list(SCALE_GAP_METHODS),
        "finite_llm_population_size": 48,
        "repetitions": repetitions,
        "interval_protocol": (
            "first_half_monte_carlo_error_quantile_calibration_then_untouched_"
            "second_half_coverage"
        ),
        "greg_auxiliary_source": (
            "frozen_emulator_probabilities"
            if external_auxiliary_complete
            else "outcome_free_uniform_prior"
        ),
        "results": results,
        "b48_exact_recovery": b48_exact,
        "b32_selected_result": selected,
        "gate": gate,
        "publication_gate_passed": gate_passed,
        "non_claims": [
            "The 48 reviewed LLM turns are an enumerated synthetic LLM population, not students.",
            "The deterministic critical-node comparison has no positive-probability population inference claim.",
        ],
    }
    body["scale_gap_sha256"] = canonical_hash(body)
    return body


__all__ = [
    "ACTIVE_ARCHETYPES",
    "ANCHOR_SELECTION_SCHEMA_VERSION",
    "EMULATOR_RELEASE_SCHEMA_VERSION",
    "EMULATOR_VALIDATION_SCHEMA_VERSION",
    "LLM_AGENT_POPULATION_SCHEMA_VERSION",
    "LLM_AGENT_POPULATION_SIZE",
    "LLMPopulationError",
    "LONGITUDINAL_ANCHOR_COUNT",
    "SCALE_GAP_BUDGETS",
    "TRACE_RELEASE_SCHEMA_VERSION",
    "compile_reviewed_llm_behavior_trace_release_v1",
    "compile_semantic_llm_agent_population_v1",
    "evaluate_scale_gap_v1",
    "fit_llm_behavior_emulator_v1",
    "predict_llm_behavior",
    "select_longitudinal_anchors",
    "validate_llm_behavior_emulator_v1",
]
