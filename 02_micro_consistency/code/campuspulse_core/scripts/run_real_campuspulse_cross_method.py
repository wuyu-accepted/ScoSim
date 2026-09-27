from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from campuspulse_core.governance_arena_v2 import STATE_FIELDS, initialize_particle_ensemble
from comparison_harness.adapters import CampusPulseAdapter
from comparison_harness.adapters.campuspulse import DIMENSION_2
from comparison_harness.campuspulse_assimilation import assimilate_isolated_turn
from comparison_harness.contracts import canonical_json
from comparison_harness.fake_executor import ObservationEnvelope, load_observations
from comparison_harness.fair_initialization import (
    FairRoleInitialization, load_fair_role_initializations,
)
from comparison_harness.prompting import COMMON_SECTIONS, SharedPromptRenderer
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider
from evaluation.dimension2 import case_id


METHOD = "campuspulse"
VARIANT = "cross_method_full"
EXPECTED_OBSERVATIONS = 512
EXPECTED_AGENTS = 350
EXPECTED_TICKS = tuple(range(10))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(STATE_FIELDS, values, strict=True)
    }


def _visible(envelope: ObservationEnvelope) -> list[dict[str, Any]]:
    return [
        {
            "message_id": item.message_id,
            "author_id": item.author_id,
            "text": item.text,
            "visibility": item.visibility,
            "tick": item.tick,
        }
        for item in envelope.observation.visible_context
    ]


def _append_sync(handle, value: Mapping[str, Any]) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _read_jsonl(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row[id_field])
            if identifier in result:
                raise ValueError(f"duplicate {id_field} at {path}:{line_number}")
            result[identifier] = row
    return result


def _validate_panel(envelopes: tuple[ObservationEnvelope, ...]) -> None:
    if len(envelopes) != EXPECTED_OBSERVATIONS:
        raise ValueError("CampusPulse horizontal runner requires exactly 512 observations")
    agents = {row.observation.agent_id for row in envelopes}
    ticks = Counter(row.observation.tick for row in envelopes)
    if len(agents) != EXPECTED_AGENTS:
        raise ValueError("CampusPulse horizontal runner requires exactly 350 agents")
    if tuple(sorted(ticks)) != EXPECTED_TICKS:
        raise ValueError("CampusPulse horizontal runner requires Tick 0 through Tick 9")
    last_tick: dict[str, int] = {}
    for envelope in envelopes:
        agent_id = envelope.observation.agent_id
        tick = envelope.observation.tick
        if tick < last_tick.get(agent_id, -1):
            raise ValueError(f"non-monotone observation order for {agent_id}")
        last_tick[agent_id] = tick


def _initialize_adapters(
    population: Mapping[str, Any],
    roles: Mapping[str, FairRoleInitialization],
    envelopes: tuple[ObservationEnvelope, ...],
    *,
    scenario_id: str,
    seed: int,
) -> dict[str, CampusPulseAdapter]:
    agents = population.get("agents")
    if not isinstance(agents, list) or len(agents) != 1000:
        raise ValueError("frozen CampusPulse population must contain 1000 agents")
    required_agent_fields = {"llm_agent_id", "state", "semantic", "stable_persona", "population_weight"}
    if any(not required_agent_fields.issubset(row) for row in agents):
        raise ValueError(
            "CampusPulse horizontal initialization requires the frozen hero bridge "
            "with llm_agent_id, state, semantic, stable_persona and population_weight"
        )
    by_id = {str(row["llm_agent_id"]): (index, row) for index, row in enumerate(agents)}
    required = {row.observation.agent_id for row in envelopes}
    if not required.issubset(by_id):
        raise ValueError("common observation agents are missing from the CampusPulse population")
    ensemble = initialize_particle_ensemble(
        population, scenario_id=scenario_id, seed=seed
    )
    means = ensemble.parent_means()
    normalized = ensemble.normalized_within_parent()
    adapters: dict[str, CampusPulseAdapter] = {}
    for agent_id in sorted(required):
        index, profile = by_id[agent_id]
        particles = [_state(row) for row in ensemble.values[index]]
        weights = normalized[index].tolist()
        adapter = CampusPulseAdapter(DIMENSION_2["full"])
        adapter.initialize(roles[agent_id].agent_seed(
            initial_state=_state(means[index]),
            initial_particles=particles,
            particle_weights=weights,
        ))
        adapters[agent_id] = adapter
    return adapters


def _capture(adapter: CampusPulseAdapter) -> tuple[dict[str, float], list[dict[str, float]], list[float]]:
    if adapter.particles is None or adapter.weights is None:
        raise ValueError("full CampusPulse adapter lost its particles")
    weights = np.asarray(adapter.weights, dtype=np.float64)
    if len(adapter.particles) != 10 or weights.shape != (10,) or np.any(weights < 0):
        raise ValueError("CampusPulse predecision particle contract drift")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("CampusPulse predecision weights must sum to one")
    matrix = np.asarray(
        [[float(row[field]) for field in STATE_FIELDS] for row in adapter.particles],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("CampusPulse predecision particles are invalid")
    return _state(np.average(matrix, axis=0, weights=weights)), [dict(row) for row in adapter.particles], weights.tolist()


def _advance(
    adapter: CampusPulseAdapter,
    *,
    decision: Mapping[str, Any],
    scenario_id: str,
    tick: int,
    seed: int,
    prompt_sha256: str,
) -> dict[str, Any]:
    before, particles, weights = _capture(adapter)
    native = assimilate_isolated_turn(
        particles=particles,
        particle_weights=weights,
        decision=decision,
        scenario_id=scenario_id,
        variant=VARIANT,
        tick=tick,
        seed=seed,
        prompt_sha256=prompt_sha256,
    )
    adapter.state = dict(native["after_state"])
    adapter.particles = [dict(row) for row in native["after_particles"]]
    adapter.weights = [float(value) for value in native["after_particle_weights"]]
    if native["before_state"] != before:
        for field in STATE_FIELDS:
            if not np.isclose(native["before_state"][field], before[field], atol=1e-6):
                raise ValueError("native assimilation did not consume the captured predecision state")
    return native


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CampusPulse full method on the frozen 512-observation horizontal panel."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--enable-real", action="store_true")
    args = parser.parse_args()
    if args.max_observations is not None and args.max_observations < 1:
        raise ValueError("max-observations must be positive")

    envelopes = load_observations(args.observations)
    _validate_panel(envelopes)
    if args.max_observations is not None:
        envelopes = envelopes[: args.max_observations]
    population = json.loads(args.population.read_text(encoding="utf-8"))
    roles = load_fair_role_initializations(args.population)
    scenario_ids = {row.observation.scenario_id for row in envelopes}
    if len(scenario_ids) != 1:
        raise ValueError("horizontal observations must contain one scenario")
    scenario_id = next(iter(scenario_ids))
    seed = 20260722
    adapters = _initialize_adapters(
        population, roles, envelopes, scenario_id=scenario_id, seed=seed
    )
    renderer = SharedPromptRenderer()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    config = json.loads(args.provider_boundary.read_text(encoding="utf-8"))
    transport = (
        OpenAIProviderTransport(
            api_key_env=config["api_key_environment_variable"],
            timeout_seconds=float(config["request_constraints"]["timeout_seconds"]),
        )
        if args.enable_real else None
    )
    provider = RealModelProvider(
        config_path=args.provider_boundary,
        auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.output_root / "provider_state",
        transport=transport,
        enable_real=bool(args.enable_real),
        budget_profile=(
            "full" if args.enable_real and args.max_observations is None else "pilot"
        ),
    )

    records_path = args.output_root / "records.jsonl"
    audits_path = args.output_root / "prompt_audit.jsonl"
    assimilation_path = args.output_root / "assimilation_sidecar.jsonl"
    existing = _read_jsonl(records_path, "source_record_id") if args.resume else {}
    existing_audits = _read_jsonl(audits_path, "source_record_id") if args.resume else {}
    existing_assimilation = _read_jsonl(assimilation_path, "source_record_id") if args.resume else {}
    if not args.resume and any(path.exists() for path in (records_path, audits_path, assimilation_path)):
        raise FileExistsError("existing CampusPulse horizontal output requires --resume")
    if set(existing) != set(existing_audits) or set(existing) != set(existing_assimilation):
        raise ValueError("resume artifacts do not cover the same settled records")

    target_count = len(envelopes)
    pending_count = target_count - len(existing)
    manifest = {
        "schema_version": "campuspulse-dimension2-cross-method-run-v1.0.0",
        "scientific_result": bool(args.enable_real and args.max_observations is None),
        "execution_mode": "real" if args.enable_real else "dry_run_no_transport",
        "scenario_id": scenario_id,
        "seed": seed,
        "method": METHOD,
        "variant": VARIANT,
        "required_record_count": target_count,
        "full_panel_record_count": EXPECTED_OBSERVATIONS,
        "agent_count": len(adapters),
        "settled_record_count": len(existing),
        "pending_record_count": pending_count,
        "preflight_call_count": pending_count,
        "network_calls_made": 0,
        "predecision_capture": "ten identity-bound particles immediately before every main decision",
        "state_transition": "campuspulse-action-assimilation-v1.0.0 after every settled decision",
        "resume_semantics": "offline replay of all settled decisions before first pending slot",
        "execution_order": "capture_prestate_then_render_then_call_then_assimilate_then_fsync_sidecar_audit_and_record_commit_marker",
        "dry_run_scope": "structural contract only; later longitudinal prompts require prior real decisions",
        "common_prompt_sections": list(COMMON_SECTIONS),
        "input_sha256": {
            "observations": _sha256(args.observations),
            "population": _sha256(args.population),
            "provider_boundary": _sha256(args.provider_boundary),
            "auxiliary_boundary": _sha256(args.auxiliary_boundary),
            "schema": _sha256(args.schema),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    initialization_path = args.output_root / "initialization_audit.jsonl"
    with initialization_path.open("w", encoding="utf-8", newline="\n") as handle:
        for agent_id in sorted(adapters):
            handle.write(canonical_json(roles[agent_id].audit_row(METHOD)) + "\n")
    manifest["initialization_audit_sha256"] = _sha256(initialization_path)
    manifest["initialization_count"] = len(adapters)
    (args.output_root / "preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    written = 0
    records_handle = audits_handle = assimilation_handle = None
    if args.enable_real:
        records_handle = records_path.open("a", encoding="utf-8", newline="\n")
        audits_handle = audits_path.open("a", encoding="utf-8", newline="\n")
        assimilation_handle = assimilation_path.open("a", encoding="utf-8", newline="\n")
    gaps: set[str] = set()
    structural_preflights = 0
    try:
        for envelope in envelopes:
            observation = envelope.observation
            adapter = adapters[observation.agent_id]
            snapshot = adapter.observe(observation)
            bundle = renderer.render(observation, snapshot)
            if envelope.observation_id in existing:
                if observation.agent_id in gaps:
                    raise ValueError(
                        "resume records must form a settled prefix for every agent"
                    )
                row = existing[envelope.observation_id]
                audit = existing_audits[envelope.observation_id]
                if (
                    row.get("method") != METHOD
                    or row.get("variant") != VARIANT
                    or audit.get("observation_sha256") != observation.observation_sha256
                    or audit.get("rendered_prompt_sha256") != bundle.rendered_sha256
                ):
                    raise ValueError("settled CampusPulse resume record drifted")
                decision = {
                    "action": row.get("action"),
                    "emotion": row.get("emotion"),
                    "public_stance": row.get("public_stance"),
                    "public_or_private": row.get("public_or_private"),
                    "text": row.get("generated_text"),
                }
                replay = _advance(
                    adapter, decision=decision, scenario_id=scenario_id,
                    tick=observation.tick, seed=seed,
                    prompt_sha256=bundle.rendered_sha256,
                )
                if replay["after_particles_sha256"] != existing_assimilation[envelope.observation_id].get("after_particles_sha256"):
                    raise ValueError("offline resume replay does not reproduce settled particle state")
                continue

            gaps.add(observation.agent_id)
            provider.preflight(
                method=METHOD,
                bundle=bundle,
                response_schema=observation.response_schema,
                allowed_actions=observation.allowed_actions,
            )
            structural_preflights += 1
            if args.dry_run:
                # The next prompt for this same agent depends on a real decision
                # that deliberately does not exist in dry-run mode.
                continue
            pre_state, particles, weights = _capture(adapter)
            result = provider.generate(
                method=METHOD,
                bundle=bundle,
                response_schema=observation.response_schema,
                allowed_actions=observation.allowed_actions,
            )
            decision = result.parsed_decision
            native = _advance(
                adapter, decision=decision, scenario_id=scenario_id,
                tick=observation.tick, seed=seed,
                prompt_sha256=bundle.rendered_sha256,
            )
            row = {
                "schema_version": "1.0.0",
                "protocol_version": "1.0.0",
                "experiment_id": "dimension2_cross_method_real_v1",
                "run_id": f"dimension2_cross_method_real_v1/{METHOD}/{seed}",
                "method": METHOD,
                "variant": VARIANT,
                "seed": seed,
                "agent_id": observation.agent_id,
                "tick": observation.tick,
                "profile_id": adapter.seed.profile_id if adapter.seed else None,
                "interest_group": adapter.seed.interest_group if adapter.seed else None,
                "visible_context": _visible(envelope),
                "pre_state": pre_state,
                "post_state": native["after_state"],
                "particles": particles,
                "particle_weights": weights,
                "generated_text": decision.get("text"),
                "action": decision.get("action"),
                "emotion": decision.get("emotion"),
                "public_stance": decision.get("public_stance"),
                "public_or_private": decision.get("public_or_private"),
                "model_status": "success",
                "failure": None,
                "source_record_id": envelope.observation_id,
            }
            errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
            if errors:
                raise ValueError(f"unified output schema failed: {errors[0].message}")
            sidecar = {
                "schema_version": "campuspulse-cross-method-assimilation-v1.0.0",
                "source_record_id": envelope.observation_id,
                "case_id": case_id(row),
                **native,
            }
            audit = {
                "schema_version": "campuspulse-cross-method-prompt-audit-v1.0.0",
                "source_record_id": envelope.observation_id,
                "agent_id": observation.agent_id,
                "tick": observation.tick,
                "observation_sha256": observation.observation_sha256,
                "common_section_sha256": {
                    name: bundle.section_sha256[name] for name in COMMON_SECTIONS
                },
                "method_context_sha256": bundle.section_sha256["method_cognitive_context"],
                "rendered_prompt_sha256": bundle.rendered_sha256,
                "predecision_particles_sha256": native["before_particles_sha256"],
                "role_card_sha256": roles[observation.agent_id].role_card_sha256,
                "request_id": result.request_id,
                "cache_key": result.cache_key,
                "cache_hit": result.cache_hit,
                "raw_archive_path": result.raw_archive_path,
                "usage": dict(result.usage),
            }
            _append_sync(assimilation_handle, sidecar)
            _append_sync(audits_handle, audit)
            # records.jsonl is the commit marker.  Sidecar and audit evidence
            # are durable before a record is counted as settled.
            _append_sync(records_handle, row)
            written += 1
    finally:
        for handle in (records_handle, audits_handle, assimilation_handle):
            if handle is not None:
                handle.close()

    if args.dry_run:
        manifest["structural_preflight_count"] = structural_preflights
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0

    manifest.update({
        "settled_record_count": len(existing) + written,
        "pending_record_count": target_count - len(existing) - written,
        "newly_written": written,
        "network_calls_made": provider.budget_snapshot()["transport_attempts"],
        "budget_used": provider.budget_snapshot(),
    })
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
