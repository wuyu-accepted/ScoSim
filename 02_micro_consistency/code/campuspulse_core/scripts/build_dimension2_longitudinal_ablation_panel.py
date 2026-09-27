from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from campuspulse_core.governance_arena_v2 import STATE_FIELDS, initialize_particle_ensemble
from comparison_harness.adapters import CampusPulseAdapter
from comparison_harness.adapters.campuspulse import DIMENSION_2
from comparison_harness.contracts import canonical_json
from comparison_harness.fake_executor import load_observations
from comparison_harness.fair_initialization import load_fair_role_initializations
from comparison_harness.prompting import COMMON_SECTIONS, SharedPromptRenderer
from evaluation.dimension2 import PRESTATE_NEUTRAL_QUANTILE, PRESTATE_NEUTRAL_RADIUS_BY_METHOD


CONDITIONS = ("context_only", "profile_only", "state_only", "full")
PRESTATE_NEUTRAL_RADIUS = PRESTATE_NEUTRAL_RADIUS_BY_METHOD["campuspulse"]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_jsonl_with_raw(path: Path, key: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows: dict[str, dict[str, Any]] = {}
    raw: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row[key])
        if identifier in rows:
            raise ValueError(f"duplicate {key} at line {number}: {identifier}")
        rows[identifier] = row
        raw[identifier] = line
    return rows, raw


def state_mean(particles: list[dict[str, Any]], weights: list[float]) -> dict[str, float]:
    matrix = np.asarray([[float(p[field]) for field in STATE_FIELDS] for p in particles])
    vector = np.average(matrix, axis=0, weights=np.asarray(weights, dtype=float))
    return {field: float(value) for field, value in zip(STATE_FIELDS, vector, strict=True)}


def normalized_particle_payload(particles: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    """Reproduce the native assimilation hash's float32/normalized boundary."""
    values = np.asarray(
        [[float(p[field]) for field in STATE_FIELDS] for p in particles], dtype=np.float32
    )
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum()
    normalized_particles = [
        {field: float(value) for field, value in zip(STATE_FIELDS, row, strict=True)}
        for row in values
    ]
    return {"particles": normalized_particles, "weights": normalized.tolist()}


def direction(value: float) -> str:
    return (
        "support" if value > PRESTATE_NEUTRAL_RADIUS
        else "oppose" if value < -PRESTATE_NEUTRAL_RADIUS
        else "neutral"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and audit the frozen 512-slot longitudinal D2 ablation panel.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--full-records", type=Path, required=True)
    parser.add_argument("--full-prompt-audit", type=Path, required=True)
    parser.add_argument("--full-assimilation-sidecar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run-slots", type=int, default=1)
    args = parser.parse_args()

    envelopes = load_observations(args.observations)
    if len(envelopes) != 512:
        raise ValueError("longitudinal ablation requires exactly 512 frozen observations")
    observations = {item.observation_id: item for item in envelopes}
    if len(observations) != 512:
        raise ValueError("observation IDs are not unique")
    full, raw_full = read_jsonl_with_raw(args.full_records, "source_record_id")
    audits, _ = read_jsonl_with_raw(args.full_prompt_audit, "source_record_id")
    sidecars, _ = read_jsonl_with_raw(args.full_assimilation_sidecar, "source_record_id")
    if set(full) != set(observations) or set(audits) != set(observations) or set(sidecars) != set(observations):
        raise ValueError("observations, full records, prompt audits and sidecars are not exactly paired")
    roles = load_fair_role_initializations(args.population)
    population = json.loads(args.population.read_text(encoding="utf-8"))
    population_agents = population.get("agents") or []
    population_index = {str(row["llm_agent_id"]): index for index, row in enumerate(population_agents)}
    scenario_ids = {item.observation.scenario_id for item in envelopes}
    seeds = {int(full[item.observation_id]["seed"]) for item in envelopes}
    if len(scenario_ids) != 1 or len(seeds) != 1:
        raise ValueError("panel must use one scenario and one seed")
    initial_ensemble = initialize_particle_ensemble(
        population, scenario_id=next(iter(scenario_ids)), seed=next(iter(seeds))
    )
    initial_means = initial_ensemble.parent_means()
    renderer = SharedPromptRenderer()
    slots: list[dict[str, Any]] = []
    direction_counts: Counter[str] = Counter()
    previous_sidecar: dict[str, dict[str, Any]] = {}

    for envelope in envelopes:
        identifier = envelope.observation_id
        row = full[identifier]
        observation = envelope.observation
        if row.get("method") != "campuspulse" or row.get("variant") != "cross_method_full":
            raise ValueError(f"invalid reused full record: {identifier}")
        if str(row.get("agent_id")) != observation.agent_id or int(row.get("tick")) != observation.tick:
            raise ValueError(f"identity/tick mismatch: {identifier}")
        particles = row.get("particles")
        weights = row.get("particle_weights")
        if not isinstance(particles, list) or len(particles) != 10 or not isinstance(weights, list) or len(weights) != 10:
            raise ValueError(f"invalid predecision particles: {identifier}")
        if not np.isclose(sum(float(value) for value in weights), 1.0, atol=1e-6):
            raise ValueError(f"invalid particle weights: {identifier}")
        mean = state_mean(particles, weights)
        scalar = mean["support"] - mean["opposition"]
        direction_counts[direction(scalar)] += 1
        reconstructed_hash = sha256_value(normalized_particle_payload(particles, weights))
        audit_hash = str(audits[identifier].get("predecision_particles_sha256"))
        if audit_hash != sidecars[identifier].get("before_particles_sha256"):
            raise ValueError(f"prompt/assimilation predecision hash drift: {identifier}")
        particle_hash_mode = "roundtrip_exact"
        if reconstructed_hash != audit_hash:
            parent = previous_sidecar.get(observation.agent_id)
            if (
                parent is None
                or parent.get("after_particles") != particles
                or parent.get("after_particle_weights") != weights
            ):
                raise ValueError(f"captured particle provenance drift: {identifier}")
            # JSON round-tripping can move a normalized float64 weight by one
            # ULP.  The signed audit/sidecar hash remains authoritative when
            # exact parent arrays and the native hash chain both agree.
            particle_hash_mode = "native_current_before_hash_with_exact_parent_arrays"
        particle_hash = audit_hash
        slots.append({
            "slot_id": identifier,
            "agent_id": observation.agent_id,
            "tick": observation.tick,
            "seed": int(row["seed"]),
            "observation_sha256": observation.observation_sha256,
            "full_record_sha256": sha256_bytes(raw_full[identifier].encode("utf-8")),
            "predecision_particles_sha256": particle_hash,
            "particle_hash_verification": particle_hash_mode,
            "pre_state_scalar": scalar,
            "pre_state_direction": direction(scalar),
            "conditions": list(CONDITIONS),
            "reused_full": True,
        })
        previous_sidecar[observation.agent_id] = sidecars[identifier]

    dry_run_rows = []
    for envelope in envelopes[: args.dry_run_slots]:
        identifier = envelope.observation_id
        row = full[identifier]
        agent_id = envelope.observation.agent_id
        earlier = [
            sidecars[item.observation_id]
            for item in envelopes
            if item.observation.agent_id == agent_id
            and item.observation.tick < envelope.observation.tick
        ]
        if earlier:
            prompt_state = dict(earlier[-1]["after_state"])
        else:
            vector = initial_means[population_index[agent_id]]
            prompt_state = {
                field: float(value)
                for field, value in zip(STATE_FIELDS, vector, strict=True)
            }
        bundles = {}
        for condition in CONDITIONS:
            adapter = CampusPulseAdapter(DIMENSION_2[condition])
            adapter.initialize(roles[envelope.observation.agent_id].agent_seed(
                initial_state=prompt_state,
                initial_particles=row["particles"],
                particle_weights=row["particle_weights"],
            ))
            bundles[condition] = renderer.render(envelope.observation, adapter.observe(envelope.observation))
        for section in COMMON_SECTIONS:
            if len({bundle.section_sha256[section] for bundle in bundles.values()}) != 1:
                raise ValueError(f"common prompt drift at {identifier}:{section}")
        if len({bundle.section_sha256["method_cognitive_context"] for bundle in bundles.values()}) != 4:
            raise ValueError(f"ablation method contexts are not distinct: {identifier}")
        if bundles["full"].rendered_sha256 != audits[identifier].get("rendered_prompt_sha256"):
            raise ValueError(f"reused full prompt is not reproducible: {identifier}")
        dry_run_rows.append({
            "slot_id": identifier,
            "network_calls_made": 0,
            "common_sections_equal": True,
            "method_contexts_distinct": True,
            "reused_full_prompt_sha256_match": True,
            "planned_new_calls": 3,
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    slots_path = args.output_root / "slots.jsonl"
    slots_path.write_text("".join(canonical_json(row) + "\n" for row in slots), encoding="utf-8", newline="\n")
    reused_path = args.output_root / "full_records.reused.jsonl"
    reused_path.write_bytes(args.full_records.read_bytes())
    dry_path = args.output_root / "dry_run_audit.jsonl"
    dry_path.write_text("".join(canonical_json(row) + "\n" for row in dry_run_rows), encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": "dimension2-longitudinal-ablation-panel-v1.1.0",
        "definitions_version": "1.6.0",
        "status": "frozen_preflight_passed",
        "slot_count": 512,
        "condition_count": 4,
        "final_record_count": 2048,
        "reused_full_record_count": 512,
        "planned_new_model_calls": 1536,
        "agent_count": len({row["agent_id"] for row in slots}),
        "ticks": sorted({row["tick"] for row in slots}),
        "pre_state_direction_counts": dict(sorted(direction_counts.items())),
        "pre_state_direction_calibration": {
            "rule": "method_native_equal_neutral_mass_quantile",
            "neutral_quantile": PRESTATE_NEUTRAL_QUANTILE,
            "campuspulse_neutral_radius": PRESTATE_NEUTRAL_RADIUS,
        },
        "dry_run_slot_count": len(dry_run_rows),
        "network_calls_made": 0,
        "input_sha256": {
            "observations": sha256_bytes(args.observations.read_bytes()),
            "population": sha256_bytes(args.population.read_bytes()),
            "full_records": sha256_bytes(args.full_records.read_bytes()),
            "full_prompt_audit": sha256_bytes(args.full_prompt_audit.read_bytes()),
            "full_assimilation_sidecar": sha256_bytes(args.full_assimilation_sidecar.read_bytes()),
        },
        "output_sha256": {
            "slots": sha256_bytes(slots_path.read_bytes()),
            "full_records_reused": sha256_bytes(reused_path.read_bytes()),
            "dry_run_audit": sha256_bytes(dry_path.read_bytes()),
        },
        "full_records_byte_identical": reused_path.read_bytes() == args.full_records.read_bytes(),
    }
    (args.output_root / "panel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
