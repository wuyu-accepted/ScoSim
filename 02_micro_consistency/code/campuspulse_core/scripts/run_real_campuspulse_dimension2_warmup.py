from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from campuspulse_core.governance_arena_v2 import STATE_FIELDS, initialize_particle_ensemble
from comparison_harness.adapters import CampusPulseAdapter
from comparison_harness.adapters.campuspulse import DIMENSION_2
from comparison_harness.contracts import CommonObservation, canonical_json
from comparison_harness.fair_initialization import load_fair_role_initializations
from comparison_harness.offline_trace import ActivationSlot, FrozenCommonTraceConnector
from comparison_harness.prompting import COMMON_SECTIONS, STANCE_PROPOSITION, SharedPromptRenderer
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider
from evaluation.dimension2 import case_id
from scripts.run_real_campuspulse_cross_method import _advance, _capture
from scripts.run_real_campuspulse_dimension2_ablation import D2_ACTIONS, DECISION_SCHEMA


TICKS = (0, 1, 2)
VARIANT = "warmup_full"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_sync(handle, value: Mapping[str, Any]) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("source_record_id") or "")
            if not source_id or source_id in result:
                raise ValueError(f"missing or duplicate source_record_id at {path}:{line_number}")
            result[source_id] = row
    return result


def _state(values: np.ndarray) -> dict[str, float]:
    return {field: float(value) for field, value in zip(STATE_FIELDS, values, strict=True)}


def _observation(connector: FrozenCommonTraceConnector, tick: int, index: int, agent_id: str) -> CommonObservation:
    base = connector._observation(ActivationSlot(tick, index, agent_id, "d2_frozen_warmup"))
    return CommonObservation(
        scenario_id=base.scenario_id, tick=base.tick, agent_id=base.agent_id,
        factual_registry=base.factual_registry, current_event=base.current_event,
        visible_context=base.visible_context, governance_artifacts=base.governance_artifacts,
        allowed_actions=D2_ACTIONS, response_schema=DECISION_SCHEMA,
    )


def _visible(observation: CommonObservation) -> list[dict[str, Any]]:
    return [{
        "message_id": item.message_id, "author_id": item.author_id,
        "text": item.text, "visibility": item.visibility, "tick": item.tick,
    } for item in observation.visible_context]


def _initialize(panel: Mapping[str, Any], population: Mapping[str, Any], roles: Mapping[str, Any]) -> dict[str, CampusPulseAdapter]:
    rows = population.get("agents")
    if not isinstance(rows, list) or len(rows) != 1000:
        raise ValueError("warmup population must contain 1000 agents")
    ensemble = initialize_particle_ensemble(population, scenario_id=str(panel["scenario_id"]), seed=int(panel["seed"]))
    means = ensemble.parent_means()
    normalized = ensemble.normalized_within_parent()
    adapters: dict[str, CampusPulseAdapter] = {}
    for selected in panel["agents"]:
        index, agent_id = int(selected["agent_index"]), str(selected["agent_id"])
        if str(rows[index].get("llm_agent_id")) != agent_id or agent_id not in roles:
            raise ValueError(f"warmup panel identity/index mismatch: {agent_id}")
        adapter = CampusPulseAdapter(DIMENSION_2["full"])
        adapter.initialize(roles[agent_id].agent_seed(
            initial_state=_state(means[index]),
            initial_particles=[_state(particle) for particle in ensemble.values[index]],
            particle_weights=normalized[index].tolist(),
        ))
        adapters[agent_id] = adapter
    return adapters


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen 12-agent Tick-0--2 CampusPulse warmup.")
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--enable-real", action="store_true")
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if panel.get("schema_version") != "campuspulse-dimension2-panel-v3" or panel.get("warmup") != {
        "condition": "full", "ticks": [0, 1, 2], "required_records": 36,
        "capture_point": "after_tick_2_native_assimilation_before_tick_3_ablation_generation",
    }:
        raise ValueError("warmup runner requires the frozen v3 panel")
    population = json.loads(args.population.read_text(encoding="utf-8"))
    roles = load_fair_role_initializations(args.population)
    adapters = _initialize(panel, population, roles)
    connector = FrozenCommonTraceConnector(args.scenario_root)
    connector.load_and_verify()
    renderer = SharedPromptRenderer()
    config = json.loads(args.provider_boundary.read_text(encoding="utf-8"))
    transport = OpenAIProviderTransport(
        api_key_env=config["api_key_environment_variable"],
        timeout_seconds=float(config["request_constraints"]["timeout_seconds"]),
    ) if args.enable_real else None
    provider = RealModelProvider(
        config_path=args.provider_boundary,
        auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.output_root / "provider_state",
        transport=transport, enable_real=bool(args.enable_real), budget_profile="pilot",
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    records_path = args.output_root / "records.jsonl"
    sidecars_path = args.output_root / "assimilation_sidecar.jsonl"
    audits_path = args.output_root / "prompt_audit.jsonl"
    existing = _read_jsonl(records_path) if args.resume else {}
    sidecars = _read_jsonl(sidecars_path) if args.resume else {}
    audits = _read_jsonl(audits_path) if args.resume else {}
    if not args.resume and any(path.exists() for path in (records_path, sidecars_path, audits_path)):
        raise FileExistsError("existing warmup output requires --resume")
    if set(existing) != set(sidecars) or set(existing) != set(audits):
        raise ValueError("warmup resume artifacts do not cover the same settled prefix")

    manifest = {
        "schema_version": "campuspulse-d2-warmup-run-v1.0.0",
        "protocol_version": "1.2.0", "scenario_id": panel["scenario_id"],
        "seed": panel["seed"], "ticks": list(TICKS), "agent_count": 12,
        "required_record_count": 36, "settled_record_count": len(existing),
        "execution_mode": "real" if args.enable_real else "dry_run_no_transport",
        "panel_sha256": _sha256(args.panel), "population_sha256": _sha256(args.population),
        "capture_point": "after_tick_2_native_assimilation_before_tick_3_ablation_generation",
        "common_prompt_sections": list(COMMON_SECTIONS), "network_calls_made": 0,
    }
    (args.output_root / "preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if args.dry_run:
        for selected in panel["agents"]:
            agent_id, index = str(selected["agent_id"]), int(selected["agent_index"])
            observation = _observation(connector, 0, index, agent_id)
            bundle = renderer.render(observation, adapters[agent_id].observe(observation))
            provider.preflight(method="campuspulse", bundle=bundle, response_schema=DECISION_SCHEMA, allowed_actions=D2_ACTIONS)
        manifest["preflight_call_count"] = 12
        (args.output_root / "preflight_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0

    with records_path.open("a", encoding="utf-8", newline="\n") as record_handle, sidecars_path.open("a", encoding="utf-8", newline="\n") as sidecar_handle, audits_path.open("a", encoding="utf-8", newline="\n") as audit_handle:
        for tick in TICKS:
            for selected in panel["agents"]:
                agent_id, index = str(selected["agent_id"]), int(selected["agent_index"])
                source_id = f"d2-warmup:{panel['seed']}:t{tick}:{agent_id}"
                observation = _observation(connector, tick, index, agent_id)
                adapter = adapters[agent_id]
                bundle = renderer.render(observation, adapter.observe(observation))
                if source_id in existing:
                    row, audit = existing[source_id], audits[source_id]
                    if audit.get("rendered_prompt_sha256") != bundle.rendered_sha256:
                        raise ValueError("warmup resume prompt drift")
                    decision = {"action": row["action"], "emotion": row["emotion"], "public_stance": row["public_stance"], "public_or_private": row["public_or_private"], "text": row["generated_text"]}
                    replay = _advance(adapter, decision=decision, scenario_id=str(panel["scenario_id"]), tick=tick, seed=int(panel["seed"]), prompt_sha256=bundle.rendered_sha256)
                    if replay["after_particles_sha256"] != sidecars[source_id].get("after_particles_sha256"):
                        raise ValueError("warmup resume particle replay drift")
                    continue
                pre_state, particles, weights = _capture(adapter)
                result = provider.generate(method="campuspulse", bundle=bundle, response_schema=DECISION_SCHEMA, allowed_actions=D2_ACTIONS)
                decision = result.parsed_decision
                native = _advance(adapter, decision=decision, scenario_id=str(panel["scenario_id"]), tick=tick, seed=int(panel["seed"]), prompt_sha256=bundle.rendered_sha256)
                assimilation_id = "assim-" + native["innovation_id"][:24]
                row = {
                    "schema_version": "campuspulse-d2-warmup-record-v1.0.0", "protocol_version": "1.2.0",
                    "experiment_id": "campuspulse_dimension2_warmup_v1", "run_id": f"campuspulse_dimension2_warmup_v1/{panel['seed']}",
                    "method": "campuspulse", "variant": VARIANT, "seed": int(panel["seed"]),
                    "agent_id": agent_id, "tick": tick, "profile_id": adapter.seed.profile_id if adapter.seed else None,
                    "interest_group": adapter.seed.interest_group if adapter.seed else None, "visible_context": _visible(observation),
                    "pre_state": pre_state, "post_state": native["after_state"], "particles": particles, "particle_weights": weights,
                    "generated_text": decision.get("text"), "action": decision.get("action"), "emotion": decision.get("emotion"),
                    "public_stance": decision.get("public_stance"), "public_or_private": decision.get("public_or_private"),
                    "model_status": "success", "failure": None, "source_record_id": source_id, "assimilation_sidecar_id": assimilation_id,
                }
                sidecar = {"schema_version": "campuspulse-assimilation-sidecar-v1.0.0", "assimilation_sidecar_id": assimilation_id, "case_id": case_id(row), "source_record_id": source_id, **native}
                audit = {"schema_version": "campuspulse-d2-warmup-prompt-audit-v1.0.0", "source_record_id": source_id, "agent_id": agent_id, "tick": tick, "rendered_prompt_sha256": bundle.rendered_sha256, "request_id": result.request_id, "cache_key": result.cache_key, "cache_hit": result.cache_hit, "raw_archive_path": result.raw_archive_path, "usage": dict(result.usage)}
                _append_sync(sidecar_handle, sidecar)
                _append_sync(audit_handle, audit)
                _append_sync(record_handle, row)

    manifest["network_calls_made"] = 36 - len(existing)
    manifest["completed_record_count"] = 36
    manifest["warmup_records_sha256"] = _sha256(records_path)
    manifest["warmup_assimilation_sha256"] = _sha256(sidecars_path)
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
