from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from comparison_harness.adapters import CampusPulseAdapter
from comparison_harness.adapters.campuspulse import DIMENSION_2
from comparison_harness.campuspulse_assimilation import assimilate_isolated_turn
from comparison_harness.contracts import AgentSeed, CommonObservation, canonical_json
from comparison_harness.offline_trace import ActivationSlot, FrozenCommonTraceConnector
from comparison_harness.prompting import (
    COMMON_SECTIONS, PUBLIC_STANCE_SEMANTICS, STANCE_PROPOSITION,
    SharedPromptRenderer,
)
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider
from campuspulse_core.governance_arena_v2 import STATE_FIELDS
from evaluation.dimension2 import case_id


CONDITIONS = ("context_only", "profile_only", "state_only", "full")
D2_ACTIONS = ("ignore", "read", "post", "reply", "seek_help", "report_risk")
D2_EMOTIONS = (
    "neutral", "calm", "curious", "concerned", "skeptical", "frustrated",
    "supportive", "urgent",
)
DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "target_id", "text", "emotion", "public_stance",
        "public_or_private",
    ],
    "properties": {
        "action": {"enum": list(D2_ACTIONS)},
        "target_id": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
        "emotion": {"enum": list(D2_EMOTIONS)},
        "public_stance": {
            "type": ["number", "null"], "minimum": -1.0, "maximum": 1.0,
            "description": PUBLIC_STANCE_SEMANTICS,
        },
        "public_or_private": {"enum": ["public", "private", "none"]},
    },
}


def load_predecision_snapshot(path: Path, panel: Mapping[str, Any], *, panel_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    """Load identity-bound particles from the frozen Tick-0--2 warmup.

    Reinitializing particles here is forbidden.  Both state-bearing ablations
    consume one shared state captured after Tick-2 assimilation and before any
    Tick-3 ablation prompt is rendered.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "campuspulse-d2-predecision-snapshot-v1.1.0":
        raise ValueError("unsupported predecision snapshot schema")
    if payload.get("protocol_version") != "1.2.0":
        raise ValueError("predecision snapshot protocol drift")
    for key in ("scenario_id", "seed", "tick"):
        if payload.get(key) != panel.get(key):
            raise ValueError(f"predecision snapshot {key} does not match panel")
    if payload.get("stance_proposition") != STANCE_PROPOSITION:
        raise ValueError("predecision snapshot stance proposition drift")
    expected_warmup = panel.get("warmup") or {}
    if (
        payload.get("source_kind") != "frozen_warmup_trajectory"
        or payload.get("warmup_condition") != "full"
        or payload.get("warmup_ticks") != [0, 1, 2]
        or payload.get("warmup_record_count") != 36
        or payload.get("capture_point") != "after_tick_2_native_assimilation_before_tick_3_ablation_generation"
        or expected_warmup.get("condition") != "full"
        or expected_warmup.get("ticks") != [0, 1, 2]
        or expected_warmup.get("required_records") != 36
    ):
        raise ValueError("predecision snapshot does not satisfy the frozen warmup contract")
    provenance = payload.get("source_provenance") or {}
    for key in ("warmup_records_sha256", "warmup_assimilation_sha256", "panel_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid warmup provenance hash: {key}")
    if panel_sha256 is not None and provenance["panel_sha256"] != panel_sha256:
        raise ValueError("warmup snapshot was built for a different panel file")
    rows = payload.get("agents")
    if not isinstance(rows, list):
        raise ValueError("predecision snapshot agents must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        agent_id = str(row.get("agent_id") or "")
        particles = row.get("particles")
        weights = row.get("particle_weights")
        if not agent_id or agent_id in by_id:
            raise ValueError("predecision snapshot has missing or duplicate agent_id")
        if not isinstance(particles, list) or len(particles) != 10:
            raise ValueError(f"{agent_id}: exactly 10 predecision particles required")
        if not isinstance(weights, list) or len(weights) != 10:
            raise ValueError(f"{agent_id}: exactly 10 particle weights required")
        if any(set(particle) != set(STATE_FIELDS) for particle in particles):
            raise ValueError(f"{agent_id}: particle state fields drift")
        numeric_weights = np.asarray(weights, dtype=np.float64)
        if np.any(numeric_weights < 0) or not np.isclose(numeric_weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{agent_id}: particle weights must be nonnegative and sum to one")
        values = np.asarray([[float(particle[field]) for field in STATE_FIELDS] for particle in particles])
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{agent_id}: particle values must be finite and bounded")
        by_id[agent_id] = {"particles": particles, "particle_weights": numeric_weights.tolist()}
    expected = {str(row["agent_id"]) for row in panel["agents"]}
    if set(by_id) != expected:
        raise ValueError("predecision snapshot must cover exactly the frozen 12-agent panel")
    scalars = []
    for row in by_id.values():
        scalars.append(sum(
            weight * (float(particle["support"]) - float(particle["opposition"]))
            for particle, weight in zip(row["particles"], row["particle_weights"], strict=True)
        ))
    if max(scalars) - min(scalars) <= 1e-6:
        raise ValueError("predecision stance is degenerate; refusing an uninformative agreement panel")
    return by_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_sync(handle, value: Mapping[str, Any]) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _state(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(STATE_FIELDS, values, strict=True)}


def _visible(observation: CommonObservation) -> list[dict[str, Any]]:
    return [
        {
            "message_id": item.message_id,
            "author_id": item.author_id,
            "text": item.text,
            "visibility": item.visibility,
            "tick": item.tick,
        }
        for item in observation.visible_context
    ]


def _profile_for_prompt(profile: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        "agent_id", "synthetic_display_id", "profile_prompt_sha256",
        "initial_state", "scenario_weights", "design_weight", "pilot_bridge",
    }
    return {key: value for key, value in profile.items() if key not in forbidden}


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                identifier = str(json.loads(line)["source_record_id"])
                if identifier in completed:
                    raise ValueError(f"duplicate source_record_id: {identifier}")
                completed.add(identifier)
    return completed


def _observation(connector: FrozenCommonTraceConnector, index: int, agent_id: str) -> CommonObservation:
    base = connector._observation(ActivationSlot(3, index, agent_id, "paired_d2_panel"))
    return CommonObservation(
        scenario_id=base.scenario_id,
        tick=base.tick,
        agent_id=base.agent_id,
        factual_registry=base.factual_registry,
        current_event=base.current_event,
        visible_context=base.visible_context,
        governance_artifacts=base.governance_artifacts,
        allowed_actions=D2_ACTIONS,
        response_schema=DECISION_SCHEMA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen 12-agent x 4-condition CampusPulse dimension-2 panel."
    )
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--predecision-snapshot", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--assimilation-schema", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-real", action="store_true")
    args = parser.parse_args()
    if args.enable_real == args.dry_run:
        raise ValueError("select exactly one of --dry-run or --enable-real")
    if args.enable_real and args.credentials is None:
        raise ValueError("--credentials is required for real execution")

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if (
        panel.get("schema_version") != "campuspulse-dimension2-panel-v3"
        or panel.get("protocol_version") != "1.2.0"
        or panel.get("conditions") != list(CONDITIONS)
        or len(panel.get("agents") or []) != 12
    ):
        raise ValueError("dimension-2 panel is invalid")
    connector = FrozenCommonTraceConnector(args.scenario_root)
    connector.load_and_verify()
    predecision_by_agent = load_predecision_snapshot(
        args.predecision_snapshot, panel, panel_sha256=_sha256(args.panel)
    )
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    assimilation_validator = Draft202012Validator(
        json.loads(args.assimilation_schema.read_text(encoding="utf-8"))
    )
    renderer = SharedPromptRenderer()
    config = json.loads(args.provider_boundary.read_text(encoding="utf-8"))
    if args.enable_real:
        credential_data = json.loads(args.credentials.read_text(encoding="utf-8-sig"))
        credential_api_key = str(credential_data.get("api_key") or "").strip()
        if not credential_api_key:
            raise ValueError("credentials JSON is missing api_key")
        # The provider retains its independent environment-based safety check;
        # materialize the JSON credential only in this child process so the
        # key is never written to manifests, caches, prompts or raw archives.
        os.environ.setdefault(config["api_key_environment_variable"], credential_api_key)
    transport = (
        OpenAIProviderTransport(
            api_key_env=config["api_key_environment_variable"],
            timeout_seconds=float(config["request_constraints"]["timeout_seconds"]),
            credentials_path=args.credentials,
        )
        if args.enable_real else None
    )
    provider = RealModelProvider(
        config_path=args.provider_boundary,
        auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.output_root / "provider_state",
        transport=transport,
        enable_real=bool(args.enable_real),
        budget_profile="pilot",
    )
    records_path = args.output_root / "records.four_conditions.jsonl"
    audits_path = args.output_root / "prompt_audit.jsonl"
    assimilation_path = args.output_root / "assimilation_sidecar.jsonl"
    if (records_path.exists() or audits_path.exists() or assimilation_path.exists()) and not args.resume:
        raise FileExistsError("existing output requires --resume")
    completed = _read_completed(records_path) if args.resume else set()
    preflights: list[dict[str, Any]] = []
    bundles_by_agent: dict[str, list[Any]] = {}
    prepared: list[tuple[Any, ...]] = []

    for selected in panel["agents"]:
        index = int(selected["agent_index"])
        agent_id = str(selected["agent_id"])
        profile = connector.profiles[index]
        if str(profile["agent_id"]) != agent_id:
            raise ValueError(f"panel agent/index mismatch: {agent_id}")
        observation = _observation(connector, index, agent_id)
        frozen_state = predecision_by_agent[agent_id]
        weights = list(frozen_state["particle_weights"])
        particles = [dict(row) for row in frozen_state["particles"]]
        particle_matrix = np.asarray([[particle[field] for field in STATE_FIELDS] for particle in particles])
        mean = np.average(particle_matrix, axis=0, weights=np.asarray(weights))
        pre_state = _state(mean)
        stake = connector.stake_by_agent[agent_id]
        interest_group = (
            "directly_affected"
            if bool(stake["fields"]["directly_affected"]["value"])
            else "bystander"
        )
        for condition in CONDITIONS:
            adapter = CampusPulseAdapter(DIMENSION_2[condition])
            adapter.initialize(AgentSeed(
                agent_id=agent_id,
                profile_id=str(profile.get("profile_prompt_sha256") or "profile-" + agent_id),
                interest_group=interest_group,
                stable_profile=_profile_for_prompt(profile),
                initial_state=pre_state,
                initial_particles=particles,
                particle_weights=weights,
            ))
            snapshot = adapter.observe(observation)
            bundle = renderer.render(observation, snapshot)
            bundles_by_agent.setdefault(agent_id, []).append(bundle)
            source_id = f"d2:{panel['seed']}:t3:{agent_id}:{condition}"
            prepared.append((condition, index, observation, adapter, bundle, pre_state, particles, weights, source_id))
            preflights.append(provider.preflight(
                method="campuspulse", bundle=bundle,
                response_schema=observation.response_schema,
                allowed_actions=observation.allowed_actions,
            ))

    for agent_id, bundles in bundles_by_agent.items():
        for section in COMMON_SECTIONS:
            if len({bundle.section_sha256[section] for bundle in bundles}) != 1:
                raise ValueError(f"common prompt section drift for {agent_id}: {section}")
        contexts = [bundle.section_text["method_cognitive_context"] for bundle in bundles]
        if len(set(contexts)) != 4:
            raise ValueError(f"ablation contexts are not distinct for {agent_id}")

    manifest = {
        "schema_version": "campuspulse-dimension2-run-manifest-v1",
        "evaluation_definitions_version": "1.6.0",
        "protocol_version": "1.2.0",
        "fairness_contract_version": "1.1.0",
        "text_evaluation_mode": "model_judge_with_stratified_human_review",
        "human_review_required": True,
        "human_review_fraction": 0.20,
        "scientific_result": not args.dry_run,
        "scenario_id": panel["scenario_id"],
        "seed": panel["seed"],
        "tick": panel["tick"],
        "agent_count": 12,
        "condition_count": 4,
        "required_record_count": 48,
        "panel_sha256": _sha256(args.panel),
        "predecision_snapshot_sha256": _sha256(args.predecision_snapshot),
        "predecision_state_source": "identity-bound state captured after the frozen Tick-0--2 full-method warmup and before any Tick-3 ablation prompt",
        "warmup_ticks": [0, 1, 2],
        "warmup_record_count": 36,
        "stance_proposition": STANCE_PROPOSITION,
        "scenario_source_hashes": dict(sorted(connector.file_hashes.items())),
        "preflight_call_count": len(preflights),
        "network_calls_made": 0,
        "common_prompt_sections_equal_within_agent": True,
        "post_state_status": "native_assimilation_registered",
        "assimilation_version": "campuspulse-action-assimilation-v1.0.0",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "preflight_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0

    written = 0
    with records_path.open("a", encoding="utf-8", newline="\n") as records_handle, audits_path.open("a", encoding="utf-8", newline="\n") as audit_handle, assimilation_path.open("a", encoding="utf-8", newline="\n") as assimilation_handle:
        for condition, index, observation, adapter, bundle, pre_state, particles, weights, source_id in prepared:
            if source_id in completed:
                continue
            result = provider.generate(
                method="campuspulse", bundle=bundle,
                response_schema=observation.response_schema,
                allowed_actions=observation.allowed_actions,
            )
            decision = result.parsed_decision
            applicable = condition in {"state_only", "full"}
            native = None
            assimilation_sidecar_id = None
            if applicable:
                native = assimilate_isolated_turn(
                    particles=particles,
                    particle_weights=weights,
                    decision=decision,
                    scenario_id=str(panel["scenario_id"]),
                    variant=condition,
                    tick=int(observation.tick),
                    seed=int(panel["seed"]),
                    prompt_sha256=bundle.rendered_sha256,
                )
                assimilation_sidecar_id = "assim-" + native["innovation_id"][:24]
            row = {
                "schema_version": "1.1.0",
                "protocol_version": "1.2.0",
                "experiment_id": "campuspulse_dimension2_ablation_real_v1",
                "run_id": f"campuspulse_dimension2_ablation_real_v1/{condition}/{panel['seed']}",
                "method": "campuspulse",
                "variant": condition,
                "seed": int(panel["seed"]),
                "agent_id": observation.agent_id,
                "tick": observation.tick,
                "profile_id": str(connector.profiles[index].get("profile_prompt_sha256")),
                "interest_group": adapter.seed.interest_group if adapter.seed else None,
                "visible_context": _visible(observation),
                "pre_state": pre_state if applicable else None,
                "post_state": native["after_state"] if native is not None else None,
                "particles": particles if applicable else None,
                "particle_weights": weights if applicable else None,
                "generated_text": decision["text"],
                "action": decision["action"],
                "emotion": decision["emotion"],
                "public_stance": decision["public_stance"],
                "public_or_private": decision["public_or_private"],
                "model_status": "success",
                "failure": None,
                "source_record_id": source_id,
                "assimilation_sidecar_id": assimilation_sidecar_id,
            }
            errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
            if errors:
                raise ValueError(f"unified schema failed: {errors[0].message}")
            sidecar = None
            if native is not None:
                sidecar = {
                    "schema_version": "campuspulse-assimilation-sidecar-v1.0.0",
                    "assimilation_sidecar_id": assimilation_sidecar_id,
                    "case_id": case_id(row),
                    "source_record_id": source_id,
                    **native,
                }
                sidecar_errors = sorted(
                    assimilation_validator.iter_errors(sidecar),
                    key=lambda error: list(error.path),
                )
                if sidecar_errors:
                    raise ValueError(
                        f"assimilation sidecar schema failed: {sidecar_errors[0].message}"
                    )
            _append_sync(records_handle, row)
            if sidecar is not None:
                _append_sync(assimilation_handle, sidecar)
            _append_sync(audit_handle, {
                "source_record_id": source_id,
                "agent_id": observation.agent_id,
                "condition": condition,
                "observation_sha256": observation.observation_sha256,
                "common_section_sha256": {
                    name: bundle.section_sha256[name] for name in COMMON_SECTIONS
                },
                "method_context_sha256": bundle.section_sha256["method_cognitive_context"],
                "rendered_prompt_sha256": bundle.rendered_sha256,
                "predecision_particle_stance": float(pre_state["support"] - pre_state["opposition"]),
                "request_id": result.request_id,
                "cache_key": result.cache_key,
                "cache_hit": result.cache_hit,
                "raw_archive_path": result.raw_archive_path,
                "usage": dict(result.usage),
            })
            completed.add(source_id)
            written += 1
    manifest.update({
        "network_calls_made": provider.budget_snapshot()["transport_attempts"],
        "completed_record_count": len(completed),
        "newly_written": written,
        "budget_used": provider.budget_snapshot(),
    })
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
