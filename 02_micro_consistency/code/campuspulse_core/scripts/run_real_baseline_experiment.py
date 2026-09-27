from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator

from comparison_harness.adapters import (
    BeliefEngineAdapter,
    ChuangOpinionDynamicsAdapter,
    GenerativeAgentsAdapter,
)
from comparison_harness.auxiliary import FrozenPromptRegistry
from comparison_harness.contracts import canonical_json
from comparison_harness.fake_executor import _record, load_observations
from comparison_harness.fair_initialization import (
    FairRoleInitialization, load_fair_role_initializations,
)
from comparison_harness.prompting import SharedPromptRenderer
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider
from comparison_harness.real_runtime import RealAuxiliaryRuntime
from evaluation.dimension2 import case_id


METHODS = ("belief_engine", "generative_agents", "chuang_opinion_dynamics")


def _adapter(
    method: str, runtime: RealAuxiliaryRuntime, chuang_instruction: str,
    role: FairRoleInitialization,
):
    if method == "belief_engine":
        return BeliefEngineAdapter(
            seed_arguments=role.belief_arguments(),
            extractor=runtime.extract,
            embedder=runtime.embedding,
        )
    if method == "generative_agents":
        return GenerativeAgentsAdapter(
            embedder=runtime.embedding,
            importance_scorer=runtime.importance,
            reflector=runtime.reflect,
        )
    if method == "chuang_opinion_dynamics":
        return ChuangOpinionDynamicsAdapter(initial_opinion=None, update_instruction=chuang_instruction)
    raise ValueError(f"unsupported baseline method: {method}")


def _existing_ids(path: Path, id_field: str) -> set[str]:
    if not path.exists():
        return set()
    values = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row[id_field])
            if identifier in values:
                raise ValueError(f"duplicate source_record_id at {path}:{number}")
            values.add(identifier)
    return values


def _append_and_sync(handle, value) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen baseline observations through the real provider with per-record checkpoints."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget-profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-real", action="store_true", required=True)
    args = parser.parse_args()
    if args.max_observations is not None and args.max_observations < 1:
        raise ValueError("max-observations must be positive")

    boundary = json.loads(args.auxiliary_boundary.read_text(encoding="utf-8"))
    registry = FrozenPromptRegistry(args.prompt_root, boundary["prompt_sha256"])
    config = json.loads(args.provider_boundary.read_text(encoding="utf-8"))
    transport = OpenAIProviderTransport(
        api_key_env=config["api_key_environment_variable"],
        timeout_seconds=float(config["request_constraints"]["timeout_seconds"]),
    )
    provider = RealModelProvider(
        config_path=args.provider_boundary,
        auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.output_root / "provider_state",
        transport=transport,
        enable_real=True,
        budget_profile=args.budget_profile,
    )
    runtime = RealAuxiliaryRuntime(provider, registry)
    renderer = SharedPromptRenderer()
    unified_schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(unified_schema)
    observations = load_observations(args.observations)
    roles = load_fair_role_initializations(args.population)
    observation_agents = {row.observation.agent_id for row in observations}
    if not observation_agents.issubset(roles):
        raise ValueError("common observations contain agents outside the fair role release")
    if args.max_observations is not None:
        observations = observations[: args.max_observations]
    chuang_instruction = registry["chuang_confirmation_update"].text.strip()
    report = {
        "schema_version": "real-baseline-experiment-v1",
        "scientific_result": args.max_observations is None,
        "seed": 20260722,
        "budget_profile": args.budget_profile,
        "input_observation_count": len(observations),
        "methods": {},
    }

    for method in METHODS:
        method_dir = args.output_root / method / "real" / "20260722"
        method_dir.mkdir(parents=True, exist_ok=True)
        records_path = method_dir / "records.jsonl"
        audit_path = method_dir / "prompt_audit.jsonl"
        state_path = method_dir / "predecision_state.jsonl"
        initialization_path = method_dir / "initialization_audit.jsonl"
        if (records_path.exists() or audit_path.exists() or state_path.exists() or initialization_path.exists()) and not args.resume:
            raise FileExistsError(f"existing output requires --resume: {method_dir}")
        completed_records = _existing_ids(records_path, "source_record_id") if args.resume else set()
        completed_audits = _existing_ids(audit_path, "observation_id") if args.resume else set()
        completed_states = (
            _existing_ids(state_path, "source_record_id")
            if args.resume and state_path.exists() else set()
        )
        completed_initializations = (
            _existing_ids(initialization_path, "agent_id")
            if args.resume and initialization_path.exists() else set()
        )
        adapters = {}
        written = 0
        with records_path.open("a", encoding="utf-8", newline="\n") as records_handle, \
             audit_path.open("a", encoding="utf-8", newline="\n") as audit_handle, \
             state_path.open("a", encoding="utf-8", newline="\n") as state_handle, \
             initialization_path.open("a", encoding="utf-8", newline="\n") as initialization_handle:
            for envelope in observations:
                observation = envelope.observation
                adapter = adapters.get(observation.agent_id)
                if adapter is None:
                    role = roles[observation.agent_id]
                    adapter = _adapter(method, runtime, chuang_instruction, role)
                    adapter.initialize(role.agent_seed())
                    adapters[observation.agent_id] = adapter
                    if observation.agent_id not in completed_initializations:
                        _append_and_sync(
                            initialization_handle, role.audit_row(method)
                        )
                        completed_initializations.add(observation.agent_id)
                snapshot = adapter.observe(observation)
                predecision_log_odds = (
                    float(adapter.log_odds) if method == "belief_engine" else None
                )
                predecision_scalar = (
                    float(adapter.stance) if method == "belief_engine" else None
                )
                bundle = renderer.render(observation, snapshot)
                result = provider.generate(
                    method=method, bundle=bundle,
                    response_schema=observation.response_schema,
                    allowed_actions=observation.allowed_actions,
                )
                adapter.commit_decision(observation, result.parsed_decision)
                recorded_decision = dict(result.parsed_decision)
                if method == "chuang_opinion_dynamics":
                    recorded_decision["public_stance"] = adapter.export_probe()
                row = _record(
                    envelope=envelope, method=method,
                    decision=recorded_decision,
                    # Unified output stores the stance expressed by this turn;
                    # method-native internal state is captured separately.
                    public_stance=recorded_decision.get("public_stance"),
                )
                row.update({
                    "experiment_id": "frozen_baseline_comparison_real_v1",
                    "run_id": f"frozen_baseline_comparison_real_v1/{method}/20260722",
                    "variant": "cross_method_full",
                    "model_status": "success",
                })
                errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
                if errors:
                    raise ValueError(f"unified schema failed for {method}: {errors[0].message}")
                if envelope.observation_id not in completed_audits:
                    _append_and_sync(audit_handle, {
                        "method": method,
                        "observation_id": envelope.observation_id,
                        "observation_sha256": observation.observation_sha256,
                        "rendered_prompt_sha256": bundle.rendered_sha256,
                        "request_id": result.request_id,
                        "cache_key": result.cache_key,
                        "cache_hit": result.cache_hit,
                        "raw_archive_path": result.raw_archive_path,
                        "usage": dict(result.usage),
                        "raw_public_stance": result.parsed_decision.get("public_stance"),
                        "recorded_public_stance": recorded_decision.get("public_stance"),
                        "role_card_sha256": roles[observation.agent_id].role_card_sha256,
                    })
                    completed_audits.add(envelope.observation_id)
                if envelope.observation_id not in completed_records:
                    _append_and_sync(records_handle, row)
                    completed_records.add(envelope.observation_id)
                    written += 1
                if (
                    method == "belief_engine"
                    and envelope.observation_id not in completed_states
                ):
                    _append_and_sync(state_handle, {
                        "schema_version": "belief-engine-predecision-state-v1.0.0",
                        "case_id": case_id(row),
                        "source_record_id": envelope.observation_id,
                        "agent_id": observation.agent_id,
                        "tick": observation.tick,
                        "log_odds": predecision_log_odds,
                        "standardized_pre_state": predecision_scalar,
                        "capture_point": "after_observation_before_main_decision",
                    })
                    completed_states.add(envelope.observation_id)
        report["methods"][method] = {
            "record_count": len(completed_records), "audit_count": len(completed_audits),
            "newly_written": written,
            "agent_count": len(adapters),
            "predecision_state_count": (
                len(completed_states) if method == "belief_engine" else 0
            ),
            "initialization_count": len(completed_initializations),
        }

    report["budget_used"] = provider.budget_snapshot()
    report["auxiliary_call_count"] = len(runtime.calls)
    report_path = args.output_root / "run_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
