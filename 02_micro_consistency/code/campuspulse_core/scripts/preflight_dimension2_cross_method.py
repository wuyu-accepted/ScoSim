from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from comparison_harness.fake_executor import load_observations


METHODS = (
    "campuspulse",
    "belief_engine",
    "generative_agents",
    "chuang_opinion_dynamics",
)
EXPECTED_OBSERVATIONS = 512
EXPECTED_AGENTS = 350
EXPECTED_TICKS = tuple(range(10))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preflight(
    observations_path: Path,
    trace_report_path: Path,
    provider_boundary_path: Path,
) -> dict[str, Any]:
    observations = load_observations(observations_path)
    trace = json.loads(trace_report_path.read_text(encoding="utf-8"))
    provider = json.loads(provider_boundary_path.read_text(encoding="utf-8"))
    agents = {row.observation.agent_id for row in observations}
    ticks = Counter(row.observation.tick for row in observations)
    ids = [row.observation_id for row in observations]
    if len(observations) != EXPECTED_OBSERVATIONS:
        raise ValueError("dimension-2 horizontal panel must contain 512 observations")
    if len(agents) != EXPECTED_AGENTS:
        raise ValueError("dimension-2 horizontal panel must contain 350 agents")
    if tuple(sorted(ticks)) != EXPECTED_TICKS:
        raise ValueError("dimension-2 horizontal panel must span Tick 0 through Tick 9")
    if len(ids) != len(set(ids)):
        raise ValueError("dimension-2 horizontal observation IDs are not unique")
    if trace.get("paired_methods") != list(METHODS):
        raise ValueError("trace method binding does not match the four frozen methods")
    if trace.get("observation_count") != EXPECTED_OBSERVATIONS:
        raise ValueError("trace report observation count drift")
    if trace.get("observations_file_sha256") != sha256(observations_path):
        raise ValueError("trace report does not bind the observations file")
    main_calls = EXPECTED_OBSERVATIONS * len(METHODS)
    result = {
        "schema_version": "dimension2-cross-method-preflight-v1.0.0",
        "scientific_result": False,
        "network_calls_made": 0,
        "execution_authorized": False,
        "scenario_id": trace["scenario_id"],
        "seed": trace["seed"],
        "methods": list(METHODS),
        "observation_count": len(observations),
        "agent_count": len(agents),
        "tick_counts": {str(key): ticks[key] for key in sorted(ticks)},
        "paired_main_decision_slots": main_calls,
        "main_decision_slots_by_method": {method: EXPECTED_OBSERVATIONS for method in METHODS},
        "auxiliary_calls": {
            "status": "data_dependent_must_be_hard_bounded_before_authorization",
            "belief_engine": "argument extraction + evidence judgement + embeddings",
            "generative_agents": "importance + reflection + embeddings",
            "campuspulse": 0,
            "chuang_opinion_dynamics": 0,
        },
        "applicability": {
            "direct_context_uptake": list(METHODS),
            "unsupported_fact_fabrication": list(METHODS),
            "expression_pre_state_agreement": ["campuspulse", "belief_engine"],
            "language_style_consistency": list(METHODS),
        },
        "input_sha256": {
            "observations": sha256(observations_path),
            "trace_report": sha256(trace_report_path),
            "provider_boundary": sha256(provider_boundary_path),
        },
        "model_id": provider.get("response_calls", {}).get("main_action", {}).get("model"),
        "gates_before_real_execution": [
            "CampusPulse full-method runner emits 512 paired records and native predecision particles",
            "Belief Engine emits one before-decision log-odds sidecar row per record",
            "all four methods use cross_method_full and identical observation hashes",
            "auxiliary request and token hard limits are frozen",
            "a separate explicit paid-run authorization is recorded",
        ],
    }
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["preflight_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen dimension-2 horizontal panel without provider calls.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_preflight(args.observations.resolve(), args.trace_report.resolve(), args.provider_boundary.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
