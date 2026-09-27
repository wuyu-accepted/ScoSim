from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison_harness.adapters import ArgumentCandidate  # noqa: E402
from comparison_harness.contracts import AgentSeed  # noqa: E402
from comparison_harness.dimension3_belief_network import BeliefNetworkPopulation  # noqa: E402


FAKE_MARKER = "FAKE_D3_BELIEF_NETWORK_CONFORMANCE_NOT_A_PAPER_RESULT"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def embedding(text: str) -> tuple[float, ...]:
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple((value / 255.0) * 2.0 - 1.0 for value in raw)


def forbidden_extractor(text: str, source_role: str):
    raise RuntimeError("conformance run accepts frozen structured arguments only")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the real Belief Engine update and fixed-network propagation without model calls.")
    parser.add_argument("--initializations", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    init_rows = [row for row in read_jsonl(args.initializations) if row["method"] == "belief_engine"]
    edge_rows = read_jsonl(args.edges)
    if len(init_rows) != 1000 or len(edge_rows) != 8000:
        raise ValueError("formal Belief conformance grid requires 1000 agents and 8000 edges")
    seeds: dict[str, AgentSeed] = {}
    seed_arguments: dict[str, tuple[ArgumentCandidate, ...]] = {}
    groups: dict[str, str] = {}
    for row in init_rows:
        agent_id = str(row["agent_id"])
        native = row["native_initialization"]
        shared_claim = canonical({"role_background": native["role_background"], "stake_position": native["stake_position"]})
        seeds[agent_id] = AgentSeed(agent_id, None, row["interest_group"], native["role_background"], None)
        groups[agent_id] = str(row["interest_group"])
        seed_arguments[agent_id] = tuple(
            ArgumentCandidate(
                claim=f"shared-initial-{item['polarity']}:{shared_claim}",
                polarity=int(item["polarity"]), strength=float(item["strength"]), source_role="seed",
            )
            for item in native["structured_argument_library"]
        )
    population = BeliefNetworkPopulation(
        seeds=seeds, seed_arguments=seed_arguments,
        directed_edges=((row["source_agent_id"], row["target_agent_id"]) for row in edge_rows),
        extractor=forbidden_extractor, embedder=embedding,
        evidence_uptake=0.2, prior_anchoring=0.4,
        similarity_threshold=0.60, arguments_per_edge=2,
    )

    states: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for row in population.population_state(0, groups, args.seed):
        states.append({**row, "experiment_status": "conformance_only_not_a_paper_result", "fake_marker": FAKE_MARKER})
    for tick in range(1, 10):
        if tick == 3:
            # Content fixture only: exercises heterogeneous event admission. It is
            # not the formal frozen argument extraction and cannot enter the paper.
            event = {
                agent_id: (ArgumentCandidate(
                    claim=("fixture: resource loss is personally consequential" if groups[agent_id] == "directly_affected" else "fixture: resource loss has no direct personal claim"),
                    polarity=(1 if groups[agent_id] == "directly_affected" else -1),
                    strength=0.8, source_role="opponent",
                ),)
                for agent_id in sorted(seeds)
            }
            population.inject_event_arguments(tick=tick, arguments_by_agent=event, event_id="conformance-resource-shock")
        deliveries = population.step(tick)
        traces.append({
            "tick": tick, "delivery_count": len(deliveries),
            "argument_count": sum(len(item.arguments) for item in deliveries),
            "delivery_sha256": hashlib.sha256(canonical([
                (item.source_agent_id, item.target_agent_id, [(a.polarity, a.strength, a.claim) for a in item.arguments])
                for item in deliveries
            ]).encode("utf-8")).hexdigest(),
        })
        if tick in {3, 9}:
            for row in population.population_state(tick, groups, args.seed):
                states.append({**row, "experiment_status": "conformance_only_not_a_paper_result", "fake_marker": FAKE_MARKER})

    keys = {(row["tick"], row["agent_id"]) for row in states}
    checks = {
        "one_thousand_agents": len(seeds) == 1000,
        "eight_thousand_fixed_edges": len(population.edges) == 8000,
        "ticks_1_through_9_propagated": population.completed_ticks == list(range(1, 10)),
        "seventy_two_thousand_edge_deliveries": len(population.delivery_log) == 72000,
        "three_thousand_measurements": len(states) == len(keys) == 3000,
        "native_log_odds_mapping_exact": all(abs(float(row["public_stance"]) - __import__("math").tanh(float(row["native_log_odds"]) / 2.0)) < 1e-12 for row in states),
        "all_rows_marked_non_paper": all(row["fake_marker"] == FAKE_MARKER for row in states),
        "real_model_calls": False,
    }
    if not all(value is True for key, value in checks.items() if key != "real_model_calls") or checks["real_model_calls"]:
        raise ValueError(checks)
    args.output_root.mkdir(parents=True, exist_ok=False)
    states_path = args.output_root / "belief_population_states.conformance.jsonl"
    trace_path = args.output_root / "belief_network_trace.conformance.jsonl"
    write_jsonl(states_path, states)
    write_jsonl(trace_path, traces)
    audit = {
        "schema_version": "dimension3-belief-network-conformance-audit-v1.0.0",
        "status": "passed_not_a_paper_result", "fake_marker": FAKE_MARKER,
        "checks": checks, "real_model_calls": 0, "real_tokens": 0,
        "output_sha256": {
            "states": hashlib.sha256(states_path.read_bytes()).hexdigest(),
            "trace": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
    }
    write_json(args.output_root / "conformance_audit.json", audit)
    print(canonical(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
