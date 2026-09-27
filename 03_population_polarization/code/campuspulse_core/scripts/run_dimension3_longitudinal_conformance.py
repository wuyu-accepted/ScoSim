from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FAKE_MARKER = "FAKE_D3_CONFORMANCE_ONLY_NOT_A_PAPER_RESULT"
METHODS = ("campuspulse", "belief_engine", "generative_agents", "chuang_opinion_dynamics")
MEASUREMENT_TICKS = (0, 3, 9)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def initial_stance(row: dict[str, Any]) -> float:
    native = row["native_initialization"]
    method = row["method"]
    if method == "campuspulse":
        state = native["initial_state_12d"]
        return clip(float(state.get("support", 0.3)) - float(state.get("opposition", 0.3)))
    if method == "belief_engine":
        return clip(math.tanh(float(native["initial_log_odds"]) / 2.0))
    if method == "generative_agents":
        return 0.0
    if method == "chuang_opinion_dynamics":
        return float(native["initial_opinion_five_level"]) / 2.0
    raise ValueError(method)


def fake_update(method: str, current: float, neighbor_mean: float, affected: bool, tick: int) -> float:
    """Deterministic transport substitute; never a scientific model result."""
    shock = 1.0 if tick >= 3 else 0.0
    stake = 0.18 if affected else -0.035
    if method == "campuspulse":
        return clip(0.72 * current + 0.20 * neighbor_mean + shock * stake)
    if method == "belief_engine":
        probability = min(1 - 1e-6, max(1e-6, (current + 1.0) / 2.0))
        log_odds = math.log(probability / (1.0 - probability))
        log_odds += 0.30 * neighbor_mean + shock * (0.38 if affected else -0.07)
        return clip(math.tanh(log_odds / 2.0))
    if method == "generative_agents":
        raw = 0.58 * current + 0.26 * neighbor_mean + shock * (0.15 if affected else -0.025)
        return min((-1.0, -0.5, 0.0, 0.5, 1.0), key=lambda value: (abs(value - raw), abs(value), value))
    if method == "chuang_opinion_dynamics":
        confirm = neighbor_mean if current == 0.0 or current * neighbor_mean >= 0 else 0.35 * neighbor_mean
        raw = 0.70 * current + 0.22 * confirm + shock * (0.16 if affected else -0.03)
        native = min((-2, -1, 0, 1, 2), key=lambda value: (abs(value - 2.0 * raw), abs(value), value))
        return native / 2.0
    raise ValueError(method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline full-grid conformance replay for the frozen D3 executor.")
    parser.add_argument("--projection-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_json(args.projection_root / "locked_dimension3_protocol.json")
    projection_audit = read_json(args.projection_root / "projection_audit.json")
    initializations = read_jsonl(args.projection_root / "method_initializations.jsonl")
    edges = read_jsonl(args.replay_root / "relationship_edges.jsonl")
    if protocol.get("status") != "frozen" or projection_audit.get("status") != "passed":
        raise ValueError("dimension-3 protocol/projection is not frozen and audited")
    if sha256(args.replay_root / "relationship_edges.jsonl") != protocol["shared_network_sha256"]:
        raise ValueError("network hash does not match the locked protocol")

    neighbors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        neighbors[str(edge["source_agent_id"])].append(str(edge["target_agent_id"]))
    if len(neighbors) != 1000 or any(len(values) != 8 for values in neighbors.values()):
        raise ValueError("frozen network must expose exactly eight outgoing neighbors per agent")

    by_method: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in METHODS}
    for row in initializations:
        by_method[row["method"]][str(row["agent_id"])] = row
    id_sets = [set(by_method[method]) for method in METHODS]
    if any(len(values) != 1000 for values in id_sets) or any(values != id_sets[0] for values in id_sets[1:]):
        raise ValueError("four method identity grids are incomplete or unequal")
    agent_ids = sorted(id_sets[0])

    outputs: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for method in METHODS:
        state = {agent_id: initial_stance(by_method[method][agent_id]) for agent_id in agent_ids}
        for tick in range(10):
            if tick > 0:
                previous = dict(state)
                for agent_id in agent_ids:
                    neighbor_mean = sum(previous[target] for target in neighbors[agent_id]) / 8.0
                    affected = by_method[method][agent_id]["interest_group"] == "directly_affected"
                    state[agent_id] = fake_update(method, previous[agent_id], neighbor_mean, affected, tick)
            trace.append({
                "fake_marker": FAKE_MARKER,
                "method": method,
                "tick": tick,
                "agent_count": len(state),
                "mean_stance": sum(state.values()) / len(state),
                "state_sha256": hashlib.sha256(canonical(state).encode("utf-8")).hexdigest(),
            })
            if tick in MEASUREMENT_TICKS:
                for agent_id in agent_ids:
                    outputs.append({
                        "schema_version": "dimension3-population-state-v1.0.0",
                        "experiment_status": "conformance_only_not_a_paper_result",
                        "fake_marker": FAKE_MARKER,
                        "method": method,
                        "variant": "full",
                        "seed": int(protocol["seed"]),
                        "tick": tick,
                        "agent_id": agent_id,
                        "interest_group": by_method[method][agent_id]["interest_group"],
                        "public_stance": state[agent_id],
                    })

    expected = int(protocol["required_output"]["expected_row_count"])
    keys = [(row["method"], row["tick"], row["agent_id"]) for row in outputs]
    checks = {
        "projection_audit_passed": True,
        "network_hash_matches_protocol": True,
        "ten_tick_trace_complete": len(trace) == 40,
        "formal_measurement_grid_shape": len(outputs) == expected == 12000,
        "measurement_keys_unique": len(keys) == len(set(keys)),
        "stance_domain_valid": all(-1.0 <= float(row["public_stance"]) <= 1.0 for row in outputs),
        "all_rows_marked_fake": all(row.get("fake_marker") == FAKE_MARKER for row in outputs),
        "no_real_model_calls": True,
    }
    if not all(checks.values()):
        raise ValueError({key: value for key, value in checks.items() if not value})

    args.output_root.mkdir(parents=True, exist_ok=False)
    states_path = args.output_root / "population_states.conformance.jsonl"
    trace_path = args.output_root / "tick_trace.conformance.jsonl"
    write_jsonl(states_path, outputs)
    write_jsonl(trace_path, trace)
    audit = {
        "schema_version": "dimension3-longitudinal-conformance-audit-v1.0.0",
        "status": "passed_not_a_paper_result",
        "fake_marker": FAKE_MARKER,
        "checks": checks,
        "population_state_rows": len(outputs),
        "tick_trace_rows": len(trace),
        "real_model_calls": 0,
        "real_tokens": 0,
        "output_sha256": {"population_states": sha256(states_path), "tick_trace": sha256(trace_path)},
    }
    write_json(args.output_root / "conformance_audit.json", audit)
    print(canonical(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
