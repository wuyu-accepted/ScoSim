from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

from campuspulse_core.governance_arena_v2 import STATE_FIELDS
from comparison_harness.prompting import STANCE_PROPOSITION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the registered Tick-3 particle snapshot from a complete frozen warmup."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--assimilation-sidecar", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if panel.get("schema_version") != "campuspulse-dimension2-panel-v3":
        raise ValueError("warmup snapshot requires the frozen v3 panel")
    expected_agents = [str(row["agent_id"]) for row in panel["agents"]]
    records = _read_jsonl(args.records)
    sidecars = _read_jsonl(args.assimilation_sidecar)
    if len(records) != 36 or len(sidecars) != 36:
        raise ValueError("warmup requires exactly 36 records and 36 assimilation rows")
    if Counter(int(row.get("tick", -1)) for row in records) != Counter({0: 12, 1: 12, 2: 12}):
        raise ValueError("warmup must contain 12 records at each of Ticks 0, 1 and 2")
    if any(row.get("method") != "campuspulse" or row.get("variant") != "warmup_full" for row in records):
        raise ValueError("warmup records must use campuspulse/warmup_full")
    keys = [(str(row.get("agent_id")), int(row.get("tick", -1))) for row in records]
    if len(set(keys)) != 36 or {agent for agent, _ in keys} != set(expected_agents):
        raise ValueError("warmup records are not the complete paired 12-agent trajectory")

    by_source: dict[str, dict[str, Any]] = {}
    for sidecar in sidecars:
        source_id = str(sidecar.get("source_record_id") or "")
        if not source_id or source_id in by_source:
            raise ValueError("warmup sidecar has missing or duplicate source_record_id")
        by_source[source_id] = sidecar
    if {str(row.get("source_record_id")) for row in records} != set(by_source):
        raise ValueError("warmup records and assimilation sidecars do not pair exactly")

    agents = []
    for agent_id in expected_agents:
        trajectory = sorted(
            (row for row in records if str(row["agent_id"]) == agent_id),
            key=lambda row: int(row["tick"]),
        )
        if [int(row["tick"]) for row in trajectory] != [0, 1, 2]:
            raise ValueError(f"incomplete warmup trajectory for {agent_id}")
        agent_sidecars = [by_source[str(row["source_record_id"])] for row in trajectory]
        for previous, current_row in zip(agent_sidecars[:-1], trajectory[1:], strict=True):
            previous_particles = np.asarray([
                [float(particle[field]) for field in STATE_FIELDS]
                for particle in previous["after_particles"]
            ])
            current_particles = np.asarray([
                [float(particle[field]) for field in STATE_FIELDS]
                for particle in current_row["particles"]
            ])
            previous_weights = np.asarray(previous["after_particle_weights"], dtype=np.float64)
            current_weights = np.asarray(current_row["particle_weights"], dtype=np.float64)
            if not np.array_equal(previous_particles, current_particles) or not np.array_equal(previous_weights, current_weights):
                raise ValueError(f"particle-chain discontinuity for {agent_id}")
            for field, value in previous["after_state"].items():
                if not np.isclose(float(value), float(current_row["pre_state"][field]), atol=1e-7):
                    raise ValueError(f"parent-state chain discontinuity for {agent_id}:{field}")
        final = agent_sidecars[-1]
        agents.append({
            "agent_id": agent_id,
            "particles": final["after_particles"],
            "particle_weights": final["after_particle_weights"],
        })

    snapshot = {
        "schema_version": "campuspulse-d2-predecision-snapshot-v1.1.0",
        "protocol_version": "1.2.0",
        "scenario_id": panel["scenario_id"],
        "seed": int(panel["seed"]),
        "tick": 3,
        "stance_proposition": STANCE_PROPOSITION,
        "source_kind": "frozen_warmup_trajectory",
        "warmup_condition": "full",
        "warmup_ticks": [0, 1, 2],
        "warmup_record_count": 36,
        "capture_point": "after_tick_2_native_assimilation_before_tick_3_ablation_generation",
        "source_provenance": {
            "warmup_records_sha256": _sha256(args.records),
            "warmup_assimilation_sha256": _sha256(args.assimilation_sidecar),
            "panel_sha256": _sha256(args.panel),
        },
        "agents": agents,
    }
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(snapshot), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"warmup snapshot schema failed: {errors[0].message}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({"agents": len(agents), "output": str(args.output), "sha256": _sha256(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
