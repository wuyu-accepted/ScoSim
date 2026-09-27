from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparison_harness.prompting import STANCE_PROPOSITION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the frozen 12-agent Tick-3 predecision state from the private formal snapshot."
    )
    parser.add_argument("--private-particle-snapshot", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, required=True,
                        help="Private JSON object mapping frozen agent_id to runtime parent index.")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    identity = json.loads(args.identity_map.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("identity map must be a JSON object")
    expected = [str(row["agent_id"]) for row in panel["agents"]]
    if set(identity) != set(expected):
        raise ValueError("identity map must cover exactly the frozen panel")

    with np.load(args.private_particle_snapshot, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        frames = metadata.get("frames") or []
        matches = [frame for frame in frames if int(frame["tick"]) == int(panel["tick"])
                   and str(frame["branch"]) == "shared_baseline"]
        if len(matches) != 1:
            raise ValueError("private snapshot must contain exactly one Tick-3 shared_baseline frame")
        prefix = str(matches[0]["prefix"])
        values = np.asarray(source[f"{prefix}_values"], dtype=np.float64)
        weights = np.asarray(source[f"{prefix}_weights"], dtype=np.float64)
        fields = [str(value) for value in source["state_fields"].tolist()]
        if values.shape != (1000, 10, 12) or weights.shape != (1000, 10):
            raise ValueError("private snapshot particle shape drift")
        agents = []
        for agent_id in expected:
            index = identity[agent_id]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 1000:
                raise ValueError(f"invalid runtime parent index for {agent_id}")
            parent_weights = weights[index]
            total = float(parent_weights.sum())
            if total <= 0:
                raise ValueError(f"nonpositive particle weight for {agent_id}")
            agents.append({
                "agent_id": agent_id,
                "particles": [
                    {field: float(value) for field, value in zip(fields, particle, strict=True)}
                    for particle in values[index]
                ],
                "particle_weights": (parent_weights / total).tolist(),
            })

    result = {
        "schema_version": "campuspulse-d2-predecision-snapshot-v1.0.0",
        "scenario_id": panel["scenario_id"], "seed": panel["seed"], "tick": panel["tick"],
        "stance_proposition": STANCE_PROPOSITION,
        "source_run_primary_slots": 366,
        "source_provenance": {
            "private_particle_snapshot_sha256": _sha256(args.private_particle_snapshot),
            "identity_map_sha256": _sha256(args.identity_map),
            "panel_sha256": _sha256(args.panel),
            "frame_branch": "shared_baseline", "frame_tick": panel["tick"],
        },
        "agents": agents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"agents": len(agents), "output_sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
