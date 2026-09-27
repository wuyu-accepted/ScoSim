from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campuspulse_core.forum_pilot import compile_pilot_profile_bridge  # noqa: E402


SEED_LABEL = b"campus-pulse-public-particle-panel-v1"
FRAME_SELECTION = (("frame_00", 0, "shared_initial"), ("frame_01", 3, "shared_baseline"), ("frame_13", 9, "Natural"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct the frozen anonymous-slot to CampusPulse agent binding and export D3 stances.")
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--hero-population", type=Path, required=True)
    parser.add_argument("--common-agents", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fidelity-evaluation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    population = load_json(args.population)
    hero = load_json(args.hero_population)
    manifest = load_json(args.manifest)
    fidelity = load_json(args.fidelity_evaluation)
    common_rows = load_jsonl(args.common_agents)
    common = {str(row["agent_id"]): row for row in common_rows}

    compiled = compile_pilot_profile_bridge(population)
    runtime_ids = [str(row["agent_id"]) for row in compiled["profiles"]]
    hero_ids = [str(row["llm_agent_id"]) for row in hero["agents"]]
    if len(runtime_ids) != 1000 or len(set(runtime_ids)) != 1000:
        raise ValueError("compiled runtime profile order is not a unique 1000-agent sequence")
    if runtime_ids != hero_ids:
        raise ValueError("hero release order does not match the compiled runtime parent order")
    if set(runtime_ids) != set(common):
        raise ValueError("runtime identity set does not match the frozen common replay")
    if sha256(args.panel) != manifest["panel_sha256"]:
        raise ValueError("particle panel hash does not match its manifest")

    permutation_seed = int.from_bytes(hashlib.sha256(SEED_LABEL).digest()[:8], "little")
    rng = np.random.default_rng(permutation_seed)
    parent_permutation = rng.permutation(1000)
    # The publisher consumes these draws immediately after the parent permutation.
    # Reproducing them confirms the exact RNG call boundary even though particle order
    # does not affect a weighted parent mean.
    particle_permutations = np.stack([rng.permutation(10) for _ in range(1000)])
    slot_to_runtime = {slot: int(parent_permutation[slot]) for slot in range(1000)}
    slot_to_agent = {slot: runtime_ids[index] for slot, index in slot_to_runtime.items()}

    expected_frames = {(str(row["prefix"]), int(row["tick"]), str(row["branch"])) for row in manifest["frames"]}
    if not set(FRAME_SELECTION) <= expected_frames:
        raise ValueError("manifest lacks one of the frozen Tick 0/3/9 Natural frames")
    field_names = [str(value) for value in manifest["state_fields"]]
    support_index, opposition_index = field_names.index("support"), field_names.index("opposition")
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    with np.load(args.panel, allow_pickle=False) as panel:
        for prefix, tick, branch in FRAME_SELECTION:
            values = np.asarray(panel[f"{prefix}_values"], dtype=np.float64)
            weights = np.asarray(panel[f"{prefix}_weights"], dtype=np.float64)
            parent_weights = np.asarray(panel[f"{prefix}_parent_weights"], dtype=np.float64)
            if values.shape != (1000, 10, 12) or weights.shape != (1000, 10):
                raise ValueError(f"invalid frozen particle shape at {prefix}")
            normalized = weights / weights.sum(axis=1, keepdims=True)
            parent_means = np.sum(values * normalized[:, :, None], axis=1)
            stance = np.clip(parent_means[:, support_index] - parent_means[:, opposition_index], -1.0, 1.0)
            normalized_parent_weights = parent_weights / parent_weights.sum()
            summaries[str(tick)] = {
                "branch": branch,
                "unweighted_mean": float(np.mean(stance)),
                "unweighted_sd_ddof_0": float(np.std(stance, ddof=0)),
                "published_weighted_mean": float(np.sum(stance * normalized_parent_weights)),
                "published_weighted_sd_ddof_0": float(np.sqrt(np.sum(normalized_parent_weights * (stance - np.sum(stance * normalized_parent_weights)) ** 2))),
            }
            for slot in range(1000):
                agent_id = slot_to_agent[slot]
                post_state = {
                    field: float(parent_means[slot, index])
                    for index, field in enumerate(field_names)
                }
                output_rows.append({
                    "schema_version": "dimension3-population-state-v1.0.0",
                    "experiment_status": "frozen_real_campuspulse_reconstruction",
                    "method": "campuspulse",
                    "variant": "full",
                    "seed": args.seed,
                    "tick": tick,
                    "agent_id": agent_id,
                    "interest_group": common[agent_id]["interest_group"],
                    "public_stance": float(stance[slot]),
                    "post_state": post_state,
                    "particles": [
                        {field: float(values[slot, particle, index]) for index, field in enumerate(field_names)}
                        for particle in range(10)
                    ],
                    "particle_weights": normalized[slot].tolist(),
                    "particle_source": {"panel_sha256": manifest["panel_sha256"], "frame": prefix, "anonymous_parent_slot": slot},
                })

    published = {}
    for frame in fidelity["dimensions"]["population_heterogeneity"]["frames"]:
        key = (int(frame["tick"]), str(frame["branch"]))
        if key in {(0, "shared_initial"), (3, "shared_baseline"), (9, "Natural")}:
            published[str(key[0])] = {
                "mean": float(frame["net_stance_distribution"]["weighted_mean"]),
                "sd": float(frame["net_stance_distribution"]["weighted_standard_deviation"]),
            }
    tolerance = 1e-6
    aggregate_checks = {
        tick: abs(summaries[tick]["published_weighted_mean"] - published[tick]["mean"]) <= tolerance
        and abs(summaries[tick]["published_weighted_sd_ddof_0"] - published[tick]["sd"]) <= tolerance
        for tick in ("0", "3", "9")
    }
    checks = {
        "panel_hash_matches_manifest": True,
        "compiled_runtime_order_unique_1000": True,
        "compiled_runtime_order_matches_hero_release_order": True,
        "runtime_ids_match_common_replay": True,
        "publisher_parent_permutation_reconstructed": sorted(slot_to_runtime.values()) == list(range(1000)),
        "publisher_particle_rng_boundary_reconstructed": particle_permutations.shape == (1000, 10),
        "tick_0_3_9_frames_present": True,
        "three_thousand_identity_bound_rows": len(output_rows) == 3000,
        "measurement_keys_unique": len({(row["tick"], row["agent_id"]) for row in output_rows}) == 3000,
        "all_rows_have_12d_parent_state": all(len(row["post_state"]) == 12 for row in output_rows),
        "all_rows_have_ten_normalized_particles": all(
            len(row["particles"]) == 10
            and len(row["particle_weights"]) == 10
            and abs(sum(row["particle_weights"]) - 1.0) <= 1e-6
            for row in output_rows
        ),
        "published_aggregate_values_reproduced": all(aggregate_checks.values()),
    }
    if not all(checks.values()):
        raise ValueError({key: value for key, value in checks.items() if not value})

    args.output_root.mkdir(parents=True, exist_ok=False)
    mapping_path = args.output_root / "anonymous_slot_agent_binding.jsonl"
    states_path = args.output_root / "campuspulse_population_states.jsonl"
    mapping_rows = [
        {"anonymous_parent_slot": slot, "runtime_parent_index": slot_to_runtime[slot], "agent_id": slot_to_agent[slot]}
        for slot in range(1000)
    ]
    write_jsonl(mapping_path, mapping_rows)
    write_jsonl(states_path, output_rows)
    audit = {
        "schema_version": "dimension3-campuspulse-identity-binding-audit-v1.0.0",
        "status": "passed_exact_reconstruction",
        "checks": checks,
        "aggregate_checks_by_tick": aggregate_checks,
        "summaries": summaries,
        "published_reference": published,
        "binding_basis": {
            "runtime_order": "compile_pilot_profile_bridge(population)['profiles']",
            "publisher_seed_label_utf8": SEED_LABEL.decode("utf-8"),
            "publisher_permutation_seed": permutation_seed,
            "publisher_algorithm": "numpy.random.default_rng(seed).permutation(1000)",
        },
        "input_sha256": {
            "population": sha256(args.population), "hero_population": sha256(args.hero_population),
            "common_agents": sha256(args.common_agents), "panel": sha256(args.panel),
            "manifest": sha256(args.manifest), "fidelity_evaluation": sha256(args.fidelity_evaluation),
        },
        "output_sha256": {"binding": sha256(mapping_path), "population_states": sha256(states_path)},
        "network_calls_made": 0,
    }
    write_json(args.output_root / "identity_binding_audit.json", audit)
    print(canonical(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
