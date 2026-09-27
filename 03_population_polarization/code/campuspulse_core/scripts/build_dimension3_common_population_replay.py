from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campuspulse_core.forum_relationships import (  # noqa: E402
    compile_scenario_stake_release_v1,
    compile_synthetic_relationship_release_v1,
)


SCENARIOS = (
    "governance_legitimacy_dispute",
    "century_gym_ghost_booking_dispute",
    "lecture_external_incident_shock",
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def field_value(position: dict[str, Any], field: str) -> Any:
    return position["fields"][field]["value"]


def two_class_group(position: dict[str, Any]) -> str:
    direct = bool(field_value(position, "directly_affected"))
    service = float(field_value(position, "service_channel_dependency"))
    resource = float(field_value(position, "resource_need_intensity"))
    return "directly_affected" if direct or max(service, resource) >= 0.65 else "bystander"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct and freeze the common D3 population, stakes, and network.")
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--hero-population", type=Path, required=True)
    parser.add_argument("--public-stakes", type=Path, required=True)
    parser.add_argument("--public-relationships", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--scenario", default="governance_legitimacy_dispute")
    args = parser.parse_args()

    population = load(args.population)
    hero = load(args.hero_population)
    public_stakes = load(args.public_stakes)
    public_relationships = load(args.public_relationships)
    stakes = compile_scenario_stake_release_v1(population=population, scenario_ids=SCENARIOS, seed=args.seed)
    relationships = compile_synthetic_relationship_release_v1(population=population, stake_release=stakes, seed=args.seed)

    semantic_ids = {str(row["agent_id"]) for row in population["agents"]}
    hero_ids = {str(row["llm_agent_id"]) for row in hero["agents"]}
    if semantic_ids != hero_ids or len(semantic_ids) != 1000:
        raise ValueError(f"hero/semantic identity mismatch: semantic={len(semantic_ids)} hero={len(hero_ids)}")

    checks = {
        "population_sha256_matches_stake_publication": stakes["population_sha256"] == public_stakes["population_sha256"],
        "population_sha256_matches_relationship_publication": relationships["population_sha256"] == public_relationships["population_sha256"],
        "stake_release_sha256_exact_match": stakes["stake_release_sha256"] == public_stakes["stake_release_sha256"],
        "relationship_release_sha256_exact_match": relationships["relationship_release_sha256"] == public_relationships["relationship_release_sha256"],
        "relationship_stake_lineage_exact_match": relationships["stake_release_sha256"] == public_relationships["stake_release_sha256"],
        "public_aggregate_matrix_exact_match": relationships["public_aggregate_matrix"] == public_relationships["public_aggregate_matrix"],
        "position_count_exact_match": len(stakes["positions"]) == int(public_stakes["position_count"]),
        "agent_count_exact_match": int(relationships["agent_count"]) == int(public_relationships["agent_count"]) == 1000,
        "edge_count_exact_match": len(relationships["edges"]) == int(public_relationships["edge_count"]) == 8000,
        "semantic_and_hero_identity_sets_exact_match": semantic_ids == hero_ids,
    }
    if not all(checks.values()):
        raise ValueError({key: value for key, value in checks.items() if not value})

    scenario_positions = [row for row in stakes["positions"] if row["scenario_id"] == args.scenario]
    if len(scenario_positions) != 1000 or {str(row["agent_id"]) for row in scenario_positions} != semantic_ids:
        raise ValueError("selected scenario does not cover the frozen 1000-agent population exactly")
    positions_by_agent = {str(row["agent_id"]): row for row in scenario_positions}
    hero_by_agent = {str(row["llm_agent_id"]): row for row in hero["agents"]}

    degree = Counter((row["source_agent_id"], row["relation_type"]) for row in relationships["edges"])
    expected_degree = {"close_friend": 2, "familiar_peer": 4, "cross_group_acquaintance": 2}
    degree_failures = [
        agent_id for agent_id in sorted(semantic_ids)
        if any(degree[(agent_id, relation)] != count for relation, count in expected_degree.items())
    ]
    if degree_failures:
        raise ValueError(f"degree contract failed for {len(degree_failures)} agents")
    checks["per_agent_degree_contract_exact_match"] = True

    agent_rows: list[dict[str, Any]] = []
    stake_rows: list[dict[str, Any]] = []
    for agent_id in sorted(semantic_ids):
        position = positions_by_agent[agent_id]
        hero_row = hero_by_agent[agent_id]
        group = two_class_group(position)
        agent_rows.append({
            "schema_version": "dimension3-common-agent-v1.0.0",
            "agent_id": agent_id,
            "profile_id": str(hero_row.get("profile_bundle_id") or f"profile-{agent_id}"),
            "primary_macro_role": str(hero_row.get("primary_archetype_id") or ""),
            "population_weight": float(hero_row.get("population_weight", 1.0 / 1000.0)),
            "interest_group": group,
            "initial_support": float((hero_row.get("state") or {}).get("support", 0.3)),
            "initial_opposition": float((hero_row.get("state") or {}).get("opposition", 0.3)),
        })
        stake_rows.append({
            "schema_version": "dimension3-common-stake-v1.0.0",
            "scenario_id": args.scenario,
            "agent_id": agent_id,
            "interest_group": group,
            "stake_position_id": position["stake_position_id"],
            "fields": position["fields"],
        })

    edge_rows = [
        {"schema_version": "dimension3-common-edge-v1.0.0", **row}
        for row in sorted(relationships["edges"], key=lambda row: (row["source_agent_id"], row["relation_type"], row["target_agent_id"]))
    ]
    group_counts = dict(sorted(Counter(row["interest_group"] for row in agent_rows).items()))

    args.output_root.mkdir(parents=True, exist_ok=False)
    agent_path = args.output_root / "agents.jsonl"
    stake_path = args.output_root / "stake_positions.jsonl"
    edge_path = args.output_root / "relationship_edges.jsonl"
    write_jsonl(agent_path, agent_rows)
    write_jsonl(stake_path, stake_rows)
    write_jsonl(edge_path, edge_rows)
    audit = {
        "schema_version": "dimension3-common-replay-audit-v1.0.0",
        "status": "passed_exact_reconstruction",
        "seed": args.seed,
        "scenario_id": args.scenario,
        "checks": checks,
        "population_count": len(agent_rows),
        "interest_group_counts": group_counts,
        "edge_count": len(edge_rows),
        "out_degree_per_agent": expected_degree,
        "source_internal_hashes": {
            "population_sha256": population["population_sha256"],
            "stake_release_sha256": stakes["stake_release_sha256"],
            "relationship_release_sha256": relationships["relationship_release_sha256"],
        },
        "source_file_sha256": {
            "population": sha256(args.population),
            "hero_population": sha256(args.hero_population),
            "public_stakes": sha256(args.public_stakes),
            "public_relationships": sha256(args.public_relationships),
        },
        "output_sha256": {
            "agents": sha256(agent_path),
            "stake_positions": sha256(stake_path),
            "relationship_edges": sha256(edge_path),
        },
        "network_calls_made": 0,
    }
    write_json(args.output_root / "reconstruction_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
