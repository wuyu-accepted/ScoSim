from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


METHODS = (
    "campuspulse",
    "belief_engine",
    "generative_agents",
    "chuang_opinion_dynamics",
)
TICKS = (0, 3, 9)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def role_card(hero: dict[str, Any]) -> dict[str, Any]:
    persona = dict(hero.get("persona") or {})
    stable = dict(hero.get("stable_persona") or {})
    semantic = dict(hero.get("semantic") or {})
    return {
        "profile_summary": persona.get("profile_summary"),
        "situational_signature": persona.get("situational_signature"),
        "uncertainty_behavior": persona.get("uncertainty_behavior"),
        "language_tendencies": list(persona.get("language_tendencies") or ()),
        "behavior_boundaries": list(persona.get("do_not_simulate") or ()),
        "interaction_role": stable.get("interaction_role"),
        "forum_tendency_distribution": stable.get("forum_tendency_distribution"),
        "trust_style": stable.get("trust_style"),
        "language_style": stable.get("language_style"),
        "activity_rhythm": stable.get("activity_rhythm"),
        "attention_budget": stable.get("attention_budget"),
        "semantic_speech_act": semantic.get("speech_act"),
        "semantic_stance": semantic.get("stance"),
        "scenario_relevance": hero.get("scenario_relevance"),
    }


def clipped_log_odds(stance: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, (stance + 1.0) / 2.0))
    return math.log(probability / (1.0 - probability))


def five_level(stance: float) -> int:
    scaled = max(-1.0, min(1.0, stance)) * 2.0
    return min((-2, -1, 0, 1, 2), key=lambda value: (abs(value - scaled), abs(value), value))


def native_payload(method: str, shared: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    stance = float(shared["initial_net_stance"])
    if method == "campuspulse":
        return {
            "stable_profile": shared["role_background"],
            "stake_position": shared["stake_position"],
            "initial_state_12d": state,
            "particle_count": 10,
            "network_edge_reference": shared["network_edge_reference"],
        }
    if method == "belief_engine":
        strength_positive = (stance + 1.0) / 2.0
        return {
            "initial_log_odds": clipped_log_odds(stance),
            "structured_argument_library": [
                {"polarity": 1, "strength": strength_positive, "source": "shared_initial_background"},
                {"polarity": -1, "strength": 1.0 - strength_positive, "source": "shared_initial_background"},
            ],
            "role_background": shared["role_background"],
            "stake_position": shared["stake_position"],
            "network_edge_reference": shared["network_edge_reference"],
        }
    if method == "generative_agents":
        return {
            "initial_memory_stream": [
                {"kind": "profile", "importance": 5.0, "content": shared["role_background"]},
                {"kind": "stake", "importance": 5.0, "content": shared["stake_position"]},
            ],
            "network_edge_reference": shared["network_edge_reference"],
            "stance_probe_required_at_ticks": list(TICKS),
        }
    if method == "chuang_opinion_dynamics":
        return {
            "initial_opinion_five_level": five_level(stance),
            "normalized_initial_opinion": five_level(stance) / 2.0,
            "closed_world_history": [
                {"kind": "profile", "content": shared["role_background"]},
                {"kind": "stake", "content": shared["stake_position"]},
            ],
            "network_edge_reference": shared["network_edge_reference"],
            "allowed_opinions": [-2, -1, 0, 1, 2],
        }
    raise ValueError(method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze method-native D3 initializations over one common population/network.")
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--hero-population", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    agents = read_jsonl(args.replay_root / "agents.jsonl")
    stakes = read_jsonl(args.replay_root / "stake_positions.jsonl")
    replay_audit = read_json(args.replay_root / "reconstruction_audit.json")
    hero_release = read_json(args.hero_population)
    hero = {str(row["llm_agent_id"]): row for row in hero_release["agents"]}
    agent_ids = {str(row["agent_id"]) for row in agents}
    stakes_by_agent = {str(row["agent_id"]): row for row in stakes}
    if replay_audit.get("status") != "passed_exact_reconstruction":
        raise ValueError("common replay layer has not passed exact reconstruction")
    if len(agents) != 1000 or agent_ids != set(hero) or agent_ids != set(stakes_by_agent):
        raise ValueError("common replay, hero, and stake identity sets must be the same frozen 1000 agents")

    network_hash = file_digest(args.replay_root / "relationship_edges.jsonl")
    rows: list[dict[str, Any]] = []
    common_hashes: dict[str, str] = {}
    for agent in sorted(agents, key=lambda row: str(row["agent_id"])):
        agent_id = str(agent["agent_id"])
        hero_row = hero[agent_id]
        state = {key: float(value) for key, value in (hero_row.get("state") or {}).items() if isinstance(value, (int, float))}
        card = role_card(hero_row)
        stake = stakes_by_agent[agent_id]
        shared = {
            "agent_id": agent_id,
            "profile_id": agent["profile_id"],
            "interest_group": agent["interest_group"],
            "role_background": card,
            "stake_position": stake["fields"],
            "initial_net_stance": float(agent["initial_support"]) - float(agent["initial_opposition"]),
            "network_edge_reference": {"sha256": network_hash, "source_agent_id": agent_id},
        }
        shared_hash = digest(shared)
        common_hashes[agent_id] = shared_hash
        for method in METHODS:
            rows.append({
                "schema_version": "dimension3-method-initialization-v1.0.0",
                "seed": args.seed,
                "scenario_id": replay_audit["scenario_id"],
                "method": method,
                "agent_id": agent_id,
                "interest_group": agent["interest_group"],
                "shared_information_sha256": shared_hash,
                "native_initialization": native_payload(method, shared, state),
            })

    protocol = {
        "schema_version": "dimension3-cross-method-protocol-v1.0.0",
        "status": "frozen",
        "seed": args.seed,
        "scenario_id": replay_audit["scenario_id"],
        "scenario_fact_contract": {
            "available_beds_before": 520,
            "available_beds_after": 320,
            "valid_applications": 536,
            "forbidden_invention": "No method may add allocation rules, rankings, deadlines, or appeal facts absent from shared observations.",
        },
        "methods": list(METHODS),
        "ticks": list(range(10)),
        "measurement_ticks": list(TICKS),
        "population_count": 1000,
        "shared_network_sha256": network_hash,
        "network_delivery": "At each tick, method-native messages propagate only along the frozen directed edge set.",
        "stance_collection": {
            "campuspulse": "weighted mean support minus weighted mean opposition over 10 particles",
            "belief_engine": "native log-odds-derived scalar mapped to [-1,1]",
            "generative_agents": "standardized five-level full-population questionnaire at ticks 0,3,9",
            "chuang_opinion_dynamics": "native integer opinion in {-2,-1,0,1,2} divided by 2",
        },
        "required_output": {
            "schema_version": "dimension3-population-state-v1.0.0",
            "one_row_per": "method x tick(0,3,9) x agent",
            "required_fields": ["method", "variant", "seed", "tick", "agent_id", "interest_group", "public_stance"],
            "expected_row_count": 12000,
        },
        "statistics": {
            "permutations": 199,
            "sarle_threshold": 0.555,
            "entropy_edges": [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "entropy_bin_count": 20,
            "entropy_degenerate_variance_threshold": 1e-10,
            "sd_ddof": 0,
        },
    }

    counts = {method: sum(row["method"] == method for row in rows) for method in METHODS}
    shared_hash_sets = {
        agent_id: {row["shared_information_sha256"] for row in rows if row["agent_id"] == agent_id}
        for agent_id in agent_ids
    }
    chuang_values = {
        row["native_initialization"]["initial_opinion_five_level"]
        for row in rows if row["method"] == "chuang_opinion_dynamics"
    }
    checks = {
        "common_replay_exact_reconstruction_passed": True,
        "four_methods_x_1000_rows": len(rows) == 4000 and all(value == 1000 for value in counts.values()),
        "identity_sets_equal": all({row["agent_id"] for row in rows if row["method"] == method} == agent_ids for method in METHODS),
        "shared_information_equal_across_methods": all(len(values) == 1 for values in shared_hash_sets.values()),
        "single_frozen_network_reference": all(row["native_initialization"]["network_edge_reference"]["sha256"] == network_hash for row in rows),
        "chuang_output_domain_valid": chuang_values <= {-2, -1, 0, 1, 2},
        "measurement_grid_is_0_3_9": protocol["measurement_ticks"] == [0, 3, 9],
    }
    if not all(checks.values()):
        raise ValueError({key: value for key, value in checks.items() if not value})

    args.output_root.mkdir(parents=True, exist_ok=False)
    rows_path = args.output_root / "method_initializations.jsonl"
    protocol_path = args.output_root / "locked_dimension3_protocol.json"
    write_jsonl(rows_path, rows)
    write_json(protocol_path, protocol)
    audit = {
        "schema_version": "dimension3-method-projection-audit-v1.0.0",
        "status": "passed",
        "checks": checks,
        "method_counts": counts,
        "interest_group_counts": replay_audit["interest_group_counts"],
        "network_sha256": network_hash,
        "input_sha256": {
            "agents": file_digest(args.replay_root / "agents.jsonl"),
            "stakes": file_digest(args.replay_root / "stake_positions.jsonl"),
            "network": network_hash,
            "reconstruction_audit": file_digest(args.replay_root / "reconstruction_audit.json"),
            "hero_population": file_digest(args.hero_population),
        },
        "output_sha256": {
            "method_initializations": file_digest(rows_path),
            "locked_protocol": file_digest(protocol_path),
        },
        "network_calls_made": 0,
    }
    write_json(args.output_root / "projection_audit.json", audit)
    print(canonical(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
