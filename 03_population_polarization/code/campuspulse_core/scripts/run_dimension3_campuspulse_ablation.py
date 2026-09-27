from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from campuspulse_core.governance_arena_v2 import (  # noqa: E402
    GraphArrays, ParticleEnsemble, STATE_FIELDS, _parent_features,
    advance_particle_tick, assimilate_forum_llm_updates_v2,
    initial_branch_state,
)
from comparison_harness.adapters import CampusPulseAdapter  # noqa: E402
from comparison_harness.adapters.campuspulse import DIMENSION_3  # noqa: E402
from comparison_harness.campuspulse_assimilation import project_shared_decision  # noqa: E402
from comparison_harness.contracts import CommonObservation, ContextItem, canonical_json  # noqa: E402
from comparison_harness.fair_initialization import load_fair_role_initializations  # noqa: E402
from comparison_harness.prompting import COMMON_SECTIONS, SharedPromptRenderer  # noqa: E402
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider  # noqa: E402


VARIANTS = ("single_particle", "no_profile", "no_profile_no_smc")
MEASUREMENT_TICKS = (0, 3, 9)
ALLOWED_ACTIONS = ("read", "post", "reply", "quote", "react", "private_message", "no_action")
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
        "target_id": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
        "emotion": {"type": "string"},
        "public_stance": {"type": ["number", "null"], "minimum": -1, "maximum": 1},
        "public_or_private": {"type": "string", "enum": ["public", "private", "none"]},
    },
    "required": ["action", "target_id", "text", "emotion", "public_stance", "public_or_private"],
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def graph_arrays(agent_ids: list[str], rows: list[dict[str, Any]], digest: str) -> tuple[GraphArrays, dict[str, list[str]]]:
    index = {agent_id: i for i, agent_id in enumerate(agent_ids)}
    outgoing: dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}
    incoming: dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}
    for row in rows:
        source, target = str(row["source_agent_id"]), str(row["target_agent_id"])
        if source not in index or target not in index or source == target:
            raise ValueError("frozen network endpoint mismatch")
        outgoing[source].append(target)
        incoming[target].append(source)
    if any(len(targets) != 8 for targets in outgoing.values()):
        raise ValueError("every frozen agent must have eight outgoing neighbors")
    targets = np.asarray([[index[target] for target in outgoing[source]] for source in agent_ids], dtype=np.int32)
    shape = targets.shape
    return GraphArrays(
        targets=targets, bridge=np.zeros(shape, dtype=bool), weight=np.ones(shape, dtype=np.float32),
        capacity=np.ones(shape, dtype=np.int8), delay=np.ones(shape, dtype=np.int8),
        graph_sha256=digest, public_identity={"graph_sha256": digest, "node_count": len(agent_ids), "out_degree": 8},
    ), incoming


def make_observation(agent_id: str, tick: int, visible: tuple[ContextItem, ...]) -> CommonObservation:
    live = tick >= 3
    facts: dict[str, Any] = {"closed_world_rule": "未展示的分配规则、排名、截止日期和申诉信息均未知。"}
    if live:
        facts.update({"available_beds_before": 520, "available_beds_after": 320, "valid_applications": 536})
    event = ({"event_id": "housing-520-320-applications-536-v1", "announcement": "宿舍床位由520个缩减至320个，目前有536份有效申请。"}
             if live else {"event_id": "pre-announcement", "announcement": "尚未发布本次资源分配公告。"})
    return CommonObservation(
        scenario_id="governance_legitimacy_dispute", tick=tick, agent_id=agent_id,
        factual_registry=facts, current_event=event, visible_context=visible,
        governance_artifacts=(), allowed_actions=ALLOWED_ACTIONS, response_schema=RESPONSE_SCHEMA,
    )


def initial_ensemble(state: Any, variant: str) -> None:
    if variant == "no_profile":
        return
    means = state.ensemble.parent_means().astype(np.float32)
    parent_weights = state.ensemble.parent_weights.copy()
    state.ensemble = ParticleEnsemble(means[:, None, :], parent_weights[:, None], parent_weights)
    state.ensemble.assert_conserved()


def state_rows(state: Any, variant: str, tick: int, seed: int, agent_ids: list[str], groups: dict[str, str]) -> list[dict[str, Any]]:
    means = state.ensemble.parent_means()
    normalized = state.ensemble.normalized_within_parent()
    rows = []
    for index, agent_id in enumerate(agent_ids):
        post_state = {field: float(means[index, offset]) for offset, field in enumerate(STATE_FIELDS)}
        rows.append({
            "schema_version": "dimension3-ablation-state-v1.0.0", "method": "campuspulse",
            "variant": variant, "seed": seed, "tick": tick, "agent_id": agent_id,
            "interest_group": groups[agent_id], "post_state": post_state,
            "public_stance": max(-1.0, min(1.0, post_state["support"] - post_state["opposition"])),
            "particles": ([{field: float(value[offset]) for offset, field in enumerate(STATE_FIELDS)} for value in state.ensemble.values[index]]
                          if variant != "no_profile_no_smc" else None),
            "particle_weights": (normalized[index].tolist() if variant != "no_profile_no_smc" else None),
        })
    return rows


def save_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    path.with_suffix(".sha256").write_text(sha256(path) + "\n", encoding="ascii")


def load_snapshot(directory: Path, variant: str, seed: int) -> dict[str, Any] | None:
    candidates = []
    for path in directory.glob("tick-*.pkl.gz") if directory.exists() else ():
        digest = path.with_suffix(".sha256")
        if digest.exists() and digest.read_text(encoding="ascii").strip() == sha256(path):
            candidates.append((int(path.name.split("-")[1].split(".")[0]), path))
    if not candidates:
        return None
    _, path = max(candidates)
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("variant") != variant or payload.get("seed") != seed:
        raise ValueError("snapshot identity mismatch")
    return payload


def prompt_isolation_check(renderer: SharedPromptRenderer, roles: dict[str, Any], state: Any, agent_ids: list[str]) -> dict[str, Any]:
    agent_id = agent_ids[0]
    index = 0
    means = state.ensemble.parent_means()
    point = means[index]
    bundles = {}
    for variant in (*VARIANTS, "full"):
        adapter = CampusPulseAdapter(DIMENSION_3[variant])
        required = DIMENSION_3[variant].particle_count
        source_particles = np.repeat(point[None, :], max(required, 1), axis=0)
        source_weights = np.repeat(1.0 / max(required, 1), max(required, 1))
        adapter.initialize(roles[agent_id].agent_seed(
            initial_state={field: float(means[index, offset]) for offset, field in enumerate(STATE_FIELDS)},
            initial_particles=[{field: float(row[offset]) for offset, field in enumerate(STATE_FIELDS)} for row in source_particles],
            particle_weights=source_weights.tolist(),
        ))
        bundles[variant] = renderer.render(make_observation(agent_id, 1, ()), adapter.observe(make_observation(agent_id, 1, ())))
    for section in COMMON_SECTIONS:
        if len({bundle.section_sha256[section] for bundle in bundles.values()}) != 1:
            raise ValueError(f"shared prompt section drift: {section}")
    contexts = {variant: bundle.section_text["method_cognitive_context"] for variant, bundle in bundles.items()}
    if len(set(contexts.values())) != 4:
        raise ValueError("four ablation cognitive contexts are not distinct")
    if "stable_profile" in contexts["no_profile"] or "stable_profile" in contexts["no_profile_no_smc"]:
        raise ValueError("profile leaked into a no-profile condition")
    if "predecision_state" in contexts["no_profile_no_smc"] or "particles" in contexts["no_profile_no_smc"]:
        raise ValueError("hidden measurement state leaked into no-profile-no-SMC prompt")
    return {variant: hashlib.sha256(context.encode("utf-8")).hexdigest() for variant, context in contexts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal CampusPulse Dimension-3 counterfactual ablation solver")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--enable-real", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be in [1,16]")
    if not args.preflight_only and not args.enable_real:
        raise ValueError("formal execution requires --enable-real")
    protocol = read_json(args.protocol)
    if protocol.get("status") != "frozen" or protocol.get("seed") != args.seed or protocol.get("measurement_ticks") != list(MEASUREMENT_TICKS):
        raise ValueError("frozen dimension-3 protocol drift")
    population = read_json(args.population)
    common = read_jsonl(args.agents)
    groups = {str(row["agent_id"]): str(row["interest_group"]) for row in common}
    native_ids = [str(row["llm_agent_id"]) for row in population["agents"]]
    if len(native_ids) != 1000 or len(set(native_ids)) != 1000 or set(native_ids) != set(groups):
        raise ValueError("population/common replay identity mismatch")
    edge_digest = sha256(args.edges)
    if edge_digest != protocol.get("shared_network_sha256"):
        raise ValueError("frozen network hash drift")
    graph, incoming = graph_arrays(native_ids, read_jsonl(args.edges), edge_digest)
    roles = load_fair_role_initializations(args.population)
    state = initial_branch_state(population, scenario_id=protocol["scenario_id"], seed=args.seed)
    initial_ensemble(state, args.variant)
    renderer = SharedPromptRenderer()
    isolation = prompt_isolation_check(renderer, roles, state, native_ids)
    preflight = {
        "status": "passed_no_api_calls", "variant": args.variant, "seed": args.seed,
        "agent_count": 1000, "ticks": list(range(10)), "measurement_ticks": list(MEASUREMENT_TICKS),
        "main_action_call_slots": 9000, "workers": args.workers,
        "particles_per_parent": int(state.ensemble.values.shape[1]),
        "stochastic_diffusion": bool(DIMENSION_3[args.variant].resampling_diffusion),
        "profile_exposed": bool(DIMENSION_3[args.variant].include_profile),
        "prompt_context_sha256": isolation, "real_calls_made": 0,
    }
    if args.preflight_only:
        print(canonical_json(preflight))
        return 0

    if args.credentials is None:
        raise ValueError("--credentials is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    snapshot = load_snapshot(args.output_root / "checkpoints", args.variant, args.seed) if args.resume else None
    messages: dict[str, list[ContextItem]] = {agent_id: [] for agent_id in native_ids}
    start_tick = 1
    if snapshot:
        state, messages, start_tick = snapshot["state"], snapshot["messages"], int(snapshot["tick"]) + 1
    elif any(args.output_root.iterdir()):
        existing_relative = {
            str(path.relative_to(args.output_root)).replace("\\", "/")
            for path in args.output_root.rglob("*") if path.is_file()
        }
        safe_precheckpoint = all(
            item == "ticks/tick-0.states.jsonl" or item.startswith("provider_state/")
            for item in existing_relative
        )
        if not (args.resume and safe_precheckpoint):
            raise FileExistsError("existing output requires --resume and a valid checkpoint")
    cfg = read_json(args.provider_boundary)
    transport = OpenAIProviderTransport(api_key_env=cfg["api_key_environment_variable"], timeout_seconds=float(cfg["request_constraints"]["timeout_seconds"]), credentials_path=args.credentials)
    if args.workers > 1:
        transport.prepare_concurrency()
    environment = dict(os.environ)
    environment["AAMAS_REAL_MODEL_ENABLED"] = "YES_I_ACCEPT_API_COST"
    environment["AAMAS_FULL_EXPERIMENT_ENABLED"] = "YES_RUN_FROZEN_FULL_EXPERIMENT"
    environment[cfg["api_key_environment_variable"]] = "CONFIGURED_VIA_CREDENTIALS_JSON"
    provider = RealModelProvider(config_path=args.provider_boundary, auxiliary_boundary_path=args.auxiliary_boundary,
                                 storage_root=args.output_root / "provider_state", transport=transport,
                                 enable_real=True, budget_profile="full", environment=environment)
    states_path, actions_path = args.output_root / "population_states.jsonl", args.output_root / "actions.jsonl"
    tick_directory = args.output_root / "ticks"
    if not snapshot:
        write_jsonl_atomic(tick_directory / "tick-0.states.jsonl", state_rows(state, args.variant, 0, args.seed, native_ids, groups))
    features = _parent_features(population, protocol["scenario_id"])
    index = {agent_id: i for i, agent_id in enumerate(native_ids)}
    for tick in range(start_tick, 10):
        advance_particle_tick(population=population, graph=graph, state=state, features=features,
                              scenario_id=protocol["scenario_id"], branch="natural", seed=args.seed, tick=tick,
                              emit_messages=False, stochastic_diffusion=DIMENSION_3[args.variant].resampling_diffusion)
        means, normalized = state.ensemble.parent_means(), state.ensemble.normalized_within_parent()
        prepared = []
        for i, agent_id in enumerate(native_ids):
            visible = tuple(item for source in incoming[agent_id] for item in messages[source][-2:])[-12:]
            observation = make_observation(agent_id, tick, visible)
            adapter = CampusPulseAdapter(DIMENSION_3[args.variant])
            adapter.initialize(roles[agent_id].agent_seed(
                initial_state={field: float(means[i, offset]) for offset, field in enumerate(STATE_FIELDS)},
                initial_particles=[{field: float(row[offset]) for offset, field in enumerate(STATE_FIELDS)} for row in state.ensemble.values[i]],
                particle_weights=normalized[i].tolist(),
            ))
            bundle = renderer.render(observation, adapter.observe(observation))
            provider.preflight(method="campuspulse", bundle=bundle, response_schema=RESPONSE_SCHEMA, allowed_actions=ALLOWED_ACTIONS)
            prepared.append((i, agent_id, observation, bundle))
        def generate(item):
            i, aid, observation, bundle = item
            result = provider.generate(method="campuspulse", bundle=bundle, response_schema=RESPONSE_SCHEMA, allowed_actions=ALLOWED_ACTIONS)
            return i, aid, observation, bundle, result
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(generate, prepared))
        updates, action_rows = {}, []
        for i, agent_id, observation, bundle, result in sorted(results):
            decision = result.parsed_decision
            updates[i] = {"prompt_sha256": bundle.rendered_sha256,
                          "response_sha256": hashlib.sha256(canonical_json(decision).encode("utf-8")).hexdigest(),
                          "decision": project_shared_decision(decision)}
            action_rows.append({"variant": args.variant, "seed": args.seed, "tick": tick, "agent_id": agent_id,
                                "decision": decision, "prompt_sha256": bundle.rendered_sha256,
                                "request_id": result.request_id, "cache_key": result.cache_key, "cache_hit": result.cache_hit,
                                "usage": dict(result.usage)})
            if decision["public_or_private"] == "public" and decision.get("text"):
                messages[agent_id].append(ContextItem(f"{args.variant}-{tick}-{agent_id}", agent_id, str(decision["text"]), "public", tick))
        update = assimilate_forum_llm_updates_v2(state=state, updates=updates, scenario_id=protocol["scenario_id"], branch="natural", tick=tick, seed=args.seed)
        if state.ensemble.values.shape[1] == 1 and update["resampling_count"] != 0:
            raise ValueError("point-state condition unexpectedly resampled")
        write_jsonl_atomic(tick_directory / f"tick-{tick}.actions.jsonl", action_rows)
        if tick in MEASUREMENT_TICKS:
            write_jsonl_atomic(tick_directory / f"tick-{tick}.states.jsonl", state_rows(state, args.variant, tick, args.seed, native_ids, groups))
        save_snapshot(args.output_root / "checkpoints" / f"tick-{tick}.pkl.gz",
                      {"variant": args.variant, "seed": args.seed, "tick": tick, "state": state, "messages": messages})
    all_actions = [row for tick in range(1, 10) for row in read_jsonl(tick_directory / f"tick-{tick}.actions.jsonl")]
    all_states = [row for tick in MEASUREMENT_TICKS for row in read_jsonl(tick_directory / f"tick-{tick}.states.jsonl")]
    write_jsonl_atomic(actions_path, all_actions)
    write_jsonl_atomic(states_path, all_states)
    report = {**preflight, "status": "complete", "budget": provider.budget_snapshot(),
              "state_rows": len(read_jsonl(states_path)), "action_rows": len(read_jsonl(actions_path))}
    (args.output_root / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
