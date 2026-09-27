from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison_harness.adapters import GenerativeAgentsAdapter
from comparison_harness.adapters.generative_agents import MemoryNode
from comparison_harness.auxiliary import FrozenPromptRegistry
from comparison_harness.contracts import AgentSeed, CommonObservation, ContextItem, canonical_json
from comparison_harness.prompting import SharedPromptRenderer
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider
from comparison_harness.real_runtime import RealAuxiliaryRuntime, RealEvaluationRuntime, load_frozen_prompt


MEASUREMENT_TICKS = (0, 3, 9)
EXPECTED_AGENTS = 1000
EXPECTED_EDGES = 8000
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["read", "post", "reply", "quote", "react", "private_message", "no_action"]},
        "target_id": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
        "emotion": {"type": "string"},
        "public_stance": {"type": ["number", "null"], "minimum": -1, "maximum": 1},
        "public_or_private": {"type": "string", "enum": ["public", "private", "none"]},
    },
    "required": ["action", "target_id", "text", "emotion", "public_stance", "public_or_private"],
}
ALLOWED_ACTIONS = ("read", "post", "reply", "quote", "react", "private_message", "no_action")
SNAPSHOT_SCHEMA = "dimension3-ga-tick-state-v1.0.0"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    for number, row in enumerate(read_jsonl(path), 1):
        key = tuple(row[field] for field in fields)
        if key in keys:
            raise ValueError(f"duplicate checkpoint key at {path}:{number}: {key}")
        keys.add(key)
    return keys


def atomic_snapshot_write(path: Path, payload: dict[str, Any]) -> None:
    """Write a local restart snapshot atomically and publish a hash manifest last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = sha256(temporary)
    os.replace(temporary, path)
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "tick": int(payload["tick"]),
        "sha256": digest,
        "file": path.name,
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)


def load_latest_snapshot(directory: Path, *, seed: int) -> dict[str, Any] | None:
    """Load only a locally generated snapshot whose manifest and digest verify."""
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    if not directory.exists():
        return None
    for manifest_path in directory.glob("tick-*.state.pkl.manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
            continue
        tick = int(manifest["tick"])
        snapshot_path = directory / str(manifest["file"])
        if snapshot_path.exists() and sha256(snapshot_path) == manifest.get("sha256"):
            candidates.append((tick, snapshot_path, manifest))
    if not candidates:
        return None
    tick, snapshot_path, _ = max(candidates, key=lambda item: item[0])
    with gzip.open(snapshot_path, "rb") as handle:
        payload = pickle.load(handle)  # trusted local file, authenticated by adjacent SHA-256 manifest
    if (not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA
            or payload.get("seed") != seed or payload.get("tick") != tick):
        raise ValueError(f"invalid tick snapshot payload: {snapshot_path}")
    return payload


def observation(agent_id: str, tick: int, visible: tuple[ContextItem, ...], event_live: bool) -> CommonObservation:
    facts = {"closed_world_rule": "No allocation rules, rankings, deadlines, or appeal facts are known unless shown here."}
    if event_live:
        facts.update({
            "available_beds_before": 520,
            "available_beds_after": 320,
            "valid_applications": 536,
        })
    event = ({
        "event_id": "housing-520-320-applications-536-v1",
        "announcement": "宿舍床位由520个缩减至320个，目前有536份有效申请。",
    } if event_live else {"event_id": "pre-announcement", "announcement": "尚未发布本次资源分配公告。"})
    return CommonObservation(
        scenario_id="governance_legitimacy_dispute", tick=tick, agent_id=agent_id,
        factual_registry=facts, current_event=event, visible_context=visible,
        governance_artifacts=(), allowed_actions=ALLOWED_ACTIONS,
        response_schema=RESPONSE_SCHEMA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal Dimension-3 Generative Agents network evolution and probes")
    parser.add_argument("--initializations", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--stance-prompt", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-agents", type=int, help="labelled plumbing audit only; omit for the formal run")
    parser.add_argument("--preflight-only", action="store_true", help="validate frozen inputs and provider boundary without API calls")
    parser.add_argument("--resume", action="store_true", help="rebuild native state from verified cached calls and append only missing checkpoints")
    parser.add_argument("--workers", type=int, default=8, help="parallel agents within each synchronized tick (1-32)")
    parser.add_argument("--enable-real", action="store_true", required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be in [1,32]")

    formal = args.max_agents is None
    protocol = read_json(args.protocol)
    if protocol.get("status") != "frozen" or protocol.get("seed") != args.seed:
        raise ValueError("dimension-3 protocol is not the frozen requested seed")
    if protocol.get("measurement_ticks") != list(MEASUREMENT_TICKS):
        raise ValueError("measurement tick drift")
    if sha256(args.edges) != protocol.get("shared_network_sha256"):
        raise ValueError("network hash drift")

    init_rows = [row for row in read_jsonl(args.initializations) if row.get("method") == "generative_agents"]
    init = {str(row["agent_id"]): row for row in init_rows}
    if len(init) != EXPECTED_AGENTS:
        raise ValueError("Generative Agents initialization must contain exactly 1000 unique agents")
    all_edges = [(str(row["source_agent_id"]), str(row["target_agent_id"])) for row in read_jsonl(args.edges)]
    if len(all_edges) != EXPECTED_EDGES or len(set(all_edges)) != EXPECTED_EDGES:
        raise ValueError("frozen network must contain 8000 unique directed edges")
    if any(source not in init or target not in init or source == target for source, target in all_edges):
        raise ValueError("network endpoint mismatch")

    agent_ids = sorted(init)
    if args.max_agents is not None:
        if not 1 <= args.max_agents <= EXPECTED_AGENTS:
            raise ValueError("max-agents must be in [1,1000]")
        agent_ids = agent_ids[:args.max_agents]
    selected = set(agent_ids)
    edges = [(source, target) for source, target in all_edges if source in selected and target in selected]
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        incoming[target].append(source)
        outgoing[source].append(target)
    if formal and (len(edges) != EXPECTED_EDGES or any(len(outgoing[aid]) != 8 for aid in agent_ids)):
        raise ValueError("formal network must give every agent exactly eight outgoing neighbors")

    provider_cfg = read_json(args.provider_boundary)
    auxiliary_cfg = read_json(args.auxiliary_boundary)
    registry = FrozenPromptRegistry(args.prompt_root, auxiliary_cfg["prompt_sha256"])
    transport = OpenAIProviderTransport(
        api_key_env=provider_cfg["api_key_environment_variable"],
        timeout_seconds=float(provider_cfg["request_constraints"]["timeout_seconds"]),
        credentials_path=args.credentials,
    )
    if args.workers > 1:
        transport.prepare_concurrency()
    environment = dict(os.environ)
    environment["AAMAS_REAL_MODEL_ENABLED"] = "YES_I_ACCEPT_API_COST"
    environment["AAMAS_FULL_EXPERIMENT_ENABLED"] = "YES_RUN_FROZEN_FULL_EXPERIMENT"
    environment[provider_cfg["api_key_environment_variable"]] = "CONFIGURED_VIA_CREDENTIALS_JSON"
    provider = RealModelProvider(
        config_path=args.provider_boundary, auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.output_root / "provider_state", transport=transport,
        enable_real=True, budget_profile="full" if formal else "pilot", environment=environment,
    )
    if args.preflight_only:
        print(canonical_json({
            "status": "passed_no_api_calls", "formal": formal,
            "agent_count": len(agent_ids), "edge_count": len(edges),
            "action_slots": len(agent_ids) * 9,
            "probe_slots": len(agent_ids) * len(MEASUREMENT_TICKS),
            "initial_memory_items": len(agent_ids) * 2,
            "measurement_ticks": list(MEASUREMENT_TICKS),
            "provider_model": provider_cfg["response_calls"]["main_action"]["model"],
            "embedding_model": provider_cfg["embedding_call"]["model"],
            "workers": args.workers,
            "real_calls_made": 0,
        }))
        return 0
    auxiliary = RealAuxiliaryRuntime(provider, registry)
    stance_prompt = load_frozen_prompt(args.stance_prompt, provider_cfg["evaluation_prompt_sha256"]["stance_probe"])
    evaluator = RealEvaluationRuntime(provider, judge_prompt="unused", stance_prompt=stance_prompt)
    renderer = SharedPromptRenderer()

    args.output_root.mkdir(parents=True, exist_ok=True)
    actions_path = args.output_root / "actions.jsonl"
    states_path = args.output_root / "population_states.jsonl"
    trace_path = args.output_root / "tick_trace.jsonl"
    report_path = args.output_root / "run_report.json"
    snapshot_root = args.output_root / "checkpoints"
    if report_path.exists():
        raise FileExistsError("completed run report already exists")
    checkpoint_paths = (actions_path, states_path, trace_path)
    if any(path.exists() for path in checkpoint_paths) and not args.resume:
        raise FileExistsError("partial output exists; pass --resume")
    completed_actions = existing_keys(actions_path, ("tick", "agent_id")) if args.resume else set()
    completed_states = existing_keys(states_path, ("tick", "agent_id")) if args.resume else set()
    completed_trace = existing_keys(trace_path, ("tick",)) if args.resume else set()
    latest_completed_tick = max((int(key[0]) for key in completed_trace), default=0)
    restored = load_latest_snapshot(snapshot_root, seed=args.seed) if args.resume else None
    if restored is not None:
        restored_tick = int(restored["tick"])
        if restored_tick > latest_completed_tick:
            raise ValueError("tick snapshot is ahead of durable JSONL checkpoints")
        expected_action_keys = {(tick, aid) for tick in range(1, restored_tick + 1) for aid in agent_ids}
        if not expected_action_keys.issubset(completed_actions):
            raise ValueError("tick snapshot has missing durable action rows")

    embedding_memo: dict[str, tuple[float, ...]] = {}
    importance_memo: dict[str, float] = {}
    memo_guard = threading.Lock()
    embedding_locks: dict[str, threading.Lock] = {}
    importance_locks: dict[str, threading.Lock] = {}
    def keyed_lock(table: dict[str, threading.Lock], text: str) -> threading.Lock:
        with memo_guard:
            return table.setdefault(text, threading.Lock())
    def embed(text: str) -> tuple[float, ...]:
        with keyed_lock(embedding_locks, text):
            if text not in embedding_memo:
                embedding_memo[text] = auxiliary.embedding(text)
            return embedding_memo[text]
    def importance(text: str) -> float:
        with keyed_lock(importance_locks, text):
            if text not in importance_memo:
                importance_memo[text] = auxiliary.importance(text)
            return importance_memo[text]

    adapters: dict[str, GenerativeAgentsAdapter] = {}
    def initialize_agent(aid: str) -> tuple[str, GenerativeAgentsAdapter]:
        row = init[aid]
        native = row["native_initialization"]
        adapter = GenerativeAgentsAdapter(embedder=embed, importance_scorer=importance, reflector=auxiliary.reflect)
        adapter.initialize(AgentSeed(aid, None, row["interest_group"], None, None))
        for item in native["initial_memory_stream"]:
            adapter._add_memory(canonical_json(item["content"]), 0, kind=str(item["kind"]), importance=float(item["importance"]))
        return aid, adapter
    previous_public: dict[str, dict[str, Any]] = {}
    state_rows: list[dict[str, Any]] = read_jsonl(states_path) if states_path.exists() else []
    action_count = 0
    delivery_count = 0
    start_tick = 0
    if restored is None:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for aid, adapter in pool.map(initialize_agent, agent_ids):
                adapters[aid] = adapter
    else:
        start_tick = int(restored["tick"])
        previous_public = dict(restored["previous_public"])
        delivery_count = int(restored["delivery_count"])
        action_count = int(restored["action_count"])
        embedding_memo.update({str(key): tuple(value) for key, value in restored["embedding_memo"].items()})
        importance_memo.update({str(key): float(value) for key, value in restored["importance_memo"].items()})
        adapter_states = restored["adapters"]
        if set(adapter_states) != set(agent_ids):
            raise ValueError("tick snapshot agent set mismatch")
        for aid in agent_ids:
            row = init[aid]
            adapter = GenerativeAgentsAdapter(embedder=embed, importance_scorer=importance, reflector=auxiliary.reflect)
            adapter.initialize(AgentSeed(aid, None, row["interest_group"], None, None))
            saved = adapter_states[aid]
            adapter.memories = [MemoryNode(
                node_id=str(node["node_id"]), text=str(node["text"]),
                created_tick=int(node["created_tick"]), last_accessed_tick=int(node["last_accessed_tick"]),
                importance=float(node["importance"]), embedding=embedding_memo[str(node["text"])],
                kind=str(node["kind"]),
            ) for node in saved["memories"]]
            adapter.importance_since_reflection = float(saved["importance_since_reflection"])
            adapter.auxiliary_log = list(saved["auxiliary_log"])
            adapters[aid] = adapter

    def save_tick_snapshot(tick: int) -> None:
        adapter_states: dict[str, Any] = {}
        for aid in agent_ids:
            adapter = adapters[aid]
            adapter_states[aid] = {
                "importance_since_reflection": adapter.importance_since_reflection,
                "auxiliary_log": list(adapter.auxiliary_log),
                "memories": [{
                    "node_id": node.node_id, "text": node.text,
                    "created_tick": node.created_tick, "last_accessed_tick": node.last_accessed_tick,
                    "importance": node.importance, "kind": node.kind,
                } for node in adapter.memories],
            }
        payload = {
            "schema_version": SNAPSHOT_SCHEMA, "seed": args.seed, "tick": tick,
            "previous_public": previous_public, "delivery_count": delivery_count,
            "action_count": action_count, "embedding_memo": embedding_memo,
            "importance_memo": importance_memo, "adapters": adapter_states,
        }
        atomic_snapshot_write(snapshot_root / f"tick-{tick}.state.pkl.gz", payload)

    def probe_tick(tick: int) -> None:
        def probe_agent(aid: str) -> dict[str, Any]:
            adapter = adapters[aid]
            probe_input = {
                "tick": tick,
                "memory_stream": [{"kind": node.kind, "text": node.text, "importance": node.importance} for node in adapter.memories],
                "visible_forum_context": [previous_public[source] for source in incoming.get(aid, ()) if source in previous_public],
            }
            stance = evaluator.stance_probe(probe_input)
            return {
                "schema_version": "dimension3-population-state-v1.0.0", "method": "generative_agents",
                "variant": "full", "seed": args.seed, "tick": tick, "agent_id": aid,
                "interest_group": init[aid]["interest_group"], "public_stance": stance,
                "native_probe_five_level": int(round(stance * 2)), "memory_count": len(adapter.memories),
            }
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(probe_agent, agent_ids))
        for row in rows:
            aid = str(row["agent_id"])
            if (tick, aid) not in completed_states:
                state_rows.append(row)
                append_jsonl(states_path, row)
                completed_states.add((tick, aid))

    if start_tick == 0 and not completed_states:
        probe_tick(0)
    for tick in range(start_tick + 1, 10):
        current_public: dict[str, dict[str, Any]] = {}
        previous_snapshot = dict(previous_public)
        def process_agent(aid: str) -> tuple[str, dict[str, Any], dict[str, Any] | None, int]:
            visible_items = []
            for source in sorted(incoming.get(aid, ())):
                message = previous_snapshot.get(source)
                if message is not None:
                    visible_items.append(ContextItem(
                        message_id=str(message["message_id"]), author_id=source,
                        text=str(message["text"]), visibility="public", tick=tick - 1,
                        thread_id="housing-allocation", parent_id=None,
                    ))
            if tick == 3:
                visible_items.append(ContextItem(
                    message_id="event:housing-520-320-applications-536-v1", author_id=None,
                    text="宿舍床位由520个缩减至320个，目前有536份有效申请。",
                    visibility="system", tick=3, thread_id="housing-allocation", parent_id=None,
                ))
            obs = observation(aid, tick, tuple(visible_items), tick >= 3)
            snapshot = adapters[aid].observe(obs)
            bundle = renderer.render(obs, snapshot)
            result = provider.generate(method="generative_agents", bundle=bundle,
                                       response_schema=RESPONSE_SCHEMA, allowed_actions=ALLOWED_ACTIONS)
            decision = dict(result.parsed_decision)
            adapters[aid].commit_decision(obs, decision)
            row = {
                "schema_version": "dimension3-ga-action-v1.0.0", "seed": args.seed,
                "tick": tick, "agent_id": aid, "interest_group": init[aid]["interest_group"],
                "observation_sha256": obs.observation_sha256, "prompt_sha256": bundle.rendered_sha256,
                "decision": decision, "request_id": result.request_id, "cache_key": result.cache_key,
                "cache_hit": result.cache_hit, "raw_archive_path": result.raw_archive_path,
            }
            public_message = None
            text = str(decision.get("text") or "").strip()
            if text and decision.get("public_or_private") == "public":
                public_message = {"message_id": f"ga:t{tick}:{aid}", "text": text}
            return aid, row, public_message, len(visible_items) - (1 if tick == 3 else 0)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tick_results = list(pool.map(process_agent, agent_ids))
        for aid, row, public_message, delivered in tick_results:
            if (tick, aid) not in completed_actions:
                append_jsonl(actions_path, row)
                completed_actions.add((tick, aid))
            action_count += 1
            delivery_count += delivered
            if public_message is not None:
                current_public[aid] = public_message
        previous_public = current_public
        trace_row = {
            "tick": tick, "agent_count": len(agent_ids), "action_count": len(agent_ids),
            "public_message_count": len(previous_public), "network_delivery_count": delivery_count,
        }
        if (tick,) not in completed_trace:
            append_jsonl(trace_path, trace_row)
            completed_trace.add((tick,))
        if tick in MEASUREMENT_TICKS:
            probe_tick(tick)
        # On the first snapshot-enabled resume, replay cached calls only as far
        # as the last durable tick before publishing a checkpoint. Thereafter,
        # every newly completed tick receives an exact native-state snapshot.
        if restored is not None or tick >= latest_completed_tick:
            save_tick_snapshot(tick)

    expected_states = len(agent_ids) * len(MEASUREMENT_TICKS)
    expected_actions = len(agent_ids) * 9
    if (len(state_rows) != expected_states or action_count != expected_actions
            or len(completed_states) != expected_states or len(completed_actions) != expected_actions
            or len(completed_trace) != 9):
        raise ValueError("formal Generative Agents output is incomplete")
    report = {
        "schema_version": "dimension3-generative-agents-formal-run-v1.0.0",
        "status": "complete", "scientific_result": formal, "seed": args.seed,
        "agent_count": len(agent_ids), "state_rows": len(state_rows), "action_rows": action_count,
        "network_deliveries": delivery_count, "unique_embedding_texts": len(embedding_memo),
        "unique_importance_texts": len(importance_memo), "logical_calls": provider.logical_calls,
        "workers": args.workers,
        "transport_attempts": provider.transport_attempts, "input_characters": provider.input_chars,
        "output_sha256": {"states": sha256(states_path), "actions": sha256(actions_path), "trace": sha256(trace_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
