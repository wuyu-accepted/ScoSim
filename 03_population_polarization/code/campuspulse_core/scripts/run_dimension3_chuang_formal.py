from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import msvcrt
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison_harness.adapters import ChuangOpinionDynamicsAdapter
from comparison_harness.auxiliary import FrozenPromptRegistry
from comparison_harness.contracts import AgentSeed, CommonObservation, ContextItem, canonical_json
from comparison_harness.prompting import SharedPromptRenderer
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider


METHOD = "chuang_opinion_dynamics"
MEASUREMENT_TICKS = (0, 3, 9)
EXPECTED_AGENTS = 1000
EXPECTED_EDGES = 8000
NATIVE_OPINIONS = (-2, -1, 0, 1, 2)
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
SNAPSHOT_SCHEMA = "dimension3-chuang-tick-state-v1.0.0"


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


def save_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = sha256(temporary)
    os.replace(temporary, path)
    manifest = {"schema_version": SNAPSHOT_SCHEMA, "tick": payload["tick"], "file": path.name, "sha256": digest}
    manifest_path = path.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)


def load_latest_snapshot(directory: Path, seed: int) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    if directory.exists():
        for manifest_path in directory.glob("tick-*.state.pkl.manifest.json"):
            manifest = read_json(manifest_path)
            snapshot_path = directory / str(manifest.get("file"))
            if (manifest.get("schema_version") == SNAPSHOT_SCHEMA and snapshot_path.exists()
                    and sha256(snapshot_path) == manifest.get("sha256")):
                candidates.append((int(manifest["tick"]), snapshot_path))
    if not candidates:
        return None
    tick, path = max(candidates, key=lambda item: item[0])
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)  # trusted local file authenticated by the adjacent digest
    if (not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA
            or payload.get("seed") != seed or payload.get("tick") != tick):
        raise ValueError(f"invalid snapshot payload: {path}")
    return payload


def observation(agent_id: str, tick: int, visible: tuple[ContextItem, ...], event_live: bool) -> CommonObservation:
    facts: dict[str, Any] = {
        "closed_world_rule": "No allocation rules, rankings, deadlines, or appeal facts are known unless shown here."
    }
    if event_live:
        facts.update({"available_beds_before": 520, "available_beds_after": 320, "valid_applications": 536})
    event = ({
        "event_id": "housing-520-320-applications-536-v1",
        "announcement": "宿舍床位由520个缩减至320个，目前有536份有效申请。",
    } if event_live else {"event_id": "pre-announcement", "announcement": "尚未发布本次资源分配公告。"})
    return CommonObservation(
        scenario_id="governance_legitimacy_dispute", tick=tick, agent_id=agent_id,
        factual_registry=facts, current_event=event, visible_context=visible,
        governance_artifacts=(), allowed_actions=ALLOWED_ACTIONS, response_schema=RESPONSE_SCHEMA,
    )


def acquire_run_lock(output_root: Path):
    """Hold a Windows byte-range lock so two paid runners cannot share outputs."""
    output_root.mkdir(parents=True, exist_ok=True)
    handle = (output_root / "formal_run.lock").open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError("formal run lock is held by another Chuang process") from exc
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal Dimension-3 Chuang et al. network evolution")
    parser.add_argument("--initializations", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics-protocol", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--enable-real", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be in [1,32]")
    if not args.preflight_only and not args.enable_real:
        raise ValueError("real execution requires --enable-real")

    protocol = read_json(args.protocol)
    metrics_protocol = read_json(args.metrics_protocol)
    if protocol.get("status") != "frozen" or protocol.get("seed") != args.seed:
        raise ValueError("dimension-3 protocol is not frozen for the requested seed")
    if protocol.get("measurement_ticks") != list(MEASUREMENT_TICKS):
        raise ValueError("measurement tick drift")
    if metrics_protocol.get("status") != "frozen" or metrics_protocol.get("schema_version") != "locked-dimension3-cross-method-metrics-v1.5.0":
        raise ValueError("dimension-3 metric definitions are not frozen at v1.5.0")
    if metrics_protocol.get("entropy", {}).get("bin_count") != 20:
        raise ValueError("dimension-3 entropy must use the frozen 20 bins")
    if sha256(args.edges) != protocol.get("shared_network_sha256"):
        raise ValueError("network hash drift")

    init_rows = [row for row in read_jsonl(args.initializations) if row.get("method") == METHOD]
    init = {str(row["agent_id"]): row for row in init_rows}
    if len(init_rows) != EXPECTED_AGENTS or len(init) != EXPECTED_AGENTS:
        raise ValueError("Chuang initialization must contain exactly 1000 unique agents")
    agent_ids = sorted(init)
    for aid in agent_ids:
        native = init[aid]["native_initialization"]
        if native.get("initial_opinion_five_level") not in NATIVE_OPINIONS:
            raise ValueError(f"invalid native initial opinion for {aid}")
        if native.get("allowed_opinions") != list(NATIVE_OPINIONS):
            raise ValueError(f"native opinion grid drift for {aid}")
        if len(native.get("closed_world_history", [])) != 2:
            raise ValueError(f"incomplete fair closed-world initialization for {aid}")

    all_edges = [(str(row["source_agent_id"]), str(row["target_agent_id"])) for row in read_jsonl(args.edges)]
    if len(all_edges) != EXPECTED_EDGES or len(set(all_edges)) != EXPECTED_EDGES:
        raise ValueError("frozen network must contain 8000 unique directed edges")
    if any(source not in init or target not in init or source == target for source, target in all_edges):
        raise ValueError("network endpoint mismatch")
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in all_edges:
        incoming[target].append(source)
        outgoing[source].append(target)
    if any(len(outgoing[aid]) != 8 for aid in agent_ids):
        raise ValueError("formal network must give every agent exactly eight outgoing neighbors")

    provider_cfg = read_json(args.provider_boundary)
    auxiliary_cfg = read_json(args.auxiliary_boundary)
    registry = FrozenPromptRegistry(args.prompt_root, auxiliary_cfg["prompt_sha256"])
    update_instruction = registry["chuang_confirmation_update"].text.strip()
    if auxiliary_cfg["methods"][METHOD]["allowed_auxiliary_calls"] != []:
        raise ValueError("Chuang formal run must not make auxiliary model calls")
    if auxiliary_cfg["methods"][METHOD]["extra_update_call_forbidden"] is not True:
        raise ValueError("Chuang extra update call boundary drift")

    initial_counts = {str(value): sum(
        init[aid]["native_initialization"]["initial_opinion_five_level"] == value for aid in agent_ids
    ) for value in NATIVE_OPINIONS}
    if args.preflight_only:
        print(canonical_json({
            "status": "passed_no_api_calls", "scientific_result": True, "seed": args.seed,
            "agent_count": len(agent_ids), "edge_count": len(all_edges), "ticks": list(range(10)),
            "measurement_ticks": list(MEASUREMENT_TICKS), "action_slots": 9000,
            "state_slots": 3000, "auxiliary_call_slots": 0, "main_action_call_slots": 9000,
            "initial_native_opinion_counts": initial_counts, "workers": args.workers,
            "metric_definitions": "1.5.0", "entropy_bins": 20, "real_calls_made": 0,
        }))
        return 0

    assert args.credentials is not None
    # Keep this handle alive for the whole process. The OS releases it even
    # after an abnormal exit, allowing the supervisor to resume safely.
    run_lock_handle = acquire_run_lock(args.output_root)
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
        enable_real=True, budget_profile="full", environment=environment,
    )
    renderer = SharedPromptRenderer()

    args.output_root.mkdir(parents=True, exist_ok=True)
    actions_path = args.output_root / "actions.jsonl"
    states_path = args.output_root / "population_states.jsonl"
    trace_path = args.output_root / "tick_trace.jsonl"
    report_path = args.output_root / "run_report.json"
    snapshot_root = args.output_root / "checkpoints"
    if report_path.exists():
        raise FileExistsError("completed run report already exists")
    if any(path.exists() for path in (actions_path, states_path, trace_path)) and not args.resume:
        raise FileExistsError("partial output exists; pass --resume")
    completed_actions = existing_keys(actions_path, ("tick", "agent_id")) if args.resume else set()
    completed_states = existing_keys(states_path, ("tick", "agent_id")) if args.resume else set()
    completed_trace = existing_keys(trace_path, ("tick",)) if args.resume else set()
    restored = load_latest_snapshot(snapshot_root, args.seed) if args.resume else None

    adapters: dict[str, ChuangOpinionDynamicsAdapter] = {}
    previous_public: dict[str, dict[str, Any]] = {}
    action_count = 0
    delivery_count = 0
    start_tick = 0
    if restored is None:
        for aid in agent_ids:
            row = init[aid]
            native = row["native_initialization"]
            adapter = ChuangOpinionDynamicsAdapter(
                initial_opinion=int(native["initial_opinion_five_level"]), update_instruction=update_instruction
            )
            adapter.initialize(AgentSeed(aid, None, row["interest_group"], None, None))
            for item in native["closed_world_history"]:
                adapter.history.append({
                    "tick": 0, "kind": str(item["kind"]), "text": canonical_json(item["content"]),
                    "closed_world": True, "source": "frozen_fair_initialization",
                })
            adapters[aid] = adapter
    else:
        start_tick = int(restored["tick"])
        if any((tick, aid) not in completed_actions for tick in range(1, start_tick + 1) for aid in agent_ids):
            raise ValueError("snapshot is ahead of durable action checkpoints")
        previous_public = dict(restored["previous_public"])
        action_count = int(restored["action_count"])
        delivery_count = int(restored["delivery_count"])
        if set(restored["adapters"]) != set(agent_ids):
            raise ValueError("snapshot agent set mismatch")
        for aid in agent_ids:
            saved = restored["adapters"][aid]
            adapter = ChuangOpinionDynamicsAdapter(initial_opinion=int(saved["current_opinion"]), update_instruction=update_instruction)
            adapter.initialize(AgentSeed(aid, None, init[aid]["interest_group"], None, None))
            adapter.history = list(saved["history"])
            adapters[aid] = adapter

    def write_population_state(tick: int) -> None:
        for aid in agent_ids:
            key = (tick, aid)
            if key in completed_states:
                continue
            opinion = adapters[aid].current_opinion
            if opinion not in NATIVE_OPINIONS:
                raise ValueError("Chuang native opinion left the registered five-level grid")
            row = {
                "schema_version": "dimension3-population-state-v1.0.0", "method": METHOD,
                "variant": "full", "seed": args.seed, "tick": tick, "agent_id": aid,
                "interest_group": init[aid]["interest_group"], "public_stance": opinion / 2.0,
                "native_opinion_five_level": opinion, "history_length": len(adapters[aid].history),
            }
            append_jsonl(states_path, row)
            completed_states.add(key)

    def write_tick_snapshot(tick: int) -> None:
        payload = {
            "schema_version": SNAPSHOT_SCHEMA, "seed": args.seed, "tick": tick,
            "previous_public": previous_public, "action_count": action_count,
            "delivery_count": delivery_count,
            "adapters": {aid: {
                "current_opinion": adapters[aid].current_opinion,
                "history": adapters[aid].history,
            } for aid in agent_ids},
        }
        save_snapshot(snapshot_root / f"tick-{tick}.state.pkl.gz", payload)

    if start_tick == 0 and not completed_states:
        write_population_state(0)
    for tick in range(start_tick + 1, 10):
        previous_snapshot = dict(previous_public)

        def process_agent(aid: str) -> tuple[str, dict[str, Any], dict[str, Any] | None, int]:
            visible: list[ContextItem] = []
            for source in sorted(incoming[aid]):
                message = previous_snapshot.get(source)
                if message is not None:
                    visible.append(ContextItem(
                        message_id=str(message["message_id"]), author_id=source, text=str(message["text"]),
                        visibility="public", tick=tick - 1, thread_id="housing-allocation",
                    ))
            network_deliveries = len(visible)
            if tick == 3:
                visible.append(ContextItem(
                    message_id="event:housing-520-320-applications-536-v1", author_id=None,
                    text="宿舍床位由520个缩减至320个，目前有536份有效申请。",
                    visibility="system", tick=3, thread_id="housing-allocation",
                ))
            obs = observation(aid, tick, tuple(visible), tick >= 3)
            pre_opinion = adapters[aid].current_opinion
            bundle = renderer.render(obs, adapters[aid].observe(obs))
            result = provider.generate(
                method=METHOD, bundle=bundle, response_schema=RESPONSE_SCHEMA, allowed_actions=ALLOWED_ACTIONS
            )
            decision = dict(result.parsed_decision)
            raw_stance = decision.get("public_stance")
            adapters[aid].commit_decision(obs, decision)
            post_opinion = adapters[aid].current_opinion
            row = {
                "schema_version": "dimension3-chuang-action-v1.0.0", "seed": args.seed,
                "tick": tick, "agent_id": aid, "interest_group": init[aid]["interest_group"],
                "observation_sha256": obs.observation_sha256, "prompt_sha256": bundle.rendered_sha256,
                "decision": decision, "pre_opinion_five_level": pre_opinion,
                "raw_model_public_stance": raw_stance, "post_opinion_five_level": post_opinion,
                "normalized_post_opinion": post_opinion / 2.0,
                "request_id": result.request_id, "cache_key": result.cache_key,
                "cache_hit": result.cache_hit, "raw_archive_path": result.raw_archive_path,
            }
            text = str(decision.get("text") or "").strip()
            public_message = None
            if text and decision.get("public_or_private") == "public":
                public_message = {"message_id": f"chuang:t{tick}:{aid}", "text": text}
            return aid, row, public_message, network_deliveries

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(process_agent, agent_ids))
        current_public: dict[str, dict[str, Any]] = {}
        for aid, row, public_message, delivered in results:
            if (tick, aid) not in completed_actions:
                append_jsonl(actions_path, row)
                completed_actions.add((tick, aid))
            action_count += 1
            delivery_count += delivered
            if public_message is not None:
                current_public[aid] = public_message
        previous_public = current_public
        if (tick,) not in completed_trace:
            append_jsonl(trace_path, {
                "tick": tick, "agent_count": EXPECTED_AGENTS, "action_count": EXPECTED_AGENTS,
                "public_message_count": len(previous_public), "network_delivery_count": delivery_count,
            })
            completed_trace.add((tick,))
        if tick in MEASUREMENT_TICKS:
            write_population_state(tick)
        write_tick_snapshot(tick)

    if len(completed_actions) != 9000 or len(completed_states) != 3000 or len(completed_trace) != 9:
        raise ValueError("formal Chuang output is incomplete")
    state_rows = read_jsonl(states_path)
    if any(row["native_opinion_five_level"] not in NATIVE_OPINIONS
           or row["public_stance"] != row["native_opinion_five_level"] / 2.0 for row in state_rows):
        raise ValueError("formal state output violates native five-level mapping")
    report = {
        "schema_version": "dimension3-chuang-formal-run-v1.0.0", "status": "complete",
        "scientific_result": True, "seed": args.seed, "agent_count": EXPECTED_AGENTS,
        "action_rows": len(completed_actions), "state_rows": len(completed_states),
        "network_deliveries": delivery_count, "workers": args.workers,
        "logical_calls": provider.logical_calls, "transport_attempts": provider.transport_attempts,
        "auxiliary_calls": 0, "native_opinion_domain": list(NATIVE_OPINIONS),
        "output_sha256": {"actions": sha256(actions_path), "states": sha256(states_path), "trace": sha256(trace_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(canonical_json(report))
    run_lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
