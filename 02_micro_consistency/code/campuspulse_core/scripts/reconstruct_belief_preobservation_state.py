from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from comparison_harness.adapters import BeliefEngineAdapter
from comparison_harness.auxiliary import FrozenPromptRegistry
from comparison_harness.fake_executor import load_observations
from comparison_harness.fair_initialization import load_fair_role_initializations
from comparison_harness.real_provider import RealModelProvider
from comparison_harness.real_runtime import RealAuxiliaryRuntime
from evaluation.dimension2 import case_id


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CacheOnlyProvider(RealModelProvider):
    """Read verified provider caches without permitting a network fallback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cache_events: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _ensure_enabled(self) -> None:
        # Cache reads are allowed.  A miss reaches the provider's transport
        # assertion because transport is deliberately None.
        return None

    def _audit(self, event: Mapping[str, Any]) -> None:
        self.cache_events.append(dict(event))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Belief Engine state immediately before the current "
            "observation, using only frozen records and verified provider caches."
        )
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--provider-cache-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--original-sidecar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    observations = load_observations(args.observations)
    records = read_jsonl(args.records)
    original = {
        str(row["source_record_id"]): row
        for row in read_jsonl(args.original_sidecar)
    }
    if len(observations) != 512 or len(records) != 512 or len(original) != 512:
        raise ValueError("time-aligned reconstruction requires the complete 512-case panel")
    by_source = {str(row["source_record_id"]): row for row in records}
    if len(by_source) != 512:
        raise ValueError("Belief Engine records contain duplicate source_record_id values")

    roles = load_fair_role_initializations(args.population)
    boundary = json.loads(args.auxiliary_boundary.read_text(encoding="utf-8"))
    registry = FrozenPromptRegistry(args.prompt_root, boundary["prompt_sha256"])
    provider = CacheOnlyProvider(
        config_path=args.provider_boundary,
        auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.provider_cache_root,
        transport=None,
        enable_real=False,
        budget_profile="full",
        environment={},
    )
    runtime = RealAuxiliaryRuntime(provider, registry)
    adapters: dict[str, BeliefEngineAdapter] = {}
    outputs: list[dict[str, Any]] = []
    max_replay_error = 0.0

    for envelope in observations:
        observation = envelope.observation
        source_id = str(envelope.observation_id)
        record = by_source.get(source_id)
        expected = original.get(source_id)
        if record is None or expected is None:
            raise ValueError(f"missing frozen evidence for {source_id}")
        if str(record["agent_id"]) != observation.agent_id or int(record["tick"]) != observation.tick:
            raise ValueError(f"record/observation identity mismatch for {source_id}")

        adapter = adapters.get(observation.agent_id)
        if adapter is None:
            role = roles[observation.agent_id]
            adapter = BeliefEngineAdapter(
                seed_arguments=role.belief_arguments(),
                extractor=runtime.extract,
                embedder=runtime.embedding,
            )
            adapter.initialize(role.agent_seed())
            adapters[observation.agent_id] = adapter

        before_log_odds = float(adapter.log_odds)
        before_stance = float(adapter.stance)
        before_active = sum(row.active for row in adapter.records)

        adapter.observe(observation)
        after_observation_log_odds = float(adapter.log_odds)
        after_observation_stance = float(adapter.stance)
        error = abs(after_observation_stance - float(expected["standardized_pre_state"]))
        max_replay_error = max(max_replay_error, error)
        if not math.isclose(
            after_observation_stance,
            float(expected["standardized_pre_state"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"cache replay diverged from original after-observation state for {source_id}: {error}"
            )

        outputs.append({
            "schema_version": "belief-engine-preobservation-state-v1.0.0",
            "case_id": case_id(record),
            "source_record_id": source_id,
            "agent_id": observation.agent_id,
            "tick": observation.tick,
            "log_odds": before_log_odds,
            "standardized_pre_state": before_stance,
            "active_argument_count": before_active,
            "capture_point": "before_current_observation_after_previous_decision",
            "original_after_observation_stance": after_observation_stance,
            "original_after_observation_log_odds": after_observation_log_odds,
        })

        adapter.commit_decision(observation, {
            "text": record.get("generated_text"),
            "action": record.get("action"),
            "emotion": record.get("emotion"),
            "public_stance": record.get("public_stance"),
            "public_or_private": record.get("public_or_private"),
        })

    if any(event.get("event") != "cache_hit" for event in provider.cache_events):
        raise ValueError("cache-only replay observed a non-cache provider event")

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "belief_engine_preobservation_state.jsonl"
    output_path.write_text(
        "".join(canonical(row) + "\n" for row in outputs),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "belief-engine-time-alignment-replay-v1.0.0",
        "record_count": len(outputs),
        "agent_count": len(adapters),
        "network_calls": 0,
        "cache_hit_count": len(provider.cache_events),
        "max_after_observation_replay_error": max_replay_error,
        "capture_point": "before_current_observation_after_previous_decision",
        "source_records_sha256": sha256(args.records),
        "source_original_sidecar_sha256": sha256(args.original_sidecar),
        "output_sha256": sha256(output_path),
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
