from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from comparison_harness.fake_executor import load_observations
from comparison_harness.fair_initialization import load_fair_role_initializations
from evaluation.common import read_jsonl
from evaluation.dimension2 import case_id
from evaluation.text_judge import build_case, load_frozen_prompt


CONDITIONS = ("context_only", "profile_only", "state_only", "full")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2048 condition-blind D2 longitudinal ablation judge cases.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--full-records", type=Path, required=True)
    parser.add_argument("--counterfactual-records", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    observations = {item.observation_id: item.observation for item in load_observations(args.observations)}
    full_rows = read_jsonl(args.full_records)
    counter_rows = read_jsonl(args.counterfactual_records)
    if len(observations) != 512 or len(full_rows) != 512 or len(counter_rows) != 1536:
        raise ValueError("judge input must contain 512 observations, 512 full records and 1536 counterfactual records")
    roles = load_fair_role_initializations(args.population)
    prompt = load_frozen_prompt(args.prompt)
    normalized: list[dict[str, Any]] = []
    for row in full_rows:
        item = dict(row)
        base_id = str(item["source_record_id"])
        item["variant"] = "full"
        item["source_record_id"] = base_id + ":ablation:full"
        normalized.append(item)
    normalized.extend(dict(row) for row in counter_rows)
    by_condition = {condition: [] for condition in CONDITIONS}
    for row in normalized:
        condition = str(row.get("variant"))
        if condition not in by_condition:
            raise ValueError(f"unexpected condition: {condition}")
        by_condition[condition].append(row)
    if any(len(rows) != 512 for rows in by_condition.values()):
        raise ValueError({condition: len(rows) for condition, rows in by_condition.items()})
    cases: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for row in sorted(by_condition[condition], key=lambda value: str(value["source_record_id"])):
            source_id = str(row["source_record_id"])
            base_id = source_id.split(":ablation:", 1)[0]
            if base_id not in observations:
                raise ValueError(f"record is not paired to frozen observation: {source_id}")
            observation = observations[base_id]
            if row.get("agent_id") != observation.agent_id or int(row.get("tick")) != observation.tick:
                raise ValueError(f"record identity drift: {source_id}")
            role = roles[observation.agent_id]
            evidence = []
            if condition in {"profile_only", "full"}:
                evidence = [{"source_id": "role-card-" + role.role_card_sha256[:24],
                             "source_type": "shared_generation_input", "content": role.role_card}]
            item = build_case(row, prompt, observation.factual_registry, observation.current_event, evidence)
            cases.append(item)
            evidence_rows.append({"case_id": item["case_id"], "evidence": evidence})
            source_rows.append({
                "case_id": item["case_id"], "source_record_id": source_id,
                "condition": condition, "observation_sha256": observation.observation_sha256,
                "fact_registry_sha256": sha(observation.factual_registry),
                "current_event_sha256": sha(observation.current_event),
                "method_evidence_sha256": sha(evidence),
            })
    ids = [str(item["case_id"]) for item in cases]
    if len(ids) != 2048 or len(set(ids)) != 2048:
        raise ValueError("longitudinal judge case IDs are not complete and unique")
    if any(not item.get("blind_fields_absent") for item in cases):
        raise ValueError("judge blindness assertion failed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("records.2048.jsonl", normalized), ("judge_cases.jsonl", cases),
                       ("method_evidence.jsonl", evidence_rows), ("judge_source_audit.jsonl", source_rows)):
        (args.output_root / name).write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": "dimension2-longitudinal-judge-case-manifest-v1.0.0",
        "evaluation_definitions_version": "1.6.0", "case_count": 2048,
        "cases_per_condition": 512, "condition_count": 4,
        "conditions": list(CONDITIONS), "paired_observation_count": 512,
        "method_name_exposed_to_judge": False, "variant_exposed_to_judge": False,
        "agent_id_exposed_to_judge": False, "predecision_state_exposed_to_judge": False,
        "network_calls_made": 0,
    }
    (args.output_root / "judge_case_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
