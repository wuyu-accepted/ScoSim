from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from comparison_harness.offline_trace import FrozenCommonTraceConnector
from evaluation.common import read_jsonl
from evaluation.dimension2 import case_id
from evaluation.text_judge import build_case, load_frozen_prompt
from scripts.run_real_campuspulse_dimension2_ablation import _observation, _profile_for_prompt


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build method-blind judge cases for the CampusPulse D2 panel.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.records)
    if len(records) != 48:
        raise ValueError(f"formal panel requires 48 records, observed {len(records)}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    selected = {str(row["agent_id"]): int(row["agent_index"]) for row in panel["agents"]}
    connector = FrozenCommonTraceConnector(args.scenario_root)
    connector.load_and_verify()
    prompt = load_frozen_prompt(args.prompt)

    cases: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for row in records:
        agent_id = str(row["agent_id"])
        index = selected.get(agent_id)
        if index is None or str(connector.profiles[index]["agent_id"]) != agent_id:
            raise ValueError(f"record is outside frozen panel: {agent_id}")
        observation = _observation(connector, index, agent_id)
        profile_enabled = str(row["variant"]) in {"profile_only", "full"}
        evidence = (
            [{
                "source_id": "synthetic_stable_profile",
                "source_type": "generation_input",
                "content": _profile_for_prompt(connector.profiles[index]),
            }]
            if profile_enabled else []
        )
        identifier = case_id(row)
        cases.append(build_case(
            row,
            prompt,
            observation.factual_registry,
            observation.current_event,
            evidence,
        ))
        evidence_rows.append({"case_id": identifier, "evidence": evidence})
        source_rows.append({
            "case_id": identifier,
            "source_record_id": row.get("source_record_id"),
            "observation_sha256": observation.observation_sha256,
            "fact_registry_sha256": hashlib.sha256(
                _canonical(observation.factual_registry).encode("utf-8")
            ).hexdigest(),
            "current_event_sha256": hashlib.sha256(
                _canonical(observation.current_event).encode("utf-8")
            ).hexdigest(),
            "method_evidence_sha256": hashlib.sha256(
                _canonical(evidence).encode("utf-8")
            ).hexdigest(),
        })

    identifiers = [str(row["case_id"]) for row in cases]
    if len(set(identifiers)) != 48:
        raise ValueError("judge case IDs are not unique")
    if any(not row.get("blind_fields_absent") for row in cases):
        raise ValueError("judge case blindness assertion failed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_root / "judge_cases.jsonl", cases)
    _write_jsonl(args.output_root / "method_evidence.jsonl", evidence_rows)
    _write_jsonl(args.output_root / "judge_source_audit.jsonl", source_rows)
    manifest = {
        "schema_version": "campuspulse-dimension2-judge-case-manifest-v1",
        "evaluation_definitions_version": "1.6.0",
        "text_evaluation_mode": "model_judge_with_stratified_human_review",
        "case_count": len(cases),
        "unique_case_count": len(set(identifiers)),
        "human_review_required": True,
        "human_review_fraction": 0.20,
        "method_name_exposed_to_judge": False,
        "variant_exposed_to_judge": False,
        "agent_id_exposed_to_judge": False,
        "particle_or_prestate_exposed_to_judge": False,
    }
    (args.output_root / "judge_case_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
