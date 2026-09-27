from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from comparison_harness.fake_executor import load_observations
from comparison_harness.fair_initialization import load_fair_role_initializations
from evaluation.common import read_jsonl
from evaluation.dimension2 import METHODS, case_id
from evaluation.text_judge import build_case, load_frozen_prompt


EXPECTED_PER_METHOD = 512


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def build_cross_method_cases(
    record_paths: list[Path], observations_path: Path, population_path: Path,
    prompt_path: Path, expected_per_method: int = EXPECTED_PER_METHOD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = [row for path in record_paths for row in read_jsonl(path)]
    counts = Counter(str(row.get("method")) for row in records)
    if counts != Counter({method: expected_per_method for method in METHODS}):
        raise ValueError(
            f"cross-method records must be {expected_per_method} per method: {dict(counts)}"
        )
    if any(row.get("variant") != "cross_method_full" for row in records):
        raise ValueError("all cross-method records must use cross_method_full")
    observations = {
        row.observation_id: row.observation
        for row in load_observations(observations_path)
    }
    if len(observations) != EXPECTED_PER_METHOD:
        raise ValueError("frozen common observation panel must contain 512 unique rows")
    roles = load_fair_role_initializations(population_path)
    by_method = {
        method: {str(row.get("source_record_id")): row for row in records if row.get("method") == method}
        for method in METHODS
    }
    reference = set(by_method[METHODS[0]])
    if len(reference) != expected_per_method or not reference.issubset(observations):
        raise ValueError("record IDs are not a valid subset of the frozen observation panel")
    if any(set(rows) != reference for rows in by_method.values()):
        raise ValueError("four methods are not paired on the same observation IDs")
    prompt = load_frozen_prompt(prompt_path)
    cases: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for source_id in sorted(reference):
            row = by_method[method][source_id]
            observation = observations[source_id]
            if row.get("agent_id") != observation.agent_id or int(row.get("tick")) != observation.tick:
                raise ValueError("record identity does not match its frozen observation")
            role = roles[observation.agent_id]
            evidence = [{
                "source_id": "role-card-" + role.role_card_sha256[:24],
                "source_type": "shared_generation_input",
                "content": role.role_card,
            }]
            item = build_case(
                row, prompt, observation.factual_registry,
                observation.current_event, evidence,
            )
            cases.append(item)
            evidence_rows.append({"case_id": item["case_id"], "evidence": evidence})
            audit_rows.append({
                "case_id": item["case_id"],
                "source_record_id": source_id,
                "observation_sha256": observation.observation_sha256,
                "role_card_sha256": role.role_card_sha256,
                "fact_registry_sha256": hashlib.sha256(
                    _canonical(observation.factual_registry).encode("utf-8")
                ).hexdigest(),
                "current_event_sha256": hashlib.sha256(
                    _canonical(observation.current_event).encode("utf-8")
                ).hexdigest(),
            })
    identifiers = [str(row["case_id"]) for row in cases]
    expected_cases = expected_per_method * len(METHODS)
    if len(cases) != expected_cases or len(set(identifiers)) != expected_cases:
        raise ValueError("cross-method blind case IDs are not complete and unique")
    if any(not row.get("blind_fields_absent") for row in cases):
        raise ValueError("judge blindness assertion failed")
    manifest = {
        "schema_version": "dimension2-cross-method-judge-cases-v1.0.0",
        "case_count": expected_cases,
        "method_count": 4,
        "cases_per_method": expected_per_method,
        "paired_observation_count": expected_per_method,
        "scientific_result": expected_per_method == EXPECTED_PER_METHOD,
        "method_name_exposed_to_judge": False,
        "variant_exposed_to_judge": False,
        "agent_id_exposed_to_judge": False,
        "predecision_state_exposed_to_judge": False,
        "shared_role_evidence_exposed_identically": True,
        "network_calls_made": 0,
    }
    return cases, evidence_rows, audit_rows, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2048 method-blind D2 horizontal judge cases.")
    parser.add_argument("--records", type=Path, nargs=4, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--pilot-one-per-method", action="store_true",
        help="Build four non-scientific audit cases from one paired observation.",
    )
    args = parser.parse_args()
    cases, evidence, audit, manifest = build_cross_method_cases(
        [path.resolve() for path in args.records], args.observations.resolve(),
        args.population.resolve(), args.prompt.resolve(),
        1 if args.pilot_one_per_method else EXPECTED_PER_METHOD,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_root / "judge_cases.jsonl", cases)
    _write_jsonl(args.output_root / "method_evidence.jsonl", evidence)
    _write_jsonl(args.output_root / "judge_source_audit.jsonl", audit)
    (args.output_root / "judge_case_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
