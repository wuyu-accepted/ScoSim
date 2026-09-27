from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .common import read_jsonl
from .dimension2 import case_id
from .inference import cohens_kappa


PROMPT_SHA256 = "98c8630f31175c05f8e9ee94280e92eb8b4484ed7a860accb6cfda56d22481b9"
PLACEHOLDERS = (
    "{{FACT_REGISTRY_JSON}}", "{{METHOD_EVIDENCE_JSON}}", "{{CURRENT_EVENT_JSON}}",
    "{{VISIBLE_CONTEXT_JSON}}", "{{GENERATED_TEXT}}", "{{CASE_ID}}",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_frozen_prompt(path: Path) -> str:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != PROMPT_SHA256:
        raise ValueError(f"judge prompt hash drift: {observed}")
    text = raw.decode("utf-8")
    if any(token not in text for token in PLACEHOLDERS):
        raise ValueError("judge prompt placeholders are incomplete")
    return text


def build_case(
    row: dict[str, Any], prompt: str, facts: Any, current_event: Any,
    method_evidence: Any,
) -> dict[str, Any]:
    identifier = case_id(row)
    replacements = {
        "{{FACT_REGISTRY_JSON}}": _canonical(facts),
        "{{METHOD_EVIDENCE_JSON}}": _canonical(method_evidence),
        "{{CURRENT_EVENT_JSON}}": _canonical(current_event),
        "{{VISIBLE_CONTEXT_JSON}}": _canonical(row.get("visible_context") or []),
        "{{GENERATED_TEXT}}": str(row.get("generated_text") or ""),
        "{{CASE_ID}}": identifier,
    }
    rendered = prompt
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return {
        "case_id": identifier,
        "prompt": rendered,
        "source_key_sha256": hashlib.sha256(_canonical([
            row.get("experiment_id"), row.get("method"), row.get("variant"),
            row.get("seed"), row.get("agent_id"), row.get("tick"),
        ]).encode("utf-8")).hexdigest(),
        "blind_fields_absent": True,
    }


def validate_judgements(path: Path, schema_path: Path, expected_case_ids: set[str]) -> list[dict[str, Any]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    rows = read_jsonl(path)
    observed = set()
    for index, row in enumerate(rows, 1):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"judgement line {index}: {errors[0].message}")
        identifier = str(row["case_id"])
        if identifier in observed:
            raise ValueError(f"duplicate judgement case_id {identifier}")
        observed.add(identifier)
    if observed != expected_case_ids:
        raise ValueError(f"judgement coverage mismatch: missing={len(expected_case_ids-observed)}, extra={len(observed-expected_case_ids)}")
    return rows


def audit_reliability(model_rows: list[dict[str, Any]], human_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model = {str(row["case_id"]): row for row in model_rows}
    human = {str(row["case_id"]): row for row in human_rows}
    overlap = sorted(set(model) & set(human))
    if not overlap:
        raise ValueError("human audit has no cases in common with model judgements")
    fields = ("context_uptake", "unsupported_fact", "expression_stance")
    metrics = {}
    for field in fields:
        left = [str(model[key][field]) for key in overlap]
        right = [str(human[key][field]) for key in overlap]
        kappa = cohens_kappa(left, right)
        metrics[field] = {
            "agreement": sum(a == b for a, b in zip(left, right)) / len(overlap),
            "cohens_kappa": kappa,
            "cohens_kappa_status": "ok" if kappa is not None else "undefined_no_label_variation",
            "disagreement_count": sum(a != b for a, b in zip(left, right)),
            "model_label_distribution": dict(sorted(Counter(left).items())),
            "human_label_distribution": dict(sorted(Counter(right).items())),
        }
    return {"schema_version": "dimension2-judge-reliability-v1.0.0", "audited_case_count": len(overlap), "metrics": metrics}


def select_human_audit_case_ids(records: list[dict[str, Any]], fraction: float = 0.20, seed: int = 20260722) -> set[str]:
    if not 0 < fraction <= 1:
        raise ValueError("audit fraction must be in (0,1]")
    if not records:
        return set()
    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in records:
        key = (row.get("method"), row.get("variant"), row.get("seed"), row.get("public_or_private"))
        strata[key].append(case_id(row))
    identifiers = [identifier for values in strata.values() for identifier in values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("audit sampling population contains duplicate case IDs")

    # Freeze one overall 20% target first.  Applying ceil independently to every
    # stratum would oversample small balanced panels (48 records became 12/48).
    target = math.ceil(fraction * len(identifiers))
    quotas = {
        key: target * len(values) / len(identifiers)
        for key, values in strata.items()
    }
    allocations = {key: math.floor(quota) for key, quota in quotas.items()}
    remaining = target - sum(allocations.values())
    ranked_strata = sorted(
        strata,
        key=lambda key: (
            -(quotas[key] - allocations[key]),
            hashlib.sha256(f"{seed}:stratum:{_canonical(key)}".encode()).hexdigest(),
        ),
    )
    for key in ranked_strata[:remaining]:
        allocations[key] += 1

    selected: set[str] = set()
    for key in sorted(strata, key=str):
        ordered = sorted(strata[key], key=lambda identifier: hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest())
        selected.update(ordered[:allocations[key]])
    if len(selected) != target:
        raise AssertionError("human audit allocation did not reach the frozen target")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate frozen method-blind dimension-2 judge packets")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("records", type=Path)
    build.add_argument("--facts", type=Path, required=True)
    build.add_argument("--event", type=Path, required=True)
    build.add_argument("--method-evidence", type=Path)
    build.add_argument("--prompt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("records", type=Path)
    check.add_argument("judgements", type=Path)
    check.add_argument("--schema", type=Path, required=True)
    check_audit = sub.add_parser("validate-audit")
    check_audit.add_argument("audit_cases", type=Path)
    check_audit.add_argument("human_judgements", type=Path)
    check_audit.add_argument("--schema", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("model_judgements", type=Path)
    audit.add_argument("human_judgements", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    sample = sub.add_parser("sample-audit")
    sample.add_argument("records", type=Path)
    sample.add_argument("judge_cases", type=Path)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--answer-sheet", type=Path)
    sample.add_argument("--fraction", type=float, default=0.20)
    import_human = sub.add_parser("import-human-csv")
    import_human.add_argument("answer_sheet", type=Path)
    import_human.add_argument("audit_cases", type=Path)
    import_human.add_argument("--schema", type=Path, required=True)
    import_human.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        records = read_jsonl(args.records)
        prompt = load_frozen_prompt(args.prompt)
        facts = json.loads(args.facts.read_text(encoding="utf-8"))
        event = json.loads(args.event.read_text(encoding="utf-8"))
        evidence = {str(row["case_id"]): row.get("evidence", []) for row in read_jsonl(args.method_evidence)} if args.method_evidence else {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                item = build_case(row, prompt, facts, event, evidence.get(case_id(row), []))
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        return 0
    if args.command == "validate":
        records = read_jsonl(args.records)
        validate_judgements(args.judgements, args.schema, {case_id(row) for row in records})
        print(f"PASSED: {len(records)} complete method-blind judgements")
        return 0
    if args.command == "validate-audit":
        audit_cases = read_jsonl(args.audit_cases)
        expected = {str(row["case_id"]) for row in audit_cases}
        if len(expected) != len(audit_cases):
            raise ValueError("human audit case file contains duplicate case IDs")
        validate_judgements(args.human_judgements, args.schema, expected)
        print(f"PASSED: {len(expected)} complete human audit judgements")
        return 0
    if args.command == "sample-audit":
        identifiers = select_human_audit_case_ids(read_jsonl(args.records), args.fraction)
        cases = [row for row in read_jsonl(args.judge_cases) if str(row["case_id"]) in identifiers]
        if len(cases) != len(identifiers):
            raise ValueError("judge case file does not cover the selected human audit sample")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in cases:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if args.answer_sheet:
            args.answer_sheet.parent.mkdir(parents=True, exist_ok=True)
            fields = [
                "case_id", "context_uptake", "unsupported_fact",
                "unsupported_spans_json", "nearest_source_ids_json",
                "expression_stance", "confidence", "brief_reason",
            ]
            with args.answer_sheet.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in cases:
                    writer.writerow({
                        "case_id": row["case_id"],
                        "unsupported_spans_json": "[]",
                        "nearest_source_ids_json": "[]",
                    })
        return 0
    if args.command == "import-human-csv":
        expected = {str(row["case_id"]) for row in read_jsonl(args.audit_cases)}
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        rows: list[dict[str, Any]] = []
        with args.answer_sheet.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, source in enumerate(csv.DictReader(handle), start=2):
                try:
                    row = {
                        "schema_version": "dimension2-text-judgement-v1.0.0",
                        "case_id": str(source["case_id"]).strip(),
                        "context_uptake": str(source["context_uptake"]).strip(),
                        "unsupported_fact": str(source["unsupported_fact"]).strip(),
                        "unsupported_spans": json.loads(source["unsupported_spans_json"]),
                        "nearest_source_ids": json.loads(source["nearest_source_ids_json"]),
                        "expression_stance": str(source["expression_stance"]).strip(),
                        "confidence": float(source["confidence"]),
                        "brief_reason": str(source["brief_reason"]).strip(),
                    }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"human answer CSV line {line_number} is incomplete: {exc}") from exc
                errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
                if errors:
                    raise ValueError(f"human answer CSV line {line_number}: {errors[0].message}")
                rows.append(row)
        observed = [str(row["case_id"]) for row in rows]
        if len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError("human answer CSV coverage does not exactly match the frozen audit cases")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"PASSED: imported {len(rows)} complete human audit judgements")
        return 0
    result = audit_reliability(read_jsonl(args.model_judgements), read_jsonl(args.human_judgements))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
