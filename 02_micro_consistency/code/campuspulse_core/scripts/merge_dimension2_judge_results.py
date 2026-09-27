from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def response_text(raw: dict) -> str:
    return "".join(
        str(content.get("text") or "")
        for item in raw.get("output", [])
        for content in item.get("content", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and validate sharded D2 judge results.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-response-dir", type=Path, action="append", default=[])
    parser.add_argument("--correct-case-id", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    expected = {str(row["case_id"]) for row in cases}
    if len(cases) != 2048 or len(expected) != 2048:
        raise ValueError("frozen judge case set must contain 2048 unique IDs")
    merged: dict[str, dict] = {}
    for path in args.judgements:
        for row in read_jsonl(path):
            identifier = str(row["case_id"])
            if identifier in merged:
                raise ValueError(f"duplicate judgement ID: {identifier}")
            merged[identifier] = row

    corrections = []
    for expected_id in args.correct_case_id:
        matches = []
        for directory in args.raw_response_dir:
            if not directory.exists():
                continue
            for path in directory.glob("*.json"):
                archive = json.loads(path.read_text(encoding="utf-8"))
                if expected_id not in canonical(archive.get("request", {})):
                    continue
                parsed = json.loads(response_text(archive.get("response", {})))
                matches.append((path, archive, parsed))
        if not matches:
            raise ValueError(f"no raw response found for correction: {expected_id}")
        path, archive, parsed = sorted(matches, key=lambda item: item[0].stat().st_mtime)[-1]
        returned_id = str(parsed.get("case_id"))
        if returned_id == expected_id:
            raise ValueError(f"correction is unnecessary: {expected_id}")
        parsed["case_id"] = expected_id
        if expected_id in merged:
            raise ValueError(f"corrected ID already exists: {expected_id}")
        merged[expected_id] = parsed
        corrections.append({
            "expected_case_id": expected_id,
            "returned_case_id": returned_id,
            "raw_archive_path": str(path.resolve()),
            "raw_response_sha256": archive.get("response_sha256"),
            "correction_scope": "case_id_metadata_only",
            "scored_fields_modified": False,
        })

    actual = set(merged)
    if actual != expected:
        raise ValueError(f"case coverage mismatch: missing={len(expected-actual)} extra={len(actual-expected)}")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for identifier, row in merged.items():
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"schema failure for {identifier}: {errors[0].message}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / "judgements.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier in sorted(merged):
            handle.write(canonical(merged[identifier]) + "\n")
    correction_path = args.output_root / "case_id_corrections.jsonl"
    with correction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in corrections:
            handle.write(canonical(row) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "dimension2-judge-merge-v1",
        "case_count": len(merged),
        "unique_case_count": len(actual),
        "correction_count": len(corrections),
        "correction_scope": "case_id_metadata_only",
        "all_scored_fields_preserved": True,
        "all_rows_schema_valid": True,
        "judgements_sha256": digest,
    }
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
