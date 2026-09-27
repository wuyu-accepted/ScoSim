from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from comparison_harness.campuspulse_assimilation import assimilate_isolated_turn
from comparison_harness.contracts import canonical_json
from evaluation.dimension2 import case_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def validate(validator: Draft202012Validator, row: dict[str, Any], label: str) -> None:
    errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"{label} schema failed: {errors[0].message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize native CampusPulse post-state without new model calls.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--prompt-audit", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-schema", type=Path, required=True)
    parser.add_argument("--sidecar-schema", type=Path, required=True)
    parser.add_argument("--scenario-id", default="governance_legitimacy_dispute")
    args = parser.parse_args()
    if args.output_records.exists() or args.output_sidecar.exists():
        raise FileExistsError("derived outputs are immutable; choose new paths")
    records = read_jsonl(args.records)
    audits = {str(row["source_record_id"]): row for row in read_jsonl(args.prompt_audit)}
    record_validator = Draft202012Validator(json.loads(args.output_schema.read_text(encoding="utf-8")))
    sidecar_validator = Draft202012Validator(json.loads(args.sidecar_schema.read_text(encoding="utf-8")))
    outputs, sidecars = [], []
    for source in records:
        row = dict(source)
        row["schema_version"] = "1.1.0"
        row["assimilation_sidecar_id"] = None
        applicable = row.get("method") == "campuspulse" and row.get("variant") in {"state_only", "full"} and row.get("model_status") == "success"
        if applicable:
            audit = audits.get(str(row.get("source_record_id")))
            if audit is None:
                raise ValueError(f"missing prompt audit for {row.get('source_record_id')}")
            decision = {key: row.get(key) for key in ("action", "emotion", "public_stance", "public_or_private", "generated_text")}
            native = assimilate_isolated_turn(
                particles=row["particles"], particle_weights=row["particle_weights"], decision=decision,
                scenario_id=args.scenario_id, variant=str(row["variant"]), tick=int(row["tick"]), seed=int(row["seed"]),
                prompt_sha256=str(audit["rendered_prompt_sha256"]),
            )
            identifier = "assim-" + native["innovation_id"][:24]
            row["post_state"] = native["after_state"]
            row["assimilation_sidecar_id"] = identifier
            sidecar = {
                "schema_version": "campuspulse-assimilation-sidecar-v1.0.0",
                "assimilation_sidecar_id": identifier,
                "case_id": case_id(row),
                "source_record_id": row["source_record_id"],
                **native,
            }
            validate(sidecar_validator, sidecar, "assimilation sidecar")
            sidecars.append(sidecar)
        validate(record_validator, row, "unified output")
        outputs.append(row)
    write_jsonl(args.output_records, outputs)
    write_jsonl(args.output_sidecar, sidecars)
    print(json.dumps({"record_count": len(outputs), "assimilation_count": len(sidecars), "network_calls": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
