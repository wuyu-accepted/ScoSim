from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


PARTICLE_VARIANTS = {"full": 10, "single_particle": 1, "no_profile": 10}
VALID_VARIANTS = {
    "campuspulse": {
        "context_only", "profile_only", "state_only", "full",
        "single_particle", "no_profile", "no_profile_no_smc",
    },
    "belief_engine": {"full"},
    "generative_agents": {"full"},
    "chuang_opinion_dynamics": {"full"},
}


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc


def semantic_errors(record: dict) -> list[str]:
    errors: list[str] = []
    method, variant = record["method"], record["variant"]
    if variant not in VALID_VARIANTS[method]:
        errors.append(f"variant {variant!r} is not registered for method {method!r}")

    particles, weights = record["particles"], record["particle_weights"]
    if (particles is None) != (weights is None):
        errors.append("particles and particle_weights must both be null or both be arrays")
    if particles is not None:
        if len(particles) != len(weights):
            errors.append("particles and particle_weights lengths differ")
        if abs(sum(weights) - 1.0) > 1e-6:
            errors.append(f"particle weights sum to {sum(weights):.12g}, expected 1")
        expected = PARTICLE_VARIANTS.get(variant)
        if method == "campuspulse" and expected is not None and len(particles) != expected:
            errors.append(f"variant {variant!r} requires {expected} particles")
    elif method == "campuspulse" and variant not in {"context_only", "profile_only", "no_profile_no_smc"}:
        errors.append(f"variant {variant!r} requires particle output")

    successful = record["model_status"] in {"success", "schema_repaired"}
    if successful and record["generated_text"] is None:
        errors.append("successful generation requires generated_text")
    if not successful and record.get("failure") is None and record["model_status"] not in {"skipped", "not_applicable"}:
        errors.append("failed generation requires failure details")
    return errors


def validate(schema_path: Path, input_path: Path) -> int:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    failures: list[str] = []
    seen: set[tuple] = set()
    ticks: dict[tuple, set[int]] = defaultdict(set)
    count = 0

    try:
        records = load_jsonl(input_path)
        for line_number, record in records:
            count += 1
            for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
                location = ".".join(map(str, error.path)) or "$"
                failures.append(f"line {line_number} [{location}]: {error.message}")
            if not isinstance(record, dict) or not all(
                key in record for key in ("experiment_id", "method", "variant", "seed", "agent_id", "tick")
            ):
                continue
            for error in semantic_errors(record):
                failures.append(f"line {line_number}: {error}")
            key = (
                record["experiment_id"], record["method"], record["variant"],
                record["seed"], record["agent_id"], record["tick"],
            )
            if key in seen:
                failures.append(f"line {line_number}: duplicate agent-tick key {key}")
            seen.add(key)
            ticks[key[:-1]].add(record["tick"])
    except ValueError as exc:
        failures.append(str(exc))

    for group, observed in ticks.items():
        expected = set(range(min(observed), max(observed) + 1))
        if observed != expected:
            failures.append(f"tick discontinuity for {group}: missing {sorted(expected - observed)}")

    if failures:
        print(f"FAILED: {len(failures)} error(s) in {count} record(s)", file=sys.stderr)
        for message in failures:
            print(f"- {message}", file=sys.stderr)
        return 1
    print(f"PASSED: {count} record(s); schema and semantic checks succeeded")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CampusPulse unified JSONL output")
    parser.add_argument("input", type=Path, help="JSONL output file")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("unified_output_v1.0.0.schema.json"),
    )
    args = parser.parse_args()
    return validate(args.schema, args.input)


if __name__ == "__main__":
    raise SystemExit(main())
