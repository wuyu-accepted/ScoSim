from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from jsonschema import Draft202012Validator
from openai import OpenAI


FIELDS = [
    "case_id", "context_uptake", "unsupported_fact",
    "unsupported_spans_json", "nearest_source_ids_json",
    "expression_stance", "confidence", "brief_reason",
]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_cases(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = [str(row["case_id"]) for row in rows]
    if len(rows) != 410 or len(set(identifiers)) != 410:
        raise ValueError("secondary blind review requires exactly 410 unique frozen cases")
    if any(not row.get("blind_fields_absent") for row in rows):
        raise ValueError("a supplied review case is not method-blind")
    return rows


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an independent second-model review over the frozen 410-case blind sample."
    )
    parser.add_argument("cases", type=Path)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=410)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 16:
        raise ValueError("max-workers must be between 1 and 16")
    if args.max_calls != 410:
        raise ValueError("formal secondary review budget must be exactly 410 calls")

    credentials = json.loads(args.credentials.read_text(encoding="utf-8-sig"))
    api_key = str(credentials.get("api_key") or "").strip()
    base_url = str(credentials.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        raise ValueError("credentials JSON must contain non-empty api_key and base_url")
    cases = read_cases(args.cases)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_root / "cache"
    raw_root = args.output_root / "raw_responses"
    cache_root.mkdir(exist_ok=True)
    raw_root.mkdir(exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout_seconds, max_retries=0)

    schema_instruction = (
        "\n\n【必须遵守的输出 JSON Schema】\n"
        + canonical(schema)
        + "\n严格按上述顶层字段输出；不得改名、嵌套或增加字段。"
    )

    def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        case_id = str(case["case_id"])
        cache_key = hashlib.sha256(
            canonical({"model": args.model, "prompt": case["prompt"], "schema": schema}).encode("utf-8")
        ).hexdigest()
        cache_path = cache_root / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            judgement = cached["judgement"]
            errors = list(validator.iter_errors(judgement))
            if not errors and str(judgement.get("case_id")) == case_id:
                return judgement
            raise ValueError(f"invalid cache entry for {case_id}")
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 2):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{
                        "role": "user",
                        "content": (
                            str(case["prompt"])
                            + schema_instruction
                            + "\n最终核对：case_id 必须逐字符复制为 "
                            + case_id
                        ),
                    }],
                    temperature=0,
                    max_tokens=args.max_output_tokens,
                    response_format={"type": "json_object"},
                    # DeepSeek-V4.1-Flash enables thinking by default.  This is a
                    # constrained classification task, so disable hidden reasoning
                    # to prevent it from consuming the entire output-token budget.
                    extra_body={"thinking": {"type": "disabled"}},
                )
                raw = response.model_dump(mode="json")
                raw_path = raw_root / f"{case_id}_attempt-{attempt:02d}.json"
                atomic_json(raw_path, {"case_id": case_id, "model": args.model, "response": raw})
                text = response.choices[0].message.content or ""
                judgement = parse_json_object(text)
                if str(judgement.get("case_id")) != case_id:
                    raise ValueError(
                        f"model changed case_id for {case_id}: {judgement.get('case_id')}"
                    )
                errors = sorted(validator.iter_errors(judgement), key=lambda error: list(error.path))
                if errors:
                    raise ValueError(f"schema failure for {case_id}: {errors[0].message}")
                atomic_json(cache_path, {
                    "cache_key": cache_key,
                    "case_id": case_id,
                    "model": args.model,
                    "judgement": judgement,
                    "raw_response_path": str(raw_path.resolve()),
                    "usage": raw.get("usage") or {},
                })
                return judgement
            except Exception as error:
                last_error = error
                if attempt > args.max_retries:
                    break
                time.sleep(min(8.0, (2 ** (attempt - 1)) + random.random()))
        assert last_error is not None
        raise last_error

    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(evaluate, case): str(case["case_id"]) for case in cases}
        for future in as_completed(futures):
            identifier = futures[future]
            completed[identifier] = future.result()

    if set(completed) != {str(row["case_id"]) for row in cases}:
        raise ValueError("secondary review did not cover the frozen sample exactly")
    jsonl_path = args.output_root / "secondary_model_judgements.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(canonical(completed[str(case["case_id"])]) + "\n")
    csv_path = args.output_root / "secondary_model_answers.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            row = completed[str(case["case_id"])]
            writer.writerow({
                "case_id": row["case_id"],
                "context_uptake": row["context_uptake"],
                "unsupported_fact": row["unsupported_fact"],
                "unsupported_spans_json": canonical(row["unsupported_spans"]),
                "nearest_source_ids_json": canonical(row["nearest_source_ids"]),
                "expression_stance": row["expression_stance"],
                "confidence": row["confidence"],
                "brief_reason": row["brief_reason"],
            })
    manifest = {
        "schema_version": "secondary-model-blind-review-v1",
        "reviewer_type": "independent_secondary_model",
        "model": args.model,
        "thinking_mode": "disabled",
        "case_count": len(completed),
        "human_review": False,
        "method_blind": True,
        "max_workers": args.max_workers,
        "judgements_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
    }
    atomic_json(args.output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
