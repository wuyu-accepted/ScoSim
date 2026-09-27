from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from comparison_harness.contracts import canonical_json
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider
from comparison_harness.real_runtime import RealEvaluationRuntime, load_frozen_prompt


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_sync(handle, value) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute frozen dimension-2 judge packets.")
    parser.add_argument("judge_cases", type=Path)
    parser.add_argument("--judge-prompt", type=Path, required=True)
    parser.add_argument("--stance-prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--auxiliary-boundary", type=Path, required=True)
    parser.add_argument("--provider-boundary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provider-state-root", type=Path)
    parser.add_argument("--budget-profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-real", action="store_true", required=True)
    args = parser.parse_args()

    config = json.loads(args.provider_boundary.read_text(encoding="utf-8"))
    hashes = config["evaluation_prompt_sha256"]
    judge_prompt = load_frozen_prompt(args.judge_prompt, hashes["dimension2_text_judge"])
    stance_prompt = load_frozen_prompt(args.stance_prompt, hashes["stance_probe"])
    transport = OpenAIProviderTransport(
        api_key_env=config["api_key_environment_variable"],
        timeout_seconds=float(config["request_constraints"]["timeout_seconds"]),
    )
    provider = RealModelProvider(
        config_path=args.provider_boundary,
        auxiliary_boundary_path=args.auxiliary_boundary,
        storage_root=args.provider_state_root or args.output_root / "provider_state",
        transport=transport, enable_real=True, budget_profile=args.budget_profile,
    )
    runtime = RealEvaluationRuntime(
        provider, judge_prompt=judge_prompt, stance_prompt=stance_prompt
    )
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    cases = _read_jsonl(args.judge_cases)
    output = args.output_root / "judgements.jsonl"
    existing = {}
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"existing output requires --resume: {output}")
        existing = {str(row["case_id"]): row for row in _read_jsonl(output)}
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            case_id = str(case["case_id"])
            if case_id in existing:
                continue
            judgement = dict(runtime.judge(str(case["prompt"]), schema))
            if str(judgement.get("case_id")) != case_id:
                # The case identifier is an evaluation-side join key, not a
                # semantic field.  Keep the raw model response in the provider
                # archive, but normalize this field to the locally assigned ID
                # so one malformed echo cannot abort a resumable shard.
                judgement["case_id"] = case_id
            _append_sync(handle, judgement)
            existing[case_id] = judgement
            written += 1
    report = {
        "schema_version": "real-dimension2-judge-run-v1",
        "case_count": len(cases), "completed_count": len(existing),
        "newly_written": written, "budget_used": provider.budget_snapshot(),
    }
    (args.output_root / "judge_run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
