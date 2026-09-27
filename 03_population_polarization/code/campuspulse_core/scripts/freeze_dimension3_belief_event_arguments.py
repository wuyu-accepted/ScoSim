from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROPOSITION = "在当前已公开信息下，现有暑期住宿分配通知具备足够程序正当性，可以按原样接受。"
CONFIRMED = "维修导致可用床位由520张降至320张，共有536份有效申请；通知列出三类优先考虑情形，但未说明类别之间及同类申请内部的具体排序。"
UNVERIFIED = "院系偏袒、先到先得、关系分配或已经内定均未得到当前可见证据支持。"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def val(fields: dict[str, Any], key: str) -> float:
    return float(fields[key]["value"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stakes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    stakes = read_jsonl(args.stakes)
    if len(stakes) != 1000:
        raise ValueError("formal event mapping requires 1000 stakes")
    rows = []
    for row in sorted(stakes, key=lambda x: x["agent_id"]):
        fields = row["fields"]
        exposure = max(float(bool(fields["directly_affected"]["value"])), val(fields, "resource_need_intensity"), val(fields, "service_channel_dependency"))
        vulnerability = (exposure + val(fields, "time_urgency") + (1.0 - val(fields, "substitute_resource_availability")) + val(fields, "appeal_cost")) / 4.0
        procedural_confidence = (val(fields, "historical_service_experience") + val(fields, "information_access_capacity") + val(fields, "eligibility_material_completeness") + val(fields, "substitute_resource_availability")) / 4.0
        negative_strength = clip(0.55 + 0.40 * vulnerability)
        positive_strength = clip(0.25 + 0.35 * procedural_confidence)
        arguments = [
            {"argument_id": "fact-scarcity-and-missing-order", "claim": CONFIRMED, "polarity": -1, "strength": negative_strength, "source_role": "opponent"},
            {"argument_id": "boundary-no-evidence-of-favoritism", "claim": UNVERIFIED, "polarity": 1, "strength": positive_strength, "source_role": "opponent"}
        ]
        rows.append({
            "schema_version": "dimension3-belief-event-arguments-v1.0.0",
            "scenario_id": row["scenario_id"], "agent_id": row["agent_id"],
            "interest_group": row["interest_group"], "stance_proposition": PROPOSITION,
            "shared_fact_sha256": hashlib.sha256(canonical({"confirmed": CONFIRMED, "unverified": UNVERIFIED}).encode()).hexdigest(),
            "interest_interpretation": {"exposure": exposure, "vulnerability": vulnerability, "procedural_confidence": procedural_confidence},
            "arguments": arguments,
        })
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "event_arguments.jsonl"
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    contract = {
        "schema_version": "dimension3-belief-event-contract-v1.0.0", "status": "frozen",
        "stance_proposition": PROPOSITION, "confirmed_facts": CONFIRMED,
        "unverified_claim_boundary": UNVERIFIED,
        "strength_mapping": {
            "exposure": "max(directly_affected, resource_need_intensity, service_channel_dependency)",
            "vulnerability": "mean(exposure,time_urgency,1-substitute_resource_availability,appeal_cost)",
            "procedural_confidence": "mean(historical_service_experience,information_access_capacity,eligibility_material_completeness,substitute_resource_availability)",
            "negative": "clip(0.55+0.40*vulnerability)", "positive": "clip(0.25+0.35*procedural_confidence)"
        },
        "agent_count": 1000, "arguments_per_agent": 2,
        "event_arguments_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "forbidden_inferences": ["具体个人能否获得床位", "未发布的排序规则", "截止时间", "申诉结果", "偏袒、关系分配或内定为事实"]
    }
    (args.output_root / "locked_event_argument_contract.json").write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(canonical(contract))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
