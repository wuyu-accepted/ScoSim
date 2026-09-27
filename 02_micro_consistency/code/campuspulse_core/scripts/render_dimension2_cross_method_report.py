from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.common import read_jsonl, write_json
from evaluation.dimension2 import METHODS, case_id, evaluate_cross_method
from evaluation.text_judge import (
    audit_reliability, select_human_audit_case_ids,
)


LABELS = {
    "campuspulse": "CampusPulse",
    "belief_engine": "Belief Engine",
    "generative_agents": "Generative Agents",
    "chuang_opinion_dynamics": "Chuang et al.",
}


def _metric(result: dict[str, Any], method: str, key: str) -> tuple[Any, int, str]:
    if key == "language_style_consistency":
        value = result["methods"][method][key]
        return value.get("value"), int(value.get("eligible_expression_count") or 0), str(value.get("status"))
    value = result["methods"][method]["text_metrics"][key]
    return value.get("rate"), int(value.get("eligible_count") or 0), str(value.get("status"))


def _display(value: Any, *, percent: bool) -> str:
    if value is None:
        return "不适用"
    return f"{100 * float(value):.1f}%" if percent else f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the four-method D2 horizontal comparison table.")
    parser.add_argument("--records", type=Path, nargs=4, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--human-judgements", type=Path, required=True)
    parser.add_argument("--belief-state-sidecar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    records = [row for path in args.records for row in read_jsonl(path)]
    judgements = read_jsonl(args.judgements)
    expected_cases = {case_id(row) for row in records}
    observed_cases = {str(row.get("case_id")) for row in judgements}
    if len(records) != 2048 or len(expected_cases) != 2048:
        raise ValueError("horizontal report requires exactly 2048 unique records")
    if observed_cases != expected_cases or len(judgements) != 2048:
        raise ValueError("horizontal report requires complete 2048-case judgements")
    human_rows = read_jsonl(args.human_judgements)
    expected_human = select_human_audit_case_ids(records, fraction=0.20)
    observed_human = {str(row.get("case_id")) for row in human_rows}
    if len(human_rows) != len(expected_human) or observed_human != expected_human:
        raise ValueError("horizontal report requires the exact frozen 20% independent blind review sample")
    reliability = audit_reliability(judgements, human_rows)
    sidecar_rows = read_jsonl(args.belief_state_sidecar)
    belief_state = {
        str(row["case_id"]): float(row["standardized_pre_state"])
        for row in sidecar_rows
    }
    if len(belief_state) != 512:
        raise ValueError("Belief Engine predecision sidecar must contain 512 cases")
    result = evaluate_cross_method(records, judgements, belief_state)
    calibration = result["stance_direction_contract"]["expression_pre_state_agreement"]
    radii = calibration["neutral_radius_by_method"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "dimension2_cross_method.json", result)
    write_json(args.output_root / "judge_reliability.json", reliability)
    specs = (
        ("direct_context_uptake", "上下文承接率", True),
        ("unsupported_fact_fabrication", "虚假事实编造率", True),
        ("expression_pre_state_agreement", "表达与前序状态吻合率", True),
        ("language_style_consistency", "语言风格一致性", False),
    )
    rows = []
    for key, label, percent in specs:
        row = {"metric": key, "metric_label": label}
        for method in METHODS:
            value, eligible, status = _metric(result, method, key)
            row[method] = value
            row[f"{method}_eligible"] = eligible
            row[f"{method}_status"] = status
        rows.append(row)
    csv_fields = ["metric", "metric_label"] + [
        field for method in METHODS
        for field in (method, f"{method}_eligible", f"{method}_status")
    ]
    with (args.output_root / "dimension2_cross_method_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 维度二横向对比实验结果", "",
        "| 评估指标 | CampusPulse | Belief Engine | Generative Agents | Chuang et al. |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for key, label, percent in specs:
        values = [
            _display(_metric(result, method, key)[0], percent=percent)
            for method in METHODS
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend([
        "", "每种方法512条记录，四种方法严格配对于同一组观察。",
        "表达与前序状态吻合率仅适用于 CampusPulse 与 Belief Engine。",
        f"该指标使用共同82%中立比例校准：CampusPulse 原生半径为 {radii['campuspulse']:.15f}，Belief Engine 原生半径为 {radii['belief_engine']:.15f}。",
        "结构化 public_stance 继续使用独立的固定 ±0.05 分箱；校准不会修改微观行为分布。",
        f"独立盲审覆盖 {len(human_rows)}/2048 条冻结样本（20%向上取整）。",
        "详细的适用样本数、状态与置信区间见 JSON 和 CSV。", "",
    ])
    (args.output_root / "FINAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"record_count": 2048, "output": str(args.output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
