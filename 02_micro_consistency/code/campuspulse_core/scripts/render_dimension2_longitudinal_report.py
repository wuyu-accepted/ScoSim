from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluation.common import read_jsonl, write_json
from evaluation.dimension2 import (
    CONDITIONS,
    PRESTATE_NEUTRAL_RADIUS_BY_METHOD,
    case_id,
    evaluate,
)


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_source_id(value: object) -> str:
    return str(value).split(":ablation:", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the frozen 2048-record D2 longitudinal ablation report.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--state-sidecars", type=Path, required=True)
    parser.add_argument("--full-sidecars", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument(
        "--reuse-classification-from", type=Path,
        help="Reuse a prior 999-permutation Panel C only after exact distribution/fold invariance checks.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.records)
    judgements = read_jsonl(args.judgements)
    if len(records) != 2048 or len(judgements) != 2048:
        raise ValueError("longitudinal report requires 2048 records and 2048 judgements")
    expected = {case_id(row) for row in records}
    observed = {str(row["case_id"]) for row in judgements}
    if len(expected) != 2048 or len(observed) != 2048 or expected != observed:
        raise ValueError(
            f"judgement coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}"
        )

    record_by_condition_source = {
        (str(row["variant"]), base_source_id(row["source_record_id"])): row for row in records
    }
    if len(record_by_condition_source) != 2048:
        raise ValueError("records are not unique by condition and frozen observation")

    normalized_sidecars: list[dict] = []
    for condition, path in (("state_only", args.state_sidecars), ("full", args.full_sidecars)):
        rows = read_jsonl(path)
        if len(rows) != 512:
            raise ValueError(f"{condition} requires exactly 512 assimilation sidecars")
        seen: set[str] = set()
        for sidecar in rows:
            source = base_source_id(sidecar["source_record_id"])
            record = record_by_condition_source.get((condition, source))
            if record is None:
                raise ValueError(f"unpaired {condition} sidecar: {source}")
            item = dict(sidecar)
            item["source_record_id"] = str(record["source_record_id"])
            item["case_id"] = case_id(record)
            if item["case_id"] in seen:
                raise ValueError(f"duplicate normalized sidecar: {item['case_id']}")
            seen.add(item["case_id"])
            normalized_sidecars.append(item)

    evaluation_permutations = 1 if args.reuse_classification_from else args.permutations
    result = evaluate(
        records,
        judgements,
        permutation_count=evaluation_permutations,
        permutation_seed=20260722,
        assimilation_rows=normalized_sidecars,
    )
    classification_reuse = None
    if args.reuse_classification_from:
        prior = json.loads(args.reuse_classification_from.read_text(encoding="utf-8"))
        if prior.get("record_count") != result.get("record_count"):
            raise ValueError("cannot reuse Panel C: record count changed")
        if prior.get("feature_count") != result.get("feature_count"):
            raise ValueError("cannot reuse Panel C: feature contract changed")
        if prior.get("distributions") != result.get("distributions"):
            raise ValueError("cannot reuse Panel C: behavior distributions changed")
        if prior.get("fold_predictions") != result.get("fold_predictions"):
            raise ValueError("cannot reuse Panel C: LOAO fold predictions changed")
        prior_classifier = prior.get("classifier", {})
        if prior_classifier.get("permutation_count") != args.permutations:
            raise ValueError("cannot reuse Panel C: prior permutation count does not match requested count")
        if prior_classifier.get("permutation_seed") != 20260722:
            raise ValueError("cannot reuse Panel C: prior permutation seed is not frozen seed 20260722")
        result["classification"] = prior["classification"]
        result["classifier"] = prior_classifier
        classification_reuse = {
            "status": "reused_after_exact_invariance_check",
            "source": str(args.reuse_classification_from.resolve()),
            "source_sha256": sha256(args.reuse_classification_from),
            "checked_equal": ["record_count", "feature_count", "distributions", "fold_predictions"],
            "reason": "protocol-1.3.0 changes only expression/pre-state direction projection; Panel C feature matrix is unchanged",
        }
        result["classification_reuse_audit"] = classification_reuse
    args.output_root.mkdir(parents=True, exist_ok=False)
    sidecar_path = args.output_root / "assimilation_sidecar.state_and_full.jsonl"
    sidecar_path.write_text(
        "".join(canonical(row) + "\n" for row in sorted(normalized_sidecars, key=lambda row: row["case_id"])),
        encoding="utf-8", newline="\n",
    )
    result_path = args.output_root / "dimension2_longitudinal_ablation.json"
    write_json(result_path, result)

    table_path = args.output_root / "dimension2_longitudinal_ablation_table.csv"
    with table_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *CONDITIONS])
        metric_keys = (
            ("上下文承接率", "direct_context_uptake"),
            ("虚假事实编造率", "unsupported_fact_fabrication"),
            ("表达与前序状态吻合率", "expression_pre_state_agreement"),
            ("决策后状态—表达自洽率", "decision_post_state_expression_consistency"),
        )
        for label, key in metric_keys:
            writer.writerow([label, *[
                result["text_metrics_by_condition"][condition][key]["rate"]
                for condition in CONDITIONS
            ]])
        writer.writerow(["去重后语言风格一致性", *[
            result["language_style_consistency_by_condition"][condition]["value"]
            for condition in CONDITIONS
        ]])
        for task, values in result["classification"].items():
            writer.writerow([f"LOAO-SVM {task} accuracy", values["accuracy"], "", "", ""])
            writer.writerow([f"LOAO-SVM {task} macro-F1", values["macro_f1"], "", "", ""])
            writer.writerow([f"LOAO-SVM {task} permutation p", values["accuracy_p_one_sided"], "", "", ""])

    md_path = args.output_root / "dimension2_longitudinal_ablation_report.md"
    labels = {
        "context_only": "纯上下文", "profile_only": "仅 Profile 约束",
        "state_only": "仅粒子状态约束", "full": "完整输入",
    }
    def pct(value: object) -> str:
        return "不适用" if value is None else f"{100 * float(value):.1f}%"
    lines = [
        "# 维度二纵向受控消融实验汇总", "",
        "## 文本质量与状态一致性", "",
        "| 指标 | 纯上下文 | 仅 Profile 约束 | 仅粒子状态约束 | 完整输入 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in metric_keys:
        values = [result["text_metrics_by_condition"][condition][key]["rate"] for condition in CONDITIONS]
        lines.append("| " + label + " | " + " | ".join(pct(value) for value in values) + " |")
    style = [result["language_style_consistency_by_condition"][condition]["value"] for condition in CONDITIONS]
    lines.append("| 去重后语言风格一致性 | " + " | ".join("不适用" if value is None else f"{float(value):.3f}" for value in style) + " |")
    lines.extend(["", "## LOAO-SVM 条件识别与置换检验", "", "| 特征任务 | Accuracy | Macro-F1 | 单侧置换 p（Accuracy） |", "| :--- | ---: | ---: | ---: |"])
    for task, values in result["classification"].items():
        lines.append(f"| {task} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} | {values['accuracy_p_one_sided']:.4f} |")
    lines.extend([
        "", "## 口径与样本审计", "",
        f"- 表达与前序状态吻合率使用共同 82% 中立比例校准后冻结的 CampusPulse 原生半径 `±{PRESTATE_NEUTRAL_RADIUS_BY_METHOD['campuspulse']:.15f}`；所有条件共用该半径，不按条件重新拟合。",
        "- 结构化 `public_stance` 的行为分布与条件分类器继续使用独立的固定 `±0.05` 分箱；决策后状态—表达自洽率也继续使用其独立的固定 `±0.05` 半径。",
        "- 四条件各 512 条，共 2048 条，基于同一组 512 条冻结观察逐槽配对。",
        "- 盲评结果 2048 条，case_id 无重复、无缺失。",
        "- 置换检验固定种子为 20260722，置换次数为 999。",
        "- 数据面板 C 的输入分布与逐条 LOAO 折预测经精确相等校验后，复用原 999 次置换结果；82% 前序状态校准不进入分类器特征。" if classification_reuse else "- 数据面板 C 在本次报告生成中重新执行全部置换检验。",
        "- 当前报告使用统一外部模型盲评；若论文协议要求独立人工 20% 复核，可靠性统计需在人工答案回收后补充。", "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": "dimension2-longitudinal-report-manifest-v1.0.0",
        "record_count": 2048, "judgement_count": 2048,
        "unique_case_count": 2048, "missing_case_count": 0, "duplicate_case_count": 0,
        "conditions": {condition: 512 for condition in CONDITIONS},
        "assimilation_sidecar_count": len(normalized_sidecars),
        "permutation_count": args.permutations, "permutation_seed": 20260722,
        "inputs": {"records_sha256": sha256(args.records), "judgements_sha256": sha256(args.judgements),
                   "state_sidecars_sha256": sha256(args.state_sidecars), "full_sidecars_sha256": sha256(args.full_sidecars),
                   "reused_classification_source_sha256": None if not args.reuse_classification_from else sha256(args.reuse_classification_from)},
        "classification_reuse_audit": classification_reuse,
        "outputs": {"evaluation_sha256": sha256(result_path), "table_sha256": sha256(table_path),
                    "report_sha256": sha256(md_path), "assimilation_sidecars_sha256": sha256(sidecar_path)},
    }
    write_json(args.output_root / "report_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
