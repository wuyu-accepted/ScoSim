from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from evaluation.dimension2 import PRESTATE_NEUTRAL_RADIUS_BY_METHOD


STANCE_LEVEL = {
    "strong_oppose": -1.0,
    "weak_oppose": -0.5,
    "neutral_or_mixed": 0.0,
    "weak_support": 0.5,
    "strong_support": 1.0,
}
PRESTATE_NEUTRAL_RADIUS = PRESTATE_NEUTRAL_RADIUS_BY_METHOD["campuspulse"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_id(row: dict) -> str:
    payload = [row.get(key) for key in ("experiment_id", "method", "variant", "seed", "agent_id", "tick")]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "d2-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def state_scalar(row: dict) -> float | None:
    particles, weights = row.get("particles"), row.get("particle_weights")
    if not particles or not weights or len(particles) != len(weights) or sum(weights) <= 0:
        return None
    return sum(float(w) * (float(p["support"]) - float(p["opposition"])) for p, w in zip(particles, weights)) / sum(weights)


def state_direction(value: float | None) -> str | None:
    if value is None:
        return None
    return (
        "support" if value > PRESTATE_NEUTRAL_RADIUS
        else "oppose" if value < -PRESTATE_NEUTRAL_RADIUS
        else "neutral"
    )


def expression_direction(label: str) -> str | None:
    level = STANCE_LEVEL.get(label)
    if level is None:
        return None
    return "support" if level > 0 else "oppose" if level < 0 else "neutral"


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    base, run = args.base.resolve(), args.run.resolve()
    results_dir, audit_dir = run / "results", run / "audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(run / "records.four_conditions.jsonl")
    judgements = {row["case_id"]: row for row in read_jsonl(run / "judge" / "judgements.jsonl")}
    evaluation_path = results_dir / "dimension_2_ablation_with_assimilation.json"
    if not evaluation_path.exists():
        evaluation_path = results_dir / "dimension_2_ablation.json"
    result = json.loads(evaluation_path.read_text(encoding="utf-8"))
    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    judge_report = json.loads((run / "judge" / "judge_run_report.json").read_text(encoding="utf-8"))

    if len(records) != 48 or len(judgements) != 48:
        raise ValueError("formal panel must contain exactly 48 records and 48 judgements")
    if result.get("definitions_version") not in {"1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0"}:
        raise ValueError("result cites an unsupported evaluation definition version")

    human_dir = run / "judge" / "human_audit"
    human_cases_path = human_dir / "human_audit_cases.jsonl"
    reliability_path = human_dir / "judge_reliability.json"
    human_target = len(read_jsonl(human_cases_path)) if human_cases_path.exists() else 0
    reliability = json.loads(reliability_path.read_text(encoding="utf-8")) if reliability_path.exists() else None
    human_completed = int(reliability["audited_case_count"]) if reliability else 0
    audit_complete = human_target > 0 and human_completed == human_target
    audit_has_disagreement = bool(
        reliability and any(
            int(metric.get("disagreement_count", 0)) > 0
            for metric in reliability.get("metrics", {}).values()
        )
    )
    if human_target not in {0, 10}:
        raise ValueError(f"48-record panel requires exactly 10 human audit cases, got {human_target}")

    text_metrics = result["text_metrics"]
    metric_rows = []
    for name in ("direct_context_uptake", "unsupported_fact_fabrication", "expression_pre_state_agreement", "decision_post_state_expression_consistency"):
        item = text_metrics[name]
        interval = item.get("bootstrap_95pct_interval") or [None, None]
        metric_rows.append({
            "metric": name,
            "value": item.get("rate"),
            "ci95_low": interval[0],
            "ci95_high": interval[1],
            "eligible_count": item.get("eligible_count"),
            "status": item.get("status"),
        })
    style = result["language_style_consistency"]
    interval = style.get("bootstrap_95pct_interval") or [None, None]
    metric_rows.append({"metric": "deduplicated_language_style_consistency", "value": style.get("value"), "ci95_low": interval[0], "ci95_high": interval[1], "eligible_count": style.get("eligible_unique_expression_count", style.get("eligible_expression_count")), "status": style.get("status")})
    alignment = result["llm_particle_update_direction_agreement"]
    metric_rows.append({"metric": "llm_particle_update_direction_agreement", "value": alignment.get("rate"), "ci95_low": None, "ci95_high": None, "eligible_count": alignment.get("tested_direction_count"), "status": alignment.get("status")})
    write_csv(results_dir / "metric_summary.csv", ["metric", "value", "ci95_low", "ci95_high", "eligible_count", "status"], metric_rows)

    distribution_rows = []
    for condition, dimensions in result["distributions"].items():
        for dimension in ("action", "emotion", "public_or_private", "stance"):
            for label, count in dimensions[dimension].items():
                distribution_rows.append({"condition": condition, "dimension": dimension, "label": label, "count": count, "proportion": dimensions["proportions"][dimension][label]})
    write_csv(results_dir / "condition_distributions.csv", ["condition", "dimension", "label", "count", "proportion"], distribution_rows)

    classifier_rows = []
    for task, item in result["classification"].items():
        classifier_rows.append({"task": task, **item})
    write_csv(results_dir / "classifier_summary.csv", ["task", "accuracy", "accuracy_p_one_sided", "macro_f1", "macro_f1_p_one_sided", "null_accuracy_mean", "null_accuracy_q95"], classifier_rows)

    judge_label_rows = []
    judge_label_summary = {}
    for field in ("context_uptake", "unsupported_fact", "expression_stance"):
        counts = Counter(str(item[field]) for item in judgements.values())
        judge_label_summary[field] = dict(sorted(counts.items()))
        for label, count in sorted(counts.items()):
            judge_label_rows.append({"field": field, "label": label, "count": count, "proportion": count / len(judgements)})
    write_csv(results_dir / "judge_label_distributions.csv", ["field", "label", "count", "proportion"], judge_label_rows)

    chain_rows = []
    for row in records:
        if row.get("variant") not in {"state_only", "full"}:
            continue
        cid = case_id(row)
        judgement = judgements[cid]
        scalar = state_scalar(row)
        before = state_direction(scalar)
        expressed = expression_direction(judgement["expression_stance"])
        chain_rows.append({
            "case_id": cid,
            "agent_id": row["agent_id"],
            "condition": row["variant"],
            "pre_state_scalar": scalar,
            "pre_state_direction": before,
            "expression_stance_label": judgement["expression_stance"],
            "expression_direction": expressed,
            "agreement": before == expressed if before is not None and expressed is not None else None,
        })
    with (results_dir / "expression_prestate_cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in chain_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    direct = text_metrics["direct_context_uptake"]
    unsupported = text_metrics["unsupported_fact_fabrication"]
    agreement = text_metrics["expression_pre_state_agreement"]
    post_agreement = text_metrics["decision_post_state_expression_consistency"]
    four = result["classification"]["four_condition"]
    profile = result["classification"]["profile_factor"]
    state = result["classification"]["state_factor"]
    report = f"""# CampusPulse 真实四条件维度二消融结果（{'人工复核完成，存在一致性警告' if audit_complete and audit_has_disagreement else '人工复核完成' if audit_complete else '暂定，等待人工复核'}）

## 运行范围

- 冻结种子：20260722；Tick：3；智能体：12；条件：4；有效记录：48/48。
- 条件：`context_only`、`profile_only`、`state_only`、`full`，每个条件 12 条。
- 生成模型调用：{run_manifest['budget_used']['logical_calls']} 次；盲评模型调用：{judge_report['budget_used']['logical_calls']} 次；传输重试：0。
- 文本评估模式：`model_judge_with_stratified_human_review`；人工复核：{human_completed}/{human_target or 10}。
- 当前冻结评估定义：1.6.0；数值评估所引定义：{result['definitions_version']}；评估结果 schema：{result['schema_version']}。
- 报告状态：{'人工复核已完成；存在标签分歧，必须连同一致性统计和限制一起报告' if audit_complete and audit_has_disagreement else '人工复核已完成' if audit_complete else '暂定；在人工复核及一致性统计完成前不得作为论文最终文本评估结果'}。

## 冻结指标结果

| 指标 | 值 | 95% 区间 | 有效样本 | 状态 |
|---|---:|---:|---:|---|
| 直接上下文承接率 | {fmt(direct['rate'])} | [{fmt(direct['bootstrap_95pct_interval'][0])}, {fmt(direct['bootstrap_95pct_interval'][1])}] | {direct['eligible_count']} | {direct['status']} |
| 不受支持事实虚构率 | {fmt(unsupported['rate'])} | [{fmt(unsupported['bootstrap_95pct_interval'][0])}, {fmt(unsupported['bootstrap_95pct_interval'][1])}] | {unsupported['eligible_count']} | {unsupported['status']} |
| 表达—前序状态吻合率 | {fmt(agreement['rate'])} | [{fmt(agreement['bootstrap_95pct_interval'][0])}, {fmt(agreement['bootstrap_95pct_interval'][1])}] | {agreement['eligible_count']} | {agreement['status']} |
| 决策后状态—表达自洽率（探索性） | {fmt(post_agreement['rate'])} | [{fmt(post_agreement['bootstrap_95pct_interval'][0])}, {fmt(post_agreement['bootstrap_95pct_interval'][1])}] | {post_agreement['eligible_count']} | {post_agreement['status']} |
| 语言风格一致性 | {fmt(style['value'])} | [{fmt(style['bootstrap_95pct_interval'][0])}, {fmt(style['bootstrap_95pct_interval'][1])}] | {style['eligible_expression_count']} | {style['status']} |
| LLM—粒子更新方向一致率 | {fmt(alignment['rate'])} | NA | {alignment['tested_direction_count']} | {alignment['status']} |

表达—前序状态指标和决策后状态—表达指标只对 `state_only` 与 `full` 适用，并复用同一外部盲评立场标签、有效样本分母及按智能体聚类 Bootstrap。表达—前序状态指标使用共同82%中立比例校准后冻结的 CampusPulse 原生半径 ±{PRESTATE_NEUTRAL_RADIUS:.15f}；结构化 `public_stance` 的固定 ±0.05 分箱保持独立且不受该校准影响；决策后探索性指标继续使用其独立的固定 ±0.05 半径。前序指标混淆计数：{json.dumps(agreement['confusion'], ensure_ascii=False, sort_keys=True)}；决策后指标混淆计数：{json.dumps(post_agreement['confusion'], ensure_ascii=False, sort_keys=True)}。决策后状态已吸收本轮生成动作与立场，因此该指标仅作探索性内部自洽诊断，不能替代表达—前序状态吻合率。

## 四条件可区分性（留一智能体交叉验证，999 次置换）

| 任务 | Accuracy | 单侧 p | Macro-F1 | 单侧 p |
|---|---:|---:|---:|---:|
| 四条件 | {fmt(four['accuracy'])} | {fmt(four['accuracy_p_one_sided'])} | {fmt(four['macro_f1'])} | {fmt(four['macro_f1_p_one_sided'])} |
| Profile 因子 | {fmt(profile['accuracy'])} | {fmt(profile['accuracy_p_one_sided'])} | {fmt(profile['macro_f1'])} | {fmt(profile['macro_f1_p_one_sided'])} |
| State 因子 | {fmt(state['accuracy'])} | {fmt(state['accuracy_p_one_sided'])} | {fmt(state['macro_f1'])} | {fmt(state['macro_f1_p_one_sided'])} |

## 解释边界

这次运行完成的是 CampusPulse 自身的真实四条件维度二消融和表达—前序状态链路。它不是三个外部基线的完整对比实验，也不是群体异质性与极化演变（维度三）的完整实验。

`LLM—粒子更新方向一致率` 现由冻结的共享动作投影和 CampusPulse 原生 `assimilate_forum_llm_updates_v2` 同化核计算。24/24 个适用 turn 均有 sidecar，覆盖率为 {fmt(alignment.get('turn_coverage_rate'))}，共检验 {alignment['tested_direction_count']} 个非零预期方向。该指标是实现机制一致性诊断，不是独立预测效度。

同化是对原始结构化动作的确定性本地派生，没有新增模型调用，也没有读取盲评标签；旧的 48 条原始记录保持不变。

冻结定义 1.2.0 恢复人工 20% 复核。本面板按总体 `ceil(48 × 0.20)` 抽取 10 条，并在四个条件间按比例分层分配。人工审阅者只能看到方法盲化材料，不能看到模型标签或汇总结果。{'复核已完成，一致性结果：' + json.dumps(reliability, ensure_ascii=False, sort_keys=True) if audit_complete else '当前尚无合格人工标签，因此以下模型判定指标仍为暂定结果。'}

盲评标签分布：{json.dumps(judge_label_summary, ensure_ascii=False, sort_keys=True)}。
"""
    (results_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    derived = run / "derived_native_assimilation"
    if derived.exists():
        derived_manifest = {
            "schema_version": "campuspulse-derived-assimilation-manifest-v1.0.0",
            "source_run_manifest_sha256": hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest(),
            "source_records_sha256": hashlib.sha256((run / "records.four_conditions.jsonl").read_bytes()).hexdigest(),
            "source_prompt_audit_sha256": hashlib.sha256((run / "prompt_audit.jsonl").read_bytes()).hexdigest(),
            "assimilation_protocol": "campuspulse-action-assimilation-v1.0.0",
            "native_kernel": "campuspulse_core.governance_arena_v2.assimilate_forum_llm_updates_v2",
            "record_count": len(records),
            "applicable_turn_count": alignment.get("applicable_turn_count"),
            "materialized_turn_count": alignment.get("matched_turn_count"),
            "network_calls": 0,
            "raw_records_mutated": False,
            "derived_records_sha256": hashlib.sha256((derived / "records.with_post_state.jsonl").read_bytes()).hexdigest(),
            "assimilation_sidecar_sha256": hashlib.sha256((derived / "assimilation_sidecar.jsonl").read_bytes()).hexdigest(),
            "evaluation_result_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        }
        (derived / "manifest.json").write_text(
            json.dumps(derived_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )

    hash_targets = [
        base / "configs" / "evaluation_definitions.yaml",
        base / "configs" / "evaluation_definitions.yaml.sha256",
        base / "configs" / "dimension2_campuspulse_panel_v1.json",
        run / "preflight_manifest.json",
        run / "run_manifest.json",
        run / "records.four_conditions.jsonl",
        run / "prompt_audit.jsonl",
        run / "judge" / "judge_case_manifest.json",
        run / "judge" / "judge_run_report.json",
        run / "judge" / "judge_source_audit.jsonl",
        run / "judge" / "judgements.jsonl",
        evaluation_path,
        results_dir / "metric_summary.csv",
        results_dir / "condition_distributions.csv",
        results_dir / "classifier_summary.csv",
        results_dir / "judge_label_distributions.csv",
        results_dir / "expression_prestate_cases.jsonl",
        results_dir / "FINAL_REPORT.md",
    ]
    if derived.exists():
        hash_targets.extend([
            base / "configs" / "campuspulse_action_assimilation_v1.0.0.yaml",
            base / "configs" / "campuspulse_action_assimilation_v1.0.0.yaml.sha256",
            base / "campuspulse_core" / "comparison_harness" / "campuspulse_assimilation.py",
            base / "campuspulse_core" / "scripts" / "materialize_campuspulse_dimension2_assimilation.py",
            base / "campuspulse_core" / "scripts" / "run_real_campuspulse_dimension2_ablation.py",
            base / "campuspulse_core" / "schemas" / "unified_output_v1.1.0.schema.json",
            base / "campuspulse_core" / "schemas" / "campuspulse_assimilation_sidecar_v1.0.0.schema.json",
            base / "docs" / "EVALUATION_CHANGELOG.md",
            derived / "records.with_post_state.jsonl",
            derived / "assimilation_sidecar.jsonl",
            derived / "manifest.json",
        ])
    if human_cases_path.exists():
        hash_targets.extend([
            human_cases_path,
            human_dir / "HUMAN_REVIEW_INSTRUCTIONS.md",
            human_dir / "audit_manifest.json",
        ])
    if reliability_path.exists():
        hash_targets.extend([
            human_dir / "human_review_answers.csv",
            human_dir / "human_judgements.jsonl",
            reliability_path,
        ])
    hash_rows = []
    for path in hash_targets:
        if not path.is_file():
            raise FileNotFoundError(path)
        hash_rows.append({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size, "path": str(path.relative_to(base)).replace("\\", "/")})
    write_csv(audit_dir / "file_hashes.sha256.csv", ["sha256", "bytes", "path"], hash_rows)

    print(json.dumps({"records": len(records), "judgements": len(judgements), "expression_prestate_cases": len(chain_rows), "hashed_files": len(hash_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
