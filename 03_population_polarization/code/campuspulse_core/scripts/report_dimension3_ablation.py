from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import read_jsonl, write_json  # noqa: E402
from evaluation.dimension3 import PRIMARY_TICKS, VARIANTS, evaluate_ablation  # noqa: E402


MICRO_FIELDS = ("trust", "satisfaction", "attention", "concern")
VARIANT_LABELS = {
    "full": "完整输入（Profile + 10 粒子 SMC）",
    "single_particle": "单粒子点估计（Profile + 1 粒子，无重采样/扩散）",
    "no_profile": "无 Profile（10 粒子 SMC）",
    "no_profile_no_smc": "双重消融（无 Profile，状态不向模型暴露）",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def load_inputs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for path in paths:
        current = read_jsonl(path)
        rows.extend(current)
        inventory.append({"path": str(path.resolve()), "sha256": sha256(path), "rows": len(current)})
    keys = [(row.get("variant"), row.get("seed"), row.get("tick"), row.get("agent_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate variant/seed/tick/agent rows across input files")
    if any(row.get("fake_marker") for row in rows):
        raise ValueError("conformance/fake records cannot be used in the formal ablation report")
    return rows, inventory


def render(report: dict[str, Any], inventory: list[dict[str, Any]]) -> str:
    groups = {(row["variant"], int(row["tick"])): row for row in report["groups"]}
    lines = [
        "# 维度三纵向机制消融实验汇总",
        "",
        "口径：每个单元为 1000 名等权父智能体的决策后 12 维状态；粒子条件先在父智能体内按权重求均值，禁止将 10000 个粒子当作 10000 人。总体标准差采用 ddof=0，利益组检验采用 199 次单侧标签置换。",
        "",
        "## 宏观结构指标",
        "",
        "| 条件 | Tick | 12维状态有效秩 | 利益组多元 R² | 置换 p |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for tick in PRIMARY_TICKS:
            row = groups[(variant, tick)]
            lines.append(f"| {VARIANT_LABELS[variant]} | {tick} | {fmt(row['state_matrix_effective_rank'])} | {fmt(row['stake_group_multivariate_r_squared_12d'])} | {fmt(row['stake_group_permutation_p'], 3)} |")
    lines += ["", "## 微观连续特征", ""]
    for field in MICRO_FIELDS:
        lines += [
            f"### {field}", "",
            "| 条件 | Tick | 均值 | 标准差 | Sarle BC | KDE峰数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for variant in VARIANTS:
            for tick in PRIMARY_TICKS:
                values = groups[(variant, tick)]["state_dimensions"][field]
                lines.append(f"| {VARIANT_LABELS[variant]} | {tick} | {fmt(values['mean'])} | {fmt(values['sd_ddof_0'])} | {fmt(values['sarle_bimodality'])} | {fmt(values['kde_peak_count'], 0)} |")
        lines.append("")
    lines += [
        "### 净立场（support − opposition）", "",
        "| 条件 | Tick | 均值 | 标准差 | Sarle BC | KDE峰数 | 20箱熵 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for tick in PRIMARY_TICKS:
            values = groups[(variant, tick)]["net_stance"]
            entropy = values.get("entropy")
            lines.append(f"| {VARIANT_LABELS[variant]} | {tick} | {fmt(values['mean'])} | {fmt(values['sd_ddof_0'])} | {fmt(values['sarle_bimodality'])} | {fmt(values['kde_peak_count'], 0)} | {fmt(None if entropy is None else entropy['raw'])} |")
    lines += ["", "## 输入来源", ""]
    lines.extend(f"- `{item['path']}`：{item['rows']} 行，SHA-256 `{item['sha256']}`" for item in inventory)
    lines += ["", "该报告只接受完整的 4 条件 × 3 Tick × 1000 人网格；缺失、重复、伪造或 conformance 数据均会使程序失败。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and report the frozen CampusPulse dimension-3 ablation grid.")
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    rows, inventory = load_inputs(args.records)
    seeds = {int(row["seed"]) for row in rows}
    if seeds != {args.seed}:
        raise ValueError(f"expected only seed {args.seed}, got {sorted(seeds)}")
    result = evaluate_ablation(rows, permutation_count=199, expected_agent_count=1000)
    args.output_root.mkdir(parents=True, exist_ok=False)
    result["input_inventory"] = inventory
    result["variant_contract"] = VARIANT_LABELS
    json_path = args.output_root / "dimension3_ablation_metrics.json"
    write_json(json_path, result)
    markdown_path = args.output_root / "dimension3_ablation_report.md"
    markdown_path.write_text(render(result, inventory), encoding="utf-8", newline="\n")
    manifest = {
        "status": "passed_complete_formal_grid",
        "input_rows": len(rows),
        "expected_rows": 4 * 3 * 1000,
        "input_inventory": inventory,
        "output_sha256": {"metrics": sha256(json_path), "report": sha256(markdown_path)},
        "network_calls": 0,
    }
    write_json(args.output_root / "report_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
