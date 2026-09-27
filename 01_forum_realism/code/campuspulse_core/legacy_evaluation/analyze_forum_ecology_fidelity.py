#!/usr/bin/env python3
"""Compare ForumTwin public worlds with the frozen full-corpus ecology release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from envs.campus_pulse_governance.forum_realism import (  # noqa: E402
    build_calibrated_realism_benchmark,
    build_governance_decision_map,
    build_realism_benchmark,
    canonical_sha256,
    extract_public_worlds,
    file_sha256,
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _deduplicate_worlds(worlds):
    """Do not count the same sealed branch twice through two evidence envelopes."""

    unique = []
    seen = set()
    for world in worlds:
        key = (
            world.scenario_id,
            world.branch,
            world.seed,
            canonical_sha256(dict(world.payload)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(world)
    return unique


def _report(result: Mapping[str, Any]) -> str:
    reference = result["reference"]
    simulation = result["simulation"]
    comparisons = result["comparisons"]
    gaps = result["observed_gaps"]
    order = ("zero", "small", "medium", "large")
    labels = {"zero": "零回复", "small": "1–3 条", "medium": "4–12 条", "large": "13 条以上"}
    rows = "\n".join(
        f"| {labels[key]} | {_percent(reference['thread_size_distribution'][key])} | "
        f"{_percent(simulation['thread_size_distribution'][key])} |"
        for key in order
    )
    gap_text = "\n".join(f"- `{gap}`" for gap in gaps) or "- 未发现结构性缺口。"
    return f"""# ForumTwin 论坛生态真实性基准

状态：**{result['status']}**  
基准哈希：`{result['benchmark_sha256']}`

本报告只比较脱敏聚合结构。它不会读取或输出原始论坛正文、源 ID、私有 Feed 或微观 Agent 身份。

## 1. 线程结构

| 回复规模 | 小喇叭全量生态 | ForumTwin 可比运行 |
|---|---:|---:|
{rows}

- 真实语料中出现回复的线程占比：{_percent(reference['reply_bearing_thread_share'])}
- 模拟中出现回复的线程占比：{_percent(simulation['reply_bearing_thread_share'])}
- 回复规模 JS divergence：{comparisons['thread_size_js_divergence']:.3f}（门槛 ≤ 0.15）
- 回复发生率相对误差：{_percent(comparisons['reply_bearing_thread_relative_error'])}（门槛 ≤ 20%）

## 2. 激活人群与互动

- 激活 Profile 回复功能 JS divergence：{comparisons['activated_profile_reply_function_js_divergence']:.3f}（门槛 ≤ 0.15）
- 点赞：{simulation['interaction_counts']['like']}
- 转发：{simulation['interaction_counts']['repost']}
- 举报：{simulation['interaction_counts']['report']}

当前生态 release 没有可对齐的真实点赞、转发和举报总体分布，因此这里只检查模拟是否实际产生这些行为，不把目标比例写死。

## 3. 当前发现

{gap_text}

优先校准顺序：先修正“热榜反馈回路与最新帖发现”的回复集中问题，再基于真实可用字段校准独立互动；不得通过补造互动数使图表好看。

## 4. 边界

""" + "\n".join(f"- {item}" for item in result["limitations"]) + "\n"


def _governance_report(result: Mapping[str, Any]) -> str:
    cards = result["branch_decision_cards"]
    table_rows = []
    for branch, card in cards.items():
        table_rows.append(
            f"| {branch} | {card['governance_message_count']} | "
            f"{card['messages_with_direct_uptake']} | "
            f"{_percent(card['direct_uptake_rate'])} | "
            f"{card['direct_resident_response_count']} |"
        )
    examples = []
    for world in result["worlds"]:
        for entry in world["entries"]:
            if not entry["direct_response_examples"]:
                continue
            response = entry["direct_response_examples"][0]
            examples.append(
                f"### {world['branch']} · Tick {entry['created_tick']} · "
                f"{entry['uptake_status']}\n\n"
                f"治理消息：{entry['visible_text']}\n\n"
                f"居民直接回应：{response['visible_text']}\n"
            )
            if len(examples) == 5:
                break
        if len(examples) == 5:
            break
    gaps = "\n".join(
        f"- **{branch}**：{card['operational_gap']}"
        for branch, card in cards.items()
    )
    return f"""# ForumTwin 治理承接与决策地图

决策地图哈希：`{result['decision_map_sha256']}`

本报告回答“治理对象发出后，居民是否真的直接回复或引用、接受还是质疑”。同线程但没有父子或引用关系的后续发言不计入承接。

## 1. 分支承接概览

| 分支 | 治理消息 | 被直接承接 | 承接率 | 居民直接回应 |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 2. 可操作缺口

{gaps}

## 3. 代表性承接

{chr(10).join(examples) if examples else '当前输入中没有严格直接承接样本。'}

## 4. 使用边界

{result['causal_boundary']}

{result['comparability_boundary']}
"""


def _calibrated_report(result: Mapping[str, Any]) -> str:
    reference = result["reference"]
    simulation = result["mature_simulation"]
    comparisons = result["comparisons"]
    labels = {"zero": "零回复", "small": "1–3 条", "medium": "4–12 条", "large": "13 条以上"}
    rows = "\n".join(
        f"| {labels[key]} | {_percent(reference['thread_size_distribution'][key])} | "
        f"{_percent(simulation['thread_size_distribution'][key])} |"
        for key in ("zero", "small", "medium", "large")
    )
    gates = {
        **result["absolute_gates"],
        **result["improvement_gates"],
    }
    gate_rows = "\n".join(
        f"- {name}: " + ("不适用" if value is None else "通过" if value else "未通过")
        for name, value in gates.items()
    )
    return f"""# ForumTwin 成熟线程生态校准

状态：**{result['status']}**
基准哈希：`{result['benchmark_sha256']}`

成熟线程定义：`{simulation['definition']}`。这避免把刚创建、尚无机会获得回复的根帖误判为零回复。

| 回复规模 | 小喇叭全量生态 | ForumTwin 成熟线程 |
|---|---:|---:|
{rows}

- 成熟线程数：{simulation['thread_count']}
- 真实回复线程占比：{_percent(reference['reply_bearing_thread_share'])}
- 模拟成熟线程回复占比：{_percent(simulation['reply_bearing_thread_share'])}
- 线程规模 JS divergence：{comparisons['mature_thread_size_js_divergence']}
- 回复占比绝对误差：{comparisons['mature_reply_bearing_absolute_error']}

## 冻结门禁

{gate_rows}

## 解释边界

{result['interpretation']}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecology", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-semantic-spine", action="store_true")
    parser.add_argument(
        "--baseline-benchmark",
        type=Path,
        help="Optional frozen v1 benchmark used for pre/post improvement gates.",
    )
    parser.add_argument("--mature-min-age-ticks", type=int, default=2)
    args = parser.parse_args()
    ecology = _load(args.ecology)
    worlds = []
    governance_worlds = []
    inputs = [
        {"kind": "ecology", "path": args.ecology.name, "sha256": file_sha256(args.ecology)}
    ]
    for path in args.result:
        inputs.append({"kind": "result", "path": path.name, "sha256": file_sha256(path)})
        payload = _load(path)
        worlds.extend(
            extract_public_worlds(
                payload,
                source_label=path.stem,
                include_semantic_spine=args.include_semantic_spine,
            )
        )
        governance_worlds.extend(
            extract_public_worlds(
                payload,
                source_label=path.stem,
                include_semantic_spine=True,
            )
        )
    worlds = _deduplicate_worlds(worlds)
    governance_worlds = _deduplicate_worlds(governance_worlds)
    result = build_realism_benchmark(ecology=ecology, worlds=worlds, input_files=inputs)
    baseline = _load(args.baseline_benchmark) if args.baseline_benchmark else None
    if args.baseline_benchmark:
        inputs.append(
            {
                "kind": "baseline_benchmark",
                "path": args.baseline_benchmark.name,
                "sha256": file_sha256(args.baseline_benchmark),
            }
        )
    calibrated = build_calibrated_realism_benchmark(
        ecology=ecology,
        worlds=worlds,
        input_files=inputs,
        baseline_benchmark=baseline,
        mature_min_age_ticks=args.mature_min_age_ticks,
    )
    governance = build_governance_decision_map(
        worlds=governance_worlds, input_files=inputs
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "forum-realism-benchmark-v1.json"
    report_path = args.output_dir / "FORUM_REALISM_BENCHMARK.md"
    governance_json_path = args.output_dir / "governance-decision-map-v1.json"
    governance_report_path = args.output_dir / "GOVERNANCE_DECISION_MAP.md"
    calibrated_json_path = args.output_dir / "forum-realism-benchmark-v2.json"
    calibrated_report_path = args.output_dir / "FORUM_REALISM_CALIBRATION.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_report(result), encoding="utf-8")
    governance_json_path.write_text(
        json.dumps(governance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    governance_report_path.write_text(
        _governance_report(governance), encoding="utf-8"
    )
    calibrated_json_path.write_text(
        json.dumps(calibrated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibrated_report_path.write_text(
        _calibrated_report(calibrated), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "world_count": result["simulation"]["world_count"],
        "benchmark_sha256": result["benchmark_sha256"],
        "json": str(json_path),
        "report": str(report_path),
        "governance_decision_map_sha256": governance["decision_map_sha256"],
        "governance_json": str(governance_json_path),
        "governance_report": str(governance_report_path),
        "calibrated_status": calibrated["status"],
        "calibrated_benchmark_sha256": calibrated["benchmark_sha256"],
        "calibrated_json": str(calibrated_json_path),
        "calibrated_report": str(calibrated_report_path),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
