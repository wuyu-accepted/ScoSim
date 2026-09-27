#!/usr/bin/env python3
"""Recompute Dimensions 1 and 4 from frozen, privacy-safe sufficient statistics.

This entry point is deliberately read-only with respect to its inputs and has
no model/provider dependency.  It does not replay the simulation.  Instead it
recomputes the publication statistics that are identifiable from the frozen
public evidence, writes results to a separate directory, and verifies that all
input bytes are unchanged before exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = (
    ROOT
    / "scenario"
    / "housing_allocation"
    / "reference_evidence"
    / "four-dimension-v5-formal-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _distribution(row: Mapping[str, Any], order: tuple[str, ...]) -> list[float]:
    values = [float(row[name]) for name in order]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("distribution contains an invalid value")
    total = sum(values)
    if total <= 0:
        raise ValueError("distribution has zero mass")
    return [value / total for value in values]


def _js_divergence(left: list[float], right: list[float], *, base: float) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("JS inputs must have equal non-zero length")
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]

    if base <= 0 or base == 1:
        raise ValueError("JS logarithm base must be positive and not equal to one")

    def kl(values: list[float]) -> float:
        return sum(
            value * (math.log(value / middle) / math.log(base))
            for value, middle in zip(values, midpoint)
            if value > 0
        )

    return 0.5 * kl(left) + 0.5 * kl(right)


def recompute_dimension1(
    public_summary: Mapping[str, Any], ecology_benchmark: Mapping[str, Any]
) -> dict[str, Any]:
    forum = dict((public_summary.get("four_dimensions") or {}).get("forum_ecology") or {})
    simulation = dict(forum.get("mature_simulation") or {})
    reference = dict(ecology_benchmark.get("reference") or {})
    order = ("zero", "small", "medium", "large")
    simulated_distribution = _distribution(
        dict(simulation.get("thread_size_distribution") or {}), order
    )
    reference_distribution = _distribution(
        dict(reference.get("thread_size_distribution") or {}), order
    )
    simulated_reply_share = float(simulation["reply_bearing_thread_share"])
    reference_reply_share = float(reference["reply_bearing_thread_share"])
    result = {
        "schema_version": "campus-pulse-dimension1-readonly-recompute-v1",
        "source_kind": "frozen_privacy_safe_sufficient_statistics",
        "not_a_simulation_replay": True,
        "mature_thread_definition": str(simulation.get("definition") or ""),
        "simulation_world_count": int(simulation["world_count"]),
        "simulation_mature_thread_count": int(simulation["thread_count"]),
        "reference_thread_count": int(reference["episode_count"]),
        "bucket_order": list(order),
        "simulation_distribution": dict(zip(order, map(_rounded, simulated_distribution))),
        "reference_distribution": dict(zip(order, map(_rounded, reference_distribution))),
        "simulation_reply_bearing_share": _rounded(simulated_reply_share),
        "reference_reply_bearing_share": _rounded(reference_reply_share),
        "reply_bearing_absolute_error": _rounded(
            abs(simulated_reply_share - reference_reply_share)
        ),
        "thread_size_js_divergence_base2": _rounded(
            _js_divergence(simulated_distribution, reference_distribution, base=2.0)
        ),
        "thread_size_js_divergence_natural_log": _rounded(
            _js_divergence(
                simulated_distribution, reference_distribution, base=math.e
            )
        ),
    }
    return result


def recompute_dimension4(public_summary: Mapping[str, Any]) -> dict[str, Any]:
    branches = dict(public_summary.get("branches") or {})
    natural = dict(branches.get("Natural") or {})
    intervention = dict(branches.get("D") or {})
    if not natural or not intervention:
        raise ValueError("public summary must contain Natural and D branches")

    def values(branch: Mapping[str, Any]) -> dict[str, float]:
        risk = dict(branch.get("risk_signals") or {})
        population = dict(branch.get("population_state") or {})
        return {
            "public_message_count": float(branch["message_count"]),
            "direct_reply_count": float(branch["reply_count"]),
            "unsupported_claim_velocity": float(risk["unsupported_claim_velocity"]),
            "help_backlog_and_sla_risk": float(risk["help_backlog_and_sla_breach"]),
            "cross_group_exposure_gap": float(risk["cross_group_exposure_gap"]),
            "population_trust": float(population["trust"]),
        }

    n_values = values(natural)
    d_values = values(intervention)
    uptake_summary = dict((public_summary.get("governance_uptake") or {}).get("summary") or {})
    governance_messages = int(uptake_summary["governance_message_count"])
    messages_with_uptake = int(uptake_summary["messages_with_direct_uptake"])
    if governance_messages <= 0 or not 0 <= messages_with_uptake <= governance_messages:
        raise ValueError("governance uptake counts are invalid")
    direct_uptake = messages_with_uptake / governance_messages
    return {
        "schema_version": "campus-pulse-dimension4-readonly-recompute-v1",
        "comparison_design": "paired_branch_final_value_difference",
        "not_standard_difference_in_differences": True,
        "shared_through_tick": int((public_summary.get("scenario") or {})["shared_through_tick"]),
        "Natural": {key: _rounded(value) for key, value in n_values.items()},
        "D": {
            **{key: _rounded(value) for key, value in d_values.items()},
            "direct_governance_question_rate": _rounded(direct_uptake),
            "direct_governance_question_numerator": messages_with_uptake,
            "direct_governance_question_denominator": governance_messages,
        },
        "D_minus_Natural": {
            key: _rounded(d_values[key] - n_values[key]) for key in n_values
        },
        "direct_governance_question_rate_difference": None,
        "direct_governance_question_rate_note": (
            "Natural has no governance messages, so the rate is undefined there "
            "and no ordinary D-Natural difference is reported."
        ),
    }


def _report(result: Mapping[str, Any]) -> str:
    d1 = result["dimension1"]
    d4 = result["dimension4"]
    lines = [
        "# 维度一与维度四只读复算报告",
        "",
        "本报告从冻结的隐私安全充分统计量复算，不执行模拟、不访问网络、不调用模型。",
        "",
        "## 维度一：论坛生态真实性",
        "",
        "| 指标 | 复算值 |",
        "|---|---:|",
        f"| 成熟模拟线程数 | {d1['simulation_mature_thread_count']} |",
        f"| 真实参照线程数 | {d1['reference_thread_count']} |",
        f"| 模拟有回复比例 | {d1['simulation_reply_bearing_share']:.6f} |",
        f"| 真实有回复比例 | {d1['reference_reply_bearing_share']:.6f} |",
        f"| 有回复比例绝对误差 | {d1['reply_bearing_absolute_error']:.6f} |",
        f"| 四档规模 JS 散度（base 2，论文主口径） | {d1['thread_size_js_divergence_base2']:.6f} |",
        f"| 四档规模 JS 散度（自然对数，敏感性值） | {d1['thread_size_js_divergence_natural_log']:.6f} |",
        "",
        "## 维度四：治理效能与次生风险",
        "",
        "| 指标 | Natural | D | D−Natural |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "public_message_count": "公开消息数",
        "direct_reply_count": "直接回复数",
        "unsupported_claim_velocity": "无依据主张流速",
        "help_backlog_and_sla_risk": "服务积压/SLA 风险",
        "cross_group_exposure_gap": "跨群曝光差",
        "population_trust": "群体信任度",
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | {d4['Natural'][key]:.6f} | {d4['D'][key]:.6f} | "
            f"{d4['D_minus_Natural'][key]:+.6f} |"
        )
    lines.extend(
        [
            f"| 干预直接质询率 | 不适用 | {d4['D']['direct_governance_question_rate']:.6f} "
            f"({d4['D']['direct_governance_question_numerator']}/{d4['D']['direct_governance_question_denominator']}) | 不作差值 |",
            "",
            "维度四的差值是共享前缀后的配对分支末值差，不是标准双重差分估计。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-summary",
        type=Path,
        default=DEFAULT_EVIDENCE / "public-summary-v1.json",
    )
    parser.add_argument(
        "--ecology-benchmark",
        type=Path,
        default=DEFAULT_EVIDENCE / "forum-ecology-benchmark-v2.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = (args.public_summary.resolve(), args.ecology_benchmark.resolve())
    output_dir = args.output_dir.resolve()
    if any(output_dir == path or output_dir in path.parents for path in inputs):
        raise ValueError("output directory must not contain or replace an input file")
    before = {str(path): _sha256(path) for path in inputs}
    public_summary = _load(inputs[0])
    ecology_benchmark = _load(inputs[1])
    result: dict[str, Any] = {
        "schema_version": "campus-pulse-dimensions-1-4-readonly-recompute-v1",
        "dimension1": recompute_dimension1(public_summary, ecology_benchmark),
        "dimension4": recompute_dimension4(public_summary),
        "inputs": [
            {"path": str(path), "sha256": before[str(path)]} for path in inputs
        ],
        "network_calls": 0,
        "model_calls": 0,
    }
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["result_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dimensions-1-4-readonly-recompute-v1.json"
    report_path = output_dir / "DIMENSIONS_1_4_READONLY_REPORT.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_report(result), encoding="utf-8")
    after = {str(path): _sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("an input changed during read-only recomputation")
    print(json.dumps({
        "status": "pass",
        "json": str(json_path),
        "report": str(report_path),
        "result_sha256": result["result_sha256"],
        "inputs_unchanged": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
