"""Deterministic uncertainty sampling and policy-robustness analysis."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import ScenarioConfig
from .experiment import run_comparison
from .policies import POLICIES, Policy
from .provenance import load_data_lineage


UNCERTAINTY_DIMENSIONS = {
    "demand_shock_scale": {
        "low": 0.80,
        "high": 1.30,
        "source_type": "stress_range",
        "target": "base_arrival_rate",
        "label": "需求冲击强度",
        "rationale": "检验集中诉求规模高于或低于基准时的策略稳定性。",
    },
    "urgency_pressure_scale": {
        "low": 0.75,
        "high": 1.35,
        "source_type": "assumption_range",
        "target": "urgency_mix.high+critical",
        "label": "高紧急诉求占比",
        "rationale": "现有数据缺少人工紧急度标签，因此使用宽区间压力测试。",
    },
    "natural_resolution_scale": {
        "low": 0.60,
        "high": 1.40,
        "source_type": "assumption_range",
        "target": "natural_resolution_probability",
        "label": "自然回应解决能力",
        "rationale": "论坛评论是否真正解决问题不可观测，采用宽区间。",
    },
    "faq_effectiveness_scale": {
        "low": 0.50,
        "high": 1.50,
        "source_type": "assumption_range",
        "target": "faq_effectiveness",
        "label": "FAQ有效性",
        "rationale": "无历史FAQ干预记录，允许效果从基准的一半到一点五倍。",
    },
    "peer_success_scale": {
        "low": 0.50,
        "high": 1.50,
        "source_type": "assumption_range",
        "target": "peer_boost_success",
        "label": "同伴补位有效性",
        "rationale": "无同伴补位实验日志，使用对称宽区间。",
    },
    "department_capacity_scale": {
        "low": 0.70,
        "high": 1.30,
        "source_type": "operator_range",
        "target": "department_capacity",
        "label": "部门基础容量",
        "rationale": "覆盖人员短缺至临时增配的运营范围。",
    },
    "faq_cost_scale": {
        "low": 0.70,
        "high": 1.60,
        "source_type": "operator_range",
        "target": "cost_weights.faq_reach",
        "label": "FAQ单位成本",
        "rationale": "无量纲成本权重尚未由试点财务数据校准。",
    },
    "peer_cost_scale": {
        "low": 0.70,
        "high": 1.60,
        "source_type": "operator_range",
        "target": "cost_weights.peer_action",
        "label": "同伴行动成本",
        "rationale": "覆盖志愿者协调和专业督导投入差异。",
    },
    "dynamic_cost_scale": {
        "low": 0.70,
        "high": 1.60,
        "source_type": "operator_range",
        "target": "cost_weights.dynamic_capacity_unit",
        "label": "动态扩容成本",
        "rationale": "覆盖低成本排班调整至高成本临时增援。",
    },
}


POLICY_LABELS = {policy_id: policy.label for policy_id, policy in POLICIES.items()}


def latin_hypercube_samples(samples: int, seed: int) -> list[dict[str, float]]:
    """Generate deterministic stratified samples over every declared dimension."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    rows = [dict() for _ in range(samples)]
    for name, definition in UNCERTAINTY_DIMENSIONS.items():
        positions = [(index + rng.random()) / samples for index in range(samples)]
        rng.shuffle(positions)
        low = float(definition["low"])
        high = float(definition["high"])
        for row, position in zip(rows, positions):
            row[name] = low + position * (high - low)
    return rows


def _scaled_probabilities(values: dict[str, float], scale: float) -> dict[str, float]:
    return {key: min(0.99, max(0.0, value * scale)) for key, value in values.items()}


def _scaled_urgency_mix(values: dict[str, float], scale: float) -> dict[str, float]:
    adjusted = dict(values)
    adjusted["high"] *= scale
    adjusted["critical"] *= scale
    total = sum(adjusted.values())
    return {key: value / total for key, value in adjusted.items()}


def apply_uncertainty_sample(
    config: ScenarioConfig,
    sample: dict[str, float],
    sample_index: int,
) -> ScenarioConfig:
    """Return a validated scenario variant while preserving evidence metadata."""

    missing = sorted(set(UNCERTAINTY_DIMENSIONS) - set(sample))
    if missing:
        raise ValueError(f"uncertainty sample is missing: {', '.join(missing)}")
    raw = copy.deepcopy(config.raw)
    parameters = raw["parameters"]
    parameters["base_arrival_rate"]["value"] = (
        config.get("base_arrival_rate") * sample["demand_shock_scale"]
    )
    parameters["urgency_mix"]["value"] = _scaled_urgency_mix(
        config.get("urgency_mix"), sample["urgency_pressure_scale"]
    )
    parameters["natural_resolution_probability"]["value"] = _scaled_probabilities(
        config.get("natural_resolution_probability"), sample["natural_resolution_scale"]
    )
    parameters["faq_effectiveness"]["value"] = _scaled_probabilities(
        config.get("faq_effectiveness"), sample["faq_effectiveness_scale"]
    )
    parameters["peer_boost_success"]["value"] = _scaled_probabilities(
        config.get("peer_boost_success"), sample["peer_success_scale"]
    )
    parameters["department_capacity"]["value"] = {
        key: max(1, int(round(value * sample["department_capacity_scale"])))
        for key, value in config.get("department_capacity").items()
    }
    costs = dict(config.get("cost_weights"))
    costs["faq_reach"] *= sample["faq_cost_scale"]
    costs["peer_action"] *= sample["peer_cost_scale"]
    costs["dynamic_capacity_unit"] *= sample["dynamic_cost_scale"]
    parameters["cost_weights"]["value"] = costs
    raw["variant"] = {
        "kind": "robustness_sample",
        "sample_index": sample_index,
        "base_scenario_fingerprint": config.fingerprint(),
        "uncertainty": sample,
    }
    return ScenarioConfig(raw)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def _threshold_curve(
    compact_samples: list[dict],
    dimension: str,
    policy_id: str,
    bins: int = 4,
) -> list[dict]:
    ordered = sorted(compact_samples, key=lambda row: row["parameters"][dimension])
    curve = []
    for bin_index in range(bins):
        start = round(bin_index * len(ordered) / bins)
        end = round((bin_index + 1) * len(ordered) / bins)
        rows = ordered[start:end]
        if not rows:
            continue
        uplifts = [
            (row["policies"][policy_id]["resolution_rate"] - row["policies"]["baseline"]["resolution_rate"])
            * 100
            for row in rows
        ]
        cost_deltas = [
            row["policies"][policy_id]["governance_cost"]
            - row["policies"]["baseline"]["governance_cost"]
            for row in rows
        ]
        curve.append(
            {
                "bin": bin_index + 1,
                "sample_count": len(rows),
                "parameter_min": min(row["parameters"][dimension] for row in rows),
                "parameter_max": max(row["parameters"][dimension] for row in rows),
                "parameter_mean": statistics.fmean(row["parameters"][dimension] for row in rows),
                "resolution_uplift_mean_pp": statistics.fmean(uplifts),
                "beats_baseline_resolution_rate": sum(value > 0 for value in uplifts) / len(uplifts),
                "governance_cost_delta_mean": statistics.fmean(cost_deltas),
            }
        )
    return curve


def run_robustness_analysis(
    config: ScenarioConfig,
    samples: int = 120,
    repetitions: int = 3,
    seed: int = 20260721,
    policies: Iterable[Policy] | None = None,
) -> dict:
    """Run policy comparisons over deterministic uncertainty samples."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    selected = list(policies) if policies is not None else list(POLICIES.values())
    policy_ids = [policy.policy_id for policy in selected]
    if "baseline" not in policy_ids:
        raise ValueError("baseline policy is required for robustness regret")
    if "combined" not in policy_ids:
        raise ValueError("combined policy is required for robustness sensitivity analysis")

    sample_parameters = latin_hypercube_samples(samples, seed)
    compact_samples: list[dict] = []
    policy_values = {
        policy_id: {
            "resolution_rate": [],
            "urgent_unresolved": [],
            "governance_cost": [],
            "fairness_gap": [],
            "peak_backlog": [],
            "rank": [],
            "resolution_regret_pp": [],
            "urgent_regret": [],
            "cost_regret": [],
        }
        for policy_id in policy_ids
    }
    wins = {policy_id: 0 for policy_id in policy_ids}
    pareto_hits = {policy_id: 0 for policy_id in policy_ids}
    common_inputs_verified = True

    for sample_index, parameters in enumerate(sample_parameters):
        variant = apply_uncertainty_sample(config, parameters, sample_index)
        comparison = run_comparison(
            variant,
            repetitions=repetitions,
            base_seed=seed,
            policies=selected,
        )
        common_inputs_verified = common_inputs_verified and comparison["common_inputs_verified"]
        wins[comparison["ranking"][0]] += 1
        for policy_id in comparison["pareto_policies"]:
            pareto_hits[policy_id] += 1

        policy_metrics = {}
        for policy_id in policy_ids:
            aggregate = comparison["aggregate"][policy_id]
            policy_metrics[policy_id] = {
                "resolution_rate": aggregate["operational_resolution_rate_mean"],
                "urgent_unresolved": aggregate["urgent_unresolved_demand_mean"],
                "governance_cost": aggregate["governance_cost_mean"],
                "fairness_gap": aggregate["coverage_fairness_gap_mean"],
                "peak_backlog": aggregate["peak_backlog_mean"],
            }
        best_resolution = max(row["resolution_rate"] for row in policy_metrics.values())
        best_urgent = min(row["urgent_unresolved"] for row in policy_metrics.values())
        lowest_cost = min(row["governance_cost"] for row in policy_metrics.values())
        rank_lookup = {policy_id: rank for rank, policy_id in enumerate(comparison["ranking"], start=1)}
        for policy_id, row in policy_metrics.items():
            values = policy_values[policy_id]
            values["resolution_rate"].append(row["resolution_rate"])
            values["urgent_unresolved"].append(row["urgent_unresolved"])
            values["governance_cost"].append(row["governance_cost"])
            values["fairness_gap"].append(row["fairness_gap"])
            values["peak_backlog"].append(row["peak_backlog"])
            values["rank"].append(float(rank_lookup[policy_id]))
            values["resolution_regret_pp"].append((best_resolution - row["resolution_rate"]) * 100)
            values["urgent_regret"].append(row["urgent_unresolved"] - best_urgent)
            values["cost_regret"].append(row["governance_cost"] - lowest_cost)
        compact_samples.append(
            {
                "sample_index": sample_index,
                "parameters": {key: round(value, 6) for key, value in parameters.items()},
                "winner": comparison["ranking"][0],
                "ranking": comparison["ranking"],
                "pareto_policies": comparison["pareto_policies"],
                "policies": policy_metrics,
            }
        )

    policy_summary = {}
    for policy_id in policy_ids:
        values = policy_values[policy_id]
        policy_summary[policy_id] = {
            "label": POLICY_LABELS.get(policy_id, policy_id),
            "win_frequency": wins[policy_id] / samples,
            "pareto_frequency": pareto_hits[policy_id] / samples,
            "mean_rank": statistics.fmean(values["rank"]),
            "resolution_rate": _distribution_summary(values["resolution_rate"]),
            "urgent_unresolved": _distribution_summary(values["urgent_unresolved"]),
            "governance_cost": _distribution_summary(values["governance_cost"]),
            "fairness_gap": _distribution_summary(values["fairness_gap"]),
            "peak_backlog": _distribution_summary(values["peak_backlog"]),
            "resolution_regret_pp": _distribution_summary(values["resolution_regret_pp"]),
            "urgent_regret": _distribution_summary(values["urgent_regret"]),
            "cost_regret": _distribution_summary(values["cost_regret"]),
        }

    combined_resolution_uplift = [
        (row["policies"]["combined"]["resolution_rate"] - row["policies"]["baseline"]["resolution_rate"])
        * 100
        for row in compact_samples
    ]
    combined_cost_delta = [
        row["policies"]["combined"]["governance_cost"]
        - row["policies"]["baseline"]["governance_cost"]
        for row in compact_samples
    ]
    sensitivity = []
    for name, definition in UNCERTAINTY_DIMENSIONS.items():
        xs = [row["parameters"][name] for row in compact_samples]
        sensitivity.append(
            {
                "parameter": name,
                "label": definition["label"],
                "resolution_uplift_correlation": _pearson(xs, combined_resolution_uplift),
                "cost_delta_correlation": _pearson(xs, combined_cost_delta),
            }
        )
    sensitivity.sort(key=lambda row: abs(row["resolution_uplift_correlation"]), reverse=True)

    threshold_analysis = {
        "faq_effectiveness_scale": {
            "policy_id": "faq_deflection",
            "curve": _threshold_curve(compact_samples, "faq_effectiveness_scale", "faq_deflection"),
        },
        "peer_success_scale": {
            "policy_id": "peer_support",
            "curve": _threshold_curve(compact_samples, "peer_success_scale", "peer_support"),
        },
        "department_capacity_scale": {
            "policy_id": "dynamic_capacity",
            "curve": _threshold_curve(compact_samples, "department_capacity_scale", "dynamic_capacity"),
        },
        "demand_shock_scale": {
            "policy_id": "combined",
            "curve": _threshold_curve(compact_samples, "demand_shock_scale", "combined"),
        },
    }
    robust_recommendation = sorted(
        policy_ids,
        key=lambda policy_id: (
            -policy_summary[policy_id]["win_frequency"],
            -policy_summary[policy_id]["pareto_frequency"],
            policy_summary[policy_id]["resolution_regret_pp"]["mean"],
            policy_summary[policy_id]["governance_cost"]["mean"],
            policy_id,
        ),
    )[0]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_id": config.scenario_id,
        "scenario_title": config.title,
        "scenario_version": config.version,
        "scenario_fingerprint": config.fingerprint(),
        "data_lineage": load_data_lineage(),
        "evidence_summary": {
            "parameter_count": len(config.parameters),
            "source_counts": config.source_counts(),
            "intervention_effect_validity": "not_empirically_validated",
        },
        "parameter_evidence": {
            name: {
                "source_type": parameter.source_type,
                "evidence": parameter.evidence,
                "limitations": parameter.limitations,
            }
            for name, parameter in sorted(config.parameters.items())
        },
        "methodology": {
            "sampling": "deterministic Latin hypercube",
            "samples": samples,
            "repetitions_per_policy_per_sample": repetitions,
            "seed": seed,
            "common_policy_inputs_verified": common_inputs_verified,
            "interpretation": "Uncertainty robustness analysis, not empirical intervention-effect validation.",
        },
        "uncertainty_dimensions": UNCERTAINTY_DIMENSIONS,
        "policy_summary": policy_summary,
        "robust_recommendation": robust_recommendation,
        "sensitivity": sensitivity,
        "threshold_analysis": threshold_analysis,
        "samples": compact_samples,
        "limitations": [
            "Uncertainty ranges are declared stress ranges, not posterior distributions.",
            "Win frequency means first place under the declared lexicographic ranking, not real-world causal superiority.",
            "Intervention effectiveness and cost ranges require pilot calibration.",
        ],
    }


def write_robustness_json(result: dict, output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return str(target)
