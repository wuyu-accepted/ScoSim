"""Paired policy experiments, aggregation, ranking, and export."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import ScenarioConfig
from .generator import generate_demand_stream
from .metrics import LIMITATION_STATEMENT, METRIC_DEFINITIONS, compute_metrics
from .policies import POLICIES, Policy
from .provenance import load_data_lineage
from .simulation import run_simulation


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _aggregate_runs(runs: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for run in runs:
        grouped.setdefault(run["policy_id"], []).append(_scalar_metrics(run["metrics"]))
    aggregate: dict[str, dict[str, float]] = {}
    for policy_id, rows in sorted(grouped.items()):
        keys = sorted(set.intersection(*(set(row) for row in rows))) if rows else []
        summary: dict[str, float] = {"replicates": float(len(rows))}
        for key in keys:
            values = [row[key] for row in rows]
            summary[f"{key}_mean"] = statistics.fmean(values)
            summary[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate[policy_id] = summary
    return aggregate


def _rank_policies(aggregate: dict[str, dict[str, float]]) -> list[str]:
    return sorted(
        aggregate,
        key=lambda policy_id: (
            aggregate[policy_id]["urgent_unresolved_demand_mean"],
            -aggregate[policy_id]["operational_resolution_rate_mean"],
            aggregate[policy_id]["governance_cost_mean"],
            policy_id,
        ),
    )


def _pareto_policies(aggregate: dict[str, dict[str, float]]) -> list[str]:
    non_dominated: list[str] = []
    for candidate, candidate_metrics in aggregate.items():
        dominated = False
        for challenger, challenger_metrics in aggregate.items():
            if candidate == challenger:
                continue
            no_worse = (
                challenger_metrics["urgent_unresolved_demand_mean"]
                <= candidate_metrics["urgent_unresolved_demand_mean"]
                and challenger_metrics["operational_resolution_rate_mean"]
                >= candidate_metrics["operational_resolution_rate_mean"]
                and challenger_metrics["governance_cost_mean"]
                <= candidate_metrics["governance_cost_mean"]
            )
            strictly_better = (
                challenger_metrics["urgent_unresolved_demand_mean"]
                < candidate_metrics["urgent_unresolved_demand_mean"]
                or challenger_metrics["operational_resolution_rate_mean"]
                > candidate_metrics["operational_resolution_rate_mean"]
                or challenger_metrics["governance_cost_mean"]
                < candidate_metrics["governance_cost_mean"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            non_dominated.append(candidate)
    return sorted(non_dominated)


def run_comparison(
    config: ScenarioConfig,
    repetitions: int = 10,
    base_seed: int = 20260721,
    policies: Iterable[Policy] | None = None,
) -> dict:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    selected = list(policies) if policies is not None else list(POLICIES.values())
    if not selected:
        raise ValueError("at least one policy is required")

    runs: list[dict] = []
    data_lineage = load_data_lineage()
    data_snapshot_id = data_lineage["dataset_snapshot_id"]
    fingerprints_by_rep: dict[str, list[str]] = {}
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    for replicate in range(repetitions):
        seed = base_seed + replicate
        demand_stream = generate_demand_stream(config, seed)
        fingerprints: list[str] = []
        for policy in selected:
            outcome = run_simulation(config, policy, seed, demand_stream)
            fingerprints.append(outcome.demand_fingerprint)
            runs.append(
                {
                    "scenario_id": config.scenario_id,
                    "scenario_version": config.version,
                    "scenario_fingerprint": config.fingerprint(),
                    "policy_id": policy.policy_id,
                    "policy_label": policy.label,
                    "replicate": replicate,
                    "seed": seed,
                    "demand_fingerprint": outcome.demand_fingerprint,
                    "generated_at_utc": generated_at_utc,
                    "parameter_source_counts": config.source_counts(),
                    "data_snapshot_id": data_snapshot_id,
                    "metric_definitions": METRIC_DEFINITIONS,
                    "metrics": compute_metrics(config, outcome),
                    "limitations": LIMITATION_STATEMENT,
                }
            )
        fingerprints_by_rep[str(replicate)] = fingerprints

    common_inputs_verified = all(len(set(values)) == 1 for values in fingerprints_by_rep.values())
    aggregate = _aggregate_runs(runs)
    ranking = _rank_policies(aggregate)
    return {
        "scenario_id": config.scenario_id,
        "scenario_version": config.version,
        "scenario_title": config.title,
        "scenario_fingerprint": config.fingerprint(),
        "repetitions": repetitions,
        "base_seed": base_seed,
        "generated_at_utc": generated_at_utc,
        "common_inputs_verified": common_inputs_verified,
        "parameter_source_counts": config.source_counts(),
        "data_lineage": data_lineage,
        "metric_definitions": METRIC_DEFINITIONS,
        "aggregate": aggregate,
        "ranking": ranking,
        "pareto_policies": _pareto_policies(aggregate),
        "limitations": [LIMITATION_STATEMENT, *config.limitations],
        "runs": runs,
    }


def write_experiment_outputs(result: dict, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs_path = destination / "runs.json"
    summary_path = destination / "summary.json"
    csv_path = destination / "metrics.csv"
    with runs_path.open("w", encoding="utf-8") as handle:
        json.dump(result["runs"], handle, ensure_ascii=False, indent=2, sort_keys=True)
    summary = {key: value for key, value in result.items() if key != "runs"}
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    scalar_names = sorted(
        {
            key
            for run in result["runs"]
            for key in _scalar_metrics(run["metrics"])
        }
    )
    fieldnames = [
        "scenario_id",
        "policy_id",
        "replicate",
        "seed",
        "data_snapshot_id",
        *scalar_names,
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in result["runs"]:
            row = {
                key: run[key]
                for key in (
                    "scenario_id",
                    "policy_id",
                    "replicate",
                    "seed",
                    "data_snapshot_id",
                )
            }
            row.update(_scalar_metrics(run["metrics"]))
            writer.writerow(row)
    return {
        "runs_json": str(runs_path),
        "summary_json": str(summary_path),
        "metrics_csv": str(csv_path),
    }
