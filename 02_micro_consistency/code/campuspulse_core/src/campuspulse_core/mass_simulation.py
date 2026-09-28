"""Budgeted finite-population primitives for CampusPulse MassSim.

The module deliberately starts with the Scale-Gap gate.  Existing reviewed
48-agent runs are treated as enumerated finite LLM populations.  Sampling
designs must recover their aggregate action, stance and topic distributions
before the larger zero-Provider population engine is eligible for use.

Only synthetic agent identifiers and reviewed aggregate fields are read.  Raw
forum text, provider credentials and private source identifiers are outside
this contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


SCALE_GAP_SCHEMA_VERSION = "campus-pulse-mass-sim-scale-gap-v1"
MASS_SIMULATION_SCHEMA_VERSION = "campus-pulse-mass-sim-v1"
SCALE_GAP_BUDGETS = (4, 8, 16, 24, 32, 48)
SCALE_GAP_METHODS = (
    "srs",
    "stratified",
    "pps",
    "greg_pps",
    "critical_replacement",
)
SCALE_GAP_DIMENSIONS = ("action", "stance", "topic_id")


class MassSimulationError(ValueError):
    """Raised when a MassSim population or estimator contract is invalid."""


@dataclass(frozen=True)
class FinitePopulationCell:
    cell_id: str
    records: tuple[dict[str, Any], ...]
    categories: dict[str, tuple[str, ...]]

    @property
    def size(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class SampleDesign:
    selected: tuple[int, ...]
    inclusion_probability: tuple[float, ...]
    strata: tuple[int, ...]
    method: str

    @property
    def budget(self) -> int:
        return len(self.selected)


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _seed_from(*parts: Any) -> int:
    return int(_canonical_hash(list(parts))[:16], 16)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(probability), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _normalized_weights(records: Sequence[Mapping[str, Any]]) -> list[float]:
    weights = []
    for record in records:
        value = record.get("population_weight")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MassSimulationError("population_weight must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise MassSimulationError(
                "population_weight must be finite and non-negative"
            )
        weights.append(number)
    total = sum(weights)
    if total <= 0.0:
        raise MassSimulationError("population weights must have positive mass")
    return [value / total for value in weights]


def _archetype_number(record: Mapping[str, Any]) -> int:
    archetype = str(record.get("archetype_id") or "")
    suffix = archetype.rsplit("-", 1)[-1]
    try:
        return max(int(suffix), 1)
    except ValueError:
        return 1 + int(_canonical_hash(archetype or "generic")[:4], 16) % 17


def _presampling_strata(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    weights = _normalized_weights(records)
    median = _quantile(weights, 0.5)
    return tuple(
        ((_archetype_number(record) - 1) % 3) * 2
        + int(weights[index] >= median)
        for index, record in enumerate(records)
    )


def _integer_allocation(capacity: Sequence[int], budget: int) -> list[int]:
    capacities = [int(value) for value in capacity]
    if any(value < 0 for value in capacities):
        raise MassSimulationError("stratum capacity cannot be negative")
    population = sum(capacities)
    if not 0 <= int(budget) <= population:
        raise MassSimulationError("budget is outside the finite population")
    allocation = [0 for _ in capacities]
    nonempty = [index for index, value in enumerate(capacities) if value]
    if budget and budget < len(nonempty):
        nonempty = sorted(nonempty, key=lambda item: -capacities[item])[:budget]
    for index in nonempty:
        if sum(allocation) >= budget:
            break
        allocation[index] = 1
    while sum(allocation) < budget:
        available = [
            index
            for index, value in enumerate(capacities)
            if allocation[index] < value
        ]
        if not available:
            raise MassSimulationError("allocation exhausted population capacity")
        target = {
            index: budget * capacities[index] / max(population, 1)
            for index in available
        }
        chosen = max(
            available,
            key=lambda index: (
                target[index] - allocation[index],
                capacities[index],
                -index,
            ),
        )
        allocation[chosen] += 1
    return allocation


def fixed_size_inclusion_probabilities(
    scores: Sequence[float],
    budget: int,
) -> tuple[float, ...]:
    """Return positive first-order probabilities that sum to a fixed budget."""

    size = [max(float(value), 1e-12) for value in scores]
    if any(not math.isfinite(value) for value in size):
        raise MassSimulationError("PPS size measure must be finite")
    count = len(size)
    if not 0 <= int(budget) <= count:
        raise MassSimulationError("PPS budget is outside the population")
    if budget == 0:
        return tuple(0.0 for _ in size)
    if budget == count:
        return tuple(1.0 for _ in size)
    probability = [0.0 for _ in size]
    remaining = set(range(count))
    mass = float(budget)
    while remaining:
        indices = sorted(remaining)
        denominator = sum(size[index] for index in indices)
        scaled = {
            index: size[index] * mass / max(denominator, 1e-12)
            for index in indices
        }
        certain = [
            index for index in indices if scaled[index] >= 1.0 - 1e-12
        ]
        if not certain:
            for index in indices:
                probability[index] = scaled[index]
            break
        for index in certain:
            probability[index] = 1.0
            remaining.remove(index)
        mass = float(budget - sum(probability))
        if mass <= 1e-12:
            break
    if not math.isclose(sum(probability), budget, abs_tol=1e-8):
        raise MassSimulationError(
            "fixed-size inclusion probabilities do not sum to budget"
        )
    return tuple(probability)


def pivotal_sample(
    inclusion_probability: Sequence[float],
    rng: random.Random,
) -> tuple[int, ...]:
    """Draw a fixed-size pivotal sample preserving first-order probabilities."""

    probability = [float(value) for value in inclusion_probability]
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probability
    ):
        raise MassSimulationError(
            "inclusion probabilities must be finite in [0,1]"
        )
    target = int(round(sum(probability)))
    active = [
        index
        for index, value in enumerate(probability)
        if 1e-12 < value < 1.0 - 1e-12
    ]
    while len(active) >= 2:
        left, right = active[0], active[1]
        total = probability[left] + probability[right]
        if total < 1.0:
            if rng.random() < probability[right] / max(total, 1e-12):
                probability[left], probability[right] = 0.0, total
            else:
                probability[left], probability[right] = total, 0.0
        else:
            denominator = max(2.0 - total, 1e-12)
            if rng.random() < (1.0 - probability[right]) / denominator:
                probability[left], probability[right] = 1.0, total - 1.0
            else:
                probability[left], probability[right] = total - 1.0, 1.0
        active = [
            index
            for index, value in enumerate(probability)
            if 1e-12 < value < 1.0 - 1e-12
        ]
    if active:
        index = active[0]
        probability[index] = float(rng.random() < probability[index])
    selected = [
        index for index, value in enumerate(probability) if value >= 1.0 - 1e-10
    ]
    if len(selected) != target:
        selected = sorted(
            range(len(probability)),
            key=lambda index: (-probability[index], index),
        )[:target]
    return tuple(sorted(selected))


def _design_for_cell(
    cell: FinitePopulationCell,
    budget: int,
    method: str,
    rng: random.Random,
) -> SampleDesign:
    if method not in SCALE_GAP_METHODS:
        raise MassSimulationError(f"unknown Scale-Gap method: {method}")
    size = cell.size
    budget = min(max(int(budget), 0), size)
    strata = _presampling_strata(cell.records)
    weights = _normalized_weights(cell.records)
    if method in {"srs", "critical_replacement"}:
        pi = tuple(budget / size for _ in range(size))
        if method == "critical_replacement":
            selected = tuple(
                sorted(
                    sorted(
                        range(size),
                        key=lambda index: (-weights[index], index),
                    )[:budget]
                )
            )
        else:
            selected = tuple(sorted(rng.sample(range(size), budget)))
        return SampleDesign(selected, pi, strata, method)

    labels = sorted(set(strata))
    capacity = [sum(value == label for value in strata) for label in labels]
    allocation = _integer_allocation(capacity, budget)
    pi = [0.0 for _ in range(size)]
    selected: list[int] = []
    for position, label in enumerate(labels):
        members = [index for index, value in enumerate(strata) if value == label]
        count = allocation[position]
        if not count:
            continue
        if method == "stratified":
            local_pi = tuple(count / len(members) for _ in members)
        else:
            local_scores = [
                weights[index]
                * (1.0 + 0.15 / math.sqrt(1 + _archetype_number(cell.records[index])))
                for index in members
            ]
            local_pi = fixed_size_inclusion_probabilities(local_scores, count)
        local_selected = pivotal_sample(local_pi, rng)
        for local_index, population_index in enumerate(members):
            pi[population_index] = local_pi[local_index]
        selected.extend(members[index] for index in local_selected)
    if len(selected) != budget:
        raise MassSimulationError("sample design did not meet the fixed budget")
    if any(pi[index] <= 0.0 for index in selected):
        raise MassSimulationError(
            "selected units require positive inclusion probability"
        )
    return SampleDesign(tuple(sorted(selected)), tuple(pi), strata, method)


def load_scale_gap_cells(
    paths: Iterable[str | Path],
) -> list[FinitePopulationCell]:
    """Load reviewed 48-agent phases without retaining generated text."""

    raw_cells: list[tuple[str, list[dict[str, Any]]]] = []
    category_values = {
        dimension: set() for dimension in SCALE_GAP_DIMENSIONS
    }
    for source_path in paths:
        path = Path(source_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment_mode") != "live_llm_agent_simulation":
            raise MassSimulationError(
                f"{path.name} is not a reviewed live-LLM simulation artifact"
            )
        scenario = str(payload.get("scenario_id") or path.stem)
        seen: set[tuple[str, str]] = set()
        for run in payload.get("policy_runs", []):
            policy_id = str(run.get("policy_id") or "unknown")
            for phase in run.get("phases", []):
                phase_id = str(phase.get("phase_id") or "")
                branch = "shared" if phase_id == "baseline" else policy_id
                key = (branch, phase_id)
                if key in seen:
                    continue
                seen.add(key)
                decisions = []
                for raw in phase.get("agent_decisions", []):
                    record = {
                        "synthetic_agent_id": str(
                            raw.get("synthetic_agent_id") or ""
                        ),
                        "archetype_id": str(raw.get("archetype_id") or ""),
                        "population_weight": raw.get("population_weight"),
                        "action": str(raw.get("action") or ""),
                        "stance": str(raw.get("stance") or ""),
                        "topic_id": str(raw.get("topic_id") or ""),
                    }
                    if not record["synthetic_agent_id"]:
                        raise MassSimulationError(
                            "Scale-Gap decision lacks a synthetic agent ID"
                        )
                    decisions.append(record)
                    for dimension in SCALE_GAP_DIMENSIONS:
                        category_values[dimension].add(record[dimension])
                if len(decisions) != 48:
                    raise MassSimulationError(
                        f"{path.name}:{branch}:{phase_id} is not a 48-agent cell"
                    )
                raw_cells.append(
                    (
                        f"{path.stem}:{scenario}:{branch}:{phase_id}",
                        decisions,
                    )
                )
    if len(raw_cells) < 2:
        raise MassSimulationError("Scale-Gap requires at least two cells")
    categories = {
        dimension: tuple(sorted(values))
        for dimension, values in category_values.items()
    }
    return [
        FinitePopulationCell(
            cell_id=cell_id,
            records=tuple(records),
            categories=categories,
        )
        for cell_id, records in raw_cells
    ]


def _true_distribution(
    cell: FinitePopulationCell,
) -> dict[str, dict[str, float]]:
    weights = _normalized_weights(cell.records)
    return {
        dimension: {
            category: sum(
                weights[index]
                for index, record in enumerate(cell.records)
                if record[dimension] == category
            )
            for category in cell.categories[dimension]
        }
        for dimension in SCALE_GAP_DIMENSIONS
    }


def _fit_surrogate(
    cells: Sequence[FinitePopulationCell],
) -> dict[str, dict[str, dict[str, float]]]:
    numerators: dict[str, dict[str, dict[str, float]]] = {
        dimension: defaultdict(lambda: defaultdict(float))
        for dimension in SCALE_GAP_DIMENSIONS
    }
    denominators: dict[str, dict[str, float]] = {
        dimension: defaultdict(float) for dimension in SCALE_GAP_DIMENSIONS
    }
    global_numerators: dict[str, dict[str, float]] = {
        dimension: defaultdict(float) for dimension in SCALE_GAP_DIMENSIONS
    }
    global_mass = {dimension: 0.0 for dimension in SCALE_GAP_DIMENSIONS}
    for cell in cells:
        weights = _normalized_weights(cell.records)
        for index, record in enumerate(cell.records):
            archetype = str(record["archetype_id"])
            for dimension in SCALE_GAP_DIMENSIONS:
                value = str(record[dimension])
                numerators[dimension][archetype][value] += weights[index]
                denominators[dimension][archetype] += weights[index]
                global_numerators[dimension][value] += weights[index]
                global_mass[dimension] += weights[index]
    model: dict[str, dict[str, dict[str, float]]] = {}
    for dimension in SCALE_GAP_DIMENSIONS:
        global_distribution = {
            category: global_numerators[dimension][category]
            / max(global_mass[dimension], 1e-12)
            for category in cells[0].categories[dimension]
        }
        model[dimension] = {"__global__": global_distribution}
        for archetype, values in numerators[dimension].items():
            mass = denominators[dimension][archetype]
            prior_strength = 1.0
            model[dimension][archetype] = {
                category: (
                    values.get(category, 0.0)
                    + prior_strength * global_distribution[category]
                )
                / max(mass + prior_strength, 1e-12)
                for category in cells[0].categories[dimension]
            }
    return model


def _surrogate_value(
    surrogate: Mapping[str, Mapping[str, Mapping[str, float]]],
    dimension: str,
    archetype: str,
    category: str,
) -> float:
    by_archetype = surrogate[dimension]
    distribution = by_archetype.get(
        archetype, by_archetype["__global__"]
    )
    return float(distribution.get(category, 0.0))


def estimate_cell_distribution(
    cell: FinitePopulationCell,
    design: SampleDesign,
    surrogate: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    """Estimate categorical population shares from one fixed sample."""

    weights = _normalized_weights(cell.records)
    selected = design.selected
    result: dict[str, dict[str, float]] = {}
    for dimension in SCALE_GAP_DIMENSIONS:
        estimates: dict[str, float] = {}
        for category in cell.categories[dimension]:
            if design.method in {"greg_pps", "critical_replacement"}:
                l0 = [
                    _surrogate_value(
                        surrogate,
                        dimension,
                        str(record["archetype_id"]),
                        category,
                    )
                    for record in cell.records
                ]
                value = sum(
                    weights[index] * l0[index] for index in range(cell.size)
                )
                if design.method == "critical_replacement":
                    value += sum(
                        weights[index]
                        * (
                            float(cell.records[index][dimension] == category)
                            - l0[index]
                        )
                        for index in selected
                    )
                else:
                    value += sum(
                        weights[index]
                        * (
                            float(cell.records[index][dimension] == category)
                            - l0[index]
                        )
                        / design.inclusion_probability[index]
                        for index in selected
                    )
            else:
                numerator = sum(
                    weights[index]
                    * float(cell.records[index][dimension] == category)
                    / design.inclusion_probability[index]
                    for index in selected
                )
                denominator = sum(
                    weights[index] / design.inclusion_probability[index]
                    for index in selected
                )
                value = numerator / max(denominator, 1e-12)
            estimates[category] = max(float(value), 0.0)
        total = sum(estimates.values())
        if total <= 0.0:
            estimates = {
                category: 1.0 / len(estimates) for category in estimates
            }
        else:
            estimates = {
                category: value / total for category, value in estimates.items()
            }
        result[dimension] = estimates
    return result


def _absolute_errors(
    estimate: Mapping[str, Mapping[str, float]],
    truth: Mapping[str, Mapping[str, float]],
) -> list[float]:
    return [
        abs(float(estimate[dimension][category]) - float(value))
        for dimension, categories in truth.items()
        for category, value in categories.items()
    ]


def _fit_test_split(
    cells: Sequence[FinitePopulationCell],
) -> tuple[list[FinitePopulationCell], list[FinitePopulationCell]]:
    fit = [
        cell
        for cell in cells
        if int(_canonical_hash(cell.cell_id)[-1], 16) % 2 == 0
    ]
    test = [cell for cell in cells if cell not in fit]
    if not fit or not test:
        midpoint = max(1, len(cells) // 2)
        fit, test = list(cells[:midpoint]), list(cells[midpoint:])
    if not test:
        raise MassSimulationError("Scale-Gap split produced no held-out cells")
    return fit, test


def evaluate_scale_gap(
    cells: Sequence[FinitePopulationCell],
    *,
    budgets: Sequence[int] = SCALE_GAP_BUDGETS,
    repetitions: int = 200,
    seed: int = 20260730,
) -> dict[str, Any]:
    """Evaluate limited-call recovery with a fit-only interval calibration."""

    if repetitions < 20:
        raise MassSimulationError("Scale-Gap needs at least 20 repetitions")
    if any(cell.size != 48 for cell in cells):
        raise MassSimulationError("Scale-Gap v1 only accepts 48-agent cells")
    fit_cells, test_cells = _fit_test_split(cells)
    surrogate = _fit_surrogate(fit_cells)
    rows = []
    for raw_budget in budgets:
        budget = min(max(int(raw_budget), 1), 48)
        for method in SCALE_GAP_METHODS:
            fit_errors = []
            for cell in fit_cells:
                truth = _true_distribution(cell)
                repeats = 1 if method == "critical_replacement" else repetitions
                for repetition in range(repeats):
                    design = _design_for_cell(
                        cell,
                        budget,
                        method,
                        random.Random(
                            _seed_from(seed, "fit", cell.cell_id, budget, method, repetition)
                        ),
                    )
                    estimate = estimate_cell_distribution(
                        cell, design, surrogate
                    )
                    fit_errors.extend(_absolute_errors(estimate, truth))
            interval_radius = _quantile(fit_errors, 0.95)
            test_errors = []
            covered = 0
            comparisons = 0
            positive_pi = True
            fixed_budget = True
            for cell in test_cells:
                truth = _true_distribution(cell)
                repeats = 1 if method == "critical_replacement" else repetitions
                for repetition in range(repeats):
                    design = _design_for_cell(
                        cell,
                        budget,
                        method,
                        random.Random(
                            _seed_from(seed, "test", cell.cell_id, budget, method, repetition)
                        ),
                    )
                    fixed_budget = fixed_budget and design.budget == budget
                    positive_pi = positive_pi and all(
                        design.inclusion_probability[index] > 0.0
                        for index in design.selected
                    )
                    estimate = estimate_cell_distribution(
                        cell, design, surrogate
                    )
                    errors = _absolute_errors(estimate, truth)
                    test_errors.extend(errors)
                    covered += sum(
                        error <= interval_radius + 1e-12 for error in errors
                    )
                    comparisons += len(errors)
            rows.append(
                {
                    "budget": budget,
                    "method": method,
                    "fit_only_interval_radius_95": interval_radius,
                    "held_out_mae": sum(test_errors) / max(len(test_errors), 1),
                    "held_out_coverage_95": covered / max(comparisons, 1),
                    "fixed_budget": fixed_budget,
                    "selected_pi_positive": positive_pi,
                    "held_out_comparisons": comparisons,
                }
            )
    selected = next(
        (
            row
            for row in rows
            if row["budget"] == 32 and row["method"] == "greg_pps"
        ),
        None,
    )
    full_budget_rows = [row for row in rows if row["budget"] == 48]
    full_budget_exact = all(
        row["held_out_mae"] <= 1e-10
        for row in full_budget_rows
        if row["method"] != "critical_replacement"
    )
    gate = {
        "selected_method": "greg_pps",
        "selected_budget": 32,
        "held_out_mae_at_or_below_0_05": bool(
            selected and selected["held_out_mae"] <= 0.05
        ),
        "coverage_in_0_90_to_0_99": bool(
            selected
            and 0.90 <= selected["held_out_coverage_95"] <= 0.99
        ),
        "fixed_budget": all(row["fixed_budget"] for row in rows),
        "selected_pi_positive": all(
            row["selected_pi_positive"] for row in rows
        ),
        "full_budget_exact": full_budget_exact,
    }
    gate["passed"] = all(gate[key] for key in (
        "held_out_mae_at_or_below_0_05",
        "coverage_in_0_90_to_0_99",
        "fixed_budget",
        "selected_pi_positive",
        "full_budget_exact",
    ))
    public_cells = [
        {
            "cell_id": cell.cell_id,
            "population_size": cell.size,
            "content_persisted": False,
        }
        for cell in cells
    ]
    body = {
        "schema_version": SCALE_GAP_SCHEMA_VERSION,
        "population_contract": (
            "Reviewed 48-agent synthetic decisions form an enumerated finite "
            "LLM population; this is not a student-population estimate."
        ),
        "provider_calls": 0,
        "fit_cells": len(fit_cells),
        "held_out_cells": len(test_cells),
        "repetitions": repetitions,
        "rows": rows,
        "gate": gate,
        "cells": public_cells,
        "privacy": {
            "synthetic_messages_persisted": False,
            "raw_forum_text_persisted": False,
            "source_identifiers_persisted": False,
        },
    }
    body["canonical_body_sha256"] = _canonical_hash(body)
    return body

