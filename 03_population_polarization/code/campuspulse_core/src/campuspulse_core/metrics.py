"""Operational, bounded metrics for CampusPulse simulations."""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import NEED_TYPES, URGENCY_LEVELS, ScenarioConfig
from .simulation import SimulationOutcome


LIMITATION_STATEMENT = (
    "All rounds and intervention outcomes are synthetic operational stress-test results. "
    "They are not estimates of real response time, psychological improvement, demographic impact, "
    "social-network diffusion, or causal policy effects."
)

METRIC_DEFINITIONS = {
    "operational_resolution_rate": (
        "Share of synthetic demands that complete a natural, boosted-peer, FAQ, or institutional response path."
    ),
    "urgent_unresolved_demand": (
        "Count of synthetic high- or critical-urgency demands without a completed response path at the horizon."
    ),
    "community_response_coverage_rate": (
        "Share of synthetic demand threads receiving at least one natural or boosted peer reply."
    ),
    "synthetic_information_reach_rate": (
        "Share of synthetic demands receiving either a thread reply or targeted official information."
    ),
    "peak_backlog": "Largest total number of queued synthetic demands observed in any decision round.",
    "capacity_utilization_rate": (
        "Institutional work units used divided by all available base and dynamic work units."
    ),
    "coverage_fairness_gap": (
        "Maximum minus minimum operational resolution rate across need types present in the run."
    ),
    "governance_cost": (
        "Dimensionless weighted sum of FAQ reach, peer actions, routing, service work, and dynamic capacity."
    ),
    "queue_accounting_error": (
        "Routed work minus completed service work minus outstanding queued work; must equal zero."
    ),
    "mean_synthetic_wait_rounds": (
        "Mean number of synthetic decision rounds from arrival to completion or simulation horizon for routed demands."
    ),
}


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percentile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return float(ordered[index])


def compute_metrics(config: ScenarioConfig, outcome: SimulationOutcome) -> dict:
    demands = list(outcome.demands.values())
    total = len(demands)
    resolved = [d for d in demands if d.resolved]
    urgent = [d for d in demands if d.urgent]
    urgent_resolved = [d for d in urgent if d.resolved]
    community_covered = [d for d in demands if d.thread.total_replies > 0]
    official_reached = [d for d in demands if d.thread.official_information_reached]
    routed = [d for d in demands if d.routed]
    waits = [d.synthetic_wait_rounds for d in routed]

    channel_counts = Counter(d.resolved_channel or "unresolved" for d in demands)
    by_need_total = Counter(d.need_type for d in demands)
    by_need_resolved = Counter(d.need_type for d in resolved)
    by_urgency_total = Counter(d.urgency for d in demands)
    by_urgency_resolved = Counter(d.urgency for d in resolved)
    coverage_by_need = {
        key: _rate(by_need_resolved[key], by_need_total[key]) for key in sorted(NEED_TYPES)
    }
    coverage_by_urgency = {
        key: _rate(by_urgency_resolved[key], by_urgency_total[key])
        for key in URGENCY_LEVELS
    }
    observed_need_rates = [coverage_by_need[key] for key in by_need_total if by_need_total[key]]
    fairness_gap = max(observed_need_rates) - min(observed_need_rates) if observed_need_rates else 0.0

    capacity_available = sum(d.total_capacity_available for d in outcome.departments.values())
    capacity_used = sum(d.total_capacity_used for d in outcome.departments.values())
    dynamic_capacity = sum(d.total_dynamic_capacity for d in outcome.departments.values())
    final_backlog = sum(len(d.queue) for d in outcome.departments.values())
    peak_backlog = max(outcome.backlog_trace, default=0)
    routed_work = sum(d.work_units for d in routed)
    outstanding_work = sum(
        outcome.demands[demand_id].work_remaining
        for department in outcome.departments.values()
        for demand_id in department.queue
    )
    queue_accounting_error = routed_work - capacity_used - outstanding_work

    costs = config.get("cost_weights")
    governance_cost = (
        outcome.faq_reach_actions * costs["faq_reach"]
        + outcome.peer_boost_actions * costs["peer_action"]
        + capacity_used * costs["service_work_unit"]
        + dynamic_capacity * costs["dynamic_capacity_unit"]
        + outcome.routing_actions * costs["routing_action"]
    )
    unique_information_reach = sum(
        1
        for demand in demands
        if demand.thread.total_replies > 0 or demand.thread.official_information_reached
    )

    return {
        "generated_demand": total,
        "resolved_demand": len(resolved),
        "unresolved_demand": total - len(resolved),
        "operational_resolution_rate": _rate(len(resolved), total),
        "urgent_demand": len(urgent),
        "urgent_unresolved_demand": len(urgent) - len(urgent_resolved),
        "urgent_resolution_rate": _rate(len(urgent_resolved), len(urgent)),
        "community_response_coverage_rate": _rate(len(community_covered), total),
        "official_information_reach_rate": _rate(len(official_reached), total),
        "synthetic_information_reach_rate": _rate(unique_information_reach, total),
        "natural_peer_resolved": channel_counts["natural_peer_response"],
        "boosted_peer_resolved": channel_counts["boosted_peer_response"],
        "faq_resolved": channel_counts["targeted_faq"],
        "institutional_completed": channel_counts["institutional_service"],
        "routed_demand": len(routed),
        "final_backlog": final_backlog,
        "peak_backlog": peak_backlog,
        "capacity_available": capacity_available,
        "capacity_used": capacity_used,
        "dynamic_capacity_added": dynamic_capacity,
        "capacity_utilization_rate": _rate(capacity_used, capacity_available),
        "mean_synthetic_wait_rounds": _rate(sum(waits), len(waits)),
        "p90_synthetic_wait_rounds": _percentile(waits, 0.9),
        "coverage_fairness_gap": fairness_gap,
        "governance_cost": float(governance_cost),
        "queue_accounting_error": queue_accounting_error,
        "coverage_by_need": coverage_by_need,
        "coverage_by_urgency": coverage_by_urgency,
        "resolved_by_channel": dict(sorted(channel_counts.items())),
        "department_metrics": {
            name: {
                "base_capacity_per_round": department.base_capacity,
                "capacity_used": department.total_capacity_used,
                "dynamic_capacity_added": department.total_dynamic_capacity,
                "completed": len(department.completed),
                "final_backlog": len(department.queue),
                "peak_backlog": department.max_backlog,
            }
            for name, department in sorted(outcome.departments.items())
        },
    }
