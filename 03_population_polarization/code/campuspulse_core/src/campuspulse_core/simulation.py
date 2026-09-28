"""Shared operational simulation engine for all CampusPulse scenarios."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass

from .config import ScenarioConfig
from .entities import DemandAgent, ServiceDepartment, URGENCY_RANK
from .generator import demand_fingerprint
from .policies import Policy


@dataclass
class SimulationOutcome:
    scenario_id: str
    scenario_version: str
    policy_id: str
    seed: int
    demand_fingerprint: str
    demands: dict[str, DemandAgent]
    departments: dict[str, ServiceDepartment]
    faq_reach_actions: int
    peer_boost_actions: int
    routing_actions: int
    backlog_trace: list[int]


def run_simulation(
    config: ScenarioConfig,
    policy: Policy,
    seed: int,
    demand_stream: list[DemandAgent],
) -> SimulationOutcome:
    """Run one policy on a private copy of a pre-generated demand stream."""

    stream_fingerprint = demand_fingerprint(demand_stream)
    copied = copy.deepcopy(demand_stream)
    demands = {d.demand_id: d for d in copied}
    arrivals: dict[int, list[DemandAgent]] = defaultdict(list)
    for demand in copied:
        arrivals[demand.arrival_round].append(demand)

    departments = {
        name: ServiceDepartment(name=name, base_capacity=int(capacity))
        for name, capacity in config.get("department_capacity").items()
    }
    natural_resolution_probability = config.get("natural_resolution_probability")
    faq_effectiveness = config.get("faq_effectiveness")
    faq_eligible = set(config.get("faq_eligible_need_types"))
    peer_success = config.get("peer_boost_success")
    peer_capacity = int(config.get("peer_boost_capacity"))
    backlog_threshold = int(config.get("dynamic_backlog_threshold"))
    dynamic_increment = int(config.get("dynamic_capacity_increment"))
    dynamic_limit = int(config.get("dynamic_capacity_limit"))
    min_urgency = URGENCY_RANK[policy.routing_min_urgency]

    faq_reach_actions = 0
    peer_boost_actions = 0
    routing_actions = 0
    backlog_trace: list[int] = []

    for current_round in range(int(config.get("rounds"))):
        for demand in arrivals.get(current_round, []):
            demand.thread.natural_replies = demand.natural_reply_count
            if (
                demand.natural_reply_count > 0
                and demand.natural_resolution_draw
                < natural_resolution_probability[demand.need_type]
            ):
                demand.resolved_channel = "natural_peer_response"

            if policy.faq_enabled and not demand.resolved and demand.need_type in faq_eligible:
                demand.thread.official_information_reached = True
                faq_reach_actions += 1
                if demand.faq_draw < faq_effectiveness[demand.need_type]:
                    demand.resolved_channel = "targeted_faq"

        if policy.peer_boost_enabled and peer_capacity > 0:
            candidates = [
                demand
                for demand in demands.values()
                if demand.arrival_round <= current_round
                and not demand.resolved
                and not demand.routed
                and demand.thread.boosted_replies == 0
            ]
            candidates.sort(
                key=lambda demand: (
                    -URGENCY_RANK[demand.urgency],
                    demand.thread.natural_replies > 0,
                    demand.arrival_round,
                    demand.demand_id,
                )
            )
            for demand in candidates[:peer_capacity]:
                demand.thread.boosted_replies = 1
                peer_boost_actions += 1
                if demand.peer_draw < peer_success[demand.need_type]:
                    demand.resolved_channel = "boosted_peer_response"

        route_candidates = [
            demand
            for demand in demands.values()
            if demand.arrival_round <= current_round
            and not demand.resolved
            and not demand.routed
            and URGENCY_RANK[demand.urgency] >= min_urgency
        ]
        route_candidates.sort(
            key=lambda demand: (
                -URGENCY_RANK[demand.urgency],
                demand.arrival_round,
                demand.demand_id,
            )
        )
        for demand in route_candidates:
            departments[demand.department].enqueue(demand, current_round)
            routing_actions += 1

        for department in departments.values():
            extra_capacity = 0
            if policy.dynamic_capacity_enabled and len(department.queue) > backlog_threshold:
                excess = len(department.queue) - backlog_threshold
                extra_capacity = min(dynamic_limit, dynamic_increment, excess)
            department.process(
                current_round=current_round,
                demands=demands,
                priority=policy.priority_queue,
                extra_capacity=extra_capacity,
            )
        backlog_trace.append(sum(len(department.queue) for department in departments.values()))

    horizon = int(config.get("rounds"))
    for demand in demands.values():
        if demand.routed and not demand.resolved:
            demand.synthetic_wait_rounds = horizon - demand.arrival_round

    return SimulationOutcome(
        scenario_id=config.scenario_id,
        scenario_version=config.version,
        policy_id=policy.policy_id,
        seed=seed,
        demand_fingerprint=stream_fingerprint,
        demands=demands,
        departments=departments,
        faq_reach_actions=faq_reach_actions,
        peer_boost_actions=peer_boost_actions,
        routing_actions=routing_actions,
        backlog_trace=backlog_trace,
    )
