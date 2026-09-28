"""Core synthetic entities and capacity-constrained service actors."""

from __future__ import annotations

from dataclasses import dataclass, field


URGENCY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class ThreadState:
    natural_replies: int = 0
    boosted_replies: int = 0
    official_information_reached: bool = False

    @property
    def total_replies(self) -> int:
        return self.natural_replies + self.boosted_replies


@dataclass
class DemandAgent:
    demand_id: str
    arrival_round: int
    need_type: str
    urgency: str
    department: str
    work_units: int
    natural_response_draw: float
    natural_reply_count: int
    natural_resolution_draw: float
    faq_draw: float
    peer_draw: float
    thread: ThreadState = field(default_factory=ThreadState)
    routed: bool = False
    queue_enter_round: int | None = None
    completed_round: int | None = None
    resolved_channel: str | None = None
    work_remaining: int = 0
    synthetic_wait_rounds: int = 0

    def __post_init__(self) -> None:
        self.work_remaining = self.work_units

    @property
    def resolved(self) -> bool:
        return self.resolved_channel is not None

    @property
    def urgent(self) -> bool:
        return self.urgency in {"high", "critical"}


@dataclass
class ServiceDepartment:
    name: str
    base_capacity: int
    queue: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    total_capacity_available: int = 0
    total_capacity_used: int = 0
    total_dynamic_capacity: int = 0
    max_backlog: int = 0

    def enqueue(self, demand: DemandAgent, current_round: int) -> None:
        if demand.demand_id in self.queue or demand.demand_id in self.completed:
            return
        demand.routed = True
        demand.queue_enter_round = current_round
        self.queue.append(demand.demand_id)
        self.max_backlog = max(self.max_backlog, len(self.queue))

    def process(
        self,
        current_round: int,
        demands: dict[str, DemandAgent],
        priority: bool,
        extra_capacity: int = 0,
    ) -> list[str]:
        extra_capacity = max(0, int(extra_capacity))
        budget = self.base_capacity + extra_capacity
        self.total_capacity_available += budget
        self.total_dynamic_capacity += extra_capacity
        if priority:
            self.queue.sort(
                key=lambda demand_id: (
                    -URGENCY_RANK[demands[demand_id].urgency],
                    demands[demand_id].arrival_round,
                    demand_id,
                )
            )
        completed_now: list[str] = []
        while budget > 0 and self.queue:
            demand = demands[self.queue[0]]
            if demand.resolved:
                self.queue.pop(0)
                continue
            consumed = min(budget, demand.work_remaining)
            demand.work_remaining -= consumed
            budget -= consumed
            self.total_capacity_used += consumed
            if demand.work_remaining == 0:
                self.queue.pop(0)
                demand.completed_round = current_round
                demand.synthetic_wait_rounds = current_round - demand.arrival_round + 1
                demand.resolved_channel = "institutional_service"
                self.completed.append(demand.demand_id)
                completed_now.append(demand.demand_id)
        self.max_backlog = max(self.max_backlog, len(self.queue))
        return completed_now
