"""Transparent governance policy definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    policy_id: str
    label: str
    routing_min_urgency: str = "high"
    priority_queue: bool = False
    faq_enabled: bool = False
    peer_boost_enabled: bool = False
    dynamic_capacity_enabled: bool = False


POLICIES: dict[str, Policy] = {
    "baseline": Policy("baseline", "Baseline: limited routing"),
    "priority_routing": Policy(
        "priority_routing",
        "Urgency-priority routing",
        priority_queue=True,
    ),
    "faq_deflection": Policy(
        "faq_deflection",
        "Targeted FAQ deflection",
        faq_enabled=True,
    ),
    "peer_support": Policy(
        "peer_support",
        "Peer response boost",
        peer_boost_enabled=True,
    ),
    "dynamic_capacity": Policy(
        "dynamic_capacity",
        "Backlog-triggered capacity",
        priority_queue=True,
        dynamic_capacity_enabled=True,
    ),
    "combined": Policy(
        "combined",
        "Combined targeted response",
        routing_min_urgency="medium",
        priority_queue=True,
        faq_enabled=True,
        peer_boost_enabled=True,
        dynamic_capacity_enabled=True,
    ),
}
