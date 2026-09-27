from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .adapters.belief_engine import ArgumentCandidate, BeliefEngineAdapter, Embedder, Extractor
from .contracts import AgentSeed


@dataclass(frozen=True)
class NetworkArgumentDelivery:
    tick: int
    source_agent_id: str
    target_agent_id: str
    message_id: str
    arguments: tuple[ArgumentCandidate, ...]


class BeliefNetworkPopulation:
    """Synchronous fixed-network Belief Engine population for Dimension 3.

    Each tick first freezes every sender's strongest active arguments, then
    delivers them along the registered directed edges. No receiver can affect
    another sender inside the same tick, preventing order-dependent updates.
    """

    def __init__(
        self, *, seeds: Mapping[str, AgentSeed],
        seed_arguments: Mapping[str, Sequence[ArgumentCandidate]],
        directed_edges: Iterable[tuple[str, str]], extractor: Extractor,
        embedder: Embedder, evidence_uptake: float = 0.2,
        prior_anchoring: float = 0.4, similarity_threshold: float = 0.60,
        arguments_per_edge: int = 2,
    ) -> None:
        self.agent_ids = tuple(sorted(seeds))
        if set(seed_arguments) != set(seeds):
            raise ValueError("seed argument identities must equal seed identities")
        self.edges = tuple(sorted((str(source), str(target)) for source, target in directed_edges))
        if len(self.edges) != len(set(self.edges)):
            raise ValueError("network contains duplicate directed edges")
        if any(source not in seeds or target not in seeds or source == target for source, target in self.edges):
            raise ValueError("network contains an invalid endpoint or self edge")
        self.arguments_per_edge = int(arguments_per_edge)
        if self.arguments_per_edge <= 0:
            raise ValueError("arguments_per_edge must be positive")
        self.adapters: dict[str, BeliefEngineAdapter] = {}
        for agent_id in self.agent_ids:
            adapter = BeliefEngineAdapter(
                seed_arguments=seed_arguments[agent_id], extractor=extractor,
                embedder=embedder, evidence_uptake=evidence_uptake,
                prior_anchoring=prior_anchoring,
                similarity_threshold=similarity_threshold,
            )
            adapter.initialize(seeds[agent_id])
            self.adapters[agent_id] = adapter
        self.completed_ticks: list[int] = []
        self.delivery_log: list[NetworkArgumentDelivery] = []

    def inject_event_arguments(
        self, *, tick: int,
        arguments_by_agent: Mapping[str, Sequence[ArgumentCandidate]],
        event_id: str,
    ) -> None:
        if set(arguments_by_agent) != set(self.agent_ids):
            raise ValueError("event injection must cover all 1000 population identities")
        for agent_id in self.agent_ids:
            self.adapters[agent_id].ingest_network_arguments(
                arguments_by_agent[agent_id], tick=tick,
                source_message_id=f"event:{event_id}:receiver:{agent_id}",
            )

    def step(self, tick: int) -> tuple[NetworkArgumentDelivery, ...]:
        if self.completed_ticks and tick != self.completed_ticks[-1] + 1:
            raise ValueError("Belief Engine ticks must be consecutive")
        if not self.completed_ticks and tick != 1:
            raise ValueError("the first network propagation tick must be 1")
        outbound = {
            agent_id: self.adapters[agent_id].export_network_arguments(self.arguments_per_edge)
            for agent_id in self.agent_ids
        }
        deliveries = tuple(
            NetworkArgumentDelivery(
                tick=tick, source_agent_id=source, target_agent_id=target,
                message_id=f"be-network:t{tick}:{source}:{target}",
                arguments=outbound[source],
            )
            for source, target in self.edges
        )
        for delivery in deliveries:
            self.adapters[delivery.target_agent_id].ingest_network_arguments(
                delivery.arguments, tick=tick, source_message_id=delivery.message_id,
            )
        self.delivery_log.extend(deliveries)
        self.completed_ticks.append(tick)
        return deliveries

    def population_state(self, tick: int, interest_groups: Mapping[str, str], seed: int) -> list[dict[str, object]]:
        if set(interest_groups) != set(self.agent_ids):
            raise ValueError("interest-group identities must equal population identities")
        return [
            {
                "schema_version": "dimension3-population-state-v1.0.0",
                "method": "belief_engine", "variant": "full", "seed": int(seed),
                "tick": int(tick), "agent_id": agent_id,
                "interest_group": interest_groups[agent_id],
                "public_stance": self.adapters[agent_id].stance,
                "native_log_odds": self.adapters[agent_id].log_odds,
                "active_argument_count": sum(row.active for row in self.adapters[agent_id].records),
                "archived_argument_count": sum(not row.active for row in self.adapters[agent_id].records),
            }
            for agent_id in self.agent_ids
        ]
