from __future__ import annotations

import hashlib
import unittest

from comparison_harness.adapters import ArgumentCandidate
from comparison_harness.contracts import AgentSeed
from comparison_harness.dimension3_belief_network import BeliefNetworkPopulation


def embedding(text: str) -> tuple[float, ...]:
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple((value / 255.0) * 2.0 - 1.0 for value in raw[:16])


def extractor(text: str, source_role: str):
    raise AssertionError("structured network propagation must not re-extract arguments")


class BeliefNetworkPopulationTests(unittest.TestCase):
    def population(self) -> BeliefNetworkPopulation:
        ids = ("a", "b", "c")
        seeds = {
            agent_id: AgentSeed(agent_id, None, None, None, {"support": 0.3, "opposition": 0.3})
            for agent_id in ids
        }
        arguments = {
            "a": (ArgumentCandidate("support-a", 1, 0.9, "seed"),),
            "b": (ArgumentCandidate("oppose-b", -1, 0.8, "seed"),),
            "c": (ArgumentCandidate("support-c", 1, 0.4, "seed"),),
        }
        return BeliefNetworkPopulation(
            seeds=seeds, seed_arguments=arguments,
            directed_edges=(("a", "b"), ("b", "c"), ("c", "a")),
            extractor=extractor, embedder=embedding, arguments_per_edge=1,
        )

    def test_synchronous_network_delivery_and_native_stance(self):
        population = self.population()
        before = {agent_id: adapter.log_odds for agent_id, adapter in population.adapters.items()}
        deliveries = population.step(1)
        self.assertEqual(len(deliveries), 3)
        self.assertEqual(len(population.delivery_log), 3)
        self.assertNotEqual(population.adapters["b"].log_odds, before["b"])
        rows = population.population_state(1, {"a": "bystander", "b": "directly_affected", "c": "bystander"}, 20260815)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(-1 <= row["public_stance"] <= 1 for row in rows))
        self.assertTrue(all(abs(row["public_stance"] - __import__("math").tanh(row["native_log_odds"] / 2)) < 1e-12 for row in rows))

    def test_duplicate_delivery_is_idempotent(self):
        population = self.population()
        candidate = (ArgumentCandidate("event", 1, 0.7, "opponent"),)
        adapter = population.adapters["a"]
        adapter.ingest_network_arguments(candidate, tick=3, source_message_id="event-1")
        count = len(adapter.records)
        adapter.ingest_network_arguments(candidate, tick=3, source_message_id="event-1")
        self.assertEqual(len(adapter.records), count)

    def test_tick_order_is_enforced(self):
        population = self.population()
        with self.assertRaises(ValueError):
            population.step(2)


if __name__ == "__main__":
    unittest.main()
