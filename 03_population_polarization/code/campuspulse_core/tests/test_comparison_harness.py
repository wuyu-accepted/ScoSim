from __future__ import annotations

import math
import unittest

from comparison_harness.adapters import (
    ArgumentCandidate,
    BeliefEngineAdapter,
    CampusPulseAdapter,
    ChuangOpinionDynamicsAdapter,
    GenerativeAgentsAdapter,
)
from comparison_harness.adapters.campuspulse import DIMENSION_2, DIMENSION_3
from comparison_harness.audit import audit_paired_prompts, audit_snapshot_information
from comparison_harness.contracts import AgentSeed, CommonObservation, ContextItem
from comparison_harness.fair_initialization import FairRoleInitialization
from comparison_harness.prompting import (
    COMMON_SECTIONS, STANCE_PROPOSITION, SharedPromptRenderer,
)


FIELDS = (
    "attention", "concern", "trust", "satisfaction", "support", "opposition",
    "activity", "voice", "service_strain", "rumor_belief", "verified_belief",
    "evidence_acceptance",
)


def vector(text: str) -> tuple[float, float, float]:
    return (float(text.count("a") + 1), float(text.count("b") + 1), float(len(text) + 1))


def seed() -> AgentSeed:
    state = {name: 0.5 for name in FIELDS}
    state["support"], state["opposition"] = 0.75, 0.25
    particles = tuple({**state, "support": 0.70 + i / 100} for i in range(10))
    return AgentSeed("agent-1", "profile-1", "group-1", {"style": "brief"}, state,
                     particles, (0.1,) * 10)


def observation() -> CommonObservation:
    return CommonObservation(
        scenario_id="governance_legitimacy_dispute", tick=3, agent_id="agent-1",
        factual_registry={"available_beds": 320, "applications": 536},
        current_event={"event_id": "allocation-update", "text": "Results were posted."},
        visible_context=(ContextItem("m-1", "agent-2", "allocation is unfair", "public", 3),),
        governance_artifacts=({"artifact_id": "notice-1", "text": "Appeal channel open"},),
        response_schema={"type": "object", "required": ["action", "text", "public_stance"]},
    )


class HarnessTests(unittest.TestCase):
    def test_fair_role_card_is_mapped_into_all_native_initializers(self):
        role = FairRoleInitialization(
            agent_id="agent-1", profile_id="profile-1",
            role_card={"language_style": "brief", "interaction_role": "requester"},
            role_card_sha256="1" * 64,
            initial_support=0.7, initial_opposition=0.3,
        )
        seeded = role.agent_seed()
        self.assertEqual(seeded.stable_profile, role.role_card)
        arguments = role.belief_arguments()
        self.assertIn("language_style", arguments[0].claim)
        self.assertGreater(arguments[0].strength, arguments[1].strength)
        ga = GenerativeAgentsAdapter(
            embedder=vector, importance_scorer=lambda _: 5.0,
            reflector=lambda _: (), reflection_threshold=999,
        )
        ga.initialize(seeded)
        self.assertEqual(ga.memories[0].kind, "profile")
        chuang = ChuangOpinionDynamicsAdapter(initial_opinion=None)
        chuang.initialize(seeded)
        self.assertEqual(chuang.current_opinion, 1)
        self.assertEqual(chuang.history[0]["kind"], "role_background")

    def test_all_common_prompt_sections_are_byte_identical(self):
        obs = observation()
        cp = CampusPulseAdapter(DIMENSION_3["full"])
        ga = GenerativeAgentsAdapter(
            embedder=vector, importance_scorer=lambda _: 5.0,
            reflector=lambda _: (), reflection_threshold=999,
        )
        ch = ChuangOpinionDynamicsAdapter(initial_opinion=1)
        be = BeliefEngineAdapter(
            seed_arguments=(ArgumentCandidate("seed claim", 1, 0.8, "seed"),),
            extractor=lambda text, role: (), embedder=vector,
        )
        adapters = (cp, ga, ch, be)
        renderer = SharedPromptRenderer()
        bundles = []
        for adapter in adapters:
            adapter.initialize(seed())
            snapshot = adapter.observe(obs)
            self.assertFalse(audit_snapshot_information(obs, snapshot))
            bundles.append(renderer.render(obs, snapshot))
        self.assertFalse(audit_paired_prompts(bundles))
        for section in COMMON_SECTIONS:
            self.assertEqual(1, len({bundle.section_sha256[section] for bundle in bundles}))
        self.assertGreater(len({bundle.section_sha256["method_cognitive_context"] for bundle in bundles}), 1)
        self.assertIn(STANCE_PROPOSITION, bundles[0].section_text["system_boundary"])

    def test_campuspulse_ablation_changes_only_registered_context(self):
        obs = observation()
        contexts = {}
        for name, spec in DIMENSION_2.items():
            adapter = CampusPulseAdapter(spec)
            adapter.initialize(seed())
            contexts[name] = adapter.observe(obs).cognitive_context
        self.assertNotIn("stable_profile", contexts["context_only"])
        self.assertNotIn("particles", contexts["context_only"])
        self.assertIsNotNone(contexts["profile_only"]["stable_profile"])
        self.assertNotIn("particles", contexts["profile_only"])
        self.assertNotIn("stable_profile", contexts["state_only"])
        self.assertEqual(10, len(contexts["state_only"]["particles"]))
        self.assertIsNotNone(contexts["full"]["stable_profile"])

    def test_generative_agents_native_retrieval_and_reflection_gate(self):
        calls = []
        adapter = GenerativeAgentsAdapter(
            embedder=vector, importance_scorer=lambda _: 6.0,
            reflector=lambda evidence: calls.append(evidence) or ("reflection",),
            reflection_threshold=6.0, retrieval_count=3,
        )
        adapter.initialize(seed())
        snap = adapter.observe(observation())
        self.assertEqual(1, len(calls))
        self.assertTrue(snap.cognitive_context["retrieved_memory_stream"])
        self.assertEqual("reflection", snap.auxiliary_calls[0]["type"])

    def test_chuang_native_grid_and_history(self):
        adapter = ChuangOpinionDynamicsAdapter(initial_opinion=-1)
        adapter.initialize(seed())
        first_snapshot = adapter.observe(observation())
        adapter.observe(observation())
        observed = [row for row in adapter.history if row.get("kind") == "observed_interaction"]
        self.assertEqual(1, len(observed))
        self.assertEqual(observation().visible_context[0].text, observed[0]["text"])
        self.assertTrue(observed[0]["closed_world"])
        self.assertFalse(any(
            row.get("kind") == "observed_interaction"
            for row in first_snapshot.cognitive_context["closed_world_interaction_memory"]
        ))
        adapter.commit_decision(observation(), {"action": "reply", "text": "I changed", "public_stance": 0.5})
        self.assertEqual(1, adapter.current_opinion)
        self.assertEqual(0.5, adapter.export_probe())
        adapter.commit_decision(observation(), {"public_stance": 0.25})
        self.assertEqual(0, adapter.current_opinion)
        self.assertEqual(0.0, adapter.export_probe())
        adapter.commit_decision(observation(), {"public_stance": None})
        self.assertEqual(0, adapter.current_opinion)

    def test_belief_engine_equation_and_stronger_duplicate_replacement(self):
        adapter = BeliefEngineAdapter(
            seed_arguments=(ArgumentCandidate("same claim", 1, 0.4, "seed"),),
            extractor=lambda text, role: (ArgumentCandidate("same claim paraphrase", 1, 0.9, role),),
            embedder=lambda _: (1.0, 0.0), evidence_uptake=0.2,
            prior_anchoring=0.4, similarity_threshold=0.6,
        )
        adapter.initialize(seed())
        adapter.observe(observation())
        expected = math.log1p(0.9 * 0.2)
        self.assertAlmostEqual(expected, adapter.log_odds)
        self.assertAlmostEqual(math.tanh(expected / 2), adapter.export_probe())
        self.assertEqual(1, sum(record.active for record in adapter.records))
        self.assertEqual(1, sum(not record.active for record in adapter.records))


if __name__ == "__main__":
    unittest.main()
