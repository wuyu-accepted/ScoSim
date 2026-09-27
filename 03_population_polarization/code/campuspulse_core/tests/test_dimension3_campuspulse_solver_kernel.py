from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campuspulse_core.governance_arena_v2 import (
    GraphArrays, ParticleEnsemble, _parent_features, advance_particle_tick,
    assimilate_forum_llm_updates_v2, initial_branch_state,
)
from scripts.run_dimension3_campuspulse_ablation import initial_ensemble, load_snapshot, save_snapshot, write_jsonl_atomic


class Dimension3CampusPulseSolverKernelTests(unittest.TestCase):
    def test_single_particle_assimilation_never_resamples(self):
        values = np.full((3, 1, 12), 0.4, dtype=np.float32)
        weights = np.full((3, 1), 1 / 3, dtype=np.float64)
        ensemble = ParticleEnsemble(values, weights, np.full(3, 1 / 3, dtype=np.float64))
        updates = {
            index: {
                "prompt_sha256": f"{index + 1:064x}",
                "response_sha256": f"{index + 11:064x}",
                "decision": {"action": "post", "stance": "support", "urgency": 0.8, "confidence": 0.7},
            }
            for index in range(3)
        }
        result = assimilate_forum_llm_updates_v2(
            state=SimpleNamespace(ensemble=ensemble), updates=updates,
            scenario_id="governance_legitimacy_dispute", branch="natural", tick=3, seed=20260815,
        )
        self.assertEqual(result["applied_count"], 3)
        self.assertEqual(result["resampling_count"], 0)
        self.assertEqual(ensemble.values.shape, (3, 1, 12))
        self.assertTrue(np.allclose(ensemble.parent_ess(), 1.0))

    def test_ten_particle_assimilation_keeps_shape_and_mass(self):
        rng = np.random.default_rng(7)
        values = rng.uniform(0.1, 0.9, (2, 10, 12)).astype(np.float32)
        parent_weights = np.asarray([0.4, 0.6])
        weights = np.repeat((parent_weights / 10)[:, None], 10, axis=1)
        ensemble = ParticleEnsemble(values, weights, parent_weights)
        assimilate_forum_llm_updates_v2(
            state=SimpleNamespace(ensemble=ensemble),
            updates={0: {"prompt_sha256": "1" * 64, "response_sha256": "2" * 64,
                         "decision": {"action": "read", "stance": "uncertain", "urgency": 0.2}}},
            scenario_id="governance_legitimacy_dispute", branch="natural", tick=1, seed=20260815,
        )
        self.assertEqual(ensemble.values.shape, (2, 10, 12))
        self.assertTrue(np.allclose(ensemble.weights.sum(axis=1), parent_weights))

    def test_small_population_tick_for_both_solver_kernels(self):
        source = json.loads((Path(__file__).resolve().parents[1] / "scenario" / "housing_allocation" / "hero-pilot-bridge-v5.json").read_text(encoding="utf-8"))
        population = copy.deepcopy(source)
        population["agents"] = population["agents"][:12]
        for agent in population["agents"]:
            agent["population_weight"] = 1 / 12
        targets = np.asarray([[(i + j + 1) % 12 for j in range(8)] for i in range(12)], dtype=np.int32)
        shape = targets.shape
        graph = GraphArrays(targets, np.zeros(shape, bool), np.ones(shape, np.float32),
                            np.ones(shape, np.int8), np.ones(shape, np.int8), "x" * 64, {})
        features = _parent_features(population, "governance_legitimacy_dispute")
        for variant, expected_particles, stochastic in (("no_profile", 10, True), ("single_particle", 1, False)):
            state = initial_branch_state(population, scenario_id="governance_legitimacy_dispute", seed=20260815)
            initial_ensemble(state, variant)
            advance_particle_tick(population=population, graph=graph, state=state, features=features,
                                  scenario_id="governance_legitimacy_dispute", branch="natural", seed=20260815,
                                  tick=1, emit_messages=False, stochastic_diffusion=stochastic)
            self.assertEqual(state.ensemble.values.shape, (12, expected_particles, 12))
            state.ensemble.assert_conserved()

    def test_snapshot_roundtrip_and_corruption_rejection(self):
        payload = {"variant": "single_particle", "seed": 20260815, "tick": 4, "state": {"x": 1}, "messages": {}}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / "tick-4.pkl.gz"
            save_snapshot(path, payload)
            self.assertEqual(load_snapshot(directory, "single_particle", 20260815), payload)
            path.write_bytes(path.read_bytes() + b"corrupt")
            self.assertIsNone(load_snapshot(directory, "single_particle", 20260815))

    def test_interrupted_tick_can_be_atomically_replaced_without_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tick-4.actions.jsonl"
            write_jsonl_atomic(path, [{"agent_id": "a", "attempt": 1}, {"agent_id": "b", "attempt": 1}])
            write_jsonl_atomic(path, [{"agent_id": "a", "attempt": 2}, {"agent_id": "b", "attempt": 2}])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["attempt"] for row in rows}, {2})


if __name__ == "__main__":
    unittest.main()
