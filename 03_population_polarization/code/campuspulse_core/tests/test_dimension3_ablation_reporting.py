from __future__ import annotations

import unittest

import numpy as np

from evaluation.common import STATE_FIELDS
from evaluation.dimension3 import evaluate_ablation_group


class Dimension3AblationReportingTests(unittest.TestCase):
    def test_macro_and_all_micro_features_are_derived_from_parent_rows(self):
        rows = []
        for index in range(8):
            state = {field: 0.05 * index + 0.01 * offset for offset, field in enumerate(STATE_FIELDS)}
            rows.append({
                "agent_id": f"a{index}",
                "interest_group": "directly_affected" if index < 4 else "bystander",
                "post_state": state,
            })
        result = evaluate_ablation_group(rows, count=7, seed=9, expected=8)
        self.assertEqual(set(result["state_dimensions"]), set(STATE_FIELDS))
        self.assertIn("state_matrix_effective_rank", result)
        self.assertIn("stake_group_multivariate_r_squared_12d", result)
        self.assertIn("stake_group_permutation_p", result)
        for field in ("trust", "satisfaction", "attention", "concern"):
            self.assertAlmostEqual(result["state_dimensions"][field]["mean"], np.mean([row["post_state"][field] for row in rows]))
        stance = [row["post_state"]["support"] - row["post_state"]["opposition"] for row in rows]
        self.assertAlmostEqual(result["net_stance"]["mean"], np.mean(stance))

    def test_parent_weighted_mean_not_particle_flattening(self):
        # The evaluator consumes exactly one post_state per parent. A duplicated
        # parent is rejected by the population uniqueness gate.
        state = {field: 0.2 for field in STATE_FIELDS}
        rows = [
            {"agent_id": "same", "interest_group": "bystander", "post_state": state},
            {"agent_id": "same", "interest_group": "bystander", "post_state": state},
        ]
        with self.assertRaises(ValueError):
            evaluate_ablation_group(rows, count=1, seed=1, expected=2)


if __name__ == "__main__":
    unittest.main()
