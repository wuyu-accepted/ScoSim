from __future__ import annotations

import unittest

from adapters.campuspulse import adapt_native_turn


class CampusPulseAdapterTests(unittest.TestCase):
    def test_native_turn_preserves_missing_values_as_null(self):
        state = {name: 0.5 for name in (
            "attention", "concern", "trust", "satisfaction", "support", "opposition",
            "activity", "voice", "service_strain", "rumor_belief", "verified_belief",
            "evidence_acceptance",
        )}
        row = adapt_native_turn({
            "record_id": "native-1", "agent_id": "a-1", "tick": 3,
            "pre_state": state, "post_state": state,
            "particles": [state] * 10, "particle_weights": [0.1] * 10,
            "decision": {"visible_text": "test", "action": "post", "emotion": "calm"},
        }, experiment_id="test", variant="full", seed=20260722)
        self.assertEqual(row["generated_text"], "test")
        self.assertIsNone(row["profile_id"])
        self.assertIsNone(row["public_stance"])


if __name__ == "__main__":
    unittest.main()
