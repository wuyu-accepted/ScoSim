from __future__ import annotations

import unittest

from evaluation.dimension2 import language_style_consistency


class Dimension2StyleDeduplicationTests(unittest.TestCase):
    def test_exact_duplicates_are_removed_within_agent(self):
        rows = []
        for index in range(10):
            agent = f"agent-{index}"
            rows.extend([
                {"agent_id": agent, "generated_text": "同一句话"},
                {"agent_id": agent, "generated_text": " 同一句话 "},
                {"agent_id": agent, "generated_text": f"不同表达{index}"},
            ])
        result = language_style_consistency(rows)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["raw_expression_count"], 30)
        self.assertEqual(result["exact_duplicate_count"], 10)
        self.assertEqual(result["eligible_agent_count"], 10)
        self.assertEqual(result["eligible_unique_expression_count"], 20)

    def test_duplicate_only_agents_do_not_become_eligible(self):
        rows = [
            {"agent_id": f"agent-{index}", "generated_text": "重复"}
            for index in range(12) for _ in range(2)
        ]
        result = language_style_consistency(rows)
        self.assertEqual(result["status"], "NA_insufficient_repeated_expression")
        self.assertEqual(result["exact_duplicate_count"], 12)
        self.assertEqual(result["eligible_agent_count"], 0)


if __name__ == "__main__":
    unittest.main()
