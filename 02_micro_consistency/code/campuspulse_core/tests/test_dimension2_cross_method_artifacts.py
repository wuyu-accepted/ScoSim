from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from comparison_harness.contracts import CommonObservation
from evaluation.dimension2 import METHODS, case_id
from scripts.build_dimension2_cross_method_judge_cases import build_cross_method_cases
from scripts.render_dimension2_cross_method_report import main as render_report
from evaluation.text_judge import select_human_audit_case_ids


FIELDS = (
    "attention", "concern", "trust", "satisfaction", "support", "opposition",
    "activity", "voice", "service_strain", "rumor_belief", "verified_belief",
    "evidence_acceptance",
)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class CrossMethodArtifactTests(unittest.TestCase):
    def test_2048_blind_cases_and_four_column_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = {"agents": []}
            for index in range(1000):
                population["agents"].append({
                    "llm_agent_id": f"a{index}", "profile_bundle_id": f"p{index}",
                    "state": {"support": 0.7, "opposition": 0.3},
                    "persona": {"profile_summary": f"role {index}"},
                    "stable_persona": {"language_style": {"language_form": "brief"}},
                    "semantic": {"speech_act": "reply", "stance": "context_dependent"},
                })
            population_path = root / "population.json"
            population_path.write_text(json.dumps(population), encoding="utf-8")

            observations = []
            keys = []
            for index in range(512):
                agent_index = index if index < 350 else index - 350
                tick = 0 if index < 350 else 1
                observation = CommonObservation(
                    scenario_id="governance_legitimacy_dispute", tick=tick,
                    agent_id=f"a{agent_index}", factual_registry={"known": "fact"},
                    current_event={"event": "allocation"}, visible_context=(),
                    governance_artifacts=(),
                    response_schema={"type": "object"},
                )
                source_id = f"obs-{index}"
                observations.append({
                    "observation_id": source_id, "activation_class": "test",
                    "observation_sha256": observation.observation_sha256,
                    "observation": observation.public_payload(),
                })
                keys.append((source_id, observation))
            observations_path = root / "observations.jsonl"
            write_jsonl(observations_path, observations)

            record_paths = []
            all_records = []
            state = {field: 0.5 for field in FIELDS}
            state.update({"support": 0.7, "opposition": 0.3})
            for method in METHODS:
                rows = []
                for source_id, observation in keys:
                    row = {
                        "experiment_id": "test", "method": method,
                        "variant": "cross_method_full", "seed": 20260722,
                        "agent_id": observation.agent_id, "tick": observation.tick,
                        "source_record_id": source_id,
                        "generated_text": f"{observation.agent_id} supports the known allocation fact",
                        "action": "post", "emotion": "neutral",
                        "public_stance": 0.5, "public_or_private": "public",
                        "model_status": "success", "visible_context": [],
                        "particles": [dict(state) for _ in range(10)] if method == "campuspulse" else None,
                        "particle_weights": [0.1] * 10 if method == "campuspulse" else None,
                    }
                    rows.append(row)
                    all_records.append(row)
                path = root / f"{method}.jsonl"
                write_jsonl(path, rows)
                record_paths.append(path)

            with patch(
                "scripts.build_dimension2_cross_method_judge_cases.load_frozen_prompt",
                return_value="facts={{FACT_REGISTRY_JSON}} evidence={{METHOD_EVIDENCE_JSON}} event={{CURRENT_EVENT_JSON}} context={{VISIBLE_CONTEXT_JSON}} text={{GENERATED_TEXT}} case={{CASE_ID}}",
            ):
                cases, evidence, audits, manifest = build_cross_method_cases(
                    record_paths, observations_path, population_path, root / "prompt.txt"
                )
            self.assertEqual(len(cases), 2048)
            self.assertEqual(len(evidence), 2048)
            self.assertEqual(len(audits), 2048)
            self.assertEqual(manifest["paired_observation_count"], 512)

            judgements = [{
                "case_id": case_id(row), "context_uptake": "yes",
                "unsupported_fact": "no", "expression_stance": "strong_support",
            } for row in all_records]
            judgements_path = root / "judgements.jsonl"
            write_jsonl(judgements_path, judgements)
            selected_human = select_human_audit_case_ids(all_records, fraction=0.20)
            human_path = root / "human.jsonl"
            write_jsonl(
                human_path,
                [row for row in judgements if row["case_id"] in selected_human],
            )
            belief_rows = [{
                "case_id": case_id(row), "standardized_pre_state": 0.4,
            } for row in all_records if row["method"] == "belief_engine"]
            belief_path = root / "belief.jsonl"
            write_jsonl(belief_path, belief_rows)
            output = root / "report"
            argv = [
                "render", "--records", *[str(path) for path in record_paths],
                "--judgements", str(judgements_path),
                "--human-judgements", str(human_path),
                "--belief-state-sidecar", str(belief_path),
                "--output-root", str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(render_report(), 0)
            report = (output / "FINAL_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("CampusPulse", report)
            self.assertIn("Belief Engine", report)
            self.assertIn("Generative Agents", report)
            self.assertIn("Chuang et al.", report)
            result = json.loads((output / "dimension2_cross_method.json").read_text(encoding="utf-8"))
            self.assertEqual(result["paired_key_count"], 512)
            self.assertEqual(
                result["methods"]["generative_agents"]["text_metrics"]
                ["expression_pre_state_agreement"]["status"],
                "NA_no_eligible_agreement_labels",
            )


if __name__ == "__main__":
    unittest.main()
