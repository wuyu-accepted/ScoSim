from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .adapters import (
    ArgumentCandidate,
    BeliefEngineAdapter,
    ChuangOpinionDynamicsAdapter,
    GenerativeAgentsAdapter,
)
from .auxiliary import FrozenPromptRegistry, sha256_text
from .contracts import AgentSeed, CommonObservation, ContextItem, MethodAdapter, canonical_json
from .prompting import SharedPromptRenderer, parse_structured_decision


BASELINE_METHODS = (
    "belief_engine", "generative_agents", "chuang_opinion_dynamics"
)
FAKE_MARKER = "FAKE_CONFORMANCE_ONLY_NOT_A_PAPER_RESULT"


@dataclass(frozen=True)
class ObservationEnvelope:
    observation_id: str
    activation_class: str
    observation: CommonObservation


class FakeAuxiliaryRuntime:
    """Deterministic local substitute that exercises frozen call boundaries."""

    def __init__(self, registry: FrozenPromptRegistry):
        self.registry = registry
        self.calls: list[dict[str, Any]] = []

    def _record(self, call_type: str, rendered_prompt: str) -> None:
        self.calls.append({
            "fake": True,
            "call_type": call_type,
            "prompt_sha256": sha256_text(rendered_prompt),
            "provider": "local_deterministic_fake",
        })

    def embedding(self, text: str) -> tuple[float, ...]:
        self.calls.append({
            "fake": True,
            "call_type": "shared_embedding",
            "input_sha256": sha256_text(text),
            "provider": "local_deterministic_fake",
        })
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((byte / 255.0) * 2.0 - 1.0) for byte in digest[:16]]
        if all(value == 0 for value in values):
            values[0] = 1.0
        return tuple(values)

    def importance(self, text: str) -> float:
        prompt = self.registry["ga_importance"].render({"INPUT_TEXT_JSON": text})
        self._record("ga_importance", prompt)
        return float(1 + hashlib.sha256(text.encode("utf-8")).digest()[0] % 10)

    def reflect(self, memories: tuple[str, ...]) -> tuple[str, ...]:
        prompt = self.registry["ga_reflection"].render({"MEMORY_STATEMENTS_JSON": memories})
        self._record("ga_reflection", prompt)
        if not memories:
            return ()
        return (FAKE_MARKER + ":reflection",)

    def extract(self, message: str, source_role: str) -> tuple[ArgumentCandidate, ...]:
        extraction = self.registry["be_argument_extraction"].render({
            "MESSAGE_JSON": message,
            "SOURCE_ROLE_JSON": source_role,
        })
        self._record("be_argument_extraction", extraction)
        polarity = 1 if hashlib.sha256(message.encode("utf-8")).digest()[0] % 2 else -1
        claim = FAKE_MARKER + ":claim:" + hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
        judgment = self.registry["be_evidence_judgment"].render({
            "CLAIM_JSON": claim,
            "MESSAGE_JSON": message,
        })
        self._record("be_evidence_judgment", judgment)
        strength = 0.25 + hashlib.sha256(claim.encode("utf-8")).digest()[0] / 510.0
        return (ArgumentCandidate(claim, polarity, strength, source_role),)


class FakeMainProvider:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, method: str, rendered_prompt: str, public_stance: float | None) -> str:
        self.calls.append({
            "fake": True,
            "method": method,
            "prompt_sha256": sha256_text(rendered_prompt),
            "provider": "local_deterministic_fake",
        })
        value = {
            "action": "read",
            "target_id": None,
            "text": None,
            "emotion": None,
            "public_stance": public_stance,
            "public_or_private": "none",
        }
        return canonical_json(value)


def _context_item(value: Mapping[str, Any]) -> ContextItem:
    return ContextItem(
        message_id=str(value["message_id"]),
        author_id=str(value["author_id"]) if value.get("author_id") is not None else None,
        text=str(value["text"]), visibility=str(value["visibility"]),
        tick=int(value["tick"]) if value.get("tick") is not None else None,
        thread_id=str(value["thread_id"]) if value.get("thread_id") is not None else None,
        parent_id=str(value["parent_id"]) if value.get("parent_id") is not None else None,
    )


def load_observations(path: Path) -> tuple[ObservationEnvelope, ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            value = raw["observation"]
            observation = CommonObservation(
                scenario_id=str(value["scenario_id"]), tick=int(value["tick"]),
                agent_id=str(value["agent_id"]), factual_registry=value["factual_registry"],
                current_event=value["current_event"],
                visible_context=tuple(_context_item(item) for item in value["visible_context"]),
                governance_artifacts=tuple(value["governance_artifacts"]),
                allowed_actions=tuple(value["allowed_actions"]),
                response_schema=value["response_schema"],
            )
            if observation.observation_sha256 != raw["observation_sha256"]:
                raise ValueError(f"observation hash mismatch at line {line_number}")
            rows.append(ObservationEnvelope(
                str(raw["observation_id"]), str(raw["activation_class"]), observation
            ))
    return tuple(rows)


def _adapter(method: str, auxiliary: FakeAuxiliaryRuntime, chuang_instruction: str) -> MethodAdapter:
    if method == "generative_agents":
        return GenerativeAgentsAdapter(
            embedder=auxiliary.embedding,
            importance_scorer=auxiliary.importance,
            reflector=auxiliary.reflect,
        )
    if method == "belief_engine":
        return BeliefEngineAdapter(
            seed_arguments=(
                ArgumentCandidate(FAKE_MARKER + ":balanced_seed_pro", 1, 0.5, "seed"),
                ArgumentCandidate(FAKE_MARKER + ":balanced_seed_con", -1, 0.5, "seed"),
            ),
            extractor=auxiliary.extract,
            embedder=auxiliary.embedding,
        )
    if method == "chuang_opinion_dynamics":
        return ChuangOpinionDynamicsAdapter(initial_opinion=0, update_instruction=chuang_instruction)
    raise ValueError(f"unsupported baseline method: {method}")


def _visible_context(observation: CommonObservation) -> list[dict[str, Any]]:
    return [
        {
            "message_id": item.message_id,
            "author_id": item.author_id,
            "text": item.text,
            "visibility": item.visibility,
            "tick": item.tick,
        }
        for item in observation.visible_context
    ]


def _record(
    *, envelope: ObservationEnvelope, method: str, decision: Mapping[str, Any],
    public_stance: float | None,
) -> dict[str, Any]:
    observation = envelope.observation
    return {
        "schema_version": "1.0.0",
        "protocol_version": "1.0.0",
        "experiment_id": "fake_baseline_chain_conformance_v1",
        "run_id": f"fake_baseline_chain_conformance_v1/{method}/fake/20260722",
        "method": method,
        "variant": "fake_conformance_only",
        "seed": 20260722,
        "agent_id": observation.agent_id,
        "tick": observation.tick,
        "profile_id": None,
        "interest_group": None,
        "visible_context": _visible_context(observation),
        "pre_state": None,
        "post_state": None,
        "particles": None,
        "particle_weights": None,
        "generated_text": decision["text"],
        "action": decision["action"],
        "emotion": decision["emotion"],
        "public_stance": public_stance,
        "public_or_private": decision["public_or_private"],
        "model_status": "not_applicable",
        "failure": None,
        "source_record_id": envelope.observation_id,
    }


def run_fake_baseline_conformance(
    *, observations_path: Path, prompt_registry: FrozenPromptRegistry,
    output_root: Path, unified_schema_path: Path,
) -> dict[str, Any]:
    envelopes = load_observations(observations_path)
    schema = json.loads(unified_schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    renderer = SharedPromptRenderer()
    auxiliary = FakeAuxiliaryRuntime(prompt_registry)
    provider = FakeMainProvider()
    method_reports: dict[str, Any] = {}
    chuang_instruction = prompt_registry["chuang_confirmation_update"].text.strip()

    for method in BASELINE_METHODS:
        adapters: dict[str, MethodAdapter] = {}
        records: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for envelope in envelopes:
            observation = envelope.observation
            adapter = adapters.get(observation.agent_id)
            if adapter is None:
                adapter = _adapter(method, auxiliary, chuang_instruction)
                adapter.initialize(AgentSeed(
                    agent_id=observation.agent_id, profile_id=None,
                    interest_group=None, stable_profile=None, initial_state=None,
                ))
                adapters[observation.agent_id] = adapter
            snapshot = adapter.observe(observation)
            bundle = renderer.render(observation, snapshot)
            raw = provider.generate(
                method=method, rendered_prompt=bundle.rendered,
                public_stance=adapter.export_probe(),
            )
            decision = parse_structured_decision(raw, observation.allowed_actions)
            adapter.commit_decision(observation, decision)
            row = _record(
                envelope=envelope, method=method, decision=decision,
                public_stance=adapter.export_probe(),
            )
            errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
            if errors:
                raise ValueError(f"unified schema failed for {method}: {errors[0].message}")
            records.append(row)
            audit_rows.append({
                "fake_marker": FAKE_MARKER,
                "method": method,
                "observation_id": envelope.observation_id,
                "observation_sha256": observation.observation_sha256,
                "rendered_prompt_sha256": bundle.rendered_sha256,
                "common_section_sha256": {
                    key: value for key, value in bundle.section_sha256.items()
                    if key != "method_cognitive_context"
                },
                "method_context_sha256": bundle.section_sha256["method_cognitive_context"],
            })
        method_dir = output_root / method / "fake" / "20260722"
        method_dir.mkdir(parents=True, exist_ok=True)
        records_path = method_dir / "records.jsonl"
        audit_path = method_dir / "prompt_audit.jsonl"
        for path, rows in ((records_path, records), (audit_path, audit_rows)):
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(canonical_json(row) + "\n")
        method_reports[method] = {
            "record_count": len(records),
            "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            "prompt_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "agent_count": len(adapters),
        }

    report = {
        "schema_version": "fake-baseline-chain-conformance-v1",
        "status": "passed_not_a_paper_result",
        "fake_marker": FAKE_MARKER,
        "model_calls_real": 0,
        "main_fake_call_count": len(provider.calls),
        "auxiliary_fake_call_count": len(auxiliary.calls),
        "input_observation_count": len(envelopes),
        "methods": method_reports,
        "frozen_prompt_sha256": {
            key: prompt.sha256 for key, prompt in sorted(prompt_registry.prompts.items())
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "conformance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
