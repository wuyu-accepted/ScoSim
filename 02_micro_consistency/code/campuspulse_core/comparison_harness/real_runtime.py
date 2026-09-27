from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .adapters import ArgumentCandidate
from .auxiliary import FrozenPromptRegistry
from .real_provider import BoundaryViolationError, RealModelProvider


IMPORTANCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"importance": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["importance"],
}
REFLECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"insights": {
        "type": "array", "maxItems": 5,
        "items": {"type": "string", "minLength": 1},
    }},
    "required": ["insights"],
}
EVIDENCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"strength": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["strength"],
}
STANCE_PROBE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"stance": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}},
    "required": ["stance"],
}


def extraction_schema(source_role: str) -> dict[str, Any]:
    if source_role not in {"self", "opponent"}:
        raise BoundaryViolationError("Belief Engine source role is outside the boundary")
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"arguments": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "polarity": {"type": "integer", "enum": [-1, 1]},
                    "source_role": {"type": "string", "enum": [source_role]},
                },
                "required": ["claim", "polarity", "source_role"],
            },
        }},
        "required": ["arguments"],
    }


def strict_api_subset(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Remove conditionals unsupported by strict Structured Outputs.

    The original complete schema is still applied locally by RealModelProvider.
    """
    forbidden = {"$schema", "$id", "title", "allOf", "not", "if", "then", "else",
                 "dependentRequired", "dependentSchemas", "uniqueItems"}

    def infer_type(value: Any) -> str | list[str] | None:
        values = value if isinstance(value, list) else [value]
        types: list[str] = []
        for item in values:
            if item is None:
                kind = "null"
            elif isinstance(item, bool):
                kind = "boolean"
            elif isinstance(item, int):
                kind = "integer"
            elif isinstance(item, float):
                kind = "number"
            elif isinstance(item, str):
                kind = "string"
            else:
                return None
            if kind not in types:
                types.append(kind)
        return types[0] if len(types) == 1 else types

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {
                str(key): clean(child)
                for key, child in value.items()
                if key not in forbidden
            }
            if "type" not in result:
                inferred = None
                if "const" in result:
                    inferred = infer_type(result["const"])
                elif "enum" in result:
                    inferred = infer_type(result["enum"])
                if inferred is not None:
                    result = {"type": inferred, **result}
            return result
        if isinstance(value, list):
            return [clean(child) for child in value]
        return value

    return clean(schema)


def load_frozen_prompt(path: Path, expected_sha256: str) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise BoundaryViolationError(f"evaluation prompt hash mismatch: {path.name}:{actual}")
    return raw.decode("utf-8")


class RealAuxiliaryRuntime:
    def __init__(self, provider: RealModelProvider, registry: FrozenPromptRegistry):
        self.provider = provider
        self.registry = registry
        self.calls: list[dict[str, Any]] = []

    def _log(self, call_id: str, result: Any) -> None:
        self.calls.append({
            "call_id": call_id, "cache_key": result.cache_key,
            "cache_hit": result.cache_hit, "raw_archive_path": result.raw_archive_path,
        })

    def embedding(self, text: str) -> tuple[float, ...]:
        result = self.provider.embed(text)
        self._log("shared_embedding", result)
        return result.vector

    def importance(self, text: str) -> float:
        prompt = self.registry["ga_importance"].render({"INPUT_TEXT_JSON": text})
        result = self.provider.call_json(
            call_id="ga_importance", prompt=prompt,
            schema=IMPORTANCE_SCHEMA, schema_name="ga_importance",
        )
        self._log("ga_importance", result)
        return float(result.parsed["importance"])

    def reflect(self, memories: tuple[str, ...]) -> tuple[str, ...]:
        prompt = self.registry["ga_reflection"].render({"MEMORY_STATEMENTS_JSON": memories})
        result = self.provider.call_json(
            call_id="ga_reflection", prompt=prompt,
            schema=REFLECTION_SCHEMA, schema_name="ga_reflection",
        )
        self._log("ga_reflection", result)
        return tuple(str(value) for value in result.parsed["insights"])

    def extract(self, message: str, source_role: str) -> tuple[ArgumentCandidate, ...]:
        prompt = self.registry["be_argument_extraction"].render({
            "MESSAGE_JSON": message, "SOURCE_ROLE_JSON": source_role,
        })
        result = self.provider.call_json(
            call_id="be_argument_extraction", prompt=prompt,
            schema=extraction_schema(source_role), schema_name="be_argument_extraction",
        )
        self._log("be_argument_extraction", result)
        candidates = []
        for raw in result.parsed["arguments"]:
            claim = str(raw["claim"])
            evidence_prompt = self.registry["be_evidence_judgment"].render({
                "CLAIM_JSON": claim, "MESSAGE_JSON": message,
            })
            evidence = self.provider.call_json(
                call_id="be_evidence_judgment", prompt=evidence_prompt,
                schema=EVIDENCE_SCHEMA, schema_name="be_evidence_judgment",
            )
            self._log("be_evidence_judgment", evidence)
            candidates.append(ArgumentCandidate(
                claim=claim, polarity=int(raw["polarity"]),
                strength=float(evidence.parsed["strength"]), source_role=source_role,
            ))
        return tuple(candidates)


class RealEvaluationRuntime:
    def __init__(self, provider: RealModelProvider, *, judge_prompt: str, stance_prompt: str):
        self.provider = provider
        self.judge_prompt = judge_prompt
        self.stance_prompt = stance_prompt

    def judge(self, rendered_case_prompt: str, full_schema: Mapping[str, Any]) -> Mapping[str, Any]:
        if not rendered_case_prompt.startswith(self.judge_prompt.split("{{FACT_REGISTRY_JSON}}", 1)[0]):
            raise BoundaryViolationError("dimension-2 judge prompt does not use the frozen prefix")
        api_schema = strict_api_subset(full_schema)
        result = self.provider.call_json(
            call_id="dimension2_text_judge", prompt=rendered_case_prompt,
            schema=api_schema, validation_schema=full_schema,
            schema_name="dimension2_text_judgement",
        )
        return result.parsed

    def stance_probe(self, memory_and_visible_context: Any) -> float:
        rendered = self.stance_prompt.replace(
            "{{MEMORY_AND_VISIBLE_CONTEXT_JSON}}",
            json.dumps(memory_and_visible_context, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")),
        )
        if "{{MEMORY_AND_VISIBLE_CONTEXT_JSON}}" in rendered:
            raise BoundaryViolationError("stance probe placeholder was not replaced")
        result = self.provider.call_json(
            call_id="stance_probe", prompt=rendered,
            schema=STANCE_PROBE_SCHEMA, schema_name="stance_probe",
        )
        return int(result.parsed["stance"]) / 2.0
