from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import CommonObservation, MethodSnapshot, canonical_json


SECTION_ORDER = (
    "system_boundary",
    "closed_world_facts",
    "scenario_and_event",
    "visible_forum_context",
    "method_cognitive_context",
    "allowed_actions",
    "structured_response_schema",
)
COMMON_SECTIONS = tuple(name for name in SECTION_ORDER if name != "method_cognitive_context")
STANCE_PROPOSITION = "The currently announced housing-allocation process is legitimate and acceptable."
PUBLIC_STANCE_SEMANTICS = (
    "public_stance is your stance toward stance_proposition: -1 means strong "
    "opposition, 0 means neutral or mixed, and +1 means strong support. The "
    "generated text and public_stance must express the same direction."
)

SYSTEM_BOUNDARY = {
    "role": "Act as one synthetic campus-forum resident in a controlled simulation.",
    "safety": "Do not claim to be a real person and do not infer facts outside the supplied closed world.",
    "decision_rule": "Use only the visible information below and return exactly one permitted forum action.",
    "stance_proposition": STANCE_PROPOSITION,
    "public_stance_semantics": PUBLIC_STANCE_SEMANTICS,
}


@dataclass(frozen=True)
class PromptBundle:
    method: str
    observation_sha256: str
    section_text: Mapping[str, str]
    section_sha256: Mapping[str, str]
    rendered: str
    rendered_sha256: str


class SharedPromptRenderer:
    """Sole renderer for every method and CampusPulse ablation.

    Common section bytes are derived only from CommonObservation. An adapter can
    influence exactly one section: method_cognitive_context.
    """

    @staticmethod
    def _dump(value: Any) -> str:
        return canonical_json(value)

    def render(
        self, observation: CommonObservation, snapshot: MethodSnapshot
    ) -> PromptBundle:
        if snapshot.agent_id != observation.agent_id or snapshot.tick != observation.tick:
            raise ValueError("snapshot does not belong to this observation")
        if snapshot.source_observation_sha256 != observation.observation_sha256:
            raise ValueError("snapshot was derived from a different observation")
        sections = {
            "system_boundary": self._dump(SYSTEM_BOUNDARY),
            "closed_world_facts": self._dump(observation.factual_registry),
            "scenario_and_event": self._dump({
                "scenario_id": observation.scenario_id,
                "tick": observation.tick,
                "current_event": observation.current_event,
                "governance_artifacts": list(observation.governance_artifacts),
            }),
            "visible_forum_context": self._dump(
                [vars(item) for item in observation.visible_context]
            ),
            "method_cognitive_context": self._dump(snapshot.cognitive_context),
            "allowed_actions": self._dump(list(observation.allowed_actions)),
            "structured_response_schema": self._dump(observation.response_schema),
        }
        hashes = {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in sections.items()
        }
        rendered = "\n".join(
            f"## {name}\n{sections[name]}" for name in SECTION_ORDER
        )
        return PromptBundle(
            method=snapshot.method,
            observation_sha256=observation.observation_sha256,
            section_text=sections,
            section_sha256=hashes,
            rendered=rendered,
            rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        )


def parse_structured_decision(text: str, allowed_actions: tuple[str, ...]) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("decision must be one JSON object")
    required = {"action", "target_id", "text", "emotion", "public_stance", "public_or_private"}
    if set(value) != required:
        raise ValueError(f"decision keys must equal {sorted(required)}")
    if value["action"] not in allowed_actions:
        raise ValueError("decision action is not allowed")
    if value["public_or_private"] not in {"public", "private", "none"}:
        raise ValueError("invalid public_or_private")
    stance = value["public_stance"]
    if stance is not None and (not isinstance(stance, (int, float)) or not -1 <= stance <= 1):
        raise ValueError("public_stance must be null or in [-1,1]")
    return value
