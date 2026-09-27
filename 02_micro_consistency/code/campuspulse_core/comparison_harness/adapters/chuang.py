from __future__ import annotations

import re
import json
from typing import Any, Mapping

from ..contracts import AgentSeed, CommonObservation, MethodAdapter, MethodSnapshot


OPINION_SPACE = (-2, -1, 0, 1, 2)
CONFIRMATION_BIAS_INSTRUCTION = (
    "Like humans, weigh belief-supporting information more readily and "
    "belief-contradicting information less readily. Rely only on the supplied "
    "closed-world forum history and report an updated integer opinion in {-2,-1,0,1,2}."
)


class ChuangOpinionDynamicsAdapter(MethodAdapter):
    """Forum wrapper for Chuang et al.'s five-level, prompt-induced dynamics."""

    method_id = "chuang_opinion_dynamics"

    def __init__(
        self, initial_opinion: int | None = None, *, update_instruction: str | None = None
    ):
        self.configured_initial_opinion = initial_opinion
        self.update_instruction = update_instruction or CONFIRMATION_BIAS_INSTRUCTION
        if not self.update_instruction.strip():
            raise ValueError("Chuang update instruction cannot be empty")
        self.current_opinion: int | None = None
        self.seed: AgentSeed | None = None
        self.history: list[dict[str, Any]] = []

    def initialize(self, seed: AgentSeed) -> None:
        self.seed = seed
        opinion = self.configured_initial_opinion
        if opinion is None and seed.initial_state is not None:
            raw = float(seed.initial_state["support"]) - float(seed.initial_state["opposition"])
            opinion = min(OPINION_SPACE, key=lambda candidate: abs(raw - candidate / 2.0))
        if opinion not in OPINION_SPACE:
            raise ValueError("initial Chuang opinion must be one of -2,-1,0,1,2")
        self.current_opinion = opinion
        self.history.clear()
        if seed.stable_profile:
            self.history.append({
                "tick": 0,
                "kind": "role_background",
                "text": json.dumps(
                    seed.stable_profile, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ),
                "closed_world": True,
            })

    def observe(self, observation: CommonObservation) -> MethodSnapshot:
        if self.current_opinion is None:
            raise RuntimeError("adapter is not initialized")
        prior_history = list(self.history)
        # Chuang et al.'s mechanism retains the interaction itself, not only
        # an identifier saying that an interaction occurred.  Admit each
        # permitted message once into the method-native closed-world history;
        # no hidden/future content is available at this boundary.
        admitted = {
            str(item.get("message_id")) for item in self.history
            if item.get("kind") == "observed_interaction"
        }
        for item in observation.visible_context:
            if item.message_id in admitted:
                continue
            self.history.append({
                "tick": observation.tick,
                "kind": "observed_interaction",
                "message_id": item.message_id,
                "author_id": item.author_id,
                "visibility": item.visibility,
                "text": item.text,
                "closed_world": True,
            })
            admitted.add(item.message_id)
        return MethodSnapshot(
            self.method_id, observation.agent_id, observation.tick,
            {"opinion": self.current_opinion, "history_length": len(self.history)},
            {
                "current_opinion": self.current_opinion,
                # Current visible items remain solely in the shared observation
                # sections for this decision. They enter native memory now and
                # become available here from the next tick onward.
                "closed_world_interaction_memory": prior_history,
                "confirmation_bias_update_instruction": self.update_instruction,
            },
            observation.observation_sha256,
        )

    @staticmethod
    def parse_opinion(value: object) -> int:
        if isinstance(value, int) and value in OPINION_SPACE:
            return value
        matches = re.findall(r"(?<!\d)[+-]?[0-2](?!\d)", str(value))
        if not matches:
            raise ValueError("no five-level opinion found")
        opinion = int(matches[-1])
        if opinion not in OPINION_SPACE:
            raise ValueError("opinion is outside native space")
        return opinion

    def commit_decision(self, observation: CommonObservation, decision: Mapping[str, Any]) -> None:
        pre = self.current_opinion
        if decision.get("public_stance") is None:
            raw_stance = None
            post = pre
        else:
            raw_stance = max(-1.0, min(1.0, float(decision["public_stance"])))
            scaled = raw_stance * 2.0
            # The shared response contract is continuous.  Project it
            # deterministically to Chuang et al.'s native {-2,-1,0,1,2}
            # instead of failing a valid shared-schema response.
            post = min(OPINION_SPACE, key=lambda value: (abs(value - scaled), abs(value), value))
        self.history.append({
            "tick": observation.tick,
            "pre_opinion": pre,
            "visible_message_ids": [item.message_id for item in observation.visible_context],
            "action": decision.get("action"),
            "text": decision.get("text"),
            "raw_public_stance": raw_stance,
            "post_opinion": post,
        })
        self.current_opinion = post

    def export_probe(self) -> float | None:
        return None if self.current_opinion is None else self.current_opinion / 2.0
