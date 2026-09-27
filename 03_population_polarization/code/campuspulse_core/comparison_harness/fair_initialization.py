from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .adapters import ArgumentCandidate
from .contracts import AgentSeed, canonical_json


SCHEMA_VERSION = "dimension2-fair-role-initialization-v1.0.0"


def _role_card(agent: Mapping[str, Any]) -> dict[str, Any]:
    persona = dict(agent.get("persona") or {})
    stable = dict(agent.get("stable_persona") or {})
    semantic = dict(agent.get("semantic") or {})
    return {
        "profile_summary": persona.get("profile_summary"),
        "situational_signature": persona.get("situational_signature"),
        "uncertainty_behavior": persona.get("uncertainty_behavior"),
        "language_tendencies": list(persona.get("language_tendencies") or ()),
        "behavior_boundaries": list(persona.get("do_not_simulate") or ()),
        "interaction_role": stable.get("interaction_role"),
        "forum_tendency_distribution": stable.get("forum_tendency_distribution"),
        "trust_style": stable.get("trust_style"),
        "language_style": stable.get("language_style"),
        "activity_rhythm": stable.get("activity_rhythm"),
        "attention_budget": stable.get("attention_budget"),
        "semantic_speech_act": semantic.get("speech_act"),
        "semantic_stance": semantic.get("stance"),
        "scenario_relevance": agent.get("scenario_relevance"),
    }


@dataclass(frozen=True)
class FairRoleInitialization:
    agent_id: str
    profile_id: str
    role_card: Mapping[str, Any]
    role_card_sha256: str
    initial_support: float
    initial_opposition: float

    @property
    def initial_net_stance(self) -> float:
        return self.initial_support - self.initial_opposition

    def agent_seed(self, *, initial_state: Mapping[str, float] | None = None,
                   initial_particles=None, particle_weights=None) -> AgentSeed:
        state = dict(initial_state) if initial_state is not None else {
            "support": self.initial_support,
            "opposition": self.initial_opposition,
        }
        return AgentSeed(
            agent_id=self.agent_id,
            profile_id=self.profile_id,
            interest_group=None,
            stable_profile=dict(self.role_card),
            initial_state=state,
            initial_particles=initial_particles,
            particle_weights=particle_weights,
        )

    def belief_arguments(self) -> tuple[ArgumentCandidate, ...]:
        # The role card is encoded symmetrically so it is available to the
        # structured argument memory without inventing an initial stance.
        text = canonical_json(self.role_card)
        support_strength = max(0.0, min(1.0, 0.5 + self.initial_net_stance / 2.0))
        opposition_strength = max(0.0, min(1.0, 0.5 - self.initial_net_stance / 2.0))
        return (
            ArgumentCandidate(
                "Role background, support-side encoding: " + text,
                1, support_strength, "seed",
            ),
            ArgumentCandidate(
                "Role background, opposition-side encoding: " + text,
                -1, opposition_strength, "seed",
            ),
        )

    def audit_row(self, method: str) -> dict[str, Any]:
        mapping = {
            "campuspulse": "stable_profile",
            "belief_engine": "symmetric_structured_seed_arguments",
            "generative_agents": "initial_profile_memory",
            "chuang_opinion_dynamics": "closed_world_role_history",
        }[method]
        return {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "agent_id": self.agent_id,
            "profile_id": self.profile_id,
            "role_card_sha256": self.role_card_sha256,
            "method_native_mapping": mapping,
            "initial_net_stance": self.initial_net_stance,
        }


def load_fair_role_initializations(path: Path) -> dict[str, FairRoleInitialization]:
    release = json.loads(path.read_text(encoding="utf-8"))
    agents = release.get("agents")
    if not isinstance(agents, list) or len(agents) != 1000:
        raise ValueError("fair initialization requires the frozen 1000-agent hero bridge")
    result: dict[str, FairRoleInitialization] = {}
    for raw in agents:
        agent_id = str(raw.get("llm_agent_id") or "")
        state = raw.get("state")
        if not agent_id or not isinstance(state, Mapping):
            raise ValueError("hero bridge agent is missing identity or state")
        card = _role_card(raw)
        card_sha = hashlib.sha256(canonical_json(card).encode("utf-8")).hexdigest()
        if agent_id in result:
            raise ValueError("duplicate hero bridge identity")
        result[agent_id] = FairRoleInitialization(
            agent_id=agent_id,
            profile_id=str(raw.get("profile_bundle_id") or f"profile-{agent_id}"),
            role_card=card,
            role_card_sha256=card_sha,
            initial_support=float(state.get("support", 0.3)),
            initial_opposition=float(state.get("opposition", 0.3)),
        )
    return result
