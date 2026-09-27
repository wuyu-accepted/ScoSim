from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ACTIONS = (
    "read", "post", "reply", "quote", "react", "private_message", "no_action"
)
EVALUATION_ACTIONS = ("ignore", "read", "post", "reply", "seek_help", "report_risk")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextItem:
    message_id: str
    author_id: str | None
    text: str
    visibility: str
    tick: int | None = None
    thread_id: str | None = None
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if self.visibility not in {"public", "private", "system"}:
            raise ValueError(f"invalid visibility: {self.visibility}")


@dataclass(frozen=True)
class AgentSeed:
    agent_id: str
    profile_id: str | None
    interest_group: str | None
    stable_profile: Mapping[str, Any] | None
    initial_state: Mapping[str, float] | None
    initial_particles: Sequence[Mapping[str, float]] | None = None
    particle_weights: Sequence[float] | None = None


@dataclass(frozen=True)
class CommonObservation:
    """Only information supplied by the shared forum kernel.

    Method adapters receive this immutable value and cannot add messages, future
    state, evaluation labels, or another condition's state.
    """

    scenario_id: str
    tick: int
    agent_id: str
    factual_registry: Mapping[str, Any]
    current_event: Mapping[str, Any]
    visible_context: tuple[ContextItem, ...]
    governance_artifacts: tuple[Mapping[str, Any], ...]
    allowed_actions: tuple[str, ...] = ACTIONS
    response_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.allowed_actions).difference(set(ACTIONS) | set(EVALUATION_ACTIONS))
        if unknown:
            raise ValueError(f"actions outside shared vocabulary: {sorted(unknown)}")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")

    def public_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tick": self.tick,
            "agent_id": self.agent_id,
            "factual_registry": self.factual_registry,
            "current_event": self.current_event,
            "visible_context": [vars(item) for item in self.visible_context],
            "governance_artifacts": list(self.governance_artifacts),
            "allowed_actions": list(self.allowed_actions),
            "response_schema": self.response_schema,
        }

    @property
    def observation_sha256(self) -> str:
        return sha256_json(self.public_payload())


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    include_profile: bool
    particle_count: int
    resampling_diffusion: bool

    def __post_init__(self) -> None:
        if self.particle_count < 0:
            raise ValueError("particle_count must be non-negative")
        if self.particle_count <= 1 and self.resampling_diffusion:
            raise ValueError("resampling/diffusion requires more than one particle")


@dataclass(frozen=True)
class MethodSnapshot:
    method: str
    agent_id: str
    tick: int
    native_state: Mapping[str, Any]
    cognitive_context: Mapping[str, Any]
    source_observation_sha256: str
    auxiliary_calls: tuple[Mapping[str, Any], ...] = ()


class MethodAdapter(abc.ABC):
    """Narrow plug-in boundary: methods own cognition, never the forum."""

    method_id: str

    @abc.abstractmethod
    def initialize(self, seed: AgentSeed) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self, observation: CommonObservation) -> MethodSnapshot:
        raise NotImplementedError

    @abc.abstractmethod
    def commit_decision(
        self, observation: CommonObservation, decision: Mapping[str, Any]
    ) -> None:
        """Update native state only after the shared action has been generated."""
        raise NotImplementedError

    @abc.abstractmethod
    def export_probe(self) -> float | None:
        """Return preregistered [-1,1] stance probe, or None when inapplicable."""
        raise NotImplementedError
