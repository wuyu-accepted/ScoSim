from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..contracts import AblationSpec, AgentSeed, CommonObservation, MethodAdapter, MethodSnapshot


DIMENSION_2 = {
    "context_only": AblationSpec("context_only", False, 0, False),
    "profile_only": AblationSpec("profile_only", True, 0, False),
    "state_only": AblationSpec("state_only", False, 10, True),
    "full": AblationSpec("full", True, 10, True),
}
DIMENSION_3 = {
    "full": AblationSpec("full", True, 10, True),
    "single_particle": AblationSpec("single_particle", True, 1, False),
    "no_profile": AblationSpec("no_profile", False, 10, True),
    "no_profile_no_smc": AblationSpec("no_profile_no_smc", False, 0, False),
}


class CampusPulseAdapter(MethodAdapter):
    """Thin view over the native CampusPulse state; no forum behavior lives here."""

    method_id = "campuspulse"

    def __init__(self, ablation: AblationSpec):
        self.ablation = ablation
        self.seed: AgentSeed | None = None
        self.state: dict[str, float] | None = None
        self.particles: list[dict[str, float]] | None = None
        self.weights: list[float] | None = None

    def initialize(self, seed: AgentSeed) -> None:
        self.seed = seed
        self.state = dict(seed.initial_state) if seed.initial_state is not None else None
        source_particles = list(seed.initial_particles or [])
        if self.ablation.particle_count == 0:
            self.particles, self.weights = None, None
        elif self.ablation.particle_count == 1:
            if seed.initial_state is None:
                raise ValueError("single-particle variant requires initial_state")
            self.particles, self.weights = [dict(seed.initial_state)], [1.0]
        else:
            if len(source_particles) != self.ablation.particle_count:
                raise ValueError("frozen particle count does not match variant")
            if seed.particle_weights is None or len(seed.particle_weights) != len(source_particles):
                raise ValueError("particle weights do not match particles")
            self.particles = [dict(row) for row in source_particles]
            self.weights = [float(value) for value in seed.particle_weights]

    def observe(self, observation: CommonObservation) -> MethodSnapshot:
        if self.seed is None:
            raise RuntimeError("adapter is not initialized")
        # Omitted components are absent, rather than described as disabled, so
        # the model never sees a condition label or an ablation hint.
        cognitive: dict[str, Any] = {}
        if self.ablation.include_profile:
            cognitive["stable_profile"] = deepcopy(self.seed.stable_profile)
        if self.ablation.particle_count:
            cognitive.update({
                "predecision_state": deepcopy(self.state),
                "particles": deepcopy(self.particles),
                "particle_weights": deepcopy(self.weights),
            })
        return MethodSnapshot(
            self.method_id, observation.agent_id, observation.tick,
            {"state": deepcopy(self.state), "particles": deepcopy(self.particles), "weights": deepcopy(self.weights)},
            cognitive, observation.observation_sha256,
        )

    def commit_decision(self, observation: CommonObservation, decision: Mapping[str, Any]) -> None:
        native = decision.get("method_update")
        if native is None:
            return
        if not isinstance(native, Mapping):
            raise ValueError("method_update must be an object")
        if "state" in native:
            self.state = {str(k): float(v) for k, v in native["state"].items()}
        if self.ablation.particle_count and "particles" in native:
            particles = [dict(row) for row in native["particles"]]
            weights = [float(value) for value in native["particle_weights"]]
            if len(particles) != self.ablation.particle_count or len(weights) != len(particles):
                raise ValueError("method update violates frozen particle count")
            self.particles, self.weights = particles, weights

    def export_probe(self) -> float | None:
        if self.state is None or not self.ablation.particle_count:
            return None
        return max(-1.0, min(1.0, float(self.state["support"]) - float(self.state["opposition"])))
