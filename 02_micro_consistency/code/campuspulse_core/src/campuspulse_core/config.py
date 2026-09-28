"""Scenario loading, evidence provenance, and strict validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_SOURCE_TYPES = {"empirical_proxy", "derived", "operator_input", "assumption"}
NEED_TYPES = {
    "academic_information",
    "emotional_support",
    "safety_report",
    "campus_service",
}
URGENCY_LEVELS = ("low", "medium", "high", "critical")

REQUIRED_PARAMETERS = {
    "rounds",
    "base_arrival_rate",
    "shock_curve",
    "need_mix",
    "urgency_mix",
    "natural_response_probability",
    "mean_replies_if_responded",
    "reply_dispersion",
    "natural_resolution_probability",
    "work_units_by_urgency",
    "routing",
    "department_capacity",
    "faq_eligible_need_types",
    "faq_effectiveness",
    "peer_boost_capacity",
    "peer_boost_success",
    "dynamic_backlog_threshold",
    "dynamic_capacity_increment",
    "dynamic_capacity_limit",
    "cost_weights",
}


class ConfigError(ValueError):
    """Raised when a scenario violates the shared contract."""


@dataclass(frozen=True)
class EvidenceParameter:
    name: str
    value: Any
    source_type: str
    evidence: str
    limitations: str = ""


class ScenarioConfig:
    """Validated scenario configuration with auditable parameter provenance."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        self.raw = raw
        self.source_path = source_path
        self.scenario_id = str(raw.get("scenario_id", ""))
        self.version = str(raw.get("version", ""))
        self.title = str(raw.get("title", ""))
        self.description = str(raw.get("description", ""))
        self.limitations = list(raw.get("limitations", []))
        params = raw.get("parameters")
        if not isinstance(params, dict):
            raise ConfigError("parameters must be an object")
        self.parameters: dict[str, EvidenceParameter] = {}
        for name, item in params.items():
            if not isinstance(item, dict):
                raise ConfigError(f"parameters.{name} must be an evidence object")
            self.parameters[name] = EvidenceParameter(
                name=name,
                value=item.get("value"),
                source_type=str(item.get("source_type", "")),
                evidence=str(item.get("evidence", "")),
                limitations=str(item.get("limitations", "")),
            )
        self.validate()

    def get(self, name: str) -> Any:
        try:
            return self.parameters[name].value
        except KeyError as exc:
            raise ConfigError(f"missing parameter: {name}") from exc

    def source_counts(self) -> dict[str, int]:
        counts = {key: 0 for key in sorted(ALLOWED_SOURCE_TYPES)}
        for param in self.parameters.values():
            counts[param.source_type] += 1
        return counts

    def fingerprint(self) -> str:
        canonical = json.dumps(self.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not self.scenario_id:
            raise ConfigError("scenario_id is required")
        if not self.version:
            raise ConfigError("version is required")
        missing = sorted(REQUIRED_PARAMETERS - self.parameters.keys())
        if missing:
            raise ConfigError(f"missing parameters: {', '.join(missing)}")
        for name, param in self.parameters.items():
            if param.source_type not in ALLOWED_SOURCE_TYPES:
                raise ConfigError(
                    f"parameters.{name}.source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}"
                )
            if not param.evidence.strip():
                raise ConfigError(f"parameters.{name}.evidence is required")

        rounds = self.get("rounds")
        if not isinstance(rounds, int) or rounds <= 0:
            raise ConfigError("parameters.rounds.value must be a positive integer")
        self._non_negative_number("base_arrival_rate")
        self._non_negative_number("reply_dispersion", strictly_positive=True)
        self._non_negative_number("peer_boost_capacity")
        self._non_negative_number("dynamic_backlog_threshold")
        self._non_negative_number("dynamic_capacity_increment")
        self._non_negative_number("dynamic_capacity_limit")

        curve = self.get("shock_curve")
        if not isinstance(curve, list) or not curve:
            raise ConfigError("shock_curve must be a non-empty list")
        if any(not isinstance(x, (int, float)) or x < 0 for x in curve):
            raise ConfigError("shock_curve entries must be non-negative numbers")

        need_mix = self._distribution("need_mix", NEED_TYPES)
        urgency_mix = self._distribution("urgency_mix", set(URGENCY_LEVELS))
        if not need_mix or not urgency_mix:
            raise ConfigError("need_mix and urgency_mix cannot be empty")

        for name in (
            "natural_response_probability",
            "natural_resolution_probability",
            "faq_effectiveness",
            "peer_boost_success",
        ):
            self._probability_mapping(name, NEED_TYPES)

        reply_means = self.get("mean_replies_if_responded")
        self._mapping_keys("mean_replies_if_responded", reply_means, NEED_TYPES)
        if any(not isinstance(v, (int, float)) or v < 1 for v in reply_means.values()):
            raise ConfigError("mean_replies_if_responded values must be >= 1")

        work_units = self.get("work_units_by_urgency")
        self._mapping_keys("work_units_by_urgency", work_units, set(URGENCY_LEVELS))
        if any(not isinstance(v, int) or v <= 0 for v in work_units.values()):
            raise ConfigError("work_units_by_urgency values must be positive integers")

        routing = self.get("routing")
        self._mapping_keys("routing", routing, NEED_TYPES)
        if any(not isinstance(v, str) or not v for v in routing.values()):
            raise ConfigError("routing values must be department names")

        capacities = self.get("department_capacity")
        if not isinstance(capacities, dict) or not capacities:
            raise ConfigError("department_capacity must be a non-empty mapping")
        if any(not isinstance(v, int) or v < 0 for v in capacities.values()):
            raise ConfigError("department capacities must be non-negative integers")
        missing_departments = sorted(set(routing.values()) - capacities.keys())
        if missing_departments:
            raise ConfigError(f"routing references missing departments: {missing_departments}")

        eligible = self.get("faq_eligible_need_types")
        if not isinstance(eligible, list) or any(x not in NEED_TYPES for x in eligible):
            raise ConfigError("faq_eligible_need_types contains an unknown need type")

        cost_weights = self.get("cost_weights")
        expected_costs = {
            "faq_reach",
            "peer_action",
            "service_work_unit",
            "dynamic_capacity_unit",
            "routing_action",
        }
        self._mapping_keys("cost_weights", cost_weights, expected_costs)
        if any(not isinstance(v, (int, float)) or v < 0 for v in cost_weights.values()):
            raise ConfigError("cost weights must be non-negative")

    def _non_negative_number(self, name: str, strictly_positive: bool = False) -> None:
        value = self.get(name)
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be numeric")
        if value < 0 or (strictly_positive and value <= 0):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ConfigError(f"{name} must be {qualifier}")

    def _mapping_keys(self, name: str, value: Any, expected: set[str]) -> None:
        if not isinstance(value, dict):
            raise ConfigError(f"{name} must be a mapping")
        if set(value) != expected:
            raise ConfigError(f"{name} keys must be exactly {sorted(expected)}")

    def _distribution(self, name: str, expected: set[str]) -> dict[str, float]:
        value = self.get(name)
        self._mapping_keys(name, value, expected)
        if any(not isinstance(v, (int, float)) or v < 0 for v in value.values()):
            raise ConfigError(f"{name} values must be non-negative")
        if abs(sum(value.values()) - 1.0) > 1e-9:
            raise ConfigError(f"{name} values must sum to 1")
        return value

    def _probability_mapping(self, name: str, expected: set[str]) -> None:
        value = self.get(name)
        self._mapping_keys(name, value, expected)
        if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in value.values()):
            raise ConfigError(f"{name} values must be probabilities in [0, 1]")


def load_scenario(path: str | Path) -> ScenarioConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ScenarioConfig(raw, source.resolve())
