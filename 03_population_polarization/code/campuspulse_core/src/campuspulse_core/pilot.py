"""Privacy-preserving evaluation for preregistered campus intervention pilots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import NEED_TYPES, URGENCY_LEVELS
from .policies import POLICIES


PILOT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCENARIOS = {
    "exam_pressure",
    "course_selection",
    "campus_safety",
    "campus_services",
}
EVENT_FIELDS = (
    "case_token",
    "scenario_id",
    "assignment_arm",
    "assigned_at",
    "observed_until",
    "need_type",
    "urgency",
    "eligible",
    "intervention_delivered",
    "outcome_observed",
    "resolved_within_sla",
    "resolution_minutes",
    "work_units",
    "governance_cost",
)
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{12,64}$")
PILOT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{5,63}$")


class PilotDataError(ValueError):
    """Raised before analysis when a pilot manifest or event log is invalid."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        preview = errors[:12]
        remaining = len(errors) - len(preview)
        message = "pilot input validation failed:\n- " + "\n- ".join(preview)
        if remaining > 0:
            message += f"\n- ... {remaining} additional error(s)"
        super().__init__(message)


@dataclass(frozen=True)
class PilotArm:
    arm_id: str
    expected_allocation: float


@dataclass(frozen=True)
class PilotManifest:
    pilot_id: str
    scenario_id: str
    preregistered_at: datetime
    pilot_start: datetime
    pilot_end: datetime
    timezone_name: str
    arms: tuple[PilotArm, ...]
    sla_minutes: float
    minimum_eligible_per_arm: int
    minimum_outcome_completeness: float
    minimum_treatment_delivery: float
    maximum_allocation_deviation: float
    minimum_cell_size: int
    cost_unit: str
    raw: dict[str, Any]

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(arm.arm_id for arm in self.arms)

    @property
    def allocation(self) -> dict[str, float]:
        return {arm.arm_id: arm.expected_allocation for arm in self.arms}


@dataclass(frozen=True)
class PilotEvent:
    scenario_id: str
    assignment_arm: str
    assigned_at: datetime
    observed_until: datetime
    need_type: str
    urgency: str
    eligible: bool
    intervention_delivered: bool
    outcome_observed: bool
    resolved_within_sla: bool | None
    resolution_minutes: float | None
    work_units: float
    governance_cost: float


def _expect_mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"manifest {path}: must be an object")
        return {}
    return value


def _expect_exact_keys(
    value: dict[str, Any], expected: set[str], path: str, errors: list[str]
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        errors.append(f"manifest {path}: missing keys {missing}")
    if extra:
        errors.append(f"manifest {path}: unexpected keys {extra}")


def _parse_timestamp(value: Any, path: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: must be a timezone-aware ISO-8601 timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{path}: must be a valid ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(f"{path}: timezone offset is required")
        return None
    return parsed


def _number(
    value: Any,
    path: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path}: must be numeric")
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        errors.append(f"{path}: must be finite")
        return None
    if minimum is not None and numeric < minimum:
        errors.append(f"{path}: must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        errors.append(f"{path}: must be <= {maximum}")
    return numeric


def _integer(
    value: Any, path: str, errors: list[str], *, minimum: int = 0
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path}: must be an integer")
        return None
    if value < minimum:
        errors.append(f"{path}: must be >= {minimum}")
    return value


def load_pilot_manifest(path: str | Path) -> PilotManifest:
    """Load a fail-closed manifest for a randomized parallel-group pilot."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotDataError([f"manifest: cannot be read as UTF-8 JSON ({exc})"]) from exc
    errors: list[str] = []
    root = _expect_mapping(raw, "root", errors)
    _expect_exact_keys(
        root,
        {
            "schema_version",
            "pilot_id",
            "scenario_id",
            "design",
            "registration",
            "outcomes",
            "readiness",
            "cost_unit",
            "privacy",
        },
        "root",
        errors,
    )
    if root.get("schema_version") != PILOT_SCHEMA_VERSION:
        errors.append(
            f"manifest schema_version: must equal {PILOT_SCHEMA_VERSION}"
        )
    pilot_id = root.get("pilot_id")
    if not isinstance(pilot_id, str) or not PILOT_ID_PATTERN.fullmatch(pilot_id):
        errors.append("manifest pilot_id: must match [a-z0-9][a-z0-9_-]{5,63}")
        pilot_id = "invalid_pilot"
    scenario_id = root.get("scenario_id")
    if scenario_id not in SUPPORTED_SCENARIOS:
        errors.append("manifest scenario_id: must be a bundled CampusPulse scenario")
        scenario_id = "invalid_scenario"

    design = _expect_mapping(root.get("design"), "design", errors)
    _expect_exact_keys(
        design,
        {"kind", "assignment_unit", "analysis_population", "arms"},
        "design",
        errors,
    )
    if design.get("kind") != "randomized_parallel":
        errors.append("manifest design.kind: only randomized_parallel is supported")
    if design.get("assignment_unit") != "demand_case":
        errors.append("manifest design.assignment_unit: must equal demand_case")
    if design.get("analysis_population") != "eligible_intention_to_treat":
        errors.append(
            "manifest design.analysis_population: must equal eligible_intention_to_treat"
        )
    arms_raw = design.get("arms")
    parsed_arms: list[PilotArm] = []
    seen_arm_ids: set[str] = set()
    if not isinstance(arms_raw, list) or len(arms_raw) < 2:
        errors.append("manifest design.arms: must contain baseline and at least one treatment")
    else:
        for index, item in enumerate(arms_raw):
            arm = _expect_mapping(item, f"design.arms[{index}]", errors)
            _expect_exact_keys(
                arm,
                {"id", "expected_allocation"},
                f"design.arms[{index}]",
                errors,
            )
            arm_id = arm.get("id")
            if arm_id not in POLICIES:
                errors.append(
                    f"manifest design.arms[{index}].id: must be a registered policy"
                )
                continue
            if arm_id in seen_arm_ids:
                errors.append(f"manifest design.arms[{index}].id: duplicate arm")
                continue
            seen_arm_ids.add(arm_id)
            allocation = _number(
                arm.get("expected_allocation"),
                f"manifest design.arms[{index}].expected_allocation",
                errors,
                minimum=0.0,
                maximum=1.0,
            )
            if allocation is not None:
                if allocation <= 0:
                    errors.append(
                        f"manifest design.arms[{index}].expected_allocation: must be > 0"
                    )
                parsed_arms.append(PilotArm(arm_id, allocation))
        if "baseline" not in seen_arm_ids:
            errors.append("manifest design.arms: baseline is required")
        if parsed_arms and abs(sum(arm.expected_allocation for arm in parsed_arms) - 1.0) > 1e-9:
            errors.append("manifest design.arms: expected allocations must sum to 1")

    registration = _expect_mapping(root.get("registration"), "registration", errors)
    _expect_exact_keys(
        registration,
        {"preregistered_at", "pilot_start", "pilot_end", "timezone"},
        "registration",
        errors,
    )
    preregistered_at = _parse_timestamp(
        registration.get("preregistered_at"),
        "manifest registration.preregistered_at",
        errors,
    )
    pilot_start = _parse_timestamp(
        registration.get("pilot_start"),
        "manifest registration.pilot_start",
        errors,
    )
    pilot_end = _parse_timestamp(
        registration.get("pilot_end"),
        "manifest registration.pilot_end",
        errors,
    )
    timezone_name = registration.get("timezone")
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        errors.append("manifest registration.timezone: must be a non-empty IANA name")
        timezone_name = "UTC"
    if preregistered_at and pilot_start and preregistered_at >= pilot_start:
        errors.append("manifest registration: preregistered_at must precede pilot_start")
    if pilot_start and pilot_end and pilot_start >= pilot_end:
        errors.append("manifest registration: pilot_start must precede pilot_end")

    outcomes = _expect_mapping(root.get("outcomes"), "outcomes", errors)
    _expect_exact_keys(outcomes, {"primary", "sla_minutes"}, "outcomes", errors)
    if outcomes.get("primary") != "resolved_within_sla":
        errors.append("manifest outcomes.primary: must equal resolved_within_sla")
    sla_minutes = _number(
        outcomes.get("sla_minutes"),
        "manifest outcomes.sla_minutes",
        errors,
        minimum=1.0,
    )

    readiness = _expect_mapping(root.get("readiness"), "readiness", errors)
    _expect_exact_keys(
        readiness,
        {
            "minimum_eligible_per_arm",
            "minimum_outcome_completeness",
            "minimum_treatment_delivery",
            "maximum_allocation_deviation",
            "minimum_cell_size",
        },
        "readiness",
        errors,
    )
    minimum_eligible = _integer(
        readiness.get("minimum_eligible_per_arm"),
        "manifest readiness.minimum_eligible_per_arm",
        errors,
        minimum=2,
    )
    minimum_completeness = _number(
        readiness.get("minimum_outcome_completeness"),
        "manifest readiness.minimum_outcome_completeness",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    minimum_treatment_delivery = _number(
        readiness.get("minimum_treatment_delivery"),
        "manifest readiness.minimum_treatment_delivery",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    maximum_allocation_deviation = _number(
        readiness.get("maximum_allocation_deviation"),
        "manifest readiness.maximum_allocation_deviation",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    minimum_cell_size = _integer(
        readiness.get("minimum_cell_size"),
        "manifest readiness.minimum_cell_size",
        errors,
        minimum=2,
    )
    if (
        minimum_eligible is not None
        and minimum_cell_size is not None
        and minimum_cell_size > minimum_eligible
    ):
        errors.append(
            "manifest readiness: minimum_cell_size cannot exceed minimum_eligible_per_arm"
        )

    cost_unit = root.get("cost_unit")
    if not isinstance(cost_unit, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{2,39}", cost_unit):
        errors.append("manifest cost_unit: must be a 3-40 character machine-safe label")
        cost_unit = "invalid_unit"

    privacy = _expect_mapping(root.get("privacy"), "privacy", errors)
    _expect_exact_keys(
        privacy,
        {
            "direct_identifiers_prohibited",
            "raw_text_prohibited",
            "case_token_persisted",
        },
        "privacy",
        errors,
    )
    if privacy.get("direct_identifiers_prohibited") is not True:
        errors.append("manifest privacy.direct_identifiers_prohibited: must be true")
    if privacy.get("raw_text_prohibited") is not True:
        errors.append("manifest privacy.raw_text_prohibited: must be true")
    if privacy.get("case_token_persisted") is not False:
        errors.append("manifest privacy.case_token_persisted: must be false")

    if errors:
        raise PilotDataError(errors)
    assert preregistered_at and pilot_start and pilot_end
    assert sla_minutes is not None
    assert minimum_eligible is not None
    assert minimum_completeness is not None
    assert minimum_treatment_delivery is not None
    assert maximum_allocation_deviation is not None
    assert minimum_cell_size is not None
    return PilotManifest(
        pilot_id=pilot_id,
        scenario_id=scenario_id,
        preregistered_at=preregistered_at,
        pilot_start=pilot_start,
        pilot_end=pilot_end,
        timezone_name=timezone_name,
        arms=tuple(parsed_arms),
        sla_minutes=sla_minutes,
        minimum_eligible_per_arm=minimum_eligible,
        minimum_outcome_completeness=minimum_completeness,
        minimum_treatment_delivery=minimum_treatment_delivery,
        maximum_allocation_deviation=maximum_allocation_deviation,
        minimum_cell_size=minimum_cell_size,
        cost_unit=cost_unit,
        raw=root,
    )


def _row_error(row_number: int, field: str, message: str) -> str:
    return f"events row {row_number} {field}: {message}"


def _csv_bool(
    value: str, row_number: int, field: str, errors: list[str], *, optional: bool = False
) -> bool | None:
    normalized = value.strip().lower()
    if optional and normalized == "":
        return None
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    errors.append(_row_error(row_number, field, "must be true/false or 1/0"))
    return None


def _csv_number(
    value: str,
    row_number: int,
    field: str,
    errors: list[str],
    *,
    optional: bool = False,
) -> float | None:
    normalized = value.strip()
    if optional and not normalized:
        return None
    try:
        numeric = float(normalized)
    except ValueError:
        errors.append(_row_error(row_number, field, "must be numeric"))
        return None
    if not math.isfinite(numeric) or numeric < 0:
        errors.append(_row_error(row_number, field, "must be finite and non-negative"))
        return None
    return numeric


def _csv_timestamp(
    value: str, row_number: int, field: str, errors: list[str]
) -> datetime | None:
    return _parse_timestamp(value, f"events row {row_number} {field}", errors)


def load_pilot_events(path: str | Path, manifest: PilotManifest) -> list[PilotEvent]:
    """Validate the entire event file before returning token-free event objects."""

    source = Path(path)
    errors: list[str] = []
    events: list[PilotEvent] = []
    seen_tokens: set[str] = set()
    try:
        handle = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PilotDataError([f"events: cannot be read ({exc})"]) from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EVENT_FIELDS):
            raise PilotDataError(
                [
                    "events header: must exactly match the documented allowlist and order; "
                    f"expected {list(EVENT_FIELDS)}"
                ]
            )
        for row_number, row in enumerate(reader, start=2):
            row_errors_before = len(errors)
            token = row["case_token"].strip()
            if not TOKEN_PATTERN.fullmatch(token):
                errors.append(
                    _row_error(row_number, "case_token", "must be an opaque 12-64 character token")
                )
            elif token in seen_tokens:
                errors.append(
                    _row_error(row_number, "case_token", "duplicates an earlier row")
                )
            else:
                seen_tokens.add(token)

            scenario_id = row["scenario_id"].strip()
            if scenario_id != manifest.scenario_id:
                errors.append(
                    _row_error(row_number, "scenario_id", "does not match the manifest")
                )
            arm = row["assignment_arm"].strip()
            if arm not in manifest.arm_ids:
                errors.append(
                    _row_error(row_number, "assignment_arm", "is not preregistered")
                )
            assigned_at = _csv_timestamp(
                row["assigned_at"], row_number, "assigned_at", errors
            )
            observed_until = _csv_timestamp(
                row["observed_until"], row_number, "observed_until", errors
            )
            if assigned_at and not manifest.pilot_start <= assigned_at < manifest.pilot_end:
                errors.append(
                    _row_error(row_number, "assigned_at", "must fall inside the pilot window")
                )
            if assigned_at and observed_until and observed_until < assigned_at:
                errors.append(
                    _row_error(row_number, "observed_until", "cannot precede assignment")
                )
            followup_minutes = (
                (observed_until - assigned_at).total_seconds() / 60
                if assigned_at and observed_until
                else None
            )

            need_type = row["need_type"].strip()
            if need_type not in NEED_TYPES:
                errors.append(_row_error(row_number, "need_type", "is not supported"))
            urgency = row["urgency"].strip()
            if urgency not in URGENCY_LEVELS:
                errors.append(_row_error(row_number, "urgency", "is not supported"))
            eligible = _csv_bool(row["eligible"], row_number, "eligible", errors)
            delivered = _csv_bool(
                row["intervention_delivered"],
                row_number,
                "intervention_delivered",
                errors,
            )
            outcome_observed = _csv_bool(
                row["outcome_observed"], row_number, "outcome_observed", errors
            )
            resolved = _csv_bool(
                row["resolved_within_sla"],
                row_number,
                "resolved_within_sla",
                errors,
                optional=True,
            )
            resolution_minutes = _csv_number(
                row["resolution_minutes"],
                row_number,
                "resolution_minutes",
                errors,
                optional=True,
            )
            work_units = _csv_number(
                row["work_units"], row_number, "work_units", errors
            )
            governance_cost = _csv_number(
                row["governance_cost"], row_number, "governance_cost", errors
            )

            if arm == "baseline" and delivered is True:
                errors.append(
                    _row_error(
                        row_number,
                        "intervention_delivered",
                        "must be false in the baseline arm",
                    )
                )
            if eligible is False and delivered is True:
                errors.append(
                    _row_error(
                        row_number,
                        "intervention_delivered",
                        "must be false for an ineligible case",
                    )
                )
            if outcome_observed is False and (resolved is not None or resolution_minutes is not None):
                errors.append(
                    _row_error(
                        row_number,
                        "outcome_observed",
                        "unobserved outcomes require blank resolution fields",
                    )
                )
            if outcome_observed is True and resolved is None:
                errors.append(
                    _row_error(
                        row_number,
                        "resolved_within_sla",
                        "is required when outcome_observed is true",
                    )
                )
            if resolved is True:
                if resolution_minutes is None:
                    errors.append(
                        _row_error(
                            row_number,
                            "resolution_minutes",
                            "is required for a resolved case",
                        )
                    )
                elif resolution_minutes > manifest.sla_minutes:
                    errors.append(
                        _row_error(
                            row_number,
                            "resolution_minutes",
                            "exceeds the SLA while resolved_within_sla is true",
                        )
                    )
                elif followup_minutes is not None and resolution_minutes > followup_minutes:
                    errors.append(
                        _row_error(
                            row_number,
                            "resolution_minutes",
                            "cannot exceed observed follow-up time",
                        )
                    )
            if resolved is False and resolution_minutes is not None and resolution_minutes <= manifest.sla_minutes:
                errors.append(
                    _row_error(
                        row_number,
                        "resolution_minutes",
                        "must exceed the SLA when resolved_within_sla is false",
                    )
                )
            if (
                outcome_observed is True
                and resolved is False
                and followup_minutes is not None
                and followup_minutes < manifest.sla_minutes
            ):
                errors.append(
                    _row_error(
                        row_number,
                        "observed_until",
                        "must cover the full SLA window for an observed unresolved outcome",
                    )
                )

            if len(errors) == row_errors_before:
                assert assigned_at and observed_until
                assert eligible is not None and delivered is not None
                assert outcome_observed is not None
                assert work_units is not None and governance_cost is not None
                events.append(
                    PilotEvent(
                        scenario_id=scenario_id,
                        assignment_arm=arm,
                        assigned_at=assigned_at,
                        observed_until=observed_until,
                        need_type=need_type,
                        urgency=urgency,
                        eligible=eligible,
                        intervention_delivered=delivered,
                        outcome_observed=outcome_observed,
                        resolved_within_sla=resolved,
                        resolution_minutes=resolution_minutes,
                        work_units=work_units,
                        governance_cost=governance_cost,
                    )
                )
    if not seen_tokens:
        errors.append("events: at least one data row is required")
    if errors:
        raise PilotDataError(errors)
    return events


def _file_identity(path: str | Path) -> dict[str, int | str]:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": source.stat().st_size}


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _distribution(events: list[PilotEvent], field: str, categories: tuple[str, ...] | set[str]) -> dict[str, float]:
    if not events:
        return {category: 0.0 for category in sorted(categories)}
    counts = Counter(getattr(event, field) for event in events)
    return {category: counts[category] / len(events) for category in sorted(categories)}


def _internal_arm_summary(
    events: list[PilotEvent], total_eligible: int
) -> dict[str, Any]:
    enrolled = len(events)
    eligible = [event for event in events if event.eligible]
    observed = [event for event in eligible if event.outcome_observed]
    successes = sum(event.resolved_within_sla is True for event in observed)
    urgent = [event for event in eligible if event.urgency in {"high", "critical"}]
    urgent_successes = sum(
        event.outcome_observed and event.resolved_within_sla is True for event in urgent
    )
    eligible_count = len(eligible)
    observed_count = len(observed)
    conservative_rate = successes / eligible_count if eligible_count else 0.0
    complete_case_rate = successes / observed_count if observed_count else 0.0
    delivery_rate = (
        sum(event.intervention_delivered for event in eligible) / eligible_count
        if eligible_count
        else 0.0
    )
    allocation_share = eligible_count / total_eligible if total_eligible else 0.0
    resolution_times = [
        event.resolution_minutes
        for event in observed
        if event.resolution_minutes is not None
    ]
    need_counts = Counter(event.need_type for event in eligible)
    urgency_counts = Counter(event.urgency for event in eligible)
    return {
        "enrolled_cases": enrolled,
        "eligible_cases": eligible_count,
        "observed_primary_outcomes": observed_count,
        "allocation_share": allocation_share,
        "delivery_rate": delivery_rate,
        "outcome_completeness": observed_count / eligible_count if eligible_count else 0.0,
        "conservative_itt_resolution_rate": conservative_rate,
        "conservative_itt_wilson_95": _wilson_interval(successes, eligible_count),
        "complete_case_resolution_rate": complete_case_rate,
        "complete_case_wilson_95": _wilson_interval(successes, observed_count),
        "urgent_eligible_cases": len(urgent),
        "urgent_conservative_resolution_rate": urgent_successes / len(urgent) if urgent else 0.0,
        "mean_work_units": _mean([event.work_units for event in eligible]),
        "mean_governance_cost": _mean([event.governance_cost for event in eligible]),
        "median_observed_resolution_minutes": _median(resolution_times),
        "need_type_counts": {key: need_counts[key] for key in sorted(NEED_TYPES)},
        "urgency_counts": {key: urgency_counts[key] for key in URGENCY_LEVELS},
        "need_type_distribution": _distribution(eligible, "need_type", NEED_TYPES),
        "urgency_distribution": _distribution(eligible, "urgency", set(URGENCY_LEVELS)),
    }


def _published_arm_summary(summary: dict[str, Any], minimum_cell_size: int) -> dict[str, Any]:
    if summary["eligible_cases"] < minimum_cell_size:
        return {
            "suppressed": True,
            "suppression_reason": "eligible arm size is below minimum_cell_size",
        }
    published = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "need_type_counts",
            "urgency_counts",
            "need_type_distribution",
            "urgency_distribution",
        }
    }
    return {"suppressed": False, **published}


def _risk_difference_interval(
    treatment_successes: int,
    treatment_total: int,
    baseline_successes: int,
    baseline_total: int,
    z: float = 1.959963984540054,
) -> list[float]:
    treatment_rate = treatment_successes / treatment_total
    baseline_rate = baseline_successes / baseline_total
    difference = treatment_rate - baseline_rate
    standard_error = math.sqrt(
        treatment_rate * (1 - treatment_rate) / treatment_total
        + baseline_rate * (1 - baseline_rate) / baseline_total
    )
    return [max(-1.0, difference - z * standard_error), min(1.0, difference + z * standard_error)]


def _arm_comparison(
    treatment: dict[str, Any], baseline: dict[str, Any], minimum_cell_size: int
) -> dict[str, Any]:
    if min(treatment["eligible_cases"], baseline["eligible_cases"]) < minimum_cell_size:
        return {
            "suppressed": True,
            "suppression_reason": "one or both arms are below minimum_cell_size",
        }
    treatment_successes = round(
        treatment["conservative_itt_resolution_rate"] * treatment["eligible_cases"]
    )
    baseline_successes = round(
        baseline["conservative_itt_resolution_rate"] * baseline["eligible_cases"]
    )
    risk_difference = (
        treatment["conservative_itt_resolution_rate"]
        - baseline["conservative_itt_resolution_rate"]
    )
    baseline_rate = baseline["conservative_itt_resolution_rate"]
    relative_risk = (
        treatment["conservative_itt_resolution_rate"] / baseline_rate
        if baseline_rate > 0
        else None
    )
    need_balance_safe = all(
        count == 0 or count >= minimum_cell_size
        for summary in (treatment, baseline)
        for count in summary["need_type_counts"].values()
    )
    urgency_balance_safe = all(
        count == 0 or count >= minimum_cell_size
        for summary in (treatment, baseline)
        for count in summary["urgency_counts"].values()
    )
    need_gap = (
        max(
            abs(
                treatment["need_type_distribution"][key]
                - baseline["need_type_distribution"][key]
            )
            for key in sorted(NEED_TYPES)
        )
        if need_balance_safe
        else None
    )
    urgency_gap = (
        max(
            abs(
                treatment["urgency_distribution"][key]
                - baseline["urgency_distribution"][key]
            )
            for key in URGENCY_LEVELS
        )
        if urgency_balance_safe
        else None
    )
    return {
        "suppressed": False,
        "conservative_resolution_risk_difference_pp": risk_difference * 100,
        "risk_difference_95_pp": [
            value * 100
            for value in _risk_difference_interval(
                treatment_successes,
                treatment["eligible_cases"],
                baseline_successes,
                baseline["eligible_cases"],
            )
        ],
        "relative_risk": relative_risk,
        "number_needed_to_treat": 1 / risk_difference if risk_difference > 0 else None,
        "delivery_rate_difference_pp": (
            treatment["delivery_rate"] - baseline["delivery_rate"]
        )
        * 100,
        "mean_governance_cost_difference": (
            treatment["mean_governance_cost"] - baseline["mean_governance_cost"]
        ),
        "maximum_need_type_balance_gap_pp": need_gap * 100 if need_gap is not None else None,
        "maximum_urgency_balance_gap_pp": urgency_gap * 100 if urgency_gap is not None else None,
        "balance_diagnostics_suppressed": {
            "need_type": not need_balance_safe,
            "urgency": not urgency_balance_safe,
        },
    }


def evaluate_pilot(manifest_path: str | Path, events_path: str | Path) -> dict[str, Any]:
    """Validate and evaluate one pilot while returning aggregate-only evidence."""

    manifest = load_pilot_manifest(manifest_path)
    events = load_pilot_events(events_path, manifest)
    by_arm = {
        arm_id: [event for event in events if event.assignment_arm == arm_id]
        for arm_id in manifest.arm_ids
    }
    total_eligible = sum(event.eligible for event in events)
    internal = {
        arm_id: _internal_arm_summary(rows, total_eligible)
        for arm_id, rows in by_arm.items()
    }
    published = {
        arm_id: _published_arm_summary(summary, manifest.minimum_cell_size)
        for arm_id, summary in internal.items()
    }

    baseline = internal["baseline"]
    comparisons = {
        arm_id: _arm_comparison(summary, baseline, manifest.minimum_cell_size)
        for arm_id, summary in internal.items()
        if arm_id != "baseline"
    }

    sample_checks = {
        arm_id: summary["eligible_cases"] >= manifest.minimum_eligible_per_arm
        for arm_id, summary in internal.items()
    }
    completeness_checks = {
        arm_id: summary["outcome_completeness"]
        >= manifest.minimum_outcome_completeness
        for arm_id, summary in internal.items()
    }
    allocation_checks = {
        arm_id: abs(
            internal[arm_id]["allocation_share"] - manifest.allocation[arm_id]
        )
        <= manifest.maximum_allocation_deviation
        for arm_id in manifest.arm_ids
    }
    delivery_checks = {
        arm_id: internal[arm_id]["delivery_rate"]
        >= manifest.minimum_treatment_delivery
        for arm_id in manifest.arm_ids
        if arm_id != "baseline"
    }
    cell_checks = {
        arm_id: summary["eligible_cases"] >= manifest.minimum_cell_size
        for arm_id, summary in internal.items()
    }
    criteria = {
        "preregistration_precedes_pilot": True,
        "minimum_eligible_per_arm": sample_checks,
        "minimum_outcome_completeness": completeness_checks,
        "minimum_treatment_delivery": delivery_checks,
        "allocation_within_declared_deviation": allocation_checks,
        "minimum_publishable_cell_size": cell_checks,
    }
    all_ready = all(
        [
            *sample_checks.values(),
            *completeness_checks.values(),
            *delivery_checks.values(),
            *allocation_checks.values(),
            *cell_checks.values(),
        ]
    )
    unmet = []
    for criterion, checks in criteria.items():
        if isinstance(checks, dict):
            unmet.extend(
                f"{criterion}:{arm_id}" for arm_id, passed in checks.items() if not passed
            )
        elif not checks:
            unmet.append(criterion)

    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot": {
            "pilot_id": manifest.pilot_id,
            "scenario_id": manifest.scenario_id,
            "design": "randomized_parallel",
            "assignment_unit": "demand_case",
            "analysis_population": "eligible_intention_to_treat",
            "primary_outcome": "resolved_within_sla",
            "sla_minutes": manifest.sla_minutes,
            "cost_unit": manifest.cost_unit,
            "arms": list(manifest.arm_ids),
            "preregistered_at": manifest.preregistered_at.isoformat(),
            "pilot_window": {
                "start": manifest.pilot_start.isoformat(),
                "end": manifest.pilot_end.isoformat(),
                "timezone": manifest.timezone_name,
            },
        },
        "input_evidence": {
            "manifest": _file_identity(manifest_path),
            "events": _file_identity(events_path),
            "note": "File fingerprints identify evaluated snapshots; paths and case tokens are not persisted.",
        },
        "data_audit": {
            "accepted_rows": len(events),
            "eligible_rows": total_eligible,
            "observed_primary_outcomes": sum(
                event.eligible and event.outcome_observed for event in events
            ),
            "ineligible_rows": sum(not event.eligible for event in events),
            "duplicate_case_tokens": 0,
            "rejected_rows": 0,
        },
        "readiness": {
            "status": (
                "evidence_candidate_pending_review"
                if all_ready
                else "descriptive_pilot_only"
            ),
            "all_automated_gates_passed": all_ready,
            "criteria": criteria,
            "unmet_criteria": unmet,
            "interpretation": (
                "Automated readiness is necessary but not sufficient for causal validity. "
                "Independent design, implementation, and statistical review is required."
            ),
        },
        "arm_summary": published,
        "comparisons_vs_baseline": comparisons,
        "privacy": {
            "aggregate_only": True,
            "case_tokens_persisted": False,
            "input_paths_persisted": False,
            "minimum_cell_size": manifest.minimum_cell_size,
        },
        "claim_status": {
            "intervention_effect_validity": "candidate_pending_independent_review"
            if all_ready
            else "not_empirically_validated",
            "simulation_effect_parameters_updated": False,
        },
        "limitations": [
            "The preregistration timestamp is self-declared unless the manifest is stored in an immutable external registry.",
            "The minimum sample threshold is a protocol input, not a power calculation performed by this evaluator.",
            "The evaluator cannot verify that random assignment was implemented without manipulation.",
            "Contamination, spillovers, selective logging, and outcome misclassification require external audit.",
            "Confidence intervals are unadjusted descriptive intervals and do not account for multiplicity or clustering.",
            "Operational resolution within SLA is not equivalent to student wellbeing or underlying-problem resolution.",
        ],
    }


def write_pilot_json(result: dict[str, Any], output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return str(target)
