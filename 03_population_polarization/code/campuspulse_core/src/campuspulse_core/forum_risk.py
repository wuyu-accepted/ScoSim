"""Interpretable ForumTwin v2 risk signals and privacy-safe actor views."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .forum_twin import ForumLedger, ForumTwinError
from .mass_population import canonical_hash


RISK_RESULT_SCHEMA_VERSION = "campus-pulse-governance-risk-result-v1"
RISK_THRESHOLD_RELEASE_SCHEMA_VERSION = "campus-pulse-risk-threshold-release-v1"
RISK_SIGNAL_NAMES = (
    "unsupported_claim_velocity",
    "private_to_public_spillover",
    "contested_claim_share",
    "cross_group_exposure_gap",
    "correction_penetration_and_delay",
    "help_backlog_and_sla_breach",
    "governance_observation_gap",
)

_DEFAULT_QUANTILES = {
    "unsupported_claim_velocity": (0.15, 0.35),
    "private_to_public_spillover": (0.08, 0.22),
    "contested_claim_share": (0.25, 0.50),
    "cross_group_exposure_gap": (0.18, 0.35),
    "correction_penetration_and_delay": (0.30, 0.55),
    "help_backlog_and_sla_breach": (0.20, 0.45),
    "governance_observation_gap": (0.15, 0.32),
}

_ACTION_MECHANISMS = {
    "unsupported_claim_velocity": ("verified_update", "evidence_dashboard"),
    "private_to_public_spillover": ("targeted_prebunk", "bridge_outreach"),
    "contested_claim_share": ("evidence_dashboard", "listening_session"),
    "cross_group_exposure_gap": ("bridge_outreach", "peer_feedback"),
    "correction_penetration_and_delay": ("targeted_prebunk", "bridge_outreach"),
    "help_backlog_and_sla_breach": ("queue_status", "support_channel"),
    "governance_observation_gap": ("listening_session", "evidence_dashboard"),
}


@dataclass(frozen=True)
class GovernanceRiskSignal:
    signal_name: str
    tick: int
    value: float
    change_velocity: float
    severity: str
    affected_groups: tuple[str, ...]
    public_evidence_thread_ids: tuple[str, ...]
    private_aggregate_event_count: int
    observable_to_governance: bool
    candidate_action_ids: tuple[str, ...]
    mechanism_details: dict[str, float]

    def __post_init__(self) -> None:
        if self.signal_name not in RISK_SIGNAL_NAMES:
            raise ForumTwinError("governance risk signal name is invalid")
        if not 0 <= self.tick <= 23:
            raise ForumTwinError("governance risk signal tick is invalid")
        if self.severity not in {"low", "watch", "high"}:
            raise ForumTwinError("governance risk severity is invalid")
        for value in (self.value, self.change_velocity, *self.mechanism_details.values()):
            if not math.isfinite(float(value)):
                raise ForumTwinError("governance risk signal is non-finite")
        if self.private_aggregate_event_count < 0:
            raise ForumTwinError("governance private aggregate count is negative")
        if len(set(self.affected_groups)) != len(self.affected_groups):
            raise ForumTwinError("governance risk affected groups repeat")
        if len(set(self.public_evidence_thread_ids)) != len(
            self.public_evidence_thread_ids
        ):
            raise ForumTwinError("governance risk public evidence threads repeat")

    def public_view(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "tick": self.tick,
            "value": round(float(self.value), 6),
            "change_velocity": round(float(self.change_velocity), 6),
            "severity": self.severity,
            "affected_groups": list(self.affected_groups),
            "public_evidence_thread_ids": list(self.public_evidence_thread_ids),
            "private_aggregate_event_count": self.private_aggregate_event_count,
            "observable_to_governance": self.observable_to_governance,
            "candidate_action_ids": list(self.candidate_action_ids),
            "mechanism_details": {
                key: round(float(value), 6)
                for key, value in sorted(self.mechanism_details.items())
            },
        }


def _bounded_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / float(denominator)))


def _severity(
    name: str,
    value: float,
    quantiles: Mapping[str, Sequence[float]] | None,
) -> str:
    raw = (quantiles or {}).get(name, _DEFAULT_QUANTILES[name])
    if not isinstance(raw, Sequence) or len(raw) != 2:
        raise ForumTwinError("risk baseline quantiles are invalid")
    watch, high = float(raw[0]), float(raw[1])
    if not 0 <= watch < high <= 1:
        raise ForumTwinError("risk baseline quantiles must be frozen probabilities")
    return "high" if value >= high else "watch" if value >= watch else "low"


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ForumTwinError("risk threshold release has no observations")
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def build_risk_threshold_release_v1(
    *,
    observations: Sequence[Mapping[str, Any]],
    source_release_hashes: Mapping[str, str],
    watch_quantile: float = 0.75,
    high_quantile: float = 0.95,
    minimum_observations_per_signal: int = 20,
) -> dict[str, Any]:
    """Freeze interpretable thresholds from shared-baseline/history rows.

    Rows carry only signal name, value and the aggregate source kind.  Record
    identities and private messages are neither accepted nor retained.
    Degenerate distributions fail closed instead of inventing separation.
    """

    if (
        not 0 < watch_quantile < high_quantile < 1
        or minimum_observations_per_signal < 20
        or not source_release_hashes
        or any(len(str(value)) != 64 for value in source_release_hashes.values())
    ):
        raise ForumTwinError("risk threshold release configuration is invalid")
    grouped: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    for row in observations:
        name = str(row.get("signal_name") or "")
        source = str(row.get("source_kind") or "")
        value = row.get("value")
        if (
            name not in RISK_SIGNAL_NAMES
            or source not in {"shared_baseline", "frozen_history"}
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ForumTwinError("risk threshold observation is invalid")
        grouped[name].append(float(value))
        sources[name][source] += 1
    if set(grouped) != set(RISK_SIGNAL_NAMES):
        raise ForumTwinError("risk threshold release requires all seven signals")
    thresholds: dict[str, Any] = {}
    for name in RISK_SIGNAL_NAMES:
        values = grouped[name]
        if len(values) < minimum_observations_per_signal:
            raise ForumTwinError("risk threshold release lacks baseline observations")
        watch = _nearest_rank(values, watch_quantile)
        high = _nearest_rank(values, high_quantile)
        if not 0 <= watch < high <= 1:
            raise ForumTwinError("risk threshold distribution cannot separate watch/high")
        thresholds[name] = {
            "watch": watch,
            "high": high,
            "observation_count": len(values),
            "source_counts": dict(sorted(sources[name].items())),
        }
    body = {
        "schema_version": RISK_THRESHOLD_RELEASE_SCHEMA_VERSION,
        "source_release_hashes": dict(sorted(source_release_hashes.items())),
        "threshold_method": "empirical_nearest_rank",
        "watch_quantile": watch_quantile,
        "high_quantile": high_quantile,
        "minimum_observations_per_signal": minimum_observations_per_signal,
        "thresholds": thresholds,
        "contains_private_text": False,
        "contains_record_ids": False,
    }
    return {**body, "threshold_release_sha256": canonical_hash(body)}


def validate_risk_threshold_release_v1(
    release: Mapping[str, Any],
) -> dict[str, Any]:
    if release.get("schema_version") != RISK_THRESHOLD_RELEASE_SCHEMA_VERSION:
        raise ForumTwinError("risk threshold release schema is invalid")
    expected = canonical_hash(
        {
            key: value
            for key, value in release.items()
            if key != "threshold_release_sha256"
        }
    )
    if release.get("threshold_release_sha256") != expected:
        raise ForumTwinError("risk threshold release hash drifted")
    rebuilt_rows = []
    for name, row in (release.get("thresholds") or {}).items():
        if name not in RISK_SIGNAL_NAMES or not isinstance(row, Mapping):
            raise ForumTwinError("risk threshold release signal is invalid")
        watch, high = row.get("watch"), row.get("high")
        count = row.get("observation_count")
        if (
            not isinstance(watch, (int, float))
            or not isinstance(high, (int, float))
            or not isinstance(count, int)
            or not 0 <= float(watch) < float(high) <= 1
            or count < int(release.get("minimum_observations_per_signal", 20))
        ):
            raise ForumTwinError("risk threshold release values are invalid")
        rebuilt_rows.append(name)
    if set(rebuilt_rows) != set(RISK_SIGNAL_NAMES):
        raise ForumTwinError("risk threshold release is incomplete")
    if (
        release.get("contains_private_text") is not False
        or release.get("contains_record_ids") is not False
    ):
        raise ForumTwinError("risk threshold release crossed its privacy boundary")
    return dict(release)


def _affected_groups(
    *,
    ledger: ForumLedger,
    tick: int,
    macro_by_display_id: Mapping[str, str],
) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for message in ledger.messages.values():
        if message.created_tick > tick or message.superseded:
            continue
        group = str(macro_by_display_id.get(message.source_agent_display_id, "unmapped"))
        counts[group] = counts.get(group, 0) + 1
    return tuple(
        key
        for key, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:6]
    )


def build_governance_risk_result_v1(
    *,
    ledger: ForumLedger,
    scenario_id: str,
    branch: str,
    tick: int,
    macro_by_display_id: Mapping[str, str],
    cross_group_exposure_rates: Mapping[str, float],
    correction_private_reach_rates: Mapping[str, float],
    help_queue: Mapping[str, float],
    population_public_state: Mapping[str, float],
    governance_visible_state: Mapping[str, float],
    previous_signal_values: Mapping[str, float] | None = None,
    baseline_quantiles: Mapping[str, Sequence[float]] | None = None,
    baseline_threshold_release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build seven separate signals; never collapse them to one total score."""

    if not scenario_id or not branch or not 0 <= tick <= 23:
        raise ForumTwinError("risk result identity is invalid")
    if baseline_quantiles is not None and baseline_threshold_release is not None:
        raise ForumTwinError("risk result cannot bind two threshold sources")
    threshold_source = "development_default_thresholds_not_publication_eligible"
    if baseline_threshold_release is not None:
        checked_release = validate_risk_threshold_release_v1(
            baseline_threshold_release
        )
        baseline_quantiles = {
            name: (float(row["watch"]), float(row["high"]))
            for name, row in checked_release["thresholds"].items()
        }
        threshold_source = str(checked_release["threshold_release_sha256"])
    elif baseline_quantiles is not None:
        threshold_source = "caller_frozen_baseline_quantiles"
    ledger.assert_consistent()
    public_messages = [
        message
        for message in ledger.messages.values()
        if message.created_tick <= tick and not message.superseded
    ]
    current_messages = [
        message for message in public_messages if message.created_tick == tick
    ]
    current_claims = [
        claim for claim in ledger.claims.values() if claim.first_seen_tick == tick
    ]
    unsupported_current = sum(
        claim.status == "unverified" for claim in current_claims
    )
    unsupported_velocity = _bounded_ratio(
        unsupported_current,
        max(1, len(current_messages)),
    )
    current_spillovers = sum(
        message.private_hearsay_origin for message in current_messages
    )
    spillover = _bounded_ratio(current_spillovers, max(1, len(current_messages)))
    active_claims = [
        claim
        for claim in ledger.claims.values()
        if claim.first_seen_tick <= tick and claim.status != "expired"
    ]
    contested = _bounded_ratio(
        sum(claim.status == "contested" for claim in active_claims),
        max(1, len(active_claims)),
    )

    exposure_values = [float(value) for value in cross_group_exposure_rates.values()]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in exposure_values):
        raise ForumTwinError("cross-group exposure rates are invalid")
    exposure_gap = (
        max(exposure_values) - min(exposure_values)
        if exposure_values
        else 0.0
    )
    correction_values = [
        float(value) for value in correction_private_reach_rates.values()
    ]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in correction_values):
        raise ForumTwinError("correction private reach rates are invalid")
    correction_penetration = (
        sum(correction_values) / len(correction_values)
        if correction_values
        else 0.0
    )
    correction_delays: list[int] = []
    for message in public_messages:
        if message.action != "correct" or message.correction_target_claim_id is None:
            continue
        target = ledger.claims.get(message.correction_target_claim_id)
        if target is not None:
            correction_delays.append(message.created_tick - target.first_seen_tick)
    mean_delay = (
        sum(correction_delays) / len(correction_delays)
        if correction_delays
        else float(tick + 1 if active_claims else 0)
    )
    correction_risk = min(
        1.0,
        0.70 * (1.0 - correction_penetration) + 0.30 * min(1.0, mean_delay / 6.0),
    )

    backlog = float(help_queue.get("backlog", 0.0))
    capacity = float(help_queue.get("capacity", 0.0))
    sla_breaches = float(help_queue.get("sla_breaches", 0.0))
    help_requests = float(help_queue.get("help_requests", 0.0))
    if any(
        not math.isfinite(value) or value < 0
        for value in (backlog, capacity, sla_breaches, help_requests)
    ):
        raise ForumTwinError("help queue risk values are invalid")
    backlog_ratio = _bounded_ratio(backlog, max(1.0, capacity + backlog))
    breach_ratio = _bounded_ratio(sla_breaches, max(1.0, help_requests))
    help_risk = min(1.0, 0.6 * backlog_ratio + 0.4 * breach_ratio)

    common_state_keys = sorted(
        set(population_public_state) & set(governance_visible_state)
    )
    if not common_state_keys:
        observation_gap = 0.0
    else:
        gaps = []
        for key in common_state_keys:
            population_value = float(population_public_state[key])
            visible_value = float(governance_visible_state[key])
            if not math.isfinite(population_value) or not math.isfinite(visible_value):
                raise ForumTwinError("governance observation gap contains non-finite values")
            gaps.append(min(1.0, abs(population_value - visible_value)))
        observation_gap = sum(gaps) / len(gaps)

    private_message_count = sum(
        message.created_tick <= tick and not message.superseded
        for message in ledger.private_messages.values()
    )
    evidence_threads = tuple(
        dict.fromkeys(
            message.thread_id
            for message in sorted(
                current_messages,
                key=lambda item: (item.created_tick, item.message_id),
            )
        )
    )[:12]
    affected = _affected_groups(
        ledger=ledger,
        tick=tick,
        macro_by_display_id=macro_by_display_id,
    )
    values_and_details = {
        "unsupported_claim_velocity": (
            unsupported_velocity,
            {"new_unsupported_claims": float(unsupported_current), "public_messages": float(len(current_messages))},
            0,
            True,
        ),
        "private_to_public_spillover": (
            spillover,
            {"spillover_messages": float(current_spillovers), "public_messages": float(len(current_messages))},
            private_message_count,
            True,
        ),
        "contested_claim_share": (
            contested,
            {"active_claims": float(len(active_claims)), "contested_claims": float(sum(claim.status == "contested" for claim in active_claims))},
            0,
            True,
        ),
        "cross_group_exposure_gap": (
            exposure_gap,
            {"maximum_rate": max(exposure_values, default=0.0), "minimum_rate": min(exposure_values, default=0.0)},
            0,
            True,
        ),
        "correction_penetration_and_delay": (
            correction_risk,
            {"penetration": correction_penetration, "mean_delay_ticks": mean_delay},
            private_message_count,
            True,
        ),
        "help_backlog_and_sla_breach": (
            help_risk,
            {"backlog_ratio": backlog_ratio, "sla_breach_ratio": breach_ratio},
            0,
            True,
        ),
        "governance_observation_gap": (
            observation_gap,
            {"compared_aggregate_fields": float(len(common_state_keys))},
            0,
            False,
        ),
    }
    signals = []
    previous = previous_signal_values or {}
    for name in RISK_SIGNAL_NAMES:
        value, details, private_count, observable = values_and_details[name]
        signal = GovernanceRiskSignal(
            signal_name=name,
            tick=tick,
            value=float(value),
            change_velocity=float(value) - float(previous.get(name, value)),
            severity=_severity(name, float(value), baseline_quantiles),
            affected_groups=affected,
            public_evidence_thread_ids=evidence_threads,
            private_aggregate_event_count=int(private_count),
            observable_to_governance=bool(observable),
            candidate_action_ids=_ACTION_MECHANISMS[name],
            mechanism_details=dict(details),
        )
        signals.append(signal.public_view())
    body = {
        "schema_version": RISK_RESULT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "branch": branch,
        "tick": tick,
        "signals": signals,
        "combined_score": None,
        "threshold_source": threshold_source,
        "privacy": {
            "contains_private_text": False,
            "contains_private_agent_ids": False,
            "contains_relationship_edges": False,
        },
    }
    return {**body, "risk_result_sha256": canonical_hash(body)}


def build_privacy_safe_actor_observations(
    risk_result: Mapping[str, Any],
    *,
    discussion_metrics: Mapping[str, float],
    service_metrics: Mapping[str, float],
    bridge_metrics: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Translate risk aggregates into role-prefixed observations, never raw chats."""

    if risk_result.get("schema_version") != RISK_RESULT_SCHEMA_VERSION:
        raise ForumTwinError("actor observations require a v1 risk result")
    if risk_result.get("privacy", {}).get("contains_private_text") is not False:
        raise ForumTwinError("governance actors cannot receive private text")
    signals = {
        str(row["signal_name"]): float(row["value"])
        for row in risk_result.get("signals", ())
    }
    if set(signals) != set(RISK_SIGNAL_NAMES):
        raise ForumTwinError("actor observations require all seven risk signals")

    def clean(prefix: str, values: Mapping[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, value in values.items():
            if not str(key).startswith(prefix):
                raise ForumTwinError("actor observation key is outside its privacy role")
            number = float(value)
            if not math.isfinite(number):
                raise ForumTwinError("actor observation value is non-finite")
            result[str(key)] = round(number, 6)
        return result

    authority = clean("discussion_", discussion_metrics)
    authority.update(
        {
            "risk_unsupported_claim_velocity": signals["unsupported_claim_velocity"],
            "risk_private_spillover_aggregate": signals["private_to_public_spillover"],
            "claim_contested_share": signals["contested_claim_share"],
            "uncertainty_observation_gap": signals["governance_observation_gap"],
        }
    )
    service = clean("service_", service_metrics)
    service["help_backlog_sla_risk"] = signals["help_backlog_and_sla_breach"]
    bridge = clean("local_", bridge_metrics)
    bridge.update(
        {
            "cross_group_exposure_gap": signals["cross_group_exposure_gap"],
            "correction_penetration_delay_risk": signals[
                "correction_penetration_and_delay"
            ],
            "participation_private_spillover_aggregate": signals[
                "private_to_public_spillover"
            ],
        }
    )
    return {
        "governance_authority": authority,
        "service_operator": service,
        "community_bridge": bridge,
    }


__all__ = [
    "RISK_RESULT_SCHEMA_VERSION",
    "RISK_THRESHOLD_RELEASE_SCHEMA_VERSION",
    "RISK_SIGNAL_NAMES",
    "GovernanceRiskSignal",
    "build_governance_risk_result_v1",
    "build_risk_threshold_release_v1",
    "build_privacy_safe_actor_observations",
    "validate_risk_threshold_release_v1",
]
