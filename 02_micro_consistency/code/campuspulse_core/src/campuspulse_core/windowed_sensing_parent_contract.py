"""Strict admission contract for the frozen windowed-sensing v1.0 parents.

The semantic and context artifacts covered here predate the v1.1 checkpoint
contract.  They are valid historical parents, but a schema string alone is not
enough to distinguish them from a newer artifact whose version was merely
changed to ``1.0.0``.  This module therefore admits exactly the two frozen
files, reconstructs both historical run fingerprints, and independently
checks the most important privacy, lineage, execution, and record-conservation
invariants.

The public compiler and private review finalizer may share this module.  New
v1.1 artifacts must be validated by their native v1.1 contract instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


FROZEN_LEGACY_SCHEMA_VERSION = "1.0.0"
FROZEN_SEMANTIC_CARD_SCHEMA_VERSION = "1.5.1"
FROZEN_SEMANTIC_PROMPT_VERSION = "1.5.1"
FROZEN_CONTEXT_PROMPT_VERSION = "1.3.1"
FROZEN_MAX_OUTPUT_TOKENS = 8_000
FROZEN_ESTIMATION_RECORD_COUNT = 1_200
FROZEN_CONTEXT_RECORD_COUNT = 75

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

WINDOW_ROLES = {
    "matched_nonshock_reference": "normal_reference",
    "tongzhou_governance_shock": "governance_shock",
    "lecture_external_shock": "exogenous_shock",
}
WINDOW_SELECTED_COUNTS = {
    "matched_nonshock_reference": 400,
    "tongzhou_governance_shock": 400,
    "lecture_external_shock": 400,
}
TEMPORAL_BLOCKS = {"baseline", "burst", "spread", "decay"}
FOCAL_PROXY_STATUSES = {
    "focal_lexical_proxy",
    "other",
    "collapsed_mixed",
}
HEURISTIC_DIFFICULTIES = {"likely_clear", "likely_difficult"}
REVISION_STATUSES = {"unchanged", "clarified", "changed", "unresolved"}

SEMANTIC_CARD_FIELDS = {
    "semantic_card_schema_version",
    "record_id",
    "input_fingerprint",
    "language_form",
    "semantic_determinacy",
    "literal_topic",
    "normalized_meaning",
    "implicit_need",
    "secondary_needs",
    "situational_pressures",
    "affected_object",
    "desired_response",
    "interpretation_basis",
    "speaker_relation",
    "ambiguity_scope",
    "cluster_assignment_risk",
    "speech_act",
    "stance",
    "risk_level",
    "local_terms",
    "alternative_interpretations",
    "model_context_needed",
    "context_needed",
    "confidence",
}
DESIGN_METADATA_FIELDS = {
    "window_key",
    "window_role",
    "segment_id",
    "temporal_block",
    "focal_proxy_status",
    "sampling_stratum_id",
    "eligible_stratum_count",
    "selected_stratum_count",
    "inclusion_probability_ppm",
    "design_weight_numerator",
    "design_weight_denominator",
    "estimation_eligible",
}
SEMANTIC_RECORD_FIELDS = SEMANTIC_CARD_FIELDS | DESIGN_METADATA_FIELDS
CONTEXT_RECORD_FIELDS = {
    "record_id",
    "comments_used",
    "revision_status",
    "context_summary",
    "before_card",
    "after_card",
}


class WindowedSensingParentContractError(ValueError):
    """Raised when a frozen legacy parent fails its admission contract."""


@dataclass(frozen=True)
class FrozenArtifactProfile:
    """Immutable byte and run identity for one historical parent."""

    sha256: str
    bytes: int
    run_fingerprint: str


@dataclass(frozen=True)
class FrozenLegacyProfile:
    """The only legacy semantic/context pair admitted into the release chain."""

    semantic: FrozenArtifactProfile
    context: FrozenArtifactProfile


@dataclass(frozen=True)
class FrozenLegacyParentValidation:
    """Safe, aggregate-only receipt returned after successful validation."""

    schema_pair: tuple[str, str]
    semantic_sha256: str
    context_sha256: str
    semantic_run_fingerprint: str
    context_run_fingerprint: str
    semantic_record_count: int
    context_record_count: int


FROZEN_LEGACY_PROFILE = FrozenLegacyProfile(
    semantic=FrozenArtifactProfile(
        sha256=(
            "9aad8186431026a3bdb9ec94646a230a"
            "9afaf6fa97b00a4165c9424163d1ac2f"
        ),
        bytes=2_278_601,
        run_fingerprint=(
            "80c87a955bfa2323ee97d072baf2b6e"
            "6f2b5399edea40b5ec1b4dbccd6838893"
        ),
    ),
    context=FrozenArtifactProfile(
        sha256=(
            "91633e17dcc7755161d3964c05a6780e"
            "adf26bf91c3f909920954570d05eca3d"
        ),
        bytes=290_247,
        run_fingerprint=(
            "973f7751eae982229e80c06d3786c3f"
            "d5862d1ff984063036c532228248319b1"
        ),
    ),
)


SEMANTIC_TOP_LEVEL_FIELDS = {
    "windowed_semantic_schema_version",
    "generated_at_utc",
    "status",
    "run_fingerprint",
    "input_evidence",
    "sampling",
    "provider",
    "execution",
    "records",
    "privacy",
    "validity",
    "summary",
}
CONTEXT_TOP_LEVEL_FIELDS = {
    "windowed_context_schema_version",
    "generated_at_utc",
    "status",
    "run_fingerprint",
    "input_evidence",
    "context_selection",
    "provider",
    "execution",
    "records",
    "privacy",
    "validity",
    "summary",
}
SEMANTIC_PROVIDER_FIELDS = {
    "api_key_environment_variable",
    "api_key_persisted",
    "base_url",
    "contact_redacted_forum_text_sent",
    "endpoint",
    "reasoning_effort",
    "requested_model",
    "source_identifiers_sent",
    "window_labels_sent",
}
CONTEXT_PROVIDER_FIELDS = {
    "api_key_environment_variable",
    "api_key_persisted",
    "base_url",
    "contact_redacted_posts_and_comments_sent",
    "endpoint",
    "reasoning_effort",
    "requested_model",
    "source_identifiers_sent",
    "window_labels_sent",
}
EXECUTION_FIELDS = {
    "batch_size",
    "max_concurrency",
    "max_total_tokens",
    "completed_batches",
    "usage",
    "token_budget",
}
BATCH_FIELDS = {
    "batch_id",
    "record_count",
    "returned_model",
    "usage",
    "provider_attempts",
    "format_repairs",
}
USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "provider_attempts",
    "format_repairs",
}
BATCH_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}
TOKEN_BUDGET_FIELDS = {
    "max_total_tokens",
    "provider_reported_or_reserved_tokens",
    "provider_calls",
    "inflight_reserved_tokens",
}


def _fail(code: str) -> None:
    raise WindowedSensingParentContractError(code)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label}_must_be_object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label}_must_be_list")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        _fail(f"{label}_keys_mismatch")


def _require_bool(value: Any, *, expected: bool, label: str) -> None:
    if not isinstance(value, bool) or value is not expected:
        _fail(f"{label}_must_be_{str(expected).lower()}")


def _require_int(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    exact: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label}_must_be_integer")
    if value < minimum:
        _fail(f"{label}_below_minimum")
    if exact is not None and value != exact:
        _fail(f"{label}_must_equal_{exact}")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}_must_be_number")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label}_must_be_finite")
    return number


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label}_must_be_sha256")
    return value


def _require_nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label}_must_be_nonempty_text")
    return value


def _canonical_hash(value: Any) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        _fail("parent_contains_noncanonical_value")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _serialized_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the exact writer used for the frozen checkpoint files."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("parent_contains_nonserializable_value")
    return {
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "bytes": len(rendered),
    }


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return the byte identity expected by the frozen-parent validator."""

    target = Path(path)
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _validate_artifact_identity(
    value: Mapping[str, Any],
    actual_identity: Mapping[str, Any],
    profile: FrozenArtifactProfile,
    *,
    label: str,
) -> None:
    identity = _require_mapping(actual_identity, label=f"{label}_file_identity")
    actual_sha = _require_sha256(
        identity.get("sha256"), label=f"{label}_file_identity_sha256"
    )
    actual_bytes = _require_int(
        identity.get("bytes"),
        label=f"{label}_file_identity_bytes",
        minimum=1,
    )
    if actual_sha != profile.sha256 or actual_bytes != profile.bytes:
        _fail(f"{label}_file_identity_not_frozen")
    reconstructed = _serialized_identity(value)
    if reconstructed != {"sha256": profile.sha256, "bytes": profile.bytes}:
        _fail(f"{label}_document_identity_not_frozen")


def _validate_generated_at(value: Any, *, label: str) -> None:
    text = _require_nonempty_text(value, label=label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{label}_must_be_iso8601")
    if parsed.tzinfo is None:
        _fail(f"{label}_must_include_timezone")


def _validate_file_identity_object(value: Any, *, label: str) -> Mapping[str, Any]:
    identity = _require_mapping(value, label=label)
    _require_exact_keys(identity, {"sha256", "bytes"}, label=label)
    _require_sha256(identity.get("sha256"), label=f"{label}_sha256")
    _require_int(identity.get("bytes"), label=f"{label}_bytes", minimum=1)
    return identity


def _validate_provider(
    value: Any,
    *,
    context: bool,
) -> Mapping[str, Any]:
    label = "context_provider" if context else "semantic_provider"
    provider = _require_mapping(value, label=label)
    _require_exact_keys(
        provider,
        CONTEXT_PROVIDER_FIELDS if context else SEMANTIC_PROVIDER_FIELDS,
        label=label,
    )
    for field in (
        "api_key_environment_variable",
        "base_url",
        "endpoint",
        "reasoning_effort",
        "requested_model",
    ):
        _require_nonempty_text(provider.get(field), label=f"{label}_{field}")
    _require_bool(
        provider.get("api_key_persisted"),
        expected=False,
        label=f"{label}_api_key_persisted",
    )
    _require_bool(
        provider.get("source_identifiers_sent"),
        expected=False,
        label=f"{label}_source_identifiers_sent",
    )
    _require_bool(
        provider.get("window_labels_sent"),
        expected=False,
        label=f"{label}_window_labels_sent",
    )
    redacted_field = (
        "contact_redacted_posts_and_comments_sent"
        if context
        else "contact_redacted_forum_text_sent"
    )
    _require_bool(
        provider.get(redacted_field),
        expected=True,
        label=f"{label}_{redacted_field}",
    )
    if provider.get("endpoint") != "/chat/completions":
        _fail(f"{label}_endpoint_mismatch")
    return provider


def _validate_execution(
    value: Any,
    *,
    stage: str,
    expected_record_count: int,
    expected_batched_record_count: int,
    expected_batch_count: int,
    batch_prefix: str,
) -> Mapping[str, Any]:
    execution = _require_mapping(value, label=f"{stage}_execution")
    _require_exact_keys(execution, EXECUTION_FIELDS, label=f"{stage}_execution")
    batch_size = _require_int(
        execution.get("batch_size"),
        label=f"{stage}_execution_batch_size",
        minimum=1,
    )
    max_total_tokens = _require_int(
        execution.get("max_total_tokens"),
        label=f"{stage}_execution_max_total_tokens",
        minimum=1,
    )
    _require_int(
        execution.get("max_concurrency"),
        label=f"{stage}_execution_max_concurrency",
        minimum=1,
    )
    batches = _require_list(
        execution.get("completed_batches"),
        label=f"{stage}_execution_completed_batches",
    )
    if len(batches) != expected_batch_count:
        _fail(f"{stage}_execution_completed_batch_count_mismatch")
    aggregate_usage = Counter(
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "provider_attempts": 0,
            "format_repairs": 0,
        }
    )
    seen_batch_ids: set[str] = set()
    batched_record_count = 0
    for index, raw_batch in enumerate(batches, start=1):
        batch = _require_mapping(
            raw_batch, label=f"{stage}_execution_batch_{index}"
        )
        _require_exact_keys(
            batch, BATCH_FIELDS, label=f"{stage}_execution_batch_{index}"
        )
        batch_id = _require_nonempty_text(
            batch.get("batch_id"),
            label=f"{stage}_execution_batch_{index}_id",
        )
        expected_id = f"{batch_prefix}{index:05d}"
        if batch_id != expected_id or batch_id in seen_batch_ids:
            _fail(f"{stage}_execution_batch_manifest_mismatch")
        seen_batch_ids.add(batch_id)
        record_count = _require_int(
            batch.get("record_count"),
            label=f"{stage}_execution_batch_{index}_record_count",
            minimum=1,
        )
        if record_count > batch_size:
            _fail(f"{stage}_execution_batch_exceeds_batch_size")
        batched_record_count += record_count
        _require_nonempty_text(
            batch.get("returned_model"),
            label=f"{stage}_execution_batch_{index}_returned_model",
        )
        usage = _require_mapping(
            batch.get("usage"),
            label=f"{stage}_execution_batch_{index}_usage",
        )
        _require_exact_keys(
            usage,
            BATCH_USAGE_FIELDS,
            label=f"{stage}_execution_batch_{index}_usage",
        )
        for field in BATCH_USAGE_FIELDS:
            aggregate_usage[field] += _require_int(
                usage.get(field),
                label=f"{stage}_execution_batch_{index}_usage_{field}",
            )
        if usage.get("total_tokens") != (
            usage.get("prompt_tokens", 0)
            + usage.get("completion_tokens", 0)
        ):
            _fail(f"{stage}_execution_batch_token_conservation_failed")
        aggregate_usage["provider_attempts"] += _require_int(
            batch.get("provider_attempts"),
            label=f"{stage}_execution_batch_{index}_provider_attempts",
            minimum=1,
        )
        aggregate_usage["format_repairs"] += _require_int(
            batch.get("format_repairs"),
            label=f"{stage}_execution_batch_{index}_format_repairs",
        )
    if batched_record_count != expected_batched_record_count:
        _fail(f"{stage}_execution_batched_record_conservation_failed")
    if expected_batched_record_count > expected_record_count:
        _fail(f"{stage}_execution_batched_records_exceed_stage_records")
    usage = _require_mapping(
        execution.get("usage"), label=f"{stage}_execution_usage"
    )
    _require_exact_keys(usage, USAGE_FIELDS, label=f"{stage}_execution_usage")
    normalized_usage = {}
    for field in USAGE_FIELDS:
        normalized_usage[field] = _require_int(
            usage.get(field), label=f"{stage}_execution_usage_{field}"
        )
    if normalized_usage != dict(aggregate_usage):
        _fail(f"{stage}_execution_usage_conservation_failed")
    if normalized_usage["total_tokens"] > max_total_tokens:
        _fail(f"{stage}_execution_usage_exceeds_budget")
    token_budget = _require_mapping(
        execution.get("token_budget"),
        label=f"{stage}_execution_token_budget",
    )
    _require_exact_keys(
        token_budget,
        TOKEN_BUDGET_FIELDS,
        label=f"{stage}_execution_token_budget",
    )
    budget_max = _require_int(
        token_budget.get("max_total_tokens"),
        label=f"{stage}_execution_token_budget_max",
        minimum=1,
    )
    if budget_max != max_total_tokens:
        _fail(f"{stage}_execution_token_budget_max_mismatch")
    reported_or_reserved = _require_int(
        token_budget.get("provider_reported_or_reserved_tokens"),
        label=f"{stage}_execution_reported_or_reserved_tokens",
    )
    if (
        reported_or_reserved < normalized_usage["total_tokens"]
        or reported_or_reserved > max_total_tokens
    ):
        _fail(f"{stage}_execution_token_budget_conservation_failed")
    provider_calls = _require_int(
        token_budget.get("provider_calls"),
        label=f"{stage}_execution_provider_calls",
    )
    if provider_calls < normalized_usage["provider_attempts"]:
        _fail(f"{stage}_execution_provider_call_conservation_failed")
    _require_int(
        token_budget.get("inflight_reserved_tokens"),
        label=f"{stage}_execution_inflight_reserved_tokens",
        exact=0,
    )
    return execution


def _validate_sampling(value: Any) -> tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[tuple[str, str], str],
]:
    sampling = _require_mapping(value, label="semantic_sampling")
    expected_keys = {
        "windowed_sampling_schema_version",
        "sampling_version",
        "seed",
        "sample_per_window",
        "sampling_unit",
        "stratification",
        "windows",
        "strata",
        "excluded_known_gap_posts",
        "sample_manifest_fingerprint",
        "privacy",
        "claim_boundary",
    }
    _require_exact_keys(sampling, expected_keys, label="semantic_sampling")
    if (
        sampling.get("windowed_sampling_schema_version") != "1.0.0"
        or sampling.get("sampling_version") != "1.0.0"
    ):
        _fail("semantic_sampling_schema_mismatch")
    _require_int(
        sampling.get("sample_per_window"),
        label="semantic_sampling_sample_per_window",
        exact=400,
    )
    _require_int(sampling.get("seed"), label="semantic_sampling_seed")
    if sampling.get("sampling_unit") != "forum_post_episode":
        _fail("semantic_sampling_unit_mismatch")
    _require_sha256(
        sampling.get("sample_manifest_fingerprint"),
        label="semantic_sampling_manifest_fingerprint",
    )
    privacy = _require_mapping(
        sampling.get("privacy"), label="semantic_sampling_privacy"
    )
    _require_exact_keys(
        privacy,
        {
            "raw_text_persisted",
            "source_identifiers_persisted",
            "contact_redaction_version",
        },
        label="semantic_sampling_privacy",
    )
    _require_bool(
        privacy.get("raw_text_persisted"),
        expected=False,
        label="semantic_sampling_raw_text_persisted",
    )
    _require_bool(
        privacy.get("source_identifiers_persisted"),
        expected=False,
        label="semantic_sampling_source_identifiers_persisted",
    )
    _require_nonempty_text(
        privacy.get("contact_redaction_version"),
        label="semantic_sampling_contact_redaction_version",
    )
    boundary = _require_mapping(
        sampling.get("claim_boundary"),
        label="semantic_sampling_claim_boundary",
    )
    _require_exact_keys(
        boundary,
        {
            "probability_population",
            "unique_people_or_students",
            "causal_effect_identified",
            "lexical_proxy_is_semantic_label",
        },
        label="semantic_sampling_claim_boundary",
    )
    _require_nonempty_text(
        boundary.get("probability_population"),
        label="semantic_sampling_probability_population",
    )
    for field in (
        "unique_people_or_students",
        "causal_effect_identified",
        "lexical_proxy_is_semantic_label",
    ):
        _require_bool(
            boundary.get(field),
            expected=False,
            label=f"semantic_sampling_{field}",
        )
    windows = _require_list(
        sampling.get("windows"), label="semantic_sampling_windows"
    )
    if len(windows) != len(WINDOW_ROLES):
        _fail("semantic_sampling_window_count_mismatch")
    segment_lookup: dict[tuple[str, str], str] = {}
    windows_by_key: dict[str, Mapping[str, Any]] = {}
    for raw_window in windows:
        window = _require_mapping(
            raw_window, label="semantic_sampling_window"
        )
        _require_exact_keys(
            window,
            {
                "window_key",
                "role",
                "eligible_count",
                "selected_count",
                "segment_standardization",
                "segments",
            },
            label="semantic_sampling_window",
        )
        window_key = _require_nonempty_text(
            window.get("window_key"),
            label="semantic_sampling_window_key",
        )
        if window_key in windows_by_key or window_key not in WINDOW_ROLES:
            _fail("semantic_sampling_window_key_mismatch")
        if window.get("role") != WINDOW_ROLES[window_key]:
            _fail("semantic_sampling_window_role_mismatch")
        selected_count = _require_int(
            window.get("selected_count"),
            label=f"semantic_sampling_{window_key}_selected_count",
            exact=WINDOW_SELECTED_COUNTS[window_key],
        )
        eligible_count = _require_int(
            window.get("eligible_count"),
            label=f"semantic_sampling_{window_key}_eligible_count",
            minimum=selected_count,
        )
        if eligible_count < selected_count:
            _fail("semantic_sampling_window_eligible_below_selected")
        segments = _require_list(
            window.get("segments"),
            label=f"semantic_sampling_{window_key}_segments",
        )
        segment_selected = 0
        segment_eligible = 0
        for raw_segment in segments:
            segment = _require_mapping(
                raw_segment,
                label=f"semantic_sampling_{window_key}_segment",
            )
            _require_exact_keys(
                segment,
                {
                    "segment_id",
                    "matched_scenario_id",
                    "start_inclusive",
                    "end_exclusive",
                    "eligible_count",
                    "selected_count",
                    "comment_collection_regime",
                    "anchor_resolution",
                },
                label=f"semantic_sampling_{window_key}_segment",
            )
            segment_id = _require_nonempty_text(
                segment.get("segment_id"),
                label=f"semantic_sampling_{window_key}_segment_id",
            )
            segment_key = (window_key, segment_id)
            if segment_key in segment_lookup:
                _fail("semantic_sampling_segment_id_duplicate")
            segment_lookup[segment_key] = window["role"]
            selected = _require_int(
                segment.get("selected_count"),
                label=f"semantic_sampling_{segment_id}_selected_count",
                minimum=1,
            )
            eligible = _require_int(
                segment.get("eligible_count"),
                label=f"semantic_sampling_{segment_id}_eligible_count",
                minimum=selected,
            )
            segment_selected += selected
            segment_eligible += eligible
            for field in (
                "start_inclusive",
                "end_exclusive",
                "comment_collection_regime",
                "anchor_resolution",
            ):
                _require_nonempty_text(
                    segment.get(field),
                    label=f"semantic_sampling_{segment_id}_{field}",
                )
        if (
            segment_selected != selected_count
            or segment_eligible != eligible_count
        ):
            _fail("semantic_sampling_segment_conservation_failed")
        windows_by_key[window_key] = window
    if set(windows_by_key) != set(WINDOW_ROLES):
        _fail("semantic_sampling_window_set_mismatch")
    strata = _require_list(
        sampling.get("strata"), label="semantic_sampling_strata"
    )
    strata_by_id: dict[str, Mapping[str, Any]] = {}
    selected_by_window: Counter[str] = Counter()
    eligible_by_window: Counter[str] = Counter()
    for raw_stratum in strata:
        stratum = _require_mapping(
            raw_stratum, label="semantic_sampling_stratum"
        )
        _require_exact_keys(
            stratum,
            {
                "stratum_id",
                "window_key",
                "segment_id",
                "temporal_block",
                "heuristic_difficulty",
                "proxy_status",
                "eligible_count",
                "selected_count",
                "inclusion_probability_ppm",
                "weight_numerator",
                "weight_denominator",
            },
            label="semantic_sampling_stratum",
        )
        stratum_id = _require_nonempty_text(
            stratum.get("stratum_id"),
            label="semantic_sampling_stratum_id",
        )
        if stratum_id in strata_by_id:
            _fail("semantic_sampling_stratum_id_duplicate")
        window_key = stratum.get("window_key")
        segment_id = stratum.get("segment_id")
        if (
            window_key not in WINDOW_ROLES
            or (window_key, segment_id) not in segment_lookup
        ):
            _fail("semantic_sampling_stratum_lineage_mismatch")
        if stratum.get("temporal_block") not in TEMPORAL_BLOCKS:
            _fail("semantic_sampling_stratum_temporal_block_invalid")
        if stratum.get("heuristic_difficulty") not in HEURISTIC_DIFFICULTIES:
            _fail("semantic_sampling_stratum_difficulty_invalid")
        if stratum.get("proxy_status") not in FOCAL_PROXY_STATUSES:
            _fail("semantic_sampling_stratum_proxy_status_invalid")
        selected = _require_int(
            stratum.get("selected_count"),
            label=f"semantic_sampling_{stratum_id}_selected_count",
            minimum=1,
        )
        eligible = _require_int(
            stratum.get("eligible_count"),
            label=f"semantic_sampling_{stratum_id}_eligible_count",
            minimum=selected,
        )
        expected_ppm = round(selected * 1_000_000 / eligible)
        if (
            stratum.get("inclusion_probability_ppm") != expected_ppm
            or stratum.get("weight_numerator") != eligible
            or stratum.get("weight_denominator") != selected
        ):
            _fail("semantic_sampling_stratum_weight_contract_mismatch")
        selected_by_window[window_key] += selected
        eligible_by_window[window_key] += eligible
        strata_by_id[stratum_id] = stratum
    for window_key, window in windows_by_key.items():
        if (
            selected_by_window[window_key] != window["selected_count"]
            or eligible_by_window[window_key] != window["eligible_count"]
        ):
            _fail("semantic_sampling_stratum_conservation_failed")
    return strata_by_id, segment_lookup


def _validate_semantic_records(
    value: Any,
    *,
    strata_by_id: Mapping[str, Mapping[str, Any]],
    segment_lookup: Mapping[tuple[str, str], str],
) -> tuple[dict[str, Mapping[str, Any]], Counter[str]]:
    records = _require_list(value, label="semantic_records")
    if len(records) != FROZEN_ESTIMATION_RECORD_COUNT:
        _fail("semantic_record_count_mismatch")
    records_by_id: dict[str, Mapping[str, Any]] = {}
    records_by_window: Counter[str] = Counter()
    records_by_stratum: Counter[str] = Counter()
    for raw_record in records:
        record = _require_mapping(raw_record, label="semantic_record")
        _require_exact_keys(
            record, SEMANTIC_RECORD_FIELDS, label="semantic_record"
        )
        record_id = _require_nonempty_text(
            record.get("record_id"), label="semantic_record_id"
        )
        if record_id in records_by_id:
            _fail("semantic_record_id_duplicate")
        _require_sha256(
            record.get("input_fingerprint"),
            label="semantic_record_input_fingerprint",
        )
        if (
            record.get("semantic_card_schema_version")
            != FROZEN_SEMANTIC_CARD_SCHEMA_VERSION
        ):
            _fail("semantic_record_card_schema_mismatch")
        confidence = _require_number(
            record.get("confidence"), label="semantic_record_confidence"
        )
        if confidence < 0 or confidence > 1:
            _fail("semantic_record_confidence_out_of_range")
        for field in ("context_needed", "model_context_needed"):
            if not isinstance(record.get(field), bool):
                _fail(f"semantic_record_{field}_must_be_boolean")
        _require_bool(
            record.get("estimation_eligible"),
            expected=True,
            label="semantic_record_estimation_eligible",
        )
        stratum_id = record.get("sampling_stratum_id")
        stratum = strata_by_id.get(stratum_id)
        if stratum is None:
            _fail("semantic_record_stratum_missing")
        window_key = record.get("window_key")
        segment_id = record.get("segment_id")
        if (
            window_key != stratum["window_key"]
            or segment_id != stratum["segment_id"]
            or record.get("window_role") != WINDOW_ROLES.get(window_key)
            or segment_lookup.get((window_key, segment_id))
            != record.get("window_role")
            or record.get("temporal_block") != stratum["temporal_block"]
            or (
                stratum["proxy_status"] != "collapsed_mixed"
                and record.get("focal_proxy_status")
                != stratum["proxy_status"]
            )
        ):
            _fail("semantic_record_design_lineage_mismatch")
        if (
            record.get("eligible_stratum_count") != stratum["eligible_count"]
            or record.get("selected_stratum_count")
            != stratum["selected_count"]
            or record.get("inclusion_probability_ppm")
            != stratum["inclusion_probability_ppm"]
            or record.get("design_weight_numerator")
            != stratum["weight_numerator"]
            or record.get("design_weight_denominator")
            != stratum["weight_denominator"]
        ):
            _fail("semantic_record_design_weight_mismatch")
        records_by_id[record_id] = record
        records_by_window[window_key] += 1
        records_by_stratum[stratum_id] += 1
    if dict(records_by_window) != WINDOW_SELECTED_COUNTS:
        _fail("semantic_record_window_conservation_failed")
    for stratum_id, stratum in strata_by_id.items():
        if records_by_stratum[stratum_id] != stratum["selected_count"]:
            _fail("semantic_record_stratum_conservation_failed")
    return records_by_id, records_by_window


def _validate_semantic_summary(
    value: Any,
    *,
    records: Sequence[Mapping[str, Any]],
    records_by_window: Counter[str],
) -> None:
    summary = _require_mapping(value, label="semantic_summary")
    _require_exact_keys(
        summary,
        {
            "record_count",
            "records_by_window",
            "language_form_counts",
            "semantic_determinacy_counts",
            "context_needed_count",
            "mean_confidence",
        },
        label="semantic_summary",
    )
    if summary.get("record_count") != len(records):
        _fail("semantic_summary_record_count_mismatch")
    expected_window_counts = dict(sorted(records_by_window.items()))
    if summary.get("records_by_window") != expected_window_counts:
        _fail("semantic_summary_window_counts_mismatch")
    language_counts = dict(
        sorted(Counter(record["language_form"] for record in records).items())
    )
    if summary.get("language_form_counts") != language_counts:
        _fail("semantic_summary_language_counts_mismatch")
    determinacy_counts = dict(
        sorted(
            Counter(
                record["semantic_determinacy"] for record in records
            ).items()
        )
    )
    if summary.get("semantic_determinacy_counts") != determinacy_counts:
        _fail("semantic_summary_determinacy_counts_mismatch")
    context_needed_count = sum(
        record["context_needed"] for record in records
    )
    if summary.get("context_needed_count") != context_needed_count:
        _fail("semantic_summary_context_needed_count_mismatch")
    expected_mean = round(
        sum(float(record["confidence"]) for record in records) / len(records),
        6,
    )
    if summary.get("mean_confidence") != expected_mean:
        _fail("semantic_summary_mean_confidence_mismatch")


def _validate_semantic(
    semantic: Mapping[str, Any],
    *,
    profile: FrozenArtifactProfile,
) -> dict[str, Mapping[str, Any]]:
    _require_exact_keys(
        semantic, SEMANTIC_TOP_LEVEL_FIELDS, label="semantic_parent"
    )
    if (
        semantic.get("windowed_semantic_schema_version")
        != FROZEN_LEGACY_SCHEMA_VERSION
    ):
        _fail("semantic_parent_schema_not_frozen_legacy")
    if semantic.get("status") != "complete":
        _fail("semantic_parent_status_not_complete")
    _validate_generated_at(
        semantic.get("generated_at_utc"), label="semantic_generated_at_utc"
    )
    if semantic.get("run_fingerprint") != profile.run_fingerprint:
        _fail("semantic_parent_run_fingerprint_not_frozen")
    input_evidence = _require_mapping(
        semantic.get("input_evidence"), label="semantic_input_evidence"
    )
    _require_exact_keys(
        input_evidence,
        {"posts_csv", "paths_persisted", "definitions_fingerprint"},
        label="semantic_input_evidence",
    )
    _validate_file_identity_object(
        input_evidence.get("posts_csv"), label="semantic_posts_csv"
    )
    _require_bool(
        input_evidence.get("paths_persisted"),
        expected=False,
        label="semantic_paths_persisted",
    )
    _require_sha256(
        input_evidence.get("definitions_fingerprint"),
        label="semantic_definitions_fingerprint",
    )
    strata_by_id, segment_lookup = _validate_sampling(
        semantic.get("sampling")
    )
    provider = _validate_provider(semantic.get("provider"), context=False)
    execution = _validate_execution(
        semantic.get("execution"),
        stage="semantic",
        expected_record_count=FROZEN_ESTIMATION_RECORD_COUNT,
        expected_batched_record_count=FROZEN_ESTIMATION_RECORD_COUNT,
        expected_batch_count=60,
        batch_prefix="batch-",
    )
    records_by_id, records_by_window = _validate_semantic_records(
        semantic.get("records"),
        strata_by_id=strata_by_id,
        segment_lookup=segment_lookup,
    )
    records = list(records_by_id.values())
    privacy = _require_mapping(
        semantic.get("privacy"), label="semantic_privacy"
    )
    _require_exact_keys(
        privacy,
        {
            "raw_text_persisted",
            "source_ids_persisted",
            "contacts_persisted",
            "redaction_version",
        },
        label="semantic_privacy",
    )
    for field in (
        "raw_text_persisted",
        "source_ids_persisted",
        "contacts_persisted",
    ):
        _require_bool(
            privacy.get(field),
            expected=False,
            label=f"semantic_privacy_{field}",
        )
    _require_nonempty_text(
        privacy.get("redaction_version"),
        label="semantic_privacy_redaction_version",
    )
    validity = _require_mapping(
        semantic.get("validity"), label="semantic_validity"
    )
    expected_validity = {
        "llm_semantic_cards_are_ground_truth": False,
        "window_identity_blinded_from_prompt": True,
        "shock_evidence_use": "evaluation_only",
        "direct_semantic_review_required": True,
    }
    if validity != expected_validity:
        _fail("semantic_validity_contract_mismatch")
    _validate_semantic_summary(
        semantic.get("summary"),
        records=records,
        records_by_window=records_by_window,
    )
    run_contract = {
        "input_evidence": input_evidence,
        "sampling": semantic["sampling"],
        "batch_size": execution["batch_size"],
        "model": provider["requested_model"],
        "reasoning_effort": provider["reasoning_effort"],
        "max_output_tokens": FROZEN_MAX_OUTPUT_TOKENS,
        "redaction_version": privacy["redaction_version"],
        "semantic_card_schema_version": (
            FROZEN_SEMANTIC_CARD_SCHEMA_VERSION
        ),
        "semantic_prompt_version": FROZEN_SEMANTIC_PROMPT_VERSION,
    }
    if _canonical_hash(run_contract) != profile.run_fingerprint:
        _fail("semantic_legacy_run_fingerprint_reconstruction_failed")
    return records_by_id


def _validate_context_selection(value: Any) -> Mapping[str, Any]:
    selection = _require_mapping(value, label="context_selection")
    _require_exact_keys(
        selection,
        {
            "eligible_context_cases",
            "available_unique_comments",
            "comments_used",
            "cases_without_linked_comments",
            "max_comments_per_post",
            "comment_timestamps_used",
            "raw_comments_persisted",
        },
        label="context_selection",
    )
    _require_int(
        selection.get("eligible_context_cases"),
        label="context_selection_eligible_cases",
        exact=FROZEN_CONTEXT_RECORD_COUNT,
    )
    available = _require_int(
        selection.get("available_unique_comments"),
        label="context_selection_available_unique_comments",
    )
    comments_used = _require_int(
        selection.get("comments_used"),
        label="context_selection_comments_used",
    )
    if comments_used > available:
        _fail("context_selection_comments_exceed_available")
    cases_without = _require_int(
        selection.get("cases_without_linked_comments"),
        label="context_selection_cases_without_comments",
    )
    if cases_without > FROZEN_CONTEXT_RECORD_COUNT:
        _fail("context_selection_cases_without_comments_exceed_cases")
    _require_int(
        selection.get("max_comments_per_post"),
        label="context_selection_max_comments_per_post",
        minimum=1,
    )
    _require_bool(
        selection.get("comment_timestamps_used"),
        expected=False,
        label="context_selection_comment_timestamps_used",
    )
    _require_bool(
        selection.get("raw_comments_persisted"),
        expected=False,
        label="context_selection_raw_comments_persisted",
    )
    return selection


def _validate_context_records(
    value: Any,
    *,
    semantic_records: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    records = _require_list(value, label="context_records")
    if len(records) != FROZEN_CONTEXT_RECORD_COUNT:
        _fail("context_record_count_mismatch")
    records_by_id: dict[str, Mapping[str, Any]] = {}
    comments_used = 0
    zero_comment_count = 0
    for raw_record in records:
        record = _require_mapping(raw_record, label="context_record")
        _require_exact_keys(
            record, CONTEXT_RECORD_FIELDS, label="context_record"
        )
        record_id = _require_nonempty_text(
            record.get("record_id"), label="context_record_id"
        )
        if record_id in records_by_id:
            _fail("context_record_id_duplicate")
        source = semantic_records.get(record_id)
        if source is None:
            _fail("context_record_not_in_semantic_parent")
        count = _require_int(
            record.get("comments_used"),
            label="context_record_comments_used",
        )
        if count > selection["max_comments_per_post"]:
            _fail("context_record_comments_exceed_maximum")
        comments_used += count
        zero_comment_count += count == 0
        if record.get("revision_status") not in REVISION_STATUSES:
            _fail("context_record_revision_status_invalid")
        if not isinstance(record.get("context_summary"), str):
            _fail("context_record_summary_must_be_text")
        before = _require_mapping(
            record.get("before_card"), label="context_record_before_card"
        )
        after = _require_mapping(
            record.get("after_card"), label="context_record_after_card"
        )
        _require_exact_keys(
            before,
            SEMANTIC_CARD_FIELDS,
            label="context_record_before_card",
        )
        _require_exact_keys(
            after,
            SEMANTIC_RECORD_FIELDS,
            label="context_record_after_card",
        )
        expected_before = {
            field: source[field] for field in SEMANTIC_CARD_FIELDS
        }
        if before != expected_before:
            _fail("context_record_before_card_not_bound_to_semantic")
        for field in DESIGN_METADATA_FIELDS | {
            "record_id",
            "input_fingerprint",
            "semantic_card_schema_version",
        }:
            if after.get(field) != source.get(field):
                _fail("context_record_after_card_lineage_mismatch")
        confidence = _require_number(
            after.get("confidence"),
            label="context_record_after_confidence",
        )
        if confidence < 0 or confidence > 1:
            _fail("context_record_after_confidence_out_of_range")
        records_by_id[record_id] = record
    expected_context_ids = {
        record_id
        for record_id, record in semantic_records.items()
        if record["context_needed"]
    }
    if set(records_by_id) != expected_context_ids:
        _fail("context_record_routing_conservation_failed")
    if comments_used != selection["comments_used"]:
        _fail("context_record_comment_conservation_failed")
    if zero_comment_count != selection["cases_without_linked_comments"]:
        _fail("context_record_zero_comment_conservation_failed")
    return list(records_by_id.values())


def _validate_context_summary(
    value: Any, *, records: Sequence[Mapping[str, Any]]
) -> None:
    summary = _require_mapping(value, label="context_summary")
    _require_exact_keys(
        summary,
        {
            "record_count",
            "revision_status_counts",
            "meaning_changed_count",
            "mean_confidence_before",
            "mean_confidence_after",
        },
        label="context_summary",
    )
    if summary.get("record_count") != len(records):
        _fail("context_summary_record_count_mismatch")
    status_counts = dict(
        sorted(Counter(record["revision_status"] for record in records).items())
    )
    if summary.get("revision_status_counts") != status_counts:
        _fail("context_summary_revision_counts_mismatch")
    changed = sum(
        record["before_card"]["normalized_meaning"]
        != record["after_card"]["normalized_meaning"]
        for record in records
    )
    if summary.get("meaning_changed_count") != changed:
        _fail("context_summary_meaning_changed_count_mismatch")
    expected_before = round(
        sum(float(record["before_card"]["confidence"]) for record in records)
        / len(records),
        6,
    )
    expected_after = round(
        sum(float(record["after_card"]["confidence"]) for record in records)
        / len(records),
        6,
    )
    if summary.get("mean_confidence_before") != expected_before:
        _fail("context_summary_mean_confidence_before_mismatch")
    if summary.get("mean_confidence_after") != expected_after:
        _fail("context_summary_mean_confidence_after_mismatch")


def _validate_context(
    context: Mapping[str, Any],
    *,
    semantic: Mapping[str, Any],
    semantic_records: Mapping[str, Mapping[str, Any]],
    semantic_identity: Mapping[str, Any],
    profile: FrozenArtifactProfile,
) -> None:
    _require_exact_keys(
        context, CONTEXT_TOP_LEVEL_FIELDS, label="context_parent"
    )
    if (
        context.get("windowed_context_schema_version")
        != FROZEN_LEGACY_SCHEMA_VERSION
    ):
        _fail("context_parent_schema_not_frozen_legacy")
    if context.get("status") != "complete":
        _fail("context_parent_status_not_complete")
    _validate_generated_at(
        context.get("generated_at_utc"), label="context_generated_at_utc"
    )
    if context.get("run_fingerprint") != profile.run_fingerprint:
        _fail("context_parent_run_fingerprint_not_frozen")
    input_evidence = _require_mapping(
        context.get("input_evidence"), label="context_input_evidence"
    )
    _require_exact_keys(
        input_evidence,
        {
            "posts_csv",
            "comments_csv",
            "semantic_result",
            "source_semantic_run_fingerprint",
            "paths_persisted",
        },
        label="context_input_evidence",
    )
    posts_identity = _validate_file_identity_object(
        input_evidence.get("posts_csv"), label="context_posts_csv"
    )
    _validate_file_identity_object(
        input_evidence.get("comments_csv"), label="context_comments_csv"
    )
    source_semantic_identity = _validate_file_identity_object(
        input_evidence.get("semantic_result"),
        label="context_semantic_result",
    )
    if posts_identity != semantic["input_evidence"]["posts_csv"]:
        _fail("context_posts_identity_not_bound_to_semantic")
    if source_semantic_identity != semantic_identity:
        _fail("context_semantic_file_identity_mismatch")
    if (
        input_evidence.get("source_semantic_run_fingerprint")
        != semantic.get("run_fingerprint")
    ):
        _fail("context_semantic_run_fingerprint_mismatch")
    _require_bool(
        input_evidence.get("paths_persisted"),
        expected=False,
        label="context_paths_persisted",
    )
    selection = _validate_context_selection(context.get("context_selection"))
    provider = _validate_provider(context.get("provider"), context=True)
    callable_record_count = (
        FROZEN_CONTEXT_RECORD_COUNT
        - selection["cases_without_linked_comments"]
    )
    execution = _validate_execution(
        context.get("execution"),
        stage="context",
        expected_record_count=FROZEN_CONTEXT_RECORD_COUNT,
        expected_batched_record_count=callable_record_count,
        expected_batch_count=12,
        batch_prefix="context-batch-",
    )
    records = _validate_context_records(
        context.get("records"),
        semantic_records=semantic_records,
        selection=selection,
    )
    privacy = _require_mapping(
        context.get("privacy"), label="context_privacy"
    )
    expected_privacy = {
        "raw_posts_or_comments_persisted": False,
        "source_ids_persisted": False,
        "contacts_persisted": False,
        "semantic_paraphrases_persisted_privately": True,
    }
    if privacy != expected_privacy:
        _fail("context_privacy_contract_mismatch")
    validity = _require_mapping(
        context.get("validity"), label="context_validity"
    )
    expected_validity = {
        "comments_are_context_not_ground_truth": True,
        "comment_timestamps_used": False,
        "direct_qualitative_review_required": True,
    }
    if validity != expected_validity:
        _fail("context_validity_contract_mismatch")
    _validate_context_summary(context.get("summary"), records=records)
    run_contract = {
        "input_evidence": input_evidence,
        "context_selection": selection,
        "batch_size": execution["batch_size"],
        "max_comments_per_post": selection["max_comments_per_post"],
        "model": provider["requested_model"],
        "reasoning_effort": provider["reasoning_effort"],
        "max_output_tokens": FROZEN_MAX_OUTPUT_TOKENS,
        "context_prompt_version": FROZEN_CONTEXT_PROMPT_VERSION,
        "window_labels_sent": provider["window_labels_sent"],
    }
    if _canonical_hash(run_contract) != profile.run_fingerprint:
        _fail("context_legacy_run_fingerprint_reconstruction_failed")


def validate_frozen_legacy_parent_pair(
    semantic: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    semantic_file_identity: Mapping[str, Any],
    context_file_identity: Mapping[str, Any],
) -> FrozenLegacyParentValidation:
    """Validate the exact frozen v1.0 semantic/context parent pair.

    ``semantic_file_identity`` and ``context_file_identity`` must come from
    the actual input files.  The parsed documents are independently rendered
    with the historical writer and checked against the same byte allowlist, so
    callers cannot pair an approved path hash with mutated in-memory content.
    """

    semantic_parent = _require_mapping(semantic, label="semantic_parent")
    context_parent = _require_mapping(context, label="context_parent")
    semantic_schema = semantic_parent.get(
        "windowed_semantic_schema_version"
    )
    context_schema = context_parent.get("windowed_context_schema_version")
    if (
        semantic_schema,
        context_schema,
    ) != (
        FROZEN_LEGACY_SCHEMA_VERSION,
        FROZEN_LEGACY_SCHEMA_VERSION,
    ):
        _fail("legacy_parent_schema_pair_mismatch")
    profile = FROZEN_LEGACY_PROFILE
    _validate_artifact_identity(
        semantic_parent,
        semantic_file_identity,
        profile.semantic,
        label="semantic",
    )
    _validate_artifact_identity(
        context_parent,
        context_file_identity,
        profile.context,
        label="context",
    )
    semantic_records = _validate_semantic(
        semantic_parent, profile=profile.semantic
    )
    _validate_context(
        context_parent,
        semantic=semantic_parent,
        semantic_records=semantic_records,
        semantic_identity={
            "sha256": profile.semantic.sha256,
            "bytes": profile.semantic.bytes,
        },
        profile=profile.context,
    )
    return FrozenLegacyParentValidation(
        schema_pair=(semantic_schema, context_schema),
        semantic_sha256=profile.semantic.sha256,
        context_sha256=profile.context.sha256,
        semantic_run_fingerprint=profile.semantic.run_fingerprint,
        context_run_fingerprint=profile.context.run_fingerprint,
        semantic_record_count=len(semantic_records),
        context_record_count=len(context_parent["records"]),
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail("parent_json_duplicate_key")
        output[key] = value
    return output


def _load_json_object(path: str | Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _: _fail("parent_json_nonfinite_number"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WindowedSensingParentContractError(
            f"{label}_json_unreadable"
        ) from exc
    return _require_mapping(value, label=label)


def validate_frozen_legacy_parent_paths(
    semantic_path: str | Path,
    context_path: str | Path,
) -> FrozenLegacyParentValidation:
    """Load and validate the exact frozen parent files from disk."""

    semantic = _load_json_object(semantic_path, label="semantic_parent")
    context = _load_json_object(context_path, label="context_parent")
    return validate_frozen_legacy_parent_pair(
        semantic,
        context,
        semantic_file_identity=file_identity(semantic_path),
        context_file_identity=file_identity(context_path),
    )


__all__ = [
    "FROZEN_LEGACY_PROFILE",
    "FrozenArtifactProfile",
    "FrozenLegacyParentValidation",
    "FrozenLegacyProfile",
    "WindowedSensingParentContractError",
    "file_identity",
    "validate_frozen_legacy_parent_pair",
    "validate_frozen_legacy_parent_paths",
]
