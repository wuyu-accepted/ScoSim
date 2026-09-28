"""Immutable reviewed ForumTwin trace-release v2 and exact replay source.

This module is deliberately transport-free.  It compiles already validated and
reviewed semantic decisions into a deterministic Merkle-bound bank, and later
replays them as the existing :class:`TurnEnvelope` with zero Provider calls.
Raw prompts, Provider bodies, credentials and corpus/source identifiers are
rejected rather than silently discarded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .forum_governance import ACTOR_TURN_V2_SCHEMA
from .forum_turns import FORUM_AGENT_TURN_SCHEMA_VERSION, OUTPUT_FIELDS
from .governance_arena_v2 import BRANCH_IDS, SCENARIO_IDS, TIMELINE
from .mass_population import canonical_hash
from .yulan_scale import ACTOR_IDS, TurnEnvelope


TRACE_RELEASE_SCHEMA_VERSION = "campus-pulse-reviewed-forum-trace-release-v2"
FORMAL_ENTRY_COUNT = 16_544
FORMAL_RESIDENT_ENTRY_COUNT = 15_776
FORMAL_ACTOR_ENTRY_COUNT = 768
TRACE_SLOT_KINDS = frozenset({"resident", "governance_actor"})
TRACE_BRANCHES = frozenset((*BRANCH_IDS, "shared_baseline"))
DECISION_TICKS = frozenset({3, 7, 13, 19})
FORMAL_SEED_COUNT = 8
FORMAL_PROVIDER_TOKEN_LIMIT = 80_000_000
RESIDENT_OUTPUT_TOKEN_LIMIT = 512
ACTOR_OUTPUT_TOKEN_LIMIT = 384
RESIDENT_ACTIVATION_ROLES = frozenset({"anchor", "interaction", "pps"})
ACTOR_ACTIVATION_ROLE = "governance_actor"

ENTRY_FIELDS = frozenset(
    {
        "slot_id",
        "slot_kind",
        "slot_identity",
        "prompt_sha256",
        "response_sha256",
        "model",
        "token_usage",
        "validated_decision",
        "review_status",
    }
)
COMPILED_ENTRY_FIELDS = ENTRY_FIELDS | {"entry_sha256"}
RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "release_status",
        "model",
        "endpoint_fingerprint",
        "run_plan_sha256",
        "renderer_sha256",
        "entry_count",
        "slot_counts",
        "entry_merkle_root",
        "entries",
        "human_review",
        "replay_contract",
        "release_sha256",
    }
)
SLOT_IDENTITY_FIELDS = frozenset(
    {
        "scenario_id",
        "seed",
        "branch",
        "tick",
        "round_id",
        "subject_id",
        "activation_role",
        "ordinal",
    }
)
TOKEN_USAGE_FIELDS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens"}
)
HUMAN_REVIEW_FIELDS = frozenset(
    {
        "status",
        "reviewer_count",
        "resident_messages_reviewed",
        "boundary_cases_reviewed",
        "anchor_trajectories_complete",
        "featured_threads_complete",
        "governance_public_messages_complete",
        "format_repairs_complete",
        "review_report_sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_SENSITIVE_TEXT_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|api[_ -]?key|password|secret|"
    r"[A-Za-z]:[\\/]|/(?:home|Users|srv|var|tmp)/|\.\.[\\/]|\\\\)",
    re.I,
)
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompt_text",
        "system_prompt",
        "agent_prompt",
        "raw_prompt",
        "provider_body",
        "provider_response",
        "response_body",
        "api_key",
        "password",
        "credential",
        "secret",
        "source_id",
        "source_ids",
        "post_code",
        "comment_id",
        "raw_text",
        "original_text",
        "filesystem_path",
        "database_path",
    }
)


class ForumTraceError(ValueError):
    """Raised when a reviewed ForumTwin trace fails closed."""


def forum_trace_slot_id(slot_identity: Mapping[str, Any]) -> str:
    """Derive the stable, non-source slot ID from its synthetic identity."""

    return "forum-slot-" + canonical_hash(slot_identity)[:32]


def _strict_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ForumTraceError("trace value is not strict JSON") from exc


def _validate_sha(value: Any, field: str) -> str:
    text = str(value or "")
    if _SHA256_RE.fullmatch(text) is None:
        raise ForumTraceError(f"{field} is not a lowercase SHA-256")
    return text


def _validate_safe_id(value: Any, field: str) -> str:
    text = str(value or "")
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise ForumTraceError(f"{field} is not a safe synthetic identifier")
    return text


def _is_forbidden_key(key: str) -> bool:
    normalized = key.strip().lower()
    return (
        normalized in _FORBIDDEN_KEYS
        or normalized.endswith("source_id")
        or normalized.endswith("source_ids")
        or normalized.startswith("raw_")
        or normalized.startswith("original_")
        or normalized.endswith("_path")
    )


def _privacy_scan(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or _is_forbidden_key(key):
                raise ForumTraceError(
                    "trace contains a forbidden field at " + ".".join(path + (str(key),))
                )
            _privacy_scan(child, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _privacy_scan(child, path + (str(index),))
    elif isinstance(value, str) and _SENSITIVE_TEXT_RE.search(value):
        raise ForumTraceError("trace contains a credential or local path")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ForumTraceError("trace contains a non-finite number")


def _validated_slot_identity(
    value: Any,
    *,
    slot_kind: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SLOT_IDENTITY_FIELDS:
        raise ForumTraceError("trace slot identity fields do not match the contract")
    scenario_id = str(value.get("scenario_id") or "")
    branch = str(value.get("branch") or "")
    if scenario_id not in SCENARIO_IDS or branch not in TRACE_BRANCHES:
        raise ForumTraceError("trace slot scenario/branch is invalid")
    seed = value.get("seed")
    tick = value.get("tick")
    ordinal = value.get("ordinal")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
        raise ForumTraceError("trace slot seed is invalid")
    if not isinstance(tick, int) or isinstance(tick, bool) or not 0 <= tick < len(TIMELINE):
        raise ForumTraceError("trace slot tick is invalid")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ForumTraceError("trace slot ordinal is invalid")
    activation_role = str(value.get("activation_role") or "")
    subject_id = _validate_safe_id(value.get("subject_id"), "slot subject_id")
    if slot_kind == "resident":
        if activation_role not in RESIDENT_ACTIVATION_ROLES:
            raise ForumTraceError("resident trace activation role is invalid")
    else:
        if activation_role != ACTOR_ACTIVATION_ROLE or subject_id not in ACTOR_IDS:
            raise ForumTraceError("governance trace actor identity is invalid")
        if branch in {"natural", "shared_baseline"} or tick not in DECISION_TICKS:
            raise ForumTraceError("governance trace branch/tick is invalid")
    if slot_kind == "resident" and branch == "shared_baseline" and tick > 2:
        raise ForumTraceError("shared-baseline resident turn must occur by T2")
    body = {
        "scenario_id": scenario_id,
        "seed": seed,
        "branch": branch,
        "tick": tick,
        "round_id": _validate_safe_id(value.get("round_id"), "slot round_id"),
        "subject_id": subject_id,
        "activation_role": activation_role,
        "ordinal": ordinal,
    }
    _privacy_scan(body)
    return body


def _validated_token_usage(value: Any, *, slot_kind: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != TOKEN_USAGE_FIELDS:
        raise ForumTraceError("trace token usage fields do not match the contract")
    result: dict[str, int] = {}
    for field in TOKEN_USAGE_FIELDS:
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ForumTraceError("trace token usage must be non-negative integers")
        result[field] = item
    if result["total_tokens"] != result["input_tokens"] + result["output_tokens"]:
        raise ForumTraceError("trace token usage does not conserve input/output tokens")
    output_limit = (
        RESIDENT_OUTPUT_TOKEN_LIMIT
        if slot_kind == "resident"
        else ACTOR_OUTPUT_TOKEN_LIMIT
    )
    if result["output_tokens"] > output_limit:
        raise ForumTraceError("trace output token usage exceeds the slot-kind limit")
    return result


def _validated_decision(value: Any, *, slot_kind: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ForumTraceError("trace validated decision must be an object")
    result = json.loads(_strict_json(value))
    schema = result.get("schema_version") or result.get("schema")
    expected_schema = (
        FORUM_AGENT_TURN_SCHEMA_VERSION
        if slot_kind == "resident"
        else ACTOR_TURN_V2_SCHEMA
    )
    if schema != expected_schema:
        raise ForumTraceError("trace validated decision schema is invalid")
    if slot_kind == "resident":
        semantic_fields = set(result) - {"schema_version", "schema"}
        if semantic_fields != OUTPUT_FIELDS:
            raise ForumTraceError("resident trace decision fields are incomplete or expanded")
    else:
        required = {
            "action_probabilities",
            "preferred_action",
            "confidence",
            "rationale_tags",
            "risk_level",
            "uncertainty",
            "public_message",
            "target_claim_id",
            "target_thread_id",
            "evidence_or_service_reference",
        }
        if not required.issubset(result):
            raise ForumTraceError("governance trace decision fields are incomplete")
    _privacy_scan(result)
    return result


def _validated_human_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != HUMAN_REVIEW_FIELDS:
        raise ForumTraceError("trace human-review fields do not match the contract")
    if value.get("status") != "passed":
        raise ForumTraceError("trace human review has not passed")
    for field, minimum in (
        ("reviewer_count", 1),
        ("resident_messages_reviewed", 640),
        ("boundary_cases_reviewed", 128),
    ):
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < minimum:
            raise ForumTraceError(f"trace human review {field} is below the frozen minimum")
    for field in (
        "anchor_trajectories_complete",
        "featured_threads_complete",
        "governance_public_messages_complete",
        "format_repairs_complete",
    ):
        if value.get(field) is not True:
            raise ForumTraceError(f"trace human review {field} is incomplete")
    result = dict(value)
    result["review_report_sha256"] = _validate_sha(
        value.get("review_report_sha256"),
        "review_report_sha256",
    )
    _privacy_scan(result)
    return result


def _compile_entry(raw: Mapping[str, Any], *, expected_model: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != ENTRY_FIELDS:
        raise ForumTraceError("trace entry stores fields outside the frozen contract")
    slot_kind = str(raw.get("slot_kind") or "")
    if slot_kind not in TRACE_SLOT_KINDS:
        raise ForumTraceError("trace slot kind is invalid")
    identity = _validated_slot_identity(raw.get("slot_identity"), slot_kind=slot_kind)
    slot_id = _validate_safe_id(raw.get("slot_id"), "slot_id")
    if slot_id != forum_trace_slot_id(identity):
        raise ForumTraceError("trace slot ID is not bound to its synthetic identity")
    model = str(raw.get("model") or "")
    if model != expected_model:
        raise ForumTraceError("trace entry model does not match the release model")
    if raw.get("review_status") != "passed":
        raise ForumTraceError("every formal trace entry must pass direct review")
    body = {
        "slot_id": slot_id,
        "slot_kind": slot_kind,
        "slot_identity": identity,
        "prompt_sha256": _validate_sha(raw.get("prompt_sha256"), "prompt_sha256"),
        "response_sha256": _validate_sha(raw.get("response_sha256"), "response_sha256"),
        "model": model,
        "token_usage": _validated_token_usage(raw.get("token_usage"), slot_kind=slot_kind),
        "validated_decision": _validated_decision(
            raw.get("validated_decision"),
            slot_kind=slot_kind,
        ),
        "review_status": "passed",
    }
    _privacy_scan(body)
    body["entry_sha256"] = canonical_hash(body)
    return body


def _merkle_root(entry_hashes: Sequence[str]) -> str:
    if not entry_hashes:
        raise ForumTraceError("trace Merkle tree cannot be empty")
    level = [bytes.fromhex(_validate_sha(value, "entry_sha256")) for value in entry_hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _validate_frozen_slot_design(entries: Sequence[Mapping[str, Any]]) -> None:
    """Prove that the complete bank follows the frozen v8 round design."""

    seeds_by_scenario = {
        scenario: {
            int(entry["slot_identity"]["seed"])
            for entry in entries
            if entry["slot_identity"]["scenario_id"] == scenario
        }
        for scenario in SCENARIO_IDS
    }
    if any(len(seeds) != FORMAL_SEED_COUNT for seeds in seeds_by_scenario.values()):
        raise ForumTraceError("formal trace must contain eight seeds per scenario")
    seed_sets = list(seeds_by_scenario.values())
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ForumTraceError("formal trace paired seeds drift across scenarios")

    groups: dict[tuple[str, int, str, int, str], list[Mapping[str, Any]]] = {}
    for entry in entries:
        identity = entry["slot_identity"]
        key = (
            identity["scenario_id"],
            int(identity["seed"]),
            identity["branch"],
            int(identity["tick"]),
            entry["slot_kind"],
        )
        groups.setdefault(key, []).append(entry)

    expected: dict[tuple[str, int, str, int, str], dict[str, int]] = {}
    hero_seed: int | None = None
    for scenario in SCENARIO_IDS:
        resident_totals = {
            seed: sum(
                len(values)
                for (item_scenario, item_seed, _branch, _tick, kind), values in groups.items()
                if item_scenario == scenario and item_seed == seed and kind == "resident"
            )
            for seed in seeds_by_scenario[scenario]
        }
        heroes = [seed for seed, count in resident_totals.items() if count == 2_512]
        if len(heroes) != 1 or any(
            count not in {2_512, 768} for count in resident_totals.values()
        ):
            raise ForumTraceError("formal trace hero/semantic-spine resident totals are invalid")
        if hero_seed is None:
            hero_seed = heroes[0]
        elif hero_seed != heroes[0]:
            raise ForumTraceError("formal trace canonical hero seed drifts across scenarios")
        for seed in seeds_by_scenario[scenario]:
            is_hero = seed == heroes[0]
            if is_hero:
                for tick, pps in ((0, 8), (1, 8), (2, 32)):
                    expected[(scenario, seed, "shared_baseline", tick, "resident")] = {
                        "anchor": 16, "interaction": 0, "pps": pps,
                    }
                for branch in ("natural", "D"):
                    for tick in range(3, 24):
                        interaction = 24 if tick <= 7 else 16 if tick <= 17 else 8
                        budget = 64 if tick <= 7 else 48 if tick <= 17 else 32
                        expected[(scenario, seed, branch, tick, "resident")] = {
                            "anchor": 16,
                            "interaction": interaction,
                            "pps": budget - 16 - interaction,
                        }
                for branch in ("A", "B", "C"):
                    for tick in (7, 13, 19):
                        expected[(scenario, seed, branch, tick, "resident")] = {
                            "anchor": 16, "interaction": 0, "pps": 32,
                        }
            else:
                expected[(scenario, seed, "shared_baseline", 2, "resident")] = {
                    "anchor": 16, "interaction": 0, "pps": 32,
                }
                for branch in BRANCH_IDS:
                    for tick in (7, 13, 19):
                        expected[(scenario, seed, branch, tick, "resident")] = {
                            "anchor": 16, "interaction": 0, "pps": 32,
                        }
            for branch in ("A", "B", "C", "D"):
                for tick in DECISION_TICKS:
                    expected[(scenario, seed, branch, tick, "governance_actor")] = {
                        "governance_actor": 3,
                    }
    if set(groups) != set(expected):
        raise ForumTraceError("formal trace round identities do not match the v8 slot plan")
    for key, values in groups.items():
        identities = [entry["slot_identity"] for entry in values]
        if len({identity["round_id"] for identity in identities}) != 1:
            raise ForumTraceError("formal trace round ID drifts within a round")
        if sorted(identity["ordinal"] for identity in identities) != list(range(len(values))):
            raise ForumTraceError("formal trace slot ordinals are not contiguous")
        actual_roles: dict[str, int] = {}
        for identity in identities:
            role = identity["activation_role"]
            actual_roles[role] = actual_roles.get(role, 0) + 1
        if actual_roles != {role: count for role, count in expected[key].items() if count}:
            raise ForumTraceError("formal trace activation-role composition drifted")
        if key[-1] == "governance_actor" and {
            identity["subject_id"] for identity in identities
        } != set(ACTOR_IDS):
            raise ForumTraceError("formal trace governance round lacks the three actors")


def compile_reviewed_forum_trace_release_v2(
    *,
    entries: Sequence[Mapping[str, Any]],
    model: str,
    endpoint_fingerprint: str,
    run_plan_sha256: str,
    renderer_sha256: str,
    human_review: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the complete 16,544-slot bank without any network operation."""

    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ForumTraceError("trace entries must be a sequence")
    if len(entries) != FORMAL_ENTRY_COUNT:
        raise ForumTraceError("formal reviewed trace requires exactly 16,544 entries")
    model_name = _validate_safe_id(model, "release model")
    endpoint_hash = _validate_sha(endpoint_fingerprint, "endpoint_fingerprint")
    run_plan_hash = _validate_sha(run_plan_sha256, "run_plan_sha256")
    renderer_hash = _validate_sha(renderer_sha256, "renderer_sha256")
    review = _validated_human_review(human_review)
    compiled = [_compile_entry(raw, expected_model=model_name) for raw in entries]
    slot_ids = [entry["slot_id"] for entry in compiled]
    prompt_ids = [entry["prompt_sha256"] for entry in compiled]
    if len(set(slot_ids)) != len(slot_ids):
        raise ForumTraceError("formal trace contains a duplicate slot identity")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ForumTraceError("formal trace contains a duplicate prompt fingerprint")
    compiled.sort(key=lambda entry: entry["slot_id"])
    counts = {
        "resident": sum(entry["slot_kind"] == "resident" for entry in compiled),
        "governance_actor": sum(
            entry["slot_kind"] == "governance_actor" for entry in compiled
        ),
    }
    if counts != {
        "resident": FORMAL_RESIDENT_ENTRY_COUNT,
        "governance_actor": FORMAL_ACTOR_ENTRY_COUNT,
    }:
        raise ForumTraceError("formal trace resident/actor slot counts are invalid")
    _validate_frozen_slot_design(compiled)
    if sum(entry["token_usage"]["total_tokens"] for entry in compiled) > FORMAL_PROVIDER_TOKEN_LIMIT:
        raise ForumTraceError("formal trace token usage exceeds the frozen hard limit")
    body = {
        "schema_version": TRACE_RELEASE_SCHEMA_VERSION,
        "release_status": "complete_reviewed",
        "model": model_name,
        "endpoint_fingerprint": endpoint_hash,
        "run_plan_sha256": run_plan_hash,
        "renderer_sha256": renderer_hash,
        "entry_count": len(compiled),
        "slot_counts": counts,
        "entry_merkle_root": _merkle_root(
            [entry["entry_sha256"] for entry in compiled]
        ),
        "entries": compiled,
        "human_review": review,
        "replay_contract": {
            "provider_call_limit": 0,
            "network_required": False,
            "lookup_key": "slot_kind+prompt_sha256",
            "prompt_body_persisted": False,
            "provider_body_persisted": False,
            "credentials_persisted": False,
            "source_ids_persisted": False,
            "paths_persisted": False,
        },
    }
    _privacy_scan(body)
    body["release_sha256"] = canonical_hash(body)
    return body


def validate_reviewed_forum_trace_release_v2(
    release: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an untrusted loaded release and return a detached JSON copy."""

    if not isinstance(release, Mapping) or set(release) != RELEASE_FIELDS:
        raise ForumTraceError("trace release fields do not match schema v2")
    detached = json.loads(_strict_json(release))
    if detached.get("schema_version") != TRACE_RELEASE_SCHEMA_VERSION:
        raise ForumTraceError("trace release schema is invalid")
    if detached.get("release_status") != "complete_reviewed":
        raise ForumTraceError("trace release is not complete and reviewed")
    if detached.get("entry_count") != FORMAL_ENTRY_COUNT:
        raise ForumTraceError("trace release entry count is invalid")
    if detached.get("slot_counts") != {
        "resident": FORMAL_RESIDENT_ENTRY_COUNT,
        "governance_actor": FORMAL_ACTOR_ENTRY_COUNT,
    }:
        raise ForumTraceError("trace release slot counts are invalid")
    model = _validate_safe_id(detached.get("model"), "release model")
    _validate_sha(detached.get("run_plan_sha256"), "run_plan_sha256")
    _validate_sha(detached.get("renderer_sha256"), "renderer_sha256")
    _validate_sha(detached.get("endpoint_fingerprint"), "endpoint_fingerprint")
    entries = detached.get("entries")
    if not isinstance(entries, list) or len(entries) != FORMAL_ENTRY_COUNT:
        raise ForumTraceError("trace release entries are invalid")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ForumTraceError("trace release contains a non-object entry")
    if [entry.get("slot_id") for entry in entries] != sorted(
        entry.get("slot_id") for entry in entries
    ):
        raise ForumTraceError("trace release entries are not canonically ordered")
    seen_slots: set[str] = set()
    seen_prompts: set[str] = set()
    counts = {"resident": 0, "governance_actor": 0}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != COMPILED_ENTRY_FIELDS:
            raise ForumTraceError("compiled trace entry fields are invalid")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        expected = _compile_entry(body, expected_model=model)
        if expected != entry:
            raise ForumTraceError("compiled trace entry hash/content drifted")
        if entry["slot_id"] in seen_slots or entry["prompt_sha256"] in seen_prompts:
            raise ForumTraceError("trace release repeats a slot or prompt")
        seen_slots.add(entry["slot_id"])
        seen_prompts.add(entry["prompt_sha256"])
        counts[entry["slot_kind"]] += 1
    if counts != detached["slot_counts"]:
        raise ForumTraceError("trace release computed slot counts drifted")
    _validate_frozen_slot_design(entries)
    if sum(entry["token_usage"]["total_tokens"] for entry in entries) > FORMAL_PROVIDER_TOKEN_LIMIT:
        raise ForumTraceError("trace release token usage exceeds the frozen hard limit")
    _validated_human_review(detached.get("human_review"))
    expected_merkle = _merkle_root([entry["entry_sha256"] for entry in entries])
    if detached.get("entry_merkle_root") != expected_merkle:
        raise ForumTraceError("trace release Merkle root drifted")
    replay_contract = detached.get("replay_contract")
    if replay_contract != {
        "provider_call_limit": 0,
        "network_required": False,
        "lookup_key": "slot_kind+prompt_sha256",
        "prompt_body_persisted": False,
        "provider_body_persisted": False,
        "credentials_persisted": False,
        "source_ids_persisted": False,
        "paths_persisted": False,
    }:
        raise ForumTraceError("trace release zero-Provider replay contract drifted")
    _privacy_scan(detached)
    expected_release_hash = canonical_hash(
        {key: value for key, value in detached.items() if key != "release_sha256"}
    )
    if detached.get("release_sha256") != expected_release_hash:
        raise ForumTraceError("trace release content hash drifted")
    return detached


class ReviewedForumTraceReplaySourceV2:
    """Exact `(slot_kind, prompt_sha256)` replay with zero Provider usage."""

    provenance = "reviewed_trace_replay"

    def __init__(
        self,
        release: Mapping[str, Any],
        *,
        expected_release_sha256: str,
        expected_model: str,
        expected_endpoint_fingerprint: str,
        expected_run_plan_sha256: str,
        expected_renderer_sha256: str,
    ) -> None:
        if not isinstance(release, Mapping):
            raise ForumTraceError("reviewed trace release must be an object")
        expected_release = _validate_sha(
            expected_release_sha256,
            "expected_release_sha256",
        )
        expected_run_plan = _validate_sha(
            expected_run_plan_sha256,
            "expected_run_plan_sha256",
        )
        expected_renderer = _validate_sha(
            expected_renderer_sha256,
            "expected_renderer_sha256",
        )
        expected_endpoint = _validate_sha(
            expected_endpoint_fingerprint,
            "expected_endpoint_fingerprint",
        )
        # Reject obvious binding mismatches before the intentionally expensive
        # full 16,544-entry validation.  A matching identity still receives the
        # complete entry/hash/Merkle/privacy audit below.
        if release.get("release_sha256") != expected_release:
            raise ForumTraceError("reviewed trace release identity mismatch")
        if release.get("model") != expected_model:
            raise ForumTraceError("reviewed trace model mismatch")
        if release.get("endpoint_fingerprint") != expected_endpoint:
            raise ForumTraceError("reviewed trace endpoint mismatch")
        if release.get("run_plan_sha256") != expected_run_plan:
            raise ForumTraceError("reviewed trace run-plan mismatch")
        if release.get("renderer_sha256") != expected_renderer:
            raise ForumTraceError("reviewed trace renderer mismatch")
        validated = validate_reviewed_forum_trace_release_v2(release)
        self.release_sha256 = validated["release_sha256"]
        self.model = validated["model"]
        self.endpoint_fingerprint = validated["endpoint_fingerprint"]
        self.run_plan_sha256 = validated["run_plan_sha256"]
        self.renderer_sha256 = validated["renderer_sha256"]
        self.provider_calls = 0
        self.network_requests = 0
        self._entries = {
            (entry["slot_kind"], entry["prompt_sha256"]): entry
            for entry in validated["entries"]
        }

    def acquire(
        self,
        *,
        slot_kind: str,
        prompt_sha256: str,
        rendered: Any | None = None,
        validator: Any | None = None,
    ) -> TurnEnvelope:
        del validator
        if slot_kind not in TRACE_SLOT_KINDS:
            raise ForumTraceError("exact replay slot kind is invalid")
        prompt_hash = _validate_sha(prompt_sha256, "prompt_sha256")
        if rendered is not None:
            if getattr(rendered, "prompt_sha256", None) != prompt_hash:
                raise ForumTraceError("exact replay rendered prompt mismatch")
            rendered_model = getattr(rendered, "expected_model", self.model)
            if rendered_model != self.model:
                raise ForumTraceError("exact replay rendered model mismatch")
            rendered_renderer = getattr(rendered, "renderer_sha256", self.renderer_sha256)
            if rendered_renderer != self.renderer_sha256:
                raise ForumTraceError("exact replay rendered renderer mismatch")
        try:
            entry = self._entries[(slot_kind, prompt_hash)]
        except KeyError as exc:
            raise ForumTraceError("exact reviewed forum trace entry is missing") from exc
        decision = {
            key: value
            for key, value in entry["validated_decision"].items()
            if key not in {"schema", "schema_version"}
        }
        usage = entry["token_usage"]
        return TurnEnvelope(
            slot_kind=slot_kind,
            prompt_sha256=prompt_hash,
            response_sha256=entry["response_sha256"],
            provenance=self.provenance,
            content=_strict_json(decision),
            returned_model=entry["model"],
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            provider_calls=0,
            cache_hit=True,
            provider_attempts=0,
            format_repairs=0,
            provider_billed_tokens=0,
        )
