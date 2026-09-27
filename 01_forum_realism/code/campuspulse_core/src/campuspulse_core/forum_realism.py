"""Aggregate-only realism diagnostics for ForumTwin public results.

The benchmark deliberately stays below the record-identifying boundary.  It
compares published simulation structure with the frozen full-corpus ecology
release and emits only distributions, rates and input hashes.  It never reads
or writes raw forum text, source identifiers or private Feed exposures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REALISM_SCHEMA_VERSION = "campus-pulse-forum-realism-benchmark-v1"
CALIBRATED_REALISM_SCHEMA_VERSION = "campus-pulse-forum-realism-benchmark-v2"
GOVERNANCE_DECISION_MAP_SCHEMA_VERSION = (
    "campus-pulse-governance-decision-map-v1"
)
FOUR_DIMENSION_FIDELITY_SCHEMA_VERSION = (
    "campus-pulse-four-dimension-fidelity-result-v1"
)
THREAD_BUCKETS = ("zero", "small", "medium", "large")
REPLY_FUNCTIONS = (
    "question",
    "request",
    "help_request",
    "complaint",
    "emotional_vent",
    "information_share",
    "discussion",
    "experience_share_or_ask",
    "other",
)
INTERACTION_KINDS = ("like", "repost", "report")


class ForumRealismError(ValueError):
    """Raised when a benchmark input violates its public aggregate contract."""


@dataclass(frozen=True)
class PublicWorld:
    source_label: str
    scenario_id: str
    branch: str
    seed: int | None
    semantic_schedule: str
    payload: Mapping[str, Any]

    @property
    def world_id(self) -> str:
        seed = "none" if self.seed is None else str(self.seed)
        return f"{self.source_label}:{self.scenario_id}:{seed}:{self.branch}"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(counts: Mapping[str, float], order: Sequence[str]) -> dict[str, float]:
    values = [max(0.0, float(counts.get(key, 0.0))) for key in order]
    total = sum(values)
    if total <= 0:
        return {key: 0.0 for key in order}
    return {key: round(value / total, 6) for key, value in zip(order, values)}


def _js_divergence(
    left: Mapping[str, float],
    right: Mapping[str, float],
    order: Sequence[str],
) -> float:
    p = [max(0.0, float(left.get(key, 0.0))) for key in order]
    q = [max(0.0, float(right.get(key, 0.0))) for key in order]
    p_total = sum(p)
    q_total = sum(q)
    if p_total <= 0 or q_total <= 0:
        raise ForumRealismError("JS divergence requires two non-empty distributions")
    p = [value / p_total for value in p]
    q = [value / q_total for value in q]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(values: Sequence[float], baseline: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / reference)
            for value, reference in zip(values, baseline)
            if value > 0 and reference > 0
        )

    return round(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint), 6)


def _thread_bucket(reply_count: int) -> str:
    if reply_count < 0:
        raise ForumRealismError("thread reply_count cannot be negative")
    if reply_count == 0:
        return "zero"
    if reply_count <= 3:
        return "small"
    if reply_count <= 12:
        return "medium"
    return "large"


def _normalized_reply_function(value: str) -> str:
    raw = value.strip().lower()
    aliases = {
        "advice_seeking": "question",
        "ask_followup": "question",
        "information_request": "question",
        "emotional_expression": "emotional_vent",
        "information_share_or_ask": "information_share",
        "risk_advice_request": "complaint",
        "risk_report": "complaint",
        "contextual_participant": "other",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in REPLY_FUNCTIONS else "other"


def _profile_reply_function(profile: Mapping[str, Any]) -> str:
    language = profile.get("language_style")
    if isinstance(language, Mapping):
        tendencies = language.get("tendencies")
        if isinstance(tendencies, Sequence) and not isinstance(tendencies, (str, bytes)):
            for value in reversed(tendencies):
                normalized = _normalized_reply_function(str(value))
                if normalized != "other" or str(value).strip().lower() == "other":
                    return normalized
        summary = str(language.get("summary") or "")
        for token in reversed(summary.replace("/", " ").replace("_", " ").split()):
            normalized = _normalized_reply_function(token)
            if normalized != "other":
                return normalized
    return "other"


def ecology_reference(ecology: Mapping[str, Any]) -> dict[str, Any]:
    if ecology.get("schema_version") != "campus-pulse-forum-ecology-release-v1":
        raise ForumRealismError("ecology release schema is unsupported")
    episode_count = int(ecology.get("episode_count", 0))
    bucket_counts = ecology.get("thread_size_bucket_counts")
    reply_distribution = ecology.get("reply_function_distribution")
    if episode_count <= 0 or not isinstance(bucket_counts, Mapping):
        raise ForumRealismError("ecology release has no thread population")
    if sum(int(bucket_counts.get(key, 0)) for key in THREAD_BUCKETS) != episode_count:
        raise ForumRealismError("ecology thread buckets do not conserve episode_count")
    if not isinstance(reply_distribution, Mapping):
        raise ForumRealismError("ecology release has no reply-function distribution")
    normalized_reply_counts: Counter[str] = Counter()
    for key, value in reply_distribution.items():
        normalized_reply_counts[_normalized_reply_function(str(key))] += float(value)
    thread_distribution = _distribution(bucket_counts, THREAD_BUCKETS)
    return {
        "episode_count": episode_count,
        "thread_size_distribution": thread_distribution,
        "reply_bearing_thread_share": round(1.0 - thread_distribution["zero"], 6),
        "reply_function_distribution": _distribution(
            normalized_reply_counts, REPLY_FUNCTIONS
        ),
        "micro_role_count": int(ecology.get("micro_role_count", 0)),
        "privacy": dict(ecology.get("privacy") or {}),
        "ecology_release_sha256": str(ecology.get("ecology_release_sha256") or ""),
    }


def extract_public_worlds(
    payload: Mapping[str, Any],
    *,
    source_label: str,
    include_semantic_spine: bool = False,
) -> list[PublicWorld]:
    result = payload.get("result", payload)
    if not isinstance(result, Mapping):
        raise ForumRealismError("result payload is not an object")
    scenario_id = str(result.get("scenario_id") or "unknown_scenario")
    worlds: list[PublicWorld] = []
    seed_results = result.get("seed_results")
    if isinstance(seed_results, Sequence) and not isinstance(seed_results, (str, bytes)):
        for seed_row in seed_results:
            if not isinstance(seed_row, Mapping):
                continue
            canonical = bool(seed_row.get("canonical_full_forum_seed"))
            branches = seed_row.get("branches")
            if not isinstance(branches, Mapping):
                continue
            for branch, branch_payload in branches.items():
                if not isinstance(branch_payload, Mapping):
                    continue
                schedule = str(branch_payload.get("semantic_schedule") or "unknown")
                if not include_semantic_spine and not canonical:
                    continue
                if not include_semantic_spine and schedule not in {
                    "canonical_full_forum",
                    "hero_tick_by_tick",
                    "full_forum",
                }:
                    continue
                worlds.append(
                    PublicWorld(
                        source_label=source_label,
                        scenario_id=scenario_id,
                        branch=str(branch),
                        seed=int(seed_row["seed"]),
                        semantic_schedule=schedule,
                        payload=branch_payload,
                    )
                )
        return worlds
    branches = result.get("branches")
    if not isinstance(branches, Mapping):
        raise ForumRealismError("result contains no public branch payloads")
    for branch, branch_payload in branches.items():
        if not isinstance(branch_payload, Mapping):
            continue
        worlds.append(
            PublicWorld(
                source_label=source_label,
                scenario_id=scenario_id,
                branch=str(branch),
                seed=int(result["seed"]) if result.get("seed") is not None else None,
                semantic_schedule=str(
                    branch_payload.get("semantic_schedule") or "hero_tick_by_tick"
                ),
                payload=branch_payload,
            )
        )
    return worlds


def world_metrics(
    world: PublicWorld,
    *,
    mature_min_age_ticks: int = 2,
) -> dict[str, Any]:
    if mature_min_age_ticks < 0:
        raise ForumRealismError("mature thread age cannot be negative")
    threads = world.payload.get("threads")
    messages = world.payload.get("messages")
    if not isinstance(threads, Sequence) or isinstance(threads, (str, bytes)):
        raise ForumRealismError(f"{world.world_id} has no public threads")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ForumRealismError(f"{world.world_id} has no public messages")
    bucket_counts: Counter[str] = Counter()
    thread_message_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    resident_message_counts: Counter[str] = Counter()
    for message in messages:
        if not isinstance(message, Mapping):
            raise ForumRealismError(f"{world.world_id} contains a malformed message")
        thread_message_counts[str(message.get("thread_id") or "")] += 1
        action_counts[str(message.get("action") or "unknown")] += 1
        display_id = str(message.get("source_agent_display_id") or "")
        if display_id and not display_id.startswith("governance:"):
            resident_message_counts[display_id] += 1
    timeline = world.payload.get("timeline")
    timeline_ticks = [
        int(row.get("tick", 0))
        for row in timeline or ()
        if isinstance(row, Mapping)
    ]
    message_ticks = [
        int(row.get("created_tick", 0))
        for row in messages
        if isinstance(row, Mapping)
    ]
    final_tick = max((*timeline_ticks, *message_ticks), default=0)
    reply_bearing = 0
    thread_sizes = []
    mature_bucket_counts: Counter[str] = Counter()
    mature_reply_bearing = 0
    mature_thread_sizes = []
    for thread in threads:
        if not isinstance(thread, Mapping):
            raise ForumRealismError(f"{world.world_id} contains a malformed thread")
        reply_count = int(thread.get("reply_count", 0))
        bucket_counts[_thread_bucket(reply_count)] += 1
        reply_bearing += int(reply_count > 0)
        size = max(1, thread_message_counts[str(thread.get("thread_id") or "")])
        thread_sizes.append(size)
        created_tick = int(thread.get("created_tick", 0) or 0)
        if final_tick - created_tick >= mature_min_age_ticks:
            mature_bucket_counts[_thread_bucket(reply_count)] += 1
            mature_reply_bearing += int(reply_count > 0)
            mature_thread_sizes.append(size)
    thread_count = len(threads)
    if thread_count <= 0:
        raise ForumRealismError(f"{world.world_id} has an empty thread list")
    top_count = max(1, math.ceil(thread_count * 0.10))
    message_total = sum(thread_sizes)
    concentration = (
        sum(sorted(thread_sizes, reverse=True)[:top_count]) / message_total
        if message_total
        else 0.0
    )
    mature_thread_count = len(mature_thread_sizes)
    mature_top_count = max(1, math.ceil(mature_thread_count * 0.10))
    mature_message_total = sum(mature_thread_sizes)
    mature_concentration = (
        sum(sorted(mature_thread_sizes, reverse=True)[:mature_top_count])
        / mature_message_total
        if mature_message_total
        else 0.0
    )
    profiles = world.payload.get("public_display_profiles")
    profile_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
        for profile in profiles:
            if isinstance(profile, Mapping):
                profile_by_id[str(profile.get("display_id") or "")] = profile
    reply_functions: Counter[str] = Counter()
    missing_profile_messages = 0
    for display_id, count in resident_message_counts.items():
        profile = profile_by_id.get(display_id)
        if profile is None:
            missing_profile_messages += count
            continue
        reply_functions[_profile_reply_function(profile)] += count
    parallel = world.payload.get("parallel_interactions")
    parallel = parallel if isinstance(parallel, Mapping) else {}
    interaction_counts = parallel.get("counts_by_kind")
    interaction_counts = interaction_counts if isinstance(interaction_counts, Mapping) else {}
    observed_interactions = {
        kind: int(interaction_counts.get(kind, 0)) for kind in INTERACTION_KINDS
    }
    return {
        "world_id": world.world_id,
        "source_label": world.source_label,
        "scenario_id": world.scenario_id,
        "branch": world.branch,
        "seed": world.seed,
        "semantic_schedule": world.semantic_schedule,
        "thread_count": thread_count,
        "message_count": len(messages),
        "resident_message_count": sum(resident_message_counts.values()),
        "unique_resident_speakers": len(resident_message_counts),
        "thread_size_counts": {key: int(bucket_counts[key]) for key in THREAD_BUCKETS},
        "thread_size_distribution": _distribution(bucket_counts, THREAD_BUCKETS),
        "reply_bearing_thread_share": round(reply_bearing / thread_count, 6),
        "top_10pct_thread_message_share": round(concentration, 6),
        "mature_thread_definition": (
            f"final_tick - created_tick >= {mature_min_age_ticks}"
        ),
        "final_tick": final_tick,
        "mature_thread_count": mature_thread_count,
        "mature_thread_size_counts": {
            key: int(mature_bucket_counts[key]) for key in THREAD_BUCKETS
        },
        "mature_thread_size_distribution": _distribution(
            mature_bucket_counts, THREAD_BUCKETS
        ),
        "mature_reply_bearing_thread_share": round(
            mature_reply_bearing / mature_thread_count
            if mature_thread_count
            else 0.0,
            6,
        ),
        "mature_top_10pct_thread_message_share": round(
            mature_concentration, 6
        ),
        "action_distribution": _distribution(action_counts, sorted(action_counts)),
        "activated_profile_reply_function_distribution": _distribution(
            reply_functions, REPLY_FUNCTIONS
        ),
        "messages_missing_public_profile": missing_profile_messages,
        "interaction_counts": observed_interactions,
        "interaction_count": sum(observed_interactions.values()),
        "compound_speech_and_engagement_turns": int(
            parallel.get("compound_speech_and_engagement_turn_count", 0)
        ),
        "multi_kind_engagement_turns": int(
            parallel.get("multi_kind_engagement_turn_count", 0)
        ),
    }


def governance_decision_world(world: PublicWorld) -> dict[str, Any]:
    """Build a public, strict-linkage map for one governance branch."""

    messages = world.payload.get("messages")
    chains = world.payload.get("mechanism_chains")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ForumRealismError(f"{world.world_id} has no public messages")
    chains = chains if isinstance(chains, Mapping) else {}
    uptake_rows = chains.get("governance_uptake_chains")
    uptake_rows = (
        uptake_rows
        if isinstance(uptake_rows, Sequence)
        and not isinstance(uptake_rows, (str, bytes))
        else []
    )
    message_by_id = {
        str(row.get("message_id")): row
        for row in messages
        if isinstance(row, Mapping) and row.get("message_id")
    }
    chain_by_governance_id = {
        str(row.get("governance_message_id")): row
        for row in uptake_rows
        if isinstance(row, Mapping) and row.get("governance_message_id")
    }
    governance_messages = [
        row
        for row in messages
        if isinstance(row, Mapping)
        and str(row.get("source_agent_display_id") or "").startswith("governance:")
        and row.get("moderation_status") != "superseded"
    ]
    governance_messages.sort(
        key=lambda row: (int(row.get("created_tick", 0)), str(row.get("message_id")))
    )
    entries = []
    artifact_counts: Counter[str] = Counter()
    artifact_uptake: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    latencies: list[int] = []
    for governance in governance_messages:
        governance_id = str(governance.get("message_id"))
        chain = chain_by_governance_id.get(governance_id, {})
        response_ids = [
            str(value) for value in list(chain.get("resident_response_ids") or [])
        ]
        responses = [message_by_id[value] for value in response_ids if value in message_by_id]
        stances = Counter(str(row.get("stance") or "unknown") for row in responses)
        operations = Counter(
            str(row.get("claim_operation") or "none") for row in responses
        )
        support = int(stances["support"] + operations["support"])
        challenge = int(stances["oppose"] + operations["challenge"])
        if support and challenge:
            status = "mixed_direct_uptake"
        elif challenge:
            status = "direct_challenge"
        elif support:
            status = "direct_support"
        elif responses:
            status = "direct_response_without_clear_alignment"
        else:
            status = "no_direct_uptake"
        status_counts[status] += 1
        artifact = governance.get("governance_artifact")
        artifact = artifact if isinstance(artifact, Mapping) else None
        artifact_kind = str(
            artifact.get("artifact_kind") if artifact is not None else "message_only"
        )
        artifact_counts[artifact_kind] += 1
        if responses:
            artifact_uptake[artifact_kind] += 1
        created_tick = int(governance.get("created_tick", 0))
        response_latencies = [
            int(row.get("created_tick", created_tick)) - created_tick for row in responses
        ]
        latencies.extend(response_latencies)
        entries.append(
            {
                "governance_message_id": governance_id,
                "created_tick": created_tick,
                "role": str(governance.get("source_agent_display_id")).removeprefix(
                    "governance:"
                ),
                "visible_text": str(governance.get("visible_text") or ""),
                "artifact": dict(artifact) if artifact is not None else None,
                "uptake_status": status,
                "direct_response_count": len(responses),
                "direct_response_latency_ticks": response_latencies,
                "direct_response_stances": dict(sorted(stances.items())),
                "direct_response_claim_operations": dict(sorted(operations.items())),
                "direct_response_examples": [
                    {
                        "message_id": str(row.get("message_id")),
                        "created_tick": int(row.get("created_tick", 0)),
                        "link_kind": (
                            "reply"
                            if row.get("parent_message_id") == governance_id
                            else "quote"
                        ),
                        "stance": str(row.get("stance") or "unknown"),
                        "visible_text": str(row.get("visible_text") or ""),
                    }
                    for row in responses[:3]
                ],
                "strict_uptake_rule": "later_direct_reply_or_quote_only",
            }
        )
    ordered_latencies = sorted(latencies)
    median_latency = None
    if ordered_latencies:
        midpoint = len(ordered_latencies) // 2
        median_latency = (
            float(ordered_latencies[midpoint])
            if len(ordered_latencies) % 2
            else round(
                (ordered_latencies[midpoint - 1] + ordered_latencies[midpoint])
                / 2.0,
                6,
            )
        )
    message_count = len(governance_messages)
    with_uptake = sum(bool(row["direct_response_count"]) for row in entries)
    return {
        "world_id": world.world_id,
        "scenario_id": world.scenario_id,
        "branch": world.branch,
        "seed": world.seed,
        "summary": {
            "governance_message_count": message_count,
            "messages_with_direct_uptake": with_uptake,
            "direct_uptake_rate": round(
                with_uptake / message_count if message_count else 0.0, 6
            ),
            "direct_resident_response_count": len(latencies),
            "median_direct_response_latency_ticks": median_latency,
            "uptake_status_counts": dict(sorted(status_counts.items())),
            "artifact_message_counts": dict(sorted(artifact_counts.items())),
            "artifact_messages_with_direct_uptake": dict(
                sorted(artifact_uptake.items())
            ),
        },
        "entries": entries,
    }


def build_governance_decision_map(
    *,
    worlds: Iterable[PublicWorld],
    input_files: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Aggregate strict uptake into branch-level operational decision cards."""

    rows = [governance_decision_world(world) for world in worlds]
    if not rows:
        raise ForumRealismError("no public worlds were supplied for governance mapping")
    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        branch = str(row["branch"])
        target = aggregate.setdefault(
            branch,
            {
                "world_count": 0,
                "governance_message_count": 0,
                "messages_with_direct_uptake": 0,
                "direct_resident_response_count": 0,
                "artifact_message_counts": Counter(),
                "artifact_messages_with_direct_uptake": Counter(),
                "uptake_status_counts": Counter(),
            },
        )
        summary = row["summary"]
        target["world_count"] += 1
        for key in (
            "governance_message_count",
            "messages_with_direct_uptake",
            "direct_resident_response_count",
        ):
            target[key] += int(summary[key])
        target["artifact_message_counts"].update(summary["artifact_message_counts"])
        target["artifact_messages_with_direct_uptake"].update(
            summary["artifact_messages_with_direct_uptake"]
        )
        target["uptake_status_counts"].update(summary["uptake_status_counts"])
    branch_cards = {}
    for branch, raw in sorted(aggregate.items()):
        messages = int(raw["governance_message_count"])
        uptake = int(raw["messages_with_direct_uptake"])
        unanswered = messages - uptake
        gap = (
            "No governance message was issued in this branch."
            if messages == 0
            else "Every governance message received a strict direct response."
            if unanswered == 0
            else f"{unanswered} governance messages received no direct reply or quote."
        )
        branch_cards[branch] = {
            "world_count": int(raw["world_count"]),
            "governance_message_count": messages,
            "messages_with_direct_uptake": uptake,
            "direct_uptake_rate": round(uptake / messages if messages else 0.0, 6),
            "direct_resident_response_count": int(
                raw["direct_resident_response_count"]
            ),
            "artifact_message_counts": dict(
                sorted(raw["artifact_message_counts"].items())
            ),
            "artifact_messages_with_direct_uptake": dict(
                sorted(raw["artifact_messages_with_direct_uptake"].items())
            ),
            "uptake_status_counts": dict(
                sorted(raw["uptake_status_counts"].items())
            ),
            "operational_gap": gap,
        }
    body: dict[str, Any] = {
        "schema_version": GOVERNANCE_DECISION_MAP_SCHEMA_VERSION,
        "input_files": [dict(row) for row in input_files],
        "branch_decision_cards": branch_cards,
        "worlds": rows,
        "strict_uptake_rule": "later_direct_reply_or_quote_only",
        "decision_use": (
            "Locate unanswered, challenged or slowly answered governance objects, then "
            "choose evidence clarification, service follow-up or outreach for a new run."
        ),
        "causal_boundary": (
            "Direct uptake is a public temporal mechanism, not a causal estimate of policy "
            "effect on real students or population state."
        ),
        "comparability_boundary": (
            "The map includes both full-forum and semantic-spine worlds. Branch uptake "
            "counts with different LLM activation densities must not be ranked as policy "
            "effects without a matched-budget comparison."
        ),
        "privacy": {
            "raw_corpus_text_read": False,
            "source_ids_persisted": False,
            "private_actor_observations_persisted": False,
        },
    }
    body["decision_map_sha256"] = canonical_sha256(body)
    return body


def build_realism_benchmark(
    *,
    ecology: Mapping[str, Any],
    worlds: Iterable[PublicWorld],
    input_files: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    reference = ecology_reference(ecology)
    rows = [world_metrics(world) for world in worlds]
    if not rows:
        raise ForumRealismError("no comparable full-forum worlds were supplied")
    pooled_threads: Counter[str] = Counter()
    pooled_reply_functions: Counter[str] = Counter()
    interaction_counts: Counter[str] = Counter()
    reply_bearing_numerator = 0.0
    thread_total = 0
    for row in rows:
        pooled_threads.update(row["thread_size_counts"])
        weight = int(row["resident_message_count"])
        for key, share in row["activated_profile_reply_function_distribution"].items():
            pooled_reply_functions[key] += float(share) * weight
        interaction_counts.update(row["interaction_counts"])
        reply_bearing_numerator += row["reply_bearing_thread_share"] * row["thread_count"]
        thread_total += int(row["thread_count"])
    simulated_thread_distribution = _distribution(pooled_threads, THREAD_BUCKETS)
    simulated_reply_function_distribution = _distribution(
        pooled_reply_functions, REPLY_FUNCTIONS
    )
    simulated_reply_share = round(reply_bearing_numerator / thread_total, 6)
    thread_js = _js_divergence(
        reference["thread_size_distribution"], simulated_thread_distribution, THREAD_BUCKETS
    )
    profile_js = _js_divergence(
        reference["reply_function_distribution"],
        simulated_reply_function_distribution,
        REPLY_FUNCTIONS,
    )
    reference_reply_share = float(reference["reply_bearing_thread_share"])
    relative_reply_error = round(
        abs(simulated_reply_share - reference_reply_share) / reference_reply_share,
        6,
    )
    gates = {
        "thread_size_js_divergence_le_0_15": thread_js <= 0.15,
        "activated_profile_reply_function_js_le_0_15": profile_js <= 0.15,
        "reply_bearing_thread_relative_error_le_0_20": relative_reply_error <= 0.20,
    }
    gaps = []
    if simulated_thread_distribution["zero"] > reference["thread_size_distribution"]["zero"] * 1.2:
        gaps.append("too_many_root_only_threads")
    if simulated_thread_distribution["large"] > reference["thread_size_distribution"]["large"] * 1.2:
        gaps.append("hot_thread_feedback_loop_too_concentrated")
    if interaction_counts["repost"] == 0:
        gaps.append("no_repost_behavior_observed")
    if interaction_counts["report"] == 0:
        gaps.append("no_report_behavior_observed")
    body: dict[str, Any] = {
        "schema_version": REALISM_SCHEMA_VERSION,
        "status": "pass" if all(gates.values()) else "needs_calibration",
        "input_files": [dict(row) for row in input_files],
        "privacy": {
            "raw_text_read": False,
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "record_level_agent_ids_persisted": False,
        },
        "reference": reference,
        "simulation": {
            "world_count": len(rows),
            "thread_count": thread_total,
            "thread_size_distribution": simulated_thread_distribution,
            "reply_bearing_thread_share": simulated_reply_share,
            "activated_profile_reply_function_distribution": simulated_reply_function_distribution,
            "interaction_counts": {
                key: int(interaction_counts[key]) for key in INTERACTION_KINDS
            },
        },
        "comparisons": {
            "thread_size_js_divergence": thread_js,
            "activated_profile_reply_function_js_divergence": profile_js,
            "reply_bearing_thread_relative_error": relative_reply_error,
        },
        "gates": gates,
        "observed_gaps": gaps,
        "worlds": rows,
        "limitations": [
            "The source corpus has no aligned public like/repost/report benchmark in this release.",
            "Comment timestamps do not support reply-latency validation.",
            "Synthetic threads run for bounded ticks while corpus threads have heterogeneous lifetimes.",
            "Profile reply-function comparison measures activated composition, not a new semantic label of each message.",
        ],
    }
    body["benchmark_sha256"] = canonical_sha256(body)
    return body


def build_calibrated_realism_benchmark(
    *,
    ecology: Mapping[str, Any],
    worlds: Iterable[PublicWorld],
    input_files: Sequence[Mapping[str, str]] = (),
    baseline_benchmark: Mapping[str, Any] | None = None,
    mature_min_age_ticks: int = 2,
) -> dict[str, Any]:
    """Evaluate mature threads and, when supplied, pre/post calibration gains.

    The old v1 benchmark is intentionally retained.  This additive result
    separates two questions that v1 mixed together: whether a thread had time
    to mature, and whether the new mechanism materially improved the frozen
    pre-calibration run.
    """

    frozen_worlds = list(worlds)
    reference = ecology_reference(ecology)
    rows = [
        world_metrics(world, mature_min_age_ticks=mature_min_age_ticks)
        for world in frozen_worlds
    ]
    if not rows:
        raise ForumRealismError("no comparable worlds were supplied")
    mature_counts: Counter[str] = Counter()
    mature_thread_total = 0
    mature_reply_total = 0.0
    mature_concentration_total = 0.0
    for row in rows:
        mature_counts.update(row["mature_thread_size_counts"])
        count = int(row["mature_thread_count"])
        mature_thread_total += count
        mature_reply_total += float(row["mature_reply_bearing_thread_share"]) * count
        mature_concentration_total += (
            float(row["mature_top_10pct_thread_message_share"]) * count
        )
    mature_distribution = _distribution(mature_counts, THREAD_BUCKETS)
    mature_reply_share = round(
        mature_reply_total / mature_thread_total if mature_thread_total else 0.0,
        6,
    )
    mature_concentration = round(
        mature_concentration_total / mature_thread_total
        if mature_thread_total
        else 0.0,
        6,
    )
    mature_js = (
        _js_divergence(
            reference["thread_size_distribution"],
            mature_distribution,
            THREAD_BUCKETS,
        )
        if mature_thread_total
        else None
    )
    reference_reply = float(reference["reply_bearing_thread_share"])
    mature_reply_absolute_error = (
        round(abs(mature_reply_share - reference_reply), 6)
        if mature_thread_total
        else None
    )

    baseline = None
    improvement_gates: dict[str, bool | None] = {
        "zero_reply_share_reduced_by_0_15": None,
        "reply_bearing_share_increased_by_0_15": None,
        "thread_size_js_reduced_by_30_percent": None,
        "top10_concentration_reduced_by_20_percent": None,
    }
    if baseline_benchmark is not None:
        baseline_simulation = baseline_benchmark.get("simulation")
        baseline_comparisons = baseline_benchmark.get("comparisons")
        baseline_worlds = baseline_benchmark.get("worlds")
        if not isinstance(baseline_simulation, Mapping) or not isinstance(
            baseline_comparisons, Mapping
        ):
            raise ForumRealismError("baseline benchmark contract is invalid")
        baseline_distribution = baseline_simulation.get("thread_size_distribution")
        if not isinstance(baseline_distribution, Mapping):
            raise ForumRealismError("baseline thread distribution is missing")
        baseline_reply = float(baseline_simulation["reply_bearing_thread_share"])
        baseline_js = float(baseline_comparisons["thread_size_js_divergence"])
        baseline_concentration_values = [
            float(row.get("top_10pct_thread_message_share", 0.0))
            for row in baseline_worlds or ()
            if isinstance(row, Mapping)
        ]
        baseline_concentration = (
            sum(baseline_concentration_values) / len(baseline_concentration_values)
            if baseline_concentration_values
            else None
        )
        baseline = {
            "benchmark_sha256": baseline_benchmark.get("benchmark_sha256"),
            "zero_reply_share": float(baseline_distribution.get("zero", 0.0)),
            "reply_bearing_thread_share": baseline_reply,
            "thread_size_js_divergence": baseline_js,
            "mean_top_10pct_thread_message_share": (
                round(baseline_concentration, 6)
                if baseline_concentration is not None
                else None
            ),
        }
        improvement_gates = {
            "zero_reply_share_reduced_by_0_15": (
                baseline["zero_reply_share"] - mature_distribution["zero"] >= 0.15
            ),
            "reply_bearing_share_increased_by_0_15": (
                mature_reply_share - baseline_reply >= 0.15
            ),
            "thread_size_js_reduced_by_30_percent": (
                mature_js is not None
                and mature_js <= baseline_js * 0.70
            ),
            "top10_concentration_reduced_by_20_percent": (
                None
                if baseline_concentration is None
                else mature_concentration <= baseline_concentration * 0.80
            ),
        }

    absolute_gates = {
        "at_least_30_mature_threads": mature_thread_total >= 30,
        "mature_thread_size_js_le_0_25": (
            mature_js is not None and mature_js <= 0.25
        ),
        "mature_reply_bearing_absolute_error_le_0_20": (
            mature_reply_absolute_error is not None
            and mature_reply_absolute_error <= 0.20
        ),
    }
    applicable_improvements = [
        value for value in improvement_gates.values() if value is not None
    ]
    passed = all(absolute_gates.values()) and (
        all(applicable_improvements) if applicable_improvements else True
    )
    body: dict[str, Any] = {
        "schema_version": CALIBRATED_REALISM_SCHEMA_VERSION,
        "status": "pass" if passed else "needs_calibration",
        "input_files": [dict(row) for row in input_files],
        "reference": reference,
        "mature_simulation": {
            "world_count": len(rows),
            "thread_count": mature_thread_total,
            "thread_size_distribution": mature_distribution,
            "reply_bearing_thread_share": mature_reply_share,
            "mean_top_10pct_thread_message_share": mature_concentration,
            "definition": f"final_tick - created_tick >= {mature_min_age_ticks}",
        },
        "comparisons": {
            "mature_thread_size_js_divergence": mature_js,
            "mature_reply_bearing_absolute_error": mature_reply_absolute_error,
        },
        "pre_calibration_baseline": baseline,
        "absolute_gates": absolute_gates,
        "improvement_gates": improvement_gates,
        "worlds": rows,
        "privacy": {
            "raw_text_read": False,
            "source_ids_persisted": False,
            "record_level_agent_ids_persisted": False,
        },
        "interpretation": (
            "These are structural ecology checks over mature synthetic threads, not "
            "evidence that generated text or policy effects match real students."
        ),
    }
    body["benchmark_sha256"] = canonical_sha256(body)
    return body


def _bounded_score(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ForumRealismError(f"{field} must be a finite number in [0, 1]")
    return float(value)


def _normalized_entropy(distribution: Mapping[str, Any]) -> float:
    values = [max(0.0, float(value)) for value in distribution.values()]
    total = sum(values)
    positive = [value / total for value in values if value > 0 and total > 0]
    if len(positive) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in positive)
    return round(entropy / math.log(len(values)), 6) if len(values) > 1 else 0.0


def _persona_dimension(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "pending",
            "reason": "no independent reviewed or judge rows supplied",
            "gate": "run the small natural-profile audit before a paired policy run",
        }
    if len(rows) < 24:
        minimum_sample_passed = False
    else:
        minimum_sample_passed = True
    profile_match = []
    context_uptake = []
    unsupported = []
    template = []
    direction = []
    state_mae = []
    for row in rows:
        for key, target in (
            ("profile_match", profile_match),
            ("direct_context_uptake", context_uptake),
            ("unsupported_fact", unsupported),
            ("template_like", template),
        ):
            if not isinstance(row.get(key), bool):
                raise ForumRealismError(f"persona row {key} must be boolean")
            target.append(int(row[key]))
        if row.get("next_tick_direction_match") is not None:
            if not isinstance(row.get("next_tick_direction_match"), bool):
                raise ForumRealismError(
                    "next_tick_direction_match must be boolean or null"
                )
            direction.append(int(row["next_tick_direction_match"]))
        if row.get("next_tick_state_mae") is not None:
            state_mae.append(
                _bounded_score(row["next_tick_state_mae"], "next_tick_state_mae")
            )
    metrics = {
        "reviewed_turn_count": len(rows),
        "profile_match_rate": round(sum(profile_match) / len(rows), 6),
        "direct_context_uptake_rate": round(sum(context_uptake) / len(rows), 6),
        "unsupported_fact_rate": round(sum(unsupported) / len(rows), 6),
        "template_like_rate": round(sum(template) / len(rows), 6),
        "next_tick_direction_match_rate": (
            round(sum(direction) / len(direction), 6) if direction else None
        ),
        "next_tick_state_mae": (
            round(sum(state_mae) / len(state_mae), 6) if state_mae else None
        ),
    }
    gates = {
        "at_least_24_independently_reviewed_turns": minimum_sample_passed,
        "profile_match_rate_at_least_0_85": metrics["profile_match_rate"] >= 0.85,
        "direct_context_uptake_at_least_0_85": (
            metrics["direct_context_uptake_rate"] >= 0.85
        ),
        "unsupported_fact_rate_at_most_0_02": (
            metrics["unsupported_fact_rate"] <= 0.02
        ),
        "template_like_rate_at_most_0_05": metrics["template_like_rate"] <= 0.05,
        "next_tick_direction_match_at_least_0_75": (
            metrics["next_tick_direction_match_rate"] is not None
            and metrics["next_tick_direction_match_rate"] >= 0.75
        ),
        "next_tick_state_mae_at_most_0_15": (
            metrics["next_tick_state_mae"] is not None
            and metrics["next_tick_state_mae"] <= 0.15
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "needs_iteration",
        "metrics": metrics,
        "gates": gates,
        "measurement_rule": (
            "Profile/text is judged independently; particle alignment uses the next "
            "tick, never the same response that just updated the particles."
        ),
    }


def _heterogeneity_dimension(
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not frames:
        return {
            "status": "pending",
            "reason": "no aggregate parent-Agent state frames supplied",
            "gate": "aggregate private checkpoints without exporting Agent IDs",
        }
    normalized = []
    for frame in frames:
        branch = str(frame.get("branch") or "")
        tick = frame.get("tick")
        if not branch or not isinstance(tick, int) or isinstance(tick, bool):
            raise ForumRealismError("population frame branch/tick is invalid")
        stance = frame.get("stance_distribution")
        state_std = frame.get("weighted_parent_state_std")
        if not isinstance(stance, Mapping) or not isinstance(state_std, Mapping):
            raise ForumRealismError("population frame distributions are missing")
        stance_values = {
            str(key): _bounded_score(value, f"stance_distribution.{key}")
            for key, value in stance.items()
        }
        if not math.isclose(sum(stance_values.values()), 1.0, abs_tol=1e-5):
            raise ForumRealismError("stance distribution must sum to one")
        std_values = {
            str(key): _bounded_score(value, f"weighted_parent_state_std.{key}")
            for key, value in state_std.items()
        }
        gap = _bounded_score(
            frame.get("maximum_between_stake_group_gap", 0.0),
            "maximum_between_stake_group_gap",
        )
        raw_groups = frame.get("stake_groups")
        raw_groups = raw_groups if isinstance(raw_groups, Mapping) else {}
        stake_groups: dict[str, dict[str, Any]] = {}
        for group_name, group in sorted(raw_groups.items()):
            if not isinstance(group, Mapping):
                raise ForumRealismError("stake-group summary must be an object")
            parent_count = group.get("parent_agent_count")
            population_weight = group.get("population_weight")
            means = group.get("state_means")
            if (
                isinstance(parent_count, bool)
                or not isinstance(parent_count, int)
                or parent_count < 0
                or not isinstance(means, Mapping)
            ):
                raise ForumRealismError("stake-group summary is invalid")
            stake_groups[str(group_name)] = {
                "parent_agent_count": parent_count,
                "population_weight": _bounded_score(
                    population_weight, "stake_group.population_weight"
                ),
                "state_means": {
                    str(key): _bounded_score(
                        value, f"stake_group.state_means.{key}"
                    )
                    for key, value in means.items()
                },
            }
        normalized.append(
            {
                "branch": branch,
                "tick": tick,
                "stance_distribution": stance_values,
                "normalized_stance_entropy": _normalized_entropy(stance_values),
                "effective_stance_count": round(
                    math.exp(
                        _normalized_entropy(stance_values)
                        * math.log(max(1, len(stance_values)))
                    ),
                    6,
                ),
                "weighted_parent_state_std": std_values,
                "stake_groups": stake_groups,
                "maximum_between_stake_group_gap": gap,
                "maximum_gap_state": frame.get("maximum_gap_state"),
                "maximum_gap_groups": list(
                    frame.get("maximum_gap_groups") or ()
                ),
                "mean_within_agent_particle_std": (
                    _bounded_score(
                        frame["mean_within_agent_particle_std"],
                        "mean_within_agent_particle_std",
                    )
                    if frame.get("mean_within_agent_particle_std") is not None
                    else None
                ),
            }
        )
    by_branch: dict[str, list[dict[str, Any]]] = {}
    for frame in normalized:
        by_branch.setdefault(frame["branch"], []).append(frame)
    final_frames = []
    maximum_std_growth = 0.0
    for branch, branch_frames in sorted(by_branch.items()):
        ordered = sorted(branch_frames, key=lambda row: row["tick"])
        first, last = ordered[0], ordered[-1]
        shared_dimensions = set(first["weighted_parent_state_std"]) & set(
            last["weighted_parent_state_std"]
        )
        growth = max(
            (
                last["weighted_parent_state_std"][key]
                - first["weighted_parent_state_std"][key]
                for key in shared_dimensions
            ),
            default=0.0,
        )
        maximum_std_growth = max(maximum_std_growth, growth)
        final_frames.append(last)
    minimum_final_entropy = min(
        frame["normalized_stance_entropy"] for frame in final_frames
    )
    maximum_group_gap = max(
        frame["maximum_between_stake_group_gap"] for frame in normalized
    )
    gates = {
        "at_least_three_time_branch_frames": len(normalized) >= 3,
        "final_stance_entropy_at_least_0_55": minimum_final_entropy >= 0.55,
        "stake_group_gap_at_least_0_10": maximum_group_gap >= 0.10,
        "at_least_one_parent_state_std_grows_by_0_05": maximum_std_growth >= 0.05,
    }
    return {
        "status": "pass" if all(gates.values()) else "needs_iteration",
        "metrics": {
            "frame_count": len(normalized),
            "minimum_final_normalized_stance_entropy": round(
                minimum_final_entropy, 6
            ),
            "maximum_between_stake_group_gap": round(maximum_group_gap, 6),
            "maximum_parent_state_std_growth": round(maximum_std_growth, 6),
        },
        "gates": gates,
        "frames": normalized,
        "measurement_rule": (
            "Heterogeneity is measured across 1,000 weighted parent Agents. Ten "
            "particles per Agent are reported only as within-Agent uncertainty and "
            "are never counted as 10,000 independent people."
        ),
    }


def _governance_dimension(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "status": "pending",
            "reason": "no matched Natural/intervention mechanism summary supplied",
            "gate": "run both branches from one exact shared checkpoint",
        }
    natural = value.get("Natural")
    intervention = value.get("D")
    if not isinstance(natural, Mapping) or not isinstance(intervention, Mapping):
        raise ForumRealismError("governance summary requires Natural and D")
    metric_names = (
        "unsupported_claim_share",
        "correction_penetration",
        "service_closure_rate",
        "governance_observation_gap",
    )
    natural_values = {
        key: _bounded_score(natural.get(key), f"Natural.{key}")
        for key in metric_names
    }
    intervention_values = {
        key: _bounded_score(intervention.get(key), f"D.{key}")
        for key in metric_names
    }
    direct_uptake = _bounded_score(
        intervention.get("direct_governance_uptake_rate"),
        "D.direct_governance_uptake_rate",
    )
    lead_ticks = intervention.get("risk_warning_lead_ticks")
    if (
        isinstance(lead_ticks, bool)
        or not isinstance(lead_ticks, (int, float))
        or not math.isfinite(float(lead_ticks))
    ):
        raise ForumRealismError("D.risk_warning_lead_ticks must be finite")
    improvements = {
        "unsupported_claim_share_reduction": round(
            natural_values["unsupported_claim_share"]
            - intervention_values["unsupported_claim_share"],
            6,
        ),
        "correction_penetration_gain": round(
            intervention_values["correction_penetration"]
            - natural_values["correction_penetration"],
            6,
        ),
        "service_closure_gain": round(
            intervention_values["service_closure_rate"]
            - natural_values["service_closure_rate"],
            6,
        ),
        "observation_gap_reduction": round(
            natural_values["governance_observation_gap"]
            - intervention_values["governance_observation_gap"],
            6,
        ),
    }
    meaningful = [item >= 0.05 for item in improvements.values()]
    gates = {
        "same_shared_checkpoint": value.get("same_shared_checkpoint") is True,
        "hidden_truth_absent": value.get("hidden_truth_absent") is True,
        "resource_accounting_conserved": (
            intervention.get("resource_accounting_conserved") is True
        ),
        "strict_direct_governance_uptake_at_least_0_25": direct_uptake >= 0.25,
        "at_least_one_target_mechanism_improves_by_0_05": any(meaningful),
        "risk_signal_precedes_or_matches_public_peak": float(lead_ticks) >= 0.0,
    }
    return {
        "status": "pass" if all(gates.values()) else "needs_iteration",
        "Natural": natural_values,
        "D": {**intervention_values, "direct_governance_uptake_rate": direct_uptake},
        "paired_improvements": improvements,
        "risk_warning_lead_ticks": float(lead_ticks),
        "gates": gates,
        "measurement_rule": (
            "Governance succeeds only through a traceable message, correction, service "
            "or observation mechanism. Lower message volume or a prettier trust curve is "
            "not itself counted as success."
        ),
    }


def four_dimension_inputs_from_hero_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive public D3/D4 inputs from one paired Hero result.

    The function accepts either the domain result or the authorized runner's
    outer artifact.  It exports aggregate frames and mechanism rates only;
    resident IDs, private text and record-level relationship edges are not
    copied.
    """

    root = result.get("result") if isinstance(result.get("result"), Mapping) else result
    branches = root.get("branches") if isinstance(root, Mapping) else None
    if not isinstance(branches, Mapping):
        raise ForumRealismError("Hero result has no branch map")
    natural = branches.get("Natural")
    intervention = branches.get("D")
    if not isinstance(natural, Mapping) or not isinstance(intervention, Mapping):
        raise ForumRealismError("four-dimensional evidence requires Natural and D")

    population_frames = []
    for branch_name, branch in (("Natural", natural), ("D", intervention)):
        timeline = branch.get("timeline")
        if not isinstance(timeline, Sequence) or isinstance(timeline, (str, bytes)):
            raise ForumRealismError("Hero branch timeline is invalid")
        for row in timeline:
            if not isinstance(row, Mapping):
                continue
            frame = row.get("population_heterogeneity")
            if not isinstance(frame, Mapping):
                continue
            public_frame = {
                key: frame[key]
                for key in (
                    "branch",
                    "tick",
                    "parent_agent_count",
                    "evidence_subject_count",
                    "particle_count",
                    "particle_evidence_subject_count",
                    "stance_distribution",
                    "weighted_parent_state_std",
                    "stake_groups",
                    "maximum_between_stake_group_gap",
                    "maximum_gap_state",
                    "maximum_gap_groups",
                    "mean_within_agent_particle_std",
                )
                if key in frame
            }
            public_frame["branch"] = branch_name
            population_frames.append(public_frame)

    def branch_mechanisms(branch: Mapping[str, Any]) -> dict[str, float]:
        claims = branch.get("claims")
        claims = (
            [row for row in claims if isinstance(row, Mapping)]
            if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes))
            else []
        )
        active = [row for row in claims if str(row.get("status")) != "expired"]
        denominator = max(1, len(active))
        unsupported = sum(str(row.get("status")) == "unverified" for row in active)
        corrected = sum(
            str(row.get("status")) in {"corrected", "verified"}
            for row in active
        )
        mechanisms = branch.get("mechanism_chains")
        mechanisms = mechanisms if isinstance(mechanisms, Mapping) else {}
        help_chains = mechanisms.get("help_service_feedback_chains")
        help_chains = (
            [row for row in help_chains if isinstance(row, Mapping)]
            if isinstance(help_chains, Sequence)
            and not isinstance(help_chains, (str, bytes))
            else []
        )
        completed_help = sum(row.get("complete") is True for row in help_chains)
        timeline = branch.get("timeline")
        timeline = (
            [row for row in timeline if isinstance(row, Mapping)]
            if isinstance(timeline, Sequence) and not isinstance(timeline, (str, bytes))
            else []
        )
        observation_gap = 0.0
        for row in reversed(timeline):
            risk = row.get("governance_risk")
            if not isinstance(risk, Mapping):
                continue
            signals = risk.get("signals")
            if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
                continue
            match = next(
                (
                    signal
                    for signal in signals
                    if isinstance(signal, Mapping)
                    and signal.get("signal_name") == "governance_observation_gap"
                ),
                None,
            )
            if match is not None:
                observation_gap = _bounded_score(
                    match.get("value"), "governance_observation_gap"
                )
                break
        return {
            "unsupported_claim_share": round(unsupported / denominator, 6),
            "correction_penetration": round(corrected / denominator, 6),
            "service_closure_rate": round(
                completed_help / len(help_chains) if help_chains else 0.0, 6
            ),
            "governance_observation_gap": round(observation_gap, 6),
        }

    intervention_mechanisms = branch_mechanisms(intervention)
    uptake = intervention.get("governance_uptake_map")
    uptake = uptake if isinstance(uptake, Mapping) else {}
    uptake_summary = uptake.get("summary")
    uptake_summary = uptake_summary if isinstance(uptake_summary, Mapping) else {}
    direct_uptake = _bounded_score(
        uptake_summary.get("direct_uptake_rate", 0.0),
        "direct_governance_uptake_rate",
    )
    governance_rounds = intervention.get("governance_rounds")
    governance_rounds = (
        [row for row in governance_rounds if isinstance(row, Mapping)]
        if isinstance(governance_rounds, Sequence)
        and not isinstance(governance_rounds, (str, bytes))
        else []
    )
    resource_conserved = bool(governance_rounds) and all(
        isinstance(round_row.get("resource_before"), Mapping)
        and isinstance(round_row.get("resource_after"), Mapping)
        and round_row["resource_before"].get("conserved") is True
        and round_row["resource_after"].get("conserved") is True
        for round_row in governance_rounds
    )
    hidden_truth_absent = all(
        round_row.get("uses_hidden_truth") is False for round_row in governance_rounds
    ) and uptake.get("hidden_truth_used") is False

    intervention_timeline = [
        row
        for row in intervention.get("timeline", ())
        if isinstance(row, Mapping) and isinstance(row.get("tick"), int)
    ]
    peak_tick = (
        max(
            intervention_timeline,
            key=lambda row: (int(row.get("messages_this_tick", 0)), -int(row["tick"])),
        )["tick"]
        if intervention_timeline
        else 0
    )
    warning_ticks = []
    for row in intervention_timeline:
        risk = row.get("governance_risk")
        signals = risk.get("signals") if isinstance(risk, Mapping) else None
        if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
            continue
        if any(
            isinstance(signal, Mapping)
            and str(signal.get("severity")) in {"watch", "high"}
            for signal in signals
        ):
            warning_ticks.append(int(row["tick"]))
    warning_lead = (
        int(peak_tick) - min(warning_ticks)
        if warning_ticks
        else -1
    )
    same_shared_checkpoint = any(
        root.get(key) is True
        for key in (
            "worlds_share_exact_T3_checkpoint",
            "worlds_share_exact_T4_checkpoint",
            "shared_checkpoint_exact",
        )
    ) or all(
        isinstance(branch.get("shared_prefix_result_sha256"), str)
        and branch.get("shared_prefix_result_sha256")
        == intervention.get("shared_prefix_result_sha256")
        for branch in (natural, intervention)
    )
    governance_summary = {
        "same_shared_checkpoint": same_shared_checkpoint,
        "hidden_truth_absent": hidden_truth_absent,
        "Natural": branch_mechanisms(natural),
        "D": {
            **intervention_mechanisms,
            "direct_governance_uptake_rate": direct_uptake,
            "risk_warning_lead_ticks": warning_lead,
            "resource_accounting_conserved": resource_conserved,
        },
    }
    return {
        "population_frames": population_frames,
        "governance_summary": governance_summary,
        "privacy": {
            "record_level_agent_ids_persisted": False,
            "private_messages_persisted": False,
            "relationship_edges_persisted": False,
        },
    }


def build_four_dimension_fidelity_report(
    *,
    ecology_benchmark: Mapping[str, Any] | None,
    persona_rows: Sequence[Mapping[str, Any]] = (),
    population_frames: Sequence[Mapping[str, Any]] = (),
    governance_summary: Mapping[str, Any] | None = None,
    input_files: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Build one fail-closed result over the four ForumTwin evidence dimensions.

    Record-level review and checkpoint rows are accepted in memory but never
    copied to the returned public report. A dimension stays ``pending`` when
    its evidence was not supplied; absence is never converted into a zero or a
    fabricated pass.
    """

    if ecology_benchmark is None:
        ecology_dimension = {
            "status": "pending",
            "reason": "no mature-thread ecology benchmark supplied",
        }
    else:
        ecology_status = str(ecology_benchmark.get("status") or "unknown")
        ecology_dimension = {
            "status": "pass" if ecology_status == "pass" else "needs_iteration",
            "source_status": ecology_status,
            "mature_simulation": dict(
                ecology_benchmark.get("mature_simulation") or {}
            ),
            "comparisons": dict(ecology_benchmark.get("comparisons") or {}),
            "gates": {
                **dict(ecology_benchmark.get("absolute_gates") or {}),
                **dict(ecology_benchmark.get("improvement_gates") or {}),
            },
        }
    dimensions = {
        "forum_ecology": ecology_dimension,
        "persona_and_one_step_state": _persona_dimension(persona_rows),
        "population_heterogeneity": _heterogeneity_dimension(population_frames),
        "network_and_governance_response": _governance_dimension(
            governance_summary
        ),
    }
    measured = [
        value["status"] for value in dimensions.values() if value["status"] != "pending"
    ]
    overall_status = (
        "pending"
        if len(measured) < 4
        else "pass"
        if all(value == "pass" for value in measured)
        else "needs_iteration"
    )
    body: dict[str, Any] = {
        "schema_version": FOUR_DIMENSION_FIDELITY_SCHEMA_VERSION,
        "status": overall_status,
        "dimensions": dimensions,
        "input_files": [dict(value) for value in input_files],
        "single_run_rule": (
            "All four dimensions must be measured from the same scenario release, "
            "population, shared baseline and Natural/D branch pair."
        ),
        "privacy": {
            "record_level_review_rows_persisted": False,
            "record_level_agent_ids_persisted": False,
            "private_messages_persisted": False,
            "raw_corpus_text_read": False,
        },
        "non_claims": [
            "The result measures internal simulation fidelity, not real student opinion.",
            "Parent-Agent heterogeneity and within-Agent particle uncertainty are distinct.",
            "A governance branch is not assumed to win before the paired run.",
        ],
    }
    body["fidelity_report_sha256"] = canonical_sha256(body)
    return body
