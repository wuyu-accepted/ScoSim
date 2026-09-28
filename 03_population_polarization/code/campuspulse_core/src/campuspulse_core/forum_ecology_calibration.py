"""Privacy-preserving forum ecology calibration for ForumTwin.

The release contains aggregate continuation probabilities inferred from the
reviewed XiaoLaba thread ecology.  It is deliberately a *selection prior*:
it may change which item an activated resident opens, but it never creates a
message, forces a reply, changes an LLM response, or encodes a policy winner.

Two build modes are supported:

``global_aggregate_fallback``
    Uses the already-public v1 ecology counts.  This mode is available in the
    repository and makes no unobserved subgroup claim.

``time_held_out_episode_calibration``
    Uses privacy-local, conflict-free episode rows.  Macro-role continuation
    estimates are enabled only when a later-month holdout improves logarithmic
    loss over the global estimate.  No episode identifier or text is emitted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


ECOLOGY_CALIBRATION_SCHEMA = "campus-pulse-forum-ecology-calibration-v2"
ECOLOGY_RELEASE_SCHEMA = "campus-pulse-forum-ecology-release-v1"
THREAD_BUCKETS = ("zero", "small", "medium", "large")
CONTINUATION_STAGES = (
    "root_to_any_reply",
    "small_to_medium",
    "medium_to_large",
)
DEFAULT_PRIOR_STRENGTH = 24.0
MINIMUM_ATTENTION_WEIGHT = 0.05


class ForumEcologyCalibrationError(ValueError):
    """Raised when a calibration input or release violates its contract."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _thread_bucket(reply_count: int) -> str:
    if reply_count < 0:
        raise ForumEcologyCalibrationError("reply count cannot be negative")
    if reply_count == 0:
        return "zero"
    if reply_count <= 3:
        return "small"
    if reply_count <= 12:
        return "medium"
    return "large"


def _stage_trials(reply_count: int) -> dict[str, tuple[int, int]]:
    """Return (success, eligible) for the three observable continuation stages."""

    return {
        "root_to_any_reply": (int(reply_count >= 1), 1),
        "small_to_medium": (int(reply_count >= 4), int(reply_count >= 1)),
        "medium_to_large": (int(reply_count >= 13), int(reply_count >= 4)),
    }


def _posterior_mean(successes: int, trials: int) -> float:
    # Jeffreys' Beta(1/2, 1/2) prior keeps every probability strictly inside
    # (0, 1) without adding pseudo-observations large enough to move 131k rows.
    return (float(successes) + 0.5) / (float(trials) + 1.0)


def _stage_counts_from_buckets(
    bucket_counts: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    zero = int(bucket_counts.get("zero", 0))
    small = int(bucket_counts.get("small", 0))
    medium = int(bucket_counts.get("medium", 0))
    large = int(bucket_counts.get("large", 0))
    return {
        "root_to_any_reply": {
            "successes": small + medium + large,
            "trials": zero + small + medium + large,
        },
        "small_to_medium": {
            "successes": medium + large,
            "trials": small + medium + large,
        },
        "medium_to_large": {
            "successes": large,
            "trials": medium + large,
        },
    }


def _stage_release(counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in CONTINUATION_STAGES:
        successes = int(counts[stage]["successes"])
        trials = int(counts[stage]["trials"])
        if not 0 <= successes <= trials or trials <= 0:
            raise ForumEcologyCalibrationError(
                f"continuation stage {stage} has invalid counts"
            )
        result[stage] = {
            "successes": successes,
            "trials": trials,
            "posterior_mean": round(_posterior_mean(successes, trials), 8),
        }
    return result


def _validate_ecology(ecology: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    if ecology.get("schema_version") != ECOLOGY_RELEASE_SCHEMA:
        raise ForumEcologyCalibrationError("ecology v1 schema is unsupported")
    episode_count = int(ecology.get("episode_count", 0))
    raw = ecology.get("thread_size_bucket_counts")
    if episode_count <= 0 or not isinstance(raw, Mapping):
        raise ForumEcologyCalibrationError("ecology release has no thread counts")
    counts = {key: int(raw.get(key, 0)) for key in THREAD_BUCKETS}
    if any(value < 0 for value in counts.values()) or sum(counts.values()) != episode_count:
        raise ForumEcologyCalibrationError("ecology thread counts do not conserve population")
    return episode_count, counts


def _episode_rows(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ForumEcologyCalibrationError("episode calibration row is not an object")
        reply_count = int(episode.get("comment_count", 0) or 0)
        if reply_count < 0:
            raise ForumEcologyCalibrationError("episode comment_count cannot be negative")
        month = str(episode.get("temporal_bucket") or "")
        macro_role = str(episode.get("primary_macro_role") or "")
        if not month or not macro_role:
            raise ForumEcologyCalibrationError(
                "episode calibration requires temporal_bucket and primary_macro_role"
            )
        rows.append(
            {
                "reply_count": reply_count,
                "month": month,
                "macro_role": macro_role,
            }
        )
    if not rows:
        raise ForumEcologyCalibrationError("episode calibration input is empty")
    return rows


def _counts_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        stage: {"successes": 0, "trials": 0} for stage in CONTINUATION_STAGES
    }
    for row in rows:
        for stage, (success, eligible) in _stage_trials(int(row["reply_count"])).items():
            counts[stage]["successes"] += success
            counts[stage]["trials"] += eligible
    return counts


def _partial_pool(
    *,
    global_probability: float,
    successes: int,
    trials: int,
    prior_strength: float,
) -> float:
    return (
        float(successes) + prior_strength * global_probability
    ) / (float(trials) + prior_strength)


def _held_out_log_loss(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_probabilities: Mapping[str, float],
    role_probabilities: Mapping[str, Mapping[str, float]],
) -> tuple[float, float, int]:
    global_loss = 0.0
    conditional_loss = 0.0
    trials = 0
    for row in rows:
        macro_role = str(row["macro_role"])
        role = role_probabilities.get(macro_role, {})
        for stage, (success, eligible) in _stage_trials(int(row["reply_count"])).items():
            if not eligible:
                continue
            target = float(success)
            global_p = min(1.0 - 1e-9, max(1e-9, global_probabilities[stage]))
            role_p = min(1.0 - 1e-9, max(1e-9, float(role.get(stage, global_p))))
            global_loss -= target * math.log(global_p) + (1.0 - target) * math.log(
                1.0 - global_p
            )
            conditional_loss -= target * math.log(role_p) + (1.0 - target) * math.log(
                1.0 - role_p
            )
            trials += 1
    if not trials:
        return 0.0, 0.0, 0
    return global_loss / trials, conditional_loss / trials, trials


def build_forum_ecology_calibration(
    *,
    ecology: Mapping[str, Any],
    source_ecology_file_sha256: str,
    episodes: Iterable[Mapping[str, Any]] | None = None,
    source_episode_file_sha256: str | None = None,
    training_months: Sequence[str] = ("2026-05", "2026-06"),
    holdout_months: Sequence[str] = ("2026-07",),
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict[str, Any]:
    """Build a hash-bound aggregate calibration release.

    Full episode rows are optional because the public repository intentionally
    does not ship the private 172 MB episode pool.  The fallback is useful and
    honest: it calibrates only the three globally identifiable continuation
    stages and records that no subgroup/time-holdout claim was possible.
    """

    if len(source_ecology_file_sha256) != 64:
        raise ForumEcologyCalibrationError("source ecology SHA-256 is invalid")
    if prior_strength <= 0:
        raise ForumEcologyCalibrationError("prior strength must be positive")
    episode_count, bucket_counts = _validate_ecology(ecology)
    aggregate_counts = _stage_counts_from_buckets(bucket_counts)
    global_stages = _stage_release(aggregate_counts)
    mode = "global_aggregate_fallback"
    macro_release: dict[str, Any] = {}
    time_holdout: dict[str, Any] = {
        "available": False,
        "reason": "private conflict-free episode rows were not supplied",
        "conditional_macro_probabilities_enabled": False,
    }
    episode_source = None

    if episodes is not None:
        if not source_episode_file_sha256 or len(source_episode_file_sha256) != 64:
            raise ForumEcologyCalibrationError("source episode SHA-256 is invalid")
        rows = _episode_rows(episodes)
        if len(rows) != episode_count:
            raise ForumEcologyCalibrationError(
                "episode rows do not match the frozen ecology population"
            )
        derived_buckets = Counter(
            _thread_bucket(int(row["reply_count"])) for row in rows
        )
        if any(int(derived_buckets[key]) != bucket_counts[key] for key in THREAD_BUCKETS):
            raise ForumEcologyCalibrationError(
                "episode rows do not reproduce frozen ecology thread buckets"
            )
        training_set = set(str(value) for value in training_months)
        holdout_set = set(str(value) for value in holdout_months)
        if not training_set or not holdout_set or training_set & holdout_set:
            raise ForumEcologyCalibrationError("training/holdout months are invalid")
        train_rows = [row for row in rows if row["month"] in training_set]
        holdout_rows = [row for row in rows if row["month"] in holdout_set]
        if not train_rows or not holdout_rows:
            raise ForumEcologyCalibrationError(
                "episode rows do not cover the frozen training and holdout months"
            )
        training_counts = _counts_for_rows(train_rows)
        training_global = _stage_release(training_counts)
        global_probabilities = {
            stage: float(training_global[stage]["posterior_mean"])
            for stage in CONTINUATION_STAGES
        }
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train_rows:
            by_role[str(row["macro_role"])].append(row)
        role_probabilities: dict[str, dict[str, float]] = {}
        for macro_role, role_rows in sorted(by_role.items()):
            counts = _counts_for_rows(role_rows)
            probabilities = {}
            stages = {}
            for stage in CONTINUATION_STAGES:
                successes = int(counts[stage]["successes"])
                trials = int(counts[stage]["trials"])
                probability = _partial_pool(
                    global_probability=global_probabilities[stage],
                    successes=successes,
                    trials=trials,
                    prior_strength=prior_strength,
                )
                probabilities[stage] = probability
                stages[stage] = {
                    "successes": successes,
                    "trials": trials,
                    "posterior_mean": round(probability, 8),
                }
            role_probabilities[macro_role] = probabilities
            macro_release[macro_role] = {
                "training_thread_count": len(role_rows),
                "continuation_stages": stages,
            }
        global_loss, conditional_loss, holdout_trials = _held_out_log_loss(
            holdout_rows,
            global_probabilities=global_probabilities,
            role_probabilities=role_probabilities,
        )
        conditional_enabled = bool(
            holdout_trials > 0 and conditional_loss < global_loss - 1e-12
        )
        mode = "time_held_out_episode_calibration"
        global_stages = training_global
        time_holdout = {
            "available": True,
            "training_months": sorted(training_set),
            "holdout_months": sorted(holdout_set),
            "training_thread_count": len(train_rows),
            "holdout_thread_count": len(holdout_rows),
            "eligible_stage_trials": holdout_trials,
            "global_log_loss": round(global_loss, 8),
            "macro_partial_pooling_log_loss": round(conditional_loss, 8),
            "relative_log_loss_improvement": round(
                (global_loss - conditional_loss) / global_loss if global_loss else 0.0,
                8,
            ),
            "conditional_macro_probabilities_enabled": conditional_enabled,
            "enablement_rule": "later-month log loss must strictly improve global calibration",
        }
        episode_source = source_episode_file_sha256

    body: dict[str, Any] = {
        "schema_version": ECOLOGY_CALIBRATION_SCHEMA,
        "calibration_mode": mode,
        "source": {
            "ecology_schema": ECOLOGY_RELEASE_SCHEMA,
            "ecology_release_sha256": str(
                ecology.get("ecology_release_sha256") or ""
            ),
            "ecology_file_sha256": source_ecology_file_sha256,
            "episode_file_sha256": episode_source,
            "thread_count": episode_count,
        },
        "thread_size_bucket_counts": bucket_counts,
        "thread_size_distribution": {
            key: round(bucket_counts[key] / episode_count, 8)
            for key in THREAD_BUCKETS
        },
        "global_continuation_stages": global_stages,
        "macro_role_partial_pooling": macro_release,
        "time_holdout": time_holdout,
        "selection_contract": {
            "use": "soft deterministic attention ordering inside the shared public board",
            "public_hot_latest_ranking_changed": False,
            "llm_action_forced": False,
            "llm_text_generated": False,
            "policy_or_branch_label_used": False,
            "minimum_attention_weight": MINIMUM_ATTENTION_WEIGHT,
            "large_thread_rule": (
                "large continuation is not identifiable from aggregate bins; use the "
                "strictly-positive exploration floor instead of inventing a target"
            ),
        },
        "method": {
            "global_prior": "Jeffreys Beta(0.5, 0.5)",
            "macro_partial_pooling_prior_strength": prior_strength,
            "stages": {
                "root_to_any_reply": "P(final replies >= 1)",
                "small_to_medium": "P(final replies >= 4 | final replies >= 1)",
                "medium_to_large": "P(final replies >= 13 | final replies >= 4)",
            },
        },
        "privacy": {
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "episode_ids_persisted": False,
            "record_level_rows_persisted": False,
        },
    }
    body["calibration_release_sha256"] = _canonical_sha256(body)
    validate_forum_ecology_calibration(body)
    return body


def validate_forum_ecology_calibration(release: Mapping[str, Any]) -> None:
    if release.get("schema_version") != ECOLOGY_CALIBRATION_SCHEMA:
        raise ForumEcologyCalibrationError("calibration schema is unsupported")
    supplied_hash = str(release.get("calibration_release_sha256") or "")
    unsigned = dict(release)
    unsigned.pop("calibration_release_sha256", None)
    if supplied_hash != _canonical_sha256(unsigned):
        raise ForumEcologyCalibrationError("calibration release hash mismatch")
    stages = release.get("global_continuation_stages")
    if not isinstance(stages, Mapping) or set(stages) != set(CONTINUATION_STAGES):
        raise ForumEcologyCalibrationError("calibration stages are incomplete")
    for stage in CONTINUATION_STAGES:
        probability = float(stages[stage]["posterior_mean"])
        if not 0.0 < probability < 1.0:
            raise ForumEcologyCalibrationError("continuation probability is not positive")
    privacy = release.get("privacy")
    if not isinstance(privacy, Mapping) or any(
        privacy.get(key) is not False
        for key in (
            "raw_text_persisted",
            "source_ids_persisted",
            "episode_ids_persisted",
            "record_level_rows_persisted",
        )
    ):
        raise ForumEcologyCalibrationError("calibration privacy contract drifted")


def continuation_probability(
    release: Mapping[str, Any],
    *,
    current_reply_count: int,
    source_macro_role: str | None = None,
) -> float:
    """Return a strictly positive probability-like attention weight."""

    validate_forum_ecology_calibration(release)
    if current_reply_count < 0:
        raise ForumEcologyCalibrationError("current reply count cannot be negative")
    if current_reply_count == 0:
        stage = "root_to_any_reply"
    elif current_reply_count <= 3:
        stage = "small_to_medium"
    elif current_reply_count <= 12:
        stage = "medium_to_large"
    else:
        return MINIMUM_ATTENTION_WEIGHT
    global_probability = float(
        release["global_continuation_stages"][stage]["posterior_mean"]
    )
    holdout = release.get("time_holdout")
    if (
        source_macro_role
        and isinstance(holdout, Mapping)
        and holdout.get("conditional_macro_probabilities_enabled") is True
    ):
        role = release.get("macro_role_partial_pooling", {}).get(source_macro_role)
        if isinstance(role, Mapping):
            role_stage = role.get("continuation_stages", {}).get(stage)
            if isinstance(role_stage, Mapping):
                global_probability = float(role_stage["posterior_mean"])
    return max(MINIMUM_ATTENTION_WEIGHT, min(1.0, global_probability))

