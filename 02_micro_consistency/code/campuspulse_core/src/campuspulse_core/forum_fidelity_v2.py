"""High-dimensional, privacy-safe fidelity evaluation for ForumTwin.

The public ForumTwin result intentionally contains only aggregates.  This
module consumes a local exact-cache replay, computes the statistics requested
by the competition report, and returns sufficient statistics only.  The
10,000 particle rows remain a private evaluation artifact: the unit of social
analysis is always the 1,000 parent Agents.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .governance_arena_v2 import STATE_FIELDS


FIDELITY_V2_SCHEMA = "campus-pulse-fidelity-evaluation-v3"
PRIVATE_PARTICLE_SNAPSHOT_SCHEMA = (
    "campus-pulse-private-particle-evaluation-snapshot-v1"
)
PUBLIC_ACTIONS = frozenset(
    {
        "post",
        "reply",
        "quote",
        "repost",
        "seek_verify",
        "correct",
        "seek_help",
        "report_risk",
    }
)


class ForumFidelityV2Error(ValueError):
    """Raised when private replay evidence violates the evaluation contract."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if tuple(result.shape) != shape or not np.all(np.isfinite(result)):
        raise ForumFidelityV2Error(f"{label} has an invalid shape or value")
    return result


class PrivateParticleFidelityCollector:
    """Capture selected full-particle frames and immediate LLM assimilation.

    This object must only be written below ``_run``.  It contains parent
    indices and complete particle state and is therefore never a public API or
    release type.
    """

    def __init__(self, *, selected_ticks: Sequence[int] = (3, 5, 10)) -> None:
        ticks = tuple(sorted({int(value) for value in selected_ticks}))
        if not ticks or ticks[0] < 0:
            raise ForumFidelityV2Error("selected particle ticks are invalid")
        self._selected_ticks = ticks
        self._snapshots: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._assimilations: list[dict[str, Any]] = []

    def record_initial(self, *, state: Any) -> None:
        self._capture(branch="shared_initial", tick=0, state=state)

    def record_checkpoint(
        self, *, branch: str, tick: int, snapshot: Mapping[str, Any]
    ) -> None:
        if int(tick) not in self._selected_ticks:
            return
        state = snapshot.get("branch_state")
        if state is None:
            raise ForumFidelityV2Error("particle checkpoint lacks branch_state")
        self._capture(branch=str(branch), tick=int(tick), state=state)

    def _capture(self, *, branch: str, tick: int, state: Any) -> None:
        ensemble = getattr(state, "ensemble", None)
        values = getattr(ensemble, "values", None)
        weights = getattr(ensemble, "weights", None)
        parent_weights = getattr(ensemble, "parent_weights", None)
        if values is None or weights is None or parent_weights is None:
            raise ForumFidelityV2Error("particle state is incomplete")
        values_array = _finite_array(
            values,
            shape=(1_000, 10, len(STATE_FIELDS)),
            label="particle values",
        )
        weights_array = _finite_array(
            weights, shape=(1_000, 10), label="particle weights"
        )
        parent_array = _finite_array(
            parent_weights, shape=(1_000,), label="parent weights"
        )
        if np.min(weights_array) < 0 or np.min(parent_array) <= 0:
            raise ForumFidelityV2Error("particle weights are not positive")
        if not np.allclose(weights_array.sum(axis=1), parent_array, atol=1e-8):
            raise ForumFidelityV2Error("within-parent particle mass drifted")
        if not math.isclose(float(parent_array.sum()), 1.0, abs_tol=1e-8):
            raise ForumFidelityV2Error("parent design weight mass drifted")
        self._snapshots[(branch, tick)] = {
            "values": values_array.astype(np.float32, copy=True),
            "weights": weights_array.astype(np.float64, copy=True),
            "parent_weights": parent_array.astype(np.float64, copy=True),
        }

    def record_assimilation(self, row: Mapping[str, Any]) -> None:
        required = {
            "branch",
            "tick",
            "parent_index",
            "prompt_sha256",
            "response_sha256",
            "innovation_id",
            "before_state",
            "after_state",
            "resampled",
        }
        if not required <= set(row):
            raise ForumFidelityV2Error("assimilation audit row is incomplete")
        before = row["before_state"]
        after = row["after_state"]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ForumFidelityV2Error("assimilation state is invalid")
        if set(before) != set(STATE_FIELDS) or set(after) != set(STATE_FIELDS):
            raise ForumFidelityV2Error("assimilation state fields drifted")
        self._assimilations.append(
            {
                "branch": str(row["branch"]),
                "tick": int(row["tick"]),
                "parent_index": int(row["parent_index"]),
                "prompt_sha256": str(row["prompt_sha256"]),
                "response_sha256": str(row["response_sha256"]),
                "innovation_id": str(row["innovation_id"]),
                "before_state": {
                    field: float(before[field]) for field in STATE_FIELDS
                },
                "after_state": {
                    field: float(after[field]) for field in STATE_FIELDS
                },
                "resampled": bool(row["resampled"]),
            }
        )

    @property
    def snapshots(self) -> Mapping[tuple[str, int], Mapping[str, np.ndarray]]:
        return self._snapshots

    @property
    def assimilations(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._assimilations)

    def write_private_npz(self, path: Path) -> dict[str, Any]:
        if not self._snapshots:
            raise ForumFidelityV2Error("no particle snapshots were captured")
        destination = path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "state_fields": np.asarray(STATE_FIELDS, dtype="U32"),
        }
        frames = []
        for ordinal, ((branch, tick), snapshot) in enumerate(
            sorted(self._snapshots.items(), key=lambda item: (item[0][1], item[0][0]))
        ):
            prefix = f"frame_{ordinal:02d}"
            arrays[f"{prefix}_values"] = snapshot["values"]
            arrays[f"{prefix}_weights"] = snapshot["weights"]
            arrays[f"{prefix}_parent_weights"] = snapshot["parent_weights"]
            frames.append(
                {
                    "prefix": prefix,
                    "branch": branch,
                    "tick": tick,
                    "parent_count": 1_000,
                    "particles_per_parent": 10,
                }
            )
        arrays["assimilation_parent_index"] = np.asarray(
            [row["parent_index"] for row in self._assimilations], dtype=np.int32
        )
        arrays["assimilation_tick"] = np.asarray(
            [row["tick"] for row in self._assimilations], dtype=np.int16
        )
        arrays["assimilation_branch"] = np.asarray(
            [row["branch"] for row in self._assimilations], dtype="U24"
        )
        arrays["assimilation_prompt_sha256"] = np.asarray(
            [row["prompt_sha256"] for row in self._assimilations], dtype="U64"
        )
        arrays["assimilation_response_sha256"] = np.asarray(
            [row["response_sha256"] for row in self._assimilations], dtype="U64"
        )
        arrays["assimilation_before"] = np.asarray(
            [
                [row["before_state"][field] for field in STATE_FIELDS]
                for row in self._assimilations
            ],
            dtype=np.float32,
        )
        arrays["assimilation_after"] = np.asarray(
            [
                [row["after_state"][field] for field in STATE_FIELDS]
                for row in self._assimilations
            ],
            dtype=np.float32,
        )
        arrays["assimilation_resampled"] = np.asarray(
            [row["resampled"] for row in self._assimilations], dtype=np.bool_
        )
        metadata = {
            "schema_version": PRIVATE_PARTICLE_SNAPSHOT_SCHEMA,
            "frames": frames,
            "state_fields": list(STATE_FIELDS),
            "assimilation_count": len(self._assimilations),
            "contains_parent_particle_mapping": True,
            "public_export_allowed": False,
        }
        arrays["metadata_json"] = np.asarray(_canonical(metadata), dtype="U")
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=destination.name + ".", suffix=".tmp",
            dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
        return {
            **metadata,
            "file_name": destination.name,
            "file_size_bytes": destination.stat().st_size,
            "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }


def _text_from_visible_context(row: Mapping[str, Any]) -> str:
    values = row.get("visible_context")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ""
    parts = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        parts.extend(
            (
                str(value.get("relation") or ""),
                str(value.get("visible_text") or value.get("text") or ""),
            )
        )
    return "\n".join(part for part in parts if part)


def _profile_text(row: Mapping[str, Any]) -> str:
    context = row.get("resident_context")
    context = context if isinstance(context, Mapping) else {}
    return "\n".join(
        str(context.get(key) or "")
        for key in (
            "forum_identity",
            "reviewed_history",
            "usual_participation",
            "communication_style",
            "handling_uncertainty",
            "evidence_basis",
            "typical_thread_entry",
        )
    ) + "\n" + str(row.get("current_situation") or "")


def _hash_features(text: str, *, dimension: int = 128) -> np.ndarray:
    result = np.zeros(dimension, dtype=np.float64)
    compact = "".join(str(text).lower().split())
    if not compact:
        return result
    for size in (1, 2, 3):
        for index in range(max(0, len(compact) - size + 1)):
            token = compact[index : index + size]
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            code = int.from_bytes(digest, "big")
            result[code % dimension] += -1.0 if code & 1 else 1.0
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 0 else result


def _decision(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("decision")
    if not isinstance(value, Mapping):
        raise ForumFidelityV2Error("persona evidence lacks a decision")
    return value


def _action_family(decision: Mapping[str, Any]) -> str:
    action = str(decision.get("action") or "read")
    if action in {"read", "ignore"}:
        return "silent_attention"
    if action in {"seek_help", "report_risk"}:
        return "service_or_risk"
    if action in {"seek_verify", "correct"}:
        return "verification"
    return "public_discussion"


def _stance_family(decision: Mapping[str, Any]) -> str:
    stance = str(decision.get("stance") or "uncertain")
    if stance in {"support", "supportive"}:
        return "support"
    if stance in {"oppose", "critical", "concerned"}:
        return "oppose_or_concern"
    return "uncertain"


def _contract_stance(decision: Mapping[str, Any]) -> str:
    """Return the five-way stance carried by the resident-turn contract.

    Predictive models above intentionally pool sparse categories, but population
    expression must not relabel ``mixed`` or ``neutral`` as ``uncertain``.  The
    former is an ambivalent position and the latter is the expected bookkeeping
    value for many silent turns; neither means epistemic uncertainty.
    """

    stance = str(decision.get("stance") or "uncertain")
    aliases = {
        "supportive": "support",
        "critical": "oppose",
        "concerned": "oppose",
    }
    stance = aliases.get(stance, stance)
    return stance if stance in {"support", "oppose", "mixed", "uncertain", "neutral"} else "uncertain"


def _evidence_family(decision: Mapping[str, Any]) -> str:
    status = str(decision.get("evidence_status") or "unverified")
    if status in {"verified", "corrected"}:
        return "verified_or_corrected"
    if status == "contested":
        return "contested"
    return "unverified_or_unspecified"


def _outcome_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result = {
        "action_family": [],
        "stance_family": [],
        "evidence_family": [],
        "public_expression": [],
        "private_action": [],
    }
    for row in rows:
        decision = _decision(row)
        result["action_family"].append(_action_family(decision))
        result["stance_family"].append(_stance_family(decision))
        result["evidence_family"].append(_evidence_family(decision))
        result["public_expression"].append(
            "yes" if str(decision.get("action") or "read") in PUBLIC_ACTIONS else "no"
        )
        private_actions = decision.get("private_actions")
        result["private_action"].append(
            "yes"
            if isinstance(private_actions, Sequence)
            and not isinstance(private_actions, (str, bytes))
            and len(private_actions) > 0
            else "no"
        )
    return result


STRUCTURED_BEHAVIOR_FIELDS = (
    "public_expression",
    "private_expression",
    "urgency",
    "confidence",
    "trust_change",
    "satisfaction_change",
    "support_signal",
    "opposition_signal",
    "verification_signal",
    "service_or_risk_signal",
    "evidence_quality",
    "expression_length",
)


def _bounded_number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(np.clip(value, 0.0, 1.0))


def _structured_behavior_vector(decision: Mapping[str, Any]) -> np.ndarray:
    action = str(decision.get("action") or "read")
    stance = _stance_family(decision)
    evidence = _evidence_family(decision)
    private_actions = decision.get("private_actions")
    has_private = (
        isinstance(private_actions, Sequence)
        and not isinstance(private_actions, (str, bytes))
        and len(private_actions) > 0
    )
    text_length = len(str(decision.get("visible_text") or "").strip())
    trust_delta = float(decision.get("trust_delta") or 0.0)
    satisfaction_delta = float(decision.get("satisfaction_delta") or 0.0)
    return np.asarray(
        (
            float(action in PUBLIC_ACTIONS),
            float(has_private),
            _bounded_number(decision.get("urgency")),
            _bounded_number(decision.get("confidence")),
            float(np.clip(0.5 + trust_delta / 0.4, 0.0, 1.0)),
            float(np.clip(0.5 + satisfaction_delta / 0.4, 0.0, 1.0)),
            float(stance == "support"),
            float(stance == "oppose_or_concern"),
            float(action in {"seek_verify", "correct"}),
            float(action in {"seek_help", "report_risk"}),
            1.0
            if evidence == "verified_or_corrected"
            else 0.5
            if evidence == "contested"
            else 0.0,
            float(np.clip(text_length / 120.0, 0.0, 1.0)),
        ),
        dtype=np.float64,
    )


def _rank_vector(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks without adding a scipy dependency."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _rank_vector(left)
    right_rank = _rank_vector(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 1e-12:
        return None
    return float(left_centered @ right_centered / denominator)


def _ridge_continuous_group_cv(
    *, matrix: np.ndarray, targets: np.ndarray, groups: Sequence[str], alpha: float = 3.0
) -> np.ndarray:
    if matrix.shape[0] != len(groups) or targets.shape[0] != len(groups):
        raise ForumFidelityV2Error("continuous grouped evidence has mismatched rows")
    folds = _stable_group_folds(groups)
    predictions = np.zeros_like(targets, dtype=np.float64)
    design = np.column_stack((np.ones(len(groups)), matrix))
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        if not np.any(test):
            raise ForumFidelityV2Error("continuous grouped folds are degenerate")
        mean = design[train, 1:].mean(axis=0)
        scale = design[train, 1:].std(axis=0)
        scale[scale < 1e-9] = 1.0
        train_design = np.column_stack(
            (np.ones(int(train.sum())), (design[train, 1:] - mean) / scale)
        )
        test_design = np.column_stack(
            (np.ones(int(test.sum())), (design[test, 1:] - mean) / scale)
        )
        xtx = train_design.T @ train_design
        penalty = np.eye(xtx.shape[0], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            xtx + penalty, train_design.T @ targets[train]
        )
        predictions[test] = np.clip(test_design @ coefficients, 0.0, 1.0)
    return predictions


def _structured_vector_alignment(
    *,
    model_matrices: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    targets = np.vstack([_structured_behavior_vector(_decision(row)) for row in rows])
    predictions = {
        name: _ridge_continuous_group_cv(
            matrix=matrix, targets=targets, groups=groups, alpha=alpha
        )
        for name, matrix in model_matrices.items()
    }
    dimensions = {}
    profile_improvements = []
    state_improvements = []
    joint_improvements = []
    for index, name in enumerate(STRUCTURED_BEHAVIOR_FIELDS):
        actual = targets[:, index]
        rmses = {
            model: float(math.sqrt(np.mean((actual - value[:, index]) ** 2)))
            for model, value in predictions.items()
        }
        profile_relative = (
            rmses["M2_context_previous_particle_state"]
            - rmses["M3_context_profile_previous_particle_state"]
        ) / max(rmses["M2_context_previous_particle_state"], 1e-12)
        state_relative = (
            rmses["M1_context_profile"]
            - rmses["M3_context_profile_previous_particle_state"]
        ) / max(rmses["M1_context_profile"], 1e-12)
        joint_relative = (
            rmses["M0_context"]
            - rmses["M3_context_profile_previous_particle_state"]
        ) / max(rmses["M0_context"], 1e-12)
        profile_improvements.append(profile_relative)
        state_improvements.append(state_relative)
        joint_improvements.append(joint_relative)
        total = float(np.sum((actual - actual.mean()) ** 2))
        r_squared = (
            1.0
            - float(
                np.sum(
                    (
                        actual
                        - predictions["M3_context_profile_previous_particle_state"][
                            :, index
                        ]
                    )
                    ** 2
                )
            )
            / total
            if total > 1e-12
            else None
        )
        rho = _spearman(
            actual,
            predictions["M3_context_profile_previous_particle_state"][:, index],
        )
        dimensions[name] = {
            "grouped_cv_rmse": {
                model: round(value, 6) for model, value in rmses.items()
            },
            "profile_increment_given_previous_state": round(profile_relative, 6),
            "previous_state_increment_given_profile": round(state_relative, 6),
            "joint_increment_over_context": round(joint_relative, 6),
            "context_only_rmse": round(rmses["M0_context"], 6),
            "context_plus_profile_and_stake_rmse": round(
                rmses["M3_context_profile_previous_particle_state"], 6
            ),
            "relative_rmse_improvement": round(joint_relative, 6),
            "grouped_cv_r_squared": None if r_squared is None else round(r_squared, 6),
            "grouped_cv_spearman_rho": None if rho is None else round(rho, 6),
        }
    row_cosines = []
    for actual, predicted in zip(
        targets, predictions["M3_context_profile_previous_particle_state"]
    ):
        denominator = float(np.linalg.norm(actual) * np.linalg.norm(predicted))
        if denominator > 1e-12:
            row_cosines.append(float(actual @ predicted / denominator))
    return {
        "method": (
            "The validated turn contract is mapped to a 12-dimensional structured "
            "behavior vector. The same M0/M1/M2/M3 comparison separates visible "
            "context, stable Profile/stake and the exact previous-Tick particle state. "
            "No Agent appears in both train and test."
        ),
        "ridge_alpha": alpha,
        "fields": list(STRUCTURED_BEHAVIOR_FIELDS),
        "dimensions": dimensions,
        "mean_profile_increment_given_previous_state": round(
            float(np.mean(profile_improvements)), 6
        ),
        "mean_previous_state_increment_given_profile": round(
            float(np.mean(state_improvements)), 6
        ),
        "mean_joint_increment_over_context": round(
            float(np.mean(joint_improvements)), 6
        ),
        "mean_relative_rmse_improvement": round(
            float(np.mean(joint_improvements)), 6
        ),
        "mean_out_of_fold_cosine_similarity": round(
            float(np.mean(row_cosines)), 6
        ),
    }


def _stable_group_folds(groups: Sequence[str], *, fold_count: int = 5) -> np.ndarray:
    unique = sorted(
        set(groups), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    assignment = {value: index % fold_count for index, value in enumerate(unique)}
    return np.asarray([assignment[value] for value in groups], dtype=np.int16)


def _ridge_multitask_cv(
    *,
    matrix: np.ndarray,
    outcomes: Mapping[str, Sequence[str]],
    groups: Sequence[str],
    alpha: float = 3.0,
) -> dict[str, Any]:
    row_count = len(groups)
    if matrix.shape[0] != row_count or row_count < 30:
        raise ForumFidelityV2Error("grouped predictive evidence is too small")
    folds = _stable_group_folds(groups)
    classes = {name: sorted(set(values)) for name, values in outcomes.items()}
    offsets: dict[str, tuple[int, int]] = {}
    column = 0
    for name in sorted(classes):
        offsets[name] = (column, column + len(classes[name]))
        column += len(classes[name])
    y = np.zeros((row_count, column), dtype=np.float64)
    for name, values in outcomes.items():
        start, _end = offsets[name]
        lookup = {value: index for index, value in enumerate(classes[name])}
        for row_index, value in enumerate(values):
            y[row_index, start + lookup[value]] = 1.0
    predictions = np.zeros_like(y)
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        if not np.any(test) or int(np.sum(train)) <= matrix.shape[1] // 3:
            raise ForumFidelityV2Error("grouped folds are degenerate")
        mean = matrix[train].mean(axis=0)
        scale = matrix[train].std(axis=0)
        scale[scale < 1e-9] = 1.0
        train_design = np.column_stack(
            (np.ones(int(train.sum())), (matrix[train] - mean) / scale)
        )
        test_design = np.column_stack(
            (np.ones(int(test.sum())), (matrix[test] - mean) / scale)
        )
        xtx = train_design.T @ train_design
        penalty = np.eye(xtx.shape[0], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            xtx + penalty,
            train_design.T @ y[train],
        )
        predictions[test] = test_design @ coefficients
    metrics: dict[str, Any] = {}
    for name in sorted(classes):
        start, end = offsets[name]
        scores = predictions[:, start:end]
        scores -= np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(np.clip(scores, -30.0, 30.0))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        lookup = {value: index for index, value in enumerate(classes[name])}
        actual = np.asarray([lookup[value] for value in outcomes[name]], dtype=np.int64)
        row_log_loss = -np.log(
            np.maximum(probabilities[np.arange(row_count), actual], 1e-12)
        )
        predicted = np.argmax(probabilities, axis=1)
        recalls = []
        for class_index in range(len(classes[name])):
            mask = actual == class_index
            recalls.append(float(np.mean(predicted[mask] == class_index)))
        metrics[name] = {
            "classes": classes[name],
            "grouped_cv_log_loss": round(float(np.mean(row_log_loss)), 6),
            "grouped_cv_accuracy": round(float(np.mean(predicted == actual)), 6),
            "grouped_cv_macro_recall": round(float(np.mean(recalls)), 6),
            "row_log_loss": row_log_loss,
        }
    return metrics


def _cluster_bootstrap_interval(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
    replicates: int = 1_000,
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[str(group)].append(index)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(keys), size=len(keys))
        indices = [index for item in sampled for index in grouped[keys[int(item)]]]
        estimates[replicate] = float(np.mean(values[indices]))
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return round(float(lower), 6), round(float(upper), 6)


def _stratified_permutation_indices(
    rows: Sequence[Mapping[str, Any]], *, rng: np.random.Generator
) -> np.ndarray:
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(str(row.get("branch") or ""), int(row.get("tick", -1)))].append(index)
    result = np.arange(len(rows), dtype=np.int64)
    for key in sorted(strata):
        indices = np.asarray(strata[key], dtype=np.int64)
        result[indices] = rng.permutation(indices)
    return result


def _block_permutation_test(
    *,
    context: np.ndarray,
    profile: np.ndarray,
    previous_state: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Sequence[str]],
    groups: Sequence[str],
    alpha: float,
    actual_joint_result: Mapping[str, Mapping[str, Any]],
    replicates: int = 49,
) -> dict[str, Any]:
    """Test whether each block beats a same-branch, same-Tick permutation."""

    outcome_names = sorted(outcomes)
    actual_loss = float(
        np.mean(
            [
                float(actual_joint_result[name]["grouped_cv_log_loss"])
                for name in outcome_names
            ]
        )
    )
    rng = np.random.default_rng(20260904)
    permuted_losses: dict[str, list[float]] = {"profile": [], "previous_state": []}
    for _ in range(replicates):
        indices = _stratified_permutation_indices(rows, rng=rng)
        for block, matrix in (
            ("profile", np.column_stack((context, profile[indices], previous_state))),
            (
                "previous_state",
                np.column_stack((context, profile, previous_state[indices])),
            ),
        ):
            result = _ridge_multitask_cv(
                matrix=matrix, outcomes=outcomes, groups=groups, alpha=alpha
            )
            permuted_losses[block].append(
                float(
                    np.mean(
                        [float(result[name]["grouped_cv_log_loss"]) for name in outcome_names]
                    )
                )
            )
    summary = {}
    for block, values in permuted_losses.items():
        array = np.asarray(values, dtype=np.float64)
        summary[block] = {
            "actual_joint_mean_log_loss": round(actual_loss, 6),
            "permuted_mean_log_loss": round(float(array.mean()), 6),
            "actual_advantage_over_permutation": round(
                float(array.mean() - actual_loss), 6
            ),
            "one_sided_p_value": round(
                float((1 + np.sum(array <= actual_loss)) / (replicates + 1)), 6
            ),
        }
    return {
        "method": (
            "49 deterministic permutations within the same branch and Tick. The test "
            "breaks the Profile or previous-state pairing while preserving the visible "
            "stage composition. Lower actual log-loss is favorable."
        ),
        "replicates": replicates,
        "blocks": summary,
    }


def _previous_particle_feature_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    assimilations: Sequence[Mapping[str, Any]],
    snapshots: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact pre-turn state inherited from the previous Tick.

    Tick 3 starts from the frozen pre-event population.  Natural/D Tick 4
    inherit the committed shared-baseline Tick 3 checkpoint; later turns use
    their own branch's immediately preceding checkpoint.  We deliberately do
    not use the post-turn state because the current LLM decision is an input
    to that update and would leak the target into the predictor.
    """

    by_fingerprint: dict[tuple[str, str], Mapping[str, Any]] = {}
    by_response: dict[str, Mapping[str, Any]] = {}
    for row in assimilations:
        key = (
            str(row.get("prompt_sha256") or ""),
            str(row.get("response_sha256") or ""),
        )
        if not all(key) or key in by_fingerprint:
            raise ForumFidelityV2Error("assimilation fingerprints are ambiguous")
        by_fingerprint[key] = row
        response = key[1]
        if response in by_response:
            raise ForumFidelityV2Error("assimilation response fingerprints are duplicated")
        by_response[response] = row

    features: list[np.ndarray] = []
    source_counts: Counter[str] = Counter()
    for turn in rows:
        key = (
            str(turn.get("prompt_sha256") or ""),
            str(turn.get("response_sha256") or ""),
        )
        audit = by_fingerprint.get(key) or by_response.get(key[1])
        if audit is None:
            raise ForumFidelityV2Error("resident turn lacks a particle audit match")
        # The archived assimilation hook used the kernel's internal branch
        # label during early continuation ticks.  The identity-bound turn
        # evidence carries the authoritative public branch/tick key.
        branch = str(turn.get("branch") or audit.get("branch") or "")
        tick = int(turn.get("tick", audit.get("tick", -1)))
        if tick == 3:
            snapshot_key = ("shared_initial", 0)
            source_label = "frozen_pre_event_population"
        elif tick == 4 and branch in {"Natural", "D"}:
            snapshot_key = ("shared_baseline", 3)
            source_label = "shared_baseline_tick_3"
        else:
            snapshot_key = (branch, tick - 1)
            source_label = f"same_branch_tick_{tick - 1}"
        snapshot = snapshots.get(snapshot_key)
        if snapshot is None:
            raise ForumFidelityV2Error(
                f"previous-Tick particle snapshot is missing: {snapshot_key}"
            )
        parent_index = int(audit.get("parent_index", -1))
        values = np.asarray(snapshot["values"], dtype=np.float64)[parent_index]
        raw_weights = np.asarray(snapshot["weights"], dtype=np.float64)[parent_index]
        mass = float(np.asarray(snapshot["parent_weights"], dtype=np.float64)[parent_index])
        if parent_index < 0 or values.shape != (10, len(STATE_FIELDS)) or mass <= 0:
            raise ForumFidelityV2Error("previous-Tick particle row is invalid")
        weights = raw_weights / mass
        if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
            raise ForumFidelityV2Error("previous-Tick within-Agent weights drifted")
        mean = np.sum(values * weights[:, None], axis=0)
        variance = np.sum(((values - mean) ** 2) * weights[:, None], axis=0)
        ess = 1.0 / float(np.sum(weights**2))
        features.append(np.concatenate((mean, np.sqrt(np.maximum(variance, 0.0)), [ess / 10.0])))
        source_counts[source_label] += 1
    matrix = np.vstack(features)
    if not np.all(np.isfinite(matrix)):
        raise ForumFidelityV2Error("previous-Tick particle features are non-finite")
    return matrix, {
        "feature_count": int(matrix.shape[1]),
        "state_fields": list(STATE_FIELDS),
        "per_field_features": ["weighted_mean", "weighted_standard_deviation"],
        "additional_features": ["within_agent_particle_ess_divided_by_10"],
        "source_turn_counts": dict(sorted(source_counts.items())),
        "target_leakage_prevented": True,
    }


def _incremental_predictive_validity(
    rows: Sequence[Mapping[str, Any]],
    *,
    assimilations: Sequence[Mapping[str, Any]],
    snapshots: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    feature_dimension = 32
    context = np.vstack(
        [
            _hash_features(
                "\n".join(
                    (
                        str(row.get("scenario_summary") or ""),
                        f"branch={row.get('branch')} tick={row.get('tick')}",
                        _text_from_visible_context(row),
                    )
                ),
                dimension=feature_dimension,
            )
            for row in rows
        ]
    )
    profile = np.vstack(
        [
            _hash_features(_profile_text(row), dimension=feature_dimension)
            for row in rows
        ]
    )
    previous_state, state_audit = _previous_particle_feature_matrix(
        rows=rows,
        assimilations=assimilations,
        snapshots=snapshots,
    )
    model_matrices = {
        "M0_context": context,
        "M1_context_profile": np.column_stack((context, profile)),
        "M2_context_previous_particle_state": np.column_stack(
            (context, previous_state)
        ),
        "M3_context_profile_previous_particle_state": np.column_stack(
            (context, profile, previous_state)
        ),
    }
    groups = [str(row.get("agent_id") or "") for row in rows]
    if any(not value for value in groups):
        raise ForumFidelityV2Error("predictive rows contain empty Agent IDs")
    alpha = float(len(set(groups)))
    outcomes = _outcome_labels(rows)
    model_results = {
        name: _ridge_multitask_cv(
            matrix=matrix, outcomes=outcomes, groups=groups, alpha=alpha
        )
        for name, matrix in model_matrices.items()
    }
    metrics = {}
    profile_improvements = []
    state_improvements = []
    joint_improvements = []
    for ordinal, name in enumerate(sorted(outcomes)):
        losses = {
            model: float(result[name]["grouped_cv_log_loss"])
            for model, result in model_results.items()
        }
        profile_rows = (
            model_results["M2_context_previous_particle_state"][name]["row_log_loss"]
            - model_results["M3_context_profile_previous_particle_state"][name]["row_log_loss"]
        )
        state_rows = (
            model_results["M1_context_profile"][name]["row_log_loss"]
            - model_results["M3_context_profile_previous_particle_state"][name]["row_log_loss"]
        )
        joint_rows = (
            model_results["M0_context"][name]["row_log_loss"]
            - model_results["M3_context_profile_previous_particle_state"][name]["row_log_loss"]
        )
        profile_relative = (
            losses["M2_context_previous_particle_state"]
            - losses["M3_context_profile_previous_particle_state"]
        ) / max(losses["M2_context_previous_particle_state"], 1e-12)
        state_relative = (
            losses["M1_context_profile"]
            - losses["M3_context_profile_previous_particle_state"]
        ) / max(losses["M1_context_profile"], 1e-12)
        joint_relative = (
            losses["M0_context"]
            - losses["M3_context_profile_previous_particle_state"]
        ) / max(losses["M0_context"], 1e-12)
        profile_improvements.append(profile_relative)
        state_improvements.append(state_relative)
        joint_improvements.append(joint_relative)
        metrics[name] = {
            "grouped_cv_log_loss": {
                model: round(value, 6) for model, value in losses.items()
            },
            "profile_increment_given_previous_state": {
                "relative_log_loss_improvement": round(profile_relative, 6),
                "cluster_bootstrap_95pct_interval": list(
                    _cluster_bootstrap_interval(
                        profile_rows, groups, seed=20260910 + ordinal
                    )
                ),
            },
            "previous_state_increment_given_profile": {
                "relative_log_loss_improvement": round(state_relative, 6),
                "cluster_bootstrap_95pct_interval": list(
                    _cluster_bootstrap_interval(
                        state_rows, groups, seed=20260920 + ordinal
                    )
                ),
            },
            "joint_increment_over_context": {
                "relative_log_loss_improvement": round(joint_relative, 6),
                "cluster_bootstrap_95pct_interval": list(
                    _cluster_bootstrap_interval(
                        joint_rows, groups, seed=20260930 + ordinal
                    )
                ),
            },
            "grouped_cv_accuracy": {
                model: result[name]["grouped_cv_accuracy"]
                for model, result in model_results.items()
            },
            "grouped_cv_macro_recall": {
                model: result[name]["grouped_cv_macro_recall"]
                for model, result in model_results.items()
            },
            "classes": model_results["M0_context"][name]["classes"],
        }
    return {
        "method": (
            "Five-fold Agent-grouped multi-task ridge classification. Four matched "
            "models separate visible context, stable Profile/stake and the exact previous-"
            "Tick 10-particle state. No Agent appears in both train and test, and no "
            "post-decision particle state is used."
        ),
        "hashed_feature_dimension_per_block": feature_dimension,
        "previous_particle_state_features": state_audit,
        "models": {
            name: {"feature_count": int(matrix.shape[1])}
            for name, matrix in model_matrices.items()
        },
        "ridge_alpha_equal_to_unique_agent_count": alpha,
        "row_count": len(rows),
        "unique_agent_count": len(set(groups)),
        "outcomes": metrics,
        "block_permutation_test": _block_permutation_test(
            context=context,
            profile=profile,
            previous_state=previous_state,
            rows=rows,
            outcomes=outcomes,
            groups=groups,
            alpha=alpha,
            actual_joint_result=model_results[
                "M3_context_profile_previous_particle_state"
            ],
        ),
        "mean_profile_increment_given_previous_state": round(
            float(np.mean(profile_improvements)), 6
        ),
        "mean_previous_state_increment_given_profile": round(
            float(np.mean(state_improvements)), 6
        ),
        "mean_joint_increment_over_context": round(
            float(np.mean(joint_improvements)), 6
        ),
        "mean_relative_log_loss_improvement": round(
            float(np.mean(joint_improvements)), 6
        ),
        "structured_12d_alignment": _structured_vector_alignment(
            model_matrices=model_matrices,
            rows=rows,
            groups=groups,
            alpha=alpha,
        ),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else 0.0


def _longitudinal_style_similarity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expressive = []
    for row in rows:
        text = str(_decision(row).get("visible_text") or "").strip()
        if text:
            expressive.append(
                {
                    "agent_id": str(row.get("agent_id") or ""),
                    "branch": str(row.get("branch") or ""),
                    "tick": int(row.get("tick", -1)),
                    "vector": _hash_features(text, dimension=256),
                }
            )
    by_agent_branch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in expressive:
        by_agent_branch[(row["agent_id"], row["branch"])].append(row)
    pairs = []
    for key in sorted(by_agent_branch):
        values = sorted(by_agent_branch[key], key=lambda row: row["tick"])
        if len(values) < 2:
            continue
        left, right = values[0], values[-1]
        candidates = [
            value
            for value in expressive
            if value["agent_id"] != left["agent_id"]
            and value["branch"] == right["branch"]
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda value: (
                abs(value["tick"] - right["tick"]),
                hashlib.sha256(
                    (left["agent_id"] + value["agent_id"]).encode("utf-8")
                ).hexdigest(),
            )
        )
        pairs.append(
            {
                "agent_id": left["agent_id"],
                "same": _cosine(left["vector"], right["vector"]),
                "matched_other": _cosine(left["vector"], candidates[0]["vector"]),
            }
        )
    if len(pairs) < 10:
        return {
            "method": "character 1-3gram cosine with branch/tick matched controls",
            "eligible_longitudinal_agent_count": len(pairs),
            "status": "insufficient_repeated_public_expression",
        }
    differences = np.asarray(
        [row["same"] - row["matched_other"] for row in pairs], dtype=np.float64
    )
    groups = [str(row["agent_id"]) for row in pairs]
    return {
        "method": (
            "Within-Agent public-language character 1-3gram cosine compared with a "
            "different Agent matched on branch and nearest Tick. One pair per Agent/branch."
        ),
        "eligible_longitudinal_agent_count": len(pairs),
        "same_agent_mean_similarity": round(
            float(np.mean([row["same"] for row in pairs])), 6
        ),
        "matched_other_mean_similarity": round(
            float(np.mean([row["matched_other"] for row in pairs])), 6
        ),
        "same_agent_advantage": round(float(np.mean(differences)), 6),
        "cluster_bootstrap_95pct_interval": list(
            _cluster_bootstrap_interval(differences, groups, seed=20260901)
        ),
    }


def _direction(value: float, *, tolerance: float = 1e-7) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def _assimilation_consistency(
    rows: Sequence[Mapping[str, Any]],
    assimilations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decisions = {
        (str(row.get("prompt_sha256") or ""), str(row.get("response_sha256") or "")):
        _decision(row)
        for row in rows
    }
    decisions_by_response = {
        str(row.get("response_sha256") or ""): _decision(row) for row in rows
    }
    if len(decisions_by_response) != len(rows):
        raise ForumFidelityV2Error("resident response fingerprints are duplicated")
    agreements = []
    by_field: dict[str, list[int]] = defaultdict(list)
    shift_norms = []
    matched = 0
    for row in assimilations:
        decision = decisions.get(
            (str(row.get("prompt_sha256") or ""), str(row.get("response_sha256") or ""))
        )
        if decision is None:
            decision = decisions_by_response.get(str(row.get("response_sha256") or ""))
        if decision is None:
            continue
        matched += 1
        before = row["before_state"]
        after = row["after_state"]
        expected: dict[str, int] = {}
        trust_delta = float(decision.get("trust_delta", 0.0))
        satisfaction_delta = float(decision.get("satisfaction_delta", 0.0))
        if _direction(trust_delta):
            expected["trust"] = _direction(trust_delta)
        if _direction(satisfaction_delta):
            expected["satisfaction"] = _direction(satisfaction_delta)
        urgency = float(decision.get("urgency", 0.0))
        if _direction(urgency - float(before["concern"])):
            expected["concern"] = _direction(urgency - float(before["concern"]))
        action = str(decision.get("action") or "read")
        activity_target = 1.0 if action in PUBLIC_ACTIONS else 0.0
        if _direction(activity_target - float(before["activity"])):
            expected["activity"] = _direction(
                activity_target - float(before["activity"])
            )
        stance = _stance_family(decision)
        if stance == "support":
            expected.update({"support": 1, "opposition": -1})
        elif stance == "oppose_or_concern":
            expected.update({"support": -1, "opposition": 1})
        if _evidence_family(decision) == "verified_or_corrected" or action == "correct":
            expected.update({"verified_belief": 1, "rumor_belief": -1})
        for field, direction in expected.items():
            observed = _direction(float(after[field]) - float(before[field]))
            agreement = int(observed == direction)
            agreements.append(agreement)
            by_field[field].append(agreement)
        shift_norms.append(
            math.sqrt(
                sum(
                    (float(after[field]) - float(before[field])) ** 2
                    for field in STATE_FIELDS
                )
            )
        )
    if matched < max(1, int(0.80 * len(assimilations))):
        raise ForumFidelityV2Error("too few assimilation rows match resident evidence")
    return {
        "interpretation": (
            "Mechanism consistency only: validated LLM fields are direct inputs to the "
            "particle update, so this is not independent predictive validation."
        ),
        "matched_turn_count": matched,
        "unmatched_terminal_turn_count": len(assimilations) - matched,
        "tested_direction_count": len(agreements),
        "direction_agreement_rate": round(float(np.mean(agreements)), 6),
        "mean_12d_parent_shift_l2": round(float(np.mean(shift_norms)), 6),
        "field_direction_agreement": {
            field: round(float(np.mean(values)), 6)
            for field, values in sorted(by_field.items())
        },
        "resampled_parent_updates": sum(bool(row.get("resampled")) for row in assimilations),
    }


def evaluate_persona_consistency_v2(
    *,
    persona_evidence: Mapping[str, Any],
    assimilations: Sequence[Mapping[str, Any]],
    snapshots: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    legacy_blind_review_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = persona_evidence.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ForumFidelityV2Error("private persona evidence is invalid")
    rows = [row for row in rows if isinstance(row, Mapping)]
    if len(rows) < 200:
        raise ForumFidelityV2Error("persona evidence is too small for grouped evaluation")
    result = {
        "evaluation_unit": "resident LLM turn grouped by persistent synthetic Agent",
        "turn_count": len(rows),
        "unique_agent_count": len({str(row.get("agent_id") or "") for row in rows}),
        "repeated_agent_count": sum(
            value > 1 for value in Counter(str(row.get("agent_id") or "") for row in rows).values()
        ),
        "joint_profile_previous_state_constraint": _incremental_predictive_validity(
            rows,
            assimilations=assimilations,
            snapshots=snapshots,
        ),
        "longitudinal_language_consistency": _longitudinal_style_similarity(rows),
        "immediate_llm_particle_assimilation": _assimilation_consistency(
            rows, assimilations
        ),
        "independent_blind_review": dict(legacy_blind_review_metrics or {}),
        "limitations": [
            "The grouped prediction test measures whether Profile/stake and previous-Tick particle state add behavioral information; it does not prove a real human identity.",
            "Character n-gram similarity measures stable expression style, not deep semantic equivalence.",
            "Immediate particle agreement is an implementation-consistency check because the structured LLM turn drives that update.",
        ],
    }
    predictive = result["joint_profile_previous_state_constraint"]
    longitudinal = result["longitudinal_language_consistency"]
    blind = result["independent_blind_review"]
    style_interval = longitudinal.get("cluster_bootstrap_95pct_interval")
    style_supported = (
        isinstance(style_interval, Sequence)
        and len(style_interval) == 2
        and float(style_interval[0]) > 0.0
    )
    profile_signal_supported = float(
        predictive["mean_profile_increment_given_previous_state"]
    ) > 0.01
    previous_state_signal_supported = float(
        predictive["mean_previous_state_increment_given_profile"]
    ) > 0.01
    joint_signal_supported = float(
        predictive["mean_joint_increment_over_context"]
    ) > 0.01
    result["component_status"] = {
        "direct_context_uptake": (
            "supported"
            if float(blind.get("direct_context_uptake_rate", 0.0)) >= 0.85
            else "needs_iteration"
        ),
        "longitudinal_expression_style": (
            "supported" if style_supported else "needs_iteration"
        ),
        "fine_grained_profile_identification": (
            "supported"
            if float(blind.get("profile_match_rate", 0.0)) >= 0.75
            else "needs_iteration"
        ),
        "out_of_agent_profile_predictive_signal": (
            "supported" if profile_signal_supported else "not_supported"
        ),
        "previous_tick_particle_state_predictive_signal": (
            "supported" if previous_state_signal_supported else "not_supported"
        ),
        "joint_profile_and_previous_state_predictive_signal": (
            "supported" if joint_signal_supported else "not_supported"
        ),
        "llm_to_particle_mechanism_consistency": (
            "supported_not_independent_validation"
            if result["immediate_llm_particle_assimilation"][
                "direction_agreement_rate"
            ]
            >= 0.90
            else "needs_iteration"
        ),
    }
    result["status"] = (
        "supported"
        if profile_signal_supported
        and previous_state_signal_supported
        and joint_signal_supported
        and style_supported
        and result["component_status"]["fine_grained_profile_identification"]
        == "supported"
        else "mixed_evidence_needs_iteration"
    )
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: Sequence[float]
) -> list[float]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= sorted_weights.sum()
    return [
        round(float(np.interp(probability, cumulative, sorted_values)), 6)
        for probability in probabilities
    ]


def _stake_scalar(row: Mapping[str, Any], name: str) -> float:
    fields = row.get("fields")
    fields = fields if isinstance(fields, Mapping) else {}
    value = fields.get(name)
    value = value.get("value") if isinstance(value, Mapping) else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _stake_group(row: Mapping[str, Any]) -> str:
    fields = row.get("fields")
    fields = fields if isinstance(fields, Mapping) else {}
    direct = fields.get("directly_affected")
    if isinstance(direct, Mapping) and direct.get("value") is True:
        return "directly_affected"
    if max(
        _stake_scalar(row, "service_channel_dependency"),
        _stake_scalar(row, "resource_need_intensity"),
    ) >= 0.65:
        return "service_dependent"
    if _stake_scalar(row, "substitute_resource_availability") >= 0.65:
        return "alternatives_available"
    return "bystander_or_mixed"


def _effective_rank(covariance: np.ndarray) -> float:
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(diagonal, diagonal)
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0.0)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    entropy = -sum(float(value * math.log(value)) for value in probabilities if value > 0)
    return float(math.exp(entropy))


def _mean_pairwise_distance(matrix: np.ndarray) -> float:
    total = 0.0
    count = 0
    for start in range(0, matrix.shape[0], 100):
        block = matrix[start : start + 100]
        distances = np.sqrt(
            np.maximum(
                np.sum((block[:, None, :] - matrix[None, :, :]) ** 2, axis=2),
                0.0,
            )
        )
        row_indices = np.arange(start, start + block.shape[0])[:, None]
        column_indices = np.arange(matrix.shape[0])[None, :]
        mask = column_indices > row_indices
        total += float(distances[mask].sum())
        count += int(mask.sum())
    return total / max(count, 1)


def _group_separation(
    matrix: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
    permutations: int = 199,
) -> dict[str, Any]:
    labels = np.asarray(groups, dtype="U32")
    unique = sorted(set(groups))
    overall = np.average(matrix, axis=0, weights=weights)
    total = float(np.sum(weights[:, None] * (matrix - overall) ** 2))

    def statistic(current: np.ndarray) -> float:
        between = 0.0
        for label in unique:
            mask = current == label
            group_weight = float(weights[mask].sum())
            if group_weight <= 0:
                continue
            mean = np.average(matrix[mask], axis=0, weights=weights[mask])
            between += group_weight * float(np.sum((mean - overall) ** 2))
        return between / max(total, 1e-12)

    observed = statistic(labels)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        if statistic(rng.permutation(labels)) >= observed - 1e-12:
            exceed += 1
    return {
        "weighted_multivariate_r_squared": round(observed, 6),
        "label_permutation_count": permutations,
        "permutation_p_value": round((exceed + 1) / (permutations + 1), 6),
        "group_count": len(unique),
        "groups": unique,
    }


def _weighted_distribution_summary(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or float(weights.sum()) <= 0:
        raise ForumFidelityV2Error("weighted distribution input is invalid")
    weights = weights / weights.sum()
    mean = float(np.average(values, weights=weights))
    centered = values - mean
    variance = float(np.average(centered**2, weights=weights))
    degenerate = variance < 1e-10
    sd = math.sqrt(max(variance, 0.0))
    if degenerate:
        skewness = None
        kurtosis = None
    else:
        standardized = centered / sd
        skewness = float(np.average(standardized**3, weights=weights))
        kurtosis = float(np.average(standardized**4, weights=weights))
    histogram, edges = np.histogram(
        values, bins=20, range=(lower, upper), weights=weights
    )
    smoothed = np.convolve(histogram, np.asarray([0.25, 0.5, 0.25]), mode="same")
    peak_count = sum(
        smoothed[index] > smoothed[index - 1]
        and smoothed[index] >= smoothed[index + 1]
        and smoothed[index] >= 0.02
        for index in range(1, len(smoothed) - 1)
    )
    return {
        "weighted_mean": round(mean, 6),
        "weighted_standard_deviation": round(sd, 6),
        "weighted_quantiles": {
            key: value
            for key, value in zip(
                ("p05", "p25", "p50", "p75", "p95"),
                _weighted_quantile(
                    values, weights, (0.05, 0.25, 0.5, 0.75, 0.95)
                ),
            )
        },
        "degenerate_distribution": degenerate,
        "skewness": None if skewness is None else round(skewness, 6),
        "kurtosis": None if kurtosis is None else round(kurtosis, 6),
        "bimodality_coefficient": (
            None
            if skewness is None or kurtosis is None or kurtosis <= 1e-12
            else round(float((skewness**2 + 1.0) / kurtosis), 6)
        ),
        "smoothed_density_peak_count": int(peak_count),
        "histogram_bin_edges": [round(float(value), 3) for value in edges],
        "histogram_weight": [round(float(value), 6) for value in histogram],
    }


def _particle_frame_statistics(
    *,
    branch: str,
    tick: int,
    snapshot: Mapping[str, np.ndarray],
    stake_groups: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(snapshot["values"], dtype=np.float64)
    weights = np.asarray(snapshot["weights"], dtype=np.float64)
    parent_weights = np.asarray(snapshot["parent_weights"], dtype=np.float64)
    normalized = weights / parent_weights[:, None]
    parent_means = np.sum(values * normalized[:, :, None], axis=1)
    overall = np.average(parent_means, axis=0, weights=parent_weights)
    centered_parent = parent_means - overall
    between_covariance = (
        centered_parent.T @ (centered_parent * parent_weights[:, None])
    )
    centered_particle = values - parent_means[:, None, :]
    within_covariance = np.einsum(
        "np,npi,npj->ij", weights, centered_particle, centered_particle
    )
    total_covariance = between_covariance + within_covariance
    total_variance = np.maximum(np.diag(total_covariance), 1e-12)
    between_variance = np.maximum(np.diag(between_covariance), 0.0)
    within_variance = np.maximum(np.diag(within_covariance), 0.0)
    standardized = centered_parent / np.sqrt(total_variance)
    stance = parent_means[:, STATE_FIELDS.index("support")] - parent_means[
        :, STATE_FIELDS.index("opposition")
    ]
    stance_distribution = _weighted_distribution_summary(
        stance, parent_weights, lower=-1.0, upper=1.0
    )
    particle_marginals = {
        field: _weighted_distribution_summary(
            values[:, :, STATE_FIELDS.index(field)].reshape(-1),
            weights.reshape(-1),
            lower=0.0,
            upper=1.0,
        )
        for field in (
            "attention",
            "concern",
            "trust",
            "satisfaction",
            "rumor_belief",
            "verified_belief",
            "evidence_acceptance",
        )
    }
    return {
        "branch": branch,
        "tick": tick,
        "parent_agent_count": 1_000,
        "particles_per_parent": 10,
        "particle_count": 10_000,
        "state_fields": list(STATE_FIELDS),
        "effective_state_rank": round(_effective_rank(total_covariance), 6),
        "mean_standardized_parent_pairwise_distance": round(
            _mean_pairwise_distance(standardized), 6
        ),
        "variance_decomposition": {
            field: {
                "between_parent_variance": round(float(between_variance[index]), 8),
                "within_parent_particle_variance": round(float(within_variance[index]), 8),
                "between_share_of_total": round(
                    float(between_variance[index] / total_variance[index]), 6
                ),
            }
            for index, field in enumerate(STATE_FIELDS)
        },
        "stake_group_separation": _group_separation(
            standardized,
            parent_weights,
            stake_groups,
            seed=20261000 + tick + (100 if branch == "D" else 0),
        ),
        "particle_state_distributions": particle_marginals,
        "net_stance_distribution": {
            **stance_distribution,
            "interpretation": (
                "Descriptive only: multiple peaks are reported when present and are not "
                "forced as a publication gate."
            ),
        },
    }


def _normalized_entropy(counter: Counter[str]) -> tuple[float, float]:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0, float(bool(total))
    probabilities = [value / total for value in counter.values()]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return entropy / math.log(len(counter)), math.exp(entropy)


def _gini(values: Sequence[int]) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if len(array) == 0 or float(array.sum()) == 0:
        return 0.0
    indices = np.arange(1, len(array) + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(indices * array) / (len(array) * array.sum()))
        - (len(array) + 1.0) / len(array)
    )


def _behavioral_heterogeneity(
    persona_rows: Sequence[Mapping[str, Any]],
    *,
    population_agent_ids: Sequence[str],
) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "action": Counter(),
        "action_family": Counter(),
        "stance": Counter(),
        "public_stance": Counter(),
        "topic": Counter(),
        "evidence_status": Counter(),
        "private_action": Counter(),
    }
    activations = Counter()
    private_action_count = 0
    for row in persona_rows:
        decision = _decision(row)
        agent_id = str(row.get("agent_id") or "")
        activations[agent_id] += 1
        counters["action"][str(decision.get("action") or "read")] += 1
        counters["action_family"][_action_family(decision)] += 1
        contract_stance = _contract_stance(decision)
        counters["stance"][contract_stance] += 1
        if str(decision.get("action") or "read") in PUBLIC_ACTIONS:
            counters["public_stance"][contract_stance] += 1
        counters["topic"][str(decision.get("topic") or "unspecified")] += 1
        counters["evidence_status"][_evidence_family(decision)] += 1
        private_actions = decision.get("private_actions")
        if isinstance(private_actions, Sequence) and not isinstance(
            private_actions, (str, bytes)
        ):
            for action in private_actions:
                if isinstance(action, Mapping):
                    counters["private_action"][
                        str(action.get("kind") or action.get("action") or "other")
                    ] += 1
                    private_action_count += 1
    distributions = {}
    for name, counter in counters.items():
        normalized, effective = _normalized_entropy(counter)
        distribution = {
            "normalized_entropy": round(normalized, 6),
            "effective_category_count": round(effective, 6),
            "category_count": len(counter),
            "top_category_share": round(
                max(counter.values(), default=0) / max(sum(counter.values()), 1), 6
            ),
        }
        # Topic strings are high-cardinality historical labels and may contain
        # legacy encoding noise.  Public evidence retains only distributional
        # sufficient statistics; it does not publish the labels themselves.
        if name != "topic":
            distribution["counts"] = dict(sorted(counter.items()))
        distributions[name] = distribution
    activation_counts = [activations.get(agent_id, 0) for agent_id in population_agent_ids]
    return {
        "resident_turn_count": len(persona_rows),
        "unique_activated_agent_count": sum(value > 0 for value in activation_counts),
        "population_activation_coverage": round(
            sum(value > 0 for value in activation_counts) / len(activation_counts), 6
        ),
        "activation_count_gini": round(_gini(activation_counts), 6),
        "private_action_count": private_action_count,
        "distributions": distributions,
    }


def evaluate_population_heterogeneity_v2(
    *,
    snapshots: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    persona_evidence: Mapping[str, Any],
    scenario_stake_release: Mapping[str, Any],
    population_agent_ids: Sequence[str],
) -> dict[str, Any]:
    positions = scenario_stake_release.get("positions")
    if not isinstance(positions, Sequence) or len(positions) != 1_000:
        raise ForumFidelityV2Error("scenario stake release is invalid")
    stake_by_agent = {
        str(row.get("agent_id") or ""): row
        for row in positions
        if isinstance(row, Mapping)
    }
    if set(stake_by_agent) != set(population_agent_ids):
        raise ForumFidelityV2Error("stake and population identities differ")
    groups = [_stake_group(stake_by_agent[agent_id]) for agent_id in population_agent_ids]
    frame_rows = [
        _particle_frame_statistics(
            branch=branch,
            tick=tick,
            snapshot=snapshot,
            stake_groups=groups,
        )
        for (branch, tick), snapshot in sorted(
            snapshots.items(), key=lambda item: (item[0][1], item[0][0])
        )
    ]
    rows = persona_evidence.get("rows")
    rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, Sequence) else []
    final_frames = [row for row in frame_rows if row["tick"] == max(v["tick"] for v in frame_rows)]
    behavioral = _behavioral_heterogeneity(
        rows, population_agent_ids=population_agent_ids
    )
    stance_entropy = float(
        behavioral["distributions"]["public_stance"]["normalized_entropy"]
    )
    non_initial_frames = [row for row in frame_rows if row["tick"] > 0]
    minimum_stake_p = min(
        row["stake_group_separation"]["permutation_p_value"]
        for row in non_initial_frames
    )
    state_supported = (
        len(frame_rows) >= 5
        and minimum_stake_p <= 0.05
        and max(row["effective_state_rank"] for row in final_frames) >= 2.0
    )
    expression_supported = stance_entropy >= 0.50
    return {
        "evaluation_unit": (
            "1,000 weighted parent Agents; 10 particles per parent quantify within-Agent uncertainty"
        ),
        "particle_rows_counted_as_people": 0,
        "frames": frame_rows,
        "behavioral_heterogeneity": behavioral,
        "summary": {
            "frame_count": len(frame_rows),
            "minimum_final_effective_state_rank": round(
                min(row["effective_state_rank"] for row in final_frames), 6
            ),
            "maximum_stake_group_multivariate_r_squared": round(
                max(
                    row["stake_group_separation"][
                        "weighted_multivariate_r_squared"
                    ]
                    for row in frame_rows
                ),
                6,
            ),
            "minimum_stake_group_permutation_p_value": round(
                min(
                    row["stake_group_separation"]["permutation_p_value"]
                    for row in frame_rows
                ),
                6,
            ),
            "maximum_mean_standardized_parent_pairwise_distance": round(
                max(row["mean_standardized_parent_pairwise_distance"] for row in frame_rows),
                6,
            ),
        },
        "component_status": {
            "continuous_parent_state_heterogeneity": (
                "supported" if state_supported else "needs_iteration"
            ),
            "stake_group_state_separation": (
                "supported"
                if minimum_stake_p <= 0.05
                else "needs_iteration"
            ),
            "expressed_stance_diversity": (
                "supported" if expression_supported else "needs_iteration"
            ),
        },
        "status": (
            "supported"
            if state_supported and expression_supported
            else "state_supported_behavioral_expression_needs_iteration"
            if state_supported
            else "needs_iteration"
        ),
        "limitations": [
            "Stake groups are synthetic issue positions, not demographic categories.",
            "Particle dispersion is uncertainty within one Agent and is not an enlarged sample size.",
            "Multimodality is described rather than required; a realistic population need not be bimodal on every state.",
            "Public stance diversity is conditional on public-expression turns; silent neutral turns remain reported separately and are not relabelled as uncertainty.",
        ],
    }


def build_fidelity_evaluation_v2(
    *,
    persona_evidence: Mapping[str, Any],
    assimilations: Sequence[Mapping[str, Any]],
    snapshots: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    scenario_stake_release: Mapping[str, Any],
    population_agent_ids: Sequence[str],
    legacy_blind_review_metrics: Mapping[str, Any] | None = None,
    replay_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    persona = evaluate_persona_consistency_v2(
        persona_evidence=persona_evidence,
        assimilations=assimilations,
        snapshots=snapshots,
        legacy_blind_review_metrics=legacy_blind_review_metrics,
    )
    heterogeneity = evaluate_population_heterogeneity_v2(
        snapshots=snapshots,
        persona_evidence=persona_evidence,
        scenario_stake_release=scenario_stake_release,
        population_agent_ids=population_agent_ids,
    )
    body = {
        "schema_version": FIDELITY_V2_SCHEMA,
        "scenario_id": "governance_legitimacy_dispute",
        "dimensions": {
            "micro_persona_consistency": persona,
            "population_heterogeneity": heterogeneity,
        },
        "replay_audit": dict(replay_audit or {}),
        "privacy": {
            "record_level_agent_ids_persisted": False,
            "record_level_turns_persisted": False,
            "full_particle_rows_persisted": False,
            "parent_particle_mapping_persisted": False,
            "private_npz_is_not_public": True,
        },
        "interpretation_boundary": (
            "This evaluates internal simulation fidelity and mechanism behavior. It does not "
            "turn synthetic Agents into real students or establish a real policy effect."
        ),
    }
    body["evaluation_sha256"] = _sha(body)
    return body
