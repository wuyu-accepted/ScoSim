from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import unicodedata
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .common import json_safe, read_jsonl, write_json
from .inference import cluster_bootstrap_interval


CONDITIONS = ("context_only", "profile_only", "state_only", "full")
ACTIONS = ("ignore", "read", "post", "reply", "seek_help", "report_risk")
ACTION_ALIASES = {"quote": "reply", "repost": "post", "seek_verify": "seek_help", "correct": "reply", "no_action": "ignore"}
EMOTIONS = ("neutral", "calm", "curious", "concerned", "skeptical", "frustrated", "supportive", "urgent")
STANCES = ("support", "oppose", "mixed", "uncertain", "neutral")
STANCE_ALIASES = {"supportive": "support", "critical": "oppose", "concerned": "oppose"}
VISIBILITY = ("public", "private", "none")
STANCE_LEVEL = {"strong_oppose": -1.0, "weak_oppose": -0.5, "neutral_or_mixed": 0.0, "weak_support": 0.5, "strong_support": 1.0}
FEATURE_COUNT = 24
PUBLIC_STANCE_DEADBAND = 0.05
POST_STATE_DEADBAND = 0.05
PRESTATE_NEUTRAL_QUANTILE = 0.82
PRESTATE_NEUTRAL_RADIUS_BY_METHOD = {
    "campuspulse": 0.041589952012686116,
    "belief_engine": 0.18032786885245905,
}
ALIGNMENT_DEADBAND = 1e-7
PUBLIC_ACTIONS = {"post", "reply", "quote", "repost", "seek_verify", "correct", "seek_help", "report_risk"}
METHODS = ("campuspulse", "belief_engine", "generative_agents", "chuang_opinion_dynamics")


def stance_direction_contract() -> dict[str, Any]:
    """Expose the frozen, deliberately separated output/state thresholds."""
    return {
        "structured_public_stance": {
            "rule": "fixed_absolute_deadband",
            "neutral_radius": PUBLIC_STANCE_DEADBAND,
            "used_by": ["behavior_distributions", "condition_classifier"],
        },
        "expression_pre_state_agreement": {
            "rule": "method_native_equal_neutral_mass_quantile",
            "neutral_quantile": PRESTATE_NEUTRAL_QUANTILE,
            "neutral_radius_by_method": dict(PRESTATE_NEUTRAL_RADIUS_BY_METHOD),
            "used_by": ["expression_pre_state_agreement"],
        },
        "decision_post_state_expression_consistency": {
            "rule": "fixed_absolute_deadband",
            "neutral_radius": POST_STATE_DEADBAND,
            "used_by": ["decision_post_state_expression_consistency"],
        },
    }


def case_id(row: dict[str, Any]) -> str:
    payload = [row.get(key) for key in ("experiment_id", "method", "variant", "seed", "agent_id", "tick")]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "d2-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _category(value: Any, vocabulary: tuple[str, ...], aliases: dict[str, str], field: str) -> str:
    normalized = aliases.get(str(value or ""), str(value or ""))
    if normalized not in vocabulary:
        raise ValueError(f"unknown {field} category {value!r}")
    return normalized


def _one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
    return [float(value == item) for item in vocabulary]


def _normalized_text(value: Any) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).lower().split())


def _stance_from_scalar(value: Any) -> str:
    if value is None:
        return "neutral"
    scalar = float(value)
    return (
        "support" if scalar > PUBLIC_STANCE_DEADBAND
        else "oppose" if scalar < -PUBLIC_STANCE_DEADBAND
        else "neutral"
    )


def _emotion_category(value: Any) -> str:
    """Normalize native/open-vocabulary affect into the frozen D2 taxonomy."""
    raw = str(value or "").strip().lower()
    if raw in EMOTIONS:
        return raw
    if not raw or raw in {"practical", "务实"}:
        return "neutral"
    if raw in {"平和"}:
        return "calm"
    if raw in {"询问"}:
        return "curious"
    # Priority is frozen for compound labels: explicit doubt, then negative
    # frustration, then concern/anxiety.
    if any(token in raw for token in ("质疑", "疑问", "疑惑", "谨慎", "审慎", "困惑", "questioning", "cautious")):
        return "skeptical"
    if any(token in raw for token in ("挫败", "不满", "烦闷", "愤怒", "失望", "疲惫", "沮丧", "无奈", "无助", "无力")):
        return "frustrated"
    if any(token in raw for token in ("担忧", "焦虑", "不安", "关切", "关注", "着急", "压抑")):
        return "concerned"
    raise ValueError(f"unknown emotion category {value!r}")


def feature_vector(row: dict[str, Any]) -> np.ndarray:
    action = _category(row.get("action"), ACTIONS, ACTION_ALIASES, "action")
    emotion = _emotion_category(row.get("emotion"))
    stance = _category(row.get("stance_label") or _stance_from_scalar(row.get("public_stance")), STANCES, STANCE_ALIASES, "stance")
    visibility = _category(row.get("public_or_private"), VISIBILITY, {}, "public_or_private")
    text = _normalized_text(row.get("generated_text"))
    vector = np.asarray([
        *_one_hot(action, ACTIONS), *_one_hot(emotion, EMOTIONS),
        *_one_hot(stance, STANCES), *_one_hot(visibility, VISIBILITY),
        float(bool(text)), min(len(text) / 512.0, 1.0),
    ], dtype=np.float64)
    if vector.shape != (FEATURE_COUNT,):
        raise AssertionError("dimension-2 feature contract drifted")
    return vector


def _loao_from_features(vectors: np.ndarray, agents: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predictions = np.empty(len(agents), dtype=object)
    for held_out in sorted(set(agents)):
        train, test = agents != held_out, agents == held_out
        if len(set(labels[train])) < 2:
            raise ValueError(f"training fold for {held_out} has fewer than two classes")
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=20260722),
        )
        model.fit(vectors[train], labels[train])
        predictions[test] = model.predict(vectors[test])
    return predictions


def _task_scores_from_features(vectors: np.ndarray, agents: np.ndarray, condition_labels: np.ndarray) -> dict[str, Any]:
    tasks = {
        "four_condition": condition_labels,
        "profile_factor": np.asarray(["present" if label in {"profile_only", "full"} else "absent" for label in condition_labels]),
        "state_factor": np.asarray(["present" if label in {"state_only", "full"} else "absent" for label in condition_labels]),
    }
    result = {}
    for name, truth in tasks.items():
        predicted = _loao_from_features(vectors, agents, truth)
        labels = CONDITIONS if name == "four_condition" else ("absent", "present")
        result[name] = {"accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro")), "truth": truth, "predicted": predicted}
    return result


def _task_scores(rows: list[dict[str, Any]], condition_labels: np.ndarray) -> dict[str, Any]:
    vectors = np.stack([feature_vector(row) for row in rows])
    agents = np.asarray([str(row["agent_id"]) for row in rows], dtype=object)
    return _task_scores_from_features(vectors, agents, condition_labels)


def _null_metrics(vectors: np.ndarray, agents: np.ndarray, labels: np.ndarray) -> dict[str, tuple[float, float]]:
    scores = _task_scores_from_features(vectors, agents, labels)
    return {name: (float(score["accuracy"]), float(score["macro_f1"])) for name, score in scores.items()}


def _hash_features(text: str) -> np.ndarray:
    vector = np.zeros(256, dtype=np.float64)
    compact = _normalized_text(text)
    for size in (1, 2, 3):
        for index in range(max(0, len(compact) - size + 1)):
            code = int.from_bytes(hashlib.blake2b(compact[index:index + size].encode("utf-8"), digest_size=8).digest(), "big")
            vector[code % 256] += -1.0 if code & 1 else 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def language_style_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    raw_expression_count = 0
    exact_duplicate_count = 0
    for row in rows:
        text = _normalized_text(row.get("generated_text"))
        if text:
            raw_expression_count += 1
            agent = str(row["agent_id"])
            if text in grouped[agent]:
                exact_duplicate_count += 1
                continue
            grouped[agent][text] = _hash_features(text)
    per_agent = []
    eligible_agents = []
    eligible_unique_expression_count = 0
    for agent, by_text in grouped.items():
        vectors = list(by_text.values())
        if len(vectors) >= 2:
            per_agent.append(float(np.mean([float(vectors[i] @ vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))])))
            eligible_agents.append(agent)
            eligible_unique_expression_count += len(vectors)
    if len(per_agent) < 10:
        return {
            "value": None,
            "status": "NA_insufficient_repeated_expression",
            "raw_expression_count": raw_expression_count,
            "exact_duplicate_count": exact_duplicate_count,
            "eligible_agent_count": len(per_agent),
            "eligible_unique_expression_count": eligible_unique_expression_count,
            "eligible_expression_count": eligible_unique_expression_count,
        }
    return {
        "value": float(np.mean(per_agent)),
        "bootstrap_95pct_interval": cluster_bootstrap_interval(per_agent, eligible_agents),
        "status": "ok",
        "raw_expression_count": raw_expression_count,
        "exact_duplicate_count": exact_duplicate_count,
        "eligible_agent_count": len(per_agent),
        "eligible_unique_expression_count": eligible_unique_expression_count,
        "eligible_expression_count": eligible_unique_expression_count,
    }


def _state_direction(row: dict[str, Any], sidecar: dict[str, float]) -> str | None:
    if row["method"] == "belief_engine":
        scalar = sidecar.get(case_id(row))
    elif row["method"] == "campuspulse":
        particles, weights = row.get("particles"), row.get("particle_weights")
        if not particles or not weights or len(particles) != len(weights) or float(sum(weights)) <= 0:
            return None
        scalar = sum(float(w) * (float(p["support"]) - float(p["opposition"])) for p, w in zip(particles, weights)) / float(sum(weights))
    else:
        return None
    if scalar is None:
        return None
    radius = PRESTATE_NEUTRAL_RADIUS_BY_METHOD.get(str(row.get("method")))
    if radius is None:
        return None
    return "support" if scalar > radius else "oppose" if scalar < -radius else "neutral"


def _post_state_direction(row: dict[str, Any]) -> str | None:
    """Map CampusPulse's native post-decision state with the frozen D2 rule."""
    if row.get("method") != "campuspulse":
        return None
    post_state = row.get("post_state")
    if not isinstance(post_state, dict):
        return None
    try:
        scalar = float(post_state["support"]) - float(post_state["opposition"])
    except (KeyError, TypeError, ValueError):
        return None
    return (
        "support" if scalar > POST_STATE_DEADBAND
        else "oppose" if scalar < -POST_STATE_DEADBAND
        else "neutral"
    )


def _expression_direction(level: Any) -> str | None:
    """Map the shared blind-judge label to direction; state thresholds stay metric-specific."""
    if level not in STANCE_LEVEL:
        return None
    value = STANCE_LEVEL[level]
    return "support" if value > 0 else "oppose" if value < 0 else "neutral"


def score_text_judgements(rows: list[dict[str, Any]], annotations: list[dict[str, Any]] | None, belief_state_sidecar: dict[str, float] | None = None) -> dict[str, Any]:
    names = ("direct_context_uptake", "unsupported_fact_fabrication", "expression_pre_state_agreement", "decision_post_state_expression_consistency")
    if annotations is None:
        return {name: {"rate": None, "eligible_count": 0, "status": "NA_not_annotated"} for name in names}
    by_case = {str(item["case_id"]): item for item in annotations}
    if len(by_case) != len(annotations):
        raise ValueError("duplicate dimension-2 judgement case_id")
    context, unsupported, agreement, post_agreement = [], [], [], []
    context_agents, unsupported_agents, agreement_agents, post_agreement_agents = [], [], [], []
    confusion: Counter[tuple[str, str]] = Counter()
    post_confusion: Counter[tuple[str, str]] = Counter()
    expressed = sum(bool(_normalized_text(row.get("generated_text"))) for row in rows)
    applicable = 0
    post_applicable = 0
    for row in rows:
        judgement = by_case.get(case_id(row))
        if judgement is None:
            continue
        if judgement["context_uptake"] in {"yes", "no"}:
            context.append(judgement["context_uptake"] == "yes")
            context_agents.append(str(row["agent_id"]))
        if judgement["unsupported_fact"] in {"yes", "no"}:
            unsupported.append(judgement["unsupported_fact"] == "yes")
            unsupported_agents.append(str(row["agent_id"]))
        # Ablation applicability is variant-specific, while horizontal
        # comparison runs use the complete state-bearing method under the
        # common `cross_method_full` label.  Do not silently mark every
        # horizontal Belief Engine/CampusPulse case as NA.
        state_bearing = row.get("variant") in {
            "state_only", "full", "cross_method_full",
        }
        if row["method"] in {"campuspulse", "belief_engine"} and state_bearing:
            applicable += 1
            state = _state_direction(row, belief_state_sidecar or {})
            expression = _expression_direction(judgement["expression_stance"])
            if state is not None and expression is not None:
                agreement.append(state == expression)
                agreement_agents.append(str(row["agent_id"]))
                confusion[(state, expression)] += 1
        # The ablation post-state metric is CampusPulse-specific.  It uses the
        # same expression extraction, direction mapping, deadband, eligibility,
        # clustering and denominator as the registered horizontal metric.
        if row.get("method") == "campuspulse" and state_bearing:
            post_applicable += 1
            post_state = _post_state_direction(row)
            expression = _expression_direction(judgement["expression_stance"])
            if post_state is not None and expression is not None:
                post_agreement.append(post_state == expression)
                post_agreement_agents.append(str(row["agent_id"]))
                post_confusion[(post_state, expression)] += 1
    def rate(values: list[bool], clusters: list[str], label: str):
        numeric = [float(value) for value in values]
        return {"rate": float(np.mean(numeric)) if numeric else None, "bootstrap_95pct_interval": cluster_bootstrap_interval(numeric, clusters) if numeric else None, "eligible_count": len(values), "status": "ok" if values else f"NA_no_eligible_{label}"}
    result = {"direct_context_uptake": rate(context, context_agents, "context_labels"), "unsupported_fact_fabrication": rate(unsupported, unsupported_agents, "fact_labels"), "expression_pre_state_agreement": rate(agreement, agreement_agents, "agreement_labels"), "decision_post_state_expression_consistency": rate(post_agreement, post_agreement_agents, "post_state_agreement_labels")}
    result["expression_pre_state_agreement"].update({"applicable_record_count": applicable, "expression_coverage_rate": expressed / len(rows) if rows else None, "confusion": {f"{a}->{b}": count for (a, b), count in sorted(confusion.items())}})
    result["decision_post_state_expression_consistency"].update({"applicable_record_count": post_applicable, "expression_coverage_rate": expressed / len(rows) if rows else None, "confusion": {f"{a}->{b}": count for (a, b), count in sorted(post_confusion.items())}, "interpretation": "exploratory_not_causal"})
    return result


def _direction(value: float, tolerance: float = ALIGNMENT_DEADBAND) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def score_particle_alignment(rows: list[dict[str, Any]], assimilation_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    if assimilation_rows is None:
        return {"rate": None, "tested_direction_count": 0, "matched_turn_count": 0, "status": "NA_no_assimilation_sidecar"}
    by_case = {str(item["case_id"]): item for item in assimilation_rows}
    agreements: list[bool] = []
    by_field: dict[str, list[bool]] = defaultdict(list)
    matched = 0
    applicable = [row for row in rows if row.get("method") == "campuspulse" and row.get("variant") in {"state_only", "full"}]
    for row in applicable:
        audit = by_case.get(case_id(row))
        if audit is None:
            continue
        before, after = audit.get("before_state"), audit.get("after_state")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("assimilation sidecar requires before_state and after_state")
        matched += 1
        expected: dict[str, int] = {}
        for field, delta_name in (("trust", "trust_delta"), ("satisfaction", "satisfaction_delta")):
            direction = _direction(float(audit.get(delta_name, 0.0)))
            if direction:
                expected[field] = direction
        concern = _direction(float(audit.get("urgency", 0.0)) - float(before["concern"]))
        if concern:
            expected["concern"] = concern
        action = str(row.get("action") or "read")
        activity = _direction(float(action in PUBLIC_ACTIONS) - float(before["activity"]))
        if activity:
            expected["activity"] = activity
        stance = _category(row.get("stance_label") or _stance_from_scalar(row.get("public_stance")), STANCES, STANCE_ALIASES, "stance")
        if stance == "support":
            expected.update({"support": 1, "opposition": -1})
        elif stance == "oppose":
            expected.update({"support": -1, "opposition": 1})
        if str(audit.get("evidence_status") or "") in {"verified", "corrected"} or action == "correct":
            expected.update({"verified_belief": 1, "rumor_belief": -1})
        for field, wanted in expected.items():
            match = _direction(float(after[field]) - float(before[field])) == wanted
            agreements.append(match)
            by_field[field].append(match)
    return {
        "rate": float(np.mean(agreements)) if agreements else None,
        "tested_direction_count": len(agreements),
        "matched_turn_count": matched,
        "applicable_turn_count": len(applicable),
        "turn_coverage_rate": matched / len(applicable) if applicable else None,
        "per_field_rates": {field: float(np.mean(values)) for field, values in sorted(by_field.items())},
        "status": "ok" if agreements else "NA_no_tested_directions",
        "interpretation": "mechanism-consistency diagnostic, not independent predictive validation",
    }


def _validate_panel(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["variant"] for row in rows)
    if set(counts) != set(CONDITIONS) or len(set(counts.values())) != 1:
        raise ValueError("dimension 2 requires an equal non-zero record count for every condition")
    agents = {str(row["agent_id"]) for row in rows}
    if len(agents) < 2:
        raise ValueError("dimension 2 requires at least two unique agents for leave-one-agent-out validation")
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        source_id = str(row.get("source_record_id") or "")
        observation_id = source_id.split(":ablation:", 1)[0]
        condition = str(row["variant"])
        if not observation_id:
            raise ValueError("dimension 2 record is missing source_record_id")
        if condition in paired[observation_id]:
            raise ValueError(f"duplicate condition for observation {observation_id}: {condition}")
        paired[observation_id][condition] = row
    for observation_id, group in paired.items():
        if set(group) != set(CONDITIONS):
            raise ValueError(f"observation {observation_id} is not paired across all conditions")
        identities = {(str(row["agent_id"]), int(row["tick"]), int(row["seed"])) for row in group.values()}
        if len(identities) != 1:
            raise ValueError(f"paired identity drift for observation {observation_id}")


def evaluate(rows: list[dict[str, Any]], annotation_rows: list[dict[str, Any]] | None = None, permutation_count: int = 999, permutation_seed: int = 20260722, belief_state_sidecar: dict[str, float] | None = None, assimilation_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = [row for row in rows if row.get("variant") in CONDITIONS]
    _validate_panel(rows)
    condition_labels = np.asarray([row["variant"] for row in rows], dtype=object)
    agent_array = np.asarray([str(row["agent_id"]) for row in rows], dtype=object)
    vectors = np.stack([feature_vector(row) for row in rows])
    observed = _task_scores_from_features(vectors, agent_array, condition_labels)
    agent_indices = {agent: np.flatnonzero(agent_array == agent) for agent in sorted(set(agent_array))}
    null = {name: {"accuracy": [], "macro_f1": []} for name in observed}
    rng = np.random.default_rng(permutation_seed)
    shuffled_labels = []
    for _ in range(permutation_count):
        shuffled = condition_labels.copy()
        for indices in agent_indices.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        shuffled_labels.append(shuffled)
    # Each permutation is independent.  Reuse the immutable feature matrix and
    # distribute exact LOAO fits across a bounded number of local processes.
    workers = max(1, min(8, (os.cpu_count() or 1)))
    null_results = Parallel(n_jobs=workers, prefer="processes")(
        delayed(_null_metrics)(vectors, agent_array, shuffled) for shuffled in shuffled_labels
    )
    for scores in null_results:
        for name, (accuracy, macro_f1) in scores.items():
            null[name]["accuracy"].append(accuracy)
            null[name]["macro_f1"].append(macro_f1)
    classifications = {}
    for name, score in observed.items():
        classifications[name] = {
            "accuracy": score["accuracy"], "macro_f1": score["macro_f1"],
            "accuracy_p_one_sided": (1 + np.sum(np.asarray(null[name]["accuracy"]) >= score["accuracy"])) / (permutation_count + 1),
            "macro_f1_p_one_sided": (1 + np.sum(np.asarray(null[name]["macro_f1"]) >= score["macro_f1"])) / (permutation_count + 1),
            "null_accuracy_mean": float(np.mean(null[name]["accuracy"])), "null_accuracy_q95": float(np.quantile(null[name]["accuracy"], 0.95)),
        }
    folds = [{"case_id": case_id(row), "agent_id": row["agent_id"], "condition": row["variant"], **{f"{name}_truth": json_safe(observed[name]["truth"][i]) for name in observed}, **{f"{name}_predicted": json_safe(observed[name]["predicted"][i]) for name in observed}} for i, row in enumerate(rows)]
    distributions = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["variant"] == condition]
        distributions[condition] = {
            "action": dict(Counter(_category(row.get("action"), ACTIONS, ACTION_ALIASES, "action") for row in subset)),
            "emotion": dict(Counter(_emotion_category(row.get("emotion")) for row in subset)),
            "stance": dict(Counter(_category(row.get("stance_label") or _stance_from_scalar(row.get("public_stance")), STANCES, STANCE_ALIASES, "stance") for row in subset)),
            "public_or_private": dict(Counter(_category(row.get("public_or_private"), VISIBILITY, {}, "public_or_private") for row in subset)),
        }
        distributions[condition]["proportions"] = {
            family: {key: count / len(subset) for key, count in distributions[condition][family].items()}
            for family in ("action", "emotion", "stance", "public_or_private")
        }
    annotation_by_case = None if annotation_rows is None else {
        str(item["case_id"]): item for item in annotation_rows
    }
    text_metrics_by_condition = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["variant"] == condition]
        condition_annotations = None if annotation_by_case is None else [
            annotation_by_case[case_id(row)]
            for row in condition_rows if case_id(row) in annotation_by_case
        ]
        text_metrics_by_condition[condition] = score_text_judgements(
            condition_rows, condition_annotations, belief_state_sidecar
        )
    return json_safe({"schema_version": "dimension-2-evaluation-v1.6.0", "definitions_version": "1.6.0", "stance_direction_contract": stance_direction_contract(), "record_count": len(rows), "agent_count": len(set(str(row["agent_id"]) for row in rows)), "feature_count": FEATURE_COUNT, "classifier": {"kind": "linear_SVM", "implementation": "sklearn.LinearSVC", "C": 1.0, "dual": "auto", "max_iter": 10000, "validation": "leave_one_agent_out", "permutation_count": permutation_count, "permutation_seed": permutation_seed}, "classification": classifications, "fold_predictions": folds, "distributions": distributions, "text_metrics": score_text_judgements(rows, annotation_rows, belief_state_sidecar), "text_metrics_by_condition": text_metrics_by_condition, "language_style_consistency": language_style_consistency(rows), "language_style_consistency_by_condition": {condition: language_style_consistency([row for row in rows if row["variant"] == condition]) for condition in CONDITIONS}, "llm_particle_update_direction_agreement": score_particle_alignment(rows, assimilation_rows)})


def evaluate_cross_method(rows: list[dict[str, Any]], annotation_rows: list[dict[str, Any]] | None = None, belief_state_sidecar: dict[str, float] | None = None, expected_methods: tuple[str, ...] = METHODS) -> dict[str, Any]:
    grouped = {method: [row for row in rows if row.get("method") == method] for method in expected_methods}
    missing = [method for method, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"missing cross-method dimension-2 records for {missing}")
    reference_keys = None
    results = {}
    for method, values in grouped.items():
        keys = {(str(row["agent_id"]), int(row["tick"]), int(row["seed"])) for row in values}
        if len(keys) != len(values):
            raise ValueError(f"duplicate cross-method records for {method}")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("cross-method dimension-2 records are not paired on agent/tick/seed")
        method_annotations = None if annotation_rows is None else [item for item in annotation_rows if str(item.get("case_id")) in {case_id(row) for row in values}]
        results[method] = {
            "record_count": len(values),
            "model_status": dict(Counter(str(row.get("model_status")) for row in values)),
            "text_metrics": score_text_judgements(values, method_annotations, belief_state_sidecar),
            "language_style_consistency": language_style_consistency(values),
        }
    return json_safe({"schema_version": "dimension-2-cross-method-v1.6.0", "definitions_version": "1.6.0", "stance_direction_contract": stance_direction_contract(), "paired_key_count": len(reference_keys or ()), "methods": results})


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen dimension-2 definitions")
    parser.add_argument("records", type=Path)
    parser.add_argument("--mode", choices=("ablation", "cross_method"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--belief-state-sidecar", type=Path)
    parser.add_argument("--assimilation-sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    annotations = read_jsonl(args.annotations) if args.annotations else None
    sidecar = {str(row["case_id"]): float(row["standardized_pre_state"]) for row in read_jsonl(args.belief_state_sidecar)} if args.belief_state_sidecar else None
    assimilation = read_jsonl(args.assimilation_sidecar) if args.assimilation_sidecar else None
    records = read_jsonl(args.records)
    result = evaluate(records, annotations, args.permutations, belief_state_sidecar=sidecar, assimilation_rows=assimilation) if args.mode == "ablation" else evaluate_cross_method(records, annotations, sidecar)
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
