from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from campuspulse_core.governance_arena_v2 import (
    STATE_FIELDS,
    ParticleEnsemble,
    assimilate_forum_llm_updates_v2,
)


ASSIMILATION_VERSION = "campuspulse-action-assimilation-v1.0.0"
STANCE_DEADBAND = 0.05
URGENCY_BY_EMOTION = {
    "calm": 0.15,
    "neutral": 0.25,
    "curious": 0.35,
    "supportive": 0.40,
    "skeptical": 0.58,
    "concerned": 0.62,
    "frustrated": 0.78,
    "urgent": 0.90,
}

# The shared output schema permits natural-language emotion labels, while the
# native update consumes this frozen eight-level urgency vocabulary.  Raw
# labels remain unchanged in the unified record; only assimilation uses this
# deterministic lexical projection.
EMOTION_KEYWORD_ALIASES = (
    (("紧急", "急迫", "焦急"), "urgent"),
    (("沮丧", "挫败", "愤怒", "恼火", "不满"), "frustrated"),
    (("担忧", "担心", "忧虑", "焦虑", "关切", "不安"), "concerned"),
    (("怀疑", "质疑", "审慎", "谨慎", "求证", "理性", "犹豫"), "skeptical"),
    (("支持", "赞同", "认同"), "supportive"),
    (("好奇", "求知", "困惑", "疑惑", "疑问", "询问", "迷茫"), "curious"),
    (("平静", "冷静"), "calm"),
    (("中立", "中性", "无明显情绪", "无明确情绪", "无特别情绪", "观望", "等待", "无事可说", "无所事事", "期待", "保留"), "neutral"),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _decision_identity(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": decision.get("action"),
        "emotion": decision.get("emotion"),
        "public_stance": decision.get("public_stance"),
        "public_or_private": decision.get("public_or_private"),
        "text": decision.get("text", decision.get("generated_text")),
    }


def _state(values: np.ndarray) -> dict[str, float]:
    return {field: float(value) for field, value in zip(STATE_FIELDS, values, strict=True)}


def project_shared_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Project the frozen shared action into the native ForumTwin update contract.

    Only fields observed in the shared action are used.  Trust, satisfaction and
    evidence status are not inferred from prose: their deltas remain zero and
    evidence remains unverified.  This prevents hidden method-specific judging.
    """

    public_stance = decision.get("public_stance")
    scalar = 0.0 if public_stance is None else float(public_stance)
    stance = "support" if scalar > STANCE_DEADBAND else "oppose" if scalar < -STANCE_DEADBAND else "uncertain"
    emotion = str(decision.get("emotion") or "neutral").strip().lower()
    # A small number of legacy cached responses were decoded with a lossy
    # console code page (the Chinese label “不确定” appears as “��ȷ��”).
    # Treat that exact artifact as the schema's neutral/uncertain emotion;
    # do not alter the raw response retained in the audit record.
    exact_aliases = {"关注": "concerned", "��ȷ��": "neutral"}
    exact_aliases.update({"uncertain": "neutral", "reserved": "neutral", "waiting": "neutral", "cautious concern": "concerned"})
    emotion = exact_aliases.get(emotion, emotion)
    # Defensive handling for any replacement-character mojibake in cached
    # Chinese neutral labels (different consoles produced slightly different
    # byte-loss variants).
    if "�" in emotion or "ȷ" in emotion:
        emotion = "neutral"
    if emotion not in URGENCY_BY_EMOTION:
        emotion = next(
            (
                native
                for keywords, native in EMOTION_KEYWORD_ALIASES
                if any(keyword in emotion for keyword in keywords)
            ),
            emotion,
        )
    if emotion not in URGENCY_BY_EMOTION:
        # Shared JSONL keeps the original natural-language label, but native
        # assimilation must never fail on a legacy/unknown label.  Project
        # such labels to the neutral urgency bucket and let the raw field
        # remain available for audit/review.
        emotion = "neutral"
    return {
        "action": str(decision.get("action") or "read"),
        "stance": stance,
        "urgency": URGENCY_BY_EMOTION[emotion],
        "trust_delta": 0.0,
        "satisfaction_delta": 0.0,
        "confidence": min(1.0, max(0.0, abs(scalar))),
        "evidence_status": "unverified",
    }


def assimilate_isolated_turn(
    *,
    particles: Sequence[Mapping[str, float]],
    particle_weights: Sequence[float],
    decision: Mapping[str, Any],
    scenario_id: str,
    variant: str,
    tick: int,
    seed: int,
    prompt_sha256: str,
) -> dict[str, Any]:
    if len(particles) != 10 or len(particle_weights) != 10:
        raise ValueError("native dimension-2 assimilation requires ten particles and ten weights")
    values = np.asarray([[float(row[field]) for field in STATE_FIELDS] for row in particles], dtype=np.float32)
    weights = np.asarray(particle_weights, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError("assimilation particles or weights are invalid")
    weights = weights / weights.sum()
    ensemble = ParticleEnsemble(values=values[None, :, :], weights=weights[None, :], parent_weights=np.asarray([1.0], dtype=np.float64))
    ensemble.assert_conserved()
    before_state = _state(ensemble.parent_means()[0])
    before_particles = [_state(row) for row in ensemble.values[0]]
    before_weights = ensemble.normalized_within_parent()[0].tolist()
    native_decision = project_shared_decision(decision)
    response_sha256 = _sha256(_decision_identity(decision))
    update = assimilate_forum_llm_updates_v2(
        state=SimpleNamespace(ensemble=ensemble),
        updates={0: {"prompt_sha256": prompt_sha256, "response_sha256": response_sha256, "decision": native_decision}},
        scenario_id=scenario_id,
        branch=f"dimension2-{variant}",
        tick=tick,
        seed=seed,
    )
    after_state = _state(ensemble.parent_means()[0])
    after_particles = [_state(row) for row in ensemble.values[0]]
    after_weights = ensemble.normalized_within_parent()[0].tolist()
    return {
        "assimilation_version": ASSIMILATION_VERSION,
        "decision_projection": native_decision,
        "decision_sha256": response_sha256,
        "prompt_sha256": prompt_sha256,
        "before_state": before_state,
        "after_state": after_state,
        "before_particles_sha256": _sha256({"particles": before_particles, "weights": before_weights}),
        "after_particles_sha256": _sha256({"particles": after_particles, "weights": after_weights}),
        "after_particles": after_particles,
        "after_particle_weights": after_weights,
        "innovation_id": update["innovation_ids"][0],
        "resampled": bool(update["resampling_count"]),
        "minimum_parent_ess": float(update["minimum_parent_ess"]),
        "trust_delta": float(native_decision["trust_delta"]),
        "satisfaction_delta": float(native_decision["satisfaction_delta"]),
        "urgency": float(native_decision["urgency"]),
        "evidence_status": str(native_decision["evidence_status"]),
    }
