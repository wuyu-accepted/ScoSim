from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STATE_FIELDS = (
    "attention", "concern", "trust", "satisfaction", "support", "opposition",
    "activity", "voice", "service_strain", "rumor_belief", "verified_belief",
    "evidence_acceptance",
)
PILOT_SEED = 20260722
DIMENSION_2_CONDITIONS = ("context_only", "profile_only", "state_only", "full")
DIMENSION_3_VARIANTS = ("full", "single_particle", "no_profile", "no_profile_no_smc")


def _state(values: Iterable[float]) -> dict[str, float]:
    return {field: float(value) for field, value in zip(STATE_FIELDS, values, strict=True)}


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        raise ValueError("particle weights must have positive mass")
    return values.astype(np.float64) / total


def adapt_native_turn(
    raw: dict[str, Any], *, experiment_id: str, variant: str, seed: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Map a native CampusPulse resident-turn record without inventing values.

    Missing concepts are null. Callers must supply pre/post state snapshots and
    particle arrays when the selected variant requires them.
    """
    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else raw
    visible = raw.get("visible_context") or raw.get("context") or []
    context = []
    for index, item in enumerate(visible):
        if isinstance(item, str):
            context.append({
                "message_id": f"context-{index:04d}", "author_id": None,
                "text": item, "visibility": "public", "tick": raw.get("tick"),
            })
        elif isinstance(item, dict):
            context.append({
                "message_id": str(item.get("message_id") or f"context-{index:04d}"),
                "author_id": str(item["author_id"]) if item.get("author_id") is not None else None,
                "text": str(item.get("text") or item.get("visible_text") or ""),
                "visibility": str(item.get("visibility") or "public"),
                "tick": item.get("tick"),
            })
    status = str(raw.get("model_status") or "success")
    failure = raw.get("failure")
    if status not in {"success", "schema_repaired"} and failure is None and status not in {"skipped", "not_applicable"}:
        failure = {"stage": "adapter", "code": "native_failure", "message": "native record reports failure", "attempts": 0}
    return {
        "schema_version": "1.0.0",
        "protocol_version": "1.0.0",
        "experiment_id": experiment_id,
        "run_id": run_id or f"{experiment_id}/campuspulse/{variant}/{seed}",
        "method": "campuspulse",
        "variant": variant,
        "seed": int(seed),
        "agent_id": str(raw.get("agent_id") or raw.get("llm_agent_id") or raw.get("source_agent_id")),
        "tick": int(raw["tick"]),
        "profile_id": str(raw["profile_id"]) if raw.get("profile_id") is not None else None,
        "interest_group": str(raw["interest_group"]) if raw.get("interest_group") is not None else None,
        "visible_context": context,
        "pre_state": raw.get("pre_state"),
        "post_state": raw.get("post_state"),
        "particles": raw.get("particles"),
        "particle_weights": raw.get("particle_weights"),
        "generated_text": decision.get("visible_text") or decision.get("generated_text"),
        "action": decision.get("action"),
        "emotion": decision.get("emotion"),
        "public_stance": decision.get("public_stance"),
        "public_or_private": str(decision.get("public_or_private") or "none"),
        "model_status": status,
        "failure": failure,
        "source_record_id": str(raw["record_id"]) if raw.get("record_id") is not None else None,
    }


def _projection_record(
    *, experiment_id: str, variant: str, agent_slot: int, tick: int,
    pre_values: np.ndarray, post_values: np.ndarray, weights: np.ndarray | None,
    particles: np.ndarray | None,
) -> dict[str, Any]:
    mean = post_values.mean(axis=0) if post_values.ndim == 2 else post_values
    public_stance = float(np.clip(mean[4] - mean[5], -1.0, 1.0))
    return {
        "schema_version": "1.0.0", "protocol_version": "1.0.0",
        "experiment_id": experiment_id,
        "run_id": f"{experiment_id}/campuspulse/{variant}/{PILOT_SEED}",
        "method": "campuspulse", "variant": variant, "seed": PILOT_SEED,
        "agent_id": f"anonymous-parent-{agent_slot:04d}", "tick": tick,
        "profile_id": None, "interest_group": None, "visible_context": [],
        "pre_state": _state(pre_values), "post_state": _state(mean),
        "particles": [_state(row) for row in particles] if particles is not None else None,
        "particle_weights": weights.tolist() if weights is not None else None,
        "generated_text": None, "action": None, "emotion": None,
        "public_stance": public_stance, "public_or_private": "none",
        "model_status": "not_applicable", "failure": None,
        "source_record_id": f"adapter_projection:slot={agent_slot}:tick={tick}",
    }


def build_particle_panel_pilot(
    panel_path: Path, manifest_path: Path, *, sample_size: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {(int(item["tick"]), str(item["branch"])): str(item["prefix"]) for item in manifest["frames"]}
    tick_prefix = {
        0: frames[(0, "shared_initial")], 1: frames[(0, "shared_initial")],
        2: frames[(0, "shared_initial")], 3: frames[(3, "shared_baseline")],
        **{tick: frames[(tick, "Natural")] for tick in range(4, 10)},
    }
    rng = np.random.default_rng(PILOT_SEED)
    slots = np.sort(rng.choice(1000, size=sample_size, replace=False))
    d2_slots = slots[:12]
    dimension2: list[dict[str, Any]] = []
    dimension3: list[dict[str, Any]] = []
    with np.load(panel_path) as panel:
        def arrays(tick: int, slot: int):
            prefix = tick_prefix[tick]
            values = panel[f"{prefix}_values"][slot]
            weights = _normalized_weights(panel[f"{prefix}_weights"][slot])
            mean = np.average(values, axis=0, weights=weights)
            return values, weights, mean

        for slot in d2_slots:
            values, weights, mean = arrays(3, int(slot))
            for condition in DIMENSION_2_CONDITIONS:
                has_particles = condition in {"state_only", "full"}
                projected = _projection_record(
                    experiment_id="dimension_2_adapter_pilot", variant=condition,
                    agent_slot=int(slot), tick=3, pre_values=mean, post_values=values,
                    weights=weights if has_particles else None,
                    particles=values if has_particles else None,
                )
                # Plumbing-only projections have no model decision. Use the
                # registered silent behavior categories so the evaluator can
                # exercise its schema without inventing expressive content.
                projected["action"] = "read"
                projected["emotion"] = "neutral"
                dimension2.append(projected)

        for slot in slots:
            previous_mean = arrays(0, int(slot))[2]
            for tick in range(10):
                values, weights, mean = arrays(tick, int(slot))
                for variant in DIMENSION_3_VARIANTS:
                    if variant == "single_particle":
                        chosen_values, chosen_weights = mean[None, :], np.asarray([1.0])
                    elif variant == "no_profile_no_smc":
                        chosen_values, chosen_weights = None, None
                    else:
                        chosen_values, chosen_weights = values, weights
                    dimension3.append(_projection_record(
                        experiment_id="dimension_3_adapter_pilot", variant=variant,
                        agent_slot=int(slot), tick=tick, pre_values=previous_mean,
                        post_values=mean, weights=chosen_weights, particles=chosen_values,
                    ))
                previous_mean = mean
    source_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    metadata = {
        "pilot_seed": PILOT_SEED, "sample_size": sample_size,
        "dimension_2_record_count": len(dimension2),
        "dimension_3_record_count": len(dimension3),
        "source_panel_sha256": source_hash,
        "status": "adapter_projection_only_not_a_paper_result",
        "limitations": [
            "Anonymous particle slots cannot be joined to profiles, messages, or interest groups.",
            "Ticks 1 and 2 repeat the frozen tick-0 state because no intermediate particle frame exists.",
            "Ablation variants are mechanical schema projections, not rerun model worlds.",
            "Dimension-2 records contain no generated text and cannot support fidelity conclusions.",
        ],
    }
    return dimension2, dimension3, metadata


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
