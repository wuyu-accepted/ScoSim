"""Evidence-bound persona inputs for representative campus-forum simulation."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .representative_taxonomy import RepresentativeTaxonomyError
from .representative_taxonomy import _call_contract
from .semantic_population import (
    SemanticBatchClient,
    SemanticTokenBudget,
    _atomic_write_json,
    _bounded_string_list,
    _bounded_text,
    _canonical_hash,
    _file_identity,
    sample_forum_posts,
)
from .llm_agents import DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL
from .shock_replay import DEFAULT_DEFINITIONS


EVIDENCE_PACK_SCHEMA_VERSION = "1.0.0"
PERSONA_PROFILE_SCHEMA_VERSION = "1.0.0"
PERSONA_PROMPT_VERSION = "1.0.0"
PERSONA_RELEASE_SCHEMA_VERSION = "1.0.0"


def _counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _compact_evidence(
    card: dict[str, Any],
    assignment: dict[str, Any],
    *,
    month: str,
    evidence_role: str,
) -> dict[str, Any]:
    return {
        "record_id": card["record_id"],
        "input_fingerprint": card["input_fingerprint"],
        "month": month,
        "evidence_role": evidence_role,
        "normalized_meaning": card["normalized_meaning"],
        "primary_need": card["implicit_need"],
        "secondary_needs": card.get("secondary_needs", []),
        "situational_pressures": card.get("situational_pressures", []),
        "desired_response": card["desired_response"],
        "speaker_relation": card["speaker_relation"],
        "speech_act": card["speech_act"],
        "stance": card["stance"],
        "risk_level": card["risk_level"],
        "language_form": card["language_form"],
        "semantic_determinacy": card["semantic_determinacy"],
        "secondary_archetype_id": assignment.get("secondary_archetype_id"),
        "release_basis": assignment["release_basis"],
        "reviewer_certainty": assignment["reviewer_certainty"],
    }


def build_persona_evidence_packs(
    *,
    posts_path: str | Path,
    semantic_result_path: str | Path,
    taxonomy_release_path: str | Path,
    output_path: str | Path,
    context_result_path: str | Path | None = None,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    representative_per_archetype: int = 10,
    contrastive_per_archetype: int = 5,
) -> dict[str, Any]:
    """Build cross-time evidence packs without persisting forum text or IDs."""

    if representative_per_archetype < 6 or contrastive_per_archetype < 2:
        raise RepresentativeTaxonomyError("persona evidence selection is too small")
    semantic = json.loads(Path(semantic_result_path).read_text(encoding="utf-8"))
    release = json.loads(Path(taxonomy_release_path).read_text(encoding="utf-8"))
    if semantic.get("status") != "complete" or release.get("status") != "complete":
        raise RepresentativeTaxonomyError("persona evidence inputs must be complete")
    cards = {item["record_id"]: item for item in semantic.get("records", [])}
    context_identity = None
    context_ids: set[str] = set()
    if context_result_path is not None:
        context = json.loads(Path(context_result_path).read_text(encoding="utf-8"))
        if context.get("status") != "complete" or context.get(
            "input_evidence", {}
        ).get("source_semantic_run_fingerprint") != semantic.get("run_fingerprint"):
            raise RepresentativeTaxonomyError("persona context evidence mismatch")
        for item in context.get("records", []):
            record_id = item.get("record_id")
            after = item.get("after_card")
            if record_id not in cards or not isinstance(after, dict):
                raise RepresentativeTaxonomyError("invalid persona context overlay")
            if after.get("input_fingerprint") != cards[record_id].get(
                "input_fingerprint"
            ):
                raise RepresentativeTaxonomyError("persona context fingerprint mismatch")
            cards[record_id] = after
            context_ids.add(record_id)
        context_identity = _file_identity(context_result_path)

    assignments = {
        item["record_id"]: item for item in release.get("assignments", [])
    }
    if not cards or set(cards) != set(assignments):
        raise RepresentativeTaxonomyError("persona evidence coverage mismatch")
    sampling = semantic.get("sampling", {})
    fit_scope = sampling.get("fit_scope", {})
    posts, sample_audit = sample_forum_posts(
        posts_path,
        definitions_path=definitions_path,
        sample_size=int(sampling.get("sample_size", len(cards))),
        seed=int(sampling.get("seed")),
        start=fit_scope.get("start_inclusive"),
        end=fit_scope.get("end_exclusive"),
        platform_marker=fit_scope.get("platform_marker", "NP"),
    )
    post_map = {item.record_id: item for item in posts}
    if set(post_map) != set(cards) or any(
        post_map[record_id].input_fingerprint != cards[record_id]["input_fingerprint"]
        for record_id in cards
    ):
        raise RepresentativeTaxonomyError("persona sample reconstruction mismatch")
    if sample_audit.get("sample_manifest_fingerprint") != sampling.get(
        "sample_manifest_fingerprint"
    ):
        raise RepresentativeTaxonomyError("persona sample manifest mismatch")

    taxonomy = release["taxonomy"]
    labels = {
        item["archetype_id"]: item["label"]
        for item in taxonomy.get("archetypes", [])
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for record_id, assignment in assignments.items():
        grouped[assignment["primary_archetype_id"]].append(record_id)
    if set(grouped) != set(labels):
        raise RepresentativeTaxonomyError("every persona archetype needs support")

    packs = []
    for archetype in taxonomy["archetypes"]:
        archetype_id = archetype["archetype_id"]
        record_ids = grouped[archetype_id]
        by_month: dict[str, list[str]] = defaultdict(list)
        for record_id in record_ids:
            by_month[post_map[record_id].month].append(record_id)

        def quality(record_id: str) -> tuple[int, int, float, str]:
            assignment = assignments[record_id]
            direct = assignment["release_basis"] == "direct_semantic_adjudication"
            certainty = {"high": 3, "medium": 2, "low": 1}.get(
                assignment.get("reviewer_certainty"), 0
            )
            base = assignment["base_model_assignment"]
            return (
                int(direct),
                certainty,
                float(base.get("confidence", 0.0)),
                record_id,
            )

        representative: list[str] = []
        for month in sorted(by_month):
            candidate = max(by_month[month], key=quality)
            representative.append(candidate)
        for record_id in sorted(record_ids, key=quality, reverse=True):
            if len(representative) >= representative_per_archetype:
                break
            if record_id not in representative:
                representative.append(record_id)
        representative = representative[:representative_per_archetype]

        contrastive_candidates = sorted(
            record_ids,
            key=lambda record_id: (
                assignments[record_id].get("secondary_archetype_id") is None,
                assignments[record_id]["base_model_assignment"].get(
                    "confidence", 1.0
                ),
                record_id,
            ),
        )
        contrastive = []
        for record_id in contrastive_candidates:
            if len(contrastive) >= contrastive_per_archetype:
                break
            if record_id not in representative:
                contrastive.append(record_id)
        if len(contrastive) < contrastive_per_archetype:
            for record_id in contrastive_candidates:
                if len(contrastive) >= contrastive_per_archetype:
                    break
                if record_id not in contrastive:
                    contrastive.append(record_id)

        all_cards = [cards[record_id] for record_id in record_ids]
        pack = {
            "archetype_id": archetype_id,
            "label": labels[archetype_id],
            "taxonomy_definition": archetype,
            "support_count": len(record_ids),
            "month_support": dict(
                sorted(Counter(post_map[r].month for r in record_ids).items())
            ),
            "observable_distributions": {
                "speech_act": _counter([card["speech_act"] for card in all_cards]),
                "stance": _counter([card["stance"] for card in all_cards]),
                "language_form": _counter(
                    [card["language_form"] for card in all_cards]
                ),
                "speaker_relation": _counter(
                    [card["speaker_relation"] for card in all_cards]
                ),
                "risk_level": _counter([card["risk_level"] for card in all_cards]),
                "semantic_determinacy": _counter(
                    [card["semantic_determinacy"] for card in all_cards]
                ),
                "situational_pressures": _counter(
                    [
                        pressure
                        for card in all_cards
                        for pressure in card.get("situational_pressures", [])
                    ]
                ),
            },
            "representative_evidence": [
                _compact_evidence(
                    cards[record_id],
                    assignments[record_id],
                    month=post_map[record_id].month,
                    evidence_role="cross_time_representative",
                )
                for record_id in representative
            ],
            "contrastive_evidence": [
                _compact_evidence(
                    cards[record_id],
                    assignments[record_id],
                    month=post_map[record_id].month,
                    evidence_role="boundary_or_contrast",
                )
                for record_id in contrastive
            ],
            "context_refined_evidence_ids": sorted(
                set(representative + contrastive) & context_ids
            ),
        }
        pack["evidence_fingerprint"] = _canonical_hash(
            {
                "archetype_id": archetype_id,
                "representative": [
                    (item["record_id"], item["input_fingerprint"])
                    for item in pack["representative_evidence"]
                ],
                "contrastive": [
                    (item["record_id"], item["input_fingerprint"])
                    for item in pack["contrastive_evidence"]
                ],
            }
        )
        packs.append(pack)

    result = {
        "persona_evidence_pack_schema_version": EVIDENCE_PACK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "run_fingerprint": _canonical_hash(
            {
                "semantic": semantic.get("run_fingerprint"),
                "taxonomy_release": release.get("run_fingerprint"),
                "context": context_identity,
                "representative_per_archetype": representative_per_archetype,
                "contrastive_per_archetype": contrastive_per_archetype,
            }
        ),
        "input_evidence": {
            "semantic_result": _file_identity(semantic_result_path),
            "taxonomy_release": _file_identity(taxonomy_release_path),
            "context_result": context_identity,
            "posts_csv": _file_identity(posts_path),
            "paths_persisted": False,
        },
        "privacy": {
            "raw_forum_text_persisted": False,
            "source_ids_persisted": False,
            "semantic_cards_only": True,
        },
        "packs": packs,
        "summary": {
            "archetype_count": len(packs),
            "record_count": len(cards),
            "representative_evidence_count": sum(
                len(pack["representative_evidence"]) for pack in packs
            ),
            "contrastive_evidence_count": sum(
                len(pack["contrastive_evidence"]) for pack in packs
            ),
            "months": sorted({post.month for post in posts}),
        },
        "validity": {
            "evidence_pack_is_persona": False,
            "persistent_real_person_identity_claimed": False,
            "demographics_inferred": False,
            "direct_persona_text_review_required": True,
        },
    }
    _atomic_write_json(result, output_path)
    return result


def persona_system_prompt() -> str:
    return (
        "你是校园论坛社会模拟的人群画像设计器。输入只代表行为episode，不代表持久真实用户。"
        "只能概括证据中可观察的触发条件、诉求、回应偏好与语言倾向。"
        "严禁推断性别、年级、专业、家庭、阶层、疾病诊断、政治身份、人格类型或稳定人生经历。"
        "不得复述姓名、课程简称、账号、联系方式或原句。只输出JSON对象。"
    )


def persona_prompt(pack: dict[str, Any]) -> str:
    evidence_ids = [
        item["record_id"]
        for key in ("representative_evidence", "contrastive_evidence")
        for item in pack[key]
    ]
    payload = {
        "task": "Create an evidence-bound behavioral persona with three within-class variants.",
        "rules": [
            "画像描述一种可被情境激活的论坛互动角色，不声称是一名真实持久用户。",
            "先提炼跨月份稳定核心，再区分类内三个有证据的变体；不能用人口属性区分。",
            "每个变体必须引用2到5个输入evidence_record_ids，且说明处境压力、反应逻辑和语言方式。",
            "语言倾向只能概括口语程度、长短、情绪强度、提问或叙事方式，不得模仿原句或专名。",
            "contrastive_evidence用于写清不应模拟成什么，不能为了整齐把边界差异抹平。",
            "simulation_instruction应能直接放入智能体prompt，要求按证据角色行动并允许不发言。",
            "不得发明政策偏好、传播概率、信任分数、人口属性或因果反应。",
        ],
        "allowed_evidence_record_ids": evidence_ids,
        "output_schema": {
            "archetype_id": pack["archetype_id"],
            "profile_summary": "20-180 Chinese characters",
            "observable_core_tendencies": ["3-6 items, <=80 characters"],
            "trigger_conditions": ["3-6 items, <=80 characters"],
            "response_expectations": ["2-5 items, <=80 characters"],
            "language_tendencies": ["3-6 items, <=80 characters"],
            "variants": [
                {
                    "variant_id": "<archetype_id>-v1/v2/v3",
                    "label": "4-18 Chinese characters",
                    "evidence_record_ids": ["2-5 allowed ids"],
                    "situational_signature": "<=100 Chinese characters",
                    "reaction_logic": "<=140 Chinese characters",
                    "language_style": "<=100 Chinese characters",
                    "uncertainty_behavior": "<=100 Chinese characters",
                    "simulation_instruction": "80-360 Chinese characters",
                }
            ],
            "do_not_simulate": ["3-7 unsupported or contrastive behaviors, <=90 characters"],
            "evidence_limitations": ["2-5 limitations, <=100 characters"],
        },
        "evidence_pack": pack,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _required_string_list(
    value: Any,
    field: str,
    *,
    minimum_items: int,
    maximum_items: int,
    maximum_length: int,
) -> list[str]:
    result = _bounded_string_list(
        value,
        field,
        maximum_items=maximum_items,
        maximum_length=maximum_length,
    )
    if len(result) < minimum_items:
        raise RepresentativeTaxonomyError(f"{field} has too few items")
    return result


def validate_persona_profile(
    raw: dict[str, Any], *, pack: dict[str, Any]
) -> dict[str, Any]:
    archetype_id = pack["archetype_id"]
    if raw.get("archetype_id") != archetype_id:
        raise RepresentativeTaxonomyError("persona archetype id mismatch")
    allowed_ids = {
        item["record_id"]
        for key in ("representative_evidence", "contrastive_evidence")
        for item in pack[key]
    }
    variants = raw.get("variants")
    if not isinstance(variants, list) or len(variants) != 3:
        raise RepresentativeTaxonomyError("persona must have exactly three variants")
    normalized_variants = []
    used_ids = set()
    for index, variant in enumerate(variants, 1):
        expected_id = f"{archetype_id}-v{index}"
        if not isinstance(variant, dict) or variant.get("variant_id") != expected_id:
            raise RepresentativeTaxonomyError("persona variant ids must be sequential")
        evidence_ids = variant.get("evidence_record_ids")
        if not isinstance(evidence_ids, list) or not 2 <= len(evidence_ids) <= 5:
            raise RepresentativeTaxonomyError("persona variant evidence count invalid")
        if len(set(evidence_ids)) != len(evidence_ids) or not set(
            evidence_ids
        ).issubset(allowed_ids):
            raise RepresentativeTaxonomyError("persona variant evidence ids invalid")
        used_ids.update(evidence_ids)
        normalized_variants.append(
            {
                "variant_id": expected_id,
                "label": _bounded_text(variant.get("label"), "label", 18),
                "evidence_record_ids": evidence_ids,
                "situational_signature": _bounded_text(
                    variant.get("situational_signature"),
                    "situational_signature",
                    100,
                ),
                "reaction_logic": _bounded_text(
                    variant.get("reaction_logic"), "reaction_logic", 140
                ),
                "language_style": _bounded_text(
                    variant.get("language_style"), "language_style", 100
                ),
                "uncertainty_behavior": _bounded_text(
                    variant.get("uncertainty_behavior"),
                    "uncertainty_behavior",
                    100,
                ),
                "simulation_instruction": _bounded_text(
                    variant.get("simulation_instruction"),
                    "simulation_instruction",
                    360,
                ),
            }
        )
    if len(used_ids) < 6:
        raise RepresentativeTaxonomyError("persona variants reuse too little evidence")
    return {
        "persona_profile_schema_version": PERSONA_PROFILE_SCHEMA_VERSION,
        "archetype_id": archetype_id,
        "profile_summary": _bounded_text(
            raw.get("profile_summary"), "profile_summary", 180
        ),
        "observable_core_tendencies": _required_string_list(
            raw.get("observable_core_tendencies"),
            "observable_core_tendencies",
            minimum_items=3,
            maximum_items=6,
            maximum_length=80,
        ),
        "trigger_conditions": _required_string_list(
            raw.get("trigger_conditions"),
            "trigger_conditions",
            minimum_items=3,
            maximum_items=6,
            maximum_length=80,
        ),
        "response_expectations": _required_string_list(
            raw.get("response_expectations"),
            "response_expectations",
            minimum_items=2,
            maximum_items=5,
            maximum_length=80,
        ),
        "language_tendencies": _required_string_list(
            raw.get("language_tendencies"),
            "language_tendencies",
            minimum_items=3,
            maximum_items=6,
            maximum_length=80,
        ),
        "variants": normalized_variants,
        "do_not_simulate": _required_string_list(
            raw.get("do_not_simulate"),
            "do_not_simulate",
            minimum_items=3,
            maximum_items=7,
            maximum_length=90,
        ),
        "evidence_limitations": _required_string_list(
            raw.get("evidence_limitations"),
            "evidence_limitations",
            minimum_items=2,
            maximum_items=5,
            maximum_length=100,
        ),
    }


def run_persona_profile_pilot(
    *,
    evidence_pack_path: str | Path,
    output_path: str | Path,
    max_concurrency: int = 8,
    max_total_tokens: int = 180000,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    max_output_tokens: int = 5000,
    timeout_seconds: float = 180.0,
    transport: Any = None,
) -> dict[str, Any]:
    evidence = json.loads(Path(evidence_pack_path).read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or not evidence.get("packs"):
        raise RepresentativeTaxonomyError("persona evidence pack must be complete")
    packs = evidence["packs"]
    if not 1 <= max_concurrency <= 12:
        raise RepresentativeTaxonomyError("invalid persona concurrency")
    contract = {
        "evidence_pack": _file_identity(evidence_pack_path),
        "evidence_run_fingerprint": evidence.get("run_fingerprint"),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "persona_prompt_version": PERSONA_PROMPT_VERSION,
    }
    run_fingerprint = _canonical_hash(contract)
    client = SemanticBatchClient.from_environment(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        transport=transport,
    )
    output = Path(output_path)
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("run_fingerprint") != run_fingerprint:
            raise RepresentativeTaxonomyError("persona checkpoint belongs to another run")
    else:
        result = {
            "persona_pilot_schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "partial",
            "run_fingerprint": run_fingerprint,
            "input_evidence": {
                "evidence_pack": _file_identity(evidence_pack_path),
                "paths_persisted": False,
            },
            "provider": {
                "base_url": client.base_url,
                "endpoint": "/chat/completions",
                "requested_model": client.model,
                "api_key_environment_variable": api_key_env,
                "api_key_persisted": False,
                "raw_forum_text_sent": False,
                "semantic_evidence_sent": True,
            },
            "execution": {
                "max_concurrency": max_concurrency,
                "max_total_tokens": max_total_tokens,
                "completed_archetypes": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "provider_attempts": 0,
                    "format_repairs": 0,
                },
            },
            "profiles": [],
            "validity": {
                "persona_is_persistent_real_person": False,
                "demographics_inferred": False,
                "direct_text_review_required": True,
                "automatic_metrics_are_fidelity_ground_truth": False,
            },
        }
    usage = result["execution"]["usage"]
    snapshot = result["execution"].get("token_budget", {})
    budget = SemanticTokenBudget(
        max_total_tokens,
        initial_spent=max(
            int(usage["total_tokens"]),
            int(snapshot.get("provider_reported_or_reserved_tokens", 0)),
        ),
        initial_calls=max(
            int(usage["provider_attempts"]), int(snapshot.get("provider_calls", 0))
        ),
    )
    complete = {item["archetype_id"] for item in result["profiles"]}
    pending = [pack for pack in packs if pack["archetype_id"] not in complete]
    failures = []
    try:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(
                    _call_contract,
                    client=client,
                    budget=budget,
                    user_prompt=persona_prompt(pack),
                    validator=lambda raw, target=pack: validate_persona_profile(
                        raw, pack=target
                    ),
                    system_message=persona_system_prompt(),
                ): pack
                for pack in pending
            }
            for future in as_completed(futures):
                pack = futures[future]
                try:
                    profile, call_usage = future.result()
                except Exception as exc:
                    failures.append((pack["archetype_id"], exc))
                    continue
                profile["evidence_fingerprint"] = pack["evidence_fingerprint"]
                result["profiles"].append(profile)
                result["profiles"].sort(key=lambda item: item["archetype_id"])
                for key in usage:
                    usage[key] += int(call_usage.get(key, 0))
                result["execution"]["completed_archetypes"].append(
                    {"archetype_id": pack["archetype_id"], "usage": call_usage}
                )
                result["execution"]["completed_archetypes"].sort(
                    key=lambda item: item["archetype_id"]
                )
                _atomic_write_json(result, output)
        if failures:
            raise RepresentativeTaxonomyError(
                f"{len(failures)} persona calls failed after checkpointing; first error: {failures[0][1]}"
            )
    except Exception:
        result["status"] = "partial"
        result["execution"]["token_budget"] = budget.snapshot()
        _atomic_write_json(result, output)
        raise
    result["status"] = "complete"
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["execution"]["token_budget"] = budget.snapshot()
    result["summary"] = {
        "profile_count": len(result["profiles"]),
        "variant_count": sum(len(item["variants"]) for item in result["profiles"]),
        "evidence_ids_used": len(
            {
                record_id
                for profile in result["profiles"]
                for variant in profile["variants"]
                for record_id in variant["evidence_record_ids"]
            }
        ),
    }
    _atomic_write_json(result, output)
    return result


def _profile_text_values(value: Any, *, key: str | None = None):
    """Yield human-facing profile strings, excluding IDs and fingerprints."""

    excluded = {
        "archetype_id",
        "variant_id",
        "evidence_record_ids",
        "evidence_fingerprint",
    }
    if key in excluded:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _profile_text_values(item, key=key)
    elif isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _profile_text_values(child_value, key=child_key)


def _persona_privacy_scan(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = {
        "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "url": re.compile(r"https?://|www\.", re.IGNORECASE),
    }
    matches = {name: 0 for name in patterns}
    for profile in profiles:
        for text in _profile_text_values(profile):
            for name, pattern in patterns.items():
                matches[name] += len(pattern.findall(text))
    return {
        "status": "pass" if not any(matches.values()) else "fail",
        "contact_or_url_match_counts": matches,
        "raw_forum_text_scanned": False,
    }


def release_reviewed_personas(
    *,
    persona_pilot_path: str | Path,
    evidence_pack_path: str | Path,
    review_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply direct human/agent text adjudication to every generated persona."""

    pilot = json.loads(Path(persona_pilot_path).read_text(encoding="utf-8"))
    evidence = json.loads(Path(evidence_pack_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_manifest_path).read_text(encoding="utf-8"))
    if pilot.get("status") != "complete" or evidence.get("status") != "complete":
        raise RepresentativeTaxonomyError("persona release inputs must be complete")
    if review.get("persona_pilot_run_fingerprint") != pilot.get("run_fingerprint"):
        raise RepresentativeTaxonomyError("persona review manifest fingerprint mismatch")
    profiles = {item["archetype_id"]: item for item in pilot.get("profiles", [])}
    packs = {item["archetype_id"]: item for item in evidence.get("packs", [])}
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise RepresentativeTaxonomyError("persona review decisions missing")
    by_id = {item.get("archetype_id"): item for item in decisions}
    if set(by_id) != set(profiles) or set(packs) != set(profiles):
        raise RepresentativeTaxonomyError("every persona must be directly reviewed")

    allowed_profile_overrides = {
        "profile_summary",
        "observable_core_tendencies",
        "trigger_conditions",
        "response_expectations",
        "language_tendencies",
        "do_not_simulate",
        "evidence_limitations",
    }
    allowed_variant_overrides = {
        "label",
        "evidence_record_ids",
        "situational_signature",
        "reaction_logic",
        "language_style",
        "uncertainty_behavior",
        "simulation_instruction",
    }
    released = []
    for archetype_id in sorted(profiles):
        decision = by_id[archetype_id]
        if decision.get("decision") not in {"accept", "revise"}:
            raise RepresentativeTaxonomyError("invalid persona review decision")
        if not decision.get("review_rationale"):
            raise RepresentativeTaxonomyError("persona review rationale missing")
        candidate = copy.deepcopy(profiles[archetype_id])
        overrides = decision.get("profile_overrides", {})
        if not isinstance(overrides, dict) or not set(overrides).issubset(
            allowed_profile_overrides
        ):
            raise RepresentativeTaxonomyError("invalid persona profile override")
        candidate.update(overrides)
        variants = {item["variant_id"]: item for item in candidate["variants"]}
        variant_overrides = decision.get("variant_overrides", {})
        if not isinstance(variant_overrides, dict):
            raise RepresentativeTaxonomyError("invalid persona variant overrides")
        for variant_id, patch in variant_overrides.items():
            if variant_id not in variants or not isinstance(patch, dict):
                raise RepresentativeTaxonomyError("unknown persona variant override")
            if not set(patch).issubset(allowed_variant_overrides):
                raise RepresentativeTaxonomyError("unsupported persona variant field")
            variants[variant_id].update(patch)
        candidate["variants"] = [
            variants[f"{archetype_id}-v{index}"] for index in range(1, 4)
        ]
        validated = validate_persona_profile(candidate, pack=packs[archetype_id])
        validated["evidence_fingerprint"] = candidate["evidence_fingerprint"]
        validated["direct_review"] = {
            "decision": decision["decision"],
            "review_rationale": decision["review_rationale"],
            "simulation_mode": decision.get(
                "simulation_mode", "active_episode_agent"
            ),
            "eligible_for_population_agent": bool(
                decision.get("eligible_for_population_agent", True)
            ),
        }
        released.append(validated)

    privacy_scan = _persona_privacy_scan(released)
    if privacy_scan["status"] != "pass":
        raise RepresentativeTaxonomyError("persona release privacy scan failed")
    result = {
        "persona_release_schema_version": PERSONA_RELEASE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "run_fingerprint": _canonical_hash(
            {
                "persona_pilot": _file_identity(persona_pilot_path),
                "evidence_pack": _file_identity(evidence_pack_path),
                "review_manifest": _file_identity(review_manifest_path),
            }
        ),
        "input_evidence": {
            "persona_pilot": _file_identity(persona_pilot_path),
            "evidence_pack": _file_identity(evidence_pack_path),
            "direct_review_manifest": _file_identity(review_manifest_path),
            "paths_persisted": False,
        },
        "profiles": released,
        "privacy_scan": privacy_scan,
        "summary": {
            "profile_count": len(released),
            "variant_count": sum(len(item["variants"]) for item in released),
            "accepted_without_text_revision": sum(
                item["direct_review"]["decision"] == "accept" for item in released
            ),
            "revised_after_direct_reading": sum(
                item["direct_review"]["decision"] == "revise" for item in released
            ),
            "population_agent_eligible_profiles": sum(
                item["direct_review"]["eligible_for_population_agent"]
                for item in released
            ),
        },
        "validity": {
            "all_profiles_directly_read": True,
            "automatic_metrics_used_as_fidelity_ground_truth": False,
            "persona_is_persistent_real_person": False,
            "demographics_inferred": False,
            "ambiguous_noise_is_separate_from_population_agents": True,
        },
    }
    _atomic_write_json(result, output_path)
    return result
