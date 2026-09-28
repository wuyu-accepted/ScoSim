"""Release a representative taxonomy after direct qualitative adjudication.

Model confidence and order agreement are useful triage signals, not fidelity
ground truth.  This module applies a fingerprint-bound semantic review manifest
to a frozen taxonomy run and produces assignments suitable for downstream
persona construction without pretending that the review is a mechanical score.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .representative_taxonomy import (
    RepresentativeTaxonomyError,
    validate_taxonomy,
)
from .semantic_population import (
    _atomic_write_json,
    _canonical_hash,
    _file_identity,
)


REVIEWED_RELEASE_SCHEMA_VERSION = "1.0.0"
TAXONOMY_REVISION_SCHEMA_VERSION = "1.0.0"
EDITABLE_TAXONOMY_FIELDS = {
    "label",
    "core_need",
    "typical_pressures",
    "expected_response",
    "inclusion_rules",
    "exclusion_rules",
    "distinguish_from_nearest",
    "governance_relevance",
}
REVIEWER_CERTAINTY = {"low", "medium", "high"}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RepresentativeTaxonomyError("expected a JSON object")
    return value


def release_reviewed_taxonomy(
    *,
    semantic_result_path: str | Path,
    taxonomy_result_path: str | Path,
    hard_pair_result_path: str | Path,
    adjudication_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply direct text-review decisions to one exact 100-record pilot.

    Every decision is bound to both a semantic-run fingerprint and the
    per-record input fingerprint.  A later sample or regenerated semantic card
    therefore cannot silently inherit an old qualitative judgment.
    """

    semantic = _read_json(semantic_result_path)
    taxonomy_run = _read_json(taxonomy_result_path)
    hard_pairs = _read_json(hard_pair_result_path)
    manifest = _read_json(adjudication_manifest_path)
    if semantic.get("status") != "complete":
        raise RepresentativeTaxonomyError("semantic result must be complete")
    if taxonomy_run.get("status") != "complete":
        raise RepresentativeTaxonomyError("taxonomy result must be complete")
    if hard_pairs.get("status") != "complete":
        raise RepresentativeTaxonomyError("hard-pair result must be complete")
    if manifest.get("adjudication_schema_version") != "1.0.0":
        raise RepresentativeTaxonomyError("unsupported adjudication manifest")

    expected = manifest.get("expected_inputs")
    if not isinstance(expected, dict):
        raise RepresentativeTaxonomyError("missing expected_inputs")
    actual_fingerprints = {
        "semantic_run_fingerprint": semantic.get("run_fingerprint"),
        "taxonomy_run_fingerprint": taxonomy_run.get("run_fingerprint"),
        "taxonomy_fingerprint": _canonical_hash(taxonomy_run.get("taxonomy")),
        "hard_pair_run_fingerprint": hard_pairs.get("run_fingerprint"),
    }
    if expected != actual_fingerprints:
        raise RepresentativeTaxonomyError(
            "adjudication manifest belongs to different semantic or taxonomy evidence"
        )
    if hard_pairs.get("taxonomy_fingerprint") != actual_fingerprints[
        "taxonomy_fingerprint"
    ]:
        raise RepresentativeTaxonomyError("hard-pair taxonomy fingerprint mismatch")

    cards = {item["record_id"]: item for item in semantic.get("records", [])}
    base_assignments = {
        item["record_id"]: item for item in taxonomy_run.get("assignments", [])
    }
    if not cards or set(cards) != set(base_assignments):
        raise RepresentativeTaxonomyError(
            "semantic cards and taxonomy assignments must have identical coverage"
        )

    taxonomy = copy.deepcopy(taxonomy_run.get("taxonomy"))
    if not isinstance(taxonomy, dict):
        raise RepresentativeTaxonomyError("missing frozen taxonomy")
    archetypes = {
        item["archetype_id"]: item for item in taxonomy.get("archetypes", [])
    }
    amendments = manifest.get("taxonomy_amendments", [])
    if not isinstance(amendments, list):
        raise RepresentativeTaxonomyError("taxonomy_amendments must be a list")
    amended_ids: list[str] = []
    for amendment in amendments:
        if not isinstance(amendment, dict):
            raise RepresentativeTaxonomyError("invalid taxonomy amendment")
        archetype_id = amendment.get("archetype_id")
        fields = amendment.get("fields")
        if archetype_id not in archetypes or not isinstance(fields, dict):
            raise RepresentativeTaxonomyError("taxonomy amendment target is invalid")
        if not fields or not set(fields).issubset(EDITABLE_TAXONOMY_FIELDS):
            raise RepresentativeTaxonomyError("taxonomy amendment edits forbidden fields")
        archetypes[archetype_id].update(copy.deepcopy(fields))
        amended_ids.append(archetype_id)
    taxonomy = validate_taxonomy(
        taxonomy,
        minimum_count=len(archetypes),
        maximum_count=len(archetypes),
    )

    raw_decisions = manifest.get("adjudications")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise RepresentativeTaxonomyError("adjudications must be a non-empty list")
    decisions: dict[str, dict[str, Any]] = {}
    taxonomy_ids = {item["archetype_id"] for item in taxonomy["archetypes"]}
    hard_pair_ids = {
        item["record_id"] for item in hard_pairs.get("judgments", [])
    }
    reviewed_hard_pair_ids = set()
    for decision in raw_decisions:
        if not isinstance(decision, dict):
            raise RepresentativeTaxonomyError("invalid adjudication")
        record_id = decision.get("record_id")
        if record_id in decisions or record_id not in cards:
            raise RepresentativeTaxonomyError("duplicate or unknown adjudication record")
        decision_fingerprint = decision.get("input_fingerprint")
        if decision_fingerprint is not None and decision_fingerprint != cards[
            record_id
        ].get("input_fingerprint"):
            raise RepresentativeTaxonomyError("adjudication input fingerprint mismatch")
        primary = decision.get("primary_archetype_id")
        secondary = decision.get("secondary_archetype_id")
        if primary not in taxonomy_ids:
            raise RepresentativeTaxonomyError("invalid adjudicated primary archetype")
        if secondary is not None and (
            secondary not in taxonomy_ids or secondary == primary
        ):
            raise RepresentativeTaxonomyError("invalid adjudicated secondary archetype")
        if decision.get("reviewer_certainty") not in REVIEWER_CERTAINTY:
            raise RepresentativeTaxonomyError("invalid reviewer certainty")
        source = decision.get("review_source")
        reason = decision.get("reason")
        if not isinstance(source, str) or not source:
            raise RepresentativeTaxonomyError("missing adjudication review source")
        if not isinstance(reason, str) or not 4 <= len(reason) <= 180:
            raise RepresentativeTaxonomyError("invalid adjudication reason")
        if source.startswith("hard_pair"):
            reviewed_hard_pair_ids.add(record_id)
        decisions[record_id] = decision
    if reviewed_hard_pair_ids != hard_pair_ids:
        missing = sorted(hard_pair_ids - reviewed_hard_pair_ids)
        extra = sorted(reviewed_hard_pair_ids - hard_pair_ids)
        raise RepresentativeTaxonomyError(
            f"direct hard-pair review coverage mismatch; missing={missing}, extra={extra}"
        )

    released_assignments = []
    primary_changes = 0
    candidate_omissions = 0
    protocol = manifest.get("review_protocol", {})
    full_membership_review = bool(
        protocol.get(
            "full_cluster_membership_reviewed",
            protocol.get("cluster_membership_reviewed", False),
        )
    )
    unreviewed_basis = (
        "retained_after_full_cluster_membership_review"
        if full_membership_review
        else "retained_after_stratified_cluster_evidence_review"
    )
    hard_pair_map = {
        item["record_id"]: item for item in hard_pairs.get("judgments", [])
    }
    for record_id in sorted(cards):
        base = base_assignments[record_id]
        decision = decisions.get(record_id)
        base_snapshot = {
            "primary_archetype_id": base["primary_archetype_id"],
            "secondary_archetype_id": base.get("secondary_archetype_id"),
            "confidence": base["confidence"],
            "boundary_case": base["boundary_case"],
        }
        if decision is None:
            released = {
                "record_id": record_id,
                "input_fingerprint": cards[record_id]["input_fingerprint"],
                "primary_archetype_id": base["primary_archetype_id"],
                "secondary_archetype_id": base.get("secondary_archetype_id"),
                "release_basis": unreviewed_basis,
                "reviewer_certainty": "not_individually_scored",
                "direct_review_reason": None,
                "base_model_assignment": base_snapshot,
            }
        else:
            primary = decision["primary_archetype_id"]
            primary_changes += primary != base["primary_archetype_id"]
            hard_pair = hard_pair_map.get(record_id)
            outside_candidates = bool(
                hard_pair and primary not in hard_pair.get("candidate_ids", [])
            )
            candidate_omissions += outside_candidates
            released = {
                "record_id": record_id,
                "input_fingerprint": cards[record_id]["input_fingerprint"],
                "primary_archetype_id": primary,
                "secondary_archetype_id": decision.get(
                    "secondary_archetype_id"
                ),
                "release_basis": "direct_semantic_adjudication",
                "reviewer_certainty": decision["reviewer_certainty"],
                "direct_review_source": decision["review_source"],
                "direct_review_reason": decision["reason"],
                "selected_outside_hard_pair_candidates": outside_candidates,
                "base_model_assignment": base_snapshot,
            }
        released_assignments.append(released)

    counts = Counter(
        item["primary_archetype_id"] for item in released_assignments
    )
    result = {
        "reviewed_release_schema_version": REVIEWED_RELEASE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "run_fingerprint": _canonical_hash(
            {
                "semantic": actual_fingerprints["semantic_run_fingerprint"],
                "taxonomy": actual_fingerprints["taxonomy_run_fingerprint"],
                "hard_pairs": actual_fingerprints["hard_pair_run_fingerprint"],
                "manifest": _file_identity(adjudication_manifest_path),
            }
        ),
        "input_evidence": {
            "semantic_result": _file_identity(semantic_result_path),
            "taxonomy_result": _file_identity(taxonomy_result_path),
            "hard_pair_result": _file_identity(hard_pair_result_path),
            "adjudication_manifest": _file_identity(adjudication_manifest_path),
            "paths_persisted": False,
        },
        "review_protocol": copy.deepcopy(manifest["review_protocol"]),
        "reviewer": manifest["reviewer"],
        "taxonomy_base_fingerprint": actual_fingerprints["taxonomy_fingerprint"],
        "taxonomy_release_fingerprint": _canonical_hash(taxonomy),
        "taxonomy": taxonomy,
        "assignments": released_assignments,
        "summary": {
            "record_count": len(released_assignments),
            "directly_adjudicated_count": len(decisions),
            "hard_pair_count": len(hard_pair_ids),
            "hard_pair_direct_review_coverage": len(reviewed_hard_pair_ids),
            "primary_assignment_change_count": primary_changes,
            "selected_outside_hard_pair_candidate_count": candidate_omissions,
            "taxonomy_amended_archetype_ids": sorted(set(amended_ids)),
            "primary_cluster_support": dict(sorted(counts.items())),
        },
        "validity": {
            "automatic_agreement_treated_as_ground_truth": False,
            "direct_semantic_review_is_mechanical_metric": False,
            "review_bound_to_exact_input_fingerprints": True,
            "release_claims_persistent_real_person_identity": False,
            "future_scaled_samples_require_new_qualitative_review": True,
        },
    }
    _atomic_write_json(result, output_path)
    return result


def revise_taxonomy_from_review(
    *,
    taxonomy_assignment_path: str | Path,
    revision_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply a cross-time qualitative taxonomy review, including new classes."""

    source = _read_json(taxonomy_assignment_path)
    manifest = _read_json(revision_manifest_path)
    if source.get("status") != "complete" or not isinstance(
        source.get("taxonomy"), dict
    ):
        raise RepresentativeTaxonomyError(
            "taxonomy assignment source must be complete"
        )
    if manifest.get("taxonomy_revision_schema_version") != (
        TAXONOMY_REVISION_SCHEMA_VERSION
    ):
        raise RepresentativeTaxonomyError("unsupported taxonomy revision manifest")
    expected = manifest.get("expected_input")
    actual = {
        "taxonomy_assignment_run_fingerprint": source.get("run_fingerprint"),
        "taxonomy_fingerprint": _canonical_hash(source["taxonomy"]),
    }
    if expected != actual:
        raise RepresentativeTaxonomyError(
            "taxonomy revision belongs to different assignment evidence"
        )
    review_protocol = manifest.get("review_protocol")
    if not isinstance(review_protocol, dict) or not review_protocol.get(
        "raw_text_persisted"
    ) is False:
        raise RepresentativeTaxonomyError("invalid taxonomy review protocol")

    taxonomy = copy.deepcopy(source["taxonomy"])
    archetypes = {
        item["archetype_id"]: item for item in taxonomy.get("archetypes", [])
    }
    amendments = manifest.get("taxonomy_amendments", [])
    additions = manifest.get("taxonomy_additions", [])
    if not isinstance(amendments, list) or not isinstance(additions, list):
        raise RepresentativeTaxonomyError("taxonomy edits must be lists")
    amendment_reasons = []
    for amendment in amendments:
        if not isinstance(amendment, dict):
            raise RepresentativeTaxonomyError("invalid taxonomy amendment")
        archetype_id = amendment.get("archetype_id")
        fields = amendment.get("fields")
        if archetype_id not in archetypes or not isinstance(fields, dict):
            raise RepresentativeTaxonomyError("invalid taxonomy amendment target")
        if not fields or not set(fields).issubset(EDITABLE_TAXONOMY_FIELDS):
            raise RepresentativeTaxonomyError("taxonomy amendment edits forbidden fields")
        reason = amendment.get("reason")
        if not isinstance(reason, str) or not 4 <= len(reason) <= 180:
            raise RepresentativeTaxonomyError("invalid taxonomy amendment reason")
        archetypes[archetype_id].update(copy.deepcopy(fields))
        amendment_reasons.append(
            {"archetype_id": archetype_id, "reason": reason}
        )

    next_index = len(archetypes) + 1
    added_reasons = []
    for addition in additions:
        if not isinstance(addition, dict):
            raise RepresentativeTaxonomyError("invalid taxonomy addition")
        expected_id = f"archetype-{next_index:02d}"
        if addition.get("archetype_id") != expected_id:
            raise RepresentativeTaxonomyError(
                "taxonomy additions must use the next sequential id"
            )
        if expected_id in archetypes:
            raise RepresentativeTaxonomyError("duplicate taxonomy addition")
        taxonomy["archetypes"].append(copy.deepcopy(addition))
        archetypes[expected_id] = taxonomy["archetypes"][-1]
        added_reasons.append(expected_id)
        next_index += 1

    taxonomy = validate_taxonomy(
        taxonomy,
        minimum_count=len(archetypes),
        maximum_count=len(archetypes),
    )
    result = {
        "taxonomy_revision_artifact_schema_version": (
            TAXONOMY_REVISION_SCHEMA_VERSION
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "revision_id": manifest.get("revision_id"),
        "reviewer": manifest.get("reviewer"),
        "review_protocol": copy.deepcopy(review_protocol),
        "input_evidence": {
            "taxonomy_assignment": _file_identity(taxonomy_assignment_path),
            "revision_manifest": _file_identity(revision_manifest_path),
            "paths_persisted": False,
        },
        "taxonomy_base_fingerprint": actual["taxonomy_fingerprint"],
        "taxonomy_release_fingerprint": _canonical_hash(taxonomy),
        "taxonomy": taxonomy,
        "summary": {
            "archetype_count": len(taxonomy["archetypes"]),
            "amended_archetype_ids": [
                item["archetype_id"] for item in amendment_reasons
            ],
            "added_archetype_ids": added_reasons,
            "amendment_reasons": amendment_reasons,
        },
        "validity": {
            "revision_is_mechanical_metric": False,
            "direct_cross_time_text_review_required": True,
            "source_assignments_automatically_released": False,
            "new_taxonomy_requires_reassignment": True,
        },
    }
    result["run_fingerprint"] = _canonical_hash(
        {
            "source": actual,
            "revision_manifest": _file_identity(revision_manifest_path),
            "taxonomy_release_fingerprint": result[
                "taxonomy_release_fingerprint"
            ],
        }
    )
    _atomic_write_json(result, output_path)
    return result
