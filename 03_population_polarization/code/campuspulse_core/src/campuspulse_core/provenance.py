"""Dataset lineage binding for every CampusPulse simulation artifact."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BUNDLED_VALIDATION = HERE / "data" / "validation" / "validation.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DataLineageError(ValueError):
    """Raised when the committed empirical snapshot cannot be verified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_fingerprint(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=4)
def load_data_lineage(validation_path: str | Path = BUNDLED_VALIDATION) -> dict:
    """Load and validate privacy-safe source hashes and empirical claim scope."""

    source = Path(validation_path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataLineageError(f"cannot load empirical validation snapshot: {exc}") from exc
    evidence = raw.get("input_evidence")
    if not isinstance(evidence, dict):
        raise DataLineageError("validation snapshot is missing input_evidence")
    normalized_files = {}
    for key in ("posts_csv", "comments_csv"):
        item = evidence.get(key)
        if not isinstance(item, dict):
            raise DataLineageError(f"validation input_evidence.{key} is missing")
        sha256 = str(item.get("sha256", "")).lower()
        size = item.get("bytes")
        if not SHA256_PATTERN.fullmatch(sha256):
            raise DataLineageError(f"validation input_evidence.{key}.sha256 is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise DataLineageError(f"validation input_evidence.{key}.bytes is invalid")
        normalized_files[key] = {"sha256": sha256, "bytes": size}

    methodology = raw.get("methodology", {})
    conclusion = raw.get("overall_conclusion", {})
    coverage = raw.get("empirical_claim_coverage", {})
    focus_evidence = raw.get("focus_evidence", {})
    focus_backtests = focus_evidence.get("scenario_backtests", {})
    field_audit = focus_evidence.get("field_audit", {})
    snapshot_payload = {
        "files": normalized_files,
        "train_window": methodology.get("train_window"),
        "holdout_window": methodology.get("holdout_window"),
        "excluded_known_gap_dates": methodology.get("excluded_known_gap_dates"),
        "platform_marker": methodology.get("platform_marker"),
    }
    return {
        "dataset_snapshot_id": _canonical_fingerprint(snapshot_payload),
        "validation_artifact_sha256": _sha256_file(source),
        "source_files": normalized_files,
        "calibration_and_validation": {
            "design": methodology.get("design"),
            "train_window": methodology.get("train_window"),
            "holdout_window": methodology.get("holdout_window"),
            "excluded_known_gap_dates": methodology.get("excluded_known_gap_dates"),
            "observable_generator_holdout_fidelity": conclusion.get(
                "observable_generator_holdout_fidelity"
            ),
            "intervention_effect_validity": conclusion.get(
                "intervention_effect_validity"
            ),
            "temporally_tested_claim_families": coverage.get(
                "temporally_tested_claim_families"
            ),
            "total_claim_families": coverage.get("total_claim_families"),
            "focus_event_backtests": {
                scenario_id: {
                    "event_count": focus_backtests.get(scenario_id, {}).get(
                        "event_count"
                    ),
                    "overall_grade": focus_backtests.get(scenario_id, {}).get(
                        "overall_grade"
                    ),
                    "evidence_scope": focus_backtests.get(scenario_id, {}).get(
                        "evidence_scope"
                    ),
                }
                for scenario_id in ("exam_pressure", "course_selection")
            },
            "field_gate_counts": field_audit.get("gate_counts"),
        },
        "interpretation": (
            "Demand arrival and natural forum response proxies are bound to this "
            "two-file snapshot. Intervention, service, urgency, and cost parameters "
            "remain assumptions or operator inputs unless separately validated."
        ),
    }
