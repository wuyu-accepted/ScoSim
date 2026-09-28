#!/usr/bin/env python3
"""Zero-network verification of the AAMAS2027 evidence release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    "01_forum_realism/housing_allocation",
    "02_micro_consistency/dimension2_cross_method",
    "02_micro_consistency/dimension2_ablation",
    "03_population_polarization/dimension3_cross_method",
    "03_population_polarization/dimension3_ablation",
    "04_governance_response/housing_natural_vs_D",
)
REQUIRED = (
    "CASE_AND_DATA.md", "parameters.json", "cost_by_method_and_call_type.csv",
    "raw_or_sufficient_statistics", "metrics", "report", "provenance.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    manifest = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for item in manifest["produced_files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            errors.append(f"root_manifest:{item['path']}")
    for relative in EXPERIMENTS:
        directory = ROOT / relative
        for name in REQUIRED:
            if not (directory / name).exists():
                errors.append(f"missing:{relative}/{name}")
        if not (directory / "provenance.json").is_file():
            continue
        local = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        for item in local["files"]:
            path = directory / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                errors.append(f"experiment_manifest:{relative}/{item['path']}")
    for dimension in ("01_forum_realism", "02_micro_consistency", "03_population_polarization", "04_governance_response"):
        code = ROOT / dimension / "code"
        source_manifest = json.loads((code / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        for item in source_manifest["files"]:
            path = code / item["snapshot"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                errors.append(f"code_manifest:{dimension}/{item['snapshot']}")
    smoke_imports = {
        "02_micro_consistency": "import adapters.campuspulse,evaluation.common,evaluation.dimension2,evaluation.text_judge,comparison_harness.real_provider,campuspulse_core.forum_pilot,campuspulse_core.governance_arena_v2",
        "03_population_polarization": "import evaluation.common,evaluation.dimension3,comparison_harness.dimension3_belief_network,campuspulse_core.forum_relationships,campuspulse_core.mass_population",
    }
    for dimension, statement in smoke_imports.items():
        base = ROOT / dimension / "code" / "campuspulse_core"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(base), str(base / "src")))
        completed = subprocess.run(
            [sys.executable, "-c", statement], env=env, capture_output=True, text=True, timeout=60
        )
        if completed.returncode:
            errors.append(f"import_smoke:{dimension}:{completed.stderr.strip().splitlines()[-1]}")
    old_answer_name = ROOT / "02_micro_consistency/dimension2_cross_method/raw_or_sufficient_statistics/independent_blind_review_answers.csv"
    template = ROOT / "02_micro_consistency/dimension2_cross_method/raw_or_sufficient_statistics/independent_blind_review_template.csv"
    if old_answer_name.exists() or not template.is_file():
        errors.append("blind_review_template_naming")
    forbidden = ("credential", "api_key", "api-key", "raw_response")
    for path in ROOT.rglob("*"):
        if path.is_file() and any(token in path.name.lower() for token in forbidden):
            errors.append(f"forbidden_filename:{path.relative_to(ROOT).as_posix()}")
    result = {"status": "pass" if not errors else "fail", "errors": errors, "checked_root_files": len(manifest["produced_files"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
