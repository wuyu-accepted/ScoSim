from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import CommonObservation, MethodSnapshot
from .prompting import COMMON_SECTIONS, PromptBundle


@dataclass(frozen=True)
class AuditFinding:
    code: str
    message: str


def audit_paired_prompts(bundles: Iterable[PromptBundle]) -> list[AuditFinding]:
    bundles = list(bundles)
    findings: list[AuditFinding] = []
    if len(bundles) < 2:
        return findings
    expected_observation = bundles[0].observation_sha256
    expected = {name: bundles[0].section_sha256[name] for name in COMMON_SECTIONS}
    for bundle in bundles[1:]:
        if bundle.observation_sha256 != expected_observation:
            findings.append(AuditFinding("observation_mismatch", bundle.method))
        for name in COMMON_SECTIONS:
            if bundle.section_sha256[name] != expected[name]:
                findings.append(AuditFinding("common_section_mismatch", f"{bundle.method}:{name}"))
    return findings


def audit_snapshot_information(
    observation: CommonObservation, snapshot: MethodSnapshot
) -> list[AuditFinding]:
    """Mechanical leak guard; semantic review remains mandatory before pilot."""
    findings: list[AuditFinding] = []
    if snapshot.source_observation_sha256 != observation.observation_sha256:
        findings.append(AuditFinding("wrong_observation", snapshot.method))
    forbidden_keys = {
        "future", "future_tick", "ground_truth", "evaluation_label",
        "expected_ranking", "other_method_state", "post_state",
    }

    def walk(value: object, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                if str(key).lower() in forbidden_keys:
                    findings.append(AuditFinding("forbidden_key", next_path))
                walk(child, next_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(snapshot.cognitive_context)
    return findings


def assert_single_ablation_difference(
    full: Mapping[str, object], variant: Mapping[str, object], allowed_paths: set[str]
) -> None:
    differences: set[str] = set()

    def walk(left: object, right: object, path: str = "") -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in set(left) | set(right):
                child_path = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    differences.add(child_path)
                else:
                    walk(left[key], right[key], child_path)
        elif left != right:
            differences.add(path)

    walk(full, variant)
    unexpected = {path for path in differences if path not in allowed_paths}
    if unexpected:
        raise AssertionError(f"ablation changed non-registered fields: {sorted(unexpected)}")
