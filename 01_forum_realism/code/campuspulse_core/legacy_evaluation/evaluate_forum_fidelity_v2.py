#!/usr/bin/env python3
"""Replay the formal ForumTwin run offline and build Fidelity Evaluation v2.

The script is intentionally incapable of network access.  It reconstructs the
312 resident turns from immutable, identity-bound settled ledger responses and
replays the six governance turns by exact prompt/model fingerprint.  The public
timeline is then checked against the original formal result through Tick 9; a
cache miss raises before any Provider could be called.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.campus_pulse.runtime_contract import canonical_sha256
from envs.campus_pulse_governance.forum_ecology_calibration import (
    validate_forum_ecology_calibration,
)
from envs.campus_pulse_governance.forum_fidelity_v2 import (
    PrivateParticleFidelityCollector,
    build_fidelity_evaluation_v2,
)
from envs.campus_pulse_governance import forum_hero as forum_hero_module
from envs.campus_pulse_governance.forum_hero import (
    SHARED_BASELINE_BRANCH,
    run_forum_hero_experiment,
)
from envs.campus_pulse_governance.forum_pilot import (
    COST_AWARE_ACTIVATION_POLICY,
    FOUR_DIMENSION_BRANCHES,
    FOUR_DIMENSION_DECISION_TICKS,
    FOUR_DIMENSION_RESIDENT_BUDGET,
    FOUR_DIMENSION_SHARED_END_TICK,
    FOUR_DIMENSION_START_TICK,
    GOVERNANCE_LEGITIMACY_SCENARIO,
    OPEN_CHOICE_MECHANISM_PROTOCOL,
    PILOT_SEED,
    compile_pilot_profile_bridge,
    scenario_weight_key,
)
from envs.campus_pulse_governance.forum_relationships import (
    compile_scenario_stake_release_v1,
    compile_synthetic_relationship_release_v1,
)
from envs.campus_pulse_governance.governance_arena_v2 import initial_branch_state
from envs.campus_pulse_governance.mass_population import canonical_hash
from envs.campus_pulse_governance.forum_turns import (
    FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
)
from envs.campus_pulse_governance.yulan_scale import TurnEnvelope, scenario_population_view
from scripts.run_forum_hero_authorized import (
    _ExactPriorTurnReplaySource,
    _load_prior_unknown_resolution,
)


DEFAULT_RUN = ROOT / "_run" / "four-dimension-v1"
DEFAULT_INITIAL_LEDGER = (
    DEFAULT_RUN
    / "formal-v5-20260830"
    / "authorized-live"
    / "authorized-live-ledger.sqlite3"
)
DEFAULT_CONTINUATION = DEFAULT_RUN / "formal-v5-20260830-continuation"
DEFAULT_CONTINUATION_LEDGER = (
    DEFAULT_CONTINUATION / "authorized-live" / "authorized-live-ledger.sqlite3"
)
DEFAULT_UNKNOWN = DEFAULT_CONTINUATION / "prior-unknown-resolution-v1.json"
DEFAULT_POPULATION = (
    ROOT
    / "data"
    / "campus_pulse_workspace"
    / "forumtwin"
    / "releases"
    / "hero-pilot-bridge-v6.json"
)
DEFAULT_FORMAL_POPULATION = (
    ROOT
    / "data"
    / "campus_pulse_workspace"
    / "forumtwin"
    / "releases"
    / "semantic-llm-forum-agent-population-v2.json"
)
DEFAULT_ECOLOGY = (
    ROOT
    / "data"
    / "campus_pulse_workspace"
    / "forumtwin"
    / "releases"
    / "forum-ecology-calibration-v2.json"
)
DEFAULT_ORIGINAL_RESULT = (
    DEFAULT_CONTINUATION / "authorized-live" / "forum-hero-result-through-T10.json"
)
DEFAULT_PERSONA_EVIDENCE = (
    DEFAULT_CONTINUATION / "authorized-live" / "private-persona-turn-evidence-v1.json"
)
DEFAULT_EXISTING_FIDELITY = (
    ROOT
    / "evidence"
    / "four-dimension-v5-formal-v1"
    / "fidelity-evaluation-v3.json"
)
DEFAULT_PUBLIC_DIR = ROOT / "evidence" / "four-dimension-v5-formal-v1"
DEFAULT_PRIVATE_DIR = DEFAULT_CONTINUATION / "fidelity-v2"
EXPECTED_MODEL = "gpt-5.6-luna"


class _OfflineCacheMissSource:
    """Fail closed on every cache miss; this class has no HTTP transport."""

    provenance = "authorized_live_llm"
    formal_evidence = True
    max_concurrency = 1

    def __init__(self) -> None:
        self.cache_miss_count = 0

    @property
    def usage_snapshot(self) -> dict[str, Any]:
        body = {
            "provider_attempts": 0,
            "provider_calls": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "semantic_turns": 0,
        }
        return {**body, "usage_sha256": canonical_sha256(body)}

    def acquire(self, **kwargs: Any) -> Any:
        self.cache_miss_count += 1
        raise RuntimeError(
            "offline fidelity replay cache miss: "
            + str(kwargs.get("slot_kind") or "unknown")
            + "/"
            + str(kwargs.get("prompt_sha256") or "")
        )


class _IdentityBoundResidentReplaySource:
    """Replay settled resident turns by the archived Agent/branch/Tick key.

    The formal V5 renderer was improved after the run, so its old prompt bytes
    are no longer present in the repository.  The private audit evidence keeps
    a hash-bound Agent/branch/Tick key for every non-terminal resident turn.
    This source joins that key to the immutable cache, rewrites only the prompt
    fingerprint to the currently rendered equivalent, and then runs the full
    current validator.  It never guesses a response and deliberately stops at
    Tick 9, before the 48 terminal turns that lack a following-state audit row.
    """

    provenance = "authorized_live_llm"
    formal_evidence = True
    max_concurrency = 1

    def __init__(
        self,
        *,
        persona_evidence: Mapping[str, Any],
        ledgers: tuple[Path, ...],
        expected_model: str,
    ) -> None:
        cache: dict[str, TurnEnvelope] = {}
        for path in ledgers:
            connection = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            try:
                rows = connection.execute(
                    """
                    SELECT slot_kind, prompt_sha256, response_sha256,
                           returned_model, prompt_tokens, completion_tokens,
                           total_tokens, content_json
                    FROM v7_live_cache WHERE slot_kind = 'resident'
                    """
                ).fetchall()
            finally:
                connection.close()
            for row in rows:
                prompt_sha = str(row["prompt_sha256"])
                if prompt_sha in cache or str(row["returned_model"]) != expected_model:
                    raise ValueError("resident replay cache is ambiguous")
                cache[prompt_sha] = TurnEnvelope(
                    slot_kind="resident",
                    prompt_sha256=prompt_sha,
                    response_sha256=str(row["response_sha256"]),
                    provenance="authorized_live_llm",
                    content=str(row["content_json"]),
                    returned_model=str(row["returned_model"]),
                    prompt_tokens=int(row["prompt_tokens"]),
                    completion_tokens=int(row["completion_tokens"]),
                    total_tokens=int(row["total_tokens"]),
                    provider_calls=0,
                    cache_hit=True,
                )
        evidence_rows = persona_evidence.get("rows")
        if not isinstance(evidence_rows, list):
            raise ValueError("private persona evidence is invalid")
        entries: dict[tuple[str, str, int], TurnEnvelope] = {}
        for row in evidence_rows:
            if not isinstance(row, Mapping):
                raise ValueError("private persona row is invalid")
            old_prompt_sha = str(row.get("prompt_sha256") or "")
            envelope = cache.get(old_prompt_sha)
            if envelope is None or envelope.response_sha256 != str(
                row.get("response_sha256") or ""
            ):
                raise ValueError("persona evidence does not match the settled cache")
            key = (
                str(row.get("agent_id") or ""),
                str(row.get("branch") or ""),
                int(row.get("tick", -1)),
            )
            if key in entries:
                raise ValueError("resident identity replay key is duplicated")
            entries[key] = envelope
        if len(entries) != 312:
            raise ValueError("resident identity replay requires 312 audited turns")
        self._entries = entries
        self._hit_count = 0
        self._miss_count = 0

    @property
    def usage_snapshot(self) -> dict[str, Any]:
        body = {
            "provider_attempts": 0,
            "provider_calls": 0,
            "total_tokens": 0,
            "cache_hits": self._hit_count,
            "semantic_turns": self._hit_count,
        }
        return {**body, "usage_sha256": canonical_sha256(body)}

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def acquire(self, **kwargs: Any) -> Any:
        if kwargs.get("slot_kind") != "resident":
            self._miss_count += 1
            raise RuntimeError("identity resident replay received a non-resident slot")
        rendered = kwargs.get("rendered")
        key = (
            str(getattr(rendered, "agent_id", "")),
            str(getattr(rendered, "branch", "")),
            int(getattr(rendered, "tick", -1)),
        )
        archived = self._entries.get(key)
        if archived is None:
            self._miss_count += 1
            raise RuntimeError("identity-bound resident replay key is missing")
        envelope = TurnEnvelope(
            slot_kind="resident",
            prompt_sha256=str(kwargs.get("prompt_sha256") or ""),
            response_sha256=archived.response_sha256,
            provenance=archived.provenance,
            content=archived.content,
            returned_model=archived.returned_model,
            prompt_tokens=archived.prompt_tokens,
            completion_tokens=archived.completion_tokens,
            total_tokens=archived.total_tokens,
            provider_calls=0,
            cache_hit=True,
        )
        validator = kwargs.get("validator")
        if validator is None:
            raise RuntimeError("identity-bound replay requires validation")
        validator(envelope)
        self._hit_count += 1
        return envelope


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=path.name + ".", suffix=".tmp",
        dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _result(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = value.get("result", value)
    if not isinstance(result, Mapping):
        raise ValueError("formal result is invalid")
    expected = str(result.get("result_sha256") or "")
    actual = canonical_sha256(
        {key: item for key, item in result.items() if key != "result_sha256"}
    )
    if expected != actual:
        raise ValueError("formal result hash mismatch")
    return result


_REPLAY_VOLATILE_KEYS = frozenset(
    {"usage", "result_sha256", "shared_prefix_result_sha256"}
)


def _semantic_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_projection(item)
            for key, item in value.items()
            if str(key) not in _REPLAY_VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_semantic_projection(item) for item in value]
    return value


def _first_difference(left: Any, right: Any, path: str = "root") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return f"{path}: keys {sorted(set(left) ^ set(right))}"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if difference:
                return difference
        return ""
    return "" if left == right else f"{path}: {left!r} != {right!r}"


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _persona_svg(evaluation: Mapping[str, Any]) -> str:
    persona = evaluation["dimensions"]["micro_persona_consistency"]
    predictive = persona["joint_profile_previous_state_constraint"]
    outcomes = predictive["outcomes"]
    labels = {
        "action_family": "Action family",
        "evidence_family": "Evidence status",
        "private_action": "Private action",
        "public_expression": "Public expression",
        "stance_family": "Stance",
    }
    bars = []
    for index, name in enumerate(sorted(outcomes)):
        profile_value = float(
            outcomes[name]["profile_increment_given_previous_state"][
                "relative_log_loss_improvement"
            ]
        )
        state_value = float(
            outcomes[name]["previous_state_increment_given_profile"][
                "relative_log_loss_improvement"
            ]
        )
        profile_width = min(180.0, abs(profile_value) * 700.0)
        state_width = min(180.0, abs(state_value) * 700.0)
        y = 165 + index * 62
        bars.append(
            f'<text x="55" y="{y + 18}" class="label">{_escape(labels[name])}</text>'
            f'<rect x="360" y="{y}" width="{profile_width:.1f}" height="12" rx="4" fill="#b80f3a"/>'
            f'<rect x="360" y="{y + 16}" width="{state_width:.1f}" height="12" rx="4" fill="#3977a8"/>'
            f'<text x="715" y="{y + 11}" class="value">P {profile_value * 100:+.1f}%</text>'
            f'<text x="715" y="{y + 27}" class="value">S {state_value * 100:+.1f}%</text>'
        )
    assimilation = persona["immediate_llm_particle_assimilation"]
    style = persona["longitudinal_language_consistency"]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620">
<style>.title{{font:700 30px Arial,sans-serif;fill:#17141a}}.sub{{font:16px Arial,sans-serif;fill:#5d5963}}.label{{font:16px Arial,sans-serif;fill:#29252d}}.value{{font:700 16px Arial,sans-serif;fill:#29252d}}.card{{fill:#faf8f5;stroke:#ddd6ce;stroke-width:1.5}}</style>
<rect width="1200" height="620" fill="#f4f1ed"/><text x="55" y="58" class="title">Profile × previous particle state · Evaluation v3</text>
<text x="55" y="91" class="sub">P = Profile increment given state · S = previous-state increment given Profile</text>
<rect x="35" y="118" width="760" height="390" rx="18" class="card"/>
<text x="360" y="140" class="sub">Relative out-of-Agent log-loss improvement</text>{''.join(bars)}
<rect x="825" y="118" width="340" height="176" rx="18" class="card"/><text x="855" y="155" class="sub">Immediate assimilation</text>
<text x="855" y="205" class="title">{float(assimilation['direction_agreement_rate']) * 100:.1f}%</text><text x="855" y="239" class="sub">{assimilation['tested_direction_count']} tested directions · mechanism check</text>
<rect x="825" y="314" width="340" height="194" rx="18" class="card"/><text x="855" y="352" class="sub">Longitudinal language</text>
<text x="855" y="401" class="title">{float(style.get('same_agent_advantage', 0)):+.3f}</text><text x="855" y="435" class="sub">same-Agent cosine advantage</text><text x="855" y="469" class="sub">{style.get('eligible_longitudinal_agent_count', 0)} repeated expressive trajectories</text>
<text x="55" y="560" class="sub">Unit: persistent synthetic Agent. Grouped folds prevent the same Agent appearing in both train and test.</text>
</svg>'''


def _heterogeneity_svg(evaluation: Mapping[str, Any]) -> str:
    dimension = evaluation["dimensions"]["population_heterogeneity"]
    frames = dimension["frames"]
    points = defaultdict(list)
    for row in frames:
        points[str(row["branch"])].append(
            (int(row["tick"]), float(row["effective_state_rank"]))
        )
    series = []
    colors = {"Natural": "#46414b", "D": "#c70b3b", "shared_initial": "#b08a39", "shared_baseline": "#b08a39"}
    for branch, values in sorted(points.items()):
        values.sort()
        path = " ".join(
            ("M" if index == 0 else "L")
            + f" {100 + tick * 70:.1f} {350 - (rank - 1) * 28:.1f}"
            for index, (tick, rank) in enumerate(values)
        )
        series.append(
            f'<path d="{path}" fill="none" stroke="{colors.get(branch, "#5079a8")}" stroke-width="4"/>'
        )
    final_tick = max(
        int(row["tick"]) for row in frames if row["branch"] in {"Natural", "D"}
    )
    final_d = next(
        row for row in frames if row["branch"] == "D" and row["tick"] == final_tick
    )
    fields = final_d["variance_decomposition"]
    stacks = []
    for index, field in enumerate(list(fields)[:12]):
        share = float(fields[field]["between_share_of_total"])
        x = 72 + index * 88
        between_height = 145 * share
        stacks.append(
            f'<rect x="{x}" y="{530-between_height:.1f}" width="44" height="{between_height:.1f}" fill="#c70b3b"/>'
            f'<rect x="{x}" y="385" width="44" height="{145-between_height:.1f}" fill="#cfc8bf"/>'
            f'<text x="{x+22}" y="552" text-anchor="middle" class="small" transform="rotate(35 {x+22} 552)">{_escape(field)}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680">
<style>.title{{font:700 30px Arial,sans-serif;fill:#17141a}}.sub{{font:16px Arial,sans-serif;fill:#5d5963}}.small{{font:12px Arial,sans-serif;fill:#4b4650}}.card{{fill:#faf8f5;stroke:#ddd6ce;stroke-width:1.5}}</style>
<rect width="1200" height="680" fill="#f4f1ed"/><text x="55" y="58" class="title">Population heterogeneity · 1,000 parents / 10,000 particles</text>
<text x="55" y="90" class="sub">Twelve-dimensional effective rank and between-parent / within-parent variance decomposition</text>
<rect x="35" y="118" width="1130" height="245" rx="18" class="card"/><text x="65" y="153" class="sub">Effective state rank over paired worlds</text>{''.join(series)}
<text x="920" y="185" class="sub" fill="#46414b">— Natural</text><text x="920" y="218" class="sub" fill="#c70b3b">— D governance</text>
<rect x="35" y="382" width="1130" height="245" rx="18" class="card"/><text x="65" y="416" class="sub">D · Tick {final_tick} variance share: parent differences (red) vs uncertainty within the same Agent (grey)</text>{''.join(stacks)}
<text x="55" y="660" class="sub">Particles are uncertainty samples, never additional people or independent observations.</text>
</svg>'''


def _distribution_svg(evaluation: Mapping[str, Any]) -> str:
    frames = evaluation["dimensions"]["population_heterogeneity"]["frames"]
    final_tick = max(
        int(row["tick"]) for row in frames if row["branch"] in {"Natural", "D"}
    )
    final = {
        str(row["branch"]): row["net_stance_distribution"]
        for row in frames
        if int(row["tick"]) == final_tick
    }
    paths = []
    for branch, color in (("Natural", "#46414b"), ("D", "#c70b3b")):
        values = final[branch]["histogram_weight"]
        maximum = max(max(values), 1e-9)
        path = " ".join(
            ("M" if index == 0 else "L")
            + f" {90 + index * 50:.1f} {480 - value / maximum * 300:.1f}"
            for index, value in enumerate(values)
        )
        paths.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="5"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620">
<style>.title{{font:700 30px Arial,sans-serif;fill:#17141a}}.sub{{font:16px Arial,sans-serif;fill:#5d5963}}.card{{fill:#faf8f5;stroke:#ddd6ce;stroke-width:1.5}}</style>
<rect width="1200" height="620" fill="#f4f1ed"/><text x="55" y="58" class="title">Net-stance distribution at Tick {final_tick}</text>
<text x="55" y="90" class="sub">Design-weighted parent Agents; descriptive density, not a forced bimodality test</text>
<rect x="35" y="118" width="1130" height="410" rx="18" class="card"/><line x1="90" y1="480" x2="1040" y2="480" stroke="#aaa29a"/>{''.join(paths)}
<text x="90" y="515" class="sub">−1 opposition</text><text x="530" y="515" class="sub">0 mixed</text><text x="930" y="515" class="sub">+1 support</text>
<text x="865" y="165" class="sub" fill="#46414b">— Natural</text><text x="865" y="200" class="sub" fill="#c70b3b">— D governance</text>
<text x="55" y="575" class="sub">Natural mean {float(final['Natural']['weighted_mean']):+.3f} · D mean {float(final['D']['weighted_mean']):+.3f}</text>
</svg>'''


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=path.name + ".",
        suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-ledger", type=Path, default=DEFAULT_INITIAL_LEDGER)
    parser.add_argument(
        "--continuation-ledger", type=Path, default=DEFAULT_CONTINUATION_LEDGER
    )
    parser.add_argument("--unknown-resolution", type=Path, default=DEFAULT_UNKNOWN)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument(
        "--formal-population", type=Path, default=DEFAULT_FORMAL_POPULATION
    )
    parser.add_argument("--ecology", type=Path, default=DEFAULT_ECOLOGY)
    parser.add_argument("--original-result", type=Path, default=DEFAULT_ORIGINAL_RESULT)
    parser.add_argument(
        "--persona-evidence", type=Path, default=DEFAULT_PERSONA_EVIDENCE
    )
    parser.add_argument(
        "--existing-fidelity", type=Path, default=DEFAULT_EXISTING_FIDELITY
    )
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    arguments = parser.parse_args()

    population = _load(arguments.population.resolve())
    formal_population = _load(arguments.formal_population.resolve())
    ecology = _load(arguments.ecology.resolve())
    validate_forum_ecology_calibration(ecology)
    stakes = compile_scenario_stake_release_v1(
        population=formal_population,
        scenario_ids=(GOVERNANCE_LEGITIMACY_SCENARIO,),
    )
    relationships = compile_synthetic_relationship_release_v1(
        population=formal_population,
        stake_release=stakes,
    )
    profile_release = compile_pilot_profile_bridge(population)
    profiles = profile_release["profiles"]
    population_agent_ids = [str(row["agent_id"]) for row in profiles]
    formal_ids = [str(row["agent_id"]) for row in formal_population["agents"]]
    if set(population_agent_ids) != set(formal_ids) or len(population_agent_ids) != 1_000:
        raise ValueError("pilot and formal population identities differ")

    persona_evidence = _load(arguments.persona_evidence.resolve())
    expected_persona_sha = str(persona_evidence.get("content_sha256") or "")
    actual_persona_sha = canonical_sha256(
        {
            key: value
            for key, value in persona_evidence.items()
            if key != "content_sha256"
        }
    )
    if expected_persona_sha != actual_persona_sha:
        raise ValueError("private persona evidence hash mismatch")

    # Validate that the first immutable ledger is safely settled and that its
    # one historical unknown attempt has the exact abandonment evidence.  The
    # resulting source is not executed; its construction is the safety audit.
    safety_miss_source = _OfflineCacheMissSource()
    unknown = _load_prior_unknown_resolution(arguments.unknown_resolution.resolve())
    _ExactPriorTurnReplaySource(
        safety_miss_source,
        arguments.initial_ledger.resolve(),
        expected_model=EXPECTED_MODEL,
        slot_kinds=frozenset({"resident"}),
        unknown_resolutions=(unknown,),
    )
    identity_source = _IdentityBoundResidentReplaySource(
        persona_evidence=persona_evidence,
        ledgers=(
            arguments.initial_ledger.resolve(),
            arguments.continuation_ledger.resolve(),
        ),
        expected_model=EXPECTED_MODEL,
    )
    source: Any = identity_source
    # Governance prompts are unchanged and remain exact prompt-fingerprint
    # replay.  There are six actor turns in the continuation ledger.
    source = _ExactPriorTurnReplaySource(
        source,
        arguments.continuation_ledger.resolve(),
        expected_model=EXPECTED_MODEL,
        slot_kinds=frozenset({"governance_actor"}),
    )
    # Every preceding checkpoint is required to evaluate whether the exact
    # previous-Tick particle state constrains the next resident LLM turn.
    particle_collector = PrivateParticleFidelityCollector(selected_ticks=range(3, 10))
    initial_population_view = scenario_population_view(
        population,
        scenario_id=scenario_weight_key(GOVERNANCE_LEGITIMACY_SCENARIO),
    )
    particle_collector.record_initial(
        state=initial_branch_state(
            initial_population_view,
            scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
            seed=PILOT_SEED,
        )
    )

    def checkpoint(branch: str, tick: int, snapshot: Mapping[str, Any]) -> None:
        particle_collector.record_checkpoint(branch=branch, tick=tick, snapshot=snapshot)

    # Add a private observation wrapper without changing the frozen kernel or
    # prompt renderer.  The wrapper records parent means immediately before
    # and after the existing assimilation function, then delegates unchanged.
    original_assimilate = forum_hero_module.assimilate_forum_llm_updates_v2

    def audited_assimilate(**kwargs: Any) -> Mapping[str, Any]:
        state = kwargs["state"]
        updates = kwargs["updates"]
        before = state.ensemble.parent_means().copy()
        before_resampling = int(state.ensemble.resampling_count)
        result = original_assimilate(**kwargs)
        after = state.ensemble.parent_means().copy()
        resampling_happened = int(state.ensemble.resampling_count) > before_resampling
        for parent_index in sorted(updates):
            row = updates[parent_index]
            particle_collector.record_assimilation(
                {
                    "branch": str(kwargs["branch"]),
                    "tick": int(kwargs["tick"]),
                    "parent_index": int(parent_index),
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "response_sha256": str(row["response_sha256"]),
                    "innovation_id": canonical_hash(
                        [
                            "forum-fidelity-v2-observation",
                            kwargs["scenario_id"],
                            kwargs["branch"],
                            kwargs["tick"],
                            kwargs["seed"],
                            int(parent_index),
                            row["prompt_sha256"],
                            row["response_sha256"],
                        ]
                    ),
                    "before_state": {
                        field: float(before[int(parent_index), ordinal])
                        for ordinal, field in enumerate(
                            forum_hero_module.PARTICLE_STATE_FIELDS
                        )
                    },
                    "after_state": {
                        field: float(after[int(parent_index), ordinal])
                        for ordinal, field in enumerate(
                            forum_hero_module.PARTICLE_STATE_FIELDS
                        )
                    },
                    "resampled": resampling_happened,
                }
            )
        return result

    shared_snapshot: dict[str, Any] = {}

    def capture_shared(_branch: str, snapshot: Mapping[str, Any]) -> None:
        shared_snapshot.update(copy.deepcopy(dict(snapshot)))

    forum_hero_module.assimilate_forum_llm_updates_v2 = audited_assimilate
    try:
        shared = run_forum_hero_experiment(
            population_release=population,
            relationship_release=relationships,
            scenario_stake_release=stakes,
            ecology_calibration_release=ecology,
            resident_source=source,
            governance_source=source,
            expected_model=EXPECTED_MODEL,
            scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
            seed=PILOT_SEED,
            start_tick=FOUR_DIMENSION_START_TICK,
            end_tick=FOUR_DIMENSION_SHARED_END_TICK,
            branches=(SHARED_BASELINE_BRANCH,),
            resident_budget_by_tick={
                FOUR_DIMENSION_START_TICK: FOUR_DIMENSION_RESIDENT_BUDGET
            },
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
            snapshot_callback=capture_shared,
            tick_snapshot_callback=checkpoint,
            governance_decision_ticks=(),
            resident_turn_schema_version=FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
            require_scenario_entry_post=True,
            relationship_attention_enabled=True,
            private_actions_enabled=True,
            public_anonymity_enabled=True,
            activation_policy=COST_AWARE_ACTIVATION_POLICY,
        )
        if int(shared_snapshot.get("tick", -1)) != FOUR_DIMENSION_SHARED_END_TICK:
            raise ValueError("offline replay did not commit shared Tick 3")
        continuation_budgets = {
            tick: FOUR_DIMENSION_RESIDENT_BUDGET for tick in range(4, 10)
        }
        action_overrides = {
            "D": (
                "verified_update",
                "evidence_dashboard",
                "listening_session",
                "support_channel",
                "queue_status",
                "peer_feedback",
                "targeted_prebunk",
                "bridge_outreach",
            )
        }
        continuations = {}
        for branch in FOUR_DIMENSION_BRANCHES:
            continuations[branch] = run_forum_hero_experiment(
                population_release=population,
                relationship_release=relationships,
                scenario_stake_release=stakes,
                ecology_calibration_release=ecology,
                resident_source=source,
                governance_source=source,
                expected_model=EXPECTED_MODEL,
                scenario_id=GOVERNANCE_LEGITIMACY_SCENARIO,
                seed=PILOT_SEED,
                start_tick=4,
                end_tick=9,
                branches=(branch,),
                resident_budget_by_tick=continuation_budgets,
                mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
                initial_snapshot=shared_snapshot,
                activation_selection_salt=canonical_hash(
                    ["four-dimension-resource-hero-paired-sample-v1", PILOT_SEED]
                ),
                governance_decision_ticks=FOUR_DIMENSION_DECISION_TICKS,
                governance_action_overrides=action_overrides,
                tick_snapshot_callback=checkpoint,
                resident_turn_schema_version=FORUM_AGENT_TURN_V5_SCHEMA_VERSION,
                relationship_attention_enabled=True,
                private_actions_enabled=True,
                public_anonymity_enabled=True,
                activation_policy=COST_AWARE_ACTIVATION_POLICY,
            )
    finally:
        forum_hero_module.assimilate_forum_llm_updates_v2 = original_assimilate
    if identity_source.miss_count or safety_miss_source.cache_miss_count:
        raise ValueError("offline replay unexpectedly missed settled evidence")
    original = _result(_load(arguments.original_result.resolve()))
    shared_timeline = shared["branches"][SHARED_BASELINE_BRANCH]["timeline"]
    replay_timelines = {
        branch: (
            copy.deepcopy(shared_timeline)
            + continuations[branch]["branches"][branch]["timeline"][4:]
        )
        for branch in FOUR_DIMENSION_BRANCHES
    }
    original_timelines = {
        branch: original["branches"][branch]["timeline"][:10]
        for branch in FOUR_DIMENSION_BRANCHES
    }
    original_projection = _semantic_projection(original_timelines)
    replay_projection = _semantic_projection(replay_timelines)
    original_projection_sha = canonical_sha256(original_projection)
    replay_projection_sha = canonical_sha256(replay_projection)
    if original_projection_sha != replay_projection_sha:
        raise ValueError(
            "offline replay changed the committed Tick 0-9 timelines: "
            + _first_difference(original_projection, replay_projection)
        )

    private_dir = arguments.private_dir.resolve()
    private_dir.mkdir(parents=True, exist_ok=True)
    private_particle = particle_collector.write_private_npz(
        private_dir / "full-particle-evaluation-snapshot-v1.npz"
    )
    existing = _load(arguments.existing_fidelity.resolve())
    dimensions = existing.get("dimensions", {})
    legacy_metrics = (
        dimensions.get("micro_persona_consistency", {}).get("independent_blind_review")
        or dimensions.get("persona_and_one_step_state", {}).get("metrics", {})
    )
    usage = source.usage_snapshot
    replay_audit = {
        "execution_mode": "identity_bound_settled_cache_replay_no_network_transport",
        "semantic_turn_count": int(usage.get("semantic_turns", 0)),
        "exact_cache_hit_count": int(usage.get("cache_hits", 0)),
        "provider_calls": int(usage.get("provider_calls", 0)),
        "provider_attempts": int(usage.get("provider_attempts", 0)),
        "provider_tokens": int(usage.get("total_tokens", 0)),
        "offline_cache_miss_count": identity_source.miss_count,
        "timeline_projection_through_tick": 9,
        "timeline_projection_matches_original": True,
        "semantic_projection_sha256": replay_projection_sha,
        "original_result_sha256": str(original["result_sha256"]),
        "private_particle_snapshot": {
            "frame_count": len(private_particle["frames"]),
            "assimilation_count": private_particle["assimilation_count"],
            "file_size_bytes": private_particle["file_size_bytes"],
            "stored_below_run_directory": True,
            "public_export_allowed": False,
        },
        "terminal_tick_note": (
            "The formal public result remains complete through Tick 10. Full-particle "
            "reconstruction stops at Tick 9 because the later improved renderer replaced "
            "the old prompt bytes and terminal turns have no next-tick persona audit key."
        ),
    }
    evaluation = build_fidelity_evaluation_v2(
        persona_evidence=persona_evidence,
        assimilations=particle_collector.assimilations,
        snapshots=particle_collector.snapshots,
        scenario_stake_release=stakes,
        population_agent_ids=population_agent_ids,
        legacy_blind_review_metrics=legacy_metrics,
        replay_audit=replay_audit,
    )
    public_dir = arguments.public_dir.resolve()
    _atomic_json(public_dir / "fidelity-evaluation-v3.json", evaluation)
    _write_text(public_dir / "persona-consistency-v3.svg", _persona_svg(evaluation))
    _write_text(
        public_dir / "population-heterogeneity-v2.svg",
        _heterogeneity_svg(evaluation),
    )
    _write_text(
        public_dir / "particle-distributions-v2.svg",
        _distribution_svg(evaluation),
    )
    summary = {
        "status": "complete",
        "public_evaluation": str(public_dir / "fidelity-evaluation-v3.json"),
        "private_particle_snapshot": str(
            private_dir / "full-particle-evaluation-snapshot-v1.npz"
        ),
        "semantic_turns": replay_audit["semantic_turn_count"],
        "provider_calls": 0,
        "timeline_projection_matches_original": True,
        "persona_status": evaluation["dimensions"]["micro_persona_consistency"][
            "status"
        ],
        "heterogeneity_status": evaluation["dimensions"][
            "population_heterogeneity"
        ]["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
