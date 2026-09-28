"""Small, auditable cross-shock acceptance panel for the variant relevance gate."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DecisionCache,
    LLMAgentError,
    TokenBudget,
    Transport,
    VectorEngineClient,
    _canonical_hash,
    _run_phase,
    build_representative_agent_population,
    load_llm_agent_inputs,
    load_representative_agent_inputs,
)
from .policies import POLICIES
from .shock_replay import DEFAULT_DEFINITIONS


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_DIRECT_REVIEW = (
    REPO_ROOT
    / "specs"
    / "004-data-grounded-representative-population"
    / "cross-shock-variant-direct-review-v1.json"
)
PANEL_SCENARIOS = (
    "tongzhou_governance_information_shock",
    "lecture_external_incident_shock",
)
PANEL_VARIANTS = (
    "archetype-03-v2",
    "archetype-12-v3",
    "archetype-07-v2",
    "archetype-02-v1",
)


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = resolved.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def load_gate_panel_review(
    path: str | Path = DEFAULT_DIRECT_REVIEW,
) -> dict[str, Any]:
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if review.get("schema_version") != "cross-shock-variant-direct-review-v1":
        raise LLMAgentError("variant gate panel requires the frozen direct review")
    indexed = {
        item.get("variant_id"): item for item in review.get("variant_reviews", [])
    }
    missing = sorted(set(PANEL_VARIANTS) - set(indexed))
    if missing:
        raise LLMAgentError(
            f"direct review is missing panel variants: {', '.join(missing)}"
        )
    selected = [indexed[variant_id] for variant_id in PANEL_VARIANTS]
    if len({item["archetype"] for item in selected}) != len(selected):
        raise LLMAgentError("gate panel variants must cover distinct archetypes")
    intents = {item["coverage_intent"] for item in selected}
    if "challenging_specificity_case" not in intents:
        raise LLMAgentError("gate panel lacks the challenging specificity case")
    if "selective_nonresponse_control" not in intents:
        raise LLMAgentError("gate panel lacks a selective nonresponse control")
    return {
        "schema_version": review["schema_version"],
        "source_artifacts": review["source_artifacts"],
        "review_method": review["review_method"],
        "variant_reviews": selected,
        "claim_boundary": review["claim_boundary"],
    }


def _select_reviewed_exemplars(
    *,
    definition: dict[str, Any],
    release: dict[str, Any],
    evidence: dict[str, Any],
    panel_review: dict[str, Any],
    scenario_id: str,
    seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    population, population_evidence = build_representative_agent_population(
        definition,
        release,
        evidence,
        agent_count=48,
        seed=seed,
    )
    by_id = {agent.agent_id: agent for agent in population}
    exemplar_field = (
        "lecture_exemplar_agent"
        if scenario_id == "lecture_external_incident_shock"
        else "tongzhou_exemplar_agent"
    )
    selected = []
    selection = []
    for reviewed in panel_review["variant_reviews"]:
        agent_id = reviewed[exemplar_field]
        agent = by_id.get(agent_id)
        if agent is None:
            raise LLMAgentError(f"reviewed exemplar is absent: {agent_id}")
        if agent.persona_variant_id != reviewed["variant_id"]:
            raise LLMAgentError(
                f"reviewed exemplar {agent_id} no longer maps to "
                f"{reviewed['variant_id']}"
            )
        selected.append(agent)
        selection.append(
            {
                "synthetic_agent_id": agent.agent_id,
                "archetype_id": agent.archetype_id,
                "archetype_label": agent.archetype_label,
                "variant_id": agent.persona_variant_id,
                "variant_label": agent.persona_variant_label,
                "population_month": agent.population_month,
                "coverage_intent": reviewed["coverage_intent"],
                "previous_cross_snapshot_rating": reviewed[
                    "cross_snapshot_rating"
                ],
                "previous_review_note": reviewed["review_note"],
                "action_horizon": {
                    key: agent.persona[key]
                    for key in (
                        "simulation_instruction",
                        "reaction_logic",
                        "uncertainty_behavior",
                    )
                },
            }
        )
    if len({agent.agent_id for agent in selected}) != len(PANEL_VARIANTS):
        raise LLMAgentError("gate panel selected duplicate exemplar agents")
    return selected, {
        "full_population_size_before_selection": len(population),
        "selection_method": (
            "exact exemplar IDs frozen by the prior cross-shock direct review"
        ),
        "population_evidence": population_evidence,
        "agents": selection,
    }


def _compact_panel_phase(phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": phase["phase_id"],
        "label": phase["label"],
        "agent_turns": phase["agent_turns"],
        "variant_route_counts": phase["variant_route_counts"],
        "contract_adjustment_counts": phase["contract_adjustment_counts"],
        "usage": phase["usage"],
        "provider_attempts": phase["provider_attempts"],
        "format_repairs": phase["format_repairs"],
        "cache_hits": phase["cache_hits"],
        "agent_decisions": phase["agent_decisions"],
    }


def run_variant_gate_acceptance_panel(
    benchmark: dict[str, Any],
    definitions: dict[str, Any],
    *,
    release: dict[str, Any],
    evidence: dict[str, Any],
    panel_review: dict[str, Any],
    client: VectorEngineClient,
    seed: int = 20260722,
    max_concurrency: int = 4,
    max_agent_turns: int = 32,
    max_total_tokens: int = 120000,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    input_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 1 <= max_concurrency <= len(PANEL_VARIANTS):
        raise LLMAgentError(
            f"panel max_concurrency must be between 1 and {len(PANEL_VARIANTS)}"
        )
    expected_turns = len(PANEL_SCENARIOS) * len(PANEL_VARIANTS) * 4
    if expected_turns > max_agent_turns:
        raise LLMAgentError(
            f"panel requires {expected_turns} turns, above max_agent_turns"
        )
    definitions_by_id = {
        item["scenario_id"]: item for item in definitions["events"]
    }
    benchmark_by_id = {
        item["scenario_id"]: item for item in benchmark["events"]
    }
    if any(
        scenario_id not in definitions_by_id or scenario_id not in benchmark_by_id
        for scenario_id in PANEL_SCENARIOS
    ):
        raise LLMAgentError("panel scenarios are missing from frozen inputs")

    budget = TokenBudget(max_total_tokens)
    scenarios = []
    for scenario_id in PANEL_SCENARIOS:
        definition = definitions_by_id[scenario_id]
        event = benchmark_by_id[scenario_id]
        selected, selection = _select_reviewed_exemplars(
            definition=definition,
            release=release,
            evidence=evidence,
            panel_review=panel_review,
            scenario_id=scenario_id,
            seed=seed,
        )

        baseline_agents = copy.deepcopy(selected)
        baseline = _run_phase(
            event=event,
            definition=definition,
            phase=event["phases"][0],
            policy=POLICIES["baseline"],
            agents=baseline_agents,
            previous_public=None,
            client=client,
            budget=budget,
            max_concurrency=max_concurrency,
        )

        event_agents = copy.deepcopy(selected)
        phases = [_compact_panel_phase(baseline)]
        previous_public = None
        for phase in event["phases"][1:]:
            result = _run_phase(
                event=event,
                definition=definition,
                phase=phase,
                policy=POLICIES["baseline"],
                agents=event_agents,
                previous_public=previous_public,
                client=client,
                budget=budget,
                max_concurrency=max_concurrency,
            )
            phases.append(_compact_panel_phase(result))
            previous_public = result
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "event_title": event["title"],
                "shock_origin": event["shock_origin"],
                "population_month": selection["agents"][0]["population_month"],
                "selection": selection,
                "baseline_personal_story_carried_into_shock": False,
                "phases": phases,
            }
        )

    result = {
        "schema_version": "variant-gate-acceptance-panel-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": "targeted_cross_shock_variant_gate_acceptance",
        "panel_design": {
            "scenario_ids": list(PANEL_SCENARIOS),
            "variant_ids": list(PANEL_VARIANTS),
            "variant_count_per_scenario": len(PANEL_VARIANTS),
            "semantic_phases": ["baseline", "burst", "spread", "decay"],
            "selection_changed_after_seeing_new_outputs": False,
            "frequency_or_population_estimation_intended": False,
        },
        "evidence": {
            "dataset_snapshot_id": benchmark["dataset_snapshot_id"],
            "analysis_snapshot_id": benchmark["analysis_snapshot_id"],
            "benchmark_fingerprint": benchmark["benchmark_fingerprint"],
            "definition_fingerprint": benchmark["definition_fingerprint"],
            "historical_targets_sent_to_provider": False,
            "raw_forum_rows_sent_to_provider": False,
            "panel_review_source": panel_review["schema_version"],
            "prior_source_artifacts": panel_review["source_artifacts"],
            "input_identities": input_identities,
        },
        "provider": {
            "base_url": client.base_url,
            "endpoint": "/chat/completions",
            "requested_model": client.model,
            "reasoning_effort": client.reasoning_effort,
            "api_key_environment_variable": api_key_env,
            "api_key_persisted": False,
        },
        "execution": {
            "seed_used_for_population_selection_only": seed,
            "required_agent_turns": expected_turns,
            "max_agent_turns": max_agent_turns,
            "max_concurrency": max_concurrency,
            "token_budget": budget.snapshot(),
            "decision_cache": (
                client.decision_cache.snapshot()
                if client.decision_cache is not None
                else {"enabled": False}
            ),
        },
        "scenarios": scenarios,
        "review_status": {
            "automatic_contracts_passed": True,
            "direct_semantic_review": "pending",
            "existing_partial_rating_changed": False,
        },
        "claim_boundary": (
            "This diagnostic panel tests gate sensitivity, specificity, and "
            "variant-mechanism preservation. It does not estimate population "
            "frequencies, historical fit, causal effects, or policy effects."
        ),
    }
    result["panel_fingerprint"] = _canonical_hash(
        {
            key: value
            for key, value in result.items()
            if key not in {"generated_at_utc", "panel_fingerprint"}
        }
    )
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if client._api_key in rendered:
        raise LLMAgentError("API credential appeared in gate panel output")
    return result


def write_variant_gate_acceptance_panel(
    result: dict[str, Any], output_path: str | Path
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(path)


def run_variant_gate_acceptance_pipeline(
    *,
    output_path: str | Path,
    persona_release_path: str | Path,
    persona_evidence_path: str | Path,
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    direct_review_path: str | Path = DEFAULT_DIRECT_REVIEW,
    decision_cache_path: str | Path | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    seed: int = 20260722,
    max_concurrency: int = 4,
    max_agent_turns: int = 32,
    max_total_tokens: int = 120000,
    max_output_tokens: int = 220,
    timeout_seconds: float = 60.0,
    transport: Transport | None = None,
) -> dict[str, Any]:
    benchmark, definitions = load_llm_agent_inputs(
        benchmark_path=benchmark_path,
        definitions_path=definitions_path,
    )
    release, evidence = load_representative_agent_inputs(
        persona_release_path=persona_release_path,
        persona_evidence_path=persona_evidence_path,
    )
    panel_review = load_gate_panel_review(direct_review_path)
    cache_path = (
        Path(decision_cache_path)
        if decision_cache_path is not None
        else Path(output_path).with_suffix(Path(output_path).suffix + ".decisions.json")
    )
    client = VectorEngineClient.from_environment(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        transport=transport,
        decision_cache=DecisionCache(cache_path),
    )
    result = run_variant_gate_acceptance_panel(
        benchmark,
        definitions,
        release=release,
        evidence=evidence,
        panel_review=panel_review,
        client=client,
        seed=seed,
        max_concurrency=max_concurrency,
        max_agent_turns=max_agent_turns,
        max_total_tokens=max_total_tokens,
        api_key_env=api_key_env,
        input_identities={
            "persona_release": _file_identity(persona_release_path),
            "persona_evidence": _file_identity(persona_evidence_path),
            "direct_review": _file_identity(direct_review_path),
            "raw_text_in_inputs": False,
        },
    )
    write_variant_gate_acceptance_panel(result, output_path)
    return result
