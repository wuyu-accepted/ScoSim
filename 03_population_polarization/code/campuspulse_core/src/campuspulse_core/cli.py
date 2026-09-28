"""Command-line interface for validation, experiments, and calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import calibrate_csvs
from .config import ConfigError, load_scenario
from .experiment import run_comparison, write_experiment_outputs
from .forum_expression import ForumExpressionError, run_forum_expression_pilot
from .lecture_thread_ecology import (
    LectureThreadError,
    run_lecture_thread_ecology_pilot,
)
from .llm_agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    LLMAgentError,
    run_live_llm_agent_pipeline,
)
from .policies import POLICIES
from .representative_taxonomy import (
    RepresentativeTaxonomyError,
    run_hard_pair_audit,
    run_taxonomy_pilot,
)
from .representative_release import (
    release_reviewed_taxonomy,
    revise_taxonomy_from_review,
)
from .representative_personas import (
    build_persona_evidence_packs,
    release_reviewed_personas,
    run_persona_profile_pilot,
)
from .pilot import PilotDataError, evaluate_pilot, write_pilot_json
from .pilot_report import generate_pilot_report
from .report import generate_html_report
from .robustness import run_robustness_analysis, write_robustness_json
from .semantic_population import (
    SemanticPopulationError,
    run_context_refinement_pilot,
    run_semantic_card_pilot,
)
from .shock_replay import (
    DEFAULT_DEFINITIONS,
    ShockReplayError,
    run_shock_replay_pipeline,
)
from .validation import evaluate_forum_data, write_validation_json
from .validation_report import generate_validation_report
from .variant_gate_acceptance import (
    DEFAULT_DIRECT_REVIEW,
    run_variant_gate_acceptance_pipeline,
)


HERE = Path(__file__).resolve().parent
SCENARIO_DIR = HERE / "scenarios"
BUNDLED_SCENARIOS = {
    path.stem: path for path in sorted(SCENARIO_DIR.glob("*.json"))
}


def resolve_scenario(value: str) -> Path:
    if value in BUNDLED_SCENARIOS:
        return BUNDLED_SCENARIOS[value]
    path = Path(value)
    if path.exists():
        return path
    raise ConfigError(
        f"unknown scenario '{value}'; available: {', '.join(sorted(BUNDLED_SCENARIOS))}"
    )


def _run_scenario(
    scenario_value: str,
    repetitions: int,
    seed: int,
    output_root: Path,
    policy_ids: list[str] | None = None,
) -> dict:
    config = load_scenario(resolve_scenario(scenario_value))
    selected = [POLICIES[policy_id] for policy_id in policy_ids] if policy_ids else None
    result = run_comparison(config, repetitions=repetitions, base_seed=seed, policies=selected)
    output_dir = output_root / config.scenario_id
    paths = write_experiment_outputs(result, output_dir)
    paths["report_html"] = generate_html_report(result, output_dir / "report.html")
    return {"scenario_id": config.scenario_id, "ranking": result["ranking"], "paths": paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campus-pulse",
        description="Data-bounded campus demand-response policy stress testing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List bundled scenarios and policies")

    validate = sub.add_parser("validate", help="Validate scenario evidence and contract")
    validate.add_argument("--scenario", default="all")

    run = sub.add_parser("run", help="Run one scenario across one or all policies")
    run.add_argument("--scenario", required=True)
    run.add_argument("--policy", choices=["all", *sorted(POLICIES)], default="all")
    run.add_argument("--repetitions", type=int, default=10)
    run.add_argument("--seed", type=int, default=20260721)
    run.add_argument("--output", type=Path, default=HERE / "runs" / "mvp")

    run_all = sub.add_parser("run-all", help="Run all bundled scenarios and policies")
    run_all.add_argument("--repetitions", type=int, default=10)
    run_all.add_argument("--seed", type=int, default=20260721)
    run_all.add_argument("--output", type=Path, default=HERE / "runs" / "mvp")

    robustness = sub.add_parser(
        "robustness",
        help="Stress-test policy rankings across declared uncertainty ranges",
    )
    robustness.add_argument("--scenario", default="all")
    robustness.add_argument("--samples", type=int, default=80)
    robustness.add_argument("--repetitions", type=int, default=2)
    robustness.add_argument("--seed", type=int, default=20260721)
    robustness.add_argument("--output", type=Path, default=HERE / "data" / "robustness")

    calibrate = sub.add_parser("calibrate", help="Create aggregate proxies from forum CSVs")
    calibrate.add_argument("--posts", required=True, type=Path)
    calibrate.add_argument("--comments", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--start", default="2025-12-01")
    calibrate.add_argument("--end", default="2026-07-21")

    replay_shocks = sub.add_parser(
        "replay-shocks",
        help="Build and score the aggregate-only offline replay for both shocks",
    )
    replay_shocks.add_argument("--posts", required=True, type=Path)
    replay_shocks.add_argument("--comments", required=True, type=Path)
    replay_shocks.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    replay_shocks.add_argument(
        "--output", type=Path, default=HERE / "runs" / "shock-replay-v1"
    )
    replay_shocks.add_argument("--repetitions", type=int, default=100)
    replay_shocks.add_argument("--seed", type=int, default=20260722)

    llm_agents = sub.add_parser(
        "run-llm-agents",
        help="Run persistent GPT-5.6 agents against one frozen shock",
    )
    llm_agents.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    llm_agents.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    llm_agents.add_argument(
        "--scenario", default="tongzhou_governance_information_shock"
    )
    llm_agents.add_argument(
        "--policies", default="baseline,faq_deflection,combined"
    )
    llm_agents.add_argument(
        "--output", type=Path, default=HERE / "runs" / "llm-agents-v1.json"
    )
    llm_agents.add_argument("--agents", type=int, default=12)
    llm_agents.add_argument("--seed", type=int, default=20260722)
    llm_agents.add_argument("--model", default=DEFAULT_MODEL)
    llm_agents.add_argument("--base-url", default=DEFAULT_BASE_URL)
    llm_agents.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    llm_agents.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    llm_agents.add_argument("--max-concurrency", type=int, default=4)
    llm_agents.add_argument("--max-agent-turns", type=int, default=500)
    llm_agents.add_argument("--max-total-tokens", type=int, default=250000)
    llm_agents.add_argument("--max-output-tokens", type=int, default=220)
    llm_agents.add_argument("--timeout-seconds", type=float, default=60.0)
    llm_agents.add_argument(
        "--persona-release",
        type=Path,
        help="Directly reviewed behavioral persona release",
    )
    llm_agents.add_argument(
        "--persona-evidence",
        type=Path,
        help="Cross-time and monthly persona evidence packs",
    )
    llm_agents.add_argument(
        "--decision-cache",
        type=Path,
        help="Optional resumable synthetic-decision cache (defaults beside output)",
    )

    gate_panel = sub.add_parser(
        "variant-gate-panel",
        help="Run the frozen four-variant cross-shock gate acceptance panel",
    )
    gate_panel.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    gate_panel.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    gate_panel.add_argument(
        "--direct-review", type=Path, default=DEFAULT_DIRECT_REVIEW
    )
    gate_panel.add_argument("--persona-release", required=True, type=Path)
    gate_panel.add_argument("--persona-evidence", required=True, type=Path)
    gate_panel.add_argument("--output", required=True, type=Path)
    gate_panel.add_argument("--decision-cache", type=Path)
    gate_panel.add_argument("--seed", type=int, default=20260722)
    gate_panel.add_argument("--model", default=DEFAULT_MODEL)
    gate_panel.add_argument("--base-url", default=DEFAULT_BASE_URL)
    gate_panel.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    gate_panel.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="none",
    )
    gate_panel.add_argument("--max-concurrency", type=int, default=4)
    gate_panel.add_argument("--max-agent-turns", type=int, default=32)
    gate_panel.add_argument("--max-total-tokens", type=int, default=120000)
    gate_panel.add_argument("--max-output-tokens", type=int, default=220)
    gate_panel.add_argument("--timeout-seconds", type=float, default=60.0)

    expression_pilot = sub.add_parser(
        "expression-pilot",
        help="Render a held-out post/reply style pilot over frozen agent decisions",
    )
    expression_pilot.add_argument("--posts", required=True, type=Path)
    expression_pilot.add_argument("--comments", required=True, type=Path)
    expression_pilot.add_argument(
        "--simulation-artifacts", required=True, nargs="+", type=Path
    )
    expression_pilot.add_argument("--persona-release", required=True, type=Path)
    expression_pilot.add_argument("--output", required=True, type=Path)
    expression_pilot.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    expression_pilot.add_argument("--per-scenario", type=int, default=8)
    expression_pilot.add_argument("--seed", type=int, default=20260722)
    expression_pilot.add_argument("--model", default=DEFAULT_MODEL)
    expression_pilot.add_argument("--base-url", default=DEFAULT_BASE_URL)
    expression_pilot.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    expression_pilot.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    expression_pilot.add_argument("--max-concurrency", type=int, default=8)
    expression_pilot.add_argument("--max-total-tokens", type=int, default=180000)
    expression_pilot.add_argument("--max-output-tokens", type=int, default=700)
    expression_pilot.add_argument("--timeout-seconds", type=float, default=120.0)
    expression_pilot.add_argument(
        "--allow-provider-text",
        action="store_true",
        help="Explicitly permit ephemeral redacted, non-shock style exemplars to be sent",
    )

    thread_ecology = sub.add_parser(
        "lecture-thread-pilot",
        help="Code observed reply functions and render a held-out lecture thread pilot",
    )
    thread_ecology.add_argument("--posts", required=True, type=Path)
    thread_ecology.add_argument("--comments", required=True, type=Path)
    thread_ecology.add_argument("--simulation-artifact", required=True, type=Path)
    thread_ecology.add_argument("--expression-artifact", required=True, type=Path)
    thread_ecology.add_argument("--persona-release", required=True, type=Path)
    thread_ecology.add_argument("--output", required=True, type=Path)
    thread_ecology.add_argument(
        "--reuse-observed-evidence",
        type=Path,
        help="Reuse aggregate-only observed function coding from an earlier completed pilot",
    )
    thread_ecology.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    thread_ecology.add_argument("--sample-per-split", type=int, default=48)
    thread_ecology.add_argument("--batch-size", type=int, default=8)
    thread_ecology.add_argument("--reply-count", type=int, default=12)
    thread_ecology.add_argument("--seed", type=int, default=20260722)
    thread_ecology.add_argument("--model", default=DEFAULT_MODEL)
    thread_ecology.add_argument("--base-url", default=DEFAULT_BASE_URL)
    thread_ecology.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    thread_ecology.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    thread_ecology.add_argument("--max-concurrency", type=int, default=8)
    thread_ecology.add_argument("--max-total-tokens", type=int, default=180000)
    thread_ecology.add_argument("--max-output-tokens", type=int, default=800)
    thread_ecology.add_argument("--timeout-seconds", type=float, default=120.0)
    thread_ecology.add_argument(
        "--allow-provider-text",
        action="store_true",
        help="Explicitly permit ephemeral redacted parent/reply pairs and style examples",
    )

    semantic_pilot = sub.add_parser(
        "semantic-pilot",
        help="Build contact-redacted GPT semantic cards for a stratified post sample",
    )
    semantic_pilot.add_argument("--posts", required=True, type=Path)
    semantic_pilot.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    semantic_pilot.add_argument("--output", required=True, type=Path)
    semantic_pilot.add_argument("--sample-size", type=int, default=1000)
    semantic_pilot.add_argument("--seed", type=int, default=20260722)
    semantic_pilot.add_argument("--start", default="2025-12-01")
    semantic_pilot.add_argument("--end", default="2026-07-21")
    semantic_pilot.add_argument("--platform-marker", default="NP")
    semantic_pilot.add_argument("--batch-size", type=int, default=20)
    semantic_pilot.add_argument("--max-concurrency", type=int, default=4)
    semantic_pilot.add_argument("--max-total-tokens", type=int, default=800000)
    semantic_pilot.add_argument("--model", default=DEFAULT_MODEL)
    semantic_pilot.add_argument("--base-url", default=DEFAULT_BASE_URL)
    semantic_pilot.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    semantic_pilot.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    semantic_pilot.add_argument("--max-output-tokens", type=int, default=8000)
    semantic_pilot.add_argument("--timeout-seconds", type=float, default=120.0)
    semantic_pilot.add_argument(
        "--allow-provider-text",
        action="store_true",
        help="Explicitly permit sending contact-redacted post text to the provider",
    )

    context_pilot = sub.add_parser(
        "semantic-context-pilot",
        help="Re-evaluate uncertain semantic cards with redacted linked-comment context",
    )
    context_pilot.add_argument("--posts", required=True, type=Path)
    context_pilot.add_argument("--comments", required=True, type=Path)
    context_pilot.add_argument("--semantic-result", required=True, type=Path)
    context_pilot.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    context_pilot.add_argument("--output", required=True, type=Path)
    context_pilot.add_argument("--batch-size", type=int, default=8)
    context_pilot.add_argument("--max-comments-per-post", type=int, default=5)
    context_pilot.add_argument("--max-concurrency", type=int, default=3)
    context_pilot.add_argument("--max-total-tokens", type=int, default=300000)
    context_pilot.add_argument("--model", default=DEFAULT_MODEL)
    context_pilot.add_argument("--base-url", default=DEFAULT_BASE_URL)
    context_pilot.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    context_pilot.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    context_pilot.add_argument("--max-output-tokens", type=int, default=8000)
    context_pilot.add_argument("--timeout-seconds", type=float, default=120.0)
    context_pilot.add_argument("--allow-provider-text", action="store_true")

    taxonomy_pilot = sub.add_parser(
        "taxonomy-pilot",
        help="Induce and assign a hierarchical representative-population taxonomy",
    )
    taxonomy_pilot.add_argument("--semantic-result", required=True, type=Path)
    taxonomy_pilot.add_argument("--context-result", type=Path)
    taxonomy_pilot.add_argument("--frozen-taxonomy", type=Path)
    taxonomy_pilot.add_argument("--output", required=True, type=Path)
    taxonomy_pilot.add_argument("--target-archetypes", type=int, default=12)
    taxonomy_pilot.add_argument("--minimum-archetypes", type=int, default=8)
    taxonomy_pilot.add_argument("--maximum-archetypes", type=int, default=16)
    taxonomy_pilot.add_argument("--batch-size", type=int, default=10)
    taxonomy_pilot.add_argument("--max-concurrency", type=int, default=8)
    taxonomy_pilot.add_argument("--max-total-tokens", type=int, default=300000)
    taxonomy_pilot.add_argument("--model", default=DEFAULT_MODEL)
    taxonomy_pilot.add_argument("--base-url", default=DEFAULT_BASE_URL)
    taxonomy_pilot.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    taxonomy_pilot.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    taxonomy_pilot.add_argument("--max-output-tokens", type=int, default=8000)
    taxonomy_pilot.add_argument("--timeout-seconds", type=float, default=120.0)

    hard_pairs = sub.add_parser(
        "taxonomy-hard-pairs",
        help="Resolve unstable taxonomy assignments with reversed candidate-order judgments",
    )
    hard_pairs.add_argument("--semantic-result", required=True, type=Path)
    hard_pairs.add_argument("--context-result", type=Path)
    hard_pairs.add_argument("--assignment-runs", required=True, nargs="+", type=Path)
    hard_pairs.add_argument("--output", required=True, type=Path)
    hard_pairs.add_argument(
        "--selection-mode",
        choices=["all_boundaries", "primary_disagreement"],
        default="all_boundaries",
    )
    hard_pairs.add_argument("--batch-size", type=int, default=10)
    hard_pairs.add_argument("--max-concurrency", type=int, default=6)
    hard_pairs.add_argument("--max-total-tokens", type=int, default=150000)
    hard_pairs.add_argument("--model", default=DEFAULT_MODEL)
    hard_pairs.add_argument("--base-url", default=DEFAULT_BASE_URL)
    hard_pairs.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    hard_pairs.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    hard_pairs.add_argument("--max-output-tokens", type=int, default=6000)
    hard_pairs.add_argument("--timeout-seconds", type=float, default=120.0)

    taxonomy_release = sub.add_parser(
        "taxonomy-release",
        help="Apply fingerprint-bound direct semantic review to a taxonomy pilot",
    )
    taxonomy_release.add_argument("--semantic-result", required=True, type=Path)
    taxonomy_release.add_argument("--taxonomy-result", required=True, type=Path)
    taxonomy_release.add_argument("--hard-pair-result", required=True, type=Path)
    taxonomy_release.add_argument("--adjudication-manifest", required=True, type=Path)
    taxonomy_release.add_argument("--output", required=True, type=Path)

    taxonomy_revise = sub.add_parser(
        "taxonomy-revise",
        help="Apply a fingerprint-bound cross-time qualitative taxonomy revision",
    )
    taxonomy_revise.add_argument("--taxonomy-assignment", required=True, type=Path)
    taxonomy_revise.add_argument("--revision-manifest", required=True, type=Path)
    taxonomy_revise.add_argument("--output", required=True, type=Path)

    persona_evidence = sub.add_parser(
        "persona-evidence",
        help="Build cross-time semantic evidence packs for representative personas",
    )
    persona_evidence.add_argument("--posts", required=True, type=Path)
    persona_evidence.add_argument("--semantic-result", required=True, type=Path)
    persona_evidence.add_argument("--context-result", type=Path)
    persona_evidence.add_argument("--taxonomy-release", required=True, type=Path)
    persona_evidence.add_argument(
        "--definitions", type=Path, default=DEFAULT_DEFINITIONS
    )
    persona_evidence.add_argument("--representative-per-archetype", type=int, default=10)
    persona_evidence.add_argument("--contrastive-per-archetype", type=int, default=5)
    persona_evidence.add_argument("--output", required=True, type=Path)

    persona_pilot = sub.add_parser(
        "persona-pilot",
        help="Generate evidence-bound representative persona variants",
    )
    persona_pilot.add_argument("--evidence-pack", required=True, type=Path)
    persona_pilot.add_argument("--output", required=True, type=Path)
    persona_pilot.add_argument("--max-concurrency", type=int, default=8)
    persona_pilot.add_argument("--max-total-tokens", type=int, default=180000)
    persona_pilot.add_argument("--model", default=DEFAULT_MODEL)
    persona_pilot.add_argument("--base-url", default=DEFAULT_BASE_URL)
    persona_pilot.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    persona_pilot.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    persona_pilot.add_argument("--max-output-tokens", type=int, default=5000)
    persona_pilot.add_argument("--timeout-seconds", type=float, default=180.0)

    persona_release = sub.add_parser(
        "persona-release",
        help="Release persona profiles after direct qualitative text review",
    )
    persona_release.add_argument("--persona-pilot", required=True, type=Path)
    persona_release.add_argument("--evidence-pack", required=True, type=Path)
    persona_release.add_argument("--review-manifest", required=True, type=Path)
    persona_release.add_argument("--output", required=True, type=Path)

    evaluate = sub.add_parser(
        "evaluate-data",
        help="Run chronological holdout checks for observable forum behavior",
    )
    evaluate.add_argument("--posts", required=True, type=Path)
    evaluate.add_argument("--comments", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--train-start", default="2026-05-01")
    evaluate.add_argument("--split", default="2026-06-16")
    evaluate.add_argument("--end", default="2026-07-16")
    evaluate.add_argument("--samples-per-need", type=int, default=50000)
    evaluate.add_argument("--minimum-posts", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=20260721)
    evaluate.add_argument("--evidence-start", default="2025-12-01")
    evaluate.add_argument("--evidence-end", default="2026-07-21")
    evaluate.add_argument("--event-window-radius-days", type=int, default=3)
    evaluate.add_argument("--max-focus-events", type=int, default=5)
    evaluate.add_argument("--minimum-event-posts", type=int, default=40)
    evaluate.add_argument("--minimum-event-separation-days", type=int, default=14)
    evaluate.add_argument("--minimum-peak-posts", type=int, default=5)
    evaluate.add_argument("--minimum-peak-lift", type=float, default=1.25)

    evaluate_pilot_parser = sub.add_parser(
        "evaluate-pilot",
        help="Validate and aggregate a preregistered de-identified pilot",
    )
    evaluate_pilot_parser.add_argument("--manifest", required=True, type=Path)
    evaluate_pilot_parser.add_argument("--events", required=True, type=Path)
    evaluate_pilot_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps({"scenarios": sorted(BUNDLED_SCENARIOS), "policies": sorted(POLICIES)}, indent=2))
        return
    if args.command == "validate":
        names = sorted(BUNDLED_SCENARIOS) if args.scenario == "all" else [args.scenario]
        result = {}
        for name in names:
            config = load_scenario(resolve_scenario(name))
            result[config.scenario_id] = {
                "valid": True,
                "version": config.version,
                "fingerprint": config.fingerprint(),
                "source_counts": config.source_counts(),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "run":
        policies = None if args.policy == "all" else [args.policy]
        output = _run_scenario(
            args.scenario, args.repetitions, args.seed, args.output, policies
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    if args.command == "run-all":
        outputs = [
            _run_scenario(name, args.repetitions, args.seed, args.output)
            for name in sorted(BUNDLED_SCENARIOS)
        ]
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return
    if args.command == "robustness":
        names = sorted(BUNDLED_SCENARIOS) if args.scenario == "all" else [args.scenario]
        outputs = []
        for name in names:
            config = load_scenario(resolve_scenario(name))
            result = run_robustness_analysis(
                config,
                samples=args.samples,
                repetitions=args.repetitions,
                seed=args.seed,
            )
            json_path = write_robustness_json(
                result, args.output / f"{config.scenario_id}.json"
            )
            outputs.append(
                {
                    "scenario_id": config.scenario_id,
                    "robust_recommendation": result["robust_recommendation"],
                    "common_policy_inputs_verified": result["methodology"][
                        "common_policy_inputs_verified"
                    ],
                    "output": json_path,
                }
            )
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return
    if args.command == "calibrate":
        result = calibrate_csvs(
            args.posts, args.comments, args.output, start=args.start, end=args.end
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "valid_posts": result["valid_posts"],
                    "linked_comments": result["linked_comments"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "replay-shocks":
        try:
            result = run_shock_replay_pipeline(
                args.posts,
                args.comments,
                args.output,
                definitions_path=args.definitions,
                repetitions=args.repetitions,
                seed=args.seed,
            )
        except (ShockReplayError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        replay = result["replay"]
        print(
            json.dumps(
                {
                    "evidence_stage": replay["overall_fidelity"][
                        "evidence_stage"
                    ],
                    "weakest_reconstruction_fit_grade": replay[
                        "overall_fidelity"
                    ][
                        "weakest_reconstruction_fit_grade"
                    ],
                    "event_reconstruction_fit_grades": replay[
                        "overall_fidelity"
                    ]["event_reconstruction_fit_grades"],
                    "external_validation_status": replay["overall_fidelity"][
                        "external_validation_status"
                    ],
                    "causal_status": replay["overall_fidelity"][
                        "causal_status"
                    ],
                    "repetitions": replay["repetitions"],
                    "dataset_snapshot_id": replay["dataset_snapshot_id"],
                    "analysis_snapshot_id": replay["analysis_snapshot_id"],
                    "outputs": result["paths"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "run-llm-agents":
        policy_ids = [
            value.strip() for value in args.policies.split(",") if value.strip()
        ]
        try:
            result = run_live_llm_agent_pipeline(
                benchmark_path=args.benchmark,
                definitions_path=args.definitions,
                output_path=args.output,
                scenario_id=args.scenario,
                policy_ids=policy_ids,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                agent_count=args.agents,
                seed=args.seed,
                max_concurrency=args.max_concurrency,
                max_agent_turns=args.max_agent_turns,
                max_total_tokens=args.max_total_tokens,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
                persona_release_path=args.persona_release,
                persona_evidence_path=args.persona_evidence,
                decision_cache_path=args.decision_cache,
            )
        except (LLMAgentError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "experiment_mode": result["experiment_mode"],
                    "scenario_id": result["scenario_id"],
                    "policies": [
                        run["policy_id"] for run in result["policy_runs"]
                    ],
                    "agent_count": result["population"]["agent_count"],
                    "agent_turns": result["execution"]["required_agent_turns"],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "accounted_tokens": result["execution"]["token_budget"][
                        "provider_reported_or_reserved_tokens"
                    ],
                    "requested_model": result["provider"]["requested_model"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "variant-gate-panel":
        try:
            result = run_variant_gate_acceptance_pipeline(
                benchmark_path=args.benchmark,
                definitions_path=args.definitions,
                direct_review_path=args.direct_review,
                persona_release_path=args.persona_release,
                persona_evidence_path=args.persona_evidence,
                output_path=args.output,
                decision_cache_path=args.decision_cache,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                seed=args.seed,
                max_concurrency=args.max_concurrency,
                max_agent_turns=args.max_agent_turns,
                max_total_tokens=args.max_total_tokens,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
        except (LLMAgentError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "experiment_mode": result["experiment_mode"],
                    "scenario_ids": result["panel_design"]["scenario_ids"],
                    "variant_ids": result["panel_design"]["variant_ids"],
                    "agent_turns": result["execution"]["required_agent_turns"],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "provider_reported_tokens": result["execution"][
                        "token_budget"
                    ]["provider_reported_or_reserved_tokens"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "expression-pilot":
        try:
            result = run_forum_expression_pilot(
                posts_path=args.posts,
                comments_path=args.comments,
                simulation_artifact_paths=args.simulation_artifacts,
                persona_release_path=args.persona_release,
                output_path=args.output,
                observed_evidence_path=args.reuse_observed_evidence,
                definitions_path=args.definitions,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                per_scenario=args.per_scenario,
                seed=args.seed,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
                allow_provider_text=args.allow_provider_text,
            )
        except (ForumExpressionError, LLMAgentError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "schema_version": result["schema_version"],
                    "case_count": result["summary"]["case_count"],
                    "accepted_render_count": result["summary"]["accepted_render_count"],
                    "fallback_count": result["summary"]["fallback_count"],
                    "accounted_tokens": result["provider"]["usage"]["total_tokens"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "lecture-thread-pilot":
        try:
            result = run_lecture_thread_ecology_pilot(
                posts_path=args.posts,
                comments_path=args.comments,
                simulation_artifact_path=args.simulation_artifact,
                expression_artifact_path=args.expression_artifact,
                persona_release_path=args.persona_release,
                output_path=args.output,
                observed_evidence_path=args.reuse_observed_evidence,
                definitions_path=args.definitions,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                sample_per_split=args.sample_per_split,
                batch_size=args.batch_size,
                reply_count=args.reply_count,
                seed=args.seed,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
                allow_provider_text=args.allow_provider_text,
            )
        except (LectureThreadError, ForumExpressionError, LLMAgentError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "schema_version": result["schema_version"],
                    "reply_case_count": result["summary"]["reply_case_count"],
                    "thread_count": result["summary"]["thread_count"],
                    "accepted_count": result["summary"]["accepted_count"],
                    "fallback_count": result["summary"]["fallback_count"],
                    "accounted_tokens": result["provider"]["usage"]["total_tokens"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "semantic-pilot":
        try:
            result = run_semantic_card_pilot(
                posts_path=args.posts,
                definitions_path=args.definitions,
                output_path=args.output,
                sample_size=args.sample_size,
                seed=args.seed,
                start=args.start,
                end=args.end,
                platform_marker=args.platform_marker,
                batch_size=args.batch_size,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
                allow_provider_text=args.allow_provider_text,
            )
        except (SemanticPopulationError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "record_count": result["summary"]["record_count"],
                    "language_form_counts": result["summary"]["language_form_counts"],
                    "semantic_determinacy_counts": result["summary"][
                        "semantic_determinacy_counts"
                    ],
                    "context_needed_count": result["summary"]["context_needed_count"],
                    "mean_confidence": result["summary"]["mean_confidence"],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "accounted_tokens": result["execution"]["token_budget"][
                        "provider_reported_or_reserved_tokens"
                    ],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "semantic-context-pilot":
        try:
            result = run_context_refinement_pilot(
                posts_path=args.posts,
                comments_path=args.comments,
                semantic_result_path=args.semantic_result,
                definitions_path=args.definitions,
                output_path=args.output,
                batch_size=args.batch_size,
                max_comments_per_post=args.max_comments_per_post,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
                allow_provider_text=args.allow_provider_text,
            )
        except (SemanticPopulationError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "record_count": result["summary"]["record_count"],
                    "revision_status_counts": result["summary"][
                        "revision_status_counts"
                    ],
                    "mean_confidence_before": result["summary"][
                        "mean_confidence_before"
                    ],
                    "mean_confidence_after": result["summary"][
                        "mean_confidence_after"
                    ],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "accounted_tokens": result["execution"]["token_budget"][
                        "provider_reported_or_reserved_tokens"
                    ],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "taxonomy-pilot":
        try:
            result = run_taxonomy_pilot(
                semantic_result_path=args.semantic_result,
                context_result_path=args.context_result,
                frozen_taxonomy_path=args.frozen_taxonomy,
                output_path=args.output,
                target_archetypes=args.target_archetypes,
                minimum_archetypes=args.minimum_archetypes,
                maximum_archetypes=args.maximum_archetypes,
                batch_size=args.batch_size,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "accounted_tokens": result["execution"]["token_budget"][
                        "provider_reported_or_reserved_tokens"
                    ],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "taxonomy-hard-pairs":
        try:
            result = run_hard_pair_audit(
                semantic_result_path=args.semantic_result,
                context_result_path=args.context_result,
                taxonomy_assignment_paths=args.assignment_runs,
                output_path=args.output,
                selection_mode=args.selection_mode,
                batch_size=args.batch_size,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "accounted_tokens": result["execution"]["token_budget"][
                        "provider_reported_or_reserved_tokens"
                    ],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "taxonomy-release":
        try:
            result = release_reviewed_taxonomy(
                semantic_result_path=args.semantic_result,
                taxonomy_result_path=args.taxonomy_result,
                hard_pair_result_path=args.hard_pair_result,
                adjudication_manifest_path=args.adjudication_manifest,
                output_path=args.output,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "taxonomy-revise":
        try:
            result = revise_taxonomy_from_review(
                taxonomy_assignment_path=args.taxonomy_assignment,
                revision_manifest_path=args.revision_manifest,
                output_path=args.output,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "persona-evidence":
        try:
            result = build_persona_evidence_packs(
                posts_path=args.posts,
                semantic_result_path=args.semantic_result,
                context_result_path=args.context_result,
                taxonomy_release_path=args.taxonomy_release,
                definitions_path=args.definitions,
                representative_per_archetype=args.representative_per_archetype,
                contrastive_per_archetype=args.contrastive_per_archetype,
                output_path=args.output,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "persona-pilot":
        try:
            result = run_persona_profile_pilot(
                evidence_pack_path=args.evidence_pack,
                output_path=args.output,
                max_concurrency=args.max_concurrency,
                max_total_tokens=args.max_total_tokens,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "provider_calls": result["execution"]["token_budget"][
                        "provider_calls"
                    ],
                    "accounted_tokens": result["execution"]["token_budget"][
                        "provider_reported_or_reserved_tokens"
                    ],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "persona-release":
        try:
            result = release_reviewed_personas(
                persona_pilot_path=args.persona_pilot,
                evidence_pack_path=args.evidence_pack,
                review_manifest_path=args.review_manifest,
                output_path=args.output,
            )
        except (RepresentativeTaxonomyError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    **result["summary"],
                    "privacy_scan": result["privacy_scan"]["status"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "evaluate-data":
        configs = [load_scenario(path) for path in BUNDLED_SCENARIOS.values()]
        result = evaluate_forum_data(
            args.posts,
            args.comments,
            configs,
            train_start=args.train_start,
            split=args.split,
            end=args.end,
            simulated_samples_per_need=args.samples_per_need,
            seed=args.seed,
            minimum_posts=args.minimum_posts,
            evidence_start=args.evidence_start,
            evidence_end=args.evidence_end,
            event_window_radius_days=args.event_window_radius_days,
            max_focus_events=args.max_focus_events,
            minimum_event_posts=args.minimum_event_posts,
            minimum_event_separation_days=args.minimum_event_separation_days,
            minimum_peak_posts=args.minimum_peak_posts,
            minimum_peak_lift=args.minimum_peak_lift,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        json_path = write_validation_json(result, args.output / "validation.json")
        html_path = generate_validation_report(result, args.output / "validation.html")
        print(
            json.dumps(
                {
                    "temporal_stability": result["overall_conclusion"][
                        "temporal_stability"
                    ],
                    "observable_generator_holdout_fidelity": result["overall_conclusion"][
                        "observable_generator_holdout_fidelity"
                    ],
                    "intervention_effect_validity": result["overall_conclusion"][
                        "intervention_effect_validity"
                    ],
                    "focus_event_backtest": result["overall_conclusion"][
                        "focus_event_backtest"
                    ],
                    "validation_json": json_path,
                    "validation_html": html_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "evaluate-pilot":
        try:
            result = evaluate_pilot(args.manifest, args.events)
        except PilotDataError as exc:
            parser.error(str(exc))
        args.output.mkdir(parents=True, exist_ok=True)
        json_path = write_pilot_json(result, args.output / "pilot_evaluation.json")
        html_path = generate_pilot_report(
            result, args.output / "pilot_evaluation.html"
        )
        print(
            json.dumps(
                {
                    "readiness_status": result["readiness"]["status"],
                    "all_automated_gates_passed": result["readiness"][
                        "all_automated_gates_passed"
                    ],
                    "pilot_json": json_path,
                    "pilot_html": html_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
