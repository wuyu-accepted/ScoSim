from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew

from .common import STATE_FIELDS, group_records, json_safe, read_jsonl, state_vector, write_json


VARIANTS = ("full", "single_particle", "no_profile", "no_profile_no_smc")
METHODS = ("campuspulse", "belief_engine", "generative_agents", "chuang_opinion_dynamics")
PRIMARY_TICKS = (0, 3, 9)
ENTROPY_EDGES = np.asarray([
    -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
], dtype=np.float64)
DEFINITIONS_VERSION = "1.6.0"


def effective_rank(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    total = float(singular_values.sum())
    if total <= 0:
        return 0.0
    probabilities = singular_values[singular_values > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def categorical_r_squared(matrix: np.ndarray, groups: np.ndarray) -> float | None:
    matrix, groups = np.asarray(matrix, dtype=np.float64), np.asarray(groups, dtype=object)
    overall = matrix.mean(axis=0)
    total_ss = float(np.sum((matrix - overall) ** 2))
    if total_ss <= 0:
        return None
    between = sum(len(matrix[groups == label]) * float(np.sum((matrix[groups == label].mean(axis=0) - overall) ** 2)) for label in sorted(set(groups)))
    return between / total_ss


def permutation_p(matrix: np.ndarray, groups: np.ndarray, observed: float | None, count: int, seed: int) -> float | None:
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(count):
        value = categorical_r_squared(matrix, rng.permutation(groups))
        exceed += value is not None and value >= observed
    return float((1 + exceed) / (count + 1))


def sarle_bimodality(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 4 or float(np.var(values)) < 1e-10:
        return None
    # Ordinary central-moment estimator. Separately bias-correcting skewness
    # and kurtosis can make their ratio slightly exceed the theoretical upper
    # bound of one.
    pearson = float(kurtosis(values, fisher=False, bias=True))
    return None if pearson <= 0 else float((float(skew(values, bias=True)) ** 2 + 1.0) / pearson)


def kde_peak_count(values: np.ndarray) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3 or float(np.var(values)) < 1e-10:
        return None
    density = gaussian_kde(values, bw_method="scott")(np.linspace(-1.0, 1.0, 1024))
    return int(len(find_peaks(density, prominence=0.05 * float(density.max()))[0]))


def discrete_entropy(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        return None
    if np.any((values < -1.0) | (values > 1.0)):
        raise ValueError("net stance lies outside [-1,1]")
    if float(np.var(values)) < 1e-10:
        return None
    counts, _ = np.histogram(values, bins=ENTROPY_EDGES)
    probabilities = counts[counts > 0] / counts.sum()
    raw = float(-np.sum(probabilities * np.log(probabilities)))
    return {"raw": raw, "normalized": raw / float(np.log(20.0))}


def univariate_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()) if len(values) else None, "sd_ddof_0": float(values.std(ddof=0)) if len(values) else None, "sarle_bimodality": sarle_bimodality(values), "kde_peak_count": kde_peak_count(values)}


def _group_label(value: Any) -> str:
    normalized = str(value or "")
    if normalized in {"directly_affected", "service_dependent", "resource_need_intense"}:
        return "directly_affected"
    if normalized in {"bystander", "bystander_or_mixed", "alternatives_available"}:
        return "bystander"
    raise ValueError(f"unregistered interest group {value!r}")


def _validate_population(rows: list[dict[str, Any]], expected: int) -> None:
    agents = [str(row["agent_id"]) for row in rows]
    if len(rows) != expected or len(set(agents)) != expected:
        raise ValueError(f"expected exactly {expected} unique parent agents, got {len(rows)} rows/{len(set(agents))} agents")


def evaluate_cross_method_group(rows: list[dict[str, Any]], count: int, seed: int, expected: int) -> dict[str, Any]:
    _validate_population(rows, expected)
    if any(row.get("public_stance") is None for row in rows):
        raise ValueError("cross-method dimension 3 requires one net stance per agent")
    stance = np.asarray([float(row["public_stance"]) for row in rows], dtype=np.float64)
    groups = np.asarray([_group_label(row.get("interest_group")) for row in rows], dtype=object)
    # A ratio of two floating-point residual sums is not scientifically
    # meaningful when the population stance is effectively constant.  Use the
    # same degeneracy boundary as BC and KDE and suppress its permutation test.
    r2 = None if float(np.var(stance)) < 1e-10 else categorical_r_squared(stance[:, None], groups)
    return {"agent_count": len(rows), "net_stance": univariate_summary(stance), "sarle_threshold": 0.555, "discrete_stance_entropy": discrete_entropy(stance), "stake_group_r_squared_1d": r2, "stake_group_permutation_p": permutation_p(stance[:, None], groups, r2, count, seed)}


def evaluate_ablation_group(rows: list[dict[str, Any]], count: int, seed: int, expected: int, allow_missing_groups: bool = False) -> dict[str, Any]:
    _validate_population(rows, expected)
    matrix = np.stack([state_vector(row) for row in rows])
    if allow_missing_groups and any(row.get("interest_group") is None for row in rows):
        groups = None
    else:
        groups = np.asarray([_group_label(row.get("interest_group")) for row in rows], dtype=object)
    stance = matrix[:, STATE_FIELDS.index("support")] - matrix[:, STATE_FIELDS.index("opposition")]
    r2 = categorical_r_squared(matrix, groups) if groups is not None else None
    return {"agent_count": len(rows), "state_matrix_effective_rank": effective_rank(matrix), "stake_group_multivariate_r_squared_12d": r2, "stake_group_permutation_p": permutation_p(matrix, groups, r2, count, seed) if groups is not None else None, "stake_group_status": "ok" if groups is not None else "NA_anonymous_panel_has_no_group_join_key", "state_dimensions": {field: univariate_summary(matrix[:, index]) for index, field in enumerate(STATE_FIELDS)}, "net_stance": {**univariate_summary(stance), "entropy": discrete_entropy(stance)}}


def _require_grid(grouped: dict[tuple[Any, ...], Any], expected_prefixes: set[tuple[Any, ...]], ticks: tuple[int, ...]) -> None:
    observed = set(grouped)
    expected = {(*prefix, tick) for prefix in expected_prefixes for tick in ticks}
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        raise ValueError(f"dimension-3 group grid mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def evaluate_cross_method(rows: list[dict[str, Any]], permutation_count: int = 199, expected_agent_count: int = 1000, ticks: tuple[int, ...] = PRIMARY_TICKS, expected_methods: tuple[str, ...] = METHODS) -> dict[str, Any]:
    selected = [row for row in rows if int(row["tick"]) in ticks]
    grouped = group_records(selected, ("method", "variant", "seed", "tick"))
    seeds = {int(row["seed"]) for row in selected}
    _require_grid(grouped, {(method, "full", seed) for method in expected_methods for seed in seeds}, ticks)
    results = []
    for (method, variant, seed, tick), group in sorted(grouped.items()):
        results.append({"method": method, "variant": variant, "seed": seed, "tick": tick, **evaluate_cross_method_group(group, permutation_count, 20260722 + int(tick), expected_agent_count)})
    fake_markers = {str(row["fake_marker"]) for row in selected if row.get("fake_marker")}
    provenance = {}
    if fake_markers:
        if len(fake_markers) != 1 or any(not row.get("fake_marker") for row in selected):
            raise ValueError("dimension-3 input cannot mix fake and real records")
        provenance = {
            "experiment_status": "conformance_only_not_a_paper_result",
            "fake_marker": next(iter(fake_markers)),
        }
    return json_safe({"schema_version": "dimension-3-cross-method-v1.6.0", "definitions_version": DEFINITIONS_VERSION, "permutation_count": permutation_count, "definitions": {"analysis_unit": "one equally weighted parent agent", "stance_vector_shape": [expected_agent_count], "sd_ddof": 0, "sarle": "(ordinary central-moment skewness^2 + 1) / ordinary central-moment Pearson kurtosis; NA when variance < 1e-10", "sarle_threshold": 0.555, "kde": "scipy gaussian_kde Scott bandwidth over 1024 points in [-1,1]; peaks require prominence >= 5% of maximum density", "stake_group_r_squared": "one-dimensional between-group sum of squares / total sum of squares on net stance; exactly two frozen labels", "permutation_p": "one-sided add-one empirical p=(1+count(permuted_R2>=observed_R2))/(199+1)", "permutation_seed": "20260722 + tick", "entropy_edges": ENTROPY_EDGES.tolist(), "entropy_bin_count": 20, "entropy_degenerate_variance_threshold": 1e-10, "entropy_log_base": "natural", "entropy_normalization": "divide by ln(20)"}, **provenance, "groups": results})


def evaluate_ablation(rows: list[dict[str, Any]], permutation_count: int = 199, expected_agent_count: int = 1000, ticks: tuple[int, ...] = PRIMARY_TICKS, expected_variants: tuple[str, ...] = VARIANTS, allow_missing_groups: bool = False) -> dict[str, Any]:
    selected = [row for row in rows if row.get("method") == "campuspulse" and row.get("variant") in expected_variants and int(row["tick"]) in ticks]
    grouped = group_records(selected, ("variant", "seed", "tick"))
    seeds = {int(row["seed"]) for row in selected}
    _require_grid(grouped, {(variant, seed) for variant in expected_variants for seed in seeds}, ticks)
    results = []
    for (variant, seed, tick), group in sorted(grouped.items()):
        results.append({"method": "campuspulse", "variant": variant, "seed": seed, "tick": tick, **evaluate_ablation_group(group, permutation_count, 20260722 + 1000 + int(tick), expected_agent_count, allow_missing_groups)})
    return json_safe({"schema_version": "dimension-3-ablation-v1.4.0", "definitions_version": "1.4.0", "permutation_count": permutation_count, "definitions": {"effective_rank": "entropy of normalized singular values of centered 1000x12 parent-state matrix", "sarle": "ordinary central-moment skewness and Pearson kurtosis", "kde_prominence": "5% of maximum density", "entropy_edges": ENTROPY_EDGES.tolist(), "sd_ddof": 0}, "groups": results})


def evaluate(rows: list[dict[str, Any]], permutation_count: int = 199, expected_agent_count: int = 1000, mode: str = "ablation", allow_missing_groups: bool = False) -> dict[str, Any]:
    return evaluate_ablation(rows, permutation_count, expected_agent_count, allow_missing_groups=allow_missing_groups) if mode == "ablation" else evaluate_cross_method(rows, permutation_count, expected_agent_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen dimension-3 definitions")
    parser.add_argument("records", type=Path)
    parser.add_argument("--mode", choices=("cross_method", "ablation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--expected-agent-count", type=int, default=1000, help="formal default is frozen at 1000; smaller values are for labelled plumbing tests only")
    parser.add_argument("--expected-methods", default=",".join(METHODS), help="cross-method completeness gate")
    args = parser.parse_args()
    rows = read_jsonl(args.records)
    if args.mode == "cross_method":
        result = evaluate_cross_method(rows, args.permutations, args.expected_agent_count, expected_methods=tuple(args.expected_methods.split(",")))
    else:
        result = evaluate_ablation(rows, args.permutations, args.expected_agent_count)
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
