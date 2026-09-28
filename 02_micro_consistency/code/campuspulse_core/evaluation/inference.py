from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260722


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-FDR adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0,1]")
    if len(values) == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def cluster_bootstrap_interval(
    values: Sequence[float], clusters: Sequence[Hashable],
    statistic: Callable[[np.ndarray], float] = lambda array: float(np.mean(array)),
    *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> list[float] | None:
    if len(values) != len(clusters) or len(values) == 0:
        return None
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(float(value))
    labels = sorted(grouped, key=str)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.choice(len(labels), size=len(labels), replace=True)
        sample = np.asarray([value for ordinal in sampled for value in grouped[labels[int(ordinal)]]], dtype=np.float64)
        estimates[index] = statistic(sample)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def paired_agent_bootstrap_difference(
    left: dict[Hashable, float], right: dict[Hashable, float],
    *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | list[float]]:
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap requires identical nonempty agent keys")
    keys = sorted(left, key=str)
    differences = np.asarray([float(left[key]) - float(right[key]) for key in keys])
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, len(keys), size=len(keys))
        estimates[index] = float(differences[sample].mean())
    return {
        "left_minus_right": float(differences.mean()),
        "bootstrap_95pct_interval": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "resamples": resamples,
        "seed": seed,
    }


def cohens_kappa(left: Iterable[str], right: Iterable[str]) -> float | None:
    first, second = list(left), list(right)
    if len(first) != len(second) or not first:
        return None
    labels = sorted(set(first) | set(second))
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    expected = sum((first.count(label) / len(first)) * (second.count(label) / len(second)) for label in labels)
    return None if expected >= 1.0 else float((observed - expected) / (1.0 - expected))
