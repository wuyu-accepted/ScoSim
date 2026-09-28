"""Deterministic synthetic demand generation with common intrinsic draws."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping

from .config import ScenarioConfig
from .entities import DemandAgent


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    draw = rng.random()
    cumulative = 0.0
    ordered = sorted(weights.items())
    for key, weight in ordered:
        cumulative += weight
        if draw <= cumulative:
            return key
    return ordered[-1][0]


def _poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0:
        return 0
    if mean < 30:
        threshold = math.exp(-mean)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= rng.random()
        return count - 1
    # Stable approximation is sufficient for operational stress tests at high rates.
    return max(0, int(round(rng.gauss(mean, math.sqrt(mean)))))


def _gamma_poisson(rng: random.Random, mean: float, dispersion: float) -> int:
    gamma_mean = rng.gammavariate(dispersion, mean / dispersion)
    return _poisson(rng, gamma_mean)


def sample_natural_reply_count(
    rng: random.Random,
    response_probability: float,
    mean_if_responded: float,
    dispersion: float,
) -> tuple[float, int]:
    """Sample the exact hurdle/count mechanism used by the production generator."""

    response_draw = rng.random()
    if response_draw < response_probability:
        return response_draw, max(1, _gamma_poisson(rng, mean_if_responded, dispersion))
    return response_draw, 0


def generate_demand_stream(config: ScenarioConfig, seed: int) -> list[DemandAgent]:
    """Generate one immutable intrinsic stream to be deep-copied across policies."""

    rng = random.Random(seed)
    rounds = config.get("rounds")
    base_rate = float(config.get("base_arrival_rate"))
    shock_curve = config.get("shock_curve")
    need_mix = config.get("need_mix")
    urgency_mix = config.get("urgency_mix")
    response_probability = config.get("natural_response_probability")
    reply_mean = config.get("mean_replies_if_responded")
    dispersion = float(config.get("reply_dispersion"))
    routing = config.get("routing")
    work_units = config.get("work_units_by_urgency")

    demands: list[DemandAgent] = []
    serial = 0
    for current_round in range(rounds):
        multiplier = float(shock_curve[current_round % len(shock_curve)])
        arrivals = _poisson(rng, base_rate * multiplier)
        for _ in range(arrivals):
            serial += 1
            need_type = _weighted_choice(rng, need_mix)
            urgency = _weighted_choice(rng, urgency_mix)
            natural_draw, replies = sample_natural_reply_count(
                rng,
                response_probability[need_type],
                reply_mean[need_type],
                dispersion,
            )
            demands.append(
                DemandAgent(
                    demand_id=f"{config.scenario_id}-{seed}-{serial:06d}",
                    arrival_round=current_round,
                    need_type=need_type,
                    urgency=urgency,
                    department=routing[need_type],
                    work_units=int(work_units[urgency]),
                    natural_response_draw=natural_draw,
                    natural_reply_count=replies,
                    natural_resolution_draw=rng.random(),
                    faq_draw=rng.random(),
                    peer_draw=rng.random(),
                )
            )
    return demands


def demand_fingerprint(demands: list[DemandAgent]) -> str:
    digest = hashlib.sha256()
    for demand in demands:
        row = (
            demand.demand_id,
            demand.arrival_round,
            demand.need_type,
            demand.urgency,
            demand.department,
            demand.work_units,
            round(demand.natural_response_draw, 15),
            demand.natural_reply_count,
            round(demand.natural_resolution_draw, 15),
            round(demand.faq_draw, 15),
            round(demand.peer_draw, 15),
        )
        digest.update(repr(row).encode("utf-8"))
    return digest.hexdigest()
