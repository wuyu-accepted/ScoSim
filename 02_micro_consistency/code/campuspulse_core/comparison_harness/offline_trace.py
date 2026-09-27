from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from campuspulse_core.forum_pilot import (
    OPEN_CHOICE_MECHANISM_PROTOCOL,
    compile_pilot_profile_bridge,
    scenario_context_for_tick,
    select_pilot_anchors,
)
from campuspulse_core.forum_relationships import (
    compile_scenario_stake_release_v1,
    compile_synthetic_relationship_release_v1,
)
from campuspulse_core.mass_population import canonical_hash

from .contracts import ACTIONS, CommonObservation, ContextItem, canonical_json


SCENARIO_ID = "governance_legitimacy_dispute"
PILOT_SEED = 20260722
TICKS = tuple(range(10))

SOURCE_FILES = {
    "population": "hero-pilot-bridge-v5.json",
    "semantic_population": "semantic-llm-forum-agent-population-v2.json",
    "relationships_public": "synthetic-relationship-release-v1.public.json",
    "stakes_public": "scenario-stake-release-v1.public.json",
    "ecology": "forum-ecology-release-v1.json",
    "activation_plan": "slot-plan.json",
}

EXPECTED_FILE_SHA256 = {
    "population": "adb811eb6be990429ee33e6986bf80d8c76ec5f3d648469d90e547d41a20a443",
    "relationships_public": "df922cd4e8b1251d50a22989136f40b95a952e9ddb002303a92e6ffb1e0ac6d7",
    "stakes_public": "59bde53b78cea55e197382339c87e27f9d30625f9dc1b2b94c69f0705459a820",
    "ecology": "2a0d10c4633ae54ea10e3d17997244318b1d7b2bb5ee13777282cc1c47e66d9e",
    "activation_plan": "fb3a3ca2f8b466e835992ee56080f14a7f42224030d32658b0f27a255785f3ed",
}

DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "target_id", "text", "emotion", "public_stance", "public_or_private"
    ],
    "properties": {
        "action": {"enum": list(ACTIONS)},
        "target_id": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
        "emotion": {"type": ["string", "null"]},
        "public_stance": {"type": ["number", "null"], "minimum": -1.0, "maximum": 1.0},
        "public_or_private": {"enum": ["public", "private", "none"]},
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class ActivationSlot:
    tick: int
    agent_index: int
    agent_id: str
    activation_class: str


@dataclass(frozen=True)
class FrozenTrace:
    observations: tuple[tuple[ActivationSlot, CommonObservation], ...]
    activation_rows: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    @property
    def observation_trace_sha256(self) -> str:
        return canonical_hash([
            {
                "tick": slot.tick,
                "agent_index": slot.agent_index,
                "agent_id": slot.agent_id,
                "activation_class": slot.activation_class,
                "observation_sha256": observation.observation_sha256,
            }
            for slot, observation in self.observations
        ])


class FrozenCommonTraceConnector:
    """Build a method-independent T0-T9 observation trace without model calls.

    Interaction-reserve slots are assigned ex ante from frozen inputs. They are
    action opportunities, not invented replies. Endogenous posts are therefore
    absent from this pre-run trace and cannot leak one method's output to another.
    """

    def __init__(self, scenario_root: Path):
        self.scenario_root = scenario_root
        self.file_hashes: dict[str, str] = {}
        self.population: dict[str, Any] = {}
        self.semantic_population: dict[str, Any] = {}
        self.relationships_public: dict[str, Any] = {}
        self.stakes_public: dict[str, Any] = {}
        self.ecology: dict[str, Any] = {}
        self.activation_plan: dict[str, Any] = {}
        self.profiles: list[dict[str, Any]] = []
        self.stake_by_agent: dict[str, Mapping[str, Any]] = {}
        self.relationships_by_agent: dict[str, tuple[Mapping[str, Any], ...]] = {}

    def load_and_verify(self) -> None:
        for key, filename in SOURCE_FILES.items():
            path = self.scenario_root / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            self.file_hashes[key] = _file_sha256(path)
            setattr(self, key, _read_json(path))
        for key, expected in EXPECTED_FILE_SHA256.items():
            if self.file_hashes[key] != expected:
                raise ValueError(f"frozen source hash mismatch: {key}")

        bridge = compile_pilot_profile_bridge(self.population)
        self.profiles = list(bridge["profiles"])
        if len(self.profiles) != 1000:
            raise ValueError("frozen population must contain 1,000 profiles")

        stakes = compile_scenario_stake_release_v1(
            population=self.semantic_population,
            scenario_ids=self.stakes_public["scenario_ids"],
            seed=int(self.stakes_public["seed"]),
        )
        if stakes["stake_release_sha256"] != self.stakes_public["stake_release_sha256"]:
            raise ValueError("record-level stake reconstruction failed public hash parity")
        relationships = compile_synthetic_relationship_release_v1(
            population=self.semantic_population,
            stake_release=stakes,
            seed=int(self.relationships_public["seed"]),
        )
        if relationships["relationship_release_sha256"] != self.relationships_public["relationship_release_sha256"]:
            raise ValueError("record-level relationship reconstruction failed public hash parity")
        self.stake_by_agent = {
            str(row["agent_id"]): row
            for row in stakes["positions"]
            if row["scenario_id"] == SCENARIO_ID
        }
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for edge in relationships["edges"]:
            grouped.setdefault(str(edge["source_agent_id"]), []).append(edge)
        self.relationships_by_agent = {
            agent_id: tuple(sorted(rows, key=lambda row: str(row["relationship_id"])))
            for agent_id, rows in grouped.items()
        }
        if len(self.stake_by_agent) != 1000 or len(self.relationships_by_agent) != 1000:
            raise ValueError("record-level reconstruction is incomplete")

    def _slot_row(self, tick: int) -> Mapping[str, Any]:
        branch = "shared_baseline" if tick <= 2 else "natural"
        matches = [
            row for row in self.activation_plan["rounds"]
            if row["scenario_id"] == SCENARIO_ID
            and int(row["seed"]) == PILOT_SEED
            and row["slot_kind"] == "resident"
            and int(row["tick"]) == tick
            and str(row["branch"]).lower() == branch
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one frozen resident slot row for tick {tick}")
        return matches[0]

    @staticmethod
    def _unit(*parts: Any) -> float:
        return (int(canonical_hash(parts)[:16], 16) + 1) / (16**16 + 1)

    def _weighted_select(
        self,
        candidates: Iterable[int],
        count: int,
        *,
        tick: int,
        label: str,
        rotation: Mapping[int, int],
    ) -> tuple[int, ...]:
        ranked = []
        for index in candidates:
            profile = self.profiles[index]
            state = profile.get("initial_state") or {}
            weight = float((profile.get("scenario_weights") or {}).get(SCENARIO_ID, 0.0))
            activity = float(state.get("activity_probability", 0.2))
            size = max(1e-12, weight * (0.25 + activity) / (1 + int(rotation.get(index, 0))) ** 2)
            race = -math.log(self._unit("paired-trace-v1", PILOT_SEED, tick, label, index)) / size
            ranked.append((race, index))
        if len(ranked) < count:
            raise ValueError(f"not enough candidates for {label} at tick {tick}")
        return tuple(index for _score, index in sorted(ranked)[:count])

    def _activation_trace(self) -> tuple[tuple[ActivationSlot, ...], tuple[Mapping[str, Any], ...]]:
        anchors = select_pilot_anchors(
            self.profiles, scenario_id=SCENARIO_ID, seed=PILOT_SEED
        )
        anchor_set = set(anchors)
        rotation: dict[int, int] = {}
        rounds: list[tuple[ActivationSlot, ...]] = []
        rows: list[Mapping[str, Any]] = []
        eligible = [
            index for index, profile in enumerate(self.profiles)
            if index not in anchor_set and profile.get("simulation_gate") != "background_signal"
        ]
        for tick in TICKS:
            plan = self._slot_row(tick)
            interactions = self._weighted_select(
                eligible, int(plan["interactions"]), tick=tick,
                label="interaction_reserve", rotation=rotation,
            )
            interaction_set = set(interactions)
            pps = self._weighted_select(
                (index for index in eligible if index not in interaction_set),
                int(plan["pps"]), tick=tick, label="pps", rotation=rotation,
            )
            for index in (*interactions, *pps):
                rotation[index] = rotation.get(index, 0) + 1
            slots = tuple(
                ActivationSlot(tick, index, str(self.profiles[index]["agent_id"]), kind)
                for kind, indices in (("anchor", anchors), ("interaction_reserve", interactions), ("pps", pps))
                for index in indices
            )
            if len(slots) != int(plan["budget"]) or len({slot.agent_index for slot in slots}) != len(slots):
                raise ValueError(f"activation budget or uniqueness drift at tick {tick}")
            rounds.append(slots)
            rows.append({
                "tick": tick,
                "branch": plan["branch"],
                "budget": plan["budget"],
                "anchor_indices": list(anchors),
                "interaction_reserve_indices": list(interactions),
                "pps_indices": list(pps),
                "round_sha256": canonical_hash([slot.__dict__ for slot in slots]),
            })
        return tuple(rounds), tuple(rows)

    def _stake_context(self, agent_id: str, tick: int) -> ContextItem | None:
        if tick < 3:
            return None
        row = self.stake_by_agent[agent_id]
        private_values = {
            key: value["value"] for key, value in sorted(row["fields"].items())
        }
        return ContextItem(
            message_id=f"private-scenario-position:{agent_id}",
            author_id=None,
            text=canonical_json({
                "label": "synthetic private scenario conditions known to this resident",
                "values": private_values,
            }),
            visibility="private",
            tick=3,
        )

    def _observation(self, slot: ActivationSlot) -> CommonObservation:
        event = scenario_context_for_tick(
            slot.tick, scenario_id=SCENARIO_ID,
            mechanism_protocol=OPEN_CHOICE_MECHANISM_PROTOCOL,
        )
        context_items: list[ContextItem] = []
        if slot.tick >= 3:
            context_items.append(ContextItem(
                message_id=f"system-event:{SCENARIO_ID}:t{slot.tick}",
                author_id=None,
                text=str(event["known_at_tick"]),
                visibility="system",
                tick=slot.tick,
            ))
            private = self._stake_context(slot.agent_id, slot.tick)
            if private is not None:
                context_items.append(private)
        facts: dict[str, Any] = {
            "synthetic_scenario": True,
            "evidence_boundary": event["evidence_boundary"],
        }
        if slot.tick >= 3:
            facts.update({
                "initial_beds": 520,
                "available_beds": 320,
                "valid_applications": 536,
                "confirmed_facts": event.get("confirmed_facts"),
                "unverified_claims": event.get("unverified_claims"),
            })
        return CommonObservation(
            scenario_id=SCENARIO_ID,
            tick=slot.tick,
            agent_id=slot.agent_id,
            factual_registry=facts,
            current_event=event,
            visible_context=tuple(context_items),
            governance_artifacts=(),
            allowed_actions=ACTIONS,
            response_schema=DECISION_SCHEMA,
        )

    def build(self) -> FrozenTrace:
        if not self.profiles:
            self.load_and_verify()
        rounds, activation_rows = self._activation_trace()
        observations = tuple(
            (slot, self._observation(slot)) for round_slots in rounds for slot in round_slots
        )
        metadata = {
            "trace_schema_version": "aamas2027-frozen-common-observation-trace-v1",
            "scenario_id": SCENARIO_ID,
            "seed": PILOT_SEED,
            "ticks": list(TICKS),
            "selection_mode": "ex_ante_method_independent_activation",
            "interaction_reserve_semantics": "action_opportunity_only_no_invented_reply_or_message",
            "model_calls": 0,
            "observation_count": len(observations),
            "source_file_sha256": dict(sorted(self.file_hashes.items())),
            "stake_release_sha256": self.stakes_public["stake_release_sha256"],
            "relationship_release_sha256": self.relationships_public["relationship_release_sha256"],
            "record_level_relationships_persisted": False,
        }
        return FrozenTrace(observations, activation_rows, metadata)


def write_frozen_trace(trace: FrozenTrace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.jsonl"
    with observations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for slot, observation in trace.observations:
            handle.write(canonical_json({
                "observation_id": f"{SCENARIO_ID}:{PILOT_SEED}:t{slot.tick}:{slot.agent_id}",
                "activation_class": slot.activation_class,
                "agent_index": slot.agent_index,
                "observation_sha256": observation.observation_sha256,
                "observation": observation.public_payload(),
            }) + "\n")
    activation_path = output_dir / "activation_trace.json"
    activation_path.write_text(
        json.dumps(list(trace.activation_rows), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    methods = (
        "campuspulse", "belief_engine", "generative_agents", "chuang_opinion_dynamics"
    )
    bindings_path = output_dir / "method_bindings.jsonl"
    with bindings_path.open("w", encoding="utf-8", newline="\n") as handle:
        for slot, observation in trace.observations:
            observation_id = f"{SCENARIO_ID}:{PILOT_SEED}:t{slot.tick}:{slot.agent_id}"
            for method in methods:
                handle.write(canonical_json({
                    "method": method,
                    "observation_id": observation_id,
                    "observation_sha256": observation.observation_sha256,
                }) + "\n")
    report = {
        **trace.metadata,
        "observation_trace_sha256": trace.observation_trace_sha256,
        "observations_file_sha256": _file_sha256(observations_path),
        "activation_file_sha256": _file_sha256(activation_path),
        "method_bindings_file_sha256": _file_sha256(bindings_path),
        "method_binding_count": len(trace.observations) * len(methods),
        "paired_methods": list(methods),
        "pairing_assertion": "each method consumes the same observation_id and observation_sha256",
    }
    report_path = output_dir / "trace_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
