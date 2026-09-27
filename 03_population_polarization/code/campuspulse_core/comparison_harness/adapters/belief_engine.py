from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..contracts import AgentSeed, CommonObservation, MethodAdapter, MethodSnapshot


Vector = Sequence[float]
Embedder = Callable[[str], Vector]


@dataclass(frozen=True)
class ArgumentCandidate:
    claim: str
    polarity: int
    strength: float
    source_role: str

    def __post_init__(self) -> None:
        if self.polarity not in {-1, 1}:
            raise ValueError("polarity must be -1 or +1")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be in [0,1]")
        if self.source_role not in {"seed", "self", "opponent"}:
            raise ValueError("invalid source role")


@dataclass
class ArgumentRecord:
    record_id: str
    claim: str
    polarity: int
    strength: float
    source_role: str
    active: bool
    tick: int
    source_message_id: str | None
    embedding: tuple[float, ...]


Extractor = Callable[[str, str], Iterable[ArgumentCandidate]]


def _cosine(left: Vector, right: Vector) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class BeliefEngineAdapter(MethodAdapter):
    """Equation-level reimplementation of Yang et al. (2026), Eq. 1.

    Author-code parity is still a pre-formal-run gate. This adapter deliberately
    exposes argument extraction and embedding as injected functions so the same
    frozen auxiliary model can be used for every paired world.
    """

    method_id = "belief_engine"

    def __init__(
        self,
        *,
        seed_arguments: Sequence[ArgumentCandidate],
        extractor: Extractor,
        embedder: Embedder,
        evidence_uptake: float = 0.2,
        prior_anchoring: float = 0.4,
        similarity_threshold: float = 0.60,
        retrieval_count: int = 5,
    ):
        if not 0 <= evidence_uptake <= 1 or not 0 <= prior_anchoring <= 1:
            raise ValueError("uptake and anchoring must be in [0,1]")
        self.seed_arguments = tuple(seed_arguments)
        self.extractor = extractor
        self.embedder = embedder
        self.u = evidence_uptake
        self.a = prior_anchoring
        self.theta = similarity_threshold
        self.retrieval_count = retrieval_count
        self.seed: AgentSeed | None = None
        self.records: list[ArgumentRecord] = []
        self.processed_messages: set[str] = set()
        self.auxiliary_log: list[Mapping[str, Any]] = []
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    def initialize(self, seed: AgentSeed) -> None:
        self.seed = seed
        self.records.clear()
        self.processed_messages.clear()
        self.auxiliary_log.clear()
        self._embedding_cache.clear()
        for candidate in self.seed_arguments:
            if candidate.source_role != "seed":
                raise ValueError("initial Belief Engine arguments must have source_role=seed")
            self._admit(candidate, tick=0, source_message_id=None)

    def _admit(self, candidate: ArgumentCandidate, tick: int, source_message_id: str | None) -> None:
        embedding = self._embedding_cache.get(candidate.claim)
        if embedding is None:
            embedding = tuple(float(value) for value in self.embedder(candidate.claim))
            self._embedding_cache[candidate.claim] = embedding
        active_same = [row for row in self.records if row.active and row.polarity == candidate.polarity]
        nearest = max(active_same, key=lambda row: _cosine(embedding, row.embedding), default=None)
        active = True
        if nearest is not None and _cosine(embedding, nearest.embedding) >= self.theta:
            if candidate.strength > nearest.strength:
                nearest.active = False
            else:
                active = False
        self.records.append(ArgumentRecord(
            record_id=f"argument-{len(self.records):06d}", claim=candidate.claim,
            polarity=candidate.polarity, strength=candidate.strength,
            source_role=candidate.source_role, active=active, tick=tick,
            source_message_id=source_message_id, embedding=embedding,
        ))

    def ingest_network_arguments(
        self, candidates: Iterable[ArgumentCandidate], *, tick: int,
        source_message_id: str,
    ) -> None:
        """Admit already structured arguments delivered by the frozen network.

        Network peers are external evidence from the receiver's perspective.
        Keeping this boundary structured avoids re-parsing a claim that was
        already extracted by the sender's native Belief Engine memory.
        """
        if self.seed is None:
            raise RuntimeError("adapter is not initialized")
        if source_message_id in self.processed_messages:
            return
        for candidate in candidates:
            self._admit(
                ArgumentCandidate(
                    claim=candidate.claim, polarity=candidate.polarity,
                    strength=candidate.strength, source_role="opponent",
                ),
                tick=tick, source_message_id=source_message_id,
            )
        self.processed_messages.add(source_message_id)

    def export_network_arguments(self, limit: int = 2) -> tuple[ArgumentCandidate, ...]:
        """Export strongest active arguments, balanced across polarities."""
        if limit <= 0:
            return ()
        active = [row for row in self.records if row.active]
        positive = sorted((row for row in active if row.polarity == 1), key=lambda row: (-row.strength, row.record_id))
        negative = sorted((row for row in active if row.polarity == -1), key=lambda row: (-row.strength, row.record_id))
        chosen: list[ArgumentRecord] = []
        while len(chosen) < limit and (positive or negative):
            for bucket in (positive, negative):
                if bucket and len(chosen) < limit:
                    chosen.append(bucket.pop(0))
        return tuple(
            ArgumentCandidate(row.claim, row.polarity, row.strength, "opponent")
            for row in chosen
        )

    @property
    def log_odds(self) -> float:
        total = 0.0
        for row in self.records:
            if not row.active:
                continue
            gamma = self.a if row.source_role == "seed" else self.u
            total += row.polarity * math.log1p(row.strength * gamma)
        return total

    @property
    def stance(self) -> float:
        # 2*sigmoid(L)-1, written with tanh for numerical stability.
        return math.tanh(self.log_odds / 2.0)

    def _retrieve(self) -> list[dict[str, Any]]:
        active = [row for row in self.records if row.active]
        positive = sorted((row for row in active if row.polarity == 1), key=lambda row: -row.strength)
        negative = sorted((row for row in active if row.polarity == -1), key=lambda row: -row.strength)
        if not active:
            positive_slots = self.retrieval_count // 2
        else:
            positive_slots = round(self.retrieval_count * len(positive) / len(active))
        negative_slots = self.retrieval_count - positive_slots
        chosen = positive[:positive_slots] + negative[:negative_slots]
        return [
            {"record_id": row.record_id, "claim": row.claim, "polarity": row.polarity,
             "strength": row.strength, "source_role": row.source_role}
            for row in chosen
        ]

    def observe(self, observation: CommonObservation) -> MethodSnapshot:
        if self.seed is None:
            raise RuntimeError("adapter is not initialized")
        for item in observation.visible_context:
            if item.message_id in self.processed_messages:
                continue
            candidates = tuple(self.extractor(item.text, "opponent"))
            self.auxiliary_log.append({
                "type": "argument_extraction", "tick": observation.tick,
                "source_message_id": item.message_id, "candidate_count": len(candidates),
            })
            for candidate in candidates:
                self._admit(candidate, observation.tick, item.message_id)
            self.processed_messages.add(item.message_id)
        return MethodSnapshot(
            self.method_id, observation.agent_id, observation.tick,
            {
                "log_odds": self.log_odds, "stance": self.stance,
                "active_argument_count": sum(row.active for row in self.records),
                "archived_argument_count": sum(not row.active for row in self.records),
            },
            {
                "maintained_stance": self.stance,
                "retrieved_active_evidence": self._retrieve(),
                "update_profile": {"evidence_uptake": self.u, "prior_anchoring": self.a},
            },
            observation.observation_sha256,
            tuple(self.auxiliary_log),
        )

    def commit_decision(self, observation: CommonObservation, decision: Mapping[str, Any]) -> None:
        text = str(decision.get("text") or "")
        if not text:
            return
        for candidate in self.extractor(text, "self"):
            self._admit(candidate, observation.tick, None)

    def export_probe(self) -> float | None:
        return self.stance
