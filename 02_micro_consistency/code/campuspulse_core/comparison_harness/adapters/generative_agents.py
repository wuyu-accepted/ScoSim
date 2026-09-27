from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import json

from ..contracts import AgentSeed, CommonObservation, MethodAdapter, MethodSnapshot


Vector = Sequence[float]
Embedder = Callable[[str], Vector]
ImportanceScorer = Callable[[str], float]
Reflector = Callable[[tuple[str, ...]], tuple[str, ...]]


@dataclass
class MemoryNode:
    node_id: str
    text: str
    created_tick: int
    last_accessed_tick: int
    importance: float
    embedding: tuple[float, ...]
    kind: str = "event"


def _cosine(left: Vector, right: Vector) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    return dot / (ln * rn) if ln and rn else 0.0


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(value - lo) / (hi - lo) for value in values]


class GenerativeAgentsAdapter(MethodAdapter):
    """Forum adaptation of Park et al.'s memory/retrieval/reflection core.

    The score preserves upstream `new_retrieve`: recency×0.5,
    relevance×3, importance×2. The shared harness, not this class, decides
    visibility, activation and available forum actions.
    """

    method_id = "generative_agents"

    def __init__(
        self,
        *,
        embedder: Embedder,
        importance_scorer: ImportanceScorer,
        reflector: Reflector,
        recency_decay: float = 0.99,
        reflection_threshold: float = 150.0,
        retrieval_count: int = 30,
    ):
        self.embedder = embedder
        self.importance_scorer = importance_scorer
        self.reflector = reflector
        self.recency_decay = recency_decay
        self.reflection_threshold = reflection_threshold
        self.retrieval_count = retrieval_count
        self.seed: AgentSeed | None = None
        self.memories: list[MemoryNode] = []
        self.importance_since_reflection = 0.0
        self.auxiliary_log: list[Mapping[str, Any]] = []

    def initialize(self, seed: AgentSeed) -> None:
        self.seed = seed
        self.memories.clear()
        self.importance_since_reflection = 0.0
        self.auxiliary_log.clear()
        if seed.stable_profile:
            profile_text = "Stable role background: " + json.dumps(
                seed.stable_profile, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )
            self._add_memory(
                profile_text, tick=0, kind="profile", importance=5.0
            )

    def _add_memory(self, text: str, tick: int, kind: str = "event", importance: float | None = None) -> None:
        score = float(self.importance_scorer(text) if importance is None else importance)
        if not 0 <= score <= 10:
            raise ValueError("Generative Agents importance must be in [0,10]")
        self.memories.append(MemoryNode(
            node_id=f"memory-{len(self.memories):06d}", text=text,
            created_tick=tick, last_accessed_tick=tick, importance=score,
            embedding=tuple(float(value) for value in self.embedder(text)), kind=kind,
        ))
        self.importance_since_reflection += score

    def _maybe_reflect(self, tick: int) -> None:
        if self.importance_since_reflection < self.reflection_threshold:
            return
        evidence = tuple(node.text for node in self.memories[-100:])
        thoughts = self.reflector(evidence)
        self.auxiliary_log.append({"type": "reflection", "tick": tick, "input_count": len(evidence)})
        self.importance_since_reflection = 0.0
        for thought in thoughts:
            self._add_memory(thought, tick, kind="thought")

    def _retrieve(self, focal_text: str, tick: int) -> list[dict[str, Any]]:
        if not self.memories:
            return []
        focal = self.embedder(focal_text)
        recency = [self.recency_decay ** max(1, tick - node.last_accessed_tick + 1) for node in self.memories]
        relevance = [_cosine(node.embedding, focal) for node in self.memories]
        importance = [node.importance for node in self.memories]
        nr, nv, ni = _normalize(recency), _normalize(relevance), _normalize(importance)
        scored = []
        for index, node in enumerate(self.memories):
            score = 0.5 * nr[index] + 3.0 * nv[index] + 2.0 * ni[index]
            scored.append((score, node))
        chosen = sorted(scored, key=lambda pair: (-pair[0], pair[1].node_id))[: self.retrieval_count]
        for _, node in chosen:
            node.last_accessed_tick = tick
        return [
            {"memory_id": node.node_id, "kind": node.kind, "text": node.text, "retrieval_score": score}
            for score, node in chosen
        ]

    def observe(self, observation: CommonObservation) -> MethodSnapshot:
        if self.seed is None:
            raise RuntimeError("adapter is not initialized")
        for item in observation.visible_context:
            marker = f"observed:{observation.tick}:{item.message_id}"
            if not any(node.node_id == marker for node in self.memories):
                score = float(self.importance_scorer(item.text))
                if not 0 <= score <= 10:
                    raise ValueError("Generative Agents importance must be in [0,10]")
                node = MemoryNode(marker, item.text, observation.tick, observation.tick, score,
                                  tuple(float(v) for v in self.embedder(item.text)))
                self.memories.append(node)
                self.importance_since_reflection += score
        self._maybe_reflect(observation.tick)
        focal = " ".join(
            [str(observation.current_event)] + [item.text for item in observation.visible_context]
        )
        retrieved = self._retrieve(focal, observation.tick)
        return MethodSnapshot(
            self.method_id, observation.agent_id, observation.tick,
            {"memory_count": len(self.memories), "importance_since_reflection": self.importance_since_reflection},
            {
                "retrieved_memory_stream": retrieved,
                "planning_instruction": "Use the retrieved memories to plan one allowed forum action.",
            },
            observation.observation_sha256,
            tuple(self.auxiliary_log),
        )

    def commit_decision(self, observation: CommonObservation, decision: Mapping[str, Any]) -> None:
        text = str(decision.get("text") or "")
        if text:
            self._add_memory(f"self action: {text}", observation.tick)

    def export_probe(self) -> float | None:
        return None
