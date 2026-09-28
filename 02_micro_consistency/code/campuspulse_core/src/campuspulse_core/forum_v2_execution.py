"""ForumTwin v2 run-plan, census and order-invariant full-population execution."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from .forum_relationships import (
    RELATIONSHIP_RELEASE_SCHEMA_VERSION,
    STAKE_RELEASE_SCHEMA_VERSION,
    private_activation_quota,
)
from .forum_turns import (
    FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
    FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
    RenderedForumAgentTurnV3,
    RenderedForumAgentTurnV4,
    ValidatedForumAgentTurnV3,
    ValidatedForumAgentTurnV4,
    materialize_forum_agent_turn_v3,
    materialize_forum_agent_turn_v4,
    validate_forum_agent_turn_v3,
    validate_forum_agent_turn_v4,
)
from .forum_twin import (
    ANCHOR_COUNT,
    INTERACTION_REASON_PRIORITY,
    AgentMemory,
    ForumActivationRound,
    ForumLedger,
    ForumTwinError,
    InteractionCandidate,
    PpsCandidate,
    resident_turn_budget,
    schedule_forum_activations,
)
from .mass_population import canonical_hash


RUN_PLAN_V9_SCHEMA_VERSION = "campus-pulse-run-plan-v9"
CHECKPOINT_V5_SCHEMA_VERSION = "campus-pulse-forum-checkpoint-v5"
AGGREGATE_RESULT_V7_SCHEMA_VERSION = "campus-pulse-live-aggregate-result-v7"
EXECUTION_MODE = "llm_forum_twin_v2"
ACTIVATION_MODES = frozenset(
    {
        "budgeted_pps",
        "full_population_keyframes",
        "full_population_every_tick",
    }
)
CENSUS_PREFLIGHT_SCHEMA_VERSION = "campus-pulse-forum-census-preflight-v1"
CENSUS_CALIBRATION_SCHEMA_VERSION = "campus-pulse-forum-census-calibration-v1"
FULL_TICK_BATCH_SCHEMA_VERSION = "campus-pulse-full-population-tick-batch-v1"
GOVERNANCE_DECISION_TICKS = (3, 7, 13, 19)

CENSUS_AGENT_COUNT = 200
CENSUS_TICKS = (
    ("shared_baseline", 4),
    ("natural", 9),
    ("D", 9),
)
CENSUS_RESIDENT_SLOTS = 600
CENSUS_ACTOR_SLOTS = 3
CENSUS_PRIMARY_SLOTS = 603
CENSUS_REQUEST_LIMIT = 1_206
CENSUS_TOKEN_LIMIT = 6_000_000


@dataclass(frozen=True)
class ActivationDesign:
    activation_mode: str
    tick: int
    agent_indices: tuple[int, ...]
    inclusion_probabilities: tuple[float, ...]
    estimator: str
    census: bool

    def __post_init__(self) -> None:
        if self.activation_mode not in ACTIVATION_MODES:
            raise ForumTwinError("ForumTwin activation mode is invalid")
        if not 0 <= self.tick <= 23:
            raise ForumTwinError("ForumTwin activation tick is invalid")
        if len(self.agent_indices) != len(self.inclusion_probabilities):
            raise ForumTwinError("ForumTwin activation probabilities do not align")
        if len(set(self.agent_indices)) != len(self.agent_indices):
            raise ForumTwinError("ForumTwin activation repeats an agent")
        if any(not 0.0 < float(value) <= 1.0 for value in self.inclusion_probabilities):
            raise ForumTwinError("ForumTwin activation has non-positive inclusion probability")
        if self.census:
            if any(value != 1.0 for value in self.inclusion_probabilities):
                raise ForumTwinError("census activation requires inclusion probability one")
            if self.estimator != "direct_design_weighted_census":
                raise ForumTwinError("census activation must disable HT/GREG")
        elif self.estimator != "greg_ht_partial_pooling":
            raise ForumTwinError("budgeted activation must use GREG/HT partial pooling")

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "activation_mode": self.activation_mode,
                "tick": self.tick,
                "agent_indices": list(self.agent_indices),
                "inclusion_probabilities": list(self.inclusion_probabilities),
                "estimator": self.estimator,
                "census": self.census,
            }
        )


@dataclass(frozen=True)
class BudgetedForumV2Activation:
    """A v2 activation round with an auditable public/private queue split."""

    round: ForumActivationRound
    public_interaction_indices: tuple[int, ...]
    private_interaction_indices: tuple[int, ...]
    private_activation_limit: int
    postponed_private_candidates: tuple[InteractionCandidate, ...]
    all_eligible_pps_positive: bool

    def __post_init__(self) -> None:
        interaction_set = set(self.round.interaction_indices)
        public_set = set(self.public_interaction_indices)
        private_set = set(self.private_interaction_indices)
        if public_set & private_set or public_set | private_set != interaction_set:
            raise ForumTwinError("v2 activation channel accounting is inconsistent")
        if len(private_set) > self.private_activation_limit:
            raise ForumTwinError("private-origin activations exceeded the forty-percent cap")
        if not self.all_eligible_pps_positive:
            raise ForumTwinError("v2 activation contains a non-positive PPS probability")

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "round_sha256": self.round.content_hash,
                "public_interaction_indices": list(self.public_interaction_indices),
                "private_interaction_indices": list(self.private_interaction_indices),
                "private_activation_limit": self.private_activation_limit,
                "postponed_private_candidates": [
                    candidate.__dict__ for candidate in self.postponed_private_candidates
                ],
                "all_eligible_pps_positive": self.all_eligible_pps_positive,
            }
        )


def _interaction_rank(candidate: InteractionCandidate) -> tuple[Any, ...]:
    return (
        INTERACTION_REASON_PRIORITY[candidate.reason],
        -float(candidate.attention_score),
        candidate.enqueued_tick,
        candidate.sequence,
        candidate.agent_index,
    )


def schedule_budgeted_forum_v2_activations(
    *,
    agent_count: int,
    tick: int,
    anchor_indices: Sequence[int],
    public_interaction_candidates: Sequence[InteractionCandidate],
    private_interaction_candidates: Sequence[InteractionCandidate],
    pps_candidates: Sequence[PpsCandidate],
    resident_budget_override: int | None = None,
) -> BudgetedForumV2Activation:
    """Use the existing scheduler while enforcing the v2 private-work quota.

    Public direct replies, quotes, corrections and service receipts retain the
    scheduler's highest priority.  Ordinary private mentions and invitations
    are admitted as active-claim work; private risk/help items retain their
    higher safety priority.  Excess private work remains queued for a later
    tick and is never converted into synthetic language by Dynamics.
    """

    budget = (
        resident_turn_budget(tick)
        if resident_budget_override is None
        else int(resident_budget_override)
    )
    if budget < ANCHOR_COUNT:
        raise ForumTwinError("v2 activation budget cannot fit the frozen anchors")
    private_limit = private_activation_quota(
        non_anchor_slots=budget - ANCHOR_COUNT,
        requested_private_activations=len(private_interaction_candidates),
    )
    anchor_set = {int(value) for value in anchor_indices}

    # An agent present in both queues is treated as a public activation.  That
    # preserves direct public/service obligations and prevents one activation
    # from consuming both channel quotas.
    public_by_agent: dict[int, InteractionCandidate] = {}
    for candidate in sorted(public_interaction_candidates, key=_interaction_rank):
        if candidate.agent_index not in anchor_set:
            public_by_agent.setdefault(candidate.agent_index, candidate)

    private_by_agent: dict[int, InteractionCandidate] = {}
    for candidate in sorted(private_interaction_candidates, key=_interaction_rank):
        if (
            candidate.agent_index not in anchor_set
            and candidate.agent_index not in public_by_agent
        ):
            private_by_agent.setdefault(candidate.agent_index, candidate)
    ranked_private = sorted(private_by_agent.values(), key=_interaction_rank)
    admitted_private = ranked_private[:private_limit]
    quota_overflow = ranked_private[private_limit:]

    # The legacy scheduler knows semantic urgency but not channel privacy.
    # Demote only ordinary private reply-like reasons so a public direct reply,
    # correction or service receipt cannot be displaced by a private mention.
    normalized_private = []
    for candidate in admitted_private:
        reason = candidate.reason
        if reason in {"direct_reply", "quote", "correction", "service_receipt"}:
            reason = "active_claim_participant"
        normalized_private.append(
            InteractionCandidate(
                agent_index=candidate.agent_index,
                reason=reason,
                attention_score=candidate.attention_score,
                enqueued_tick=candidate.enqueued_tick,
                sequence=candidate.sequence,
            )
        )

    scheduled = schedule_forum_activations(
        agent_count=agent_count,
        tick=tick,
        anchor_indices=anchor_indices,
        interaction_candidates=(
            tuple(public_by_agent.values()) + tuple(normalized_private)
        ),
        pps_candidates=pps_candidates,
        resident_budget_override=resident_budget_override,
    )
    selected_set = set(scheduled.interaction_indices)
    private_agent_set = set(private_by_agent)
    selected_private = tuple(
        index for index in scheduled.interaction_indices if index in private_agent_set
    )
    selected_public = tuple(
        index for index in scheduled.interaction_indices if index not in private_agent_set
    )
    postponed_by_agent = {
        candidate.agent_index: candidate for candidate in quota_overflow
    }
    for candidate in admitted_private:
        if candidate.agent_index not in selected_set:
            postponed_by_agent.setdefault(candidate.agent_index, candidate)
    postponed = tuple(
        sorted(postponed_by_agent.values(), key=_interaction_rank)
    )
    return BudgetedForumV2Activation(
        round=scheduled,
        public_interaction_indices=selected_public,
        private_interaction_indices=selected_private,
        private_activation_limit=private_limit,
        postponed_private_candidates=postponed,
        all_eligible_pps_positive=all(
            float(candidate.inclusion_probability) > 0 for candidate in pps_candidates
        ),
    )


def activation_design_for_tick(
    *,
    activation_mode: str,
    tick: int,
    population_size: int,
    budgeted_indices: Sequence[int],
    budgeted_inclusion_probabilities: Sequence[float],
    keyframe_ticks: Sequence[int] = (),
) -> ActivationDesign:
    if activation_mode not in ACTIVATION_MODES or population_size <= 0:
        raise ForumTwinError("activation design configuration is invalid")
    keyframes = set(int(value) for value in keyframe_ticks)
    census = activation_mode == "full_population_every_tick" or (
        activation_mode == "full_population_keyframes" and tick in keyframes
    )
    if census:
        return ActivationDesign(
            activation_mode=activation_mode,
            tick=tick,
            agent_indices=tuple(range(population_size)),
            inclusion_probabilities=(1.0,) * population_size,
            estimator="direct_design_weighted_census",
            census=True,
        )
    indices = tuple(int(value) for value in budgeted_indices)
    probabilities = tuple(float(value) for value in budgeted_inclusion_probabilities)
    if not indices or len(indices) > resident_turn_budget(tick):
        raise ForumTwinError("budgeted activation violates the tick budget")
    if any(not 0 <= index < population_size for index in indices):
        raise ForumTwinError("budgeted activation contains an invalid agent index")
    return ActivationDesign(
        activation_mode=activation_mode,
        tick=tick,
        agent_indices=indices,
        inclusion_probabilities=probabilities,
        estimator="greg_ht_partial_pooling",
        census=False,
    )


@dataclass(frozen=True)
class FrozenPopulationTick:
    run_id: str
    scenario_id: str
    branch: str
    seed: int
    tick: int
    agent_ids: tuple[str, ...]
    public_board_sha256: str
    private_inboxes_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.scenario_id or not self.branch:
            raise ForumTwinError("frozen full-population tick identity is incomplete")
        if not 0 <= self.tick <= 23 or not self.agent_ids:
            raise ForumTwinError("frozen full-population tick is invalid")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ForumTwinError("frozen full-population tick repeats an agent")
        for value in (
            self.public_board_sha256,
            self.private_inboxes_sha256,
            self.state_sha256,
        ):
            if len(value) != 64:
                raise ForumTwinError("frozen full-population snapshot hash is invalid")

    @property
    def snapshot_sha256(self) -> str:
        return canonical_hash(
            {
                "run_id": self.run_id,
                "scenario_id": self.scenario_id,
                "branch": self.branch,
                "seed": self.seed,
                "tick": self.tick,
                "agent_ids": list(self.agent_ids),
                "public_board_sha256": self.public_board_sha256,
                "private_inboxes_sha256": self.private_inboxes_sha256,
                "state_sha256": self.state_sha256,
            }
        )


class FullPopulationTickTransaction:
    """Collect all settled outputs before one order-invariant batch commit."""

    def __init__(self, snapshot: FrozenPopulationTick) -> None:
        self.snapshot = snapshot
        self._expected = set(snapshot.agent_ids)
        self._outputs: dict[str, dict[str, Any]] = {}
        self._sealed = False

    @property
    def completed_count(self) -> int:
        return len(self._outputs)

    @property
    def remaining_count(self) -> int:
        return len(self._expected) - len(self._outputs)

    def submit(
        self,
        *,
        agent_id: str,
        settled_status: str,
        prompt_sha256: str,
        response_sha256: str,
        validated_output: Mapping[str, Any],
    ) -> None:
        if self._sealed:
            raise ForumTwinError("full-population tick is already sealed")
        if agent_id not in self._expected or agent_id in self._outputs:
            raise ForumTwinError("full-population tick agent is unknown or duplicated")
        if settled_status != "succeeded":
            raise ForumTwinError("unknown or failed output cannot enter a full tick batch")
        if len(prompt_sha256) != 64 or len(response_sha256) != 64:
            raise ForumTwinError("full-population tick response hashes are invalid")
        output = dict(validated_output)
        self._outputs[agent_id] = {
            "agent_id": agent_id,
            "prompt_sha256": prompt_sha256,
            "response_sha256": response_sha256,
            "validated_output": output,
            "validated_output_sha256": canonical_hash(output),
        }

    def seal(self) -> dict[str, Any]:
        if self._sealed:
            raise ForumTwinError("full-population tick batch was already sealed")
        if set(self._outputs) != self._expected:
            raise ForumTwinError("full-population tick cannot commit before every slot settles")
        ordered = [self._outputs[agent_id] for agent_id in sorted(self._outputs)]
        body = {
            "schema_version": FULL_TICK_BATCH_SCHEMA_VERSION,
            "snapshot_sha256": self.snapshot.snapshot_sha256,
            "run_id": self.snapshot.run_id,
            "scenario_id": self.snapshot.scenario_id,
            "branch": self.snapshot.branch,
            "seed": self.snapshot.seed,
            "tick": self.snapshot.tick,
            "slot_count": len(ordered),
            "outputs": ordered,
            "commit_order": "agent_id_ascending_after_all_validated",
        }
        self._sealed = True
        return {**body, "batch_sha256": canonical_hash(body)}

    def commit(
        self,
        materialize_batch: Callable[[tuple[Mapping[str, Any], ...]], Any],
    ) -> tuple[dict[str, Any], Any]:
        batch = self.seal()
        # The callback receives one immutable sorted batch. It is responsible
        # for applying it to a cloned state and swapping only after success.
        new_state = materialize_batch(tuple(batch["outputs"]))
        return batch, new_state


def _v3_validated_output(value: ValidatedForumAgentTurnV3) -> dict[str, Any]:
    """Return the exact structured decision without prompts or channel text metadata."""

    decision = asdict(value.decision)
    return {
        "schema_version": FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        "decision": decision,
        "engagements": [asdict(item) for item in value.interactions],
        "private_actions": [
            {
                **asdict(item),
                "target_agent_ids": list(item.target_agent_ids),
                "topic_tags": list(item.topic_tags),
            }
            for item in value.private_actions
        ],
        "provenance": {
            "kind": value.provenance.kind,
            "prompt_sha256": value.provenance.prompt_sha256,
            "response_sha256": value.provenance.response_sha256,
            "model": value.provenance.model,
            "trace_release_sha256": value.provenance.trace_release_sha256,
        },
    }


def execute_forum_v3_population_tick(
    *,
    snapshot: FrozenPopulationTick,
    rendered_by_agent: Mapping[str, RenderedForumAgentTurnV3],
    source: Any,
    ledger: ForumLedger,
    memories_by_agent: Mapping[str, AgentMemory],
    max_workers: int = 32,
    validated_output_guard: Callable[[ValidatedForumAgentTurnV3], None] | None = None,
    turn_validator: Callable[..., ValidatedForumAgentTurnV3] | None = None,
) -> dict[str, Any]:
    """Acquire one frozen V3 population tick, then commit in agent-id order.

    Every prompt is rendered before this function is called. Provider returns
    are validated against the same pre-tick ledger. No output becomes visible
    until every slot has settled; materialization happens on cloned state and
    is swapped only if the whole deterministic batch succeeds.
    """

    expected = set(snapshot.agent_ids)
    if (
        set(rendered_by_agent) != expected
        or set(memories_by_agent) != expected
        or not 1 <= int(max_workers) <= 32
        or any(
            rendered.agent_id != agent_id
            or rendered.scenario_id != snapshot.scenario_id
            or rendered.branch != snapshot.branch
            or rendered.tick != snapshot.tick
            for agent_id, rendered in rendered_by_agent.items()
        )
        or any(memory.agent_id != agent_id for agent_id, memory in memories_by_agent.items())
    ):
        raise ForumTwinError("frozen V3 population tick inputs do not align")
    transaction = FullPopulationTickTransaction(snapshot)
    validate_turn = turn_validator or validate_forum_agent_turn_v3

    def acquire(agent_id: str) -> tuple[str, Any, ValidatedForumAgentTurnV3]:
        rendered = rendered_by_agent[agent_id]
        def validator(envelope: Any) -> ValidatedForumAgentTurnV3:
            validated = validate_turn(
                envelope, rendered=rendered, ledger=ledger
            )
            if validated_output_guard is not None:
                validated_output_guard(validated)
            return validated
        envelope = source.acquire(
            slot_kind="resident",
            prompt_sha256=rendered.prompt_sha256,
            rendered=rendered,
            validator=validator,
        )
        validated = validator(envelope)
        return agent_id, envelope, validated

    settled: dict[str, tuple[Any, ValidatedForumAgentTurnV3]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(acquire, agent_id): agent_id
            for agent_id in snapshot.agent_ids
        }
        for future in as_completed(futures):
            agent_id, envelope, validated = future.result()
            settled[agent_id] = (envelope, validated)
            transaction.submit(
                agent_id=agent_id,
                settled_status="succeeded",
                prompt_sha256=rendered_by_agent[agent_id].prompt_sha256,
                response_sha256=envelope.response_sha256,
                validated_output=_v3_validated_output(validated),
            )

    def materialize(_ordered: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
        working_ledger = copy.deepcopy(ledger)
        working_memories = copy.deepcopy(dict(memories_by_agent))
        materializations = []
        for agent_id in sorted(snapshot.agent_ids):
            _envelope, validated = settled[agent_id]
            result = materialize_forum_agent_turn_v3(
                validated=validated,
                ledger=working_ledger,
                memory=working_memories[agent_id],
            )
            materializations.append(
                {
                    "agent_id": agent_id,
                    "action": result.action,
                    "message_id": result.message_id,
                    "thread_id": result.thread_id,
                    "claim_id": result.claim_id,
                    "interaction_count": len(result.interaction_ids),
                    "private_action_count": result.private_action_count,
                    "content_hash": result.content_hash,
                }
            )
        working_ledger.assert_consistent()
        return {
            "ledger": working_ledger,
            "memories_by_agent": working_memories,
            "materializations": materializations,
        }

    batch, state = transaction.commit(materialize)
    return {
        "batch": batch,
        "ledger": state["ledger"],
        "memories_by_agent": state["memories_by_agent"],
        "materializations": state["materializations"],
    }


def _v4_validated_output(value: ValidatedForumAgentTurnV4) -> dict[str, Any]:
    """Project a validated v4 response without private author identity."""

    base = _v3_validated_output(value.base_validated)
    return {
        **base,
        "schema_version": FORUM_AGENT_TURN_V4_SCHEMA_VERSION,
        "public_identity_mode": value.public_identity_mode,
    }


def execute_forum_v4_population_tick(
    *,
    snapshot: FrozenPopulationTick,
    rendered_by_agent: Mapping[str, RenderedForumAgentTurnV4],
    source: Any,
    ledger: ForumLedger,
    memories_by_agent: Mapping[str, AgentMemory],
    max_workers: int = 32,
    validated_output_guard: Callable[[ValidatedForumAgentTurnV4], None] | None = None,
) -> dict[str, Any]:
    """Execute an order-invariant full-population tick with optional anonymity.

    All public/private inputs are frozen before acquisition.  Anonymous author
    faces are materialized only after every slot has validated, so Provider
    completion order cannot leak same-tick speech or change the next board.
    """

    expected = set(snapshot.agent_ids)
    if (
        set(rendered_by_agent) != expected
        or set(memories_by_agent) != expected
        or not 1 <= int(max_workers) <= 32
        or any(
            rendered.agent_id != agent_id
            or rendered.scenario_id != snapshot.scenario_id
            or rendered.branch != snapshot.branch
            or rendered.tick != snapshot.tick
            for agent_id, rendered in rendered_by_agent.items()
        )
        or any(memory.agent_id != agent_id for agent_id, memory in memories_by_agent.items())
    ):
        raise ForumTwinError("frozen V4 population tick inputs do not align")
    transaction = FullPopulationTickTransaction(snapshot)

    def acquire(agent_id: str) -> tuple[str, Any, ValidatedForumAgentTurnV4]:
        rendered = rendered_by_agent[agent_id]

        def validator(envelope: Any) -> ValidatedForumAgentTurnV4:
            validated = validate_forum_agent_turn_v4(
                envelope, rendered=rendered, ledger=ledger
            )
            if validated_output_guard is not None:
                validated_output_guard(validated)
            return validated

        envelope = source.acquire(
            slot_kind="resident",
            prompt_sha256=rendered.prompt_sha256,
            rendered=rendered,
            validator=validator,
        )
        validated = validator(envelope)
        return agent_id, envelope, validated

    settled: dict[str, tuple[Any, ValidatedForumAgentTurnV4]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(acquire, agent_id): agent_id
            for agent_id in snapshot.agent_ids
        }
        for future in as_completed(futures):
            agent_id, envelope, validated = future.result()
            settled[agent_id] = (envelope, validated)
            transaction.submit(
                agent_id=agent_id,
                settled_status="succeeded",
                prompt_sha256=rendered_by_agent[agent_id].prompt_sha256,
                response_sha256=envelope.response_sha256,
                validated_output=_v4_validated_output(validated),
            )

    def materialize(_ordered: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
        working_ledger = copy.deepcopy(ledger)
        working_memories = copy.deepcopy(dict(memories_by_agent))
        materializations = []
        for agent_id in sorted(snapshot.agent_ids):
            _envelope, validated = settled[agent_id]
            result = materialize_forum_agent_turn_v4(
                validated=validated,
                ledger=working_ledger,
                memory=working_memories[agent_id],
            )
            materializations.append(
                {
                    "agent_id": agent_id,
                    "action": result.action,
                    "message_id": result.message_id,
                    "thread_id": result.thread_id,
                    "claim_id": result.claim_id,
                    "public_identity_mode": result.public_identity_mode,
                    "public_pseudonym_id": result.public_pseudonym_id,
                    "interaction_count": len(result.interaction_ids),
                    "private_action_count": result.private_action_count,
                    "content_hash": result.content_hash,
                }
            )
        working_ledger.assert_consistent()
        return {
            "ledger": working_ledger,
            "memories_by_agent": working_memories,
            "materializations": materializations,
        }

    batch, state = transaction.commit(materialize)
    return {
        "batch": batch,
        "ledger": state["ledger"],
        "memories_by_agent": state["memories_by_agent"],
        "materializations": state["materializations"],
    }


def require_private_shadow_output(value: ValidatedForumAgentTurnV3) -> None:
    """Reject any attempt to mutate the public channel in a shadow replay."""

    if value.decision.action not in {"read", "ignore"}:
        raise ForumTwinError("private shadow output attempted a public primary action")
    if value.interactions:
        raise ForumTwinError("private shadow output attempted a public engagement")


def planned_slot_counts(
    *,
    scenario_count: int,
    branch_count: int,
    governed_branch_count: int,
    seed_count: int,
    activation_mode: str,
    keyframe_ticks: Sequence[int] = (),
    tick_count: int = 24,
) -> dict[str, int]:
    """Return the exact shared-baseline/branch/actor work contract.

    T0-T2 are evaluated once and cloned into every branch.  T3 onward is
    evaluated separately for each branch.  This is the convention behind the
    frozen 19,006 / 45,012 / 108,048 full-population examples.
    """

    if (
        activation_mode not in ACTIVATION_MODES
        or scenario_count < 1
        or branch_count < 1
        or not 0 <= governed_branch_count <= branch_count
        or seed_count < 1
        or tick_count != 24
    ):
        raise ForumTwinError("ForumTwin slot-count configuration is invalid")
    keyframes = {int(value) for value in keyframe_ticks}
    if activation_mode == "full_population_keyframes":
        if not keyframes or any(not 0 <= value < tick_count for value in keyframes):
            raise ForumTwinError("keyframe slot count requires frozen ticks")
    elif keyframes:
        raise ForumTwinError("non-keyframe slot count cannot declare keyframes")

    def resident_slots(tick: int) -> int:
        if activation_mode == "full_population_every_tick" or (
            activation_mode == "full_population_keyframes" and tick in keyframes
        ):
            return 1_000
        return resident_turn_budget(tick)

    shared_resident = sum(resident_slots(tick) for tick in range(3))
    per_branch_resident = sum(
        resident_slots(tick) for tick in range(3, tick_count)
    )
    per_scenario_seed_resident = (
        shared_resident + branch_count * per_branch_resident
    )
    per_scenario_seed_actor = (
        governed_branch_count * len(GOVERNANCE_DECISION_TICKS) * 3
    )
    multiplier = scenario_count * seed_count
    resident = per_scenario_seed_resident * multiplier
    actor = per_scenario_seed_actor * multiplier
    return {
        "shared_baseline_resident_slots_per_scenario_seed": shared_resident,
        "post_baseline_resident_slots_per_branch_scenario_seed": per_branch_resident,
        "resident_slots": resident,
        "governance_actor_slots": actor,
        "primary_slots": resident + actor,
        "request_hard_limit": 2 * (resident + actor),
    }


def build_forum_twin_run_plan_v9(
    *,
    population_sha256: str,
    profile_release_sha256: str,
    relationship_release_sha256: str,
    stake_release_sha256: str,
    model_component_hashes: Mapping[str, str],
    scenario_ids: Sequence[str],
    branches: Sequence[str],
    seeds: Sequence[int],
    activation_mode: str,
    keyframe_ticks: Sequence[int] = (),
    tick_count: int = 24,
    execution_provenance: str = "emulator_only_development",
    provider_slot_limit: int = 0,
    provider_request_limit: int = 0,
    provider_token_limit: int = 0,
) -> dict[str, Any]:
    if activation_mode not in ACTIVATION_MODES:
        raise ForumTwinError("run-plan v9 activation mode is invalid")
    if tick_count != 24:
        raise ForumTwinError("run-plan v9 freezes the 24-tick timeline")
    scenarios = tuple(dict.fromkeys(str(value) for value in scenario_ids))
    branch_values = tuple(dict.fromkeys(str(value) for value in branches))
    seed_values = tuple(dict.fromkeys(int(value) for value in seeds))
    if not scenarios or not seed_values or not branch_values:
        raise ForumTwinError("run-plan v9 scenario, branch and seed sets are required")
    if set(branch_values) - {"natural", "A", "B", "C", "D"}:
        raise ForumTwinError("run-plan v9 branch set is invalid")
    keyframes = tuple(sorted(set(int(value) for value in keyframe_ticks)))
    if activation_mode == "full_population_keyframes":
        if not keyframes or any(not 0 <= value < tick_count for value in keyframes):
            raise ForumTwinError("keyframe mode requires frozen in-range census ticks")
    elif keyframes:
        raise ForumTwinError("only full_population_keyframes may declare keyframe ticks")
    for value in (
        population_sha256,
        profile_release_sha256,
        relationship_release_sha256,
        stake_release_sha256,
        *model_component_hashes.values(),
    ):
        if len(str(value)) != 64:
            raise ForumTwinError("run-plan v9 component hash is invalid")
    if execution_provenance not in {
        "emulator_only_development",
        "reviewed_trace_replay",
        "authorized_live_llm",
    }:
        raise ForumTwinError("run-plan v9 execution provenance is invalid")
    if min(provider_slot_limit, provider_request_limit, provider_token_limit) < 0:
        raise ForumTwinError("run-plan v9 provider limits cannot be negative")
    counts = planned_slot_counts(
        scenario_count=len(scenarios),
        branch_count=len(branch_values),
        governed_branch_count=sum(value != "natural" for value in branch_values),
        seed_count=len(seed_values),
        activation_mode=activation_mode,
        keyframe_ticks=keyframes,
        tick_count=tick_count,
    )
    if execution_provenance == "emulator_only_development":
        if any(
            value != 0
            for value in (
                provider_slot_limit,
                provider_request_limit,
                provider_token_limit,
            )
        ):
            raise ForumTwinError("development plan cannot reserve Provider budget")
    elif (
        provider_slot_limit != counts["primary_slots"]
        or provider_request_limit != counts["request_hard_limit"]
        or provider_token_limit <= 0
    ):
        raise ForumTwinError("formal run-plan v9 Provider limits do not match work")
    body = {
        "schema_version": RUN_PLAN_V9_SCHEMA_VERSION,
        "execution_mode": EXECUTION_MODE,
        "activation_mode": activation_mode,
        "population": {
            "agent_count": 1_000,
            "particles_per_agent": 10,
            "population_sha256": population_sha256,
            "profile_release_sha256": profile_release_sha256,
        },
        "relationships": {
            "schema_version": RELATIONSHIP_RELEASE_SCHEMA_VERSION,
            "relationship_release_sha256": relationship_release_sha256,
            "private_chat_enabled": True,
            "dynamic_groups_enabled": True,
            "cross_group_bridge_enabled": True,
            "public_ranking_is_global": True,
        },
        "stakes": {
            "schema_version": STAKE_RELEASE_SCHEMA_VERSION,
            "stake_release_sha256": stake_release_sha256,
        },
        "resident_turn_schema_version": FORUM_AGENT_TURN_V3_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_V5_SCHEMA_VERSION,
        "aggregate_result_schema_version": AGGREGATE_RESULT_V7_SCHEMA_VERSION,
        "model_component_hashes": dict(sorted(model_component_hashes.items())),
        "timeline": {"tick_count": tick_count, "keyframe_ticks": list(keyframes)},
        "scenarios": list(scenarios),
        "branches": list(branch_values),
        "seeds": list(seed_values),
        "execution_provenance": execution_provenance,
        "activation_contract": {
            "anchors_per_budgeted_tick": 16,
            "resident_budget_by_tick": [
                resident_turn_budget(tick) for tick in range(tick_count)
            ],
            "private_origin_share_of_non_anchor_slots_max": 0.40,
            "private_overflow_carries_to_next_tick": True,
            "census_inclusion_probability": 1.0,
            "census_uses_greg_ht": False,
            "same_tick_two_phase_commit": True,
        },
        "resident_output_contract": {
            "public_messages_per_agent_tick_max": 1,
            "public_interactions_per_agent_tick_max": 6,
            "private_actions_per_agent_tick_max": 2,
            "dynamics_may_generate_public_or_private_text": False,
        },
        "risk_contract": {
            "schema_version": "campus-pulse-governance-risk-result-v1",
            "signal_names": [
                "unsupported_claim_velocity",
                "private_to_public_spillover",
                "contested_claim_share",
                "cross_group_exposure_gap",
                "correction_penetration_and_delay",
                "help_backlog_and_sla_breach",
                "governance_observation_gap",
            ],
            "combined_score": None,
            "governance_reads_private_text": False,
        },
        "provider_limits": {
            "primary_slots": provider_slot_limit,
            "requests": provider_request_limit,
            "tokens": provider_token_limit,
            "network_retry_limit": 0,
            "format_repair_limit": 1,
        },
        "work_contract": counts,
        "privacy": {
            "governance_reads_private_text": False,
            "public_exports_private_agent_ids": False,
            "dynamics_generates_language": False,
        },
    }
    return {**body, "plan_sha256": canonical_hash(body)}


def validate_forum_twin_run_plan_v9(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != RUN_PLAN_V9_SCHEMA_VERSION:
        raise ForumTwinError("run-plan v9 schema is invalid")
    expected = canonical_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected:
        raise ForumTwinError("run-plan v9 hash drifted")
    rebuilt = build_forum_twin_run_plan_v9(
        population_sha256=str(plan["population"]["population_sha256"]),
        profile_release_sha256=str(plan["population"]["profile_release_sha256"]),
        relationship_release_sha256=str(
            plan["relationships"]["relationship_release_sha256"]
        ),
        stake_release_sha256=str(plan["stakes"]["stake_release_sha256"]),
        model_component_hashes=plan["model_component_hashes"],
        scenario_ids=plan["scenarios"],
        branches=plan["branches"],
        seeds=plan["seeds"],
        activation_mode=str(plan["activation_mode"]),
        keyframe_ticks=plan["timeline"]["keyframe_ticks"],
        tick_count=int(plan["timeline"]["tick_count"]),
        execution_provenance=str(plan["execution_provenance"]),
        provider_slot_limit=int(plan["provider_limits"]["primary_slots"]),
        provider_request_limit=int(plan["provider_limits"]["requests"]),
        provider_token_limit=int(plan["provider_limits"]["tokens"]),
    )
    if rebuilt != dict(plan):
        raise ForumTwinError("run-plan v9 contains non-canonical or unsupported fields")
    return rebuilt


def select_stratified_census_panel_200(
    *,
    agents: Sequence[Mapping[str, Any]],
    stake_positions: Sequence[Mapping[str, Any]],
    seed: int = 20260722,
) -> dict[str, Any]:
    """Select 200 agents by macro role, micro role, stake and activity."""

    if len(agents) != 1_000:
        raise ForumTwinError("census panel selection requires 1,000 agents")
    stakes = {str(row["agent_id"]): row for row in stake_positions}
    if len(stakes) != 1_000:
        raise ForumTwinError("census panel selection requires one stake position per agent")
    strata: dict[tuple[str, str, bool, str], list[Mapping[str, Any]]] = defaultdict(list)
    for agent in agents:
        agent_id = str(agent.get("agent_id") or "")
        if agent_id not in stakes:
            raise ForumTwinError("census panel agent lacks a stake position")
        fields = stakes[agent_id]["fields"]
        direct = bool(fields["directly_affected"]["value"])
        attention = float(agent.get("attention_budget", 0.5))
        activity_bin = "high" if attention >= 0.5 else "low"
        stratum = (
            str(agent.get("primary_macro_role") or ""),
            str(agent.get("micro_role_id") or ""),
            direct,
            activity_bin,
        )
        strata[stratum].append(agent)
    total = len(agents)
    quotas: dict[tuple[str, str, bool, str], int] = {}
    remainders: list[tuple[float, str, tuple[str, str, bool, str]]] = []
    for stratum, rows in strata.items():
        exact = CENSUS_AGENT_COUNT * len(rows) / total
        quota = min(len(rows), math.floor(exact))
        quotas[stratum] = quota
        remainders.append(
            (exact - quota, canonical_hash([seed, *stratum]), stratum)
        )
    remaining = CENSUS_AGENT_COUNT - sum(quotas.values())
    for _fraction, _tie, stratum in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if quotas[stratum] < len(strata[stratum]):
            quotas[stratum] += 1
            remaining -= 1
    if remaining:
        for stratum in sorted(strata, key=lambda value: canonical_hash([seed, *value])):
            while remaining and quotas[stratum] < len(strata[stratum]):
                quotas[stratum] += 1
                remaining -= 1
    selected: list[dict[str, Any]] = []
    for stratum in sorted(strata, key=str):
        rows = sorted(
            strata[stratum],
            key=lambda agent: canonical_hash(
                ["census-panel-v1", seed, agent["agent_id"]]
            ),
        )
        inclusion_probability = min(
            1.0, quotas[stratum] / len(rows) if rows else 0.0
        )
        for agent in rows[: quotas[stratum]]:
            selected.append(
                {
                    "agent_id": str(agent["agent_id"]),
                    "macro_role": str(agent.get("primary_macro_role") or ""),
                    "micro_role": str(agent.get("micro_role_id") or ""),
                    "directly_affected": stratum[2],
                    "activity_bin": stratum[3],
                    "panel_inclusion_probability": inclusion_probability,
                    "census_round_inclusion_probability": 1.0,
                }
            )
    selected.sort(key=lambda row: row["agent_id"])
    if len(selected) != CENSUS_AGENT_COUNT:
        raise ForumTwinError("census panel did not select exactly 200 agents")
    body = {
        "schema_version": "campus-pulse-forum-census-panel-v1",
        "seed": seed,
        "population_size": len(agents),
        "panel_size": len(selected),
        "stratification": [
            "macro_role",
            "micro_role",
            "directly_affected",
            "activity_bin",
        ],
        "members": selected,
    }
    return {**body, "panel_sha256": canonical_hash(body)}


def build_census_preflight_v1(
    *,
    plan_sha256: str,
    panel_sha256: str,
    endpoint: str,
    model: str = "gpt-5.6-luna",
) -> dict[str, Any]:
    if len(plan_sha256) != 64 or len(panel_sha256) != 64:
        raise ForumTwinError("census preflight release hash is invalid")
    if not endpoint.startswith("https://") or not model:
        raise ForumTwinError("census preflight Provider profile is invalid")
    body = {
        "schema_version": CENSUS_PREFLIGHT_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "panel_sha256": panel_sha256,
        "scenario_id": "governance_legitimacy_dispute",
        "seed": 20260722,
        "rounds": [
            {"branch": branch, "tick": tick, "resident_slots": 200}
            for branch, tick in CENSUS_TICKS
        ],
        "resident_slots": CENSUS_RESIDENT_SLOTS,
        "governance_actor_slots": CENSUS_ACTOR_SLOTS,
        "primary_slots": CENSUS_PRIMARY_SLOTS,
        "request_hard_limit": CENSUS_REQUEST_LIMIT,
        "token_hard_limit": CENSUS_TOKEN_LIMIT,
        "format_repair_limit": 1,
        "network_retry_limit": 0,
        "endpoint": endpoint,
        "model": model,
        "reasoning": "none",
        "contains_api_key": False,
        "ready": True,
    }
    return {**body, "preflight_sha256": canonical_hash(body)}


def _category_shares(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total = 0.0
    for row in rows:
        weight = float((weights or {}).get(str(row["agent_id"]), row.get("design_weight", 1.0)))
        category = str(row[field])
        totals[category] += weight
        total += weight
    return {key: value / total for key, value in sorted(totals.items())} if total else {}


def _pivotal_sample(
    rows: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    seed: int,
) -> tuple[list[Mapping[str, Any]], dict[str, float]]:
    if not 0 < budget <= len(rows):
        raise ForumTwinError("census resample budget is invalid")
    if budget == len(rows):
        return list(rows), {str(row["agent_id"]): 1.0 for row in rows}
    measures = [max(1e-12, float(row.get("design_weight", 1.0))) for row in rows]
    probabilities = [0.0] * len(rows)
    active = set(range(len(rows)))
    remaining = float(budget)
    while active:
        mass = sum(measures[index] for index in active)
        scale = remaining / mass
        certainty = [index for index in active if measures[index] * scale >= 1.0 - 1e-12]
        if not certainty:
            for index in active:
                probabilities[index] = measures[index] * scale
            break
        for index in certainty:
            probabilities[index] = 1.0
            active.remove(index)
            remaining -= 1.0
        if remaining <= 1e-12:
            break
    working = probabilities[:]
    step = 0
    while True:
        fractional = [i for i, value in enumerate(working) if 1e-12 < value < 1 - 1e-12]
        if len(fractional) < 2:
            break
        left, right = fractional[0], fractional[1]
        a, b = working[left], working[right]
        uniform = (
            int(canonical_hash(["census-pivotal", seed, step, rows[left]["agent_id"], rows[right]["agent_id"]])[:16], 16)
            + 0.5
        ) / 16**16
        pair_sum = a + b
        if pair_sum < 1:
            if uniform <= a / pair_sum:
                working[left], working[right] = pair_sum, 0.0
            else:
                working[left], working[right] = 0.0, pair_sum
        else:
            if uniform <= (1.0 - b) / (2.0 - pair_sum):
                working[left], working[right] = 1.0, pair_sum - 1.0
            else:
                working[left], working[right] = pair_sum - 1.0, 1.0
        step += 1
    selected = [row for row, final in zip(rows, working) if final >= 1.0 - 1e-9]
    if len(selected) != budget or any(value <= 0 for value in probabilities):
        raise ForumTwinError("census pivotal sample violated exact positive probability")
    return selected, {
        str(row["agent_id"]): float(probability)
        for row, probability in zip(rows, probabilities)
    }


def _greg_share(
    rows: Sequence[Mapping[str, Any]],
    sample: Sequence[Mapping[str, Any]],
    inclusion: Mapping[str, float],
    field: str,
) -> dict[str, float]:
    categories = sorted({str(row[field]) for row in rows})
    total_weight = sum(float(row.get("design_weight", 1.0)) for row in rows)
    totals: dict[str, float] = {}
    surrogate_field = f"surrogate_{field}_probabilities"
    for category in categories:
        model_total = sum(
            float(row.get("design_weight", 1.0))
            * float((row.get(surrogate_field) or {}).get(category, 0.0))
            for row in rows
        )
        correction = 0.0
        for row in sample:
            agent_id = str(row["agent_id"])
            weight = float(row.get("design_weight", 1.0))
            observed = 1.0 if str(row[field]) == category else 0.0
            predicted = float((row.get(surrogate_field) or {}).get(category, 0.0))
            correction += weight / float(inclusion[agent_id]) * (observed - predicted)
        totals[category] = min(total_weight, max(0.0, model_total + correction))
    normalizer = sum(totals.values())
    if normalizer <= 0:
        return _category_shares(sample, field)
    return {category: totals[category] / normalizer for category in categories}


def evaluate_budgeted_sampling_against_census(
    *,
    census_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int] = (16, 24, 32, 48),
    repetitions: int = 80,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Treat the 200 live decisions as a finite LLM population for calibration."""

    if len(census_rows) != CENSUS_AGENT_COUNT or repetitions < 20:
        raise ForumTwinError("census calibration requires 200 rows and at least 20 repetitions")
    required = {"agent_id", "action", "stance", "topic", "private_action", "design_weight"}
    if any(not required.issubset(row) for row in census_rows):
        raise ForumTwinError("census decision row is incomplete")
    fields = ("action", "stance", "topic", "private_action")
    truth = {field: _category_shares(census_rows, field) for field in fields}
    budget_results = []
    for budget in budgets:
        estimates = {field: [] for field in fields}
        positive_pi = True
        recalls = []
        for repetition in range(repetitions):
            sample, inclusion = _pivotal_sample(
                census_rows,
                int(budget),
                seed=seed + repetition,
            )
            positive_pi = positive_pi and all(value > 0 for value in inclusion.values())
            for field in fields:
                estimates[field].append(
                    _greg_share(census_rows, sample, inclusion, field)
                )
            should_act = [row for row in census_rows if bool(row.get("should_public_action"))]
            if should_act:
                recalls.append(
                    sum(bool(row.get("scheduler_selected")) for row in should_act)
                    / len(should_act)
                )
        field_mae: dict[str, float] = {}
        coverage_hits = 0
        coverage_total = 0
        for field in fields:
            categories = sorted(truth[field])
            errors = []
            for estimate in estimates[field]:
                errors.extend(
                    abs(float(estimate.get(category, 0.0)) - truth[field][category])
                    for category in categories
                )
            field_mae[field] = sum(errors) / len(errors) if errors else 0.0
            for category in categories:
                values = sorted(float(estimate.get(category, 0.0)) for estimate in estimates[field])
                lower = values[max(0, math.floor(0.025 * (len(values) - 1)))]
                upper = values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))]
                coverage_hits += lower - 1e-12 <= truth[field][category] <= upper + 1e-12
                coverage_total += 1
        mean_mae = sum(field_mae.values()) / len(field_mae)
        coverage = coverage_hits / coverage_total if coverage_total else 0.0
        scheduler_recall = sum(recalls) / len(recalls) if recalls else 1.0
        budget_results.append(
            {
                "budget": int(budget),
                "field_mae": field_mae,
                "mean_mae": mean_mae,
                "coverage_95": coverage,
                "positive_inclusion_probabilities": positive_pi,
                "public_action_scheduler_recall": scheduler_recall,
            }
        )
    by_budget = {row["budget"]: row for row in budget_results}
    gates = {
        "b24_all_field_mae_le_0_06": 24 in by_budget
        and all(value <= 0.06 for value in by_budget[24]["field_mae"].values()),
        "b32_all_field_mae_le_0_05": 32 in by_budget
        and all(value <= 0.05 for value in by_budget[32]["field_mae"].values()),
        "coverage_0_90_to_0_99": 32 in by_budget
        and 0.90 <= by_budget[32]["coverage_95"] <= 0.99,
        "scheduler_recall_ge_0_90": 32 in by_budget
        and by_budget[32]["public_action_scheduler_recall"] >= 0.90,
        "positive_inclusion_probabilities": all(
            row["positive_inclusion_probabilities"] for row in budget_results
        ),
    }
    body = {
        "schema_version": CENSUS_CALIBRATION_SCHEMA_VERSION,
        "finite_population_size": len(census_rows),
        "budgets": budget_results,
        "truth": truth,
        "gates": gates,
        "status": "pass" if all(gates.values()) else "fail",
    }
    return {**body, "calibration_sha256": canonical_hash(body)}


def _scan_public_result(value: Any, path: str = "$") -> None:
    forbidden = (
        "private_text",
        "private_agent_id",
        "conversation_id",
        "record_level_relationship_edges",
        "relationship_edge_records",
        "raw_prompt",
        "provider_body",
        "api_key",
        "credential",
        "source_id",
        "hidden_truth",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden):
                if child is not False:
                    raise ForumTwinError(
                        f"aggregate result v7 contains private material at {path}.{key}"
                    )
            _scan_public_result(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _scan_public_result(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ForumTwinError("aggregate result v7 contains a non-finite number")


def validate_aggregate_result_v7(
    result: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_plan_sha256: str | None = None,
    expected_launch_fingerprint: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_id",
        "launch_fingerprint",
        "plan_sha256",
        "execution_provenance",
        "activation_mode",
        "publication_eligible",
        "public_metrics",
        "private_aggregate_metrics",
        "risk_results",
        "governance",
        "census_calibration",
        "relationship_ablation",
        "completeness",
        "usage",
        "privacy",
        "non_claims",
        "result_sha256",
    }
    if not isinstance(result, Mapping) or set(result) != required:
        raise ForumTwinError("aggregate result v7 fields drifted")
    if result.get("schema_version") != AGGREGATE_RESULT_V7_SCHEMA_VERSION:
        raise ForumTwinError("aggregate result v7 schema is invalid")
    if result.get("activation_mode") not in ACTIVATION_MODES:
        raise ForumTwinError("aggregate result v7 activation mode is invalid")
    if result.get("execution_provenance") not in {
        "emulator_only_development",
        "reviewed_trace_replay",
        "authorized_live_llm",
    } or not isinstance(result.get("publication_eligible"), bool):
        raise ForumTwinError("aggregate result v7 provenance is invalid")
    for value, label in (
        (result.get("plan_sha256"), "plan"),
        (result.get("launch_fingerprint"), "launch"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ForumTwinError(f"aggregate result v7 {label} hash is invalid")
    if expected_run_id is not None and result.get("run_id") != expected_run_id:
        raise ForumTwinError("aggregate result v7 run identity drifted")
    if (
        expected_plan_sha256 is not None
        and result.get("plan_sha256") != expected_plan_sha256
    ):
        raise ForumTwinError("aggregate result v7 plan identity drifted")
    if (
        expected_launch_fingerprint is not None
        and result.get("launch_fingerprint") != expected_launch_fingerprint
    ):
        raise ForumTwinError("aggregate result v7 launch identity drifted")
    completeness = result.get("completeness")
    if (
        not isinstance(completeness, Mapping)
        or set(completeness)
        != {"required_primary_slots", "completed_primary_slots", "status"}
        or not isinstance(completeness.get("required_primary_slots"), int)
        or not isinstance(completeness.get("completed_primary_slots"), int)
        or not 0
        <= completeness["completed_primary_slots"]
        <= completeness["required_primary_slots"]
    ):
        raise ForumTwinError("aggregate result v7 completeness is invalid")
    if result["publication_eligible"] and (
        result["execution_provenance"] == "emulator_only_development"
        or completeness["status"] != "complete"
        or completeness["completed_primary_slots"]
        != completeness["required_primary_slots"]
    ):
        raise ForumTwinError("aggregate result v7 is not publication eligible")
    privacy = result.get("privacy")
    if not isinstance(privacy, Mapping) or any(
        privacy.get(key) is not False
        for key in (
            "contains_private_text",
            "contains_private_agent_ids",
            "contains_relationship_edges",
            "contains_credentials",
        )
    ):
        raise ForumTwinError("aggregate result v7 privacy boundary failed")
    ablation = result.get("relationship_ablation")
    if ablation is not None:
        if (
            not isinstance(ablation, Mapping)
            or ablation.get("schema_version")
            != "campus-pulse-forum-relationship-ablation-public-result-v1"
            or not isinstance(ablation.get("public_result_sha256"), str)
            or ablation.get("public_result_sha256")
            != canonical_hash(
                {
                    key: value
                    for key, value in ablation.items()
                    if key != "public_result_sha256"
                }
            )
            or ablation.get("privacy", {}).get("contains_private_text") is not False
        ):
            raise ForumTwinError("aggregate result v7 relationship ablation is invalid")
    _scan_public_result(result)
    expected = canonical_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    if result.get("result_sha256") != expected:
        raise ForumTwinError("aggregate result v7 hash drifted")
    return dict(result)


def seal_aggregate_result_v7(body: Mapping[str, Any]) -> dict[str, Any]:
    if "result_sha256" in body:
        raise ForumTwinError("aggregate result v7 body is already sealed")
    result = {**dict(body), "result_sha256": canonical_hash(body)}
    return validate_aggregate_result_v7(result)


def build_aggregate_result_v7(
    *,
    run_id: str,
    launch_fingerprint: str,
    plan_sha256: str,
    execution_provenance: str,
    activation_mode: str,
    publication_eligible: bool,
    public_metrics: Mapping[str, Any],
    private_aggregate_metrics: Mapping[str, Any],
    risk_results: Sequence[Mapping[str, Any]],
    governance: Mapping[str, Any],
    census_calibration: Mapping[str, Any] | None,
    relationship_ablation: Mapping[str, Any] | None,
    required_primary_slots: int,
    completed_primary_slots: int,
    completeness_status: str,
    usage: Mapping[str, Any],
    non_claims: Sequence[str],
) -> dict[str, Any]:
    """Build the sole public result-v7 projection.

    Private messages, member identities and record-level relationships are
    deliberately not accepted as parameters.  This keeps every executor and
    evidence compiler on the same aggregate-only boundary.
    """

    if (
        not isinstance(required_primary_slots, int)
        or not isinstance(completed_primary_slots, int)
        or not 0 <= completed_primary_slots <= required_primary_slots
        or not isinstance(completeness_status, str)
        or not completeness_status
        or any(not isinstance(value, str) or not value for value in non_claims)
    ):
        raise ForumTwinError("aggregate result v7 completeness or non-claims are invalid")
    checked_risks: list[dict[str, Any]] = []
    for risk in risk_results:
        if (
            not isinstance(risk, Mapping)
            or risk.get("schema_version")
            != "campus-pulse-governance-risk-result-v1"
            or not isinstance(risk.get("signals"), Sequence)
            or len(risk["signals"]) != 7
            or risk.get("combined_score") is not None
            or risk.get("privacy", {}).get("contains_private_text") is not False
        ):
            raise ForumTwinError("aggregate result v7 risk projection is invalid")
        checked_risks.append(dict(risk))
    body = {
        "schema_version": AGGREGATE_RESULT_V7_SCHEMA_VERSION,
        "run_id": run_id,
        "launch_fingerprint": launch_fingerprint,
        "plan_sha256": plan_sha256,
        "execution_provenance": execution_provenance,
        "activation_mode": activation_mode,
        "publication_eligible": bool(publication_eligible),
        "public_metrics": dict(public_metrics),
        "private_aggregate_metrics": dict(private_aggregate_metrics),
        "risk_results": checked_risks,
        "governance": dict(governance),
        "census_calibration": (
            None if census_calibration is None else dict(census_calibration)
        ),
        "relationship_ablation": (
            None if relationship_ablation is None else dict(relationship_ablation)
        ),
        "completeness": {
            "required_primary_slots": required_primary_slots,
            "completed_primary_slots": completed_primary_slots,
            "status": completeness_status,
        },
        "usage": dict(usage),
        "privacy": {
            "contains_private_text": False,
            "contains_private_agent_ids": False,
            "contains_relationship_edges": False,
            "contains_credentials": False,
        },
        "non_claims": list(non_claims),
    }
    return seal_aggregate_result_v7(body)


__all__ = [
    "ACTIVATION_MODES",
    "AGGREGATE_RESULT_V7_SCHEMA_VERSION",
    "CHECKPOINT_V5_SCHEMA_VERSION",
    "CENSUS_PRIMARY_SLOTS",
    "CENSUS_REQUEST_LIMIT",
    "CENSUS_TOKEN_LIMIT",
    "RUN_PLAN_V9_SCHEMA_VERSION",
    "ActivationDesign",
    "BudgetedForumV2Activation",
    "FrozenPopulationTick",
    "FullPopulationTickTransaction",
    "activation_design_for_tick",
    "build_census_preflight_v1",
    "build_aggregate_result_v7",
    "build_forum_twin_run_plan_v9",
    "evaluate_budgeted_sampling_against_census",
    "execute_forum_v3_population_tick",
    "execute_forum_v4_population_tick",
    "planned_slot_counts",
    "schedule_budgeted_forum_v2_activations",
    "seal_aggregate_result_v7",
    "select_stratified_census_panel_200",
    "validate_aggregate_result_v7",
    "validate_forum_twin_run_plan_v9",
]
