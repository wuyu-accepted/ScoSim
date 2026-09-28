"""Independent micro-persona audit for the four-dimensional ForumTwin run.

Resident Agents still produce natural language; this module never adds labels
to their prompts.  After a run, it builds blinded four-way persona retrieval
cases from the locally retained, sanitized turn evidence.  An independent
reviewer judges the generated text and direct context uptake.  The scorer then
compares that judgement with a private answer key and the following-tick state
transition.  Record-level rows stay private; only aggregate rates are intended
for a public fidelity report.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from typing import Any, Mapping, Sequence

from .governance_arena_v2 import STATE_FIELDS


PERSONA_AUDIT_CASE_SCHEMA = "campus-pulse-persona-audit-case-v1"
PERSONA_AUDIT_RELEASE_SCHEMA = "campus-pulse-persona-audit-release-v1"
PERSONA_JUDGEMENT_SCHEMA = "campus-pulse-persona-judgement-v1"
PERSONA_REVIEW_ROWS_SCHEMA = "campus-pulse-private-persona-review-rows-v1"
AUDITED_STATE_FIELDS = ("attention", "concern", "trust", "satisfaction")
STATE_DIRECTIONS = ("down", "stable", "up")


class ForumPersonaAuditError(ValueError):
    """Raised when blinded persona evidence violates its frozen contract."""


class PrivatePersonaEvidenceCollector:
    """Collect the minimum local evidence needed for a blinded D2 audit.

    The collector is deliberately not a public result type.  It retains a
    sanitized V5 model view, the validated structured response and four state
    values from the current and following committed ticks.  API credentials,
    HTTP bodies, source records and raw corpus text are never accepted.
    """

    def __init__(self, *, agent_ids_by_index: Sequence[str]) -> None:
        ids = tuple(str(value) for value in agent_ids_by_index)
        if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ForumPersonaAuditError("persona collector Agent index is invalid")
        self._agent_ids_by_index = ids
        self._turns: list[dict[str, Any]] = []
        self._states: dict[tuple[str, int, str], dict[str, float]] = {}
        self._lock = threading.Lock()

    def record_resident_turn(self, *, rendered: Any, envelope: Any) -> None:
        try:
            prompt = json.loads(str(rendered.agent_prompt))
            decision = json.loads(str(envelope.content))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ForumPersonaAuditError(
                "persona collector received a non-JSON resident turn"
            ) from exc
        if prompt.get("model_visible_view") != "campus-pulse-natural-resident-view-v1":
            return
        if "profile" in prompt or "posterior_state" in prompt:
            raise ForumPersonaAuditError("persona collector received a non-natural model view")
        for forbidden in ("api_key", "password", "secret", "provider_body"):
            if forbidden in _canonical(prompt).lower():
                raise ForumPersonaAuditError("persona collector received a forbidden field")
        scenario_context = prompt.get("scenario_context")
        scenario_context = (
            scenario_context if isinstance(scenario_context, Mapping) else {}
        )
        current_situation = "".join(
            value
            for value in (
                str(prompt.get("current_experience") or ""),
                str(scenario_context.get("your_current_situation") or ""),
            )
            if value
        )
        visible_context = prompt.get("forum_context")
        visible_context = (
            [dict(value) for value in visible_context if isinstance(value, Mapping)]
            if isinstance(visible_context, Sequence)
            and not isinstance(visible_context, (str, bytes))
            else []
        )
        row = {
            "agent_id": str(rendered.agent_id),
            "branch": str(rendered.branch),
            "tick": int(rendered.tick),
            "scenario_summary": str(
                scenario_context.get("event_summary")
                or scenario_context.get("scenario_brief")
                or "校园公共资源分配争议"
            ),
            "current_situation": current_situation,
            "resident_context": dict(prompt.get("resident_context") or {}),
            "visible_context": visible_context,
            "decision": decision,
            "prompt_sha256": str(rendered.prompt_sha256),
            "response_sha256": str(envelope.response_sha256),
            "provenance": str(envelope.provenance),
            "returned_model": str(envelope.returned_model),
        }
        with self._lock:
            self._turns.append(row)

    def record_checkpoint(
        self,
        *,
        branch: str,
        tick: int,
        snapshot: Mapping[str, Any],
    ) -> None:
        branch_state = snapshot.get("branch_state")
        ensemble = getattr(branch_state, "ensemble", None)
        if ensemble is None or not hasattr(ensemble, "parent_means"):
            raise ForumPersonaAuditError("persona checkpoint lacks a particle ensemble")
        means = ensemble.parent_means()
        if tuple(means.shape) != (len(self._agent_ids_by_index), len(STATE_FIELDS)):
            raise ForumPersonaAuditError("persona checkpoint state shape drifted")
        with self._lock:
            relevant = {str(row["agent_id"]) for row in self._turns}
            for index, agent_id in enumerate(self._agent_ids_by_index):
                if agent_id not in relevant:
                    continue
                values = {
                    field: round(float(means[index, ordinal]), 8)
                    for ordinal, field in enumerate(STATE_FIELDS)
                }
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in values.values()
                ):
                    raise ForumPersonaAuditError("persona checkpoint state is invalid")
                self._states[(str(branch), int(tick), agent_id)] = values

    def private_release(self) -> dict[str, Any]:
        with self._lock:
            turns = [dict(value) for value in self._turns]
            states = dict(self._states)
        rows = []
        for row in turns:
            branch = str(row["branch"])
            tick = int(row["tick"])
            agent_id = str(row["agent_id"])
            same_key = (branch, tick, agent_id)
            next_key = (branch, tick + 1, agent_id)
            if branch == "shared_baseline":
                next_key = ("Natural", tick + 1, agent_id)
            same_state = states.get(same_key)
            next_state = states.get(next_key)
            if same_state is None or next_state is None:
                continue
            rows.append(
                {
                    **row,
                    "same_tick_state": {
                        field: same_state[field] for field in AUDITED_STATE_FIELDS
                    },
                    "next_tick_state": {
                        field: next_state[field] for field in AUDITED_STATE_FIELDS
                    },
                }
            )
        body = {
            "schema_version": "campus-pulse-private-persona-turn-evidence-v1",
            "rows": rows,
            "captured_resident_turn_count": len(turns),
            "next_tick_eligible_turn_count": len(rows),
            "raw_provider_body_persisted": False,
            "api_credentials_persisted": False,
            "raw_corpus_text_persisted": False,
            "public_export_allowed": False,
        }
        body["content_sha256"] = _sha(body)
        return body


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _public_decision(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    decision = row.get("decision")
    if not isinstance(decision, Mapping):
        return None
    text = decision.get("visible_text")
    action = str(decision.get("action") or "")
    if not isinstance(text, str) or not text.strip() or action in {"read", "ignore"}:
        return None
    return decision


def _profile_card(row: Mapping[str, Any]) -> dict[str, str]:
    context = row.get("resident_context")
    if not isinstance(context, Mapping):
        raise ForumPersonaAuditError("turn evidence lacks natural resident_context")
    allowed = (
        "forum_identity",
        "reviewed_history",
        "usual_participation",
        "communication_style",
        "handling_uncertainty",
    )
    card = {}
    for key in allowed:
        value = context.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ForumPersonaAuditError(f"resident_context.{key} is missing")
        card[key] = value.strip()
    return card


def _visible_context(row: Mapping[str, Any]) -> list[dict[str, str]]:
    values = row.get("visible_context")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result = []
    for value in values[:6]:
        if not isinstance(value, Mapping):
            continue
        text = value.get("visible_text")
        relation = value.get("relation")
        if not isinstance(text, str) or not text.strip():
            continue
        result.append(
            {
                "relation": str(relation or "可见帖子"),
                "text": text.strip()[:240],
            }
        )
    return result


def _candidate_pool_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("branch") or ""),
        int(row.get("tick", -1)),
        str(row.get("agent_id") or ""),
    )


def _profile_similarity(left: Mapping[str, str], right: Mapping[str, str]) -> float:
    """Rank plausible distractors without using outputs or hidden state."""

    def bigrams(card: Mapping[str, str]) -> set[str]:
        text = "".join(str(card[key]) for key in sorted(card))
        return {text[index : index + 2] for index in range(max(0, len(text) - 1))}

    left_tokens = bigrams(left)
    right_tokens = bigrams(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _selected_rows(
    rows: Sequence[Mapping[str, Any]], *, sample_size: int
) -> list[Mapping[str, Any]]:
    eligible = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or _public_decision(row) is None:
            continue
        agent_id = str(row.get("agent_id") or "")
        branch = str(row.get("branch") or "")
        tick = row.get("tick")
        same_state = row.get("same_tick_state")
        next_state = row.get("next_tick_state")
        if (
            not agent_id
            or not branch
            or isinstance(tick, bool)
            or not isinstance(tick, int)
            or not isinstance(same_state, Mapping)
            or not isinstance(next_state, Mapping)
        ):
            continue
        _profile_card(row)
        identity = (branch, tick, agent_id)
        if identity in seen:
            continue
        seen.add(identity)
        eligible.append(row)
    if len(eligible) < sample_size:
        raise ForumPersonaAuditError(
            f"persona audit needs {sample_size} public turns with next-tick state; "
            f"only {len(eligible)} are eligible"
        )

    # Preserve time/branch variety without selecting on the generated answer.
    buckets: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in eligible:
        buckets.setdefault((str(row["branch"]), int(row["tick"])), []).append(row)
    for values in buckets.values():
        values.sort(key=lambda item: _sha(_candidate_pool_key(item)))
    selected = []
    keys = sorted(buckets)
    while len(selected) < sample_size and keys:
        remaining = []
        for key in keys:
            values = buckets[key]
            if values and len(selected) < sample_size:
                selected.append(values.pop(0))
            if values:
                remaining.append(key)
        keys = remaining
    return selected


def build_blinded_persona_audit_release(
    *,
    turn_evidence_rows: Sequence[Mapping[str, Any]],
    sample_size: int = 24,
) -> dict[str, Any]:
    """Create Provider-safe judge cases plus a separate private answer key."""

    if isinstance(sample_size, bool) or not 8 <= sample_size <= 96:
        raise ForumPersonaAuditError("persona audit sample_size must be from 8 through 96")
    selected = _selected_rows(turn_evidence_rows, sample_size=sample_size)
    profile_rows = []
    seen_cards = set()
    for row in turn_evidence_rows:
        if not isinstance(row, Mapping):
            continue
        try:
            card = _profile_card(row)
        except ForumPersonaAuditError:
            continue
        card_sha = _sha(card)
        if card_sha in seen_cards:
            continue
        seen_cards.add(card_sha)
        profile_rows.append(
            (
                str(row.get("agent_id") or ""),
                card,
                card_sha,
                card["forum_identity"],
            )
        )
    if len(profile_rows) < 4:
        raise ForumPersonaAuditError("persona audit needs at least four distinct profile cards")

    cases = []
    answers = []
    labels = ("A", "B", "C", "D")
    for ordinal, row in enumerate(selected, start=1):
        true_agent = str(row["agent_id"])
        true_card = _profile_card(row)
        true_sha = _sha(true_card)
        true_stratum = true_card["forum_identity"]
        distractors = [
            item
            for item in profile_rows
            if item[0] != true_agent
            and item[2] != true_sha
        ]
        distractors.sort(
            key=lambda item: (
                item[3] != true_stratum,
                -_profile_similarity(true_card, item[1]),
                _sha(["persona-distractor-v1", ordinal, item[2]]),
            )
        )
        if len(distractors) < 3:
            raise ForumPersonaAuditError("persona audit lacks distinct distractor cards")
        cards = [
            (true_agent, true_card, true_sha, true_stratum),
            *distractors[:3],
        ]
        rotation = int(_sha(["persona-label-v1", ordinal])[:8], 16) % 4
        cards = cards[rotation:] + cards[:rotation]
        correct_label = labels[
            next(index for index, item in enumerate(cards) if item[2] == true_sha)
        ]
        decision = _public_decision(row)
        assert decision is not None
        case_id = f"persona-case-{ordinal:03d}"
        case = {
            "schema_version": PERSONA_AUDIT_CASE_SCHEMA,
            "case_id": case_id,
            "task": (
                "根据四张候选人物卡、这个人当时看到的合成论坛内容和随后发出的文字，"
                "判断哪张人物卡最吻合；同时检查是否直接承接上下文、是否补造事实、"
                "是否像套话，并判断文字所表达的四项状态方向。"
            ),
            "scenario": str(row.get("scenario_summary") or "校园公共资源分配争议"),
            "current_situation": str(row.get("current_situation") or ""),
            "visible_forum_context": _visible_context(row),
            "observed_action": {
                "action": str(decision.get("action") or ""),
                "stance": str(decision.get("stance") or ""),
                "emotion": str(decision.get("emotion") or ""),
                "visible_text": str(decision.get("visible_text") or "").strip(),
            },
            "candidate_profiles": {
                label: card for label, (_, card, _, _) in zip(labels, cards)
            },
            "output_contract": {
                "schema_version": PERSONA_JUDGEMENT_SCHEMA,
                "case_id": case_id,
                "likely_profile": "A|B|C|D",
                "direct_context_uptake": "boolean",
                "unsupported_fact": "boolean",
                "template_like": "boolean",
                "state_direction": {
                    field: "down|stable|up" for field in AUDITED_STATE_FIELDS
                },
                "brief_reason": "不超过80个中文字符",
            },
        }
        cases.append(case)
        answers.append(
            {
                "case_id": case_id,
                "correct_profile": correct_label,
                "agent_id": true_agent,
                "branch": str(row["branch"]),
                "tick": int(row["tick"]),
                "same_tick_state": {
                    field: float(row["same_tick_state"][field])
                    for field in AUDITED_STATE_FIELDS
                },
                "next_tick_state": {
                    field: float(row["next_tick_state"][field])
                    for field in AUDITED_STATE_FIELDS
                },
            }
        )
    public_body = {
        "schema_version": PERSONA_AUDIT_RELEASE_SCHEMA,
        "sample_size": len(cases),
        "cases": cases,
        "distractor_selection": (
            "same broad forum role first, then nearest natural-language profile; "
            "generated response and hidden state never enter distractor selection"
        ),
        "raw_state_values_visible_to_judge": False,
        "agent_ids_visible_to_judge": False,
        "release_hashes_visible_to_judge": False,
    }
    public_body["release_sha256"] = _sha(public_body)
    private_key = {
        "schema_version": "campus-pulse-private-persona-answer-key-v1",
        "audit_release_sha256": public_body["release_sha256"],
        "answers": answers,
        "public_export_allowed": False,
    }
    private_key["answer_key_sha256"] = _sha(private_key)
    return {"judge_cases": public_body, "private_answer_key": private_key}


def _validate_judgement(value: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    expected = {
        "schema_version",
        "case_id",
        "likely_profile",
        "direct_context_uptake",
        "unsupported_fact",
        "template_like",
        "state_direction",
        "brief_reason",
    }
    if set(value) != expected or value.get("schema_version") != PERSONA_JUDGEMENT_SCHEMA:
        raise ForumPersonaAuditError("persona judgement fields do not match the contract")
    if value.get("case_id") != case_id or value.get("likely_profile") not in {
        "A",
        "B",
        "C",
        "D",
    }:
        raise ForumPersonaAuditError("persona judgement identity is invalid")
    for field in ("direct_context_uptake", "unsupported_fact", "template_like"):
        if not isinstance(value.get(field), bool):
            raise ForumPersonaAuditError(f"persona judgement {field} must be boolean")
    directions = value.get("state_direction")
    if not isinstance(directions, Mapping) or set(directions) != set(AUDITED_STATE_FIELDS):
        raise ForumPersonaAuditError("persona judgement state directions are incomplete")
    if any(str(directions[field]) not in STATE_DIRECTIONS for field in AUDITED_STATE_FIELDS):
        raise ForumPersonaAuditError("persona judgement state direction is invalid")
    reason = value.get("brief_reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 80:
        raise ForumPersonaAuditError("persona judgement brief_reason is invalid")
    return dict(value)


def _observed_direction(before: float, after: float, *, deadband: float = 0.015) -> str:
    delta = after - before
    if delta > deadband:
        return "up"
    if delta < -deadband:
        return "down"
    return "stable"


def score_persona_audit(
    *,
    audit_release: Mapping[str, Any],
    private_answer_key: Mapping[str, Any],
    judgements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Produce private rows accepted by ``build_four_dimension_fidelity_report``."""

    if audit_release.get("schema_version") != PERSONA_AUDIT_RELEASE_SCHEMA:
        raise ForumPersonaAuditError("persona audit release schema is invalid")
    if private_answer_key.get("audit_release_sha256") != audit_release.get(
        "release_sha256"
    ):
        raise ForumPersonaAuditError("persona answer key belongs to another audit release")
    cases = {
        str(row.get("case_id")): row
        for row in audit_release.get("cases") or ()
        if isinstance(row, Mapping)
    }
    answers = {
        str(row.get("case_id")): row
        for row in private_answer_key.get("answers") or ()
        if isinstance(row, Mapping)
    }
    judged = {}
    for raw in judgements:
        if not isinstance(raw, Mapping):
            raise ForumPersonaAuditError("persona judgement must be an object")
        case_id = str(raw.get("case_id") or "")
        if case_id not in cases or case_id in judged:
            raise ForumPersonaAuditError("persona judgement case is missing or duplicated")
        judged[case_id] = _validate_judgement(raw, case_id=case_id)
    if set(judged) != set(cases) or set(answers) != set(cases):
        raise ForumPersonaAuditError("persona audit is incomplete")

    rows = []
    for case_id in sorted(cases):
        judgement = judged[case_id]
        answer = answers[case_id]
        expected_codes = {"down": -1, "stable": 0, "up": 1}
        agreements = []
        errors = []
        for field in AUDITED_STATE_FIELDS:
            before = float(answer["same_tick_state"][field])
            after = float(answer["next_tick_state"][field])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (before, after)):
                raise ForumPersonaAuditError("private next-tick state is invalid")
            actual = _observed_direction(before, after)
            predicted = str(judgement["state_direction"][field])
            agreements.append(actual == predicted)
            errors.append(abs(expected_codes[actual] - expected_codes[predicted]) / 2.0)
        rows.append(
            {
                "profile_match": judgement["likely_profile"]
                == answer["correct_profile"],
                "direct_context_uptake": judgement["direct_context_uptake"],
                "unsupported_fact": judgement["unsupported_fact"],
                "template_like": judgement["template_like"],
                "next_tick_direction_match": sum(agreements) >= 3,
                "next_tick_state_mae": round(sum(errors) / len(errors), 6),
            }
        )
    body = {
        "schema_version": PERSONA_REVIEW_ROWS_SCHEMA,
        "audit_release_sha256": audit_release["release_sha256"],
        "row_count": len(rows),
        "rows": rows,
        "record_level_judgements_persisted_privately": True,
        "public_export_allowed": False,
        "measurement_rule": (
            "Four-way profile retrieval and context/fact/template checks are independent; "
            "state direction is compared with the following committed tick, never exposed "
            "to the judge and never the same assimilation step."
        ),
    }
    body["content_sha256"] = _sha(body)
    return body
