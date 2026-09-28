"""Private ForumTwin corpus, review-selection, and profile compilation contracts.

This module owns the zero-Provider part of ForumTwin data preparation.  Raw
rows are represented only by explicitly private dataclasses; every release
builder emits hashes, aggregate counts and synthetic semantic identifiers, but
never source paths, source identifiers, or source text.  Embedding functions
validate manifests only: downloading or running an embedding model belongs to
the separately pinned data-prep environment.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from .mass_population import canonical_hash
from .semantic_population import CONTACT_PATTERNS, generalize_local_person_names, heuristic_difficulty, redact_for_provider


CORPUS_SNAPSHOT_SCHEMA = "campus-pulse-forum-corpus-snapshot-v1"
CORPUS_INDEX_SCHEMA = "campus-pulse-forum-corpus-index-v1"
EMBEDDING_SHARD_PLAN_SCHEMA = "campus-pulse-forum-embedding-shard-plan-v1"
EMBEDDING_RESUME_SCHEMA = "campus-pulse-forum-embedding-resume-manifest-v1"
PRIVATE_THREAD_MANIFEST_SCHEMA = "campus-pulse-private-forum-thread-manifest-v1"
REVIEW_PAYLOAD_SAMPLE_SCHEMA = "campus-pulse-forum-review-payload-sample-v1"
ACTIVE_REVIEW_PLAN_SCHEMA = "campus-pulse-forum-active-review-plan-v1"
ACTIVE_REVIEW_PREFLIGHT_SCHEMA = "campus-pulse-forum-review-preflight-v1"
MICRO_ROLE_SCHEMA = "campus-pulse-forum-micro-role-release-v1"
PROFILE_RELEASE_SCHEMA = "campus-pulse-forum-profile-release-v1"
POPULATION_SCHEMA = "semantic-llm-forum-agent-population-v2"
REVIEWED_BEHAVIOR_PORTFOLIO_SCHEMA = (
    "campus-pulse-reviewed-behavior-portfolio-v1"
)

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512
EMBEDDING_DTYPE = "float16"

INITIAL_REVIEW_THREADS = 1_500
REVIEW_ROUND_THREADS = 500
MAX_REVIEW_THREADS = 3_000
MAX_REVIEW_COMMENTS = 6
MIN_CLUSTER_COVERAGE = 0.95
MIN_RARE_FUNCTION_REVIEWS = 50
MAX_MACRO_SHARE_DELTA = 0.01

MIN_MICRO_ROLES = 80
MAX_MICRO_ROLES = 120
MIN_ROLES_PER_MACRO = 4
MAX_ROLES_PER_MACRO = 10
MIN_EPISODES_PER_ROLE = 30
MACRO_ROLE_COUNT = 17
MIN_PROFILE_BUNDLES = 300
MIN_EFFECTIVE_BUNDLES = 250.0
MAX_PROMPT_CLONE_SHARE = 0.01
MIN_EVIDENCE_MATCH_RATE = 0.90
MAX_FALLBACK_RATE = 0.10
FORUM_AGENT_COUNT = 1_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s]+")
_POSIX_PATH = re.compile(r"(?:^|\s)/(?:home|Users|var|tmp|opt)/[^\s]+")


class ForumDataError(ValueError):
    """Raised when a ForumTwin private-data contract cannot fail closed."""


def _require_sha256(value: Any, field: str) -> str:
    result = str(value or "").lower()
    if not _SHA256.fullmatch(result):
        raise ForumDataError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _require_nonempty(value: Any, field: str, *, maximum: int = 240) -> str:
    result = " ".join(str(value or "").split())
    if not result or len(result) > maximum:
        raise ForumDataError(f"{field} must contain 1-{maximum} characters")
    return result


def _parse_iso(value: Any, field: str) -> str:
    text = str(value or "")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForumDataError(f"{field} must be ISO-8601") from exc
    return text


def _safe_profile_text(value: Any, field: str, *, maximum: int = 240) -> str:
    text = _require_nonempty(value, field, maximum=maximum)
    if _EMAIL.search(text) or _PHONE.search(text) or _WINDOWS_PATH.search(text) or _POSIX_PATH.search(text):
        raise ForumDataError(f"{field} contains prohibited identifying or path-like text")
    return text


def _normalized_string_list(
    value: Any,
    field: str,
    *,
    minimum: int = 1,
    maximum: int = 8,
    item_maximum: int = 80,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ForumDataError(f"{field} must be a list")
    items = tuple(
        dict.fromkeys(
            _safe_profile_text(item, f"{field} item", maximum=item_maximum)
            for item in value
        )
    )
    if not minimum <= len(items) <= maximum:
        raise ForumDataError(f"{field} must contain {minimum}-{maximum} unique items")
    return items


def _probability_map(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ForumDataError(f"{field} must be a nonempty object")
    result: dict[str, float] = {}
    for key, raw in sorted(value.items()):
        label = _require_nonempty(key, f"{field} key", maximum=80)
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ForumDataError(f"{field}.{label} must be numeric") from exc
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ForumDataError(f"{field}.{label} must be a finite value in [0, 1]")
        result[label] = round(number, 4)
    return result


def _release_hash(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = canonical_hash(result)
    return result


def build_corpus_snapshot_release_v1(
    *,
    posts: Mapping[str, Any],
    comments: Mapping[str, Any],
    captured_at_utc: str,
    coverage_start: str,
    coverage_end: str,
    collection_regimes: Sequence[Mapping[str, Any]],
    known_gap_dates: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a path-free immutable snapshot description from local file audit data."""

    def source(value: Mapping[str, Any], role: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ForumDataError(f"{role} source audit must be an object")
        fields = value.get("fields")
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ForumDataError(f"{role} fields must be a nonempty list")
        row_count = int(value.get("row_count", -1))
        byte_size = int(value.get("byte_size", -1))
        if row_count < 0 or byte_size < 0:
            raise ForumDataError(f"{role} row_count/byte_size must be nonnegative")
        return {
            "role": role,
            "row_count": row_count,
            "byte_size": byte_size,
            "sha256": _require_sha256(value.get("sha256"), f"{role}.sha256"),
            "modified_at_utc": _parse_iso(value.get("modified_at_utc"), f"{role}.modified_at_utc"),
            "encoding": _require_nonempty(value.get("encoding"), f"{role}.encoding", maximum=32),
            "fields": sorted({_require_nonempty(item, f"{role}.field", maximum=80) for item in fields}),
        }

    regimes: list[dict[str, str]] = []
    for index, item in enumerate(collection_regimes):
        if not isinstance(item, Mapping):
            raise ForumDataError("collection regime must be an object")
        regimes.append(
            {
                "regime_id": _require_nonempty(item.get("regime_id"), f"regime[{index}].id", maximum=80),
                "start_inclusive": _parse_iso(item.get("start_inclusive"), f"regime[{index}].start"),
                "end_exclusive": _parse_iso(item.get("end_exclusive"), f"regime[{index}].end"),
                "comparability": _require_nonempty(item.get("comparability"), f"regime[{index}].comparability", maximum=120),
            }
        )
    body = {
        "schema_version": CORPUS_SNAPSHOT_SCHEMA,
        "captured_at_utc": _parse_iso(captured_at_utc, "captured_at_utc"),
        "sources": [source(posts, "posts"), source(comments, "comments")],
        "coverage": {
            "start_inclusive": _parse_iso(coverage_start, "coverage_start"),
            "end_inclusive": _parse_iso(coverage_end, "coverage_end"),
            "known_gap_dates": sorted({_parse_iso(item, "known_gap_date") for item in known_gap_dates}),
            "collection_regimes": regimes,
            "comment_dates_inherit_parent_date": True,
            "reply_latency_claim_allowed": False,
        },
        "privacy": {
            "source_paths_persisted": False,
            "source_record_ids_persisted": False,
            "raw_text_persisted": False,
            "contact_details_persisted": False,
        },
    }
    return _release_hash(body, "snapshot_sha256")


def audit_csv_file_v1(
    path: str | Path,
    *,
    role: str,
    encoding: str = "auto",
) -> dict[str, Any]:
    """Read-only CSV inventory whose return value deliberately omits ``path``."""

    source = Path(path)
    if role not in {"posts", "comments"}:
        raise ForumDataError("CSV role must be posts or comments")
    if not source.is_file():
        raise ForumDataError(f"{role} CSV does not exist")
    with source.open("rb") as prefix_handle:
        prefix = prefix_handle.read(4)
    if encoding == "auto":
        if prefix.startswith(b"\xef\xbb\xbf"):
            chosen_encoding = "utf-8-sig"
        elif prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
            chosen_encoding = "utf-16"
        else:
            chosen_encoding = "utf-8"
    else:
        chosen_encoding = _require_nonempty(encoding, "encoding", maximum=32)
    digest = hashlib.sha256()
    byte_size = 0
    with source.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_size += len(block)
    try:
        with source.open("r", encoding=chosen_encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            required = (
                {"post_code", "real_date", "content"}
                if role == "posts"
                else {"post_code", "comment_id", "real_date", "floor", "content"}
            )
            if not required.issubset(fields):
                raise ForumDataError(f"{role} CSV is missing required fields: {sorted(required - set(fields))}")
            row_count = 0
            parseable_dates = 0
            unparseable_dates = 0
            earliest: str | None = None
            latest: str | None = None
            for row in reader:
                row_count += 1
                value = str(row.get("real_date") or "")
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    unparseable_dates += 1
                    continue
                parseable_dates += 1
                normalized = parsed.date().isoformat()
                earliest = normalized if earliest is None else min(earliest, normalized)
                latest = normalized if latest is None else max(latest, normalized)
    except UnicodeError as exc:
        raise ForumDataError(f"{role} CSV cannot be decoded as {chosen_encoding}") from exc
    stat = source.stat()
    return {
        "role": role,
        "row_count": row_count,
        "byte_size": byte_size,
        "sha256": digest.hexdigest(),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "encoding": chosen_encoding,
        "fields": fields,
        "date_audit": {
            "parseable_count": parseable_dates,
            "unparseable_count": unparseable_dates,
            "earliest": earliest,
            "latest": latest,
        },
        "source_path_persisted": False,
    }


def read_csv_rows_private_v1(
    path: str | Path,
    *,
    encoding: str,
) -> Iterable[dict[str, str]]:
    """Yield private rows.  Callers must not serialize these dictionaries."""

    with Path(path).open("r", encoding=encoding, newline="") as handle:
        yield from csv.DictReader(handle)


@dataclass(frozen=True)
class PrivatePostRecord:
    """An in-process row.  It must never be serialized as a public release."""

    private_post_code: str
    semantic_row_id: str
    content: str
    content_sha256: str
    parent_date: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class PrivateCommentRecord:
    """An in-process comment linked by a private source code."""

    private_post_code: str
    private_comment_id: str
    semantic_row_id: str
    content: str
    content_sha256: str
    floor: int
    parent_date: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class DeduplicationResult:
    records: tuple[PrivatePostRecord | PrivateCommentRecord, ...]
    conflicts: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]


def _content_hash(value: str) -> str:
    return canonical_hash({"content": value})


def deduplicate_posts_v1(rows: Iterable[Mapping[str, Any]]) -> DeduplicationResult:
    """Deduplicate by private post_code+content hash and retain safe conflict markers."""

    unique: dict[tuple[str, str], PrivatePostRecord] = {}
    by_code: dict[str, set[str]] = defaultdict(set)
    input_rows = 0
    for row in rows:
        input_rows += 1
        code = _require_nonempty(row.get("post_code"), "post_code", maximum=200)
        content = str(row.get("content") or "")
        digest = _content_hash(content)
        key = (code, digest)
        by_code[code].add(digest)
        unique.setdefault(
            key,
            PrivatePostRecord(
                private_post_code=code,
                semantic_row_id="post-" + canonical_hash(["forumtwin-post", code, digest])[:24],
                content=content,
                content_sha256=digest,
                parent_date=_parse_iso(row.get("real_date"), "post.real_date"),
                payload={
                    key: value
                    for key, value in row.items()
                    if key not in {"post_code", "content", "id", "post_hash"}
                },
            ),
        )
    conflict_codes = {code: values for code, values in by_code.items() if len(values) > 1}
    conflicts = tuple(
        {
            "conflict_id": "post-conflict-" + canonical_hash(["post", code])[:20],
            "candidate_count": len(digests),
            "requires_human_review": True,
        }
        for code, digests in sorted(conflict_codes.items())
    )
    records = tuple(sorted(unique.values(), key=lambda item: item.semantic_row_id))
    audit = {
        "kind": "posts",
        "input_rows": input_rows,
        "unique_rows": len(records),
        "exact_duplicate_rows": input_rows - len(records),
        "conflict_groups": len(conflicts),
        "conflict_candidate_rows": sum(len(value) for value in conflict_codes.values()),
        "raw_text_persisted": False,
        "source_ids_persisted": False,
        "audit_sha256": canonical_hash([record.semantic_row_id for record in records]),
    }
    return DeduplicationResult(records, conflicts, audit)


def _floor(value: Any) -> int:
    try:
        floor = int(value)
    except (TypeError, ValueError):
        return 2**31 - 1
    return floor if floor >= 0 else 2**31 - 1


def deduplicate_comments_v1(rows: Iterable[Mapping[str, Any]]) -> DeduplicationResult:
    """Deduplicate comments and preserve same-ID/different-body conflict groups."""

    unique: dict[tuple[str, str, str], PrivateCommentRecord] = {}
    by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    input_rows = 0
    for row in rows:
        input_rows += 1
        code = _require_nonempty(row.get("post_code"), "comment.post_code", maximum=200)
        comment_id = _require_nonempty(row.get("comment_id"), "comment.comment_id", maximum=200)
        content = str(row.get("content") or "")
        digest = _content_hash(content)
        key = (code, comment_id, digest)
        by_identity[(code, comment_id)].add(digest)
        unique.setdefault(
            key,
            PrivateCommentRecord(
                private_post_code=code,
                private_comment_id=comment_id,
                semantic_row_id="comment-" + canonical_hash(["forumtwin-comment", code, comment_id, digest])[:24],
                content=content,
                content_sha256=digest,
                floor=_floor(row.get("floor")),
                parent_date=_parse_iso(row.get("real_date"), "comment.real_date"),
                payload={
                    key: value
                    for key, value in row.items()
                    if key not in {"post_code", "comment_id", "content", "id", "comment_hash"}
                },
            ),
        )
    conflict_groups = {key: values for key, values in by_identity.items() if len(values) > 1}
    conflicts = tuple(
        {
            "conflict_id": "comment-conflict-" + canonical_hash(["comment", *key])[:20],
            "candidate_count": len(digests),
            "requires_human_review": True,
        }
        for key, digests in sorted(conflict_groups.items())
    )
    records = tuple(
        sorted(unique.values(), key=lambda item: (item.private_post_code, item.floor, item.semantic_row_id))
    )
    audit = {
        "kind": "comments",
        "input_rows": input_rows,
        "unique_rows": len(records),
        "exact_duplicate_rows": input_rows - len(records),
        "conflict_groups": len(conflicts),
        "conflict_candidate_rows": sum(len(value) for value in conflict_groups.values()),
        "raw_text_persisted": False,
        "source_ids_persisted": False,
        "audit_sha256": canonical_hash([record.semantic_row_id for record in records]),
    }
    return DeduplicationResult(records, conflicts, audit)


@dataclass(frozen=True)
class PrivateForumThread:
    semantic_thread_id: str
    private_post_code: str
    parent_date: str
    root_candidates: tuple[PrivatePostRecord, ...]
    comments: tuple[PrivateCommentRecord, ...]
    requires_human_review: bool


def rebuild_private_threads_v1(
    posts: DeduplicationResult,
    comments: DeduplicationResult,
) -> tuple[tuple[PrivateForumThread, ...], dict[str, Any]]:
    """Rebuild private threads; comment dates are replaced by the parent date."""

    post_groups: dict[str, list[PrivatePostRecord]] = defaultdict(list)
    for record in posts.records:
        if not isinstance(record, PrivatePostRecord):
            raise ForumDataError("posts result contains a non-post record")
        post_groups[record.private_post_code].append(record)
    comment_groups: dict[str, list[PrivateCommentRecord]] = defaultdict(list)
    orphan_comments = 0
    for record in comments.records:
        if not isinstance(record, PrivateCommentRecord):
            raise ForumDataError("comments result contains a non-comment record")
        if record.private_post_code not in post_groups:
            orphan_comments += 1
            continue
        comment_groups[record.private_post_code].append(record)
    threads: list[PrivateForumThread] = []
    inherited_dates = 0
    for code, roots in sorted(post_groups.items()):
        parent_date = min(root.parent_date for root in roots)
        rebuilt_comments = []
        for comment in comment_groups.get(code, []):
            if comment.parent_date != parent_date:
                inherited_dates += 1
            rebuilt_comments.append(
                PrivateCommentRecord(
                    private_post_code=comment.private_post_code,
                    private_comment_id=comment.private_comment_id,
                    semantic_row_id=comment.semantic_row_id,
                    content=comment.content,
                    content_sha256=comment.content_sha256,
                    floor=comment.floor,
                    parent_date=parent_date,
                    payload=comment.payload,
                )
            )
        identity_counts = Counter(item.private_comment_id for item in rebuilt_comments)
        threads.append(
            PrivateForumThread(
                semantic_thread_id="thread-" + canonical_hash(["forumtwin-thread", code])[:24],
                private_post_code=code,
                parent_date=parent_date,
                root_candidates=tuple(sorted(roots, key=lambda item: item.semantic_row_id)),
                comments=tuple(sorted(rebuilt_comments, key=lambda item: (item.floor, item.semantic_row_id))),
                requires_human_review=(len(roots) != 1 or any(count > 1 for count in identity_counts.values())),
            )
        )
    audit = {
        "thread_count": len(threads),
        "orphan_comment_rows": orphan_comments,
        "comment_dates_replaced_by_parent_date": len(comments.records) - orphan_comments,
        "comment_dates_that_differed": inherited_dates,
        "reply_latency_inferred": False,
        "conflict_threads": sum(thread.requires_human_review for thread in threads),
        "thread_manifest_sha256": canonical_hash(
            [
                {
                    "semantic_thread_id": thread.semantic_thread_id,
                    "root_count": len(thread.root_candidates),
                    "comment_ids": [comment.semantic_row_id for comment in thread.comments],
                }
                for thread in threads
            ]
        ),
    }
    return tuple(threads), audit


def build_private_thread_manifest_v1(
    *,
    corpus_snapshot_sha256: str,
    threads: Sequence[PrivateForumThread],
) -> dict[str, Any]:
    """Serialize thread structure without private linkage keys or source text."""

    safe_threads = []
    for thread in sorted(threads, key=lambda item: item.semantic_thread_id):
        safe_threads.append(
            {
                "semantic_thread_id": thread.semantic_thread_id,
                "parent_date": thread.parent_date,
                "root_semantic_row_ids": [item.semantic_row_id for item in thread.root_candidates],
                "comments": [
                    {
                        "semantic_row_id": item.semantic_row_id,
                        "floor": item.floor,
                    }
                    for item in thread.comments
                ],
                "requires_human_review": thread.requires_human_review,
            }
        )
    body = {
        "schema_version": PRIVATE_THREAD_MANIFEST_SCHEMA,
        "corpus_snapshot_sha256": _require_sha256(corpus_snapshot_sha256, "corpus_snapshot_sha256"),
        "thread_count": len(safe_threads),
        "comment_count": sum(len(item["comments"]) for item in safe_threads),
        "conflict_thread_count": sum(item["requires_human_review"] for item in safe_threads),
        "threads": safe_threads,
        "privacy": {
            "source_linkage_keys_persisted": False,
            "raw_text_persisted": False,
            "source_paths_persisted": False,
        },
    }
    return _release_hash(body, "thread_manifest_sha256")


@dataclass(frozen=True)
class EmbeddingShard:
    shard_id: str
    row_ids: tuple[str, ...]
    vector_count: int
    dimension: int
    dtype: str
    payload_sha256: str


def build_embedding_shard_plan_v1(
    *,
    corpus_snapshot_sha256: str,
    semantic_row_ids: Sequence[str],
    model_revision: str,
    model_weights_sha256: str,
    tokenizer_sha256: str,
    model_config_sha256: str,
    shard_size: int = 50_000,
) -> dict[str, Any]:
    """Plan full local embedding work without loading or downloading a model."""

    if not 1 <= shard_size <= 100_000:
        raise ForumDataError("embedding shard_size must be between 1 and 100000")
    row_ids = [str(item) for item in semantic_row_ids]
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise ForumDataError("embedding plan requires nonempty unique semantic row ids")
    shards = []
    for start in range(0, len(row_ids), shard_size):
        values = row_ids[start : start + shard_size]
        shards.append(
            {
                "shard_id": f"embedding-shard-{start // shard_size + 1:05d}",
                "start_row": start,
                "end_row_exclusive": start + len(values),
                "row_count": len(values),
                "semantic_row_ids": values,
                "row_id_sha256": canonical_hash(values),
                "state": "pending",
            }
        )
    body = {
        "schema_version": EMBEDDING_SHARD_PLAN_SCHEMA,
        "corpus_snapshot_sha256": _require_sha256(corpus_snapshot_sha256, "corpus_snapshot_sha256"),
        "embedding": {
            "model": EMBEDDING_MODEL,
            "model_revision": _require_nonempty(model_revision, "model_revision", maximum=160),
            "model_weights_sha256": _require_sha256(model_weights_sha256, "model_weights_sha256"),
            "tokenizer_sha256": _require_sha256(tokenizer_sha256, "tokenizer_sha256"),
            "model_config_sha256": _require_sha256(model_config_sha256, "model_config_sha256"),
            "dimension": EMBEDDING_DIMENSION,
            "dtype": EMBEDDING_DTYPE,
            "model_download_performed": False,
            "embedding_inference_performed": False,
        },
        "rows": {
            "count": len(row_ids),
            "semantic_row_order_sha256": canonical_hash(row_ids),
        },
        "shard_size": shard_size,
        "shards": shards,
        "privacy": {
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "source_paths_persisted": False,
        },
    }
    return _release_hash(body, "plan_sha256")


def build_embedding_resume_manifest_v1(
    plan: Mapping[str, Any],
    *,
    completed_shards: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate resumable shard completion metadata; no vector bytes are read."""

    if plan.get("schema_version") != EMBEDDING_SHARD_PLAN_SCHEMA:
        raise ForumDataError("embedding shard plan schema mismatch")
    expected_plan_hash = canonical_hash({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("plan_sha256") != expected_plan_hash:
        raise ForumDataError("embedding shard plan hash mismatch")
    planned = {str(item["shard_id"]): item for item in plan.get("shards", [])}
    completed: dict[str, dict[str, Any]] = {}
    for item in completed_shards:
        shard_id = str(item.get("shard_id") or "")
        if shard_id not in planned or shard_id in completed:
            raise ForumDataError("completed embedding shard is unknown or duplicated")
        expected = planned[shard_id]
        row_count = int(item.get("vector_count", -1))
        dimension = int(item.get("dimension", -1))
        dtype = str(item.get("dtype") or "")
        row_id_sha256 = str(item.get("row_id_sha256") or "")
        if (
            row_count != int(expected["row_count"])
            or dimension != EMBEDDING_DIMENSION
            or dtype != EMBEDDING_DTYPE
            or row_id_sha256 != expected["row_id_sha256"]
        ):
            raise ForumDataError("completed embedding shard layout does not match its plan")
        completed[shard_id] = {
            "shard_id": shard_id,
            "vector_count": row_count,
            "dimension": dimension,
            "dtype": dtype,
            "row_id_sha256": row_id_sha256,
            "payload_sha256": _require_sha256(item.get("payload_sha256"), "payload_sha256"),
        }
    completed_ordered = [completed[key] for key in planned if key in completed]
    pending = [key for key in planned if key not in completed]
    body = {
        "schema_version": EMBEDDING_RESUME_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "completed_shards": completed_ordered,
        "pending_shard_ids": pending,
        "completed_row_count": sum(item["vector_count"] for item in completed_ordered),
        "total_row_count": int(plan["rows"]["count"]),
        "merge_ready": not pending,
        "merge_identity_sha256": canonical_hash(completed_ordered) if not pending else None,
        "provider_calls": 0,
        "model_download_performed": False,
        "embedding_inference_performed_by_manifest_builder": False,
    }
    return _release_hash(body, "resume_manifest_sha256")


def build_embedding_index_release_v1(
    *,
    corpus_snapshot_sha256: str,
    model_revision: str,
    model_weights_sha256: str,
    tokenizer_sha256: str,
    model_config_sha256: str,
    expected_row_ids: Sequence[str],
    shards: Sequence[EmbeddingShard],
) -> dict[str, Any]:
    """Validate completed embedding shards and bind their deterministic merge order."""

    expected = tuple(str(item) for item in expected_row_ids)
    if len(expected) != len(set(expected)):
        raise ForumDataError("expected embedding row ids must be unique")
    seen: list[str] = []
    safe_shards: list[dict[str, Any]] = []
    shard_ids = set()
    for shard in shards:
        if shard.shard_id in shard_ids:
            raise ForumDataError("embedding shard ids must be unique")
        shard_ids.add(shard.shard_id)
        if shard.vector_count != len(shard.row_ids):
            raise ForumDataError("embedding shard vector_count does not match row ids")
        if shard.dimension != EMBEDDING_DIMENSION or shard.dtype != EMBEDDING_DTYPE:
            raise ForumDataError("embedding shard layout is incompatible")
        _require_sha256(shard.payload_sha256, "embedding shard payload_sha256")
        seen.extend(shard.row_ids)
        safe_shards.append(
            {
                "shard_id": _require_nonempty(shard.shard_id, "shard_id", maximum=80),
                "row_count": shard.vector_count,
                "row_id_sha256": canonical_hash(list(shard.row_ids)),
                "payload_sha256": shard.payload_sha256,
            }
        )
    if tuple(seen) != expected:
        raise ForumDataError("embedding shards do not exactly cover expected rows in order")
    body = {
        "schema_version": CORPUS_INDEX_SCHEMA,
        "corpus_snapshot_sha256": _require_sha256(corpus_snapshot_sha256, "corpus_snapshot_sha256"),
        "embedding": {
            "model": EMBEDDING_MODEL,
            "model_revision": _require_nonempty(model_revision, "model_revision", maximum=160),
            "model_weights_sha256": _require_sha256(model_weights_sha256, "model_weights_sha256"),
            "tokenizer_sha256": _require_sha256(tokenizer_sha256, "tokenizer_sha256"),
            "model_config_sha256": _require_sha256(model_config_sha256, "model_config_sha256"),
            "dimension": EMBEDDING_DIMENSION,
            "dtype": EMBEDDING_DTYPE,
            "full_text_local_only": True,
        },
        "rows": {
            "count": len(expected),
            "semantic_row_order_sha256": canonical_hash(list(expected)),
            "complete": True,
        },
        "shards": safe_shards,
        "merge": {
            "order": [item["shard_id"] for item in safe_shards],
            "merge_sha256": canonical_hash(safe_shards),
            "idempotent": True,
        },
        "privacy": {
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "source_paths_persisted": False,
        },
    }
    return _release_hash(body, "index_sha256")


def verify_embedding_index_release_v1(
    release: Mapping[str, Any],
    *,
    expected_row_ids: Sequence[str],
    shards: Sequence[EmbeddingShard],
) -> None:
    rebuilt = build_embedding_index_release_v1(
        corpus_snapshot_sha256=str(release.get("corpus_snapshot_sha256") or ""),
        model_revision=str(release.get("embedding", {}).get("model_revision") or ""),
        model_weights_sha256=str(release.get("embedding", {}).get("model_weights_sha256") or ""),
        tokenizer_sha256=str(release.get("embedding", {}).get("tokenizer_sha256") or ""),
        model_config_sha256=str(release.get("embedding", {}).get("model_config_sha256") or ""),
        expected_row_ids=expected_row_ids,
        shards=shards,
    )
    if dict(release) != rebuilt:
        raise ForumDataError("embedding index release is not an idempotent rebuild")


def _review_stratum(item: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get(field) or "unknown")
        for field in (
            "month",
            "topic_cluster",
            "thread_size_bucket",
            "language_difficulty",
            "reply_function",
        )
    )


def select_active_review_threads_v1(
    threads: Sequence[Mapping[str, Any]],
    *,
    already_selected: Sequence[str] = (),
    round_index: int = 0,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Select a deterministic, stratified initial or active-learning review round."""

    if round_index < 0 or round_index > 3:
        raise ForumDataError("review round_index must be between 0 and 3")
    selected_before = set(str(item) for item in already_selected)
    if len(selected_before) > MAX_REVIEW_THREADS:
        raise ForumDataError("review selection already exceeds the frozen cap")
    budget = INITIAL_REVIEW_THREADS if round_index == 0 else REVIEW_ROUND_THREADS
    budget = min(budget, MAX_REVIEW_THREADS - len(selected_before))
    eligible = []
    for item in threads:
        semantic_id = str(item.get("semantic_thread_id") or "")
        if not semantic_id or semantic_id in selected_before:
            continue
        if item.get("requires_human_review") is True:
            continue
        priority = (
            float(item.get("uncertainty", 0.0))
            + float(item.get("rare_cluster_score", 0.0))
            + float(item.get("temporal_drift_score", 0.0))
        )
        eligible.append((item, semantic_id, _review_stratum(item), priority))
    if budget > len(eligible):
        raise ForumDataError("not enough eligible threads for the frozen review budget")
    groups: dict[tuple[str, ...], list[tuple[Mapping[str, Any], str, tuple[str, ...], float]]] = defaultdict(list)
    for row in eligible:
        groups[row[2]].append(row)
    ranked_groups = []
    for stratum, rows in groups.items():
        rows.sort(
            key=lambda row: (
                -row[3] if round_index else 0.0,
                canonical_hash([seed, round_index, row[1]]),
            )
        )
        ranked_groups.append((stratum, rows))
    ranked_groups.sort(key=lambda item: (len(item[1]), item[0]))
    chosen: list[tuple[Mapping[str, Any], str, tuple[str, ...], float]] = []
    # Cover every represented stratum first when the budget permits.
    for _stratum, rows in ranked_groups:
        if len(chosen) == budget:
            break
        chosen.append(rows.pop(0))
    remaining = [row for _stratum, rows in ranked_groups for row in rows]
    remaining.sort(
        key=lambda row: (
            -row[3] if round_index else 0.0,
            canonical_hash([seed, round_index, row[1]]),
        )
    )
    chosen.extend(remaining[: budget - len(chosen)])
    chosen.sort(key=lambda row: row[1])
    selected_ids = [row[1] for row in chosen]
    if len(selected_ids) != budget or len(set(selected_ids)) != budget:
        raise ForumDataError("active-review selection violated its fixed budget")
    body = {
        "schema_version": ACTIVE_REVIEW_PLAN_SCHEMA,
        "round_index": round_index,
        "seed": seed,
        "budget": budget,
        "selected_before_count": len(selected_before),
        "selected_total_after_count": len(selected_before) + budget,
        "semantic_thread_ids": selected_ids,
        "stratum_counts": {
            "|".join(stratum): count
            for stratum, count in sorted(Counter(row[2] for row in chosen).items())
        },
        "priority_inputs": ["uncertainty", "rare_cluster_score", "temporal_drift_score"],
        "source_ids_included": False,
        "raw_text_included": False,
    }
    return _release_hash(body, "selection_sha256")


def build_structural_review_descriptors_v1(
    threads: Sequence[PrivateForumThread],
) -> list[dict[str, Any]]:
    """Build a pre-embedding dry-run frame, explicitly not a semantic selection."""

    descriptors = []
    for thread in threads:
        if not thread.root_candidates:
            continue
        root = thread.root_candidates[0]
        size = len(thread.comments)
        if size == 0:
            size_bucket = "zero"
        elif size <= 3:
            size_bucket = "small"
        elif size <= 12:
            size_bucket = "medium"
        else:
            size_bucket = "large"
        redacted, _ = redact_for_provider(root.content, max_characters=600)
        descriptors.append(
            {
                "semantic_thread_id": thread.semantic_thread_id,
                "month": thread.parent_date[:7],
                "topic_cluster": "pending-local-embedding",
                "thread_size_bucket": size_bucket,
                "language_difficulty": heuristic_difficulty(redacted),
                "reply_function": "pending-active-review",
                "uncertainty": 1.0,
                "rare_cluster_score": 0.0,
                "temporal_drift_score": 0.0,
                "requires_human_review": thread.requires_human_review,
                "selection_basis": "structural_pre_embedding_dry_run",
            }
        )
    return descriptors


def _privacy_findings(value: Any, *, key: str = "") -> Counter[str]:
    findings: Counter[str] = Counter()
    forbidden_keys = {
        "post_code",
        "comment_id",
        "source_id",
        "source_key",
        "source_path",
        "path",
        "api_key",
        "credential",
        "raw_text",
    }
    if key.casefold() in forbidden_keys:
        findings["source_ids_detected" if "id" in key or "code" in key or "key" in key else "paths_detected"] += 1
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            findings.update(_privacy_findings(child, key=str(child_key)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            findings.update(_privacy_findings(child, key=key))
    elif isinstance(value, str):
        findings["contacts_detected"] += sum(bool(pattern.search(value)) for pattern in CONTACT_PATTERNS)
        findings["contacts_detected"] += bool(_EMAIL.search(value)) + bool(_PHONE.search(value))
        findings["paths_detected"] += bool(_WINDOWS_PATH.search(value)) + bool(_POSIX_PATH.search(value))
        findings["credentials_detected"] += bool(re.search(r"(?i)(?:sk-|bearer\s+)[A-Za-z0-9_-]{12,}", value))
    return findings


def build_review_payload_sample_v1(
    *,
    threads: Sequence[PrivateForumThread],
    selected_semantic_thread_ids: Sequence[str],
    sample_size: int = 8,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Render a bounded redacted sample solely for preflight privacy inspection."""

    if not 1 <= sample_size <= 24:
        raise ForumDataError("review payload sample_size must be between 1 and 24")
    selected = set(str(item) for item in selected_semantic_thread_ids)
    eligible = [
        thread
        for thread in threads
        if thread.semantic_thread_id in selected and not thread.requires_human_review and len(thread.root_candidates) == 1
    ]
    eligible.sort(key=lambda item: canonical_hash([seed, item.semantic_thread_id]))
    if len(eligible) < sample_size:
        raise ForumDataError("not enough conflict-free selected threads for payload sampling")
    payloads = []
    for thread in eligible[:sample_size]:
        root, root_truncated = redact_for_provider(thread.root_candidates[0].content, max_characters=600)
        root = generalize_local_person_names(root)
        comments = []
        for comment in thread.comments[:MAX_REVIEW_COMMENTS]:
            text, truncated = redact_for_provider(comment.content, max_characters=240)
            comments.append(
                {
                    "text": generalize_local_person_names(text),
                    "floor_bucket": min(6, max(1, comment.floor if comment.floor < 2**31 - 1 else 6)),
                    "truncated": truncated,
                }
            )
        payloads.append(
            {
                "semantic_thread_id": thread.semantic_thread_id,
                "redacted_root": root,
                "root_truncated": root_truncated,
                "representative_redacted_comments": comments,
                "temporal_bucket": thread.parent_date[:7],
                "thread_structure_features": {
                    "comment_count": len(thread.comments),
                    "representative_comment_count": len(comments),
                },
            }
        )
    findings = _privacy_findings(payloads)
    scan = {
        "status": "pass" if not sum(findings.values()) else "fail",
        "source_ids_detected": int(findings["source_ids_detected"]),
        "credentials_detected": int(findings["credentials_detected"]),
        "paths_detected": int(findings["paths_detected"]),
        "contacts_detected": int(findings["contacts_detected"]),
    }
    if scan["status"] != "pass":
        raise ForumDataError(f"redacted active-review sample failed privacy scan: {scan}")
    body = {
        "schema_version": REVIEW_PAYLOAD_SAMPLE_SCHEMA,
        "sample_size": sample_size,
        "payloads": payloads,
        "payload_fingerprints": [canonical_hash(item) for item in payloads],
        "privacy_scan": scan,
        "provider_calls": 0,
        "authorization_present": False,
        "source_ids_included": False,
        "raw_unredacted_text_included": False,
    }
    return _release_hash(body, "sample_sha256")


def evaluate_active_review_gate_v1(
    *,
    covered_cluster_weight: float,
    rare_function_counts: Mapping[str, int],
    monthly_distribution_stable: bool,
    consecutive_macro_share_deltas: Sequence[float],
    reviewed_thread_count: int,
) -> dict[str, Any]:
    coverage = float(covered_cluster_weight)
    if not 0 <= coverage <= 1:
        raise ForumDataError("covered_cluster_weight must be in [0, 1]")
    if not 0 <= reviewed_thread_count <= MAX_REVIEW_THREADS:
        raise ForumDataError("reviewed_thread_count is outside the frozen cap")
    counts = {str(key): int(value) for key, value in rare_function_counts.items()}
    if any(value < 0 for value in counts.values()):
        raise ForumDataError("rare-function counts must be nonnegative")
    deltas = [float(value) for value in consecutive_macro_share_deltas]
    two_round_stable = len(deltas) >= 2 and all(value < MAX_MACRO_SHARE_DELTA for value in deltas[-2:])
    checks = {
        "cluster_coverage": coverage >= MIN_CLUSTER_COVERAGE,
        "rare_reply_functions": bool(counts) and min(counts.values()) >= MIN_RARE_FUNCTION_REVIEWS,
        "monthly_distribution_stable": monthly_distribution_stable is True,
        "two_consecutive_macro_deltas": two_round_stable,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "observed": {
            "covered_cluster_weight": coverage,
            "rare_function_counts": dict(sorted(counts.items())),
            "monthly_distribution_stable": monthly_distribution_stable is True,
            "consecutive_macro_share_deltas": deltas,
            "reviewed_thread_count": reviewed_thread_count,
        },
        "thresholds": {
            "minimum_cluster_coverage": MIN_CLUSTER_COVERAGE,
            "minimum_rare_function_reviews": MIN_RARE_FUNCTION_REVIEWS,
            "maximum_macro_share_delta_strict": MAX_MACRO_SHARE_DELTA,
            "maximum_review_threads": MAX_REVIEW_THREADS,
        },
        "population_compilation_allowed": passed,
        "cap_reached_without_pass": reviewed_thread_count == MAX_REVIEW_THREADS and not passed,
        "gate_sha256": canonical_hash(checks),
    }


def build_active_review_preflight_v1(
    *,
    corpus_snapshot_sha256: str,
    index_sha256: str | None,
    selection_sha256: str,
    endpoint: str = "https://yunwu.ai/v1",
    model: str = "gpt-5.4-mini",
    selected_thread_count: int,
    sample_payload_fingerprints: Sequence[str] = (),
    privacy_scan: Mapping[str, Any] | None = None,
    semantic_selection_ready: bool = True,
    embedding_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a read-only, secret-free preflight.  It never authorizes transport."""

    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ForumDataError("active-review endpoint must be credential-free HTTPS")
    if not 1 <= selected_thread_count <= MAX_REVIEW_THREADS:
        raise ForumDataError("selected_thread_count must be between 1 and 3000")
    if semantic_selection_ready and index_sha256 is None:
        raise ForumDataError("an immutable completed index is required for a semantic review preflight")
    safe_index_sha256 = (
        _require_sha256(index_sha256, "index_sha256") if index_sha256 is not None else None
    )
    safe_plan_sha256 = (
        _require_sha256(embedding_plan_sha256, "embedding_plan_sha256")
        if embedding_plan_sha256 is not None
        else None
    )
    if not semantic_selection_ready and safe_plan_sha256 is None:
        raise ForumDataError("a dry-run preflight must bind its embedding shard plan")
    fingerprints = [_require_sha256(item, "sample payload fingerprint") for item in sample_payload_fingerprints]
    scan = {
        "status": "pending_payload_scan",
        "source_ids_detected": 0,
        "credentials_detected": 0,
        "paths_detected": 0,
        "contacts_detected": 0,
    }
    if privacy_scan is not None:
        if not isinstance(privacy_scan, Mapping):
            raise ForumDataError("privacy_scan must be an object")
        scan = {
            "status": str(privacy_scan.get("status") or ""),
            "source_ids_detected": int(privacy_scan.get("source_ids_detected", -1)),
            "credentials_detected": int(privacy_scan.get("credentials_detected", -1)),
            "paths_detected": int(privacy_scan.get("paths_detected", -1)),
            "contacts_detected": int(privacy_scan.get("contacts_detected", -1)),
        }
        if scan["status"] != "pass" or any(scan[key] != 0 for key in scan if key != "status"):
            raise ForumDataError("active-review payload privacy scan must pass with zero findings")
    body = {
        "schema_version": ACTIVE_REVIEW_PREFLIGHT_SCHEMA,
        "inputs": {
            "corpus_snapshot_sha256": _require_sha256(corpus_snapshot_sha256, "corpus_snapshot_sha256"),
            "index_sha256": safe_index_sha256,
            "embedding_plan_sha256": safe_plan_sha256,
            "selection_sha256": _require_sha256(selection_sha256, "selection_sha256"),
        },
        "provider": {
            "endpoint": endpoint.rstrip("/"),
            "model": _require_nonempty(model, "model", maximum=120),
            "primary_call_limit": MAX_REVIEW_THREADS,
            "format_repair_limit_per_slot": 1,
            "network_retry_limit": 0,
            "request_hard_limit": MAX_REVIEW_THREADS * 2,
            "token_hard_limit": 20_000_000,
            "max_concurrency": 24,
        },
        "selection": {
            "selected_thread_count": selected_thread_count,
            "max_comments_per_thread": MAX_REVIEW_COMMENTS,
            "sample_payload_fingerprints": fingerprints,
            "semantic_selection_ready": semantic_selection_ready is True,
        },
        "sent_fields": [
            "semantic_thread_id",
            "redacted_root",
            "up_to_six_representative_redacted_comments",
            "temporal_bucket",
            "thread_structure_features",
            "json_output_contract",
        ],
        "excluded_fields": [
            "raw_unredacted_text",
            "source_id",
            "real_identity",
            "source_path",
            "credentials",
            "hidden_population_truth",
        ],
        "privacy_scan": scan,
        "authorization_required": True,
        "authorization_present": False,
        "provider_calls_executed": 0,
        "preflight_ready_for_authorization": (
            semantic_selection_ready is True and scan["status"] == "pass"
        ),
    }
    return _release_hash(body, "preflight_sha256")


def _episode_field(episode: Mapping[str, Any], field: str) -> Any:
    if field not in episode:
        raise ForumDataError(f"reviewed episode missing {field}")
    return episode[field]


def _reviewed_evidence_id(semantic_episode_id: Any) -> str:
    semantic_id = _require_nonempty(
        semantic_episode_id, "semantic_episode_id", maximum=120
    )
    return "evidence-" + canonical_hash(["forum-evidence", semantic_id])[:20]


def compile_micro_role_release_v1(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate reviewed micro-role assignments and compile aggregate role descriptors."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    semantic_ids = set()
    for episode in episodes:
        semantic_id = _require_nonempty(_episode_field(episode, "semantic_episode_id"), "semantic_episode_id", maximum=120)
        if semantic_id in semantic_ids:
            raise ForumDataError("semantic episode ids must be unique")
        semantic_ids.add(semantic_id)
        macro = _safe_profile_text(_episode_field(episode, "primary_macro_role"), "primary_macro_role", maximum=80)
        micro = _safe_profile_text(_episode_field(episode, "micro_interaction_role"), "micro_interaction_role", maximum=100)
        groups[(macro, micro)].append(episode)
    by_macro = Counter(macro for macro, _micro in groups)
    if len(by_macro) != MACRO_ROLE_COUNT:
        raise ForumDataError("micro-role release requires exactly 17 macro roles")
    if any(not MIN_ROLES_PER_MACRO <= count <= MAX_ROLES_PER_MACRO for count in by_macro.values()):
        raise ForumDataError("each macro role must contain 4-10 micro roles")
    if not MIN_MICRO_ROLES <= len(groups) <= MAX_MICRO_ROLES:
        raise ForumDataError("total micro-role count must be between 80 and 120")
    if any(len(rows) < MIN_EPISODES_PER_ROLE for rows in groups.values()):
        raise ForumDataError("each micro role must contain at least 30 reviewed episodes")
    roles = []
    for (macro, micro), rows in sorted(groups.items()):
        topic_counts = Counter(topic for row in rows for topic in row.get("topic_portfolio", []))
        need_counts = Counter(need for row in rows for need in row.get("need_portfolio", []))
        function_counts = Counter(str(row.get("reply_function") or "other") for row in rows)
        evidence_ids = sorted(str(row["semantic_episode_id"]) for row in rows)
        roles.append(
            {
                "micro_role_id": "micro-" + canonical_hash([macro, micro])[:20],
                "primary_macro_role": macro,
                "micro_interaction_role": micro,
                "reviewed_episode_count": len(rows),
                "topic_portfolio": [key for key, _ in topic_counts.most_common(6)],
                "need_portfolio": [key for key, _ in need_counts.most_common(6)],
                "reply_function_distribution": {
                    key: count / len(rows) for key, count in sorted(function_counts.items())
                },
                "reviewed_evidence_pool": [
                    _reviewed_evidence_id(semantic_id)
                    for semantic_id in evidence_ids[:12]
                ],
            }
        )
    body = {
        "schema_version": MICRO_ROLE_SCHEMA,
        "macro_role_count": len(by_macro),
        "micro_role_count": len(roles),
        "role_counts_by_macro": dict(sorted(by_macro.items())),
        "roles": roles,
        "source_reviewed_episode_count": len(episodes),
        "source_ids_persisted": False,
        "raw_text_persisted": False,
    }
    return _release_hash(body, "micro_role_release_sha256")


def _bundle_payload(episode: Mapping[str, Any], role: Mapping[str, Any]) -> dict[str, Any]:
    style = episode.get("interaction_style")
    tendencies = episode.get("forum_tendencies")
    trust = episode.get("trust_style")
    if not isinstance(style, Mapping) or not isinstance(tendencies, Mapping) or not isinstance(trust, Mapping):
        raise ForumDataError("interaction_style, forum_tendencies and trust_style must be objects")
    return {
        "primary_macro_role": role["primary_macro_role"],
        "secondary_macro_role": _safe_profile_text(episode.get("secondary_macro_role") or role["primary_macro_role"], "secondary_macro_role", maximum=80),
        "micro_role_id": role["micro_role_id"],
        "micro_interaction_role": role["micro_interaction_role"],
        "episode_focus": _safe_profile_text(_episode_field(episode, "episode_focus"), "episode_focus", maximum=180),
        "topic_portfolio": list(_normalized_string_list(_episode_field(episode, "topic_portfolio"), "topic_portfolio", maximum=6)),
        "need_portfolio": list(_normalized_string_list(_episode_field(episode, "need_portfolio"), "need_portfolio", maximum=6)),
        "forum_tendencies": _probability_map(tendencies, "forum_tendencies"),
        "trust_style": _probability_map(trust, "trust_style"),
        "uncertainty_handling": _safe_profile_text(_episode_field(episode, "uncertainty_handling"), "uncertainty_handling", maximum=100),
        "language_form": _safe_profile_text(style.get("language_form"), "language_form", maximum=60),
        "reply_function": _safe_profile_text(style.get("reply_function"), "reply_function", maximum=80),
        "information_density": _safe_profile_text(style.get("information_density"), "information_density", maximum=60),
        "activity_rhythm": _safe_profile_text(_episode_field(episode, "activity_rhythm"), "activity_rhythm", maximum=80),
        "attention_budget": _probability_map(
            {"value": _episode_field(episode, "attention_budget")},
            "attention_budget",
        )["value"],
        "long_term_memory_seed": _safe_profile_text(_episode_field(episode, "long_term_memory_seed"), "long_term_memory_seed", maximum=180),
        "behavior_boundaries": list(_normalized_string_list(_episode_field(episode, "behavior_boundaries"), "behavior_boundaries", maximum=6)),
        "safety_gates": list(_normalized_string_list(_episode_field(episode, "safety_gates"), "safety_gates", maximum=6)),
    }


def compile_forum_profile_release_v1(
    *,
    episodes: Sequence[Mapping[str, Any]],
    micro_role_release: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile 300+ genuinely distinct prompt bundles from reviewed episodes."""

    if micro_role_release.get("schema_version") != MICRO_ROLE_SCHEMA:
        raise ForumDataError("micro-role release schema mismatch")
    if micro_role_release.get("micro_role_release_sha256") != canonical_hash(
        {key: value for key, value in micro_role_release.items() if key != "micro_role_release_sha256"}
    ):
        raise ForumDataError("micro-role release hash mismatch")
    roles = {
        (str(item["primary_macro_role"]), str(item["micro_interaction_role"])): item
        for item in micro_role_release.get("roles", [])
    }
    candidates: dict[str, dict[str, Any]] = {}
    evidence_by_bundle: dict[str, list[str]] = defaultdict(list)
    match_by_bundle: Counter[str] = Counter()
    count_by_bundle: Counter[str] = Counter()
    for episode in episodes:
        key = (str(episode.get("primary_macro_role")), str(episode.get("micro_interaction_role")))
        if key not in roles:
            raise ForumDataError("episode references a micro role outside the frozen release")
        payload = _bundle_payload(episode, roles[key])
        signature = canonical_hash(payload)
        candidates.setdefault(signature, payload)
        count_by_bundle[signature] += 1
        if episode.get("reviewed_evidence_match") is True:
            match_by_bundle[signature] += 1
            evidence_by_bundle[signature].append(
                _reviewed_evidence_id(episode["semantic_episode_id"])
            )
    if len(candidates) < MIN_PROFILE_BUNDLES:
        raise ForumDataError("profile release requires at least 300 distinct prompt bundles")
    role_evidence = {item["micro_role_id"]: item["reviewed_evidence_pool"] for item in roles.values()}
    bundles = []
    for signature, payload in sorted(candidates.items()):
        evidence = list(dict.fromkeys(evidence_by_bundle[signature] + role_evidence[payload["micro_role_id"]]))[:5]
        if len(evidence) < 3:
            raise ForumDataError("every profile bundle requires 3-5 reviewed evidence references")
        match_rate = match_by_bundle[signature] / count_by_bundle[signature]
        bundles.append(
            {
                "profile_bundle_id": "bundle-" + signature[:20],
                **payload,
                "reviewed_evidence_refs": evidence,
                "reviewed_evidence_match_rate": round(match_rate, 6),
                "profile_prompt_sha256": signature,
            }
        )
    body = {
        "schema_version": PROFILE_RELEASE_SCHEMA,
        "micro_role_release_sha256": micro_role_release["micro_role_release_sha256"],
        "profile_bundle_count": len(bundles),
        "bundles": bundles,
        "privacy": {
            "source_ids_persisted": False,
            "raw_text_persisted": False,
            "inferred_demographics": False,
            "real_longitudinal_users_inferred": False,
        },
    }
    return _release_hash(body, "profile_release_sha256")


def _effective_count(counts: Iterable[int]) -> float:
    values = [value for value in counts if value > 0]
    total = sum(values)
    if not total:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return math.exp(entropy)


def compile_forum_agent_population_v2(
    *,
    profile_release: Mapping[str, Any],
    micro_role_release: Mapping[str, Any],
    scenario_ids: Sequence[str],
    seed: int = 20260803,
    hero_agent_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile exactly 1,000 callable profiles and enforce the heterogeneity gates."""

    if profile_release.get("schema_version") != PROFILE_RELEASE_SCHEMA:
        raise ForumDataError("profile release schema mismatch")
    expected_hash = canonical_hash(
        {key: value for key, value in profile_release.items() if key != "profile_release_sha256"}
    )
    if profile_release.get("profile_release_sha256") != expected_hash:
        raise ForumDataError("profile release hash mismatch")
    bundles = list(profile_release.get("bundles", []))
    if len(bundles) < MIN_PROFILE_BUNDLES:
        raise ForumDataError("population requires at least 300 profile bundles")
    scenarios = tuple(dict.fromkeys(_require_nonempty(item, "scenario_id", maximum=100) for item in scenario_ids))
    if not scenarios:
        raise ForumDataError("at least one scenario is required")
    ordered = sorted(bundles, key=lambda bundle: canonical_hash([seed, bundle["profile_bundle_id"]]))
    agents = []
    bundle_counts: Counter[str] = Counter()
    for index in range(FORUM_AGENT_COUNT):
        bundle = ordered[index % len(ordered)]
        bundle_id = str(bundle["profile_bundle_id"])
        bundle_counts[bundle_id] += 1
        evidence_match = float(bundle["reviewed_evidence_match_rate"]) >= MIN_EVIDENCE_MATCH_RATE
        agent_id = "forum-agent-" + canonical_hash(
            [profile_release["profile_release_sha256"], seed, index]
        )[:20]
        agents.append(
            {
                "agent_id": agent_id,
                "synthetic_display_id": f"FT-{index + 1:04d}",
                "profile_bundle_id": bundle_id,
                "profile_prompt_sha256": bundle["profile_prompt_sha256"],
                "primary_macro_role": bundle["primary_macro_role"],
                "secondary_macro_role": bundle["secondary_macro_role"],
                "micro_role_id": bundle["micro_role_id"],
                "micro_interaction_role": bundle["micro_interaction_role"],
                "episode_focus": bundle["episode_focus"],
                "topic_portfolio": bundle["topic_portfolio"],
                "need_portfolio": bundle["need_portfolio"],
                "forum_tendencies": bundle["forum_tendencies"],
                "trust_style": bundle["trust_style"],
                "uncertainty_handling": bundle["uncertainty_handling"],
                "language_style": {
                    "language_form": bundle["language_form"],
                    "reply_function": bundle["reply_function"],
                    "information_density": bundle["information_density"],
                },
                "activity_rhythm": bundle["activity_rhythm"],
                "attention_budget": bundle["attention_budget"],
                "long_term_memory_seed": bundle["long_term_memory_seed"],
                "behavior_boundaries": bundle["behavior_boundaries"],
                "safety_gates": bundle["safety_gates"],
                "reviewed_evidence_refs": bundle["reviewed_evidence_refs"],
                "evidence_binding": "reviewed_evidence_match" if evidence_match else "reviewed_profile_fallback",
                "scenario_weights": {scenario: 1.0 / FORUM_AGENT_COUNT for scenario in scenarios},
                "provider_turn_contract": "campus-pulse-forum-agent-turn-v1",
                "llm_memory": [],
                "private_state_history": [],
                "message_inbox": [],
                "particle_count": 10,
            }
        )
    clone_share = max(bundle_counts.values()) / FORUM_AGENT_COUNT
    effective = _effective_count(bundle_counts.values())
    match_count = sum(agent["evidence_binding"] == "reviewed_evidence_match" for agent in agents)
    match_rate = match_count / FORUM_AGENT_COUNT
    fallback_rate = 1.0 - match_rate
    hero_set = set(hero_agent_ids)
    unknown_hero = hero_set - {agent["agent_id"] for agent in agents}
    if unknown_hero:
        raise ForumDataError("hero_agent_ids include agents outside this population")
    hero_fallback = [
        agent["agent_id"] for agent in agents
        if agent["agent_id"] in hero_set and agent["evidence_binding"] != "reviewed_evidence_match"
    ]
    checks = {
        "agent_count": len(agents) == FORUM_AGENT_COUNT,
        "bundle_count": len(bundle_counts) >= MIN_PROFILE_BUNDLES,
        "effective_bundle_count": effective >= MIN_EFFECTIVE_BUNDLES,
        "prompt_clone_share": clone_share <= MAX_PROMPT_CLONE_SHARE,
        "reviewed_evidence_match_rate": match_rate >= MIN_EVIDENCE_MATCH_RATE,
        "fallback_rate": fallback_rate <= MAX_FALLBACK_RATE,
        "hero_profiles_have_no_fallback": not hero_fallback,
    }
    if not all(checks.values()):
        raise ForumDataError(f"ForumTwin profile gate failed: {checks}")
    body = {
        "schema_version": POPULATION_SCHEMA,
        "population_size": FORUM_AGENT_COUNT,
        "particles_per_agent": 10,
        "profile_release_sha256": profile_release["profile_release_sha256"],
        "micro_role_release_sha256": micro_role_release.get("micro_role_release_sha256"),
        "seed": seed,
        "scenario_ids": list(scenarios),
        "agents": agents,
        "profile_gate": {
            "status": "pass",
            "checks": checks,
            "distinct_profile_bundles_used": len(bundle_counts),
            "effective_profile_bundle_count": round(effective, 6),
            "maximum_identical_prompt_profile_share": round(clone_share, 6),
            "reviewed_evidence_match_rate": round(match_rate, 6),
            "fallback_rate": round(fallback_rate, 6),
        },
        "privacy": {
            "source_ids_persisted": False,
            "raw_text_persisted": False,
            "inferred_gender_department_age_or_identity": False,
            "real_longitudinal_relationship_inferred": False,
        },
    }
    return _release_hash(body, "population_sha256")


_PARTICIPATION_PHRASES = {
    "question": "遇到信息缺口时会直接追问关键细节",
    "information_request": "会向知道情况的人询问具体信息",
    "request": "会直接说明自己需要什么以及希望得到怎样的回应",
    "help_request": "遇到实际困难时会讲清处境并寻求可执行的帮助",
    "complaint": "认为规则或服务不合理时会明确表达不满",
    "emotional_expression": "受到影响时会先表达当下的真实感受",
    "emotional_vent": "受到影响时会比较直接地吐槽或宣泄情绪",
    "information_share": "掌握相关信息时会补充自己知道的内容",
    "discussion": "会接着别人的观点继续讨论，而不是另起话题",
    "experience_share_or_ask": "会结合自己的经历分享或向相似经历者求证",
    "correction": "发现说法有误时会指出问题并补充可核验信息",
    "risk_report": "察觉风险时会提醒他人或向合适渠道报告",
    "other": "会先看帖子是否与自己有关，再决定参与还是离开",
}

_LANGUAGE_FORM_PHRASES = {
    "colloquial": "说话偏口语，像日常校园论坛交流",
    "formal": "表达相对完整，会把理由和条件说清楚",
    "concise": "通常用较短的话直奔重点",
    "narrative": "更习惯按事情经过来讲述",
    "mixed": "会随对象和场合在简短口语与完整说明之间切换",
}

_INFORMATION_DENSITY_PHRASES = {
    "low": "通常只说当前最在意的一点",
    "medium": "通常会给出足以让别人接话的背景",
    "high": "往往会补充较多条件、依据或过程",
    "mixed": "信息多少取决于事情是否直接影响自己",
}


def _reviewed_behavior_example(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one reviewed semantic card into model-readable past behavior.

    The result deliberately contains neither the semantic episode identifier nor
    the evidence fingerprint. Those identifiers remain in the surrounding
    release for audit joins, while the Provider receives only reviewed semantic
    descriptions.
    """

    style = episode.get("interaction_style")
    style = style if isinstance(style, Mapping) else {}
    speech_act = str(
        episode.get("speech_act") or episode.get("reply_function") or "other"
    ).strip()
    reply_function = str(episode.get("reply_function") or speech_act).strip()
    language_form = str(style.get("language_form") or "mixed").strip()
    density = str(style.get("information_density") or "mixed").strip()
    uncertainty = _safe_profile_text(
        episode.get("uncertainty_handling") or "根据当时看到的信息作出判断",
        "uncertainty_handling",
        maximum=120,
    )
    participation = _PARTICIPATION_PHRASES.get(
        speech_act,
        _PARTICIPATION_PHRASES.get(
            reply_function, _PARTICIPATION_PHRASES["other"]
        ),
    )
    return {
        "past_situation": _safe_profile_text(
            _episode_field(episode, "episode_focus"),
            "episode_focus",
            maximum=180,
        ),
        "usual_participation": participation,
        "communication_style": "；".join(
            (
                _LANGUAGE_FORM_PHRASES.get(
                    language_form, "表达方式会随具体帖子和交流对象变化"
                ),
                _INFORMATION_DENSITY_PHRASES.get(
                    density, "会根据事情是否与自己相关决定说多少"
                ),
            )
        ),
        "information_posture": uncertainty,
        "recurring_topics": list(
            _normalized_string_list(
                _episode_field(episode, "topic_portfolio"),
                "topic_portfolio",
                maximum=6,
            )
        ),
        "recurring_needs": list(
            _normalized_string_list(
                _episode_field(episode, "need_portfolio"),
                "need_portfolio",
                maximum=6,
            )
        ),
    }


def compile_reviewed_behavior_portfolio_v1(
    *,
    episodes: Sequence[Mapping[str, Any]],
    population_release: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind 3-5 readable reviewed behavior examples to every profile bundle.

    This additive release does not alter the frozen population-v2 hash. It is
    a presentation adapter: internal evidence fingerprints prove where each
    example came from, while the model-visible profile receives only the
    reviewed semantic description.
    """

    if population_release.get("schema_version") != POPULATION_SCHEMA:
        raise ForumDataError("reviewed behavior portfolio requires population-v2")
    expected_hash = canonical_hash(
        {
            key: value
            for key, value in population_release.items()
            if key != "population_sha256"
        }
    )
    if population_release.get("population_sha256") != expected_hash:
        raise ForumDataError("population-v2 hash mismatch")

    episode_by_evidence: dict[str, Mapping[str, Any]] = {}
    reviewed_by_role: dict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    reviewed_by_macro: dict[str, list[tuple[str, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for episode in episodes:
        evidence_id = _reviewed_evidence_id(
            _episode_field(episode, "semantic_episode_id")
        )
        if evidence_id in episode_by_evidence:
            raise ForumDataError("reviewed semantic episode ids must be unique")
        episode_by_evidence[evidence_id] = episode
        macro = str(episode.get("primary_macro_role") or "").strip()
        micro = re.sub(
            r"_\d{2}$", "", str(episode.get("micro_interaction_role") or "").strip()
        )
        reviewed_by_role[(macro, micro)].append((evidence_id, episode))
        reviewed_by_macro[macro].append((evidence_id, episode))

    refs_by_bundle: dict[str, tuple[str, ...]] = {}
    role_by_bundle: dict[str, tuple[str, str]] = {}
    for agent in population_release.get("agents", []):
        if not isinstance(agent, Mapping):
            raise ForumDataError("population-v2 contains a malformed Agent")
        bundle_id = _require_nonempty(
            agent.get("profile_bundle_id"), "profile_bundle_id", maximum=100
        )
        refs = tuple(
            str(value)
            for value in agent.get("reviewed_evidence_refs", [])
            if isinstance(value, str) and value.startswith("evidence-")
        )[:5]
        if not 3 <= len(refs) <= 5:
            raise ForumDataError("every profile bundle requires 3-5 evidence refs")
        existing = refs_by_bundle.setdefault(bundle_id, refs)
        if existing != refs:
            raise ForumDataError(
                "one profile bundle cannot have conflicting evidence refs"
            )
        role_key = (
            str(agent.get("primary_macro_role") or "").strip(),
            re.sub(
                r"_\d{2}$",
                "",
                str(agent.get("micro_interaction_role") or "").strip(),
            ),
        )
        existing_role = role_by_bundle.setdefault(bundle_id, role_key)
        if existing_role != role_key:
            raise ForumDataError(
                "one profile bundle cannot map to conflicting reviewed roles"
            )

    portfolios = []
    unresolved_exact_refs = 0
    same_role_fallback_count = 0
    same_macro_fallback_count = 0
    for bundle_id, refs in sorted(refs_by_bundle.items()):
        selected: list[tuple[str, Mapping[str, Any], str]] = []
        selected_ids: set[str] = set()
        for evidence_id in refs:
            episode = episode_by_evidence.get(evidence_id)
            if episode is None:
                unresolved_exact_refs += 1
                continue
            selected.append((evidence_id, episode, "exact_profile_evidence"))
            selected_ids.add(evidence_id)
        if len(selected) < 3:
            candidates = sorted(
                reviewed_by_role.get(role_by_bundle[bundle_id], []),
                key=lambda row: canonical_hash(
                    ["reviewed-role-example", bundle_id, row[0]]
                ),
            )
            for evidence_id, episode in candidates:
                if evidence_id in selected_ids:
                    continue
                selected.append(
                    (evidence_id, episode, "same_micro_role_reviewed_example")
                )
                selected_ids.add(evidence_id)
                same_role_fallback_count += 1
                if len(selected) >= 3:
                    break
        if len(selected) < 3:
            candidates = sorted(
                reviewed_by_macro.get(role_by_bundle[bundle_id][0], []),
                key=lambda row: canonical_hash(
                    ["reviewed-macro-example", bundle_id, row[0]]
                ),
            )
            for evidence_id, episode in candidates:
                if evidence_id in selected_ids:
                    continue
                selected.append(
                    (evidence_id, episode, "same_macro_role_reviewed_example")
                )
                selected_ids.add(evidence_id)
                same_macro_fallback_count += 1
                if len(selected) >= 3:
                    break
        if len(selected) < 3:
            raise ForumDataError(
                f"profile bundle {bundle_id} has fewer than three reviewed role examples"
            )
        selected = selected[:5]
        portfolios.append(
            {
                "profile_bundle_id": bundle_id,
                "reviewed_evidence_refs": list(refs),
                "reviewed_example_bindings": [
                    {"evidence_ref": evidence_id, "basis": basis}
                    for evidence_id, _episode, basis in selected
                ],
                "reviewed_behavior_examples": [
                    _reviewed_behavior_example(episode)
                    for _evidence_id, episode, _basis in selected
                ],
            }
        )
    body = {
        "schema_version": REVIEWED_BEHAVIOR_PORTFOLIO_SCHEMA,
        "source_population_sha256": population_release["population_sha256"],
        "source_reviewed_episode_count": len(episodes),
        "profile_bundle_count": len(portfolios),
        "binding_summary": {
            "unresolved_exact_profile_evidence_refs": unresolved_exact_refs,
            "same_micro_role_reviewed_example_count": same_role_fallback_count,
            "same_macro_role_reviewed_example_count": same_macro_fallback_count,
            "minimum_examples_per_bundle": 3,
        },
        "portfolios": portfolios,
        "privacy": {
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "semantic_episode_ids_persisted": False,
            "model_visible_evidence_fingerprints": False,
        },
    }
    return _release_hash(body, "behavior_portfolio_sha256")
