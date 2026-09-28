"""Offline, resumable BGE embedding execution for ForumTwin shard plans.

The executor never imports a model runtime, downloads weights, or performs
network/Provider calls.  A caller in the separately pinned data-prep
environment injects both a private text loader and an already initialized
encoder.  Only float16 vector bytes and path/text-free manifests are written.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .forum_data import (
    CORPUS_INDEX_SCHEMA,
    EMBEDDING_DIMENSION,
    EMBEDDING_DTYPE,
    EMBEDDING_MODEL,
    EMBEDDING_SHARD_PLAN_SCHEMA,
    EmbeddingShard,
    ForumDataError,
    build_embedding_index_release_v1,
    build_embedding_resume_manifest_v1,
)
from .mass_population import canonical_hash


LOCAL_BGE_BINDING_SCHEMA = "campus-pulse-local-bge-execution-binding-v1"
LOCAL_BGE_SHARD_SCHEMA = "campus-pulse-local-bge-shard-artifact-v1"
LOCAL_BGE_MERGE_SCHEMA = "campus-pulse-local-bge-merge-artifact-v1"
LOCAL_BGE_EXECUTION_SCHEMA = "campus-pulse-local-bge-execution-release-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ID_RE = re.compile(r"^embedding-shard-[0-9]{5}$")
_VECTOR_ITEM_BYTES = np.dtype("<f2").itemsize


class ForumEmbeddingError(ForumDataError):
    """Raised when local embedding work cannot be resumed safely."""


@dataclass(frozen=True)
class LocalBGEEncoderIdentity:
    """Exact immutable identity of the already available local encoder."""

    model: str
    model_revision: str
    model_weights_sha256: str
    tokenizer_sha256: str
    model_config_sha256: str
    dimension: int = EMBEDDING_DIMENSION
    dtype: str = EMBEDDING_DTYPE

    def validated_view(self) -> dict[str, Any]:
        if self.model != EMBEDDING_MODEL:
            raise ForumEmbeddingError(
                f"local encoder model must be exactly {EMBEDDING_MODEL}"
            )
        revision = " ".join(str(self.model_revision or "").split())
        if not revision or len(revision) > 160:
            raise ForumEmbeddingError("local encoder revision is invalid")
        for value, label in (
            (self.model_weights_sha256, "model weights"),
            (self.tokenizer_sha256, "tokenizer"),
            (self.model_config_sha256, "model config"),
        ):
            if _SHA256_RE.fullmatch(str(value or "")) is None:
                raise ForumEmbeddingError(f"local encoder {label} hash is invalid")
        if self.dimension != EMBEDDING_DIMENSION or self.dtype != EMBEDDING_DTYPE:
            raise ForumEmbeddingError("local encoder must emit float16 512d vectors")
        return {
            "model": self.model,
            "model_revision": revision,
            "model_weights_sha256": self.model_weights_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_config_sha256": self.model_config_sha256,
            "dimension": self.dimension,
            "dtype": self.dtype,
        }


class LocalEmbeddingEncoder(Protocol):
    """Injected local encoder; implementations own model loading and batching."""

    identity: LocalBGEEncoderIdentity

    def encode(self, texts: Sequence[str]) -> Any: ...


class PrivateSemanticTextLoader(Protocol):
    """Injected private lookup.  Returned text is never persisted by this module."""

    def load_texts(self, semantic_row_ids: Sequence[str]) -> Mapping[str, str]: ...


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForumEmbeddingError(f"{label} is not readable canonical JSON") from exc
    if not isinstance(value, Mapping):
        raise ForumEmbeddingError(f"{label} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                byte_size += len(block)
    except OSError as exc:
        raise ForumEmbeddingError("local vector artifact cannot be read") from exc
    return digest.hexdigest(), byte_size


def _hash_bound(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = canonical_hash(result)
    return result


def _validate_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if plan.get("schema_version") != EMBEDDING_SHARD_PLAN_SCHEMA:
        raise ForumEmbeddingError("embedding shard plan schema mismatch")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != canonical_hash(body):
        raise ForumEmbeddingError("embedding shard plan hash mismatch")
    embedding = plan.get("embedding")
    rows = plan.get("rows")
    shards = plan.get("shards")
    privacy = plan.get("privacy")
    if not isinstance(embedding, Mapping) or not isinstance(rows, Mapping):
        raise ForumEmbeddingError("embedding plan identity or row frame is invalid")
    if not isinstance(shards, list) or not shards:
        raise ForumEmbeddingError("embedding plan must contain explicit shards")
    if not isinstance(privacy, Mapping) or any(
        privacy.get(key) is not False
        for key in ("raw_text_persisted", "source_ids_persisted", "source_paths_persisted")
    ):
        raise ForumEmbeddingError("embedding plan privacy boundary is invalid")
    if (
        embedding.get("model") != EMBEDDING_MODEL
        or embedding.get("dimension") != EMBEDDING_DIMENSION
        or embedding.get("dtype") != EMBEDDING_DTYPE
        or embedding.get("model_download_performed") is not False
        or embedding.get("embedding_inference_performed") is not False
    ):
        raise ForumEmbeddingError("embedding plan is not the frozen offline BGE layout")
    identity = LocalBGEEncoderIdentity(
        model=str(embedding.get("model") or ""),
        model_revision=str(embedding.get("model_revision") or ""),
        model_weights_sha256=str(embedding.get("model_weights_sha256") or ""),
        tokenizer_sha256=str(embedding.get("tokenizer_sha256") or ""),
        model_config_sha256=str(embedding.get("model_config_sha256") or ""),
        dimension=int(embedding.get("dimension", -1)),
        dtype=str(embedding.get("dtype") or ""),
    ).validated_view()
    ordered_row_ids: list[str] = []
    normalized_shards: list[dict[str, Any]] = []
    expected_start = 0
    for raw in shards:
        if not isinstance(raw, Mapping):
            raise ForumEmbeddingError("embedding shard plan row is invalid")
        shard_id = str(raw.get("shard_id") or "")
        if _SHARD_ID_RE.fullmatch(shard_id) is None:
            raise ForumEmbeddingError("embedding shard id is unsafe")
        row_ids = raw.get("semantic_row_ids")
        if not isinstance(row_ids, list) or not row_ids:
            raise ForumEmbeddingError("embedding shard must contain explicit row ids")
        row_ids = [str(value) for value in row_ids]
        start = int(raw.get("start_row", -1))
        end = int(raw.get("end_row_exclusive", -1))
        count = int(raw.get("row_count", -1))
        if (
            start != expected_start
            or end != start + len(row_ids)
            or count != len(row_ids)
            or raw.get("row_id_sha256") != canonical_hash(row_ids)
            or raw.get("state") != "pending"
        ):
            raise ForumEmbeddingError("embedding shard layout drifted from its plan")
        ordered_row_ids.extend(row_ids)
        normalized_shards.append(dict(raw))
        expected_start = end
    if len(ordered_row_ids) != len(set(ordered_row_ids)):
        raise ForumEmbeddingError("embedding shard plan contains duplicate row ids")
    if (
        int(rows.get("count", -1)) != len(ordered_row_ids)
        or rows.get("semantic_row_order_sha256") != canonical_hash(ordered_row_ids)
    ):
        raise ForumEmbeddingError("embedding plan row coverage is incomplete")
    return identity, normalized_shards


def load_embedding_shard_plan_v1(path: str | Path) -> dict[str, Any]:
    """Read and fully validate an explicit plan produced by data preparation."""

    plan = dict(_read_json(Path(path), "embedding shard plan"))
    _validate_plan(plan)
    return plan


def _binding(plan: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": LOCAL_BGE_BINDING_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "corpus_snapshot_sha256": plan["corpus_snapshot_sha256"],
        "embedding": dict(identity),
        "execution": {
            "network_allowed": False,
            "model_download_allowed": False,
            "provider_calls": 0,
            "encoder_injected": True,
        },
        "privacy": {
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "source_paths_persisted": False,
        },
    }
    return _hash_bound(body, "binding_sha256")


def _ensure_binding(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        if dict(_read_json(path, "embedding execution binding")) != dict(expected):
            raise ForumEmbeddingError(
                "embedding output directory belongs to a different immutable plan"
            )
        return
    _atomic_json(path, expected)


def _shard_body(
    *,
    plan: Mapping[str, Any],
    planned_shard: Mapping[str, Any],
    payload_sha256: str,
    byte_size: int,
) -> dict[str, Any]:
    expected_bytes = (
        int(planned_shard["row_count"]) * EMBEDDING_DIMENSION * _VECTOR_ITEM_BYTES
    )
    if byte_size != expected_bytes or _SHA256_RE.fullmatch(payload_sha256) is None:
        raise ForumEmbeddingError("embedding shard vector payload layout is invalid")
    body = {
        "schema_version": LOCAL_BGE_SHARD_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "shard_id": planned_shard["shard_id"],
        "vector_count": int(planned_shard["row_count"]),
        "dimension": EMBEDDING_DIMENSION,
        "dtype": EMBEDDING_DTYPE,
        "row_id_sha256": planned_shard["row_id_sha256"],
        "payload_sha256": payload_sha256,
        "byte_size": byte_size,
        "vector_file": f"{planned_shard['shard_id']}.f16",
        "raw_text_persisted": False,
        "provider_calls": 0,
    }
    return _hash_bound(body, "shard_artifact_sha256")


def _existing_or_recovered_shard(
    *,
    plan: Mapping[str, Any],
    planned_shard: Mapping[str, Any],
    vector_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, bool]:
    if manifest_path.exists() and not vector_path.exists():
        raise ForumEmbeddingError("completed embedding shard vector payload is missing")
    if not vector_path.exists():
        return None, False
    payload_sha256, byte_size = _sha256_file(vector_path)
    rebuilt = _shard_body(
        plan=plan,
        planned_shard=planned_shard,
        payload_sha256=payload_sha256,
        byte_size=byte_size,
    )
    if manifest_path.exists():
        if dict(_read_json(manifest_path, "embedding shard manifest")) != rebuilt:
            raise ForumEmbeddingError("completed embedding shard hash or identity changed")
        return rebuilt, False
    # The vector file is atomically renamed before its metadata.  Rebuilding the
    # path-free manifest closes that single crash window without re-encoding.
    _atomic_json(manifest_path, rebuilt)
    return rebuilt, True


def _encode_shard(
    *,
    plan: Mapping[str, Any],
    planned_shard: Mapping[str, Any],
    vector_path: Path,
    manifest_path: Path,
    encoder: LocalEmbeddingEncoder,
    text_loader: PrivateSemanticTextLoader,
    encode_batch_size: int,
) -> tuple[dict[str, Any], int]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=vector_path.name + ".", suffix=".tmp", dir=vector_path.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_size = 0
    batch_calls = 0
    row_ids = list(planned_shard["semantic_row_ids"])
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for start in range(0, len(row_ids), encode_batch_size):
                batch_ids = row_ids[start : start + encode_batch_size]
                loaded = text_loader.load_texts(tuple(batch_ids))
                if not isinstance(loaded, Mapping) or set(loaded) != set(batch_ids):
                    raise ForumEmbeddingError(
                        "private text loader did not return the requested semantic rows exactly"
                    )
                texts = [loaded[row_id] for row_id in batch_ids]
                if any(not isinstance(text, str) for text in texts):
                    raise ForumEmbeddingError("private text loader returned a non-text value")
                vectors = np.asarray(encoder.encode(tuple(texts)))
                if vectors.shape != (len(batch_ids), EMBEDDING_DIMENSION):
                    raise ForumEmbeddingError("local encoder returned an invalid vector shape")
                if not np.issubdtype(vectors.dtype, np.number):
                    raise ForumEmbeddingError("local encoder returned nonnumeric vectors")
                vectors = np.ascontiguousarray(vectors, dtype="<f2")
                if not np.isfinite(vectors).all():
                    raise ForumEmbeddingError("local encoder returned nonfinite float16 vectors")
                payload = vectors.tobytes(order="C")
                handle.write(payload)
                digest.update(payload)
                byte_size += len(payload)
                batch_calls += 1
            handle.flush()
            os.fsync(handle.fileno())
        artifact = _shard_body(
            plan=plan,
            planned_shard=planned_shard,
            payload_sha256=digest.hexdigest(),
            byte_size=byte_size,
        )
        os.replace(temporary, vector_path)
        _atomic_json(manifest_path, artifact)
        return artifact, batch_calls
    finally:
        if temporary.exists():
            temporary.unlink()


def _merge_body(
    *,
    plan: Mapping[str, Any],
    shard_artifacts: Sequence[Mapping[str, Any]],
    payload_sha256: str,
    byte_size: int,
) -> dict[str, Any]:
    expected_bytes = int(plan["rows"]["count"]) * EMBEDDING_DIMENSION * _VECTOR_ITEM_BYTES
    if byte_size != expected_bytes or _SHA256_RE.fullmatch(payload_sha256) is None:
        raise ForumEmbeddingError("merged embedding vector payload layout is invalid")
    body = {
        "schema_version": LOCAL_BGE_MERGE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "ordered_shard_ids": [item["shard_id"] for item in shard_artifacts],
        "ordered_shard_payload_sha256s": [
            item["payload_sha256"] for item in shard_artifacts
        ],
        "vector_count": int(plan["rows"]["count"]),
        "dimension": EMBEDDING_DIMENSION,
        "dtype": EMBEDDING_DTYPE,
        "payload_sha256": payload_sha256,
        "byte_size": byte_size,
        "vector_file": "forum-corpus-embeddings.f16",
        "raw_text_persisted": False,
        "provider_calls": 0,
    }
    return _hash_bound(body, "merge_artifact_sha256")


def _merge_shards(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    shard_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    vector_path = output_dir / "forum-corpus-embeddings.f16"
    manifest_path = output_dir / "forum-corpus-embeddings.json"
    if manifest_path.exists() and not vector_path.exists():
        raise ForumEmbeddingError("merged embedding manifest has no vector payload")
    if vector_path.exists():
        payload_sha256, byte_size = _sha256_file(vector_path)
        rebuilt = _merge_body(
            plan=plan,
            shard_artifacts=shard_artifacts,
            payload_sha256=payload_sha256,
            byte_size=byte_size,
        )
        if manifest_path.exists():
            if dict(_read_json(manifest_path, "merged embedding manifest")) != rebuilt:
                raise ForumEmbeddingError("merged embedding hash or identity changed")
            return rebuilt, False
        _atomic_json(manifest_path, rebuilt)
        return rebuilt, True
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=vector_path.name + ".", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            for artifact in shard_artifacts:
                source_path = output_dir / str(artifact["vector_file"])
                with source_path.open("rb") as source:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        byte_size += len(block)
            target.flush()
            os.fsync(target.fileno())
        artifact = _merge_body(
            plan=plan,
            shard_artifacts=shard_artifacts,
            payload_sha256=digest.hexdigest(),
            byte_size=byte_size,
        )
        os.replace(temporary, vector_path)
        _atomic_json(manifest_path, artifact)
        return artifact, False
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_local_bge_embedding_plan_v1(
    *,
    plan_path: str | Path,
    output_dir: str | Path,
    encoder: LocalEmbeddingEncoder,
    text_loader: PrivateSemanticTextLoader,
    encode_batch_size: int = 256,
) -> dict[str, Any]:
    """Execute one immutable local shard plan with exact crash-safe resume."""

    if not isinstance(encode_batch_size, int) or isinstance(encode_batch_size, bool):
        raise ForumEmbeddingError("encode_batch_size must be an integer")
    if not 1 <= encode_batch_size <= 4096:
        raise ForumEmbeddingError("encode_batch_size must be between 1 and 4096")
    plan = load_embedding_shard_plan_v1(plan_path)
    planned_identity, planned_shards = _validate_plan(plan)
    if not isinstance(getattr(encoder, "identity", None), LocalBGEEncoderIdentity):
        raise ForumEmbeddingError("injected encoder must expose a frozen identity")
    encoder_identity = encoder.identity.validated_view()
    if encoder_identity != planned_identity:
        raise ForumEmbeddingError("injected encoder identity does not match the shard plan")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    binding = _binding(plan, encoder_identity)
    _ensure_binding(destination / "embedding-execution-binding.json", binding)

    shard_artifacts: list[dict[str, Any]] = []
    newly_encoded_shards = 0
    recovered_shard_manifests = 0
    encoder_batch_calls = 0
    for planned_shard in planned_shards:
        shard_id = str(planned_shard["shard_id"])
        vector_path = destination / f"{shard_id}.f16"
        manifest_path = destination / f"{shard_id}.json"
        artifact, recovered = _existing_or_recovered_shard(
            plan=plan,
            planned_shard=planned_shard,
            vector_path=vector_path,
            manifest_path=manifest_path,
        )
        if artifact is None:
            artifact, batch_calls = _encode_shard(
                plan=plan,
                planned_shard=planned_shard,
                vector_path=vector_path,
                manifest_path=manifest_path,
                encoder=encoder,
                text_loader=text_loader,
                encode_batch_size=encode_batch_size,
            )
            newly_encoded_shards += 1
            encoder_batch_calls += batch_calls
        elif recovered:
            recovered_shard_manifests += 1
        shard_artifacts.append(artifact)

    completed = [
        {
            "shard_id": item["shard_id"],
            "vector_count": item["vector_count"],
            "dimension": item["dimension"],
            "dtype": item["dtype"],
            "row_id_sha256": item["row_id_sha256"],
            "payload_sha256": item["payload_sha256"],
        }
        for item in shard_artifacts
    ]
    resume_manifest = build_embedding_resume_manifest_v1(
        plan, completed_shards=completed
    )
    if not resume_manifest["merge_ready"]:
        raise ForumEmbeddingError("completed local shards did not cover the full plan")
    _atomic_json(destination / "embedding-resume-manifest.json", resume_manifest)

    merged, recovered_merge_manifest = _merge_shards(
        output_dir=destination,
        plan=plan,
        shard_artifacts=shard_artifacts,
    )
    row_ids = [
        row_id
        for planned_shard in planned_shards
        for row_id in planned_shard["semantic_row_ids"]
    ]
    index_release = build_embedding_index_release_v1(
        corpus_snapshot_sha256=str(plan["corpus_snapshot_sha256"]),
        model_revision=str(planned_identity["model_revision"]),
        model_weights_sha256=str(planned_identity["model_weights_sha256"]),
        tokenizer_sha256=str(planned_identity["tokenizer_sha256"]),
        model_config_sha256=str(planned_identity["model_config_sha256"]),
        expected_row_ids=row_ids,
        shards=[
            EmbeddingShard(
                shard_id=str(planned_shard["shard_id"]),
                row_ids=tuple(str(row_id) for row_id in planned_shard["semantic_row_ids"]),
                vector_count=int(artifact["vector_count"]),
                dimension=int(artifact["dimension"]),
                dtype=str(artifact["dtype"]),
                payload_sha256=str(artifact["payload_sha256"]),
            )
            for planned_shard, artifact in zip(planned_shards, shard_artifacts)
        ],
    )
    if index_release["schema_version"] != CORPUS_INDEX_SCHEMA:
        raise ForumEmbeddingError("local embedding index release schema drifted")
    _atomic_json(destination / "corpus-index.json", index_release)

    release_body = {
        "schema_version": LOCAL_BGE_EXECUTION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "corpus_snapshot_sha256": plan["corpus_snapshot_sha256"],
        "embedding": planned_identity,
        "rows": {
            "count": int(plan["rows"]["count"]),
            "semantic_row_order_sha256": plan["rows"]["semantic_row_order_sha256"],
        },
        "shards": [
            {
                key: item[key]
                for key in (
                    "shard_id",
                    "vector_count",
                    "dimension",
                    "dtype",
                    "row_id_sha256",
                    "payload_sha256",
                    "byte_size",
                    "shard_artifact_sha256",
                )
            }
            for item in shard_artifacts
        ],
        "merge": {
            "payload_sha256": merged["payload_sha256"],
            "byte_size": merged["byte_size"],
            "merge_artifact_sha256": merged["merge_artifact_sha256"],
            "ordered_shard_payload_sha256s": merged[
                "ordered_shard_payload_sha256s"
            ],
        },
        "resume_manifest_sha256": resume_manifest["resume_manifest_sha256"],
        "index_sha256": index_release["index_sha256"],
        "execution": {
            "network_calls": 0,
            "provider_calls": 0,
            "model_download_performed": False,
            "encoder_injected": True,
            "idempotent": True,
        },
        "privacy": {
            "raw_text_persisted": False,
            "source_ids_persisted": False,
            "source_paths_persisted": False,
        },
    }
    release = _hash_bound(release_body, "execution_release_sha256")
    release_path = destination / "embedding-execution-release.json"
    if release_path.exists():
        if dict(_read_json(release_path, "embedding execution release")) != release:
            raise ForumEmbeddingError("existing embedding execution release changed")
    else:
        _atomic_json(release_path, release)
    return {
        "release": release,
        "index_release": index_release,
        "resume_manifest": resume_manifest,
        "run_stats": {
            "newly_encoded_shards": newly_encoded_shards,
            "reused_shards": len(planned_shards) - newly_encoded_shards,
            "recovered_shard_manifests": recovered_shard_manifests,
            "recovered_merge_manifest": recovered_merge_manifest,
            "encoder_batch_calls": encoder_batch_calls,
            "provider_calls": 0,
            "network_calls": 0,
        },
    }
