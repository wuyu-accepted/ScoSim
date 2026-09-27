from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from jsonschema import Draft202012Validator

from .contracts import canonical_json
from .prompting import PromptBundle, parse_structured_decision


PROVIDER_BOUNDARY_V2_SHA256 = "78d6529e7110612df72c56a13a3c8f8fcf010dc0eaa33a184014de71efc4d217"


class ProviderDisabledError(RuntimeError):
    pass


class BoundaryViolationError(RuntimeError):
    pass


class BudgetExceededError(RuntimeError):
    pass


class ModelResponseError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransportResponse:
    output_text: str
    raw_response: Mapping[str, Any]
    request_id: str | None
    status: str
    usage: Mapping[str, int]
    refusal: str | None = None


@dataclass(frozen=True)
class EmbeddingTransportResponse:
    vector: tuple[float, ...]
    raw_response: Mapping[str, Any]
    request_id: str | None
    usage: Mapping[str, int]


class ProviderTransport(Protocol):
    def create_response(self, payload: Mapping[str, Any]) -> TransportResponse:
        raise NotImplementedError

    def create_embedding(self, payload: Mapping[str, Any]) -> EmbeddingTransportResponse:
        raise NotImplementedError


@dataclass(frozen=True)
class ProviderResult:
    output_text: str
    parsed: Mapping[str, Any]
    cache_hit: bool
    request_id: str | None
    usage: Mapping[str, int]
    raw_archive_path: str
    cache_key: str

    @property
    def parsed_decision(self) -> Mapping[str, Any]:
        return self.parsed


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    cache_hit: bool
    request_id: str | None
    usage: Mapping[str, int]
    raw_archive_path: str
    cache_key: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class OpenAIProviderTransport:
    """OpenAI Responses and Embeddings transport with SDK retries disabled."""

    def __init__(self, *, api_key_env: str, timeout_seconds: float, credentials_path: Path | None = None):
        credentials: dict[str, Any] = {}
        if credentials_path is not None:
            credentials = json.loads(credentials_path.read_text(encoding="utf-8-sig"))
            if not isinstance(credentials, dict):
                raise ProviderDisabledError("credentials JSON must be an object")
        api_key = os.environ.get(api_key_env) or str(credentials.get("api_key") or "").strip()
        if not api_key:
            raise ProviderDisabledError(f"missing API key in {api_key_env} or credentials JSON")
        base_url = str(credentials.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
        if base_url and not base_url.startswith("https://"):
            raise ProviderDisabledError("base_url must use https")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install requirements-real-provider.txt before real execution") from error
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def prepare_concurrency(self) -> None:
        """Resolve SDK lazy resource imports on the caller thread.

        The OpenAI SDK exposes resources through cached properties.  Resolving
        different properties for the first time in separate worker threads can
        deadlock Python's module locks.  This method performs no network I/O.
        """
        _ = self._client.responses
        _ = self._client.embeddings

    def create_response(self, payload: Mapping[str, Any]) -> TransportResponse:
        response = self._client.responses.create(**dict(payload))
        raw = response.model_dump(mode="json")
        usage_value = raw.get("usage") or {}
        refusal = None
        for item in raw.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") == "refusal":
                    refusal = str(content.get("refusal") or "model refusal")
                    break
        return TransportResponse(
            output_text=str(getattr(response, "output_text", "") or ""),
            raw_response=raw,
            request_id=getattr(response, "id", None),
            status=str(getattr(response, "status", "unknown")),
            usage={
                "input_tokens": int(usage_value.get("input_tokens") or 0),
                "output_tokens": int(usage_value.get("output_tokens") or 0),
                "total_tokens": int(usage_value.get("total_tokens") or 0),
            },
            refusal=refusal,
        )

    def create_embedding(self, payload: Mapping[str, Any]) -> EmbeddingTransportResponse:
        response = self._client.embeddings.create(**dict(payload))
        raw = response.model_dump(mode="json")
        usage_value = raw.get("usage") or {}
        data = raw.get("data") or []
        vector_value = data[0].get("embedding") if len(data) == 1 else []
        if not isinstance(vector_value, list):
            vector_value = []
        return EmbeddingTransportResponse(
            vector=tuple(float(value) for value in vector_value),
            raw_response=raw,
            request_id=getattr(response, "id", None),
            usage={
                "input_tokens": int(usage_value.get("prompt_tokens") or 0),
                "output_tokens": 0,
                "total_tokens": int(usage_value.get("total_tokens") or 0),
            },
        )


OpenAIResponsesTransport = OpenAIProviderTransport


class RealModelProvider:
    """Frozen, gated provider for main, auxiliary, judge, probe and embedding calls."""

    def __init__(
        self,
        *,
        config_path: Path,
        auxiliary_boundary_path: Path,
        storage_root: Path,
        transport: ProviderTransport | None = None,
        enable_real: bool = False,
        budget_profile: str = "handshake",
        environment: Mapping[str, str] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config_path = config_path
        self.boundary_path = auxiliary_boundary_path
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.boundary = json.loads(auxiliary_boundary_path.read_text(encoding="utf-8"))
        self.storage_root = storage_root.resolve()
        self.transport = transport
        self.enable_real = enable_real
        self.budget_profile = budget_profile
        self.environment = dict(os.environ if environment is None else environment)
        self.sleeper = sleeper
        self.logical_calls = 0
        self.transport_attempts = 0
        self.input_chars = 0
        self.requested_output_tokens = 0
        self.actual_output_tokens = 0
        self.embedding_inputs = 0
        self._state_lock = threading.Lock()
        self._validate_frozen_configuration()

    @property
    def limits(self) -> Mapping[str, int]:
        return self.config["budget_profiles"][self.budget_profile]

    def _validate_frozen_configuration(self) -> None:
        actual_config_hash = _sha256_bytes(self.config_path.read_bytes())
        if actual_config_hash != PROVIDER_BOUNDARY_V2_SHA256:
            raise BoundaryViolationError(f"provider boundary hash mismatch: {actual_config_hash}")
        if self.config.get("boundary_version") != "2.0.0" or self.config.get("status") != "frozen":
            raise BoundaryViolationError("provider v2 boundary must be frozen")
        if self.config.get("default_enabled") is not False:
            raise BoundaryViolationError("real provider must be disabled by default")
        if self.budget_profile not in self.config.get("budget_profiles", {}):
            raise BoundaryViolationError(f"unknown budget profile: {self.budget_profile}")
        actual_auxiliary_hash = _sha256_bytes(self.boundary_path.read_bytes())
        if actual_auxiliary_hash != self.config.get("auxiliary_boundary_sha256"):
            raise BoundaryViolationError("linked auxiliary boundary hash mismatch")
        main = self.boundary["main_action_call"]
        frozen = self.config["response_calls"]["main_action"]
        for key in ("model", "temperature", "top_p", "max_output_tokens"):
            if main.get(key) != frozen.get(key):
                raise BoundaryViolationError(f"main action boundary drift: {key}")
        constraints = self.config["request_constraints"]
        if constraints.get("store") is not False or constraints.get("tools") != []:
            raise BoundaryViolationError("storage and model tools must remain disabled")
        expected_auxiliary: dict[str, Mapping[str, Any]] = {}
        for method in self.boundary["methods"].values():
            for item in method["allowed_auxiliary_calls"]:
                if item["id"] != "shared_embedding":
                    expected_auxiliary[item["id"]] = item
        for call_id, item in expected_auxiliary.items():
            configured = self.config["response_calls"].get(call_id)
            if configured is None:
                raise BoundaryViolationError(f"missing auxiliary call boundary: {call_id}")
            for key in ("model", "temperature", "max_output_tokens"):
                if configured.get(key) != item.get(key):
                    raise BoundaryViolationError(f"auxiliary call boundary drift: {call_id}:{key}")
        embedding = self.config["embedding_call"]
        if embedding["model"] != self.boundary["shared_embedding_call"]["model"]:
            raise BoundaryViolationError("embedding model drift")

    def _ensure_enabled(self) -> None:
        gate = self.config["enable_gate"]
        if not self.enable_real:
            raise ProviderDisabledError("real provider is disabled; pass the explicit execution flag")
        if self.environment.get(gate["environment_variable"]) != gate["required_value"]:
            raise ProviderDisabledError("real provider confirmation environment value is absent")
        if self.budget_profile == "full":
            full_gate = self.config["full_run_gate"]
            if self.environment.get(full_gate["environment_variable"]) != full_gate["required_value"]:
                raise ProviderDisabledError("full experiment confirmation environment value is absent")
        if not self.environment.get(self.config["api_key_environment_variable"]):
            raise ProviderDisabledError("real provider API key environment variable is absent")
        if self.transport is None:
            raise ProviderDisabledError("no real provider transport was configured")

    def _reserve(self, *, input_chars: int, requested_output: int = 0, embedding_inputs: int = 0) -> None:
        with self._state_lock:
            limits = self.limits
            if self.logical_calls + 1 > limits["max_logical_calls"]:
                raise BudgetExceededError("logical call budget exceeded")
            if self.input_chars + input_chars > limits["max_input_characters"]:
                raise BudgetExceededError("input character budget exceeded")
            if self.requested_output_tokens + requested_output > limits["max_requested_output_tokens"]:
                raise BudgetExceededError("requested output token budget exceeded")
            if self.embedding_inputs + embedding_inputs > limits["max_embedding_inputs"]:
                raise BudgetExceededError("embedding input budget exceeded")
            self.logical_calls += 1
            self.input_chars += input_chars
            self.requested_output_tokens += requested_output
            self.embedding_inputs += embedding_inputs

    def _reserve_transport_attempt(self) -> None:
        with self._state_lock:
            if self.transport_attempts + 1 > self.limits["max_transport_attempts"]:
                raise BudgetExceededError("transport attempt budget exceeded")
            self.transport_attempts += 1

    def _cache_key(self, kind: str, payload: Mapping[str, Any]) -> str:
        return _sha256_json({
            "provider_boundary_sha256": PROVIDER_BOUNDARY_V2_SHA256,
            "auxiliary_boundary_sha256": self.config["auxiliary_boundary_sha256"],
            "kind": kind,
            "request": payload,
        })

    def _audit(self, event: Mapping[str, Any]) -> None:
        path = self.storage_root / "provider_audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_lock:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")

    def _archive(self, *, kind: str, call_id: str, cache_key: str, attempt: int,
                 payload: Mapping[str, Any], raw_response: Mapping[str, Any]) -> str:
        archive_dir = self.storage_root / "raw_responses"
        archive_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{time.time_ns()}_{kind}_{call_id}_call-{self.logical_calls:06d}_"
            f"attempt-{attempt:02d}_{cache_key[:16]}.json"
        )
        path = archive_dir / filename
        archive = {
            "archive_schema_version": "2.0.0",
            "kind": kind,
            "call_id": call_id,
            "cache_key": cache_key,
            "request": payload,
            "response": raw_response,
            "response_sha256": _sha256_json(raw_response),
        }
        _atomic_json(path, archive)
        return path.relative_to(self.storage_root).as_posix()

    def _cache_write(self, cache_key: str, entry: Mapping[str, Any]) -> None:
        wrapped = {"entry": dict(entry), "entry_sha256": _sha256_json(entry)}
        _atomic_json(self.storage_root / "cache" / f"{cache_key}.json", wrapped)

    def _cache_read(self, cache_key: str) -> Mapping[str, Any] | None:
        path = self.storage_root / "cache" / f"{cache_key}.json"
        if not path.exists():
            return None
        wrapped = json.loads(path.read_text(encoding="utf-8"))
        entry = wrapped.get("entry")
        if not isinstance(entry, Mapping) or wrapped.get("entry_sha256") != _sha256_json(entry):
            raise BoundaryViolationError("cache entry digest mismatch")
        if entry.get("cache_key") != cache_key:
            raise BoundaryViolationError("cache key mismatch")
        archive_path = (self.storage_root / str(entry.get("raw_archive_path", ""))).resolve()
        if not archive_path.is_file() or self.storage_root not in archive_path.parents:
            raise BoundaryViolationError("cache raw archive is absent or outside storage root")
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        if archive.get("cache_key") != cache_key:
            raise BoundaryViolationError("raw archive cache key mismatch")
        if archive.get("response_sha256") != _sha256_json(archive.get("response")):
            raise BoundaryViolationError("raw archive response digest mismatch")
        if entry.get("raw_response_sha256") != archive.get("response_sha256"):
            raise BoundaryViolationError("cache and raw archive response digests differ")
        return entry

    def _retryable(self, error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        configured = set(self.config["retry_policy"]["retryable_status_codes"])
        if status in configured:
            return True
        name = type(error).__name__.lower()
        return any(token in name for token in ("timeout", "connection", "ratelimit"))

    def _run_with_retries(self, *, call_id: str, cache_key: str,
                          operation: Callable[[int], Any]) -> Any:
        policy = self.config["retry_policy"]
        max_attempts = 1 + int(policy["max_retries"])
        for attempt in range(1, max_attempts + 1):
            self._reserve_transport_attempt()
            try:
                return operation(attempt)
            except Exception as error:
                retryable = self._retryable(error)
                self._audit({
                    "event": "network_or_validation_error", "call_id": call_id,
                    "cache_key": cache_key, "attempt": attempt,
                    "error_type": type(error).__name__, "retryable": retryable,
                })
                if not retryable or attempt >= max_attempts:
                    raise
                delay = min(
                    float(policy["max_delay_seconds"]),
                    float(policy["base_delay_seconds"]) * (2 ** (attempt - 1)),
                )
                self.sleeper(delay)
        raise AssertionError("unreachable retry state")

    def _response_payload(self, *, call_id: str, prompt: str,
                          schema: Mapping[str, Any], schema_name: str) -> dict[str, Any]:
        if call_id not in self.config["response_calls"]:
            raise BoundaryViolationError(f"unregistered response call: {call_id}")
        spec = self.config["response_calls"][call_id]
        constraints = self.config["request_constraints"]
        payload = {
            "model": spec["model"], "input": prompt,
            "temperature": spec["temperature"], "top_p": spec["top_p"],
            "max_output_tokens": spec["max_output_tokens"],
            "reasoning": {"effort": constraints["reasoning_effort"]},
            "store": False, "tools": [], "parallel_tool_calls": False,
            "truncation": "disabled",
            "prompt_cache_key": _sha256_bytes(prompt.encode("utf-8")),
            "text": {"format": {
                "type": "json_schema", "name": schema_name,
                "strict": True, "schema": schema,
            }},
        }
        if spec.get("omit_top_p"):
            payload.pop("top_p", None)
        return payload

    def preflight_json(self, *, call_id: str, prompt: str,
                       schema: Mapping[str, Any], schema_name: str) -> dict[str, Any]:
        payload = self._response_payload(
            call_id=call_id, prompt=prompt, schema=schema, schema_name=schema_name
        )
        return {
            "call_id": call_id, "model": payload["model"],
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "schema_sha256": _sha256_json(schema),
            "max_output_tokens": payload["max_output_tokens"],
            "network_call_permitted": bool(self.enable_real),
        }

    def call_json(self, *, call_id: str, prompt: str, schema: Mapping[str, Any],
                  schema_name: str,
                  validation_schema: Mapping[str, Any] | None = None) -> ProviderResult:
        self._ensure_enabled()
        payload = self._response_payload(
            call_id=call_id, prompt=prompt, schema=schema, schema_name=schema_name
        )
        requested_output = int(payload["max_output_tokens"])
        self._reserve(input_chars=len(prompt), requested_output=requested_output)
        effective_schema = validation_schema or schema
        cache_key = self._cache_key("response", {
            "request": payload,
            "validation_schema_sha256": _sha256_json(effective_schema),
        })
        cached = self._cache_read(cache_key)
        # Some OpenAI-compatible gateways reject an explicit top_p=1.0 for
        # selected models even though 1.0 is the endpoint default.  Preserve
        # all existing full-payload cache entries; for a new request only,
        # serialize the same effective sampling boundary by omitting the
        # unsupported default-valued field.
        if cached is None and payload.get("top_p") == 1.0:
            compatible_payload = dict(payload)
            compatible_payload.pop("top_p", None)
            compatible_key = self._cache_key("response", {
                "request": compatible_payload,
                "validation_schema_sha256": _sha256_json(effective_schema),
            })
            compatible_cached = self._cache_read(compatible_key)
            payload, cache_key, cached = compatible_payload, compatible_key, compatible_cached
            self._audit({
                "event": "default_parameter_omitted", "call_id": call_id,
                "parameter": "top_p", "effective_value": 1.0,
                "cache_key": cache_key,
            })
        # The configured 32-token importance cap was sufficient on the
        # original endpoint.  The compatibility gateway can consume all 32 as
        # opaque reasoning tokens and return no JSON.  Keep every successful
        # 32-token cache entry; for a new request, reserve and expose enough
        # room for the same one-integer schema to be emitted.
        if cached is None and call_id == "ga_importance" and int(payload["max_output_tokens"]) == 32:
            extra_output = 96
            with self._state_lock:
                if self.requested_output_tokens + extra_output > self.limits["max_requested_output_tokens"]:
                    raise BudgetExceededError("requested output token budget exceeded")
                self.requested_output_tokens += extra_output
            compatible_payload = dict(payload)
            compatible_payload["max_output_tokens"] = 128
            compatible_key = self._cache_key("response", {
                "request": compatible_payload,
                "validation_schema_sha256": _sha256_json(effective_schema),
            })
            compatible_cached = self._cache_read(compatible_key)
            payload, cache_key, cached = compatible_payload, compatible_key, compatible_cached
            self._audit({
                "event": "output_cap_compatibility_amendment", "call_id": call_id,
                "configured_max_output_tokens": 32, "effective_max_output_tokens": 128,
                "reason": "gateway_reasoning_tokens_exhausted_original_cap",
                "cache_key": cache_key,
            })
        # A rare response can also consume the 128-token compatibility cap in
        # reasoning before emitting the one-integer JSON object.  Escalate only
        # that uncached request; successful 32/128-token results remain frozen
        # and reusable under their original cache keys.
        if cached is None and call_id == "ga_importance" and int(payload["max_output_tokens"]) == 128:
            extra_output = 128
            with self._state_lock:
                if self.requested_output_tokens + extra_output > self.limits["max_requested_output_tokens"]:
                    raise BudgetExceededError("requested output token budget exceeded")
                self.requested_output_tokens += extra_output
            compatible_payload = dict(payload)
            compatible_payload["max_output_tokens"] = 256
            compatible_key = self._cache_key("response", {
                "request": compatible_payload,
                "validation_schema_sha256": _sha256_json(effective_schema),
            })
            compatible_cached = self._cache_read(compatible_key)
            payload, cache_key, cached = compatible_payload, compatible_key, compatible_cached
            self._audit({
                "event": "output_cap_compatibility_amendment", "call_id": call_id,
                "configured_max_output_tokens": 128, "effective_max_output_tokens": 256,
                "reason": "gateway_reasoning_tokens_exhausted_compatibility_cap",
                "cache_key": cache_key,
            })
        # Reflections contain several structured insight strings.  On this
        # endpoint a small minority exhaust the frozen 256-token allowance
        # (status=incomplete, reason=max_output_tokens).  Preserve every
        # completed 256-token cache entry, but give only new reflection calls
        # enough output room to finish the same frozen schema.
        if cached is None and call_id == "ga_reflection" and int(payload["max_output_tokens"]) == 256:
            extra_output = 256
            with self._state_lock:
                if self.requested_output_tokens + extra_output > self.limits["max_requested_output_tokens"]:
                    raise BudgetExceededError("requested output token budget exceeded")
                self.requested_output_tokens += extra_output
            compatible_payload = dict(payload)
            compatible_payload["max_output_tokens"] = 512
            compatible_key = self._cache_key("response", {
                "request": compatible_payload,
                "validation_schema_sha256": _sha256_json(effective_schema),
            })
            compatible_cached = self._cache_read(compatible_key)
            payload, cache_key, cached = compatible_payload, compatible_key, compatible_cached
            self._audit({
                "event": "output_cap_compatibility_amendment", "call_id": call_id,
                "configured_max_output_tokens": 256, "effective_max_output_tokens": 512,
                "reason": "gateway_max_output_tokens_interrupted_structured_reflection",
                "cache_key": cache_key,
            })
        validator = Draft202012Validator(effective_schema)
        if cached is not None:
            parsed = json.loads(str(cached["output_text"]))
            errors = sorted(validator.iter_errors(parsed), key=lambda error: list(error.path))
            if errors:
                raise BoundaryViolationError(f"cached structured output failed schema: {errors[0].message}")
            self._audit({"event": "cache_hit", "call_id": call_id, "cache_key": cache_key})
            return ProviderResult(
                str(cached["output_text"]), parsed, True, cached.get("request_id"),
                cached.get("usage", {}), str(cached["raw_archive_path"]), cache_key,
            )

        def operation(attempt: int) -> ProviderResult:
            assert self.transport is not None
            response = self.transport.create_response(payload)
            archive_path = self._archive(
                kind="response", call_id=call_id, cache_key=cache_key, attempt=attempt,
                payload=payload, raw_response=response.raw_response,
            )
            if response.status != "completed":
                raise ModelResponseError(f"model response status is {response.status}")
            if response.refusal:
                raise ModelResponseError("model returned an explicit refusal")
            parsed = json.loads(response.output_text)
            errors = sorted(validator.iter_errors(parsed), key=lambda error: list(error.path))
            if errors:
                raise ModelResponseError(f"structured output failed schema: {errors[0].message}")
            output_tokens = int(response.usage.get("output_tokens", 0))
            if self.actual_output_tokens + output_tokens > self.limits["max_actual_output_tokens"]:
                raise BudgetExceededError("actual output token budget exceeded")
            self.actual_output_tokens += output_tokens
            entry = {
                "cache_schema_version": "2.0.0", "cache_key": cache_key,
                "output_text": response.output_text, "request_id": response.request_id,
                "usage": dict(response.usage), "raw_archive_path": archive_path,
                "raw_response_sha256": _sha256_json(response.raw_response),
            }
            self._cache_write(cache_key, entry)
            self._audit({
                "event": "network_success", "call_id": call_id, "cache_key": cache_key,
                "attempt": attempt, "request_id": response.request_id,
                "usage": dict(response.usage), "raw_archive_path": archive_path,
            })
            return ProviderResult(
                response.output_text, parsed, False, response.request_id,
                dict(response.usage), archive_path, cache_key,
            )

        return self._run_with_retries(call_id=call_id, cache_key=cache_key, operation=operation)

    def _guard_bundle(self, *, method: str, bundle: PromptBundle,
                      response_schema: Mapping[str, Any], allowed_actions: tuple[str, ...]) -> None:
        if method not in self.config["allowed_methods"]:
            raise BoundaryViolationError(f"method outside provider boundary: {method}")
        if bundle.method != method:
            raise BoundaryViolationError("prompt method does not match requested method")
        if _sha256_bytes(bundle.rendered.encode("utf-8")) != bundle.rendered_sha256:
            raise BoundaryViolationError("rendered prompt hash mismatch")
        if response_schema.get("properties", {}).get("action", {}).get("enum") != list(allowed_actions):
            raise BoundaryViolationError("response schema action vocabulary drift")
        expected = {"action", "target_id", "text", "emotion", "public_stance", "public_or_private"}
        if set(response_schema.get("required", [])) != expected:
            raise BoundaryViolationError("response schema required fields drift")
        if response_schema.get("additionalProperties") is not False:
            raise BoundaryViolationError("response schema must forbid extra fields")

    def preflight(self, *, method: str, bundle: PromptBundle,
                  response_schema: Mapping[str, Any], allowed_actions: tuple[str, ...]) -> dict[str, Any]:
        self._guard_bundle(
            method=method, bundle=bundle, response_schema=response_schema,
            allowed_actions=allowed_actions,
        )
        result = self.preflight_json(
            call_id="main_action", prompt=bundle.rendered,
            schema=response_schema, schema_name="forum_action",
        )
        result.update({"method": method, "observation_sha256": bundle.observation_sha256})
        return result

    def generate(self, *, method: str, bundle: PromptBundle,
                 response_schema: Mapping[str, Any], allowed_actions: tuple[str, ...]) -> ProviderResult:
        self._guard_bundle(
            method=method, bundle=bundle, response_schema=response_schema,
            allowed_actions=allowed_actions,
        )
        result = self.call_json(
            call_id="main_action", prompt=bundle.rendered,
            schema=response_schema, schema_name="forum_action",
        )
        decision = parse_structured_decision(result.output_text, allowed_actions)
        return ProviderResult(
            result.output_text, decision, result.cache_hit, result.request_id,
            result.usage, result.raw_archive_path, result.cache_key,
        )

    def embed(self, text: str) -> EmbeddingResult:
        self._ensure_enabled()
        if not text:
            raise BoundaryViolationError("embedding input must not be empty")
        spec = self.config["embedding_call"]
        payload = {"model": spec["model"], "input": text, "dimensions": spec["dimensions"]}
        self._reserve(input_chars=len(text), embedding_inputs=1)
        cache_key = self._cache_key("embedding", payload)
        cached = self._cache_read(cache_key)
        # Preserve every previously frozen full-text cache entry exactly.  For a
        # new, unusually large retrieval query, bound only the representation
        # sent to the embedding endpoint.  Memory text, prompts, decisions and
        # stance probes retain the complete original.  A head/tail view keeps
        # evidence from both ends and records the original digest for audit.
        if cached is None and len(text) > 6000:
            original_sha256 = _sha256_bytes(text.encode("utf-8"))
            marker = f"\n[deterministic embedding elision; original_sha256={original_sha256}]\n"
            bounded = text[:2950] + marker + text[-2950:]
            payload = {"model": spec["model"], "input": bounded, "dimensions": spec["dimensions"]}
            cache_key = self._cache_key("embedding", payload)
            cached = self._cache_read(cache_key)
            self._audit({
                "event": "embedding_input_bounded", "call_id": "shared_embedding",
                "original_sha256": original_sha256, "original_characters": len(text),
                "bounded_characters": len(bounded), "cache_key": cache_key,
            })
        if cached is not None:
            vector = tuple(float(value) for value in cached["vector"])
            if len(vector) != int(spec["dimensions"]):
                raise BoundaryViolationError("cached embedding dimension mismatch")
            self._audit({"event": "cache_hit", "call_id": "shared_embedding", "cache_key": cache_key})
            return EmbeddingResult(
                vector, True, cached.get("request_id"), cached.get("usage", {}),
                str(cached["raw_archive_path"]), cache_key,
            )

        def operation(attempt: int) -> EmbeddingResult:
            assert self.transport is not None
            response = self.transport.create_embedding(payload)
            archive_path = self._archive(
                kind="embedding", call_id="shared_embedding", cache_key=cache_key,
                attempt=attempt, payload=payload, raw_response=response.raw_response,
            )
            if len(response.vector) != int(spec["dimensions"]):
                raise ModelResponseError("embedding dimension mismatch")
            entry = {
                "cache_schema_version": "2.0.0", "cache_key": cache_key,
                "vector": list(response.vector), "request_id": response.request_id,
                "usage": dict(response.usage), "raw_archive_path": archive_path,
                "raw_response_sha256": _sha256_json(response.raw_response),
            }
            self._cache_write(cache_key, entry)
            self._audit({
                "event": "network_success", "call_id": "shared_embedding",
                "cache_key": cache_key, "attempt": attempt,
                "request_id": response.request_id, "usage": dict(response.usage),
                "raw_archive_path": archive_path,
            })
            return EmbeddingResult(
                response.vector, False, response.request_id, dict(response.usage),
                archive_path, cache_key,
            )

        return self._run_with_retries(
            call_id="shared_embedding", cache_key=cache_key, operation=operation
        )

    def budget_snapshot(self) -> dict[str, int | str]:
        return {
            "budget_profile": self.budget_profile,
            "logical_calls": self.logical_calls,
            "transport_attempts": self.transport_attempts,
            "input_characters": self.input_chars,
            "requested_output_tokens": self.requested_output_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "embedding_inputs": self.embedding_inputs,
        }
