from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from comparison_harness.contracts import CommonObservation, MethodSnapshot, canonical_json
from comparison_harness.prompting import SharedPromptRenderer
from comparison_harness.real_provider import (
    BoundaryViolationError,
    EmbeddingTransportResponse,
    ModelResponseError,
    ProviderDisabledError,
    RealModelProvider,
    TransportResponse,
)
from comparison_harness.real_runtime import strict_api_subset


CORE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CORE_ROOT.parent
CONFIG = PROJECT_ROOT / "configs" / "real_model_provider_boundary_v2.json"
AUXILIARY = PROJECT_ROOT / "configs" / "auxiliary_call_boundary_v1.json"

ACTION = {
    "action": "read", "target_id": None, "text": None, "emotion": None,
    "public_stance": None, "public_or_private": "none",
}


class RecordingTransport:
    def __init__(self, *, response=None, failures=0, embedding_size=1536):
        self.response = ACTION if response is None else response
        self.failures = failures
        self.embedding_size = embedding_size
        self.response_calls = []
        self.embedding_calls = []

    def create_response(self, payload):
        self.response_calls.append(payload)
        if len(self.response_calls) <= self.failures:
            raise TimeoutError("synthetic timeout")
        if isinstance(self.response, TransportResponse):
            return self.response
        output = canonical_json(self.response)
        return TransportResponse(
            output_text=output,
            raw_response={"id": "resp_test", "status": "completed", "output_text": output},
            request_id="resp_test", status="completed",
            usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        )

    def create_embedding(self, payload):
        self.embedding_calls.append(payload)
        vector = tuple(0.0 for _ in range(self.embedding_size))
        return EmbeddingTransportResponse(
            vector=vector,
            raw_response={"id": "emb_test", "data": [{"embedding": list(vector)}]},
            request_id="emb_test",
            usage={"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        )


class RealProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.transport = RecordingTransport()
        self.observation = CommonObservation(
            scenario_id="audit", tick=0, agent_id="agent-1",
            factual_registry={"synthetic": True}, current_event={"event": "baseline"},
            visible_context=(), governance_artifacts=(),
            response_schema={
                "type": "object", "additionalProperties": False,
                "properties": {
                    "action": {"enum": ["read", "post", "reply", "quote", "react", "private_message", "no_action"]},
                    "target_id": {"type": ["string", "null"]},
                    "text": {"type": ["string", "null"]},
                    "emotion": {"type": ["string", "null"]},
                    "public_stance": {"type": ["number", "null"], "minimum": -1, "maximum": 1},
                    "public_or_private": {"enum": ["public", "private", "none"]},
                },
                "required": ["action", "target_id", "text", "emotion", "public_stance", "public_or_private"],
            },
        )
        self.bundle = self._bundle("belief_engine")

    def tearDown(self):
        self.temp.cleanup()

    def _bundle(self, method):
        snapshot = MethodSnapshot(
            method=method, agent_id="agent-1", tick=0, native_state={},
            cognitive_context={"belief": 0},
            source_observation_sha256=self.observation.observation_sha256,
        )
        return SharedPromptRenderer().render(self.observation, snapshot)

    def provider(self, *, enabled=True, transport=None, environment=None, profile="handshake", config=CONFIG):
        env = {
            "AAMAS_REAL_MODEL_ENABLED": "YES_I_ACCEPT_API_COST",
            "OPENAI_API_KEY": "test-key-never-sent",
        } if environment is None else environment
        return RealModelProvider(
            config_path=config, auxiliary_boundary_path=AUXILIARY,
            storage_root=Path(self.temp.name),
            transport=self.transport if transport is None else transport,
            enable_real=enabled, budget_profile=profile, environment=env,
            sleeper=lambda _: None,
        )

    def generate(self, provider, *, method="belief_engine", bundle=None):
        return provider.generate(
            method=method, bundle=self.bundle if bundle is None else bundle,
            response_schema=self.observation.response_schema,
            allowed_actions=self.observation.allowed_actions,
        )

    def test_default_disabled_blocks_before_transport(self):
        with self.assertRaises(ProviderDisabledError):
            self.generate(self.provider(enabled=False))
        self.assertEqual([], self.transport.response_calls)

    def test_confirmation_api_key_and_full_gates_are_enforced(self):
        for environment in ({"OPENAI_API_KEY": "x"}, {"AAMAS_REAL_MODEL_ENABLED": "YES_I_ACCEPT_API_COST"}):
            with self.subTest(environment=environment):
                with self.assertRaises(ProviderDisabledError):
                    self.generate(self.provider(environment=environment))
        with self.assertRaises(ProviderDisabledError):
            self.generate(self.provider(profile="full"))

    def test_frozen_config_hash_tamper_is_rejected(self):
        bad = Path(self.temp.name) / "bad.json"
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        value["default_enabled"] = True
        bad.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(BoundaryViolationError):
            self.provider(config=bad)

    def test_method_and_prompt_boundary_including_campuspulse(self):
        provider = self.provider()
        with self.assertRaises(BoundaryViolationError):
            provider.preflight(
                method="belief_engine",
                bundle=replace(self.bundle, rendered=self.bundle.rendered + "tampered"),
                response_schema=self.observation.response_schema,
                allowed_actions=self.observation.allowed_actions,
            )
        campuspulse = self._bundle("campuspulse")
        self.assertEqual("campuspulse", provider.preflight(
            method="campuspulse", bundle=campuspulse,
            response_schema=self.observation.response_schema,
            allowed_actions=self.observation.allowed_actions,
        )["method"])

    def test_retry_archive_cache_and_cache_integrity(self):
        transport = RecordingTransport(failures=1)
        provider = self.provider(transport=transport)
        first = self.generate(provider)
        second = self.generate(provider)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(2, len(transport.response_calls))
        cache_path = Path(self.temp.name) / "cache" / f"{first.cache_key}.json"
        wrapped = json.loads(cache_path.read_text(encoding="utf-8"))
        wrapped["entry"]["output_text"] = "{}"
        cache_path.write_text(json.dumps(wrapped), encoding="utf-8")
        with self.assertRaises(BoundaryViolationError):
            self.generate(provider)

    def test_refusal_incomplete_and_bad_schema_are_archived_before_error(self):
        cases = [
            TransportResponse("", {"status": "completed", "refusal": "no"}, "r1", "completed", {}, "no"),
            TransportResponse("", {"status": "incomplete"}, "r2", "incomplete", {}),
            TransportResponse("{}", {"status": "completed", "output_text": "{}"}, "r3", "completed", {}),
        ]
        for index, response in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as storage:
                provider = RealModelProvider(
                    config_path=CONFIG, auxiliary_boundary_path=AUXILIARY,
                    storage_root=Path(storage), transport=RecordingTransport(response=response),
                    enable_real=True, environment={
                        "AAMAS_REAL_MODEL_ENABLED": "YES_I_ACCEPT_API_COST", "OPENAI_API_KEY": "x"
                    }, sleeper=lambda _: None,
                )
                with self.assertRaises(ModelResponseError):
                    self.generate(provider)
                self.assertEqual(1, len(list((Path(storage) / "raw_responses").glob("*.json"))))

    def test_embedding_archive_cache_and_dimension_validation(self):
        provider = self.provider()
        first = provider.embed("hello")
        second = provider.embed("hello")
        self.assertEqual(1536, len(first.vector))
        self.assertTrue(second.cache_hit)
        self.assertEqual(1, len(self.transport.embedding_calls))
        bad_provider = self.provider(transport=RecordingTransport(embedding_size=7))
        with self.assertRaises(ModelResponseError):
            bad_provider.embed("bad dimension")
        self.assertGreaterEqual(len(list((Path(self.temp.name) / "raw_responses").glob("*.json"))), 2)

    def test_api_schema_subset_keeps_full_schema_for_local_validation(self):
        full = {
            "type": "object", "properties": {
                "x": {"type": "integer"},
                "version": {"const": "v1"},
                "choice": {"enum": ["a", "b"]},
                "items": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
            },
            "required": ["x"],
            "allOf": [{"if": {"properties": {"x": {"const": 1}}}, "then": {}}],
        }
        subset = strict_api_subset(full)
        self.assertNotIn("allOf", subset)
        self.assertEqual("string", subset["properties"]["version"]["type"])
        self.assertEqual("string", subset["properties"]["choice"]["type"])
        self.assertNotIn("uniqueItems", subset["properties"]["items"])
        provider = self.provider(transport=RecordingTransport(response={"x": "wrong"}))
        with self.assertRaises(ModelResponseError):
            provider.call_json(
                call_id="stance_probe", prompt="probe", schema=subset,
                validation_schema=full, schema_name="test_full_local_schema",
            )


if __name__ == "__main__":
    unittest.main()
