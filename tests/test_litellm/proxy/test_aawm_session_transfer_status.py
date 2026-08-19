import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLMRoutes, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.aawm_session_transfer.endpoints import (
    caller_may_read_session_transfer,
    get_session_transfer_status,
    router,
)
from litellm.proxy.aawm_session_transfer.hooks import publish_adapter_transfer_event
from litellm.proxy.aawm_session_transfer.identity import extract_transfer_identity
from litellm.proxy.aawm_session_transfer.registry import (
    SessionTransferRegistry,
    reset_session_transfer_registry,
    set_session_transfer_registry_override,
)
from litellm.proxy.aawm_session_transfer.schema import (
    MAX_QUERY_RESULTS,
    SCHEMA_VERSION,
    TRANSFER_PERMISSION,
    TRANSFER_ROUTE,
    assert_content_free,
    public_transfer_record,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


class FakeDualCache:
    def __init__(self) -> None:
        self.values = {}

    async def async_get_cache(self, key, **kwargs):
        return self.values.get(key)

    async def async_set_cache(self, key, value, ttl=None, **kwargs):
        self.values[key] = value

    async def async_batch_get_cache(self, keys, **kwargs):
        return [self.values.get(key) for key in keys]


def _admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )


def _service_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="transcript",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        permissions={TRANSFER_PERMISSION: True},
    )


def _denied_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="other",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        permissions={},
    )


@pytest.fixture
def registry():
    store = FakeDualCache()
    item = SessionTransferRegistry(
        cache=store,
        source_instance="worker-a",
        now_fn=lambda: datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc),
    )
    set_session_transfer_registry_override(item)
    yield item
    reset_session_transfer_registry()


def _identity(**overrides):
    payload = {
        "litellm_call_id": "call-1",
        "trace_id": "trace-1",
        "canonical_session_id": "sess-canon",
        "session_id": "sess-canon",
        "codex_session_id": "codex-sess-1",
        "agent_id": "agent-1",
        "agent_name": "worker",
        "parent_agent_id": "parent-1",
        "parent_session_id": "parent-sess",
        "provider": "openai",
        "model": "gpt-5.4",
        "route": "https://chatgpt.com/backend-api/codex/responses",
        "stream_path": "pass_through",
        "context_window": 128000,
        "estimated_input_tokens": 1200,
        "estimated_output_tokens": 40,
        "provider_input_tokens": 1180,
        "provider_output_tokens": 32,
        "remaining_tokens": 126820,
        "request_count": 3,
        "cumulative_input_tokens": 3600,
        "repeated_prefix_tokens": 800,
        "prompt_category_tokens": {
            "system": 200,
            "tool_advertisement": 300,
            "conversation": 500,
            "other": 100,
            "residual": 20,
            "system_behavior": 80,
            "system_safety": 40,
            "system_instructional": 60,
            "system_unclassified": 20,
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_codex_session_query_preserves_exact_identity(registry):
    await registry.mark_phase(_identity(), "response_streaming")
    result = await registry.query(codex_session_id="codex-sess-1")
    assert result["result_count"] == 1
    transfer = result["transfers"][0]
    assert transfer["codex_session_id"] == "codex-sess-1"
    assert transfer["canonical_session_id"] == "sess-canon"
    assert transfer["litellm_call_id"] == "call-1"
    assert transfer["agent_id"] == "agent-1"
    assert transfer["parent_agent_id"] == "parent-1"
    assert transfer["trace_id"] == "trace-1"
    assert transfer["context"]["request_count"] == 3
    assert transfer["context"]["repeated_prefix_tokens"] == 800
    assert transfer["context"]["prompt_category_tokens"]["system"] == 200
    assert_content_free(result)


@pytest.mark.asyncio
async def test_multi_worker_aggregation_uses_shared_store():
    store = FakeDualCache()
    worker_a = SessionTransferRegistry(cache=store, source_instance="worker-a")
    worker_b = SessionTransferRegistry(cache=store, source_instance="worker-b")
    await worker_a.upsert(_identity(litellm_call_id="call-a", agent_id="agent-1"))
    await worker_b.upsert(_identity(litellm_call_id="call-b", agent_id="agent-1"))
    result = await worker_a.query(agent_id="agent-1")
    call_ids = {item["litellm_call_id"] for item in result["transfers"]}
    assert call_ids == {"call-a", "call-b"}


@pytest.mark.asyncio
async def test_stale_active_record_does_not_claim_terminal(registry):
    await registry.upsert(_identity(), {"phase": "response_streaming"})
    stale_now = datetime(2026, 8, 18, 12, 2, tzinfo=timezone.utc)
    registry._now_fn = lambda: stale_now
    result = await registry.query(litellm_call_id="call-1")
    transfer = result["transfers"][0]
    assert transfer["stale"] is True
    assert transfer["freshness"] == "stale"
    assert transfer["phase"] == "response_streaming"
    assert transfer["terminal_state"] is None
    assert transfer["active"] is False


@pytest.mark.asyncio
async def test_redis_degradation_never_fabricates_terminal():
    class FailingCache:
        async def async_get_cache(self, key, **kwargs):
            raise ConnectionError("redis down")

        async def async_set_cache(self, key, value, ttl=None, **kwargs):
            raise ConnectionError("redis down")

    registry = SessionTransferRegistry(cache=FailingCache(), source_instance="worker-a")
    await registry.upsert(_identity(), {"phase": "awaiting_upstream"})
    result = await registry.query(litellm_call_id="call-1")
    assert result["registry"]["state"] in {"degraded", "unavailable"}
    assert result["transfers"][0]["phase"] != "completed"
    assert result["transfers"][0]["redis_degraded"] is True
    assert result["transfers"][0]["freshness"] in {"live", "unavailable", "stale"}


@pytest.mark.asyncio
async def test_query_bounds_results(registry):
    for index in range(MAX_QUERY_RESULTS + 5):
        await registry.upsert(
            _identity(
                litellm_call_id=f"call-{index}",
                agent_id="agent-bound",
            )
        )
    result = await registry.query(agent_id="agent-bound", limit=999)
    assert result["result_count"] == MAX_QUERY_RESULTS
    assert result["truncated"] is True


def test_public_record_strips_content_and_redis_keys():
    record = _identity()
    record["prompt"] = "secret prompt"
    record["redis_key"] = "aawm:session-transfer:hidden"
    record["error_message"] = "full traceback"
    public = public_transfer_record(record)
    assert "prompt" not in public
    assert "redis_key" not in json.dumps(public)
    assert_content_free(public)
    with pytest.raises(ValueError):
        assert_content_free({"transfers": [{"prompt": "nope"}]})


def test_identity_extraction_keeps_codex_session_without_content():
    identity = extract_transfer_identity(
        request_body={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "do not leak"}],
            "metadata": {
                "codex_session_id": "codex-sess-9",
                "agent_id": "agent-9",
                "prompt_overhead": {
                    "input_system_tokens_estimated": 11,
                    "input_tool_advertisement_tokens_estimated": 22,
                },
            },
        },
        logging_obj=SimpleNamespace(
            litellm_call_id="call-9",
            model_call_details={"trace_id": "trace-9"},
        ),
        litellm_call_id="call-9",
        custom_llm_provider="openai",
        stream_path="pass_through",
    )
    assert identity["codex_session_id"] == "codex-sess-9"
    assert identity["agent_id"] == "agent-9"
    assert identity["prompt_category_tokens"]["system"] == 11
    assert "do not leak" not in json.dumps(identity)


def test_authorization_allows_admin_and_permission_only():
    assert caller_may_read_session_transfer(_admin_user()) is True
    assert caller_may_read_session_transfer(_service_user()) is True
    assert caller_may_read_session_transfer(_denied_user()) is False


@pytest.mark.asyncio
async def test_endpoint_rejects_unauthorized_and_missing_filter(registry):
    with pytest.raises(HTTPException) as denied:
        await get_session_transfer_status(
            session_id="sess-canon",
            user_api_key_dict=_denied_user(),
        )
    assert denied.value.status_code == 403

    with pytest.raises(HTTPException) as missing:
        await get_session_transfer_status(user_api_key_dict=_service_user())
    assert missing.value.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_returns_content_free_status(registry):
    await registry.record_chunks(
        _identity(),
        upstream_chunks=4,
        upstream_bytes=40,
        downstream_chunks=3,
        downstream_bytes=30,
        first_upstream=True,
        first_downstream=True,
    )
    payload = await get_session_transfer_status(
        codex_session_id="codex-sess-1",
        user_api_key_dict=_service_user(),
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    transfer = payload["transfers"][0]
    assert transfer["upstream_chunk_count"] == 4
    assert transfer["downstream_chunk_count"] == 3
    assert transfer["phase"] == "response_streaming"
    assert_content_free(payload)


def test_route_is_self_managed_and_virtual_key_can_allow_it():
    assert TRANSFER_ROUTE in LiteLLMRoutes.self_managed_routes.value
    allowed = UserAPIKeyAuth(
        user_id="transcript",
        allowed_routes=[TRANSFER_ROUTE],
    )
    assert (
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route=TRANSFER_ROUTE,
            valid_token=allowed,
        )
        is True
    )
    denied = UserAPIKeyAuth(
        user_id="other",
        allowed_routes=["/health"],
    )
    with pytest.raises(HTTPException):
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route=TRANSFER_ROUTE,
            valid_token=denied,
        )


@pytest.mark.asyncio
async def test_pass_through_stream_path_updates_counters(registry):
    chunks = [b"data: one\n\n", b"data: two\n\n"]

    async def _aiter_bytes():
        for chunk in chunks:
            yield chunk

    response = MagicMock()
    response.aiter_bytes = _aiter_bytes
    logging_obj = SimpleNamespace(
        litellm_call_id="stream-call",
        model_call_details={},
        _update_completion_start_time=lambda **kwargs: None,
    )
    success_kwargs = {
        "litellm_call_id": "stream-call",
        "litellm_params": {
            "metadata": {
                "codex_session_id": "codex-stream",
                "agent_id": "agent-stream",
            }
        },
    }

    emitted = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "gpt-5.4", "metadata": {"session_id": "codex-stream"}},
        litellm_logging_obj=logging_obj,
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="https://example.test/v1/responses",
        success_handler_kwargs=success_kwargs,
        custom_llm_provider="openai",
    ):
        emitted.append(chunk)

    assert emitted == chunks
    result = await registry.query(litellm_call_id="stream-call")
    transfer = result["transfers"][0]
    assert transfer["stream_path"] == "pass_through"
    assert transfer["codex_session_id"] == "codex-stream"
    assert transfer["upstream_chunk_count"] == 2
    assert transfer["downstream_chunk_count"] == 2
    assert transfer["upstream_byte_count"] == sum(len(chunk) for chunk in chunks)
    assert transfer["phase"] == "completed"
    assert transfer["terminal_state"] == "completed"


@pytest.mark.asyncio
async def test_pass_through_timeout_marks_timed_out(registry):
    async def _aiter_bytes():
        raise httpx.ReadTimeout("slow upstream")
        yield b""  # pragma: no cover

    response = MagicMock()
    response.aiter_bytes = _aiter_bytes
    logging_obj = SimpleNamespace(
        litellm_call_id="timeout-call",
        model_call_details={},
        _update_completion_start_time=lambda **kwargs: None,
    )
    success_kwargs = {
        "litellm_call_id": "timeout-call",
        "litellm_params": {"metadata": {"codex_session_id": "codex-timeout"}},
    }
    with pytest.raises(httpx.ReadTimeout):
        async for _chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(),
            url_route="https://example.test/v1/responses",
            success_handler_kwargs=success_kwargs,
            custom_llm_provider="openai",
        ):
            pass
    result = await registry.query(litellm_call_id="timeout-call")
    transfer = result["transfers"][0]
    assert transfer["phase"] == "timed_out"
    assert transfer["timeout_kind"] == "upstream_read"
    assert transfer["error_code"] == "timeout"
    assert transfer["upstream_chunk_count"] == 0


@pytest.mark.asyncio
async def test_adapter_hook_uses_same_registry(registry):
    await publish_adapter_transfer_event(
        request_body={"metadata": {"codex_session_id": "codex-adapter"}},
        litellm_call_id="adapter-call",
        custom_llm_provider="openrouter",
        phase="response_streaming",
        extra={"upstream_chunk_count": 1, "downstream_chunk_count": 0},
    )
    result = await registry.query(codex_session_id="codex-adapter")
    transfer = result["transfers"][0]
    assert transfer["stream_path"] == "adapter"
    assert transfer["litellm_call_id"] == "adapter-call"
    assert transfer["phase"] == "response_streaming"
    assert transfer["upstream_chunk_count"] == 1
    assert transfer["downstream_chunk_count"] == 0


def test_http_endpoint_auth_via_dependency(registry):
    app = FastAPI()
    app.include_router(router)

    async def _auth_ok():
        return _service_user()

    async def _auth_denied():
        return _denied_user()

    client = TestClient(app)
    app.dependency_overrides[user_api_key_auth] = _auth_denied
    denied = client.get(TRANSFER_ROUTE, params={"session_id": "sess-canon"})
    assert denied.status_code == 403

    app.dependency_overrides[user_api_key_auth] = _auth_ok
    missing = client.get(TRANSFER_ROUTE)
    assert missing.status_code == 400
