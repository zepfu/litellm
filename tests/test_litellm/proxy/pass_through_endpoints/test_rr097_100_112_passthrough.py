"""RR-097 / RR-100 / RR-112: passthrough body rewrite, transfer terminal, Grok peek.

Collection-safe against the RR-093 adapter-runtime import cycle: package
``__init__`` is stubbed so owned modules can load. Product code is not changed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers, QueryParams

from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,
    is_malformed_grok_literal_tool_label_transcript_text,
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    composer_repair,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.aawm_session_transfer.registry import (
    STALE_AFTER_SECONDS,
    SessionTransferRegistry,
    reset_session_transfer_registry,
    set_session_transfer_registry_override,
)
from litellm.proxy.aawm_session_transfer.schema import TERMINAL_PHASE_SET
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    provider_shaping as _aawm_provider_shaping,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import streaming as _aawm_alias_streaming
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _tool_definition_name,
    _tool_definition_parameters,
)
from litellm.proxy.pass_through_endpoints.providers.grok import (
    direct_responses_validation as grok_direct,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _install_cycle_safe_package_stubs() -> None:
    """Stub eager package inits so RR-093 does not block this file's collection."""

    runtime_name = "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime"
    if runtime_name not in sys.modules or not hasattr(
        sys.modules[runtime_name], "__path__"
    ):
        runtime = ModuleType(runtime_name)
        runtime.__path__ = [
            str(_REPO_ROOT / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime")
        ]
        runtime.__package__ = runtime_name
        runtime.install = lambda host_globals: None  # type: ignore[method-assign]
        runtime.install_wave6f = lambda host_globals: None  # type: ignore[method-assign]
        sys.modules[runtime_name] = runtime

    alias_name = "litellm.proxy.pass_through_endpoints.aawm_alias_routing"
    if alias_name not in sys.modules or not hasattr(sys.modules[alias_name], "__path__"):
        alias = ModuleType(alias_name)
        alias.__path__ = [
            str(_REPO_ROOT / "litellm/proxy/pass_through_endpoints/aawm_alias_routing")
        ]
        alias.__package__ = alias_name
        sys.modules[alias_name] = alias


_install_cycle_safe_package_stubs()

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (  # noqa: E402
    payload_validation as _payload_validation,
    request_build as _request_build,
    sse as _sse,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    pass_through_request,
)

ZWSP = "\u200b"
STALE_CONTENT_LENGTH = "1"
RR097_ERP_CIPHERTEXT = "gAAAAABrr097ORIGINAL_REASONING_BYTES=="


def _normalize_openai_function_tool_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    normalized = dict(parameters)
    if normalized.get("type") is None:
        normalized["type"] = "object"
    return normalized


def _advertised_tools_index(request_body: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(request_body, dict):
        return {}
    tools_index: dict[str, dict[str, Any]] = {}
    for source in (request_body.get("tools"), request_body.get("functions")):
        if not isinstance(source, list):
            continue
        for tool in source:
            if not isinstance(tool, dict):
                continue
            tool_name = _tool_definition_name(tool)
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            parameters = _tool_definition_parameters(tool)
            tools_index[tool_name] = _normalize_openai_function_tool_parameters(
                parameters
            )
    return tools_index


def _repair_runtime() -> composer_repair.Runtime:
    return composer_repair.Runtime(
        decode_json_prefix=_aawm_provider_shaping.decode_json_prefix,
        strip_text_spans=_request_build._strip_text_spans,
        build_advertised_function_tools_index=_advertised_tools_index,
        validate_tool_arguments=_request_build._validate_tool_arguments_against_openai_parameters,
        is_malformed_composer_literal_text=is_malformed_composer_call_literal_text,
        is_malformed_tool_call_text_output=_payload_validation._is_codex_auto_agent_malformed_tool_call_text_output,
    )


def _grok_host(
    *,
    max_chunks: int = 5000,
    max_bytes: int = 8 * 1024 * 1024,
) -> SimpleNamespace:
    def _decode_http_response_body(body: Any) -> str:
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        if isinstance(body, str):
            return body
        return str(body)

    def _intake_context(request, request_body, **kwargs):
        del request
        advertised = _advertised_tools_index(
            request_body if isinstance(request_body, dict) else None
        )
        return {
            "adapter": grok_direct.DIRECT_GROK_RESPONSES_ADAPTER,
            "provider": kwargs.get("provider") or "xai",
            "model_alias": kwargs.get("model_alias"),
            "advertised_tool_count": len(advertised),
        }

    return SimpleNamespace(
        _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS=max_chunks,
        _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES=max_bytes,
        _aawm_alias_streaming=_aawm_alias_streaming,
        _decode_http_response_body=_decode_http_response_body,
        _get_anthropic_grok_composer_repair_runtime=_repair_runtime,
        _is_codex_auto_agent_malformed_tool_call_text_output=(
            _payload_validation._is_codex_auto_agent_malformed_tool_call_text_output
        ),
        _build_malformed_tool_call_intake_context=_intake_context,
        _build_advertised_openai_function_tools_index=_advertised_tools_index,
        _collect_responses_response_from_stream=_collect_completed_response_from_stream,
        _responses_sse_from_repaired_response_body=_sse._responses_sse_from_repaired_response_body,
    )


async def _collect_completed_response_from_stream(
    response: StreamingResponse,
    event_summaries: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    rendered_parts: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            rendered_parts.append(chunk.decode("utf-8", errors="replace"))
        else:
            rendered_parts.append(str(chunk))
    rendered = "".join(rendered_parts)
    completed: Optional[dict[str, Any]] = None
    for line in rendered.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if event_summaries is not None and len(event_summaries) < 50:
            event_summaries.append({"type": parsed.get("type")})
        if parsed.get("type") == "response.completed" and isinstance(
            parsed.get("response"), dict
        ):
            completed = parsed["response"]
    if completed is None:
        raise ValueError("no response.completed event")
    return completed


def _patch_grok_host(*, max_chunks: int = 5000, max_bytes: int = 8 * 1024 * 1024):
    return patch.object(
        grok_direct,
        "_passthrough_host",
        return_value=_grok_host(max_chunks=max_chunks, max_bytes=max_bytes),
    )


class _FakeDualCache:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def async_get_cache(self, key, **kwargs):
        return self.values.get(key)

    async def async_set_cache(self, key, value, ttl=None, **kwargs):
        self.values[key] = value

    async def async_batch_get_cache(self, keys, **kwargs):
        return [self.values.get(key) for key in keys]


@pytest.fixture
def transfer_clock() -> dict[str, datetime]:
    return {"now": datetime(2026, 8, 21, 18, 0, tzinfo=timezone.utc)}


@pytest.fixture
def transfer_registry(transfer_clock: dict[str, datetime]):
    registry = SessionTransferRegistry(
        cache=_FakeDualCache(),
        source_instance="rr097-100-112-tester",
        now_fn=lambda: transfer_clock["now"],
    )
    set_session_transfer_registry_override(registry)
    yield registry
    reset_session_transfer_registry()


def _content_length_header(response: Response) -> Optional[str]:
    for name, value in response.headers.items():
        if name.lower() == "content-length":
            return str(value)
    return None


def _assert_rewritten_content_length(response: Response, *, original_len: int) -> None:
    new_len = len(response.body)
    assert new_len != original_len
    header = _content_length_header(response)
    if header is None:
        return
    assert header != STALE_CONTENT_LENGTH
    assert header != str(original_len)
    assert int(header) == new_len


def _passthrough_request() -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = "http://localhost:4000/openai/v1/responses"
    request.headers = Headers({"content-type": "application/json"})
    request.query_params = QueryParams({})
    request.cookies = {}
    request.body = AsyncMock(return_value=b"{}")
    request.state = SimpleNamespace()
    return request


def _user_api_key_dict() -> MagicMock:
    user = MagicMock()
    user.api_key = "test-api-key"
    user.key_alias = "test-alias"
    user.user_email = "test@example.com"
    user.user_id = "test-user-id"
    user.team_id = "test-team-id"
    user.org_id = "test-org-id"
    user.team_alias = "test-team-alias"
    user.end_user_id = "test-end-user-id"
    user.request_route = "/openai/v1/responses"
    user.spend = 0.0
    return user


def _sanitize_watermark_config() -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
        load_text_watermark_config,
    )

    return load_text_watermark_config(
        {
            "mode": "sanitize",
            "directions": {"request": False, "response": True},
            "unicode": {
                "enabled": True,
                "policy": "conservative",
                "normalize_spaces": True,
                "nfkc": False,
            },
            "removal": {
                "enabled": True,
                "stream_policy": "audit_only",
                "on_unremovable": "allow",
            },
            "statistical_detectors": [],
        }
    )


def _watermarked_responses_body() -> dict[str, Any]:
    visible = f"hello{ZWSP}world"
    return {
        "id": "resp_rr097_watermark",
        "object": "response",
        "status": "completed",
        "output_text": visible,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": visible}],
            }
        ],
    }


def _encrypted_reasoning_body() -> dict[str, Any]:
    return {
        "id": "resp_rr097_erp",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_rr097",
                "encrypted_content": RR097_ERP_CIPHERTEXT,
            }
        ],
    }


def _search_replace_tool_request_body() -> dict[str, Any]:
    return {
        "model": "grok-build",
        "input": "Apply the advertised search_replace call.",
        "stream": False,
        "tools": [
            {
                "type": "function",
                "name": "search_replace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                    "additionalProperties": False,
                },
            }
        ],
    }


def _search_replace_literal_text() -> tuple[str, str, dict[str, str]]:
    payload = {
        "file_path": "/tmp/rr112.py",
        "old_string": "alpha",
        "new_string": "alpha-fixed",
    }
    call_id = "call-rr112-search_replace-1"
    preface = "I'll update the advertised file now."
    text = (
        f"{preface}\n"
        "[Context note - prior assistant step; not an executable tool invocation]\n"
        "Tool label: search_replace\n"
        f"Correlation ref: {call_id}\n"
        f"Input payload: {json.dumps(payload, ensure_ascii=False)}"
    )
    return text, call_id, payload


def _literal_tool_response_payload(literal_text: str) -> dict[str, Any]:
    return {
        "id": "resp_rr112_direct_grok",
        "object": "response",
        "status": "completed",
        "model": "grok-build",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": literal_text}],
            }
        ],
    }


def _direct_grok_request(*, endpoint: str = "v1/responses") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = f"http://localhost:4000/grok/{endpoint}"
    request.headers = Headers(
        {
            "authorization": "Bearer oidc-token",
            "x-litellm-api-key": "litellm-test-key",
            "content-type": "application/json",
        }
    )
    request.query_params = QueryParams({})
    request.cookies = {}
    return request


async def _collect_stream_text(response: StreamingResponse) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


def _sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_type}\n"
        "data: "
        + json.dumps({"type": event_type, **payload}, ensure_ascii=False)
        + "\n\n"
    ).encode("utf-8")


def _assert_malformed_reject(exc: ProxyException) -> None:
    assert str(exc.code) == "502"
    detail = exc.detail
    assert isinstance(detail, dict)
    error = detail.get("error")
    assert isinstance(error, dict)
    assert error["code"] == "aawm_auto_agent_malformed_tool_call_text"
    failure_kind = detail.get("failure_kind") or error.get("failure_kind")
    assert failure_kind == "malformed_tool_call"


async def _invoke_nonstream_pass_through(
    *,
    request_body: dict[str, Any],
    upstream: httpx.Response,
    call_id: str,
    watermark_config: Any = None,
    custom_llm_provider: str = "openai",
) -> Any:
    request = _passthrough_request()
    logging_obj = MagicMock()
    logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
    logging_obj.post_call_failure_hook = AsyncMock()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
                return_value=MagicMock(client=MagicMock()),
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                "HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
                new=AsyncMock(return_value=upstream),
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj",
                logging_obj,
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                "pass_through_endpoint_logging.pass_through_async_success_handler",
                new=AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                "ProxyBaseLLMRequestProcessing.get_custom_headers",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                "_aawm_session_owner_pre_send_guard",
                new=AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                "_aawm_session_owner_on_upstream_result",
                new=AsyncMock(),
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.capture_passthrough_shape",
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.uuid.uuid4",
                return_value=call_id,
            )
        )
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                "_direct_capture_xai_passthrough_failure",
                new=AsyncMock(),
            )
        )
        if watermark_config is not None:
            stack.enter_context(
                patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
                    "_get_runtime_text_watermark_config",
                    return_value=watermark_config,
                )
            )
        return await pass_through_request(
            request=request,
            target="https://api.openai.com/v1/responses",
            custom_headers={"authorization": "Bearer test"},
            user_api_key_dict=_user_api_key_dict(),
            custom_body=request_body,
            custom_llm_provider=custom_llm_provider,
            stream=False,
            caller_managed_hidden_retry=True,
        )


def _count_terminal_registry_writes(registry: SessionTransferRegistry) -> list[str]:
    writes: list[str] = []
    original_finalize = registry.finalize
    original_mark = registry.mark_phase

    async def _finalize(identity, phase, extra=None):
        writes.append(f"finalize:{phase}")
        return await original_finalize(identity, phase, extra)

    async def _mark_phase(identity, phase, extra=None):
        if str(phase) in TERMINAL_PHASE_SET:
            writes.append(f"mark_phase:{phase}")
        return await original_mark(identity, phase, extra)

    registry.finalize = _finalize  # type: ignore[method-assign]
    registry.mark_phase = _mark_phase  # type: ignore[method-assign]
    return writes


async def _assert_inactive_terminal_without_stale_expiry(
    registry: SessionTransferRegistry,
    *,
    call_id: str,
    clock: dict[str, datetime],
    expected_phase: str,
) -> dict[str, Any]:
    record = await registry.get_by_call_id(call_id)
    assert record is not None, (
        f"RR-100: expected an immediate session-transfer record for {call_id}"
    )
    assert record["phase"] == expected_phase
    assert record["terminal_state"] == expected_phase
    assert record["freshness"] == "terminal"
    assert record["active"] is False
    assert record["stale"] is False, (
        "RR-100: inactivity must come from a terminal phase, not stale-record expiry"
    )
    assert record.get("finalized_at")
    clock["now"] = clock["now"] + timedelta(seconds=STALE_AFTER_SECONDS + 90)
    later = await registry.get_by_call_id(call_id)
    assert later is not None
    assert later["phase"] == expected_phase
    assert later["terminal_state"] == expected_phase
    assert later["freshness"] == "terminal"
    assert later["active"] is False
    assert later["stale"] is False
    return record


@pytest.mark.asyncio
async def test_rr097_watermark_sanitize_drops_stale_content_length():
    body = _watermarked_responses_body()
    original = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    upstream = httpx.Response(
        status_code=200,
        content=original,
        headers={
            "content-type": "application/json",
            "content-length": STALE_CONTENT_LENGTH,
            "x-request-id": "rr097-watermark",
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    response = await _invoke_nonstream_pass_through(
        request_body={"model": "gpt-5.4", "input": "hi"},
        upstream=upstream,
        call_id="rr097-watermark-call",
        watermark_config=_sanitize_watermark_config(),
    )

    assert isinstance(response, Response)
    rendered = response.body.decode("utf-8")
    assert ZWSP not in rendered
    assert "hello" in rendered and "world" in rendered
    _assert_rewritten_content_length(response, original_len=len(original))
    assert response.headers.get("x-request-id") == "rr097-watermark"


@pytest.mark.asyncio
async def test_rr097_reasoning_stamp_drops_stale_content_length():
    body = _encrypted_reasoning_body()
    original = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    upstream = httpx.Response(
        status_code=200,
        content=original,
        headers={
            "content-type": "application/json",
            "content-length": STALE_CONTENT_LENGTH,
            "x-request-id": "rr097-erp",
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    response = await _invoke_nonstream_pass_through(
        request_body={"model": "gpt-5.4", "input": "hi"},
        upstream=upstream,
        call_id="rr097-erp-call",
        custom_llm_provider="openai",
    )

    assert isinstance(response, Response)
    stamped = json.loads(response.body)
    item = stamped["output"][0]
    assert item["type"] == "reasoning"
    assert item.get("aawm_encrypted_reasoning_provenance")
    assert str(item.get("encrypted_content") or "").startswith("aawm_erp:")
    _assert_rewritten_content_length(response, original_len=len(original))
    assert response.headers.get("x-request-id") == "rr097-erp"


@pytest.mark.asyncio
async def test_rr097_direct_grok_json_repair_drops_stale_content_length():
    literal_text, call_id, payload = _search_replace_literal_text()
    original_payload = _literal_tool_response_payload(literal_text)
    original = json.dumps(original_payload).encode("utf-8")
    upstream = Response(
        content=original,
        media_type="application/json",
        status_code=200,
        headers={
            "content-length": STALE_CONTENT_LENGTH,
            "x-request-id": "rr097-grok",
        },
    )

    with _patch_grok_host():
        response = await grok_direct.validate_direct_grok_responses_payload(
            upstream,
            request=_direct_grok_request(),
            request_body=_search_replace_tool_request_body(),
            endpoint="v1/responses",
        )

    assert isinstance(response, Response)
    repaired = json.loads(response.body)
    assert "Tool label:" not in json.dumps(repaired)
    function_calls = [
        item
        for item in repaired.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    assert len(function_calls) == 1
    assert function_calls[0]["call_id"] == call_id
    assert json.loads(function_calls[0]["arguments"]) == payload
    _assert_rewritten_content_length(response, original_len=len(original))
    assert response.headers.get("x-request-id") == "rr097-grok"


@pytest.mark.asyncio
async def test_rr097_reconstructed_grok_stream_does_not_copy_stale_content_length():
    literal_text, call_id, payload = _search_replace_literal_text()
    response_body = _literal_tool_response_payload(literal_text)

    async def _chunks():
        yield _sse_event("response.completed", {"response": response_body})

    upstream = StreamingResponse(
        _chunks(),
        media_type="text/event-stream",
        status_code=200,
        headers={
            "content-type": "text/event-stream",
            "content-length": STALE_CONTENT_LENGTH,
            "x-request-id": "rr097-grok-stream",
        },
    )
    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True

    with _patch_grok_host():
        response = await grok_direct.validate_direct_grok_responses_payload(
            upstream,
            request=_direct_grok_request(),
            request_body=request_body,
            endpoint="v1/responses",
        )

    assert isinstance(response, StreamingResponse)
    rendered = await _collect_stream_text(response)
    assert '"type": "function_call"' in rendered
    assert call_id in rendered
    assert payload["file_path"] in rendered
    header = None
    for name, value in response.headers.items():
        if name.lower() == "content-length":
            header = str(value)
            break
    assert header != STALE_CONTENT_LENGTH, (
        "RR-097: reconstructed Grok SSE copied stale content-length from upstream"
    )
    assert response.headers.get("x-request-id") == "rr097-grok-stream"


@pytest.mark.asyncio
async def test_rr100_nonstream_success_terminalizes_immediately(
    transfer_registry: SessionTransferRegistry,
    transfer_clock: dict[str, datetime],
):
    writes = _count_terminal_registry_writes(transfer_registry)
    body = {"id": "resp_rr100_ok", "status": "completed", "output": []}
    content = json.dumps(body).encode("utf-8")
    upstream = httpx.Response(
        status_code=200,
        content=content,
        headers={"content-type": "application/json", "content-length": str(len(content))},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    call_id = "rr100-nonstream-success"

    response = await _invoke_nonstream_pass_through(
        request_body={"model": "gpt-5.4", "input": "hi", "session_id": "sess-rr100"},
        upstream=upstream,
        call_id=call_id,
    )

    assert isinstance(response, Response)
    await _assert_inactive_terminal_without_stale_expiry(
        transfer_registry,
        call_id=call_id,
        clock=transfer_clock,
        expected_phase="completed",
    )
    assert writes == ["finalize:completed"]


@pytest.mark.asyncio
async def test_rr100_nonstream_handled_error_terminalizes_immediately(
    transfer_registry: SessionTransferRegistry,
    transfer_clock: dict[str, datetime],
):
    writes = _count_terminal_registry_writes(transfer_registry)
    error_body = b'{"error":"upstream failed"}'
    upstream = httpx.Response(
        status_code=429,
        content=error_body,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    call_id = "rr100-nonstream-error"

    with pytest.raises(ProxyException) as exc_info:
        await _invoke_nonstream_pass_through(
            request_body={
                "model": "gpt-5.4",
                "input": "hi",
                "session_id": "sess-rr100-error",
            },
            upstream=upstream,
            call_id=call_id,
        )

    assert str(exc_info.value.code) == "429"
    await _assert_inactive_terminal_without_stale_expiry(
        transfer_registry,
        call_id=call_id,
        clock=transfer_clock,
        expected_phase="failed",
    )
    assert writes == ["finalize:failed"]


def test_rr112_direct_grok_literal_repair_documents_bounded_best_effort_window():
    docs = "\n".join(
        part
        for part in (
            inspect.getdoc(grok_direct),
            inspect.getdoc(grok_direct._validate_direct_grok_streaming_response),
        )
        if part
    ).lower()
    assert "peek window" in docs
    assert "forward" in docs
    assert "ceiling" in docs
    assert "best-effort" in docs or "best effort" in docs


@pytest.mark.asyncio
async def test_rr112_marker_inside_peek_window_repairs_literal_tool_call():
    literal_text, call_id, payload = _search_replace_literal_text()
    response_body = _literal_tool_response_payload(literal_text)
    chunks = [
        _sse_event(
            "response.output_text.delta",
            {"delta": "I'll update the advertised file now.\n"},
        ),
        _sse_event(
            "response.output_text.delta",
            {
                "delta": (
                    "Tool label: search_replace\n"
                    f"Correlation ref: {call_id}\n"
                    f"Input payload: {json.dumps(payload, ensure_ascii=False)}"
                )
            },
        ),
        _sse_event("response.completed", {"response": response_body}),
    ]

    async def _chunks():
        for chunk in chunks:
            yield chunk

    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True
    with _patch_grok_host(max_chunks=8):
        response = await grok_direct.validate_direct_grok_responses_payload(
            StreamingResponse(_chunks(), media_type="text/event-stream"),
            request=_direct_grok_request(),
            request_body=request_body,
            endpoint="v1/responses",
        )

    assert isinstance(response, StreamingResponse)
    rendered = await _collect_stream_text(response)
    assert "Tool label:" not in rendered
    assert '"type": "function_call"' in rendered
    assert call_id in rendered
    assert "event: response.completed" in rendered


@pytest.mark.asyncio
async def test_rr112_marker_after_peek_window_is_forwarded_not_repaired():
    literal_text, _call_id, _payload = _search_replace_literal_text()
    response_body = _literal_tool_response_payload(literal_text)
    first = _sse_event(
        "response.output_text.delta",
        {"delta": "ordinary preface with no tool marker yet"},
    )
    late_marker = _sse_event(
        "response.output_text.delta",
        {"delta": "Tool label: search_replace\nCorrelation ref: call-late\n"},
    )
    completed = _sse_event("response.completed", {"response": response_body})
    release = asyncio.Event()

    async def _chunks():
        yield first
        await release.wait()
        yield late_marker
        yield completed

    async def _release_later() -> None:
        await asyncio.sleep(0.05)
        release.set()

    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True
    releaser = asyncio.create_task(_release_later())
    try:
        with _patch_grok_host(max_chunks=1):
            response = await asyncio.wait_for(
                grok_direct.validate_direct_grok_responses_payload(
                    StreamingResponse(_chunks(), media_type="text/event-stream"),
                    request=_direct_grok_request(),
                    request_body=request_body,
                    endpoint="v1/responses",
                ),
                timeout=1.0,
            )
        assert isinstance(response, StreamingResponse)
        release.set()
        rendered = await _collect_stream_text(response)
    finally:
        release.set()
        releaser.cancel()

    assert "Tool label: search_replace" in rendered
    assert '"type": "function_call"' not in rendered


@pytest.mark.asyncio
async def test_rr112_collection_ceiling_rejects_marked_stream(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")

    marker = _sse_event(
        "response.output_text.delta",
        {
            "delta": (
                "Tool label: search_replace\n"
                "Correlation ref: call-rr112-ceiling\n"
            )
        },
    )
    filler = _sse_event(
        "response.output_text.delta",
        {"delta": "still generating more literal tool text"},
    )
    release = asyncio.Event()

    async def _chunks():
        yield marker
        await release.wait()
        for _ in range(8):
            yield filler

    async def _release_later() -> None:
        await asyncio.sleep(0.05)
        release.set()

    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True
    releaser = asyncio.create_task(_release_later())
    try:
        with _patch_grok_host(max_chunks=3):
            with pytest.raises(ProxyException) as exc_info:
                response = await asyncio.wait_for(
                    grok_direct.validate_direct_grok_responses_payload(
                        StreamingResponse(_chunks(), media_type="text/event-stream"),
                        request=_direct_grok_request(),
                        request_body=request_body,
                        endpoint="v1/responses",
                    ),
                    timeout=1.0,
                )
                if isinstance(response, StreamingResponse):
                    release.set()
                    await _collect_stream_text(response)
            _assert_malformed_reject(exc_info.value)
    finally:
        release.set()
        releaser.cancel()
