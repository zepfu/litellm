"""D1-634: recover Responses streams that fail before the first client byte."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    _RESPONSES_PRE_COMMIT_TRANSIENT_CLASSES,
    plan_responses_pre_commit_retry,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _classify_codex_auto_agent_retryable_exhaustion,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _execute_passthrough_pre_first_byte_with_hidden_retries,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
    ResponsesStreamPreCommitFailure,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_type}\ndata: "
        + json.dumps(payload, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


def _failed_lifecycle_stream(
    *,
    code: str = "server_overloaded",
    message: str = "The server is currently overloaded. Please try again later.",
) -> list[bytes]:
    return [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "code": code,
                    "message": message,
                },
            },
        ),
        _sse(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "status": "failed",
                    "model": "gpt-5.4",
                    "output": [],
                    "error": {
                        "type": "server_error",
                        "code": code,
                        "message": message,
                    },
                },
            },
        ),
    ]


class _FakeUpstreamStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.aiter_calls = 0
        self.status_code = 200
        self.headers = httpx.Headers({"content-type": "text/event-stream"})
        self.request = httpx.Request(
            "POST",
            "https://chatgpt.com/backend-api/codex/responses",
        )

    async def aiter_bytes(self):
        self.aiter_calls += 1
        for chunk in self._chunks:
            yield chunk


def _route_kwargs() -> dict[str, Any]:
    return {
        "litellm_params": {
            "metadata": {
                "aawm_route_rollup_context": {
                    "group_header_label": "litellm#Codex[0.141.0]",
                    "incoming_endpoint": "/openai_passthrough/responses",
                    "outgoing_target": "chatgpt.com/backend-api/codex/responses",
                    "model_label": "gpt-5.4",
                    "reasoning_effort": "high",
                }
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }


@pytest.mark.asyncio
async def test_early_response_failed_skips_success_callbacks_and_normal_turn():
    chunks = _failed_lifecycle_stream()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj._update_completion_start_time = MagicMock()
    failure_called = asyncio.Event()

    async def _capture_failure(**kwargs):
        failure_called.set()

    logging_obj.async_success_handler = AsyncMock()
    logging_obj.async_failure_handler = AsyncMock(side_effect=_capture_failure)
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = _route_kwargs()
    response = _FakeUpstreamStream(chunks)

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup_turn"
    ) as record_turn, patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.emit_aawm_route_status_event"
    ) as emit_status, patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup"
    ) as record_rollup:
        emitted = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            emitted.append(chunk)
        await asyncio.wait_for(failure_called.wait(), timeout=1)

    logging_obj.async_success_handler.assert_not_awaited()
    logging_obj.async_failure_handler.assert_awaited()
    record_turn.assert_not_called()
    emit_status.assert_called()
    assert emit_status.call_args.kwargs["status"] == "Failed"
    record_rollup.assert_called()
    assert record_rollup.call_args.kwargs["status"] == "Failed"
    assert record_rollup.call_args.kwargs["turns"] == 0
    rendered = b"".join(emitted).decode("utf-8")
    assert "response.failed" in rendered
    metadata = success_handler_kwargs["litellm_params"]["metadata"]
    assert metadata["aawm_route_rollup_turn_suppressed"] is True
    assert metadata["aawm_responses_stream_failed"] is True
    assert metadata["aawm_responses_stream_failure_class"] == "server_overloaded"


@pytest.mark.asyncio
async def test_peek_holds_lifecycle_until_failed_without_downstream_commit():
    response = _FakeUpstreamStream(_failed_lifecycle_stream())
    peeked, failure = await PassThroughStreamingHandler.peek_responses_pre_commit_stream(
        response
    )
    assert failure is not None
    assert failure.error_class == "server_overloaded"
    assert failure.retryable is True
    assert failure.classification == "transient_capacity"
    assert isinstance(peeked, object)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "error_code",
        "error_message",
        "expected_error_class",
        "expected_status_code",
    ),
    [
        (
            "token_invalidated",
            "The access token has been invalidated.",
            "token_invalidated",
            401,
        ),
        (
            "invalid_request_error",
            (
                "Item with id 'rs_abc123' not found. "
                "Items are not persisted when store is set to false. "
                "Try again with store set to true."
            ),
            "openai_responses_unpersisted_item_not_found",
            400,
        ),
    ],
    ids=["token-invalidated", "unpersisted-rs-item"],
)
async def test_peek_classifies_native_codex_recovery_errors_before_commit(
    error_code: str,
    error_message: str,
    expected_error_class: str,
    expected_status_code: int,
) -> None:
    chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_recovery",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": error_code,
                    "message": error_message,
                },
            },
        ),
    ]

    peeked, failure = await PassThroughStreamingHandler.peek_responses_pre_commit_stream(
        _FakeUpstreamStream(chunks)
    )

    assert failure is not None
    assert failure.error_class == expected_error_class
    assert failure.status_code == expected_status_code
    assert failure.provider_returned is True
    http_exc = failure.as_http_exception()
    assert http_exc.status_code == expected_status_code
    assert getattr(http_exc, "_aawm_provider_returned", False) is True
    assert [chunk async for chunk in peeked.aiter_bytes()] == chunks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "error_code",
        "error_message",
        "expected_error_class",
        "raw_error_marker",
    ),
    [
        (
            "token_invalidated",
            "The access token has been invalidated.",
            "token_invalidated",
            "The access token has been invalidated.",
        ),
        (
            "invalid_request_error",
            (
                "Item with id 'rs_abc123' not found. "
                "Items are not persisted when store is set to false. "
                "Try again with store set to true."
            ),
            "openai_responses_unpersisted_item_not_found",
            "Items are not persisted when store is set to false.",
        ),
    ],
    ids=["token-invalidated", "unpersisted-rs-item"],
)
async def test_chunk_processor_terminalizes_native_recovery_once_after_commit(
    error_code: str,
    error_message: str,
    expected_error_class: str,
    raw_error_marker: str,
) -> None:
    response_chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_recovery",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "hello",
            },
        ),
        _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": error_code,
                    "message": error_message,
                },
            },
        ),
    ]
    response = _FakeUpstreamStream(response_chunks)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    success_handler_kwargs = _route_kwargs()
    success_handler_kwargs["litellm_params"]["metadata"].update(
        {
            "codex_auto_agent_selected_provider": "openai",
            "codex_auto_agent_selected_model": "gpt-5.4",
            "codex_auto_agent_selected_account_label": "account1",
            "codex_auto_agent_selected_account_hash": "hash-account-1",
            "codex_auto_agent_selected_account_lane": (
                "codex-oauth:account1:hash-account-1"
            ),
            "model_alias_label": "codex-auto-review",
            "canonical_session_identity": "session-1",
        }
    )
    finalize = AsyncMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=finalize,
    ), patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.emit_aawm_route_status_event"
    ), patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup"
    ):
        emitted = [
            chunk
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body={
                    "model": "gpt-5.4",
                    "previous_response_id": "resp-previous",
                    "stream": True,
                },
                litellm_logging_obj=logging_obj,
                endpoint_type=EndpointType.OPENAI,
                start_time=datetime.now(),
                passthrough_success_handler_obj=MagicMock(
                    spec=PassThroughEndpointLogging
                ),
                url_route="https://chatgpt.com/backend-api/codex/responses",
                custom_llm_provider="openai",
                success_handler_kwargs=success_handler_kwargs,
            )
        ]
        await asyncio.sleep(0)

    rendered = b"".join(emitted).decode("utf-8")
    assert rendered.count("event: response.failed") == 1
    assert rendered.count("data: [DONE]") == 1
    assert rendered.count('"delta":"hello"') == 1
    assert raw_error_marker not in rendered
    metadata = success_handler_kwargs["litellm_params"]["metadata"]
    assert metadata["error_class"] == expected_error_class
    assert metadata["stream_hidden_retry_safe"] is False
    assert response.aiter_calls == 1
    finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_peek_replays_substantive_prefix_then_remainder():
    chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_ok",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "hello",
            },
        ),
        _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {"id": "resp_ok", "status": "completed", "output": []},
            },
        ),
    ]
    response = _FakeUpstreamStream(chunks)
    peeked, failure = await PassThroughStreamingHandler.peek_responses_pre_commit_stream(
        response
    )
    assert failure is None
    replayed = [chunk async for chunk in peeked.aiter_bytes()]
    assert replayed == chunks


def test_no_replay_after_substantive_output():
    chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_ok",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "hello",
            },
        ),
        _sse(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_ok",
                    "status": "failed",
                    "error": {"code": "server_overloaded", "message": "overloaded"},
                },
            },
        ),
    ]
    decision, error_payload, event_type = (
        PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(chunks)
    )
    assert decision == "substantive"
    assert error_payload is None
    assert event_type == "response.output_text.delta"


def test_plan_retries_same_account_for_transient_capacity():
    first = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=1,
    )
    assert first["action"] == "retry_same_account"
    assert first["retry_same_account"] is True
    assert first["apply_account_exhaustion_cooldown"] is False
    assert first["wait_seconds"] == 10.0
    assert first["http_status"] == 503
    assert first["retryable"] is True


def test_plan_rotates_account_for_usage_limit():
    plan = plan_responses_pre_commit_retry(
        error_class="usage_limit_reached",
        same_account_transient_attempts=1,
    )
    assert plan["action"] == "rotate_account"
    assert plan["retry_same_account"] is False
    assert plan["apply_account_exhaustion_cooldown"] is True
    assert plan["wait_seconds"] == 0.0


def test_plan_returns_pre_stream_503_after_two_transient_failures():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=2,
    )
    assert plan["action"] == "pre_stream_unavailable"
    assert plan["retry_same_account"] is False
    assert plan["apply_account_exhaustion_cooldown"] is False
    assert plan["http_status"] == 503
    assert plan["retryable"] is True
    assert plan["wait_seconds"] == 10.0


def _opencode_go_empty_success_proxy_exception() -> ProxyException:
    """Match production `_raise_codex_auto_agent_empty_success_response`."""
    exc = ProxyException(
        message=(
            "Codex auto-agent OpenCode Go candidate returned an empty successful "
            "Responses payload."
        ),
        type="upstream_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_codex_auto_agent_empty_success",
                "status": "EMPTY_SUCCESS_RESPONSE",
                "type": "upstream_error",
            }
        },
    )
    return exc


def test_empty_success_502_is_not_pre_commit_transient():
    """Live Ohmypi stream=true `basic` 503s empty OpenCode Go success.

    `_raise_codex_auto_agent_empty_success_response` fail-closes with HTTP 502
    `aawm_codex_auto_agent_empty_success`. Mapping that 502 through
    `_CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES` makes
    `plan_responses_pre_commit_retry` treat emptiness as same-account
    pre-commit capacity and 503 the whole alias after two attempts.
    Empty success must leave the candidate loop instead.
    """
    exc = _opencode_go_empty_success_proxy_exception()

    classified = _classify_codex_auto_agent_retryable_exhaustion(exc)
    assert classified != "upstream_transient_internal"
    assert classified is not None
    assert classified not in _RESPONSES_PRE_COMMIT_TRANSIENT_CLASSES

    plan = plan_responses_pre_commit_retry(
        error_class=classified,
        same_account_transient_attempts=2,
    )
    assert plan["action"] not in {
        "pre_stream_unavailable",
        "retry_same_account",
    }


def test_pre_commit_failure_http_exception_is_503_with_retry_after():
    exc = ResponsesStreamPreCommitFailure(
        error_class="server_overloaded",
        classification="transient_capacity",
        retryable=True,
        pre_commit_retry_exhausted=True,
        message="server_overloaded",
    )
    http_exc = exc.as_http_exception()
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 503
    assert http_exc.headers["Retry-After"] == "10"
    assert http_exc.detail["error"]["retryable"] is True
    assert http_exc.detail["error"]["type"] == "server_overloaded"


@pytest.mark.asyncio
async def test_hidden_retry_retries_same_account_then_returns_503():
    attempts: list[int] = []

    async def operation():
        attempts.append(1)
        raise ResponsesStreamPreCommitFailure(
            error_class="server_overloaded",
            classification="transient_capacity",
            retryable=True,
            message="server_overloaded",
        )

    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    kwargs: dict[str, Any] = {}
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
        new=fake_sleep,
    ):
        with pytest.raises(ResponsesStreamPreCommitFailure) as raised:
            await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs=kwargs,
                operation_name="stream_pre_first_byte",
                operation=operation,
                caller_managed_hidden_retry=False,
            )

    assert len(attempts) == 2
    assert sleep_calls == [10.0]
    assert raised.value.pre_commit_retry_exhausted is True
    http_exc = raised.value.as_http_exception()
    assert http_exc.status_code == 503
    assert http_exc.headers["Retry-After"] == "10"


@pytest.mark.asyncio
async def test_hidden_retry_does_not_retry_usage_limit():
    async def operation():
        raise ResponsesStreamPreCommitFailure(
            error_class="usage_limit_reached",
            classification="usage_limit_reached",
            retryable=False,
            message="usage_limit_reached",
        )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._passthrough_hidden_retry_sleep",
        new=AsyncMock(),
    ) as mock_sleep:
        with pytest.raises(ResponsesStreamPreCommitFailure):
            await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs={},
                operation_name="stream_pre_first_byte",
                operation=operation,
                caller_managed_hidden_retry=False,
            )

    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_stream_still_dispatches_success():
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_success_handler = AsyncMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = _route_kwargs()
    completed_event = {
        "type": "response.completed",
        "response": {"status": "completed", "output": []},
    }
    precomputed_lines = [
        'data: {"type":"response.output_text.delta","delta":"ok"}',
        f"data: {json.dumps(completed_event)}",
    ]

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks",
        return_value={"result": {"response": "ok"}, "kwargs": {}},
    ), patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup_turn"
    ) as record_turn:
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            response=httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://chatgpt.com/backend-api/codex/responses",
                ),
            ),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4"},
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now() - timedelta(milliseconds=10),
            raw_bytes=[],
            precomputed_lines=precomputed_lines,
            end_time=datetime.now(),
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        )

    logging_obj.async_success_handler.assert_awaited_once()
    record_turn.assert_called_once()


def test_reconcile_error_and_response_failed_without_duplicate_payload():
    chunks = [
        'event: error',
        'data: {"type":"error","error":{"code":"server_overloaded","message":"overloaded"}}',
        'event: response.failed',
        'data: {"type":"response.failed","response":{"status":"failed","error":{"code":"server_overloaded","message":"overloaded"}}}',
    ]
    payload = PassThroughStreamingHandler._reconcile_responses_stream_error_payload(
        all_chunks=chunks,
        terminal_payload={
            "status": "failed",
            "error": {"code": "server_overloaded", "message": "overloaded"},
        },
    )
    assert payload is not None
    assert payload.get("code") == "server_overloaded"
    error_class, classification, retryable = (
        PassThroughStreamingHandler._classify_responses_pre_commit_error(payload)
    )
    assert error_class == "server_overloaded"
    assert classification == "transient_capacity"
    assert retryable is True


def test_inspect_pre_commit_chunks_does_not_raise_on_truncated_utf8_tail():
    """T-4: inspect/peek must treat mid-codepoint SSE tails as incomplete text,
    not dump UnicodeDecodeError from _chunk_lines/finish()."""
    complete = _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": "resp_ok",
                "status": "in_progress",
                "model": "gpt-5.4",
                "output": [],
            },
        },
    )
    chunks = [complete + b"\xe2\x82"]
    decision, error_payload, event_type = (
        PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(chunks)
    )
    assert decision == "lifecycle"
    assert error_payload is None
    assert event_type is None


def test_inspect_pre_commit_chunks_still_classifies_valid_utf8_sse():
    chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_ok",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "hello",
            },
        ),
    ]
    decision, error_payload, event_type = (
        PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(chunks)
    )
    assert decision == "substantive"
    assert error_payload is None
    assert event_type == "response.output_text.delta"


@pytest.mark.asyncio
async def test_peek_does_not_raise_on_truncated_utf8_at_end_of_stream():
    """T-4: a lone truncated multi-byte sequence as the last peeked chunk must
    not raise UnicodeDecodeError from peek_responses_pre_commit_stream."""
    chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_ok",
                    "status": "in_progress",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
        b"\xc3",
    ]
    response = _FakeUpstreamStream(chunks)
    peeked, failure = await PassThroughStreamingHandler.peek_responses_pre_commit_stream(
        response
    )
    assert failure is None
    replayed = [chunk async for chunk in peeked.aiter_bytes()]
    assert replayed == chunks
