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

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    plan_responses_pre_commit_retry,
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
        self.status_code = 200
        self.headers = httpx.Headers({"content-type": "text/event-stream"})
        self.request = httpx.Request(
            "POST",
            "https://chatgpt.com/backend-api/codex/responses",
        )

    async def aiter_bytes(self):
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
