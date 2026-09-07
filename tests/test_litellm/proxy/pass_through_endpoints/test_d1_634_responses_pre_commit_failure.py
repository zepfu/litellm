"""D1-634: recover Responses streams that fail before the first client byte."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    _RESPONSES_PRE_COMMIT_TRANSIENT_CLASSES,
    _is_openai_alpha_capacity_retry_enabled,
    plan_responses_pre_commit_retry,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _classify_codex_auto_agent_retryable_exhaustion,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _execute_passthrough_pre_first_byte_with_hidden_retries,
    _is_openai_alpha_capacity_retry_target,
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


class _ClosableAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.close_calls = 0

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.close_calls += 1


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


def _responses_request(body: dict[str, Any]) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/openai_passthrough/v1/responses",
            "raw_path": b"/openai_passthrough/v1/responses",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )
    request.scope["parsed_body"] = (tuple(body.keys()), body)
    return request


def _direct_codex_selection() -> dict[str, Any]:
    candidate = {
        "provider": "openai",
        "model": "gpt-5.4",
        "route_family": "codex_responses",
        "codex_oauth_account_label": "account1",
        "codex_oauth_account_hash": "hash-account-1",
        "codex_oauth_lane_key": "codex-oauth:account1:hash-account-1",
    }
    return {
        "candidate": candidate,
        "lane_key": candidate["codex_oauth_lane_key"],
        "cooldown_key": "openai:gpt-5.4:codex-oauth:account1:hash-account-1",
        "request_mode": "ordinary_continuation",
        "canonical_session_identity": "session-1",
        "session_owner_identity": "session-1",
    }


async def _run_direct_responses_runtime_case(
    body: dict[str, Any],
    base_handler: Any,
    *,
    redispatch_side_effect: Any = None,
) -> tuple[list[dict[str, Any]], AsyncMock, Any, BaseException | None]:
    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        codex_oauth,
        session_affinity,
    )

    request = _responses_request(body)
    seen_bodies: list[dict[str, Any]] = []

    async def _get_body(current_request: Request) -> dict[str, Any]:
        _keys, current_body = current_request.scope["parsed_body"]
        body_copy = dict(current_body)
        if isinstance(current_body.get("input"), list):
            body_copy["input"] = list(current_body["input"])
        seen_bodies.append(body_copy)
        return current_body

    selected_auth = SimpleNamespace(
        headers={
            "Authorization": "Bearer server-token-account1",
            "ChatGPT-Account-Id": "server-account1",
        }
    )
    selection = _direct_codex_selection()
    mock_bind = AsyncMock(return_value=(selected_auth, selection, body))

    with patch.object(lpe, "get_request_body", new=_get_body), patch.object(
        lpe,
        "_resolve_codex_auto_agent_alias_model",
        return_value=None,
    ), patch.object(
        lpe, "_is_oa_xai_request_body", return_value=False
    ), patch.object(
        lpe, "_is_grok_native_oauth_request_body", return_value=False
    ), patch.object(
        lpe, "_should_use_direct_codex_oauth_inventory", return_value=True
    ), patch.object(
        codex_oauth,
        "select_and_bind_direct_codex_oauth_inventory",
        new=mock_bind,
    ), patch.object(
        lpe.BaseOpenAIPassThroughHandler,
        "_base_openai_pass_through_handler",
        new=base_handler,
    ), patch.object(
        session_affinity,
        "raise_session_owner_redispatch_required",
        side_effect=redispatch_side_effect,
    ) as mock_redispatch:
        try:
            response = await lpe.openai_proxy_route(
                endpoint="v1/responses",
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=object(),  # type: ignore[arg-type]
            )
        except BaseException as exc:
            return seen_bodies, mock_redispatch, None, exc

    return seen_bodies, mock_redispatch, response, None


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
        "fragmented",
    ),
    [
        (
            "token_invalidated",
            "The access token has been invalidated.",
            "token_invalidated",
            401,
            False,
        ),
        (
            "token_invalidated",
            "The access token has been invalidated.",
            "token_invalidated",
            401,
            True,
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
            False,
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
            True,
        ),
    ],
    ids=[
        "token-invalidated",
        "token-invalidated-fragmented",
        "unpersisted-rs-item",
        "unpersisted-rs-item-fragmented",
    ],
)
async def test_peek_classifies_native_codex_recovery_errors_before_commit(
    error_code: str,
    error_message: str,
    expected_error_class: str,
    expected_status_code: int,
    fragmented: bool,
) -> None:
    error_chunk = _sse(
        "error",
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": error_code,
                "message": error_message,
            },
        },
    )
    if fragmented:
        split_at = error_chunk.index(b'"message"') + len(b'"message"')
        error_chunks = [error_chunk[:split_at], error_chunk[split_at:]]
        partial_decision, _, _ = (
            PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(
                [error_chunks[0]]
            )
        )
        assert partial_decision != "failed"
    else:
        error_chunks = [error_chunk]

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
        *error_chunks,
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
        "fragmented",
    ),
    [
        (
            "token_invalidated",
            "The access token has been invalidated.",
            "token_invalidated",
            "The access token has been invalidated.",
            False,
        ),
        (
            "token_invalidated",
            "The access token has been invalidated.",
            "token_invalidated",
            "The access token has been invalidated.",
            True,
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
            False,
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
            True,
        ),
    ],
    ids=[
        "token-invalidated",
        "token-invalidated-fragmented",
        "unpersisted-rs-item",
        "unpersisted-rs-item-fragmented",
    ],
)
async def test_chunk_processor_terminalizes_native_recovery_once_after_commit(
    error_code: str,
    error_message: str,
    expected_error_class: str,
    raw_error_marker: str,
    fragmented: bool,
) -> None:
    error_chunk = _sse(
        "error",
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": error_code,
                "message": error_message,
            },
        },
    )
    if fragmented:
        marker_end = error_chunk.index(raw_error_marker.encode()) + len(
            raw_error_marker
        )
        second_split = marker_end + max(1, (len(error_chunk) - marker_end) // 2)
        error_chunks = [
            error_chunk[:marker_end],
            error_chunk[marker_end:second_split],
            error_chunk[second_split:],
        ]
    else:
        error_chunks = [error_chunk]

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
    ]
    response_chunks.extend(error_chunks)
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
    assert error_message not in rendered
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
        same_account_transient_attempts=0,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert first["action"] == "retry_same_account"
    assert first["retry_same_account"] is True
    assert first["apply_account_exhaustion_cooldown"] is False
    # Progressive schedule: attempt 0 -> 15s
    assert first["wait_seconds"] == 15.0
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
    """Deadline exhaustion replaces the old fixed-2-attempt-then-503 model."""
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        elapsed_seconds=7195.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=7200.0),
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "deadline_exhausted"
    assert plan["retry_same_account"] is False
    assert plan["apply_account_exhaustion_cooldown"] is False
    assert plan["http_status"] == 503
    assert plan["retryable"] is True
    assert plan["wait_seconds"] == 0.0


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


# ---------------------------------------------------------------------------
# OPENAI-054: progressive capacity retry schedule and deadline
# ---------------------------------------------------------------------------

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.retry import (
    OpenAIAlphaCapacityRetryBudget,
    openai_alpha_capacity_retry_wait_seconds,
    openai_alpha_capacity_retry_within_deadline,
)


def test_progressive_schedule_returns_15_30_60_120_240_then_repeats():
    assert openai_alpha_capacity_retry_wait_seconds(0) == 15.0
    assert openai_alpha_capacity_retry_wait_seconds(1) == 30.0
    assert openai_alpha_capacity_retry_wait_seconds(2) == 60.0
    assert openai_alpha_capacity_retry_wait_seconds(3) == 120.0
    assert openai_alpha_capacity_retry_wait_seconds(4) == 240.0
    assert openai_alpha_capacity_retry_wait_seconds(5) == 240.0
    assert openai_alpha_capacity_retry_wait_seconds(6) == 240.0
    assert openai_alpha_capacity_retry_wait_seconds(10) == 240.0
    assert openai_alpha_capacity_retry_wait_seconds(100) == 240.0


def test_negative_attempt_number_clamped_to_zero():
    assert openai_alpha_capacity_retry_wait_seconds(-1) == 15.0
    assert openai_alpha_capacity_retry_wait_seconds(-100) == 15.0


def test_deadline_allows_retry_when_within_budget():
    assert openai_alpha_capacity_retry_within_deadline(
        elapsed_seconds=0.0,
        next_wait_seconds=15.0,
        deadline_seconds=7200.0,
    ) is True


def test_deadline_rejects_retry_when_exceeded():
    assert openai_alpha_capacity_retry_within_deadline(
        elapsed_seconds=7200.0,
        next_wait_seconds=1.0,
        deadline_seconds=7200.0,
    ) is False


def test_deadline_rejects_when_projected_exceeds():
    assert openai_alpha_capacity_retry_within_deadline(
        elapsed_seconds=7190.0,
        next_wait_seconds=15.0,
        deadline_seconds=7200.0,
    ) is False


def test_deadline_rejects_when_zero():
    assert openai_alpha_capacity_retry_within_deadline(
        elapsed_seconds=0.0,
        next_wait_seconds=1.0,
        deadline_seconds=0.0,
    ) is False


def test_deadline_rejects_when_negative():
    assert openai_alpha_capacity_retry_within_deadline(
        elapsed_seconds=0.0,
        next_wait_seconds=1.0,
        deadline_seconds=-1.0,
    ) is False


def test_plan_progressive_schedule_first_retry_15s():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["retry_same_account"] is True
    assert plan["wait_seconds"] == 15.0
    assert plan["retryable"] is True


def test_plan_progressive_schedule_second_retry_30s():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=1,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 30.0


def test_plan_progressive_schedule_third_retry_60s():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=2,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 60.0


def test_plan_progressive_schedule_fourth_retry_120s():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=3,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 120.0


def test_plan_progressive_schedule_fifth_retry_240s():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=4,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 240.0


def test_plan_progressive_schedule_sixth_retry_repeats_240s():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=5,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 240.0


def test_plan_deadline_exhausted_when_projected_exceeds():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        elapsed_seconds=7190.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=7200.0),
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "deadline_exhausted"
    assert plan["retry_same_account"] is False
    assert plan["retryable"] is True


def test_plan_no_deadline_when_already_exceeded():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=4,
        elapsed_seconds=7205.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=7200.0),
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "deadline_exhausted"


def test_plan_usage_limit_still_rotates():
    """Non-capacity exhaustion is still excluded from retries."""
    plan = plan_responses_pre_commit_retry(
        error_class="usage_limit_reached",
        same_account_transient_attempts=0,
    )
    assert plan["action"] == "rotate_account"
    assert plan["retry_same_account"] is False
    assert plan["apply_account_exhaustion_cooldown"] is True


def test_plan_terminal_for_non_capacity():
    plan = plan_responses_pre_commit_retry(
        error_class="token_invalidated",
        same_account_transient_attempts=0,
    )
    assert plan["action"] == "terminal"
    assert plan["retryable"] is False


def test_plan_terminal_for_none_error_class():
    plan = plan_responses_pre_commit_retry(
        error_class=None,
        same_account_transient_attempts=0,
    )
    assert plan["action"] == "terminal"
    assert plan["retryable"] is False


def test_plan_capacity_exhausted_uses_progressive_schedule():
    plan = plan_responses_pre_commit_retry(
        error_class="capacity_exhausted",
        same_account_transient_attempts=0,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 15.0


def test_plan_upstream_transient_internal_uses_progressive_schedule():
    plan = plan_responses_pre_commit_retry(
        error_class="upstream_transient_internal",
        same_account_transient_attempts=1,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 30.0


def test_plan_default_deadline_is_7200():
    """Without an explicit budget, the 7200s default deadline is used."""
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        elapsed_seconds=8000.0,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert plan["action"] == "deadline_exhausted"


def test_plan_custom_budget_short_deadline():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        elapsed_seconds=30.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=40.0),
        openai_alpha_capacity_retry_enabled=True,
    )
    # 30 + 15 = 45 > 40, so deadline exhausted
    assert plan["action"] == "deadline_exhausted"


def test_plan_custom_budget_still_retries():
    plan = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        elapsed_seconds=10.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=40.0),
        openai_alpha_capacity_retry_enabled=True,
    )
    # 10 + 15 = 25 <= 40
    assert plan["action"] == "retry_same_account"
    assert plan["wait_seconds"] == 15.0


def test_plan_existing_alpha_extension_caller_still_works():
    """The explicit alpha extension still accepts the request-wide budget."""
    first = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=1,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert first["action"] == "retry_same_account"
    assert first["wait_seconds"] == 30.0

    second = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=2,
        openai_alpha_capacity_retry_enabled=True,
    )
    assert second["action"] == "retry_same_account"
    assert second["wait_seconds"] == 60.0


def test_plan_preserves_legacy_policy_without_alpha_extension():
    first = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=0,
        elapsed_seconds=7195.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=7200.0),
    )
    assert first["action"] == "retry_same_account"
    assert first["wait_seconds"] == 10.0

    exhausted = plan_responses_pre_commit_retry(
        error_class="server_overloaded",
        same_account_transient_attempts=2,
        elapsed_seconds=0.0,
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=7200.0),
    )
    assert exhausted["action"] == "pre_stream_unavailable"
    assert exhausted["wait_seconds"] == 10.0


def test_openai_alpha_capacity_budget_defaults():
    budget = OpenAIAlphaCapacityRetryBudget()
    assert budget.schedule == (15.0, 30.0, 60.0, 120.0, 240.0)
    assert budget.deadline_seconds == 7200.0


def test_openai_alpha_capacity_budget_custom():
    budget = OpenAIAlphaCapacityRetryBudget(
        schedule=(10.0, 20.0),
        deadline_seconds=3600.0,
    )
    assert budget.schedule == (10.0, 20.0)
    assert budget.deadline_seconds == 3600.0


# ---------------------------------------------------------------------------
# OPENAI-054: capacity retry coordinator tests
# ---------------------------------------------------------------------------

import time as _time_module

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    pre_commit_retry as pre_commit_retry_module,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry import (
    _LOCAL_CAPACITY_WAKEUP_EVENTS,
    OpenAIAlphaCapacityRetryCoordinator,
    _build_openai_capacity_success_redis_key,
    _build_openai_capacity_target_identity,
    _hash_target_identity,
    _emit_capacity_retry_log,
    _emit_openai_capacity_terminal_log,
    _resolve_redis_for_capacity_wakeup,
    _signal_openai_capacity_success,
    CapacityRetryLogEntry,
    get_or_create_openai_alpha_capacity_retry_coordinator,
)


class TestOpenAIAlphaCapacityRetrySchedule:
    def test_schedule_exact_steps(self):
        """Progressive: 15, 30, 60, 120, 240, 240, 240..."""
        assert openai_alpha_capacity_retry_wait_seconds(0) == 15.0
        assert openai_alpha_capacity_retry_wait_seconds(1) == 30.0
        assert openai_alpha_capacity_retry_wait_seconds(2) == 60.0
        assert openai_alpha_capacity_retry_wait_seconds(3) == 120.0
        assert openai_alpha_capacity_retry_wait_seconds(4) == 240.0
        assert openai_alpha_capacity_retry_wait_seconds(5) == 240.0
        assert openai_alpha_capacity_retry_wait_seconds(6) == 240.0
        assert openai_alpha_capacity_retry_wait_seconds(100) == 240.0

    def test_deadline_within(self):
        assert openai_alpha_capacity_retry_within_deadline(
            elapsed_seconds=0.0, next_wait_seconds=15.0, deadline_seconds=7200.0
        ) is True

    def test_deadline_exceeded(self):
        assert openai_alpha_capacity_retry_within_deadline(
            elapsed_seconds=7190.0, next_wait_seconds=15.0, deadline_seconds=7200.0
        ) is False

    def test_deadline_at_boundary(self):
        assert openai_alpha_capacity_retry_within_deadline(
            elapsed_seconds=7185.0, next_wait_seconds=15.0, deadline_seconds=7200.0
        ) is True

    def test_deadline_zero_disabled(self):
        assert openai_alpha_capacity_retry_within_deadline(
            elapsed_seconds=0.0, next_wait_seconds=1.0, deadline_seconds=0.0
        ) is False

    def test_deadline_negative_disabled(self):
        assert openai_alpha_capacity_retry_within_deadline(
            elapsed_seconds=0.0, next_wait_seconds=1.0, deadline_seconds=-1.0
        ) is False


class TestTargetIdentity:
    def test_identity_preserves_selected_model_and_upstream_route(self):
        assert _build_openai_capacity_target_identity(
            provider="openai",
            model="GPT-5.6-Astra",
            upstream_url="https://api.openai.com/v1/responses/",
        ) == "openai:gpt-5.6-astra@api.openai.com/v1/responses"

    def test_different_models_and_upstream_hosts_are_distinct(self):
        api_target = _build_openai_capacity_target_identity(
            model="gpt-5.6-astra",
            upstream_url="https://api.openai.com/v1/responses",
        )
        codex_target = _build_openai_capacity_target_identity(
            model="gpt-5.6-astra",
            upstream_url="https://chatgpt.com/backend-api/codex/responses",
        )
        other_model = _build_openai_capacity_target_identity(
            model="gpt-5.4",
            upstream_url="https://api.openai.com/v1/responses",
        )

        assert api_target != codex_target
        assert api_target != other_model

    def test_same_target_normalizes_model_and_route(self):
        first = _build_openai_capacity_target_identity(
            model=" OpenAI/GPT-5.6-Astra ",
            upstream_url="https://API.OPENAI.COM/v1/responses/",
        )
        second = _build_openai_capacity_target_identity(
            model="gpt-5.6-astra",
            upstream_url="https://api.openai.com/v1/responses",
        )

        assert first == second

    def test_identity_and_redis_key_are_credential_free(self):
        target_identity = _build_openai_capacity_target_identity(
            provider="openai",
            model="gpt-5.6-astra",
            upstream_url="https://api.openai.com/v1/responses",
        )
        redis_key = _build_openai_capacity_success_redis_key(
            target_identity, namespace="aawm-routing-alpha-v1"
        )

        assert redis_key == (
            "aawm:openai_capacity_success:"
            "aawm-routing-alpha-v1:openai:gpt-5.6-astra@"
            "api.openai.com/v1/responses"
        )
        assert "account" not in redis_key
        assert "secret" not in redis_key

    def test_hash_is_stable(self):
        h1 = _hash_target_identity("openai:gpt")
        h2 = _hash_target_identity("openai:gpt")
        assert h1 == h2
        assert len(h1) == 12


class TestCoordinatorBasic:
    def test_elapsed_increases(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        t0 = coordinator.elapsed_seconds
        _time_module.sleep(0.01)
        t1 = coordinator.elapsed_seconds
        assert t1 > t0

    def test_retry_count_starts_zero(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.retry_count == 0

    def test_next_wait_seconds_matches_schedule(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.next_wait_seconds() == 15.0

    def test_within_deadline_fresh(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.within_deadline() is True

    def test_budget_property(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
            budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=3600.0),
        )
        assert coordinator.budget.deadline_seconds == 3600.0
        assert coordinator.deadline_seconds == 3600.0

    def test_default_budget(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.deadline_seconds == 7200.0


class TestCoordinatorRequestCarrier:
    @staticmethod
    def _request() -> Request:
        return Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": "/openai_passthrough/responses",
                "headers": [],
            }
        )

    def test_same_request_reuses_one_coordinator_and_preserves_ledger(
        self, monkeypatch
    ):
        clock = [100.0]
        monkeypatch.setattr(
            pre_commit_retry_module.time,
            "monotonic",
            lambda: clock[0],
        )
        request = self._request()

        coordinator = get_or_create_openai_alpha_capacity_retry_coordinator(
            request,
            target_identity="openai:first",
            namespace="alpha-v1",
            budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=7200.0),
        )
        coordinator.record_retry("timer")
        clock[0] = 137.5

        rebound = get_or_create_openai_alpha_capacity_retry_coordinator(
            request,
            target_identity="openai:second",
            namespace="alpha-v2",
            budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=1.0),
        )

        assert rebound is coordinator
        assert request.state.aawm_openai_capacity_retry is coordinator
        assert coordinator.target_identity == "openai:second"
        assert coordinator.namespace == "alpha-v2"
        assert coordinator._start_monotonic == 100.0
        assert coordinator.elapsed_seconds == pytest.approx(37.5)
        assert coordinator.deadline_seconds == 7200.0
        assert coordinator.retry_count == 1
        assert coordinator.next_wait_seconds() == 30.0

    def test_different_requests_get_independent_coordinators(self):
        first_request = self._request()
        second_request = self._request()

        first = get_or_create_openai_alpha_capacity_retry_coordinator(
            first_request,
            target_identity="openai:gpt",
        )
        repeated = get_or_create_openai_alpha_capacity_retry_coordinator(
            first_request,
            target_identity="openai:gpt",
        )
        second = get_or_create_openai_alpha_capacity_retry_coordinator(
            second_request,
            target_identity="openai:gpt",
        )

        first.record_retry("timer")

        assert repeated is first
        assert second is not first
        assert first_request.state.aawm_openai_capacity_retry is first
        assert second_request.state.aawm_openai_capacity_retry is second
        assert first.retry_count == 1
        assert second.retry_count == 0


class TestCoordinatorSleepWakeup:
    @pytest.mark.asyncio
    async def test_sleep_zero_returns_timer(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        reason = await coordinator.sleep_with_wakeup(0.0)
        assert reason == "timer"

    @pytest.mark.asyncio
    async def test_sleep_timer(self):
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry."
            "_resolve_redis_for_capacity_wakeup",
            return_value=None,
        ):
            coordinator = OpenAIAlphaCapacityRetryCoordinator(
                target_identity="openai:gpt",
            )
            reason = await coordinator.sleep_with_wakeup(0.05)
        assert reason == "timer"

    @pytest.mark.asyncio
    async def test_sleep_peer_success_via_event(self):
        target_identity = "openai:local-event"
        namespace = "aawm-routing-test-v2"

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry."
            "_resolve_redis_for_capacity_wakeup",
            return_value=None,
        ):
            coordinator = OpenAIAlphaCapacityRetryCoordinator(
                target_identity=target_identity,
                namespace=namespace,
            )
            sleep_task = asyncio.create_task(
                coordinator.sleep_with_wakeup(1.0)
            )
            for _ in range(3):
                await asyncio.sleep(0)
                if _LOCAL_CAPACITY_WAKEUP_EVENTS.get((namespace, target_identity)):
                    break
            assert _LOCAL_CAPACITY_WAKEUP_EVENTS.get((namespace, target_identity))

            await _signal_openai_capacity_success(target_identity, namespace)
            reason = await asyncio.wait_for(sleep_task, timeout=0.2)

        assert reason == "peer_success"

    @pytest.mark.asyncio
    async def test_wakeup_event_returns_event(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
            namespace="aawm-routing-test-v2",
        )
        ev = coordinator.wakeup_event()
        assert not ev.is_set()
        ev.set()
        assert ev.is_set()
        coordinator._local_events.discard(ev)
        if not coordinator._local_events:
            _LOCAL_CAPACITY_WAKEUP_EVENTS.pop(coordinator._local_event_key, None)

    def test_redis_resolver_uses_public_manager(self):
        redis_client = SimpleNamespace()
        redis_cache = SimpleNamespace(
            init_async_client=MagicMock(return_value=redis_client)
        )
        manager = SimpleNamespace(
            get_dual_cache=MagicMock(
                return_value=SimpleNamespace(redis_cache=redis_cache)
            )
        )

        with patch(
            "litellm.proxy.aawm_alias_routing_redis."
            "aawm_alias_routing_redis_manager",
            manager,
        ):
            assert _resolve_redis_for_capacity_wakeup() is redis_cache

        manager.get_dual_cache.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_signal_success_increments_redis_epoch_and_refreshes_ttl(self):
        target_identity = "openai:redis-signal"
        namespace = "aawm-routing-alpha-v1"
        redis_client = SimpleNamespace(
            incr=AsyncMock(return_value=4),
            expire=AsyncMock(return_value=True),
        )
        redis_cache = SimpleNamespace(
            init_async_client=MagicMock(return_value=redis_client)
        )
        manager = SimpleNamespace(
            get_dual_cache=MagicMock(
                return_value=SimpleNamespace(redis_cache=redis_cache)
            )
        )
        event = asyncio.Event()
        _LOCAL_CAPACITY_WAKEUP_EVENTS.setdefault(
            (namespace, target_identity), set()
        ).add(event)

        try:
            with patch(
                "litellm.proxy.aawm_alias_routing_redis."
                "aawm_alias_routing_redis_manager",
                manager,
            ):
                await _signal_openai_capacity_success(
                    target_identity, namespace
                )
        finally:
            _LOCAL_CAPACITY_WAKEUP_EVENTS.get(
                (namespace, target_identity), set()
            ).discard(event)
            if not _LOCAL_CAPACITY_WAKEUP_EVENTS.get((namespace, target_identity)):
                _LOCAL_CAPACITY_WAKEUP_EVENTS.pop(
                    (namespace, target_identity), None
                )

        redis_key = _build_openai_capacity_success_redis_key(
            target_identity, namespace
        )
        assert event.is_set()
        redis_client.incr.assert_awaited_once_with(redis_key)
        redis_client.expire.assert_awaited_once_with(redis_key, 300)

    @pytest.mark.asyncio
    async def test_sleep_with_wakeup_reads_redis_epoch_change(self):
        target_identity = "openai:redis-read"
        namespace = "aawm-routing-alpha-v1"
        redis_client = SimpleNamespace(
            get=AsyncMock(side_effect=[b"7", b"8"]),
        )
        redis_cache = SimpleNamespace(
            init_async_client=MagicMock(return_value=redis_client)
        )
        manager = SimpleNamespace(
            get_dual_cache=MagicMock(
                return_value=SimpleNamespace(redis_cache=redis_cache)
            )
        )

        with patch(
            "litellm.proxy.aawm_alias_routing_redis."
            "aawm_alias_routing_redis_manager",
            manager,
        ):
            coordinator = OpenAIAlphaCapacityRetryCoordinator(
                target_identity=target_identity,
                namespace=namespace,
            )
            reason = await coordinator.sleep_with_wakeup(1.0)

        redis_key = _build_openai_capacity_success_redis_key(
            target_identity, namespace
        )
        assert reason == "peer_success"
        assert redis_client.get.await_count == 2
        assert all(
            call.args == (redis_key,)
            for call in redis_client.get.await_args_list
        )

    @pytest.mark.asyncio
    async def test_sleep_with_wakeup_does_not_treat_expired_epoch_as_success(self):
        target_identity = "openai:redis-expiry"
        namespace = "aawm-routing-alpha-v1"
        redis_client = SimpleNamespace(
            get=AsyncMock(side_effect=[b"7", None]),
        )
        redis_cache = SimpleNamespace(
            init_async_client=MagicMock(return_value=redis_client)
        )
        manager = SimpleNamespace(
            get_dual_cache=MagicMock(
                return_value=SimpleNamespace(redis_cache=redis_cache)
            )
        )

        with patch(
            "litellm.proxy.aawm_alias_routing_redis."
            "aawm_alias_routing_redis_manager",
            manager,
        ):
            coordinator = OpenAIAlphaCapacityRetryCoordinator(
                target_identity=target_identity,
                namespace=namespace,
            )
            reason = await coordinator.sleep_with_wakeup(0.01)

        assert reason == "timer"
        assert redis_client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_sleep_with_wakeup_does_not_treat_redis_recovery_as_success(self):
        target_identity = "openai:redis-recovery"
        namespace = "aawm-routing-alpha-v1"
        redis_client = SimpleNamespace(
            get=AsyncMock(side_effect=[RuntimeError("redis unavailable"), b"7"]),
        )
        redis_cache = SimpleNamespace(
            init_async_client=MagicMock(return_value=redis_client)
        )
        manager = SimpleNamespace(
            get_dual_cache=MagicMock(
                return_value=SimpleNamespace(redis_cache=redis_cache)
            )
        )

        with patch(
            "litellm.proxy.aawm_alias_routing_redis."
            "aawm_alias_routing_redis_manager",
            manager,
        ):
            coordinator = OpenAIAlphaCapacityRetryCoordinator(
                target_identity=target_identity,
                namespace=namespace,
            )
            reason = await coordinator.sleep_with_wakeup(0.01)

        assert reason == "timer"
        assert redis_client.get.await_count == 2


class TestCoordinatorLogging:
    def test_record_retry_increments_count(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.retry_count == 0
        coordinator.record_retry("timer")
        assert coordinator.retry_count == 1
        coordinator.record_retry("peer_success")
        assert coordinator.retry_count == 2

    def test_record_terminal_sets_reason(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.terminal_reason == ""
        coordinator.record_terminal("deadline_exhausted")
        assert coordinator.terminal_reason == "deadline_exhausted"

    def test_emit_capacity_retry_log(self, caplog):
        import logging as _logging
        with caplog.at_level(_logging.INFO, logger="LiteLLMProxy"):
            _emit_capacity_retry_log(
                CapacityRetryLogEntry(
                    target_class="openai:gpt",
                    target_hash="abc123",
                    retry_ordinal=0,
                    wait_seconds=15.0,
                    elapsed_seconds=0.0,
                    deadline_seconds=7200.0,
                    remaining_seconds=7200.0,
                    wakeup_reason="timer",
                    terminal_reason="",
                    commit_state="pre_commit",
                    phase="pre_wait",
                    error_class="server_overloaded",
                    status_code=502,
                )
            )
        assert "openai_alpha_capacity_retry" in caplog.text
        assert "phase=pre_wait" in caplog.text
        assert "target_class=openai:gpt" in caplog.text
        assert "target_hash=abc123" in caplog.text
        assert "error_class=server_overloaded" in caplog.text
        assert "status_code=502" in caplog.text
        assert "ordinal=0" in caplog.text
        assert "selected_delay=15.0" in caplog.text
        assert "wait=15.0" in caplog.text
        assert "wakeup=timer" in caplog.text
        assert "commit=pre_commit" in caplog.text

    @pytest.mark.asyncio
    async def test_sleep_emits_sanitized_pre_wait_before_wait_and_retry(
        self, monkeypatch
    ):
        clock = [100.0]
        emitted = []

        def emit(entry):
            emitted.append(entry)

        async def fake_wait_for(awaitable, timeout):
            emitted.append(("wait", timeout))
            awaitable.close()
            clock[0] += timeout
            raise asyncio.TimeoutError

        monkeypatch.setattr(
            pre_commit_retry_module.time,
            "monotonic",
            lambda: clock[0],
        )
        monkeypatch.setattr(
            pre_commit_retry_module,
            "_emit_capacity_retry_log",
            emit,
        )
        monkeypatch.setattr(
            pre_commit_retry_module.asyncio,
            "wait_for",
            fake_wait_for,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.pre_commit_retry."
            "_resolve_redis_for_capacity_wakeup",
            return_value=None,
        ):
            coordinator = OpenAIAlphaCapacityRetryCoordinator(
                target_identity="openai:gpt\nsecret",
            )
            wakeup_reason = await coordinator.sleep_with_wakeup(
                0.05,
                error_class="server_overloaded\nsecret",
                status_code=502,
            )
            coordinator.record_retry(wakeup_reason)

        assert wakeup_reason == "timer"
        assert emitted[0].phase == "pre_wait"
        assert emitted[0].target_class == "openai:gpt_secret"
        assert emitted[0].error_class == "server_overloaded_secret"
        assert emitted[0].status_code == 502
        assert emitted[0].retry_ordinal == 0
        assert emitted[0].wait_seconds == 0.05
        assert emitted[0].wakeup_reason == "pending"
        assert emitted[1][0] == "wait"
        assert emitted[2].phase == "retry"
        assert emitted[2].retry_ordinal == 0
        assert emitted[2].wait_seconds == 0.05
        assert emitted[2].wakeup_reason == "timer"
        assert emitted[2].error_class == "server_overloaded_secret"
        assert emitted[2].status_code == 502

    def test_emit_terminal_log(self, caplog):
        import logging as _logging
        with caplog.at_level(_logging.INFO, logger="LiteLLMProxy"):
            _emit_openai_capacity_terminal_log(
                target_class="openai:gpt",
                target_hash="abc123",
                total_retries=5,
                elapsed_seconds=100.0,
                deadline_seconds=7200.0,
                terminal_reason="deadline_exhausted",
                error_class="server_overloaded",
                status_code=503,
            )
        assert "openai_alpha_capacity_terminal" in caplog.text
        assert "phase=terminal" in caplog.text
        assert "target_class=openai:gpt" in caplog.text
        assert "error_class=server_overloaded" in caplog.text
        assert "status_code=503" in caplog.text
        assert "ordinal=5" in caplog.text
        assert "retries=5" in caplog.text
        assert "remaining=7100.0" in caplog.text
        assert "reason=deadline_exhausted" in caplog.text


class TestPlanIntegrationWithCoordinator:
    def test_plan_with_coordinator_elapsed_respects_budget(self):
        """Integration: plan_responses_pre_commit_retry with coordinator budget."""
        budget = OpenAIAlphaCapacityRetryBudget(deadline_seconds=40.0)
        plan = plan_responses_pre_commit_retry(
            error_class="server_overloaded",
            same_account_transient_attempts=0,
            elapsed_seconds=30.0,  # 30 + 15 = 45 > 40
            budget=budget,
            openai_alpha_capacity_retry_enabled=True,
        )
        assert plan["action"] == "deadline_exhausted"

    def test_plan_with_coordinator_budget_allows_retry(self):
        budget = OpenAIAlphaCapacityRetryBudget(deadline_seconds=40.0)
        plan = plan_responses_pre_commit_retry(
            error_class="server_overloaded",
            same_account_transient_attempts=0,
            elapsed_seconds=10.0,  # 10 + 15 = 25 <= 40
            budget=budget,
            openai_alpha_capacity_retry_enabled=True,
        )
        assert plan["action"] == "retry_same_account"
        assert plan["wait_seconds"] == 15.0

    def test_plan_non_capacity_errors_unchanged(self):
        """Non-capacity errors are still excluded."""
        budget = OpenAIAlphaCapacityRetryBudget()
        plan = plan_responses_pre_commit_retry(
            error_class="usage_limit_reached",
            same_account_transient_attempts=0,
            elapsed_seconds=0.0,
            budget=budget,
        )
        assert plan["action"] == "rotate_account"

    def test_plan_none_error_class_unchanged(self):
        budget = OpenAIAlphaCapacityRetryBudget()
        plan = plan_responses_pre_commit_retry(
            error_class=None,
            same_account_transient_attempts=0,
            elapsed_seconds=0.0,
            budget=budget,
        )
        assert plan["action"] == "terminal"

    def test_plan_repeating_240s_within_deadline(self):
        budget = OpenAIAlphaCapacityRetryBudget()
        plan = plan_responses_pre_commit_retry(
            error_class="server_overloaded",
            same_account_transient_attempts=5,  # 6th attempt
            elapsed_seconds=100.0,
            budget=budget,
            openai_alpha_capacity_retry_enabled=True,
        )
        assert plan["action"] == "retry_same_account"
        assert plan["wait_seconds"] == 240.0

    def test_plan_deadline_exhausted_has_retryable_true(self):
        budget = OpenAIAlphaCapacityRetryBudget(deadline_seconds=10.0)
        plan = plan_responses_pre_commit_retry(
            error_class="server_overloaded",
            same_account_transient_attempts=0,
            elapsed_seconds=5.0,  # 5 + 15 = 20 > 10
            budget=budget,
            openai_alpha_capacity_retry_enabled=True,
        )
        assert plan["action"] == "deadline_exhausted"
        assert plan["retryable"] is True
        assert plan["http_status"] == 503


class TestCoordinatorNoReplayBoundary:
    """Ensure the coordinator does not replay after substantive output."""
    def test_coordinator_does_not_retry_after_record_terminal(self):
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity="openai:gpt",
        )
        assert coordinator.terminal_reason == ""
        coordinator.record_terminal("deadline_exhausted")
        assert coordinator.terminal_reason == "deadline_exhausted"
        # After terminal, retry_count is unchanged (record_terminal doesn't
        # increment it; only record_retry does).
        assert coordinator.retry_count == 0


def test_central_capacity_retry_target_is_scoped_to_alpha_openai_responses(
    monkeypatch,
):
    request = _responses_request({"model": "gpt-5.4"})
    monkeypatch.delenv("AAWM_LITELLM_ENVIRONMENT", raising=False)
    assert not _is_openai_alpha_capacity_retry_target(
        request=request,
        url=httpx.URL("https://api.openai.com/v1/responses"),
        endpoint_type=EndpointType.OPENAI,
    )

    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    assert _is_openai_alpha_capacity_retry_target(
        request=request,
        url=httpx.URL("https://api.openai.com/v1/responses"),
        endpoint_type=EndpointType.OPENAI,
    )
    assert _is_openai_alpha_capacity_retry_target(
        request=request,
        url=httpx.URL("https://chatgpt.com/backend-api/codex/responses"),
        endpoint_type=EndpointType.OPENAI,
    )
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-dev")
    assert not _is_openai_alpha_capacity_retry_target(
        request=request,
        url=httpx.URL("https://api.openai.com/v1/responses"),
        endpoint_type=EndpointType.OPENAI,
    )


def test_alpha_capacity_planner_gate_requires_codex_openai_route(monkeypatch):
    request = _responses_request({"model": "gpt-5.4"})
    candidate = {
        "provider": "openai",
        "route_family": "codex_responses",
    }
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    assert _is_openai_alpha_capacity_retry_enabled(
        request=request,
        candidate=candidate,
        is_codex_alias=True,
    )
    assert not _is_openai_alpha_capacity_retry_enabled(
        request=request,
        candidate={**candidate, "provider": "openrouter"},
        is_codex_alias=True,
    )
    assert not _is_openai_alpha_capacity_retry_enabled(
        request=request,
        candidate={**candidate, "route_family": "anthropic_openai_responses_adapter"},
        is_codex_alias=True,
    )
    assert not _is_openai_alpha_capacity_retry_enabled(
        request=request,
        candidate=candidate,
        is_codex_alias=False,
    )
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-dev")
    assert not _is_openai_alpha_capacity_retry_enabled(
        request=request,
        candidate=candidate,
        is_codex_alias=True,
    )


def test_alpha_capacity_planner_gate_rejects_non_responses_route(monkeypatch):
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/openai_passthrough/chat/completions",
            "raw_path": b"/openai_passthrough/chat/completions",
            "query_string": b"",
            "headers": [],
        }
    )
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    assert not _is_openai_alpha_capacity_retry_enabled(
        request=request,
        candidate={
            "provider": "openai",
            "route_family": "codex_responses",
        },
        is_codex_alias=True,
    )


def test_central_capacity_retry_target_rejects_non_openai_upstream_in_alpha(
    monkeypatch,
):
    request = _responses_request({"model": "gpt-5.4"})
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    assert not _is_openai_alpha_capacity_retry_target(
        request=request,
        url=httpx.URL("https://api.anthropic.com/v1/messages"),
        endpoint_type=EndpointType.ANTHROPIC,
    )
    assert not _is_openai_alpha_capacity_retry_target(
        request=request,
        url=httpx.URL("https://api.openai.com/v1/chat/completions"),
        endpoint_type=EndpointType.OPENAI,
    )


@pytest.mark.asyncio
async def test_central_coordinator_replaces_legacy_precommit_cap():
    coordinator = OpenAIAlphaCapacityRetryCoordinator(
        target_identity="openai:gpt",
        budget=OpenAIAlphaCapacityRetryBudget(deadline_seconds=60.0),
    )
    coordinator.within_deadline = MagicMock(return_value=True)
    coordinator.sleep_with_wakeup = AsyncMock(return_value="timer")
    coordinator.record_retry = MagicMock()
    coordinator.record_terminal = MagicMock()
    coordinator.signal_success = AsyncMock()

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ResponsesStreamPreCommitFailure(
                error_class="server_overloaded",
                classification="transient_capacity",
                retryable=True,
                message="overloaded",
            )
        return "committed"

    result = await _execute_passthrough_pre_first_byte_with_hidden_retries(
        kwargs={},
        operation_name="stream_pre_first_byte",
        operation=operation,
        caller_managed_hidden_retry=False,
        openai_capacity_coordinator=coordinator,
    )

    assert result == "committed"
    assert attempts == 3
    assert coordinator.sleep_with_wakeup.await_count == 2
    assert [call.args[0] for call in coordinator.record_retry.call_args_list] == [
        "timer",
        "timer",
    ]
    assert [
        call.kwargs for call in coordinator.sleep_with_wakeup.call_args_list
    ] == [
        {"error_class": "server_overloaded", "status_code": 503},
        {"error_class": "server_overloaded", "status_code": 503},
    ]
    assert [
        call.kwargs for call in coordinator.record_retry.call_args_list
    ] == [
        {"error_class": "server_overloaded", "status_code": 503},
        {"error_class": "server_overloaded", "status_code": 503},
    ]
    coordinator.record_terminal.assert_called_once_with(
        "success",
        error_class="success",
        status_code=200,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_kind", "status_code", "error_field", "error_value"),
    [
        ("http_exception", 429, "code", "server_overloaded"),
        ("http_status_error", 502, "type", "server_overloaded"),
        ("http_exception", 503, "code", "server_overloaded"),
        ("http_exception", 500, "code", "server_is_overloaded"),
        ("http_status_error", 529, "type", "server_is_overloaded"),
        ("http_status_error", 504, "code", "capacity_exhausted"),
        ("http_exception", 500, "type", "capacity_exhausted"),
    ],
)
async def test_central_coordinator_retries_raw_http_overload_then_succeeds(
    exception_kind: str,
    status_code: int,
    error_field: str,
    error_value: str,
):
    overload_payload = {
        "error": {
            error_field: error_value,
            "message": "Please try again later.",
        }
    }
    headers = {"Retry-After": "1", "X-Upstream": "capacity"}
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status_code,
        request=request,
        content=json.dumps(overload_payload).encode("utf-8"),
        headers=headers,
    )
    overload_exception: Exception
    if exception_kind == "http_exception":
        overload_exception = HTTPException(
            status_code=status_code,
            detail=overload_payload,
            headers=headers,
        )
    else:
        overload_exception = httpx.HTTPStatusError(
            "upstream overload",
            request=request,
            response=response,
        )

    coordinator = MagicMock()
    coordinator.deadline_seconds = 7200.0
    coordinator.remaining_seconds = 7200.0
    coordinator.within_deadline.return_value = True
    coordinator.next_wait_seconds.return_value = 15.0
    coordinator.sleep_with_wakeup = AsyncMock(return_value="timer")
    coordinator.record_retry = MagicMock()
    coordinator.record_terminal = MagicMock()
    coordinator.signal_success = AsyncMock()

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise overload_exception
        return "committed"

    result = await _execute_passthrough_pre_first_byte_with_hidden_retries(
        kwargs={},
        operation_name="stream_pre_first_byte",
        operation=operation,
        caller_managed_hidden_retry=False,
        openai_capacity_coordinator=coordinator,
    )

    assert result == "committed"
    assert attempts == 2
    coordinator.sleep_with_wakeup.assert_awaited_once_with(
        15.0,
        error_class=f"http_status_{status_code}",
        status_code=status_code,
    )
    coordinator.record_retry.assert_called_once_with(
        "timer",
        error_class=f"http_status_{status_code}",
        status_code=status_code,
    )
    coordinator.signal_success.assert_awaited_once()
    coordinator.record_terminal.assert_called_once_with(
        "success",
        error_class="success",
        status_code=200,
    )
    if isinstance(overload_exception, HTTPException):
        assert overload_exception.status_code == status_code
        assert overload_exception.detail == overload_payload
        assert overload_exception.headers == headers
    else:
        assert overload_exception.response.status_code == status_code
        assert overload_exception.response.content == json.dumps(
            overload_payload
        ).encode("utf-8")
        assert overload_exception.response.headers["retry-after"] == "1"
        assert overload_exception.response.headers["x-upstream"] == "capacity"


@pytest.mark.asyncio
async def test_central_coordinator_preserves_last_capacity_failure_on_expiry():
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    terminal_payload = {
        "error": {
            "type": "server_overloaded",
            "code": "server_overloaded",
            "message": "The upstream server is overloaded.",
        }
    }
    terminal_exception = HTTPException(
        status_code=502,
        detail=terminal_payload,
        headers={"Retry-After": "999999"},
    )
    coordinator = MagicMock()
    coordinator.deadline_seconds = 7200.0
    coordinator.remaining_seconds = 7200.0
    coordinator.within_deadline.return_value = True
    coordinator.next_wait_seconds.return_value = 0.0
    coordinator.sleep_with_wakeup = AsyncMock(return_value="timer")
    coordinator.record_retry = MagicMock()

    await_calls = 0

    async def operation():
        nonlocal await_calls
        await_calls += 1
        if await_calls == 1:
            raise terminal_exception
        return "unreachable"

    async def fake_await(
        operation,
        *,
        timeout_seconds,
        operation_name,
    ):
        if await_calls == 0:
            return await operation()
        raise pte._PassthroughHiddenRetryBudgetTimeout("expired")

    with patch.object(
        pte,
        "_await_passthrough_pre_first_byte_operation",
        new=fake_await,
    ):
        with pytest.raises(HTTPException) as raised:
            await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs={},
                operation_name="stream_pre_first_byte",
                operation=operation,
                caller_managed_hidden_retry=False,
                openai_capacity_coordinator=coordinator,
            )

    assert raised.value is terminal_exception
    assert raised.value.status_code == 502
    assert raised.value.detail == terminal_payload
    assert raised.value.headers == {"Retry-After": "999999"}
    coordinator.record_terminal.assert_called_once_with(
        "deadline_exhausted", error_class="server_overloaded", status_code=502
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_capacity", [False, True])
async def test_legacy_budget_timeout_preserves_original_exception(prior_capacity):
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    timeout = pte._PassthroughHiddenRetryBudgetTimeout("expired")
    capacity = HTTPException(
        status_code=503,
        detail={"error": {"code": "server_overloaded"}},
        headers={"Retry-After": "20"},
    )
    outcomes = [capacity, timeout] if prior_capacity else [timeout]

    async def fake_await(*args, **kwargs):
        raise outcomes.pop(0)

    with patch.object(
        pte, "_await_passthrough_pre_first_byte_operation", new=fake_await
    ), patch.object(pte.asyncio, "sleep", new=AsyncMock()):
        with pytest.raises(pte._PassthroughHiddenRetryBudgetTimeout) as raised:
            await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs={},
                operation=AsyncMock(),
                operation_name="non_stream_pre_first_byte",
                caller_managed_hidden_retry=False,
                custom_llm_provider="other",
            )
    assert raised.value is timeout
    assert not outcomes
    assert pte._get_passthrough_terminal_wire_headers(capacity) == {}


@pytest.mark.asyncio
async def test_central_coordinator_expiry_without_prior_failure_returns_504():
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    coordinator = MagicMock()
    coordinator.deadline_seconds = 7200.0
    coordinator.remaining_seconds = 7200.0

    async def fake_await(
        operation,
        *,
        timeout_seconds,
        operation_name,
    ):
        raise pte._PassthroughHiddenRetryBudgetTimeout(
            "Pass-through stream_pre_first_byte hidden retry budget exhausted"
        )

    with patch.object(
        pte,
        "_await_passthrough_pre_first_byte_operation",
        new=fake_await,
    ):
        with pytest.raises(HTTPException) as raised:
            await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs={},
                operation=AsyncMock(),
                operation_name="stream_pre_first_byte",
                caller_managed_hidden_retry=False,
                openai_capacity_coordinator=coordinator,
            )

    assert raised.value.status_code == 504
    assert "hidden retry budget exhausted" in str(raised.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_kind", "status_code", "error_payload"),
    [
        (
            "http_exception",
            429,
            {
                "error": {
                    "type": "usage_limit_reached",
                    "code": "usage_limit_reached",
                    "message": "Quota exhausted.",
                }
            },
        ),
        (
            "http_status_error",
            503,
            {
                "error": {
                    "type": "invalid_api_key",
                    "code": "invalid_api_key",
                    "message": "Authentication failed.",
                }
            },
        ),
        (
            "http_exception", 500,
            {"error": {"code": "capacity_exhausted", "message": "Quota exhausted."}},
        ),
        (
            "http_status_error", 529,
            {"error": {"type": "server_is_overloaded", "code": "invalid_api_key"}},
        ),
        (
            "http_exception", 503,
            {"error": {"code": "server_is_overloaded", "type": "usage_limit_reached"}},
        ),
        (
            "http_status_error", 500,
            {"error": {"type": "capacity_exhausted", "code": "token_invalidated"}},
        ),
        (
            "http_exception", 429,
            {"error": {"code": "rate_limit_exceeded", "message": "Local request limit."}},
        ),
        (
            "http_status_error", 504,
            {"error": {"code": "upstream_timeout", "message": "Gateway timeout."}},
        ),
        (
            "http_exception", 500,
            {"error": {"code": "internal_error", "message": "Internal error."}},
        ),
    ],
)
async def test_central_coordinator_exits_raw_http_quota_or_auth_immediately(
    exception_kind: str,
    status_code: int,
    error_payload: dict[str, Any],
):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status_code,
        request=request,
        content=json.dumps(error_payload).encode("utf-8"),
    )
    if exception_kind == "http_exception":
        exception: Exception = HTTPException(
            status_code=status_code,
            detail=error_payload,
            headers={"X-Upstream": "quota"},
        )
    else:
        exception = httpx.HTTPStatusError(
            "upstream terminal error",
            request=request,
            response=response,
        )

    coordinator = MagicMock()
    coordinator.deadline_seconds = 7200.0
    coordinator.remaining_seconds = 7200.0
    coordinator.sleep_with_wakeup = AsyncMock(
        side_effect=AssertionError("non-capacity HTTP error must not sleep")
    )
    coordinator.record_retry = MagicMock()
    coordinator.record_terminal = MagicMock()

    async def operation():
        raise exception

    with pytest.raises(type(exception)) as raised:
        await _execute_passthrough_pre_first_byte_with_hidden_retries(
            kwargs={},
            operation_name="stream_pre_first_byte",
            operation=operation,
            caller_managed_hidden_retry=False,
            openai_capacity_coordinator=coordinator,
        )

    assert raised.value is exception
    coordinator.sleep_with_wakeup.assert_not_awaited()
    coordinator.record_retry.assert_not_called()
    coordinator.record_terminal.assert_called_once_with(
        "non_capacity_error",
        error_class=f"http_status_{status_code}",
        status_code=status_code,
    )


@pytest.mark.asyncio
async def test_central_coordinator_uses_remaining_deadline_once_per_attempt():
    coordinator = OpenAIAlphaCapacityRetryCoordinator(
        target_identity="openai:gpt",
    )
    coordinator.within_deadline = MagicMock(return_value=True)
    coordinator.next_wait_seconds = MagicMock(return_value=15.0)
    coordinator.sleep_with_wakeup = AsyncMock(return_value="timer")
    coordinator.record_retry = MagicMock()
    coordinator.record_terminal = MagicMock()
    coordinator.signal_success = AsyncMock()

    attempts = 0
    timeout_values: list[Optional[float]] = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ResponsesStreamPreCommitFailure(
                error_class="server_overloaded",
                classification="transient_capacity",
                retryable=True,
                message="overloaded",
            )
        return "committed"

    async def fake_await(
        operation,
        *,
        timeout_seconds,
        operation_name,
    ):
        timeout_values.append(timeout_seconds)
        return await operation()

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints."
            "_await_passthrough_pre_first_byte_operation",
            new=fake_await,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.time.monotonic",
            side_effect=[0.0, 15.0, 15.0],
        ),
        patch.object(
            type(coordinator),
            "remaining_seconds",
            new_callable=PropertyMock,
        ) as remaining_seconds,
    ):
        remaining_seconds.side_effect = [7200.0, 7185.0]
        result = await _execute_passthrough_pre_first_byte_with_hidden_retries(
            kwargs={},
            operation_name="stream_pre_first_byte",
            operation=operation,
            caller_managed_hidden_retry=False,
            openai_capacity_coordinator=coordinator,
        )

    assert result == "committed"
    assert timeout_values == [7200.0, 7185.0]
    coordinator.record_terminal.assert_called_once_with(
        "success",
        error_class="success",
        status_code=200,
    )


@pytest.mark.asyncio
async def test_central_coordinator_uses_repeating_capacity_schedule():
    coordinator = MagicMock()
    coordinator.deadline_seconds = 7200.0
    coordinator.remaining_seconds = 7200.0
    coordinator.within_deadline.return_value = True
    coordinator.next_wait_seconds.side_effect = [
        15.0,
        30.0,
        60.0,
        120.0,
        240.0,
        240.0,
    ]
    coordinator.sleep_with_wakeup = AsyncMock(return_value="timer")
    coordinator.record_retry = MagicMock()
    coordinator.record_terminal = MagicMock()
    coordinator.signal_success = AsyncMock()

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts <= 6:
            raise ResponsesStreamPreCommitFailure(
                error_class="server_overloaded",
                classification="transient_capacity",
                retryable=True,
                message="overloaded",
            )
        return "committed"

    result = await _execute_passthrough_pre_first_byte_with_hidden_retries(
        kwargs={},
        operation_name="stream_pre_first_byte",
        operation=operation,
        caller_managed_hidden_retry=False,
        openai_capacity_coordinator=coordinator,
    )

    assert result == "committed"
    assert attempts == 7
    assert [
        call.args[0]
        for call in coordinator.sleep_with_wakeup.call_args_list
    ] == [15.0, 30.0, 60.0, 120.0, 240.0, 240.0]
    assert coordinator.record_retry.call_count == 6
    assert all(
        call.kwargs == {"error_class": "server_overloaded", "status_code": 503}
        for call in coordinator.sleep_with_wakeup.call_args_list
    )
    assert all(
        call.kwargs == {"error_class": "server_overloaded", "status_code": 503}
        for call in coordinator.record_retry.call_args_list
    )
    coordinator.record_terminal.assert_called_once_with(
        "success",
        error_class="success",
        status_code=200,
    )


@pytest.mark.asyncio
async def test_central_coordinator_does_not_tight_loop_non_capacity_retryable():
    coordinator = MagicMock()
    coordinator.deadline_seconds = 7200.0
    coordinator.remaining_seconds = 7200.0
    coordinator.sleep_with_wakeup = AsyncMock(
        side_effect=AssertionError("non-capacity retry must not sleep")
    )
    coordinator.record_retry = MagicMock()
    coordinator.record_terminal = MagicMock()

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("connection reset")

    with pytest.raises(httpx.ReadError):
        await _execute_passthrough_pre_first_byte_with_hidden_retries(
            kwargs={},
            operation_name="stream_pre_first_byte",
            operation=operation,
            caller_managed_hidden_retry=False,
            openai_capacity_coordinator=coordinator,
        )

    assert attempts == 1
    coordinator.sleep_with_wakeup.assert_not_awaited()
    coordinator.record_retry.assert_not_called()
    coordinator.record_terminal.assert_called_once_with(
        "non_capacity_error",
        error_class="upstream_connectivity_failure",
        status_code=None,
    )


def test_candidate_loop_planner_calls_pass_live_elapsed_and_deadline():
    """Candidate-loop planner calls must use live request-wide elapsed/deadline.

    Passing only per-slot attempt counts lets ``plan_responses_pre_commit_retry``
    default ``elapsed_seconds=0.0`` and reuse a stale schedule instead of the
    request-wide two-hour deadline.
    """
    import ast
    from pathlib import Path

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    tree = ast.parse(
        Path(candidate_loop.__file__).read_text(encoding="utf-8"),
        filename=candidate_loop.__file__,
    )
    handle = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "handle_alias_route"
    )

    assigned_names = {
        target.id
        for node in ast.walk(handle)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    annotated_names = {
        node.target.id
        for node in ast.walk(handle)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "request_retry_started_at" in assigned_names
    assert "request_retry_budget" in assigned_names | annotated_names

    calls = [
        node
        for node in ast.walk(handle)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plan_responses_pre_commit_retry"
    ]
    assert len(calls) == 2

    def _name_ids(node: ast.AST) -> set[str]:
        return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}

    def _attr_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                names.add(child.attr)
        return names

    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        assert set(keywords) >= {
            "error_class",
            "same_account_transient_attempts",
            "elapsed_seconds",
            "budget",
            "openai_alpha_capacity_retry_enabled",
        }
        elapsed_names = _name_ids(keywords["elapsed_seconds"])
        elapsed_attrs = _attr_names(keywords["elapsed_seconds"])
        budget_names = _name_ids(keywords["budget"])
        assert "request_retry_started_at" in elapsed_names
        assert "monotonic" in elapsed_attrs
        assert "_is_openai_alpha_capacity_retry_enabled" in _attr_names(
            keywords["openai_alpha_capacity_retry_enabled"]
        )
        assert "same_account_transient_attempts_by_slot" not in elapsed_names
        assert "request_retry_budget" in budget_names
        assert "same_account_transient_attempts_by_slot" not in budget_names


@pytest.mark.asyncio
async def test_nonstream_openai_passthrough_responses_hands_off_capacity_coordinator(
    monkeypatch,
):
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as sa,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = SimpleNamespace(path="/openai_passthrough/v1/responses")
    mock_request.headers = {"content-type": "application/json"}
    mock_request.query_params = {}
    mock_request.state = SimpleNamespace()
    custom_body = {"model": "gpt-5.4"}
    monkeypatch.setenv(
        "AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "aawm-routing-test-v2"
    )
    upstream_response = MagicMock()
    upstream_response.status_code = 200
    upstream_response.headers = {"content-type": "application/json"}
    upstream_response.aiter_bytes = AsyncMock(return_value=[b'{"ok": true}'])
    upstream_response.aread = AsyncMock(return_value=b'{"ok": true}')
    captured: dict[str, Any] = {}

    async def execute_hidden_retries(**kwargs):
        captured.update(kwargs)
        return upstream_response

    async def run_with_renewal(_lease, operation):
        return await operation()

    with patch.object(
        pte,
        "_aawm_session_owner_pre_send_guard",
        new=AsyncMock(),
    ), patch.object(
        sa,
        "get_request_session_owner_lease",
        return_value=None,
    ), patch.object(
        sa,
        "run_with_session_owner_lease_renewal",
        new=run_with_renewal,
    ), patch.object(
        sa,
        "finalize_request_session_owner_lease",
        new=AsyncMock(),
    ), patch.object(
        pte,
        "_execute_passthrough_pre_first_byte_with_hidden_retries",
        new=execute_hidden_retries,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj, patch.object(
        pte.pass_through_endpoint_logging,
        "pass_through_async_success_handler",
        new_callable=AsyncMock,
    ):
        mock_client_obj = MagicMock()
        mock_client_obj.client = MagicMock()
        mock_get_client.return_value = mock_client_obj
        mock_logging_obj.pre_call_hook = AsyncMock(return_value=custom_body)
        mock_logging_obj.post_call_success_hook = AsyncMock()
        mock_logging_obj.post_call_failure_hook = AsyncMock()

        await pass_through_request(
            request=mock_request,
            target="https://api.openai.com/v1/responses",
            custom_headers={},
            user_api_key_dict=MagicMock(),
            custom_body=custom_body,
            custom_llm_provider="openai",
            stream=False,
        )

    coordinator = captured.get("openai_capacity_coordinator")
    assert captured.get("operation_name") == "non_stream_pre_first_byte"
    assert isinstance(coordinator, OpenAIAlphaCapacityRetryCoordinator)
    assert coordinator.deadline_seconds == 7200.0
    assert coordinator._namespace == "aawm-routing-test-v2"
    assert coordinator._target_identity == (
        "openai:gpt-5.4@api.openai.com/v1/responses"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_request", [False, True])
@pytest.mark.parametrize("error_code", ["server_overloaded", "server_is_overloaded", "capacity_exhausted"])
@pytest.mark.parametrize("event_type", ["error", "response.failed"])
@pytest.mark.parametrize(
    "target",
    ["https://api.openai.com/v1/responses", "https://chatgpt.com/backend-api/codex/responses"],
)
async def test_alpha_openai_sse_precommit_overload_closes_and_propagates(
    monkeypatch, stream_request, error_code, event_type, target,
):
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as sa,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = SimpleNamespace(path="/openai_passthrough/v1/responses")
    mock_request.headers = {"content-type": "application/json"}
    mock_request.query_params = {}
    mock_request.state = SimpleNamespace()
    custom_body = {"model": "gpt-5.4"}
    error = {"code": error_code, "type": "server_error", "message": "Try again."}
    payload = (
        {"type": event_type, "response": {"status": "failed", "error": error}}
        if event_type == "response.failed"
        else {"type": event_type, "error": error}
    )
    stream = _ClosableAsyncByteStream([_sse(event_type, payload)])
    upstream_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", target),
    )
    captured: dict[str, Any] = {}

    async def execute_hidden_retries(**kwargs):
        captured.update(kwargs)
        try:
            return await kwargs["operation"]()
        except BaseException as exc:
            captured["exception"] = exc
            raise

    async def run_with_renewal(_lease, operation):
        return await operation()

    with patch.object(
        pte,
        "_aawm_session_owner_pre_send_guard",
        new=AsyncMock(),
    ), patch.object(
        sa,
        "get_request_session_owner_lease",
        return_value=None,
    ), patch.object(
        sa,
        "run_with_session_owner_lease_renewal",
        new=run_with_renewal,
    ), patch.object(
        sa,
        "finalize_request_session_owner_lease",
        new=AsyncMock(),
    ), patch.object(
        pte,
        "_execute_passthrough_pre_first_byte_with_hidden_retries",
        new=execute_hidden_retries,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=upstream_response),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj:
        mock_client_obj = MagicMock()
        mock_client_obj.client = MagicMock()
        mock_client_obj.client.send = AsyncMock(return_value=upstream_response)
        mock_client_obj.client.build_request.return_value = upstream_response.request
        mock_get_client.return_value = mock_client_obj
        mock_logging_obj.pre_call_hook = AsyncMock(return_value=custom_body)
        mock_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

        with pytest.raises(ProxyException):
            await pass_through_request(
                request=mock_request,
                target=target,
                custom_headers={},
                user_api_key_dict=MagicMock(),
                custom_body=custom_body,
                custom_llm_provider="openai",
                stream=stream_request,
            )

    assert isinstance(captured["exception"], ResponsesStreamPreCommitFailure)
    assert captured["exception"].error_class == "server_overloaded"
    assert captured["exception"].retryable is True
    assert captured["exception"].error_code == error_code
    assert stream.close_calls == 1
    assert upstream_response.is_closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["server_is_overloaded", "capacity_exhausted"])
@pytest.mark.parametrize(
    ("enabled", "error_type", "message", "substantive"),
    [
        (False, "server_error", "Try again.", False),
        (True, "server_error", "Quota exceeded", False),
        (True, "insufficient_quota", "Try again.", False),
        (True, "authentication_error", "Invalid credentials", False),
        (True, "invalid_request_error", "Invalid request", False),
        (True, "server_error", "Try again.", True),
    ],
)
async def test_exact_capacity_sse_codes_preserve_peek_exclusions_and_bytes(
    error_code, enabled, error_type, message, substantive,
):
    chunks = []
    if substantive:
        chunks.append(_sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "hello"},
        ))
    chunks.append(_sse(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"code": error_code, "type": error_type, "message": message},
            },
        },
    ))
    response = httpx.Response(
        200,
        stream=_ClosableAsyncByteStream(chunks),
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    peeked, failure = await PassThroughStreamingHandler.peek_responses_pre_commit_stream(
        response, openai_alpha_capacity_retry_enabled=enabled,
    )
    if substantive:
        assert failure is None
    else:
        assert failure is not None
        assert failure.retryable is False
        assert failure.error_code == error_code
    assert b"".join([chunk async for chunk in peeked.aiter_bytes()]) == b"".join(chunks)
    await peeked.aclose()


@pytest.mark.asyncio
async def test_nonstream_alpha_openai_sse_substantive_bytes_survive_precommit_peek(
    monkeypatch,
):
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as sa,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = SimpleNamespace(path="/openai_passthrough/v1/responses")
    mock_request.headers = {"content-type": "application/json"}
    mock_request.query_params = {}
    mock_request.state = SimpleNamespace()
    custom_body = {"model": "gpt-5.4"}
    chunks = [
        _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_substantive",
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
                "response": {
                    "id": "resp_substantive",
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
        ),
    ]
    stream = _ClosableAsyncByteStream(chunks)
    upstream_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    captured: dict[str, Any] = {}

    class _StopAfterPeek(Exception):
        pass

    async def run_with_renewal(_lease, operation):
        captured["response"] = await operation()
        raise _StopAfterPeek

    with patch.object(
        pte,
        "_aawm_session_owner_pre_send_guard",
        new=AsyncMock(),
    ), patch.object(
        sa,
        "get_request_session_owner_lease",
        return_value=None,
    ), patch.object(
        sa,
        "run_with_session_owner_lease_renewal",
        new=run_with_renewal,
    ), patch.object(
        sa,
        "finalize_request_session_owner_lease",
        new=AsyncMock(),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
        new=AsyncMock(return_value=upstream_response),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj, patch.object(
        OpenAIAlphaCapacityRetryCoordinator,
        "signal_success",
        new_callable=AsyncMock,
    ) as signal_success:
        mock_client_obj = MagicMock()
        mock_client_obj.client = MagicMock()
        mock_get_client.return_value = mock_client_obj
        mock_logging_obj.pre_call_hook = AsyncMock(return_value=custom_body)
        mock_logging_obj.post_call_success_hook = AsyncMock()
        mock_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

        with pytest.raises(ProxyException):
            await pass_through_request(
                request=mock_request,
                target="https://api.openai.com/v1/responses",
                custom_headers={},
                user_api_key_dict=MagicMock(),
                custom_body=custom_body,
                custom_llm_provider="openai",
                stream=False,
            )

    replayed = [chunk async for chunk in captured["response"].aiter_bytes()]
    await captured["response"].aclose()
    assert replayed == chunks
    signal_success.assert_awaited_once()
    assert stream.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "upstream_retry_after", "expired", "expected_retry_after"),
    [
        ("sse", "999999", True, "7200"),
        ("raw", None, True, "10"),
        ("raw", "17", True, "17"),
        ("http", "17", False, None),
    ],
)
async def test_stream_capacity_failure_retry_after_reaches_proxy_wire_headers(
    monkeypatch, kind, upstream_retry_after, expired, expected_retry_after,
):
    monkeypatch.setenv("AAWM_LITELLM_ENVIRONMENT", "litellm-alpha")
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as sa,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )
    from litellm.proxy.proxy_server import openai_exception_handler

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = SimpleNamespace(path="/openai_passthrough/v1/responses")
    mock_request.headers = {"content-type": "application/json"}
    mock_request.query_params = {}
    mock_request.state = SimpleNamespace()
    custom_body = {"model": "gpt-5.4", "stream": True}
    terminal_failure = ResponsesStreamPreCommitFailure(
        error_class="server_overloaded",
        classification="transient_capacity",
        retryable=True,
        retry_after_seconds=999999.0,
        pre_commit_retry_exhausted=True,
        message="The upstream server is overloaded.",
    )
    if kind == "raw":
        terminal_failure = httpx.HTTPStatusError(
            "capacity",
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            response=httpx.Response(
                503,
                json={"error": {"code": "server_overloaded", "message": "capacity"}},
                headers=(
                    {"Retry-After": upstream_retry_after}
                    if upstream_retry_after is not None else {}
                ),
            ),
        )
    elif kind == "http":
        terminal_failure = HTTPException(
            status_code=503, detail="capacity",
            headers={"Retry-After": upstream_retry_after},
        )

    async def execute_hidden_retries(**kwargs):
        if not expired:
            raise terminal_failure
        coordinator = MagicMock()
        coordinator.deadline_seconds = 7200.0
        coordinator.remaining_seconds = 1.0
        coordinator.elapsed_seconds = 7200.0
        coordinator.within_deadline.return_value = False
        return await _execute_passthrough_pre_first_byte_with_hidden_retries(
            kwargs={},
            operation=AsyncMock(side_effect=terminal_failure),
            operation_name="stream_pre_first_byte",
            caller_managed_hidden_retry=False,
            openai_capacity_coordinator=coordinator,
        )

    async def run_with_renewal(_lease, operation):
        return await operation()

    with patch.object(
        pte,
        "_aawm_session_owner_pre_send_guard",
        new=AsyncMock(),
    ), patch.object(
        sa,
        "get_request_session_owner_lease",
        return_value=None,
    ), patch.object(
        sa,
        "run_with_session_owner_lease_renewal",
        new=run_with_renewal,
    ), patch.object(
        sa,
        "finalize_request_session_owner_lease",
        new=AsyncMock(),
    ), patch.object(
        pte,
        "_execute_passthrough_pre_first_byte_with_hidden_retries",
        new=execute_hidden_retries,
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging_obj:
        mock_client_obj = MagicMock()
        mock_client_obj.client = MagicMock()
        mock_get_client.return_value = mock_client_obj
        mock_logging_obj.pre_call_hook = AsyncMock(return_value=custom_body)
        mock_logging_obj.post_call_failure_hook = AsyncMock()

        with pytest.raises(ProxyException) as raised:
            await pass_through_request(
                request=mock_request,
                target="https://api.openai.com/v1/responses",
                custom_headers={},
                user_api_key_dict=MagicMock(),
                custom_body=custom_body,
                custom_llm_provider="openai",
                stream=True,
            )

    assert raised.value.code == "503"
    expected_detail = (
        terminal_failure.response.text if kind == "raw" else terminal_failure.detail
    )
    assert raised.value.detail == expected_detail
    assert raised.value.headers.get("Retry-After") == expected_retry_after

    response = await openai_exception_handler(mock_request, raised.value)
    assert response.status_code == 503
    assert response.headers.get("retry-after") == expected_retry_after
    assert json.loads(response.body)["error"]["message"] == str(expected_detail)
