"""D1-634: recover Responses streams that fail before the first client byte."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    _RESPONSES_PRE_COMMIT_TRANSIENT_CLASSES,
    _extract_openai_responses_unpersisted_item_id,
    plan_responses_pre_commit_retry,
    remove_openai_responses_unpersisted_input_item,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import candidate_loop
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
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


def _unpersisted_item_error(item_id: str = "rs_stale") -> ProxyException:
    message = (
        f"Item with id '{item_id}' not found. "
        "Items are not persisted when store is set to false. "
        "Try again with store set to true."
    )
    exc = ProxyException(
        message=message,
        type="invalid_request_error",
        param="input",
        code=400,
    )
    exc.detail = {
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_request_error",
            "message": message,
        }
    }
    return exc


def _responses_body_with_unpersisted_item(
    item_id: str = "rs_stale",
) -> dict[str, Any]:
    message = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "continue"}],
    }
    tool_call = {
        "type": "function_call",
        "id": "fc_keep",
        "call_id": "call_keep",
        "name": "exec_command",
        "arguments": "{}",
    }
    tool_output = {
        "type": "function_call_output",
        "call_id": "call_keep",
        "output": "ok",
    }
    stale_reasoning = {
        "type": "reasoning",
        "id": item_id,
        "summary": [],
    }
    unrelated_reasoning = {
        "type": "reasoning",
        "id": "rs_keep",
        "summary": [{"type": "summary_text", "text": "keep"}],
    }
    return {
        "model": "gpt-5.4",
        "input": [
            message,
            tool_call,
            tool_output,
            stale_reasoning,
            unrelated_reasoning,
        ],
        "previous_response_id": "resp_previous",
        "store": False,
        "stream": True,
        "metadata": {"trace": "d1-634"},
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


def test_extract_openai_responses_unpersisted_item_id_returns_exact_id():
    exc = _unpersisted_item_error("rs_Abc-123_x")

    assert _extract_openai_responses_unpersisted_item_id(exc) == "rs_Abc-123_x"


@pytest.mark.parametrize(
    "exc",
    [
        HTTPException(
            status_code=400,
            detail=(
                "Item with id 'rs_stale' not found. "
                "Items are not persisted when store is set to false."
            ),
        ),
        HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Item with id 'rs_stale' not found.",
                }
            },
        ),
        HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "Item with id 'rs_stale' not found. "
                        "Items are not persisted when store is set to true."
                    ),
                }
            },
        ),
        HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "Item with id 'rs_stale' not found. "
                        "Items are not persisted when store is set to false."
                    ),
                }
            },
        ),
    ],
    ids=["unstructured", "missing-message-shape", "nonmatching-message", "wrong-status"],
)
def test_extract_openai_responses_unpersisted_item_id_rejects_unstructured_or_nonmatching(
    exc: Exception,
):
    assert _extract_openai_responses_unpersisted_item_id(exc) is None


def test_remove_openai_responses_unpersisted_input_item_copies_and_preserves_body():
    body = _responses_body_with_unpersisted_item()
    original_input = body["input"]
    original_metadata = body["metadata"]
    expected_items = [
        item for item in original_input if item.get("id") != "rs_stale"
    ]

    repaired = remove_openai_responses_unpersisted_input_item(body, "rs_stale")

    assert repaired is not None
    assert repaired is not body
    assert repaired["input"] is not original_input
    assert repaired["input"] == expected_items
    assert body["input"] is original_input
    assert body["input"][-2]["id"] == "rs_stale"
    assert repaired["metadata"] is original_metadata
    assert repaired["previous_response_id"] == body["previous_response_id"]
    assert repaired["store"] is False
    assert repaired["stream"] is True
    assert all(
        repaired_item is original_item
        for repaired_item, original_item in zip(
            repaired["input"], original_input[:3] + original_input[4:]
        )
    )


@pytest.mark.parametrize(
    ("input_value", "item_id"),
    [
        ({"id": "rs_stale"}, "rs_stale"),
        ([{"id": "rs_duplicate"}, {"id": "rs_duplicate"}], "rs_duplicate"),
        ([{"id": "rs_other"}], "rs_stale"),
    ],
    ids=["input-not-list", "duplicate-match", "no-match"],
)
def test_remove_openai_responses_unpersisted_input_item_refuses_ambiguous_or_missing(
    input_value: Any,
    item_id: str,
):
    body = {"model": "gpt-5.4", "input": input_value, "store": False}
    before = dict(body)

    assert remove_openai_responses_unpersisted_input_item(body, item_id) is None
    assert body == before


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
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
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
async def test_direct_responses_runtime_repairs_once_then_succeeds():
    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )

    body = _responses_body_with_unpersisted_item()
    success = Response(content=b"ok", status_code=200)
    calls: list[dict[str, Any]] = []

    async def _base_handler(**kwargs: Any) -> Response:
        current_body = await lpe.get_request_body(kwargs["request"])
        calls.append(current_body)
        if len(calls) == 1:
            raise _unpersisted_item_error()
        return success

    seen_bodies, mock_redispatch, response, raised = (
        await _run_direct_responses_runtime_case(body, _base_handler)
    )

    assert raised is None
    assert response is success
    assert len(calls) == 2
    assert [item.get("id") for item in calls[0]["input"]] == [
        item.get("id") for item in body["input"]
    ]
    assert all(item.get("id") != "rs_stale" for item in calls[1]["input"])
    assert len(seen_bodies) == 3
    mock_redispatch.assert_not_called()


@pytest.mark.asyncio
async def test_direct_responses_runtime_no_match_keeps_current_failure():
    body = _responses_body_with_unpersisted_item("rs_other")
    redispatch_error = HTTPException(
        status_code=409,
        detail={"error": {"code": "aawm_session_owner_redispatch_required"}},
    )

    async def _base_handler(**_kwargs: Any) -> Response:
        raise _unpersisted_item_error()

    seen_bodies, mock_redispatch, response, raised = (
        await _run_direct_responses_runtime_case(
            body,
            _base_handler,
            redispatch_side_effect=redispatch_error,
        )
    )

    assert response is None
    assert raised is redispatch_error
    assert len(seen_bodies) == 1
    mock_redispatch.assert_called_once()


@pytest.mark.asyncio
async def test_direct_responses_runtime_second_failure_does_not_loop():
    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )

    body = _responses_body_with_unpersisted_item()
    redispatch_error = HTTPException(
        status_code=409,
        detail={"error": {"code": "aawm_session_owner_redispatch_required"}},
    )
    calls: list[dict[str, Any]] = []

    async def _base_handler(**kwargs: Any) -> Response:
        current_body = await lpe.get_request_body(kwargs["request"])
        calls.append(current_body)
        raise _unpersisted_item_error()

    seen_bodies, mock_redispatch, response, raised = (
        await _run_direct_responses_runtime_case(
            body,
            _base_handler,
            redispatch_side_effect=redispatch_error,
        )
    )

    assert response is None
    assert raised is redispatch_error
    assert len(calls) == 2
    assert all(item.get("id") != "rs_stale" for item in calls[1]["input"])
    assert len(seen_bodies) == 3
    mock_redispatch.assert_called_once()


async def _run_alias_responses_runtime_case(
    body: dict[str, Any],
    *,
    succeeds_after_repair: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], int, Any, BaseException | None]:
    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        attempt_records,
    )

    request = _responses_request(body)
    selection = _direct_codex_selection()
    provider_calls: list[dict[str, Any]] = []
    selection_calls = 0
    success = object()
    redispatch_error = HTTPException(
        status_code=409,
        detail={"error": {"code": "aawm_session_owner_redispatch_required"}},
    )

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        nonlocal selection_calls
        selection_calls += 1
        return selection

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Any:
        del candidate
        provider_calls.append(
            {
                "body": dict(candidate_body),
                "input": list(candidate_body.get("input", [])),
            }
        )
        if succeeds_after_repair and len(provider_calls) == 2:
            return success
        raise _unpersisted_item_error()

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: Any) -> None:
            return None

    async def _owner_guard(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    fake_session_affinity = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: "session-1",
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        reset_released_request_session_owner_guard=lambda _request: False,
        raise_session_owner_redispatch_required=lambda **_kwargs: (
            (_ for _ in ()).throw(redispatch_error)
        ),
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(
        candidate_loop,
        "register_aawm_route_rollup_access_log_replacement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        attempt_records,
        "_bind_auto_agent_alias_request_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(candidate_loop, "_session_affinity_mod", lambda: fake_session_affinity)
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        candidate_loop._dev_fault_plan,
        "_raise_if_openai_fault_plan_slot_fails",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        lambda **kwargs: kwargs["prepared_request_body"],
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        lambda **kwargs: kwargs["prepared_request_body"],
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_success",
        lambda **kwargs: kwargs["prepared_request_body"],
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda *_args, **_kwargs: None,
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=MagicMock(),
        publish_cooldown_memory_fn=MagicMock(),
        persist_cooldown_fn=_noop_async,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda candidate_body, **_kwargs: candidate_body,
        raise_redispatch_fn=MagicMock(),
    )

    try:
        response = await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )
    except BaseException as exc:
        return provider_calls, selection_calls, None, exc
    return provider_calls, selection_calls, response, None


@pytest.mark.asyncio
async def test_alias_responses_runtime_repairs_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    provider_calls, selection_calls, response, raised = (
        await _run_alias_responses_runtime_case(
            _responses_body_with_unpersisted_item(),
            succeeds_after_repair=True,
            monkeypatch=monkeypatch,
        )
    )

    assert raised is None
    assert response is not None
    assert selection_calls == 1
    assert len(provider_calls) == 2
    assert all(
        item.get("id") != "rs_stale" for item in provider_calls[1]["input"]
    )


@pytest.mark.asyncio
async def test_alias_responses_runtime_no_match_keeps_current_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    provider_calls, selection_calls, response, raised = (
        await _run_alias_responses_runtime_case(
            _responses_body_with_unpersisted_item("rs_other"),
            succeeds_after_repair=False,
            monkeypatch=monkeypatch,
        )
    )

    assert response is None
    assert isinstance(raised, HTTPException)
    assert raised.status_code == 409
    assert selection_calls == 1
    assert len(provider_calls) == 1


@pytest.mark.asyncio
async def test_alias_responses_runtime_second_failure_does_not_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    provider_calls, selection_calls, response, raised = (
        await _run_alias_responses_runtime_case(
            _responses_body_with_unpersisted_item(),
            succeeds_after_repair=False,
            monkeypatch=monkeypatch,
        )
    )

    assert response is None
    assert isinstance(raised, HTTPException)
    assert raised.status_code == 409
    assert selection_calls == 1
    assert len(provider_calls) == 2
    assert all(
        item.get("id") != "rs_stale" for item in provider_calls[1]["input"]
    )


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
