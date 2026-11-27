import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Request, Response

from litellm.integrations.aawm_session_history import writer
from litellm.integrations.aawm_session_history.waits import (
    SessionHistoryWaitMiddleware,
    record_candidate_selection_failure,
    record_responses_terminal_delivery,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_wire import (
    OpenAIResponsesStreamingResponse,
    wrap_openai_responses_stream,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "selector_failed"),
    [(422, False), (None, False), (500, True), (200, True)],
)
async def test_terminal_wait_checkpoint_retains_sent_status(monkeypatch, status_code, selector_failed):
    checkpoints = []
    sent_messages = []
    monkeypatch.setattr(writer, "_build_session_history_dsn", lambda: "test-dsn")
    monkeypatch.setattr(writer, "_enqueue_session_history_record", checkpoints.append)

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    async def send(message):
        sent_messages.append(message)

    async def app(scope, receive, send):
        await receive()
        if status_code is None:
            raise RuntimeError("Failed before headers")
        if selector_failed:
            try:
                raise ValueError("selection diagnostic with sensitive text")
            except ValueError as error:
                record_candidate_selection_failure(error)
        await send({"type": "http.response.start", "status": status_code, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = SessionHistoryWaitMiddleware(app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [(b"x-litellm-call-id", b"test-call")],
    }
    if status_code is None:
        with pytest.raises(RuntimeError, match="Failed before headers"):
            await middleware(scope, receive, send)
        assert not sent_messages
    else:
        await middleware(scope, receive, send)
        assert sent_messages[0]["status"] == status_code

    checkpoint = checkpoints[-1]
    assert checkpoint["litellm_call_id"] == "test-call"
    wait = checkpoint["metadata"]["wait_accounting"]
    assert wait["status"] == ("completed" if status_code == 200 else "failed")
    if status_code is None:
        assert "response_status_code" not in wait
    else:
        assert wait["response_status_code"] == status_code
    if selector_failed and status_code == 500:
        assert wait["candidate_selection_failure"] == {"exception_type": "ValueError"}
    else:
        assert "candidate_selection_failure" not in wait


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_status", "disconnect_after_terminal", "expected"),
    [
        ("completed", True, "completed"),
        ("completed", False, "disconnected"),
        ("failed", True, "disconnected"),
    ],
)
async def test_disconnect_after_delivered_responses_terminal(
    monkeypatch, response_status, disconnect_after_terminal, expected
):
    checkpoints = []
    completed_sends = []
    disconnect = asyncio.Event()
    pending_send = asyncio.Event()
    monkeypatch.setattr(writer, "_build_session_history_dsn", lambda: "test-dsn")
    monkeypatch.setattr(writer, "_enqueue_session_history_record", checkpoints.append)

    event_type = f"response.{response_status}"
    payload = {"type": event_type, "response": {"status": response_status}}
    terminal_frame = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

    async def source():
        yield terminal_frame

    async def app(scope, receive, send):
        chunks, trace = wrap_openai_responses_stream(
            source(), upstream_response=httpx.Response(200)
        )
        trace.register_post_finalization_callback(record_responses_terminal_delivery)
        await OpenAIResponsesStreamingResponse(chunks, wire_trace=trace)(scope, receive, send)

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] != "http.response.body":
            return
        disconnect.set()
        if disconnect_after_terminal and message["body"] == terminal_frame:
            completed_sends.append(message["body"])
            return
        await pending_send.wait()

    await asyncio.wait_for(
        SessionHistoryWaitMiddleware(app)(
            {"type": "http", "method": "POST", "path": "/openai_passthrough/responses", "headers": []},
            receive,
            send,
        ),
        timeout=2,
    )

    assert completed_sends == ([terminal_frame] if disconnect_after_terminal else [])
    assert checkpoints[-1]["metadata"]["wait_accounting"]["status"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_returned", [False, True])
async def test_codex_alias_http500_retains_only_local_failure_origin(monkeypatch, tmp_path, provider_returned):
    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as routes
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import candidate_loop, config_startup

    checkpoints = []
    sent_messages = []
    monkeypatch.setattr(writer, "_build_session_history_dsn", lambda: "test-dsn")
    monkeypatch.setattr(writer, "_enqueue_session_history_record", checkpoints.append)
    monkeypatch.setattr(routes, "get_request_body", AsyncMock(return_value={"model": "work"}))
    monkeypatch.setattr(routes, "maybe_proxy_muse_code_responses", AsyncMock(return_value=None))
    monkeypatch.setattr(routes, "_resolve_codex_auto_agent_alias_model", lambda *a, **kw: "work")
    monkeypatch.setattr(routes, "_should_use_direct_codex_oauth_inventory", lambda *a, **kw: False)

    def fail_preflight():
        try:
            config_startup.compile_directory(tmp_path / "missing")
        except config_startup.ConfigDirectoryError as error:
            raise RuntimeError("local preflight failed") from error

    async def fail_after_selection(**kwargs):
        try:
            fail_preflight()
        except RuntimeError as wrapped:
            error = candidate_loop._proxy_exception_for_unclassified_probe_failure(wrapped)
            error._aawm_provider_returned = provider_returned
            raise error

    monkeypatch.setattr(
        routes.BaseOpenAIPassThroughHandler,
        "_base_openai_pass_through_handler",
        fail_after_selection,
    )

    async def app(scope, receive, send):
        request = Request(scope, receive=receive)
        with pytest.raises(ProxyException) as error:
            await routes.openai_proxy_route("responses", request, Response(), UserAPIKeyAuth())
        await Response(status_code=int(error.value.code))(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    async def send(message):
        sent_messages.append(message)

    await SessionHistoryWaitMiddleware(app)(
        {"type": "http", "method": "POST", "path": "/openai_passthrough/responses", "headers": []},
        receive,
        send,
    )

    assert sent_messages[0]["status"] == 500
    wait = checkpoints[-1]["metadata"]["wait_accounting"]
    assert wait["response_status_code"] == 500
    if provider_returned:
        assert "terminal_alias_failure" not in wait
    else:
        failure = wait["terminal_alias_failure"]
        assert set(failure) == {"exception_type", "module", "function", "line"}
        assert failure["exception_type"] == "ConfigDirectoryError"
        assert failure["module"] == config_startup.__name__
        assert failure["function"] and failure["line"] > 0
