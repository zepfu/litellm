"""Shipped inbound Cursor Agent CLI Connect auth, proxy, and session_history."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from litellm.llms.cursor_agent.connect import (
    _ProtoConnectFrameDecoder,
    _encode_proto_message_field,
    _encode_proto_varint_field,
    decode_connect_proto_frames,
    encode_connect_proto_frame,
    encode_cursor_run_request,
)
from litellm.llms.cursor_agent.constants import CURSOR_CLI_KEY_ENV
from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
    CURSOR_AGENT_CLI_INBOUND_PROVIDER,
    CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY,
    CURSOR_AGENT_CLI_INBOUND_TRACE_NAME,
    CursorAgentCliInboundMiddleware,
    InboundCursorAgentCliAuthError,
    _Http1LaneRegistry,
    _connect_header_flush_frame,
    _request_context_exec_replies,
    _summarize_connect_chunk,
    _upstream_request_headers,
    build_inbound_cli_session_history_kwargs,
    encode_bidi_append_request,
    encode_bidi_request_id,
    inbound_cli_auth_error_payload,
    is_cursor_agent_cli_bidi_append_scope,
    is_cursor_agent_cli_run_scope,
    is_cursor_agent_cli_runsse_scope,
    parse_bidi_append_request,
    parse_bidi_request_id,
    proxy_inbound_cli_bidi_append,
    proxy_inbound_cli_run,
    proxy_inbound_cli_runsse,
    require_inbound_cli_bearer,
)


def _run_frame(model_id: str = "composer-2.5", conversation_id: str = "conv-1") -> bytes:
    return encode_cursor_run_request(
        {
            "runRequest": {
                "conversationState": {},
                "action": {
                    "userMessageAction": {
                        "userMessage": {
                            "text": "ping",
                            "messageId": "message-1",
                            "selectedContext": {},
                            "mode": "AGENT_MODE_AGENT",
                        }
                    }
                },
                "requestedModel": {"modelId": model_id},
                "mcpTools": {},
                "conversationId": conversation_id,
                "conversationGroupId": conversation_id,
                "runId": "run-inbound-1",
            }
        }
    )


def test_require_inbound_cli_bearer_accepts_access_token() -> None:
    token = require_inbound_cli_bearer({"Authorization": "Bearer cursor-access-token"})
    assert token == "cursor-access-token"


def test_require_inbound_cli_bearer_rejects_missing() -> None:
    with pytest.raises(InboundCursorAgentCliAuthError) as exc:
        require_inbound_cli_bearer({})
    assert exc.value.status_code == 401
    assert exc.value.reason == "missing_authorization"
    payload = inbound_cli_auth_error_payload(exc.value)
    assert payload["error"] == "cursor_agent_cli_inbound_auth"
    assert payload["reason"] == "missing_authorization"


def test_require_inbound_cli_bearer_rejects_cloud_agents_basic() -> None:
    basic = base64.b64encode(b"cursor-cloud-key:").decode("ascii")
    with pytest.raises(InboundCursorAgentCliAuthError) as exc:
        require_inbound_cli_bearer({"Authorization": f"Basic {basic}"})
    assert exc.value.reason == "cloud_agents_basic_not_accepted"
    assert exc.value.status_code == 401


def test_require_inbound_cli_bearer_ignores_cli_key_env(monkeypatch) -> None:
    monkeypatch.setenv(CURSOR_CLI_KEY_ENV, "cli-key-must-be-ignored")
    with pytest.raises(InboundCursorAgentCliAuthError) as exc:
        require_inbound_cli_bearer({}, environ={CURSOR_CLI_KEY_ENV: "cli-key-must-be-ignored"})
    assert exc.value.reason == "missing_authorization"


def test_require_inbound_cli_bearer_does_not_use_raw_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "raw-key-must-not-authenticate")
    with pytest.raises(InboundCursorAgentCliAuthError) as exc:
        require_inbound_cli_bearer({})
    assert exc.value.reason == "missing_authorization"


def test_inbound_session_history_kwargs_are_inbound_cli_not_adapters() -> None:
    kwargs = build_inbound_cli_session_history_kwargs(
        call_id="call-1",
        headers={
            "user-agent": "Cursor-CLI/2026.09.08-6caf4ff (linux x64)",
            "x-request-id": "req-1",
            "authorization": "Bearer secret-must-not-be-copied",
        },
        model_id="composer-2.5",
        conversation_id="conv-1",
        run_id="run-1",
        client_host="127.0.0.1",
    )
    metadata = kwargs["litellm_params"]["metadata"]
    tags = kwargs["standard_logging_object"]["request_tags"]
    assert kwargs["custom_llm_provider"] == CURSOR_AGENT_CLI_INBOUND_PROVIDER
    assert kwargs["model"] == "composer-2.5"
    assert kwargs["call_type"] == "pass_through_endpoint"
    assert metadata["inbound_versus_outbound"] == "inbound"
    assert metadata["passthrough_route_family"] == CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY
    assert metadata["trace_name"] == CURSOR_AGENT_CLI_INBOUND_TRACE_NAME
    assert metadata["cursor_connect_method"] == "Run"
    assert "route:cursor_agent_cli_inbound" in tags
    assert "codex-cursor-agent-aiserver-adapter" not in tags
    assert "cursor:agent:create" not in tags
    assert "authorization" not in kwargs["litellm_params"]["proxy_server_request"]["headers"]
    assert kwargs["standard_logging_object"]["session_id"] == "conv-1"


def test_build_session_history_record_for_inbound_cli() -> None:
    from litellm.integrations.aawm_agent_identity import _build_session_history_record

    kwargs = build_inbound_cli_session_history_kwargs(
        call_id="call-history-1",
        headers={"user-agent": "Cursor-CLI/2026.09.08-6caf4ff (linux x64)"},
        model_id="composer-2.5",
        conversation_id="conv-history",
        run_id="run-history",
        client_host="127.0.0.1",
    )
    record = _build_session_history_record(
        kwargs=kwargs,
        result={
            "id": "run-history",
            "object": "cursor_agent_cli_inbound.run",
            "status": "completed",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        start_time="2026-09-10T16:00:00Z",
        end_time="2026-09-10T16:00:01Z",
    )
    assert record is not None
    assert record["session_id"] == "conv-history"
    assert record["provider"] == CURSOR_AGENT_CLI_INBOUND_PROVIDER
    assert record["model"] == "composer-2.5"
    metadata = record["metadata"]
    assert metadata["inbound_versus_outbound"] == "inbound"
    assert metadata["passthrough_route_family"] == CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY
    assert metadata["cursor_connect_method"] == "Run"
    assert metadata["cursor_connect_http_version"] == "2"
    assert metadata["cursor_agent_run_id"] == "run-history"
    assert metadata["cursor_agent_conversation_id"] == "conv-history"
    assert metadata["aawm_passthrough_endpoint_type"] == CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY
    assert "inbound_cli_error" not in metadata
    request_tags = metadata.get("request_tags") or []
    assert "route:cursor_agent_cli_inbound" in request_tags
    assert "codex-cursor-agent-aiserver-adapter" not in request_tags
    assert "cursor:agent:create" not in request_tags
    assert record["client_name"] == "cursor-cli"
    assert record["client_version"] == "2026.09.08-6caf4ff"
    assert record["client_user_agent"].startswith("Cursor-CLI/")
    assert record["litellm_environment"] or record["litellm_version"]


class _FakeSession:
    def __init__(self) -> None:
        self.written: List[bytes] = []
        self.ended = False
        self.opened_headers: Optional[List] = None
        self.response_status = 200
        self.closed = False
        self._chunks = [b"upstream-connect-bytes"]

    async def open(self, request_headers):
        self.opened_headers = request_headers

    async def write_request(self, data: bytes, *, end_stream: bool = False) -> None:
        if data:
            self.written.append(data)
        if end_stream:
            self.ended = True

    async def iter_response_data(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_proxy_inbound_cli_run_forwards_bearer_and_body() -> None:
    session = _FakeSession()
    sent: List[Dict[str, Any]] = []
    request_messages = [
        {"type": "http.request", "body": _run_frame(), "more_body": False},
    ]

    async def receive():
        return request_messages.pop(0)

    async def send(message):
        sent.append(message)

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await proxy_inbound_cli_run(
            {
                "type": "http",
                "http_version": "2",
                "method": "POST",
                "path": "/agent.v1.AgentService/Run",
                "headers": [
                    (b"authorization", b"Bearer cursor-access-token"),
                    (b"content-type", b"application/connect+proto"),
                    (b"user-agent", b"Cursor-CLI/2026.09.08-6caf4ff (linux x64)"),
                ],
                "client": ("127.0.0.1", 9),
            },
            receive,
            send,
            session_factory=lambda: session,
        )

    assert session.opened_headers is not None
    assert ("authorization", "Bearer cursor-access-token") in session.opened_headers
    assert ("te", "trailers") in session.opened_headers
    assert any(b"cursor-access-token" != chunk for chunk in session.written) or session.written
    assert b"raw-key" not in b"".join(session.written)
    assert session.ended is True
    assert session.closed is True
    bodies = [message.get("body") for message in sent if message.get("type") == "http.response.body"]
    assert b"upstream-connect-bytes" in bodies
    persist.assert_awaited()
    persist_kwargs = persist.await_args.kwargs
    assert persist_kwargs["sniffed"]["model_id"] == "composer-2.5"
    assert persist_kwargs["sniffed"]["conversation_id"] == "conv-1"
    assert persist_kwargs["http_version"] == "2"
    assert persist_kwargs["connect_method"] == "Run"
    assert persist_kwargs["status_code"] == 200


@pytest.mark.asyncio
async def test_proxy_inbound_cli_run_auth_error_is_not_404() -> None:
    sent: List[Dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await proxy_inbound_cli_run(
            {
                "type": "http",
                "http_version": "2",
                "method": "POST",
                "path": "/agent.v1.AgentService/Run",
                "headers": [(b"authorization", b"Basic Y3Vyc29yLWtleTo=")],
                "client": ("127.0.0.1", 9),
            },
            receive,
            send,
            session_factory=lambda: _FakeSession(),
        )

    start = next(message for message in sent if message.get("type") == "http.response.start")
    assert start["status"] == 401
    assert start["status"] != 404
    body = next(message["body"] for message in sent if message.get("type") == "http.response.body")
    assert b"cloud_agents_basic_not_accepted" in body
    persist.assert_awaited()


@pytest.mark.asyncio
async def test_proxy_inbound_cli_run_rejects_http1() -> None:
    sent: List[Dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await proxy_inbound_cli_run(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": "/agent.v1.AgentService/Run",
                "headers": [(b"authorization", b"Bearer token")],
            },
            receive,
            send,
        )
    start = next(message for message in sent if message.get("type") == "http.response.start")
    assert start["status"] == 505


def test_connect_header_flush_frame_is_empty_asgi_body() -> None:
    frame = _connect_header_flush_frame()
    assert frame == b""


def _heartbeat_chunk() -> bytes:
    interaction = _encode_proto_message_field(13, b"")
    return encode_connect_proto_frame(_encode_proto_message_field(1, interaction))


def _request_context_chunk() -> bytes:
    exec_server = b"".join(
        (
            _encode_proto_message_field(10, b""),
            _encode_proto_varint_field(19, 1),
            _encode_proto_message_field(55, b""),
        )
    )
    return encode_connect_proto_frame(_encode_proto_message_field(2, exec_server))


def test_summarize_connect_chunk_reports_field_numbers_not_payload() -> None:
    chunk = _request_context_chunk()
    summary = _summarize_connect_chunk(chunk)
    assert "flags=0" in summary
    assert "fields=2" in summary
    assert "nested=2:[10,19,55]" in summary
    assert chunk.hex() not in summary


def test_request_context_exec_replies_answers_agentn_query() -> None:
    decoder = _ProtoConnectFrameDecoder()
    replies, forwarded = _request_context_exec_replies(_request_context_chunk(), decoder)
    assert len(replies) == 2
    assert forwarded == b""
    payloads = [frame.payload for frame in decode_connect_proto_frames(b"".join(replies))]
    assert [_decode_top_fields(payload) for payload in payloads] == [[2], [5]]
    heartbeat = _heartbeat_chunk()
    heartbeat_replies, heartbeat_forwarded = _request_context_exec_replies(
        heartbeat,
        _ProtoConnectFrameDecoder(),
    )
    assert heartbeat_replies == []
    assert heartbeat_forwarded == heartbeat


def test_request_context_exec_replies_does_not_duplicate_split_heartbeat() -> None:
    """A 9-byte heartbeat split at 6 bytes must reassemble to 9, not 15."""
    heartbeat = _heartbeat_chunk()
    assert len(heartbeat) == 9
    decoder = _ProtoConnectFrameDecoder()
    first, second = heartbeat[:6], heartbeat[6:]
    replies1, forwarded1 = _request_context_exec_replies(first, decoder)
    assert replies1 == []
    assert forwarded1 == b""
    assert bytes(decoder.buffer) == first
    replies2, forwarded2 = _request_context_exec_replies(second, decoder)
    assert replies2 == []
    assert forwarded2 == heartbeat
    assert len(forwarded1) + len(forwarded2) == 9
    assert bytes(decoder.buffer) == b""


def _decode_top_fields(payload: bytes) -> List[int]:
    from litellm.llms.cursor_agent.connect import _decode_proto_fields

    return [number for number, _wire, _value in _decode_proto_fields(payload)]


@pytest.mark.asyncio
async def test_proxy_inbound_cli_run_starts_response_before_client_end_body() -> None:
    session = _FakeSession()
    sent: List[Dict[str, Any]] = []
    response_started = asyncio.Event()
    client_ended = asyncio.Event()
    first_chunk = {"type": "http.request", "body": _run_frame(), "more_body": True}

    async def receive():
        if first_chunk:
            message = dict(first_chunk)
            first_chunk.clear()
            return message
        await response_started.wait()
        client_ended.set()
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)
        if message.get("type") == "http.response.start":
            response_started.set()

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await asyncio.wait_for(
            proxy_inbound_cli_run(
                {
                    "type": "http",
                    "http_version": "2",
                    "method": "POST",
                    "path": "/agent.v1.AgentService/Run",
                    "headers": [
                        (b"authorization", b"Bearer cursor-access-token"),
                        (b"content-type", b"application/connect+proto"),
                    ],
                    "client": ("127.0.0.1", 9),
                },
                receive,
                send,
                session_factory=lambda: session,
            ),
            timeout=2,
        )

    assert response_started.is_set()
    assert client_ended.is_set()
    start = next(message for message in sent if message.get("type") == "http.response.start")
    assert start["status"] == 200
    body_events = [message for message in sent if message.get("type") == "http.response.body"]
    assert body_events[0]["body"] == b""
    assert body_events[0]["more_body"] is True
    bodies = [message.get("body") for message in body_events]
    assert b"upstream-connect-bytes" in bodies
    persist.assert_awaited()


@pytest.mark.asyncio
async def test_inbound_middleware_bypasses_fastapi_before_end_body() -> None:
    inner_called = False

    async def inner(scope, receive, send):
        nonlocal inner_called
        inner_called = True
        await asyncio.sleep(3600)

    middleware = CursorAgentCliInboundMiddleware(inner)
    session = _FakeSession()
    sent: List[Dict[str, Any]] = []
    response_started = asyncio.Event()
    first_chunk = {"type": "http.request", "body": _run_frame(), "more_body": True}

    async def receive():
        if first_chunk:
            message = dict(first_chunk)
            first_chunk.clear()
            return message
        await response_started.wait()
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)
        if message.get("type") == "http.response.start":
            response_started.set()

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ), patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._AgentnH2Session",
        side_effect=lambda *args, **kwargs: session,
    ):
        await asyncio.wait_for(
            middleware(
                {
                    "type": "http",
                    "http_version": "2",
                    "method": "POST",
                    "path": "/agent.v1.AgentService/Run",
                    "headers": [
                        (b"authorization", b"Bearer cursor-access-token"),
                        (b"content-type", b"application/connect+proto"),
                    ],
                    "client": ("127.0.0.1", 9),
                },
                receive,
                send,
            ),
            timeout=2,
        )

    assert inner_called is False
    assert any(message.get("type") == "http.response.start" for message in sent)


@pytest.mark.asyncio
async def test_proxy_inbound_cli_run_persists_when_client_disconnects_before_upstream() -> None:
    class _HangUntilClosedSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self._closed = asyncio.Event()
            self._chunks = []

        async def iter_response_data(self):
            await self._closed.wait()
            if False:
                yield b""

        async def aclose(self) -> None:
            self.closed = True
            self._closed.set()

    session = _HangUntilClosedSession()
    sent: List[Dict[str, Any]] = []
    response_started = asyncio.Event()
    first_chunk = {"type": "http.request", "body": _run_frame(), "more_body": True}

    async def receive():
        if first_chunk:
            message = dict(first_chunk)
            first_chunk.clear()
            return message
        await response_started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message.get("type") == "http.response.start":
            response_started.set()

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await asyncio.wait_for(
            proxy_inbound_cli_run(
                {
                    "type": "http",
                    "http_version": "2",
                    "method": "POST",
                    "path": "/agent.v1.AgentService/Run",
                    "headers": [
                        (b"authorization", b"Bearer cursor-access-token"),
                        (b"content-type", b"application/connect+proto"),
                    ],
                    "client": ("127.0.0.1", 9),
                },
                receive,
                send,
                session_factory=lambda: session,
            ),
            timeout=2,
        )

    assert session.closed is True
    persist.assert_awaited()


@pytest.mark.asyncio
async def test_agentn_session_aclose_does_not_wait_for_peer() -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    class _HangingWriter:
        def __init__(self) -> None:
            self.closed = False
            self.aborted = False

        def close(self) -> None:
            self.closed = True

        def abort(self) -> None:
            self.aborted = True

        async def wait_closed(self) -> None:
            await asyncio.Event().wait()

    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    writer = _HangingWriter()
    session.writer = writer
    session.reader = object()
    await asyncio.wait_for(session.aclose(), timeout=2)
    assert writer.closed is True
    assert writer.aborted is True
    assert session.writer is None


def test_agentn_session_dispatches_data_received_without_dropping() -> None:
    from h2.events import DataReceived, ResponseReceived

    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    class _AckConnection:
        def acknowledge_received_data(self, *_args, **_kwargs) -> None:
            return None

    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = _AckConnection()
    headers = object.__new__(ResponseReceived)
    headers.headers = [(":status", "200"), ("content-type", "application/connect+proto")]
    payload = _request_context_chunk()
    data = object.__new__(DataReceived)
    data.data = payload
    data.flow_controlled_length = len(data.data)
    data.stream_id = 1
    chunks, ended = session._dispatch_h2_events([headers, data])
    assert session.response_status == 200
    assert chunks == []
    assert ended is False
    assert len(session._auto_replies) == 2


def test_proxy_server_wraps_inbound_cli_run_as_raw_asgi() -> None:
    source = (
        Path(__file__).resolve().parents[4] / "litellm" / "proxy" / "proxy_server.py"
    ).read_text(encoding="utf-8")
    assert "CursorAgentCliInboundMiddleware" in source
    assert "add_middleware(CursorAgentCliInboundMiddleware)" in source


def test_parse_cursor_cli_user_agent() -> None:
    from litellm.integrations.aawm_agent_identity import _parse_client_identity_from_user_agent

    name, version = _parse_client_identity_from_user_agent(
        "Cursor-CLI/2026.09.08-6caf4ff (linux x64)"
    )
    assert name == "cursor-cli"
    assert version == "2026.09.08-6caf4ff"


def test_http2_run_scope_does_not_match_runsse_or_bidiappend() -> None:
    run_scope = {
        "type": "http",
        "method": "POST",
        "path": "/agent.v1.AgentService/Run",
    }
    runsse_scope = {
        "type": "http",
        "method": "POST",
        "path": "/agent.v1.AgentService/RunSSE",
    }
    bidi_scope = {
        "type": "http",
        "method": "POST",
        "path": "/aiserver.v1.BidiService/BidiAppend",
    }
    poll_scope = {
        "type": "http",
        "method": "POST",
        "path": "/agent.v1.AgentService/RunPoll",
    }
    assert is_cursor_agent_cli_run_scope(run_scope) is True
    assert is_cursor_agent_cli_run_scope(runsse_scope) is False
    assert is_cursor_agent_cli_run_scope(bidi_scope) is False
    assert is_cursor_agent_cli_run_scope(poll_scope) is False
    assert is_cursor_agent_cli_runsse_scope(runsse_scope) is True
    assert is_cursor_agent_cli_runsse_scope(run_scope) is False
    assert is_cursor_agent_cli_bidi_append_scope(bidi_scope) is True
    assert is_cursor_agent_cli_bidi_append_scope(run_scope) is False


def test_upstream_http2_headers_omit_cursor_streaming() -> None:
    forwarded = _upstream_request_headers(
        {
            "authorization": "Bearer cursor-access-token",
            "content-type": "application/connect+proto",
            "x-cursor-streaming": "true",
            "x-request-id": "req-http2",
        },
        "cursor-access-token",
    )
    names = {name for name, _value in forwarded}
    assert ("authorization", "Bearer cursor-access-token") in forwarded
    assert "x-cursor-streaming" not in names
    assert ("x-request-id", "req-http2") in forwarded


def test_bidi_append_round_trip_binary_and_hex() -> None:
    payload = decode_connect_proto_frames(_run_frame())[0].payload
    encoded = encode_bidi_append_request("req-lane-1", 0, payload, binary=True)
    parsed = parse_bidi_append_request(encoded, "application/proto")
    assert parsed["request_id"] == "req-lane-1"
    assert parsed["append_seqno"] == 0
    assert parsed["binary"] is True
    assert parsed["client_message"] == payload

    hex_encoded = encode_bidi_append_request("req-lane-1", 1, payload, binary=False)
    hex_parsed = parse_bidi_append_request(hex_encoded, "application/proto")
    assert hex_parsed["request_id"] == "req-lane-1"
    assert hex_parsed["append_seqno"] == 1
    assert hex_parsed["binary"] is False
    assert hex_parsed["client_message"] == payload


def test_parse_bidi_request_id_from_connect_envelope() -> None:
    body = encode_bidi_request_id("req-lane-1")
    assert parse_bidi_request_id(body, "application/connect+proto") == "req-lane-1"


@pytest.mark.asyncio
async def test_proxy_inbound_cli_runsse_rejects_http2() -> None:
    sent: List[Dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await proxy_inbound_cli_runsse(
            {
                "type": "http",
                "http_version": "2",
                "method": "POST",
                "path": "/agent.v1.AgentService/RunSSE",
                "headers": [(b"authorization", b"Bearer cursor-access-token")],
            },
            receive,
            send,
        )
    start = next(message for message in sent if message.get("type") == "http.response.start")
    assert start["status"] == 505
    persist.assert_awaited()
    assert persist.await_args.kwargs["connect_method"] == "RunSSE"


@pytest.mark.asyncio
async def test_proxy_inbound_cli_runsse_and_bidiappend_share_agentn_lane() -> None:
    class _LaneSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self._wrote = asyncio.Event()
            self._closed = asyncio.Event()
            self._chunks = [b"upstream-connect-bytes"]

        async def write_request(self, data: bytes, *, end_stream: bool = False) -> None:
            await super().write_request(data, end_stream=end_stream)
            if data:
                self._wrote.set()

        async def iter_response_data(self):
            await self._wrote.wait()
            for chunk in list(self._chunks):
                yield chunk

        async def aclose(self) -> None:
            self.closed = True
            self._closed.set()

    session = _LaneSession()
    lanes = _Http1LaneRegistry()
    persist = AsyncMock()
    run_payload = decode_connect_proto_frames(_run_frame())[0].payload
    request_id = "req-lane-1"
    expected_agentn_frame = encode_connect_proto_frame(run_payload)

    runsse_messages = [
        {
            "type": "http.request",
            "body": encode_bidi_request_id(request_id),
            "more_body": False,
        }
    ]
    append_messages = [
        {
            "type": "http.request",
            "body": encode_bidi_append_request(request_id, 0, run_payload, binary=True),
            "more_body": False,
        }
    ]
    sent_runsse: List[Dict[str, Any]] = []
    sent_append: List[Dict[str, Any]] = []
    response_started = asyncio.Event()

    async def receive_runsse():
        if runsse_messages:
            return runsse_messages.pop(0)
        await response_started.wait()
        return {"type": "http.disconnect"}

    async def send_runsse(message):
        sent_runsse.append(message)
        if message.get("type") == "http.response.start":
            response_started.set()

    async def receive_append():
        return append_messages.pop(0)

    async def send_append(message):
        sent_append.append(message)

    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        runsse_task = asyncio.create_task(
            proxy_inbound_cli_runsse(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "method": "POST",
                    "path": "/agent.v1.AgentService/RunSSE",
                    "headers": [
                        (b"authorization", b"Bearer cursor-access-token"),
                        (b"content-type", b"application/connect+proto"),
                        (b"x-cursor-streaming", b"true"),
                        (b"x-request-id", request_id.encode("ascii")),
                    ],
                    "client": ("127.0.0.1", 9),
                },
                receive_runsse,
                send_runsse,
                session_factory=lambda: session,
                lanes=lanes,
            )
        )
        await asyncio.wait_for(response_started.wait(), timeout=2)
        await proxy_inbound_cli_bidi_append(
            {
                "type": "http",
                "method": "POST",
                "path": "/aiserver.v1.BidiService/BidiAppend",
                "headers": [
                    (b"authorization", b"Bearer cursor-access-token"),
                    (b"content-type", b"application/proto"),
                    (b"x-cursor-streaming", b"true"),
                    (b"x-request-id", request_id.encode("ascii")),
                ],
            },
            receive_append,
            send_append,
            lanes=lanes,
        )
        await asyncio.wait_for(runsse_task, timeout=2)

    assert session.opened_headers is not None
    names = {name for name, _value in session.opened_headers}
    assert ("authorization", "Bearer cursor-access-token") in session.opened_headers
    assert "x-cursor-streaming" not in names
    assert expected_agentn_frame in session.written
    append_start = next(
        message for message in sent_append if message.get("type") == "http.response.start"
    )
    assert append_start["status"] == 200
    runsse_start = next(
        message for message in sent_runsse if message.get("type") == "http.response.start"
    )
    assert runsse_start["status"] == 200
    persist.assert_awaited()
    persist_kwargs = persist.await_args.kwargs
    assert persist_kwargs["connect_method"] == "RunSSE"
    assert persist_kwargs["http_version"] == "1.1"
    assert persist_kwargs["sniffed"]["model_id"] == "composer-2.5"


@pytest.mark.asyncio
async def test_inbound_middleware_dispatches_http1_without_fastapi() -> None:
    inner_called = False

    async def inner(scope, receive, send):
        nonlocal inner_called
        inner_called = True
        await asyncio.sleep(3600)

    middleware = CursorAgentCliInboundMiddleware(inner)
    sent: List[Dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    persist = AsyncMock()
    with patch(
        "litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound._persist_inbound_cli_turn",
        persist,
    ):
        await middleware(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": "/agent.v1.AgentService/RunSSE",
                "headers": [(b"authorization", b"Basic Y3Vyc29yLWtleTo=")],
            },
            receive,
            send,
        )
    assert inner_called is False
    start = next(message for message in sent if message.get("type") == "http.response.start")
    assert start["status"] == 401
