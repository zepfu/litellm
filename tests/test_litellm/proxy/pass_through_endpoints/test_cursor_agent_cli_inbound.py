"""Shipped inbound Cursor Agent CLI Connect auth, proxy, and session_history."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from litellm.llms.cursor_agent.connect import (
    CONNECT_COMPRESSED_FLAG,
    _ProtoConnectFrameDecoder,
    _encode_proto_message_field,
    _encode_proto_varint_field,
    decode_connect_proto_frames,
    encode_connect_proto_frame,
    encode_cursor_run_request,
)
from litellm.llms.cursor_agent.constants import CURSOR_CLI_KEY_ENV
from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
    AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG,
    CURSOR_AGENT_CLI_INBOUND_PROVIDER,
    CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY,
    CURSOR_AGENT_CLI_INBOUND_TRACE_NAME,
    CursorAgentCliInboundMiddleware,
    InboundCursorAgentCliAuthError,
    InboundCursorAgentCliProtocolError,
    _Http1LaneRegistry,
    _InboundRunProvenance,
    _cli_connect_envelope,
    _connect_header_flush_frame,
    _log_inbound_run_terminal,
    _observe_upstream_headers,
    _request_context_exec_replies,
    _rewrite_cli_connect_bytes,
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


def test_require_inbound_cli_bearer_does_not_use_auth_token_env(monkeypatch) -> None:
    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "stored-access-token-is-not-a-header")
    with pytest.raises(InboundCursorAgentCliAuthError) as exc:
        require_inbound_cli_bearer({})
    assert exc.value.reason == "missing_authorization"


def test_require_inbound_cli_bearer_rejects_empty_bearer() -> None:
    with pytest.raises(InboundCursorAgentCliAuthError) as exc:
        require_inbound_cli_bearer({"Authorization": "Bearer   "})
    assert exc.value.reason == "empty_bearer"
    assert exc.value.status_code == 401


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
    assert "inbound-versus-outbound:inbound" in tags
    assert "codex-cursor-agent-aiserver-adapter" not in tags
    assert "anthropic-cursor-agent-aiserver-adapter" not in tags
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
    assert "inbound-versus-outbound:inbound" in request_tags
    assert "codex-cursor-agent-aiserver-adapter" not in request_tags
    assert "anthropic-cursor-agent-aiserver-adapter" not in request_tags
    assert "cursor:agent:create" not in request_tags
    assert record["client_name"] == "cursor-cli"
    assert record["client_version"] == "2026.09.08-6caf4ff"
    assert record["client_user_agent"].startswith("Cursor-CLI/")
    assert record["litellm_environment"] or record["litellm_version"]
    assert record["provider"] not in {"cursor_agent", "cursor"}
    assert record.get("tool_activity") in (None, [])
    assert (record.get("tool_call_count") or 0) == 0


class _FakeSession:
    def __init__(self) -> None:
        self.written: List[bytes] = []
        self.ended = False
        self.opened_headers: Optional[List] = None
        self.response_status = 200
        self.closed = False
        self._chunks = [b"upstream-connect-bytes"]
        self.reader_termination_event = asyncio.Event()
        self._flush_termination_event = asyncio.Event()
        self.upstream_termination_reason: Optional[str] = None
        self.flush_termination_reason: Optional[str] = None

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

    async def aclose(self, reason: str = "normal_response") -> bool:
        self.closed = True
        return True


@pytest.mark.asyncio
async def test_proxy_inbound_cli_run_forwards_bearer_and_body(monkeypatch) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "raw-key-must-not-be-sent")
    monkeypatch.setenv(CURSOR_CLI_KEY_ENV, "cli-key-must-be-ignored")
    session = _FakeSession()
    sent: List[Dict[str, Any]] = []
    request_messages = [
        {"type": "http.request", "body": _run_frame(), "more_body": False},
    ]
    parked = asyncio.Event()

    async def receive():
        if request_messages:
            return request_messages.pop(0)
        await parked.wait()
        return {"type": "http.disconnect"}

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
    assert ("authorization", "Bearer raw-key-must-not-be-sent") not in session.opened_headers
    assert all("raw-key-must-not-be-sent" not in value for _name, value in session.opened_headers)
    assert all("cli-key-must-be-ignored" not in value for _name, value in session.opened_headers)
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
    assert start["status"] != 404
    body = next(message["body"] for message in sent if message.get("type") == "http.response.body")
    assert b"http2_required" in body
    persist.assert_awaited()
    persist_kwargs = persist.await_args.kwargs
    assert persist_kwargs["connect_method"] == "Run"
    assert persist_kwargs["http_version"] == "1.1"


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


def test_request_context_exec_replies_forwards_gzip_uncompressed() -> None:
    """Agentn gzip Connect frames must reach the CLI without compression bit 0.

    The stock CLI errors with ``received compressed envelope, but do not know
    how to decompress`` when inbound forwards gzip envelopes. The shipped
    helper must decode then re-encode uncompressed.
    """
    heartbeat = _heartbeat_chunk()
    payload = decode_connect_proto_frames(heartbeat)[0].payload
    compressed = encode_connect_proto_frame(payload, compress=True)
    assert compressed[0] & CONNECT_COMPRESSED_FLAG
    assert compressed != heartbeat
    replies, forwarded = _request_context_exec_replies(
        compressed,
        _ProtoConnectFrameDecoder(),
    )
    assert replies == []
    assert forwarded[0] & CONNECT_COMPRESSED_FLAG == 0
    frames = decode_connect_proto_frames(forwarded)
    assert len(frames) == 1
    assert frames[0].payload == payload
    assert frames[0].flags & CONNECT_COMPRESSED_FLAG == 0
    assert forwarded == heartbeat


def test_request_context_exec_replies_answers_gzip_query_without_forward() -> None:
    query = _request_context_chunk()
    payload = decode_connect_proto_frames(query)[0].payload
    compressed = encode_connect_proto_frame(payload, compress=True)
    assert compressed[0] & CONNECT_COMPRESSED_FLAG
    replies, forwarded = _request_context_exec_replies(
        compressed,
        _ProtoConnectFrameDecoder(),
    )
    assert forwarded == b""
    assert len(replies) == 2
    for reply in replies:
        assert reply[0] & CONNECT_COMPRESSED_FLAG == 0


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

    parked = asyncio.Event()

    async def receive():
        if first_chunk:
            message = dict(first_chunk)
            first_chunk.clear()
            return message
        await response_started.wait()
        if not client_ended.is_set():
            client_ended.set()
            return {"type": "http.request", "body": b"", "more_body": False}
        await parked.wait()
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
    parked = asyncio.Event()
    body_ended = False

    async def receive():
        nonlocal body_ended
        if first_chunk:
            message = dict(first_chunk)
            first_chunk.clear()
            return message
        await response_started.wait()
        if not body_ended:
            body_ended = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await parked.wait()
        return {"type": "http.disconnect"}

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

        async def aclose(self, reason: str = "normal_response") -> bool:
            self.closed = True
            self._closed.set()
            return await super().aclose(reason=reason)

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


def test_agentn_session_connection_terminated_marks_session_unusable(caplog) -> None:
    from h2.events import ConnectionTerminated, DataReceived

    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    class _AckConnection:
        def acknowledge_received_data(self, *_args, **_kwargs) -> None:
            return None

    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = _AckConnection()
    session.stream_id = 1
    goaway = object.__new__(ConnectionTerminated)
    goaway.error_code = 0
    goaway.last_stream_id = 0
    goaway.additional_data = None
    stale = object.__new__(DataReceived)
    stale.data = _heartbeat_chunk()
    stale.flow_controlled_length = len(stale.data)
    stale.stream_id = 1
    chunks, ended = session._dispatch_h2_events([goaway, stale])
    assert ended is True
    assert chunks == []
    assert session._connection_terminated is True
    assert session._closed is True
    assert session.upstream_termination_reason == "upstream_reset"
    with pytest.raises(asyncio.QueueEmpty):
        session._incoming.get_nowait()
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert any(
        "cursor_agent_cli_inbound agentn stream closed event=ConnectionTerminated"
        in message
        for message in warning_messages
    ), warning_messages


class _CountingWriter:
    def __init__(self) -> None:
        self.writes = 0
        self.closed = False
        self.aborted = False

    def write(self, _data: bytes) -> None:
        self.writes += 1

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True

    async def wait_closed(self) -> None:
        return None


def _agentn_response_then_goaway(*, end_stream: bool = True) -> tuple[Any, bytes, bytes]:
    from h2.config import H2Configuration
    from h2.connection import H2Connection

    client = H2Connection(
        config=H2Configuration(client_side=True, header_encoding="utf-8")
    )
    server = H2Connection(
        config=H2Configuration(client_side=False, header_encoding="utf-8")
    )
    client.initiate_connection()
    server.initiate_connection()
    server.receive_data(client.data_to_send())
    client.receive_data(server.data_to_send())
    stream_id = client.get_next_available_stream_id()
    client.send_headers(
        stream_id,
        [
            (":method", "POST"),
            (":scheme", "https"),
            (":authority", "agentn.global.api5.cursor.sh"),
            (":path", "/agent.v1.AgentService/Run"),
        ],
        end_stream=False,
    )
    server.receive_data(client.data_to_send())
    payload = _heartbeat_chunk()
    server.send_headers(
        stream_id,
        [(":status", "200"), ("content-type", "application/connect+proto")],
        end_stream=False,
    )
    server.send_data(stream_id, payload, end_stream=end_stream)
    data_frames = server.data_to_send()
    server.close_connection()
    goaway = server.data_to_send()
    return client, data_frames, goaway


@pytest.mark.asyncio
async def test_agentn_session_receive_data_goaway_stops_read_loop() -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    client, _data_frames, goaway = _agentn_response_then_goaway()
    assert goaway

    class _Reader:
        def __init__(self) -> None:
            self._payloads = [goaway, b""]

        async def read(self, _size: int) -> bytes:
            if self._payloads:
                return self._payloads.pop(0)
            return b""

    writer = _CountingWriter()
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = client
    session.reader = _Reader()
    session.writer = writer
    session.stream_id = 1
    writes_before = writer.writes
    await asyncio.wait_for(session._read_loop(), timeout=2)
    assert session._connection_terminated is True
    assert session._closed is True
    assert session.upstream_termination_reason == "upstream_reset"
    assert session.reader_termination_event.is_set()
    assert writer.writes == writes_before
    await asyncio.wait_for(session.aclose(reason="upstream_reset"), timeout=2)
    await asyncio.wait_for(session.aclose(reason="upstream_reset"), timeout=2)
    assert writer.closed is True
    assert writer.aborted is True


@pytest.mark.asyncio
async def test_agentn_session_delivers_payload_before_coalesced_goaway() -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    client, data_frames, goaway = _agentn_response_then_goaway()
    coalesced = data_frames + goaway

    class _Reader:
        def __init__(self) -> None:
            self._payloads = [coalesced, b""]

        async def read(self, _size: int) -> bytes:
            if self._payloads:
                return self._payloads.pop(0)
            return b""

    writer = _CountingWriter()
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = client
    session.reader = _Reader()
    session.writer = writer
    session.stream_id = 1
    collected: List[Optional[bytes]] = []

    async def _consume() -> None:
        async for chunk in session.iter_response_data():
            collected.append(chunk)

    consumer = asyncio.create_task(_consume())
    await asyncio.wait_for(session._read_loop(), timeout=2)
    await asyncio.wait_for(consumer, timeout=2)
    assert collected == [_heartbeat_chunk()]
    assert session._connection_terminated is True
    assert writer.writes == 0


@pytest.mark.asyncio
async def test_agentn_session_delivers_payload_before_fragmented_goaway() -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    client, data_frames, goaway = _agentn_response_then_goaway(end_stream=False)

    class _Reader:
        def __init__(self) -> None:
            self._payloads = [data_frames, goaway, b""]

        async def read(self, _size: int) -> bytes:
            if self._payloads:
                return self._payloads.pop(0)
            return b""

    writer = _CountingWriter()
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = client
    session.reader = _Reader()
    session.writer = writer
    session.stream_id = 1
    collected: List[Optional[bytes]] = []

    async def _consume() -> None:
        async for chunk in session.iter_response_data():
            collected.append(chunk)

    consumer = asyncio.create_task(_consume())
    await asyncio.wait_for(session._read_loop(), timeout=2)
    await asyncio.wait_for(consumer, timeout=2)
    assert collected == [_heartbeat_chunk()]
    assert session._connection_terminated is True
    assert writer.writes == 0


def test_agentn_session_stream_reset_on_other_stream_does_not_close_session() -> None:
    from h2.events import StreamReset

    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.stream_id = 1
    other = object.__new__(StreamReset)
    other.stream_id = 3
    other.error_code = 8
    chunks, ended = session._dispatch_h2_events([other])
    assert ended is False
    assert chunks == []
    assert session._closed is False
    assert session._connection_terminated is False
    assert session.upstream_termination_reason is None


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


def test_upstream_http2_headers_preserve_inbound_checksum() -> None:
    forwarded = _upstream_request_headers(
        {
            "authorization": "Bearer cursor-access-token",
            "content-type": "application/connect+proto",
            "x-cursor-checksum": "opaque-checksum-value",
            "x-cursor-streaming": "true",
            "x-request-id": "req-http2",
            "user-agent": "Cursor-CLI/2026.09.15-d2fe57e (linux x64)",
        },
        "cursor-access-token",
    )
    names = {name for name, _value in forwarded}
    assert ("x-cursor-checksum", "opaque-checksum-value") in forwarded
    assert "x-cursor-streaming" not in names
    assert (
        "user-agent",
        "Cursor-CLI/2026.09.15-d2fe57e (linux x64)",
    ) in forwarded
    provenance = _InboundRunProvenance(call_id="call-1")
    _observe_upstream_headers(
        {
            "authorization": "Bearer cursor-access-token",
            "x-cursor-checksum": "opaque-checksum-value",
            "x-cursor-streaming": "true",
            "user-agent": "Cursor-CLI/2026.09.15-d2fe57e (linux x64)",
        },
        "cursor-access-token",
        forwarded,
        provenance,
    )
    assert provenance.checksum_in is True
    assert provenance.checksum_out is True
    assert provenance.checksum_in_eq_out is True
    assert provenance.streaming_in is True
    assert provenance.ua_source == "inbound"
    assert provenance.bearer_in_eq_out is True


def test_upstream_http2_headers_do_not_synthesize_checksum() -> None:
    forwarded = _upstream_request_headers(
        {
            "authorization": "Bearer cursor-access-token",
            "content-type": "application/connect+proto",
        },
        "cursor-access-token",
    )
    names = {name for name, _value in forwarded}
    assert "x-cursor-checksum" not in names


def test_sequential_runs_keep_distinct_bearers_and_checksums() -> None:
    first = _upstream_request_headers(
        {
            "authorization": "Bearer token-a",
            "content-type": "application/connect+proto",
            "x-cursor-checksum": "checksum-a",
        },
        "token-a",
    )
    second = _upstream_request_headers(
        {
            "authorization": "Bearer token-b",
            "content-type": "application/connect+proto",
            "x-cursor-checksum": "checksum-b",
        },
        "token-b",
    )
    assert ("authorization", "Bearer token-a") in first
    assert ("x-cursor-checksum", "checksum-a") in first
    assert ("authorization", "Bearer token-b") in second
    assert ("x-cursor-checksum", "checksum-b") in second
    assert ("authorization", "Bearer token-b") not in first
    assert ("x-cursor-checksum", "checksum-b") not in first


def test_upstream_http2_headers_force_identity_encoding() -> None:
    forwarded = _upstream_request_headers(
        {
            "authorization": "Bearer cursor-access-token",
            "content-type": "application/connect+proto",
            "accept-encoding": "gzip, deflate, br",
            "connect-accept-encoding": "gzip",
            "grpc-accept-encoding": "gzip",
        },
        "cursor-access-token",
    )
    names = {name for name, _value in forwarded}
    assert ("accept-encoding", "identity") in forwarded
    assert "connect-accept-encoding" not in names
    assert "grpc-accept-encoding" not in names
    assert "grpc-encoding" not in names
    encodings = [value for name, value in forwarded if name == "accept-encoding"]
    assert encodings == ["identity"]


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

        async def aclose(self, reason: str = "normal_response") -> bool:
            self.closed = True
            self._closed.set()
            return True

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

    parked = asyncio.Event()

    async def receive_runsse():
        if runsse_messages:
            return runsse_messages.pop(0)
        await response_started.wait()
        await parked.wait()
        return {"type": "http.disconnect"}

    async def send_runsse(message):
        sent_runsse.append(message)
        if message.get("type") == "http.response.start":
            response_started.set()

    append_parked = asyncio.Event()

    async def receive_append():
        if append_messages:
            return append_messages.pop(0)
        await append_parked.wait()
        return {"type": "http.disconnect"}

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


_HEALTHY_INBOUND_LOG_MARKERS = (
    "cursor_agent_cli_inbound lifecycle",
    "cursor_agent_cli_inbound Run http_version=",
    "cursor_agent_cli_inbound RunSSE http_version=",
    "cursor_agent_cli_inbound opened agentn stream_id=",
    "cursor_agent_cli_inbound agentn response status=",
    "cursor_agent_cli_inbound agentn data bytes=",
    "cursor_agent_cli_inbound agentn read EOF",
    "cursor_agent_cli_inbound auto-answered request_context",
    "cursor_agent_cli_inbound wrote agentn bytes=",
    "cursor_agent_cli_inbound forwarded bytes=",
)


def _assert_healthy_inbound_logs_not_info(caplog) -> None:
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "INFO"
    ]
    hits = [
        message
        for message in info_messages
        if any(marker in message for marker in _HEALTHY_INBOUND_LOG_MARKERS)
    ]
    assert hits == [], hits


def test_healthy_inbound_debug_log_stays_debug_by_default(monkeypatch, caplog) -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _cursor_inbound_debug_log,
        _log_inbound_cli_lifecycle,
    )

    monkeypatch.delenv(AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG, raising=False)
    caplog.set_level("DEBUG", logger="LiteLLM Proxy")
    _cursor_inbound_debug_log(
        "cursor_agent_cli_inbound forwarded bytes=%s flags=%s", 25936, 0
    )
    _cursor_inbound_debug_log("cursor_agent_cli_inbound agentn data bytes=%s", 5266)
    _cursor_inbound_debug_log(
        "cursor_agent_cli_inbound wrote agentn bytes=%s end_stream=%s pending=%s",
        12,
        False,
        12,
    )
    _log_inbound_cli_lifecycle(
        call_id="call-1",
        event="cleanup_finished",
        reason="normal_response",
        http_version="2",
    )
    _assert_healthy_inbound_logs_not_info(caplog)
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "DEBUG"
    ]
    assert any("forwarded bytes=25936" in message for message in debug_messages)
    assert any("agentn data bytes=5266" in message for message in debug_messages)
    assert any("wrote agentn bytes=12" in message for message in debug_messages)
    assert any(
        "lifecycle call_id=call-1 event=cleanup_finished reason=normal_response"
        in message
        for message in debug_messages
    )


@pytest.mark.parametrize("value", ["0", "true", "yes", "DEBUG", "info"])
def test_healthy_inbound_debug_log_ignores_non_exact_one(
    monkeypatch, caplog, value: str
) -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _cursor_inbound_debug_log,
    )

    monkeypatch.setenv(AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG, value)
    caplog.set_level("DEBUG", logger="LiteLLM Proxy")
    _cursor_inbound_debug_log("cursor_agent_cli_inbound forwarded bytes=%s flags=%s", 16, 0)
    _assert_healthy_inbound_logs_not_info(caplog)


def test_healthy_inbound_debug_log_promotes_on_exact_one(monkeypatch, caplog) -> None:
    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _cursor_inbound_debug_log,
        _log_inbound_cli_lifecycle,
    )

    monkeypatch.setenv(AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG, "1")
    caplog.set_level("DEBUG", logger="LiteLLM Proxy")
    _cursor_inbound_debug_log(
        "cursor_agent_cli_inbound forwarded bytes=%s flags=%s", 25936, 0
    )
    _log_inbound_cli_lifecycle(
        call_id="call-1",
        event="cleanup_finished",
        reason="normal_response",
        http_version="2",
    )
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "INFO"
    ]
    assert any("forwarded bytes=25936" in message for message in info_messages)
    assert any(
        "lifecycle call_id=call-1 event=cleanup_finished reason=normal_response"
        in message
        for message in info_messages
    )


def test_agentn_data_dispatch_is_not_info_by_default(monkeypatch, caplog) -> None:
    from h2.events import DataReceived, ResponseReceived

    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    class _AckConnection:
        def acknowledge_received_data(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.delenv(AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG, raising=False)
    caplog.set_level("DEBUG", logger="LiteLLM Proxy")
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = _AckConnection()
    headers = object.__new__(ResponseReceived)
    headers.headers = [(":status", "200"), ("content-type", "application/connect+proto")]
    data = object.__new__(DataReceived)
    data.data = _heartbeat_chunk()
    data.flow_controlled_length = len(data.data)
    data.stream_id = 1
    session._dispatch_h2_events([headers, data])
    _assert_healthy_inbound_logs_not_info(caplog)
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "DEBUG"
    ]
    assert any("agentn response status=200" in message for message in debug_messages)
    assert any("agentn data bytes=" in message for message in debug_messages)


def test_agentn_stream_closed_warning_stays_visible_with_debug_gate_off(
    monkeypatch, caplog
) -> None:
    from h2.events import ConnectionTerminated, DataReceived

    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    class _AckConnection:
        def acknowledge_received_data(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.delenv(AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG, raising=False)
    caplog.set_level("DEBUG", logger="LiteLLM Proxy")
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.connection = _AckConnection()
    session.stream_id = 1
    goaway = object.__new__(ConnectionTerminated)
    goaway.error_code = 0
    goaway.last_stream_id = 0
    goaway.additional_data = None
    stale = object.__new__(DataReceived)
    stale.data = _heartbeat_chunk()
    stale.flow_controlled_length = len(stale.data)
    stale.stream_id = 1
    session._dispatch_h2_events([goaway, stale])
    _assert_healthy_inbound_logs_not_info(caplog)
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert any(
        "cursor_agent_cli_inbound agentn stream closed event=ConnectionTerminated"
        in message
        for message in warning_messages
    ), warning_messages


def test_named_compose_files_default_inbound_debug_off() -> None:
    repo = Path(__file__).resolve().parents[4]
    alpha = (repo / "docker-compose.alpha.yml").read_text(encoding="utf-8")
    dev = (repo / "docker-compose.dev.yml").read_text(encoding="utf-8")
    expected = "AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG=${AAWM_CURSOR_AGENT_CLI_INBOUND_DEBUG:-0}"
    assert expected in alpha
    assert expected in dev


def test_rewrite_cli_connect_bytes_buffers_partial_frame() -> None:
    heartbeat = _heartbeat_chunk()
    decoder = _ProtoConnectFrameDecoder()
    assert _rewrite_cli_connect_bytes(heartbeat[:6], decoder) == b""
    assert bytes(decoder.buffer) == heartbeat[:6]
    assert _rewrite_cli_connect_bytes(heartbeat[6:], decoder) == heartbeat


def test_rewrite_cli_connect_bytes_fail_closed_on_decoder_error() -> None:
    provenance = _InboundRunProvenance(call_id="call-1")
    decoder = _ProtoConnectFrameDecoder()
    with pytest.raises(InboundCursorAgentCliProtocolError) as exc_info:
        _rewrite_cli_connect_bytes(
            b"\x00\xff\xff\xff\xffnot-a-frame",
            decoder,
            provenance=provenance,
        )
    assert exc_info.value.reason == "invalid_request"
    assert exc_info.value.direction == "client"
    assert provenance.decoder_failed is True
    assert provenance.decoder_direction == "client"


def test_request_context_exec_replies_fail_closed_on_decoder_error() -> None:
    provenance = _InboundRunProvenance(call_id="call-1")
    decoder = _ProtoConnectFrameDecoder()
    with pytest.raises(InboundCursorAgentCliProtocolError) as exc_info:
        _request_context_exec_replies(
            b"\x00\xff\xff\xff\xffnot-a-frame",
            decoder,
            provenance=provenance,
        )
    assert exc_info.value.direction == "agentn"
    assert provenance.decoder_failed is True


def test_agentn_stream_reset_records_error_code_and_remote_reset(caplog) -> None:
    from h2.events import StreamReset

    from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
        _AgentnH2Session,
    )

    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    session = _AgentnH2Session("https://agentn.global.api5.cursor.sh")
    session.stream_id = 1
    session.provenance = _InboundRunProvenance(call_id="call-reset")
    reset = object.__new__(StreamReset)
    reset.stream_id = 1
    reset.error_code = 8
    reset.remote_reset = True
    chunks, ended = session._dispatch_h2_events([reset])
    assert ended is True
    assert chunks == []
    assert session.provenance.stream_reset_code == 8
    assert session.provenance.remote_reset is True
    assert session.provenance.first_close_actor == "agentn"
    assert session.provenance.first_close_event == "StreamReset"
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert any(
        "event=StreamReset" in message
        and "error_code=8" in message
        and "remote_reset=true" in message
        for message in warning_messages
    ), warning_messages


def test_run_terminal_log_omits_secrets(caplog) -> None:
    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    provenance = _InboundRunProvenance(call_id="call-secret")
    provenance.checksum_in = True
    provenance.checksum_out = True
    provenance.checksum_in_eq_out = True
    provenance.streaming_in = False
    provenance.ua_source = "inbound"
    provenance.bearer_in_eq_out = True
    provenance.http_status = 200
    provenance.stream_reset_code = 8
    provenance.remote_reset = True
    provenance.connect_endstream = "error"
    provenance.connect_endstream_code = "unauthenticated"
    _log_inbound_run_terminal(
        provenance,
        termination_reason="upstream_reset",
        http_version="2",
    )
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    joined = "\n".join(messages)
    assert "cursor_agent_cli_inbound run_terminal" in joined
    assert "checksum_in=true" in joined
    assert "connect_endstream_code=unauthenticated" in joined
    assert "opaque" not in joined
    assert "Bearer" not in joined
    assert "eyJ" not in joined
    assert "first_close_actor=unknown" in joined
    assert "endstream_forwarded=unknown" in joined


def test_cli_connect_envelope_marks_endstream_forwarded() -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectProtoFrame

    provenance = _InboundRunProvenance(call_id="call-endstream")
    frame = CursorConnectProtoFrame(
        flags=2, payload=b'{"ok":true}', is_end_stream=True
    )
    encoded = _cli_connect_envelope(frame, provenance=provenance)
    assert provenance.connect_endstream == "ok"
    assert provenance.endstream_forwarded is True
    assert encoded.endswith(b'{"ok":true}')
    assert encoded[0] == 2


def test_run_terminal_records_first_close_booleans(caplog) -> None:
    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    provenance = _InboundRunProvenance(call_id="call-close")
    provenance.first_close_actor = "agentn"
    provenance.first_close_event = "ConnectionTerminated"
    provenance.endstream_forwarded = False
    provenance.connect_endstream = "not_seen"
    _log_inbound_run_terminal(
        provenance,
        termination_reason="upstream_reset",
        http_version="2",
    )
    joined = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    )
    assert "first_close_actor=agentn" in joined
    assert "first_close_event=ConnectionTerminated" in joined
    assert "endstream_forwarded=false" in joined
    assert "connect_endstream=not_seen" in joined


def test_run_terminal_logs_successful_endstream_forwarded(caplog) -> None:
    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    provenance = _InboundRunProvenance(call_id="call-ok")
    provenance.endstream_forwarded = True
    provenance.connect_endstream = "ok"
    _log_inbound_run_terminal(
        provenance,
        termination_reason="normal_response",
        http_version="2",
    )
    joined = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    )
    assert "cursor_agent_cli_inbound run_terminal" in joined
    assert "reason=normal_response" in joined
    assert "endstream_forwarded=true" in joined
    assert "connect_endstream=ok" in joined


def test_run_terminal_stays_silent_for_success_without_endstream(caplog) -> None:
    caplog.set_level("WARNING", logger="LiteLLM Proxy")
    provenance = _InboundRunProvenance(call_id="call-silent")
    provenance.connect_endstream = "not_seen"
    _log_inbound_run_terminal(
        provenance,
        termination_reason="normal_response",
        http_version="2",
    )
    joined = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    )
    assert "run_terminal" not in joined


def test_outbound_turn_headers_still_omit_checksum() -> None:
    from litellm.llms.cursor_agent.dashboard import build_turn_headers

    headers = build_turn_headers(
        "access-token",
        extra_headers={
            "x-cursor-checksum": "must-not-pass",
            "X-Cursor-Streaming": "true",
        },
        request_id="req-1",
        http2=True,
    )
    names = {key.lower() for key in headers}
    assert "x-cursor-checksum" not in names
    assert "x-cursor-streaming" not in names
