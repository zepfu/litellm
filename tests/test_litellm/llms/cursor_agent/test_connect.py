from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import DataReceived, RequestReceived, StreamEnded, WindowUpdated

from litellm.llms.cursor_agent import connect as cursor_connect
from litellm.llms.cursor_agent.connect import (
    CursorAgentAuth,
    CursorAgentConnectClient,
    CursorConnectError,
    CursorConnectProtocolError,
    access_token_is_fresh,
    decode_connect_proto_frames,
    encode_connect_proto_frame,
    parse_cursor_agent_payloads,
    require_http2_response,
)


def _jwt_with_expiry(exp: int) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': exp})}.signature"


def _end_stream_frame() -> bytes:
    return bytes((2, 0, 0, 0, 2)) + b"{}"


def _run_request(run_id: str = "run-1") -> dict[str, Any]:
    return {
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
            "requestedModel": {"modelId": "grok-4.6"},
            "mcpTools": {},
            "conversationId": "conversation-1",
            "conversationGroupId": "conversation-1",
            "runId": run_id,
        }
    }


def _message(field_number: int, payload: bytes) -> bytes:
    return cursor_connect._encode_proto_message_field(field_number, payload)


def _varint(field_number: int, value: int) -> bytes:
    return cursor_connect._encode_proto_varint_field(field_number, value)


def _string(field_number: int, value: str) -> bytes:
    return cursor_connect._encode_proto_string_field(field_number, value)


def _server_text_frame(text: str) -> bytes:
    text_delta = _string(1, text)
    interaction_update = _message(1, text_delta)
    return encode_connect_proto_frame(_message(1, interaction_update))


def _server_turn_ended_frame(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> bytes:
    turn_ended = _varint(1, input_tokens) + _varint(2, output_tokens)
    interaction_update = _message(14, turn_ended)
    return encode_connect_proto_frame(_message(1, interaction_update))


def _read_envelope(
    path: str,
    *,
    content: str = "",
    status: str = "ok",
    total_lines: int = 0,
    file_size: int = 0,
    truncated: bool = False,
    range_applied: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "version": 1,
        "status": status,
        "path": path,
    }
    if status == "ok":
        payload.update(
            {
                "content_b64": base64.b64encode(content.encode()).decode(),
                "total_lines": total_lines,
                "file_size": file_size,
                "truncated": truncated,
                "range_applied": range_applied,
            }
        )
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"LITELLM_CURSOR_READ_V1:{encoded}"


def _codex_exec_output(
    output: str,
    *,
    exit_code: int = 0,
    marker: str = "Final output:",
    token_count: int | None = None,
) -> str:
    lines = []
    if token_count is not None:
        lines.append(
            f"Warning: truncated output (original token count: {token_count})"
        )
    lines.extend(
        [
            "Chunk ID: cursor-read-test",
            "Wall time: 0.001 seconds",
            f"Process exited with code {exit_code}",
            marker,
            output.rstrip("\r\n"),
        ]
    )
    return "\n".join(lines)


def _proto_value(value: Any) -> bytes:
    if isinstance(value, str):
        return _string(3, value)
    if isinstance(value, bool):
        return _varint(4, int(value))
    raise AssertionError(f"unsupported test protobuf value: {value!r}")


def _mcp_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> bytes:
    args = _string(1, name) + _string(3, call_id) + _string(5, name)
    for key, value in arguments.items():
        entry = _string(1, key) + _message(2, _proto_value(value))
        args += _message(2, entry)
    mcp_tool_call = _message(1, args)
    return _message(15, mcp_tool_call) + _string(57, call_id)


def _server_mcp_tool_frame(
    event_field: int,
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    arguments_delta: str | None = None,
) -> bytes:
    update = _string(1, call_id) + _message(
        2,
        _mcp_tool_call(
            call_id=call_id,
            name=name,
            arguments=arguments,
        ),
    )
    if arguments_delta is not None:
        update += _string(3, arguments_delta)
    interaction_update = _message(event_field, update)
    return encode_connect_proto_frame(_message(1, interaction_update))


def _top_level_fields(payload: bytes) -> list[tuple[int, int, Any]]:
    return cursor_connect._decode_proto_fields(payload)


def _last_field(
    payload: bytes,
    field_number: int,
    *,
    wire_type: int,
) -> Any:
    return cursor_connect._proto_last_field(
        _top_level_fields(payload),
        field_number,
        wire_type=wire_type,
    )


class _FakeTLS:
    @staticmethod
    def selected_alpn_protocol() -> str:
        return "h2"


class _H2LoopbackPeer:
    def __init__(self, *, send_terminal: bool) -> None:
        self.send_terminal = send_terminal
        self.server = H2Connection(
            config=H2Configuration(
                client_side=False,
                header_encoding="utf-8",
            )
        )
        self.server.initiate_connection()
        self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self.incoming.put_nowait(self.server.data_to_send())
        self.request_body = bytearray()
        self.request_headers: dict[str, str] = {}
        self.request_stream_ended = False
        self.response_sent = False
        self.closed = False

    async def open_connection(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        return self, self

    async def read(self, _size: int) -> bytes:
        return await self.incoming.get()

    def write(self, data: bytes) -> None:
        for event in self.server.receive_data(data):
            if isinstance(event, RequestReceived):
                self.request_headers = dict(event.headers)
                self.server.send_headers(
                    event.stream_id,
                    [
                        (":status", "200"),
                        ("content-type", "application/connect+proto"),
                    ],
                    end_stream=False,
                )
            elif isinstance(event, DataReceived):
                self.server.acknowledge_received_data(
                    event.flow_controlled_length,
                    event.stream_id,
                )
                self.request_body.extend(event.data)
                if self.send_terminal and not self.response_sent:
                    response_body = b"".join(
                        [
                            _server_text_frame("ok"),
                            _server_turn_ended_frame(),
                            _end_stream_frame(),
                        ]
                    )
                    self.server.send_data(
                        event.stream_id,
                        response_body,
                        end_stream=True,
                    )
                    self.response_sent = True
            elif isinstance(event, StreamEnded):
                self.request_stream_ended = True
        outbound = self.server.data_to_send()
        if outbound:
            self.incoming.put_nowait(outbound)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str) -> Any:
        return _FakeTLS() if name == "ssl_object" else None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _H2WindowUpdatePeer:
    def __init__(self) -> None:
        self.server = H2Connection(
            config=H2Configuration(
                client_side=False,
                header_encoding="utf-8",
            )
        )
        self.server.initiate_connection()
        self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self.incoming.put_nowait(self.server.data_to_send())
        self.stream_id = 0
        self.window_updates: list[int] = []
        self.terminal_sent = False
        self.closed = False

    async def open_connection(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        return self, self

    async def read(self, _size: int) -> bytes:
        return await self.incoming.get()

    def write(self, data: bytes) -> None:
        for event in self.server.receive_data(data):
            if isinstance(event, RequestReceived):
                self.stream_id = event.stream_id
                self.server.send_headers(
                    event.stream_id,
                    [
                        (":status", "200"),
                        ("content-type", "application/connect+proto"),
                    ],
                    end_stream=False,
                )
                for _ in range(3):
                    self.server.send_data(
                        event.stream_id,
                        _server_text_frame("x" * 16_000),
                        end_stream=False,
                    )
            elif isinstance(event, DataReceived):
                self.server.acknowledge_received_data(
                    event.flow_controlled_length,
                    event.stream_id,
                )
            elif isinstance(event, WindowUpdated):
                self.window_updates.append(event.stream_id)
                if event.stream_id == self.stream_id and not self.terminal_sent:
                    self.server.send_data(
                        event.stream_id,
                        _server_turn_ended_frame() + _end_stream_frame(),
                        end_stream=True,
                    )
                    self.terminal_sent = True
        outbound = self.server.data_to_send()
        if outbound:
            self.incoming.put_nowait(outbound)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str) -> Any:
        return _FakeTLS() if name == "ssl_object" else None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _H2StartupPeer:
    def __init__(self) -> None:
        self.server = H2Connection(
            config=H2Configuration(
                client_side=False,
                header_encoding="utf-8",
            )
        )
        self.server.initiate_connection()
        self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self.incoming.put_nowait(self.server.data_to_send())
        self.decoder = cursor_connect._ProtoConnectFrameDecoder()
        self.client_payloads: list[bytes] = []
        self.stream_id = 0
        self.stage = "run"
        self.saw_exec_response = False
        self.saw_exec_close = False
        self.request_stream_ended = False
        self.closed = False

    async def open_connection(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        return self, self

    async def read(self, _size: int) -> bytes:
        return await self.incoming.get()

    def _send(self, body: bytes, *, end_stream: bool = False) -> None:
        self.server.send_data(
            self.stream_id,
            body,
            end_stream=end_stream,
        )

    def _send_request_context(self) -> None:
        exec_request = _varint(1, 7) + _string(15, "exec-7") + _message(10, b"")
        self._send(encode_connect_proto_frame(_message(2, exec_request)))

    def _send_set_blob(self) -> None:
        set_blob_args = cursor_connect._encode_proto_bytes_field(
            1, b"blob-1"
        ) + cursor_connect._encode_proto_bytes_field(2, b"value")
        kv_request = _varint(1, 8) + _message(3, set_blob_args)
        self._send(encode_connect_proto_frame(_message(4, kv_request)))

    def _send_get_blob(self) -> None:
        get_blob_args = cursor_connect._encode_proto_bytes_field(1, b"blob-1")
        kv_request = _varint(1, 9) + _message(2, get_blob_args)
        self._send(encode_connect_proto_frame(_message(4, kv_request)))

    def _handle_client_payload(self, payload: bytes) -> None:
        self.client_payloads.append(payload)
        if self.stage == "run" and _last_field(payload, 1, wire_type=2) is not None:
            self.stage = "exec"
            self._send_request_context()
            return

        if self.stage == "exec":
            exec_message = _last_field(payload, 2, wire_type=2)
            if isinstance(exec_message, bytes):
                exec_fields = _top_level_fields(exec_message)
                self.saw_exec_response = (
                    cursor_connect._proto_last_field(
                        exec_fields,
                        1,
                        wire_type=0,
                    )
                    == 7
                    and cursor_connect._proto_last_field(
                        exec_fields,
                        10,
                        wire_type=2,
                    )
                    is not None
                )
            exec_control = _last_field(payload, 5, wire_type=2)
            if isinstance(exec_control, bytes):
                stream_close = _last_field(
                    exec_control,
                    1,
                    wire_type=2,
                )
                self.saw_exec_close = isinstance(stream_close, bytes) and _last_field(stream_close, 1, wire_type=0) == 7
            if self.saw_exec_response and self.saw_exec_close:
                self.stage = "set_blob"
                self._send_set_blob()
            return

        kv_message = _last_field(payload, 3, wire_type=2)
        if not isinstance(kv_message, bytes):
            return
        request_id = _last_field(kv_message, 1, wire_type=0)
        if self.stage == "set_blob" and request_id == 8:
            assert _last_field(kv_message, 3, wire_type=2) == b""
            self.stage = "get_blob"
            self._send_get_blob()
            return
        if self.stage == "get_blob" and request_id == 9:
            get_result = _last_field(kv_message, 2, wire_type=2)
            assert isinstance(get_result, bytes)
            assert _last_field(get_result, 1, wire_type=2) == b"value"
            self.stage = "terminal"
            self._send(
                _server_text_frame("native startup ok")
                + _server_turn_ended_frame(
                    input_tokens=12,
                    output_tokens=3,
                )
                + _end_stream_frame(),
                end_stream=True,
            )

    def write(self, data: bytes) -> None:
        for event in self.server.receive_data(data):
            if isinstance(event, RequestReceived):
                self.stream_id = event.stream_id
                self.server.send_headers(
                    event.stream_id,
                    [
                        (":status", "200"),
                        ("content-type", "application/connect+proto"),
                    ],
                    end_stream=False,
                )
            elif isinstance(event, DataReceived):
                self.server.acknowledge_received_data(
                    event.flow_controlled_length,
                    event.stream_id,
                )
                for frame in self.decoder.feed(event.data):
                    assert frame.is_end_stream is False
                    self._handle_client_payload(frame.payload)
            elif isinstance(event, StreamEnded):
                self.request_stream_ended = True
        outbound = self.server.data_to_send()
        if outbound:
            self.incoming.put_nowait(outbound)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str) -> Any:
        return _FakeTLS() if name == "ssl_object" else None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _H2McpExecPeer:
    def __init__(
        self,
        *,
        local_exec: bool = False,
        local_exec_field: int = 2,
        fragment_local_exec: bool = False,
    ) -> None:
        self.local_exec = local_exec
        self.local_exec_field = local_exec_field
        self.fragment_local_exec = fragment_local_exec
        self.server = H2Connection(
            config=H2Configuration(
                client_side=False,
                header_encoding="utf-8",
            )
        )
        self.server.initiate_connection()
        self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self.incoming.put_nowait(self.server.data_to_send())
        self.decoder = cursor_connect._ProtoConnectFrameDecoder()
        self.client_payloads: list[bytes] = []
        self.stream_id = 0
        self.sent_tool_call = False
        self.request_stream_ended = False
        self.closed = False

    async def open_connection(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        return self, self

    async def read(self, _size: int) -> bytes:
        return await self.incoming.get()

    def _send_tool_call(self) -> None:
        if self.local_exec:
            shell_args = (
                _string(1, "pwd")
                + _string(2, "/workspace")
                + _varint(3, 30_000)
                + _string(4, "cursor-tool-call")
            )
            exec_request = (
                _varint(1, 11)
                + _message(self.local_exec_field, shell_args)
                + _string(15, "exec-local")
            )
            frame = encode_connect_proto_frame(_message(2, exec_request))
            if self.fragment_local_exec:
                for chunk in (frame[:3], frame[3:8], frame[8:]):
                    self.server.send_data(
                        self.stream_id,
                        chunk,
                        end_stream=False,
                    )
            else:
                self.server.send_data(
                    self.stream_id,
                    frame,
                    end_stream=False,
                )
            self.sent_tool_call = True
            return

        mcp_args = (
            _string(1, "exec_command")
            + _message(
                2,
                _string(1, "cmd") + _message(2, _proto_value("pwd")),
            )
            + _message(
                2,
                _string(1, "sandboxed") + _message(2, _proto_value(True)),
            )
            + _string(
                3,
                "call-3e993636-853f-474a-b20c-439984662b5d-0"
                "\nfc_d772cb13-2b1b-9884-b49a-e0ba78733f62_0",
            )
            + _string(5, "exec_command")
        )
        exec_request = (
            _varint(1, 11)
            + _message(11, mcp_args)
            + _string(15, "exec-11")
        )
        self.server.send_data(
            self.stream_id,
            encode_connect_proto_frame(_message(2, exec_request)),
            end_stream=False,
        )
        self.sent_tool_call = True

    def write(self, data: bytes) -> None:
        for event in self.server.receive_data(data):
            if isinstance(event, RequestReceived):
                self.stream_id = event.stream_id
                self.server.send_headers(
                    event.stream_id,
                    [
                        (":status", "200"),
                        ("content-type", "application/connect+proto"),
                    ],
                    end_stream=False,
                )
            elif isinstance(event, DataReceived):
                self.server.acknowledge_received_data(
                    event.flow_controlled_length,
                    event.stream_id,
                )
                for frame in self.decoder.feed(event.data):
                    assert frame.is_end_stream is False
                    self.client_payloads.append(frame.payload)
                    if (
                        not self.sent_tool_call
                        and _last_field(frame.payload, 1, wire_type=2)
                        is not None
                    ):
                        self._send_tool_call()
            elif isinstance(event, StreamEnded):
                self.request_stream_ended = True
        outbound = self.server.data_to_send()
        if outbound:
            self.incoming.put_nowait(outbound)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str) -> Any:
        return _FakeTLS() if name == "ssl_object" else None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _H2ReadContinuationPeer:
    def __init__(self) -> None:
        self.server = H2Connection(
            config=H2Configuration(
                client_side=False,
                header_encoding="utf-8",
            )
        )
        self.server.initiate_connection()
        self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self.incoming.put_nowait(self.server.data_to_send())
        self.decoder = cursor_connect._ProtoConnectFrameDecoder()
        self.client_payloads: list[bytes] = []
        self.stream_id = 0
        self.connection_count = 0
        self.path = "/workspace/file with spaces.txt"
        self.sent_read_call = False
        self.saw_read_result = False
        self.saw_stream_close = False
        self.read_result: bytes | None = None
        self.response_sent = False
        self.request_stream_ended = False
        self.closed = False

    async def open_connection(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        self.connection_count += 1
        return self, self

    async def read(self, _size: int) -> bytes:
        return await self.incoming.get()

    def _send_read_call(self) -> None:
        read_args = _string(1, self.path) + _string(2, "read-call")
        exec_request = (
            _varint(1, 17)
            + _message(7, read_args)
            + _string(15, "exec-read")
        )
        self.server.send_data(
            self.stream_id,
            encode_connect_proto_frame(_message(2, exec_request)),
            end_stream=False,
        )
        self.sent_read_call = True

    def _record_client_result(self, payload: bytes) -> None:
        exec_message = _last_field(payload, 2, wire_type=2)
        if isinstance(exec_message, bytes):
            if _last_field(exec_message, 1, wire_type=0) == 17:
                read_result = _last_field(exec_message, 7, wire_type=2)
                if isinstance(read_result, bytes):
                    self.saw_read_result = True
                    self.read_result = read_result
        exec_control = _last_field(payload, 5, wire_type=2)
        if isinstance(exec_control, bytes):
            stream_close = _last_field(exec_control, 1, wire_type=2)
            if (
                isinstance(stream_close, bytes)
                and _last_field(stream_close, 1, wire_type=0) == 17
            ):
                self.saw_stream_close = True
        if self.saw_read_result and self.saw_stream_close and not self.response_sent:
            self.response_sent = True
            self.server.send_data(
                self.stream_id,
                _server_text_frame("read complete")
                + _server_turn_ended_frame()
                + _end_stream_frame(),
                end_stream=True,
            )

    def write(self, data: bytes) -> None:
        if self.closed:
            raise RuntimeError("write attempted after the retained stream closed")
        for event in self.server.receive_data(data):
            if isinstance(event, RequestReceived):
                self.stream_id = event.stream_id
                self.server.send_headers(
                    event.stream_id,
                    [
                        (":status", "200"),
                        ("content-type", "application/connect+proto"),
                    ],
                    end_stream=False,
                )
            elif isinstance(event, DataReceived):
                self.server.acknowledge_received_data(
                    event.flow_controlled_length,
                    event.stream_id,
                )
                for frame in self.decoder.feed(event.data):
                    assert frame.is_end_stream is False
                    self.client_payloads.append(frame.payload)
                    if (
                        not self.sent_read_call
                        and _last_field(frame.payload, 1, wire_type=2) is not None
                    ):
                        self._send_read_call()
                    elif self.sent_read_call:
                        self._record_client_result(frame.payload)
            elif isinstance(event, StreamEnded):
                self.request_stream_ended = True
        outbound = self.server.data_to_send()
        if outbound:
            self.incoming.put_nowait(outbound)

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str) -> Any:
        return _FakeTLS() if name == "ssl_object" else None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


def test_connect_proto_frames_round_trip_gzip_and_parse_end_stream() -> None:
    message = _message(7, b"")
    body = encode_connect_proto_frame(message, compress=True) + _end_stream_frame()

    frames = decode_connect_proto_frames(body)

    assert frames[0].payload == message
    assert frames[0].is_end_stream is False
    assert frames[1].payload == b"{}"
    assert frames[1].is_end_stream is True


def test_connect_rejects_http_compression_and_http_downgrade() -> None:
    response = httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        extensions={"http_version": b"HTTP/2"},
    )
    with pytest.raises(CursorConnectProtocolError, match="Compressed"):
        require_http2_response(response)

    downgraded = httpx.Response(
        200,
        extensions={"http_version": b"HTTP/1.1"},
    )
    with pytest.raises(CursorConnectError, match="HTTP/2"):
        require_http2_response(downgraded)


def test_h2_bidi_receives_terminal_before_request_half_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2LoopbackPeer(send_terminal=True)
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))

    result = asyncio.run(
        client.run(
            _run_request(),
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
        )
    )

    assert result.text == "ok"
    assert result.turn_ended is True
    assert peer.request_stream_ended is False
    assert peer.request_headers["x-request-id"] == "run-1"
    assert peer.request_headers["x-original-request-id"] == "run-1"
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        peer.request_headers["x-blob-encryption-key"],
    )
    assert peer.request_headers["content-type"] == "application/connect+proto"
    request_frames = decode_connect_proto_frames(bytes(peer.request_body))
    assert _top_level_fields(request_frames[0].payload)[0][0] == 1


def test_h2_bidi_flushes_response_window_updates_without_pending_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2WindowUpdatePeer()
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))

    result = asyncio.run(
        client.run(
            _run_request(),
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
        )
    )

    assert result.turn_ended is True
    assert peer.stream_id in peer.window_updates
    assert peer.terminal_sent is True


def test_h2_bidi_completes_server_driven_request_context_and_kv_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2StartupPeer()
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))

    result = asyncio.run(
        client.run(
            _run_request(),
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
        )
    )

    assert result.text == "native startup ok"
    assert result.turn_ended is True
    assert result.usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    assert peer.stage == "terminal"
    assert peer.saw_exec_response is True
    assert peer.saw_exec_close is True
    assert peer.request_stream_ended is False


def test_h2_bidi_surfaces_mcp_exec_as_external_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2McpExecPeer()
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))

    result = asyncio.run(
        client.run(
            _run_request(),
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
            stop_on_tool_call=True,
            retain_on_tool_call=True,
        )
    )

    assert result.tool_calls == [
        {
            "call_id": "call-3e993636-853f-474a-b20c-439984662b5d-0",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd","sandboxed":true}',
            "id": "fc_d772cb13-2b1b-9884-b49a-e0ba78733f62_0",
        }
    ]
    assert peer.sent_tool_call is True
    assert peer.request_stream_ended is False
    assert peer.closed is True
    assert result.retained_session is None
    assert all(
        _last_field(payload, 2, wire_type=2) is None
        and _last_field(payload, 5, wire_type=2) is None
        for payload in peer.client_payloads
    )


def test_h2_bidi_surfaces_local_shell_as_advertised_external_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2McpExecPeer(local_exec=True)
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))
    request = _run_request()
    request["runRequest"]["mcpTools"] = {
        "mcpTools": [{"name": "exec_command"}],
    }

    result = asyncio.run(
        client.run(
            request,
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
            stop_on_tool_call=True,
            retain_on_tool_call=True,
        )
    )

    assert result.tool_calls == [
        {
            "call_id": "exec-local",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
            "id": "fc_exec-local",
        }
    ]
    assert peer.sent_tool_call is True
    assert peer.request_stream_ended is False
    assert peer.closed is True
    assert result.retained_session is None
    assert all(
        _last_field(payload, 2, wire_type=2) is None
        and _last_field(payload, 5, wire_type=2) is None
        for payload in peer.client_payloads
    )


def test_h2_bidi_surfaces_fragmented_shell_stream_as_external_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2McpExecPeer(
        local_exec=True,
        local_exec_field=14,
        fragment_local_exec=True,
    )
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))
    request = _run_request()
    request["runRequest"]["mcpTools"] = {
        "mcpTools": [{"name": "exec_command"}],
    }

    result = asyncio.run(
        client.run(
            request,
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
            stop_on_tool_call=True,
            retain_on_tool_call=True,
        )
    )

    assert result.tool_calls == [
        {
            "call_id": "exec-local",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
            "id": "fc_exec-local",
        }
    ]
    assert peer.sent_tool_call is True
    assert peer.request_stream_ended is False
    assert peer.closed is True
    assert result.retained_session is None
    assert all(
        _last_field(payload, 2, wire_type=2) is None
        and _last_field(payload, 5, wire_type=2) is None
        for payload in peer.client_payloads
    )


def test_h2_bidi_retains_read_run_until_external_output_then_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2ReadContinuationPeer()
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    request = _run_request()
    request["runRequest"]["mcpTools"] = {
        "mcpTools": [{"name": "exec_command"}],
    }

    async def exercise() -> tuple[Any, Any]:
        client = CursorAgentConnectClient(
            auth=CursorAgentAuth("access-token")
        )
        first = await client.run(
            request,
            url="https://cursor.test/agent.v1.AgentService/Run",
            timeout=0.5,
            stop_on_tool_call=True,
            retain_on_tool_call=True,
        )
        session = first.retained_session
        assert session is not None
        assert peer.closed is False
        second = await session.continue_with_tool_outputs(
            [
                (
                    "read-call",
                    _codex_exec_output(
                        _read_envelope(
                            peer.path,
                            content="line one\nline two",
                            total_lines=2,
                            file_size=17,
                        )
                    ),
                )
            ],
            timeout=0.5,
        )
        await session.aclose()
        return first, second

    first, second = asyncio.run(exercise())

    assert peer.connection_count == 1
    assert first.tool_calls[0]["call_id"] == "read-call"
    command = json.loads(first.tool_calls[0]["arguments"])["cmd"]
    assert command.endswith(f"'{peer.path}'")
    assert peer.saw_read_result is True
    assert peer.saw_stream_close is True
    assert peer.request_stream_ended is False
    assert peer.closed is True
    assert second.text == "read complete"
    assert second.turn_ended is True


@pytest.mark.parametrize("message_field", [2, 14])
def test_proto_bidi_bridges_advertised_local_shell_arguments(
    message_field: int,
) -> None:
    shell_args = _string(1, "pwd") + _string(2, "/workspace")
    exec_request = (
        _varint(1, 17)
        + _message(message_field, shell_args)
        + _string(15, "exec-local")
    )
    normalized, client_messages = cursor_connect._process_agent_server_message(
        _message(2, exec_request),
        {},
        local_exec_tool_name="exec_command",
    )

    assert normalized == {
        "interactionUpdate": {
            "toolCallCompleted": {
                "callId": "exec-local",
                "toolName": "exec_command",
                "argsJson": '{"cmd":"pwd","workdir":"/workspace"}',
                "itemId": "fc_exec-local",
            }
        }
    }
    assert client_messages == []


def test_proto_bidi_bridges_read_args_through_external_exec_command() -> None:
    path = "/workspace/file with spaces;$(touch should-not-run).txt"
    read_args = _string(1, path) + _string(2, "read-call")
    exec_request = (
        _varint(1, 23)
        + _message(7, read_args)
        + _string(15, "exec-read")
    )

    normalized, client_messages = cursor_connect._process_agent_server_message(
        _message(2, exec_request),
        {},
        local_exec_tool_name="exec_command",
    )

    tool_call = normalized["interactionUpdate"]["toolCallCompleted"]
    command = json.loads(tool_call["argsJson"])["cmd"]
    assert tool_call["callId"] == "read-call"
    assert tool_call["toolName"] == "exec_command"
    assert command.startswith("python3 -c ")
    assert command.endswith(f"'{path}'")
    assert client_messages == []


@pytest.mark.parametrize(
    ("marker", "token_count"),
    [
        ("Final output:", None),
        ("Output:", 3),
    ],
)
def test_read_command_executes_utf8_file_and_emits_valid_envelope(
    tmp_path: Path,
    marker: str,
    token_count: int | None,
) -> None:
    path = tmp_path / "file with spaces.txt"
    content = "alpha\nbeta\n"
    path.write_text(content, encoding="utf-8")
    command = cursor_connect._read_command(str(path))

    completed = subprocess.run(
        shlex.split(command),
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    decoded = cursor_connect._decode_cursor_read_envelope(
        _codex_exec_output(
            completed.stdout,
            marker=marker,
            token_count=token_count,
        ),
        expected_path=str(path),
    )
    assert decoded == {
        "status": "ok",
        "path": str(path),
        "content": content,
        "total_lines": 2,
        "file_size": len(content.encode("utf-8")),
        "truncated": False,
        "range_applied": False,
    }


@pytest.mark.parametrize("rejection", ["nonzero", "multiple", "extra"])
def test_decode_cursor_read_envelope_rejects_ambiguous_exec_output(
    rejection: str,
) -> None:
    path = "/workspace/file.txt"
    envelope = _read_envelope(path, content="alpha", total_lines=1, file_size=5)
    if rejection == "nonzero":
        output = _codex_exec_output(envelope, exit_code=7)
        expected_reason = "nonzero status"
    elif rejection == "multiple":
        output = _codex_exec_output(f"{envelope}\n{envelope}")
        expected_reason = "exactly one"
    else:
        output = _codex_exec_output(f"{envelope}\nunexpected trailing content")
        expected_reason = "ambiguous extra content"

    decoded = cursor_connect._decode_cursor_read_envelope(
        output,
        expected_path=path,
    )

    assert decoded["status"] == "invalid_file"
    assert expected_reason in decoded["reason"]


@pytest.mark.parametrize(
    "unsupported_args",
    [
        _varint(4, 1),
        _varint(5, 1),
        _string(6, "latin1"),
    ],
)
def test_proto_bidi_rejects_nondefault_read_range_or_encoding(
    unsupported_args: bytes,
) -> None:
    read_args = _string(1, "/workspace/file.txt") + unsupported_args
    exec_request = _varint(1, 23) + _message(7, read_args)

    with pytest.raises(
        CursorConnectProtocolError,
        match="nondefault",
    ):
        cursor_connect._process_agent_server_message(
            _message(2, exec_request),
            {},
            local_exec_tool_name="exec_command",
        )


def test_proto_bidi_encodes_read_result_success_and_invalid_envelope() -> None:
    path = "/workspace/file.txt"
    exec_fields = _top_level_fields(
        _varint(1, 23) + _string(15, "exec-read")
    )
    request = {
        "exec_fields": exec_fields,
        "message_field": 7,
        "path": path,
    }

    messages = cursor_connect._encode_read_terminal_result(
        request,
        _codex_exec_output(
            _read_envelope(
                path,
                content="alpha\nbeta",
                total_lines=2,
                file_size=10,
                truncated=True,
            )
        ),
    )
    exec_message = _last_field(messages[0], 2, wire_type=2)
    assert isinstance(exec_message, bytes)
    assert _last_field(exec_message, 1, wire_type=0) == 23
    assert _last_field(exec_message, 15, wire_type=2) == b"exec-read"
    read_result = _last_field(exec_message, 7, wire_type=2)
    assert isinstance(read_result, bytes)
    success = _last_field(read_result, 1, wire_type=2)
    assert isinstance(success, bytes)
    assert _last_field(success, 1, wire_type=2) == path.encode()
    assert _last_field(success, 2, wire_type=2) == b"alpha\nbeta"
    assert _last_field(success, 3, wire_type=0) == 2
    assert _last_field(success, 4, wire_type=0) == 10
    assert _last_field(success, 6, wire_type=0) == 1
    assert _last_field(messages[1], 5, wire_type=2) is not None

    invalid_messages = cursor_connect._encode_read_terminal_result(
        request,
        _codex_exec_output("not-a-read-envelope"),
    )
    invalid_exec_message = _last_field(
        invalid_messages[0],
        2,
        wire_type=2,
    )
    assert isinstance(invalid_exec_message, bytes)
    invalid_result = _last_field(
        invalid_exec_message,
        7,
        wire_type=2,
    )
    assert isinstance(invalid_result, bytes)
    invalid_file = _last_field(invalid_result, 6, wire_type=2)
    assert isinstance(invalid_file, bytes)
    assert _last_field(invalid_file, 1, wire_type=2) == path.encode()
    assert b"invalid envelope" in (
        _last_field(invalid_file, 2, wire_type=2) or b""
    )


@pytest.mark.parametrize("message_field", [2, 14])
def test_proto_bidi_rejects_local_exec_without_advertised_command_tool(
    message_field: int,
) -> None:
    shell_exec = _varint(1, 4) + _message(message_field, b"")
    server_message = _message(2, shell_exec)

    with pytest.raises(
        CursorConnectProtocolError,
        match=f"unsupported local exec operation field {message_field}",
    ):
        cursor_connect._process_agent_server_message(server_message, {})


@pytest.mark.parametrize("message_field", [2, 14])
def test_proto_bidi_rejects_advertised_local_exec_without_command(
    message_field: int,
) -> None:
    shell_exec = _varint(1, 4) + _message(message_field, _string(2, "/workspace"))

    with pytest.raises(
        CursorConnectProtocolError,
        match="does not contain a command",
    ):
        cursor_connect._process_agent_server_message(
            _message(2, shell_exec),
            {},
            local_exec_tool_name="exec_command",
        )


def test_proto_bidi_rejects_known_unsupported_local_exec_without_execution(
) -> None:
    message_field = 5
    operation_name = "grep_args"
    exec_request = _varint(1, 17) + _string(15, "exec-local") + _message(message_field, b"")
    server_message = _message(2, exec_request)

    normalized, client_messages = cursor_connect._process_agent_server_message(
        server_message,
        {},
    )

    assert normalized == {
        "execServerMessage": {
            "id": 17,
            "execId": "exec-local",
            "messageField": message_field,
        }
    }
    assert len(client_messages) == 2

    throw_control = _last_field(client_messages[0], 5, wire_type=2)
    assert isinstance(throw_control, bytes)
    throw = _last_field(throw_control, 2, wire_type=2)
    assert isinstance(throw, bytes)
    assert _last_field(throw, 1, wire_type=0) == 17
    error = cursor_connect._decode_proto_string(
        _last_field(throw, 2, wire_type=2)
    )
    assert operation_name in error
    assert f"field {message_field}" in error
    assert "no local execution was performed" in error

    stream_close_control = _last_field(client_messages[1], 5, wire_type=2)
    assert isinstance(stream_close_control, bytes)
    stream_close = _last_field(stream_close_control, 1, wire_type=2)
    assert isinstance(stream_close, bytes)
    assert _last_field(stream_close, 1, wire_type=0) == 17


def test_proto_startup_responses_match_native_empty_id_frames() -> None:
    exec_fields = cursor_connect._decode_proto_fields(_message(10, b""))
    exec_response, stream_close = cursor_connect._encode_request_context_exec_response(exec_fields)
    assert exec_response.hex() == "120652040a020a00"
    assert stream_close.hex() == "2a020a00"

    kv_fields = cursor_connect._decode_proto_fields(_message(3, b""))
    assert cursor_connect._encode_kv_response(kv_fields, {}).hex() == "1a021a00"


def test_h2_bidi_heartbeats_and_times_out_without_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _H2LoopbackPeer(send_terminal=False)
    monkeypatch.setattr(asyncio, "open_connection", peer.open_connection)
    monkeypatch.setattr(
        cursor_connect,
        "CURSOR_CONNECT_HEARTBEAT_SECONDS",
        0.01,
    )
    client = CursorAgentConnectClient(auth=CursorAgentAuth("access-token"))

    with pytest.raises(
        CursorConnectError,
        match="timed out.*without turnEnded or a completed tool call",
    ) as exc_info:
        asyncio.run(
            client.run(
                _run_request(),
                url="https://cursor.test/agent.v1.AgentService/Run",
                timeout=0.04,
            )
        )

    assert exc_info.value.status_code == 504
    assert peer.request_stream_ended is False
    request_frames = decode_connect_proto_frames(bytes(peer.request_body))
    assert _top_level_fields(request_frames[0].payload)[0][0] == 1
    assert any(frame.payload == _message(7, b"") for frame in request_frames[1:])


def test_cursor_events_parse_camel_case_tool_calls_and_int64_usage() -> None:
    result = parse_cursor_agent_payloads(
        [
            {"interactionUpdate": {"textDelta": {"text": "hello "}}},
            {
                "interactionUpdate": {
                    "toolCall": {
                        "callId": "call-1",
                        "toolName": "exec_command",
                        "argsJson": '{"cmd":"pwd"}',
                    }
                }
            },
            {"execServerMessage": {"status": "started"}},
            {
                "usage": {
                    "inputTokens": 999,
                    "outputTokens": 999,
                }
            },
            {
                "interactionUpdate": {
                    "turnEnded": {
                        "inputTokens": "12",
                        "outputTokens": 3,
                        "cacheReadTokens": "4",
                        "reasoningTokens": 2,
                        "cacheWriteTokens": "5",
                    }
                }
            },
        ]
    )

    assert result.text == "hello "
    assert result.turn_ended is True
    assert result.tool_calls == [
        {
            "call_id": "call-1",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
            "id": "fc_call-1",
        }
    ]
    assert result.usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens_details": {"reasoning_tokens": 2},
    }
    assert result.provider_metadata == {"cache_write_input_tokens": 5}
    assert result.exec_server_messages == [{"status": "started"}]


def test_stop_on_tool_call_waits_for_completed_arguments() -> None:
    class Response:
        status_code = 200
        headers = {"content-type": "application/connect+proto"}
        http_version = "HTTP/2"
        content = b"".join(
            [
                _server_mcp_tool_frame(
                    2,
                    call_id="call-1",
                    name="exec_command",
                    arguments={},
                ),
                _server_mcp_tool_frame(
                    7,
                    call_id="call-1",
                    name="exec_command",
                    arguments={},
                    arguments_delta='{"cmd":"',
                ),
                _server_mcp_tool_frame(
                    7,
                    call_id="call-1",
                    name="exec_command",
                    arguments={},
                    arguments_delta='pwd"}',
                ),
                _server_mcp_tool_frame(
                    3,
                    call_id="call-1",
                    name="exec_command",
                    arguments={"cmd": "pwd"},
                ),
                _end_stream_frame(),
            ]
        )

    class Client:
        async def post(self, _url: str, **_kwargs: Any) -> Response:
            return Response()

        async def aclose(self) -> None:
            return None

    client = CursorAgentConnectClient(
        auth=CursorAgentAuth("access-token"),
        client_factory=Client,
    )
    result = asyncio.run(
        client.run(
            _run_request(),
            stop_on_tool_call=True,
        )
    )

    assert result.tool_calls == [
        {
            "call_id": "call-1",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
            "id": "fc_call-1",
        }
    ]


def test_auth_refresh_window_and_file_precedence(tmp_path: Path, monkeypatch) -> None:
    now = 1_000
    assert access_token_is_fresh(_jwt_with_expiry(1_301), now=now) is True
    assert access_token_is_fresh(_jwt_with_expiry(1_300), now=now) is False
    assert access_token_is_fresh(_jwt_with_expiry(900), now=now) is False

    auth_file = tmp_path / "cursor-auth.json"
    auth_file.write_text(
        json.dumps({"accessToken": "file-access", "apiKey": "key-file"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CURSOR_AGENT_AUTH_FILE", str(auth_file))
    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "env-access")
    monkeypatch.setenv("CURSOR_API_KEY", "key-env")

    exchanged: list[str] = []

    async def exchange(raw_api_key: str) -> str:
        exchanged.append(raw_api_key)
        return f"exchanged-{raw_api_key}"

    auth = CursorAgentAuth(exchange=exchange)
    assert asyncio.run(auth.resolve()) == "env-access"
    assert exchanged == []

    monkeypatch.delenv("CURSOR_AUTH_TOKEN")
    assert asyncio.run(auth.resolve(force_refresh=True)) == "exchanged-key-env"
    assert exchanged == ["key-env"]

    monkeypatch.delenv("CURSOR_API_KEY")
    auth = CursorAgentAuth(exchange=exchange)
    assert asyncio.run(auth.resolve()) == "file-access"

    auth_file.write_text(json.dumps({"apiKey": "key-file"}), encoding="utf-8")
    auth = CursorAgentAuth(exchange=exchange)
    with pytest.raises(CursorConnectError, match="request-time API key exchange is not supported"):
        asyncio.run(auth.resolve())
    assert exchanged == ["key-env"]


def test_managed_auth_file_replacement_invalidates_cache_before_egress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    auth_file = tmp_path / "cursor-auth.json"
    auth_file.write_text(json.dumps({"accessToken": "expired-access"}), encoding="utf-8")
    monkeypatch.delenv("CURSOR_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_CURSOR_AGENT_AUTH_FILE", str(auth_file))

    auth = CursorAgentAuth(exchange=lambda key: (_ for _ in ()).throw(AssertionError("exchange called")))
    assert asyncio.run(auth.resolve()) == "expired-access"
    replacement_file = tmp_path / "cursor-auth-replacement.json"
    replacement_file.write_text(
        json.dumps({"accessToken": "replacement-access"}),
        encoding="utf-8",
    )
    os.replace(replacement_file, auth_file)

    class Response:
        status_code = 200
        headers = {"content-type": "application/connect+proto"}
        http_version = "HTTP/2"
        content = _server_text_frame("ok") + _server_turn_ended_frame() + _end_stream_frame()

    class Client:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def post(self, _url: str, **kwargs: Any) -> Response:
            self.requests.append(kwargs)
            return Response()

        async def aclose(self) -> None:
            return None

    transport = Client()
    client = CursorAgentConnectClient(auth=auth, client_factory=lambda: transport)
    assert asyncio.run(client.run(_run_request())).text == "ok"
    assert transport.requests[0]["headers"]["authorization"] == "Bearer replacement-access"


def test_managed_auth_file_rejection_fails_closed_and_sanitized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    auth_file = tmp_path / "cursor-auth.json"
    auth_file.write_text(
        json.dumps({"accessToken": "rejected-access", "apiKey": "key-file"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("CURSOR_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_CURSOR_AGENT_AUTH_FILE", str(auth_file))

    exchange_calls: list[str] = []

    async def exchange(raw_api_key: str) -> str:
        exchange_calls.append(raw_api_key)
        return "sidecar-access"

    auth = CursorAgentAuth(exchange=exchange)
    assert asyncio.run(auth.resolve()) == "rejected-access"
    auth.invalidate("rejected-access")
    auth_file.write_text(json.dumps({"apiKey": "key-file"}), encoding="utf-8")

    with pytest.raises(CursorConnectError, match="request-time API key exchange is not supported") as exc_info:
        asyncio.run(auth.resolve(force_refresh=True, rejected_token="rejected-access"))
    assert exchange_calls == []
    assert str(exc_info.value) == exc_info.value.message
    assert exc_info.value.body is None
    assert "key-file" not in str(exc_info.value)
    assert "rejected-access" not in str(exc_info.value)


def test_auth_exchange_is_singleflight_and_401_retries_once() -> None:
    exchange_calls: list[str] = []

    async def exchange(raw_api_key: str) -> str:
        exchange_calls.append(raw_api_key)
        await asyncio.sleep(0)
        return f"access-{len(exchange_calls)}"

    auth = CursorAgentAuth("key-shared", exchange=exchange)

    async def resolve_twice() -> list[str]:
        return list(await asyncio.gather(auth.resolve(), auth.resolve()))

    resolved = asyncio.run(resolve_twice())
    assert resolved == ["access-1", "access-1"]
    assert exchange_calls == ["key-shared"]

    class Response:
        def __init__(self, status_code: int, body: bytes) -> None:
            self.status_code = status_code
            self.content = body
            self.headers = {}
            self.http_version = "HTTP/2"

    class Client:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def post(self, url: str, **kwargs: Any) -> Response:
            self.requests.append({"url": url, **kwargs})
            if len(self.requests) == 1:
                return Response(401, b"")
            body = _server_text_frame("ok") + _server_turn_ended_frame() + _end_stream_frame()
            return Response(200, body)

        async def aclose(self) -> None:
            return None

    client = Client()
    auth = CursorAgentAuth("key-retry", exchange=exchange)
    connect_client = CursorAgentConnectClient(
        auth=auth,
        client_factory=lambda: client,
    )
    result = asyncio.run(connect_client.run(_run_request()))

    assert result.text == "ok"
    assert len(client.requests) == 2
    assert [request["headers"]["authorization"] for request in client.requests] == [
        "Bearer access-2",
        "Bearer access-3",
    ]
    first_headers = client.requests[0]["headers"]
    retry_headers = client.requests[1]["headers"]
    assert first_headers["x-request-id"] == "run-1"
    assert first_headers["x-original-request-id"] == "run-1"
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        first_headers["x-blob-encryption-key"],
    )
    assert retry_headers["x-blob-encryption-key"] == first_headers["x-blob-encryption-key"]
    assert exchange_calls[-2:] == ["key-retry", "key-retry"]


def test_connect_headers_preserve_explicit_request_identity() -> None:
    class Response:
        status_code = 200
        headers = {"content-type": "application/connect+proto"}
        http_version = "HTTP/2"
        content = _server_turn_ended_frame() + _end_stream_frame()

    class Client:
        def __init__(self) -> None:
            self.request: dict[str, Any] = {}

        async def post(self, _url: str, **kwargs: Any) -> Response:
            self.request = kwargs
            return Response()

        async def aclose(self) -> None:
            return None

    transport = Client()
    client = CursorAgentConnectClient(
        auth=CursorAgentAuth("access-token"),
        client_factory=lambda: transport,
    )

    asyncio.run(
        client.run(
            _run_request("run-native"),
            extra_headers={
                "x-request-id": "request-explicit",
                "x-original-request-id": "original-explicit",
                "x-blob-encryption-key": "a" * 64,
            },
        )
    )

    headers = transport.request["headers"]
    assert headers["x-request-id"] == "request-explicit"
    assert headers["x-original-request-id"] == "original-explicit"
    assert headers["x-blob-encryption-key"] == "a" * 64


def test_rejected_explicit_raw_api_key_is_not_returned_as_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CURSOR_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_CURSOR_AGENT_AUTH_FILE", raising=False)

    async def exchange(_raw_api_key: str) -> str:
        return "rejected-access"

    auth = CursorAgentAuth("key-explicit", exchange=exchange)
    auth.invalidate("rejected-access")

    with pytest.raises(CursorConnectError, match="valid access token"):
        asyncio.run(auth.resolve(force_refresh=True))


def test_connect_client_rejects_incomplete_text_without_turn_ended() -> None:
    class Response:
        status_code = 200
        headers = {"content-type": "application/connect+proto"}
        http_version = "HTTP/2"
        content = _server_text_frame("partial") + _end_stream_frame()

    class Client:
        async def post(self, _url: str, **_kwargs: Any) -> Response:
            return Response()

        async def aclose(self) -> None:
            return None

    auth = CursorAgentAuth("access-token")
    client = CursorAgentConnectClient(
        auth=auth,
        client_factory=Client,
    )

    with pytest.raises(CursorConnectProtocolError, match="turnEnded"):
        asyncio.run(client.run(_run_request()))
