"""Cursor Agent Connect framing, authentication, and Run client.

The AgentService route is a Connect streaming RPC.  Keep its wire handling in
one small module so the direct provider and AAWM adapters cannot drift apart.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import os
import secrets
import ssl
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)
from urllib.parse import urlsplit

import httpx

from .constants import (
    CURSOR_AGENT_AUTH_FILE_ENV,
    CURSOR_AGENT_AUTH_REFRESH_SECONDS,
    CURSOR_AGENT_AUTH_EXCHANGE_PATH,
    CURSOR_AGENT_CONNECT_CONTENT_TYPE,
    CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES,
    CURSOR_AGENT_DASHBOARD_HOST,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_AGENT_TURN_HOST,
    CURSOR_API_KEY_ENV,
    CURSOR_AUTH_TOKEN_ENV,
)
from .dashboard import build_turn_headers, cursor_agent_user_agent

# Connect envelopes reserve bit 0 for compression and bit 1 for EndStream.
CONNECT_COMPRESSED_FLAG = 0x01
CONNECT_END_STREAM_FLAG = 0x02
CONNECT_KNOWN_FLAGS = CONNECT_END_STREAM_FLAG | CONNECT_COMPRESSED_FLAG
CURSOR_CONNECT_HEARTBEAT_SECONDS = 5.0
CURSOR_CONNECT_TERMINAL_TIMEOUT_SECONDS = 120.0


class CursorConnectError(Exception):
    """Transport or protocol failure for a Cursor Connect request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        headers: Optional[Mapping[str, Any]] = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.body = body


class CursorConnectProtocolError(CursorConnectError):
    """Malformed or unsupported Connect framing."""


def ensure_cursor_http2_available() -> None:
    """Fail before egress when the HTTP/2 dependency is unavailable."""
    try:
        import h2  # noqa: F401
    except ImportError as exc:
        raise CursorConnectError(
            "Cursor Agent requires the h2 package for HTTP/2 Connect.",
            status_code=500,
        ) from exc


def require_http2_response(response: Any) -> None:
    """Require an uncompressed response negotiated as HTTP/2."""
    version = getattr(response, "http_version", None)
    if isinstance(version, bytes):
        version = version.decode("ascii", "replace")
    if str(version or "").upper() != "HTTP/2":
        raise CursorConnectError(
            "Cursor Agent Connect requires negotiated HTTP/2; "
            f"received {version or 'unknown'}. HTTP downgrade is rejected.",
            status_code=502,
            headers=getattr(response, "headers", {}),
        )
    response_headers = getattr(response, "headers", {})
    get_header = getattr(response_headers, "get", None)
    content_encoding = str(get_header("content-encoding", "") if callable(get_header) else "").lower().strip()
    if content_encoding not in {"", "identity"}:
        raise CursorConnectProtocolError(
            "Compressed HTTP response bodies are not supported for Cursor Connect.",
            status_code=502,
            headers=response_headers,
        )


@dataclass(frozen=True)
class CursorConnectProtoFrame:
    """One decoded Connect-proto envelope."""

    flags: int
    payload: bytes
    is_end_stream: bool = False


def _bounded_gzip_decompress(payload: bytes) -> bytes:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    decoded = decompressor.decompress(
        payload,
        CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES + 1,
    )
    if len(decoded) > CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES or decompressor.unconsumed_tail:
        raise CursorConnectProtocolError("Compressed Cursor Connect frame exceeds the maximum supported size.")
    decoded += decompressor.flush()
    if len(decoded) > CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES or not decompressor.eof or decompressor.unused_data:
        raise CursorConnectProtocolError("Cursor Connect response contains an invalid gzip frame.")
    return decoded


def encode_connect_proto_frame(
    payload: bytes,
    *,
    flags: int = 0,
    compress: bool = False,
) -> bytes:
    """Encode one Connect-proto message envelope."""
    if flags & ~CONNECT_KNOWN_FLAGS:
        raise CursorConnectProtocolError(f"Unsupported Cursor Connect frame flags: 0x{flags:02x}.")
    if flags & CONNECT_END_STREAM_FLAG:
        raise CursorConnectProtocolError("Use an EndStream JSON envelope instead of a protobuf message.")
    encoded = bytes(payload)
    if compress:
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        encoded = compressor.compress(encoded) + compressor.flush()
        flags |= CONNECT_COMPRESSED_FLAG
    elif flags & CONNECT_COMPRESSED_FLAG:
        raise CursorConnectProtocolError("Compressed Cursor Connect frames require compress=True.")
    if len(encoded) > CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES:
        raise CursorConnectProtocolError("Cursor Connect request frame exceeds the maximum supported size.")
    return bytes((flags,)) + len(encoded).to_bytes(4, "big") + encoded


def _decode_connect_proto_payload(
    *,
    flags: int,
    payload: bytes,
) -> CursorConnectProtoFrame:
    if flags & ~CONNECT_KNOWN_FLAGS:
        raise CursorConnectProtocolError(f"Unsupported Cursor Connect frame flags: 0x{flags:02x}.")
    is_end_stream = bool(flags & CONNECT_END_STREAM_FLAG)
    if is_end_stream and flags & CONNECT_COMPRESSED_FLAG:
        raise CursorConnectProtocolError("Cursor Connect EndStream frames cannot be compressed.")
    decoded = _bounded_gzip_decompress(payload) if flags & CONNECT_COMPRESSED_FLAG else payload
    return CursorConnectProtoFrame(
        flags=flags,
        payload=decoded,
        is_end_stream=is_end_stream,
    )


def decode_connect_proto_frames(body: bytes) -> List[CursorConnectProtoFrame]:
    """Decode all complete Connect-proto envelopes in a buffered body."""
    decoder = _ProtoConnectFrameDecoder()
    frames = decoder.feed(body)
    decoder.finish()
    return frames


async def iter_connect_proto_frames(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[CursorConnectProtoFrame]:
    """Incrementally decode Connect-proto envelopes from a response body."""
    decoder = _ProtoConnectFrameDecoder()
    async for chunk in chunks:
        if not chunk:
            continue
        if not isinstance(chunk, (bytes, bytearray)):
            raise CursorConnectProtocolError("Cursor Connect response body yielded a non-byte chunk.")
        for frame in decoder.feed(bytes(chunk)):
            yield frame
        if decoder.saw_end_stream:
            break
    decoder.finish()


class _ProtoConnectFrameDecoder:
    """Incremental Connect-proto envelope decoder for raw HTTP/2."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.saw_end_stream = False

    def feed(self, chunk: bytes) -> List[CursorConnectProtoFrame]:
        if self.saw_end_stream and chunk:
            raise CursorConnectProtocolError("Cursor Connect response has bytes after its EndStream frame.")
        self.buffer.extend(chunk)
        frames: List[CursorConnectProtoFrame] = []
        while len(self.buffer) >= 5:
            flags = self.buffer[0]
            frame_length = int.from_bytes(self.buffer[1:5], "big")
            if frame_length > CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES:
                raise CursorConnectProtocolError("Cursor Connect response frame exceeds the maximum supported size.")
            frame_end = 5 + frame_length
            if len(self.buffer) < frame_end:
                break
            payload = bytes(self.buffer[5:frame_end])
            del self.buffer[:frame_end]
            frame = _decode_connect_proto_payload(flags=flags, payload=payload)
            frames.append(frame)
            if frame.is_end_stream:
                self.saw_end_stream = True
                if self.buffer:
                    raise CursorConnectProtocolError("Cursor Connect response has bytes after its EndStream frame.")
                break
        return frames

    def finish(self) -> None:
        if self.buffer:
            raise CursorConnectProtocolError("Cursor Connect response ended with an incomplete frame.")


def _encode_proto_varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _encode_proto_key(field_number: int, wire_type: int) -> bytes:
    if field_number <= 0:
        raise CursorConnectProtocolError("Protobuf field numbers must be positive.")
    return _encode_proto_varint((field_number << 3) | wire_type)


def _encode_proto_varint_field(
    field_number: int,
    value: Any,
    *,
    include_default: bool = False,
) -> bytes:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise CursorConnectProtocolError(f"Cursor protobuf field {field_number} requires an integer.") from exc
    if normalized == 0 and not include_default:
        return b""
    return _encode_proto_key(field_number, 0) + _encode_proto_varint(normalized)


def _encode_proto_bytes_field(
    field_number: int,
    value: bytes,
    *,
    include_empty: bool = False,
) -> bytes:
    encoded = bytes(value)
    if not encoded and not include_empty:
        return b""
    return _encode_proto_key(field_number, 2) + _encode_proto_varint(len(encoded)) + encoded


def _encode_proto_string_field(
    field_number: int,
    value: Any,
    *,
    include_empty: bool = False,
) -> bytes:
    if value is None:
        return b""
    return _encode_proto_bytes_field(
        field_number,
        str(value).encode("utf-8"),
        include_empty=include_empty,
    )


def _encode_proto_message_field(
    field_number: int,
    payload: bytes,
    *,
    include_empty: bool = True,
) -> bytes:
    return _encode_proto_bytes_field(
        field_number,
        payload,
        include_empty=include_empty,
    )


def _decode_proto_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise CursorConnectProtocolError("Cursor protobuf contains an invalid varint.")


def _decode_proto_fields(data: bytes) -> List[tuple[int, int, Any]]:
    fields: List[tuple[int, int, Any]] = []
    offset = 0
    while offset < len(data):
        key, offset = _decode_proto_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number <= 0:
            raise CursorConnectProtocolError("Cursor protobuf contains field number zero.")
        if wire_type == 0:
            value, offset = _decode_proto_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise CursorConnectProtocolError("Cursor protobuf ended inside a fixed64 field.")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _decode_proto_varint(data, offset)
            if length > CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES:
                raise CursorConnectProtocolError("Cursor protobuf field exceeds the maximum supported size.")
            end = offset + length
            if end > len(data):
                raise CursorConnectProtocolError("Cursor protobuf ended inside a length-delimited field.")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise CursorConnectProtocolError("Cursor protobuf ended inside a fixed32 field.")
            value = data[offset:end]
            offset = end
        else:
            raise CursorConnectProtocolError(f"Cursor protobuf uses unsupported wire type {wire_type}.")
        fields.append((field_number, wire_type, value))
    return fields


def _proto_field_values(
    fields: Iterable[tuple[int, int, Any]],
    field_number: int,
    *,
    wire_type: Optional[int] = None,
) -> List[Any]:
    return [
        value
        for number, actual_wire_type, value in fields
        if number == field_number and (wire_type is None or actual_wire_type == wire_type)
    ]


def _proto_last_field(
    fields: Iterable[tuple[int, int, Any]],
    field_number: int,
    *,
    wire_type: Optional[int] = None,
) -> Any:
    values = _proto_field_values(
        fields,
        field_number,
        wire_type=wire_type,
    )
    return values[-1] if values else None


def _decode_proto_string(value: Any) -> str:
    if not isinstance(value, bytes):
        return ""
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CursorConnectProtocolError("Cursor protobuf contains invalid UTF-8.") from exc


def _proto_mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


_AGENT_MODE_VALUES = {
    "AGENT_MODE_UNSPECIFIED": 0,
    "AGENT_MODE_AGENT": 1,
    "AGENT_MODE_ASK": 2,
    "AGENT_MODE_PLAN": 3,
    "AGENT_MODE_DEBUG": 4,
    "AGENT_MODE_TRIAGE": 5,
    "AGENT_MODE_PROJECT": 6,
    "AGENT_MODE_MULTITASK": 7,
    "AGENT_MODE_CUSTOM": 8,
}


def _encode_conversation_history_text(text: Any) -> bytes:
    return _encode_proto_string_field(1, text)


def _encode_conversation_history_content(
    content: Mapping[str, Any],
    *,
    assistant: bool,
) -> bytes:
    text = content.get("text")
    if isinstance(text, Mapping):
        return _encode_proto_message_field(
            1,
            _encode_conversation_history_text(text.get("text")),
        )
    tool_call = _proto_mapping_value(content, "toolCall", "tool_call")
    if assistant and isinstance(tool_call, Mapping):
        payload = b"".join(
            (
                _encode_proto_string_field(
                    1,
                    _proto_mapping_value(
                        tool_call,
                        "toolCallId",
                        "tool_call_id",
                    ),
                ),
                _encode_proto_string_field(
                    2,
                    _proto_mapping_value(tool_call, "toolName", "tool_name"),
                ),
                _encode_proto_string_field(
                    3,
                    _proto_mapping_value(tool_call, "argsJson", "args_json"),
                ),
            )
        )
        return _encode_proto_message_field(4, payload)
    raise CursorConnectProtocolError("Cursor conversation history contains unsupported content.")


def _encode_conversation_history_message(message: Mapping[str, Any]) -> bytes:
    user = message.get("user")
    if isinstance(user, Mapping):
        payload = b"".join(
            _encode_proto_message_field(
                1,
                _encode_conversation_history_content(content, assistant=False),
            )
            for content in user.get("content", [])
            if isinstance(content, Mapping)
        )
        return _encode_proto_message_field(1, payload)

    assistant = message.get("assistant")
    if isinstance(assistant, Mapping):
        payload = b"".join(
            _encode_proto_message_field(
                1,
                _encode_conversation_history_content(content, assistant=True),
            )
            for content in assistant.get("content", [])
            if isinstance(content, Mapping)
        )
        return _encode_proto_message_field(2, payload)

    tool = message.get("tool")
    if isinstance(tool, Mapping):
        payload = b"".join(
            (
                _encode_proto_string_field(
                    1,
                    _proto_mapping_value(tool, "toolCallId", "tool_call_id"),
                ),
                _encode_proto_string_field(
                    2,
                    _proto_mapping_value(tool, "toolName", "tool_name"),
                ),
                b"".join(
                    _encode_proto_message_field(
                        3,
                        _encode_conversation_history_content(
                            content,
                            assistant=False,
                        ),
                    )
                    for content in tool.get("content", [])
                    if isinstance(content, Mapping)
                ),
            )
        )
        return _encode_proto_message_field(3, payload)

    raise CursorConnectProtocolError("Cursor conversation history message has no supported role.")


def _encode_conversation_history(history: Mapping[str, Any]) -> bytes:
    payload = b"".join(
        _encode_proto_message_field(
            1,
            _encode_conversation_history_message(message),
        )
        for message in history.get("messages", [])
        if isinstance(message, Mapping)
    )
    replace_user_info = _proto_mapping_value(
        history,
        "replaceUserInfo",
        "replace_user_info",
    )
    if replace_user_info is not None:
        payload += _encode_proto_varint_field(
            2,
            bool(replace_user_info),
            include_default=True,
        )
    return payload


def _encode_user_message(user_message: Mapping[str, Any]) -> bytes:
    mode = _proto_mapping_value(user_message, "mode")
    if isinstance(mode, str):
        if mode not in _AGENT_MODE_VALUES:
            raise CursorConnectProtocolError(f"Unsupported Cursor Agent mode {mode!r}.")
        mode = _AGENT_MODE_VALUES[mode]
    return b"".join(
        (
            _encode_proto_string_field(1, user_message.get("text")),
            _encode_proto_string_field(
                2,
                _proto_mapping_value(user_message, "messageId", "message_id"),
            ),
            _encode_proto_message_field(3, b""),
            _encode_proto_varint_field(4, mode or 0),
        )
    )


def _encode_user_message_action(action: Mapping[str, Any]) -> bytes:
    user_message = _proto_mapping_value(
        action,
        "userMessage",
        "user_message",
    )
    if not isinstance(user_message, Mapping):
        raise CursorConnectProtocolError("Cursor Run requires action.userMessageAction.userMessage.")
    payload = _encode_proto_message_field(
        1,
        _encode_user_message(user_message),
    )
    history = _proto_mapping_value(
        action,
        "conversationHistory",
        "conversation_history",
    )
    if isinstance(history, Mapping):
        payload += _encode_proto_message_field(
            7,
            _encode_conversation_history(history),
        )
    return payload


def _encode_conversation_action(action: Mapping[str, Any]) -> bytes:
    user_message_action = _proto_mapping_value(
        action,
        "userMessageAction",
        "user_message_action",
    )
    if not isinstance(user_message_action, Mapping):
        raise CursorConnectProtocolError("Cursor Run supports only userMessageAction for this adapter.")
    return _encode_proto_message_field(
        1,
        _encode_user_message_action(user_message_action),
    )


def _encode_model_details(model: Mapping[str, Any]) -> bytes:
    payload = b"".join(
        (
            _encode_proto_string_field(
                1,
                _proto_mapping_value(model, "modelId", "model_id"),
            ),
            _encode_proto_string_field(
                3,
                _proto_mapping_value(model, "displayModelId", "display_model_id"),
            ),
            _encode_proto_string_field(
                4,
                _proto_mapping_value(model, "displayName", "display_name"),
            ),
            _encode_proto_string_field(
                5,
                _proto_mapping_value(
                    model,
                    "displayNameShort",
                    "display_name_short",
                ),
            ),
        )
    )
    for alias in model.get("aliases", []):
        payload += _encode_proto_string_field(6, alias)
    max_mode = _proto_mapping_value(model, "maxMode", "max_mode")
    if max_mode is not None:
        payload += _encode_proto_varint_field(
            7,
            bool(max_mode),
            include_default=True,
        )
    return payload


def _encode_requested_model(model: Mapping[str, Any]) -> bytes:
    payload = _encode_proto_string_field(
        1,
        _proto_mapping_value(model, "modelId", "model_id"),
    )
    max_mode = _proto_mapping_value(model, "maxMode", "max_mode")
    if max_mode is not None:
        payload += _encode_proto_varint_field(
            2,
            bool(max_mode),
            include_default=True,
        )
    for parameter in model.get("parameters", []):
        if not isinstance(parameter, Mapping):
            continue
        parameter_payload = b"".join(
            (
                _encode_proto_string_field(1, parameter.get("id")),
                _encode_proto_string_field(2, parameter.get("value")),
            )
        )
        payload += _encode_proto_message_field(3, parameter_payload)
    return payload


def _encode_mcp_tools(mcp_tools: Mapping[str, Any]) -> bytes:
    payload = b""
    definitions = _proto_mapping_value(mcp_tools, "mcpTools", "mcp_tools") or []
    for definition in definitions:
        if not isinstance(definition, Mapping):
            continue
        definition_payload = b"".join(
            (
                _encode_proto_string_field(1, definition.get("name")),
                _encode_proto_string_field(2, definition.get("description")),
                _encode_proto_string_field(
                    4,
                    _proto_mapping_value(
                        definition,
                        "providerIdentifier",
                        "provider_identifier",
                    ),
                ),
                _encode_proto_string_field(
                    5,
                    _proto_mapping_value(
                        definition,
                        "toolName",
                        "tool_name",
                    ),
                ),
                _encode_proto_string_field(
                    6,
                    _proto_mapping_value(
                        definition,
                        "inputSchemaJson",
                        "input_schema_json",
                    ),
                ),
            )
        )
        payload += _encode_proto_message_field(1, definition_payload)
    return payload


def _encode_conversation_state(state: Mapping[str, Any]) -> bytes:
    payload = b""
    byte_fields = {
        1: ("rootPromptMessagesJson", "root_prompt_messages_json"),
        8: ("turns",),
        3: ("todos",),
        13: ("summaryArchives", "summary_archives"),
    }
    for field_number, keys in byte_fields.items():
        values = _proto_mapping_value(state, *keys) or []
        for value in values:
            encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
            payload += _encode_proto_bytes_field(field_number, encoded)
    for value in _proto_mapping_value(state, "pendingToolCalls", "pending_tool_calls") or []:
        payload += _encode_proto_string_field(4, value)
    for value in (
        _proto_mapping_value(
            state,
            "previousWorkspaceUris",
            "previous_workspace_uris",
        )
        or []
    ):
        payload += _encode_proto_string_field(9, value)
    mode = state.get("mode")
    if isinstance(mode, str):
        mode = _AGENT_MODE_VALUES.get(mode)
    if mode is not None:
        payload += _encode_proto_varint_field(
            10,
            mode,
            include_default=True,
        )
    return payload


def encode_agent_client_message(request_payload: Mapping[str, Any]) -> bytes:
    """Encode the supported AgentClientMessage variants."""
    run_request = _proto_mapping_value(
        request_payload,
        "runRequest",
        "run_request",
    )
    if not isinstance(run_request, Mapping):
        raise CursorConnectProtocolError("Cursor Agent client message requires runRequest.")
    conversation_state = _proto_mapping_value(
        run_request,
        "conversationState",
        "conversation_state",
    )
    action = run_request.get("action")
    requested_model = _proto_mapping_value(
        run_request,
        "requestedModel",
        "requested_model",
    )
    mcp_tools = _proto_mapping_value(run_request, "mcpTools", "mcp_tools")
    if not isinstance(conversation_state, Mapping):
        conversation_state = {}
    if not isinstance(action, Mapping):
        raise CursorConnectProtocolError("Cursor Run requires action.")
    if not isinstance(requested_model, Mapping):
        raise CursorConnectProtocolError("Cursor Run requires requestedModel.")
    if not isinstance(mcp_tools, Mapping):
        mcp_tools = {}

    run_payload = b"".join(
        (
            _encode_proto_message_field(
                1,
                _encode_conversation_state(conversation_state),
            ),
            _encode_proto_message_field(
                2,
                _encode_conversation_action(action),
            ),
        )
    )
    model_details = _proto_mapping_value(
        run_request,
        "modelDetails",
        "model_details",
    )
    if isinstance(model_details, Mapping):
        run_payload += _encode_proto_message_field(
            3,
            _encode_model_details(model_details),
        )
    run_payload += _encode_proto_message_field(
        9,
        _encode_requested_model(requested_model),
    )
    run_payload += _encode_proto_message_field(
        4,
        _encode_mcp_tools(mcp_tools),
    )
    run_payload += b"".join(
        (
            _encode_proto_string_field(
                5,
                _proto_mapping_value(
                    run_request,
                    "conversationId",
                    "conversation_id",
                ),
            ),
            _encode_proto_string_field(
                16,
                _proto_mapping_value(
                    run_request,
                    "conversationGroupId",
                    "conversation_group_id",
                ),
            ),
            _encode_proto_string_field(
                25,
                _proto_mapping_value(run_request, "runId", "run_id"),
            ),
            _encode_proto_string_field(
                26,
                _proto_mapping_value(
                    run_request,
                    "agentSessionId",
                    "agent_session_id",
                ),
            ),
        )
    )
    return _encode_proto_message_field(1, run_payload)


def encode_cursor_run_request(request_payload: Mapping[str, Any]) -> bytes:
    """Encode a lowerCamelCase Run request as one Connect-proto frame."""
    return encode_connect_proto_frame(encode_agent_client_message(request_payload))


def _decode_proto_value(payload: bytes) -> Any:
    fields = _decode_proto_fields(payload)
    null_value = _proto_last_field(fields, 1, wire_type=0)
    if null_value is not None:
        return None
    number_value = _proto_last_field(fields, 2, wire_type=1)
    if isinstance(number_value, bytes):
        return struct.unpack("<d", number_value)[0]
    string_value = _proto_last_field(fields, 3, wire_type=2)
    if string_value is not None:
        return _decode_proto_string(string_value)
    bool_value = _proto_last_field(fields, 4, wire_type=0)
    if bool_value is not None:
        return bool(bool_value)
    struct_value = _proto_last_field(fields, 5, wire_type=2)
    if isinstance(struct_value, bytes):
        return _decode_proto_struct(struct_value)
    list_value = _proto_last_field(fields, 6, wire_type=2)
    if isinstance(list_value, bytes):
        return [
            _decode_proto_value(value)
            for value in _proto_field_values(
                _decode_proto_fields(list_value),
                1,
                wire_type=2,
            )
        ]
    return None


def _decode_proto_struct(payload: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for entry in _proto_field_values(
        _decode_proto_fields(payload),
        1,
        wire_type=2,
    ):
        entry_fields = _decode_proto_fields(entry)
        key = _decode_proto_string(_proto_last_field(entry_fields, 1, wire_type=2))
        value = _proto_last_field(entry_fields, 2, wire_type=2)
        if key and isinstance(value, bytes):
            result[key] = _decode_proto_value(value)
    return result


def _decode_mcp_tool_identity(value: Any) -> tuple[str, str]:
    packed_id = _decode_proto_string(value)
    call_id, separator, item_id = packed_id.partition("\n")
    if (
        separator
        and call_id
        and item_id.startswith("fc_")
        and "\n" not in item_id
        and "\r" not in item_id
    ):
        return call_id, item_id
    return packed_id, ""


def _decode_mcp_args(payload: bytes) -> Dict[str, Any]:
    fields = _decode_proto_fields(payload)
    args: Dict[str, Any] = {}
    entries = _proto_field_values(fields, 2, wire_type=2)
    for entry in entries:
        entry_fields = _decode_proto_fields(entry)
        key = _decode_proto_string(_proto_last_field(entry_fields, 1, wire_type=2))
        value = _proto_last_field(entry_fields, 2, wire_type=2)
        if key and isinstance(value, bytes):
            args[key] = _decode_proto_value(value)
    call_id, item_id = _decode_mcp_tool_identity(
        _proto_last_field(fields, 3, wire_type=2)
    )
    return {
        "name": _decode_proto_string(
            _proto_last_field(fields, 5, wire_type=2) or _proto_last_field(fields, 1, wire_type=2)
        ),
        "call_id": call_id,
        "id": item_id,
        "arguments": json.dumps(
            args,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "arguments_present": bool(entries),
    }


def _decode_tool_call(payload: bytes) -> Dict[str, Any]:
    fields = _decode_proto_fields(payload)
    call_id = _decode_proto_string(_proto_last_field(fields, 57, wire_type=2))
    mcp_tool_call = _proto_last_field(fields, 15, wire_type=2)
    if not isinstance(mcp_tool_call, bytes):
        return {
            "call_id": call_id,
            "name": "",
            "arguments": "",
            "arguments_present": False,
            "supported": False,
        }
    mcp_fields = _decode_proto_fields(mcp_tool_call)
    args_payload = _proto_last_field(mcp_fields, 1, wire_type=2)
    decoded = (
        _decode_mcp_args(args_payload)
        if isinstance(args_payload, bytes)
        else {
            "name": "",
            "call_id": "",
            "arguments": "{}",
            "arguments_present": False,
        }
    )
    decoded["call_id"] = call_id or decoded["call_id"]
    decoded["supported"] = True
    return decoded


def _decode_tool_update(
    payload: bytes,
    *,
    partial: bool = False,
    completed: bool = False,
) -> Dict[str, Any]:
    fields = _decode_proto_fields(payload)
    call_id = _decode_proto_string(_proto_last_field(fields, 1, wire_type=2))
    tool_call_payload = _proto_last_field(fields, 2, wire_type=2)
    tool_call = (
        _decode_tool_call(tool_call_payload)
        if isinstance(tool_call_payload, bytes)
        else {
            "call_id": "",
            "name": "",
            "arguments": "",
            "arguments_present": False,
            "supported": False,
        }
    )
    result: Dict[str, Any] = {
        "callId": call_id or tool_call["call_id"],
        "toolName": tool_call["name"],
    }
    if partial:
        result["argumentsDelta"] = _decode_proto_string(_proto_last_field(fields, 3, wire_type=2))
    elif tool_call["supported"] and (completed or tool_call["arguments_present"]):
        result["argsJson"] = tool_call["arguments"]
    return result


def _decode_turn_ended(payload: bytes) -> Dict[str, Any]:
    fields = _decode_proto_fields(payload)
    names = {
        1: "inputTokens",
        2: "outputTokens",
        3: "cacheReadTokens",
        4: "cacheWriteTokens",
        5: "reasoningTokens",
    }
    result: Dict[str, Any] = {}
    for field_number, name in names.items():
        value = _proto_last_field(fields, field_number, wire_type=0)
        if value is not None:
            result[name] = value
    return result


def _decode_interaction_update(payload: bytes) -> Dict[str, Any]:
    fields = _decode_proto_fields(payload)
    text_delta = _proto_last_field(fields, 1, wire_type=2)
    if isinstance(text_delta, bytes):
        text_fields = _decode_proto_fields(text_delta)
        return {"textDelta": {"text": _decode_proto_string(_proto_last_field(text_fields, 1, wire_type=2))}}
    partial_tool_call = _proto_last_field(fields, 7, wire_type=2)
    if isinstance(partial_tool_call, bytes):
        return {
            "partialToolCall": _decode_tool_update(
                partial_tool_call,
                partial=True,
            )
        }
    tool_call_started = _proto_last_field(fields, 2, wire_type=2)
    if isinstance(tool_call_started, bytes):
        return {"toolCallStarted": _decode_tool_update(tool_call_started)}
    tool_call_completed = _proto_last_field(fields, 3, wire_type=2)
    if isinstance(tool_call_completed, bytes):
        return {
            "toolCallCompleted": _decode_tool_update(
                tool_call_completed,
                completed=True,
            )
        }
    turn_ended = _proto_last_field(fields, 14, wire_type=2)
    if isinstance(turn_ended, bytes):
        return {"turnEnded": _decode_turn_ended(turn_ended)}
    return {}


def decode_agent_server_message(payload: bytes) -> Dict[str, Any]:
    """Decode the AgentServerMessage variants used by the external adapter."""
    fields = _decode_proto_fields(payload)
    interaction_update = _proto_last_field(fields, 1, wire_type=2)
    if isinstance(interaction_update, bytes):
        return {"interactionUpdate": _decode_interaction_update(interaction_update)}

    exec_server_message = _proto_last_field(fields, 2, wire_type=2)
    if isinstance(exec_server_message, bytes):
        exec_fields = _decode_proto_fields(exec_server_message)
        message_field = next(
            (number for number, wire_type, _value in exec_fields if wire_type == 2 and number not in {15}),
            None,
        )
        return {
            "execServerMessage": {
                "id": _proto_last_field(exec_fields, 1, wire_type=0) or 0,
                "execId": _decode_proto_string(_proto_last_field(exec_fields, 15, wire_type=2)),
                "messageField": message_field,
            }
        }

    kv_server_message = _proto_last_field(fields, 4, wire_type=2)
    if isinstance(kv_server_message, bytes):
        kv_fields = _decode_proto_fields(kv_server_message)
        message_field = next(
            (number for number, wire_type, _value in kv_fields if wire_type == 2 and number in {2, 3}),
            None,
        )
        return {
            "kvServerMessage": {
                "id": _proto_last_field(kv_fields, 1, wire_type=0) or 0,
                "messageField": message_field,
            }
        }
    return {}


def _encode_request_context_exec_response(
    exec_fields: List[tuple[int, int, Any]],
) -> List[bytes]:
    request_id = _proto_last_field(exec_fields, 1, wire_type=0) or 0
    exec_id = _decode_proto_string(_proto_last_field(exec_fields, 15, wire_type=2))
    request_context_success = _encode_proto_message_field(1, b"")
    request_context_result = _encode_proto_message_field(
        1,
        request_context_success,
    )
    exec_client_message = b"".join(
        (
            _encode_proto_varint_field(1, request_id),
            _encode_proto_string_field(15, exec_id),
            _encode_proto_message_field(10, request_context_result),
        )
    )
    stream_close = _encode_proto_varint_field(1, request_id)
    exec_client_control = _encode_proto_message_field(1, stream_close)
    return [
        _encode_proto_message_field(2, exec_client_message),
        _encode_proto_message_field(5, exec_client_control),
    ]


def _encode_kv_response(
    kv_fields: List[tuple[int, int, Any]],
    blobs: Dict[bytes, bytes],
) -> bytes:
    request_id = _proto_last_field(kv_fields, 1, wire_type=0) or 0
    get_blob_args = _proto_last_field(kv_fields, 2, wire_type=2)
    if isinstance(get_blob_args, bytes):
        args_fields = _decode_proto_fields(get_blob_args)
        blob_id = _proto_last_field(args_fields, 1, wire_type=2) or b""
        if blob_id in blobs:
            result = _encode_proto_bytes_field(
                1,
                blobs[blob_id],
                include_empty=True,
            )
        else:
            error = _encode_proto_string_field(
                1,
                "Cursor blob was not found in the current Run.",
            )
            result = _encode_proto_message_field(2, error)
        kv_client_message = _encode_proto_varint_field(1, request_id) + _encode_proto_message_field(2, result)
        return _encode_proto_message_field(3, kv_client_message)

    set_blob_args = _proto_last_field(kv_fields, 3, wire_type=2)
    if isinstance(set_blob_args, bytes):
        args_fields = _decode_proto_fields(set_blob_args)
        blob_id = _proto_last_field(args_fields, 1, wire_type=2) or b""
        blob_data = _proto_last_field(args_fields, 2, wire_type=2) or b""
        blobs[bytes(blob_id)] = bytes(blob_data)
        kv_client_message = _encode_proto_varint_field(1, request_id) + _encode_proto_message_field(3, b"")
        return _encode_proto_message_field(3, kv_client_message)

    raise CursorConnectProtocolError("Cursor Agent sent an unsupported KV server message.")


def _process_agent_server_message(
    payload: bytes,
    blobs: Dict[bytes, bytes],
) -> tuple[Dict[str, Any], List[bytes]]:
    fields = _decode_proto_fields(payload)
    exec_server_message = _proto_last_field(fields, 2, wire_type=2)
    if isinstance(exec_server_message, bytes):
        exec_fields = _decode_proto_fields(exec_server_message)
        mcp_args = _proto_last_field(
            exec_fields,
            11,
            wire_type=2,
        )
        if isinstance(mcp_args, bytes):
            tool_call = _decode_mcp_args(mcp_args)
            call_id = tool_call["call_id"]
            name = tool_call["name"]
            if not call_id or not name:
                raise CursorConnectProtocolError(
                    "Cursor Agent requested an MCP tool call without a "
                    "tool_call_id or tool name."
                )
            completed_tool_call = {
                "callId": call_id,
                "toolName": name,
                "argsJson": tool_call["arguments"],
            }
            if tool_call["id"]:
                completed_tool_call["itemId"] = tool_call["id"]
            return (
                {
                    "interactionUpdate": {
                        "toolCallCompleted": completed_tool_call,
                    }
                },
                [],
            )
        request_context_args = _proto_last_field(
            exec_fields,
            10,
            wire_type=2,
        )
        if not isinstance(request_context_args, bytes):
            message_field = next(
                (number for number, wire_type, _value in exec_fields if wire_type == 2 and number != 15),
                "unknown",
            )
            raise CursorConnectProtocolError(
                "Cursor Agent requested unsupported local exec operation " f"field {message_field}."
            )
        return (
            decode_agent_server_message(payload),
            _encode_request_context_exec_response(exec_fields),
        )

    kv_server_message = _proto_last_field(fields, 4, wire_type=2)
    if isinstance(kv_server_message, bytes):
        kv_fields = _decode_proto_fields(kv_server_message)
        return (
            decode_agent_server_message(payload),
            [_encode_kv_response(kv_fields, blobs)],
        )

    exec_server_control = _proto_last_field(fields, 5, wire_type=2)
    if isinstance(exec_server_control, bytes):
        raise CursorConnectProtocolError("Cursor Agent aborted a local exec request.")

    interaction_query = _proto_last_field(fields, 7, wire_type=2)
    if isinstance(interaction_query, bytes):
        raise CursorConnectProtocolError("Cursor Agent requested an unsupported interactive client response.")

    return decode_agent_server_message(payload), []


def _raise_for_connect_end_stream(payload: bytes) -> None:
    if not payload:
        return
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CursorConnectProtocolError(f"Cursor Connect EndStream frame is not valid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise CursorConnectProtocolError("Cursor Connect EndStream frame must contain a JSON object.")
    error = decoded.get("error")
    if error:
        message = (
            error.get("message", "Cursor Agent Connect EndStream error.") if isinstance(error, Mapping) else str(error)
        )
        raise CursorConnectError(message, status_code=502, body=decoded)


def decode_cursor_agent_response_payloads(body: bytes) -> List[Dict[str, Any]]:
    """Decode a completed response body that did not require bidi callbacks."""
    payloads: List[Dict[str, Any]] = []
    blobs: Dict[bytes, bytes] = {}
    for frame in decode_connect_proto_frames(body):
        if frame.is_end_stream:
            _raise_for_connect_end_stream(frame.payload)
            continue
        normalized, client_messages = _process_agent_server_message(
            frame.payload,
            blobs,
        )
        if client_messages:
            raise CursorConnectProtocolError(
                "Cursor Agent Connect received a bidi exec/KV request " "on a completed response body."
            )
        if normalized:
            payloads.append(normalized)
    return payloads


def _mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _walk_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_mappings(nested)


def _json_arguments(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, int, float, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _tool_call_from_mapping(
    mapping: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    nested: Mapping[str, Any] = mapping
    for key in (
        "toolCall",
        "mcpToolCall",
        "functionCall",
        "toolUse",
        "tool_call",
        "mcp_tool_call",
        "function_call",
    ):
        candidate = mapping.get(key)
        if isinstance(candidate, Mapping):
            nested = candidate
            break

    call_id = _mapping_value(
        nested,
        "callId",
        "toolCallId",
        "functionCallId",
        "id",
        "requestId",
        "call_id",
        "tool_call_id",
    )
    name = _mapping_value(
        nested,
        "name",
        "toolName",
        "functionName",
        "tool_name",
        "function_name",
    )
    arguments_key = None
    arguments = None
    for key in (
        "argsJson",
        "arguments",
        "args",
        "input",
        "parameters",
        "args_json",
    ):
        if key in nested:
            arguments_key = key
            arguments = nested[key]
            break
    if arguments is None:
        for key in ("argumentsDelta", "argsTextDelta", "delta", "arguments_delta"):
            if key in nested:
                arguments_key = key
                arguments = nested[key]
                break

    if call_id is None:
        return None
    normalized_arguments = _json_arguments(arguments)
    return {
        "call_id": str(call_id),
        "name": str(name or ""),
        "arguments": normalized_arguments if normalized_arguments is not None else "",
        "arguments_present": arguments_key is not None,
        "arguments_is_delta": arguments_key in {"argumentsDelta", "argsTextDelta", "delta", "arguments_delta"},
        "id": str(_mapping_value(nested, "itemId", "outputItemId", "item_id") or f"fc_{call_id}"),
    }


_TOOL_EVENT_KEYS = {
    "toolCall",
    "toolCallStarted",
    "toolCallDelta",
    "toolCallCompleted",
    "mcpToolCall",
    "mcpToolCallStarted",
    "mcpToolCallDelta",
    "mcpToolCallCompleted",
    "functionCall",
    "toolUse",
    "partialToolCall",
    "tool_call",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_completed",
    "mcp_tool_call",
    "function_call",
}
_PARTIAL_TOOL_EVENT_KEYS = {
    "toolCallStarted",
    "toolCallDelta",
    "mcpToolCallStarted",
    "mcpToolCallDelta",
    "partialToolCall",
    "tool_call_started",
    "tool_call_delta",
}


@dataclass
class CursorAgentRunResult:
    """Normalized output collected from one Cursor Run invocation."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turn_ended: bool = False
    end_stream: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    exec_server_messages: List[Dict[str, Any]] = field(default_factory=list)
    _pending_tool_calls: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )

    def add_payload(self, payload: Mapping[str, Any]) -> None:
        self.events.append(dict(payload))
        for mapping in _walk_mappings(payload):
            text_delta = mapping.get("textDelta")
            if isinstance(text_delta, Mapping):
                text = text_delta.get("text")
                if isinstance(text, str):
                    self.text += text
            elif isinstance(text_delta, str):
                self.text += text_delta
            if "turnEnded" in mapping:
                self.turn_ended = True
                turn_ended = mapping.get("turnEnded")
                if isinstance(turn_ended, Mapping):
                    self._merge_turn_ended_usage(turn_ended)
            for key, value in mapping.items():
                if key == "execServerMessage" and isinstance(value, Mapping):
                    self.exec_server_messages.append(dict(value))
                if key not in _TOOL_EVENT_KEYS:
                    continue
                if isinstance(value, Mapping):
                    tool_call = _tool_call_from_mapping(value)
                    if tool_call is not None:
                        if key in _PARTIAL_TOOL_EVENT_KEYS:
                            self._merge_pending_tool_call(tool_call)
                        else:
                            self._complete_tool_call(tool_call)

    @staticmethod
    def _merge_tool_call_values(
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> None:
        if incoming.get("name"):
            existing["name"] = incoming["name"]
        arguments = str(incoming.get("arguments") or "")
        if incoming.get("arguments_is_delta"):
            existing["arguments"] = f"{existing.get('arguments', '')}{arguments}"
        elif incoming.get("arguments_present"):
            existing["arguments"] = arguments
        if incoming.get("id"):
            existing["id"] = incoming["id"]

    def _merge_tool_call(self, incoming: Dict[str, Any]) -> None:
        call_id = incoming["call_id"]
        for existing in self.tool_calls:
            if existing.get("call_id") != call_id:
                continue
            self._merge_tool_call_values(existing, incoming)
            return
        completed = dict(incoming)
        completed["arguments"] = str(completed.get("arguments") or "{}")
        completed.pop("arguments_is_delta", None)
        completed.pop("arguments_present", None)
        self.tool_calls.append(completed)

    def _merge_pending_tool_call(self, incoming: Dict[str, Any]) -> None:
        call_id = incoming["call_id"]
        existing = self._pending_tool_calls.get(call_id)
        if existing is None:
            self._pending_tool_calls[call_id] = dict(incoming)
            return
        self._merge_tool_call_values(existing, incoming)

    def _complete_tool_call(self, incoming: Dict[str, Any]) -> None:
        call_id = incoming["call_id"]
        completed = self._pending_tool_calls.pop(call_id, None)
        if completed is None:
            completed = dict(incoming)
        else:
            self._merge_tool_call_values(completed, incoming)
        if not completed.get("name"):
            return
        self._merge_tool_call(completed)

    @staticmethod
    def _usage_int(usage: Mapping[str, Any], *keys: str) -> Optional[int]:
        for key in keys:
            value = usage.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _merge_turn_ended_usage(self, turn_ended: Mapping[str, Any]) -> None:
        nested_usage = turn_ended.get("usage")
        usage = dict(nested_usage) if isinstance(nested_usage, Mapping) else {}
        usage.update(turn_ended)

        input_tokens = self._usage_int(
            usage,
            "inputTokens",
            "inputTokenCount",
            "promptTokens",
            "input_tokens",
        )
        output_tokens = self._usage_int(
            usage,
            "outputTokens",
            "outputTokenCount",
            "completionTokens",
            "output_tokens",
        )
        total_tokens = self._usage_int(
            usage,
            "totalTokens",
            "totalTokenCount",
            "total_tokens",
        )
        cache_read_tokens = self._usage_int(
            usage,
            "cacheReadTokens",
            "cacheReadInputTokens",
            "cacheReadInputTokenCount",
            "cachedTokens",
            "cache_read_input_tokens",
        )
        reasoning_tokens = self._usage_int(
            usage,
            "reasoningTokens",
            "reasoningTokenCount",
            "reasoning_tokens",
        )
        cache_write_tokens = self._usage_int(
            usage,
            "cacheWriteTokens",
            "cacheWriteInputTokens",
            "cacheWriteInputTokenCount",
            "cache_write_input_tokens",
        )

        if input_tokens is not None:
            self.usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            self.usage["output_tokens"] = output_tokens
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if total_tokens is not None:
            self.usage["total_tokens"] = total_tokens
        if cache_read_tokens is not None:
            self.usage["input_tokens_details"] = {
                "cached_tokens": cache_read_tokens,
            }
        if reasoning_tokens is not None:
            self.usage["output_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens,
            }
        if cache_write_tokens is not None:
            self.provider_metadata["cache_write_input_tokens"] = cache_write_tokens

    def validate_terminal(self) -> None:
        if self.tool_calls or self.turn_ended:
            return
        if self.text:
            raise CursorConnectProtocolError("Cursor Agent Connect ended with incomplete text before turnEnded.")
        raise CursorConnectProtocolError("Cursor Agent Connect ended without turnEnded or a completed tool call.")


def parse_cursor_agent_payloads(
    payloads: Iterable[Mapping[str, Any]],
) -> CursorAgentRunResult:
    """Normalize decoded AgentServerMessage JSON payloads."""
    result = CursorAgentRunResult()
    for payload in payloads:
        result.add_payload(payload)
    return result


def _decode_jwt_exp(token: str) -> Optional[float]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (
        ValueError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        binascii.Error,
    ):
        return None
    if not isinstance(payload, Mapping):
        return None
    exp = payload.get("exp")
    if isinstance(exp, bool):
        return None
    try:
        return float(exp)
    except (TypeError, ValueError):
        return None


def access_token_is_fresh(
    token: str,
    *,
    now: Optional[float] = None,
    refresh_seconds: int = CURSOR_AGENT_AUTH_REFRESH_SECONDS,
) -> bool:
    """Check JWT expiry, refreshing before the configured early window."""
    if not token or not token.strip():
        return False
    exp = _decode_jwt_exp(token.strip())
    if exp is None:
        return True
    return exp > (time.time() if now is None else now) + refresh_seconds


def _looks_like_raw_api_key(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("key_", "key-", "cursor_", "cursor-", "sk-"))


def _read_secret_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        from litellm.secret_managers.main import get_secret_str

        return (get_secret_str(name) or "").strip()
    except Exception:
        return ""


def _read_auth_file() -> Dict[str, Any]:
    path = _read_secret_env(CURSOR_AGENT_AUTH_FILE_ENV)
    if not path:
        return {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


async def exchange_cursor_api_key_for_access_token(
    raw_api_key: str,
    *,
    dashboard_base: Optional[str] = None,
    http_client: Optional[Any] = None,
) -> str:
    """Exchange a raw Cursor user API key without exposing it in errors."""
    url = f"{(dashboard_base or CURSOR_AGENT_DASHBOARD_HOST).rstrip('/')}" f"{CURSOR_AGENT_AUTH_EXCHANGE_PATH}"
    headers = {
        "authorization": f"Bearer {raw_api_key}",
        "content-type": "application/json",
        "user-agent": cursor_agent_user_agent(),
    }
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(http2=True, timeout=30.0)
    try:
        response = await client.post(url, headers=headers, json={})
        status_code = int(getattr(response, "status_code", 502))
        if status_code >= 400:
            raise CursorConnectError(
                "Cursor API key exchange failed.",
                status_code=status_code,
                headers=getattr(response, "headers", {}),
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise CursorConnectError(
                f"Cursor API key exchange returned invalid JSON: {exc}",
                status_code=502,
            ) from exc
        access_token = payload.get("accessToken") if isinstance(payload, Mapping) else None
        if not isinstance(access_token, str) or not access_token.strip():
            raise CursorConnectError(
                "Cursor API key exchange did not return accessToken.",
                status_code=401,
            )
        return access_token.strip()
    finally:
        if owned_client:
            await client.aclose()


def exchange_cursor_api_key_for_access_token_sync(
    raw_api_key: str,
    *,
    dashboard_base: Optional[str] = None,
) -> str:
    """Synchronous counterpart used by direct provider configuration."""
    response = httpx.post(
        f"{(dashboard_base or CURSOR_AGENT_DASHBOARD_HOST).rstrip('/')}" f"{CURSOR_AGENT_AUTH_EXCHANGE_PATH}",
        headers={
            "authorization": f"Bearer {raw_api_key}",
            "content-type": "application/json",
            "user-agent": cursor_agent_user_agent(),
        },
        json={},
        timeout=30.0,
    )
    if response.status_code >= 400:
        raise CursorConnectError(
            "Cursor API key exchange failed.",
            status_code=response.status_code,
            headers=response.headers,
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise CursorConnectError(
            f"Cursor API key exchange returned invalid JSON: {exc}",
            status_code=502,
        ) from exc
    token = payload.get("accessToken") if isinstance(payload, Mapping) else None
    if not isinstance(token, str) or not token.strip():
        raise CursorConnectError(
            "Cursor API key exchange did not return accessToken.",
            status_code=401,
        )
    return token.strip()


class CursorAgentAuth:
    """Async singleflight resolver for Cursor access credentials."""

    def __init__(
        self,
        explicit_credential: Optional[str] = None,
        *,
        dashboard_base: Optional[str] = None,
        exchange: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        self.explicit_credential = (explicit_credential or "").strip() or None
        self.dashboard_base = dashboard_base
        self._exchange = exchange
        self._cached_token: Optional[str] = None
        self._lock = asyncio.Lock()
        self._rejected_tokens: set[str] = set()

    async def resolve(
        self,
        *,
        force_refresh: bool = False,
        rejected_token: Optional[str] = None,
    ) -> str:
        if rejected_token:
            self._rejected_tokens.add(rejected_token)
            force_refresh = True
        if (
            not force_refresh
            and self._cached_token
            and self._cached_token not in self._rejected_tokens
            and access_token_is_fresh(self._cached_token)
        ):
            return self._cached_token
        async with self._lock:
            if (
                not force_refresh
                and self._cached_token
                and self._cached_token not in self._rejected_tokens
                and access_token_is_fresh(self._cached_token)
            ):
                return self._cached_token
            token = await self._resolve_uncached()
            self._cached_token = token
            return token

    async def _resolve_uncached(self) -> str:
        explicit = self.explicit_credential
        if explicit and explicit not in self._rejected_tokens:
            if _looks_like_raw_api_key(explicit):
                exchanged = await self._exchange_key(explicit)
                if exchanged not in self._rejected_tokens:
                    return exchanged
            elif access_token_is_fresh(explicit):
                return explicit

        env_access = _read_secret_env(CURSOR_AUTH_TOKEN_ENV)
        if env_access and env_access not in self._rejected_tokens:
            if access_token_is_fresh(env_access):
                return env_access

        env_api_key = _read_secret_env(CURSOR_API_KEY_ENV)
        if env_api_key:
            exchanged = await self._exchange_key(env_api_key)
            if exchanged not in self._rejected_tokens:
                return exchanged

        auth_file = _read_auth_file()
        file_access = str(auth_file.get("accessToken") or auth_file.get("access_token") or "").strip()
        if file_access and file_access not in self._rejected_tokens:
            if access_token_is_fresh(file_access):
                return file_access

        file_api_key = str(auth_file.get("apiKey") or auth_file.get("api_key") or "").strip()
        if file_api_key:
            exchanged = await self._exchange_key(file_api_key)
            if exchanged not in self._rejected_tokens:
                return exchanged

        raise CursorConnectError(
            "Cursor Agent requires a valid access token or raw API key.",
            status_code=401,
        )

    async def _exchange_key(self, raw_api_key: str) -> str:
        if self._exchange is not None:
            token = await self._exchange(raw_api_key)
        else:
            token = await exchange_cursor_api_key_for_access_token(
                raw_api_key,
                dashboard_base=self.dashboard_base,
            )
        if not isinstance(token, str) or not token.strip():
            raise CursorConnectError(
                "Cursor API key exchange did not return accessToken.",
                status_code=401,
            )
        return token.strip()

    def invalidate(self, token: Optional[str] = None) -> None:
        value = token or self._cached_token
        if value:
            self._rejected_tokens.add(value)
        if token is None or token == self._cached_token:
            self._cached_token = None


_SHARED_AUTH_CACHE: Dict[str, CursorAgentAuth] = {}


def get_shared_cursor_agent_auth(
    explicit_credential: Optional[str] = None,
    *,
    dashboard_base: Optional[str] = None,
) -> CursorAgentAuth:
    """Return the process-local singleflight auth resolver for one route."""
    cache_key = hashlib.sha256(f"{dashboard_base or ''}\0{explicit_credential or ''}".encode("utf-8")).hexdigest()
    cached = _SHARED_AUTH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    auth = CursorAgentAuth(
        explicit_credential,
        dashboard_base=dashboard_base,
    )
    _SHARED_AUTH_CACHE[cache_key] = auth
    return auth


def clear_shared_cursor_agent_auth_cache() -> None:
    """Clear cached auth objects for isolated tests and credential rotation."""
    _SHARED_AUTH_CACHE.clear()


def resolve_cursor_access_token_sync(
    explicit_credential: Optional[str] = None,
    *,
    allow_exchange: bool = False,
    dashboard_base: Optional[str] = None,
) -> str:
    """Synchronous provider-config resolver for direct LiteLLM calls."""
    explicit = (explicit_credential or "").strip()
    if explicit and _looks_like_raw_api_key(explicit):
        if not allow_exchange:
            raise CursorConnectError(
                "Cursor API key exchange is required for an explicit raw API key.",
                status_code=401,
            )
        return exchange_cursor_api_key_for_access_token_sync(
            explicit,
            dashboard_base=dashboard_base,
        )
    if explicit and access_token_is_fresh(explicit):
        return explicit
    env_access = _read_secret_env(CURSOR_AUTH_TOKEN_ENV)
    if env_access and access_token_is_fresh(env_access):
        return env_access
    raw_key = _read_secret_env(CURSOR_API_KEY_ENV)
    if raw_key:
        if not allow_exchange:
            raise CursorConnectError(
                "Cursor API key exchange is required for CURSOR_API_KEY.",
                status_code=401,
            )
        return exchange_cursor_api_key_for_access_token_sync(
            raw_key,
            dashboard_base=dashboard_base,
        )
    auth_file = _read_auth_file()
    file_access = str(auth_file.get("accessToken") or auth_file.get("access_token") or "").strip()
    if file_access and access_token_is_fresh(file_access):
        return file_access
    file_api_key = str(auth_file.get("apiKey") or auth_file.get("api_key") or "").strip()
    if file_api_key and allow_exchange:
        return exchange_cursor_api_key_for_access_token_sync(
            file_api_key,
            dashboard_base=dashboard_base,
        )
    if file_api_key:
        raise CursorConnectError(
            "Cursor API key exchange is required for the Cursor auth file API key.",
            status_code=401,
        )
    raise CursorConnectError(
        "Cursor Agent requires a fresh access token or an exchangeable API key. " "CURSOR_CLI_KEY is ignored.",
        status_code=401,
    )


class CursorAgentConnectClient:
    """HTTP/2 Connect client for AgentService/Run."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        dashboard_base: Optional[str] = None,
        http_client: Optional[Any] = None,
        auth: Optional[CursorAgentAuth] = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.auth = auth or get_shared_cursor_agent_auth(
            api_key,
            dashboard_base=dashboard_base,
        )
        self.http_client = http_client
        self.dashboard_base = dashboard_base
        self.client_factory = client_factory

    @staticmethod
    def _ensure_http2_dependency() -> None:
        ensure_cursor_http2_available()

    @staticmethod
    def _require_http2(response: Any) -> None:
        require_http2_response(response)

    @staticmethod
    async def _read_response_body(response: Any) -> bytes:
        aread = getattr(response, "aread", None)
        if callable(aread):
            body = await aread()
            return body if isinstance(body, bytes) else bytes(body or b"")
        content = getattr(response, "content", b"")
        return content if isinstance(content, bytes) else bytes(content or b"")

    async def _response_chunks(self, response: Any) -> AsyncIterator[bytes]:
        aiter_bytes = getattr(response, "aiter_bytes", None)
        if callable(aiter_bytes):
            async for chunk in aiter_bytes():
                yield chunk
            return
        yield await self._read_response_body(response)

    @staticmethod
    def _h2_response_headers(
        headers: Iterable[tuple[Any, Any]],
    ) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for raw_name, raw_value in headers:
            name = raw_name.decode("ascii", "replace") if isinstance(raw_name, bytes) else str(raw_name)
            value = raw_value.decode("utf-8", "replace") if isinstance(raw_value, bytes) else str(raw_value)
            normalized[name.lower()] = value
        return normalized

    @staticmethod
    def _validate_h2_response_headers(headers: Mapping[str, str]) -> int:
        try:
            status_code = int(headers.get(":status", "502"))
        except (TypeError, ValueError):
            status_code = 502
        content_encoding = headers.get("content-encoding", "").lower().strip()
        if content_encoding not in {"", "identity"}:
            raise CursorConnectProtocolError(
                "Compressed HTTP response bodies are not supported for Cursor Connect.",
                status_code=502,
                headers=headers,
            )
        if status_code == 401:
            raise CursorConnectError(
                "Cursor Agent rejected the access token.",
                status_code=401,
                headers=headers,
            )
        if status_code >= 400:
            raise CursorConnectError(
                f"Cursor Agent Connect request failed with HTTP {status_code}.",
                status_code=status_code,
                headers=headers,
            )
        content_type = headers.get("content-type", "").lower()
        if CURSOR_AGENT_CONNECT_CONTENT_TYPE not in content_type:
            raise CursorConnectProtocolError(
                "Cursor Agent Connect returned an unexpected content-type.",
                status_code=502,
                headers=headers,
            )
        return status_code

    @staticmethod
    def _flush_h2_request_data(
        connection: Any,
        stream_id: int,
        pending: bytearray,
    ) -> bytes:
        while pending:
            window = int(connection.local_flow_control_window(stream_id))
            if window <= 0:
                break
            chunk_size = min(
                len(pending),
                window,
                int(connection.max_outbound_frame_size),
            )
            connection.send_data(
                stream_id,
                bytes(pending[:chunk_size]),
                end_stream=False,
            )
            del pending[:chunk_size]
        return connection.data_to_send()

    async def _run_h2_bidi_once(  # noqa: PLR0915
        self,
        *,
        url: str,
        request_body: bytes,
        headers: Mapping[str, str],
        stop_on_tool_call: bool,
        timeout: Optional[float],
    ) -> CursorAgentRunResult:
        """Run Cursor's true bidi RPC without half-closing the request."""
        from h2 import events as h2_events
        from h2.config import H2Configuration
        from h2.connection import H2Connection

        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise CursorConnectError(
                "Cursor Agent Connect requires an HTTPS URL without userinfo.",
                status_code=500,
            )

        terminal_timeout = float(timeout) if timeout is not None else CURSOR_CONNECT_TERMINAL_TIMEOUT_SECONDS
        if terminal_timeout <= 0:
            raise CursorConnectError(
                "Cursor Agent Connect timeout must be greater than zero.",
                status_code=500,
            )

        port = parsed.port or 443
        authority = parsed.hostname
        if port != 443:
            authority = f"{authority}:{port}"
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        ssl_context = ssl.create_default_context()
        ssl_context.set_alpn_protocols(["h2"])
        loop = asyncio.get_running_loop()
        deadline = loop.time() + terminal_timeout
        writer: Any = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    parsed.hostname,
                    port,
                    ssl=ssl_context,
                    server_hostname=parsed.hostname,
                ),
                timeout=terminal_timeout,
            )
            ssl_object = writer.get_extra_info("ssl_object")
            selected_alpn = ssl_object.selected_alpn_protocol() if ssl_object is not None else None
            if selected_alpn != "h2":
                raise CursorConnectError(
                    "Cursor Agent Connect requires negotiated HTTP/2; "
                    f"received {selected_alpn or 'unknown'}. "
                    "HTTP downgrade is rejected.",
                    status_code=502,
                )

            connection = H2Connection(
                config=H2Configuration(
                    client_side=True,
                    header_encoding="utf-8",
                )
            )
            connection.initiate_connection()
            stream_id = connection.get_next_available_stream_id()
            request_headers = [
                (":method", "POST"),
                (":scheme", "https"),
                (":authority", authority),
                (":path", path),
            ]
            for raw_name, raw_value in headers.items():
                name = str(raw_name).lower()
                if name.startswith(":") or name in {
                    "connection",
                    "content-length",
                    "host",
                    "transfer-encoding",
                }:
                    continue
                request_headers.append((name, str(raw_value)))
            connection.send_headers(
                stream_id,
                request_headers,
                end_stream=False,
            )

            pending = bytearray(request_body)
            writer.write(connection.data_to_send())
            await writer.drain()
            result = CursorAgentRunResult()
            decoder = _ProtoConnectFrameDecoder()
            blobs: Dict[bytes, bytes] = {}
            next_heartbeat = loop.time() + CURSOR_CONNECT_HEARTBEAT_SECONDS
            saw_response_headers = False

            while True:
                now = loop.time()
                if now >= deadline:
                    raise CursorConnectError(
                        "Cursor Agent Connect timed out after "
                        f"{terminal_timeout:g}s without turnEnded or a "
                        "completed tool call.",
                        status_code=504,
                    )
                if now >= next_heartbeat:
                    pending.extend(encode_connect_proto_frame(_encode_proto_message_field(7, b"")))
                    next_heartbeat = now + CURSOR_CONNECT_HEARTBEAT_SECONDS

                outbound = self._flush_h2_request_data(
                    connection,
                    stream_id,
                    pending,
                )
                if outbound:
                    writer.write(outbound)
                    await writer.drain()

                wait_seconds = min(
                    deadline - loop.time(),
                    max(0.0, next_heartbeat - loop.time()),
                )
                try:
                    incoming = await asyncio.wait_for(
                        reader.read(64 * 1024),
                        timeout=wait_seconds,
                    )
                except TimeoutError:
                    continue
                if not incoming:
                    decoder.finish()
                    if not saw_response_headers:
                        raise CursorConnectError(
                            "Cursor Agent HTTP/2 stream closed before response headers.",
                            status_code=502,
                        )
                    result.validate_terminal()
                    return result

                for event in connection.receive_data(incoming):
                    if isinstance(event, h2_events.ResponseReceived):
                        response_headers = self._h2_response_headers(event.headers)
                        self._validate_h2_response_headers(response_headers)
                        saw_response_headers = True
                        continue
                    if isinstance(event, h2_events.DataReceived):
                        connection.acknowledge_received_data(
                            event.flow_controlled_length,
                            event.stream_id,
                        )
                        for frame in decoder.feed(event.data):
                            if frame.is_end_stream:
                                _raise_for_connect_end_stream(frame.payload)
                                result.end_stream = True
                                result.validate_terminal()
                                return result
                            normalized, client_messages = _process_agent_server_message(
                                frame.payload,
                                blobs,
                            )
                            for client_message in client_messages:
                                pending.extend(encode_connect_proto_frame(client_message))
                            if normalized:
                                result.add_payload(normalized)
                            if stop_on_tool_call and result.tool_calls:
                                return result
                            if result.turn_ended:
                                return result
                        continue
                    if isinstance(event, h2_events.TrailersReceived):
                        trailer_headers = self._h2_response_headers(event.headers)
                        error_code = trailer_headers.get("connect-error-code")
                        if error_code and error_code != "0":
                            raise CursorConnectError(
                                trailer_headers.get(
                                    "connect-error-message",
                                    f"Cursor Agent Connect error {error_code}.",
                                ),
                                status_code=502,
                                headers=trailer_headers,
                            )
                        continue
                    if isinstance(event, h2_events.StreamEnded):
                        decoder.finish()
                        result.validate_terminal()
                        return result
                    if isinstance(event, h2_events.StreamReset):
                        raise CursorConnectError(
                            "Cursor Agent HTTP/2 stream was reset " f"(error_code={event.error_code}).",
                            status_code=502,
                        )
                    if isinstance(event, h2_events.ConnectionTerminated):
                        raise CursorConnectError(
                            "Cursor Agent HTTP/2 connection terminated " f"(error_code={event.error_code}).",
                            status_code=502,
                        )
        except CursorConnectError:
            raise
        except TimeoutError as exc:
            raise CursorConnectError(
                "Cursor Agent Connect timed out before the HTTP/2 stream opened.",
                status_code=504,
            ) from exc
        except (OSError, ssl.SSLError) as exc:
            raise CursorConnectError(
                f"Cursor Agent Connect transport failed: {exc}",
                status_code=502,
            ) from exc
        except Exception as exc:
            raise CursorConnectError(
                f"Cursor Agent Connect request failed: {exc}",
                status_code=502,
            ) from exc
        finally:
            if writer is not None:
                writer.close()
                wait_closed = getattr(writer, "wait_closed", None)
                if callable(wait_closed):
                    try:
                        await wait_closed()
                    except Exception:
                        pass

    async def _consume_response(
        self,
        response: Any,
        *,
        stop_on_tool_call: bool,
    ) -> CursorAgentRunResult:
        self._require_http2(response)
        status_code = int(getattr(response, "status_code", 502))
        if status_code == 401:
            body = await self._read_response_body(response)
            raise CursorConnectError(
                "Cursor Agent rejected the access token.",
                status_code=401,
                headers=getattr(response, "headers", {}),
                body=body,
            )
        if status_code >= 400:
            body = await self._read_response_body(response)
            raise CursorConnectError(
                f"Cursor Agent Connect request failed with HTTP {status_code}.",
                status_code=status_code,
                headers=getattr(response, "headers", {}),
                body=body,
            )
        result = CursorAgentRunResult()
        blobs: Dict[bytes, bytes] = {}
        async for frame in iter_connect_proto_frames(self._response_chunks(response)):
            if frame.is_end_stream:
                _raise_for_connect_end_stream(frame.payload)
                result.end_stream = True
                break
            normalized, client_messages = _process_agent_server_message(
                frame.payload,
                blobs,
            )
            if client_messages:
                raise CursorConnectProtocolError(
                    "Cursor Agent Connect received a bidi exec/KV request " "on a non-bidi transport."
                )
            if normalized:
                result.add_payload(normalized)
            if stop_on_tool_call and result.tool_calls:
                break
        result.validate_terminal()
        return result

    async def _run_once(
        self,
        *,
        url: str,
        request_body: bytes,
        headers: Mapping[str, str],
        stop_on_tool_call: bool,
        timeout: Optional[float],
    ) -> CursorAgentRunResult:
        if self.http_client is None and self.client_factory is None:
            self._ensure_http2_dependency()
            return await self._run_h2_bidi_once(
                url=url,
                request_body=request_body,
                headers=headers,
                stop_on_tool_call=stop_on_tool_call,
                timeout=timeout,
            )

        owned_client = self.http_client is None
        if owned_client:
            self._ensure_http2_dependency()
            if self.client_factory is not None:
                client = self.client_factory()
            else:
                client = httpx.AsyncClient(
                    http2=True,
                    timeout=timeout or 600.0,
                    follow_redirects=False,
                )
        else:
            client = self.http_client

        request_kwargs = {
            "headers": dict(headers),
            "content": request_body,
        }
        if timeout is not None:
            request_kwargs["timeout"] = timeout

        try:
            stream_method = getattr(client, "stream", None)
            if callable(stream_method):
                context = stream_method(
                    "POST",
                    url,
                    **request_kwargs,
                )
                if inspect.isawaitable(context):
                    context = await context
                if hasattr(context, "__aenter__"):
                    async with context as response:
                        return await self._consume_response(
                            response,
                            stop_on_tool_call=stop_on_tool_call,
                        )
                response = context
                try:
                    return await self._consume_response(
                        response,
                        stop_on_tool_call=stop_on_tool_call,
                    )
                finally:
                    aclose = getattr(response, "aclose", None)
                    if callable(aclose):
                        await aclose()
            response = await client.post(url, **request_kwargs)
            return await self._consume_response(
                response,
                stop_on_tool_call=stop_on_tool_call,
            )
        except CursorConnectError:
            raise
        except httpx.HTTPError as exc:
            raise CursorConnectError(
                f"Cursor Agent Connect transport failed: {exc}",
                status_code=502,
            ) from exc
        except Exception as exc:
            if isinstance(exc, CursorConnectError):
                raise
            raise CursorConnectError(
                f"Cursor Agent Connect request failed: {exc}",
                status_code=502,
            ) from exc
        finally:
            if owned_client:
                await client.aclose()

    async def run(
        self,
        request_payload: Mapping[str, Any],
        *,
        url: Optional[str] = None,
        extra_headers: Optional[Mapping[str, Any]] = None,
        stop_on_tool_call: bool = False,
        timeout: Optional[float] = None,
    ) -> CursorAgentRunResult:
        """Run one Connect request, retrying one 401 with refreshed auth."""
        self._ensure_http2_dependency()
        target_url = (url or "").strip()
        if not target_url:
            target_url = f"{CURSOR_AGENT_TURN_HOST}{CURSOR_AGENT_RUN_PATH}"
        request_body = encode_cursor_run_request(request_payload)
        run_request = request_payload.get("runRequest")
        run_id = str(run_request.get("runId") or "").strip() if isinstance(run_request, Mapping) else ""
        turn_extra_headers = {str(key).lower(): value for key, value in dict(extra_headers or {}).items()}
        if run_id:
            turn_extra_headers.setdefault("x-original-request-id", run_id)
        turn_extra_headers.setdefault(
            "x-blob-encryption-key",
            secrets.token_hex(32),
        )
        token = await self.auth.resolve()
        headers = build_turn_headers(
            token,
            extra_headers=turn_extra_headers,
            request_id=run_id or None,
            http2=True,
        )
        headers["accept"] = CURSOR_AGENT_CONNECT_CONTENT_TYPE
        headers["content-type"] = CURSOR_AGENT_CONNECT_CONTENT_TYPE
        headers["accept-encoding"] = "identity"
        try:
            return await self._run_once(
                url=target_url,
                request_body=request_body,
                headers=headers,
                stop_on_tool_call=stop_on_tool_call,
                timeout=timeout,
            )
        except CursorConnectError as exc:
            if exc.status_code != 401:
                raise
            self.auth.invalidate(token)
            retry_token = await self.auth.resolve(
                force_refresh=True,
                rejected_token=token,
            )
            retry_headers = build_turn_headers(
                retry_token,
                extra_headers=turn_extra_headers,
                request_id=run_id or None,
                http2=True,
            )
            retry_headers["accept"] = CURSOR_AGENT_CONNECT_CONTENT_TYPE
            retry_headers["content-type"] = CURSOR_AGENT_CONNECT_CONTENT_TYPE
            retry_headers["accept-encoding"] = "identity"
            return await self._run_once(
                url=target_url,
                request_body=request_body,
                headers=retry_headers,
                stop_on_tool_call=stop_on_tool_call,
                timeout=timeout,
            )


__all__ = [
    "CONNECT_COMPRESSED_FLAG",
    "CONNECT_END_STREAM_FLAG",
    "CursorAgentAuth",
    "CursorAgentConnectClient",
    "CursorAgentRunResult",
    "CursorConnectError",
    "CursorConnectProtoFrame",
    "CursorConnectProtocolError",
    "access_token_is_fresh",
    "clear_shared_cursor_agent_auth_cache",
    "decode_agent_server_message",
    "decode_connect_proto_frames",
    "decode_cursor_agent_response_payloads",
    "ensure_cursor_http2_available",
    "encode_agent_client_message",
    "encode_connect_proto_frame",
    "encode_cursor_run_request",
    "exchange_cursor_api_key_for_access_token",
    "exchange_cursor_api_key_for_access_token_sync",
    "get_shared_cursor_agent_auth",
    "iter_connect_proto_frames",
    "parse_cursor_agent_payloads",
    "require_http2_response",
    "resolve_cursor_access_token_sync",
]
