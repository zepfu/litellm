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
import re
import secrets
import shlex
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
    cast,
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
    exclude_workspace_context = _proto_mapping_value(
        run_request,
        "excludeWorkspaceContext",
        "exclude_workspace_context",
    )
    if exclude_workspace_context is not None:
        run_payload += _encode_proto_varint_field(
            12,
            bool(exclude_workspace_context),
            include_default=True,
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


_UNSUPPORTED_LOCAL_EXEC_FIELDS = {
    5: "grep_args",
}
# Cursor's generated ExecServerMessage schema types both fields as ShellArgs.
# Field 14 streams ShellStream responses only after client-side execution.
_LOCAL_EXEC_BRIDGE_FIELDS = {
    2: "shell_args",
    14: "shell_stream_args",
}
_LOCAL_EXEC_TOOL_NAMES = (
    "exec_command",
    "shell",
    "bash",
    "run_shell_command",
)
_CURSOR_READ_ENVELOPE_PREFIX = "LITELLM_CURSOR_READ_V1:"
_CURSOR_READ_MAX_OUTPUT_CHARS = 8 * 1024 * 1024
_CURSOR_READ_ENVELOPE_MAX_BYTES = 16 * 1024 * 1024
_CURSOR_EXEC_CHUNK_ID_RE = re.compile(r"Chunk ID: \S+")
_CURSOR_EXEC_WALL_TIME_RE = re.compile(r"Wall time: \d+(?:\.\d+)? seconds")
_CURSOR_EXEC_PROCESS_EXIT_RE = re.compile(r"Process exited with code (-?\d+)")
_CURSOR_EXEC_TRUNCATION_WARNING_RE = re.compile(
    r"Warning: truncated output \(original token count: \d+\)"
)
_CURSOR_EXEC_TOKEN_COUNT_RE = re.compile(r"(?:Token count|Tokens): \d+")
_CURSOR_READ_COMMAND_SCRIPT = r"""import base64
import codecs
import json
import os
import stat
import sys

MAX_OUTPUT = 8388608
PREFIX = "LITELLM_CURSOR_READ_V1:"
BINARY_EXTENSIONS = {
    ".7z", ".avi", ".bmp", ".class", ".dll", ".dmg", ".doc", ".docx",
    ".eot", ".gif", ".gz", ".ico", ".jpeg", ".jpg", ".lock", ".mp3",
    ".mp4", ".otf", ".pdf", ".png", ".pyc", ".so", ".sqlite", ".tar",
    ".ttf", ".wav", ".webp", ".woff", ".woff2", ".xls", ".xlsx", ".zip",
}

def emit(status, **extra):
    payload = {"version": 1, "status": status}
    payload.update(extra)
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    print(PREFIX + encoded, flush=True)

if len(sys.argv) != 2:
    emit("invalid_file", path="", reason="Reader requires exactly one path argument.")
    raise SystemExit(2)

path = sys.argv[1]
try:
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        emit("invalid_file", path=path, reason="Path is not a regular file.")
        raise SystemExit(2)
    if os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS:
        emit("invalid_file", path=path, reason="Binary files are not supported by the read bridge.")
        raise SystemExit(2)

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    parts = []
    captured = 0
    total_lines = 0
    saw_text = False
    ends_with_newline = False
    truncated = False
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            text = decoder.decode(chunk, final=False)
            if "\x00" in text or any(
                (ord(char) < 9 or 13 < ord(char) < 32 or ord(char) == 127)
                for char in text
            ):
                raise ValueError("Binary or non-text content is not supported.")
            total_lines += text.count("\n")
            if text:
                saw_text = True
                ends_with_newline = text.endswith("\n")
            if captured < MAX_OUTPUT:
                part = text[: MAX_OUTPUT - captured]
                parts.append(part)
                captured += len(part)
                if len(part) != len(text):
                    truncated = True
            elif text:
                truncated = True
        tail = decoder.decode(b"", final=True)
        if "\x00" in tail or any(
            (ord(char) < 9 or 13 < ord(char) < 32 or ord(char) == 127)
            for char in tail
        ):
            raise ValueError("Binary or non-text content is not supported.")
        total_lines += tail.count("\n")
        if tail:
            saw_text = True
            ends_with_newline = tail.endswith("\n")
        if captured < MAX_OUTPUT:
            part = tail[: MAX_OUTPUT - captured]
            parts.append(part)
            captured += len(part)
            if len(part) != len(tail):
                truncated = True
        elif tail:
            truncated = True

    content = "".join(parts)
    if saw_text and not ends_with_newline:
        total_lines += 1
    if content.startswith("\ufeff"):
        content = content[1:]
    emit(
        "ok",
        path=path,
        content_b64=base64.b64encode(content.encode("utf-8")).decode("ascii"),
        total_lines=total_lines,
        file_size=int(info.st_size),
        truncated=truncated,
        range_applied=False,
    )
except FileNotFoundError:
    emit("file_not_found", path=path)
except PermissionError:
    emit("permission_denied", path=path)
except (UnicodeDecodeError, ValueError) as exc:
    emit("invalid_file", path=path, reason=str(exc))
except OSError as exc:
    emit("error", path=path, error=str(exc))
except Exception as exc:
    emit("error", path=path, error=str(exc))
"""


def _decode_proto_int32(value: Any) -> Optional[int]:
    if value is None:
        return None
    normalized = int(value) & 0xFFFFFFFF
    if normalized & 0x80000000:
        normalized -= 1 << 32
    return normalized


def _exec_request_identity(
    exec_fields: List[tuple[int, int, Any]],
) -> tuple[int, str]:
    return (
        _proto_last_field(exec_fields, 1, wire_type=0) or 0,
        _decode_proto_string(_proto_last_field(exec_fields, 15, wire_type=2)),
    )


def _encode_exec_client_message(
    exec_fields: List[tuple[int, int, Any]],
    *,
    message_field: int,
    message_payload: bytes,
) -> bytes:
    request_id, exec_id = _exec_request_identity(exec_fields)
    exec_client_message = b"".join(
        (
            _encode_proto_varint_field(1, request_id),
            _encode_proto_string_field(15, exec_id),
            _encode_proto_message_field(message_field, message_payload),
        )
    )
    return _encode_proto_message_field(2, exec_client_message)


def _encode_exec_stream_close(
    exec_fields: List[tuple[int, int, Any]],
) -> bytes:
    request_id, _exec_id = _exec_request_identity(exec_fields)
    return _encode_proto_message_field(
        5,
        _encode_proto_message_field(1, _encode_proto_varint_field(1, request_id)),
    )


def _encode_read_result(
    *,
    path: str,
    status: str,
    content: str = "",
    total_lines: int = 0,
    file_size: int = 0,
    truncated: bool = False,
    range_applied: bool = False,
    error: str = "",
    reason: str = "",
) -> bytes:
    if status == "ok":
        success = b"".join(
            (
                _encode_proto_string_field(1, path),
                _encode_proto_string_field(2, content, include_empty=True),
                _encode_proto_varint_field(3, total_lines),
                _encode_proto_varint_field(4, file_size),
                _encode_proto_varint_field(6, truncated),
                _encode_proto_varint_field(8, range_applied),
            )
        )
        return _encode_proto_message_field(1, success)
    if status == "file_not_found":
        return _encode_proto_message_field(
            4,
            _encode_proto_string_field(1, path),
        )
    if status == "permission_denied":
        return _encode_proto_message_field(
            5,
            _encode_proto_string_field(1, path),
        )
    if status == "rejected":
        rejected = _encode_proto_string_field(1, path) + _encode_proto_string_field(2, reason)
        return _encode_proto_message_field(3, rejected)
    if status == "invalid_file":
        invalid = _encode_proto_string_field(1, path) + _encode_proto_string_field(2, reason)
        return _encode_proto_message_field(6, invalid)
    if status == "error":
        read_error = _encode_proto_string_field(1, path) + _encode_proto_string_field(2, error)
        return _encode_proto_message_field(2, read_error)
    raise CursorConnectProtocolError(f"Unsupported Cursor read result status {status!r}.")


def _decode_read_args(
    exec_fields: List[tuple[int, int, Any]],
) -> Dict[str, Any]:
    args_payload = _proto_last_field(exec_fields, 7, wire_type=2)
    if not isinstance(args_payload, bytes):
        raise CursorConnectProtocolError(
            "Cursor Agent read operation does not contain read arguments."
        )
    args_fields = _decode_proto_fields(args_payload)
    path = _decode_proto_string(_proto_last_field(args_fields, 1, wire_type=2))
    if not path:
        raise CursorConnectProtocolError(
            "Cursor Agent read operation does not contain a path."
        )
    limit_value = _proto_last_field(args_fields, 5, wire_type=0)
    if limit_value is not None and int(limit_value) > 0xFFFFFFFF:
        raise CursorConnectProtocolError(
            "Cursor Agent read operation contains an invalid line limit."
        )
    offset = _decode_proto_int32(
        _proto_last_field(args_fields, 4, wire_type=0)
    ) or 0
    limit = int(limit_value or 0)
    encoding_hint = _decode_proto_string(
        _proto_last_field(args_fields, 6, wire_type=2)
    )
    unsupported = []
    if offset:
        unsupported.append("nondefault line offset")
    if limit:
        unsupported.append("nondefault line limit")
    if encoding_hint:
        unsupported.append("nondefault encoding_hint")
    if unsupported:
        raise CursorConnectProtocolError(
            "Cursor Agent read operation requests "
            f"{', '.join(unsupported)}; the external read bridge supports "
            "only full UTF-8 text reads."
        )
    request_id, exec_id = _exec_request_identity(exec_fields)
    return {
        "path": path,
        "tool_call_id": _decode_proto_string(
            _proto_last_field(args_fields, 2, wire_type=2)
        ),
        "offset": offset,
        "limit": limit,
        "encoding_hint": encoding_hint,
        "request_id": request_id,
        "exec_id": exec_id,
    }


def _read_command(path: str) -> str:
    return shlex.join(
        [
            "python3",
            "-c",
            _CURSOR_READ_COMMAND_SCRIPT,
            path,
        ]
    )


def _decode_cursor_read_envelope(  # noqa: PLR0915
    output: Any,
    *,
    expected_path: str,
) -> Dict[str, Any]:
    raw_output = output if isinstance(output, str) else str(output or "")
    if len(raw_output.encode("utf-8", "replace")) > _CURSOR_READ_ENVELOPE_MAX_BYTES:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read envelope exceeds the supported size.",
        }
    lines = raw_output.splitlines()
    header_line_index = 0
    if lines and _CURSOR_EXEC_TRUNCATION_WARNING_RE.fullmatch(lines[0]):
        header_line_index = 1
    if len(lines) < header_line_index + 5:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned invalid exec_command framing.",
        }
    if not _CURSOR_EXEC_CHUNK_ID_RE.fullmatch(lines[header_line_index]):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned invalid exec_command framing.",
        }
    if not _CURSOR_EXEC_WALL_TIME_RE.fullmatch(lines[header_line_index + 1]):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned invalid exec_command framing.",
        }
    process_line_index = header_line_index + 2
    if _CURSOR_EXEC_TOKEN_COUNT_RE.fullmatch(lines[process_line_index]):
        process_line_index += 1
    process_match = _CURSOR_EXEC_PROCESS_EXIT_RE.fullmatch(
        lines[process_line_index]
    )
    if process_match is None:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned invalid exec_command framing.",
        }
    exit_code = int(process_match.group(1))
    if exit_code != 0:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": f"The external read command exited with nonzero status {exit_code}.",
        }
    output_marker_index = process_line_index + 1
    if output_marker_index >= len(lines) or lines[output_marker_index] not in {
        "Final output:",
        "Output:",
    }:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned invalid exec_command framing.",
        }
    envelope_lines = lines[output_marker_index + 1 :]
    envelope_occurrences = sum(
        line.count(_CURSOR_READ_ENVELOPE_PREFIX) for line in envelope_lines
    )
    if envelope_occurrences == 0 and len(envelope_lines) == 1:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned an invalid envelope.",
        }
    if envelope_occurrences > 1:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": (
                "The external read command must return exactly one "
                "LITELLM_CURSOR_READ_V1: envelope."
            ),
        }
    if len(envelope_lines) != 1:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned ambiguous extra content.",
        }
    envelope_line = envelope_lines[0]
    if not envelope_line.startswith(_CURSOR_READ_ENVELOPE_PREFIX):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned ambiguous extra content.",
        }
    encoded = envelope_line[len(_CURSOR_READ_ENVELOPE_PREFIX) :]
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = base64.b64decode(
            encoded + padding,
            altchars=b"-_",
            validate=True,
        )
        decoded = json.loads(payload.decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned an invalid envelope.",
        }
    if not isinstance(decoded, Mapping) or decoded.get("version") != 1:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned an unsupported envelope.",
        }
    if decoded.get("path") != expected_path:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read envelope path did not match the request.",
        }
    status = decoded.get("status")
    if status in {"file_not_found", "permission_denied"}:
        return {"status": status, "path": expected_path}
    if status == "error":
        return {
            "status": "error",
            "path": expected_path,
            "error": str(decoded.get("error") or "External read failed."),
        }
    if status == "invalid_file":
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": str(decoded.get("reason") or "External read returned invalid content."),
        }
    if status != "ok":
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read command returned an unsupported status.",
        }
    content_b64 = decoded.get("content_b64")
    if not isinstance(content_b64, str):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read envelope omitted content.",
        }
    try:
        padding = "=" * (-len(content_b64) % 4)
        content_bytes = base64.b64decode(
            content_b64 + padding,
            validate=True,
        )
        content = content_bytes.decode("utf-8")
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read content was not valid UTF-8.",
        }
    if len(content) > _CURSOR_READ_MAX_OUTPUT_CHARS:
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read content exceeds the supported size.",
        }
    total_lines = decoded.get("total_lines")
    file_size = decoded.get("file_size")
    truncated = decoded.get("truncated")
    range_applied = decoded.get("range_applied")
    if (
        isinstance(total_lines, bool)
        or not isinstance(total_lines, int)
        or total_lines < 0
        or isinstance(file_size, bool)
        or not isinstance(file_size, int)
        or file_size < 0
        or not isinstance(truncated, bool)
        or not isinstance(range_applied, bool)
    ):
        return {
            "status": "invalid_file",
            "path": expected_path,
            "reason": "The external read envelope contained invalid metadata.",
        }
    return {
        "status": "ok",
        "path": expected_path,
        "content": content,
        "total_lines": total_lines,
        "file_size": file_size,
        "truncated": truncated,
        "range_applied": range_applied,
    }


def _encode_read_terminal_result(
    exec_request: Mapping[str, Any],
    output: Any,
) -> List[bytes]:
    expected_path = str(exec_request["path"])
    envelope = _decode_cursor_read_envelope(
        output,
        expected_path=expected_path,
    )
    read_result = _encode_read_result(**envelope)
    exec_fields = cast(List[tuple[int, int, Any]], exec_request["exec_fields"])
    return [
        _encode_exec_client_message(
            exec_fields,
            message_field=7,
            message_payload=read_result,
        ),
        _encode_exec_stream_close(exec_fields),
    ]


def _encode_external_exec_terminal_result(
    exec_request: Mapping[str, Any],
    output: Any,
) -> List[bytes]:
    message_field = int(exec_request["message_field"])
    if message_field == 7:
        return _encode_read_terminal_result(exec_request, output)
    raise CursorConnectProtocolError(
        f"Cursor Agent requested unsupported external exec field {message_field}."
    )


def _encode_unsupported_local_exec_response(
    exec_fields: List[tuple[int, int, Any]],
    message_field: int,
) -> List[bytes]:
    operation_name = _UNSUPPORTED_LOCAL_EXEC_FIELDS[message_field]
    request_id = _proto_last_field(exec_fields, 1, wire_type=0) or 0
    error = (
        "Cursor Agent local exec operation "
        f"{operation_name} (field {message_field}) is unsupported by this adapter; "
        "no local execution was performed."
    )
    exec_client_throw = b"".join(
        (
            _encode_proto_varint_field(1, request_id),
            _encode_proto_string_field(2, error),
        )
    )
    exec_client_stream_close = _encode_proto_varint_field(1, request_id)
    return [
        _encode_proto_message_field(
            5,
            _encode_proto_message_field(2, exec_client_throw),
        ),
        _encode_proto_message_field(
            5,
            _encode_proto_message_field(1, exec_client_stream_close),
        ),
    ]


def _advertised_local_exec_tool_name(
    request_payload: Mapping[str, Any],
) -> Optional[str]:
    run_request = _proto_mapping_value(
        request_payload,
        "runRequest",
        "run_request",
    )
    if not isinstance(run_request, Mapping):
        return None
    mcp_tools = _proto_mapping_value(run_request, "mcpTools", "mcp_tools")
    if not isinstance(mcp_tools, Mapping):
        return None
    definitions = _proto_mapping_value(mcp_tools, "mcpTools", "mcp_tools")
    if not isinstance(definitions, list):
        return None

    for preferred_name in _LOCAL_EXEC_TOOL_NAMES:
        for definition in definitions:
            if not isinstance(definition, Mapping):
                continue
            name = _proto_mapping_value(
                definition,
                "name",
                "toolName",
                "tool_name",
            )
            if isinstance(name, str) and name.casefold() == preferred_name:
                return name
    return None


def _decode_local_exec_tool_call(
    exec_fields: List[tuple[int, int, Any]],
    *,
    message_field: int,
    tool_name: str,
) -> Dict[str, Any]:
    return _decode_local_exec_request(
        exec_fields,
        message_field=message_field,
        tool_name=tool_name,
    )["normalized"]


def _decode_local_exec_request(
    exec_fields: List[tuple[int, int, Any]],
    *,
    message_field: int,
    tool_name: str,
) -> Dict[str, Any]:
    args_payload = _proto_last_field(
        exec_fields,
        message_field,
        wire_type=2,
    )
    if not isinstance(args_payload, bytes):
        raise CursorConnectProtocolError(
            "Cursor Agent local exec operation does not contain shell arguments."
        )
    args_fields = _decode_proto_fields(args_payload)
    command = _decode_proto_string(
        _proto_last_field(args_fields, 1, wire_type=2)
    )
    if not command:
        raise CursorConnectProtocolError(
            "Cursor Agent local exec operation does not contain a command."
        )
    working_directory = _decode_proto_string(
        _proto_last_field(args_fields, 2, wire_type=2)
    )
    request_id = _proto_last_field(exec_fields, 1, wire_type=0) or 0
    exec_id = _decode_proto_string(
        _proto_last_field(exec_fields, 15, wire_type=2)
    )
    if not exec_id and not request_id:
        raise CursorConnectProtocolError(
            "Cursor Agent local exec operation does not contain a replayable "
            "request identity."
        )

    argument_name = (
        "command"
        if tool_name.casefold() in {"bash", "shell", "run_shell_command"}
        else "cmd"
    )
    arguments: Dict[str, Any] = {argument_name: command}
    if working_directory:
        arguments["workdir"] = working_directory
    call_id = exec_id or f"cursor-exec-{request_id}"
    normalized = {
        "interactionUpdate": {
            "toolCallCompleted": {
                "callId": call_id,
                "toolName": tool_name,
                "argsJson": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "itemId": f"fc_{call_id}",
            }
        }
    }
    return {
        "normalized": normalized,
        "exec_request": {
            "call_id": call_id,
            "message_field": message_field,
            "exec_fields": exec_fields,
            "command": command,
            "working_directory": working_directory,
        },
    }


def _decode_read_tool_call(
    exec_fields: List[tuple[int, int, Any]],
    *,
    tool_name: str,
) -> Dict[str, Any]:
    read_args = _decode_read_args(exec_fields)
    request_id = int(read_args["request_id"])
    exec_id = str(read_args["exec_id"])
    call_id = str(read_args["tool_call_id"] or exec_id or f"cursor-read-{request_id}")
    if not call_id:
        raise CursorConnectProtocolError(
            "Cursor Agent read operation does not contain a replayable request identity."
        )
    command = _read_command(str(read_args["path"]))
    normalized = {
        "interactionUpdate": {
            "toolCallCompleted": {
                "callId": call_id,
                "toolName": tool_name,
                "argsJson": json.dumps(
                    {"cmd": command},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "itemId": f"fc_{call_id}",
            }
        }
    }
    return {
        "normalized": normalized,
        "exec_request": {
            "call_id": call_id,
            "message_field": 7,
            "exec_fields": exec_fields,
            "path": str(read_args["path"]),
        },
    }


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
    *,
    local_exec_tool_name: Optional[str] = None,
    external_exec_requests: Optional[List[Dict[str, Any]]] = None,
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
                None,
            )
            if (
                message_field in _LOCAL_EXEC_BRIDGE_FIELDS
                and local_exec_tool_name
            ):
                local_exec_request = _decode_local_exec_request(
                    exec_fields,
                    message_field=message_field,
                    tool_name=local_exec_tool_name,
                )
                return (
                    local_exec_request["normalized"],
                    [],
                )
            if message_field == 7 and local_exec_tool_name:
                read_tool_call = _decode_read_tool_call(
                    exec_fields,
                    tool_name=local_exec_tool_name,
                )
                if external_exec_requests is not None:
                    external_exec_requests.append(read_tool_call["exec_request"])
                return read_tool_call["normalized"], []
            if message_field in _UNSUPPORTED_LOCAL_EXEC_FIELDS:
                return (
                    decode_agent_server_message(payload),
                    _encode_unsupported_local_exec_response(
                        exec_fields,
                        message_field,
                    ),
                )
            raise CursorConnectProtocolError(
                "Cursor Agent requested unsupported local exec operation "
                f"field {message_field or 'unknown'}."
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


class CursorAgentRetainedSession:
    """A live native Run retained while an external tool executes."""

    def __init__(
        self,
        *,
        reader: Any,
        writer: Any,
        connection: Any,
        stream_id: int,
        decoder: _ProtoConnectFrameDecoder,
        blobs: Dict[bytes, bytes],
        local_exec_tool_name: Optional[str],
        saw_response_headers: bool,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.connection = connection
        self.stream_id = stream_id
        self.decoder = decoder
        self.blobs = blobs
        self.local_exec_tool_name = local_exec_tool_name
        self.saw_response_headers = saw_response_headers
        self.pending = bytearray()
        self._buffered_frames: List[CursorConnectProtoFrame] = []
        self._external_exec_requests: Dict[str, Dict[str, Any]] = {}
        self._closed = False
        self._wait_closed_started = False

    @property
    def can_continue(self) -> bool:
        return bool(self._external_exec_requests) and all(
            int(request.get("message_field") or 0) == 7
            for request in self._external_exec_requests.values()
        ) and not self._closed

    def register_external_exec(self, exec_request: Mapping[str, Any]) -> None:
        call_id = str(exec_request.get("call_id") or "")
        if not call_id:
            raise CursorConnectProtocolError(
                "Cursor Agent external exec request does not contain a call id."
            )
        if int(exec_request.get("message_field") or 0) != 7:
            raise CursorConnectProtocolError(
                "Cursor Agent retained continuation supports only field-7 read requests."
            )
        self._external_exec_requests[call_id] = dict(exec_request)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.writer, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    async def aclose(self) -> None:
        self.close()
        if self._wait_closed_started:
            return
        self._wait_closed_started = True
        wait_closed = getattr(self.writer, "wait_closed", None)
        if callable(wait_closed):
            try:
                await wait_closed()
            except Exception:
                pass

    async def continue_with_tool_outputs(
        self,
        outputs: List[tuple[str, Any]],
        *,
        timeout: Optional[float] = None,
    ) -> "CursorAgentRunResult":
        if self._closed:
            raise CursorConnectError(
                "Cursor Agent retained continuation session is closed.",
                status_code=409,
            )
        if not outputs:
            raise CursorConnectError(
                "Cursor Agent continuation requires a function_call_output.",
                status_code=400,
            )
        requests: List[tuple[Dict[str, Any], Any]] = []
        seen_call_ids: set[str] = set()
        for call_id, output in outputs:
            normalized_call_id = str(call_id)
            if normalized_call_id in seen_call_ids:
                raise CursorConnectError(
                    "Cursor Agent continuation contains a duplicate function_call_output.",
                    status_code=400,
                )
            seen_call_ids.add(normalized_call_id)
            exec_request = self._external_exec_requests.get(normalized_call_id)
            if exec_request is None:
                raise CursorConnectError(
                    "Cursor Agent continuation does not match a pending external tool call.",
                    status_code=409,
                )
            requests.append((exec_request, output))

        for call_id, _output in outputs:
            self._external_exec_requests.pop(str(call_id), None)
        for exec_request, output in requests:
            for client_message in _encode_external_exec_terminal_result(
                exec_request,
                output,
            ):
                self.pending.extend(encode_connect_proto_frame(client_message))
        result = await self._read_until_boundary(
            stop_on_tool_call=True,
            timeout=timeout,
        )
        if result.tool_calls and self.can_continue:
            result.retained_session = self
        return result

    async def _flush_pending(self) -> None:
        while True:
            outbound = CursorAgentConnectClient._flush_h2_request_data(
                self.connection,
                self.stream_id,
                self.pending,
            )
            if not outbound:
                return
            self.writer.write(outbound)
            await self.writer.drain()

    def _handle_frame(
        self,
        frame: CursorConnectProtoFrame,
        result: "CursorAgentRunResult",
        *,
        stop_on_tool_call: bool,
    ) -> bool:
        if frame.is_end_stream:
            _raise_for_connect_end_stream(frame.payload)
            result.end_stream = True
            return True
        external_exec_requests: List[Dict[str, Any]] = []
        normalized, client_messages = _process_agent_server_message(
            frame.payload,
            self.blobs,
            local_exec_tool_name=self.local_exec_tool_name,
            external_exec_requests=external_exec_requests,
        )
        for exec_request in external_exec_requests:
            self.register_external_exec(exec_request)
        for client_message in client_messages:
            self.pending.extend(encode_connect_proto_frame(client_message))
        if normalized:
            result.add_payload(normalized)
        if result.turn_ended:
            return True
        return bool(
            result.tool_calls
            and (stop_on_tool_call or self._external_exec_requests)
        )

    async def _read_until_boundary(  # noqa: PLR0915
        self,
        *,
        stop_on_tool_call: bool,
        timeout: Optional[float],
    ) -> "CursorAgentRunResult":
        loop = asyncio.get_running_loop()
        terminal_timeout = (
            float(timeout)
            if timeout is not None
            else CURSOR_CONNECT_TERMINAL_TIMEOUT_SECONDS
        )
        if terminal_timeout <= 0:
            raise CursorConnectError(
                "Cursor Agent Connect timeout must be greater than zero.",
                status_code=500,
            )
        deadline = loop.time() + terminal_timeout
        next_heartbeat = loop.time() + CURSOR_CONNECT_HEARTBEAT_SECONDS
        result = CursorAgentRunResult()

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
                self.pending.extend(
                    encode_connect_proto_frame(_encode_proto_message_field(7, b""))
                )
                next_heartbeat = now + CURSOR_CONNECT_HEARTBEAT_SECONDS
            await self._flush_pending()

            if self._buffered_frames:
                frame = self._buffered_frames.pop(0)
                if self._handle_frame(
                    frame,
                    result,
                    stop_on_tool_call=stop_on_tool_call,
                ):
                    return result
                continue

            wait_seconds = min(
                deadline - loop.time(),
                max(0.0, next_heartbeat - loop.time()),
            )
            try:
                incoming = await asyncio.wait_for(
                    self.reader.read(64 * 1024),
                    timeout=wait_seconds,
                )
            except TimeoutError:
                continue
            if not incoming:
                self.decoder.finish()
                if not self.saw_response_headers:
                    raise CursorConnectError(
                        "Cursor Agent HTTP/2 stream closed before response headers.",
                        status_code=502,
                    )
                result.validate_terminal()
                return result

            from h2 import events as h2_events

            for event in self.connection.receive_data(incoming):
                if isinstance(event, h2_events.ResponseReceived):
                    response_headers = CursorAgentConnectClient._h2_response_headers(
                        event.headers
                    )
                    CursorAgentConnectClient._validate_h2_response_headers(
                        response_headers
                    )
                    self.saw_response_headers = True
                    continue
                if isinstance(event, h2_events.DataReceived):
                    self.connection.acknowledge_received_data(
                        event.flow_controlled_length,
                        event.stream_id,
                    )
                    frames = self.decoder.feed(event.data)
                    for index, frame in enumerate(frames):
                        if self._handle_frame(
                            frame,
                            result,
                            stop_on_tool_call=stop_on_tool_call,
                        ):
                            self._buffered_frames.extend(frames[index + 1 :])
                            return result
                    continue
                if isinstance(event, h2_events.TrailersReceived):
                    trailer_headers = CursorAgentConnectClient._h2_response_headers(
                        event.headers
                    )
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
                    self.decoder.finish()
                    result.validate_terminal()
                    return result
                if isinstance(event, h2_events.StreamReset):
                    raise CursorConnectError(
                        "Cursor Agent HTTP/2 stream was reset "
                        f"(error_code={event.error_code}).",
                        status_code=502,
                    )
                if isinstance(event, h2_events.ConnectionTerminated):
                    raise CursorConnectError(
                        "Cursor Agent HTTP/2 connection terminated "
                        f"(error_code={event.error_code}).",
                        status_code=502,
                    )

    async def start(
        self,
        request_body: bytes,
        *,
        timeout: Optional[float],
        stop_on_tool_call: bool,
    ) -> "CursorAgentRunResult":
        self.pending.extend(request_body)
        return await self._read_until_boundary(
            stop_on_tool_call=stop_on_tool_call,
            timeout=timeout,
        )


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
    retained_session: Optional["CursorAgentRetainedSession"] = field(
        default=None,
        repr=False,
    )
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
    return _read_managed_auth_file().auth_file


def _managed_auth_file_path() -> Optional[str]:
    path = _read_secret_env(CURSOR_AGENT_AUTH_FILE_ENV)
    return path or None


@dataclass(frozen=True)
class _CursorAgentAuthFileIdentity:
    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    content_digest: str


@dataclass(frozen=True)
class _CursorAgentAuthFileRead:
    auth_file: Dict[str, Any]
    identity: Optional[_CursorAgentAuthFileIdentity]
    error: Optional[str] = None


class _CursorAgentAuthFileHandle:
    def __init__(self, path: str) -> None:
        self._handle = Path(path).open("rb")

    def __enter__(self) -> "_CursorAgentAuthFileHandle":
        return self

    def __exit__(self, *_args: Any) -> None:
        self._handle.close()

    def read(self) -> bytes:
        return self._handle.read()

    def stat(self) -> os.stat_result:
        return os.fstat(self._handle.fileno())


def _read_managed_auth_file() -> _CursorAgentAuthFileRead:
    """Read the sidecar-owned credential without writing or refreshing it."""
    path = _managed_auth_file_path()
    if not path:
        return _CursorAgentAuthFileRead({}, None, "unavailable")
    try:
        with _CursorAgentAuthFileHandle(path) as auth_handle:
            raw_auth_file = auth_handle.read()
            file_stat = auth_handle.stat()
    except (OSError, ValueError):
        return _CursorAgentAuthFileRead({}, None, "unreadable")

    identity = _CursorAgentAuthFileIdentity(
        path=path,
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        content_digest=hashlib.sha256(raw_auth_file).hexdigest(),
    )
    try:
        loaded = json.loads(raw_auth_file.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _CursorAgentAuthFileRead({}, identity, "invalid-json")
    if not isinstance(loaded, Mapping):
        return _CursorAgentAuthFileRead({}, identity, "invalid-shape")
    return _CursorAgentAuthFileRead(dict(loaded), identity)


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
        self._file_identity: Optional[_CursorAgentAuthFileIdentity] = None
        self._file_error: Optional[str] = None

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
            not _managed_auth_file_path()
            and not force_refresh
            and self._cached_token
            and self._cached_token not in self._rejected_tokens
            and access_token_is_fresh(self._cached_token)
        ):
            return self._cached_token
        async with self._lock:
            auth_read: Optional[_CursorAgentAuthFileRead] = None
            if _managed_auth_file_path():
                auth_read = self._refresh_file_cache_metadata()
            if (
                not force_refresh
                and self._cached_token
                and self._cached_token not in self._rejected_tokens
                and access_token_is_fresh(self._cached_token)
            ):
                return self._cached_token
            token = await self._resolve_uncached(auth_read=auth_read)
            self._cached_token = token
            return token

    def _refresh_file_cache_metadata(self) -> _CursorAgentAuthFileRead:
        auth_read = _read_managed_auth_file()
        changed = self._file_identity != auth_read.identity or self._file_error != auth_read.error
        if changed:
            self._cached_token = None
            self._rejected_tokens.clear()
        self._file_identity = auth_read.identity
        self._file_error = auth_read.error
        return auth_read

    async def _resolve_uncached(
        self,
        *,
        auth_read: Optional[_CursorAgentAuthFileRead] = None,
    ) -> str:
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

        if auth_read is None:
            auth_read = _read_managed_auth_file()
        auth_file = auth_read.auth_file
        file_access = str(auth_file.get("accessToken") or auth_file.get("access_token") or "").strip()
        if file_access and file_access not in self._rejected_tokens:
            if access_token_is_fresh(file_access):
                return file_access

        if _managed_auth_file_path():
            raise CursorConnectError(
                "Cursor Agent auth file does not contain a fresh accessToken; "
                "request-time API key exchange is not supported in managed "
                "auth-file mode.",
                status_code=401,
            )
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
    if _managed_auth_file_path():
        raise CursorConnectError(
            "Cursor Agent auth file does not contain a fresh accessToken. "
            "CURSOR_CLI_KEY is ignored and auth-file API keys are not exchanged "
            "at request time.",
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
        local_exec_tool_name: Optional[str],
        retain_on_tool_call: bool,
    ) -> CursorAgentRunResult:
        """Run Cursor's true bidi RPC without half-closing the request."""
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
        writer: Any = None
        session: Optional[CursorAgentRetainedSession] = None
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
            session = CursorAgentRetainedSession(
                reader=reader,
                writer=writer,
                connection=connection,
                stream_id=stream_id,
                decoder=_ProtoConnectFrameDecoder(),
                blobs={},
                local_exec_tool_name=local_exec_tool_name,
                saw_response_headers=False,
            )
            writer.write(connection.data_to_send())
            await writer.drain()
            result = await session.start(
                request_body,
                timeout=terminal_timeout,
                stop_on_tool_call=stop_on_tool_call,
            )
            if retain_on_tool_call and result.tool_calls and session.can_continue:
                result.retained_session = session
                session = None
                writer = None
            return result
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
            if session is not None:
                await session.aclose()
            elif writer is not None:
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
        local_exec_tool_name: Optional[str],
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
                local_exec_tool_name=local_exec_tool_name,
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
        local_exec_tool_name: Optional[str],
        retain_on_tool_call: bool,
    ) -> CursorAgentRunResult:
        if self.http_client is None and self.client_factory is None:
            self._ensure_http2_dependency()
            return await self._run_h2_bidi_once(
                url=url,
                request_body=request_body,
                headers=headers,
                stop_on_tool_call=stop_on_tool_call,
                timeout=timeout,
                local_exec_tool_name=local_exec_tool_name,
                retain_on_tool_call=retain_on_tool_call,
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
                            local_exec_tool_name=local_exec_tool_name,
                        )
                response = context
                try:
                    return await self._consume_response(
                        response,
                        stop_on_tool_call=stop_on_tool_call,
                        local_exec_tool_name=local_exec_tool_name,
                    )
                finally:
                    aclose = getattr(response, "aclose", None)
                    if callable(aclose):
                        await aclose()
            response = await client.post(url, **request_kwargs)
            return await self._consume_response(
                response,
                stop_on_tool_call=stop_on_tool_call,
                local_exec_tool_name=local_exec_tool_name,
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
        retain_on_tool_call: bool = False,
    ) -> CursorAgentRunResult:
        """Run one Connect request, retrying one 401 with refreshed auth."""
        self._ensure_http2_dependency()
        target_url = (url or "").strip()
        if not target_url:
            target_url = f"{CURSOR_AGENT_TURN_HOST}{CURSOR_AGENT_RUN_PATH}"
        request_body = encode_cursor_run_request(request_payload)
        run_request = request_payload.get("runRequest")
        run_id = str(run_request.get("runId") or "").strip() if isinstance(run_request, Mapping) else ""
        local_exec_tool_name = _advertised_local_exec_tool_name(request_payload)
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
                local_exec_tool_name=local_exec_tool_name,
                retain_on_tool_call=retain_on_tool_call,
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
                local_exec_tool_name=local_exec_tool_name,
                retain_on_tool_call=retain_on_tool_call,
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
