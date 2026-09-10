"""Inbound Cursor Agent CLI Connect surface.

This is the default HTTP/2 bidi turn the host ``cursoral`` / ``cursoralt`` /
``cursorala`` launchers aim at LiteLLM with hidden ``--agent-endpoint``.

It is not Cloud Agents ``/cursor`` (Basic ``API_KEY:`` to ``api.cursor.com``)
and not outbound ``cursor_agent`` adapters (Codex/Anthropic calling
``agentn``). Dashboard login stays on ``api2``; this module does not read
``CURSOR_API_ENDPOINT``.

Auth is the CLI's existing Bearer access token. ``CURSOR_CLI_KEY`` is ignored.
Raw ``CURSOR_API_KEY`` is not used as the Connect credential.
"""

from __future__ import annotations

import asyncio
import ssl
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import Request
from starlette.responses import Response

from litellm._logging import verbose_proxy_logger
from litellm.llms.cursor_agent.constants import (
    CURSOR_AGENT_CONNECT_CONTENT_TYPE,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_AGENT_TURN_HOST,
    CURSOR_CLI_KEY_ENV,
)
from litellm.llms.cursor_agent.connect import (
    CursorAgentConnectClient,
    CursorConnectError,
    _ProtoConnectFrameDecoder,
    _decode_proto_fields,
    _decode_proto_string,
    _encode_request_context_exec_response,
    _proto_last_field,
    decode_connect_proto_frames,
    encode_connect_proto_frame,
    ensure_cursor_http2_available,
)
from litellm.llms.cursor_agent.dashboard import cursor_agent_user_agent
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

CURSOR_AGENT_CLI_INBOUND_PROVIDER = "cursor_agent_cli_inbound"
CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY = "cursor_agent_cli_inbound"
CURSOR_AGENT_CLI_INBOUND_TRACE_NAME = "cursor-agent-cli-inbound"
CURSOR_AGENT_CLI_INBOUND_TAGS = (
    "route:cursor_agent_cli_inbound",
    "cursor-agent-cli-inbound",
    "inbound-versus-outbound:inbound",
)
CURSOR_AGENT_RUNSSE_PATH = "/agent.v1.AgentService/RunSSE"
CURSOR_AGENT_BIDI_APPEND_PATH = "/aiserver.v1.BidiService/BidiAppend"

_HOP_BY_HOP = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_FORWARDED_REQUEST_HEADERS = {
    "accept",
    "connect-protocol-version",
    "content-type",
    "user-agent",
    "x-cursor-client-type",
    "x-cursor-client-version",
    "x-cursor-streaming",
    "x-ghost-mode",
    "x-request-id",
}
_MAX_LOGGED_AGENTN_DATA_CHUNKS = 8


def _connect_header_flush_frame() -> bytes:
    """Empty ASGI body that still flushes Hypercorn HTTP/2 HEADERS.

    Hypercorn sends HTTP/2 HEADERS on the first ``http.response.body`` event
    even when that body is empty; empty DATA is not forwarded. A non-empty
    dummy Connect envelope would poison the CLI stream.
    """
    return b""


def _summarize_connect_chunk(chunk: bytes) -> str:
    """Describe Connect envelopes without retaining protobuf field values."""
    try:
        frames = decode_connect_proto_frames(chunk)
    except Exception:
        return f"head={chunk[:24].hex()}"
    if not frames:
        return f"head={chunk[:24].hex()}"
    parts: List[str] = []
    for frame in frames:
        field_numbers: List[str] = []
        nested: List[str] = []
        if not frame.is_end_stream:
            try:
                decoded = _decode_proto_fields(frame.payload)
            except Exception:
                decoded = []
            field_numbers = [str(number) for number, _wire, _value in decoded]
            for number, wire_type, value in decoded:
                if wire_type != 2 or not isinstance(value, bytes) or number not in {1, 2, 4, 5, 7}:
                    continue
                try:
                    inner = _decode_proto_fields(value)
                except Exception:
                    continue
                nested.append(
                    f"{number}:[{','.join(str(inner_number) for inner_number, _w, _v in inner)}]"
                )
        parts.append(
            "flags={flags} end={end} len={length} fields={fields} nested={nested}".format(
                flags=frame.flags,
                end=int(frame.is_end_stream),
                length=len(frame.payload),
                fields=",".join(field_numbers) or "-",
                nested=";".join(nested) or "-",
            )
        )
    return " ".join(parts)


def _request_context_exec_replies(
    chunk: bytes,
    decoder: _ProtoConnectFrameDecoder,
) -> List[bytes]:
    """Answer agentn exec_server_message request-context queries.

    Native CLI answers these on a direct agentn stream. Through the inbound
    proxy the CLI never emits that reply, so agentn waits on heartbeats.
    """
    try:
        frames = decoder.feed(chunk)
    except Exception:
        return []
    replies: List[bytes] = []
    for frame in frames:
        if frame.is_end_stream:
            continue
        try:
            fields = _decode_proto_fields(frame.payload)
        except Exception:
            continue
        exec_server = _proto_last_field(fields, 2, wire_type=2)
        if not isinstance(exec_server, bytes):
            continue
        try:
            exec_fields = _decode_proto_fields(exec_server)
        except Exception:
            continue
        request_context_args = _proto_last_field(exec_fields, 10, wire_type=2)
        if not isinstance(request_context_args, bytes):
            continue
        for payload in _encode_request_context_exec_response(exec_fields):
            replies.append(encode_connect_proto_frame(payload))
    return replies


def is_cursor_agent_cli_run_scope(scope: Mapping[str, Any]) -> bool:
    if scope.get("type") != "http":
        return False
    if str(scope.get("method") or "").upper() != "POST":
        return False
    path = str(scope.get("path") or "")
    return path == CURSOR_AGENT_RUN_PATH or path.endswith(CURSOR_AGENT_RUN_PATH)


class InboundCursorAgentCliAuthError(Exception):
    """Attributable inbound CLI auth failure (never a 404)."""

    def __init__(self, message: str, *, status_code: int = 401, reason: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.reason = reason


def _header_map(headers: Any) -> Dict[str, str]:
    if headers is None:
        return {}
    if isinstance(headers, Mapping):
        return {str(key): str(value) for key, value in headers.items() if value is not None}
    result: Dict[str, str] = {}
    for item in headers:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        key, value = item[0], item[1]
        if isinstance(key, bytes):
            key = key.decode("latin1")
        if isinstance(value, bytes):
            value = value.decode("latin1")
        result[str(key)] = str(value)
    return result


def _get_header(headers: Mapping[str, str], name: str) -> str:
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return ""


def require_inbound_cli_bearer(
    headers: Any,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the CLI Bearer access token from inbound Connect headers.

    Cloud Agents Basic auth is rejected. ``CURSOR_CLI_KEY`` is ignored even if
    present in the process environment. The raw ``CURSOR_API_KEY`` env value is
    never used as the Connect credential.
    """
    _ = environ.get(CURSOR_CLI_KEY_ENV) if environ is not None else None
    header_map = _header_map(headers)
    authorization = _get_header(header_map, "authorization").strip()
    if not authorization:
        raise InboundCursorAgentCliAuthError(
            "Cursor Agent CLI Connect requires Authorization: Bearer <accessToken>.",
            reason="missing_authorization",
        )
    scheme, _, remainder = authorization.partition(" ")
    scheme_lower = scheme.lower()
    if scheme_lower == "basic":
        raise InboundCursorAgentCliAuthError(
            "Cloud Agents Basic authentication is not accepted on the Agent CLI Connect route.",
            reason="cloud_agents_basic_not_accepted",
        )
    if scheme_lower != "bearer":
        raise InboundCursorAgentCliAuthError(
            "Cursor Agent CLI Connect requires Authorization: Bearer <accessToken>.",
            reason="non_bearer_authorization",
        )
    token = remainder.strip()
    if not token:
        raise InboundCursorAgentCliAuthError(
            "Cursor Agent CLI Connect Bearer access token is empty.",
            reason="empty_bearer",
        )
    return token


def inbound_cli_auth_error_payload(exc: InboundCursorAgentCliAuthError) -> Dict[str, str]:
    return {
        "error": "cursor_agent_cli_inbound_auth",
        "reason": exc.reason,
        "detail": exc.message,
    }


def _sniff_run_metadata(buffer: bytes) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    try:
        frames = decode_connect_proto_frames(buffer)
    except Exception:
        return metadata
    for frame in frames:
        if frame.is_end_stream:
            continue
        try:
            fields = _decode_proto_fields(frame.payload)
        except Exception:
            continue
        run_payload = _proto_last_field(fields, 1, wire_type=2)
        if not isinstance(run_payload, bytes):
            continue
        try:
            run_fields = _decode_proto_fields(run_payload)
        except Exception:
            continue
        requested = _proto_last_field(run_fields, 9, wire_type=2)
        if isinstance(requested, bytes):
            try:
                model_id = _decode_proto_string(
                    _proto_last_field(_decode_proto_fields(requested), 1, wire_type=2)
                )
            except Exception:
                model_id = ""
            if model_id:
                metadata["model_id"] = model_id
        conversation_id = _decode_proto_string(_proto_last_field(run_fields, 5, wire_type=2))
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        run_id = _decode_proto_string(_proto_last_field(run_fields, 25, wire_type=2))
        if run_id:
            metadata["run_id"] = run_id
        if metadata:
            return metadata
    return metadata


def build_inbound_cli_session_history_kwargs(
    *,
    call_id: str,
    headers: Mapping[str, str],
    model_id: str,
    conversation_id: str,
    run_id: str,
    connect_method: str = "Run",
    http_version: str = "2",
    client_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Kwargs for the shipped session_history builder / success callback."""
    user_agent = _get_header(headers, "user-agent")
    request_id = _get_header(headers, "x-request-id")
    session_id = conversation_id or request_id or call_id
    safe_headers = {
        key: value
        for key, value in headers.items()
        if str(key).lower() != "authorization"
    }
    tags = list(CURSOR_AGENT_CLI_INBOUND_TAGS)
    metadata = {
        "trace_name": CURSOR_AGENT_CLI_INBOUND_TRACE_NAME,
        "passthrough_route_family": CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY,
        "aawm_passthrough_endpoint_type": CURSOR_AGENT_CLI_INBOUND_ROUTE_FAMILY,
        "inbound_versus_outbound": "inbound",
        "cursor_connect_method": connect_method,
        "cursor_connect_http_version": http_version,
        "tags": tags,
        "client_user_agent": user_agent or None,
        "session_id": session_id,
        "cursor_agent_run_id": run_id or None,
        "cursor_agent_conversation_id": conversation_id or None,
    }
    return {
        "model": model_id or "cursor_agent_cli",
        "custom_llm_provider": CURSOR_AGENT_CLI_INBOUND_PROVIDER,
        "call_type": "pass_through_endpoint",
        "litellm_call_id": call_id,
        "litellm_params": {
            "api_base": CURSOR_AGENT_TURN_HOST,
            "custom_llm_provider": CURSOR_AGENT_CLI_INBOUND_PROVIDER,
            "metadata": metadata,
            "proxy_server_request": {
                "headers": safe_headers,
                "url": CURSOR_AGENT_RUN_PATH,
                "method": "POST",
            },
        },
        "standard_logging_object": {
            "metadata": dict(metadata),
            "request_tags": tags,
            "session_id": session_id,
            "custom_llm_provider": CURSOR_AGENT_CLI_INBOUND_PROVIDER,
        },
        "passthrough_logging_payload": {
            "request_headers": safe_headers,
        },
        "client_ip": client_host,
    }


def _asgi_headers(scope: Mapping[str, Any]) -> Dict[str, str]:
    return _header_map(scope.get("headers") or [])


def _client_host(scope: Mapping[str, Any]) -> Optional[str]:
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        return str(client[0])
    return None


def _http_version(scope: Mapping[str, Any]) -> str:
    version = str(scope.get("http_version") or "")
    if version.startswith("2"):
        return "2"
    return version or "1.1"


async def _send_json_error(
    send: Callable[[Mapping[str, Any]], Awaitable[None]],
    *,
    status_code: int,
    payload: Mapping[str, str],
) -> None:
    import json

    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _persist_inbound_cli_turn(
    *,
    call_id: str,
    headers: Mapping[str, str],
    sniffed: Mapping[str, str],
    http_version: str,
    client_host: Optional[str],
    status_code: int,
    start_time: datetime,
    error: Optional[str] = None,
) -> None:
    kwargs = build_inbound_cli_session_history_kwargs(
        call_id=call_id,
        headers=headers,
        model_id=sniffed.get("model_id") or "",
        conversation_id=sniffed.get("conversation_id") or "",
        run_id=sniffed.get("run_id") or "",
        http_version=http_version,
        client_host=client_host,
    )
    end_time = datetime.now(timezone.utc)
    result: Dict[str, Any] = {
        "id": sniffed.get("run_id") or call_id,
        "object": "cursor_agent_cli_inbound.run",
        "status": "completed" if 200 <= status_code < 300 and not error else "failed",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if error:
        result["error"] = {"message": error, "type": "cursor_agent_cli_inbound"}
        kwargs["litellm_params"]["metadata"]["inbound_cli_error"] = error
        kwargs["standard_logging_object"]["metadata"]["inbound_cli_error"] = error
    try:
        logging_obj = LiteLLMLoggingObj(
            model=str(kwargs.get("model") or "cursor_agent_cli"),
            messages=None,
            stream=True,
            call_type="pass_through_endpoint",
            start_time=start_time,
            litellm_call_id=call_id,
            function_id="cursor_agent_cli_inbound",
            kwargs=kwargs,
        )
        logging_obj.model_call_details.update(
            {
                "custom_llm_provider": CURSOR_AGENT_CLI_INBOUND_PROVIDER,
                "call_type": "pass_through_endpoint",
                **kwargs,
            }
        )
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "cursor_agent_cli_inbound session_history/Langfuse persist failed: %s",
            exc,
        )


class _AgentnH2Session:
    """HTTP/2 Connect client session to the Cursor Agent turn host."""

    def __init__(self, turn_base: str) -> None:
        parsed = urlsplit(turn_base)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise CursorConnectError(
                "Inbound Cursor Agent CLI egress requires HTTPS agentn without userinfo.",
                status_code=500,
            )
        self.hostname = parsed.hostname
        self.port = parsed.port or 443
        self.authority = self.hostname if self.port == 443 else f"{self.hostname}:{self.port}"
        self.path = CURSOR_AGENT_RUN_PATH
        self.reader: Any = None
        self.writer: Any = None
        self.connection: Any = None
        self.stream_id = 0
        self.pending = bytearray()
        self.response_status = 200
        self._headers_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._incoming: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._read_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._logged_data_chunks = 0
        self._agentn_body_decoder = _ProtoConnectFrameDecoder()
        self._auto_replies: List[bytes] = []

    async def _flush_connection(self) -> None:
        writer = self.writer
        connection = self.connection
        if writer is None or connection is None:
            return
        outbound = connection.data_to_send()
        if outbound:
            writer.write(outbound)
            await writer.drain()

    async def _acknowledge_remote_settings(self) -> None:
        """Read agentn SETTINGS and ACK before sending request DATA."""
        from h2.events import RemoteSettingsChanged, SettingsAcknowledged, WindowUpdated

        reader = self.reader
        if reader is None or self.connection is None:
            raise CursorConnectError(
                "Inbound Cursor Agent CLI egress lost the HTTP/2 socket during handshake.",
                status_code=502,
            )
        deadline = asyncio.get_running_loop().time() + 10.0
        while asyncio.get_running_loop().time() < deadline:
            timeout = max(0.05, deadline - asyncio.get_running_loop().time())
            incoming = await asyncio.wait_for(reader.read(64 * 1024), timeout=timeout)
            if not incoming:
                raise CursorConnectError(
                    "Inbound Cursor Agent CLI egress closed during HTTP/2 handshake.",
                    status_code=502,
                )
            events = self.connection.receive_data(incoming)
            await self._flush_connection()
            if any(
                isinstance(event, (RemoteSettingsChanged, SettingsAcknowledged, WindowUpdated))
                for event in events
            ):
                return
        raise CursorConnectError(
            "Inbound Cursor Agent CLI egress timed out waiting for HTTP/2 SETTINGS.",
            status_code=504,
        )

    async def open(self, request_headers: List[Tuple[str, str]]) -> None:
        from h2.config import H2Configuration
        from h2.connection import H2Connection

        ensure_cursor_http2_available()
        ssl_context = ssl.create_default_context()
        ssl_context.set_alpn_protocols(["h2"])
        self.reader, self.writer = await asyncio.open_connection(
            self.hostname,
            self.port,
            ssl=ssl_context,
            server_hostname=self.hostname,
        )
        ssl_object = self.writer.get_extra_info("ssl_object")
        selected = ssl_object.selected_alpn_protocol() if ssl_object is not None else None
        if selected != "h2":
            raise CursorConnectError(
                "Inbound Cursor Agent CLI egress requires negotiated HTTP/2; "
                f"received {selected or 'unknown'}.",
                status_code=502,
            )
        self.connection = H2Connection(
            config=H2Configuration(client_side=True, header_encoding="utf-8")
        )
        self.connection.initiate_connection()
        await self._flush_connection()
        await self._acknowledge_remote_settings()
        self.stream_id = self.connection.get_next_available_stream_id()
        self.connection.send_headers(
            self.stream_id,
            [
                (":method", "POST"),
                (":scheme", "https"),
                (":authority", self.authority),
                (":path", self.path),
                *request_headers,
            ],
            end_stream=False,
        )
        await self._flush_connection()
        verbose_proxy_logger.info(
            "cursor_agent_cli_inbound opened agentn stream_id=%s",
            self.stream_id,
        )
        self._read_task = asyncio.create_task(self._read_loop())

    def _dispatch_h2_events(self, events: List[Any]) -> Tuple[List[bytes], bool]:
        from h2.events import (
            ConnectionTerminated,
            DataReceived,
            ResponseReceived,
            StreamEnded,
            StreamReset,
            TrailersReceived,
            WindowUpdated,
        )

        chunks: List[bytes] = []
        ended = False
        for event in events:
            if isinstance(event, ResponseReceived):
                status = "200"
                header_names: List[str] = []
                for name, value in event.headers:
                    if name == ":status":
                        status = value
                    elif not str(name).startswith(":"):
                        header_names.append(str(name))
                try:
                    self.response_status = int(status)
                except ValueError:
                    self.response_status = 502
                self._headers_event.set()
                verbose_proxy_logger.info(
                    "cursor_agent_cli_inbound agentn response status=%s headers=%s",
                    self.response_status,
                    ",".join(header_names),
                )
            elif isinstance(event, DataReceived):
                if event.data:
                    payload = bytes(event.data)
                    chunks.append(payload)
                    self._auto_replies.extend(
                        _request_context_exec_replies(payload, self._agentn_body_decoder)
                    )
                    if self._logged_data_chunks < _MAX_LOGGED_AGENTN_DATA_CHUNKS:
                        self._logged_data_chunks += 1
                        verbose_proxy_logger.info(
                            "cursor_agent_cli_inbound agentn data bytes=%s %s",
                            len(payload),
                            _summarize_connect_chunk(payload),
                        )
                    else:
                        verbose_proxy_logger.info(
                            "cursor_agent_cli_inbound agentn data bytes=%s",
                            len(payload),
                        )
                if self.connection is not None:
                    self.connection.acknowledge_received_data(
                        event.flow_controlled_length,
                        event.stream_id,
                    )
            elif isinstance(event, (StreamEnded, TrailersReceived)):
                ended = True
            elif isinstance(event, (StreamReset, ConnectionTerminated)):
                verbose_proxy_logger.warning(
                    "cursor_agent_cli_inbound agentn stream closed: %s",
                    type(event).__name__,
                )
                ended = True
            elif isinstance(event, WindowUpdated):
                if self.connection is not None:
                    CursorAgentConnectClient._flush_h2_request_data(
                        self.connection,
                        self.stream_id,
                        self.pending,
                    )
        return chunks, ended

    async def _read_loop(self) -> None:
        reader = self.reader
        if reader is None:
            await self._incoming.put(None)
            return
        try:
            while not self._closed:
                incoming = await reader.read(64 * 1024)
                if not incoming:
                    verbose_proxy_logger.info(
                        "cursor_agent_cli_inbound agentn read EOF status=%s",
                        self.response_status,
                    )
                    break
                async with self._lock:
                    if self.connection is None:
                        break
                    events = self.connection.receive_data(incoming)
                    chunks, ended = self._dispatch_h2_events(events)
                    auto_replies = self._auto_replies
                    self._auto_replies = []
                    await self._flush_connection()
                for reply in auto_replies:
                    verbose_proxy_logger.info(
                        "cursor_agent_cli_inbound auto-answered request_context bytes=%s",
                        len(reply),
                    )
                    await self.write_request(reply, end_stream=False)
                for chunk in chunks:
                    await self._incoming.put(chunk)
                if ended:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            verbose_proxy_logger.warning(
                "cursor_agent_cli_inbound agentn read loop failed: %s",
                exc,
            )
        finally:
            await self._incoming.put(None)

    async def write_request(self, data: bytes, *, end_stream: bool = False) -> None:
        async with self._lock:
            writer = self.writer
            connection = self.connection
            if writer is None or connection is None:
                return
            if data:
                self.pending.extend(data)
            outbound = CursorAgentConnectClient._flush_h2_request_data(
                connection,
                self.stream_id,
                self.pending,
            )
            if outbound:
                writer.write(outbound)
                await writer.drain()
            if end_stream:
                connection.end_stream(self.stream_id)
                leftover = connection.data_to_send()
                if leftover:
                    writer.write(leftover)
                    await writer.drain()
        verbose_proxy_logger.info(
            "cursor_agent_cli_inbound wrote agentn bytes=%s end_stream=%s pending=%s",
            len(data),
            end_stream,
            len(self.pending),
        )

    async def iter_response_data(self):
        while True:
            chunk = await self._incoming.get()
            if chunk is None:
                return
            yield chunk

    async def aclose(self) -> None:
        self._closed = True
        read_task = self._read_task
        self._read_task = None
        if read_task is not None and not read_task.done():
            read_task.cancel()
            try:
                await read_task
            except (asyncio.CancelledError, Exception):
                pass
        writer = self.writer
        self.writer = None
        self.reader = None
        try:
            self._incoming.put_nowait(None)
        except Exception:
            pass
        if writer is None:
            return
        try:
            writer.close()
            abort_fn = getattr(writer, "abort", None)
            if callable(abort_fn):
                abort_fn()
            wait_closed = getattr(writer, "wait_closed", None)
            if callable(wait_closed):
                await asyncio.wait_for(wait_closed(), timeout=1.0)
        except Exception:
            pass



def _upstream_request_headers(
    headers: Mapping[str, str],
    access_token: str,
) -> List[Tuple[str, str]]:
    forwarded: List[Tuple[str, str]] = [
        ("authorization", f"Bearer {access_token}"),
        ("content-type", _get_header(headers, "content-type") or CURSOR_AGENT_CONNECT_CONTENT_TYPE),
        (
            "connect-protocol-version",
            _get_header(headers, "connect-protocol-version") or "1",
        ),
        ("user-agent", _get_header(headers, "user-agent") or cursor_agent_user_agent()),
    ]
    seen = {name for name, _value in forwarded}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered in seen or lowered in _HOP_BY_HOP or lowered == "authorization":
            continue
        if lowered in _FORWARDED_REQUEST_HEADERS or lowered.startswith("x-cursor-"):
            if lowered == "x-cursor-checksum":
                continue
            forwarded.append((lowered, str(value)))
            seen.add(lowered)
    if "te" not in seen:
        inbound_te = _get_header(headers, "te")
        forwarded.append(("te", inbound_te or "trailers"))
    return forwarded


async def proxy_inbound_cli_run(
    scope: Mapping[str, Any],
    receive: Callable[[], Awaitable[Mapping[str, Any]]],
    send: Callable[[Mapping[str, Any]], Awaitable[None]],
    *,
    session_factory: Optional[Callable[[], _AgentnH2Session]] = None,
    turn_base: str = CURSOR_AGENT_TURN_HOST,
) -> None:
    start_time = datetime.now(timezone.utc)
    call_id = str(uuid.uuid4())
    headers = _asgi_headers(scope)
    http_version = _http_version(scope)
    client_host = _client_host(scope)
    sniffed: Dict[str, str] = {}
    sniff_buffer = bytearray()
    status_code = 500
    error_message: Optional[str] = None
    session: Optional[_AgentnH2Session] = None
    response_started = False

    try:
        if http_version != "2":
            status_code = 505
            error_message = "HTTP/2 required for AgentService/Run"
            await _send_json_error(
                send,
                status_code=505,
                payload={
                    "error": "cursor_agent_cli_inbound_http_version",
                    "reason": "http2_required",
                    "detail": error_message,
                },
            )
            return
        try:
            access_token = require_inbound_cli_bearer(headers)
        except InboundCursorAgentCliAuthError as exc:
            status_code = exc.status_code
            error_message = exc.reason
            await _send_json_error(
                send,
                status_code=exc.status_code,
                payload=inbound_cli_auth_error_payload(exc),
            )
            return

        verbose_proxy_logger.info(
            "cursor_agent_cli_inbound Run http_version=%s client=%s",
            http_version,
            client_host,
        )
        session = session_factory() if session_factory is not None else _AgentnH2Session(turn_base)
        await session.open(_upstream_request_headers(headers, access_token))
        to_agentn: asyncio.Queue[Optional[Tuple[bytes, bool]]] = asyncio.Queue()

        async def pump_client() -> None:
            try:
                while True:
                    message = await receive()
                    message_type = message.get("type")
                    if message_type == "http.disconnect":
                        await to_agentn.put((b"", True))
                        await session.aclose()
                        return
                    if message_type != "http.request":
                        continue
                    body = message.get("body") or b""
                    end_stream = not message.get("more_body", False)
                    if body:
                        if len(sniff_buffer) < 65536:
                            remaining = 65536 - len(sniff_buffer)
                            sniff_buffer.extend(body[:remaining])
                            sniffed.update(_sniff_run_metadata(bytes(sniff_buffer)))
                    await to_agentn.put((bytes(body), end_stream))
                    if end_stream:
                        return
            finally:
                await to_agentn.put(None)

        async def pump_to_agentn() -> None:
            while True:
                item = await to_agentn.get()
                if item is None:
                    return
                data, end_stream = item
                await session.write_request(data, end_stream=end_stream)
                if end_stream:
                    return

        async def pump_upstream() -> None:
            nonlocal response_started, status_code
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", CURSOR_AGENT_CONNECT_CONTENT_TYPE.encode("ascii")),
                        (b"connect-protocol-version", b"1"),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": _connect_header_flush_frame(),
                    "more_body": True,
                }
            )
            response_started = True
            status_code = 200
            try:
                async for chunk in session.iter_response_data():
                    if chunk:
                        verbose_proxy_logger.info(
                            "cursor_agent_cli_inbound forwarded bytes=%s",
                            len(chunk),
                        )
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": True,
                            }
                        )
                    if session.response_status >= 400:
                        status_code = session.response_status
            finally:
                await send({"type": "http.response.body", "body": b"", "more_body": False})

        await asyncio.gather(
            pump_client(),
            pump_to_agentn(),
            pump_upstream(),
            return_exceptions=True,
        )
        if session.response_status >= 400:
            status_code = session.response_status
            error_message = f"upstream_http_{session.response_status}"
    except InboundCursorAgentCliAuthError as exc:
        status_code = exc.status_code
        error_message = exc.reason
        if not response_started:
            await _send_json_error(
                send,
                status_code=exc.status_code,
                payload=inbound_cli_auth_error_payload(exc),
            )
    except CursorConnectError as exc:
        status_code = int(getattr(exc, "status_code", 502) or 502)
        error_message = "upstream_connect_error"
        verbose_proxy_logger.warning("cursor_agent_cli_inbound connect error: %s", exc.message)
        if not response_started:
            await _send_json_error(
                send,
                status_code=status_code,
                payload={
                    "error": "cursor_agent_cli_inbound_upstream",
                    "reason": error_message,
                    "detail": "Cursor Agent CLI inbound egress failed.",
                },
            )
    except Exception as exc:
        status_code = 502
        error_message = "inbound_proxy_error"
        verbose_proxy_logger.warning("cursor_agent_cli_inbound proxy error: %s", exc)
        if not response_started:
            await _send_json_error(
                send,
                status_code=502,
                payload={
                    "error": "cursor_agent_cli_inbound_upstream",
                    "reason": error_message,
                    "detail": "Cursor Agent CLI inbound proxy failed.",
                },
            )
    finally:
        if session is not None:
            await session.aclose()
        await _persist_inbound_cli_turn(
            call_id=call_id,
            headers=headers,
            sniffed=sniffed,
            http_version=http_version,
            client_host=client_host,
            status_code=status_code,
            start_time=start_time,
            error=error_message,
        )



class CursorAgentCliInboundMiddleware:
    """Outermost ASGI intercept for inbound HTTP/2 Connect ``Run``.

    FastAPI ``request_response`` waits for the handler to return before the
    custom Response ASGI cycle starts. Connect bidi does not EndBody until
    response headers arrive, so the FastAPI route deadlocks. This middleware
    bypasses FastAPI for ``POST /agent.v1.AgentService/Run``.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Awaitable[Mapping[str, Any]]],
        send: Callable[[Mapping[str, Any]], Awaitable[None]],
    ) -> None:
        if is_cursor_agent_cli_run_scope(scope):
            await proxy_inbound_cli_run(scope, receive, send)
            return
        await self.app(scope, receive, send)


class CursorAgentCliInboundResponse(Response):
    """Starlette response that takes over the ASGI cycle for bidi Connect."""

    media_type = CURSOR_AGENT_CONNECT_CONTENT_TYPE

    def __init__(
        self,
        *,
        session_factory: Optional[Callable[[], _AgentnH2Session]] = None,
        turn_base: str = CURSOR_AGENT_TURN_HOST,
    ) -> None:
        super().__init__(content=b"", media_type=self.media_type)
        self.session_factory = session_factory
        self.turn_base = turn_base

    async def __call__(self, scope, receive, send) -> None:
        await proxy_inbound_cli_run(
            scope,
            receive,
            send,
            session_factory=self.session_factory,
            turn_base=self.turn_base,
        )


async def cursor_agent_cli_run_endpoint(request: Request) -> Response:
    """FastAPI entry for inbound HTTP/2 Connect ``Run``.

    Does not depend on ``user_api_key_auth``: the CLI sends a Cursor access
    token, not a LiteLLM virtual key.
    """
    return CursorAgentCliInboundResponse()
