"""Hypercorn HTTP/2 settings for inbound Cursor Agent CLI Connect."""

from __future__ import annotations

import asyncio
import importlib.metadata
from typing import Any, List, Optional

CURSOR_AGENT_CLI_HTTP2_KEEPALIVE_SECONDS = 600
CURSOR_AGENT_CLI_H2_MAX_INBOUND_FRAME_SIZE = 16 * 1024 * 1024
SUPPORTED_HYPERCORN_VERSION_PREFIX = "0.15."

_H2_RECEIVE_DISPATCH_GUARDS_INSTALLED = False
_ORIGINAL_H2_HANDLE: Any = None
_ORIGINAL_H2_HANDLE_EVENTS: Any = None
_ORIGINAL_H2_CLOSE_STREAM: Any = None
_ORIGINAL_H2_SEND_DATA: Any = None
_ORIGINAL_HTTP_HANDLE: Any = None
_ORIGINAL_WS_HANDLE_EVENTS: Any = None


class UnsupportedHypercornVersion(RuntimeError):
    """Cursor inbound HTTP/2 guards require the pinned Hypercorn 0.15 line."""


def hypercorn_h2_receive_dispatch_guards_installed() -> bool:
    return _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED


def original_hypercorn_h2_handle_events() -> Any:
    return _ORIGINAL_H2_HANDLE_EVENTS


def original_hypercorn_h2_handle() -> Any:
    return _ORIGINAL_H2_HANDLE


def require_supported_hypercorn_version(version: Optional[str] = None) -> str:
    installed = version if version is not None else importlib.metadata.version("hypercorn")
    if not str(installed).startswith(SUPPORTED_HYPERCORN_VERSION_PREFIX):
        raise UnsupportedHypercornVersion(
            "Cursor Agent CLI inbound requires hypercorn "
            f"{SUPPORTED_HYPERCORN_VERSION_PREFIX}x; found {installed}."
        )
    return str(installed)


def restore_hypercorn_h2_receive_dispatch_guards() -> None:
    """Restore stock Hypercorn H2Protocol methods. Tests only."""
    global _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED
    global _ORIGINAL_H2_HANDLE, _ORIGINAL_H2_HANDLE_EVENTS
    global _ORIGINAL_H2_CLOSE_STREAM, _ORIGINAL_H2_SEND_DATA
    global _ORIGINAL_HTTP_HANDLE, _ORIGINAL_WS_HANDLE_EVENTS
    from hypercorn.protocol.h2 import H2Protocol
    from hypercorn.protocol.http_stream import HTTPStream
    from hypercorn.protocol.ws_stream import WSStream

    if _ORIGINAL_H2_HANDLE is not None:
        H2Protocol.handle = _ORIGINAL_H2_HANDLE
    if _ORIGINAL_H2_HANDLE_EVENTS is not None:
        H2Protocol._handle_events = _ORIGINAL_H2_HANDLE_EVENTS
    if _ORIGINAL_H2_CLOSE_STREAM is not None:
        H2Protocol._close_stream = _ORIGINAL_H2_CLOSE_STREAM
    if _ORIGINAL_H2_SEND_DATA is not None:
        H2Protocol._send_data = _ORIGINAL_H2_SEND_DATA
    if _ORIGINAL_HTTP_HANDLE is not None:
        HTTPStream.handle = _ORIGINAL_HTTP_HANDLE
    if _ORIGINAL_WS_HANDLE_EVENTS is not None:
        WSStream._handle_events = _ORIGINAL_WS_HANDLE_EVENTS
    if hasattr(H2Protocol, "_aawm_cursor_h2_guards"):
        delattr(H2Protocol, "_aawm_cursor_h2_guards")
    if hasattr(H2Protocol, "_aawm_hypercorn_version"):
        delattr(H2Protocol, "_aawm_hypercorn_version")
    _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED = False
    _ORIGINAL_H2_HANDLE = None
    _ORIGINAL_H2_HANDLE_EVENTS = None
    _ORIGINAL_H2_CLOSE_STREAM = None
    _ORIGINAL_H2_SEND_DATA = None
    _ORIGINAL_HTTP_HANDLE = None
    _ORIGINAL_WS_HANDLE_EVENTS = None


def install_hypercorn_h2_receive_dispatch_guards() -> None:  # noqa: PLR0915
    """Guard Hypercorn 0.15 receive dispatch against retired-stream KeyError.

    Alpha pins ``hypercorn==0.15.0``. That dispatcher indexes
    ``streams[event.stream_id]`` for ``DataReceived`` / ``StreamEnded`` and
    continues the event batch after ``ConnectionTerminated``. A client GOAWAY
    or stream close then raises ``KeyError: 1`` from
    ``TCPServer._read_data`` as an unhandled ``client_connected_cb`` error.

    Do not wrap ``TCPServer.run`` in ``except KeyError``. A KeyError from a
    live stream handler must still propagate.
    """
    global _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED
    global _ORIGINAL_H2_HANDLE, _ORIGINAL_H2_HANDLE_EVENTS
    global _ORIGINAL_H2_CLOSE_STREAM, _ORIGINAL_H2_SEND_DATA
    global _ORIGINAL_HTTP_HANDLE, _ORIGINAL_WS_HANDLE_EVENTS
    if _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED:
        return

    version = require_supported_hypercorn_version()

    import h2.events
    import h2.exceptions
    import h2.settings
    import priority as priority_lib

    from hypercorn.events import Closed, RawData, Updated
    from hypercorn.protocol.events import Body, EndBody, StreamClosed
    from hypercorn.protocol.h2 import H2Protocol
    from hypercorn.protocol.http_stream import HTTPStream
    from hypercorn.protocol.ws_stream import WSStream

    _ORIGINAL_H2_HANDLE = H2Protocol.handle
    _ORIGINAL_H2_HANDLE_EVENTS = H2Protocol._handle_events
    _ORIGINAL_H2_CLOSE_STREAM = H2Protocol._close_stream
    _ORIGINAL_H2_SEND_DATA = H2Protocol._send_data
    _ORIGINAL_HTTP_HANDLE = HTTPStream.handle
    _ORIGINAL_WS_HANDLE_EVENTS = WSStream._handle_events

    def _retired_ids(protocol: Any) -> set:
        retired = getattr(protocol, "_aawm_retired_stream_ids", None)
        if retired is None:
            retired = set()
            protocol._aawm_retired_stream_ids = retired
        return retired

    def _stream_known_retired(protocol: Any, stream_id: int) -> bool:
        return bool(
            getattr(protocol, "closed", False) or stream_id in _retired_ids(protocol)
        )

    def _stream_put_lock(stream: Any) -> asyncio.Lock:
        lock = getattr(stream, "_aawm_put_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            stream._aawm_put_lock = lock
        return lock

    def _stream_space_event(stream: Any) -> asyncio.Event:
        event = getattr(stream, "_aawm_queue_space", None)
        if event is None:
            event = asyncio.Event()
            stream._aawm_queue_space = event
        return event

    def _app_queue(stream: Any) -> Optional[asyncio.Queue]:
        app_put = getattr(stream, "app_put", None)
        if app_put is not None:
            queue = getattr(app_put, "_aawm_queue", None)
            if isinstance(queue, asyncio.Queue):
                return queue
            queue = getattr(app_put, "__self__", None)
            if isinstance(queue, asyncio.Queue):
                return queue
        bound = getattr(stream, "_aawm_app_queue", None)
        if isinstance(bound, asyncio.Queue):
            return bound
        return None

    def _notify_queue_space(queue: asyncio.Queue) -> None:
        for owner in list(getattr(queue, "_aawm_owner_streams", set())):
            space = getattr(owner, "_aawm_queue_space", None)
            if space is not None:
                space.set()

    def _bind_queue_to_stream(stream: Any, queue: asyncio.Queue) -> None:
        owners = getattr(queue, "_aawm_owner_streams", None)
        if owners is None:
            owners = set()
            queue._aawm_owner_streams = owners
        owners.add(stream)
        stream._aawm_app_queue = queue
        if getattr(queue, "_aawm_get_wrapped", False):
            return
        original_get = queue.get
        original_get_nowait = queue.get_nowait

        async def _wrapped_get():
            item = await original_get()
            _notify_queue_space(queue)
            return item

        def _wrapped_get_nowait():
            item = original_get_nowait()
            _notify_queue_space(queue)
            return item

        queue.get = _wrapped_get
        queue.get_nowait = _wrapped_get_nowait
        queue._aawm_get_wrapped = True

    async def _put_app_event(stream: Any, event: dict) -> None:
        """Queue an ASGI receive event without blocking the connection reader.

        If the application queue is full, wait on a per-stream space event
        that close() can set. A retired stream discards the event instead of
        cancelling the shared reader.
        """
        if getattr(stream, "closed", False):
            return
        app_put = getattr(stream, "app_put", None)
        if app_put is None:
            return
        queue = _app_queue(stream)
        if isinstance(queue, asyncio.Queue):
            _bind_queue_to_stream(stream, queue)
            _stream_space_event(stream)
        while not getattr(stream, "closed", False):
            async with _stream_put_lock(stream):
                if getattr(stream, "closed", False):
                    return
                if queue is None:
                    await app_put(event)
                    return
                try:
                    queue.put_nowait(event)
                    return
                except asyncio.QueueFull:
                    space = _stream_space_event(stream)
                    space.clear()
                    if queue.maxsize <= 0 or queue.qsize() < queue.maxsize:
                        continue
            await space.wait()

    async def _http_handle(self, event: Any) -> None:
        if getattr(self, "closed", False) and not isinstance(event, StreamClosed):
            return
        if isinstance(event, Body) or isinstance(event, EndBody):
            queue = _app_queue(self)
            if queue is not None:
                _bind_queue_to_stream(self, queue)
        if isinstance(event, Body):
            await _put_app_event(
                self,
                {
                    "type": "http.request",
                    "body": bytes(event.data),
                    "more_body": True,
                },
            )
            return
        if isinstance(event, EndBody):
            await _put_app_event(
                self,
                {"type": "http.request", "body": b"", "more_body": False},
            )
            return
        await _ORIGINAL_HTTP_HANDLE(self, event)
        queue = _app_queue(self)
        if queue is not None:
            _bind_queue_to_stream(self, queue)

    async def _ws_handle_events(self) -> None:
        from wsproto.connection import ConnectionState
        from wsproto.events import CloseConnection, Message, Ping
        from wsproto.frame_protocol import CloseReason

        from hypercorn.protocol.events import StreamClosed as H2StreamClosed
        from hypercorn.protocol.ws_stream import FrameTooLargeError

        for event in self.connection.events():
            if isinstance(event, Message):
                try:
                    self.buffer.extend(event)
                except FrameTooLargeError:
                    await self._send_wsproto_event(
                        CloseConnection(code=CloseReason.MESSAGE_TOO_BIG)
                    )
                    break
                if event.message_finished:
                    message = self.buffer.to_message()
                    self.buffer.clear()
                    await _put_app_event(self, message)
            elif isinstance(event, Ping):
                await self._send_wsproto_event(event.response())
            elif isinstance(event, CloseConnection):
                if self.connection.state == ConnectionState.REMOTE_CLOSING:
                    await self._send_wsproto_event(event.response())
                await self.send(H2StreamClosed(stream_id=self.stream_id))

    async def _release_stream_buffers(self) -> None:
        buffers = getattr(self, "stream_buffers", None)
        if not isinstance(buffers, dict):
            return
        for _stream_id, buffer in list(buffers.items()):
            close = getattr(buffer, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
        has_data = getattr(self, "has_data", None)
        if has_data is not None:
            await has_data.set()

    def _disconnect_event_for_stream(stream: Any) -> dict:
        scope = getattr(stream, "scope", None)
        is_websocket = type(stream).__name__ == "WSStream" or (
            isinstance(scope, dict) and scope.get("type") == "websocket"
        )
        if not is_websocket:
            return {"type": "http.disconnect"}
        state = getattr(stream, "state", None)
        state_name = getattr(state, "name", "") or str(state)
        code = 1000 if state_name in {"CLOSED", "HTTPCLOSED"} else 1006
        return {"type": "websocket.disconnect", "code": code}

    def _deliver_terminal_disconnect(stream: Any) -> None:
        """Enqueue stream-specific disconnect without waiting for the app.

        Hypercorn 0.15 HTTPStream/WSStream handle(StreamClosed) awaits
        Queue.put. A full request queue then deadlocks GOAWAY cleanup, and
        a disconnect already waiting on that put is not visible in
        ``self.streams``. HTTP streams get ``http.disconnect``; HTTP/2
        CONNECT WebSocket streams keep ``websocket.disconnect`` plus the
        stock close code.
        """
        app_put = getattr(stream, "app_put", None)
        if app_put is None:
            return
        event = _disconnect_event_for_stream(stream)
        queue = getattr(app_put, "__self__", None)
        if isinstance(queue, asyncio.Queue):
            try:
                queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                    return
                except asyncio.QueueFull:
                    pass
        put_nowait = getattr(app_put, "put_nowait", None)
        if callable(put_nowait):
            try:
                put_nowait(event)
            except Exception:
                pass

    async def _enter_connection_close(self, *, notify_transport: bool) -> None:
        already_closed = bool(getattr(self, "closed", False))
        self.closed = True
        # Unblock response push/drain waiters before HTTPStream disconnect
        # delivery, which can itself wait on a full application queue.
        await _release_stream_buffers(self)
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            await _close_stream_terminal(self, stream_id)
        if notify_transport and not already_closed:
            await self.send(Closed())

    async def _close_stream(self, stream_id: int) -> None:
        # Ordinary StreamClosed must not await a full application queue.
        # Response draining and END_STREAM stay in _send_data.
        await _close_stream_terminal(self, stream_id)

    async def _close_stream_terminal(self, stream_id: int) -> None:
        _retired_ids(self).add(stream_id)
        stream = self.streams.pop(stream_id, None)
        if stream is None:
            await self.has_data.set()
            return
        stream.closed = True
        _stream_space_event(stream).set()
        _deliver_terminal_disconnect(stream)
        await self.has_data.set()

    def _drop_priority(self, stream_id: int) -> None:
        try:
            self.priority.remove_stream(stream_id)
        except (priority_lib.MissingStreamError, KeyError):
            pass

    async def _send_data(self, stream_id: int) -> None:
        try:
            buffer = self.stream_buffers.get(stream_id)
            if buffer is None or getattr(self, "closed", False):
                if buffer is not None:
                    close = getattr(buffer, "close", None)
                    if callable(close):
                        try:
                            await close()
                        except Exception:
                            pass
                _drop_priority(self, stream_id)
                return
            chunk_size = min(
                self.connection.local_flow_control_window(stream_id),
                self.connection.max_outbound_frame_size,
            )
            chunk_size = max(0, chunk_size)
            data = await buffer.pop(chunk_size)
            if data:
                self.connection.send_data(stream_id, data)
                await self._flush()
            else:
                self.priority.block(stream_id)
            buffer = self.stream_buffers.get(stream_id)
            if buffer is None:
                _drop_priority(self, stream_id)
                return
            if buffer.complete:
                if not getattr(self, "closed", False):
                    self.connection.end_stream(stream_id)
                    await self._flush()
                    self.stream_buffers.pop(stream_id, None)
                    _drop_priority(self, stream_id)
        except (h2.exceptions.StreamClosedError, KeyError, h2.exceptions.ProtocolError):
            buffer = self.stream_buffers.pop(stream_id, None)
            if buffer is not None:
                close = getattr(buffer, "close", None)
                if callable(close):
                    try:
                        await close()
                    except Exception:
                        pass
            _drop_priority(self, stream_id)

    def _batch_is_connection_terminal(events: List[Any]) -> bool:
        return any(isinstance(event, h2.events.ConnectionTerminated) for event in events)

    async def _handle(self, event: Any) -> None:
        if isinstance(event, RawData):
            if getattr(self, "closed", False):
                return
            try:
                events = self.connection.receive_data(event.data)
            except h2.exceptions.ProtocolError:
                await _enter_connection_close(self, notify_transport=True)
                return
            if _batch_is_connection_terminal(events):
                await _enter_connection_close(self, notify_transport=True)
                return
            await self._handle_events(events)
            return
        if isinstance(event, Closed):
            await _enter_connection_close(self, notify_transport=False)
            return
        await _ORIGINAL_H2_HANDLE(self, event)

    async def _handle_events(self, events: List[Any]) -> None:
        if _batch_is_connection_terminal(events):
            await _enter_connection_close(self, notify_transport=True)
            return
        for event in events:
            if getattr(self, "closed", False):
                return
            if isinstance(event, h2.events.RequestReceived):
                if self.context.terminated.is_set():
                    if getattr(self, "closed", False):
                        return
                    self.connection.reset_stream(event.stream_id)
                    self.connection.update_settings(
                        {h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 0}
                    )
                else:
                    await self._create_stream(event)
                    await self.send(Updated(idle=False))
            elif isinstance(event, h2.events.DataReceived):
                stream = self.streams.get(event.stream_id)
                if stream is None:
                    if _stream_known_retired(self, event.stream_id):
                        if not getattr(self, "closed", False):
                            try:
                                self.connection.acknowledge_received_data(
                                    event.flow_controlled_length,
                                    event.stream_id,
                                )
                            except (
                                h2.exceptions.ProtocolError,
                                h2.exceptions.StreamClosedError,
                            ):
                                pass
                        continue
                    raise KeyError(event.stream_id)
                await stream.handle(Body(stream_id=event.stream_id, data=event.data))
                if not getattr(self, "closed", False):
                    try:
                        self.connection.acknowledge_received_data(
                            event.flow_controlled_length,
                            event.stream_id,
                        )
                    except (
                        h2.exceptions.ProtocolError,
                        h2.exceptions.StreamClosedError,
                    ):
                        pass
            elif isinstance(event, h2.events.StreamEnded):
                stream = self.streams.get(event.stream_id)
                if stream is None:
                    if _stream_known_retired(self, event.stream_id):
                        continue
                    raise KeyError(event.stream_id)
                await stream.handle(EndBody(stream_id=event.stream_id))
            elif isinstance(event, h2.events.StreamReset):
                await self._close_stream(event.stream_id)
                if not getattr(self, "closed", False):
                    await self._window_updated(event.stream_id)
            elif isinstance(event, h2.events.WindowUpdated):
                await self._window_updated(event.stream_id)
            elif isinstance(event, h2.events.PriorityUpdated):
                await self._priority_updated(event)
            elif isinstance(event, h2.events.RemoteSettingsChanged):
                if (
                    h2.settings.SettingCodes.INITIAL_WINDOW_SIZE
                    in event.changed_settings
                ):
                    await self._window_updated(None)
            elif isinstance(event, h2.events.ConnectionTerminated):
                await _enter_connection_close(self, notify_transport=True)
                return
        if not getattr(self, "closed", False):
            await self._flush()

    H2Protocol.handle = _handle
    H2Protocol._handle_events = _handle_events
    H2Protocol._close_stream = _close_stream
    H2Protocol._send_data = _send_data
    HTTPStream.handle = _http_handle
    WSStream._handle_events = _ws_handle_events
    H2Protocol._aawm_cursor_h2_guards = True
    H2Protocol._aawm_hypercorn_version = version
    _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED = True


def configure_hypercorn_for_cursor_agent_cli(config: Any) -> Any:
    """Keep HTTP/1.1 available while enabling HTTP/2 bidi for Agent CLI turns.

    TLS uses ALPN ``h2`` / ``http/1.1``. Cleartext ``http://`` listeners start
    as HTTP/1.1 and Hypercorn upgrades prior-knowledge ``PRI *`` to HTTP/2,
    which is the default Cursor Agent CLI ``Run`` path on ``--agent-endpoint``.
    """
    config.keep_alive_timeout = float(CURSOR_AGENT_CLI_HTTP2_KEEPALIVE_SECONDS)
    config.h2_max_inbound_frame_size = int(CURSOR_AGENT_CLI_H2_MAX_INBOUND_FRAME_SIZE)
    alpn = list(getattr(config, "alpn_protocols", []) or [])
    if "h2" not in alpn:
        alpn.insert(0, "h2")
    if "http/1.1" not in alpn:
        alpn.append("http/1.1")
    config.alpn_protocols = alpn
    install_hypercorn_h2_receive_dispatch_guards()
    return config
