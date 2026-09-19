"""Hypercorn HTTP/2 settings for inbound Cursor Agent CLI Connect."""

from __future__ import annotations

import importlib.metadata
from typing import Any, List, Optional

CURSOR_AGENT_CLI_HTTP2_KEEPALIVE_SECONDS = 600
CURSOR_AGENT_CLI_H2_MAX_INBOUND_FRAME_SIZE = 16 * 1024 * 1024
SUPPORTED_HYPERCORN_VERSION_PREFIX = "0.15."

_H2_RECEIVE_DISPATCH_GUARDS_INSTALLED = False
_ORIGINAL_H2_HANDLE: Any = None
_ORIGINAL_H2_HANDLE_EVENTS: Any = None
_ORIGINAL_H2_CLOSE_STREAM: Any = None


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
    global _ORIGINAL_H2_HANDLE, _ORIGINAL_H2_HANDLE_EVENTS, _ORIGINAL_H2_CLOSE_STREAM
    from hypercorn.protocol.h2 import H2Protocol

    if _ORIGINAL_H2_HANDLE is not None:
        H2Protocol.handle = _ORIGINAL_H2_HANDLE
    if _ORIGINAL_H2_HANDLE_EVENTS is not None:
        H2Protocol._handle_events = _ORIGINAL_H2_HANDLE_EVENTS
    if _ORIGINAL_H2_CLOSE_STREAM is not None:
        H2Protocol._close_stream = _ORIGINAL_H2_CLOSE_STREAM
    if hasattr(H2Protocol, "_aawm_cursor_h2_guards"):
        delattr(H2Protocol, "_aawm_cursor_h2_guards")
    if hasattr(H2Protocol, "_aawm_hypercorn_version"):
        delattr(H2Protocol, "_aawm_hypercorn_version")
    _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED = False
    _ORIGINAL_H2_HANDLE = None
    _ORIGINAL_H2_HANDLE_EVENTS = None
    _ORIGINAL_H2_CLOSE_STREAM = None


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
    global _ORIGINAL_H2_HANDLE, _ORIGINAL_H2_HANDLE_EVENTS, _ORIGINAL_H2_CLOSE_STREAM
    if _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED:
        return

    version = require_supported_hypercorn_version()

    import h2.events
    import h2.exceptions
    import h2.settings

    from hypercorn.events import Closed, RawData, Updated
    from hypercorn.protocol.events import Body, EndBody
    from hypercorn.protocol.h2 import H2Protocol

    _ORIGINAL_H2_HANDLE = H2Protocol.handle
    _ORIGINAL_H2_HANDLE_EVENTS = H2Protocol._handle_events
    _ORIGINAL_H2_CLOSE_STREAM = H2Protocol._close_stream

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

    async def _release_stream_buffers(self) -> None:
        buffers = getattr(self, "stream_buffers", None)
        if not isinstance(buffers, dict):
            return
        for stream_id, buffer in list(buffers.items()):
            close = getattr(buffer, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
            buffers.pop(stream_id, None)
        has_data = getattr(self, "has_data", None)
        if has_data is not None:
            await has_data.set()

    async def _enter_connection_close(self, *, notify_transport: bool) -> None:
        already_closed = bool(getattr(self, "closed", False))
        self.closed = True
        # Unblock response push/drain waiters before HTTPStream disconnect
        # delivery, which can itself wait on a full application queue.
        await _release_stream_buffers(self)
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            await self._close_stream(stream_id)
        if notify_transport and not already_closed:
            await self.send(Closed())

    async def _close_stream(self, stream_id: int) -> None:
        _retired_ids(self).add(stream_id)
        buffer = getattr(self, "stream_buffers", {}).get(stream_id)
        await _ORIGINAL_H2_CLOSE_STREAM(self, stream_id)
        if buffer is not None:
            close = getattr(buffer, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
            getattr(self, "stream_buffers", {}).pop(stream_id, None)

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
                    self.connection.acknowledge_received_data(
                        event.flow_controlled_length,
                        event.stream_id,
                    )
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
