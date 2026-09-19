"""Hypercorn HTTP/2 settings for inbound Cursor Agent CLI Connect."""

from __future__ import annotations

import importlib.metadata
from typing import Any, List

CURSOR_AGENT_CLI_HTTP2_KEEPALIVE_SECONDS = 600
CURSOR_AGENT_CLI_H2_MAX_INBOUND_FRAME_SIZE = 16 * 1024 * 1024

_H2_RECEIVE_DISPATCH_GUARDS_INSTALLED = False
_ORIGINAL_H2_HANDLE: Any = None
_ORIGINAL_H2_HANDLE_EVENTS: Any = None
_ORIGINAL_H2_CLOSE_STREAM: Any = None


def hypercorn_h2_receive_dispatch_guards_installed() -> bool:
    return _H2_RECEIVE_DISPATCH_GUARDS_INSTALLED


def original_hypercorn_h2_handle_events() -> Any:
    return _ORIGINAL_H2_HANDLE_EVENTS


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

    async def _close_stream(self, stream_id: int) -> None:
        _retired_ids(self).add(stream_id)
        await _ORIGINAL_H2_CLOSE_STREAM(self, stream_id)

    async def _handle(self, event: Any) -> None:
        if isinstance(event, RawData):
            if getattr(self, "closed", False):
                return
            try:
                events = self.connection.receive_data(event.data)
            except h2.exceptions.ProtocolError:
                await self._flush()
                await self.send(Closed())
            else:
                await self._handle_events(events)
            return
        if isinstance(event, Closed):
            self.closed = True
            stream_ids = list(self.streams.keys())
            for stream_id in stream_ids:
                await self._close_stream(stream_id)
            await self.has_data.set()
            return
        await _ORIGINAL_H2_HANDLE(self, event)

    async def _handle_events(self, events: List[Any]) -> None:
        for event in events:
            if getattr(self, "closed", False):
                return
            if isinstance(event, h2.events.RequestReceived):
                if self.context.terminated.is_set():
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
                self.closed = True
                stream_ids = list(self.streams.keys())
                for stream_id in stream_ids:
                    await self._close_stream(stream_id)
                await self.has_data.set()
                await self.send(Closed())
                return
        if not getattr(self, "closed", False):
            await self._flush()

    H2Protocol.handle = _handle
    H2Protocol._handle_events = _handle_events
    H2Protocol._close_stream = _close_stream
    H2Protocol._aawm_cursor_h2_guards = True
    H2Protocol._aawm_hypercorn_version = importlib.metadata.version("hypercorn")
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
