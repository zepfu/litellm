"""Named LiteLLM containers start Hypercorn for inbound Cursor Agent CLI HTTP/2."""

from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import h2.config
import h2.connection
import h2.events
import pytest

from litellm.proxy.pass_through_endpoints.cursor_agent_cli_hypercorn import (
    UnsupportedHypercornVersion,
    configure_hypercorn_for_cursor_agent_cli,
    hypercorn_h2_receive_dispatch_guards_installed,
    install_hypercorn_h2_receive_dispatch_guards,
    original_hypercorn_h2_handle,
    original_hypercorn_h2_handle_events,
    require_supported_hypercorn_version,
    restore_hypercorn_h2_receive_dispatch_guards,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_configure_hypercorn_keeps_http1_and_lengthens_keepalive() -> None:
    config = MagicMock()
    config.alpn_protocols = ["h2", "http/1.1"]
    config.keep_alive_timeout = 5
    config.h2_max_inbound_frame_size = 16384
    configure_hypercorn_for_cursor_agent_cli(config)
    assert config.keep_alive_timeout == 600
    assert "h2" in config.alpn_protocols
    assert "http/1.1" in config.alpn_protocols
    assert config.h2_max_inbound_frame_size == 16 * 1024 * 1024


def test_hypercorn_helper_is_used_by_proxy_cli() -> None:
    source = _read("litellm/proxy/proxy_cli.py")
    assert "configure_hypercorn_for_cursor_agent_cli" in source
    assert "cursor_agent_cli_hypercorn" in source
    assert "--run_hypercorn" in source


def test_named_container_start_commands_enable_hypercorn() -> None:
    """Drive the shipped compose/Dockerfiles; do not pin content hashes."""
    compose_dev = _read("docker-compose.dev.yml")
    compose_alpha = _read("docker-compose.alpha.yml")
    dockerfile_dev = _read("Dockerfile.dev")
    dockerfile_alpha = _read("Dockerfile.alpha")
    requirements = _read("requirements.txt")

    assert "container_name: litellm-dev" in compose_dev
    assert "container_name: litellm-alpha" in compose_alpha
    assert "--run_hypercorn" in compose_dev
    assert "--run_hypercorn" in compose_alpha
    assert "--run_hypercorn" in dockerfile_dev
    assert "--run_hypercorn" in dockerfile_alpha
    assert "hypercorn==0.15.0" in requirements


class _LoopExceptionTrap:
    def __init__(self) -> None:
        self.exceptions: List[BaseException] = []

    def handler(self, _loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        if isinstance(exc, BaseException):
            self.exceptions.append(exc)


class _RecordingStream:
    def __init__(self) -> None:
        self.events: List[Any] = []
        self.idle = False
        self.fail_with: Optional[BaseException] = None

    async def handle(self, event: Any) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.events.append(event)


def _new_h2_client() -> h2.connection.H2Connection:
    client = h2.connection.H2Connection(
        config=h2.config.H2Configuration(client_side=True, header_encoding=None)
    )
    client.initiate_connection()
    return client


def _open_run(client: h2.connection.H2Connection, *, end_stream: bool = False) -> bytes:
    stream_id = client.get_next_available_stream_id()
    client.send_headers(
        stream_id,
        [
            (b":method", b"POST"),
            (b":scheme", b"http"),
            (b":authority", b"localhost"),
            (b":path", b"/agent.v1.AgentService/Run"),
            (b"content-type", b"application/connect+proto"),
        ],
        end_stream=False,
    )
    client.send_data(stream_id, b"hello-run", end_stream=end_stream)
    return client.data_to_send()


def _goaway(client: h2.connection.H2Connection) -> bytes:
    client.close_connection()
    return client.data_to_send()


async def _make_h2_protocol(*, install_guards: bool = True):
    from hypercorn.asyncio.task_group import TaskGroup
    from hypercorn.asyncio.worker_context import WorkerContext
    from hypercorn.app_wrappers import ASGIWrapper
    from hypercorn.config import Config
    from hypercorn.protocol.h2 import H2Protocol

    if install_guards:
        install_hypercorn_h2_receive_dispatch_guards()

    sent: List[Any] = []

    async def send(event: Any) -> None:
        sent.append(event)

    async def app(scope, receive, send_app):
        await send_app({"type": "http.response.start", "status": 200, "headers": []})
        await send_app(
            {"type": "http.response.body", "body": b"ok", "more_body": False}
        )

    loop = asyncio.get_running_loop()
    task_group = TaskGroup(loop)
    await task_group.__aenter__()
    protocol = H2Protocol(
        ASGIWrapper(app),
        Config(),
        WorkerContext(),
        task_group,
        False,
        ("127.0.0.1", 9),
        ("127.0.0.1", 4011),
        send,
    )
    await protocol.initiate()
    return protocol, sent, task_group


async def _close_protocol(protocol: Any, task_group: Any) -> None:
    from hypercorn.events import Closed

    protocol.closed = True
    try:
        await protocol.has_data.set()
    except Exception:
        pass
    try:
        await protocol.handle(Closed())
    except Exception:
        pass
    inner = getattr(task_group, "_task_group", None)
    tasks = list(getattr(inner, "_tasks", ())) if inner is not None else []
    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        await asyncio.wait_for(task_group.__aexit__(None, None, None), timeout=1)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
        try:
            await task_group.__aexit__(None, None, None)
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_hypercorn_h2_goaway_does_not_keyerror_retired_stream() -> None:
    import importlib.metadata

    from hypercorn.events import Closed, RawData
    from hypercorn.protocol.h2 import H2Protocol

    assert importlib.metadata.version("hypercorn").startswith("0.15.")
    protocol, sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        assert getattr(H2Protocol, "_aawm_cursor_h2_guards", False) is True
        client = _new_h2_client()
        await protocol.handle(RawData(data=_open_run(client)))
        trap = _LoopExceptionTrap()
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(trap.handler)
        try:
            await protocol.handle(RawData(data=_goaway(client)))
            await protocol.handle(RawData(data=b"ignored-after-close"))
        finally:
            loop.set_exception_handler(previous)
        assert protocol.closed is True
        assert 1 not in protocol.streams
        assert any(isinstance(event, Closed) for event in sent)
        assert trap.exceptions == []
    finally:
        await _close_protocol(protocol, task_group)


@pytest.mark.asyncio
async def test_hypercorn_h2_end_stream_then_goaway_does_not_keyerror() -> None:
    from hypercorn.events import RawData

    protocol, _sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        client = _new_h2_client()
        first = _open_run(client, end_stream=True)
        rest = _goaway(client)
        await protocol.handle(RawData(data=first + rest))
        assert protocol.closed is True
        assert 1 not in protocol.streams
    finally:
        await _close_protocol(protocol, task_group)


@pytest.mark.asyncio
async def test_hypercorn_h2_stale_data_after_close_does_not_keyerror() -> None:
    from hypercorn.events import RawData
    from hypercorn.protocol.events import Body, EndBody

    protocol, _sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        client = _new_h2_client()
        await protocol.handle(RawData(data=_open_run(client)))
        recording = _RecordingStream()
        protocol.streams[1] = recording
        data_event = h2.events.DataReceived(
            stream_id=1,
            data=b"tail",
            flow_controlled_length=4,
        )
        end_event = h2.events.StreamEnded(stream_id=1)
        goaway = h2.events.ConnectionTerminated()
        goaway.error_code = 0
        goaway.last_stream_id = 0
        await protocol._handle_events([goaway, data_event, end_event])
        assert protocol.closed is True
        assert 1 not in protocol.streams
        assert not any(isinstance(event, (Body, EndBody)) for event in recording.events)
    finally:
        await _close_protocol(protocol, task_group)


@pytest.mark.asyncio
async def test_hypercorn_h2_live_stream_keyerror_still_surfaces() -> None:
    from hypercorn.events import RawData

    protocol, _sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        client = _new_h2_client()
        await protocol.handle(RawData(data=_open_run(client)))
        recording = _RecordingStream()
        recording.fail_with = KeyError("live-handler")
        protocol.streams[1] = recording
        data_event = h2.events.DataReceived(
            stream_id=1,
            data=b"live",
            flow_controlled_length=4,
        )
        with pytest.raises(KeyError, match="live-handler"):
            await protocol._handle_events([data_event])
    finally:
        await _close_protocol(protocol, task_group)


@pytest.mark.asyncio
async def test_stock_hypercorn_h2_goaway_then_stale_data_keyerrors() -> None:
    """Unpatched 0.15 dispatcher is the live fingerprint; keep that as red proof."""
    from hypercorn.events import RawData

    original = original_hypercorn_h2_handle_events()
    protocol, _sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        if original is None:
            pytest.skip("original Hypercorn dispatcher was not captured")
        client = _new_h2_client()
        await protocol.handle(RawData(data=_open_run(client)))
        data_event = h2.events.DataReceived(
            stream_id=1,
            data=b"stale",
            flow_controlled_length=5,
        )
        goaway = h2.events.ConnectionTerminated()
        goaway.error_code = 0
        goaway.last_stream_id = 0
        protocol.streams.pop(1, None)
        with pytest.raises(KeyError):
            await original(protocol, [goaway, data_event])
    finally:
        await _close_protocol(protocol, task_group)


def test_configure_hypercorn_installs_receive_dispatch_guards() -> None:
    from hypercorn.protocol.h2 import H2Protocol

    config = MagicMock()
    config.alpn_protocols = ["h2", "http/1.1"]
    configure_hypercorn_for_cursor_agent_cli(config)
    assert hypercorn_h2_receive_dispatch_guards_installed() is True
    assert getattr(H2Protocol, "_aawm_cursor_h2_guards", False) is True
    assert str(getattr(H2Protocol, "_aawm_hypercorn_version", "")).startswith("0.15.")
    assert importlib.metadata.version("hypercorn").startswith("0.15.")
    first_handle = H2Protocol.handle
    captured = original_hypercorn_h2_handle()
    install_hypercorn_h2_receive_dispatch_guards()
    assert H2Protocol.handle is first_handle
    assert original_hypercorn_h2_handle() is captured


def test_unsupported_hypercorn_version_leaves_methods_untouched() -> None:
    from hypercorn.protocol.h2 import H2Protocol

    restore_hypercorn_h2_receive_dispatch_guards()
    handle_before = H2Protocol.handle
    events_before = H2Protocol._handle_events
    close_before = H2Protocol._close_stream
    with pytest.raises(UnsupportedHypercornVersion):
        require_supported_hypercorn_version("0.18.0")
    with pytest.raises(UnsupportedHypercornVersion):
        from unittest.mock import patch as _patch

        with _patch(
            "litellm.proxy.pass_through_endpoints.cursor_agent_cli_hypercorn.importlib.metadata.version",
            return_value="0.18.0",
        ):
            install_hypercorn_h2_receive_dispatch_guards()
    assert H2Protocol.handle is handle_before
    assert H2Protocol._handle_events is events_before
    assert H2Protocol._close_stream is close_before
    assert hypercorn_h2_receive_dispatch_guards_installed() is False
    install_hypercorn_h2_receive_dispatch_guards()


@pytest.mark.asyncio
async def test_hypercorn_h2_coalesced_request_goaway_during_worker_termination() -> None:
    from hypercorn.events import Closed, RawData

    protocol, sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        await protocol.context.terminated.set()
        client = _new_h2_client()
        first = _open_run(client, end_stream=True)
        rest = _goaway(client)
        await protocol.handle(RawData(data=first + rest))
        assert protocol.closed is True
        assert 1 not in protocol.streams
        assert any(isinstance(event, Closed) for event in sent)
    finally:
        await _close_protocol(protocol, task_group)


@pytest.mark.asyncio
async def test_hypercorn_h2_goaway_releases_blocked_response_buffer() -> None:
    from hypercorn.events import RawData
    from hypercorn.protocol.h2 import BUFFER_HIGH_WATER, StreamBuffer

    protocol, _sent, task_group = await _make_h2_protocol(install_guards=True)
    try:
        client = _new_h2_client()
        await protocol.handle(RawData(data=client.data_to_send()))
        buffer = StreamBuffer(protocol.context.event_class)
        protocol.stream_buffers[1] = buffer
        blocked = asyncio.create_task(buffer.push(b"x" * int(BUFFER_HIGH_WATER)))
        await asyncio.sleep(0)
        assert blocked.done() is False
        await protocol.handle(RawData(data=_goaway(client)))
        await asyncio.wait_for(blocked, timeout=1)
        assert protocol.closed is True
        assert 1 not in protocol.stream_buffers
    finally:
        await _close_protocol(protocol, task_group)
