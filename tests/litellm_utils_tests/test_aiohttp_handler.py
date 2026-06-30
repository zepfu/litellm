import asyncio
import os
import sys

import aiohttp

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # Adds the parent directory to the system-path
import pytest

from litellm.llms.custom_httpx.aiohttp_transport import (
    get_litellm_aiohttp_session_attribution,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


def _assert_session_attribution_shape(attribution: dict) -> None:
    assert set(attribution.keys()) == {
        "owner_kind",
        "creation_site",
        "pid",
        "container_hostname",
        "event_loop_id",
        "session_id",
        "litellm_owns_session",
    }
    assert attribution["owner_kind"] in {"custom_httpx", "custom_httpx_transport"}
    assert isinstance(attribution["owner_kind"], str)
    assert attribution["pid"] > 0
    assert isinstance(attribution["session_id"], int)


@pytest.mark.asyncio
async def test_shared_aiohttp_session_receives_attribution():
    session = aiohttp.ClientSession()
    try:
        transport = AsyncHTTPHandler._create_aiohttp_transport(shared_session=session)
        attribution = get_litellm_aiohttp_session_attribution(session)

        assert attribution is not None
        _assert_session_attribution_shape(attribution)
        assert attribution["owner_kind"] == "custom_httpx"
        assert (
            attribution["creation_site"]
            == "AsyncHTTPHandler._create_aiohttp_transport:shared_session"
        )
        assert attribution["litellm_owns_session"] is False
        assert attribution["session_id"] == id(session)
        assert attribution["event_loop_id"] == id(asyncio.get_running_loop())
        assert attribution["pid"] == os.getpid()
    finally:
        await transport.aclose()
        await session.close()


@pytest.mark.asyncio
async def test_owned_aiohttp_transport_session_receives_attribution():
    transport = AsyncHTTPHandler._create_aiohttp_transport()
    session = transport._get_valid_client_session()  # type: ignore
    attribution = get_litellm_aiohttp_session_attribution(session)

    assert attribution is not None
    _assert_session_attribution_shape(attribution)
    assert attribution["owner_kind"] == "custom_httpx_transport"
    assert (
        attribution["creation_site"]
        == "LiteLLMAiohttpTransport._create_replacement_client_session"
    )
    assert attribution["litellm_owns_session"] is True
    assert attribution["session_id"] == id(session)
    assert attribution["event_loop_id"] == id(asyncio.get_running_loop())
    assert attribution["pid"] == os.getpid()

    await transport.aclose()


@pytest.mark.asyncio
async def test_client_session_helper():
    """Test that the client session helper handles event loop changes correctly"""
    transport = AsyncHTTPHandler._create_aiohttp_transport()

    assert transport is not None
    assert hasattr(transport, "_get_valid_client_session")
    session1 = transport._get_valid_client_session()  # type: ignore
    session2 = transport._get_valid_client_session()  # type: ignore

    assert session1 is not None
    assert session1 is session2

    await transport.aclose()


@pytest.mark.asyncio
async def test_event_loop_robustness():
    """Test behavior when event loops change (simulating CI/CD scenario)"""
    transport = AsyncHTTPHandler._create_aiohttp_transport()

    assert transport is not None
    assert hasattr(transport, "_get_valid_client_session")
    session = transport._get_valid_client_session()  # type: ignore
    assert session is not None

    transport.client = lambda: aiohttp.ClientSession()  # type: ignore
    session2 = transport._get_valid_client_session()  # type: ignore
    assert session2 is not None

    await transport.aclose()


@pytest.mark.asyncio
async def test_httpx_request_simulation():
    """Test that the transport can handle a simulated HTTP request"""
    transport = AsyncHTTPHandler._create_aiohttp_transport()

    assert transport is not None
    assert hasattr(transport, "_get_valid_client_session")
    session = transport._get_valid_client_session()  # type: ignore

    assert session is not None
    assert hasattr(session, "request")

    await transport.aclose()
