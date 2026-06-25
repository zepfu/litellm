import asyncio
import aiohttp
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.mark.asyncio
async def test_aiohttp_handler_cleanup():
    """Test BaseLLMAIOHTTPHandler cleanup is deterministic via close()"""
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    handler_session = aiohttp.ClientSession()
    transport_session = aiohttp.ClientSession()

    transport = Mock()
    transport.client = transport_session
    transport._get_valid_client_session = Mock(return_value=handler_session)
    transport._owns_session = True
    transport.aclose = AsyncMock()

    handler = BaseLLMAIOHTTPHandler(transport=transport)
    handler._owns_transport = True

    session = handler._get_async_client_session()
    await handler.close()

    assert session is handler_session
    assert handler_session.closed
    assert transport_session.closed
    transport.aclose.assert_awaited_once()
    await handler_session.close()
    await transport_session.close()


@pytest.mark.asyncio
async def test_shared_transport_sessions_are_not_owned_by_default():
    """Test that shared transport ownership is preserved for externally-owned sessions."""
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    shared_session = Mock()
    shared_session.closed = False
    shared_session.close = AsyncMock()

    shared_transport = Mock()
    shared_transport.client = shared_session
    shared_transport._get_valid_client_session = Mock(return_value=shared_session)
    shared_transport._owns_session = False
    shared_transport.aclose = AsyncMock()

    handler = BaseLLMAIOHTTPHandler(transport=shared_transport)
    session = handler._get_async_client_session()

    assert handler._owns_session is False
    await handler.close()

    assert session is shared_session
    assert shared_session.closed is False
    shared_session.close.assert_not_called()
    shared_transport.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_close_litellm_async_clients_includes_owned_transport_handler():
    """Test global cleanup handles handlers that own transport sessions."""
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
    from litellm.llms.custom_httpx.async_client_cleanup import (
        close_litellm_async_clients,
    )
    import litellm

    transport_session = aiohttp.ClientSession()

    owned_transport = Mock()
    owned_transport.client = transport_session
    owned_transport._owns_session = True
    owned_transport.aclose = AsyncMock()

    owned_handler = BaseLLMAIOHTTPHandler(transport=owned_transport)
    owned_handler._owns_transport = True
    owned_handler._get_async_client_session()

    cache = SimpleNamespace(cache_dict={"owned-handler": owned_handler})

    with patch.object(
        litellm, "in_memory_llm_clients_cache", cache
    ), patch.object(litellm, "base_llm_aiohttp_handler", None):
        await close_litellm_async_clients()

    assert transport_session.closed
    owned_transport.aclose.assert_awaited_once()
    await transport_session.close()


def test_atexit_cleanup_via_async_loop():
    """Test that cleanup can run in a fresh event loop context."""
    from litellm.llms.custom_httpx.async_client_cleanup import (
        close_litellm_async_clients,
    )

    try:
        asyncio.get_running_loop()
        pytest.skip("Cannot test atexit scenario when event loop is running")
    except RuntimeError:
        pass

    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    try:
        new_loop.run_until_complete(close_litellm_async_clients())
    finally:
        new_loop.close()
