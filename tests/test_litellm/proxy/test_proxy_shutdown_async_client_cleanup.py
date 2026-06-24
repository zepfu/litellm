from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_proxy_shutdown_event_closes_cached_async_clients():
    from litellm.proxy import proxy_server as proxy_server_module

    close_cached_clients = AsyncMock(name="close_litellm_async_clients")

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=close_cached_clients,
    ):
        await proxy_server_module.proxy_shutdown_event()

    close_cached_clients.assert_awaited_once()


@pytest.mark.asyncio
async def test_proxy_shutdown_event_logs_error_when_cached_async_client_cleanup_fails():
    from litellm.proxy import proxy_server as proxy_server_module

    close_cached_clients = AsyncMock(
        name="close_litellm_async_clients",
        side_effect=RuntimeError("cleanup failed"),
    )

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=close_cached_clients,
    ), patch.object(
        proxy_server_module.verbose_proxy_logger,
        "error",
    ) as mock_error:
        await proxy_server_module.proxy_shutdown_event()

    close_cached_clients.assert_awaited_once()
    mock_error.assert_called()
    assert "cached LiteLLM async HTTP clients" in str(mock_error.call_args)
