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
async def test_proxy_shutdown_event_runs_guardrail_shutdown_hooks():
    from litellm.proxy import proxy_server as proxy_server_module

    class GuardrailWithShutdown:
        guardrail_name = "session-owning-guardrail"

        def __init__(self):
            self.shutdown_calls = 0

        async def async_shutdown_hook(self):
            self.shutdown_calls += 1

    guardrail = GuardrailWithShutdown()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch.object(proxy_server_module.litellm, "callbacks", [guardrail]):
        await proxy_server_module.proxy_shutdown_event()

    assert guardrail.shutdown_calls == 1


@pytest.mark.asyncio
async def test_proxy_shutdown_event_logs_guardrail_shutdown_hook_errors():
    from litellm.proxy import proxy_server as proxy_server_module

    class FailingGuardrailShutdown:
        guardrail_name = "failing-guardrail"

        async def async_shutdown_hook(self):
            raise RuntimeError("cleanup failed")

    guardrail = FailingGuardrailShutdown()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch.object(
        proxy_server_module.litellm, "callbacks", [guardrail]
    ), patch.object(
        proxy_server_module.verbose_proxy_logger,
        "error",
    ) as mock_error:
        await proxy_server_module.proxy_shutdown_event()

    assert any(
        "guardrail shutdown hook" in str(call)
        for call in mock_error.call_args_list
    )


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
