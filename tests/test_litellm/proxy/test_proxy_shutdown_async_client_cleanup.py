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

    empty_handler = type(
        "Handler",
        (),
        {"guardrail_id_to_custom_guardrail": {}},
    )()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        empty_handler,
    ), patch.object(
        proxy_server_module.litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        return_value=[guardrail],
    ):
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

    empty_handler = type(
        "Handler",
        (),
        {"guardrail_id_to_custom_guardrail": {}},
    )()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        empty_handler,
    ), patch.object(
        proxy_server_module.litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        return_value=[guardrail],
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

@pytest.mark.asyncio
async def test_proxy_shutdown_event_closes_guardrail_from_in_memory_registry_only():
    """Guardrails reachable only via IN_MEMORY_GUARDRAIL_HANDLER still shut down."""
    from litellm.proxy import proxy_server as proxy_server_module

    class RegistryOnlyGuardrail:
        guardrail_name = "registry-only-guardrail"

        def __init__(self):
            self.shutdown_calls = 0

        async def async_shutdown_hook(self):
            self.shutdown_calls += 1

    guardrail = RegistryOnlyGuardrail()
    mock_handler = type(
        "Handler",
        (),
        {"guardrail_id_to_custom_guardrail": {"gid-1": guardrail}},
    )()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch.object(
        proxy_server_module.litellm,
        "callbacks",
        [],
    ), patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_handler,
    ), patch.object(
        proxy_server_module.litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        return_value=[],
    ):
        await proxy_server_module.proxy_shutdown_event()

    assert guardrail.shutdown_calls == 1


@pytest.mark.asyncio
async def test_proxy_shutdown_event_closes_guardrail_from_logging_callback_manager_only():
    """Session-owning guardrails registered only on logging_callback_manager close once."""
    from litellm.proxy import proxy_server as proxy_server_module

    class ManagerOnlyGuardrail:
        guardrail_name = "manager-only-guardrail"

        def __init__(self):
            self.shutdown_calls = 0

        async def async_shutdown_hook(self):
            self.shutdown_calls += 1

    guardrail = ManagerOnlyGuardrail()
    empty_handler = type(
        "Handler",
        (),
        {"guardrail_id_to_custom_guardrail": {}},
    )()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch.object(
        proxy_server_module.litellm,
        "callbacks",
        [],
    ), patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        empty_handler,
    ), patch.object(
        proxy_server_module.litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        return_value=[guardrail],
    ):
        await proxy_server_module.proxy_shutdown_event()

    assert guardrail.shutdown_calls == 1


@pytest.mark.asyncio
async def test_proxy_shutdown_event_deduplicates_guardrail_shutdown_hooks():
    from litellm.proxy import proxy_server as proxy_server_module

    class SharedGuardrail:
        guardrail_name = "deduped-guardrail"

        def __init__(self):
            self.shutdown_calls = 0

        async def async_shutdown_hook(self):
            self.shutdown_calls += 1

    guardrail = SharedGuardrail()
    mock_handler = type(
        "Handler",
        (),
        {"guardrail_id_to_custom_guardrail": {"gid-1": guardrail}},
    )()

    with patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=AsyncMock(name="close_litellm_async_clients"),
    ), patch.object(
        proxy_server_module.litellm,
        "callbacks",
        [guardrail],
    ), patch(
        "litellm.proxy.guardrails.guardrail_registry.IN_MEMORY_GUARDRAIL_HANDLER",
        mock_handler,
    ), patch.object(
        proxy_server_module.litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        return_value=[guardrail],
    ), patch.object(
        proxy_server_module,
        "llm_router",
        type(
            "Router",
            (),
            {
                "guardrail_list": [
                    {"callback": guardrail, "guardrail_name": "deduped-guardrail"}
                ]
            },
        )(),
    ):
        await proxy_server_module.proxy_shutdown_event()

    assert guardrail.shutdown_calls == 1


@pytest.mark.asyncio
async def test_proxy_shutdown_event_continues_after_aawm_alias_redis_shutdown_failure():
    """AAWM alias Redis cleanup failures must not skip later proxy shutdown steps."""
    from litellm.proxy import proxy_server as proxy_server_module

    close_cached_clients = AsyncMock(name="close_litellm_async_clients")
    shutdown_alias_redis = AsyncMock(
        name="shutdown_aawm_alias_routing_redis",
        side_effect=RuntimeError("alias redis cleanup failed"),
    )

    with patch.object(
        proxy_server_module,
        "shutdown_aawm_alias_routing_redis",
        new=shutdown_alias_redis,
    ), patch.object(
        proxy_server_module.litellm,
        "close_litellm_async_clients",
        new=close_cached_clients,
    ), patch.object(
        proxy_server_module.verbose_proxy_logger,
        "error",
    ) as mock_error, patch.object(
        proxy_server_module,
        "_close_guardrail_shutdown_hooks",
        new=AsyncMock(name="_close_guardrail_shutdown_hooks"),
    ) as close_guardrails:
        await proxy_server_module.proxy_shutdown_event()

    shutdown_alias_redis.assert_awaited_once()
    close_cached_clients.assert_awaited_once()
    close_guardrails.assert_awaited_once()
    mock_error.assert_called()
    assert "AAWM alias routing Redis" in str(mock_error.call_args)
