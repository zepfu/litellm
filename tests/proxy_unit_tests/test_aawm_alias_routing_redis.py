import json
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy import aawm_alias_routing_redis
from litellm.proxy.aawm_alias_routing_redis import AAWMAliasRoutingRedisManager
from litellm.proxy.health_endpoints import _health_endpoints
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _build_aawm_alias_routing_durable_cache_key,
    _read_aawm_alias_routing_durable_payload,
    _write_aawm_alias_routing_durable_payload,
)


def _build_fake_aawm_alias_cache():
    fake_cache = MagicMock()
    fake_cache.ping = AsyncMock(return_value=True)
    fake_cache.disconnect = AsyncMock()
    return fake_cache


def _build_proxy_logging_obj(dual_cache=None):
    """Standalone proxy_logging_obj sentinel that init/reset/shutdown must not touch."""
    if dual_cache is None:
        dual_cache = SimpleNamespace(redis_cache=None)
    return SimpleNamespace(
        internal_usage_cache=SimpleNamespace(
            dual_cache=dual_cache,
        )
    )


def _assert_no_secret_leakage_from_status(status: dict, secret: str) -> None:
    payload = json.dumps(status)
    assert secret not in payload


def _reset_aawm_alias_env(monkeypatch):
    for key in [
        "AAWM_ALIAS_ROUTING_REDIS_URL",
        "AAWM_ALIAS_ROUTING_REDIS_HOST",
        "AAWM_ALIAS_ROUTING_REDIS_PORT",
        "AAWM_ALIAS_ROUTING_REDIS_PASSWORD",
        "AAWM_ALIAS_ROUTING_REDIS_USERNAME",
        "AAWM_ALIAS_ROUTING_REDIS_DB",
        "AAWM_ALIAS_ROUTING_REDIS_SSL",
        "AAWM_ALIAS_ROUTING_STATE_NAMESPACE",
        "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
        "LITELLM_AAWM_ERROR_LOG_ENV",
    ]:
        monkeypatch.delenv(key, raising=False)


class _RedisNamespaceProbe:
    def __init__(self, namespace=None):
        self.namespace = namespace
        self.disconnect = AsyncMock()
        self.last_set_key = None
        self.last_get_key = None
        self.last_value = None

    async def async_set_cache(self, key, value, **kwargs):
        self.last_set_key = RedisCache.check_and_fix_namespace(self, key=key)
        self.last_value = value
        return None

    async def async_get_cache(self, key, **kwargs):
        self.last_get_key = RedisCache.check_and_fix_namespace(self, key=key)
        if self.last_value is None:
            return None
        payload = dict(self.last_value)
        payload["expires_at_epoch"] = time.time() + 10
        return payload


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_parses_url_config_and_builds_alias_cache(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    redis_url = "redis://:secret-token@aawm-cache.local:6380/0"
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_URL", redis_url)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "url-routing-plane")
    # Host is present to assert URL precedence when both are configured.
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "should-be-ignored")

    fake_cache = _build_fake_aawm_alias_cache()

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ) as mock_redis_ctor:
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

        assert fake_cache.ping.await_count == 1
        assert fake_cache.ping.awaited()
        _, cache_kwargs = mock_redis_ctor.call_args
        assert cache_kwargs["url"] == redis_url
        assert "namespace" not in cache_kwargs
        assert "host" not in cache_kwargs
        assert cache_kwargs["socket_timeout"] == 5.0
        status = manager.get_status()
        assert status["mode"] == "redis"
        assert status["config_mode"] == "url"
        assert status["reachable"] is True
        assert "secret-token" not in str(status)

        dual_cache = manager.get_dual_cache()
        assert dual_cache is not None
        assert isinstance(dual_cache, DualCache)
        assert dual_cache.redis_cache is fake_cache


@pytest.mark.parametrize(
    ("ssl_value", "expected_ssl"), [("true", True), ("false", False), ("0", False)]
)
@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_parses_host_config_port_ssl(
    monkeypatch, ssl_value, expected_ssl
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_PORT", "6381")
    if ssl_value is None:
        monkeypatch.delenv("AAWM_ALIAS_ROUTING_REDIS_SSL", raising=False)
    else:
        monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_SSL", ssl_value)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "host-routing-plane")

    fake_cache = _build_fake_aawm_alias_cache()

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ) as mock_redis_ctor:
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

        _, cache_kwargs = mock_redis_ctor.call_args
        assert cache_kwargs["host"] == "aawm-host"
        assert cache_kwargs["port"] == 6381
        if expected_ssl:
            assert cache_kwargs["ssl"] is True
        else:
            assert "ssl" not in cache_kwargs
        assert "namespace" not in cache_kwargs
        assert cache_kwargs["password"] is None
        status = manager.get_status()
        assert status["mode"] == "redis"
        assert status["config_mode"] == "host"
        dual_cache = manager.get_dual_cache()
        assert dual_cache is not None
        assert dual_cache.redis_cache is fake_cache


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_unconfigured_status_reports_memory(monkeypatch):
    _reset_aawm_alias_env(monkeypatch)

    manager = AAWMAliasRoutingRedisManager()
    status = manager.get_status()

    assert status["configured"] is False
    assert status["mode"] == "memory"
    assert status["config_mode"] == "unconfigured"
    assert status["reachable"] == "unknown"
    assert status["namespace"] == "default"
    assert manager.get_dual_cache() is None


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_initialization_failure_records_error_type_without_secret_leakage(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    redis_url = "redis://:forbidden-token@aawm-cache.local:6379/0"
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_URL", redis_url)

    fake_cache = _build_fake_aawm_alias_cache()
    fake_cache.ping = AsyncMock(side_effect=RuntimeError("redis rejected"))

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ):
        manager = AAWMAliasRoutingRedisManager()
        manager.STARTUP_RETRY_DELAY_SECONDS = 0
        await manager.initialize()

        status = manager.get_status()
        assert status["configured"] is True
        assert status["reachable"] is False
        assert status["mode"] == "memory"
        assert status["config_mode"] == "url"
        assert status["error_type"] == "RuntimeError"
        assert manager.get_dual_cache() is None
        _assert_no_secret_leakage_from_status(status, "forbidden-token")


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_tracks_namespace_and_key_prefix(monkeypatch):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "custom-routing-plane")

    fake_cache = _build_fake_aawm_alias_cache()

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

        status = manager.get_status()
        assert status["namespace"] == "custom-routing-plane"
        assert status["key_prefix"] == "aawm:alias-routing:custom-routing-plane"


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_initialize_is_idempotent(monkeypatch):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    fake_cache = _build_fake_aawm_alias_cache()

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ) as mock_redis_ctor:
        manager = AAWMAliasRoutingRedisManager()

        await manager.initialize()
        dual_cache_first = manager.get_dual_cache()
        await manager.initialize()
        dual_cache_second = manager.get_dual_cache()

        assert mock_redis_ctor.call_count == 1
        assert manager.get_status()["reachable"] is True
        assert dual_cache_first is dual_cache_second
        assert dual_cache_first is not None
        assert dual_cache_first.redis_cache is fake_cache


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_module_wrappers_share_health_status(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "wrapper-plane")

    fake_cache = _build_fake_aawm_alias_cache()

    aawm_alias_routing_redis.reset()
    try:
        with patch(
            "litellm.proxy.aawm_alias_routing_redis.RedisCache",
            side_effect=lambda *args, **kwargs: fake_cache,
        ) as mock_redis_ctor:
            await aawm_alias_routing_redis.initialize()

            status = aawm_alias_routing_redis.get_status()
            health_status = _health_endpoints._get_aawm_alias_routing_cache_status()

            # Manager/module singleton is the source of truth and includes config_mode.
            assert status == {
                "configured": True,
                "config_mode": "host",
                "mode": "redis",
                "state_source": "durable_cache",
                "reachable": True,
                "namespace": "wrapper-plane",
                "key_prefix": "aawm:alias-routing:wrapper-plane",
                "error_type": None,
            }

            # Production health helper re-maps status fields and currently omits
            # config_mode. Assert the actual helper output rather than stripping
            # manager fields to force a false equality that would mask that gap.
            assert health_status == {
                "configured": True,
                "mode": "redis",
                "state_source": "durable_cache",
                "reachable": True,
                "namespace": "wrapper-plane",
                "key_prefix": "aawm:alias-routing:wrapper-plane",
                "error_type": None,
            }
            assert "config_mode" not in health_status
            # Shared fields should still come from the singleton status payload.
            for field in (
                "configured",
                "mode",
                "state_source",
                "reachable",
                "namespace",
                "key_prefix",
                "error_type",
            ):
                assert health_status[field] == status[field]

            dual_cache = aawm_alias_routing_redis.get_dual_cache()
            assert dual_cache is not None
            assert dual_cache.redis_cache is fake_cache
            _, cache_kwargs = mock_redis_ctor.call_args
            assert "namespace" not in cache_kwargs
    finally:
        await aawm_alias_routing_redis.shutdown()


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_key_prefix_not_double_prefixed_through_set_get_probe(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "state-plane")

    probe = _RedisNamespaceProbe(namespace=None)
    dual_cache = SimpleNamespace(
        async_set_cache=probe.async_set_cache,
        async_get_cache=probe.async_get_cache,
        redis_cache=probe,
    )
    expected_key = _build_aawm_alias_routing_durable_cache_key(
        alias_family="GROK", state_kind="Cooldown", state_key="Team-Alpha:request-9"
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_aawm_alias_routing_dual_cache",
        return_value=dual_cache,
    ):
        write_ok = await _write_aawm_alias_routing_durable_payload(
            alias_family="GROK",
            state_kind="Cooldown",
            state_key="Team-Alpha:request-9",
            payload={"status": "cooldown"},
            ttl_seconds=30,
        )
        read_payload = await _read_aawm_alias_routing_durable_payload(
            alias_family="GROK", state_kind="Cooldown", state_key="Team-Alpha:request-9"
        )

    assert write_ok is True
    assert probe.last_set_key == expected_key
    assert probe.last_get_key == expected_key
    assert probe.last_set_key.count("aawm:alias-routing:state-plane") == 1

    # The payload can be re-materialized from durable cache through mocked get path.
    assert read_payload["status"] == "cooldown"

    # RedisCache must not prepend namespace again when key already contains the alias-routing
    # prefix (no double-prefixed key variants).
    assert probe.last_set_key == expected_key


@pytest.mark.asyncio
async def test_aawm_alias_routing_durable_write_surfaces_redis_failure(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "failure-plane")

    redis_cache = SimpleNamespace(
        async_set_cache=AsyncMock(side_effect=RuntimeError("redis write failed"))
    )
    dual_cache = SimpleNamespace(
        redis_cache=redis_cache,
        async_set_cache=AsyncMock(),
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._get_aawm_alias_routing_dual_cache",
        return_value=dual_cache,
    ):
        write_ok = await _write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="candidate-key",
            payload={"cooldown_key": "candidate-key"},
            ttl_seconds=30,
        )

    assert write_ok is False
    redis_cache.async_set_cache.assert_awaited_once()
    dual_cache.async_set_cache.assert_not_awaited()


@pytest.mark.parametrize(
    ("runtime_environment", "expected_namespace"),
    [
        ("dev", "aawm-routing-dev-v1"),
        ("development", "aawm-routing-dev-v1"),
        ("prod", "aawm-routing-prod-v1"),
        ("production", "aawm-routing-prod-v1"),
    ],
)
@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_derives_isolated_environment_namespace(
    monkeypatch, runtime_environment, expected_namespace
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")
    monkeypatch.setenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT", runtime_environment)

    fake_cache = _build_fake_aawm_alias_cache()
    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

    status = manager.get_status()
    assert status["namespace"] == expected_namespace
    assert status["key_prefix"] == (f"aawm:alias-routing:{expected_namespace}")
    assert status["state_source"] == "durable_cache"


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_reset_after_successful_wiring_detaches_owned_client(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    fake_cache = _build_fake_aawm_alias_cache()
    sentinel_redis = object()
    proxy_logging_obj = _build_proxy_logging_obj(
        dual_cache=SimpleNamespace(redis_cache=sentinel_redis)
    )

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

        assert manager.get_status()["mode"] == "redis"
        dual_cache = manager.get_dual_cache()
        assert dual_cache is not None
        assert dual_cache.redis_cache is fake_cache

        manager.reset()
        assert manager.get_status()["configured"] is False
        assert manager.get_status()["reachable"] == "unknown"
        assert manager.get_status()["mode"] == "memory"
        assert manager.get_status()["namespace"] == "default"
        assert manager.get_dual_cache() is None
        # reset detaches manager-owned DualCache only; does not disconnect live clients
        fake_cache.disconnect.assert_not_awaited()
        # shared proxy logging cache slot remains untouched
        assert (
            proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache
            is sentinel_redis
        )


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_does_not_mutate_proxy_logging_internal_usage_cache(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    fake_cache = _build_fake_aawm_alias_cache()
    sentinel_redis = object()
    proxy_logging_obj = _build_proxy_logging_obj(
        dual_cache=SimpleNamespace(redis_cache=sentinel_redis)
    )
    original_internal_usage_cache = proxy_logging_obj.internal_usage_cache
    original_dual_cache = original_internal_usage_cache.dual_cache

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()
        assert manager.get_dual_cache() is not None
        assert manager.get_dual_cache().redis_cache is fake_cache

        manager.reset()
        await manager.initialize()
        await manager.shutdown()

    assert proxy_logging_obj.internal_usage_cache is original_internal_usage_cache
    assert proxy_logging_obj.internal_usage_cache.dual_cache is original_dual_cache
    assert (
        proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache is sentinel_redis
    )


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_failed_initialization_then_retry_succeeds(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    failed_cache = _build_fake_aawm_alias_cache()
    failed_cache.ping = AsyncMock(side_effect=RuntimeError("redis unavailable"))
    healthy_cache = _build_fake_aawm_alias_cache()
    attempts = AAWMAliasRoutingRedisManager.STARTUP_CONNECT_ATTEMPTS

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        # Exhaust the in-call bounded retries, then recover on a later initialize().
        side_effect=[failed_cache] * attempts + [healthy_cache],
    ):
        manager = AAWMAliasRoutingRedisManager()
        manager.STARTUP_RETRY_DELAY_SECONDS = 0
        await manager.initialize()

        first_status = manager.get_status()
        assert first_status["reachable"] is False
        assert first_status["mode"] == "memory"
        assert manager.get_dual_cache() is None

        await manager.initialize()
        second_status = manager.get_status()
        assert second_status["reachable"] is True
        assert second_status["mode"] == "redis"
        assert manager._cache is healthy_cache
        dual_cache = manager.get_dual_cache()
        assert dual_cache is not None
        assert dual_cache.redis_cache is healthy_cache

    assert failed_cache.disconnect.await_count == attempts
    assert healthy_cache.disconnect.await_count == 0


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_shutdown_twice_after_success_and_after_failure(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    success_cache = _build_fake_aawm_alias_cache()

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: success_cache,
    ):
        success_manager = AAWMAliasRoutingRedisManager()
        await success_manager.initialize()
        assert success_manager.get_dual_cache() is not None
        await success_manager.shutdown()
        await success_manager.shutdown()

    assert success_cache.disconnect.await_count == 1
    assert success_manager.get_dual_cache() is None

    failed_cache = _build_fake_aawm_alias_cache()
    failed_cache.ping = AsyncMock(return_value=False)

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: failed_cache,
    ):
        failed_manager = AAWMAliasRoutingRedisManager()
        failed_manager.STARTUP_RETRY_DELAY_SECONDS = 0
        await failed_manager.initialize()
        await failed_manager.shutdown()
        await failed_manager.shutdown()

    # unreachable client attempts disconnect during initialize, not during shutdown
    assert failed_cache.disconnect.await_count == (
        AAWMAliasRoutingRedisManager.STARTUP_CONNECT_ATTEMPTS
    )
    assert failed_manager.get_status()["mode"] == "memory"
    assert failed_manager.get_dual_cache() is None


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_replacement_client_remains_wired_on_shutdown(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    first_cache = _build_fake_aawm_alias_cache()
    second_cache = _build_fake_aawm_alias_cache()
    sentinel_redis = object()
    proxy_logging_obj = _build_proxy_logging_obj(
        dual_cache=SimpleNamespace(redis_cache=sentinel_redis)
    )

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=[first_cache, second_cache],
    ):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()
        first_dual = manager.get_dual_cache()
        assert first_dual is not None
        assert first_dual.redis_cache is first_cache

        manager.reset()
        assert manager.get_dual_cache() is None
        # reset does not disconnect; re-init will disconnect previous owned client
        assert first_cache.disconnect.await_count == 0

        await manager.initialize()

        assert first_cache.disconnect.await_count == 1
        assert second_cache.disconnect.await_count == 0
        second_dual = manager.get_dual_cache()
        assert second_dual is not None
        assert second_dual.redis_cache is second_cache
        assert manager.get_status()["mode"] == "redis"
        assert (
            proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache
            is sentinel_redis
        )

        await manager.shutdown()

    assert second_cache.disconnect.await_count == 1
    assert manager.get_dual_cache() is None
    # only manager-owned fakes are disconnected; shared proxy logging slot untouched
    assert (
        proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache is sentinel_redis
    )


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_initialize_does_not_mutate_cache_or_router_response_cache(
    monkeypatch,
):
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    fake_cache = _build_fake_aawm_alias_cache()
    original_litellm_cache = litellm.cache
    original_router = SimpleNamespace(cache_responses=True)

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: fake_cache,
    ), patch("litellm.proxy.proxy_server.llm_router", original_router):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

        assert litellm.cache is original_litellm_cache
        assert original_router.cache_responses is True
        dual_cache = manager.get_dual_cache()
        assert dual_cache is not None
        assert dual_cache.redis_cache is fake_cache


@pytest.mark.asyncio
async def test_health_readiness_includes_sanitized_aawm_alias_routing_cache():
    alias_cache_status = {
        "configured": True,
        "mode": "memory",
        "state_source": "local_fallback",
        "reachable": "unknown",
        "namespace": "custom-routing-plane",
        "key_prefix": "aawm:alias-routing:custom-routing-plane",
        "error_type": "RuntimeError",
    }
    litellm.success_callback = []

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
        "litellm.proxy.proxy_server.version", "0.0.0-test"
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints._db_health_readiness_check",
        AsyncMock(return_value={"status": "connected"}),
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints._get_aawm_alias_routing_cache_status",
        return_value=alias_cache_status,
    ):
        response = await _health_endpoints.health_readiness()

    assert response["aawm_alias_routing_cache"] == alias_cache_status
    assert response["db"] == "connected"
    assert response["aawm_alias_routing_cache"]["error_type"] == "RuntimeError"
    assert response["aawm_alias_routing_cache"]["state_source"] == "local_fallback"


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_startup_event_uses_wrapper_target(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    old_max_budget = litellm.max_budget
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "use_background_health_checks", False)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    litellm.max_budget = 0

    try:
        fake_prisma = SimpleNamespace(db=None)
        with patch(
            "litellm.proxy.proxy_server.get_secret",
            side_effect=lambda key, default=None: None,
        ), patch(
            "litellm.proxy.proxy_server.get_secret_str",
            side_effect=lambda key, default=None: None,
        ), patch(
            "litellm.proxy.proxy_server.proxy_config.get_config_state",
            return_value={},
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._setup_prisma_client",
            AsyncMock(return_value=fake_prisma),
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._initialize_startup_logging",
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._validate_redis_transaction_buffer_config",
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._initialize_semantic_tool_filter",
            AsyncMock(),
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._initialize_jwt_auth",
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent.initialize_scheduled_background_jobs",
            AsyncMock(),
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._update_default_team_member_budget",
            AsyncMock(),
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._sync_ui_settings_to_general_settings",
            AsyncMock(),
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._init_dd_tracer",
        ), patch(
            "litellm.proxy.proxy_server.ProxyStartupEvent._init_pyroscope",
        ), patch(
            "litellm.proxy.proxy_server._initialize_shared_aiohttp_session",
            AsyncMock(return_value=None),
        ), patch(
            "litellm.proxy.proxy_server.proxy_shutdown_event",
            AsyncMock(),
        ), patch(
            "litellm.proxy.proxy_server.initialize_aawm_alias_routing_redis",
            AsyncMock(),
        ) as mock_init_alias:
            async with proxy_server.proxy_startup_event(app=None):
                pass

        mock_init_alias.assert_awaited_once_with()
    finally:
        litellm.max_budget = old_max_budget


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_transient_startup_failure_then_success(
    monkeypatch,
):
    """A single initialize() recovers after a transient first connect failure."""
    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    failed_cache = _build_fake_aawm_alias_cache()
    failed_cache.ping = AsyncMock(side_effect=RuntimeError("redis warming up"))
    healthy_cache = _build_fake_aawm_alias_cache()

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=[failed_cache, healthy_cache],
    ) as mock_redis_ctor:
        manager = AAWMAliasRoutingRedisManager()
        manager.STARTUP_RETRY_DELAY_SECONDS = 0
        await manager.initialize()

        status = manager.get_status()
        assert status["reachable"] is True
        assert status["mode"] == "redis"
        assert status["state_source"] == "durable_cache"
        assert status["error_type"] is None
        assert manager.get_dual_cache() is not None
        assert manager._cache is healthy_cache
        assert mock_redis_ctor.call_count == 2

    assert failed_cache.disconnect.await_count == 1
    assert healthy_cache.disconnect.await_count == 0
    assert failed_cache.ping.await_count == 1
    assert healthy_cache.ping.await_count == 1


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_exhausted_startup_retry_falls_back_to_memory(
    monkeypatch,
):
    """When all bounded startup attempts fail, fallback stays visible and sanitized."""
    _reset_aawm_alias_env(monkeypatch)
    redis_url = "redis://:retry-secret@aawm-cache.local:6379/0"
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_URL", redis_url)

    failed_cache = _build_fake_aawm_alias_cache()
    failed_cache.ping = AsyncMock(side_effect=ConnectionError("connection refused"))
    attempts = AAWMAliasRoutingRedisManager.STARTUP_CONNECT_ATTEMPTS

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=lambda *args, **kwargs: failed_cache,
    ) as mock_redis_ctor:
        manager = AAWMAliasRoutingRedisManager()
        manager.STARTUP_RETRY_DELAY_SECONDS = 0
        await manager.initialize()

        status = manager.get_status()
        assert status["configured"] is True
        assert status["reachable"] is False
        assert status["mode"] == "memory"
        assert status["state_source"] == "local_fallback"
        assert status["error_type"] == "ConnectionError"
        assert manager.get_dual_cache() is None
        assert mock_redis_ctor.call_count == attempts
        _assert_no_secret_leakage_from_status(status, "retry-secret")

    assert failed_cache.ping.await_count == attempts
    assert failed_cache.disconnect.await_count == attempts


@pytest.mark.asyncio
async def test_aawm_alias_routing_redis_constructs_cache_off_running_event_loop(
    monkeypatch,
):
    """Constructor runs off the proxy event loop so RedisCache health-ping tasks are not scheduled on it."""
    import asyncio

    _reset_aawm_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    constructed_on_running_loop = []
    fake_cache = _build_fake_aawm_alias_cache()

    def _ctor(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            constructed_on_running_loop.append(True)
        except RuntimeError:
            constructed_on_running_loop.append(False)
        return fake_cache

    with patch(
        "litellm.proxy.aawm_alias_routing_redis.RedisCache",
        side_effect=_ctor,
    ):
        manager = AAWMAliasRoutingRedisManager()
        await manager.initialize()

        assert manager.get_status()["mode"] == "redis"
        assert constructed_on_running_loop == [False]
