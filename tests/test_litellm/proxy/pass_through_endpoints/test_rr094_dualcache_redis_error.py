"""RR-094 / RR-118: DualCache Redis errors must not look like confirmed misses.

Existing D1-551 tests mock ``DualCache.async_get_cache`` to raise. That skips
the real DualCache path, which currently swallows Redis ``ConnectionError``
and returns ``None``. Routing then treats the failure as ``confirmed_miss``.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state as cooldown_state_mod,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
    configure_cooldown_state_runtime,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)


class _RaisingRedisCacheStub:
    """RedisCache stand-in whose async get raises. No live Redis."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.get_calls = 0

    async def async_get_cache(self, key, parent_otel_span=None, **kwargs):
        self.get_calls += 1
        raise self.error


def _real_dual_cache_with_redis_error(
    error: BaseException | None = None,
) -> tuple[DualCache, _RaisingRedisCacheStub]:
    redis_cache = _RaisingRedisCacheStub(
        error if error is not None else ConnectionError("redis unavailable")
    )
    dual = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,  # type: ignore[arg-type]
    )
    return dual, redis_cache


def _local_affinity(*, model: str) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "route_family": "openai_responses",
        "last_resort": False,
        "expires_at_monotonic": time.monotonic() + 3600,
        "affinity_state_source": "memory",
    }


@pytest.mark.asyncio
async def test_rr094_real_dualcache_redis_connection_error_is_durable_error_not_confirmed_miss() -> None:
    dual, redis_cache = _real_dual_cache_with_redis_error()
    result = await durable_mod.read_aawm_alias_routing_state(
        alias_family="codex",
        state_kind="cooldown",
        state_key="rr094-dualcache-connection-error",
        last_good_local=None,
        dual_cache=dual,
    )

    assert redis_cache.get_calls >= 1
    assert result["durable_error"] is True
    assert result["confirmed_miss"] is False
    assert result.get("durable_miss") is not True
    assert result["source"] == "durable_error"


@pytest.mark.asyncio
async def test_rr094_real_dualcache_redis_error_keeps_last_good_local_as_degraded() -> None:
    dual, redis_cache = _real_dual_cache_with_redis_error()
    last_good = _local_affinity(model="gpt-5.4-mini")
    result = await durable_mod.read_aawm_alias_routing_state(
        alias_family="codex",
        state_kind="affinity",
        state_key="rr094-dualcache-degraded-local",
        last_good_local=last_good,
        dual_cache=dual,
    )

    assert redis_cache.get_calls >= 1
    assert result["durable_error"] is True
    assert result["confirmed_miss"] is False
    assert result["source"] == "degraded_local"
    assert result["payload"]["model"] == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_rr094_cooldown_does_not_negative_cache_dualcache_redis_error() -> None:
    dual, redis_cache = _real_dual_cache_with_redis_error()
    previous_manager = cooldown_state_mod._manager
    manager = AliasRoutingStateManager()
    configure_cooldown_state_runtime(manager=manager)
    cooldown_key = "rr094-codex-cd-redis-error"
    try:
        seconds, source = await cooldown_state_mod._get_codex_auto_agent_active_cooldown_state(
            cooldown_key,
            _dual_cache_fn=lambda: dual,
        )
        assert redis_cache.get_calls >= 1
        assert seconds == 0.0
        assert source != "negative_cache"
        assert manager.codex.is_negative_cached(cooldown_key) is False
        assert cooldown_key not in manager.codex.cooldown_negative_until_monotonic_by_key

        seconds_again, source_again = (
            await cooldown_state_mod._get_codex_auto_agent_active_cooldown_state(
                cooldown_key,
                _dual_cache_fn=lambda: dual,
            )
        )
        assert redis_cache.get_calls >= 2
        assert seconds_again == 0.0
        assert source_again != "negative_cache"
        assert manager.codex.is_negative_cached(cooldown_key) is False
    finally:
        cooldown_state_mod._manager = previous_manager


@pytest.mark.asyncio
async def test_rr118_confirmed_miss_source_labels_distinguish_local_lease_from_miss() -> None:
    class _EmptyRedisCache:
        async def async_get_cache(self, key, parent_otel_span=None, **kwargs):
            return None

    dual = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=_EmptyRedisCache(),  # type: ignore[arg-type]
    )
    miss_without_local = await durable_mod.read_aawm_alias_routing_state(
        alias_family="codex",
        state_kind="affinity",
        state_key="rr118-miss-no-local",
        last_good_local=None,
        dual_cache=dual,
    )
    miss_with_local = await durable_mod.read_aawm_alias_routing_state(
        alias_family="codex",
        state_kind="affinity",
        state_key="rr118-miss-with-local",
        last_good_local=_local_affinity(model="gpt-5.4"),
        dual_cache=dual,
    )

    assert miss_without_local["confirmed_miss"] is True
    assert miss_with_local["confirmed_miss"] is True
    assert miss_without_local["durable_error"] is False
    assert miss_with_local["durable_error"] is False
    assert miss_without_local["source"] != miss_with_local["source"]
    assert miss_without_local["source"] in {"confirmed_miss", "durable_miss", "miss"}
    assert miss_with_local["source"] in {"local_lease", "memory"}
    assert miss_without_local["source"] != "memory"


def test_rr118_durable_reader_does_not_use_identical_memory_ternary() -> None:
    source = inspect.getsource(durable_mod.read_aawm_alias_routing_state)
    assert '"memory" if last_good_local else "memory"' not in source
    assert "'memory' if last_good_local else 'memory'" not in source
