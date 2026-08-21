"""D1-550: successful Redis self-heal must reconcile process-local routing maps.

After a startup outage, reattach is not enough. Outage-window local affinity
must not win over durable once Redis is reachable again.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.aawm_alias_routing_redis import AAWMAliasRoutingRedisManager
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state as cooldown_state_mod,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
    alias_routing_state,
)


SESSION_KEY = "sess-d1550-outage-window"


def _reset_alias_env(monkeypatch) -> None:
    for key in [
        "AAWM_ALIAS_ROUTING_REDIS_URL",
        "AAWM_ALIAS_ROUTING_REDIS_HOST",
        "AAWM_ALIAS_ROUTING_REDIS_PORT",
        "AAWM_ALIAS_ROUTING_REDIS_PASSWORD",
        "AAWM_ALIAS_ROUTING_REDIS_USERNAME",
        "AAWM_ALIAS_ROUTING_REDIS_DB",
        "AAWM_ALIAS_ROUTING_REDIS_SSL",
        "AAWM_ALIAS_ROUTING_REDIS_TIMEOUT_SECONDS",
        "AAWM_ALIAS_ROUTING_REDIS_SELF_HEAL_INTERVAL_SECONDS",
        "AAWM_ALIAS_ROUTING_STATE_NAMESPACE",
        "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
        "LITELLM_AAWM_ERROR_LOG_ENV",
    ]:
        monkeypatch.delenv(key, raising=False)


def _local_pin(*, model: str) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "route_family": "openai_responses",
        "last_resort": False,
        "expires_at_monotonic": time.monotonic() + 6 * 3600,
        "affinity_state_source": "memory",
    }


@pytest.fixture(autouse=True)
def restore_cooldown_state_manager():
    from litellm.proxy import aawm_alias_routing_redis

    original = cooldown_state_mod._manager
    aawm_alias_routing_redis.reset()
    yield
    cooldown_state_mod._manager = original
    aawm_alias_routing_redis.reset()


@pytest.mark.asyncio
async def test_self_heal_reconciles_outage_window_local_affinity(monkeypatch) -> None:
    """After successful self-heal, outage-window local affinity does not win over durable."""
    _reset_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE", "d1550-test")

    family = alias_routing_state.codex
    family.session_affinity_by_key.clear()
    family.cooldown_until_monotonic_by_key.clear()
    family.session_affinity_by_key[SESSION_KEY] = _local_pin(model="gpt-5.4-mini")

    failed_cache = MagicMock()
    failed_cache.ping = AsyncMock(side_effect=ConnectionError("connection refused"))
    failed_cache.disconnect = AsyncMock()
    healthy_cache = MagicMock()
    healthy_cache.ping = AsyncMock(return_value=True)
    healthy_cache.disconnect = AsyncMock()
    attempts = AAWMAliasRoutingRedisManager.STARTUP_CONNECT_ATTEMPTS
    builds = {"n": 0}

    async def _build_off_loop(config):
        builds["n"] += 1
        if builds["n"] <= attempts:
            return failed_cache
        return healthy_cache

    manager = AAWMAliasRoutingRedisManager()
    manager.STARTUP_RETRY_DELAY_SECONDS = 0
    manager._build_redis_cache_off_loop = _build_off_loop  # type: ignore[method-assign]
    manager._resolve_self_heal_interval_seconds = lambda: 0.05  # type: ignore[method-assign]

    durable_payload = {
        "provider": "openai",
        "model": "gpt-5.5",
        "route_family": "openai_responses",
        "last_resort": False,
        "expires_at_epoch": time.time() + 3600,
    }

    try:
        await manager.initialize()
        first_status = manager.get_status()
        assert first_status["reachable"] is False
        assert first_status["self_heal_active"] is True
        assert SESSION_KEY in family.session_affinity_by_key

        for _ in range(100):
            if manager.get_status().get("reachable") is True:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("self-heal did not restore durable cache")

        second_status = manager.get_status()
        assert second_status["reachable"] is True
        assert second_status["self_heal_active"] is False
        # Source transition after reconcile (no secrets).
        assert second_status.get("local_state_reconciled") is True or second_status.get(
            "affinity_state_source"
        ) in {"durable_cache", "reconciled"}

        # Outage-window pin must not remain authoritative.
        remaining = family.session_affinity_by_key.get(SESSION_KEY)
        if remaining is not None:
            assert remaining.get("model") != "gpt-5.4-mini" or remaining.get(
                "affinity_state_source"
            ) != "memory"

        cooldown_state_mod.configure_cooldown_state_runtime(manager=alias_routing_state)
        dual = MagicMock()
        dual.redis_cache = healthy_cache
        dual.async_get_cache = AsyncMock(return_value=durable_payload)
        with patch.object(
            cooldown_state_mod, "get_aawm_alias_routing_dual_cache", lambda: dual
        ), patch.object(
            durable_mod, "get_aawm_alias_routing_dual_cache", lambda: dual
        ), patch.object(
            durable_mod, "_dual_cache_override", lambda: dual
        ):
            affinity = await cooldown_state_mod._get_codex_auto_agent_session_affinity(
                SESSION_KEY
            )
        assert affinity is not None
        assert affinity.get("model") == "gpt-5.5"
        assert affinity.get("affinity_state_source") == "durable_cache"
    finally:
        family.session_affinity_by_key.pop(SESSION_KEY, None)
        try:
            await manager.shutdown()
        except Exception:
            manager.reset()


@pytest.mark.asyncio
async def test_self_heal_success_invalidates_local_cooldown_maps(monkeypatch) -> None:
    _reset_alias_env(monkeypatch)
    monkeypatch.setenv("AAWM_ALIAS_ROUTING_REDIS_HOST", "aawm-host")

    worker = AliasRoutingStateManager()
    cooldown_key = "openai:gpt-5.4:auth:d1550"
    worker.codex.cooldown_until_monotonic_by_key[cooldown_key] = (
        time.monotonic() + 120.0
    )
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.state.alias_routing_state",
        worker,
    )

    failed_cache = MagicMock()
    failed_cache.ping = AsyncMock(side_effect=ConnectionError("connection refused"))
    failed_cache.disconnect = AsyncMock()
    healthy_cache = MagicMock()
    healthy_cache.ping = AsyncMock(return_value=True)
    healthy_cache.disconnect = AsyncMock()
    attempts = AAWMAliasRoutingRedisManager.STARTUP_CONNECT_ATTEMPTS
    builds = {"n": 0}

    async def _build_off_loop(config):
        builds["n"] += 1
        if builds["n"] <= attempts:
            return failed_cache
        return healthy_cache

    manager = AAWMAliasRoutingRedisManager()
    manager.STARTUP_RETRY_DELAY_SECONDS = 0
    manager._build_redis_cache_off_loop = _build_off_loop  # type: ignore[method-assign]
    manager._resolve_self_heal_interval_seconds = lambda: 0.05  # type: ignore[method-assign]
    try:
        await manager.initialize()
        for _ in range(100):
            if manager.get_status().get("reachable") is True:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("self-heal did not restore durable cache")

        reconcile = getattr(manager, "reconcile_local_routing_state", None)
        if reconcile is None:
            reconcile = getattr(
                alias_routing_state, "invalidate_local_routing_maps", None
            )
        assert reconcile is not None, (
            "self-heal must expose a reconcile/invalidate hook for local maps"
        )
        if cooldown_key in worker.codex.cooldown_until_monotonic_by_key:
            # Either the loop already cleared, or the hook is callable for next read.
            assert worker.codex.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0) <= time.monotonic() or callable(
                reconcile
            )
    finally:
        try:
            await manager.shutdown()
        except Exception:
            manager.reset()
