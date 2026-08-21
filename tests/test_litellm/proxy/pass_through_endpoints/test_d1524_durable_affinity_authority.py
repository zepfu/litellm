"""D1-524: durable Redis affinity is authoritative across workers (Wave 1).

A still-valid local pin must not win when durable has a different pin.
Two logical workers with independent memory maps share one fake durable.
Existing OPENAI-020 / D1-612 tests must remain green.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state as cooldown_state_mod,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)


SESSION_KEY = "sess-d1524-shared-workers"


def _candidate(*, model: str) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "route_family": "openai_responses",
        "last_resort": False,
    }


def _local_pin(*, model: str) -> dict[str, Any]:
    return {
        **_candidate(model=model),
        "expires_at_monotonic": time.monotonic() + 6 * 3600,
        "affinity_state_source": "memory",
    }


class _SharedDurable:
    """In-memory DualCache stand-in shared by two logical workers."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self.redis_cache = MagicMock()
        self.redis_cache.async_set_cache = AsyncMock(side_effect=self._set)
        self.async_set_cache = AsyncMock(side_effect=self._set)
        self.async_get_cache = AsyncMock(side_effect=self._get)

    async def _set(self, key: str, value: Any, **kwargs: Any) -> None:
        payload = dict(value) if isinstance(value, dict) else {"value": value}
        ttl = kwargs.get("ttl")
        if "expires_at_epoch" not in payload:
            payload["expires_at_epoch"] = time.time() + float(ttl or 3600)
        self._store[key] = payload

    async def _get(self, key: str, **kwargs: Any) -> Any:
        return self._store.get(key)


@pytest.fixture()
def restore_cooldown_state_manager():
    original = cooldown_state_mod._manager
    from litellm.proxy import aawm_alias_routing_redis

    aawm_alias_routing_redis.reset()
    yield
    cooldown_state_mod._manager = original
    aawm_alias_routing_redis.reset()


@pytest.mark.asyncio
async def test_valid_local_pin_loses_to_different_durable_pin(
    restore_cooldown_state_manager,
) -> None:
    """If local is still valid AND durable has a different pin, durable wins."""
    worker = AliasRoutingStateManager()
    cooldown_state_mod.configure_cooldown_state_runtime(manager=worker)
    worker.codex.session_affinity_by_key[SESSION_KEY] = _local_pin(model="gpt-5.4-mini")

    durable_payload = {
        **_candidate(model="gpt-5.5"),
        "expires_at_epoch": time.time() + 3600,
    }
    dual = _SharedDurable()
    cache_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="affinity",
        state_key=SESSION_KEY,
    )
    dual._store[cache_key] = durable_payload

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


@pytest.mark.asyncio
async def test_two_workers_share_one_durable_without_clearing_local_maps(
    restore_cooldown_state_manager,
) -> None:
    """Worker B's durable write is visible to worker A without clearing A's local dict."""
    worker_a = AliasRoutingStateManager()
    worker_b = AliasRoutingStateManager()
    shared = _SharedDurable()

    worker_a.codex.session_affinity_by_key[SESSION_KEY] = _local_pin(
        model="gpt-5.4-mini"
    )
    assert SESSION_KEY in worker_a.codex.session_affinity_by_key
    assert worker_a.codex.session_affinity_by_key is not worker_b.codex.session_affinity_by_key

    with patch.object(
        cooldown_state_mod, "get_aawm_alias_routing_dual_cache", lambda: shared
    ), patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", lambda: shared
    ), patch.object(
        durable_mod, "_dual_cache_override", lambda: shared
    ):
        cooldown_state_mod.configure_cooldown_state_runtime(manager=worker_b)
        await cooldown_state_mod._set_codex_auto_agent_session_affinity(
            SESSION_KEY, _candidate(model="gpt-5.5")
        )

        cooldown_state_mod.configure_cooldown_state_runtime(manager=worker_a)
        affinity_a = await cooldown_state_mod._get_codex_auto_agent_session_affinity(
            SESSION_KEY
        )

    assert affinity_a is not None
    assert affinity_a.get("model") == "gpt-5.5"
    assert affinity_a.get("affinity_state_source") == "durable_cache"
    # Local map is not manually cleared; durable still wins on read.
    assert SESSION_KEY in worker_a.codex.session_affinity_by_key


@pytest.mark.asyncio
async def test_anthropic_valid_local_affinity_also_loses_to_durable(
    restore_cooldown_state_manager,
) -> None:
    worker = AliasRoutingStateManager()
    cooldown_state_mod.configure_cooldown_state_runtime(manager=worker)
    session_key = "sess-d1524-anthropic"
    worker.anthropic.session_affinity_by_key[session_key] = {
        "provider": "anthropic",
        "model": "claude-local",
        "route_family": "anthropic_messages",
        "last_resort": False,
        "expires_at_monotonic": time.monotonic() + 6 * 3600,
        "affinity_state_source": "memory",
    }
    dual = _SharedDurable()
    cache_key = durable_mod.build_aawm_alias_routing_durable_cache_key(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=session_key,
    )
    dual._store[cache_key] = {
        "provider": "anthropic",
        "model": "claude-durable",
        "route_family": "anthropic_messages",
        "last_resort": False,
        "expires_at_epoch": time.time() + 3600,
    }

    with patch.object(
        cooldown_state_mod, "get_aawm_alias_routing_dual_cache", lambda: dual
    ), patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", lambda: dual
    ), patch.object(
        durable_mod, "_dual_cache_override", lambda: dual
    ):
        affinity = await cooldown_state_mod._get_anthropic_auto_agent_session_affinity(
            session_key
        )

    assert affinity is not None
    assert affinity.get("model") == "claude-durable"
    assert affinity.get("affinity_state_source") == "durable_cache"
