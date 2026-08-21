"""D1-528: periodic prune + eviction/overflow metrics on bounded routing maps.

Expired entries must prune without being read. FIFO still caps at
``DEFAULT_MEMORY_STATE_MAX_SIZE``. Eviction/prune counters must exist.
"""

from __future__ import annotations

import time
from typing import Any, MutableMapping

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
    DEFAULT_MEMORY_STATE_MAX_SIZE,
    bound_memory_map,
)


def _require_prune() -> Any:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import memory as memory_mod

    prune = getattr(memory_mod, "prune_expired_memory_map", None)
    if prune is None:
        prune = getattr(memory_mod, "prune_expired_monotonic_entries", None)
    assert prune is not None, (
        "memory.py must expose prune_expired_memory_map / prune_expired_monotonic_entries"
    )
    return prune, memory_mod


def _counter(module: Any, *names: str) -> int:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, int):
            return value
        if hasattr(value, "value"):
            return int(value.value)
        if isinstance(value, dict) and "count" in value:
            return int(value["count"])
    metrics = getattr(module, "MEMORY_STATE_METRICS", None) or getattr(
        module, "memory_state_metrics", None
    )
    if isinstance(metrics, dict):
        for name in names:
            if name in metrics:
                return int(metrics[name])
    raise AssertionError(f"missing eviction/prune counter among {names}")


def test_expired_entries_prune_without_being_read() -> None:
    prune, memory_mod = _require_prune()
    now = time.monotonic()
    cache: MutableMapping[str, float] = {
        "expired-unread-1": now - 10.0,
        "expired-unread-2": now - 1.0,
        "still-valid": now + 3600.0,
    }
    before_prunes = _counter(
        memory_mod, "expired_prune_count", "memory_expired_prune_count", "pruned"
    )
    prune(cache, now=now)
    assert "expired-unread-1" not in cache
    assert "expired-unread-2" not in cache
    assert cache["still-valid"] == now + 3600.0
    after_prunes = _counter(
        memory_mod, "expired_prune_count", "memory_expired_prune_count", "pruned"
    )
    assert after_prunes >= before_prunes + 2


def test_fifo_still_caps_at_default_memory_state_max_size() -> None:
    prune, memory_mod = _require_prune()
    now = time.monotonic()
    cache: dict[str, float] = {}
    for i in range(DEFAULT_MEMORY_STATE_MAX_SIZE + 5):
        cache[f"k{i}"] = now + 60.0
        bound_memory_map(cache, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
    assert len(cache) == DEFAULT_MEMORY_STATE_MAX_SIZE
    assert "k0" not in cache
    assert f"k{DEFAULT_MEMORY_STATE_MAX_SIZE + 4}" in cache
    prune(cache, now=now)
    assert len(cache) <= DEFAULT_MEMORY_STATE_MAX_SIZE
    # Still-valid pins must not be expired-pruned.
    assert all(until > now for until in cache.values())


def test_eviction_and_prune_counters_exist_and_increment() -> None:
    prune, memory_mod = _require_prune()
    before_evictions = _counter(
        memory_mod,
        "fifo_eviction_count",
        "memory_fifo_eviction_count",
        "evicted",
        "overflow_eviction_count",
    )
    cache = {f"k{i}": time.monotonic() + 10.0 for i in range(4)}
    bound_memory_map(cache, max_size=2)
    after_evictions = _counter(
        memory_mod,
        "fifo_eviction_count",
        "memory_fifo_eviction_count",
        "evicted",
        "overflow_eviction_count",
    )
    assert after_evictions >= before_evictions + 2

    now = time.monotonic()
    expired = {"old": now - 5.0, "live": now + 5.0}
    before_prunes = _counter(
        memory_mod, "expired_prune_count", "memory_expired_prune_count", "pruned"
    )
    prune(expired, now=now)
    after_prunes = _counter(
        memory_mod, "expired_prune_count", "memory_expired_prune_count", "pruned"
    )
    assert after_prunes >= before_prunes + 1
    assert "live" in expired
    assert "old" not in expired


def test_affinity_payload_map_prunes_expired_monotonic_without_read() -> None:
    prune, _memory_mod = _require_prune()
    now = time.monotonic()
    cache: dict[str, dict[str, Any]] = {
        "expired-session": {
            "model": "gpt-5.4-mini",
            "expires_at_monotonic": now - 1.0,
        },
        "live-session": {
            "model": "gpt-5.5",
            "expires_at_monotonic": now + 100.0,
        },
    }
    prune(cache, now=now)
    assert "expired-session" not in cache
    assert cache["live-session"]["model"] == "gpt-5.5"


def test_rr114_production_bound_path_prunes_expired_before_fifo() -> None:
    """Expired keys must leave via prune, not FIFO, on the production bound path.

    Direct ``prune_expired_memory_map`` coverage is not enough: cooldown and
    affinity writers call ``bound_memory_map`` without pruning first, so expired
    entries still occupy FIFO slots and can displace live keys.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import memory as memory_mod
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        bound_memory_map,
    )

    now = time.monotonic()
    # Live keys are oldest. FIFO-only eviction would drop them and keep expired
    # entries. Prune-before-FIFO must keep the live keys instead.
    cache: dict[str, float] = {
        "live-keep": now + 3600.0,
        "live-keep-2": now + 3600.0,
        "expired-newest": now - 10.0,
        "expired-newest-2": now - 1.0,
    }
    before_prunes = _counter(
        memory_mod, "expired_prune_count", "memory_expired_prune_count", "pruned"
    )
    before_evictions = _counter(
        memory_mod,
        "fifo_eviction_count",
        "memory_fifo_eviction_count",
        "evicted",
        "overflow_eviction_count",
    )
    bound_memory_map(cache, max_size=2)
    after_prunes = _counter(
        memory_mod, "expired_prune_count", "memory_expired_prune_count", "pruned"
    )
    after_evictions = _counter(
        memory_mod,
        "fifo_eviction_count",
        "memory_fifo_eviction_count",
        "evicted",
        "overflow_eviction_count",
    )
    assert "expired-newest" not in cache
    assert "expired-newest-2" not in cache
    assert "live-keep" in cache
    assert "live-keep-2" in cache
    assert after_prunes >= before_prunes + 2
    assert after_evictions == before_evictions


@pytest.mark.asyncio
async def test_rr114_cooldown_writer_prunes_expired_before_fifo_on_production_path() -> None:
    """Production cooldown writes must prune expired entries before FIFO eviction."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        cooldown_state as cooldown_state_mod,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
        configure_cooldown_state_runtime,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasRoutingStateManager,
    )

    previous_manager = cooldown_state_mod._manager
    manager = AliasRoutingStateManager(max_size=2)
    configure_cooldown_state_runtime(manager=manager)
    now = time.monotonic()
    family = manager.codex
    # Insert live first so FIFO-without-prune would evict it to make room.
    family.cooldown_until_monotonic_by_key["live-keep"] = now + 3600.0
    family.cooldown_until_monotonic_by_key["expired-newest"] = now - 30.0
    try:
        with patch.object(
            cooldown_state_mod,
            "write_aawm_alias_routing_durable_payload",
            new=AsyncMock(return_value=True),
        ), patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state.DEFAULT_MEMORY_STATE_MAX_SIZE",
            2,
        ):
            await cooldown_state_mod._set_codex_auto_agent_cooldown("live-new", 60.0)
        remaining = family.cooldown_until_monotonic_by_key
        assert "expired-newest" not in remaining
        assert "live-keep" in remaining
        assert "live-new" in remaining
        assert len(remaining) <= 2
    finally:
        cooldown_state_mod._manager = previous_manager
