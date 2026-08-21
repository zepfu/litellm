"""D1-528: periodic prune + eviction/overflow metrics on bounded routing maps.

Expired entries must prune without being read. FIFO still caps at
``DEFAULT_MEMORY_STATE_MAX_SIZE``. Eviction/prune counters must exist.
"""

from __future__ import annotations

import time
from typing import Any, MutableMapping

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
