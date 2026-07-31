"""Wave 7 owner tests for runtime-memory functions (D1-591)."""

from __future__ import annotations

import inspect
import time
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    runtime_memory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.runtime_memory import (
    RuntimeMemoryRuntime,
    bound_aawm_alias_routing_memory_map,
    configure_runtime_memory,
    hydrate_aawm_alias_routing_affinity_memory,
    hydrate_aawm_alias_routing_cooldown_memory,
    replace_request_body_in_place,
    should_log_aawm_alias_routing_event,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runtime() -> RuntimeMemoryRuntime:
    previous = runtime_memory._runtime
    log_map: dict[str, float] = {}
    rt = RuntimeMemoryRuntime(log_until_map=log_map, max_size=8)
    configure_runtime_memory(runtime=rt)
    try:
        yield rt
    finally:
        runtime_memory._runtime = previous


# ---------------------------------------------------------------------------
# configure / require
# ---------------------------------------------------------------------------


def test_configure_runtime_memory_is_sync() -> None:
    assert not inspect.iscoroutinefunction(configure_runtime_memory)


def test_should_log_requires_runtime() -> None:
    previous = runtime_memory._runtime
    runtime_memory._runtime = None
    try:
        with pytest.raises(RuntimeError, match="runtime_memory runtime not configured"):
            should_log_aawm_alias_routing_event("k")
    finally:
        runtime_memory._runtime = previous


def test_bound_map_requires_runtime_when_no_explicit_max() -> None:
    previous = runtime_memory._runtime
    runtime_memory._runtime = None
    try:
        with pytest.raises(RuntimeError, match="runtime_memory runtime not configured"):
            bound_aawm_alias_routing_memory_map({"a": 1})
    finally:
        runtime_memory._runtime = previous


# ---------------------------------------------------------------------------
# should_log_aawm_alias_routing_event
# ---------------------------------------------------------------------------


def test_should_log_first_call_returns_true(runtime: RuntimeMemoryRuntime) -> None:
    assert should_log_aawm_alias_routing_event("key-a") is True


def test_should_log_second_call_suppressed(runtime: RuntimeMemoryRuntime) -> None:
    assert should_log_aawm_alias_routing_event("key-a") is True
    assert should_log_aawm_alias_routing_event("key-a") is False


def test_should_log_different_keys_independent(runtime: RuntimeMemoryRuntime) -> None:
    assert should_log_aawm_alias_routing_event("key-a") is True
    assert should_log_aawm_alias_routing_event("key-b") is True


def test_should_log_expires_after_window(runtime: RuntimeMemoryRuntime) -> None:
    assert should_log_aawm_alias_routing_event("key-a") is True
    # Simulate window expiry by rewinding the stored until value.
    runtime.log_until_map["key-a"] = time.monotonic() - 1.0
    assert should_log_aawm_alias_routing_event("key-a") is True


def test_should_log_bounds_map(runtime: RuntimeMemoryRuntime) -> None:
    # Fill beyond max_size=8
    for i in range(12):
        should_log_aawm_alias_routing_event(f"key-{i}")
    assert len(runtime.log_until_map) <= 8


# ---------------------------------------------------------------------------
# replace_request_body_in_place
# ---------------------------------------------------------------------------


def test_replace_body_same_object_noop() -> None:
    body: dict[str, Any] = {"model": "x"}
    replace_request_body_in_place(body, body)
    assert body == {"model": "x"}


def test_replace_body_different_object() -> None:
    body: dict[str, Any] = {"model": "old", "extra": 1}
    updated: dict[str, Any] = {"model": "new"}
    replace_request_body_in_place(body, updated)
    assert body == {"model": "new"}
    # Original dict identity preserved.
    assert id(body) != id(updated)


def test_replace_body_is_sync() -> None:
    assert not inspect.iscoroutinefunction(replace_request_body_in_place)


# ---------------------------------------------------------------------------
# bound_aawm_alias_routing_memory_map
# ---------------------------------------------------------------------------


def test_bound_map_trims_fifo(runtime: RuntimeMemoryRuntime) -> None:
    cache: dict[str, int] = {f"k{i}": i for i in range(20)}
    bound_aawm_alias_routing_memory_map(cache)
    assert len(cache) == 8
    # Oldest keys removed.
    assert "k0" not in cache
    assert "k19" in cache


def test_bound_map_explicit_max_size() -> None:
    cache: dict[str, int] = {f"k{i}": i for i in range(10)}
    bound_aawm_alias_routing_memory_map(cache, max_size=3)
    assert len(cache) == 3


# ---------------------------------------------------------------------------
# hydrate_aawm_alias_routing_cooldown_memory
# ---------------------------------------------------------------------------


def test_hydrate_cooldown_future(runtime: RuntimeMemoryRuntime) -> None:
    mmap: dict[str, float] = {}
    future_epoch = time.time() + 60.0
    hydrate_aawm_alias_routing_cooldown_memory(
        memory_map=mmap,
        cooldown_key="ck",
        expires_at_epoch=future_epoch,
    )
    assert "ck" in mmap
    assert mmap["ck"] > time.monotonic()


def test_hydrate_cooldown_past_is_noop(runtime: RuntimeMemoryRuntime) -> None:
    mmap: dict[str, float] = {}
    past_epoch = time.time() - 10.0
    hydrate_aawm_alias_routing_cooldown_memory(
        memory_map=mmap,
        cooldown_key="ck",
        expires_at_epoch=past_epoch,
    )
    assert "ck" not in mmap


def test_hydrate_cooldown_does_not_downgrade(runtime: RuntimeMemoryRuntime) -> None:
    mmap: dict[str, float] = {}
    far_future = time.time() + 120.0
    near_future = time.time() + 10.0
    hydrate_aawm_alias_routing_cooldown_memory(
        memory_map=mmap, cooldown_key="ck", expires_at_epoch=far_future
    )
    first_until = mmap["ck"]
    hydrate_aawm_alias_routing_cooldown_memory(
        memory_map=mmap, cooldown_key="ck", expires_at_epoch=near_future
    )
    assert mmap["ck"] == first_until


# ---------------------------------------------------------------------------
# hydrate_aawm_alias_routing_affinity_memory
# ---------------------------------------------------------------------------


def test_hydrate_affinity_future(runtime: RuntimeMemoryRuntime) -> None:
    mmap: dict[str, dict[str, Any]] = {}
    result = hydrate_aawm_alias_routing_affinity_memory(
        memory_map=mmap,
        session_key="s1",
        payload={"provider": "openai", "model": "gpt-4o", "route_family": "rf"},
        expires_at_epoch=time.time() + 60.0,
    )
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o"
    assert "s1" in mmap


def test_hydrate_affinity_past_returns_empty(runtime: RuntimeMemoryRuntime) -> None:
    mmap: dict[str, dict[str, Any]] = {}
    result = hydrate_aawm_alias_routing_affinity_memory(
        memory_map=mmap,
        session_key="s1",
        payload={"provider": "openai"},
        expires_at_epoch=time.time() - 5.0,
    )
    assert result == {}
    assert "s1" not in mmap


def test_hydrate_affinity_does_not_clobber_fresher(runtime: RuntimeMemoryRuntime) -> None:
    mmap: dict[str, dict[str, Any]] = {}
    far_epoch = time.time() + 120.0
    near_epoch = time.time() + 10.0
    hydrate_aawm_alias_routing_affinity_memory(
        memory_map=mmap,
        session_key="s1",
        payload={"provider": "openai", "model": "a"},
        expires_at_epoch=far_epoch,
    )
    second = hydrate_aawm_alias_routing_affinity_memory(
        memory_map=mmap,
        session_key="s1",
        payload={"provider": "anthropic", "model": "b"},
        expires_at_epoch=near_epoch,
    )
    # Fresher entry wins.
    assert second["provider"] == "openai"
    assert second["model"] == "a"


def test_hydrate_affinity_is_sync() -> None:
    assert not inspect.iscoroutinefunction(hydrate_aawm_alias_routing_affinity_memory)
