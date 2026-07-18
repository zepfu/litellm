"""RR-054 operational residual behavioral tests (#17/#18, #26, #30, #42).

Production-only assertions for map bounds, DualCache write-through coherency,
cooldown negative-cache repeat reads, and async terminal persistence.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
    bound_memory_map,
)


# ---------------------------------------------------------------------------
# #17 / #18 map bounds (cooldown + session affinity for both families)
# ---------------------------------------------------------------------------


def test_rr054_issue17_18_bound_memory_map_fifo_trims_oldest() -> None:
    cache: dict[str, float] = {f"k{i}": float(i) for i in range(6)}
    bound_memory_map(cache, max_size=3)
    assert len(cache) == 3
    assert "k0" not in cache
    assert "k1" not in cache
    assert "k2" not in cache
    assert cache == {"k3": 3.0, "k4": 4.0, "k5": 5.0}


def test_rr054_issue17_18_godfile_bound_helper_trims_maps() -> None:
    cache: dict[str, Any] = {f"s{i}": {"model": f"m{i}"} for i in range(5)}
    lpe._bound_aawm_alias_routing_memory_map(cache, max_size=2)
    assert len(cache) == 2
    assert "s0" not in cache
    assert "s4" in cache


@pytest.mark.asyncio
async def test_rr054_issue17_18_codex_cooldown_set_bounds_map() -> None:
    key_prefix = "rr054-17-18-codex-cd"
    maps_to_clear = [
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key,
        lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key,
    ]
    for m in maps_to_clear:
        m.clear()

    with patch.object(
        lpe,
        "_write_aawm_alias_routing_durable_payload",
        new=AsyncMock(return_value=True),
    ):
        async def _set_with_bound(cooldown_key: str, cooldown_seconds: float) -> None:
            ttl_seconds = max(0.0, float(cooldown_seconds))
            async with lpe._codex_auto_agent_lock:
                until = time.monotonic() + ttl_seconds
                current_until = lpe._codex_auto_agent_cooldown_until_monotonic_by_key.get(
                    cooldown_key, 0.0
                )
                if until > current_until:
                    lpe._codex_auto_agent_cooldown_until_monotonic_by_key[
                        cooldown_key
                    ] = until
                    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.pop(
                        cooldown_key, None
                    )
                    bound_memory_map(
                        lpe._codex_auto_agent_cooldown_until_monotonic_by_key,
                        max_size=3,
                    )
            if ttl_seconds > 0:
                await lpe._write_aawm_alias_routing_durable_payload(
                    alias_family="codex",
                    state_kind="cooldown",
                    state_key=cooldown_key,
                    payload={"cooldown_key": cooldown_key},
                    ttl_seconds=ttl_seconds,
                )

        for i in range(5):
            await _set_with_bound(f"{key_prefix}-{i}", 30.0)

    assert len(lpe._codex_auto_agent_cooldown_until_monotonic_by_key) == 3
    assert f"{key_prefix}-0" not in lpe._codex_auto_agent_cooldown_until_monotonic_by_key
    assert f"{key_prefix}-4" in lpe._codex_auto_agent_cooldown_until_monotonic_by_key
    for m in maps_to_clear:
        m.clear()


@pytest.mark.asyncio
async def test_rr054_issue17_18_anthropic_and_codex_affinity_sets_bound() -> None:
    candidate = {
        "provider": "openai",
        "model": "gpt-test",
        "route_family": "codex_openai_responses",
        "last_resort": False,
    }
    anth_candidate = {
        "provider": "anthropic",
        "model": "claude-test",
        "route_family": "anthropic_native",
        "last_resort": False,
    }
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()

    with patch.object(
        lpe,
        "_write_aawm_alias_routing_durable_payload",
        new=AsyncMock(return_value=True),
    ):
        # Seed beyond default max so bound_memory_map path is exercised on writes.
        for i in range(lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE + 3):
            await lpe._set_codex_auto_agent_session_affinity(
                f"rr054-aff-codex-{i}", candidate
            )
            await lpe._set_anthropic_auto_agent_session_affinity(
                f"rr054-aff-anth-{i}", anth_candidate
            )

    assert (
        len(lpe._codex_auto_agent_session_affinity_by_key)
        <= lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE
    )
    assert (
        len(lpe._anthropic_auto_agent_session_affinity_by_key)
        <= lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE
    )
    # Oldest FIFO keys should be gone; newest retained.
    assert "rr054-aff-codex-0" not in lpe._codex_auto_agent_session_affinity_by_key
    assert (
        f"rr054-aff-codex-{lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE + 2}"
        in lpe._codex_auto_agent_session_affinity_by_key
    )
    assert "rr054-aff-anth-0" not in lpe._anthropic_auto_agent_session_affinity_by_key
    assert (
        f"rr054-aff-anth-{lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE + 2}"
        in lpe._anthropic_auto_agent_session_affinity_by_key
    )

    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe._anthropic_auto_agent_session_affinity_by_key.clear()


@pytest.mark.asyncio
async def test_rr054_issue17_18_anthropic_cooldown_set_bounds_via_real_helper() -> None:
    """Anthropic cooldown writes call the shared bound helper on the process map."""
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key.clear()

    bound_calls: list[dict] = []

    def _tracking_bound(cache, *, max_size=None, **kwargs):
        bound_calls.append(cache)
        size = (
            max_size
            if max_size is not None
            else lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE
        )
        bound_memory_map(cache, max_size=size)

    with patch.object(
        lpe,
        "_write_aawm_alias_routing_durable_payload",
        new=AsyncMock(return_value=True),
    ), patch.object(
        lpe, "_bound_aawm_alias_routing_memory_map", side_effect=_tracking_bound
    ):
        await lpe._set_anthropic_auto_agent_cooldown("rr054-anth-cd-a", 12.0)
        await lpe._set_anthropic_auto_agent_cooldown("rr054-anth-cd-b", 12.0)

    assert bound_calls
    assert any("rr054-anth-cd-b" in c for c in bound_calls if isinstance(c, dict))
    assert "rr054-anth-cd-a" in lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key.clear()


# ---------------------------------------------------------------------------
# #26 DualCache coherency (write-through memory after redis write)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_issue26_durable_write_writes_through_dualcache_memory() -> None:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    dual.async_get_cache = AsyncMock(return_value=None)
    dual.async_set_cache = AsyncMock(return_value=None)

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:dual:cache-key",
    ):
        ok = await lpe._write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="rr054-dual-coherency",
            payload={"cooldown_key": "rr054-dual-coherency"},
            ttl_seconds=45.0,
        )

    assert ok is True
    dual.redis_cache.async_set_cache.assert_awaited()
    dual.async_set_cache.assert_awaited()
    redis_kwargs = dual.redis_cache.async_set_cache.await_args.kwargs
    mem_kwargs = dual.async_set_cache.await_args.kwargs
    assert mem_kwargs["key"] == redis_kwargs["key"] == "rr054:dual:cache-key"
    assert mem_kwargs["value"] == redis_kwargs["value"]
    assert mem_kwargs["value"]["cooldown_key"] == "rr054-dual-coherency"
    assert "expires_at_epoch" in mem_kwargs["value"]
    assert mem_kwargs["ttl"] == pytest.approx(redis_kwargs["ttl"], abs=1.0)


@pytest.mark.asyncio
async def test_rr054_issue26_durable_write_falls_back_to_sync_set_cache() -> None:
    dual = MagicMock(spec=["redis_cache", "async_get_cache", "set_cache"])
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    dual.async_get_cache = AsyncMock(return_value=None)
    dual.set_cache = MagicMock(return_value=None)

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:dual:sync-key",
    ):
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="anthropic",
            state_kind="affinity",
            state_key="rr054-sync-set",
            payload={"provider": "anthropic", "model": "claude-x"},
            ttl_seconds=20.0,
        )

    assert ok is True
    dual.redis_cache.async_set_cache.assert_awaited()
    dual.set_cache.assert_called_once()
    call_kwargs = dual.set_cache.call_args.kwargs
    assert call_kwargs["key"] == "rr054:dual:sync-key"
    assert call_kwargs["value"]["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_rr054_issue26_memory_write_through_failure_does_not_fail_redis_write() -> None:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    dual.async_get_cache = AsyncMock(return_value=None)
    dual.async_set_cache = AsyncMock(side_effect=RuntimeError("memory layer down"))

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:dual:mem-fail",
    ):
        ok = await lpe._write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="rr054-mem-fail",
            payload={"cooldown_key": "rr054-mem-fail"},
            ttl_seconds=10.0,
        )

    assert ok is True
    dual.redis_cache.async_set_cache.assert_awaited()


# ---------------------------------------------------------------------------
# #30 negative-cache repeat read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_issue30_codex_negative_cache_skips_repeat_redis_read() -> None:
    key = "rr054-neg-cache-codex"
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(key, None)
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.pop(key, None)

    dual = MagicMock()
    dual.redis_cache = MagicMock()
    read_mock = AsyncMock(return_value=None)

    with patch.object(
        lpe, "_get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        lpe, "_read_aawm_alias_routing_durable_payload", new=read_mock
    ), patch.object(lpe, "_AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS", 30.0):
        first = await lpe._get_codex_auto_agent_active_cooldown_state(key)
        second = await lpe._get_codex_auto_agent_active_cooldown_state(key)

    assert first == (0.0, "local_fallback")
    assert second == (0.0, "negative_cache")
    assert read_mock.await_count == 1
    assert key in lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key

    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.pop(key, None)


@pytest.mark.asyncio
async def test_rr054_issue30_anthropic_negative_cache_skips_repeat_redis_read() -> None:
    key = "rr054-neg-cache-anth"
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.pop(key, None)
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key.pop(key, None)

    dual = MagicMock()
    dual.redis_cache = MagicMock()
    read_mock = AsyncMock(return_value=None)

    with patch.object(
        lpe, "_get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        lpe, "_read_aawm_alias_routing_durable_payload", new=read_mock
    ), patch.object(lpe, "_AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS", 30.0):
        first = await lpe._get_anthropic_auto_agent_active_cooldown_state(key)
        second = await lpe._get_anthropic_auto_agent_active_cooldown_state(key)

    assert first == (0.0, "local_fallback")
    assert second == (0.0, "negative_cache")
    assert read_mock.await_count == 1
    assert key in lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key

    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key.pop(key, None)


@pytest.mark.asyncio
async def test_rr054_issue30_positive_cooldown_clears_negative_cache() -> None:
    key = "rr054-neg-then-set"
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(key, None)
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key[key] = (
        time.monotonic() + 60.0
    )

    with patch.object(
        lpe,
        "_write_aawm_alias_routing_durable_payload",
        new=AsyncMock(return_value=True),
    ):
        await lpe._set_codex_auto_agent_cooldown(key, 25.0)

    assert key not in lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key
    seconds, source = await lpe._get_codex_auto_agent_active_cooldown_state(key)
    assert seconds > 0
    assert source == "memory"

    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(key, None)
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key.pop(key, None)


# ---------------------------------------------------------------------------
# #42 async terminal persistence (do not block event loop)
# ---------------------------------------------------------------------------


def _build_minimal_request() -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/openai_passthrough/responses",
        "raw_path": b"/openai_passthrough/responses",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_rr054_issue42_terminal_persist_uses_executor_when_loop_running() -> None:
    request = _build_minimal_request()
    body = {
        "model": "aawm-code",
        "litellm_metadata": {
            "session_id": "rr054-terminal-session",
            "agent_id": "agent-rr054-terminal",
        },
    }
    persisted: list[dict[str, Any]] = []
    executor_jobs: list[Any] = []

    def _fake_persist(**kwargs: Any) -> bool:
        persisted.append(kwargs)
        return True

    real_loop = asyncio.get_running_loop()

    def _tracking_run_in_executor(executor, fn, *args):
        executor_jobs.append((executor, fn, args))
        # Execute immediately so assertions can observe the payload without thread races.
        result = fn(*args) if args else fn()
        fut = real_loop.create_future()
        fut.set_result(result)
        return fut

    mock_loop = MagicMock()
    mock_loop.run_in_executor = MagicMock(side_effect=_tracking_run_in_executor)

    with patch.object(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        return_value=None,
    ), patch.object(
        lpe, "_emit_auto_agent_alias_route_event", return_value=None
    ), patch(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        side_effect=_fake_persist,
    ), patch.object(asyncio, "get_running_loop", return_value=mock_loop):
        lpe._emit_auto_agent_alias_no_candidate_event(
            alias_family="codex_auto_agent",
            alias_model="aawm-code",
            request=request,
            request_body=body,
            exc=HTTPException(
                status_code=429,
                detail={"candidates": [], "error": {"code": "all_unavailable"}},
            ),
            attempts=[
                {
                    "provider": "openai",
                    "model": "gpt-test",
                    "route_family": "codex_openai_responses",
                    "error_class": "rate_limit",
                }
            ],
        )

    assert executor_jobs, "persist_agent_terminal_error must be scheduled via executor"
    assert mock_loop.run_in_executor.call_count == 1
    # First arg is executor (None); second is the zero-arg callable.
    assert mock_loop.run_in_executor.call_args.args[0] is None
    assert callable(mock_loop.run_in_executor.call_args.args[1])
    assert len(persisted) == 1
    terminal = persisted[0]
    assert terminal["terminal_outcome"] == "agent_session_terminated"
    assert terminal["fallback_result"] == "no_candidate_available"
    assert terminal["redispatch_required"] is False
    assert terminal["agent_session_killed"] is True
    assert terminal["error_context"]["failure_kind"] == "agent_alias_no_candidate"
    assert terminal["error_context"]["model_alias"] == "aawm-code"


def test_rr054_issue42_terminal_persist_source_uses_run_in_executor() -> None:
    import inspect

    source = inspect.getsource(lpe._emit_auto_agent_alias_no_candidate_event)
    assert "run_in_executor" in source
    assert "persist_agent_terminal_error" in source
    # Must not only call persist synchronously on the hot path when a loop exists.
    assert "get_running_loop" in source
