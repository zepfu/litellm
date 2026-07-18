"""RR-054 durable max-semantics tests.

Focused coverage for:
- cooldown durable writes with max-expiry (never shrink longer existing expiry)
- bounded retry on retryable Redis failures; no retry on non-retryable
- DualCache memory write-through after successful Redis SET
- affinity hydration race protection (do not clobber fresher in-memory state)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
    hydrate_affinity_memory,
    hydrate_cooldown_memory,
)


def _dual_cache(
    *,
    existing: Any = None,
    set_side_effect: Any = None,
    set_return: Any = None,
    mem_side_effect: Any = None,
) -> MagicMock:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    if set_side_effect is not None:
        dual.redis_cache.async_set_cache = AsyncMock(side_effect=set_side_effect)
    else:
        dual.redis_cache.async_set_cache = AsyncMock(return_value=set_return)
    dual.async_get_cache = AsyncMock(return_value=existing)
    if mem_side_effect is not None:
        dual.async_set_cache = AsyncMock(side_effect=mem_side_effect)
    else:
        dual.async_set_cache = AsyncMock(return_value=None)
    return dual


# ---------------------------------------------------------------------------
# Cooldown max-expiry writes (RR-054 #2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_durable_max_expiry_keeps_longer_existing_cooldown() -> None:
    existing_expires = time.time() + 3600.0
    dual = _dual_cache(
        existing={
            "cooldown_key": "cand-a",
            "failure_class": "capacity",
            "expires_at_epoch": existing_expires,
        }
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:max:keep-long",
    ):
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="cand-a",
            payload={
                "cooldown_key": "cand-a",
                "failure_class": "rate_limit",
            },
            ttl_seconds=30.0,
        )

    assert ok is True
    kwargs = dual.redis_cache.async_set_cache.await_args.kwargs
    written = kwargs["value"]
    assert written["expires_at_epoch"] == pytest.approx(existing_expires, abs=1.0)
    assert kwargs["ttl"] == pytest.approx(3600.0, abs=2.0)
    # Non-expiry payload fields from the new write are merged onto the longer
    # existing durable record (max-expiry, not max-payload-freeze).
    assert written["cooldown_key"] == "cand-a"
    assert written["failure_class"] == "rate_limit"
    dual.async_set_cache.assert_awaited()
    assert dual.async_set_cache.await_args.kwargs["value"] == written


@pytest.mark.asyncio
async def test_rr054_durable_max_expiry_extends_when_new_ttl_is_longer() -> None:
    existing_expires = time.time() + 10.0
    dual = _dual_cache(
        existing={
            "cooldown_key": "cand-b",
            "failure_class": "capacity",
            "expires_at_epoch": existing_expires,
        }
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:max:extend",
    ):
        before = time.time()
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="anthropic",
            state_kind="cooldown",
            state_key="cand-b",
            payload={
                "cooldown_key": "cand-b",
                "failure_class": "usage_limit",
            },
            ttl_seconds=120.0,
        )
        after = time.time()

    assert ok is True
    written = dual.redis_cache.async_set_cache.await_args.kwargs["value"]
    assert written["expires_at_epoch"] >= before + 119.0
    assert written["expires_at_epoch"] <= after + 121.0
    assert written["failure_class"] == "usage_limit"
    # Existing non-expiry fields survive when the new write extends expiry.
    assert written["cooldown_key"] == "cand-b"
    assert dual.redis_cache.async_set_cache.await_args.kwargs["ttl"] == pytest.approx(
        120.0, abs=1.0
    )


@pytest.mark.asyncio
async def test_rr054_durable_max_expiry_fresh_write_when_no_existing() -> None:
    dual = _dual_cache(existing=None)

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:max:fresh",
    ):
        before = time.time()
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="cand-fresh",
            payload={"cooldown_key": "cand-fresh"},
            ttl_seconds=45.0,
        )
        after = time.time()

    assert ok is True
    written = dual.redis_cache.async_set_cache.await_args.kwargs["value"]
    assert written["cooldown_key"] == "cand-fresh"
    assert written["expires_at_epoch"] >= before + 44.0
    assert written["expires_at_epoch"] <= after + 46.0
    assert dual.redis_cache.async_set_cache.await_args.kwargs["ttl"] == pytest.approx(
        45.0, abs=1.0
    )


@pytest.mark.asyncio
async def test_rr054_durable_max_expiry_via_lpe_reexport() -> None:
    """God-file reexport must keep the same max-expiry semantics."""
    existing_expires = time.time() + 1800.0
    dual = _dual_cache(
        existing={"cooldown_key": "k", "expires_at_epoch": existing_expires}
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:max:lpe",
    ):
        ok = await lpe._write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="k",
            payload={"cooldown_key": "k"},
            ttl_seconds=15.0,
        )

    assert ok is True
    written = dual.redis_cache.async_set_cache.await_args.kwargs["value"]
    assert written["expires_at_epoch"] == pytest.approx(existing_expires, abs=1.0)


# ---------------------------------------------------------------------------
# Retryable Redis failures (bounded retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_durable_write_retries_once_on_timeout_then_succeeds() -> None:
    dual = _dual_cache(
        existing=None,
        set_side_effect=[TimeoutError("redis timed out"), None],
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:retry:timeout-ok",
    ), patch.object(
        durable_mod, "get_durable_write_retry_attempts", return_value=1
    ), patch.object(
        durable_mod, "get_durable_write_retry_backoff_seconds", return_value=0.05
    ), patch.object(
        durable_mod.asyncio, "sleep", new=AsyncMock()
    ) as mock_sleep:
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="retry-timeout",
            payload={"cooldown_key": "retry-timeout"},
            ttl_seconds=30.0,
        )

    assert ok is True
    assert dual.redis_cache.async_set_cache.await_count == 2
    assert mock_sleep.await_count == 1
    assert mock_sleep.await_args.args == (0.05,)
    # Memory write-through only after successful Redis write.
    dual.async_set_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_rr054_durable_write_retries_once_on_connection_error_then_succeeds() -> (
    None
):
    dual = _dual_cache(
        existing=None,
        set_side_effect=[ConnectionError("redis connection reset"), None],
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:retry:conn-ok",
    ), patch.object(
        durable_mod, "get_durable_write_retry_attempts", return_value=1
    ), patch.object(
        durable_mod, "get_durable_write_retry_backoff_seconds", return_value=0.01
    ), patch.object(
        durable_mod.asyncio, "sleep", new=AsyncMock()
    ) as mock_sleep:
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="anthropic",
            state_kind="cooldown",
            state_key="retry-conn",
            payload={"cooldown_key": "retry-conn"},
            ttl_seconds=20.0,
        )

    assert ok is True
    assert dual.redis_cache.async_set_cache.await_count == 2
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
async def test_rr054_durable_write_exhausts_retry_on_repeated_timeout() -> None:
    dual = _dual_cache(
        existing=None,
        set_side_effect=[TimeoutError("t1"), TimeoutError("t2")],
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:retry:exhaust",
    ), patch.object(
        durable_mod, "get_durable_write_retry_attempts", return_value=1
    ), patch.object(
        durable_mod, "get_durable_write_retry_backoff_seconds", return_value=0.05
    ), patch.object(
        durable_mod.asyncio, "sleep", new=AsyncMock()
    ) as mock_sleep:
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="retry-exhaust",
            payload={"cooldown_key": "retry-exhaust"},
            ttl_seconds=30.0,
        )

    assert ok is False
    assert dual.redis_cache.async_set_cache.await_count == 2
    assert mock_sleep.await_count == 1
    dual.async_set_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_rr054_durable_write_does_not_retry_non_retryable_error() -> None:
    dual = _dual_cache(
        existing=None,
        set_side_effect=RuntimeError("redis write failed"),
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:retry:nonretry",
    ), patch.object(
        durable_mod, "get_durable_write_retry_attempts", return_value=1
    ), patch.object(
        durable_mod, "get_durable_write_retry_backoff_seconds", return_value=0.05
    ), patch.object(
        durable_mod.asyncio, "sleep", new=AsyncMock()
    ) as mock_sleep:
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="retry-nonretry",
            payload={"cooldown_key": "retry-nonretry"},
            ttl_seconds=30.0,
        )

    assert ok is False
    dual.redis_cache.async_set_cache.assert_awaited_once()
    assert mock_sleep.await_count == 0
    dual.async_set_cache.assert_not_awaited()


def test_rr054_is_retryable_redis_error_classification() -> None:
    from litellm.proxy.aawm_alias_routing_redis import is_retryable_redis_error

    assert is_retryable_redis_error(TimeoutError("t")) is True
    assert is_retryable_redis_error(ConnectionError("c")) is True
    assert is_retryable_redis_error(RuntimeError("x")) is False
    assert is_retryable_redis_error(ValueError("bad")) is False


# ---------------------------------------------------------------------------
# Memory write-through (DualCache coherency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_durable_write_through_dualcache_memory_matches_redis() -> None:
    dual = _dual_cache(existing=None)

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:wt:match",
    ):
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="wt-match",
            payload={"cooldown_key": "wt-match"},
            ttl_seconds=45.0,
        )

    assert ok is True
    dual.redis_cache.async_set_cache.assert_awaited()
    dual.async_set_cache.assert_awaited()
    redis_kwargs = dual.redis_cache.async_set_cache.await_args.kwargs
    mem_kwargs = dual.async_set_cache.await_args.kwargs
    assert mem_kwargs["key"] == redis_kwargs["key"] == "rr054:wt:match"
    assert mem_kwargs["value"] == redis_kwargs["value"]
    assert mem_kwargs["value"]["cooldown_key"] == "wt-match"
    assert "expires_at_epoch" in mem_kwargs["value"]
    assert mem_kwargs["ttl"] == pytest.approx(redis_kwargs["ttl"], abs=1.0)


@pytest.mark.asyncio
async def test_rr054_durable_write_through_falls_back_to_sync_set_cache() -> None:
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
        return_value="rr054:wt:sync",
    ):
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="anthropic",
            state_kind="affinity",
            state_key="wt-sync",
            payload={"provider": "anthropic", "model": "claude-x"},
            ttl_seconds=20.0,
        )

    assert ok is True
    dual.redis_cache.async_set_cache.assert_awaited()
    dual.set_cache.assert_called_once()
    call_kwargs = dual.set_cache.call_args.kwargs
    assert call_kwargs["key"] == "rr054:wt:sync"
    assert call_kwargs["value"]["provider"] == "anthropic"
    assert "expires_at_epoch" in call_kwargs["value"]


@pytest.mark.asyncio
async def test_rr054_memory_write_through_failure_does_not_fail_redis_write() -> None:
    dual = _dual_cache(
        existing=None,
        mem_side_effect=RuntimeError("memory layer down"),
    )

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch.object(
        durable_mod,
        "build_aawm_alias_routing_durable_cache_key",
        return_value="rr054:wt:mem-fail",
    ):
        ok = await durable_mod.write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="wt-mem-fail",
            payload={"cooldown_key": "wt-mem-fail"},
            ttl_seconds=10.0,
        )

    assert ok is True
    dual.redis_cache.async_set_cache.assert_awaited()


# ---------------------------------------------------------------------------
# Affinity hydration race protection
# ---------------------------------------------------------------------------


def test_rr054_affinity_hydrate_keeps_fresher_in_memory_state() -> None:
    """Stale Redis payload must not clobber a fresher local affinity write."""
    memory: dict[str, dict[str, Any]] = {
        "session-1": {
            "provider": "local",
            "model": "m-local",
            "route_family": "codex",
            "last_resort": False,
            "expires_at_monotonic": time.monotonic() + 9999.0,
        }
    }
    out = hydrate_affinity_memory(
        memory_map=memory,
        session_key="session-1",
        payload={
            "provider": "remote",
            "model": "m-remote",
            "route_family": "codex",
            "last_resort": False,
        },
        expires_at_epoch=time.time() + 10.0,
    )
    assert out["provider"] == "local"
    assert out["model"] == "m-local"
    assert memory["session-1"]["provider"] == "local"


def test_rr054_affinity_hydrate_applies_when_memory_is_staler() -> None:
    memory: dict[str, dict[str, Any]] = {
        "session-2": {
            "provider": "stale-local",
            "model": "m-stale",
            "route_family": "anthropic",
            "last_resort": False,
            "expires_at_monotonic": time.monotonic() + 5.0,
        }
    }
    out = hydrate_affinity_memory(
        memory_map=memory,
        session_key="session-2",
        payload={
            "provider": "fresh-remote",
            "model": "m-fresh",
            "route_family": "anthropic",
            "last_resort": True,
        },
        expires_at_epoch=time.time() + 120.0,
    )
    assert out["provider"] == "fresh-remote"
    assert out["model"] == "m-fresh"
    assert out["last_resort"] is True
    assert memory["session-2"]["provider"] == "fresh-remote"
    assert memory["session-2"]["expires_at_monotonic"] > time.monotonic()


def test_rr054_affinity_hydrate_applies_when_memory_empty() -> None:
    memory: dict[str, dict[str, Any]] = {}
    out = hydrate_affinity_memory(
        memory_map=memory,
        session_key="session-empty",
        payload={
            "provider": "openai",
            "model": "gpt-x",
            "route_family": "codex",
            "last_resort": False,
        },
        expires_at_epoch=time.time() + 60.0,
    )
    assert out["provider"] == "openai"
    assert "session-empty" in memory
    assert memory["session-empty"]["model"] == "gpt-x"


def test_rr054_affinity_hydrate_race_protection_via_lpe_reexport() -> None:
    memory: dict[str, dict[str, Any]] = {
        "s1": {
            "provider": "local",
            "model": "m1",
            "route_family": "r",
            "last_resort": False,
            "expires_at_monotonic": time.monotonic() + 9999.0,
        }
    }
    out = lpe._hydrate_aawm_alias_routing_affinity_memory(
        memory_map=memory,
        session_key="s1",
        payload={
            "provider": "remote",
            "model": "m2",
            "route_family": "r",
            "last_resort": False,
        },
        expires_at_epoch=time.time() + 10.0,
    )
    assert out["provider"] == "local"
    assert memory["s1"]["provider"] == "local"


def test_rr054_cooldown_hydrate_also_uses_max_semantics() -> None:
    """Cooldown memory hydration must not shrink a longer local until."""
    key = "cand-cd"
    memory: dict[str, float] = {key: time.monotonic() + 500.0}
    before = memory[key]
    hydrate_cooldown_memory(
        memory_map=memory,
        cooldown_key=key,
        expires_at_epoch=time.time() + 10.0,
    )
    assert memory[key] == before

    hydrate_cooldown_memory(
        memory_map=memory,
        cooldown_key=key,
        expires_at_epoch=time.time() + 900.0,
    )
    assert memory[key] > before
