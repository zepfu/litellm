"""D1-551 / D1-532: durable routing-state policy helper (Wave 0).

A durable read exception must not be treated as empty state. Last-good local
is returned as ``degraded_local``. A confirmed durable miss may still use a
valid local lease. Affinity and cooldown share one helper.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    cooldown_state as cooldown_state_mod,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as durable_mod


def _require_attr(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    assert value is not None, f"{module.__name__}.{name} is required for D1-551/D1-532"
    return value


def _result_source(result: Any) -> Optional[str]:
    if isinstance(result, dict):
        return (
            result.get("source")
            or result.get("state_source")
            or result.get("affinity_state_source")
        )
    return (
        getattr(result, "source", None)
        or getattr(result, "state_source", None)
        or getattr(result, "affinity_state_source", None)
    )


def _result_payload(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("payload", result)
    return getattr(result, "payload", result)


def _result_flag(result: Any, *names: str) -> Any:
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
        return None
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return None


def _local_affinity(*, model: str) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "route_family": "openai_responses",
        "last_resort": False,
        "expires_at_monotonic": time.monotonic() + 3600,
        "affinity_state_source": "memory",
    }


def _durable_affinity(*, model: str) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": model,
        "route_family": "openai_responses",
        "last_resort": False,
        "expires_at_epoch": time.time() + 3600,
    }


def _dual_cache(*, get_side_effect: Any = None, get_return: Any = None) -> MagicMock:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    if get_side_effect is not None:
        dual.async_get_cache = AsyncMock(side_effect=get_side_effect)
    else:
        dual.async_get_cache = AsyncMock(return_value=get_return)
    return dual


@pytest.mark.asyncio
async def test_durable_read_exception_is_not_treated_as_empty_state() -> None:
    """Redis/read errors must be distinguishable from a confirmed durable miss."""
    read_state = _require_attr(durable_mod, "read_aawm_alias_routing_state")
    last_good = _local_affinity(model="gpt-5.4-mini")
    dual = _dual_cache(get_side_effect=ConnectionError("redis read failed"))

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ):
        result = await read_state(
            alias_family="codex",
            state_kind="affinity",
            state_key="sess-d1551-error",
            last_good_local=last_good,
        )

    assert _result_source(result) == "degraded_local"
    payload = _result_payload(result)
    assert payload is not None
    assert payload.get("model") == "gpt-5.4-mini"
    assert _result_flag(result, "confirmed_miss", "durable_miss") is not True
    assert _result_flag(result, "durable_error", "read_error") is not False


@pytest.mark.asyncio
async def test_durable_read_exception_without_local_does_not_invent_a_pin() -> None:
    read_state = _require_attr(durable_mod, "read_aawm_alias_routing_state")
    dual = _dual_cache(get_side_effect=TimeoutError("redis timeout"))

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ):
        result = await read_state(
            alias_family="codex",
            state_kind="affinity",
            state_key="sess-d1551-error-empty",
            last_good_local=None,
        )

    assert _result_payload(result) in (None, {}, [])
    assert _result_source(result) in {"degraded_local", "durable_error", "unavailable"}
    assert _result_flag(result, "confirmed_miss", "durable_miss") is not True


@pytest.mark.asyncio
async def test_last_good_local_returned_with_degraded_source_on_read_error() -> None:
    read_state = _require_attr(durable_mod, "read_aawm_alias_routing_state")
    last_good = _local_affinity(model="gpt-5.3-codex-spark")
    dual = _dual_cache(get_side_effect=OSError("broken pipe"))

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ):
        result = await read_state(
            alias_family="anthropic",
            state_kind="cooldown",
            state_key="anthropic:claude:auth:d1551",
            last_good_local=last_good,
        )

    assert _result_source(result) == "degraded_local"
    assert _result_payload(result).get("model") == "gpt-5.3-codex-spark"


@pytest.mark.asyncio
async def test_confirmed_durable_miss_may_use_local_lease() -> None:
    read_state = _require_attr(durable_mod, "read_aawm_alias_routing_state")
    last_good = _local_affinity(model="gpt-5.4")
    dual = _dual_cache(get_return=None)

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ):
        result = await read_state(
            alias_family="codex",
            state_kind="affinity",
            state_key="sess-d1551-miss",
            last_good_local=last_good,
        )

    assert _result_flag(result, "confirmed_miss", "durable_miss") is True
    assert _result_flag(result, "durable_error", "read_error") is not True
    payload = _result_payload(result)
    assert payload is not None
    assert payload.get("model") == "gpt-5.4"
    assert _result_source(result) in {"memory", "local_lease"}


@pytest.mark.asyncio
async def test_durable_payload_wins_over_still_valid_local() -> None:
    read_state = _require_attr(durable_mod, "read_aawm_alias_routing_state")
    last_good = _local_affinity(model="gpt-5.4-mini")
    durable_payload = _durable_affinity(model="gpt-5.5")
    dual = _dual_cache(get_side_effect=[durable_payload])

    with patch.object(
        durable_mod, "get_aawm_alias_routing_dual_cache", return_value=dual
    ):
        result = await read_state(
            alias_family="codex",
            state_kind="affinity",
            state_key="sess-d1551-durable-wins",
            last_good_local=last_good,
        )

    payload = _result_payload(result)
    assert payload is not None
    assert payload.get("model") == "gpt-5.5"
    assert _result_source(result) == "durable_cache"


def test_affinity_and_cooldown_getters_share_the_wave0_helper() -> None:
    """D1-532: one helper, not a second copy-paste memory-first path."""
    helper_name = "read_aawm_alias_routing_state"
    affinity_src = inspect.getsource(
        cooldown_state_mod._get_codex_auto_agent_session_affinity
    )
    cooldown_src = inspect.getsource(
        cooldown_state_mod._get_anthropic_auto_agent_active_cooldown_state
    )
    assert helper_name in affinity_src
    assert helper_name in cooldown_src
    assert hasattr(durable_mod, helper_name)
