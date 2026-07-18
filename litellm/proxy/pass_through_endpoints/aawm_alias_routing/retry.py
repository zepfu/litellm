"""Shared adapter cooldown / hidden-retry primitives (RR-054 #12).

Preserves provider-specific multi-budget semantics (e.g. Google capacity vs
rate-limit vs transient) by only extracting map ops and wait-key helpers that
are genuinely shared.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, Sequence, TypeVar, Union

from .memory import DEFAULT_MEMORY_STATE_MAX_SIZE
from .state import MonotonicCooldownMap

logger = logging.getLogger("LiteLLMProxy")
RetryResultT = TypeVar("RetryResultT")


@dataclass(frozen=True)
class AdapterRetryPolicy(Generic[RetryResultT]):
    """Strategy hooks for the shared cross-family retry sequence."""

    before_attempt: Callable[[int], Awaitable[None]]
    on_failure: Callable[[Exception, int], Awaitable[bool]]
    on_success: Optional[Callable[[RetryResultT, int], Awaitable[None]]] = None


async def run_adapter_retry_policy(
    operation: Callable[[], Awaitable[RetryResultT]],
    *,
    policy: AdapterRetryPolicy[RetryResultT],
) -> RetryResultT:
    """Own the common attempt/wait/execute/classify/retry sequence.

    Provider strategies retain classification, budget accounting, cooldown
    side effects, and retry decisions. This preserves Google's independent
    capacity, generic rate-limit, and transient budgets.
    """
    attempt = 0
    while True:
        attempt += 1
        await policy.before_attempt(attempt)
        try:
            result = await operation()
        except Exception as exc:
            if await policy.on_failure(exc, attempt):
                continue
            raise
        if policy.on_success is not None:
            await policy.on_success(result, attempt)
        return result


def normalize_cooldown_keys(
    rate_limit_keys: Union[str, Sequence[str], None],
    *,
    default_key: str = "__default__",
) -> list[str]:
    if isinstance(rate_limit_keys, str):
        keys = [rate_limit_keys] if rate_limit_keys else []
    elif rate_limit_keys is None:
        keys = []
    else:
        keys = [key for key in rate_limit_keys if isinstance(key, str) and key]
    return keys or [default_key]


async def wait_for_monotonic_cooldown_map(
    cooldown_map: MonotonicCooldownMap,
    rate_limit_keys: Union[str, Sequence[str], None],
    *,
    log_label: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_active: Optional[Callable[[list[str], float], Awaitable[None]]] = None,
    default_key: str = "__default__",
) -> float:
    """Sleep until all keys in ``cooldown_map`` are clear. Returns waited seconds."""
    keys = normalize_cooldown_keys(rate_limit_keys, default_key=default_key)
    async with cooldown_map.lock:
        wait_seconds = cooldown_map.max_remaining(keys)
    if wait_seconds <= 0:
        return 0.0
    if on_active is not None:
        await on_active(keys, wait_seconds)
    logger.warning(
        "%s cooldown active for %s; sleeping %.1fs before upstream request",
        log_label,
        ", ".join(keys),
        wait_seconds,
    )
    await sleep(wait_seconds)
    return wait_seconds


async def set_monotonic_cooldown_map(
    cooldown_map: MonotonicCooldownMap,
    rate_limit_keys: Union[str, Sequence[str], None],
    wait_seconds: float,
    *,
    max_size: Optional[int] = DEFAULT_MEMORY_STATE_MAX_SIZE,
    default_key: str = "__default__",
) -> None:
    keys = normalize_cooldown_keys(rate_limit_keys, default_key=default_key)
    async with cooldown_map.lock:
        for key in keys:
            cooldown_map.extend(key, wait_seconds, max_size=max_size)


def projected_hidden_retry_within_budget(
    *,
    accumulated_hidden_wait_seconds: float,
    next_wait_seconds: float,
    hidden_retry_budget_seconds: float,
) -> tuple[float, bool]:
    """Return (projected_total, within_budget) for hidden client-side retries."""
    projected = accumulated_hidden_wait_seconds + max(0.0, float(next_wait_seconds))
    within = (
        hidden_retry_budget_seconds > 0
        and projected <= float(hidden_retry_budget_seconds)
    )
    return projected, within


def parse_non_negative_float_env(
    env_name: str,
    *,
    default: float,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
    getenv=None,
) -> float:
    """Shared env float parse for adapter retry/backoff budgets (RR-054 #12)."""
    import os

    raw = (getenv or os.getenv)(env_name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except Exception:
        return default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def parse_non_negative_int_env(
    env_name: str,
    *,
    default: int,
    minimum: int = 0,
    maximum: Optional[int] = None,
    getenv=None,
) -> int:
    import os

    raw = (getenv or os.getenv)(env_name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except Exception:
        return default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def exponential_backoff_seconds(
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float,
    jitter_seconds: float = 0.0,
) -> float:
    """Generic exponential backoff used by adapter hidden-retry loops."""
    import random

    attempt = max(1, int(attempt))
    delay = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
    if jitter_seconds > 0:
        delay += random.uniform(0.0, jitter_seconds)
    return max(0.0, float(delay))
