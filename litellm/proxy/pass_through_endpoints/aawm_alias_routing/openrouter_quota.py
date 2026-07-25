"""OpenRouter free-daily-quota probe, durable cooldown helpers, and alias-probe gate.

Wave 5A extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

The quota cache tuple and lock (``_openrouter_free_daily_quota_cache``,
``_openrouter_free_daily_quota_lock``) remain owned by the god module for
Wave 5B.  Cache reads/writes go through injected getter/setter callbacks;
the lock is injected as a shared object reference.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Never, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException

from .policy import (
    CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
    OPENROUTER_FREE_DAILY_QUOTA_MODELS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OPENROUTER_DURABLE_QUOTA_DAILY_KEY = "openrouter_free_daily_requests:requests"
_OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS = 30.0
_OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS = 0.5
_OPENROUTER_FREE_DAILY_QUOTA_MODELS = OPENROUTER_FREE_DAILY_QUOTA_MODELS

# ---------------------------------------------------------------------------
# Injected runtime seams (deferred to Wave 5B for state-manager ownership)
# ---------------------------------------------------------------------------
_get_quota_cache: Optional[Callable[[], tuple[Optional[float], float]]] = None
_set_quota_cache: Optional[Callable[[tuple[Optional[float], float]], None]] = None
_quota_lock: Optional[asyncio.Lock] = None
_get_dynamic_injection_pool: Optional[Callable[[], Awaitable[Any]]] = None
_get_adapter_active_cooldown_seconds: Optional[Callable[[Optional[str]], Awaitable[float]]] = None
_get_adapter_rate_limit_key: Optional[Callable[[Optional[str]], str]] = None
_fetch_quota_row: Optional[Callable[[], Awaitable[Optional[Any]]]] = None
_get_free_daily_quota_exhausted_cooldown_seconds: Optional[
    Callable[[], Awaitable[float]]
] = None


def configure_openrouter_quota_runtime(
    *,
    get_quota_cache: Callable[[], tuple[Optional[float], float]],
    set_quota_cache: Callable[[tuple[Optional[float], float]], None],
    quota_lock: asyncio.Lock,
    get_dynamic_injection_pool: Callable[[], Awaitable[Any]],
    get_adapter_active_cooldown_seconds: Callable[[Optional[str]], Awaitable[float]],
    get_adapter_rate_limit_key: Callable[[Optional[str]], str],
    fetch_quota_row: Callable[[], Awaitable[Optional[Any]]],
    get_free_daily_quota_exhausted_cooldown_seconds: Callable[
        [], Awaitable[float]
    ],
) -> None:
    """Bind god-module-owned quota cache accessors and adapter helpers."""
    global _get_quota_cache, _set_quota_cache, _quota_lock
    global _get_dynamic_injection_pool
    global _get_adapter_active_cooldown_seconds, _get_adapter_rate_limit_key
    global _fetch_quota_row
    global _get_free_daily_quota_exhausted_cooldown_seconds
    _get_quota_cache = get_quota_cache
    _set_quota_cache = set_quota_cache
    _quota_lock = quota_lock
    _get_dynamic_injection_pool = get_dynamic_injection_pool
    _get_adapter_active_cooldown_seconds = get_adapter_active_cooldown_seconds
    _get_adapter_rate_limit_key = get_adapter_rate_limit_key
    _fetch_quota_row = fetch_quota_row
    _get_free_daily_quota_exhausted_cooldown_seconds = (
        get_free_daily_quota_exhausted_cooldown_seconds
    )


def _reset_openrouter_free_daily_quota_cache() -> None:
    assert _set_quota_cache is not None
    _set_quota_cache((None, 0.0))


def _parse_openrouter_free_daily_quota_reset_timestamp(
    expected_reset_at: Any,
) -> Optional[float]:
    if isinstance(expected_reset_at, (int, float)):
        return float(expected_reset_at)
    if isinstance(expected_reset_at, datetime):
        reset_dt = expected_reset_at
        if reset_dt.tzinfo is None:
            reset_dt = reset_dt.replace(tzinfo=timezone.utc)
        return reset_dt.timestamp()
    if isinstance(expected_reset_at, str):
        raw_reset = expected_reset_at.strip()
        if not raw_reset:
            return None
        if raw_reset.endswith("Z"):
            raw_reset = raw_reset[:-1] + "+00:00"
        try:
            reset_dt = datetime.fromisoformat(raw_reset)
        except (TypeError, ValueError):
            return None
        if reset_dt.tzinfo is None:
            reset_dt = reset_dt.replace(tzinfo=timezone.utc)
        return reset_dt.timestamp()
    return None


# ---------------------------------------------------------------------------
# Quota probe / fetch
# ---------------------------------------------------------------------------


async def _fetch_openrouter_free_daily_quota_row() -> Optional[Any]:
    """Direct DB fetch (used when no injected callback is configured)."""
    assert _get_dynamic_injection_pool is not None
    pool = await _get_dynamic_injection_pool()
    return await pool.fetchrow(
        """
        SELECT expected_reset_at, remaining_pct
        FROM public.rate_limit_observations
        WHERE provider = $1
          AND quota_key = $2
        ORDER BY observed_at DESC
        LIMIT 1
        """,
        CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        _OPENROUTER_DURABLE_QUOTA_DAILY_KEY,
    )


async def _get_openrouter_free_daily_quota_exhausted_cooldown_seconds() -> float:
    assert _get_quota_cache is not None
    assert _set_quota_cache is not None
    assert _quota_lock is not None
    assert _fetch_quota_row is not None

    now_monotonic = time.monotonic()
    cached_reset_at, cached_until = _get_quota_cache()
    if cached_until > now_monotonic:
        if cached_reset_at is None:
            return 0.0
        return max(0.0, cached_reset_at - time.time())

    async with _quota_lock:
        cached_reset_at, cached_until = _get_quota_cache()
        if cached_until > time.monotonic():
            if cached_reset_at is None:
                return 0.0
            return max(0.0, cached_reset_at - time.time())

        reset_at_ts: Optional[float] = None
        try:
            row = await asyncio.wait_for(
                _fetch_quota_row(),
                timeout=_OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS,
            )
            if row is not None:
                remaining_pct = row["remaining_pct"]
                try:
                    remaining_pct_float = float(remaining_pct) if remaining_pct is not None else None
                except (TypeError, ValueError):
                    remaining_pct_float = None
                if remaining_pct_float is not None and remaining_pct_float <= 0:
                    reset_at_ts = _parse_openrouter_free_daily_quota_reset_timestamp(row["expected_reset_at"])
        except Exception:
            verbose_proxy_logger.debug(
                "OpenRouter durable quota check failed; failing open for alias selection",
                exc_info=True,
            )
            reset_at_ts = None

        if reset_at_ts is not None and reset_at_ts <= time.time():
            reset_at_ts = None
        _set_quota_cache((
            reset_at_ts,
            time.monotonic() + _OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS,
        ))
        if reset_at_ts is None:
            return 0.0
        return max(0.0, reset_at_ts - time.time())


# ---------------------------------------------------------------------------
# Candidate classification
# ---------------------------------------------------------------------------


def _is_openrouter_free_quota_candidate(candidate: dict[str, Any]) -> bool:
    if candidate["provider"] != CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        return False
    model = str(candidate.get("model") or "")
    return model in _OPENROUTER_FREE_DAILY_QUOTA_MODELS


# ---------------------------------------------------------------------------
# Durable cooldown application
# ---------------------------------------------------------------------------


async def _apply_openrouter_durable_quota_candidate_cooldown(
    *,
    candidate: dict[str, Any],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
) -> tuple[float, Optional[str], Optional[str]]:
    if not _is_openrouter_free_quota_candidate(candidate):
        return cooldown_seconds, cooldown_state_source, skip_reason

    assert _get_free_daily_quota_exhausted_cooldown_seconds is not None
    durable_cooldown = (
        await _get_free_daily_quota_exhausted_cooldown_seconds()
    )
    if durable_cooldown <= 0:
        return cooldown_seconds, cooldown_state_source, skip_reason

    if durable_cooldown > cooldown_seconds:
        cooldown_seconds = durable_cooldown
        cooldown_state_source = "durable_quota"
        skip_reason = "durable_quota_exhausted"
    return cooldown_seconds, cooldown_state_source, skip_reason


# ---------------------------------------------------------------------------
# Alias-probe cooldown gate
# ---------------------------------------------------------------------------


def _raise_openrouter_auto_agent_candidate_unavailable(message: str) -> Never:
    exc = ProxyException(
        message=message,
        type="invalid_request_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise exc


async def _maybe_raise_openrouter_adapter_alias_probe_cooldown(
    adapter_model: Optional[str],
    *,
    use_alias_candidate_probe: bool = False,
) -> None:
    if not use_alias_candidate_probe:
        return
    assert _get_adapter_active_cooldown_seconds is not None
    assert _get_adapter_rate_limit_key is not None
    cooldown_seconds = await _get_adapter_active_cooldown_seconds(adapter_model)
    if cooldown_seconds <= 0:
        return
    rounded_wait = max(1, int(cooldown_seconds))
    model_label = _get_adapter_rate_limit_key(adapter_model)
    _raise_openrouter_auto_agent_candidate_unavailable(
        (
            f"OpenRouter auto-agent candidate {model_label} is temporarily cooling down "
            f"on the adapter. Retry after ~{rounded_wait}s."
        )
    )
