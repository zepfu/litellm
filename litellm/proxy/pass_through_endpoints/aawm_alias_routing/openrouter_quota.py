"""OpenRouter free-daily-quota probe and pre-transport admission.

The policy classifier governs free-model quota checks for alias selection and
all adapter transports. Exhaustion skips alias candidates and refuses direct
requests with a local 429 before transport; paid models bypass the quota probe.
Adapter cooldown refusal remains alias-probe-only. Cache and lock access use
the injected runtime seams.

Durable free-quota observations are bound to one authoritative deployment
identity so environments, credentials/accounts, and observation sources never
contaminate one another:

- ``provider``: ``openrouter``
- ``quota_key``: ``openrouter_free_daily_requests:requests``
- ``client``: ``openrouter`` (stored ``client_family``)
- ``account_hash``: synthetic shared-pool hash
  ``sha256(b"openrouter_free_daily_shared_pool")[:12]``
- ``source``: ``openrouter_free_daily_local_meter``
- ``environment``: configured runtime environment, matched with
  ``NULLIF(BTRIM(evidence->>'environment'), '') IS NOT DISTINCT FROM $6``
- ``model``: ``NULL`` (provider-wide ``:free`` shared pool)

Model is deliberately unbound as a per-model lookup key: OpenRouter documents
free-model quota as account-level, and writers persist ``model IS NULL`` with
``raw_provider_fields.model_scope = "openrouter_:free_shared_pool"``. The gate
applies that pooled observation to exactly the models in
``OPENROUTER_FREE_DAILY_QUOTA_MODELS``. The SQL ``model IS NULL`` predicate
keeps a hypothetical model-specific row from contaminating the shared pool.

An unconfigured environment still reads, but only rows whose evidence
environment is empty or absent. That preserves unstamped historical rows
without a migration and still prevents configured environments from reading
one another. The process cache slot remains a ``(Optional[float], float)``
tuple; a module-local identity shadow invalidates it when any bound dimension
changes. Each lookup resolves one six-way identity under the quota lock and
reuses that same tuple for the injected fetch, SQL arguments, and cache
publication.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Never, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException

from .policy import (
    CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
    OPENROUTER_FREE_DAILY_QUOTA_MODELS,
    is_openrouter_free_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OPENROUTER_DURABLE_QUOTA_DAILY_KEY = "openrouter_free_daily_requests:requests"
_OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS = 30.0
_OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS = 0.5
_OPENROUTER_FREE_DAILY_QUOTA_MODELS = OPENROUTER_FREE_DAILY_QUOTA_MODELS
_OPENROUTER_DURABLE_QUOTA_CLIENT = "openrouter"
_OPENROUTER_DURABLE_QUOTA_SOURCE = "openrouter_free_daily_local_meter"
_OPENROUTER_DURABLE_QUOTA_ACCOUNT_HASH = hashlib.sha256(
    b"openrouter_free_daily_shared_pool"
).hexdigest()[:12]

_OpenRouterDurableQuotaIdentity = tuple[str, str, str, str, str, Optional[str]]

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
_get_observation_environment: Optional[Callable[[], Optional[str]]] = None
_quota_cache_identity: Optional[_OpenRouterDurableQuotaIdentity] = None
_quota_lookup_identity: Optional[_OpenRouterDurableQuotaIdentity] = None


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
    get_observation_environment: Callable[[], Optional[str]],
) -> None:
    """Bind god-module-owned quota cache accessors and adapter helpers."""
    global _get_quota_cache, _set_quota_cache, _quota_lock
    global _get_dynamic_injection_pool
    global _get_adapter_active_cooldown_seconds, _get_adapter_rate_limit_key
    global _fetch_quota_row
    global _get_free_daily_quota_exhausted_cooldown_seconds
    global _get_observation_environment, _quota_cache_identity
    global _quota_lookup_identity
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
    _get_observation_environment = get_observation_environment
    _quota_cache_identity = None
    _quota_lookup_identity = None


def _reset_openrouter_free_daily_quota_cache() -> None:
    global _quota_cache_identity, _quota_lookup_identity
    assert _set_quota_cache is not None
    _quota_cache_identity = None
    _quota_lookup_identity = None
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


def _openrouter_durable_quota_environment() -> Optional[str]:
    if _get_observation_environment is None:
        return None
    try:
        raw_environment = _get_observation_environment()
    except Exception:
        verbose_proxy_logger.debug(
            "OpenRouter durable quota environment resolution failed; "
            "failing closed for cross-environment reads",
            exc_info=True,
        )
        return None
    if raw_environment is None:
        return None
    environment = str(raw_environment).strip()
    return environment or None


def _openrouter_durable_quota_identity() -> _OpenRouterDurableQuotaIdentity:
    """Return the six-way read identity, including a nullable environment."""
    return (
        CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        _OPENROUTER_DURABLE_QUOTA_DAILY_KEY,
        _OPENROUTER_DURABLE_QUOTA_CLIENT,
        _OPENROUTER_DURABLE_QUOTA_ACCOUNT_HASH,
        _OPENROUTER_DURABLE_QUOTA_SOURCE,
        _openrouter_durable_quota_environment(),
    )


# ---------------------------------------------------------------------------
# Quota probe / fetch
# ---------------------------------------------------------------------------


async def _fetch_openrouter_quota_row_with_identity(
    identity: _OpenRouterDurableQuotaIdentity,
) -> Optional[Any]:
    """Invoke the injected fetch while binding ``identity`` for SQL arguments."""
    global _quota_lookup_identity
    assert _fetch_quota_row is not None
    _quota_lookup_identity = identity
    try:
        return await _fetch_quota_row()
    finally:
        _quota_lookup_identity = None


async def _fetch_openrouter_free_daily_quota_row(
    identity: Optional[_OpenRouterDurableQuotaIdentity] = None,
) -> Optional[Any]:
    """Direct DB fetch bound to the complete durable-quota identity.

    SQL arguments use the lookup-attempt identity. The injected fetch seam is
    zero-arg, so the cooldown binds that tuple before calling it. This function
    does not recapture ``get_observation_environment``.
    """
    lookup_identity = identity if identity is not None else _quota_lookup_identity
    assert lookup_identity is not None
    assert _get_dynamic_injection_pool is not None
    pool = await _get_dynamic_injection_pool()
    provider, quota_key, client, account_hash, source, environment = lookup_identity
    return await pool.fetchrow(
        """
        SELECT expected_reset_at, remaining_pct
        FROM public.rate_limit_observations
        WHERE provider = $1
          AND quota_key = $2
          AND client = $3
          AND account_hash = $4
          AND source = $5
          AND model IS NULL
          AND NULLIF(BTRIM(evidence->>'environment'), '') IS NOT DISTINCT FROM $6
        ORDER BY observed_at DESC
        LIMIT 1
        """,
        provider,
        quota_key,
        client,
        account_hash,
        source,
        environment,
    )


def _cached_exhausted_cooldown_seconds(
    cached_reset_at: Optional[float],
) -> float:
    if cached_reset_at is None:
        return 0.0
    return max(0.0, cached_reset_at - time.time())


async def _get_openrouter_free_daily_quota_exhausted_cooldown_seconds() -> float:
    global _quota_cache_identity
    assert _get_quota_cache is not None
    assert _set_quota_cache is not None
    assert _quota_lock is not None
    assert _fetch_quota_row is not None

    identity = _openrouter_durable_quota_identity()
    now_monotonic = time.monotonic()
    cached_reset_at, cached_until = _get_quota_cache()
    if _quota_cache_identity == identity and cached_until > now_monotonic:
        return _cached_exhausted_cooldown_seconds(cached_reset_at)

    async with _quota_lock:
        identity = _openrouter_durable_quota_identity()
        cached_reset_at, cached_until = _get_quota_cache()
        if _quota_cache_identity == identity and cached_until > time.monotonic():
            return _cached_exhausted_cooldown_seconds(cached_reset_at)

        reset_at_ts: Optional[float] = None
        try:
            row = await asyncio.wait_for(
                _fetch_openrouter_quota_row_with_identity(identity),
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
                "OpenRouter durable quota check failed; failing open for quota admission",
                exc_info=True,
            )
            reset_at_ts = None

        if reset_at_ts is not None and reset_at_ts <= time.time():
            reset_at_ts = None
        _quota_cache_identity = identity
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
    return is_openrouter_free_model(candidate.get("model"))


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
# Free-quota admission and alias-probe cooldown gate
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
    # The retry transport invokes this gate for direct and alias routes alike.
    # Daily free quota applies to both; adapter cooldown remains probe-only.
    if is_openrouter_free_model(adapter_model):
        assert _get_free_daily_quota_exhausted_cooldown_seconds is not None
        quota_wait = await _get_free_daily_quota_exhausted_cooldown_seconds()
        if quota_wait > 0:
            rounded_wait = max(1, int(quota_wait))
            message = (
                f"OpenRouter free daily quota is exhausted for {adapter_model}. "
                f"Retry after ~{rounded_wait}s."
            )
            if use_alias_candidate_probe:
                _raise_openrouter_auto_agent_candidate_unavailable(message)
            exc = ProxyException(
                message=message,
                type="rate_limit_error",
                param="model",
                code=429,
            )
            setattr(exc, "attempted_provider_call", False)
            raise exc
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
