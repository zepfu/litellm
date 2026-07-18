"""Durable alias-routing cooldown/affinity Redis helpers (RR-054 #1/#2).

Owns DualCache selection, cache-key construction, payload expiry parsing, and
durable read/write with max-expiry (never truncate a longer existing cooldown).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Callable, Optional

from litellm.proxy.aawm_alias_routing_redis import (
    get_dual_cache as _redis_get_dual_cache,
    get_durable_write_retry_attempts,
    get_durable_write_retry_backoff_seconds,
    get_status as _redis_get_status,
    is_retryable_redis_error,
    resolve_alias_routing_state_namespace,
)

from .types import Payload

logger = logging.getLogger("LiteLLMProxy")

AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX = "aawm:alias-routing"
AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT = "aawm-routing-v1"

# Rate-limit Redis failure logs to avoid hot-path spam.
_DURABLE_FAILURE_LOG_INTERVAL_SECONDS = 30.0
_durable_failure_log_until_monotonic_by_key: dict[str, float] = {}
_durable_affinity_key_until_epoch: dict[str, float] = {}

_clean_value: Optional[Callable[[object], Optional[str]]] = None
_dual_cache_override: Optional[Callable[[], Optional[object]]] = None


def configure_durable_runtime(
    *,
    clean_value: Callable[[object], Optional[str]],
    get_dual_cache_override: Optional[Callable[[], Optional[object]]] = None,
) -> None:
    global _clean_value, _dual_cache_override
    _clean_value = clean_value
    _dual_cache_override = get_dual_cache_override


def _clean(value: object) -> Optional[str]:
    if _clean_value is not None:
        return _clean_value(value)
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _should_log_durable_failure(log_key: str) -> bool:
    now = time.monotonic()
    until = _durable_failure_log_until_monotonic_by_key.get(log_key, 0.0)
    if now < until:
        return False
    _durable_failure_log_until_monotonic_by_key[log_key] = (
        now + _DURABLE_FAILURE_LOG_INTERVAL_SECONDS
    )
    # Bound map size cheaply.
    if len(_durable_failure_log_until_monotonic_by_key) > 512:
        try:
            oldest = next(iter(_durable_failure_log_until_monotonic_by_key))
            _durable_failure_log_until_monotonic_by_key.pop(oldest, None)
        except StopIteration:
            pass
    return True


def get_aawm_alias_routing_state_namespace() -> str:
    try:
        return resolve_alias_routing_state_namespace()
    except Exception:
        raw = _clean(os.getenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE"))
        if raw is not None:
            return raw
    return AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT


def build_aawm_alias_routing_durable_cache_key(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> str:
    namespace = get_aawm_alias_routing_state_namespace()
    normalized_family = alias_family.strip().lower()
    normalized_kind = state_kind.strip().lower()
    opaque_state_key = hashlib.sha256(state_key.encode("utf-8")).hexdigest()
    return (
        f"{AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX}:{namespace}:"
        f"{normalized_family}:{normalized_kind}:{opaque_state_key}"
    )


def _get_durable_affinity_key_limit() -> int:
    raw = _clean(os.getenv("AAWM_ALIAS_ROUTING_DURABLE_AFFINITY_KEY_LIMIT"))
    if raw is None:
        return 4096
    try:
        return max(1, int(raw))
    except Exception:
        return 4096


def _reserve_durable_affinity_key(cache_key: str, *, expires_at_epoch: float) -> bool:
    now = time.time()
    for key, expiry in list(_durable_affinity_key_until_epoch.items()):
        if expiry <= now:
            _durable_affinity_key_until_epoch.pop(key, None)
    if cache_key in _durable_affinity_key_until_epoch:
        _durable_affinity_key_until_epoch[cache_key] = max(
            _durable_affinity_key_until_epoch[cache_key],
            expires_at_epoch,
        )
        return True
    if len(_durable_affinity_key_until_epoch) >= _get_durable_affinity_key_limit():
        return False
    _durable_affinity_key_until_epoch[cache_key] = expires_at_epoch
    return True


def get_aawm_alias_routing_dual_cache() -> Optional[object]:
    """Return DualCache for alias-routing durable state.

    Prefer dedicated alias-routing Redis. If configured but unhealthy, return
    None (do not poison shared internal_usage_cache). Legacy shared fallback is
    only used when dedicated routing Redis is unconfigured.
    """
    if _dual_cache_override is not None:
        try:
            override = _dual_cache_override()
            if override is not None:
                return override
        except Exception:
            pass
    try:
        try:
            dual_cache = _redis_get_dual_cache()
            if (
                dual_cache is not None
                and getattr(dual_cache, "redis_cache", None) is not None
            ):
                return dual_cache
        except Exception:
            dual_cache = None

        try:
            status = _redis_get_status()
            if isinstance(status, dict) and status.get("configured") is True:
                return None
        except Exception:
            pass
    except Exception:
        pass

    try:
        from litellm.proxy.proxy_server import proxy_logging_obj
    except Exception:
        return None
    if proxy_logging_obj is None:
        return None
    internal_usage_cache = getattr(proxy_logging_obj, "internal_usage_cache", None)
    if internal_usage_cache is None:
        return None
    dual_cache = getattr(internal_usage_cache, "dual_cache", None)
    if dual_cache is None or getattr(dual_cache, "redis_cache", None) is None:
        return None
    return dual_cache


def parse_aawm_alias_routing_durable_expiry(payload: object) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    expires_at = payload.get("expires_at_epoch")
    if not isinstance(expires_at, (int, float)):
        return None
    if float(expires_at) <= time.time():
        return None
    return float(expires_at)


async def read_aawm_alias_routing_durable_payload(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> Optional[Payload]:
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None
    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=alias_family,
        state_kind=state_kind,
        state_key=state_key,
    )
    try:
        async_get_cache = getattr(dual_cache, "async_get_cache", None)
        if not callable(async_get_cache):
            return None
        payload = await async_get_cache(key=cache_key)
    except Exception:
        if _should_log_durable_failure(f"read:{alias_family}:{state_kind}"):
            logger.warning(
                "AAWM alias routing durable read failed for family=%s kind=%s",
                alias_family,
                state_kind,
                exc_info=True,
            )
        return None
    if not isinstance(payload, dict):
        return None
    if parse_aawm_alias_routing_durable_expiry(payload) is None:
        return None
    return dict(payload)


async def write_aawm_alias_routing_durable_payload(  # noqa: PLR0915
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
    payload: Payload,
    ttl_seconds: float,
) -> bool:
    """Write durable payload with max-expiry semantics (RR-054 #2).

    Never truncate a longer existing expires_at_epoch with a shorter write.
    Also write-through DualCache memory when available for process coherency.
    """
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return False
    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=alias_family,
        state_kind=state_kind,
        state_key=state_key,
    )
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        return False

    now = time.time()
    new_expires = now + max(0.0, float(ttl_seconds))
    durable_payload = dict(payload)
    durable_payload["expires_at_epoch"] = new_expires
    ttl = max(1.0, float(ttl_seconds))
    if state_kind.strip().lower() == "affinity" and not _reserve_durable_affinity_key(
        cache_key,
        expires_at_epoch=new_expires,
    ):
        if _should_log_durable_failure(f"affinity-cardinality:{alias_family}"):
            logger.warning(
                "AAWM alias routing durable affinity write skipped at cardinality cap "
                "for family=%s",
                alias_family,
            )
        return False

    # Max-expiry: keep longer existing durable expiry when present.
    existing_payload: Optional[Payload] = None
    try:
        async_get_cache = getattr(dual_cache, "async_get_cache", None)
        existing_raw = (
            await async_get_cache(key=cache_key)
            if callable(async_get_cache)
            else None
        )
        if isinstance(existing_raw, dict):
            existing_payload = dict(existing_raw)
    except Exception:
        if _should_log_durable_failure(f"read-before-write:{alias_family}:{state_kind}"):
            logger.warning(
                "AAWM alias routing durable pre-write read failed for family=%s kind=%s",
                alias_family,
                state_kind,
                exc_info=True,
            )
        existing_payload = None

    if existing_payload is not None:
        existing_expires = parse_aawm_alias_routing_durable_expiry(existing_payload)
        if existing_expires is not None and existing_expires >= new_expires:
            durable_payload = dict(existing_payload)
            # Preserve longer expiry; merge new payload fields without shrinking expiry.
            for key, value in payload.items():
                if key == "expires_at_epoch":
                    continue
                durable_payload[key] = value
            durable_payload["expires_at_epoch"] = existing_expires
            ttl = max(1.0, existing_expires - now)
        elif existing_expires is not None:
            # Extending: keep non-expiry fields from existing when not overwritten.
            merged = dict(existing_payload)
            merged.update(payload)
            merged["expires_at_epoch"] = new_expires
            durable_payload = merged

    max_attempts = 1 + int(get_durable_write_retry_attempts())
    retry_backoff_seconds = float(get_durable_write_retry_backoff_seconds())
    for attempt in range(max_attempts):
        try:
            await redis_cache.async_set_cache(
                key=cache_key,
                value=durable_payload,
                ttl=ttl,
                raise_on_error=True,
            )
            # DualCache memory coherency: write-through in-process cache when available.
            try:
                set_cache = getattr(dual_cache, "async_set_cache", None)
                if callable(set_cache):
                    await set_cache(
                        key=cache_key,
                        value=durable_payload,
                        ttl=ttl,
                        local_only=True,
                    )
                else:
                    sync_set = getattr(dual_cache, "set_cache", None)
                    if callable(sync_set):
                        sync_set(
                            key=cache_key,
                            value=durable_payload,
                            ttl=ttl,
                            local_only=True,
                        )
            except Exception:
                # Memory write-through is best-effort; Redis write already succeeded.
                pass
            return True
        except Exception as exc:
            if attempt >= max_attempts - 1:
                if _should_log_durable_failure(
                    f"write-exhaust:{alias_family}:{state_kind}"
                ):
                    logger.warning(
                        "AAWM alias routing durable write failed after retry exhaustion for family=%s kind=%s",
                        alias_family,
                        state_kind,
                        exc_info=True,
                    )
                return False
            if not is_retryable_redis_error(exc):
                if _should_log_durable_failure(
                    f"write-nonretry:{alias_family}:{state_kind}"
                ):
                    logger.warning(
                        "AAWM alias routing durable write failed with non-retryable error for family=%s kind=%s",
                        alias_family,
                        state_kind,
                        exc_info=True,
                    )
                return False
            if _should_log_durable_failure(f"write-retry:{alias_family}:{state_kind}"):
                logger.warning(
                    "AAWM alias routing durable write retrying after timeout/connection error for family=%s kind=%s",
                    alias_family,
                    state_kind,
                    exc_info=True,
                )
            if retry_backoff_seconds > 0:
                await asyncio.sleep(retry_backoff_seconds)
            continue
    return False
