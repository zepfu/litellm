"""Durable alias-routing cooldown/affinity Redis helpers (RR-054 #1/#2).

Owns DualCache selection, cache-key construction, payload expiry parsing, and
durable read/write with max-expiry (never truncate a longer existing cooldown).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# CFG-004: Durable payload inspection, deletion, and absence verification
# ---------------------------------------------------------------------------
#
# These helpers are deliberately stricter than the best-effort read/write path
# above.  ``DualCache.async_get_cache`` and ``RedisCache.async_get_cache`` both
# swallow Redis exceptions (log + return ``None``), which would let a Redis
# outage masquerade as a clean "absent" result.  Inspection / deletion /
# absence-verification must FAIL CLOSED instead: they talk to the underlying
# configured Redis client directly and propagate errors as ``RuntimeError``.

_VALID_ALIAS_FAMILIES = frozenset({"codex", "anthropic"})


@dataclass
class DurableKeyInspection:
    """Result of inspecting a durable cache key (CFG-004)."""

    cache_key: str
    exists: bool
    payload: Optional[Payload] = None
    expires_at_epoch: Optional[float] = None
    ttl_remaining_seconds: Optional[float] = None


def _validate_alias_family(alias_family: str, context: str) -> str:
    """Normalize and validate ``alias_family``; unknown families raise.

    Never defaults an unknown family to codex.
    """
    normalized = (alias_family or "").strip().lower()
    if normalized not in _VALID_ALIAS_FAMILIES:
        raise ValueError(
            f"{context}: unknown alias_family {alias_family!r}; "
            f"expected one of {sorted(_VALID_ALIAS_FAMILIES)}"
        )
    return normalized


def _strict_redis_context(dual_cache: object, cache_key: str, context: str):
    """Resolve a strict Redis client + namespaced key, raising on any gap.

    Verifies the dual cache actually has a usable Redis backend and that the
    backend exposes the methods we need.  Any missing piece is a RuntimeError
    (fail closed) rather than a silent absence.
    """
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: dual cache has no redis_cache"
        )
    init_async_client = getattr(redis_cache, "init_async_client", None)
    if not callable(init_async_client):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis_cache missing init_async_client"
        )
    try:
        client = init_async_client()
    except Exception as exc:  # noqa: BLE001 - fail closed with context
        raise RuntimeError(
            f"AAWM alias routing durable {context}: failed to init redis client: {exc}"
        ) from exc
    if client is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client unavailable"
        )
    fix_namespace = getattr(redis_cache, "check_and_fix_namespace", None)
    namespaced_key = (
        fix_namespace(key=cache_key) if callable(fix_namespace) else cache_key
    )
    return redis_cache, client, namespaced_key


async def _strict_redis_get_value(
    client: object,
    redis_cache: object,
    namespaced_key: str,
    context: str,
) -> tuple[Optional[Payload], bool]:
    """Strictly read a key.  Returns ``(payload, present)``.

    Propagates Redis errors as ``RuntimeError``.  A present-but-malformed
    value (not a dict, or undecodable) raises ``RuntimeError`` rather than
    being reported as absent.
    """
    get_fn = getattr(client, "get", None)
    if not callable(get_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client missing get"
        )
    try:
        raw = await get_fn(namespaced_key)
    except Exception as exc:  # noqa: BLE001 - fail closed with context
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis get failed: {exc}"
        ) from exc
    if raw is None:
        return None, False
    decode = getattr(redis_cache, "_get_cache_logic", None)
    try:
        parsed = decode(cached_response=raw) if callable(decode) else raw
    except Exception as exc:  # noqa: BLE001 - malformed stored value
        raise RuntimeError(
            f"AAWM alias routing durable {context}: malformed cached value: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: malformed cached value "
            f"(expected dict, got {type(parsed).__name__})"
        )
    return dict(parsed), True


async def _strict_redis_ttl(
    client: object,
    namespaced_key: str,
    context: str,
) -> Optional[int]:
    """Strictly read the actual remaining TTL (seconds) for a key.

    Mirrors ``RedisCache.async_get_ttl`` semantics (``None`` when the key has
    no expiry or is missing) but propagates Redis errors as ``RuntimeError``.
    """
    ttl_fn = getattr(client, "ttl", None)
    if not callable(ttl_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client missing ttl"
        )
    try:
        ttl_value = await ttl_fn(namespaced_key)
    except Exception as exc:  # noqa: BLE001 - fail closed with context
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis ttl failed: {exc}"
        ) from exc
    if ttl_value is None or ttl_value < 0:
        return None
    return int(ttl_value)


def _strict_in_memory_get_value(
    dual_cache: object,
    cache_key: str,
    context: str,
) -> tuple[Optional[Payload], bool]:
    """Strictly read a key from the DualCache in-memory tier.

    Resolves ``dual_cache.in_memory_cache``, requires a targeted ``get_cache``
    method, and propagates any exception as ``RuntimeError``.  Returns
    ``(payload, present)``.  A present-but-non-dict value raises
    ``RuntimeError`` (malformed).
    """
    in_memory_cache = getattr(dual_cache, "in_memory_cache", None)
    if in_memory_cache is None:
        # No in-memory tier configured -- nothing to check.
        return None, False
    get_fn = getattr(in_memory_cache, "get_cache", None)
    if not callable(get_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "in_memory_cache missing get_cache"
        )
    try:
        value = get_fn(key=cache_key)
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            f"in-memory get failed: {exc}"
        ) from exc
    if value is None:
        return None, False
    if not isinstance(value, dict):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            f"malformed in-memory value (expected dict, got {type(value).__name__})"
        )
    return dict(value), True


async def inspect_aawm_alias_routing_durable_key(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> DurableKeyInspection:
    """Inspect a durable key: existence, payload, and actual TTL (CFG-004).

    Fails closed: missing cache/methods, Redis errors, and malformed present
    values all raise ``RuntimeError``.  Unknown ``alias_family`` raises
    ``ValueError``.  ``ttl_remaining_seconds`` prefers the actual cache TTL and
    only falls back to the payload expiry when the cache reports no TTL.
    """
    context = f"inspect family={alias_family} kind={state_kind}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )
    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family,
        state_kind=state_kind,
        state_key=state_key,
    )
    redis_cache, client, namespaced_key = _strict_redis_context(
        dual_cache, cache_key, context
    )
    payload, present = await _strict_redis_get_value(
        client, redis_cache, namespaced_key, context
    )
    if not present:
        return DurableKeyInspection(cache_key=cache_key, exists=False)

    expires_at = parse_aawm_alias_routing_durable_expiry(payload)
    actual_ttl = await _strict_redis_ttl(client, namespaced_key, context)
    if actual_ttl is not None:
        ttl_remaining: Optional[float] = float(actual_ttl)
    elif expires_at is not None:
        ttl_remaining = max(0.0, expires_at - time.time())
    else:
        ttl_remaining = None
    return DurableKeyInspection(
        cache_key=cache_key,
        exists=True,
        payload=payload,
        expires_at_epoch=expires_at,
        ttl_remaining_seconds=ttl_remaining,
    )


async def delete_aawm_alias_routing_durable_key(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> bool:
    """Delete a durable cache key from both Redis and the DualCache memory tier.

    Returns True iff the key existed in Redis prior to deletion.

    Fails closed: missing cache/methods, Redis errors, malformed present
    values, or failure to clear/verify either layer raise ``RuntimeError``.
    Unknown ``alias_family`` raises ``ValueError``.

    Post-deletion verification proves both raw Redis absence and normal
    DualCache read-path absence (the path ``read_aawm_alias_routing_durable_payload``
    uses).  No clients are closed and no broad flush is performed.
    """
    context = f"delete family={alias_family} kind={state_kind}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )
    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family,
        state_kind=state_kind,
        state_key=state_key,
    )
    redis_cache, client, namespaced_key = _strict_redis_context(
        dual_cache, cache_key, context
    )

    # 1. Strict existence check + malformed-value validation via raw Redis.
    _payload, present = await _strict_redis_get_value(
        client, redis_cache, namespaced_key, context
    )

    # 2. Delete from raw Redis (strict).
    delete_fn = getattr(client, "delete", None)
    if not callable(delete_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client missing delete"
        )
    try:
        await delete_fn(namespaced_key)
    except Exception as exc:  # noqa: BLE001 - fail closed with context
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis delete failed: {exc}"
        ) from exc

    # 3. Delete from the DualCache in-memory tier (strict, targeted).
    in_memory_cache = getattr(dual_cache, "in_memory_cache", None)
    if in_memory_cache is not None:
        mem_delete = getattr(in_memory_cache, "delete_cache", None)
        if not callable(mem_delete):
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "in_memory_cache missing delete_cache"
            )
        try:
            mem_delete(cache_key)
        except Exception as exc:  # noqa: BLE001 - fail closed with context
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                f"in-memory delete failed: {exc}"
            ) from exc

    # 4. Verify raw Redis absence (strict).
    _post_raw, post_present = await _strict_redis_get_value(
        client, redis_cache, namespaced_key, context
    )
    if post_present:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "key still present in Redis after delete"
        )

    # 5. Verify in-memory tier absence (strict, underlying method).
    _mem_val, mem_present = _strict_in_memory_get_value(
        dual_cache, cache_key, context
    )
    if mem_present:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "key still present in in-memory tier after delete"
        )

    return present


async def verify_aawm_alias_routing_durable_absence(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> bool:
    """Verify a durable key is absent from BOTH Redis and the in-memory tier.

    Returns True only when both tiers confirm absence.  Fails closed: missing
    cache/methods, Redis errors, in-memory read errors, and malformed present
    values raise ``RuntimeError`` rather than returning a false-positive
    absence.  Unknown ``alias_family`` raises ``ValueError``.
    """
    context = f"absence-check family={alias_family} kind={state_kind}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )
    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family,
        state_kind=state_kind,
        state_key=state_key,
    )
    redis_cache, client, namespaced_key = _strict_redis_context(
        dual_cache, cache_key, context
    )
    # Strict raw Redis check.
    _payload, redis_present = await _strict_redis_get_value(
        client, redis_cache, namespaced_key, context
    )
    if redis_present:
        return False
    # Strict in-memory tier check.
    _mem_val, mem_present = _strict_in_memory_get_value(
        dual_cache, cache_key, context
    )
    if mem_present:
        return False
    return True
