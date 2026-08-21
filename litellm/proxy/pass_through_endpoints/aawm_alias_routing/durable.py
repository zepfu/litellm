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
from typing import Any, Callable, Optional, Union

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



# ---------------------------------------------------------------------------
# Persistent (unbounded) cooldown marker
# ---------------------------------------------------------------------------
#
# Replaces the former 10-year expires_at_epoch sentinel for Redis TTL -1.
# Payloads use "persistent": true as an explicit JSON-safe marker.  The
# internal UNBOUNDED_EXPIRY singleton represents unbounded expiry in
# parser results, inspection dataclasses, and max-expiry comparisons.

PERSISTENT_MARKER = "persistent"


class UnboundedExpiry:
    """Typed singleton representing unbounded (persistent) expiry.

    Compares greater than any finite epoch so max-expiry logic naturally
    preserves persistent state over finite writes.
    """

    _instance: Optional["UnboundedExpiry"] = None

    def __new__(cls) -> "UnboundedExpiry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNBOUNDED_EXPIRY"

    def __ge__(self, other: object) -> bool:
        if isinstance(other, UnboundedExpiry):
            return True
        if isinstance(other, (int, float)):
            return True
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, UnboundedExpiry):
            return False
        if isinstance(other, (int, float)):
            return True
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, UnboundedExpiry):
            return True
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnboundedExpiry)

    def __hash__(self) -> int:
        return hash("UnboundedExpiry")


UNBOUNDED_EXPIRY = UnboundedExpiry()


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
        return str(resolve_alias_routing_state_namespace())
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
            from litellm.proxy import aawm_alias_routing_redis as live_redis

            redis_get_dual_cache = getattr(live_redis, "get_dual_cache", _redis_get_dual_cache)
            dual_cache = redis_get_dual_cache() if callable(redis_get_dual_cache) else _redis_get_dual_cache()
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


def parse_aawm_alias_routing_durable_expiry(
    payload: object,
) -> Union[float, UnboundedExpiry, None]:
    """Parse durable payload expiry.

    Returns:
    - UNBOUNDED_EXPIRY for explicitly persistent payloads
      ("persistent": true marker)
    - A finite future epoch float for time-bounded payloads
    - None for absent, malformed, or expired payloads
    """
    if not isinstance(payload, dict):
        return None
    # Explicit persistent marker (JSON-safe, no Infinity or arbitrary date).
    if payload.get(PERSISTENT_MARKER) is True:
        return UNBOUNDED_EXPIRY
    expires_at = payload.get("expires_at_epoch")
    if not isinstance(expires_at, (int, float)):
        return None
    epoch = float(expires_at)
    if epoch <= time.time():
        return None
    return epoch


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
    allow_ttl_shrink: bool = False,
) -> bool:
    """Write durable payload with max-expiry semantics (RR-054 #2).

    Never truncate a longer existing expires_at_epoch with a shorter write
    unless the caller explicitly allows a usage-limit replacement.
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
        if existing_expires is UNBOUNDED_EXPIRY:
            # Persistent existing key: preserve persistent state; merge new
            # payload fields without introducing a finite expiry.
            durable_payload = dict(existing_payload)
            for key, value in payload.items():
                if key in ("expires_at_epoch", PERSISTENT_MARKER):
                    continue
                durable_payload[key] = value
            durable_payload.pop("expires_at_epoch", None)
            durable_payload[PERSISTENT_MARKER] = True
            ttl = -1.0  # Signal persistent to Redis (no expiry).
        elif (
            existing_expires is not None
            and existing_expires >= new_expires
            and not allow_ttl_shrink
        ):
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
            if ttl < 0:
                # Persistent write: bypass async_set_cache (which always
                # applies a TTL) and use the raw Redis client directly.
                import json as _json_mod

                init_fn = getattr(redis_cache, "init_async_client", None)
                if not callable(init_fn):
                    return False
                raw_client = init_fn()
                if raw_client is None:
                    return False
                fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
                ns_key = (
                    fix_ns(key=cache_key) if callable(fix_ns) else cache_key
                )
                await raw_client.set(
                    name=ns_key,
                    value=_json_mod.dumps(durable_payload),
                )
                await raw_client.persist(ns_key)
            else:
                await redis_cache.async_set_cache(
                    key=cache_key,
                    value=durable_payload,
                    ttl=ttl,
                    raise_on_error=True,
                )
            # DualCache memory coherency: write-through in-process cache when
            # available.  Skip for persistent (ttl < 0) writes: the in-memory
            # cache interprets negative TTL as immediately expired, which would
            # silently discard the just-written persistent entry.  Persistent
            # state lives authoritatively in Redis and is hydrated on read.
            if ttl < 0:
                return True
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

# Production family labels accepted by durable clear/publish APIs.
# Canonicalized to the internal short form before any key construction.
_FAMILY_LABEL_ALIASES: dict[str, str] = {
    "codex_auto_agent": "codex",
    "anthropic_auto_agent": "anthropic",
}


@dataclass
class DurableKeyInspection:
    """Result of inspecting a durable cache key (CFG-004)."""

    cache_key: str
    exists: bool
    payload: Optional[Payload] = None
    expires_at_epoch: Union[float, UnboundedExpiry, None] = None
    ttl_remaining_seconds: Union[float, UnboundedExpiry, None] = None


def _validate_alias_family(alias_family: str, context: str) -> str:
    """Normalize and validate ``alias_family``; unknown families raise.

    Accepts production family labels (``codex_auto_agent``,
    ``anthropic_auto_agent``) and canonicalizes them to the internal short
    form.  Never defaults an unknown family to codex.
    """
    normalized = (alias_family or "").strip().lower()
    if normalized in _FAMILY_LABEL_ALIASES:
        normalized = _FAMILY_LABEL_ALIASES[normalized]
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
        ttl_remaining: Union[float, UnboundedExpiry, None] = float(actual_ttl)
    elif expires_at is UNBOUNDED_EXPIRY:
        ttl_remaining = UNBOUNDED_EXPIRY
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




# ---------------------------------------------------------------------------
# CFG-004: Atomic cooldown publication transaction
# ---------------------------------------------------------------------------
#
# Single strict atomic Redis Lua transaction that:
#   1. Preflights bounded index capacity (reject before mutation)
#   2. Journals exact pre-images (GET each cooldown key)
#   3. Registers all lane members (SADD)
#   4. Writes all cooldown values with monotonic TTL
#   5. Writes a transaction receipt atomically
#
# TTL rules:
#   - existing TTL == -1 (persistent): remains persistent
#   - existing TTL == -2 (absent): gets ceil(requested)
#   - existing TTL >= 0: only extends (never shortens)
#
# No non-Lua fallback.  EVAL unavailable -> fail closed.
# Identity is reconstructible from public candidate+ingress/route family.

import json as _json
import math as _math
import uuid as _uuid


# ---------------------------------------------------------------------------
# Sanitized error types (no lane key, Redis key, credentials, or raw error)
# ---------------------------------------------------------------------------


class PublicationTransactionError(RuntimeError):
    """Base for publication transaction failures.  Sanitized message."""

    def __init__(
        self,
        *,
        phase: str,
        family: str,
        transaction_id_prefix: str,
        identity_prefix: str,
        key_count: int,
        exception_classes: tuple[str, ...],
        detail: str = "",
    ) -> None:
        self.phase = phase
        self.family = family
        self.transaction_id_prefix = transaction_id_prefix
        self.identity_prefix = identity_prefix
        self.key_count = key_count
        self.exception_classes = exception_classes
        msg = (
            f"publication transaction {detail}".strip()
            + f" [phase={phase} family={family}"
            + f" txn={transaction_id_prefix}"
            + f" keys={key_count}"
            + f" errors={','.join(exception_classes) or "none"}]"
        )
        super().__init__(msg)


class RollbackFailedError(PublicationTransactionError):
    """Rollback failed; state is indeterminate."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("detail", "rollback failed; state indeterminate")
        super().__init__(**kwargs)  # type: ignore[arg-type]


class CapacityRejectedError(PublicationTransactionError):
    """Index capacity preflight rejected the transaction."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("detail", "capacity rejected")
        super().__init__(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Transaction phases
# ---------------------------------------------------------------------------

PHASE_PREPARED = "PREPARED"
PHASE_DURABLE_COMMITTED = "DURABLE_COMMITTED"
PHASE_LOCAL_COMMITTED = "LOCAL_COMMITTED"


# ---------------------------------------------------------------------------
# Lua script: atomic publish_cooldown_transaction
# ---------------------------------------------------------------------------
#
# KEYS layout (all passed by caller):
#   KEYS[1]           = receipt key
#   KEYS[2..N+1]      = cooldown keys (N = num_cooldown_keys)
#   KEYS[N+2..N+M+1]  = identity set keys (M = num_identity_keys)
#
# ARGV layout:
#   ARGV[1]  = num_cooldown_keys (N)
#   ARGV[2]  = num_identity_keys (M)
#   ARGV[3]  = requested_ttl_seconds (ceiled by caller)
#   ARGV[4]  = max_lanes_per_identity (bound)
#   ARGV[5]  = transaction_id
#   ARGV[6]  = receipt_json (serialized receipt payload)
#   ARGV[7]  = cooldown_value_json (serialized cooldown payload)
#   ARGV[8]           = allow_ttl_shrink (1/0)
#   ARGV[9..9+N-1]    = lane_key members for identity registration
#                       (one per identity key, cycled if M < N)
#   ARGV[9..9+M-1]    = lane member for each identity key
#
# Returns:
#   1  -> success (all mutations applied)
#   -1 -> capacity rejected (no mutations applied)
#   -2 -> unexpected internal error
#
# The script journals pre-images internally and stores them in the receipt
# so the caller can reconstruct them for rollback.

_LUA_PUBLISH_COOLDOWN_TRANSACTION = """
local receipt_key = KEYS[1]
local num_cd = tonumber(ARGV[1])
local num_id = tonumber(ARGV[2])
local req_ttl = tonumber(ARGV[3])
local max_lanes = tonumber(ARGV[4])
local txn_id = ARGV[5]
local receipt_json = ARGV[6]
local cd_value_json = ARGV[7]
local allow_ttl_shrink = tonumber(ARGV[8]) == 1

-- Phase 1: Aggregate unique-member capacity preflight.
-- All identity keys in one transaction map to the same identity set.
-- Count unique NEW members and reject if total would exceed max_lanes.
if num_id > 0 then
    local id_key = KEYS[1 + num_cd + 1]
    local card = redis.call('SCARD', id_key)
    local seen = {}
    local new_count = 0
    for i = 1, num_id do
        local member = ARGV[8 + i]
        if not seen[member] then
            seen[member] = true
            local is_member = redis.call('SISMEMBER', id_key, member)
            if is_member == 0 then
                new_count = new_count + 1
            end
        end
    end
    if card + new_count > max_lanes then
        return -1
    end
end

-- Phase 2: Journal pre-images for cooldown keys
local preimages = {}
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local val = redis.call('GET', cd_key)
    local ttl = redis.call('TTL', cd_key)
    preimages[i] = {val, ttl}
end

-- Phase 2b: Journal identity-set membership and TTL before mutation.
local id_preimages = {}
if num_id > 0 then
    local id_key = KEYS[1 + num_cd + 1]
    local members = redis.call('SMEMBERS', id_key)
    local id_ttl = redis.call('TTL', id_key)
    id_preimages = {members, id_ttl}
end

-- Phase 3: Register all lane members
for i = 1, num_id do
    local id_key = KEYS[1 + num_cd + i]
    local member = ARGV[8 + i]
    -- Read pre-image TTL BEFORE SADD: an absent key returns -2; after
    -- SADD creates it, TTL would return -1 which must NOT be confused
    -- with a genuinely persistent key.
    local pre_ttl = redis.call('TTL', id_key)
    redis.call('SADD', id_key, member)
    -- Monotonic TTL on identity key using pre-image TTL:
    --   -2 (absent)  -> apply ceil(requested finite TTL)
    --   -1 (persist) -> remain persistent (no EXPIRE)
    --   positive     -> monotonic max(pre_ttl, req_ttl)
    if pre_ttl == -2 then
        if req_ttl > 0 then
            redis.call('EXPIRE', id_key, req_ttl)
        end
    elseif pre_ttl == -1 then
        -- genuinely persistent: leave without EXPIRE
    else
        local target = req_ttl
        if pre_ttl > target then
            target = pre_ttl
        end
        if target > 0 then
            redis.call('EXPIRE', id_key, target)
        end
    end
end

-- Phase 4: Write cooldown values with monotonic TTL.
-- Align expires_at_epoch in the payload with the effective preserved TTL
-- so the existing durable reader remains valid for the full Redis lifetime.
local now_ts = tonumber(redis.call('TIME')[1])
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local old_ttl = preimages[i][2]
    local effective_ttl = req_ttl
    if old_ttl == -1 then
        effective_ttl = -1
    elseif old_ttl == -2 or allow_ttl_shrink then
        effective_ttl = req_ttl
    else
        if old_ttl > effective_ttl then
            effective_ttl = old_ttl
        end
    end
    local payload = cjson.decode(cd_value_json)
    if effective_ttl == -1 then
        -- Persistent: explicit JSON-safe marker (no arbitrary date)
        payload['persistent'] = true
        payload['expires_at_epoch'] = cjson.null
    else
        payload['expires_at_epoch'] = now_ts + effective_ttl
    end
    redis.call('SET', cd_key, cjson.encode(payload))
    if effective_ttl == -1 then
        redis.call('PERSIST', cd_key)
    elseif effective_ttl > 0 then
        redis.call('EXPIRE', cd_key, effective_ttl)
    end
end

-- Phase 5: Journal post-images (state after mutation) for rollback drift check.
local postimages = {}
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local pval = redis.call('GET', cd_key)
    local pttl = redis.call('TTL', cd_key)
    postimages[i] = {pval, pttl}
end
local id_post_members = {}
local id_post_ttl = -2
if num_id > 0 then
    local id_key = KEYS[1 + num_cd + 1]
    local id_post_card = redis.call('SCARD', id_key)
    if id_post_card > 0 then
        id_post_members = redis.call('SMEMBERS', id_key)
    end
    id_post_ttl = redis.call('TTL', id_key)
end

-- Phase 6: Write receipt with exact pre-images and post-images for rollback.
local preimage_arr = {}
for i = 1, num_cd do
    local entry = {}
    entry['v'] = preimages[i][1]
    entry['t'] = preimages[i][2]
    preimage_arr[i] = entry
end
local postimage_arr = {}
for i = 1, num_cd do
    local entry = {}
    entry['v'] = postimages[i][1]
    entry['t'] = postimages[i][2]
    postimage_arr[i] = entry
end
local receipt = cjson.decode(receipt_json)
receipt['preimages'] = preimage_arr
receipt['postimages'] = postimage_arr
if num_id > 0 then
    local id_entry = {}
    id_entry['members'] = id_preimages[1]
    id_entry['ttl'] = id_preimages[2]
    id_entry['key'] = KEYS[1 + num_cd + 1]
    receipt['identity_preimage'] = id_entry
    local id_post_entry = {}
    id_post_entry['members'] = id_post_members
    id_post_entry['ttl'] = id_post_ttl
    id_post_entry['key'] = KEYS[1 + num_cd + 1]
    receipt['identity_postimage'] = id_post_entry
end
redis.call('SET', receipt_key, cjson.encode(receipt))
if req_ttl > 0 then
    redis.call('EXPIRE', receipt_key, req_ttl + 60)
end

return 1
"""


# ---------------------------------------------------------------------------
# Transaction data structures
# ---------------------------------------------------------------------------


@dataclass
class CooldownTransactionJournal:
    """Exact pre-images journaled by the Lua transaction."""

    transaction_id: str
    phase: str
    alias_family: str
    identity_hash: str
    cooldown_keys: list[str]
    identity_keys: list[str]
    lane_members: list[str]
    preimages: list[tuple[Optional[str], int]]  # (raw_value, ttl) per cooldown key
    receipt_key: str
    requested_ttl: int


@dataclass
class CooldownTransactionResult:
    """Result of a successful publish_cooldown_transaction."""

    transaction_id: str
    phase: str
    journal: CooldownTransactionJournal


def _ceil_ttl(ttl_seconds: float) -> int:
    """Ceil fractional TTL to whole seconds, minimum 1."""
    return max(1, int(_math.ceil(float(ttl_seconds))))


def _sanitize_exception_classes(*excs: BaseException) -> tuple[str, ...]:
    return tuple(type(e).__name__ for e in excs)


# ---------------------------------------------------------------------------
# publish_cooldown_transaction
# ---------------------------------------------------------------------------


async def publish_cooldown_transaction(  # noqa: PLR0915
    *,
    alias_family: str,
    identity_hash: str,
    cooldown_keys: list[str],
    lane_members: list[str],
    ttl_seconds: float,
    max_lanes_per_identity: int = 64,
    cooldown_payload: Optional[dict] = None,
    allow_ttl_shrink: bool = False,
) -> CooldownTransactionResult:
    """Execute the atomic cooldown publication transaction.

    One Lua EVAL: preflight capacity, journal pre-images, register lane
    members, write cooldown values with monotonic TTL, write receipt.

    Fails closed: no Redis backend, missing EVAL, or Redis error raises
    RuntimeError.  Capacity rejection raises CapacityRejectedError.

    Returns CooldownTransactionResult with phase=DURABLE_COMMITTED.
    """
    context = f"publish-txn family={alias_family}"
    family = _validate_alias_family(alias_family, context)
    transaction_id = _uuid.uuid4().hex

    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )

    # Build namespaced keys
    receipt_state_key = f"txn-receipt:{transaction_id}"
    receipt_cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family, state_kind="txn_receipt", state_key=receipt_state_key
    )
    cd_cache_keys = [
        build_aawm_alias_routing_durable_cache_key(
            alias_family=family, state_kind="cooldown", state_key=k
        )
        for k in cooldown_keys
    ]
    id_cache_keys = [
        build_aawm_alias_routing_durable_cache_key(
            alias_family=family, state_kind="lane_identity", state_key=identity_hash
        )
        for _ in lane_members
    ]

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
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: failed to init redis client"
        ) from exc
    if client is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client unavailable"
        )

    eval_fn = getattr(client, "eval", None)
    if not callable(eval_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "redis client missing eval (atomic transaction unavailable)"
        )

    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    def _ns(key: str) -> str:
        return fix_ns(key=key) if callable(fix_ns) else key

    ns_receipt = _ns(receipt_cache_key)
    ns_cd_keys = [_ns(k) for k in cd_cache_keys]
    ns_id_keys = [_ns(k) for k in id_cache_keys]

    req_ttl = _ceil_ttl(ttl_seconds)
    num_cd = len(cooldown_keys)
    num_id = len(lane_members)

    # Serialize payload.  Include expires_at_epoch so the existing durable
    # reader (read_aawm_alias_routing_durable_payload / parse_aawm_alias_routing_durable_expiry)
    # can validate the payload after restart or local-memory loss.
    cd_payload = cooldown_payload or {"cooldown_keys": cooldown_keys}
    if "expires_at_epoch" not in cd_payload:
        cd_payload["expires_at_epoch"] = time.time() + float(ttl_seconds)
    cd_value_json = _json.dumps(cd_payload, separators=(",", ":"))

    # Build receipt with pre-image placeholders (Lua fills actual pre-images
    # internally; we store the structure for reconciliation)
    receipt = {
        "txn_id": transaction_id,
        "family": family,
        "num_keys": num_cd,
        "ttl": req_ttl,
    }
    receipt_json = _json.dumps(receipt, separators=(",", ":"))

    # KEYS: [receipt, cd_keys..., id_keys...]
    keys = [ns_receipt] + ns_cd_keys + ns_id_keys
    # ARGV: [num_cd, num_id, req_ttl, max_lanes, txn_id, receipt_json,
    #        cd_value_json, allow_ttl_shrink, lane_members...]
    argv = [
        str(num_cd),
        str(num_id),
        str(req_ttl),
        str(max(1, int(max_lanes_per_identity))),
        transaction_id,
        receipt_json,
        cd_value_json,
        "1" if allow_ttl_shrink else "0",
    ] + list(lane_members)

    try:
        result = await eval_fn(
            _LUA_PUBLISH_COOLDOWN_TRANSACTION,
            len(keys),
            *keys,
            *argv,
        )
    except Exception as exc:
        # Commit-then-raise reconciliation: the EVAL may have committed
        # server-side but the response was lost.  Check receipt presence.
        try:
            committed = await reconcile_cooldown_transaction(
                alias_family=family,
                transaction_id=transaction_id,
                cooldown_cache_keys=ns_cd_keys,
                identity_cache_key=ns_id_keys[0] if ns_id_keys else None,
                lane_members=list(lane_members),
            )
            if committed:
                journal = CooldownTransactionJournal(
                    transaction_id=transaction_id,
                    phase=PHASE_DURABLE_COMMITTED,
                    alias_family=family,
                    identity_hash=identity_hash,
                    cooldown_keys=list(cooldown_keys),
                    identity_keys=id_cache_keys,
                    lane_members=list(lane_members),
                    preimages=[],
                    receipt_key=receipt_cache_key,
                    requested_ttl=req_ttl,
                )
                return CooldownTransactionResult(
                    transaction_id=transaction_id,
                    phase=PHASE_DURABLE_COMMITTED,
                    journal=journal,
                )
        except Exception:
            pass  # reconciliation itself failed; fall through to raise
        raise RuntimeError(
            f"AAWM alias routing durable {context}: lua transaction failed"
        ) from exc

    code = int(result)
    if code == -1:
        raise CapacityRejectedError(
            phase=PHASE_PREPARED,
            family=family,
            transaction_id_prefix=transaction_id[:12],
            identity_prefix=identity_hash[:12],
            key_count=num_cd,
            exception_classes=(),
        )
    if code != 1:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: unexpected lua result {code}"
        )

    journal = CooldownTransactionJournal(
        transaction_id=transaction_id,
        phase=PHASE_DURABLE_COMMITTED,
        alias_family=family,
        identity_hash=identity_hash,
        cooldown_keys=list(cooldown_keys),
        identity_keys=id_cache_keys,
        lane_members=list(lane_members),
        preimages=[],  # pre-images are internal to Lua; rollback uses receipt
        receipt_key=receipt_cache_key,
        requested_ttl=req_ttl,
    )
    return CooldownTransactionResult(
        transaction_id=transaction_id,
        phase=PHASE_DURABLE_COMMITTED,
        journal=journal,
    )


# ---------------------------------------------------------------------------
# reconcile_cooldown_transaction
# ---------------------------------------------------------------------------


async def reconcile_cooldown_transaction(
    *,
    alias_family: str,
    transaction_id: str,
    cooldown_cache_keys: list[str],
    identity_cache_key: str,
    lane_members: list[str],
) -> bool:
    """Reconcile after EVAL exception or commit-then-raise for publication.

    Checks whether the receipt key exists (commit evidence only).  When the
    receipt is present, verifies the full published post-image: every
    cooldown key must be present and every lane member must be registered
    in the identity set.  Receipt existence alone never authorizes success.
    Returns True only when receipt AND all postconditions hold.
    Returns False if the receipt is absent (no commit occurred).
    Fails closed on Redis errors or postcondition violations.
    """
    context = f"reconcile-txn family={alias_family} txn={transaction_id[:12]}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )
    receipt_state_key = f"txn-receipt:{transaction_id}"
    receipt_cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family, state_kind="txn_receipt", state_key=receipt_state_key
    )
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: dual cache has no redis_cache"
        )
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: missing init_async_client"
        )
    try:
        client = init_fn()
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: client init failed"
        ) from exc
    if client is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: client unavailable"
        )
    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    ns_key = fix_ns(key=receipt_cache_key) if callable(fix_ns) else receipt_cache_key
    get_fn = getattr(client, "get", None)
    if not callable(get_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: client missing get"
        )
    try:
        raw = await get_fn(ns_key)
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: receipt check failed"
        ) from exc
    if raw is None:
        return False

    # Strict postcondition: verify published post-image.
    # Receipt is commit evidence only; success requires full post-image.
    for cd_key in cooldown_cache_keys:
        try:
            cd_raw = await get_fn(cd_key)
        except Exception as exc:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "publication postcondition check failed"
            ) from exc
        if cd_raw is None:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "cooldown key absent after reconciled publication"
            )
    sismember_fn = getattr(client, "sismember", None)
    if not callable(sismember_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "postcondition check unavailable (missing sismember)"
        )
    for member in lane_members:
        try:
            is_member = await sismember_fn(identity_cache_key, member)
        except Exception as exc:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "publication membership postcondition failed"
            ) from exc
        if not is_member:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "lane member absent from identity set after "
                "reconciled publication"
            )
    return True


# ---------------------------------------------------------------------------
# rollback_cooldown_transaction
# ---------------------------------------------------------------------------

# Atomic Lua rollback script.  Restores exact pre-images from the receipt,
# removes lane members, restores identity-key TTL, and deletes the receipt.
#
# KEYS layout:
#   KEYS[1]           = receipt key
#   KEYS[2..N+1]      = cooldown keys (N = num_cooldown_keys)
#   KEYS[N+2]         = identity set key (single identity)
#
# ARGV layout:
#   ARGV[1]  = num_cooldown_keys (N)
#   ARGV[2..N+1] = lane members to SREM from identity set
#
# Returns:
#   1  -> success (all pre-images restored, receipt deleted)
#   -1 -> receipt missing (nothing to restore)
#   -2 -> partial restoration error (receipt RETAINED for forensics)
#
# On error the receipt is NOT deleted so operators can inspect what failed.

_LUA_ROLLBACK_COOLDOWN_TRANSACTION = """
local receipt_key = KEYS[1]
local num_cd = tonumber(ARGV[1])

-- Phase 1: Read receipt
local receipt_raw = redis.call('GET', receipt_key)
if not receipt_raw then
    return -1
end

local ok, receipt = pcall(cjson.decode, receipt_raw)
if not ok then
    return -2
end

local preimages = receipt['preimages']
if not preimages then
    return -2
end

-- Phase 2: Drift check -- compare exact current state to post-image.
local postimages = receipt['postimages']
if postimages then
    for i = 1, num_cd do
        local cd_key = KEYS[1 + i]
        local cur_val = redis.call('GET', cd_key)
        local cur_ttl = redis.call('TTL', cd_key)
        local post = postimages[i]
        if post then
            local post_val = post['v']
            local post_ttl = tonumber(post['t'])
            local cur_absent = (cur_val == false or cur_val == nil)
            local post_absent = (post_val == false or post_val == nil)
            if cur_absent ~= post_absent then
                return -3
            end
            if not cur_absent and cur_val ~= post_val then
                return -3
            end
            if post_ttl >= 0 and cur_ttl >= 0 then
                if math.abs(cur_ttl - post_ttl) > 1 then
                    return -3
                end
            elseif post_ttl ~= cur_ttl then
                if not (post_ttl < 0 and cur_ttl < 0) then
                    return -3
                end
            end
        end
    end
end

-- Phase 2b: Identity set drift check against post-image.
local id_postimage = receipt['identity_postimage']
if id_postimage and id_postimage['key'] then
    local id_key = id_postimage['key']
    local post_members = id_postimage['members']
    local post_set = {}
    local post_count = 0
    if post_members then
        for _, m in ipairs(post_members) do
            post_set[m] = true
            post_count = post_count + 1
        end
    end
    local cur_card = redis.call('SCARD', id_key)
    if cur_card ~= post_count then
        return -3
    end
    if cur_card > 0 then
        local cur_members = redis.call('SMEMBERS', id_key)
        for _, m in ipairs(cur_members) do
            if not post_set[m] then
                return -3
            end
        end
    end
end

-- Phase 3: Restore cooldown key pre-images
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local entry = preimages[i]
    if entry then
        local val = entry['v']
        local ttl = tonumber(entry['t'])
        if ttl == -2 or val == false or val == nil then
            redis.call('DEL', cd_key)
        else
            redis.call('SET', cd_key, val)
            if ttl == -1 then
                redis.call('PERSIST', cd_key)
            elseif ttl > 0 then
                redis.call('EXPIRE', cd_key, ttl)
            end
        end
    else
        redis.call('DEL', cd_key)
    end
end

-- Phase 4: Restore prior identity membership (differential SREM).
local id_preimage = receipt['identity_preimage']
if id_preimage and id_preimage['key'] then
    local id_key = id_preimage['key']
    local prior_members = id_preimage['members']
    local prior_set = {}
    if prior_members then
        for _, m in ipairs(prior_members) do
            prior_set[m] = true
        end
    end
    for i = 1, num_cd do
        local member = ARGV[1 + i]
        if member and not prior_set[member] then
            redis.call('SREM', id_key, member)
        end
    end
    local id_ttl = tonumber(id_preimage['ttl'])
    if id_ttl == -2 then
        redis.call('DEL', id_key)
    elseif id_ttl == -1 then
        redis.call('PERSIST', id_key)
    elseif id_ttl and id_ttl > 0 then
        redis.call('EXPIRE', id_key, id_ttl)
    end
end

-- Phase 5: Delete receipt (only on full success)
redis.call('DEL', receipt_key)

return 1
"""


async def rollback_cooldown_transaction(
    *,
    alias_family: str,
    journal: CooldownTransactionJournal,
) -> None:
    """Restore exact pre-images from a committed transaction atomically.

    Executes a single Lua EVAL that reads the receipt, restores cooldown
    key pre-images (including TTL -1 persistent and -2 absent semantics),
    removes lane members from the identity set, restores identity-key TTL,
    and deletes the receipt.

    On restoration error the receipt is RETAINED (not deleted) so operators
    can inspect the failure.  Any failure raises RollbackFailedError with
    sanitized context -- this error is NEVER suppressed by callers.
    """
    context = f"rollback-txn family={alias_family} txn={journal.transaction_id[:12]}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    try:
        client = init_fn()
    except Exception as exc:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=_sanitize_exception_classes(exc),
        ) from None
    if client is None:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )

    eval_fn = getattr(client, "eval", None)
    if not callable(eval_fn):
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )

    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)

    def _ns(key: str) -> str:
        return fix_ns(key=key) if callable(fix_ns) else key

    num_cd = len(journal.cooldown_keys)

    # Build namespaced keys: [receipt, cooldown_keys..., identity_key]
    ns_receipt = _ns(journal.receipt_key)
    ns_cd_keys = [
        _ns(
            build_aawm_alias_routing_durable_cache_key(
                alias_family=family, state_kind="cooldown", state_key=k
            )
        )
        for k in journal.cooldown_keys
    ]
    # Single identity key (all lane members map to one identity set).
    ns_id_key = (
        _ns(journal.identity_keys[0]) if journal.identity_keys else None
    )

    keys = [ns_receipt] + ns_cd_keys
    if ns_id_key is not None:
        keys.append(ns_id_key)

    # ARGV: [num_cd, lane_members...]
    argv = [str(num_cd)] + list(journal.lane_members)

    try:
        result = await eval_fn(
            _LUA_ROLLBACK_COOLDOWN_TRANSACTION,
            len(keys),
            *keys,
            *argv,
        )
    except Exception as exc:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=_sanitize_exception_classes(exc),
        ) from None

    code = int(result)
    if code == -1:
        raise RollbackReceiptMissingError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=(),
        )
    if code == -3:
        raise RollbackDriftError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=(),
        )
    if code != 1:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=("LuaRollbackError",),
        )


# ---------------------------------------------------------------------------
# CFG-004 Wave B: Bounded identity-set inspection and atomic compare-and-clear
# ---------------------------------------------------------------------------
#
# Strict identity-set inspection (bounded SMEMBERS, no SCAN/KEYS), atomic
# Lua compare-and-clear with exact expected membership verification, pre-image
# journaling, DualCache invalidation, postcondition verification, and
# reconciliation for lost EVAL responses.


class MembershipDriftError(PublicationTransactionError):
    """Identity set membership drifted from expected; no mutations applied."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("detail", "membership drift detected")
        super().__init__(**kwargs)  # type: ignore[arg-type]


class ClearIndeterminateError(PublicationTransactionError):
    """Clear transaction outcome is indeterminate after lost EVAL response."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("detail", "clear outcome indeterminate")
        super().__init__(**kwargs)  # type: ignore[arg-type]


class RollbackDriftError(PublicationTransactionError):
    """Rollback rejected: current state drifted from receipt post-image."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("detail", "rollback drift detected; state modified since commit")
        super().__init__(**kwargs)  # type: ignore[arg-type]


class RollbackReceiptMissingError(PublicationTransactionError):
    """Rollback cannot proceed: receipt missing or expired (indeterminate)."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("detail", "receipt missing or expired; outcome indeterminate")
        super().__init__(**kwargs)  # type: ignore[arg-type]


class IdentitySetOverBoundError(RuntimeError):
    """Identity set cardinality exceeds the inspection bound."""

    def __init__(self, *, cardinality: int, bound: int, context: str) -> None:
        self.cardinality = cardinality
        self.bound = bound
        super().__init__(
            f"AAWM alias routing durable {context}: "
            f"identity set cardinality {cardinality} exceeds bound {bound}"
        )


PHASE_CLEAR_COMMITTED = "CLEAR_COMMITTED"

_DEFAULT_CLEAR_RECEIPT_TTL = 300


@dataclass
class IdentitySetInspection:
    """Result of bounded identity-set inspection (CFG-004 Wave B)."""

    identity_key: str
    exists: bool
    members: frozenset
    cardinality: int
    ttl_remaining_seconds: Union[float, UnboundedExpiry, None] = None


@dataclass
class ClearTransactionJournal:
    """Journal for a committed clear transaction."""

    transaction_id: str
    phase: str
    alias_family: str
    identity_hash: str
    cooldown_keys: list
    lane_members: list
    expected_members: list
    identity_key: str
    receipt_key: str
    receipt_ttl: int


@dataclass
class ClearTransactionResult:
    """Result of a successful clear_cooldown_transaction."""

    transaction_id: str
    phase: str
    journal: ClearTransactionJournal
    keys_deleted: int
    members_removed: int


def _invalidate_in_memory_keys(
    dual_cache: object,
    cache_keys: list,
    context: str,
) -> None:
    """Delete each cache key from the DualCache in-memory tier (strict)."""
    in_memory_cache = getattr(dual_cache, "in_memory_cache", None)
    if in_memory_cache is None:
        return
    mem_delete = getattr(in_memory_cache, "delete_cache", None)
    if not callable(mem_delete):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "in_memory_cache missing delete_cache"
        )
    for key in cache_keys:
        try:
            mem_delete(key)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                f"in-memory invalidation failed: {exc}"
            ) from exc


_LUA_INSPECT_IDENTITY_SET = """
local id_key = KEYS[1]
local max_members = tonumber(ARGV[1])

-- Atomic bound check: SCARD before SMEMBERS.
local card = redis.call('SCARD', id_key)
if card > max_members then
    return {-1, card, -2}
end

local members = {}
if card > 0 then
    members = redis.call('SMEMBERS', id_key)
end
local ttl = redis.call('TTL', id_key)

-- Return: {1, cardinality, ttl, member1, member2, ...}
local result = {1, card, ttl}
for i, m in ipairs(members) do
    result[3 + i] = m
end
return result
"""


async def inspect_identity_set(
    *,
    alias_family: str,
    identity_hash: str,
    max_members: int = 1024,
) -> IdentitySetInspection:
    """Bounded strict inspection of an identity set via one atomic Lua EVAL.

    A single Lua operation checks SCARD before SMEMBERS and returns no
    oversized member list.  Raises IdentitySetOverBoundError if the set
    exceeds ``max_members``, RuntimeError on any Redis error, ValueError
    on unknown ``alias_family``.
    """
    context = f"inspect-identity family={alias_family}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )
    identity_cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family, state_kind="lane_identity", state_key=identity_hash
    )
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: dual cache has no redis_cache"
        )
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: missing init_async_client"
        )
    try:
        client = init_fn()
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: failed to init redis client"
        ) from exc
    if client is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client unavailable"
        )
    eval_fn = getattr(client, "eval", None)
    if not callable(eval_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "redis client missing eval (atomic inspection unavailable)"
        )
    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    ns_key = fix_ns(key=identity_cache_key) if callable(fix_ns) else identity_cache_key

    bound = max(1, int(max_members))

    try:
        result = await eval_fn(_LUA_INSPECT_IDENTITY_SET, 1, ns_key, str(bound))
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: lua inspection failed"
        ) from exc

    if not isinstance(result, (list, tuple)) or len(result) < 3:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: unexpected lua result format"
        )

    status = int(result[0])
    cardinality = int(result[1])
    ttl_int = int(result[2])

    if status == -1:
        raise IdentitySetOverBoundError(
            cardinality=cardinality, bound=bound, context=context
        )
    if status != 1:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: unexpected lua status {status}"
        )

    raw_members = result[3:]
    members = frozenset(
        m.decode("utf-8") if isinstance(m, bytes) else str(m)
        for m in raw_members
    )

    exists = cardinality > 0 or ttl_int != -2

    ttl_remaining: Union[float, UnboundedExpiry, None]
    if ttl_int == -2:
        ttl_remaining = None
    elif ttl_int == -1:
        ttl_remaining = UNBOUNDED_EXPIRY
    else:
        ttl_remaining = float(ttl_int)

    return IdentitySetInspection(
        identity_key=identity_cache_key,
        exists=exists,
        members=members,
        cardinality=cardinality,
        ttl_remaining_seconds=ttl_remaining,
    )


# ---------------------------------------------------------------------------
# Lua script: atomic compare-and-clear
# ---------------------------------------------------------------------------
#
# KEYS layout:
#   KEYS[1]              = receipt key
#   KEYS[2..N+1]         = cooldown keys (N = num_cd)
#   KEYS[N+2]            = identity set key
#
# ARGV layout:
#   ARGV[1]  = num_cd (N)
#   ARGV[2]  = num_expected (E)
#   ARGV[3]  = num_remove (R)
#   ARGV[4]  = receipt_ttl_seconds
#   ARGV[5]  = receipt_json
#   ARGV[6..5+E]         = expected members (drift check)
#   ARGV[6+E..5+E+R]     = members to SREM
#
# Returns:
#   1  -> success (all mutations applied)
#   -1 -> membership drift detected (no mutations applied)

_LUA_CLEAR_COOLDOWN_TRANSACTION = """
local receipt_key = KEYS[1]
local num_cd = tonumber(ARGV[1])
local num_expected = tonumber(ARGV[2])
local num_remove = tonumber(ARGV[3])
local receipt_ttl = tonumber(ARGV[4])
local receipt_json = ARGV[5]

local id_key = KEYS[2 + num_cd]

-- Phase 1: Drift check -- verify exact membership.
local actual_members = redis.call('SMEMBERS', id_key)
local actual_set = {}
for _, m in ipairs(actual_members) do
    actual_set[m] = true
end
local expected_set = {}
for i = 1, num_expected do
    local member = ARGV[5 + i]
    expected_set[member] = true
    if not actual_set[member] then
        return -1
    end
end
for _, m in ipairs(actual_members) do
    if not expected_set[m] then
        return -1
    end
end

-- Phase 2: Journal pre-images (value + TTL) for cooldown keys.
local preimages = {}
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local val = redis.call('GET', cd_key)
    local ttl = redis.call('TTL', cd_key)
    preimages[i] = {val, ttl}
end

-- Phase 2b: Journal identity-set TTL before mutation.
local id_ttl = redis.call('TTL', id_key)

-- Phase 3: Delete selected cooldown keys.
for i = 1, num_cd do
    redis.call('DEL', KEYS[1 + i])
end

-- Phase 4: Remove selected members from identity set.
for i = 1, num_remove do
    local member = ARGV[5 + num_expected + i]
    redis.call('SREM', id_key, member)
end

-- Phase 5: Delete identity set if now empty.
if redis.call('SCARD', id_key) == 0 then
    redis.call('DEL', id_key)
end

-- Phase 6: Journal post-images (state after mutation) for rollback drift check.
local postimages = {}
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local pval = redis.call('GET', cd_key)
    local pttl = redis.call('TTL', cd_key)
    postimages[i] = {pval, pttl}
end
local id_post_members = {}
local id_post_card = redis.call('SCARD', id_key)
if id_post_card > 0 then
    id_post_members = redis.call('SMEMBERS', id_key)
end
local id_post_ttl = redis.call('TTL', id_key)

-- Phase 7: Write receipt with exact pre-images and post-images for rollback.
local preimage_arr = {}
for i = 1, num_cd do
    local entry = {}
    entry['v'] = preimages[i][1]
    entry['t'] = preimages[i][2]
    preimage_arr[i] = entry
end
local postimage_arr = {}
for i = 1, num_cd do
    local entry = {}
    entry['v'] = postimages[i][1]
    entry['t'] = postimages[i][2]
    postimage_arr[i] = entry
end
local receipt = cjson.decode(receipt_json)
receipt['preimages'] = preimage_arr
receipt['postimages'] = postimage_arr
local id_entry = {}
id_entry['members'] = actual_members
id_entry['ttl'] = id_ttl
id_entry['key'] = id_key
receipt['identity_preimage'] = id_entry
local id_post_entry = {}
id_post_entry['members'] = id_post_members
id_post_entry['ttl'] = id_post_ttl
id_post_entry['key'] = id_key
receipt['identity_postimage'] = id_post_entry
local removed_arr = {}
for i = 1, num_remove do
    removed_arr[i] = ARGV[5 + num_expected + i]
end
receipt['removed_members'] = removed_arr
redis.call('SET', receipt_key, cjson.encode(receipt))
if receipt_ttl > 0 then
    redis.call('EXPIRE', receipt_key, receipt_ttl)
end

return 1
"""


async def clear_cooldown_transaction(  # noqa: PLR0915
    *,
    alias_family: str,
    identity_hash: str,
    cooldown_keys: list,
    expected_members: list,
    lane_members: Optional[list] = None,
    receipt_ttl_seconds: int = _DEFAULT_CLEAR_RECEIPT_TTL,
) -> ClearTransactionResult:
    """Atomic compare-and-clear: verify identity membership, delete cooldown
    keys, remove selected members, delete empty sets.

    One Lua EVAL: drift check, journal pre-images, DEL cooldown keys, SREM
    members, DEL empty identity set, write receipt.

    Fails closed: no Redis backend, missing EVAL, or Redis error raises
    RuntimeError.  Membership drift raises MembershipDriftError.  Lost EVAL
    response triggers reconciliation; indeterminate outcome raises
    ClearIndeterminateError.

    On success: strict postcondition verification, then DualCache in-memory
    invalidation for each cooldown cache key.
    """
    if lane_members is None:
        lane_members = list(expected_members)
    context = f"clear-txn family={alias_family}"
    family = _validate_alias_family(alias_family, context)
    transaction_id = _uuid.uuid4().hex

    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )

    # Build namespaced keys.
    receipt_state_key = f"clear-receipt:{transaction_id}"
    receipt_cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family, state_kind="txn_receipt", state_key=receipt_state_key
    )
    cd_cache_keys = [
        build_aawm_alias_routing_durable_cache_key(
            alias_family=family, state_kind="cooldown", state_key=k
        )
        for k in cooldown_keys
    ]
    identity_cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family, state_kind="lane_identity", state_key=identity_hash
    )

    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: dual cache has no redis_cache"
        )
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: missing init_async_client"
        )
    try:
        client = init_fn()
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: failed to init redis client"
        ) from exc
    if client is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: redis client unavailable"
        )
    eval_fn = getattr(client, "eval", None)
    if not callable(eval_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "redis client missing eval (atomic transaction unavailable)"
        )

    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)

    def _ns(key: str) -> str:
        return fix_ns(key=key) if callable(fix_ns) else key

    ns_receipt = _ns(receipt_cache_key)
    ns_cd_keys = [_ns(k) for k in cd_cache_keys]
    ns_id_key = _ns(identity_cache_key)

    num_cd = len(cooldown_keys)
    num_expected = len(expected_members)
    num_remove = len(lane_members)
    rcpt_ttl = max(1, int(receipt_ttl_seconds))

    receipt = {
        "txn_id": transaction_id,
        "family": family,
        "num_keys": num_cd,
        "operation": "clear",
    }
    receipt_json = _json.dumps(receipt, separators=(",", ":"))

    # KEYS: [receipt, cd_keys..., id_key]
    keys = [ns_receipt] + ns_cd_keys + [ns_id_key]
    # ARGV: [num_cd, num_expected, num_remove, receipt_ttl, receipt_json,
    #         expected_members..., lane_members...]
    argv = (
        [str(num_cd), str(num_expected), str(num_remove), str(rcpt_ttl), receipt_json]
        + list(expected_members)
        + list(lane_members)
    )

    try:
        result = await eval_fn(
            _LUA_CLEAR_COOLDOWN_TRANSACTION,
            len(keys),
            *keys,
            *argv,
        )
    except Exception as exc:
        # Lost EVAL response: reconcile via receipt presence.
        try:
            committed = await reconcile_clear_transaction(
                alias_family=family,
                transaction_id=transaction_id,
                cooldown_cache_keys=ns_cd_keys,
                identity_cache_key=ns_id_key,
                lane_members=list(lane_members),
            )
            if committed:
                # Run ALL strict Redis postconditions (never skip on
                # reconciled commit).
                get_fn_r = getattr(client, "get", None)
                if not callable(get_fn_r):
                    raise RuntimeError(
                        f"AAWM alias routing durable {context}: "
                        "postcondition check unavailable (missing get)"
                    )
                for ns_key_r in ns_cd_keys:
                    raw_r = await get_fn_r(ns_key_r)
                    if raw_r is not None:
                        raise RuntimeError(
                            f"AAWM alias routing durable {context}: "
                            "cooldown key still present after reconciled clear"
                        )
                sismember_fn_r = getattr(client, "sismember", None)
                if not callable(sismember_fn_r):
                    raise RuntimeError(
                        f"AAWM alias routing durable {context}: "
                        "postcondition check unavailable (missing sismember)"
                    )
                for member_r in lane_members:
                    still_r = await sismember_fn_r(ns_id_key, member_r)
                    if still_r:
                        raise RuntimeError(
                            f"AAWM alias routing durable {context}: "
                            "member still in identity set after reconciled clear"
                        )
                # Mandatory DualCache invalidation (never swallow errors).
                _invalidate_in_memory_keys(dual_cache, cd_cache_keys, context)
                journal = ClearTransactionJournal(
                    transaction_id=transaction_id,
                    phase=PHASE_CLEAR_COMMITTED,
                    alias_family=family,
                    identity_hash=identity_hash,
                    cooldown_keys=list(cooldown_keys),
                    lane_members=list(lane_members),
                    expected_members=list(expected_members),
                    identity_key=identity_cache_key,
                    receipt_key=receipt_cache_key,
                    receipt_ttl=rcpt_ttl,
                )
                return ClearTransactionResult(
                    transaction_id=transaction_id,
                    phase=PHASE_CLEAR_COMMITTED,
                    journal=journal,
                    keys_deleted=num_cd,
                    members_removed=num_remove,
                )
        except (ClearIndeterminateError, RollbackFailedError, RuntimeError):
            raise
        except Exception:
            pass
        raise ClearIndeterminateError(
            phase=PHASE_PREPARED,
            family=family,
            transaction_id_prefix=transaction_id[:12],
            identity_prefix=identity_hash[:12],
            key_count=num_cd,
            exception_classes=_sanitize_exception_classes(exc),
        ) from None

    code = int(result)
    if code == -1:
        raise MembershipDriftError(
            phase=PHASE_PREPARED,
            family=family,
            transaction_id_prefix=transaction_id[:12],
            identity_prefix=identity_hash[:12],
            key_count=num_cd,
            exception_classes=(),
        )
    if code != 1:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: unexpected lua result {code}"
        )

    # Strict postcondition verification.
    get_fn = getattr(client, "get", None)
    if not callable(get_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "postcondition check unavailable (missing get)"
        )
    for ns_key in ns_cd_keys:
        try:
            raw = await get_fn(ns_key)
        except Exception as exc:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "postcondition check failed"
            ) from exc
        if raw is not None:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "cooldown key still present after clear"
            )
    sismember_fn = getattr(client, "sismember", None)
    if not callable(sismember_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "postcondition check unavailable (missing sismember)"
        )
    for member in lane_members:
        try:
            still = await sismember_fn(ns_id_key, member)
        except Exception as exc:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "postcondition membership check failed"
            ) from exc
        if still:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "member still in identity set after clear"
            )

    # DualCache in-memory invalidation.
    _invalidate_in_memory_keys(dual_cache, cd_cache_keys, context)

    journal = ClearTransactionJournal(
        transaction_id=transaction_id,
        phase=PHASE_CLEAR_COMMITTED,
        alias_family=family,
        identity_hash=identity_hash,
        cooldown_keys=list(cooldown_keys),
        lane_members=list(lane_members),
        expected_members=list(expected_members),
        identity_key=identity_cache_key,
        receipt_key=receipt_cache_key,
        receipt_ttl=rcpt_ttl,
    )
    return ClearTransactionResult(
        transaction_id=transaction_id,
        phase=PHASE_CLEAR_COMMITTED,
        journal=journal,
        keys_deleted=num_cd,
        members_removed=num_remove,
    )


async def reconcile_clear_transaction(
    *,
    alias_family: str,
    transaction_id: str,
    cooldown_cache_keys: list[str],
    identity_cache_key: str,
    lane_members: list[str],
) -> bool:
    """Reconcile after lost EVAL response for a clear transaction.

    Checks whether the clear receipt key exists (commit evidence only).
    When the receipt is present, verifies the full clear post-image:
    every cooldown key must be absent and every lane member must be
    absent from the identity set.  Receipt existence alone never
    authorizes success.
    Returns True only when receipt AND all postconditions hold.
    Returns False if the receipt is absent (no commit occurred).
    Fails closed on Redis errors or postcondition violations.
    """
    context = f"reconcile-clear family={alias_family} txn={transaction_id[:12]}"
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: no Redis cache available"
        )
    receipt_state_key = f"clear-receipt:{transaction_id}"
    receipt_cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=family, state_kind="txn_receipt", state_key=receipt_state_key
    )
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: dual cache has no redis_cache"
        )
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: missing init_async_client"
        )
    try:
        client = init_fn()
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: client init failed"
        ) from exc
    if client is None:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: client unavailable"
        )
    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    ns_key = fix_ns(key=receipt_cache_key) if callable(fix_ns) else receipt_cache_key
    get_fn = getattr(client, "get", None)
    if not callable(get_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: client missing get"
        )
    try:
        raw = await get_fn(ns_key)
    except Exception as exc:
        raise RuntimeError(
            f"AAWM alias routing durable {context}: receipt check failed"
        ) from exc
    if raw is None:
        return False

    # Strict postcondition: verify clear post-image.
    # Receipt is commit evidence only; success requires full post-image.
    for cd_key in cooldown_cache_keys:
        try:
            cd_raw = await get_fn(cd_key)
        except Exception as exc:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "clear postcondition check failed"
            ) from exc
        if cd_raw is not None:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "cooldown key still present after reconciled clear"
            )
    sismember_fn = getattr(client, "sismember", None)
    if not callable(sismember_fn):
        raise RuntimeError(
            f"AAWM alias routing durable {context}: "
            "postcondition check unavailable (missing sismember)"
        )
    for member in lane_members:
        try:
            still = await sismember_fn(identity_cache_key, member)
        except Exception as exc:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "clear membership postcondition failed"
            ) from exc
        if still:
            raise RuntimeError(
                f"AAWM alias routing durable {context}: "
                "member still in identity set after reconciled clear"
            )
    return True


# ---------------------------------------------------------------------------
# Lua script: rollback clear transaction
# ---------------------------------------------------------------------------
#
# Restores exact pre-images from the clear receipt: cooldown key values
# and TTLs (including -1 persistent and -2 absent), identity set members,
# and identity key TTL.  Deletes the receipt on full success.
#
# KEYS layout:
#   KEYS[1]           = receipt key
#   KEYS[2..N+1]      = cooldown keys (N = num_cd)
#   KEYS[N+2]         = identity set key
#
# ARGV layout:
#   ARGV[1]  = num_cd (N)
#
# Returns:
#   1  -> success
#   -1 -> receipt missing (nothing to restore)
#   -2 -> parse/partial error (receipt RETAINED for forensics)

_LUA_ROLLBACK_CLEAR_TRANSACTION = """
local receipt_key = KEYS[1]
local num_cd = tonumber(ARGV[1])

local receipt_raw = redis.call('GET', receipt_key)
if not receipt_raw then
    return -1
end

local ok, receipt = pcall(cjson.decode, receipt_raw)
if not ok then
    return -2
end

local preimages = receipt['preimages']
if not preimages then
    return -2
end

-- Phase 1: Drift check -- compare exact current state to post-image.
-- If current state does not match the recorded post-image, another writer
-- modified state after the commit; rollback must reject.
local postimages = receipt['postimages']
if postimages then
    for i = 1, num_cd do
        local cd_key = KEYS[1 + i]
        local cur_val = redis.call('GET', cd_key)
        local cur_ttl = redis.call('TTL', cd_key)
        local post = postimages[i]
        if post then
            local post_val = post['v']
            local post_ttl = tonumber(post['t'])
            -- Compare values (nil/false equivalence for absent keys)
            local cur_absent = (cur_val == false or cur_val == nil)
            local post_absent = (post_val == false or post_val == nil)
            if cur_absent ~= post_absent then
                return -3
            end
            if not cur_absent and cur_val ~= post_val then
                return -3
            end
            -- TTL comparison: allow 1-second tolerance for elapsed time
            if post_ttl >= 0 and cur_ttl >= 0 then
                if math.abs(cur_ttl - post_ttl) > 1 then
                    return -3
                end
            elseif post_ttl ~= cur_ttl then
                -- -1 vs -2 or -1 vs positive: drift
                if not (post_ttl < 0 and cur_ttl < 0) then
                    return -3
                end
            end
        end
    end
end

-- Phase 1b: Identity set drift check against post-image.
local id_postimage = receipt['identity_postimage']
if id_postimage and id_postimage['key'] then
    local id_key = id_postimage['key']
    local post_members = id_postimage['members']
    local post_set = {}
    local post_count = 0
    if post_members then
        for _, m in ipairs(post_members) do
            post_set[m] = true
            post_count = post_count + 1
        end
    end
    local cur_card = redis.call('SCARD', id_key)
    if cur_card ~= post_count then
        return -3
    end
    if cur_card > 0 then
        local cur_members = redis.call('SMEMBERS', id_key)
        for _, m in ipairs(cur_members) do
            if not post_set[m] then
                return -3
            end
        end
    end
end

-- Phase 2: Restore cooldown key pre-images.
for i = 1, num_cd do
    local cd_key = KEYS[1 + i]
    local entry = preimages[i]
    if entry then
        local val = entry['v']
        local ttl = tonumber(entry['t'])
        if ttl == -2 or val == false or val == nil then
            redis.call('DEL', cd_key)
        else
            redis.call('SET', cd_key, val)
            if ttl == -1 then
                redis.call('PERSIST', cd_key)
            elseif ttl > 0 then
                redis.call('EXPIRE', cd_key, ttl)
            end
        end
    else
        redis.call('DEL', cd_key)
    end
end

-- Phase 3: Restore identity set from pre-image.
local id_preimage = receipt['identity_preimage']
if id_preimage and id_preimage['key'] then
    local id_key = id_preimage['key']
    -- Clear current membership and restore exact pre-image.
    redis.call('DEL', id_key)
    local prior_members = id_preimage['members']
    if prior_members then
        for _, m in ipairs(prior_members) do
            redis.call('SADD', id_key, m)
        end
    end
    local id_ttl = tonumber(id_preimage['ttl'])
    if id_ttl == -2 then
        redis.call('DEL', id_key)
    elseif id_ttl == -1 then
        redis.call('PERSIST', id_key)
    elseif id_ttl and id_ttl > 0 then
        redis.call('EXPIRE', id_key, id_ttl)
    end
end

-- Phase 4: Delete receipt (only on full success).
redis.call('DEL', receipt_key)

return 1
"""


async def rollback_clear_transaction(
    *,
    alias_family: str,
    journal: ClearTransactionJournal,
) -> None:
    """Restore exact pre-images from a committed clear transaction atomically.

    Executes a single Lua EVAL that reads the receipt, restores cooldown
    key pre-images (including TTL -1 persistent and -2 absent semantics),
    restores identity set members and TTL, and deletes the receipt.

    On restoration error the receipt is RETAINED (not deleted) so operators
    can inspect the failure.  Any failure raises RollbackFailedError with
    sanitized context.
    """
    context = (
        f"rollback-clear family={alias_family} txn={journal.transaction_id[:12]}"
    )
    family = _validate_alias_family(alias_family, context)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    try:
        client = init_fn()
    except Exception as exc:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=_sanitize_exception_classes(exc),
        ) from None
    if client is None:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )
    eval_fn = getattr(client, "eval", None)
    if not callable(eval_fn):
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=len(journal.cooldown_keys),
            exception_classes=("RuntimeError",),
        )

    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)

    def _ns(key: str) -> str:
        return fix_ns(key=key) if callable(fix_ns) else key

    num_cd = len(journal.cooldown_keys)
    ns_receipt = _ns(journal.receipt_key)
    ns_cd_keys = [
        _ns(
            build_aawm_alias_routing_durable_cache_key(
                alias_family=family, state_kind="cooldown", state_key=k
            )
        )
        for k in journal.cooldown_keys
    ]
    ns_id_key = _ns(journal.identity_key)

    keys = [ns_receipt] + ns_cd_keys + [ns_id_key]
    argv = [str(num_cd)]

    try:
        result = await eval_fn(
            _LUA_ROLLBACK_CLEAR_TRANSACTION,
            len(keys),
            *keys,
            *argv,
        )
    except Exception as exc:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=_sanitize_exception_classes(exc),
        ) from None

    code = int(result)
    if code == -1:
        raise RollbackReceiptMissingError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=(),
        )
    if code == -3:
        raise RollbackDriftError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=(),
        )
    if code != 1:
        raise RollbackFailedError(
            phase=journal.phase,
            family=family,
            transaction_id_prefix=journal.transaction_id[:12],
            identity_prefix=journal.identity_hash[:12],
            key_count=num_cd,
            exception_classes=("LuaRollbackError",),
        )


async def read_aawm_alias_routing_state(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
    last_good_local: Any = None,
    dual_cache: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Durable-first routing-state reader used by affinity and cooldown getters.

    A successful Redis miss is ``confirmed_miss``. A Redis exception is
    ``degraded_local`` / ``durable_error`` and must not be treated as empty.
    Callers should pass the DualCache they already resolved so getter-level
    patches and per-worker fakes are not lost to a second lookup.
    """

    def _local_payload() -> Any:
        if isinstance(last_good_local, dict):
            return dict(last_good_local)
        return last_good_local

    if dual_cache is None:
        dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        if last_good_local:
            payload = _local_payload()
            return {
                "payload": payload,
                "source": "memory",
                "affinity_state_source": "memory",
                "confirmed_miss": False,
                "durable_miss": False,
                "durable_error": False,
                "read_error": False,
            }
        return {
            "payload": None,
            "source": "unavailable",
            "affinity_state_source": "unavailable",
            "confirmed_miss": False,
            "durable_miss": False,
            "durable_error": False,
            "read_error": False,
        }

    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=alias_family,
        state_kind=state_kind,
        state_key=state_key,
    )
    try:
        async_get_cache = getattr(dual_cache, "async_get_cache", None)
        if not callable(async_get_cache):
            raise RuntimeError("redis client missing async_get_cache")
        payload = await async_get_cache(key=cache_key, raise_on_error=True)
    except Exception:
        payload = _local_payload()
        source = "degraded_local" if last_good_local else "durable_error"
        return {
            "payload": payload,
            "source": source,
            "affinity_state_source": source,
            "confirmed_miss": False,
            "durable_miss": False,
            "durable_error": True,
            "read_error": True,
        }

    if isinstance(payload, dict) and parse_aawm_alias_routing_durable_expiry(payload) is not None:
        return {
            "payload": dict(payload),
            "source": "durable_cache",
            "affinity_state_source": "durable_cache",
            "confirmed_miss": False,
            "durable_miss": False,
            "durable_error": False,
            "read_error": False,
        }

    local_payload = _local_payload() if last_good_local else None
    miss_source = "local_lease" if last_good_local else "confirmed_miss"
    return {
        "payload": local_payload,
        "source": miss_source,
        "affinity_state_source": miss_source,
        "confirmed_miss": True,
        "durable_miss": True,
        "durable_error": False,
        "read_error": False,
    }
