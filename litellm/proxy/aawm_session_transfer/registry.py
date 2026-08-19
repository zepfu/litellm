"""Shared multi-worker session-transfer registry.

Uses the existing AAWM alias-routing Redis DualCache connection. Records live
in a dedicated transfer namespace and never share cooldown/affinity keys.
Telemetry failures never raise into serving traffic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from litellm.proxy.aawm_alias_routing_redis import (
    get_dual_cache,
    get_status as get_alias_routing_redis_status,
)
from litellm.proxy.aawm_session_transfer.schema import (
    DEFAULT_QUERY_LIMIT,
    MAX_INDEX_MEMBERS,
    SCHEMA_VERSION,
    TERMINAL_PHASE_SET,
    clamp_limit,
    coerce_non_negative_int,
    is_terminal_phase,
    iter_identity_values,
    merge_records,
    new_transfer_record,
    normalize_phase,
    public_transfer_record,
    sanitize_identity,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

TRANSFER_KEY_PREFIX = "aawm:session-transfer"
TRANSFER_NAMESPACE_ENV = "AAWM_SESSION_TRANSFER_STATE_NAMESPACE"
DEFAULT_NAMESPACE = "aawm-transfer-v1"
ACTIVE_TTL_SECONDS = 180
TERMINAL_TTL_SECONDS = 45
STALE_AFTER_SECONDS = 30
HEARTBEAT_MIN_INTERVAL_SECONDS = 0.75
INDEX_TTL_SECONDS = 180

_registry_override: Optional["SessionTransferRegistry"] = None
_default_registry: Optional["SessionTransferRegistry"] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return utc_now_iso(_now()) or ""


def resolve_transfer_namespace() -> str:
    explicit = (os.getenv(TRANSFER_NAMESPACE_ENV) or "").strip()
    if explicit:
        return explicit
    runtime_environment = (
        (
            os.getenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT")
            or os.getenv("LITELLM_AAWM_ERROR_LOG_ENV")
            or ""
        )
        .strip()
        .lower()
    )
    if runtime_environment in {"dev", "development"}:
        return "aawm-transfer-dev-v1"
    if runtime_environment in {"prod", "production"}:
        return "aawm-transfer-prod-v1"
    return DEFAULT_NAMESPACE


def _hash_identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class InMemoryTransferStore:
    """Process-local fallback used when Redis is unavailable."""

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [key for key, until in self._expiry.items() if until <= now]
        for key in expired:
            self._values.pop(key, None)
            self._expiry.pop(key, None)

    async def async_get_cache(self, key: str, **_: Any) -> Any:
        self._purge_expired()
        return self._values.get(key)

    async def async_set_cache(self, key: str, value: Any, ttl: Optional[float] = None, **_: Any) -> None:
        self._purge_expired()
        self._values[key] = value
        if ttl is not None:
            self._expiry[key] = time.time() + float(ttl)
        else:
            self._expiry.pop(key, None)

    async def async_batch_get_cache(self, keys: list, **_: Any) -> List[Any]:
        self._purge_expired()
        return [self._values.get(key) for key in keys]


class SessionTransferRegistry:
    def __init__(
        self,
        *,
        cache: Any = None,
        source_instance: Optional[str] = None,
        now_fn=None,
    ) -> None:
        self._cache = cache
        self._memory = InMemoryTransferStore()
        self._source_instance = source_instance or socket.gethostname()
        self._now_fn = now_fn or _now
        self._last_heartbeat_mono: Dict[str, float] = {}
        self._last_error_class: Optional[str] = None
        self._degraded = False

    def bind_cache(self, cache: Any) -> None:
        self._cache = cache

    def _store(self) -> Any:
        if self._cache is not None:
            return self._cache
        try:
            cache = get_dual_cache()
        except Exception:
            cache = None
        if cache is not None:
            self._cache = cache
            return cache
        return self._memory

    def _redis_status(self) -> Dict[str, Any]:
        try:
            return get_alias_routing_redis_status()
        except Exception:
            return {"mode": "memory", "state_source": "memory", "reachable": False}

    def registry_status(self) -> Dict[str, Any]:
        redis_status = self._redis_status()
        attached = self._store() is not self._memory and self._cache is not None
        reachable = redis_status.get("reachable")
        if attached and reachable is True and not self._degraded:
            state = "ok"
        elif attached:
            state = "degraded"
        else:
            state = "unavailable"
        return {
            "state": state,
            "mode": "redis" if attached else "memory",
            "state_source": redis_status.get("state_source") or (
                "durable_cache" if attached else "memory"
            ),
            "reachable": reachable,
            "error_class": self._last_error_class,
        }

    def _record_key(self, call_id: str) -> str:
        namespace = resolve_transfer_namespace()
        return f"{TRANSFER_KEY_PREFIX}:{namespace}:call:{_hash_identity(call_id)}"

    def _index_key(self, field: str, value: str) -> str:
        namespace = resolve_transfer_namespace()
        return (
            f"{TRANSFER_KEY_PREFIX}:{namespace}:idx:{field}:{_hash_identity(value)}"
        )

    def _decode_record(self, raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            return dict(raw)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return None

    async def _read_raw(self, key: str) -> Any:
        store = self._store()
        try:
            return await store.async_get_cache(key=key)
        except Exception as exc:
            self._degraded = True
            self._last_error_class = type(exc).__name__
            logger.debug("session-transfer registry read failed", exc_info=True)
            if store is not self._memory:
                try:
                    return await self._memory.async_get_cache(key=key)
                except Exception:
                    return None
            return None

    async def _write_raw(self, key: str, value: Any, ttl: int) -> bool:
        encoded = json.dumps(value, separators=(",", ":"), default=str)
        store = self._store()
        try:
            await store.async_set_cache(key=key, value=encoded, ttl=ttl)
            if store is not self._memory:
                await self._memory.async_set_cache(key=key, value=encoded, ttl=ttl)
            return True
        except Exception as exc:
            self._degraded = True
            self._last_error_class = type(exc).__name__
            logger.debug("session-transfer registry write failed", exc_info=True)
            try:
                await self._memory.async_set_cache(key=key, value=encoded, ttl=ttl)
            except Exception:
                pass
            return False

    async def _read_index(self, field: str, value: str) -> List[str]:
        raw = await self._read_raw(self._index_key(field, value))
        if raw is None:
            return []
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, str)][:MAX_INDEX_MEMBERS]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, str)][:MAX_INDEX_MEMBERS]
        return []

    async def _write_index(self, field: str, value: str, call_ids: List[str], ttl: int) -> None:
        bounded = call_ids[-MAX_INDEX_MEMBERS:]
        await self._write_raw(self._index_key(field, value), bounded, ttl)

    def _annotate_freshness(self, record: Dict[str, Any]) -> Dict[str, Any]:
        phase = normalize_phase(record.get("phase"))
        now = self._now_fn()
        last_heartbeat = record.get("last_heartbeat_at")
        stale = False
        if phase not in TERMINAL_PHASE_SET and last_heartbeat:
            try:
                heartbeat_dt = datetime.fromisoformat(
                    str(last_heartbeat).replace("Z", "+00:00")
                )
                age = (now - heartbeat_dt).total_seconds()
                stale = age > STALE_AFTER_SECONDS
            except (TypeError, ValueError):
                stale = True
        record["phase"] = phase
        record["stale"] = stale
        record["active"] = phase not in TERMINAL_PHASE_SET and not stale
        if phase in TERMINAL_PHASE_SET:
            record["freshness"] = "terminal"
            record["terminal_state"] = phase
        elif stale:
            record["freshness"] = "stale"
        else:
            record["freshness"] = "live"
        if self._degraded:
            record["redis_degraded"] = True
        return record

    async def upsert(
        self,
        identity: Mapping[str, Any],
        updates: Optional[Mapping[str, Any]] = None,
        *,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        try:
            call_id = sanitize_identity(identity.get("litellm_call_id"))
            if not call_id:
                return None
            existing = self._decode_record(await self._read_raw(self._record_key(call_id)))
            record = new_transfer_record() if existing is None else existing
            record = merge_records(record, identity)
            if updates:
                record = merge_records(record, updates)
            now_iso = utc_now_iso(self._now_fn())
            record["schema_version"] = SCHEMA_VERSION
            record["litellm_call_id"] = call_id
            record["source_instance"] = sanitize_identity(self._source_instance)
            if not record.get("received_at"):
                record["received_at"] = now_iso
            last_mono = self._last_heartbeat_mono.get(call_id, 0.0)
            now_mono = time.monotonic()
            if force or now_mono - last_mono >= HEARTBEAT_MIN_INTERVAL_SECONDS:
                record["last_heartbeat_at"] = now_iso
                self._last_heartbeat_mono[call_id] = now_mono
            record = self._annotate_freshness(record)
            ttl = (
                TERMINAL_TTL_SECONDS
                if is_terminal_phase(normalize_phase(record.get("phase")))
                else ACTIVE_TTL_SECONDS
            )
            wrote = await self._write_raw(self._record_key(call_id), record, ttl)
            if not wrote:
                record["redis_degraded"] = True
                record["freshness"] = (
                    "unavailable"
                    if record.get("freshness") != "terminal"
                    else record.get("freshness")
                )
            for field, value in iter_identity_values(record):
                if field == "litellm_call_id":
                    continue
                members = await self._read_index(field, value)
                if call_id not in members:
                    members.append(call_id)
                await self._write_index(field, value, members, INDEX_TTL_SECONDS)
            return record
        except Exception:
            logger.debug("session-transfer upsert failed", exc_info=True)
            return None

    async def mark_phase(
        self,
        identity: Mapping[str, Any],
        phase: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {"phase": normalize_phase(phase)}
        timestamp_key = {
            "request_preparing": "preparing_at",
            "awaiting_upstream": "awaiting_upstream_at",
            "response_streaming": "first_upstream_chunk_at",
            "completed": "finalized_at",
            "failed": "finalized_at",
            "cancelled": "finalized_at",
            "disconnected": "finalized_at",
            "timed_out": "finalized_at",
        }.get(updates["phase"])
        if timestamp_key:
            updates[timestamp_key] = utc_now_iso(self._now_fn())
        if extra:
            updates.update(dict(extra))
        if is_terminal_phase(updates["phase"]):
            updates["active"] = False
            updates["terminal_state"] = updates["phase"]
            updates["freshness"] = "terminal"
        return await self.upsert(identity, updates, force=True)

    async def record_chunks(
        self,
        identity: Mapping[str, Any],
        *,
        upstream_chunks: int = 0,
        upstream_bytes: int = 0,
        downstream_chunks: int = 0,
        downstream_bytes: int = 0,
        first_upstream: bool = False,
        first_downstream: bool = False,
    ) -> Optional[Dict[str, Any]]:
        now_iso = utc_now_iso(self._now_fn())
        updates: Dict[str, Any] = {
            "phase": "response_streaming",
            "upstream_chunk_count": coerce_non_negative_int(upstream_chunks) or 0,
            "upstream_byte_count": coerce_non_negative_int(upstream_bytes) or 0,
            "downstream_chunk_count": coerce_non_negative_int(downstream_chunks) or 0,
            "downstream_byte_count": coerce_non_negative_int(downstream_bytes) or 0,
        }
        if first_upstream:
            updates["first_upstream_chunk_at"] = now_iso
        if first_downstream:
            updates["first_downstream_chunk_at"] = now_iso
        return await self.upsert(identity, updates)

    async def finalize(
        self,
        identity: Mapping[str, Any],
        phase: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        terminal = normalize_phase(phase)
        if terminal not in TERMINAL_PHASE_SET:
            terminal = "failed"
        updates = {
            "phase": terminal,
            "active": False,
            "stale": False,
            "freshness": "terminal",
            "terminal_state": terminal,
            "finalized_at": utc_now_iso(self._now_fn()),
        }
        if extra:
            updates.update(dict(extra))
        return await self.upsert(identity, updates, force=True)

    async def get_by_call_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        cleaned = sanitize_identity(call_id)
        if not cleaned:
            return None
        record = self._decode_record(await self._read_raw(self._record_key(cleaned)))
        if record is None:
            return None
        return self._annotate_freshness(record)

    async def query(
        self,
        *,
        litellm_call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        codex_session_id: Optional[str] = None,
        canonical_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        active_only: bool = False,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> Dict[str, Any]:
        status = self.registry_status()
        wanted = clamp_limit(limit)
        records: List[Dict[str, Any]] = []
        seen: set[str] = set()

        async def _add(record: Optional[Dict[str, Any]]) -> None:
            if record is None:
                return
            annotated = self._annotate_freshness(record)
            call_id = sanitize_identity(annotated.get("litellm_call_id"))
            if not call_id or call_id in seen:
                return
            if active_only and not annotated.get("active"):
                return
            seen.add(call_id)
            records.append(annotated)

        if litellm_call_id:
            await _add(await self.get_by_call_id(litellm_call_id))
        identity_filters = (
            ("canonical_session_id", canonical_session_id),
            ("codex_session_id", codex_session_id),
            ("session_id", session_id),
            ("agent_id", agent_id),
        )
        truncated = False
        for field, value in identity_filters:
            cleaned = sanitize_identity(value)
            if not cleaned:
                continue
            members = await self._read_index(field, cleaned)
            if len(members) > wanted:
                truncated = True
            for call_id in members:
                if len(records) >= wanted:
                    truncated = True
                    break
                await _add(await self.get_by_call_id(call_id))
            if len(records) >= wanted:
                break

        records.sort(
            key=lambda item: str(item.get("last_heartbeat_at") or item.get("received_at") or ""),
            reverse=True,
        )
        bounded = records[:wanted]
        public_records = [public_transfer_record(item) for item in bounded]
        return {
            "schema_version": SCHEMA_VERSION,
            "registry": status,
            "result_count": len(public_records),
            "truncated": truncated or len(records) > wanted,
            "transfers": public_records,
        }


def get_session_transfer_registry() -> SessionTransferRegistry:
    global _default_registry
    if _registry_override is not None:
        return _registry_override
    if _default_registry is None:
        _default_registry = SessionTransferRegistry()
    return _default_registry


def set_session_transfer_registry_override(
    registry: Optional[SessionTransferRegistry],
) -> None:
    global _registry_override
    _registry_override = registry


def reset_session_transfer_registry() -> None:
    global _default_registry, _registry_override
    _registry_override = None
    _default_registry = SessionTransferRegistry()


async def safe_upsert(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    try:
        return await get_session_transfer_registry().upsert(*args, **kwargs)
    except Exception:
        logger.debug("session-transfer safe_upsert failed", exc_info=True)
        return None


async def safe_mark_phase(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    try:
        return await get_session_transfer_registry().mark_phase(*args, **kwargs)
    except Exception:
        logger.debug("session-transfer safe_mark_phase failed", exc_info=True)
        return None


async def safe_record_chunks(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    try:
        return await get_session_transfer_registry().record_chunks(*args, **kwargs)
    except Exception:
        logger.debug("session-transfer safe_record_chunks failed", exc_info=True)
        return None


async def safe_finalize(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    try:
        return await get_session_transfer_registry().finalize(*args, **kwargs)
    except Exception:
        logger.debug("session-transfer safe_finalize failed", exc_info=True)
        return None
