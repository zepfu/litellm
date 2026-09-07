"""Central pre-first-byte retry coordinator for OpenAI alpha capacity errors.

Owns the request-wide two-hour deadline, progressive sleep schedule, structured
Docker-log progress, and cross-worker wakeup via Redis-backed epoch plus local
``asyncio.Event`` with timer fallback when Redis is unavailable.

Scope: ``/openai_passthrough/responses`` OpenAI/Codex upstream capacity/overload
handling only.  Anthropic code, routes, adapters, fallbacks, probes, and logging
are out of scope.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

from .durable import get_aawm_alias_routing_state_namespace
from .retry import (
    OpenAIAlphaCapacityRetryBudget,
    openai_alpha_capacity_retry_wait_seconds,
    openai_alpha_capacity_retry_within_deadline,
)

logger = logging.getLogger("LiteLLMProxy")

# ---------------------------------------------------------------------------
# Target identity (credential-free)
# ---------------------------------------------------------------------------

_OPENAI_CAPACITY_SUCCESS_KEY_PREFIX = "aawm:openai_capacity_success"
_OPENAI_CAPACITY_SUCCESS_EPOCH_TTL_SECONDS = 300  # 5 min stale-success expiry
_OPENAI_CAPACITY_SUCCESS_POLL_SECONDS = 1.0
_LOCAL_CAPACITY_WAKEUP_EVENTS: dict[tuple[str, str], set[asyncio.Event]] = {}


def _resolve_openai_capacity_namespace(namespace: Optional[str]) -> str:
    if namespace is not None:
        normalized = str(namespace).strip()
        if normalized:
            return normalized
    return get_aawm_alias_routing_state_namespace()


def _normalize_selected_model(model: Optional[str], provider: str) -> str:
    normalized = " ".join(str(model or "").strip().split()).lower()
    if not normalized:
        return "unknown"

    provider_prefix = provider.strip().lower()
    for prefix in (f"{provider_prefix}/", f"{provider_prefix}:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    return normalized or "unknown"


def _normalize_upstream_host_path(upstream_url: Optional[str]) -> str:
    parsed = urlsplit(str(upstream_url or ""))
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        hostname = "unknown"

    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None and not (
        (parsed.scheme.lower() == "https" and port == 443)
        or (parsed.scheme.lower() == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"

    path = "/" + "/".join(part for part in parsed.path.split("/") if part)
    return f"{hostname}{path.lower()}"


def _capacity_wakeup_scope_key(
    target_identity: str,
    namespace: str,
) -> tuple[str, str]:
    return (namespace, target_identity)


def _build_openai_capacity_target_identity(
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    upstream_url: Optional[str] = None,
) -> str:
    """Build a credential-free target identity for capacity wakeup.

    The selected model and upstream host/path distinguish compatible targets.
    URL credentials, query parameters, and request bodies are excluded.
    """
    normalized_provider = provider.strip().lower() or "openai"
    normalized_model = _normalize_selected_model(model, normalized_provider)
    normalized_target = _normalize_upstream_host_path(upstream_url)
    return f"{normalized_provider}:{normalized_model}@{normalized_target}"


def _build_openai_capacity_success_redis_key(
    target_identity: str,
    namespace: str = "default",
) -> str:
    """Build the Redis key for the capacity-success epoch."""
    return f"{_OPENAI_CAPACITY_SUCCESS_KEY_PREFIX}:{namespace}:{target_identity}"


def _hash_target_identity(target_identity: str) -> str:
    """Short hash for log-safe target identification."""
    return hashlib.sha256(target_identity.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityRetryLogEntry:
    """Structured Docker-log record emitted for every wait and retry."""

    target_class: str
    target_hash: str
    retry_ordinal: int
    wait_seconds: float
    elapsed_seconds: float
    deadline_seconds: float
    remaining_seconds: float
    wakeup_reason: str  # "timer" | "peer_success"
    terminal_reason: str  # empty unless terminal
    commit_state: str  # "pre_commit" | "deadline_exhausted" | "terminal"


def _emit_capacity_retry_log(entry: CapacityRetryLogEntry) -> None:
    logger.info(
        "openai_alpha_capacity_retry "
        "target_class=%s target_hash=%s ordinal=%d "
        "wait=%.1fs elapsed=%.1fs deadline=%.1fs remaining=%.1fs "
        "wakeup=%s terminal=%s commit=%s",
        entry.target_class,
        entry.target_hash,
        entry.retry_ordinal,
        entry.wait_seconds,
        entry.elapsed_seconds,
        entry.deadline_seconds,
        entry.remaining_seconds,
        entry.wakeup_reason,
        entry.terminal_reason,
        entry.commit_state,
    )


def _emit_openai_capacity_terminal_log(
    *,
    target_class: str,
    target_hash: str,
    total_retries: int,
    elapsed_seconds: float,
    deadline_seconds: float,
    terminal_reason: str,
) -> None:
    logger.info(
        "openai_alpha_capacity_terminal "
        "target_class=%s target_hash=%s retries=%d "
        "elapsed=%.1fs deadline=%.1fs reason=%s",
        target_class,
        target_hash,
        total_retries,
        elapsed_seconds,
        deadline_seconds,
        terminal_reason,
    )


# ---------------------------------------------------------------------------
# Cross-worker wakeup via Redis epoch + local asyncio.Event
# ---------------------------------------------------------------------------


@dataclass
class _CapacityWakeupState:
    """Per-waiter state for cross-worker success wakeup."""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    starting_epoch: Optional[int] = None
    epoch_baseline_known: bool = True
    redis_key: Optional[str] = None
    wakeup_reason: str = "timer"


def _resolve_redis_for_capacity_wakeup() -> Optional[Any]:
    """Resolve the alias-routing Redis manager if available."""
    try:
        from litellm.proxy.aawm_alias_routing_redis import (
            aawm_alias_routing_redis_manager,
        )

        manager = aawm_alias_routing_redis_manager
        if manager is None:
            return None
        dual_cache = manager.get_dual_cache()
        if dual_cache is None:
            return None
        redis_cache = getattr(dual_cache, "redis_cache", None)
        if redis_cache is None:
            return None
        return redis_cache
    except Exception:
        return None


async def _signal_openai_capacity_success(
    target_identity: str,
    namespace: Optional[str] = None,
) -> None:
    """Increment the capacity-success epoch for cross-worker wakeup."""
    resolved_namespace = _resolve_openai_capacity_namespace(namespace)
    scope_key = _capacity_wakeup_scope_key(target_identity, resolved_namespace)
    for event in tuple(_LOCAL_CAPACITY_WAKEUP_EVENTS.get(scope_key, ())):
        event.set()
    redis_cache = _resolve_redis_for_capacity_wakeup()
    if redis_cache is None:
        return
    key = _build_openai_capacity_success_redis_key(
        target_identity, resolved_namespace
    )
    try:
        client = redis_cache.init_async_client()
        if client is None:
            return
        await client.incr(key)
        await client.expire(key, _OPENAI_CAPACITY_SUCCESS_EPOCH_TTL_SECONDS)
    except Exception:
        pass


async def _fetch_capacity_success_epoch(
    redis_cache: Any,
    key: str,
) -> Optional[int]:
    client = redis_cache.init_async_client()
    if client is None:
        raise RuntimeError("capacity wakeup Redis client unavailable")
    value = await client.get(key)
    if value is None:
        return None
    epoch = int(value)
    return epoch if epoch > 0 else None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class OpenAIAlphaCapacityRetryCoordinator:
    """Central pre-first-byte retry owner for one OpenAI alpha request.

    Owns the request-wide two-hour deadline, progressive sleep with cross-worker
    wakeup, and structured Docker-log progress.
    """

    def __init__(
        self,
        *,
        target_identity: str,
        budget: Optional[OpenAIAlphaCapacityRetryBudget] = None,
        namespace: Optional[str] = None,
    ):
        self._target_identity = target_identity
        self._target_hash = _hash_target_identity(target_identity)
        self._budget = budget or OpenAIAlphaCapacityRetryBudget()
        self._namespace = _resolve_openai_capacity_namespace(namespace)
        self._local_event_key = _capacity_wakeup_scope_key(
            target_identity, self._namespace
        )
        self._start_monotonic = time.monotonic()
        self._retry_count = 0
        self._terminal_reason: str = ""
        self._redis_cache = _resolve_redis_for_capacity_wakeup()
        self._redis_key = (
            _build_openai_capacity_success_redis_key(target_identity, namespace)
            if self._redis_cache is not None
            else None
        )
        self._local_events = _LOCAL_CAPACITY_WAKEUP_EVENTS.setdefault(
            self._local_event_key, set()
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    @property
    def deadline_seconds(self) -> float:
        return self._budget.deadline_seconds

    @property
    def budget(self) -> OpenAIAlphaCapacityRetryBudget:
        return self._budget

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - self.elapsed_seconds)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def terminal_reason(self) -> str:
        return self._terminal_reason

    def next_wait_seconds(self) -> float:
        """Return the wait for the next retry attempt."""
        return openai_alpha_capacity_retry_wait_seconds(self._retry_count)

    def within_deadline(self) -> bool:
        return openai_alpha_capacity_retry_within_deadline(
            elapsed_seconds=self.elapsed_seconds,
            next_wait_seconds=self.next_wait_seconds(),
            deadline_seconds=self.deadline_seconds,
        )

    async def signal_success(self) -> None:
        """Signal a successful connection to wake waiting requests."""
        await _signal_openai_capacity_success(
            self._target_identity, self._namespace
        )

    async def sleep_with_wakeup(self, wait_seconds: float) -> str:
        """Sleep for *wait_seconds* with cross-worker success wakeup.

        Returns ``"timer"`` if the full interval elapsed, ``"peer_success"``
        if a same-upstream success woke this waiter early.
        """
        if wait_seconds <= 0:
            return "timer"

        wakeup = _CapacityWakeupState()
        self._local_events.add(wakeup.event)
        redis_cache = self._redis_cache
        redis_key = self._redis_key

        try:
            if redis_cache is not None and redis_key is not None:
                try:
                    wakeup.starting_epoch = await _fetch_capacity_success_epoch(
                        redis_cache, redis_key
                    )
                except Exception:
                    wakeup.starting_epoch = None
                    wakeup.epoch_baseline_known = False

            deadline = time.monotonic() + wait_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wakeup.wakeup_reason = "timer"
                    break

                if wakeup.event.is_set():
                    wakeup.wakeup_reason = "peer_success"
                    break

                poll_seconds = min(_OPENAI_CAPACITY_SUCCESS_POLL_SECONDS, remaining)
                if redis_cache is not None and redis_key is not None:
                    try:
                        current_epoch = await _fetch_capacity_success_epoch(
                            redis_cache, redis_key
                        )
                    except Exception:
                        current_epoch = None
                    if not wakeup.epoch_baseline_known:
                        if current_epoch is not None:
                            wakeup.starting_epoch = current_epoch
                            wakeup.epoch_baseline_known = True
                        current_epoch = None
                    if (
                        current_epoch is not None
                        and (
                            wakeup.starting_epoch is None
                            or current_epoch != wakeup.starting_epoch
                        )
                    ):
                        wakeup.wakeup_reason = "peer_success"
                        break

                try:
                    await asyncio.wait_for(wakeup.event.wait(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._local_events.discard(wakeup.event)
            if not self._local_events:
                _LOCAL_CAPACITY_WAKEUP_EVENTS.pop(self._local_event_key, None)

        return wakeup.wakeup_reason

    def record_retry(self, wakeup_reason: str) -> None:
        """Record a retry attempt and emit structured log."""
        ordinal = self._retry_count
        wait_seconds = self.next_wait_seconds()
        self._retry_count += 1

        _emit_capacity_retry_log(
            CapacityRetryLogEntry(
                target_class=self._target_identity,
                target_hash=self._target_hash,
                retry_ordinal=ordinal,
                wait_seconds=wait_seconds,
                elapsed_seconds=self.elapsed_seconds,
                deadline_seconds=self.deadline_seconds,
                remaining_seconds=self.remaining_seconds,
                wakeup_reason=wakeup_reason,
                terminal_reason="",
                commit_state="pre_commit",
            )
        )

    def record_terminal(self, reason: str) -> None:
        """Record terminal exhaustion and emit structured log."""
        self._terminal_reason = reason
        _emit_openai_capacity_terminal_log(
            target_class=self._target_identity,
            target_hash=self._target_hash,
            total_retries=self._retry_count,
            elapsed_seconds=self.elapsed_seconds,
            deadline_seconds=self.deadline_seconds,
            terminal_reason=reason,
        )

    def wakeup_event(self) -> asyncio.Event:
        """Return a local event for same-worker wakeup.

        Callers in the same worker can set this event to wake this waiter.
        """
        event = asyncio.Event()
        self._local_events.add(event)
        return event
