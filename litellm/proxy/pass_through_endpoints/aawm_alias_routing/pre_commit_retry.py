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
_LOCAL_CAPACITY_WAKEUP_EVENTS: dict[str, set[asyncio.Event]] = {}


def _build_openai_capacity_target_identity(
    *,
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """Build a credential-free target identity for capacity wakeup.

    Uses provider + optional model prefix (first segment before '/' or '-').
    Does not include account labels, hashes, or credentials.
    """
    model_prefix = ""
    if model:
        stripped = model.strip()
        for sep in ("/", "-"):
            if sep in stripped:
                model_prefix = stripped.split(sep)[0]
                break
        if not model_prefix:
            model_prefix = stripped
    if model_prefix:
        return f"{provider}:{model_prefix}"
    return provider


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
    redis_key: Optional[str] = None
    wakeup_reason: str = "timer"


def _resolve_redis_for_capacity_wakeup() -> Optional[Any]:
    """Resolve the alias-routing Redis manager if available."""
    try:
        from litellm.proxy.aawm_alias_routing_redis import (
            _aawm_alias_routing_redis_manager,
        )

        manager = _aawm_alias_routing_redis_manager
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
    namespace: str = "default",
) -> None:
    """Increment the capacity-success epoch for cross-worker wakeup."""
    for event in tuple(_LOCAL_CAPACITY_WAKEUP_EVENTS.get(target_identity, ())):
        event.set()
    redis_cache = _resolve_redis_for_capacity_wakeup()
    if redis_cache is None:
        return
    key = _build_openai_capacity_success_redis_key(target_identity, namespace)
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
    try:
        client = redis_cache.init_async_client()
        if client is None:
            return None
        value = await client.get(key)
        if value is None:
            return 0
        return int(value)
    except Exception:
        return None


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
        namespace: str = "default",
    ):
        self._target_identity = target_identity
        self._target_hash = _hash_target_identity(target_identity)
        self._budget = budget or OpenAIAlphaCapacityRetryBudget()
        self._namespace = namespace
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
            target_identity, set()
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
                    wakeup.starting_epoch = 0

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
                    if (
                        current_epoch is not None
                        and current_epoch != wakeup.starting_epoch
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
                _LOCAL_CAPACITY_WAKEUP_EVENTS.pop(self._target_identity, None)

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
