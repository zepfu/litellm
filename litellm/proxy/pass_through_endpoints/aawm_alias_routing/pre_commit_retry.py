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
from typing import Any, Awaitable, Callable, Optional, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.requests import Request

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
_OPENAI_CAPACITY_RETRY_STATE_KEY = "aawm_openai_capacity_retry"
_LOCAL_CAPACITY_WAKEUP_EVENTS: dict[tuple[str, str], set[asyncio.Event]] = {}
_CLIENT_DISCONNECT_POLL_SECONDS = 0.25
_T = TypeVar("_T")


class ClientDisconnectedCancellation(asyncio.CancelledError):
    """Cancellation raised when the parsed client request disconnects."""


async def _wait_for_client_disconnect(request: Request) -> None:
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(_CLIENT_DISCONNECT_POLL_SECONDS)


async def await_with_client_disconnect(
    operation: Callable[[], Awaitable[_T]],
    *,
    request: Request,
) -> _T:
    """Run an awaitable while canceling it when the parsed request disconnects.

    ``operation`` is an unstarted factory so an already-disconnected request
    cannot begin another egress. The request body must be parsed before this
    helper is called; it only polls ``Request.is_disconnected()``. On
    disconnect, the operation is canceled and awaited before
    ``ClientDisconnectedCancellation`` is propagated. Caller cancellation
    remains a normal ``asyncio.CancelledError``. The watcher is canceled and
    awaited on every other exit path as well.
    """
    if await request.is_disconnected():
        raise ClientDisconnectedCancellation("client disconnected")

    disconnect_task = asyncio.create_task(_wait_for_client_disconnect(request))
    try:
        operation_task: asyncio.Future[Any] = asyncio.ensure_future(operation())
    except BaseException:
        disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)
        raise

    try:
        done, _ = await asyncio.wait(
            {operation_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            disconnect_task.result()
            if not operation_task.done():
                operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise ClientDisconnectedCancellation("client disconnected")
        return operation_task.result()
    finally:
        if not operation_task.done():
            operation_task.cancel()
        if not disconnect_task.done():
            disconnect_task.cancel()
        await asyncio.gather(
            operation_task,
            disconnect_task,
            return_exceptions=True,
        )


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


_LOG_SAFE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "._:/@-"
)
_LOG_VALUE_MAX_LENGTH = 160


def _sanitize_log_value(
    value: Any,
    *,
    fallback: str = "unknown",
    max_length: int = _LOG_VALUE_MAX_LENGTH,
) -> str:
    """Return a bounded, single-line log token without request data."""
    if value is None:
        return fallback
    try:
        text = str(value).strip()
    except Exception:
        return fallback
    if not text:
        return fallback
    sanitized = "".join(
        character if character in _LOG_SAFE_CHARACTERS else "_" for character in text
    )
    return sanitized[:max_length] or fallback


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
    wakeup_reason: str  # "pending" | "timer" | "peer_success" | "none"
    terminal_reason: str  # empty unless terminal
    commit_state: str  # "pre_commit" | "deadline_exhausted" | "terminal"
    phase: str = "retry"  # "pre_wait" | "retry"
    error_class: str = "unknown"
    status_code: Optional[int] = None


def _emit_capacity_retry_log(entry: CapacityRetryLogEntry) -> None:
    target_class = _sanitize_log_value(entry.target_class)
    target_hash = _sanitize_log_value(entry.target_hash)
    error_class = _sanitize_log_value(entry.error_class)
    status_code = _sanitize_log_value(entry.status_code)
    phase = _sanitize_log_value(entry.phase, fallback="retry")
    wakeup_reason = _sanitize_log_value(entry.wakeup_reason)
    terminal_reason = _sanitize_log_value(
        entry.terminal_reason,
        fallback="",
    )
    commit_state = _sanitize_log_value(
        entry.commit_state,
        fallback="pre_commit",
    )
    logger.info(
        "openai_alpha_capacity_retry "
        "phase=%s target_class=%s target_hash=%s "
        "error_class=%s status_code=%s ordinal=%d "
        "selected_delay=%.1fs wait=%.1fs "
        "elapsed=%.1fs deadline=%.1fs remaining=%.1fs "
        "wakeup=%s terminal=%s commit=%s",
        phase,
        target_class,
        target_hash,
        error_class,
        status_code,
        entry.retry_ordinal,
        entry.wait_seconds,
        entry.wait_seconds,
        entry.elapsed_seconds,
        entry.deadline_seconds,
        entry.remaining_seconds,
        wakeup_reason,
        terminal_reason,
        commit_state,
    )


def _emit_openai_capacity_terminal_log(
    *,
    target_class: str,
    target_hash: str,
    total_retries: int,
    elapsed_seconds: float,
    deadline_seconds: float,
    terminal_reason: str,
    error_class: Optional[str] = None,
    status_code: Optional[int] = None,
    wakeup_reason: str = "none",
    wait_seconds: float = 0.0,
    remaining_seconds: Optional[float] = None,
) -> None:
    if remaining_seconds is None:
        remaining_seconds = max(0.0, deadline_seconds - elapsed_seconds)
    logger.info(
        "openai_alpha_capacity_terminal "
        "phase=terminal target_class=%s target_hash=%s "
        "error_class=%s status_code=%s ordinal=%d "
        "selected_delay=%.1fs wait=%.1fs "
        "elapsed=%.1fs deadline=%.1fs remaining=%.1fs "
        "wakeup=%s terminal=%s reason=%s commit=terminal retries=%d",
        _sanitize_log_value(target_class),
        _sanitize_log_value(target_hash),
        _sanitize_log_value(error_class),
        _sanitize_log_value(status_code),
        total_retries,
        wait_seconds,
        wait_seconds,
        elapsed_seconds,
        deadline_seconds,
        remaining_seconds,
        _sanitize_log_value(wakeup_reason),
        _sanitize_log_value(terminal_reason, fallback=""),
        _sanitize_log_value(terminal_reason, fallback=""),
        total_retries,
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
    """Publish a fresh capacity-success generation for cross-worker wakeup."""
    resolved_namespace = _resolve_openai_capacity_namespace(namespace)
    scope_key = _capacity_wakeup_scope_key(target_identity, resolved_namespace)
    for event in tuple(_LOCAL_CAPACITY_WAKEUP_EVENTS.get(scope_key, ())):
        event.set()
    redis_cache = _resolve_redis_for_capacity_wakeup()
    if redis_cache is None:
        return
    key = _build_openai_capacity_success_redis_key(target_identity, resolved_namespace)
    try:
        client = redis_cache.init_async_client()
        if client is None:
            return
        # A counter can reuse an old waiter's epoch after the key expires.
        await client.set(
            key, uuid4().int, ex=_OPENAI_CAPACITY_SUCCESS_EPOCH_TTL_SECONDS
        )
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


async def _fetch_capacity_success_epoch_with_deadline(
    redis_cache: Any,
    key: str,
    *,
    deadline: float,
) -> Optional[int]:
    """Read the wakeup epoch without exceeding the remaining wait budget."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError

    fetch_task = asyncio.create_task(_fetch_capacity_success_epoch(redis_cache, key))
    try:
        return await asyncio.wait_for(fetch_task, timeout=remaining)
    finally:
        if not fetch_task.done():
            fetch_task.cancel()
        await asyncio.gather(fetch_task, return_exceptions=True)


async def _read_capacity_success_epoch_for_wait(
    redis_cache: Any,
    key: str,
    *,
    deadline: float,
) -> tuple[Optional[int], bool]:
    """Return an epoch and whether the bounded Redis read completed."""
    try:
        return (
            await _fetch_capacity_success_epoch_with_deadline(
                redis_cache,
                key,
                deadline=deadline,
            ),
            True,
        )
    except Exception:
        return None, False


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
        self._target_class = _sanitize_log_value(target_identity)
        self._budget = budget or OpenAIAlphaCapacityRetryBudget()
        self._namespace = _resolve_openai_capacity_namespace(namespace)
        self._local_event_key = _capacity_wakeup_scope_key(
            target_identity, self._namespace
        )
        self._start_monotonic = time.monotonic()
        self._retry_count = 0
        self._terminal_reason: str = ""
        self._last_error_class: Optional[str] = None
        self._last_status_code: Optional[int] = None
        self._pending_wait_seconds: Optional[float] = None
        self._redis_cache = _resolve_redis_for_capacity_wakeup()
        self._redis_key = (
            _build_openai_capacity_success_redis_key(target_identity, self._namespace)
            if self._redis_cache is not None
            else None
        )
        self._local_events = _LOCAL_CAPACITY_WAKEUP_EVENTS.setdefault(
            self._local_event_key, set()
        )

    @property
    def target_identity(self) -> str:
        return self._target_identity

    @property
    def namespace(self) -> str:
        return self._namespace

    def rebind_target(
        self,
        *,
        target_identity: str,
        namespace: Optional[str] = None,
    ) -> None:
        """Update target-scoped wakeup state without resetting request state."""
        resolved_namespace = (
            self._namespace
            if namespace is None
            else _resolve_openai_capacity_namespace(namespace)
        )
        if (
            target_identity == self._target_identity
            and resolved_namespace == self._namespace
        ):
            return

        previous_local_event_key = self._local_event_key
        previous_local_events = self._local_events
        self._local_event_key = _capacity_wakeup_scope_key(
            target_identity, resolved_namespace
        )
        if self._local_event_key != previous_local_event_key:
            if (
                not previous_local_events
                and _LOCAL_CAPACITY_WAKEUP_EVENTS.get(previous_local_event_key)
                is previous_local_events
            ):
                _LOCAL_CAPACITY_WAKEUP_EVENTS.pop(previous_local_event_key, None)
            self._local_events = _LOCAL_CAPACITY_WAKEUP_EVENTS.setdefault(
                self._local_event_key, set()
            )

        self._target_identity = target_identity
        self._target_hash = _hash_target_identity(target_identity)
        self._target_class = _sanitize_log_value(target_identity)
        self._namespace = resolved_namespace
        self._redis_cache = _resolve_redis_for_capacity_wakeup()
        self._redis_key = (
            _build_openai_capacity_success_redis_key(
                target_identity, resolved_namespace
            )
            if self._redis_cache is not None
            else None
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
        await _signal_openai_capacity_success(self._target_identity, self._namespace)

    async def sleep_with_wakeup(
        self,
        wait_seconds: float,
        *,
        error_class: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> str:
        """Sleep for *wait_seconds* with cross-worker success wakeup.

        Returns ``"timer"`` if the full interval elapsed, ``"peer_success"``
        if a same-upstream success woke this waiter early.
        """
        if wait_seconds <= 0:
            return "timer"

        deadline = min(
            time.monotonic() + wait_seconds,
            self._start_monotonic + self.deadline_seconds,
        )
        self.record_pre_wait(
            wait_seconds,
            error_class=error_class,
            status_code=status_code,
        )
        wakeup = _CapacityWakeupState()
        sleep_local_event_key = self._local_event_key
        local_events = _LOCAL_CAPACITY_WAKEUP_EVENTS.setdefault(
            sleep_local_event_key, set()
        )
        self._local_events = local_events
        local_events.add(wakeup.event)
        redis_cache = self._redis_cache
        redis_key = self._redis_key

        try:
            if redis_cache is not None and redis_key is not None:
                (
                    wakeup.starting_epoch,
                    wakeup.epoch_baseline_known,
                ) = await _read_capacity_success_epoch_for_wait(
                    redis_cache,
                    redis_key,
                    deadline=deadline,
                )

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wakeup.wakeup_reason = "timer"
                    break

                if wakeup.event.is_set():
                    wakeup.wakeup_reason = "peer_success"
                    break

                if redis_cache is not None and redis_key is not None:
                    current_epoch, _ = await _read_capacity_success_epoch_for_wait(
                        redis_cache,
                        redis_key,
                        deadline=deadline,
                    )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        wakeup.wakeup_reason = "timer"
                        break
                    if not wakeup.epoch_baseline_known:
                        if current_epoch is not None:
                            wakeup.starting_epoch = current_epoch
                            wakeup.epoch_baseline_known = True
                        current_epoch = None
                    if current_epoch is not None and (
                        wakeup.starting_epoch is None
                        or current_epoch != wakeup.starting_epoch
                    ):
                        wakeup.wakeup_reason = "peer_success"
                        break
                    if wakeup.event.is_set():
                        wakeup.wakeup_reason = "peer_success"
                        break

                poll_seconds = min(_OPENAI_CAPACITY_SUCCESS_POLL_SECONDS, remaining)
                try:
                    await asyncio.wait_for(wakeup.event.wait(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            local_events.discard(wakeup.event)
            if (
                not local_events
                and _LOCAL_CAPACITY_WAKEUP_EVENTS.get(sleep_local_event_key)
                is local_events
            ):
                _LOCAL_CAPACITY_WAKEUP_EVENTS.pop(sleep_local_event_key, None)

        return wakeup.wakeup_reason

    def _remember_log_metadata(
        self,
        *,
        error_class: Optional[str],
        status_code: Optional[int],
    ) -> tuple[str, Optional[int]]:
        if error_class is not None:
            self._last_error_class = error_class
        if status_code is not None:
            self._last_status_code = status_code
        return (
            _sanitize_log_value(self._last_error_class),
            self._last_status_code,
        )

    def record_pre_wait(
        self,
        wait_seconds: float,
        *,
        error_class: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Emit the retry record before entering the existing wait loop."""
        wait_seconds = float(wait_seconds)
        resolved_error_class, resolved_status_code = self._remember_log_metadata(
            error_class=error_class,
            status_code=status_code,
        )
        self._pending_wait_seconds = wait_seconds
        _emit_capacity_retry_log(
            CapacityRetryLogEntry(
                target_class=self._target_class,
                target_hash=self._target_hash,
                retry_ordinal=self._retry_count,
                wait_seconds=wait_seconds,
                elapsed_seconds=self.elapsed_seconds,
                deadline_seconds=self.deadline_seconds,
                remaining_seconds=self.remaining_seconds,
                wakeup_reason="pending",
                terminal_reason="",
                commit_state="pre_commit",
                phase="pre_wait",
                error_class=resolved_error_class,
                status_code=resolved_status_code,
            )
        )

    def record_retry(
        self,
        wakeup_reason: str,
        *,
        error_class: Optional[str] = None,
        status_code: Optional[int] = None,
        terminal_reason: Optional[str] = None,
    ) -> None:
        """Record a retry attempt and emit structured log."""
        resolved_error_class, resolved_status_code = self._remember_log_metadata(
            error_class=error_class,
            status_code=status_code,
        )
        ordinal = self._retry_count
        wait_seconds = (
            self._pending_wait_seconds
            if self._pending_wait_seconds is not None
            else self.next_wait_seconds()
        )
        self._pending_wait_seconds = None
        self._retry_count += 1

        _emit_capacity_retry_log(
            CapacityRetryLogEntry(
                target_class=self._target_class,
                target_hash=self._target_hash,
                retry_ordinal=ordinal,
                wait_seconds=wait_seconds,
                elapsed_seconds=self.elapsed_seconds,
                deadline_seconds=self.deadline_seconds,
                remaining_seconds=self.remaining_seconds,
                wakeup_reason=wakeup_reason,
                terminal_reason=(
                    terminal_reason if terminal_reason is not None else ""
                ),
                commit_state="pre_commit",
                phase="retry",
                error_class=resolved_error_class,
                status_code=resolved_status_code,
            )
        )

    def record_terminal(
        self,
        reason: str,
        *,
        error_class: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Record terminal exhaustion and emit structured log."""
        _, resolved_status_code = self._remember_log_metadata(
            error_class=error_class,
            status_code=status_code,
        )
        self._terminal_reason = reason
        _emit_openai_capacity_terminal_log(
            target_class=self._target_class,
            target_hash=self._target_hash,
            total_retries=self._retry_count,
            elapsed_seconds=self.elapsed_seconds,
            deadline_seconds=self.deadline_seconds,
            terminal_reason=reason,
            error_class=self._last_error_class,
            status_code=resolved_status_code,
            remaining_seconds=self.remaining_seconds,
        )

    def wakeup_event(self) -> asyncio.Event:
        """Return a local event for same-worker wakeup.

        Callers in the same worker can set this event to wake this waiter.
        """
        event = asyncio.Event()
        self._local_events.add(event)
        return event


def get_or_create_openai_alpha_capacity_retry_coordinator(
    request: Request,
    *,
    target_identity: str,
    namespace: str = "default",
    budget: Optional[OpenAIAlphaCapacityRetryBudget] = None,
) -> OpenAIAlphaCapacityRetryCoordinator:
    """Return the one capacity retry coordinator carried by *request*."""
    coordinator = getattr(
        request.state,
        _OPENAI_CAPACITY_RETRY_STATE_KEY,
        None,
    )
    if coordinator is None:
        coordinator = OpenAIAlphaCapacityRetryCoordinator(
            target_identity=target_identity,
            budget=budget,
            namespace=namespace,
        )
        setattr(request.state, _OPENAI_CAPACITY_RETRY_STATE_KEY, coordinator)
    else:
        coordinator.rebind_target(
            target_identity=target_identity,
            namespace=namespace,
        )
    return coordinator
