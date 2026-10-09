"""Bounded cancellation joins and the native precommit egress fence."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

from fastapi import HTTPException

from litellm.integrations.aawm_session_history.waits import wait_for

logger = logging.getLogger("LiteLLMProxy")
PRECOMMIT_CLEANUP_TIMEOUT_SECONDS = 1.0
_T = TypeVar("_T")
_retained_tasks: set[asyncio.Future[Any]] = set()


@dataclass
class PrecommitFence:
    stopped: bool = False


_fence: ContextVar[Optional[PrecommitFence]] = ContextVar(
    "aawm_precommit_fence", default=None
)
_cleanup_request: ContextVar[Any] = ContextVar("aawm_cleanup_request", default=None)


class PrecommitCleanupTimeout(HTTPException):
    """Local retryable HTTP failure; never authorizes an internal replay."""

    aawm_cleanup_timeout = True

    def __init__(self) -> None:
        self.type = "server_error"
        self.message = "aawm_cleanup_timeout: precommit cancellation cleanup timed out"
        super().__init__(
            status_code=503,
            detail={
                "error": {
                    "type": self.type,
                    "code": "aawm_cleanup_timeout",
                    "message": self.message,
                }
            },
            headers={"Retry-After": "1"},
        )


def _has_pending_cancellation(task: Optional[asyncio.Future[Any]]) -> bool:
    # Task.cancelling is available from Python 3.11; the shared fence also
    # protects supported older Python runtimes without inspecting private fields.
    cancelling = getattr(task, "cancelling", None)
    return cancelling is not None and bool(cancelling())


def ensure_precommit_send_active() -> None:
    """A cancelled operation cannot send again even if cancellation was swallowed."""
    fence = _fence.get()
    task = asyncio.current_task()
    if (fence is not None and fence.stopped) or _has_pending_cancellation(task):
        raise asyncio.CancelledError


async def run_fenced_precommit(
    operation: Callable[[], Awaitable[_T]], fence: PrecommitFence
) -> _T:
    token = _fence.set(fence)
    try:
        ensure_precommit_send_active()
        return await operation()
    finally:
        _fence.reset(token)


async def run_with_precommit_cleanup_owner(
    request: Any, operation: Callable[[], Awaitable[_T]]
) -> _T:
    """Keep cleanup ownership across nested and caller-managed retry helpers."""
    token = _cleanup_request.set(request)
    try:
        return await operation()
    finally:
        _cleanup_request.reset(token)


def _observe_task(task: asyncio.Future[Any]) -> None:
    _retained_tasks.discard(task)
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            logger.debug("Precommit cleanup task settled with %s", type(error).__name__)


def retain_precommit_task(task: asyncio.Future[Any]) -> None:
    if task not in _retained_tasks:
        _retained_tasks.add(task)
        task.add_done_callback(_observe_task)


async def cancel_and_drain_precommit(
    *tasks: asyncio.Future[Any],
    request: Any = None,
    operation_task: Optional[asyncio.Future[Any]] = None,
) -> set[asyncio.Future[Any]]:
    """Join for at most one second; retain unfinished work until it settles.

    asyncio.wait bounds observation without waiting for cancellation acknowledgement.
    The lease owns unfinished provider cleanup; watchers only need observation.
    """
    active = set(tasks)
    if request is None:
        request = _cleanup_request.get()
    for task in active:
        if not task.done() and not _has_pending_cancellation(task):
            task.cancel()
        retain_precommit_task(task)
    pending = {task for task in active if not task.done()}
    try:
        if pending:
            _, pending = await wait_for(
                "cancellation_cleanup",
                asyncio.wait(pending, timeout=PRECOMMIT_CLEANUP_TIMEOUT_SECONDS),
            )
        return pending
    finally:
        if (
            operation_task is not None
            and not operation_task.done()
            and request is not None
        ):
            # Lazy import avoids the alias-routing package's initialization cycle.
            from .session_affinity import defer_session_owner_lease_until_cleanup

            defer_session_owner_lease_until_cleanup(request, operation_task)
