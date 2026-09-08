"""Deferred terminal work for validated passthrough responses."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional


DEFERRED_SUCCESS_HOLDER_ATTR = "_aawm_deferred_success_holder"
DeferredSuccessFinalizer = Callable[[], Awaitable[None]]
DeferredFailureFinalizer = Callable[[str], Awaitable[None]]


class DeferredPassthroughSuccess:
    """Finalize success or failure once without running the opposite callback."""

    def __init__(self) -> None:
        self._finalizer: Optional[DeferredSuccessFinalizer] = None
        self._failure_finalizer: Optional[DeferredFailureFinalizer] = None
        self._finalization_task: Optional[asyncio.Task[Any]] = None

    def set_finalizer(self, finalizer: DeferredSuccessFinalizer) -> None:
        self._finalizer = finalizer

    def set_failure_finalizer(self, finalizer: DeferredFailureFinalizer) -> None:
        self._failure_finalizer = finalizer

    async def finalize(self) -> None:
        if self._finalization_task is None:
            if self._finalizer is None:
                return
            self._finalization_task = asyncio.create_task(self._finalizer())
        await asyncio.shield(self._finalization_task)

    async def finalize_failure(self, *, phase: str = "failed") -> None:
        if self._finalization_task is None:
            if self._failure_finalizer is None:
                return
            if phase not in {"failed", "cancelled", "disconnected", "timed_out"}:
                phase = "failed"
            self._finalization_task = asyncio.create_task(
                self._failure_finalizer(phase)
            )
        await asyncio.shield(self._finalization_task)


def bind_deferred_success_holder(
    response: Any,
    holder: Optional[DeferredPassthroughSuccess],
) -> Any:
    if holder is not None:
        setattr(response, DEFERRED_SUCCESS_HOLDER_ATTR, holder)
    return response


def inherit_deferred_success_holder(
    response: Any,
    *,
    source_response: Any = None,
) -> Any:
    if getattr(response, DEFERRED_SUCCESS_HOLDER_ATTR, None) is None:
        holder = getattr(source_response, DEFERRED_SUCCESS_HOLDER_ATTR, None)
        if holder is not None:
            setattr(response, DEFERRED_SUCCESS_HOLDER_ATTR, holder)
    return response


async def finalize_deferred_success(response: Any) -> None:
    holder = getattr(response, DEFERRED_SUCCESS_HOLDER_ATTR, None)
    finalizer = getattr(holder, "finalize", None)
    if callable(finalizer):
        await finalizer()


async def finalize_deferred_failure(
    response: Any, *, phase: str = "failed"
) -> None:
    holder = getattr(response, DEFERRED_SUCCESS_HOLDER_ATTR, None)
    finalizer = getattr(holder, "finalize_failure", None)
    if callable(finalizer):
        await finalizer(phase=phase)


__all__ = [
    "DEFERRED_SUCCESS_HOLDER_ATTR",
    "DeferredPassthroughSuccess",
    "bind_deferred_success_holder",
    "finalize_deferred_failure",
    "finalize_deferred_success",
    "inherit_deferred_success_holder",
]
