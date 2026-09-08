"""Deferred success work for alias-candidate passthrough responses."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional


DEFERRED_SUCCESS_HOLDER_ATTR = "_aawm_deferred_success_holder"
DeferredSuccessFinalizer = Callable[[], Awaitable[None]]


class DeferredPassthroughSuccess:
    """Carry success work until the candidate response is committed."""

    def __init__(self) -> None:
        self._finalizer: Optional[DeferredSuccessFinalizer] = None
        self._finalization_task: Optional[asyncio.Task[Any]] = None

    def set_finalizer(self, finalizer: DeferredSuccessFinalizer) -> None:
        self._finalizer = finalizer

    async def finalize(self) -> None:
        if self._finalization_task is None:
            if self._finalizer is None:
                return
            self._finalization_task = asyncio.create_task(self._finalizer())
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


__all__ = [
    "DEFERRED_SUCCESS_HOLDER_ATTR",
    "DeferredPassthroughSuccess",
    "bind_deferred_success_holder",
    "finalize_deferred_success",
    "inherit_deferred_success_holder",
]
