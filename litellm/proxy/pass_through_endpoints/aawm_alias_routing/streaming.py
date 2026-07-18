"""Bounded streaming-response peeking for alias candidate validation (RR-054 #1/#14)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Coroutine

from fastapi.responses import StreamingResponse


@dataclass(frozen=True)
class BoundedStreamPeek:
    """Result of consuming a stream only while it remains validation-bounded."""

    response: StreamingResponse
    buffered_chunks: list[Any]
    buffered_bytes: int
    exhausted: bool


def _chunk_size(chunk: object) -> int:
    if isinstance(chunk, (bytes, bytearray)):
        return len(chunk)
    return len(str(chunk).encode("utf-8", errors="replace"))


async def peek_streaming_response(
    response: StreamingResponse,
    *,
    max_chunks: int,
    max_bytes: int,
) -> BoundedStreamPeek:
    """Buffer a small stream, or return a lossless lazy continuation on overflow."""
    iterator = response.body_iterator.__aiter__()
    buffered_chunks: list[Any] = []
    buffered_bytes = 0
    try:
        chunk = await iterator.__anext__()
    except StopAsyncIteration:
        chunk = None
    while True:
        if chunk is None:
            async def _replay_buffered() -> Any:
                for buffered in buffered_chunks:
                    yield buffered

            return BoundedStreamPeek(
                response=StreamingResponse(
                    _replay_buffered(),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                exhausted=True,
            )

        chunk_bytes = _chunk_size(chunk)
        if (
            len(buffered_chunks) >= max(0, max_chunks)
            or buffered_bytes + chunk_bytes > max(0, max_bytes)
        ):
            async def _continue_losslessly() -> Any:
                for buffered in buffered_chunks:
                    yield buffered
                yield chunk
                async for remaining in iterator:
                    yield remaining

            return BoundedStreamPeek(
                response=StreamingResponse(
                    _continue_losslessly(),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                exhausted=False,
            )
        buffered_chunks.append(chunk)
        buffered_bytes += chunk_bytes

        # create_task requires a Coroutine; AsyncIterator.__anext__ is typed as
        # Awaitable, so wrap it without changing scheduling/read semantics.
        async def _await_next_chunk() -> Any:
            return await iterator.__anext__()

        next_chunk_coro: Coroutine[Any, Any, Any] = _await_next_chunk()
        next_chunk_task: asyncio.Task[Any] = asyncio.create_task(next_chunk_coro)
        await asyncio.sleep(0)
        if not next_chunk_task.done():
            async def _continue_pending() -> Any:
                try:
                    for buffered in buffered_chunks:
                        yield buffered
                    try:
                        pending_chunk = await next_chunk_task
                    except StopAsyncIteration:
                        return
                    yield pending_chunk
                    async for remaining in iterator:
                        yield remaining
                finally:
                    if not next_chunk_task.done():
                        next_chunk_task.cancel()
                        try:
                            await next_chunk_task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass

            return BoundedStreamPeek(
                response=StreamingResponse(
                    _continue_pending(),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                exhausted=False,
            )
        try:
            chunk = next_chunk_task.result()
        except StopAsyncIteration:
            chunk = None
