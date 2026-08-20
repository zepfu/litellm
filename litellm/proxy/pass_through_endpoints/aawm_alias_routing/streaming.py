"""Bounded streaming-response peeking for alias candidate validation (RR-054 #1/#14)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Coroutine, Literal, Optional

import aiohttp
import httpx
from fastapi.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.repetitive_output import (
    inherit_or_wrap_passthrough_streaming_response,
    maybe_wrap_passthrough_responses_stream,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.output_guard_config import (
    OutputGuardRequestContext,
)

StreamPeekStopReason = Literal[
    "stream_exhausted",
    "pending_stream",
    "chunk_limit",
    "byte_limit",
]


@dataclass(frozen=True)
class BoundedStreamPeek:
    """Result of consuming a stream only while it remains validation-bounded."""

    response: StreamingResponse
    buffered_chunks: list[Any]
    buffered_bytes: int
    stop_reason: StreamPeekStopReason

    @property
    def exhausted(self) -> bool:
        return self.stop_reason == "stream_exhausted"


@dataclass(frozen=True)
class StreamingTimeoutProgress:
    emitted_bytes: bool
    chunk_count: int
    total_emitted_bytes: int
    last_emission_timestamp: float | None


StreamTimeoutTerminalizer = Callable[[BaseException, StreamingTimeoutProgress], Awaitable[Any]]
_STREAM_TIMEOUT_TERMINALIZER_ATTR = "_aawm_stream_timeout_terminalizer"


def _bind_stream_timeout_terminalizer(
    response: StreamingResponse,
    terminalizer: StreamTimeoutTerminalizer,
) -> StreamingResponse:
    setattr(response, _STREAM_TIMEOUT_TERMINALIZER_ATTR, terminalizer)
    return response


def _guard_reconstructed_passthrough_streaming_response(
    reconstructed: StreamingResponse,
    *,
    source_response: StreamingResponse,
    request_context: Optional[OutputGuardRequestContext] = None,
) -> StreamingResponse:
    """Keep CFG-025 live-forward wrapping across peek/replay reconstructions."""
    guarded = inherit_or_wrap_passthrough_streaming_response(
        reconstructed,
        source_response=source_response,
        request_context=request_context,
    )
    if isinstance(guarded, StreamingResponse):
        return guarded
    if request_context is None:
        return reconstructed
    wrapped_iter = maybe_wrap_passthrough_responses_stream(
        reconstructed.body_iterator,
        request_context=request_context,
    )
    if wrapped_iter is reconstructed.body_iterator:
        return reconstructed
    return StreamingResponse(
        wrapped_iter,
        headers=dict(reconstructed.headers),
        status_code=reconstructed.status_code,
        media_type=reconstructed.media_type or "text/event-stream",
    )


def _get_stream_timeout_terminalizer(
    response: StreamingResponse,
) -> Optional[StreamTimeoutTerminalizer]:
    terminalizer = getattr(response, _STREAM_TIMEOUT_TERMINALIZER_ATTR, None)
    return terminalizer if callable(terminalizer) else None


def _chunk_size(chunk: object) -> int:
    if isinstance(chunk, (bytes, bytearray)):
        return len(chunk)
    return len(str(chunk).encode("utf-8", errors="replace"))


def _as_protocol_chunks(chunks: Any) -> AsyncGenerator[Any, None]:
    async def _iter() -> AsyncGenerator[Any, None]:
        if chunks is None:
            return

        if isinstance(chunks, (bytes, bytearray, memoryview, str)):
            yield chunks
            return

        if isinstance(chunks, AsyncIterable):
            async for chunk in chunks:
                yield chunk
            return

        for chunk in chunks:
            yield chunk

    return _iter()


async def peek_streaming_response(  # noqa: PLR0915
    response: StreamingResponse,
    *,
    max_chunks: int,
    max_bytes: int,
    terminalizer: Optional[StreamTimeoutTerminalizer] = None,
) -> BoundedStreamPeek:
    """Buffer a small stream, or return a lossless lazy continuation on overflow."""

    terminalizer = terminalizer or _get_stream_timeout_terminalizer(response)
    timeout_types = (httpx.ReadTimeout, aiohttp.client_exceptions.SocketTimeoutError)
    iterator = response.body_iterator.__aiter__()
    buffered_chunks: list[Any] = []
    buffered_bytes = 0

    async def _streaming_continuation(
        *,
        initial_chunk: Any = None,
        next_chunk_task: Optional[asyncio.Task[Any]] = None,
        terminal_exception: Optional[BaseException] = None,
    ) -> AsyncGenerator[Any, None]:
        emitted_chunks = 0
        emitted_bytes = 0
        last_emission_timestamp: float | None = None

        def _mark_emit(chunk: Any) -> None:
            nonlocal emitted_chunks, emitted_bytes, last_emission_timestamp
            emitted_chunks += 1
            emitted_bytes += _chunk_size(chunk)
            last_emission_timestamp = time.monotonic()

        async def _yield_terminalizer_chunks(
            exc: BaseException,
        ) -> AsyncGenerator[Any, None]:
            if terminalizer is None or emitted_chunks < 1:
                raise exc

            protocol_chunks = await terminalizer(
                exc,
                StreamingTimeoutProgress(
                    emitted_bytes=emitted_bytes > 0,
                    chunk_count=emitted_chunks,
                    total_emitted_bytes=emitted_bytes,
                    last_emission_timestamp=last_emission_timestamp,
                ),
            )

            async for chunk in _as_protocol_chunks(protocol_chunks):
                yield chunk

        try:
            for buffered in buffered_chunks:
                _mark_emit(buffered)
                yield buffered

            if terminal_exception is not None:
                async for terminal_chunk in _yield_terminalizer_chunks(
                    terminal_exception,
                ):
                    yield terminal_chunk
                return

            if initial_chunk is not None:
                _mark_emit(initial_chunk)
                yield initial_chunk

            if next_chunk_task is not None:
                try:
                    pending_chunk = await next_chunk_task
                except StopAsyncIteration:
                    pass
                except timeout_types as exc:
                    async for terminal_chunk in _yield_terminalizer_chunks(exc):
                        yield terminal_chunk
                    return
                else:
                    _mark_emit(pending_chunk)
                    yield pending_chunk

            async for remaining in iterator:
                _mark_emit(remaining)
                yield remaining
        except timeout_types as exc:
            async for terminal_chunk in _yield_terminalizer_chunks(exc):
                yield terminal_chunk
            return
        finally:
            if next_chunk_task is not None and not next_chunk_task.done():
                next_chunk_task.cancel()
                try:
                    await next_chunk_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass

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
                response=_guard_reconstructed_passthrough_streaming_response(
                    StreamingResponse(
                        _replay_buffered(),
                        headers=dict(response.headers),
                        status_code=response.status_code,
                        media_type=response.media_type or "text/event-stream",
                    ),
                    source_response=response,
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                stop_reason="stream_exhausted",
            )

        chunk_bytes = _chunk_size(chunk)
        stop_reason: Optional[StreamPeekStopReason] = None
        if len(buffered_chunks) >= max(0, max_chunks):
            stop_reason = "chunk_limit"
        elif buffered_bytes + chunk_bytes > max(0, max_bytes):
            stop_reason = "byte_limit"

        if stop_reason is not None:
            continuation_response = StreamingResponse(
                _streaming_continuation(initial_chunk=chunk),
                headers=dict(response.headers),
                status_code=response.status_code,
                media_type=response.media_type or "text/event-stream",
            )
            if terminalizer is not None:
                _bind_stream_timeout_terminalizer(
                    continuation_response,
                    terminalizer,
                )
            return BoundedStreamPeek(
                response=_guard_reconstructed_passthrough_streaming_response(
                    continuation_response,
                    source_response=response,
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                stop_reason=stop_reason,
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
            continuation_response = StreamingResponse(
                _streaming_continuation(next_chunk_task=next_chunk_task),
                headers=dict(response.headers),
                status_code=response.status_code,
                media_type=response.media_type or "text/event-stream",
            )
            if terminalizer is not None:
                _bind_stream_timeout_terminalizer(
                    continuation_response,
                    terminalizer,
                )
            return BoundedStreamPeek(
                response=_guard_reconstructed_passthrough_streaming_response(
                    continuation_response,
                    source_response=response,
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                stop_reason="pending_stream",
            )

        try:
            chunk = next_chunk_task.result()
        except timeout_types as exc:
            if terminalizer is None:
                raise exc

            continuation_response = StreamingResponse(
                _streaming_continuation(
                    terminal_exception=exc,
                    next_chunk_task=None,
                ),
                headers=dict(response.headers),
                status_code=response.status_code,
                media_type=response.media_type or "text/event-stream",
            )
            _bind_stream_timeout_terminalizer(
                continuation_response,
                terminalizer,
            )
            return BoundedStreamPeek(
                response=_guard_reconstructed_passthrough_streaming_response(
                    continuation_response,
                    source_response=response,
                ),
                buffered_chunks=buffered_chunks,
                buffered_bytes=buffered_bytes,
                stop_reason="stream_exhausted",
            )
        except StopAsyncIteration:
            chunk = None
