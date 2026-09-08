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

from litellm._logging import verbose_proxy_logger
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.repetitive_output import (
    _STREAM_CLEANUP_ATTR,
    _CleanupBoundAsyncIterator,
    _compose_stream_cleanups,
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
_RESPONSES_PREFETCH_ABORT_ATTR = "_aawm_responses_prefetch_abort"
_PREFETCH_ABORT_REGISTER_ATTR = "_aawm_register_prefetch_continuation"


def _bind_stream_timeout_terminalizer(
    response: StreamingResponse,
    terminalizer: StreamTimeoutTerminalizer,
) -> StreamingResponse:
    setattr(response, _STREAM_TIMEOUT_TERMINALIZER_ATTR, terminalizer)
    return response


def _bind_stream_cleanup(
    response: StreamingResponse,
    cleanup: Callable[[], Awaitable[None]],
) -> StreamingResponse:
    combined = _compose_stream_cleanups(
        getattr(response, _STREAM_CLEANUP_ATTR, None),
        cleanup,
    )
    if combined is not None:
        setattr(response, _STREAM_CLEANUP_ATTR, combined)
    return response


def _guard_reconstructed_passthrough_streaming_response(
    reconstructed: StreamingResponse,
    *,
    source_response: StreamingResponse,
    request_context: Optional[OutputGuardRequestContext] = None,
) -> StreamingResponse:
    """Keep CFG-025 live-forward wrapping across peek/replay reconstructions."""
    cleanup = _compose_stream_cleanups(
        getattr(reconstructed, _STREAM_CLEANUP_ATTR, None),
        getattr(source_response, _STREAM_CLEANUP_ATTR, None),
    )

    def _inherit_cleanup(target: StreamingResponse) -> StreamingResponse:
        if cleanup is not None:
            _bind_stream_cleanup(target, cleanup)
        return target

    guarded = inherit_or_wrap_passthrough_streaming_response(
        reconstructed,
        source_response=source_response,
        request_context=request_context,
    )
    if isinstance(guarded, StreamingResponse):
        return _inherit_cleanup(guarded)
    if request_context is None:
        return _inherit_cleanup(reconstructed)
    wrapped_iter = maybe_wrap_passthrough_responses_stream(
        reconstructed.body_iterator,
        request_context=request_context,
    )
    if wrapped_iter is reconstructed.body_iterator:
        return _inherit_cleanup(reconstructed)
    return _inherit_cleanup(
        StreamingResponse(
            wrapped_iter,
            headers=dict(reconstructed.headers),
            status_code=reconstructed.status_code,
            media_type=reconstructed.media_type or "text/event-stream",
        )
    )


def _get_stream_timeout_terminalizer(
    response: StreamingResponse,
) -> Optional[StreamTimeoutTerminalizer]:
    terminalizer = getattr(response, _STREAM_TIMEOUT_TERMINALIZER_ATTR, None)
    return terminalizer if callable(terminalizer) else None


def _register_prefetch_abort_continuation(
    source_response: StreamingResponse,
    continuation_response: StreamingResponse,
    cleanup: Optional[Callable[[], Awaitable[None]]],
) -> None:
    """Attach continuation cleanup to the response-local abort owner."""
    if cleanup is None:
        return
    owner = getattr(source_response, _RESPONSES_PREFETCH_ABORT_ATTR, None)
    if not callable(owner):
        return
    register = getattr(owner, _PREFETCH_ABORT_REGISTER_ATTR, None)
    if callable(register):
        register(continuation_response, cleanup)
        return
    owner_target = getattr(owner, "_aawm_prefetch_abort_target", None)
    _bind_stream_cleanup(owner_target or source_response, cleanup)


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
    body_iterator = response.body_iterator
    iterator = body_iterator.__aiter__()
    inherited_cleanup = getattr(response, _STREAM_CLEANUP_ATTR, None)
    buffered_chunks: list[Any] = []
    buffered_bytes = 0

    def _make_continuation_cleanup(
        next_chunk_task: Optional[asyncio.Task[Any]],
    ) -> Callable[[], Awaitable[None]]:
        cleaned = False

        async def _close_continuation_resources() -> None:
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            if next_chunk_task is not None:
                if not next_chunk_task.done():
                    next_chunk_task.cancel()
                try:
                    await next_chunk_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                except BaseException:
                    verbose_proxy_logger.debug(
                        "Failed to settle peeked streaming response read task",
                        exc_info=True,
                    )
            candidates = [iterator]
            if body_iterator is not iterator:
                candidates.append(body_iterator)
            for candidate in candidates:
                close = getattr(candidate, "aclose", None)
                if not callable(close):
                    continue
                try:
                    await close()
                except BaseException:
                    verbose_proxy_logger.debug(
                        "Failed to close peeked streaming response iterator",
                        exc_info=True,
                    )
        cleanup = _compose_stream_cleanups(
            _close_continuation_resources,
            inherited_cleanup,
        )
        if cleanup is None:
            return _close_continuation_resources
        return cleanup

    async def _streaming_continuation(
        *,
        initial_chunk: Any = None,
        next_chunk_task: Optional[asyncio.Task[Any]] = None,
        terminal_exception: Optional[BaseException] = None,
        cleanup: Optional[Callable[[], Awaitable[None]]] = None,
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
            if cleanup is not None:
                await cleanup()

    active_cleanup = _make_continuation_cleanup(None)
    try:
        try:
            chunk = await iterator.__anext__()
        except StopAsyncIteration:
            chunk = None

        while True:
            if chunk is None:
                cleanup = _make_continuation_cleanup(None)
                active_cleanup = cleanup

                async def _replay_buffered() -> AsyncGenerator[Any, None]:
                    for buffered in buffered_chunks:
                        yield buffered

                replay_response = StreamingResponse(
                    _CleanupBoundAsyncIterator(
                        _replay_buffered(),
                        cleanup,
                    ),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                )
                _bind_stream_cleanup(replay_response, cleanup)
                reconstructed = _guard_reconstructed_passthrough_streaming_response(
                    replay_response,
                    source_response=response,
                )
                _register_prefetch_abort_continuation(
                    response,
                    reconstructed,
                    cleanup,
                )
                active_cleanup = None
                return BoundedStreamPeek(
                    response=reconstructed,
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
                cleanup = _make_continuation_cleanup(None)
                active_cleanup = cleanup
                continuation_response = StreamingResponse(
                    _CleanupBoundAsyncIterator(
                        _streaming_continuation(
                            initial_chunk=chunk,
                            cleanup=cleanup,
                        ),
                        cleanup,
                    ),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                )
                _bind_stream_cleanup(continuation_response, cleanup)
                if terminalizer is not None:
                    _bind_stream_timeout_terminalizer(
                        continuation_response,
                        terminalizer,
                    )
                reconstructed = _guard_reconstructed_passthrough_streaming_response(
                    continuation_response,
                    source_response=response,
                )
                _register_prefetch_abort_continuation(
                    response,
                    reconstructed,
                    cleanup,
                )
                active_cleanup = None
                return BoundedStreamPeek(
                    response=reconstructed,
                    buffered_chunks=buffered_chunks,
                    buffered_bytes=buffered_bytes,
                    stop_reason=stop_reason,
                )

            buffered_chunks.append(chunk)
            buffered_bytes += chunk_bytes

            # create_task requires a Coroutine; AsyncIterator.__anext__ is typed
            # as Awaitable, so wrap it without changing scheduling/read semantics.
            async def _await_next_chunk() -> Any:
                return await iterator.__anext__()

            next_chunk_coro: Coroutine[Any, Any, Any] = _await_next_chunk()
            next_chunk_task: asyncio.Task[Any] = asyncio.create_task(next_chunk_coro)
            active_cleanup = _make_continuation_cleanup(next_chunk_task)
            await asyncio.sleep(0)

            if not next_chunk_task.done():
                cleanup = active_cleanup
                continuation_response = StreamingResponse(
                    _CleanupBoundAsyncIterator(
                        _streaming_continuation(
                            next_chunk_task=next_chunk_task,
                            cleanup=cleanup,
                        ),
                        cleanup,
                    ),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                )
                _bind_stream_cleanup(continuation_response, cleanup)
                if terminalizer is not None:
                    _bind_stream_timeout_terminalizer(
                        continuation_response,
                        terminalizer,
                    )
                reconstructed = _guard_reconstructed_passthrough_streaming_response(
                    continuation_response,
                    source_response=response,
                )
                _register_prefetch_abort_continuation(
                    response,
                    reconstructed,
                    cleanup,
                )
                active_cleanup = None
                return BoundedStreamPeek(
                    response=reconstructed,
                    buffered_chunks=buffered_chunks,
                    buffered_bytes=buffered_bytes,
                    stop_reason="pending_stream",
                )

            try:
                chunk = next_chunk_task.result()
            except timeout_types as exc:
                if terminalizer is None:
                    raise exc

                cleanup = _make_continuation_cleanup(None)
                active_cleanup = cleanup
                continuation_response = StreamingResponse(
                    _CleanupBoundAsyncIterator(
                        _streaming_continuation(
                            terminal_exception=exc,
                            next_chunk_task=None,
                            cleanup=cleanup,
                        ),
                        cleanup,
                    ),
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type=response.media_type or "text/event-stream",
                )
                _bind_stream_cleanup(continuation_response, cleanup)
                _bind_stream_timeout_terminalizer(
                    continuation_response,
                    terminalizer,
                )
                reconstructed = _guard_reconstructed_passthrough_streaming_response(
                    continuation_response,
                    source_response=response,
                )
                _register_prefetch_abort_continuation(
                    response,
                    reconstructed,
                    cleanup,
                )
                active_cleanup = None
                return BoundedStreamPeek(
                    response=reconstructed,
                    buffered_chunks=buffered_chunks,
                    buffered_bytes=buffered_bytes,
                    stop_reason="stream_exhausted",
                )
            except StopAsyncIteration:
                chunk = None
    except BaseException:
        if active_cleanup is not None:
            await active_cleanup()
        raise
