"""OpenCode Go Chat Completions → Responses incremental streaming (OC-021).

Replaces complete-upstream buffering and the alias-wide generation ceiling
with a first-chunk peek/replay adapter and separate connect, first-event,
and read-idle timeouts.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)

_GO_STREAM_EMPTY = object()

OPENCODE_GO_CONNECT_TIMEOUT_SECONDS = 30.0
OPENCODE_GO_FIRST_EVENT_TIMEOUT_SECONDS = 30.0
OPENCODE_GO_READ_IDLE_TIMEOUT_SECONDS = 600.0
OPENCODE_GO_FIRST_EVENT_PEEK_MAX_BYTES = 65536
OPENCODE_GO_FIRST_EVENT_PEEK_MAX_CHUNKS = 1


def build_opencode_go_stream_timeout() -> httpx.Timeout:
    """HTTP timeouts: connect vs idle-read. Generation wall-clock is not capped."""
    return httpx.Timeout(
        connect=OPENCODE_GO_CONNECT_TIMEOUT_SECONDS,
        read=OPENCODE_GO_READ_IDLE_TIMEOUT_SECONDS,
        write=OPENCODE_GO_CONNECT_TIMEOUT_SECONDS,
        pool=OPENCODE_GO_CONNECT_TIMEOUT_SECONDS,
    )


class OpenCodeGoUpstreamChunkAdapter:
    """Peek one Chat Completions chunk, then replay it exactly once."""

    def __init__(self, upstream: Any) -> None:
        self._upstream = upstream
        self._first_chunk: Any = _GO_STREAM_EMPTY
        self._primed = False
        self._replay_pending = False
        self._closed = False
        self.prime_count = 0
        self.close_count = 0

    @property
    def logging_obj(self) -> Any:
        return getattr(self._upstream, "logging_obj", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)

    def __aiter__(self) -> "OpenCodeGoUpstreamChunkAdapter":
        return self

    async def prime_first_chunk(self) -> Any:
        if self._primed:
            return None if self._first_chunk is _GO_STREAM_EMPTY else self._first_chunk
        self.prime_count += 1
        try:
            self._first_chunk = await self._upstream.__anext__()
            self._replay_pending = True
        except StopAsyncIteration:
            self._first_chunk = _GO_STREAM_EMPTY
            self._replay_pending = False
        self._primed = True
        return None if self._first_chunk is _GO_STREAM_EMPTY else self._first_chunk

    @property
    def exhausted_before_first_chunk(self) -> bool:
        return self._primed and self._first_chunk is _GO_STREAM_EMPTY

    async def __anext__(self) -> Any:
        if not self._primed:
            await self.prime_first_chunk()
        if self._first_chunk is _GO_STREAM_EMPTY:
            raise StopAsyncIteration
        if self._replay_pending:
            self._replay_pending = False
            return self._first_chunk
        return await self._upstream.__anext__()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_count += 1
        close = getattr(self._upstream, "aclose", None)
        if not callable(close):
            close = getattr(self._upstream, "close", None)
        if not callable(close):
            completion_stream = getattr(self._upstream, "completion_stream", None)
            close = getattr(completion_stream, "aclose", None) or getattr(
                completion_stream, "close", None
            )
        if not callable(close):
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


async def close_opencode_go_stream_resource(resource: Any) -> None:
    """Close a Go upstream stream or adapter once. Never close a shared session."""
    if resource is None:
        return
    close = getattr(resource, "aclose", None)
    if not callable(close):
        close = getattr(resource, "close", None)
    if not callable(close):
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


class OpenCodeGoChatCompletionsToResponsesAdapter(LiteLLMCompletionStreamingIterator):
    """Go-specific chunk adapter: Chat Completions deltas to Responses SSE events."""

    def __init__(
        self,
        *,
        model: str,
        litellm_custom_stream_wrapper: Any,
        request_input: Any,
        responses_api_request: Any,
        litellm_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            litellm_custom_stream_wrapper=litellm_custom_stream_wrapper,
            request_input=request_input,
            responses_api_request=responses_api_request,
            custom_llm_provider="openai",
            litellm_metadata=litellm_metadata,
        )

    async def aclose(self) -> None:
        wrapper = self.litellm_custom_stream_wrapper
        close = getattr(wrapper, "aclose", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
