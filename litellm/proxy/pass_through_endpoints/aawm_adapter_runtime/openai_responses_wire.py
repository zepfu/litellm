"""Final-wire coordination for native OpenAI Responses SSE streams.

The transport and policy wrappers may observe a stream before Starlette sends
it.  This module owns the last iterator boundary so terminal selection,
``[DONE]`` ordering, upstream closure, and the request-scoped owner decision
share one first-terminal-wins state.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Dict, Optional

from starlette.responses import StreamingResponse


class OpenAIResponsesWireDisposition(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    DISCONNECTED = "disconnected"


class OpenAIResponsesWireState(str, Enum):
    NEW = "new"
    HEADERS_STARTED = "headers_started"
    BODY_STARTED = "body_started"
    TERMINAL_SELECTED = "terminal_selected"
    TERMINAL_SENT = "terminal_sent"
    DONE_SENT = "done_sent"
    CLOSED = "closed"


@dataclass
class OpenAIResponsesWireTrace:
    """Bounded lifecycle trace shared by the ASGI and body iterators."""

    state: OpenAIResponsesWireState = OpenAIResponsesWireState.NEW
    response_start_sent: bool = False
    first_body_sent: bool = False
    commitment: str = "none"
    terminal_event_type: Optional[str] = None
    disposition: Optional[OpenAIResponsesWireDisposition] = None
    terminal_selected: bool = False
    terminal_sent: bool = False
    done_sent: bool = False
    terminal_wire_committed: bool = False
    done_wire_committed: bool = False
    partial_frame_discarded: bool = False
    duplicate_terminal_suppressed: int = 0
    upstream_done_suppressed: int = 0
    close_error: Optional[str] = None
    finalization_started: bool = False
    finalized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    _request: Any = field(default=None, repr=False, compare=False)
    _finalize_transport: Optional[
        Callable[[OpenAIResponsesWireDisposition], Awaitable[None]]
    ] = field(default=None, repr=False, compare=False)

    def publish_request_commitment(self) -> None:
        request = self._request
        state = getattr(request, "state", None)
        if state is None:
            return
        try:
            setattr(state, "_aawm_openai_responses_wire_trace", self)
            setattr(
                state,
                "_aawm_openai_responses_wire_commitment",
                self.snapshot(),
            )
        except Exception:
            return

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "response_start_sent": self.response_start_sent,
            "first_body_sent": self.first_body_sent,
            "commitment": self.commitment,
            "terminal_event_type": self.terminal_event_type,
            "disposition": (
                self.disposition.value if self.disposition is not None else None
            ),
            "terminal_selected": self.terminal_selected,
            "terminal_sent": self.terminal_sent,
            "done_sent": self.done_sent,
            "terminal_wire_committed": self.terminal_wire_committed,
            "done_wire_committed": self.done_wire_committed,
            "partial_frame_discarded": self.partial_frame_discarded,
            "duplicate_terminal_suppressed": self.duplicate_terminal_suppressed,
            "upstream_done_suppressed": self.upstream_done_suppressed,
            "close_error": self.close_error,
            "finalization_started": self.finalization_started,
            "finalized": self.finalized,
        }


WireDispositionCallback = Callable[
    [OpenAIResponsesWireDisposition, OpenAIResponsesWireTrace],
    Awaitable[None],
]


async def _await_shielded(awaitable: Awaitable[Any]) -> Any:
    """Complete cleanup/finalization before propagating cancellation."""

    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
    return task.result()


def bind_openai_responses_wire_trace_to_request(
    request: Any,
    trace: OpenAIResponsesWireTrace,
) -> None:
    """Expose the live wire commitment to request-scoped ledger consumers."""

    trace._request = request
    trace.publish_request_commitment()


def _line_event_type(raw_line: str) -> Optional[str]:
    line = raw_line.strip()
    if not line.startswith("event:"):
        return None
    return line.removeprefix("event:").strip() or None


def _sse_block_end(value: bytes) -> tuple[int, int]:
    lf_boundary = value.find(b"\n\n")
    crlf_boundary = value.find(b"\r\n\r\n")
    if lf_boundary < 0:
        return crlf_boundary, 4
    if crlf_boundary < 0 or lf_boundary < crlf_boundary:
        return lf_boundary, 2
    return crlf_boundary, 4


def _split_sse_blocks(value: bytes) -> tuple[list[bytes], bytes]:
    blocks: list[bytes] = []
    remainder = value
    while remainder:
        boundary, boundary_length = _sse_block_end(remainder)
        if boundary < 0:
            break
        end = boundary + boundary_length
        blocks.append(remainder[:end])
        remainder = remainder[end:]
    return blocks, remainder


def _parse_sse_block(block: bytes) -> tuple[Optional[str], Optional[dict], bool]:
    try:
        decoded = block.decode("utf-8")
    except UnicodeDecodeError:
        return None, None, False

    lines = decoded.replace("\r\n", "\n").splitlines()
    event_line_indexes = [
        index
        for index, raw_line in enumerate(lines)
        if _line_event_type(raw_line) is not None
    ]
    relevant_lines = (
        lines[event_line_indexes[-1] :]
        if event_line_indexes
        else lines
    )
    event_type = (
        _line_event_type(lines[event_line_indexes[-1]])
        if event_line_indexes
        else None
    )
    data_lines: list[str] = []
    saw_done = False
    for raw_line in relevant_lines:
        line = raw_line.strip()
        if line.startswith("data:"):
            payload_text = line.removeprefix("data:").strip()
            if payload_text == "[DONE]":
                saw_done = True
            else:
                data_lines.append(payload_text)

    payload: Optional[dict] = None
    if data_lines:
        try:
            decoded_payload = json.loads("\n".join(data_lines))
        except (TypeError, json.JSONDecodeError):
            decoded_payload = None
        if isinstance(decoded_payload, dict):
            payload = decoded_payload
            payload_type = decoded_payload.get("type")
            if (
                event_type is not None
                and isinstance(payload_type, str)
                and event_type != payload_type.strip()
            ):
                return None, None, saw_done
            if event_type is None and isinstance(payload_type, str):
                event_type = payload_type.strip() or None
    return event_type, payload, saw_done


def _terminal_block_suffix(
    block: bytes,
    *,
    event_type: Optional[str],
) -> tuple[bytes, bool]:
    """Drop an incomplete preceding SSE event when a terminal follows it."""

    if event_type is None:
        return block, False
    lines = block.splitlines(keepends=True)
    event_line_indexes = [
        index
        for index, raw_line in enumerate(lines)
        if _line_event_type(raw_line.decode("utf-8", errors="replace"))
        == event_type
    ]
    if not event_line_indexes:
        return block, False
    index = event_line_indexes[-1]
    prefix = b"".join(lines[:index])
    if not prefix.strip():
        return block, False
    return b"".join(lines[index:]), True


def _remove_done_lines(block: bytes) -> tuple[bytes, bool]:
    """Remove an inline ``[DONE]`` marker from a terminal SSE block."""

    lines = block.splitlines(keepends=True)
    filtered: list[bytes] = []
    removed = False
    for line in lines:
        if line.strip() in {b"data: [DONE]", b"data:[DONE]"}:
            removed = True
            continue
        filtered.append(line)
    return b"".join(filtered), removed


def _terminal_disposition(
    event_type: Optional[str],
    payload: Optional[dict],
) -> Optional[OpenAIResponsesWireDisposition]:
    if not isinstance(payload, dict):
        return None
    payload_type = payload.get("type")
    if (
        event_type is not None
        and isinstance(payload_type, str)
        and event_type != payload_type.strip()
    ):
        return None
    effective_event_type = event_type or (
        payload_type.strip() if isinstance(payload_type, str) else None
    )
    if effective_event_type not in {
        "response.completed",
        "response.failed",
        "response.incomplete",
    }:
        return None
    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        return None
    status = response_payload.get("status")
    if effective_event_type == "response.completed" and status == "completed":
        return OpenAIResponsesWireDisposition.COMPLETED
    if effective_event_type == "response.failed" and status == "failed":
        return OpenAIResponsesWireDisposition.FAILED
    if effective_event_type == "response.incomplete" and status == "incomplete":
        return OpenAIResponsesWireDisposition.INCOMPLETE
    return None


def _synthetic_terminal(
    *,
    disposition: OpenAIResponsesWireDisposition,
    model: Optional[str],
    reason: str,
) -> bytes:
    if disposition is OpenAIResponsesWireDisposition.FAILED:
        event_type = "response.failed"
        response_status = "failed"
    else:
        event_type = "response.incomplete"
        response_status = "incomplete"
    payload: Dict[str, Any] = {
        "type": event_type,
        "response": {
            "object": "response",
            "status": response_status,
            "model": model or "unknown",
        },
    }
    if disposition is OpenAIResponsesWireDisposition.INCOMPLETE:
        payload["response"]["incomplete_details"] = {"reason": reason}
    else:
        payload["response"]["error"] = {"code": reason, "type": reason}
    return (
        f"event: {event_type}\ndata: "
        + json.dumps(payload, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


class OpenAIResponsesWireCoordinator:
    """Normalize one native Responses body iterator at the final boundary."""

    _DONE = b"data: [DONE]\n\n"

    def __init__(
        self,
        source: AsyncIterable[bytes],
        *,
        upstream_response: Any = None,
        on_disposition: Optional[WireDispositionCallback] = None,
        trace: Optional[OpenAIResponsesWireTrace] = None,
        model: Optional[str] = None,
    ) -> None:
        self._source = source
        self._upstream_response = upstream_response
        self._on_disposition = on_disposition
        self.trace = trace or OpenAIResponsesWireTrace()
        self._model = model
        self._buffer = b""
        self._closed = False

    async def _close_source(self) -> None:
        close_errors: list[str] = []
        for value in (self._source, self._upstream_response):
            close = getattr(value, "aclose", None)
            if not callable(close):
                continue
            try:
                await close()
            except Exception as exc:  # noqa: BLE001
                close_errors.append(type(exc).__name__)
        if close_errors:
            self.trace.close_error = ",".join(close_errors)
        self._closed = True
        self.trace.state = OpenAIResponsesWireState.CLOSED
        self.trace.publish_request_commitment()

    async def finalize_transport(
        self,
        disposition: OpenAIResponsesWireDisposition,
    ) -> None:
        """Finalize one transport outcome and close both iterator layers."""

        try:
            if not self.trace.finalized:
                await self._notify(disposition)
        finally:
            if not self._closed:
                await _await_shielded(self._close_source())

    def _discard_partial_buffer(self) -> None:
        if self._buffer:
            self.trace.partial_frame_discarded = True
            self._buffer = b""

    async def _notify(
        self,
        disposition: OpenAIResponsesWireDisposition,
    ) -> None:
        if self.trace.finalized or self.trace.finalization_started:
            return
        self.trace.finalization_started = True
        self.trace.disposition = disposition
        self.trace.publish_request_commitment()
        try:
            if self._on_disposition is not None:
                try:
                    await _await_shielded(
                        self._on_disposition(disposition, self.trace)
                    )
                except Exception as exc:  # noqa: BLE001
                    # The terminal and [DONE] have already been delivered. A
                    # callback failure must not turn that committed stream into
                    # a transport error; retain only a bounded error class.
                    self.trace.metadata["disposition_callback_error"] = type(
                        exc
                    ).__name__
        finally:
            self.trace.finalized = True
            self.trace.commitment = "finalized"
            self.trace.publish_request_commitment()

    def _select_terminal(
        self,
        *,
        event_type: str,
        disposition: OpenAIResponsesWireDisposition,
    ) -> None:
        self.trace.terminal_selected = True
        self.trace.terminal_event_type = event_type
        self.trace.disposition = disposition
        self.trace.state = OpenAIResponsesWireState.TERMINAL_SELECTED
        self.trace.publish_request_commitment()

    async def _emit_terminal(
        self,
        block: bytes,
        *,
        event_type: str,
        disposition: OpenAIResponsesWireDisposition,
    ) -> AsyncIterator[bytes]:
        self._select_terminal(event_type=event_type, disposition=disposition)
        yield block
        self.trace.terminal_sent = True
        self.trace.terminal_wire_committed = True
        self.trace.state = OpenAIResponsesWireState.TERMINAL_SENT
        self.trace.commitment = "terminal"
        self.trace.publish_request_commitment()
        yield self._DONE
        self.trace.done_sent = True
        self.trace.done_wire_committed = True
        self.trace.state = OpenAIResponsesWireState.DONE_SENT
        self.trace.commitment = "done"
        self.trace.publish_request_commitment()
        await self._notify(disposition)

    async def _emit_synthetic_terminal(
        self,
        *,
        disposition: OpenAIResponsesWireDisposition,
        reason: str,
    ) -> AsyncIterator[bytes]:
        event_type = (
            "response.failed"
            if disposition is OpenAIResponsesWireDisposition.FAILED
            else "response.incomplete"
        )
        block = _synthetic_terminal(
            disposition=disposition,
            model=self._model,
            reason=reason,
        )
        async for emitted in self._emit_terminal(
            block,
            event_type=event_type,
            disposition=disposition,
        ):
            yield emitted

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for raw_chunk in self._source:
                if not raw_chunk:
                    continue
                self._buffer += bytes(raw_chunk)
                blocks, self._buffer = _split_sse_blocks(self._buffer)
                for block in blocks:
                    event_type, payload, saw_done = _parse_sse_block(block)
                    if self.trace.terminal_selected:
                        if _terminal_disposition(event_type, payload) is not None:
                            self.trace.duplicate_terminal_suppressed += 1
                        if saw_done:
                            self.trace.upstream_done_suppressed += 1
                        continue
                    disposition = _terminal_disposition(event_type, payload)
                    if disposition is not None:
                        if saw_done:
                            self.trace.upstream_done_suppressed += 1
                        terminal_block, partial_prefix_discarded = (
                            _terminal_block_suffix(
                                block,
                                event_type=event_type,
                            )
                        )
                        if partial_prefix_discarded:
                            self.trace.partial_frame_discarded = True
                        terminal_event_type = event_type or "response.incomplete"
                        response_payload = (
                            payload.get("response")
                            if isinstance(payload, dict)
                            else None
                        )
                        if (
                            disposition is OpenAIResponsesWireDisposition.INCOMPLETE
                            and (
                                terminal_event_type != "response.incomplete"
                                or not isinstance(response_payload, dict)
                            )
                        ):
                            terminal_event_type = "response.incomplete"
                            terminal_block = _synthetic_terminal(
                                disposition=disposition,
                                model=self._model,
                                reason=(
                                    "provider_terminal_payload_malformed"
                                    if not isinstance(response_payload, dict)
                                    else "provider_completed_status_not_completed"
                                ),
                            )
                        elif (
                            disposition is OpenAIResponsesWireDisposition.FAILED
                            and not isinstance(response_payload, dict)
                        ):
                            terminal_event_type = "response.failed"
                            terminal_block = _synthetic_terminal(
                                disposition=disposition,
                                model=self._model,
                                reason="provider_terminal_payload_malformed",
                            )
                        if saw_done:
                            cleaned_block, removed_done = _remove_done_lines(
                                terminal_block
                            )
                            if removed_done:
                                terminal_block = cleaned_block
                        async for emitted in self._emit_terminal(
                            terminal_block,
                            event_type=terminal_event_type,
                            disposition=disposition,
                        ):
                            yield emitted
                        continue
                    if saw_done:
                        self.trace.upstream_done_suppressed += 1
                        continue
                    yield block

            self._discard_partial_buffer()
            if not self.trace.terminal_selected:
                async for emitted in self._emit_synthetic_terminal(
                    disposition=OpenAIResponsesWireDisposition.INCOMPLETE,
                    reason="upstream_stream_ended_without_terminal_event",
                ):
                    yield emitted
            await _await_shielded(self._close_source())
        except asyncio.CancelledError:
            self._discard_partial_buffer()
            try:
                await self.finalize_transport(OpenAIResponsesWireDisposition.CANCELLED)
            finally:
                raise
        except (BrokenPipeError, ConnectionResetError):
            self._discard_partial_buffer()
            await self.finalize_transport(OpenAIResponsesWireDisposition.DISCONNECTED)
            raise
        except Exception:
            self._discard_partial_buffer()
            if self.trace.terminal_selected:
                await self.finalize_transport(OpenAIResponsesWireDisposition.FAILED)
                return
            if self.trace.first_body_sent or self.trace.response_start_sent:
                async for emitted in self._emit_synthetic_terminal(
                    disposition=OpenAIResponsesWireDisposition.FAILED,
                    reason="openai_responses_wire_source_error",
                ):
                    yield emitted
                await _await_shielded(self._close_source())
                return
            await self.finalize_transport(OpenAIResponsesWireDisposition.FAILED)
            raise
        finally:
            if not self._closed:
                await _await_shielded(self._close_source())


class OpenAIResponsesStreamingResponse(StreamingResponse):
    """Starlette response that records ASGI commitment separately from yields."""

    def __init__(
        self,
        content: AsyncIterable[bytes],
        *,
        wire_trace: OpenAIResponsesWireTrace,
        on_disposition: Optional[WireDispositionCallback] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(content, **kwargs)
        self.wire_trace = wire_trace
        self._on_disposition = on_disposition

    async def _close_body_iterator(self) -> None:
        close = getattr(self.body_iterator, "aclose", None)
        if callable(close):
            await _await_shielded(close())

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async def tracked_send(message: Dict[str, Any]) -> None:
            message_type = message.get("type")
            await send(message)
            if message_type == "http.response.start":
                self.wire_trace.response_start_sent = True
                self.wire_trace.commitment = "headers"
                self.wire_trace.state = OpenAIResponsesWireState.HEADERS_STARTED
            elif message_type == "http.response.body":
                self.wire_trace.first_body_sent = True
                if self.wire_trace.state is OpenAIResponsesWireState.HEADERS_STARTED:
                    self.wire_trace.state = OpenAIResponsesWireState.BODY_STARTED
                if self.wire_trace.commitment == "none":
                    self.wire_trace.commitment = "body"
            self.wire_trace.publish_request_commitment()

        try:
            await super().__call__(scope, receive, tracked_send)
        except BaseException as exc:
            self.wire_trace.metadata["asgi_error"] = type(exc).__name__
            try:
                await self._close_body_iterator()
            except BaseException as close_exc:  # noqa: BLE001
                self.wire_trace.metadata["body_iterator_close_error"] = type(
                    close_exc
                ).__name__
            raise
        finally:
            if not self.wire_trace.finalized:
                self.wire_trace.metadata["asgi_closed_before_terminal"] = True
                finalizer = self.wire_trace._finalize_transport
                if finalizer is not None:
                    try:
                        await _await_shielded(
                            finalizer(OpenAIResponsesWireDisposition.DISCONNECTED)
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.wire_trace.metadata[
                            "disposition_callback_error"
                        ] = type(exc).__name__
                elif self._on_disposition is not None:
                    try:
                        await _await_shielded(
                            self._on_disposition(
                                OpenAIResponsesWireDisposition.DISCONNECTED,
                                self.wire_trace,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.wire_trace.metadata[
                            "disposition_callback_error"
                        ] = type(exc).__name__
                self.wire_trace.finalized = True
                self.wire_trace.commitment = "finalized"
                self.wire_trace.publish_request_commitment()
            self.wire_trace.metadata.update(self.wire_trace.snapshot())


def wrap_openai_responses_stream(
    source: AsyncIterable[bytes],
    *,
    upstream_response: Any = None,
    on_disposition: Optional[WireDispositionCallback] = None,
    model: Optional[str] = None,
) -> tuple[AsyncIterator[bytes], OpenAIResponsesWireTrace]:
    """Wrap a processed stream and return its iterator plus lifecycle trace."""

    trace = OpenAIResponsesWireTrace()
    coordinator = OpenAIResponsesWireCoordinator(
        source,
        upstream_response=upstream_response,
        on_disposition=on_disposition,
        trace=trace,
        model=model,
    )
    trace._finalize_transport = coordinator.finalize_transport
    return coordinator.__aiter__(), trace
