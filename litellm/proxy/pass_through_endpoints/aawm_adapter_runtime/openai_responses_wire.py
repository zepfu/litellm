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
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Optional,
)

from starlette.responses import Response, StreamingResponse


_POLICY_FAILURE_VALUE_MAX_LENGTH = 128


def _bounded_policy_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized[:_POLICY_FAILURE_VALUE_MAX_LENGTH]


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


_KNOWN_POLICY_FAILURE_CODES = frozenset(
    {
        "aawm_repetitive_output_loop",
        "aawm_watermark_output_rejected",
    }
)


def _is_local_policy_failure_code(code: Any) -> bool:
    return str(code or "").strip() in _KNOWN_POLICY_FAILURE_CODES


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
    _terminal_body: Optional[bytes] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _done_body: Optional[bytes] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _finalization_task: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _post_finalization_callbacks: list[
        Callable[[Dict[str, Any]], Awaitable[None]]
    ] = field(
        default_factory=list,
        repr=False,
        compare=False,
    )
    _post_finalization_task: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _finalized_snapshot: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        compare=False,
    )
    _delivered_snapshot: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        compare=False,
    )
    asgi_delivery_complete: bool = False

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
        try:
            from .provider_call_ledger import publish_wire_commitment_snapshot

            publish_wire_commitment_snapshot(
                request,
                commitment=self.snapshot(),
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
            "asgi_delivery_complete": self.asgi_delivery_complete,
            "policy_failure_kind": self.metadata.get("policy_failure_kind"),
            "policy_failure_code": self.metadata.get("policy_failure_code"),
            "policy_failure_class": self.metadata.get("policy_failure_class"),
        }

    def record_policy_failure(
        self,
        *,
        kind: Any = None,
        code: Any = None,
        classification: Any = None,
    ) -> bool:
        """Freeze a bounded policy cause before delivered consumers run."""

        values = {
            "policy_failure_kind": _bounded_policy_value(kind),
            "policy_failure_code": _bounded_policy_value(code),
            "policy_failure_class": _bounded_policy_value(classification),
        }
        if not any(values.values()):
            return False
        for key, value in values.items():
            if value is not None and self.metadata.get(key) is None:
                self.metadata[key] = value
        self.publish_request_commitment()
        return True

    def record_policy_failure_from_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        response_payload = payload.get("response")
        if not isinstance(response_payload, dict):
            return False
        metadata = response_payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return False
        error = response_payload.get("error")
        if not isinstance(error, dict):
            error = payload.get("error")
        if not isinstance(error, dict):
            error = {}
        code = error.get("code") or metadata.get("error_code")
        if not _is_local_policy_failure_code(code):
            return False
        kind = metadata.get("failure_kind")
        classification = (
            metadata.get("failure_class")
            or metadata.get("policy_failure_class")
            or kind
        )
        return self.record_policy_failure(
            kind=kind,
            code=code,
            classification=classification,
        )

    def record_policy_failure_from_exception(self, exc: BaseException) -> bool:
        marker = getattr(exc, "_aawm_policy_failure", None)
        if not isinstance(marker, dict):
            detail = getattr(exc, "detail", None)
            if isinstance(detail, dict):
                marker = detail
        if not isinstance(marker, dict):
            return False
        metadata = marker.get("metadata")
        if not isinstance(metadata, dict):
            metadata = marker
        error = marker.get("error")
        if not isinstance(error, dict):
            error = {}
        marker_code = (
            error.get("code")
            or metadata.get("error_code")
            or marker.get("policy_failure_code")
        )
        if not _is_local_policy_failure_code(marker_code):
            return False
        return self.record_policy_failure(
            kind=(
                metadata.get("failure_kind")
                or metadata.get("policy_failure_kind")
                or marker.get("policy_failure_kind")
            ),
            code=marker_code,
            classification=(
                metadata.get("failure_class")
                or metadata.get("policy_failure_class")
                or marker.get("policy_failure_class")
                or metadata.get("failure_kind")
            ),
        )

    def record_response_start_delivery(self) -> None:
        """Record headers only after the ASGI send completed."""

        self.response_start_sent = True
        self.commitment = "headers"
        self.state = OpenAIResponsesWireState.HEADERS_STARTED
        self.publish_request_commitment()

    def record_body_delivery(self, body: Any) -> None:
        """Record body/terminal delivery only after the ASGI send completed."""

        if isinstance(body, memoryview):
            body = body.tobytes()
        elif isinstance(body, bytearray):
            body = bytes(body)
        if not isinstance(body, bytes):
            body = b""

        self.first_body_sent = True
        if self._terminal_body is not None and self._terminal_body in body:
            self.terminal_sent = True
            self.terminal_wire_committed = True
            self.commitment = "terminal"
            self.state = OpenAIResponsesWireState.TERMINAL_SENT
        if self._done_body is not None and self._done_body in body:
            self.done_sent = True
            self.done_wire_committed = True
            self.commitment = "done"
            self.state = OpenAIResponsesWireState.DONE_SENT
        elif self.commitment in {"none", "headers"}:
            self.commitment = "body"
            if self.state is OpenAIResponsesWireState.HEADERS_STARTED:
                self.state = OpenAIResponsesWireState.BODY_STARTED
        self.publish_request_commitment()

    def register_post_finalization_callback(
        self,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Register a consumer that runs after final disposition delivery."""

        if self._delivered_snapshot is not None:
            return
        self._post_finalization_callbacks.append(callback)

    def record_asgi_delivery_complete(self) -> None:
        """Freeze the delivered snapshot after the response attempt finishes."""

        self.asgi_delivery_complete = True
        if self.finalized:
            delivered_snapshot = dict(
                self._finalized_snapshot or self.snapshot()
            )
            delivered_snapshot["asgi_delivery_complete"] = True
            self._delivered_snapshot = delivered_snapshot
            request = self._request
            state = getattr(request, "state", None)
            if state is not None:
                try:
                    setattr(
                        state,
                        "_aawm_openai_responses_delivered_snapshot",
                        dict(delivered_snapshot),
                    )
                except Exception:
                    pass
        self.publish_request_commitment()

    async def run_post_finalization_callbacks(self) -> None:
        """Run terminal consumers against one immutable delivered snapshot."""

        if not self.finalized or not self.asgi_delivery_complete:
            return
        if self._post_finalization_task is None:
            snapshot = dict(
                self._delivered_snapshot
                or self._finalized_snapshot
                or self.snapshot()
            )
            callbacks = tuple(self._post_finalization_callbacks)

            async def _run_callbacks() -> None:
                for callback in callbacks:
                    try:
                        await callback(dict(snapshot))
                    except Exception as exc:  # noqa: BLE001
                        self.metadata["post_finalization_callback_error"] = (
                            type(exc).__name__
                        )

            self._post_finalization_task = asyncio.create_task(_run_callbacks())
        await _await_shielded(self._post_finalization_task)

    async def _finalize_disposition(
        self,
        disposition: OpenAIResponsesWireDisposition,
        callback: Optional[Callable[..., Awaitable[None]]],
    ) -> None:
        """Run one shielded disposition callback and publish terminal state."""

        if self.finalized:
            return
        finalization_task = self._finalization_task
        if finalization_task is None:
            self.finalization_started = True
            self.disposition = disposition
            self.publish_request_commitment()

            async def _run_finalization() -> None:
                try:
                    if callback is not None:
                        try:
                            await callback(disposition, self)
                        except Exception as exc:  # noqa: BLE001
                            self.metadata["disposition_callback_error"] = type(
                                exc
                            ).__name__
                finally:
                    self.finalized = True
                    self.commitment = "finalized"
                    self._finalized_snapshot = self.snapshot()
                    self.publish_request_commitment()

            finalization_task = asyncio.create_task(_run_finalization())
            self._finalization_task = finalization_task
        await _await_shielded(finalization_task)


WireDispositionCallback = Callable[
    [OpenAIResponsesWireDisposition, OpenAIResponsesWireTrace],
    Awaitable[None],
]


async def _await_shielded(awaitable: Awaitable[Any]) -> Any:
    """Complete cleanup/finalization before propagating cancellation."""

    task = asyncio.ensure_future(awaitable)
    cancellation_requested = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A caller can cancel the waiting task repeatedly. Keep the
            # cleanup task shielded until it reaches a terminal state.
            cancellation_requested = True
            continue
    if cancellation_requested:
        raise asyncio.CancelledError
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


def _parse_sse_block(
    block: bytes,
) -> tuple[Optional[str], Optional[dict], bool, bool]:
    """Parse one complete SSE block and flag invalid Responses framing.

    Plain SSE comments/fields without a data payload are valid keep-alives.
    Once a data payload or native terminal event is present, malformed JSON,
    non-object JSON, or an event/payload type disagreement is a protocol
    failure rather than an opaque chunk that can be followed by success.
    """

    try:
        decoded = block.decode("utf-8")
    except UnicodeDecodeError:
        return None, None, False, True

    lines = decoded.replace("\r\n", "\n").splitlines()
    event_line_indexes = [
        index
        for index, raw_line in enumerate(lines)
        if _line_event_type(raw_line) is not None
    ]
    # Validate every field in the complete SSE block before selecting a
    # terminal.  Looking only at the last event line can hide a malformed
    # partial event that precedes a valid terminal in the same block.
    relevant_lines = lines
    event_type = (
        _line_event_type(lines[event_line_indexes[-1]])
        if event_line_indexes
        else None
    )
    data_lines: list[str] = []
    saw_done = False
    saw_data_line = False
    for raw_line in relevant_lines:
        line = raw_line.strip()
        if line.startswith("data:"):
            saw_data_line = True
            payload_text = line.removeprefix("data:").strip()
            if payload_text == "[DONE]":
                saw_done = True
            else:
                data_lines.append(payload_text)

    payload: Optional[dict] = None
    malformed = False
    if data_lines:
        try:
            decoded_payload = json.loads("\n".join(data_lines))
        except (TypeError, json.JSONDecodeError):
            decoded_payload = None
            malformed = True
        if isinstance(decoded_payload, dict):
            payload = decoded_payload
            payload_type = decoded_payload.get("type")
            if (
                event_type is not None
                and (
                    not isinstance(payload_type, str)
                    or event_type != payload_type.strip()
                )
            ):
                malformed = True
            if event_type is None:
                if not isinstance(payload_type, str) or not payload_type.strip():
                    malformed = True
                else:
                    event_type = payload_type.strip()
        else:
            malformed = True
    elif saw_data_line and not saw_done:
        malformed = True
    elif event_type is not None:
        malformed = True

    if saw_done and event_type is not None:
        malformed = True

    if event_type in {
        "response.completed",
        "response.failed",
        "response.incomplete",
    } and _terminal_disposition(event_type, payload) is None:
        malformed = True

    return event_type, payload, saw_done, malformed


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
        self._source_iterator: Optional[AsyncIterator[bytes]] = None

    async def _close_source(self) -> None:
        if self.trace.done_wire_committed:
            extensions = getattr(self._upstream_response, "extensions", None)
            if isinstance(extensions, dict):
                extensions[
                    "aawm_openai_responses_terminal_close_expected"
                ] = True
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
        await self.trace._finalize_disposition(disposition, self._on_disposition)

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
        self.trace._terminal_body = block
        self.trace._done_body = self._DONE
        self._select_terminal(event_type=event_type, disposition=disposition)
        yield block
        yield self._DONE
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
            self._source_iterator = self._source.__aiter__()
            async for raw_chunk in self._source_iterator:
                if not raw_chunk:
                    continue
                self._buffer += bytes(raw_chunk)
                blocks, self._buffer = _split_sse_blocks(self._buffer)
                for block in blocks:
                    event_type, payload, saw_done, malformed = _parse_sse_block(block)
                    if self.trace.terminal_selected:
                        if _terminal_disposition(event_type, payload) is not None:
                            self.trace.duplicate_terminal_suppressed += 1
                        if saw_done:
                            self.trace.upstream_done_suppressed += 1
                        continue
                    if malformed:
                        self.trace.metadata["malformed_frame"] = True
                        async for emitted in self._emit_synthetic_terminal(
                            disposition=OpenAIResponsesWireDisposition.FAILED,
                            reason="malformed_sse_frame",
                        ):
                            yield emitted
                        return
                    disposition = _terminal_disposition(event_type, payload)
                    if disposition is not None:
                        self.trace.record_policy_failure_from_payload(payload)
                        if saw_done:
                            self.trace.upstream_done_suppressed += 1
                        terminal_response_payload = (
                            payload.get("response")
                            if isinstance(payload, dict)
                            else None
                        )
                        incomplete_details = (
                            terminal_response_payload.get("incomplete_details")
                            if isinstance(terminal_response_payload, dict)
                            else None
                        )
                        if (
                            disposition
                            is OpenAIResponsesWireDisposition.INCOMPLETE
                            and isinstance(incomplete_details, dict)
                            and incomplete_details.get("reason")
                            == "upstream_stream_partial_frame"
                        ):
                            self.trace.partial_frame_discarded = True
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
                        return
                    if saw_done:
                        self.trace.upstream_done_suppressed += 1
                        async for emitted in self._emit_synthetic_terminal(
                            disposition=OpenAIResponsesWireDisposition.INCOMPLETE,
                            reason="provider_done_before_terminal_event",
                        ):
                            yield emitted
                        return
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
        except Exception as exc:
            self._discard_partial_buffer()
            policy_failure_recorded = (
                self.trace.record_policy_failure_from_exception(exc)
            )
            if self.trace.terminal_selected:
                await self.finalize_transport(OpenAIResponsesWireDisposition.FAILED)
                return
            if self.trace.first_body_sent or self.trace.response_start_sent:
                async for emitted in self._emit_synthetic_terminal(
                    disposition=OpenAIResponsesWireDisposition.FAILED,
                    reason=(
                        self.trace.metadata.get("policy_failure_code")
                        or self.trace.metadata.get("policy_failure_kind")
                        if policy_failure_recorded
                        else "openai_responses_wire_source_error"
                    ),
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
                self.wire_trace.record_response_start_delivery()
            elif message_type == "http.response.body":
                self.wire_trace.record_body_delivery(message.get("body"))

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
                    except BaseException as exc:  # noqa: BLE001
                        self.wire_trace.metadata[
                            "disposition_callback_error"
                        ] = type(exc).__name__
                else:
                    try:
                        await self.wire_trace._finalize_disposition(
                            OpenAIResponsesWireDisposition.DISCONNECTED,
                            self._on_disposition,
                        )
                    except BaseException as exc:  # noqa: BLE001
                        self.wire_trace.metadata[
                            "disposition_callback_error"
                        ] = type(exc).__name__
            self.wire_trace.record_asgi_delivery_complete()
            try:
                await _await_shielded(
                    self.wire_trace.run_post_finalization_callbacks()
                )
            except BaseException as exc:  # noqa: BLE001
                self.wire_trace.metadata[
                    "post_finalization_callback_error"
                ] = type(exc).__name__
            self.wire_trace.metadata.update(self.wire_trace.snapshot())


class OpenAIResponsesBufferedResponse(Response):
    """Delay native Responses ownership finalization until body delivery."""

    def __init__(
        self,
        content: Any = None,
        *,
        wire_trace: OpenAIResponsesWireTrace,
        disposition: OpenAIResponsesWireDisposition,
        on_disposition: WireDispositionCallback,
        **kwargs: Any,
    ) -> None:
        super().__init__(content=content, **kwargs)
        self.wire_trace = wire_trace
        self._disposition = disposition
        self._on_disposition = on_disposition
        # A buffered Responses JSON body is the terminal payload itself. Bind
        # it before ASGI delivery so the post-send trace records commitment.
        if isinstance(self.body, bytes):
            self.wire_trace._terminal_body = self.body

    async def _finalize(
        self,
        disposition: OpenAIResponsesWireDisposition,
    ) -> None:
        finalizer = self.wire_trace._finalize_transport
        if finalizer is not None:
            await finalizer(disposition)
            return
        await self.wire_trace._finalize_disposition(disposition, self._on_disposition)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async def tracked_send(message: Dict[str, Any]) -> None:
            await send(message)
            message_type = message.get("type")
            if message_type == "http.response.start":
                self.wire_trace.record_response_start_delivery()
            elif message_type == "http.response.body":
                self.wire_trace.record_body_delivery(message.get("body"))

        try:
            await super().__call__(scope, receive, tracked_send)
        except asyncio.CancelledError:
            await self._finalize(OpenAIResponsesWireDisposition.CANCELLED)
            raise
        except (BrokenPipeError, ConnectionResetError):
            await self._finalize(OpenAIResponsesWireDisposition.DISCONNECTED)
            raise
        except BaseException:
            await self._finalize(OpenAIResponsesWireDisposition.DISCONNECTED)
            raise
        else:
            await self._finalize(self._disposition)
        finally:
            self.wire_trace.record_asgi_delivery_complete()
            try:
                await _await_shielded(
                    self.wire_trace.run_post_finalization_callbacks()
                )
            except BaseException as exc:  # noqa: BLE001
                self.wire_trace.metadata[
                    "post_finalization_callback_error"
                ] = type(exc).__name__
            self.wire_trace.metadata.update(self.wire_trace.snapshot())


def wrap_openai_responses_stream(
    source: AsyncIterable[bytes],
    *,
    upstream_response: Any = None,
    on_disposition: Optional[WireDispositionCallback] = None,
    trace: Optional[OpenAIResponsesWireTrace] = None,
    model: Optional[str] = None,
) -> tuple[AsyncIterator[bytes], OpenAIResponsesWireTrace]:
    """Wrap a processed stream and return its iterator plus lifecycle trace."""

    trace = trace or OpenAIResponsesWireTrace()
    coordinator = OpenAIResponsesWireCoordinator(
        source,
        upstream_response=upstream_response,
        on_disposition=on_disposition,
        trace=trace,
        model=model,
    )
    trace._finalize_transport = coordinator.finalize_transport
    return coordinator.__aiter__(), trace
