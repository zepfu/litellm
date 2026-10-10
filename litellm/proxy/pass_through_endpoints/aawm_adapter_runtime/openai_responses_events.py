"""Bounded, attempt-scoped parsing for OpenAI Responses SSE events.

The Responses path has several compatibility consumers, but it must have one
admission boundary. ``ResponsesEventPipeline`` owns framing, JSON decoding,
prefix replay, and bounded diagnostic retention. Consumers receive the same
``ParsedResponsesEvent`` instance; legacy byte entry points can still create a
standalone pipeline at the perimeter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional


DEFAULT_MAX_EVENT_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_RAW_EVENTS = 64
DEFAULT_MAX_RAW_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_STREAM_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_STREAM_EVENTS = 100_000
DEFAULT_MAX_HELD_EVENTS = 256
DEFAULT_MAX_HELD_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_OUTPUT_ITEMS = 128
DEFAULT_MAX_EVENT_TYPES = 64
DEFAULT_MAX_CONTROL_EVENTS = 8
DEFAULT_MAX_CONTROL_BYTES = 256 * 1024
DEFAULT_MAX_DELTA_KEYS = 256

EVENT_BYTES_ENV = "AAWM_RESPONSES_MAX_EVENT_BYTES"
RAW_EVENTS_ENV = "AAWM_RESPONSES_MAX_RAW_EVENTS"
RAW_BYTES_ENV = "AAWM_RESPONSES_MAX_RAW_BYTES"
STREAM_BYTES_ENV = "AAWM_RESPONSES_MAX_STREAM_BYTES"
STREAM_EVENTS_ENV = "AAWM_RESPONSES_MAX_STREAM_EVENTS"
HELD_EVENTS_ENV = "AAWM_RESPONSES_MAX_HELD_EVENTS"
HELD_BYTES_ENV = "AAWM_RESPONSES_MAX_HELD_BYTES"
OUTPUT_BYTES_ENV = "AAWM_RESPONSES_MAX_OUTPUT_BYTES"
OUTPUT_ITEMS_ENV = "AAWM_RESPONSES_MAX_OUTPUT_ITEMS"
PARSED_RESPONSES_EVENTS_ATTR = "aawm_parsed_responses_events"
RESPONSES_EVENTS_ADMITTED_ATTR = "aawm_responses_events_admitted"
RESPONSES_WIRE_EMISSIONS_ATTR = "aawm_responses_wire_emissions"
RESPONSES_EVENT_PIPELINE_ATTR = "aawm_responses_event_pipeline"


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """Read an integer limit, rejecting values below the supported minimum."""
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value.strip())
    except ValueError:
        return default
    return value if value >= minimum else default


def resolve_responses_event_limits() -> tuple[int, int]:
    """Resolve per-event and diagnostic raw-retention bounds."""
    return (
        _env_int(EVENT_BYTES_ENV, DEFAULT_MAX_EVENT_BYTES, minimum=1),
        _env_int(RAW_EVENTS_ENV, DEFAULT_MAX_RAW_EVENTS, minimum=0),
    )


def resolve_responses_raw_retention_limits() -> tuple[int, int]:
    """Resolve diagnostic raw-event count and byte bounds."""
    return (
        _env_int(RAW_EVENTS_ENV, DEFAULT_MAX_RAW_EVENTS, minimum=0),
        _env_int(RAW_BYTES_ENV, DEFAULT_MAX_RAW_BYTES, minimum=0),
    )


def resolve_responses_stream_limits() -> tuple[int, int]:
    """Resolve aggregate stream and event-count bounds."""
    return (
        _env_int(STREAM_BYTES_ENV, DEFAULT_MAX_STREAM_BYTES, minimum=1),
        _env_int(STREAM_EVENTS_ENV, DEFAULT_MAX_STREAM_EVENTS, minimum=1),
    )


def resolve_responses_retention_limits() -> tuple[int, int, int, int]:
    """Resolve held-event and event-driven output retention bounds."""
    return (
        _env_int(HELD_EVENTS_ENV, DEFAULT_MAX_HELD_EVENTS, minimum=0),
        _env_int(HELD_BYTES_ENV, DEFAULT_MAX_HELD_BYTES, minimum=0),
        _env_int(OUTPUT_BYTES_ENV, DEFAULT_MAX_OUTPUT_BYTES, minimum=1),
        _env_int(OUTPUT_ITEMS_ENV, DEFAULT_MAX_OUTPUT_ITEMS, minimum=1),
    )


def _coerce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _line_ending_at(buffer: bytes, index: int) -> Optional[bytes]:
    """Return the one line ending starting at ``index``, treating CRLF as atomic."""
    if index < 0 or index >= len(buffer):
        return None
    if buffer.startswith(b"\r\n", index):
        return b"\r\n"
    if buffer[index : index + 1] in {b"\n", b"\r"}:
        return buffer[index : index + 1]
    return None


def _line_ending_ending_at(buffer: bytes) -> Optional[bytes]:
    """Return the one line ending that terminates ``buffer``, if present."""
    if buffer.endswith(b"\r\n"):
        return b"\r\n"
    if buffer.endswith(b"\n") or buffer.endswith(b"\r"):
        return buffer[-1:]
    return None


def _scan_event_separator(
    buffer: bytes,
    *,
    start: int = 0,
    pending_line_ending: bytes = b"",
) -> Optional[tuple[int, int, bytes]]:
    """Find the first two consecutive line endings without splitting CRLF.

    A pending CR from a prior read is completed as CRLF when the next byte is
    LF. That completed CRLF is one line ending, not a blank-line separator.
    """
    index = max(0, int(start))
    first_ending = pending_line_ending if pending_line_ending in {b"\r", b"\n"} else b""
    if first_ending == b"\r" and buffer.startswith(b"\n"):
        first_ending = b"\r\n"
        index = 1
    while index < len(buffer):
        ending = _line_ending_at(buffer, index)
        if ending is None:
            first_ending = b""
            index += 1
            continue
        if not first_ending:
            first_ending = ending
            index += len(ending)
            continue
        complete_end = index + len(ending)
        return complete_end, complete_end - (index - len(first_ending)), first_ending + ending
    return None


def _separator_length_at_end(buffer: bytes) -> int:
    """Return the exact two-line-ending separator ending ``buffer``."""
    if not buffer:
        return 0
    second = _line_ending_ending_at(buffer)
    if second is None:
        return 0
    first = _line_ending_ending_at(buffer[: -len(second)])
    if first is None:
        return 0
    return len(first) + len(second)


def _find_event_boundary(buffer: bytes) -> Optional[tuple[int, int]]:
    """Find the first blank SSE line without treating CRLF as two endings."""
    match = _scan_event_separator(buffer)
    if match is None:
        return None
    complete_end, separator_length, separator = match
    # A trailing CR may still form one CRLF with the next read. Wait rather
    # than treating that CR as a completed second line ending.
    if (
        complete_end == len(buffer)
        and separator.endswith(b"\r")
        and not separator.endswith(b"\r\n")
    ):
        return None
    return complete_end, separator_length


def _json_size(value: Any) -> Optional[int]:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return None


def bounded_identity_key(value: Any, *, limit: int = 1024) -> Optional[str]:
    """Return a stored identity that still distinguishes over-limit strings."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    encoded = normalized.encode("utf-8")
    if len(encoded) <= limit:
        return normalized
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _line_ending_from_bytes(line: bytes) -> tuple[bytes, bytes]:
    if line.endswith(b"\r\n"):
        return line[:-2], b"\r\n"
    if line.endswith(b"\n"):
        return line[:-1], b"\n"
    if line.endswith(b"\r"):
        return line[:-1], b"\r"
    return line, b""


def render_responses_event_wire(
    event: "ParsedResponsesEvent",
    payload: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Serialize a payload with the source event's retained framing.

    Unchanged frames keep the original bytes. Transformed payloads replace the
    concatenated ``data:`` body and preserve comments, event type, and the
    original line-ending / blank-line separator.
    """
    wire = event.forward_event or event.raw_event or b""
    if payload is None:
        payload = event.payload
    if not wire or payload is None:
        return wire
    if payload is event.payload:
        return wire

    lines = wire.splitlines(keepends=True)
    data_indexes: list[int] = []
    for index, line in enumerate(lines):
        content, _ending = _line_ending_from_bytes(line)
        if content.lstrip().startswith(b"data:"):
            data_indexes.append(index)
    if not data_indexes:
        return wire

    first_data_index = data_indexes[0]
    first_content, line_ending = _line_ending_from_bytes(lines[first_data_index])
    data_offset = first_content.find(b"data:")
    if data_offset < 0:
        return wire
    prefix = first_content[: data_offset + len(b"data:")]
    remainder = first_content[data_offset + len(b"data:") :]
    whitespace_length = len(remainder) - len(remainder.lstrip(b" "))
    prefix += remainder[:whitespace_length]
    rendered = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    data_indexes_set = set(data_indexes)
    updated: list[bytes] = []
    for index, line in enumerate(lines):
        if index not in data_indexes_set:
            updated.append(line)
        elif index == first_data_index:
            updated.append(prefix + rendered + line_ending)
    serialized = b"".join(updated)
    separator = event.separator
    if not separator:
        separator_length = _separator_length_at_end(wire)
        separator = wire[-separator_length:] if separator_length else b""
    if separator and not serialized.endswith(separator):
        serialized += separator
    return serialized


class ResponsesEventChunk(bytes):
    """Bytes-compatible chunk carrying one ordered association per frame."""

    def __new__(
        cls,
        value: bytes,
        events: Iterable["ParsedResponsesEvent"] = (),
        emissions: Iterable["ResponsesWireEmission"] = (),
    ) -> "ResponsesEventChunk":
        instance = bytes.__new__(cls, value)
        emission_list = list(emissions)
        event_list = [
            emission.source_event if isinstance(emission, ResponsesWireEmission) else emission
            for emission in (emission_list or list(events))
        ]
        setattr(instance, PARSED_RESPONSES_EVENTS_ATTR, event_list)
        setattr(instance, RESPONSES_WIRE_EMISSIONS_ATTR, emission_list)
        setattr(instance, RESPONSES_EVENTS_ADMITTED_ATTR, True)
        return instance

    @property
    def parsed_responses_events(self) -> list["ParsedResponsesEvent"]:
        value = getattr(self, PARSED_RESPONSES_EVENTS_ATTR, None)
        return value if isinstance(value, list) else []

    @property
    def responses_wire_emissions(self) -> list["ResponsesWireEmission"]:
        value = getattr(self, RESPONSES_WIRE_EMISSIONS_ATTR, None)
        return value if isinstance(value, list) else []


def _unique_source_events(
    events: Iterable["ParsedResponsesEvent"],
) -> list["ParsedResponsesEvent"]:
    unique: list[ParsedResponsesEvent] = []
    seen: set[int] = set()
    for event in events:
        if not isinstance(event, ParsedResponsesEvent):
            continue
        marker = id(event)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(event)
    return unique


def _emissions_from_events(
    events: Iterable["ParsedResponsesEvent"],
) -> list["ResponsesWireEmission"]:
    return [
        event.emission()
        for event in _unique_source_events(events)
    ]


def _normalize_wire_emissions(
    events: Iterable[Any] = (),
    emissions: Iterable["ResponsesWireEmission"] = (),
) -> list["ResponsesWireEmission"]:
    normalized: list[ResponsesWireEmission] = []
    seen: set[int] = set()
    for item in (*emissions, *events):
        if isinstance(item, ResponsesWireEmission):
            emission = item
        elif isinstance(item, ParsedResponsesEvent):
            emission = item.emission()
        else:
            continue
        marker = id(emission.source_event)
        if marker in seen:
            continue
        seen.add(marker)
        normalized.append(emission)
    return normalized


def attach_responses_events(
    chunk: bytes,
    events: Iterable["ParsedResponsesEvent"] = (),
    *,
    emissions: Iterable["ResponsesWireEmission"] = (),
    replace: bool = False,
) -> bytes:
    """Attach one ordered association per emitted frame without changing bytes."""
    existing_emissions = getattr(chunk, RESPONSES_WIRE_EMISSIONS_ATTR, None)
    existing_events = getattr(chunk, PARSED_RESPONSES_EVENTS_ATTR, None)
    if replace:
        emission_list = _normalize_wire_emissions(events, emissions)
    else:
        prior_events = existing_events if isinstance(existing_events, list) else []
        prior_emissions = (
            existing_emissions if isinstance(existing_emissions, list) else ()
        )
        emission_list = _normalize_wire_emissions(
            events,
            (*prior_emissions, *emissions),
        )
        if not emission_list and prior_events:
            emission_list = _normalize_wire_emissions(prior_events)
    if (
        getattr(chunk, RESPONSES_EVENTS_ADMITTED_ATTR, False) is True
        and isinstance(existing_events, list)
        and not emission_list
        and not events
        and not emissions
    ):
        return chunk
    return ResponsesEventChunk(
        bytes(chunk),
        [emission.source_event for emission in emission_list],
        emission_list,
    )


def attach_responses_emissions(
    chunk: bytes,
    emissions: Iterable["ResponsesWireEmission"],
) -> bytes:
    """Replace the carrier with the current effective emissions."""
    emission_list = _normalize_wire_emissions(emissions=emissions)
    wire = b"".join(emission.wire_bytes for emission in emission_list)
    return attach_responses_events(wire, emissions=emission_list, replace=True)


def responses_events_from_chunk(
    chunk: Any,
) -> Optional[list["ParsedResponsesEvent"]]:
    """Return the typed events associated with an admitted byte carrier."""
    if getattr(chunk, RESPONSES_EVENTS_ADMITTED_ATTR, False) is not True:
        return None
    events = getattr(chunk, PARSED_RESPONSES_EVENTS_ATTR, None)
    return list(events) if isinstance(events, list) else []


def responses_emissions_from_chunk(
    chunk: Any,
) -> Optional[list["ResponsesWireEmission"]]:
    """Return the current effective emissions associated with a carrier."""
    if getattr(chunk, RESPONSES_EVENTS_ADMITTED_ATTR, False) is not True:
        return None
    emissions = getattr(chunk, RESPONSES_WIRE_EMISSIONS_ATTR, None)
    if isinstance(emissions, list) and emissions:
        return list(emissions)
    events = responses_events_from_chunk(chunk)
    if events is None:
        return None
    return _emissions_from_events(events)


def responses_event_pipeline_from_response(
    response: Any,
) -> Optional["ResponsesEventPipeline"]:
    """Return the attempt-scoped pipeline attached to a response wrapper."""
    pipeline = getattr(response, RESPONSES_EVENT_PIPELINE_ATTR, None)
    return pipeline if isinstance(pipeline, ResponsesEventPipeline) else None


@dataclass(frozen=True)
class ParsedResponsesEvent:
    """One complete Responses SSE event with its decoded payload."""

    sequence_number: int
    event_type: Optional[str]
    payload: Optional[Dict[str, Any]]
    is_done_marker: bool
    is_malformed: bool
    byte_count: int
    observed_at: datetime
    forward_event: Optional[bytes] = None
    raw_event: Optional[bytes] = None
    separator: bytes = b""

    @property
    def type(self) -> str:
        return self.event_type or ""

    @property
    def response_payload(self) -> Optional[Dict[str, Any]]:
        response = self.payload.get("response") if self.payload else None
        return response if isinstance(response, dict) else None

    @property
    def usage_payload(self) -> Optional[Dict[str, Any]]:
        if not self.payload:
            return None
        direct_usage = self.payload.get("usage")
        if isinstance(direct_usage, dict):
            return direct_usage
        response = self.response_payload
        if not response:
            return None
        usage = response.get("usage")
        return usage if isinstance(usage, dict) else None

    def field(self, name: str, default: Any = None) -> Any:
        """Read a top-level event field without exposing parser fallbacks."""
        if self.payload is None:
            return default
        return self.payload.get(name, default)

    def emission(
        self,
        *,
        payload: Optional[Dict[str, Any]] = None,
        wire_bytes: Optional[bytes] = None,
    ) -> "ResponsesWireEmission":
        """Return the original frame or an owned transformed projection."""
        effective_payload = self.payload if payload is None else payload
        if wire_bytes is None:
            wire_bytes = render_responses_event_wire(self, effective_payload)
        return ResponsesWireEmission(
            source_event=self,
            effective_payload=effective_payload,
            wire_bytes=wire_bytes,
        )


class ResponsesEventTooLarge(ValueError):
    """An SSE event exceeded the configured event bound."""


class ResponsesStreamTooLarge(ValueError):
    """A Responses stream exceeded an aggregate configured bound."""


class ResponsesEventRetentionTooLarge(ResponsesStreamTooLarge):
    """Required typed-event retention exceeded its configured bound."""


class ResponsesBoundFailure(Exception):
    """Typed admission failure with the accepted prefix and an error payload."""

    def __init__(
        self,
        cause: BaseException,
        *,
        accepted_events: Optional[Iterable["ParsedResponsesEvent"]] = None,
        accepted_bytes: int = 0,
    ) -> None:
        self.cause = cause
        self.accepted_events = list(accepted_events or ())
        self.accepted_bytes = max(0, int(accepted_bytes))
        if isinstance(cause, ResponsesEventTooLarge):
            self.kind = "event_too_large"
            self.code = "openai_responses_stream_event_too_large"
            self.incomplete_reason = "upstream_stream_event_too_large"
        elif isinstance(cause, ResponsesEventRetentionTooLarge):
            self.kind = "retention_too_large"
            self.code = "openai_responses_stream_retention_too_large"
            self.incomplete_reason = "upstream_stream_retention_too_large"
        else:
            self.kind = "stream_too_large"
            self.code = "openai_responses_stream_too_large"
            self.incomplete_reason = "upstream_stream_too_large"
        self.error_payload = {
            "code": self.code,
            "type": "invalid_response",
            "message": str(cause),
        }
        super().__init__(str(cause))

    @classmethod
    def from_exception(cls, exc: BaseException) -> "ResponsesBoundFailure":
        if isinstance(exc, ResponsesBoundFailure):
            return exc
        return cls(
            exc,
            accepted_events=getattr(exc, "accepted_events", ()),
            accepted_bytes=int(getattr(exc, "accepted_bytes", 0) or 0),
        )


@dataclass(frozen=True)
class ResponsesWireEmission:
    """One output frame referring to the shared source event."""

    source_event: "ParsedResponsesEvent"
    effective_payload: Optional[Dict[str, Any]]
    wire_bytes: bytes

    def transformed(
        self,
        payload: Optional[Dict[str, Any]] = None,
        *,
        wire_bytes: Optional[bytes] = None,
    ) -> "ResponsesWireEmission":
        """Keep the original source event and replace the current effective wire."""
        effective_payload = (
            self.effective_payload if payload is None else payload
        )
        if wire_bytes is None:
            wire_bytes = render_responses_event_wire(
                self.source_event,
                effective_payload,
            )
        return ResponsesWireEmission(
            source_event=self.source_event,
            effective_payload=effective_payload,
            wire_bytes=wire_bytes,
        )


class ResponsesEventPipeline:
    """The single bounded admission and replay owner for one stream attempt."""

    def __init__(
        self,
        *,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_raw_events: int = DEFAULT_MAX_RAW_EVENTS,
        max_raw_bytes: int = DEFAULT_MAX_RAW_BYTES,
        max_stream_bytes: int = DEFAULT_MAX_STREAM_BYTES,
        max_stream_events: int = DEFAULT_MAX_STREAM_EVENTS,
        max_held_events: int = DEFAULT_MAX_HELD_EVENTS,
        max_held_bytes: int = DEFAULT_MAX_HELD_BYTES,
        max_control_events: int = DEFAULT_MAX_CONTROL_EVENTS,
        max_control_bytes: int = DEFAULT_MAX_CONTROL_BYTES,
        retain_raw_events: bool = False,
        retain_events: bool = False,
        summary_reducer: Optional["ResponsesEventSummaryReducer"] = None,
    ) -> None:
        if max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be positive")
        if max_raw_events < 0:
            raise ValueError("max_raw_events must not be negative")
        if max_raw_bytes < 0:
            raise ValueError("max_raw_bytes must not be negative")
        if max_stream_bytes <= 0:
            raise ValueError("max_stream_bytes must be positive")
        if max_stream_events <= 0:
            raise ValueError("max_stream_events must be positive")
        if max_held_events < 0:
            raise ValueError("max_held_events must not be negative")
        if max_held_bytes < 0:
            raise ValueError("max_held_bytes must not be negative")
        if max_control_events < 0:
            raise ValueError("max_control_events must not be negative")
        if max_control_bytes < 0:
            raise ValueError("max_control_bytes must not be negative")
        self._assign_limits(
            max_event_bytes=max_event_bytes,
            max_raw_events=max_raw_events,
            max_raw_bytes=max_raw_bytes,
            max_stream_bytes=max_stream_bytes,
            max_stream_events=max_stream_events,
            max_held_events=max_held_events,
            max_held_bytes=max_held_bytes,
            max_control_events=max_control_events,
            max_control_bytes=max_control_bytes,
            retain_raw_events=retain_raw_events,
            retain_events=retain_events,
            summary_reducer=summary_reducer,
        )

        self._buffer = bytearray()
        self._pending_cr = False
        self._next_sequence_number = 1
        self._raw_events: Deque[bytes] = deque()
        self._raw_event_bytes = 0
        self._held_events: Deque[ParsedResponsesEvent] = deque()
        self._held_event_bytes = 0
        self._replay_remaining_bytes = 0
        self._replay_consumed_bytes = 0
        self._replay_consumed_event_bytes = 0
        self._replay_buffer: Optional[bytes] = None
        self._last_feed_admitted_bytes = 0
        self._observers: list[Callable[[ParsedResponsesEvent], None]] = []

        self.event_count = 0
        self.malformed_event_count = 0
        self.done_marker_count = 0
        self.dropped_raw_event_count = 0
        self.dropped_raw_event_bytes = 0
        self.omitted_held_event_count = 0
        self.control_event_count = 0
        self.control_event_bytes = 0
        self.total_stream_bytes = 0
        self.total_event_count = 0
        self.last_event_observed_at: Optional[datetime] = None
        if summary_reducer is not None:
            self.add_observer(summary_reducer.consume)

    def _assign_limits(
        self,
        *,
        max_event_bytes: int,
        max_raw_events: int,
        max_raw_bytes: int,
        max_stream_bytes: int,
        max_stream_events: int,
        max_held_events: int,
        max_held_bytes: int,
        max_control_events: int,
        max_control_bytes: int,
        retain_raw_events: bool,
        retain_events: bool,
        summary_reducer: Optional["ResponsesEventSummaryReducer"],
    ) -> None:
        self.max_event_bytes = int(max_event_bytes)
        self.max_raw_events = int(max_raw_events)
        self.max_raw_bytes = int(max_raw_bytes)
        self.max_stream_bytes = int(max_stream_bytes)
        self.max_stream_events = int(max_stream_events)
        self.max_held_events = int(max_held_events)
        self.max_held_bytes = int(max_held_bytes)
        self.max_control_events = int(max_control_events)
        self.max_control_bytes = int(max_control_bytes)
        self.retain_raw_events = bool(retain_raw_events)
        self.retain_events = bool(retain_events)
        self.summary_reducer = summary_reducer

    @property
    def buffered_event_bytes(self) -> int:
        return len(self._buffer)

    @property
    def held_event_count(self) -> int:
        return len(self._held_events)

    @property
    def held_event_bytes(self) -> int:
        return self._held_event_bytes

    @property
    def replay_remaining_bytes(self) -> int:
        return self._replay_remaining_bytes

    @property
    def last_feed_admitted_bytes(self) -> int:
        return self._last_feed_admitted_bytes

    @property
    def raw_event_bytes(self) -> int:
        return self._raw_event_bytes

    def add_observer(
        self,
        observer: Callable[[ParsedResponsesEvent], None],
    ) -> None:
        """Register a bounded number of post-admission typed observers."""
        if not callable(observer):
            raise TypeError("observer must be callable")
        if len(self._observers) >= 8:
            raise ResponsesEventRetentionTooLarge(
                "Responses event observer limit exceeded"
            )
        self._observers.append(observer)

    @staticmethod
    def _event_boundary(buffer: bytes) -> Optional[tuple[int, int]]:
        return _find_event_boundary(buffer)

    def _check_stream_admission(self, chunk_length: int) -> None:
        if self.total_stream_bytes + chunk_length > self.max_stream_bytes:
            raise ResponsesStreamTooLarge(
                f"Responses SSE stream exceeded {self.max_stream_bytes} bytes"
            )

    def _retain_raw(self, raw_event: bytes) -> Optional[bytes]:
        if not self.retain_raw_events:
            return None
        if self.max_raw_events == 0 or self.max_raw_bytes == 0:
            self.dropped_raw_event_count += 1
            self.dropped_raw_event_bytes += len(raw_event)
            return None

        self._raw_events.append(raw_event)
        self._raw_event_bytes += len(raw_event)
        while (
            len(self._raw_events) > self.max_raw_events
            or self._raw_event_bytes > self.max_raw_bytes
        ):
            dropped = self._raw_events.popleft()
            self._raw_event_bytes -= len(dropped)
            self.dropped_raw_event_count += 1
            self.dropped_raw_event_bytes += len(dropped)
        return raw_event

    def _held_representation_bytes(self, event: ParsedResponsesEvent) -> int:
        """Charge the retained decoded payload and carrier, not just raw bytes."""
        total = 0
        if event.forward_event:
            total += len(event.forward_event)
        elif event.raw_event:
            total += len(event.raw_event)
        else:
            total += max(0, int(event.byte_count))
        payload_size = _json_size(event.payload) if event.payload is not None else 0
        if payload_size:
            total += payload_size
        return total

    def _hold_event(
        self,
        event: ParsedResponsesEvent | List[ParsedResponsesEvent],
        *,
        operator: Optional[ResponsesBoundFailure] = None,
    ) -> None:
        if not self.retain_events:
            return
        # The first upstream read can coalesce many events. Treat it as one
        # transaction so a terminal in that batch is discoverable even when
        # it contains more events than the individual hold capacity.
        batch = [event] if isinstance(event, ParsedResponsesEvent) else list(event)
        batch_bytes = sum(self._held_representation_bytes(item) for item in batch)
        projected_count = len(self._held_events) + len(batch)
        projected_bytes = self._held_event_bytes + batch_bytes
        if (
            self.max_held_events == 0
            or self.max_held_bytes == 0
            or projected_count > self.max_held_events
            or projected_bytes > self.max_held_bytes
        ):
            if operator is not None:
                operator.accepted_events.extend(self._held_events)
                # The coalesced prefix is valid and already parsed. Carry it
                # for a later bound terminal, then stop holding the live
                # remainder so ordinary streams can pass the hold capacity.
                operator.accepted_events.extend(batch)
                self._held_events.clear()
                self._held_event_bytes = 0
                self.retain_events = False
                return
            else:
                self.omitted_held_event_count += len(batch)
                raise ResponsesEventRetentionTooLarge(
                    "Responses typed-event replay retention exceeded configured bound"
                )
        else:
            self._held_events.extend(batch)
            self._held_event_bytes = projected_bytes

    def _emit(self, event: ParsedResponsesEvent) -> None:
        for observer in tuple(self._observers):
            observer(event)

    def _parse_event_lines(
        self,
        raw_event: bytes,
        *,
        malformed_tail: bool,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]], bool, bool, bytes]:
        event_type: Optional[str] = None
        payload: Optional[Dict[str, Any]] = None
        is_done_marker = False
        is_malformed = malformed_tail
        data_lines: list[str] = []

        try:
            text = raw_event.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_event.decode("utf-8", errors="replace")
            is_malformed = True

        for raw_line in text.splitlines(keepends=False):
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip() or None
                continue
            if not line.startswith("data:"):
                continue
            data_lines.append(line[5:].lstrip(" "))

        if data_lines and data_lines[-1] == "[DONE]":
            is_done_marker = True
        elif data_lines:
            try:
                decoded = json.loads("\n".join(data_lines))
            except (TypeError, json.JSONDecodeError):
                is_malformed = True
            else:
                if isinstance(decoded, dict):
                    payload = decoded
                    payload_type = decoded.get("type")
                    if isinstance(payload_type, str) and payload_type.strip():
                        event_type = payload_type.strip()
                else:
                    is_malformed = True

        separator_length = _separator_length_at_end(raw_event)
        separator = raw_event[-separator_length:] if separator_length else b""
        return event_type, payload, is_done_marker, is_malformed, separator

    def _parse_event(
        self,
        raw_event: bytes,
        *,
        malformed_tail: bool = False,
    ) -> ParsedResponsesEvent:
        if len(raw_event) > self.max_event_bytes:
            raise ResponsesEventTooLarge(
                f"Responses SSE event exceeded {self.max_event_bytes} bytes"
            )
        if self.total_event_count >= self.max_stream_events:
            raise ResponsesStreamTooLarge(
                f"Responses SSE stream exceeded {self.max_stream_events} events"
            )

        observed_at = datetime.now()
        sequence_number = self._next_sequence_number
        self._next_sequence_number += 1
        self.event_count += 1
        self.total_event_count += 1
        self.last_event_observed_at = observed_at

        (
            event_type,
            payload,
            is_done_marker,
            is_malformed,
            separator,
        ) = self._parse_event_lines(raw_event, malformed_tail=malformed_tail)
        if is_done_marker:
            self.done_marker_count += 1
        if is_malformed:
            self.malformed_event_count += 1

        retained_raw = self._retain_raw(raw_event)
        event = ParsedResponsesEvent(
            sequence_number=sequence_number,
            event_type=event_type,
            payload=payload,
            is_done_marker=is_done_marker,
            is_malformed=is_malformed,
            byte_count=len(raw_event),
            observed_at=observed_at,
            forward_event=raw_event,
            raw_event=retained_raw,
            separator=separator,
        )
        self._hold_event(event)
        self._emit(event)
        if self.summary_reducer is not None and self.summary_reducer.overflow:
            raise ResponsesEventRetentionTooLarge(
                "Responses event summary retention exceeded configured bound"
            )
        return event

    def _emit_if_separator_at_end(self) -> Optional[ParsedResponsesEvent]:
        if self._pending_cr:
            return None
        separator_length = _separator_length_at_end(bytes(self._buffer[-4:]))
        if not separator_length:
            return None
        raw_event = bytes(self._buffer)
        self._buffer.clear()
        return self._parse_event(raw_event)

    def _admit_byte(self, value: int, *, count_stream_bytes: bool) -> None:
        if count_stream_bytes:
            if self.total_stream_bytes >= self.max_stream_bytes:
                raise ResponsesStreamTooLarge(
                    f"Responses SSE stream exceeded {self.max_stream_bytes} bytes"
                )
            self.total_stream_bytes += 1
        if len(self._buffer) >= self.max_event_bytes:
            raise ResponsesEventTooLarge(
                f"Responses SSE event exceeded {self.max_event_bytes} bytes"
            )
        self._buffer.append(value)

    def _consume_buffered_bytes(
        self,
        chunk: bytes,
        *,
        count_stream_bytes: bool,
    ) -> list[ParsedResponsesEvent]:
        if not chunk:
            self._last_feed_admitted_bytes = 0
            return []
        events: list[ParsedResponsesEvent] = []
        admitted_bytes = 0
        self._last_feed_admitted_bytes = 0
        try:
            for value in chunk:
                if self._pending_cr and value == 10:
                    self._admit_byte(value, count_stream_bytes=count_stream_bytes)
                    admitted_bytes += 1
                    self._pending_cr = False
                    completed = self._emit_if_separator_at_end()
                    if completed is not None:
                        events.append(completed)
                    continue
                if self._pending_cr:
                    self._pending_cr = False
                    completed = self._emit_if_separator_at_end()
                    if completed is not None:
                        events.append(completed)
                self._admit_byte(value, count_stream_bytes=count_stream_bytes)
                admitted_bytes += 1
                self._pending_cr = value == 13
                if self._pending_cr:
                    continue
                completed = self._emit_if_separator_at_end()
                if completed is not None:
                    events.append(completed)
        except (
            ResponsesBoundFailure,
            ResponsesEventTooLarge,
            ResponsesStreamTooLarge,
        ) as exception:
            # Callers can still deliver the valid prefix of a transport batch
            # before constructing a terminal for the rejected suffix.
            bound_failure = ResponsesBoundFailure.from_exception(exception)
            if not bound_failure.accepted_events:
                bound_failure.accepted_events = events
            if not bound_failure.accepted_bytes:
                bound_failure.accepted_bytes = admitted_bytes
            self._last_feed_admitted_bytes = admitted_bytes
            raise bound_failure from exception
        self._last_feed_admitted_bytes = admitted_bytes
        return events

    def feed(self, chunk: bytes) -> list[ParsedResponsesEvent]:
        """Admit bytes incrementally and emit each complete event once."""
        data = _coerce_bytes(chunk)
        return self._consume_buffered_bytes(data, count_stream_bytes=True)

    def finish(self) -> list[ParsedResponsesEvent]:
        """Return an unterminated UTF-8-complete tail as one malformed event.

        A leftover truncated multi-byte sequence is an incomplete transport
        split, not a Responses event. Drop it instead of emitting a
        replacement-decoded malformed frame.
        """
        if self._pending_cr:
            self._pending_cr = False
            completed = self._emit_if_separator_at_end()
            if completed is not None:
                leftover = self.finish()
                return [completed, *leftover]
        if not self._buffer:
            return []
        tail = bytes(self._buffer)
        self._buffer.clear()
        try:
            tail.decode("utf-8")
        except UnicodeDecodeError:
            return []
        return [self._parse_event(tail, malformed_tail=True)]

    def parse_event(self, raw_event: bytes) -> ParsedResponsesEvent:
        """Admit one already-framed event for legacy callers."""
        data = _coerce_bytes(raw_event)
        if not data:
            raise ValueError("raw_event must not be empty")
        if len(data) > self.max_event_bytes:
            raise ResponsesEventTooLarge(
                f"Responses SSE event exceeded {self.max_event_bytes} bytes"
            )
        self._check_stream_admission(len(data))
        self.total_stream_bytes += len(data)
        return self._parse_event(data)

    def adopt_event(
        self,
        payload: Optional[Dict[str, Any]],
        *,
        event_type: Optional[str] = None,
        wire_event: Optional[bytes] = None,
        is_done_marker: bool = False,
        is_malformed: bool = False,
        control: bool = False,
    ) -> ParsedResponsesEvent:
        """Register a structured/generated event without decoding JSON again.

        Adapter and terminal coordinators sometimes already own a decoded
        payload.  Re-encoding it only to feed it back through ``parse_event``
        would create a second parser boundary and can change its framing.  The
        adopted event still participates in the bounded event/reducer contract;
        its wire bytes, when present, are retained only as the output carrier.
        """
        if payload is not None and not isinstance(payload, dict):
            raise TypeError("payload must be a mapping or None")
        data = _coerce_bytes(wire_event) if wire_event is not None else b""
        if control:
            if self.control_event_count >= self.max_control_events:
                raise ResponsesStreamTooLarge(
                    "Responses generated control-event capacity exceeded"
                )
            if (
                self.control_event_bytes + len(data) > self.max_control_bytes
                or (data and len(data) > self.max_event_bytes)
            ):
                raise ResponsesEventTooLarge(
                    "Responses generated control event exceeded configured bound"
                )
        else:
            if data and len(data) > self.max_event_bytes:
                raise ResponsesEventTooLarge(
                    f"Responses SSE event exceeded {self.max_event_bytes} bytes"
                )
            if self.total_event_count >= self.max_stream_events:
                raise ResponsesStreamTooLarge(
                    f"Responses SSE stream exceeded {self.max_stream_events} events"
                )
            if data:
                self._check_stream_admission(len(data))
                self.total_stream_bytes += len(data)
        observed_at = datetime.now()
        sequence_number = self._next_sequence_number
        self._next_sequence_number += 1
        if not control:
            self.event_count += 1
            self.total_event_count += 1
        else:
            self.control_event_count += 1
            self.control_event_bytes += len(data)
        self.last_event_observed_at = observed_at
        if is_done_marker:
            self.done_marker_count += 1
        if is_malformed:
            self.malformed_event_count += 1
        separator_length = _separator_length_at_end(data)
        event = ParsedResponsesEvent(
            sequence_number=sequence_number,
            event_type=event_type,
            payload=payload,
            is_done_marker=is_done_marker,
            is_malformed=is_malformed,
            byte_count=len(data),
            observed_at=observed_at,
            forward_event=data or None,
            raw_event=None,
            separator=data[-separator_length:] if separator_length else b"",
        )
        if not control:
            self._hold_event(event)
            self._emit(event)
            if self.summary_reducer is not None and self.summary_reducer.overflow:
                raise ResponsesEventRetentionTooLarge(
                    "Responses event summary retention exceeded configured bound"
                )
        return event

    def prepare_replay(self, prefix_bytes: int) -> None:
        """Mark the exact prefix that will be replayed by the byte wrapper."""
        if prefix_bytes < 0:
            raise ValueError("prefix_bytes must not be negative")
        if prefix_bytes > self.total_stream_bytes:
            raise ValueError(
                "prefix_bytes cannot exceed bytes already admitted by the pipeline"
            )
        self._replay_remaining_bytes = int(prefix_bytes)
        self._replay_consumed_bytes = 0
        self._replay_consumed_event_bytes = 0
        self._replay_buffer = bytes(self._buffer)
        # Already-held events are the precommit prefix. Live remainder is
        # bounded by max_stream_events, not the hold budget.
        self.retain_events = False

    def _drain_replayed_events(self) -> list[ParsedResponsesEvent]:
        ready: list[ParsedResponsesEvent] = []
        while self._held_events:
            event = self._held_events[0]
            next_bytes = self._replay_consumed_event_bytes + event.byte_count
            if next_bytes > self._replay_consumed_bytes:
                break
            ready.append(self._held_events.popleft())
            self._held_event_bytes -= self._held_representation_bytes(event)
            self._replay_consumed_event_bytes = next_bytes
        return ready

    def feed_replayed(self, chunk: bytes) -> list[ParsedResponsesEvent]:
        """Consume a replayed prefix without decoding it a second time."""
        data = _coerce_bytes(chunk)
        if not data:
            return []
        if self._replay_remaining_bytes <= 0:
            self.retain_events = False
            return self.feed(data)

        prefix_length = min(len(data), self._replay_remaining_bytes)
        self._replay_remaining_bytes -= prefix_length
        self._replay_consumed_bytes += prefix_length
        events = self._drain_replayed_events()
        if self._replay_remaining_bytes == 0 and self._replay_buffer is not None:
            self._buffer = bytearray(self._replay_buffer)
            self._pending_cr = self._buffer.endswith(b"\r") and not self._buffer.endswith(
                b"\r\n"
            )
            self._replay_buffer = None
        if self._replay_remaining_bytes == 0:
            self.retain_events = False
        remainder = data[prefix_length:]
        if remainder:
            events.extend(self.feed(remainder))
        return events

    def attach_to_response(self, response: Any) -> Any:
        """Publish this attempt-scoped owner on a compatible response object."""
        try:
            setattr(response, RESPONSES_EVENT_PIPELINE_ATTR, self)
        except Exception:
            pass
        return response

    def drain_held_events(self) -> list[ParsedResponsesEvent]:
        """Return held precommit events and release their replay references."""
        events = list(self._held_events)
        self._held_events.clear()
        self._held_event_bytes = 0
        self.retain_events = False
        return events

    def diagnostics(self) -> Dict[str, Any]:
        diagnostics = {
            "event_count": self.event_count,
            "malformed_event_count": self.malformed_event_count,
            "done_marker_count": self.done_marker_count,
            "buffered_event_bytes": self.buffered_event_bytes,
            "max_event_bytes": self.max_event_bytes,
            "raw_event_count": len(self._raw_events),
            "raw_event_bytes": self.raw_event_bytes,
            "max_raw_events": self.max_raw_events,
            "max_raw_bytes": self.max_raw_bytes,
            "dropped_raw_event_count": self.dropped_raw_event_count,
            "dropped_raw_event_bytes": self.dropped_raw_event_bytes,
            "retains_raw_events": self.retain_raw_events,
            "total_stream_bytes": self.total_stream_bytes,
            "max_stream_bytes": self.max_stream_bytes,
            "total_event_count": self.total_event_count,
            "max_stream_events": self.max_stream_events,
            "held_event_count": self.held_event_count,
            "held_event_bytes": self.held_event_bytes,
            "max_held_events": self.max_held_events,
            "max_held_bytes": self.max_held_bytes,
            "omitted_held_event_count": self.omitted_held_event_count,
            "control_event_count": self.control_event_count,
            "control_event_bytes": self.control_event_bytes,
            "max_control_events": self.max_control_events,
            "max_control_bytes": self.max_control_bytes,
            "replay_remaining_bytes": self.replay_remaining_bytes,
        }
        if self.summary_reducer is not None:
            diagnostics["summary"] = self.summary_reducer.snapshot()
        return diagnostics

    def summary_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the bounded event-driven summary, when configured."""
        if self.summary_reducer is None:
            return None
        return self.summary_reducer.snapshot()


class ResponsesEventSummaryReducer:
    """Bounded event-driven summary used by logging and terminal consumers."""

    _IDENTITY_BYTES_LIMIT = 1024
    _ALIAS_LIMIT = 128
    _EVENT_TYPE_LIMIT = DEFAULT_MAX_EVENT_TYPES
    _DELTA_KEY_LIMIT = DEFAULT_MAX_DELTA_KEYS
    _USAGE_BYTES_LIMIT = 16 * 1024

    def __init__(
        self,
        *,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_output_items: int = DEFAULT_MAX_OUTPUT_ITEMS,
    ) -> None:
        self.max_output_bytes = max(1, int(max_output_bytes))
        self.max_output_items = max(1, int(max_output_items))
        self._event_count = 0
        self._event_types: list[str] = []
        self._event_counts: dict[str, int] = {}
        self._terminal_event_type: Optional[str] = None
        self._terminal_response_payload: Optional[dict[str, Any]] = None
        self._first_error_payload: Optional[dict[str, Any]] = None
        self._failed_error_payload: Optional[dict[str, Any]] = None
        self._response_id: Optional[str] = None
        self._response_model: Optional[str] = None
        self._usage_payload: Optional[dict[str, Any]] = None
        self._token_count_payload: Optional[dict[str, Any]] = None
        self._output_items: dict[str, dict[str, Any]] = {}
        self._ordered_keys: list[str] = []
        self._aliases: dict[str, str] = {}
        self._key_by_output_index: dict[int, str] = {}
        self._delta_keys_seen: set[str] = set()
        self._delta_key_order: Deque[str] = deque()
        self._stream_text_parts: list[str] = []
        self._stream_text_bytes = 0
        self._output_bytes = 0
        self._overflow = False
        self._omitted_items = 0
        self._omitted_output_bytes = 0

    @property
    def overflow(self) -> bool:
        return self._overflow

    @property
    def event_count(self) -> int:
        return self._event_count

    @staticmethod
    def _event_payload(event: Any) -> Optional[dict[str, Any]]:
        if isinstance(event, ParsedResponsesEvent):
            return event.payload
        return event if isinstance(event, dict) else None

    @staticmethod
    def _event_type(event: Any, payload: Optional[dict[str, Any]]) -> Optional[str]:
        if isinstance(event, ParsedResponsesEvent):
            return event.event_type
        event_type = payload.get("type") if isinstance(payload, dict) else None
        return event_type if isinstance(event_type, str) else None

    @staticmethod
    def _bounded_identity(value: Any) -> Optional[str]:
        return bounded_identity_key(
            value,
            limit=ResponsesEventSummaryReducer._IDENTITY_BYTES_LIMIT,
        )

    @classmethod
    def _item_key(cls, payload: dict[str, Any], *, fallback: int) -> str:
        item = payload.get("item")
        candidates: Iterable[Any] = (
            item.get("id") if isinstance(item, dict) else None,
            item.get("call_id") if isinstance(item, dict) else None,
            payload.get("item_id"),
        )
        for candidate in candidates:
            bounded = cls._bounded_identity(candidate)
            if bounded:
                return bounded
        output_index = payload.get("output_index")
        if isinstance(output_index, int) and not isinstance(output_index, bool):
            return f"output:{output_index}"
        return f"fallback:{fallback}"

    def _remember_delta_key(self, key: str) -> None:
        if key in self._delta_keys_seen:
            return
        if len(self._delta_keys_seen) >= self._DELTA_KEY_LIMIT:
            evicted = self._delta_key_order.popleft()
            self._delta_keys_seen.discard(evicted)
        self._delta_keys_seen.add(key)
        self._delta_key_order.append(key)

    def _record_event_type(self, event_type: Optional[str]) -> None:
        if not event_type:
            return
        if (
            event_type not in self._event_counts
            and len(self._event_counts) >= self._EVENT_TYPE_LIMIT
        ):
            self._overflow = True
            return
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        if (
            event_type not in self._event_types
            and len(self._event_types) < self._EVENT_TYPE_LIMIT
        ):
            self._event_types.append(event_type)

    def _bounded_copy(self, value: Any, limit: int) -> Optional[Any]:
        size = _json_size(value)
        if size is None or size > limit:
            self._overflow = True
            return None
        try:
            return copy.deepcopy(value)
        except (TypeError, ValueError):
            self._overflow = True
            return None

    def _record_terminal(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        response_payload = payload.get("response")
        if not isinstance(response_payload, dict):
            return
        self._terminal_event_type = event_type
        copied = self._bounded_copy(response_payload, self.max_output_bytes)
        if isinstance(copied, dict):
            self._terminal_response_payload = copied
            response_id = self._bounded_identity(copied.get("id"))
            response_model = self._bounded_identity(copied.get("model"))
            if response_id:
                self._response_id = response_id
            if response_model:
                self._response_model = response_model
        if event_type == "response.failed":
            error = response_payload.get("error")
            if isinstance(error, dict):
                copied_error = self._bounded_copy(error, self._USAGE_BYTES_LIMIT)
                if isinstance(copied_error, dict):
                    self._failed_error_payload = copied_error

    def _record_usage(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            response = payload.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return
        copied = self._bounded_copy(usage, self._USAGE_BYTES_LIMIT)
        if isinstance(copied, dict):
            self._usage_payload = copied

    def _item_slot(self, key: str) -> Optional[tuple[dict[str, Any], bool]]:
        existing = self._output_items.get(key)
        if existing is not None:
            return existing, False
        if len(self._output_items) >= self.max_output_items:
            self._overflow = True
            self._omitted_items += 1
            return None
        return {"id": key}, True

    def _commit_item(
        self,
        key: str,
        candidate: dict[str, Any],
        *,
        created: bool,
    ) -> None:
        if created:
            self._ordered_keys.append(key)
        self._output_items[key] = candidate

    def _charge_delta(self, previous_size: int, new_size: Optional[int]) -> bool:
        if new_size is None:
            self._overflow = True
            return False
        projected = self._output_bytes - previous_size + new_size
        if projected > self.max_output_bytes:
            self._overflow = True
            self._omitted_output_bytes += max(0, projected - self.max_output_bytes)
            return False
        self._output_bytes = projected
        return True

    def _merge_item(self, payload: dict[str, Any]) -> None:
        item = payload.get("item")
        if not isinstance(item, dict):
            return
        key = self._item_key(payload, fallback=len(self._ordered_keys))
        index = payload.get("output_index")
        if isinstance(index, int) and not isinstance(index, bool):
            existing_key = self._key_by_output_index.get(index)
            if existing_key:
                key = existing_key
            elif len(self._key_by_output_index) < self._ALIAS_LIMIT:
                self._key_by_output_index[index] = key
        for alias in (
            self._bounded_identity(item.get("id")),
            self._bounded_identity(item.get("call_id")),
            self._bounded_identity(payload.get("item_id")),
        ):
            if alias and (
                alias in self._aliases or len(self._aliases) < self._ALIAS_LIMIT
            ):
                self._aliases[alias] = key
        slot = self._item_slot(key)
        if slot is None:
            return
        target, created = slot
        previous_size = _json_size(target) or 0
        candidate = {**copy.deepcopy(target), **copy.deepcopy(item)}
        if not self._charge_delta(previous_size, _json_size(candidate)):
            return
        self._commit_item(key, candidate, created=created)

    def _append_text(self, key: str, value: Any, *, done: bool) -> None:
        if not isinstance(value, str):
            return
        if done and key in self._delta_keys_seen:
            return
        encoded = value.encode("utf-8")
        slot = self._item_slot(key)
        if slot is None:
            return
        target, created = slot
        content = target.get("content")
        if not isinstance(content, list):
            content = []
            output_text = None
        else:
            content = [copy.deepcopy(part) for part in content]
            output_text = None
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    output_text = part
                    break
        if output_text is None:
            output_text = {"type": "output_text", "text": ""}
            content.append(output_text)
        next_text = value if done else f"{output_text.get('text', '')}{value}"
        candidate = {
            **copy.deepcopy(target),
            "content": content,
            "type": target.get("type") or "message",
            "role": target.get("role") or "assistant",
        }
        for part in candidate["content"]:
            if isinstance(part, dict) and part.get("type") == "output_text":
                part["text"] = next_text
                break
        if not done and self._stream_text_bytes + len(encoded) > self.max_output_bytes:
            self._overflow = True
            self._omitted_output_bytes += (
                self._stream_text_bytes + len(encoded) - self.max_output_bytes
            )
            return
        if done and key not in self._delta_keys_seen:
            if self._stream_text_bytes + len(encoded) > self.max_output_bytes:
                self._overflow = True
                self._omitted_output_bytes += (
                    self._stream_text_bytes + len(encoded) - self.max_output_bytes
                )
                return
        previous_size = _json_size(target) or 0
        if not self._charge_delta(previous_size, _json_size(candidate)):
            return
        if not done or key not in self._delta_keys_seen:
            self._stream_text_bytes += len(encoded)
            self._stream_text_parts.append(value)
            self._remember_delta_key(key)
        self._commit_item(key, candidate, created=created)

    def _append_arguments(self, payload: dict[str, Any], *, done: bool) -> None:
        raw_alias = self._bounded_identity(payload.get("item_id"))
        key = self._aliases.get(
            raw_alias or "",
            self._item_key(payload, fallback=len(self._ordered_keys)),
        )
        slot = self._item_slot(key)
        if slot is None:
            return
        target, created = slot
        value = payload.get("arguments")
        if not isinstance(value, str):
            value = payload.get("delta")
        if not isinstance(value, str):
            return
        current = target.get("arguments")
        candidate = {
            **copy.deepcopy(target),
            "arguments": value if done else f"{current or ''}{value}",
            "type": target.get("type") or "function_call",
        }
        previous_size = _json_size(target) or 0
        if not self._charge_delta(previous_size, _json_size(candidate)):
            return
        self._commit_item(key, candidate, created=created)

    def consume(self, event: Any) -> None:
        payload = self._event_payload(event)
        if not isinstance(payload, dict):
            return
        self._event_count += 1
        event_type = self._event_type(event, payload)
        self._record_event_type(event_type)
        self._record_usage(payload)

        if event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            self._record_terminal(event_type, payload)
            return
        if event_type == "error":
            if self._first_error_payload is None:
                nested = payload.get("error")
                source = nested if isinstance(nested, dict) else payload
                copied = self._bounded_copy(source, self._USAGE_BYTES_LIMIT)
                if isinstance(copied, dict):
                    self._first_error_payload = copied
            return
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            self._merge_item(payload)
            return
        if event_type in {
            "response.output_text.delta",
            "response.output_text.done",
        }:
            raw_alias = self._bounded_identity(payload.get("item_id"))
            key = self._aliases.get(
                raw_alias or "",
                self._item_key(payload, fallback=len(self._ordered_keys)),
            )
            self._append_text(
                key,
                payload.get("delta")
                if event_type.endswith(".delta")
                else payload.get("text"),
                done=event_type.endswith(".done"),
            )
            return
        if event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
        }:
            self._append_arguments(payload, done=event_type.endswith(".done"))
            return
        if event_type == "token_count":
            copied = self._bounded_copy(payload, self._USAGE_BYTES_LIMIT)
            if isinstance(copied, dict):
                self._token_count_payload = copied

    @property
    def terminal_event_type(self) -> Optional[str]:
        return self._terminal_event_type

    @property
    def terminal_response_payload(self) -> Optional[dict[str, Any]]:
        return self._terminal_response_payload

    @property
    def first_error_payload(self) -> Optional[dict[str, Any]]:
        return self._first_error_payload

    @property
    def failed_error_payload(self) -> Optional[dict[str, Any]]:
        return self._failed_error_payload

    def snapshot(self) -> dict[str, Any]:
        output = [
            self._output_items[key]
            for key in self._ordered_keys
            if key in self._output_items
        ]
        return {
            "event_count": self._event_count,
            "event_types": list(self._event_types),
            "event_counts": dict(self._event_counts),
            "terminal_event_type": self._terminal_event_type,
            "terminal_response_payload": self._terminal_response_payload,
            "first_error_payload": self._first_error_payload,
            "failed_error_payload": self._failed_error_payload,
            "response_id": self._response_id,
            "response_model": self._response_model,
            "usage": self._usage_payload,
            "token_count": self._token_count_payload,
            "output": output,
            "stream_text": "".join(self._stream_text_parts),
            "output_bytes": self._output_bytes,
            "overflow": self._overflow,
            "omitted_items": self._omitted_items,
            "omitted_output_bytes": self._omitted_output_bytes,
        }


def create_responses_event_pipeline(
    *,
    retain_events: bool = False,
    summary_reducer: Optional[ResponsesEventSummaryReducer] = None,
) -> ResponsesEventPipeline:
    """Create one pipeline from the process-level Responses limits."""
    max_event_bytes, max_raw_events = resolve_responses_event_limits()
    max_raw_events, max_raw_bytes = resolve_responses_raw_retention_limits()
    max_stream_bytes, max_stream_events = resolve_responses_stream_limits()
    max_held_events, max_held_bytes, max_output_bytes, max_output_items = (
        resolve_responses_retention_limits()
    )
    reducer = summary_reducer or ResponsesEventSummaryReducer(
        max_output_bytes=max_output_bytes,
        max_output_items=max_output_items,
    )
    retain_raw_events = str(
        os.environ.get("AAWM_RESPONSES_EVENT_RAW_DIAGNOSTICS", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    return ResponsesEventPipeline(
        max_event_bytes=max_event_bytes,
        max_raw_events=max_raw_events,
        max_raw_bytes=max_raw_bytes,
        max_stream_bytes=max_stream_bytes,
        max_stream_events=max_stream_events,
        max_held_events=max_held_events,
        max_held_bytes=max_held_bytes,
        retain_raw_events=retain_raw_events,
        retain_events=retain_events,
        summary_reducer=reducer,
    )


def summarize_responses_events(
    events: Iterable[ParsedResponsesEvent],
    *,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_output_items: int = DEFAULT_MAX_OUTPUT_ITEMS,
) -> dict[str, Any]:
    """Build a bounded logging summary from already-parsed events."""
    reducer = ResponsesEventSummaryReducer(
        max_output_bytes=max_output_bytes,
        max_output_items=max_output_items,
    )
    for event in events:
        reducer.consume(event)
    return reducer.snapshot()
