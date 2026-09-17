"""Bounded typed parsing for OpenAI Responses SSE events."""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Optional


DEFAULT_MAX_EVENT_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_RAW_EVENTS = 64
DEFAULT_MAX_STREAM_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_STREAM_EVENTS = 100_000
EVENT_BYTES_ENV = "AAWM_RESPONSES_MAX_EVENT_BYTES"
RAW_EVENTS_ENV = "AAWM_RESPONSES_MAX_RAW_EVENTS"
STREAM_BYTES_ENV = "AAWM_RESPONSES_MAX_STREAM_BYTES"
STREAM_EVENTS_ENV = "AAWM_RESPONSES_MAX_STREAM_EVENTS"


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
    """Resolve configured per-event and raw-retention bounds."""
    return (
        _env_int(EVENT_BYTES_ENV, DEFAULT_MAX_EVENT_BYTES, minimum=1),
        _env_int(RAW_EVENTS_ENV, DEFAULT_MAX_RAW_EVENTS, minimum=0),
    )


def resolve_responses_stream_limits() -> tuple[int, int]:
    """Resolve configured aggregate stream and event-count bounds."""
    return (
        _env_int(STREAM_BYTES_ENV, DEFAULT_MAX_STREAM_BYTES, minimum=1),
        _env_int(STREAM_EVENTS_ENV, DEFAULT_MAX_STREAM_EVENTS, minimum=1),
    )


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


class ResponsesEventTooLarge(ValueError):
    """An SSE event exceeded the configured event bound."""


class ResponsesStreamTooLarge(ValueError):
    """A Responses stream exceeded an aggregate configured bound."""


@dataclass
class ResponsesEventPipeline:
    """Incremental SSE parser with explicit event and raw-retention bounds."""

    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES
    max_raw_events: int = DEFAULT_MAX_RAW_EVENTS
    max_stream_bytes: int = DEFAULT_MAX_STREAM_BYTES
    max_stream_events: int = DEFAULT_MAX_STREAM_EVENTS
    retain_raw_events: bool = False

    _buffer: bytearray = field(default_factory=bytearray, init=False)
    _decoded_event_type: Optional[str] = None
    _next_sequence_number: int = 1
    _raw_events: Deque[bytes] = field(default_factory=deque, init=False)
    event_count: int = 0
    malformed_event_count: int = 0
    done_marker_count: int = 0
    dropped_raw_event_count: int = 0
    total_stream_bytes: int = 0
    total_event_count: int = 0
    last_event_observed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be positive")
        if self.max_raw_events < 0:
            raise ValueError("max_raw_events must not be negative")
        if self.max_stream_bytes <= 0:
            raise ValueError("max_stream_bytes must be positive")
        if self.max_stream_events <= 0:
            raise ValueError("max_stream_events must be positive")

    @property
    def buffered_event_bytes(self) -> int:
        return len(self._buffer)

    @staticmethod
    def _event_boundary(buffer: bytes) -> Optional[tuple[int, int]]:
        crlf_boundary = buffer.find(b"\r\n\r\n")
        lf_boundary = buffer.find(b"\n\n")
        if crlf_boundary < 0 and lf_boundary < 0:
            return None
        if crlf_boundary < 0 or (lf_boundary >= 0 and lf_boundary < crlf_boundary):
            boundary_start, boundary_length = lf_boundary, 2
        else:
            boundary_start, boundary_length = crlf_boundary, 4
        return boundary_start + boundary_length, boundary_length

    def feed(self, chunk: bytes) -> list[ParsedResponsesEvent]:
        """Consume raw bytes once and emit each complete event exactly once."""
        if not chunk:
            return []
        self._buffer.extend(chunk)
        self.total_stream_bytes += len(chunk)
        if self.total_stream_bytes > self.max_stream_bytes:
            raise ResponsesStreamTooLarge(
                f"Responses SSE stream exceeded {self.max_stream_bytes} bytes"
            )
        if len(self._buffer) > self.max_event_bytes:
            raise ResponsesEventTooLarge(
                f"Responses SSE event exceeded {self.max_event_bytes} bytes"
            )

        events: list[ParsedResponsesEvent] = []
        while True:
            boundary = self._event_boundary(self._buffer)
            if boundary is None:
                break
            boundary_end, _ = boundary
            if boundary_end > self.max_event_bytes:
                raise ResponsesEventTooLarge(
                    f"Responses SSE event exceeded {self.max_event_bytes} bytes"
                )
            raw_event = bytes(self._buffer[:boundary_end])
            del self._buffer[:boundary_end]
            events.append(self._parse_event(raw_event))

        if self.total_event_count > self.max_stream_events:
            raise ResponsesStreamTooLarge(
                f"Responses SSE stream exceeded {self.max_stream_events} events"
            )

        if len(self._buffer) > self.max_event_bytes:
            raise ResponsesEventTooLarge(
                f"Responses SSE event exceeded {self.max_event_bytes} bytes"
            )
        return events

    def finish(self) -> list[ParsedResponsesEvent]:
        """Return an unterminated tail as malformed, then clear framing state."""
        tail = bytes(self._buffer)
        self._buffer.clear()
        if not tail:
            return []
        return [self._parse_event(tail, malformed_tail=True)]

    def parse_event(self, raw_event: bytes) -> ParsedResponsesEvent:
        """Parse one already-framed event for bounded pre-commit inspection."""
        return self._parse_event(raw_event)

    def _parse_event_lines(self, raw_event: bytes, *, malformed_tail: bool) -> tuple[
        Optional[str],
        Optional[Dict[str, Any]],
        bool,
        bool,
    ]:
        event_type = self._decoded_event_type
        self._decoded_event_type = None
        payload: Optional[Dict[str, Any]] = None
        is_done_marker = False
        is_malformed = malformed_tail
        data_lines: list[str] = []

        try:
            text = raw_event.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_event.decode("utf-8", errors="replace")
            is_malformed = True

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip() or None
                continue
            if not line.startswith("data:"):
                continue
            data_lines.append(line.removeprefix("data:").lstrip(" "))

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
        return event_type, payload, is_done_marker, is_malformed

    def _parse_event(
        self, raw_event: bytes, *, malformed_tail: bool = False
    ) -> ParsedResponsesEvent:
        observed_at = datetime.now()
        sequence_number = self._next_sequence_number
        self._next_sequence_number += 1
        self.event_count += 1
        self.total_event_count += 1
        self.last_event_observed_at = observed_at

        event_type, payload, is_done_marker, is_malformed = (
            self._parse_event_lines(raw_event, malformed_tail=malformed_tail)
        )
        if is_done_marker:
            self.done_marker_count += 1

        if is_malformed:
            self.malformed_event_count += 1

        retained_raw: Optional[bytes] = None
        if self.retain_raw_events:
            if self.max_raw_events == 0:
                self.dropped_raw_event_count += 1
            else:
                retained_raw = raw_event
                self._raw_events.append(raw_event)
                if len(self._raw_events) > self.max_raw_events:
                    self._raw_events.popleft()
                    self.dropped_raw_event_count += 1

        return ParsedResponsesEvent(
            sequence_number=sequence_number,
            event_type=event_type,
            payload=payload,
            is_done_marker=is_done_marker,
            is_malformed=is_malformed,
            byte_count=len(raw_event),
            observed_at=observed_at,
            forward_event=raw_event,
            raw_event=retained_raw,
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "event_count": self.event_count,
            "malformed_event_count": self.malformed_event_count,
            "done_marker_count": self.done_marker_count,
            "buffered_event_bytes": self.buffered_event_bytes,
            "max_event_bytes": self.max_event_bytes,
            "raw_event_count": len(self._raw_events),
            "max_raw_events": self.max_raw_events,
            "dropped_raw_event_count": self.dropped_raw_event_count,
            "retains_raw_events": self.retain_raw_events,
            "total_stream_bytes": self.total_stream_bytes,
            "max_stream_bytes": self.max_stream_bytes,
            "total_event_count": self.total_event_count,
            "max_stream_events": self.max_stream_events,
        }
