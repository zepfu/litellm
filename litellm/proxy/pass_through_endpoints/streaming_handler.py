import asyncio
import codecs
import inspect
import json
import os
import struct
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

import litellm
from litellm.integrations.aawm_session_history.waits import (
    wait_for,
    wait_iterator,
    enrich_request_waits,
)
from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_stream_shape,
    diagnostic_payload_capture_enabled,
    passthrough_full_payload_capture_enabled,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.cohere.cancellation import (
    aclose_upstream_response_once,
    is_cohere_api_host,
    is_cohere_provider_name,
)
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.aawm_route_logging import (
    _AAWM_PARSED_CODEX_REVIEW_DECISIONS_KWARGS_KEY,
    emit_aawm_route_status_event,
    record_aawm_route_rollup,
    record_aawm_route_rollup_failure,
    record_aawm_route_rollup_turn,
)
from litellm.proxy.aawm_session_transfer.identity import extract_transfer_identity
from litellm.proxy.aawm_session_transfer.registry import (
    safe_finalize,
    safe_mark_phase,
    safe_record_chunks,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_delivered_disposition import (
    annotate_delivered_wire_failure,
    get_delivered_wire_disposition,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardPassThroughResponseObject

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .aawm_alias_routing.audit_persist import _emit_aawm_terminal_error
from .aawm_adapter_runtime.deferred_success import DeferredPassthroughSuccess
from .aawm_adapter_runtime.openai_responses_events import (
    ParsedResponsesEvent,
    ResponsesBoundFailure,
    ResponsesEventRetentionTooLarge,
    ResponsesEventPipeline,
    ResponsesEventTooLarge,
    ResponsesStreamTooLarge,
    attach_responses_emissions,
    attach_responses_events,
    create_responses_event_pipeline,
    responses_emissions_from_chunk,
    responses_event_pipeline_from_response,
    responses_events_from_chunk,
)
from .aawm_adapter_runtime.provider_call_ledger import (
    get_responses_replay_commitment_reason,
    publish_responses_replay_commitment,
    responses_effective_provider_tools_allowed,
)
from .success_handler import PassThroughEndpointLogging



def _truthy_env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_RESPONSES_NON_SUBSTANTIVE_LIFECYCLE_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
    }
)
_RESPONSES_PRE_COMMIT_FAILURE_EVENTS = frozenset({"response.failed", "error"})
_RECOGNIZED_PROVIDER_TERMINAL_EVENTS = frozenset(
    {
        "response.failed",
        "response.incomplete",
        "error",
    }
)

# Recognized terminals for metadata (includes response.completed, which
# is committed separately in _inspect_responses_events).
_RECOGNIZED_TERMINAL_METADATA_EVENTS = frozenset(
    {
        "response.completed",
        "response.failed",
        "response.incomplete",
        "error",
    }
)
# OPENAI-080: Oracle-frozen exact typed allowlist for replay-safe private events.
# These are exact event-name-and-shape gates, not prefix matching.
_RESPONSES_SAFE_ITEM_ADDED_TYPES = frozenset({"message", "function_call", "custom_tool_call", "reasoning"})
_RESPONSES_SAFE_CONTENT_PART_ADDED_TYPES = frozenset({"output_text", "refusal", "reasoning_text"})
_RESPONSES_SUBSTANTIVE_EVENT_PREFIXES = (
    "response.output_text",
    "response.output_item",
    "response.content_part",
    "response.refusal",
    "response.reasoning",
    "response.function_call",
    "response.custom_tool_call",
    "response.mcp_call",
    "response.file_search",
    "response.web_search",
    "response.code_interpreter",
    "response.image_generation",
    "response.computer",
    "response.apply_patch",
    "response.local_shell",
    "response.shell",
)
_RESPONSES_PRE_COMMIT_MAX_BUFFER_BYTES = 64 * 1024
RESPONSES_PRE_COMMIT_TRANSIENT_RETRY_WAIT_SECONDS = 10.0
RESPONSES_PRE_COMMIT_TRANSIENT_MAX_ATTEMPTS = 2
_RESPONSES_TRANSIENT_CAPACITY_CLASSES = frozenset(
    {
        "server_overloaded",
        "upstream_overloaded",
        "capacity_exhausted",
        "upstream_transient_internal",
    }
)
_RESPONSES_ORDINARY_RETRY_CLASSES = frozenset(
    {
        "openai_response_protection_unavailable",
    }
)
_RESPONSES_TRANSIENT_STREAM_CLASSES = frozenset(
    {
        "stream_interrupted",
    }
)
_RESPONSES_ACCOUNT_EXHAUSTION_CLASSES = frozenset(
    {
        "usage_limit_reached",
    }
)
_OPENAI_RESPONSE_PROTECTION_UNAVAILABLE_ERROR = (
    "response protection is unavailable"
)
_RESPONSES_STREAM_REDISPATCH_CLASSES = frozenset(
    {
        "token_invalidated",
        "openai_responses_unpersisted_item_not_found",
    }
)


class ResponsesStreamPreCommitFailure(Exception):
    """Upstream Responses stream failed before any substantive client byte."""

    def __init__(
        self,
        *,
        error_class: str,
        classification: str,
        retryable: bool,
        retry_after_seconds: float = RESPONSES_PRE_COMMIT_TRANSIENT_RETRY_WAIT_SECONDS,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        message: Optional[str] = None,
        status_code: Optional[int] = None,
        pre_commit_retry_exhausted: bool = False,
        error_payload: Optional[Dict[str, Any]] = None,
        provider_returned: bool = True,
        failure_origin: Optional[str] = None,
    ) -> None:
        self.error_class = error_class
        self.classification = classification
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.error_code = error_code
        self.error_type = error_type
        self.status_code = status_code
        self.pre_commit_retry_exhausted = pre_commit_retry_exhausted
        self.error_payload = error_payload if isinstance(error_payload, dict) else None
        self.message = message or classification
        self.provider_returned = provider_returned
        self.failure_origin = failure_origin
        self.detail = {
            "error": {
                "message": self.message,
                "type": error_type or error_class,
                "code": error_code or classification,
                "retryable": retryable,
            }
        }
        super().__init__(self.message)

    def as_http_exception(self):
        from fastapi import HTTPException, status as fastapi_status

        if self.status_code is not None:
            status_code = int(self.status_code)
        elif self.error_class in _RESPONSES_ACCOUNT_EXHAUSTION_CLASSES:
            status_code = fastapi_status.HTTP_429_TOO_MANY_REQUESTS
        elif self.retryable or self.error_class in _RESPONSES_TRANSIENT_CAPACITY_CLASSES:
            status_code = fastapi_status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            status_code = fastapi_status.HTTP_502_BAD_GATEWAY
        headers = None
        if self.retryable or self.pre_commit_retry_exhausted:
            retry_after = self.retry_after_seconds
            if retry_after is None or retry_after <= 0:
                retry_after = RESPONSES_PRE_COMMIT_TRANSIENT_RETRY_WAIT_SECONDS
            if retry_after == int(retry_after):
                retry_after_header = str(int(retry_after))
            else:
                retry_after_header = str(retry_after)
            headers = {"Retry-After": retry_after_header}
        http_exception = HTTPException(
            status_code=status_code,
            detail=self.detail,
            headers=headers,
        )
        setattr(http_exception, "_aawm_provider_returned", self.provider_returned)
        if self.error_class == "openai_response_protection_unavailable":
            setattr(
                http_exception,
                "_aawm_typed_precommit_error_class",
                self.error_class,
            )
        return http_exception


class _PrefixedHttpxByteStream:
    """Replay peeked SSE bytes, then continue the original upstream iterator."""

    def __init__(
        self,
        response: httpx.Response,
        prefix: Any,
        remainder: Any = None,
        *,
        event_pipeline: Optional[ResponsesEventPipeline] = None,
        bound_failure: Optional[ResponsesBoundFailure] = None,
    ) -> None:
        self._response = response
        self._prefix = prefix
        self._remainder = remainder
        self._aawm_responses_event_pipeline = event_pipeline
        self._bound_failure = bound_failure
        if event_pipeline is not None:
            prepare_replay = getattr(event_pipeline, "prepare_replay", None)
            prefix_bytes = getattr(prefix, "byte_count", 0)
            if callable(prepare_replay):
                # feed_until may stop in the middle of a coalesced HTTPX
                # read. Replay only the bytes the parser has already admitted;
                # the wrapper routes the stored suffix through live admission
                # after the accepted prefix has been yielded.
                admitted_bytes = int(
                    getattr(event_pipeline, "total_stream_bytes", prefix_bytes)
                )
                prepare_replay(min(int(prefix_bytes), admitted_bytes))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _close_prefix(self) -> None:
        if getattr(self, "_prefix_closed", False):
            return
        self._prefix_closed = True
        close = getattr(self._prefix, "close", None)
        if callable(close):
            close()

    async def aiter_bytes(self):
        async def _iter_live_chunk(chunk: bytes):
            pipeline = self._aawm_responses_event_pipeline
            if pipeline is None:
                yield chunk
                return
            try:
                events = pipeline.feed(chunk)
            except (
                ResponsesBoundFailure,
                ResponsesEventTooLarge,
                ResponsesStreamTooLarge,
            ) as exc:
                bound_failure = ResponsesBoundFailure.from_exception(exc)
                accepted_frames = [
                    event.forward_event or event.raw_event or b""
                    for event in bound_failure.accepted_events
                    if (event.forward_event or event.raw_event)
                ]
                if accepted_frames:
                    yield attach_responses_events(
                        b"".join(accepted_frames),
                        bound_failure.accepted_events,
                    )
                raise bound_failure from exc
            yield attach_responses_events(chunk, events)

        try:
            if hasattr(self._prefix, "iter_chunks"):
                prefix_chunks = self._prefix.iter_chunks()
            else:
                prefix_chunks = iter(self._prefix)
            for chunk in prefix_chunks:
                if chunk:
                    pipeline = self._aawm_responses_event_pipeline
                    if pipeline is None:
                        yield chunk
                        continue
                    replay_length = min(
                        len(chunk),
                        int(getattr(pipeline, "replay_remaining_bytes", 0)),
                    )
                    replayed = chunk[:replay_length]
                    if replayed:
                        events = pipeline.feed_replayed(replayed)
                        yield attach_responses_events(replayed, events)
                    live_suffix = chunk[replay_length:]
                    if live_suffix:
                        async for live_chunk in _iter_live_chunk(live_suffix):
                            yield live_chunk
            # Retire prefix storage before pulling the live remainder so
            # spooled files are closed during streaming, not after.
            self._close_prefix()
            if self._bound_failure is not None:
                bound_failure = self._bound_failure
                self._bound_failure = None
                raise ResponsesBoundFailure(bound_failure.cause) from bound_failure
            if self._remainder is None:
                return
            if self._aawm_responses_event_pipeline is not None:
                self._aawm_responses_event_pipeline.retain_events = False
            async for chunk in self._remainder:
                if not chunk:
                    continue
                async for live_chunk in _iter_live_chunk(bytes(chunk)):
                    yield live_chunk
        except BaseException:
            await self.aclose()
            raise
        finally:
            self._close_prefix()

    async def aclose(self) -> None:
        self._close_prefix()
        close = getattr(self._response, "aclose", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


class _ResponsesPreCommitPrefixStore:
    """Bounded-memory storage for an exact replayable response prefix."""

    _CHUNK_LENGTH = struct.Struct("!Q")

    def __init__(
        self,
        max_buffer_bytes: int,
        *,
        max_storage_bytes: Optional[int] = None,
    ) -> None:
        memory_limit = max(0, int(max_buffer_bytes))
        self._max_storage_bytes = (
            None
            if max_storage_bytes is None
            else max(0, int(max_storage_bytes))
        )
        self._byte_count = 0
        self._data = tempfile.SpooledTemporaryFile(
            max_size=memory_limit,
            mode="w+b",
        )
        self._lengths = tempfile.SpooledTemporaryFile(
            max_size=memory_limit,
            mode="w+b",
        )
        self._closed = False

    def append(self, chunk: bytes) -> None:
        if self._closed:
            raise RuntimeError("Responses pre-commit prefix store is closed")
        if not chunk:
            return
        if (
            self._max_storage_bytes is not None
            and self._byte_count + len(chunk) > self._max_storage_bytes
        ):
            raise ResponsesStreamTooLarge(
                "Responses pre-commit prefix exceeded configured storage bound"
            )
        self._data.seek(0, os.SEEK_END)
        self._data.write(chunk)
        self._lengths.seek(0, os.SEEK_END)
        self._lengths.write(self._CHUNK_LENGTH.pack(len(chunk)))
        self._byte_count += len(chunk)

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def read_range(self, start: int, end: int) -> bytes:
        if self._closed or end <= start:
            return b""
        self._data.seek(start)
        return self._data.read(end - start)

    def iter_chunks(self):
        if self._closed:
            return
        self._data.seek(0)
        self._lengths.seek(0)
        while True:
            encoded_length = self._lengths.read(self._CHUNK_LENGTH.size)
            if not encoded_length:
                return
            if len(encoded_length) != self._CHUNK_LENGTH.size:
                raise RuntimeError("Corrupt Responses pre-commit prefix metadata")
            chunk_length = self._CHUNK_LENGTH.unpack(encoded_length)[0]
            chunk = self._data.read(chunk_length)
            if len(chunk) != chunk_length:
                raise RuntimeError("Corrupt Responses pre-commit prefix data")
            if chunk:
                yield chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._data.close()
        finally:
            self._lengths.close()


class _ResponsesPreCommitSSEParser:
    """Inspect events from the attempt-scoped pipeline in arrival order."""

    def __init__(
        self,
        prefix: _ResponsesPreCommitPrefixStore,
        pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> None:
        # Keep the prefix reference for compatibility and diagnostics. Framing
        # and JSON admission are owned exclusively by ``pipeline``.
        self._prefix = prefix
        self._pipeline = pipeline or create_responses_event_pipeline(
            retain_events=True,
        )
        self._saw_lifecycle = False
        self._first_error_payload: Optional[Dict[str, Any]] = None
        self._first_error_event: Optional[str] = None
        self._saw_substantive = False
        self._substantive_event: Optional[str] = None
        self._first_terminal_decision: Optional[str] = None
        self._commitment_reason: Optional[str] = None
        self._response_id: Optional[str] = None
        self._output_items: Dict[int, tuple[str, str]] = {}

    def _consume_event(self, parsed_event: ParsedResponsesEvent) -> None:
        decision, error_payload, event_type, commitment_reason = (
            PassThroughStreamingHandler._inspect_responses_events(
                parsed_event
            )
        )
        association_valid = self._validate_event_association(
            parsed_event,
            record=decision == "lifecycle",
        )
        if not association_valid:
            commitment_reason = "unknown_semantics"
            if decision != "failed":
                decision = "substantive"
        if decision in {"failed", "substantive"}:
            # The first terminal/substantive event wins. A later event in the
            # same HTTPX read must not erase the earlier precommit outcome.
            if self._first_terminal_decision is None:
                self._first_terminal_decision = decision
        if decision == "substantive" and not self._saw_substantive:
            self._saw_substantive = True
            self._substantive_event = event_type
            self._commitment_reason = commitment_reason
        elif decision == "failed" and self._first_error_payload is None:
            self._first_error_payload = error_payload
            self._first_error_event = event_type
            self._commitment_reason = commitment_reason
        if decision == "lifecycle":
            self._saw_lifecycle = True

    @staticmethod
    def _is_nonnegative_index(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def _validate_event_association(
        self,
        parsed_event: ParsedResponsesEvent,
        *,
        record: bool,
    ) -> bool:
        event_type = parsed_event.event_type
        payload = parsed_event.payload
        if parsed_event.is_malformed or not isinstance(payload, dict):
            return event_type not in {
                "response.created",
                "response.in_progress",
                "response.failed",
                "response.incomplete",
                "response.output_item.added",
                "response.output_item.done",
                "response.content_part.added",
                "response.content_part.done",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
            }

        response = payload.get("response")
        if event_type in {
            "response.created",
            "response.in_progress",
            "response.failed",
            "response.incomplete",
            "response.completed",
        }:
            if not isinstance(response, dict):
                return event_type not in {
                    "response.created",
                    "response.in_progress",
                    "response.failed",
                    "response.incomplete",
                    "response.completed",
                }
            response_id = response.get("id")
            if not isinstance(response_id, str) or not response_id:
                return False
            if self._response_id is not None and response_id != self._response_id:
                return False
            if (
                event_type in {"response.created", "response.in_progress"}
                and record
            ):
                self._response_id = response_id
            if event_type == "response.failed":
                output = response.get("output")
                if not isinstance(output, list):
                    return False
                for output_index, item in enumerate(output):
                    known_item = self._output_items.get(output_index)
                    if known_item is not None and (
                        not isinstance(item, dict)
                        or (item.get("id"), item.get("type")) != known_item
                    ):
                        return False
                    if known_item is None and self._output_items:
                        return False
            return True

        if event_type not in {
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
        }:
            return True
        if self._response_id is None:
            return False

        output_index = payload.get("output_index")
        if not self._is_nonnegative_index(output_index):
            return False
        if event_type == "response.output_item.added":
            item = payload.get("item")
            if not isinstance(item, dict):
                return False
            item_id = item.get("id")
            item_type = item.get("type")
            if not isinstance(item_id, str) or not item_id:
                return False
            if not isinstance(item_type, str) or not item_type:
                return False
            existing = self._output_items.get(output_index)
            if existing is not None:
                return False
            if any(existing_id == item_id for existing_id, _ in self._output_items.values()):
                return False
            if record:
                self._output_items[output_index] = (item_id, item_type)
            return True

        item_id = payload.get("item_id")
        item = payload.get("item") if event_type == "response.output_item.done" else None
        if item is not None:
            if not isinstance(item, dict):
                return False
            item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            return False
        associated_item = self._output_items.get(output_index)
        if associated_item is None or associated_item[0] != item_id:
            return False
        if item is not None and item.get("type") != associated_item[1]:
            return False

        expected_item_type: Optional[str] = None
        if event_type in {
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
        }:
            expected_item_type = "reasoning"
        elif event_type in {
            "response.content_part.added",
            "response.content_part.done",
        }:
            part = payload.get("part")
            part_type = part.get("type") if isinstance(part, dict) else None
            expected_item_type = (
                "reasoning" if part_type == "reasoning_text" else "message"
            )
        if expected_item_type is not None and associated_item[1] != expected_item_type:
            return False
        return True

    def feed(
        self,
        chunk: bytes,
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        if not chunk:
            return self.decision()
        self._pipeline.feed_until(
            chunk,
            inspect_event=self._consume_until_decisive,
        )
        return self.decision()

    def _consume_until_decisive(self, parsed_event: ParsedResponsesEvent) -> bool:
        self._consume_event(parsed_event)
        return self._first_terminal_decision is not None

    def finish(self) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        for parsed_event in self._pipeline.finish():
            self._consume_event(parsed_event)
        return self.decision()

    def decision(
        self,
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        if self._first_terminal_decision == "substantive":
            return "substantive", None, self._substantive_event
        if self._first_terminal_decision == "failed":
            return (
                "failed",
                self._first_error_payload,
                self._first_error_event,
            )
        if self._saw_lifecycle:
            return "lifecycle", None, None
        return "empty", None, None

    @property
    def commitment_reason(self) -> Optional[str]:
        return self._commitment_reason


class _PassThroughStreamLineAccumulator:
    """Incrementally decode raw stream bytes into non-empty SSE/log lines."""

    __slots__ = ("_decoder", "_pending", "lines")

    def __init__(self) -> None:
        # Incomplete UTF-8 at a stream boundary is a mid-codepoint split.
        # Drop the leftover truncated sequence instead of raising on finish().
        # ``errors="ignore"`` is not enough on every CPython incremental
        # decoder: ``final=True`` can still raise UnicodeDecodeError for a
        # truncated multi-byte sequence (T-4).
        self._decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        self._pending = ""
        self.lines: List[str] = []

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        try:
            decoded = self._decoder.decode(chunk)
        except UnicodeDecodeError:
            decoded = chunk.decode("utf-8", errors="ignore")
            self._decoder.reset()
        self._pending += decoded
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self.lines.append(stripped)

    def finish(self) -> List[str]:
        try:
            tail = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            # T-4: peeked Responses SSE can end mid-codepoint. Treat the
            # leftover truncated sequence as incomplete text, not a crash.
            self._decoder.reset()
            tail = ""
        if tail:
            self._pending += tail
        stripped = self._pending.strip()
        if stripped:
            self.lines.append(stripped)
        self._pending = ""
        return self.lines

    def has_pending_frame(self) -> bool:
        buffered_bytes, _ = self._decoder.getstate()
        return bool(self._pending or buffered_bytes)


def _strip_chunk_line(raw_line: str) -> str:
    return raw_line.strip()
_PROVIDER_EXECUTABLE_TOOL_TYPES_HINT = frozenset({
    "mcp",
    "web_search",
    "web_search_preview",
    "web_search_call",
    "code_interpreter",
    "computer",
    "image_generation",
    "file_search",
})


class PassThroughStreamingHandler:
    _AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV = "AAWM_STREAM_SUMMARY_FIRST_FINALIZE"

    _ANTHROPIC_RATE_LIMIT_HEADER_PREFIXES = (
        "anthropic-ratelimit-",
        "x-ratelimit-",
    )
    _ANTHROPIC_RATE_LIMIT_HEADER_NAMES = {
        "retry-after",
    }
    _CODEX_RATE_LIMIT_HEADER_PREFIXES = (
        "x-codex-",
    )
    _CODEX_RATE_LIMIT_HEADER_NAMES = {
        "x-oai-request-id",
    }
    _XAI_OAUTH_RATE_LIMIT_HEADER_PREFIXES = (
        "x-ratelimit-",
    )
    _XAI_OAUTH_RATE_LIMIT_HEADER_NAMES = {
        "retry-after",
    }
    _XAI_OAUTH_RATE_LIMIT_HEADERS_KEY = "xai_oauth_response_headers"
    _XAI_GROK_OIDC_RATE_LIMIT_HEADERS_KEY = "xai_grok_oidc_response_headers"
    _XAI_OAUTH_RATE_LIMIT_SOURCE = "xai_oauth_response_headers"
    _XAI_GROK_OIDC_RATE_LIMIT_SOURCE = "xai_grok_oidc_response_headers"
    _CLEAN_EOF_INCOMPLETE_REASON = "upstream_stream_ended_without_terminal_event"


    _RESPONSES_TERMINAL_EVENTS = {
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
    _RESPONSES_NON_SUBSTANTIVE_LIFECYCLE_EVENTS = (
        _RESPONSES_NON_SUBSTANTIVE_LIFECYCLE_EVENTS
    )

    @staticmethod
    def _is_openai_capacity_retry_policy_enabled() -> bool:
        policy_value = os.getenv("AAWM_OPENAI_CAPACITY_RETRY_ENABLED")
        return not (
            policy_value is not None
            and policy_value.strip().lower() in {"0", "false", "no", "off"}
        )

    @staticmethod
    def _is_openai_responses_stream(
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> bool:
        parsed_url = urlparse(url_route)
        provider_name = (
            custom_llm_provider.value
            if isinstance(custom_llm_provider, litellm.LlmProviders)
            else custom_llm_provider
        )
        return (
            endpoint_type == EndpointType.OPENAI
            and provider_name == "openai"
            and parsed_url.hostname in {"api.openai.com", "chatgpt.com"}
            and OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        )

    @staticmethod
    def _is_xai_responses_route(
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> bool:
        provider_name = (
            custom_llm_provider.value
            if isinstance(custom_llm_provider, litellm.LlmProviders)
            else custom_llm_provider
        )
        return (
            endpoint_type == EndpointType.OPENAI
            and provider_name == litellm.LlmProviders.XAI.value
            and OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        )

    @staticmethod
    def _is_responses_wire_owned_stream(
        *,
        response: Any,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> bool:
        """Recognize xAI streams owned by the shared final-wire coordinator."""
        if not PassThroughStreamingHandler._is_xai_responses_route(
            endpoint_type=endpoint_type,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
        ):
            return False
        extensions = getattr(response, "extensions", None)
        return isinstance(extensions, dict) and (
            extensions.get("aawm_responses_wire_owned") is True
        )

    @staticmethod
    def _stamp_encrypted_reasoning_in_responses_sse_chunk(
        chunk: bytes,
        *,
        request_body: Optional[Dict[str, Any]],
        custom_llm_provider: Optional[str],
        events: Optional[List[ParsedResponsesEvent]] = None,
    ) -> bytes:
        """OPENAI-006: delegate SSE provenance stamping to the helper module."""
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            stamp_encrypted_reasoning_in_responses_sse_chunk,
        )

        return stamp_encrypted_reasoning_in_responses_sse_chunk(
            chunk,
            request_body=request_body,
            custom_llm_provider=custom_llm_provider,
            events=events,
        )

    @staticmethod
    def _is_done_chunk(chunk: bytes) -> bool:
        try:
            decoded = chunk.decode("utf-8", errors="ignore")
        except (AttributeError, UnicodeDecodeError):
            return False
        for raw_line in decoded.splitlines():
            if _strip_chunk_line(raw_line) == "data: [DONE]":
                return True
        return False

    @staticmethod
    def _split_trailing_done_chunk(chunk: bytes) -> tuple[bytes, bytes]:
        def _split_complete_done_suffix(value: bytes) -> tuple[bytes, bytes]:
            lines = value.splitlines(keepends=True)
            first_done_line: Optional[int] = None
            for index in range(len(lines) - 1, -1, -1):
                stripped = lines[index].strip()
                if not stripped:
                    continue
                if stripped == b"data: [DONE]":
                    first_done_line = index
                    continue
                break
            if first_done_line is None:
                return value, b""
            return (
                b"".join(lines[:first_done_line]),
                b"".join(lines[first_done_line:]),
            )

        prefix, done_suffix = _split_complete_done_suffix(chunk)
        if done_suffix:
            return prefix, done_suffix

        done_marker = b"data: [DONE]"
        for marker_length in range(
            min(len(chunk), len(done_marker) - 1),
            0,
            -1,
        ):
            if not chunk.endswith(done_marker[:marker_length]):
                continue
            marker_start = len(chunk) - marker_length
            if marker_start > 0 and chunk[marker_start - 1 : marker_start] not in {
                b"\n",
                b"\r",
            }:
                continue
            prefix, prior_done_suffix = _split_complete_done_suffix(
                chunk[:marker_start]
            )
            return prefix, prior_done_suffix + chunk[marker_start:]

        return chunk, b""

    @staticmethod
    def _split_responses_sse_event_buffer(
        chunk: bytes,
    ) -> tuple[bytes, bytes]:
        """Return complete SSE events and retain the trailing partial event."""
        if not chunk:
            return b"", b""

        lf_boundary = chunk.rfind(b"\n\n")
        crlf_boundary = chunk.rfind(b"\r\n\r\n")
        if crlf_boundary >= lf_boundary:
            boundary_start, boundary_length = crlf_boundary, 4
        else:
            boundary_start, boundary_length = lf_boundary, 2
        if boundary_start < 0:
            return b"", chunk
        boundary_end = boundary_start + boundary_length
        return chunk[:boundary_end], chunk[boundary_end:]

    @staticmethod
    def _partition_responses_emissions_by_count(
        prefix: bytes,
        suffix: bytes,
        emissions: List[Any],
    ) -> tuple[List[Any], List[Any]]:
        """Keep one ordered emission per frame after a done-marker split."""
        del suffix
        if not emissions:
            return [], []
        prefix_count = 0
        remaining = bytes(prefix)
        for emission in emissions:
            wire = getattr(emission, "wire_bytes", b"") or b""
            if not wire or not remaining.startswith(wire):
                break
            remaining = remaining[len(wire):]
            prefix_count += 1
        return emissions[:prefix_count], emissions[prefix_count:]

    @staticmethod
    def _chunk_lines(chunks: List[bytes]) -> List[str]:
        accumulator = _PassThroughStreamLineAccumulator()
        for chunk in chunks:
            accumulator.feed(chunk)
        return accumulator.finish()

    @staticmethod
    def _responses_lines_have_terminal_event(lines: List[str]) -> bool:
        for raw_line in lines:
            line = _strip_chunk_line(raw_line)
            event_type: Optional[str] = None
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload_text = line.removeprefix("data:").strip()
                if payload_text == "[DONE]":
                    continue
                try:
                    payload = json.loads(payload_text)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict):
                    payload_type = payload.get("type")
                    if isinstance(payload_type, str):
                        event_type = payload_type
            if event_type in PassThroughStreamingHandler._RESPONSES_TERMINAL_EVENTS:
                return True
        return False

    @staticmethod
    def _has_only_fields(value: Dict[str, Any], fields: set[str]) -> bool:
        return set(value).issubset(fields)

    @staticmethod
    def _is_nonnegative_index(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _is_nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value)

    @staticmethod
    def _is_safe_reasoning_item(
        item: Any,
        *,
        allowed_statuses: set[Optional[str]],
    ) -> bool:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            return False
        if not PassThroughStreamingHandler._is_nonempty_string(item.get("id")):
            return False
        if item.get("status") not in allowed_statuses:
            return False
        if not PassThroughStreamingHandler._has_only_fields(
            item,
            {"id", "type", "status", "summary", "content", "encrypted_content"},
        ):
            return False
        summary = item.get("summary", [])
        if not isinstance(summary, list) or not all(
            isinstance(part, dict)
            and set(part) == {"type", "text"}
            and part.get("type") == "summary_text"
            and isinstance(part.get("text"), str)
            for part in summary
        ):
            return False
        content = item.get("content", [])
        if not isinstance(content, list) or not all(
            isinstance(part, dict)
            and set(part) == {"type", "text"}
            and part.get("type") == "reasoning_text"
            and isinstance(part.get("text"), str)
            for part in content
        ):
            return False
        encrypted_content = item.get("encrypted_content")
        return encrypted_content is None or isinstance(encrypted_content, str)

    @staticmethod
    def _is_safe_output_item_added_shape(item: Any) -> bool:
        """Return True when *item* is one of the four permitted safe-private
        ``response.output_item.added`` shapes (OPENAI-080 closed set)."""
        if not isinstance(item, dict):
            return False
        item_type = item.get("type")
        if item_type not in _RESPONSES_SAFE_ITEM_ADDED_TYPES:
            return False

        if item_type == "message":
            return (
                PassThroughStreamingHandler._has_only_fields(
                    item, {"id", "type", "role", "status", "content"}
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("id")
                )
                and item.get("role") == "assistant"
                and item.get("status") == "in_progress"
                and item.get("content") == []
            )

        if item_type == "function_call":
            return (
                PassThroughStreamingHandler._has_only_fields(
                    item,
                    {
                        "id", "type", "status", "call_id", "name", "arguments",
                    },
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("id")
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("call_id")
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("name")
                )
                and item.get("arguments") == ""
                and item.get("status") in (None, "in_progress")
            )

        if item_type == "custom_tool_call":
            return (
                PassThroughStreamingHandler._has_only_fields(
                    item,
                    {"id", "type", "status", "call_id", "name", "input"},
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("id")
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("call_id")
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    item.get("name")
                )
                and item.get("input") == ""
                and item.get("status") in (None, "in_progress")
            )

        if item_type == "reasoning":
            return PassThroughStreamingHandler._is_safe_reasoning_item(
                item,
                allowed_statuses={None, "in_progress"},
            )

        return False

    @staticmethod
    def _is_safe_content_part_added_shape(part: Any) -> bool:
        """Return True when *part* is one of the three permitted safe-private
        ``response.content_part.added`` shapes (OPENAI-080 closed set)."""
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type not in _RESPONSES_SAFE_CONTENT_PART_ADDED_TYPES:
            return False

        if part_type == "output_text":
            if not PassThroughStreamingHandler._has_only_fields(
                part, {"type", "text", "annotations", "logprobs"}
            ) or part.get("text") != "":
                return False
            annotations = part.get("annotations")
            if annotations not in (None, []) or (
                annotations is not None and not isinstance(annotations, list)
            ):
                return False
            logprobs = part.get("logprobs")
            if logprobs not in (None, []) or (
                logprobs is not None and not isinstance(logprobs, list)
            ):
                return False
            return True

        if part_type == "refusal":
            return (
                PassThroughStreamingHandler._has_only_fields(
                    part, {"type", "refusal"}
                )
                and part.get("refusal") == ""
            )

        if part_type == "reasoning_text":
            return (
                PassThroughStreamingHandler._has_only_fields(
                    part, {"type", "text"}
                )
                and isinstance(part.get("text"), str)
            )

        return False

    @staticmethod
    def _is_responses_non_substantive_lifecycle_event(
        event_type: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Frozen Oracle allowlist: return True only when the exact event name and
        full semantic shape are unconditionally safe to discard on a precommit failure.

        This is the OPENAI-080 closed set.  Any unrecognised event, unexpected field,
        or schema conflict commits the attempt.
        """
        if event_type is None or not isinstance(payload, dict):
            return False
        if payload.get("type") != event_type:
            return False
        sequence_number = payload.get("sequence_number")
        if sequence_number is not None and not (
            PassThroughStreamingHandler._is_nonnegative_index(sequence_number)
        ):
            return False

        if event_type in ("response.created", "response.in_progress"):
            response_payload = payload.get("response")
            if not isinstance(response_payload, dict):
                return False
            if not PassThroughStreamingHandler._has_only_fields(
                payload, {"type", "sequence_number", "response"}
            ):
                return False
            response_fields = {
                "id", "created_at", "error", "incomplete_details",
                "instructions", "metadata", "model", "object", "output",
                "parallel_tool_calls", "temperature", "tool_choice", "tools",
                "top_p", "max_output_tokens", "previous_response_id",
                "reasoning", "status", "text", "truncation", "usage", "user",
                "store", "background", "billing", "context_management",
                "conversation", "max_tool_calls", "prompt",
                "prompt_cache_key", "prompt_cache_retention",
                "safety_identifier", "service_tier", "top_logprobs",
            }
            if not PassThroughStreamingHandler._has_only_fields(
                response_payload, response_fields
            ):
                return False
            if not PassThroughStreamingHandler._is_nonempty_string(
                response_payload.get("id")
            ):
                return False
            if response_payload.get("status") != "in_progress":
                return False
            if response_payload.get("error") is not None or response_payload.get(
                "incomplete_details"
            ) is not None:
                return False
            output = response_payload.get("output")
            if not isinstance(output, list) or output:
                return False
            created_at = response_payload.get("created_at")
            if created_at is not None and not (
                PassThroughStreamingHandler._is_nonnegative_index(created_at)
            ):
                return False
            if response_payload.get("object") not in (None, "response"):
                return False
            model = response_payload.get("model")
            if model is not None and not isinstance(model, str):
                return False
            return True

        if event_type == "response.output_item.added":
            return (
                PassThroughStreamingHandler._has_only_fields(
                    payload, {"type", "sequence_number", "output_index", "item"}
                )
                and PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get("output_index")
                )
                and PassThroughStreamingHandler._is_safe_output_item_added_shape(
                    payload.get("item")
                )
            )

        if event_type == "response.output_item.done":
            if not PassThroughStreamingHandler._has_only_fields(
                payload, {"type", "sequence_number", "output_index", "item"}
            ) or not PassThroughStreamingHandler._is_nonnegative_index(
                payload.get("output_index")
            ):
                return False
            item = payload.get("item")
            return PassThroughStreamingHandler._is_safe_reasoning_item(
                item,
                allowed_statuses={"completed"},
            )

        if event_type in {
            "response.content_part.added",
            "response.content_part.done",
        }:
            if not PassThroughStreamingHandler._has_only_fields(
                payload,
                {
                    "type", "sequence_number", "item_id", "output_index",
                    "content_index", "part",
                },
            ):
                return False
            if not (
                PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get("output_index")
                )
                and PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get("content_index")
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    payload.get("item_id")
                )
            ):
                return False
            part = payload.get("part")
            if event_type == "response.content_part.added":
                return PassThroughStreamingHandler._is_safe_content_part_added_shape(
                    part
                )
            return (
                isinstance(part, dict)
                and set(part) == {"type", "text"}
                and part.get("type") == "reasoning_text"
                and isinstance(part.get("text"), str)
            )

        if event_type in {
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            if not PassThroughStreamingHandler._has_only_fields(
                payload,
                {
                    "type", "sequence_number", "item_id", "output_index",
                    "summary_index", "part",
                },
            ):
                return False
            if not (
                PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get("output_index")
                )
                and PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get("summary_index")
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    payload.get("item_id")
                )
            ):
                return False
            part = payload.get("part")
            if (
                not isinstance(part, dict)
                or set(part) != {"type", "text"}
                or part.get("type") != "summary_text"
                or not isinstance(part.get("text"), str)
            ):
                return False
            return True

        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
        }:
            is_summary = "summary" in event_type
            value_field = "delta" if event_type.endswith(".delta") else "text"
            fields = {
                "type", "sequence_number", "item_id", "output_index",
                "summary_index" if is_summary else "content_index",
                value_field,
            }
            if not PassThroughStreamingHandler._has_only_fields(payload, fields):
                return False
            index_field = "summary_index" if is_summary else "content_index"
            if not (
                PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get("output_index")
                )
                and PassThroughStreamingHandler._is_nonnegative_index(
                    payload.get(index_field)
                )
                and PassThroughStreamingHandler._is_nonempty_string(
                    payload.get("item_id")
                )
                and isinstance(payload.get(value_field), str)
            ):
                return False
            return True

        return False

    @staticmethod
    def _is_responses_substantive_event(
        event_type: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Normal-delivery taxonomy: return True when the event carries
        client-visible content (text, tool data, media, effects).

        Uses the original prefix-based classification, separate from the
        OPENAI-080 precommit replay-commit allowlist.  Callers using
        precommit replay decisions must go through
        _inspect_responses_events or
        _is_responses_non_substantive_lifecycle_event directly.
        """
        if event_type is None or event_type == "[DONE]":
            return False
        if event_type in _RESPONSES_PRE_COMMIT_FAILURE_EVENTS:
            # Failure events carry their own classification; they are
            # not substantive for delivery-tracking purposes.
            return False
        if event_type in PassThroughStreamingHandler._RESPONSES_TERMINAL_EVENTS:
            # response.completed is a substantive delivery marker;
            # response.failed / response.incomplete are not.
            return event_type == "response.completed"
        if event_type in _RESPONSES_NON_SUBSTANTIVE_LIFECYCLE_EVENTS:
            # Lifecycle events that are unconditionally safe: check the
            # OPENAI-080 frozen-shape allowlist for shape-level safety.
            return not PassThroughStreamingHandler._is_responses_non_substantive_lifecycle_event(
                event_type,
                payload,
            )
        if event_type.startswith(_RESPONSES_SUBSTANTIVE_EVENT_PREFIXES):
            return True
        return event_type.startswith("response.")

    @staticmethod
    def _extract_responses_stream_error_payload(
        event_type: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        if event_type == "error":
            nested = payload.get("error")
            if isinstance(nested, dict):
                return nested
            return payload
        if event_type == "response.failed":
            response_payload = payload.get("response")
            if isinstance(response_payload, dict):
                nested = response_payload.get("error")
                if isinstance(nested, dict):
                    return {**nested, "response": response_payload}
                if response_payload.get("status") == "failed":
                    return {"response": response_payload, "code": "response.failed"}
            nested = payload.get("error")
            if isinstance(nested, dict):
                return nested
            return payload
        response_payload = payload.get("response")
        if isinstance(response_payload, dict) and (
            response_payload.get("status") == "failed"
            or response_payload.get("error") is not None
        ):
            nested = response_payload.get("error")
            if isinstance(nested, dict):
                return {**nested, "response": response_payload}
            return {"response": response_payload, "code": "response.failed"}
        return None

    @staticmethod
    def _extract_responses_stream_error_fields(
        error_payload: Optional[Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if not isinstance(error_payload, dict):
            return None, None, None

        candidates: List[Dict[str, Any]] = [error_payload]
        nested_error = error_payload.get("error")
        if isinstance(nested_error, dict):
            candidates.append(nested_error)
        response_payload = error_payload.get("response")
        if isinstance(response_payload, dict):
            response_error = response_payload.get("error")
            if isinstance(response_error, dict):
                candidates.append(response_error)

        code: Optional[str] = None
        error_type: Optional[str] = None
        message: Optional[str] = None
        for candidate in candidates:
            if code is None:
                raw_code = candidate.get("code")
                if isinstance(raw_code, str) and raw_code.strip():
                    code = raw_code.strip()
            if error_type is None:
                raw_type = candidate.get("type")
                if isinstance(raw_type, str) and raw_type.strip():
                    error_type = raw_type.strip()
            if message is None:
                raw_message = candidate.get("message")
                if isinstance(raw_message, str) and raw_message.strip():
                    message = raw_message.strip()
            if code is not None and error_type is not None and message is not None:
                break
        return code, error_type, message

    @staticmethod
    def _is_responses_unpersisted_item_not_found_payload(
        error_payload: Optional[Dict[str, Any]],
    ) -> bool:
        code, error_type, message = (
            PassThroughStreamingHandler._extract_responses_stream_error_fields(
                error_payload
            )
        )
        if not isinstance(message, str):
            return False
        if error_type not in {"invalid_request_error", "error", None}:
            return False
        if (
            error_type != "invalid_request_error"
            and code != "invalid_request_error"
        ):
            return False
        from .aawm_alias_routing.error_signals import (
            _OPENAI_RESPONSES_UNPERSISTED_ITEM_NOT_FOUND_RE,
        )

        return (
            _OPENAI_RESPONSES_UNPERSISTED_ITEM_NOT_FOUND_RE.fullmatch(message)
            is not None
        )

    @staticmethod
    def _is_openai_response_protection_unavailable_payload(
        error_payload: Optional[Dict[str, Any]],
    ) -> bool:
        """Match the observed OpenAI protection-unavailable provider shape."""
        if not isinstance(error_payload, dict):
            return False
        direct_error = error_payload.get("error")
        candidates: list[Any] = [error_payload]
        if isinstance(direct_error, dict):
            candidates.append(direct_error)
        nested_response = error_payload.get("response")
        if isinstance(nested_response, dict):
            nested_error = nested_response.get("error")
            if isinstance(nested_error, dict):
                candidates.append(nested_error)
        for candidate in candidates:
            message = candidate.get("message")
            if (
                isinstance(message, str)
                and message.strip().lower()
                == _OPENAI_RESPONSE_PROTECTION_UNAVAILABLE_ERROR
                and candidate.get("type") == "internal_error"
                and candidate.get("code") == "error"
            ):
                return True
        return False

    @staticmethod
    def _classify_responses_pre_commit_error(
        error_payload: Optional[Dict[str, Any]],
        *,
        openai_alpha_capacity_retry_enabled: bool = False,
    ) -> tuple[str, str, bool]:
        error_code, error_type, error_message = (
            PassThroughStreamingHandler._extract_responses_stream_error_fields(
                error_payload
            )
        )
        if PassThroughStreamingHandler._is_openai_response_protection_unavailable_payload(
            error_payload
        ):
            return (
                "openai_response_protection_unavailable",
                "transient_upstream",
                True,
            )
        if error_code == "token_invalidated":
            return "token_invalidated", "token_invalidated", False
        if PassThroughStreamingHandler._is_responses_unpersisted_item_not_found_payload(
            error_payload
        ):
            return (
                "openai_responses_unpersisted_item_not_found",
                "openai_responses_unpersisted_item_not_found",
                False,
            )

        tokens: set[str] = set()
        if isinstance(error_payload, dict):
            for key in ("code", "type", "param"):
                value = error_payload.get(key)
                if isinstance(value, str) and value.strip():
                    tokens.add(value.strip().lower())
            message = error_payload.get("message")
            if isinstance(message, str) and message.strip():
                tokens.add(message.strip().lower())
            response_payload = error_payload.get("response")
            if isinstance(response_payload, dict):
                nested = response_payload.get("error")
                if isinstance(nested, dict):
                    for key in ("code", "type"):
                        value = nested.get(key)
                        if isinstance(value, str) and value.strip():
                            tokens.add(value.strip().lower())
                    nested_message = nested.get("message")
                    if isinstance(nested_message, str) and nested_message.strip():
                        tokens.add(nested_message.strip().lower())
        for value in (error_code, error_type, error_message):
            if isinstance(value, str) and value.strip():
                tokens.add(value.strip().lower())
        joined = " ".join(sorted(tokens))
        if any(
            marker in joined
            for marker in (
                "usage_limit_reached",
                "usage limit",
                "quota exceeded",
                "quota exhausted",
                "insufficient_quota",
                "quota_limit",
                "weekly limit",
            )
        ):
            return "usage_limit_reached", "usage_limit_reached", False
        if error_code in {
            "authentication_error",
            "authorization_error",
            "permission_error",
            "invalid_request_error",
            "insufficient_quota",
            "usage_limit_reached",
        } or error_type in {
            "authentication_error",
            "authorization_error",
            "permission_error",
            "invalid_request_error",
            "insufficient_quota",
            "usage_limit_reached",
        }:
            return "provider_terminal_error", "provider_terminal_error", False
        if openai_alpha_capacity_retry_enabled and (
            error_code in {
                "rate_limit_exceeded",
                "rate_limited",
                "too_many_requests",
            }
            or error_type in {"rate_limit_error", "rate_limited"}
            or any(
                marker in joined
                for marker in (
                    "rate_limit",
                    "rate limit",
                    "too many requests",
                    "429",
                )
            )
        ):
            return "server_overloaded", "transient_capacity", True
        if (
            error_code in {"server_is_overloaded", "capacity_exhausted"}
            and error_type not in {
                "authentication_error",
                "authorization_error",
                "permission_error",
                "invalid_request_error",
                "token_invalidated",
                "insufficient_quota",
                "usage_limit_reached",
            }
        ):
            return "server_overloaded", "transient_capacity", True
        if any(
            marker in joined
            for marker in (
                "server_overloaded",
                "overloaded_error",
                "model is overloaded",
                "upstream_overloaded",
                "high demand",
                "model_capacity_exhausted",
                "selected model is at capacity",
                "model is at capacity",
                "model at capacity",
                "upstream busy",
            )
        ):
            return "server_overloaded", "transient_capacity", True
        if (
            {
                "server_error",
                "internal_server_error",
                "internal_error",
                "api_error",
                "upstream_error",
                "service_unavailable",
            }
            & tokens
            or "server error" in joined
            or "internal server error" in joined
        ):
            return "upstream_transient_internal", "transient_upstream", True
        return "provider_terminal_error", "provider_terminal_error", False

    @staticmethod
    def _inspect_responses_pre_commit_chunks(
        chunks: List[bytes],
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """OPENAI-080: return (decision, error_payload, event_type, commitment_reason)."""
        buffered = b"".join(chunks)
        pipeline = ResponsesEventPipeline(
            max_raw_events=0,
        )
        events = pipeline.feed(buffered)
        if pipeline.buffered_event_bytes:
            events.extend(pipeline.finish())
        if not events:
            return "empty", None, None, None
        parser = _ResponsesPreCommitSSEParser(
            _ResponsesPreCommitPrefixStore(0),
            pipeline,
        )
        for event in events:
            parser._consume_event(event)
        decision = parser.decision()
        parsed_reason = parser.commitment_reason
        effective_reason = parsed_reason or "unknown_semantics"
        return decision[0], decision[1], decision[2], effective_reason


    @staticmethod
    def _inspect_failure_snapshot_for_hidden_output(
        payload: Optional[Dict[str, Any]],
        *,
        event_type: Optional[str] = None,
    ) -> Optional[str]:
        """Return the commitment reason when *payload* carries output items
        hidden inside a terminal or failure snapshot envelope (OPENAI-080)."""
        if not isinstance(payload, dict):
            return (
                "unknown_semantics"
                if event_type == "response.failed"
                else None
            )
        response_payload = payload.get("response")
        if not isinstance(response_payload, dict):
            return (
                "unknown_semantics"
                if event_type == "response.failed"
                else None
            )
        output = response_payload.get("output")
        if not isinstance(output, list):
            return "unknown_semantics"
        if not output:
            return None
        for item in output:
            if not isinstance(item, dict):
                return "unknown_semantics"
            item_type = item.get("type")
            if not isinstance(item_type, str):
                # Unknown type in snapshot: conservative commit
                return "unknown_semantics"
            if item_type == "message":
                # Any message item in a snapshot commits — even empty content
                # represents a structural completion.
                return "assistant_output"
            if item_type in ("function_call", "custom_tool_call"):
                # Tool items in a failure/terminal snapshot always commit,
                # even with empty arguments — structural completion evidence.
                return "client_tool_payload"
            if item_type in _PROVIDER_EXECUTABLE_TOOL_TYPES_HINT:
                return "provider_execution_possible"
            # Unknown item type in snapshot: conservative commit
            return "unknown_semantics"
        return None

    @staticmethod
    def _inspect_responses_events(
        parsed_event: ParsedResponsesEvent,
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """OPENAI-080: evaluate replay safety before failure classification.

        Returns (decision, error_payload, event_type, commitment_reason).
        commitment_reason is None for safe-private (lifecycle) events and set
        for every committing event.  The caller must set the monotonic replay
        latch *before* yielding committed output.
        """
        if parsed_event.is_malformed or parsed_event.is_done_marker:
            if parsed_event.is_done_marker:
                return "substantive", None, "[DONE]", "terminal_marker"
            # A malformed frame must veto replay; do not forward the bad bytes.
            return "substantive", None, None, "unknown_semantics"

        event_type = parsed_event.event_type
        payload = parsed_event.payload

        # --- Step 1: check if this event commits the attempt ---
        commitment_reason: Optional[str] = None

        if event_type in (None, ""):
            # Empty event with no type: treat as unknown semantics
            return "substantive", None, event_type, "unknown_semantics"

        # [DONE] is handled above via is_done_marker

        # Unknown / unrecognised event -> commit
        if event_type == "response.incomplete":
            commitment_reason = "unknown_semantics"
        elif event_type not in _RECOGNIZED_PROVIDER_TERMINAL_EVENTS and not (
            PassThroughStreamingHandler._is_responses_non_substantive_lifecycle_event(
                event_type, payload
            )
        ):
            # This event is not in the safe allowlist.  Classify the reason.
            if event_type == "response.completed":
                commitment_reason = "successful_terminal"
            elif event_type and (
                event_type.startswith("response.output_text.")
                or event_type.startswith("response.refusal.")
                or event_type.startswith("response.output_item.")
            ):
                # Inspect output items for assistant output vs client tool
                item = payload.get("item") if isinstance(payload, dict) else None
                if isinstance(item, dict) and item.get("type") == "message":
                    commitment_reason = "assistant_output"
                elif (
                    event_type.startswith("response.output_text.")
                    or event_type.startswith("response.refusal.")
                ):
                    commitment_reason = "assistant_output"
                elif event_type == "response.output_item.done":
                    commitment_reason = "assistant_output"
                else:
                    commitment_reason = "assistant_output"
            elif event_type and (
                event_type.startswith("response.function_call.")
                or event_type.startswith("response.custom_tool_call.")
            ):
                commitment_reason = "client_tool_payload"
            elif event_type and event_type.startswith("response.mcp_call."):
                commitment_reason = "provider_execution_possible"
            elif event_type and (
                event_type.startswith("response.web_search.")
                or event_type.startswith("response.file_search.")
                or event_type.startswith("response.code_interpreter.")
                or event_type.startswith("response.computer.")
                or event_type.startswith("response.image_generation.")
            ):
                commitment_reason = "provider_execution_possible"
            elif event_type and event_type.startswith("response."):
                # Any other response.* event (e.g. response.apply_patch,
                # response.local_shell, response.shell, response.reasoning
                # items that aren't in the safe allowlist, etc.)
                commitment_reason = "unknown_semantics"
            else:
                commitment_reason = "unknown_semantics"

        # --- Step 2: inspect semantic payload for hidden commits ---
        # A failure envelope must not hide text, tool data, or possible effects.
        hidden_reason = (
            PassThroughStreamingHandler._inspect_failure_snapshot_for_hidden_output(
                payload,
                event_type=event_type,
            )
        )
        if hidden_reason is not None:
            commitment_reason = hidden_reason

        # --- Step 3: extract error payload ---
        error_payload = (
            PassThroughStreamingHandler._extract_responses_stream_error_payload(
                event_type,
                payload,
            )
        )

        # --- Step 4: return decision ---
        if error_payload is not None or event_type in _RESPONSES_PRE_COMMIT_FAILURE_EVENTS:
            return "failed", error_payload or payload, event_type, commitment_reason

        if commitment_reason is not None:
            return "substantive", None, event_type, commitment_reason

        # Safe private event
        return "lifecycle", None, None, None

    @staticmethod
    def _iter_responses_sse_events(
        lines: List[str],
    ) -> List[tuple[Optional[str], Optional[Dict[str, Any]]]]:
        pipeline = ResponsesEventPipeline()
        text = "\n\n".join(lines)
        encoded = text.encode("utf-8") if text else b""
        if not text.endswith("\n\n"):
            encoded += b"\n\n"
        events = pipeline.feed(encoded)
        events.extend(pipeline.finish())
        converted: List[tuple[Optional[str], Optional[Dict[str, Any]]]] = []
        for parsed_event in events:
            if parsed_event.is_malformed or parsed_event.is_done_marker:
                event_type: Optional[str] = None
                payload: Optional[Dict[str, Any]] = None
            else:
                event_type = parsed_event.event_type
                payload = parsed_event.payload
            if event_type is None and payload is None:
                continue
            converted.append((event_type, payload))
        return converted

    @staticmethod
    def _build_responses_pre_commit_failure(
        *,
        error_payload: Optional[Dict[str, Any]],
        event_type: Optional[str],
        pre_commit_retry_exhausted: bool = False,
        retry_after_seconds: float = RESPONSES_PRE_COMMIT_TRANSIENT_RETRY_WAIT_SECONDS,
        openai_alpha_capacity_retry_enabled: bool = False,
        provider_returned: bool = True,
        failure_origin: Optional[str] = None,
    ) -> ResponsesStreamPreCommitFailure:
        error_class, classification, retryable = (
            PassThroughStreamingHandler._classify_responses_pre_commit_error(
                error_payload,
                openai_alpha_capacity_retry_enabled=openai_alpha_capacity_retry_enabled,
            )
        )
        extracted_code, extracted_type, extracted_message = (
            PassThroughStreamingHandler._extract_responses_stream_error_fields(
                error_payload
            )
        )
        sanitized_message = None
        error_code = None
        error_type = None
        status_code = None
        if isinstance(error_payload, dict):
            sanitized_message = (
                OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_error_for_logging(
                    error_payload
                )
            )
            raw_status_code = error_payload.get("status_code")
            if isinstance(raw_status_code, int):
                status_code = raw_status_code
            elif isinstance(raw_status_code, str) and raw_status_code.strip():
                try:
                    status_code = int(raw_status_code.strip())
                except ValueError:
                    status_code = None
        if extracted_code:
            error_code = extracted_code
        if extracted_type:
            error_type = extracted_type
        if extracted_message:
            sanitized_message = extracted_message
        if error_class == "token_invalidated":
            status_code = 401
        elif error_class == "openai_responses_unpersisted_item_not_found":
            status_code = 400
            error_type = "invalid_request_error"
            error_code = error_code or "invalid_request_error"
        elif error_class == "openai_response_protection_unavailable":
            status_code = 503
        elif error_class == "server_overloaded" and (
            error_code in {
                "rate_limit_exceeded",
                "rate_limited",
                "too_many_requests",
            }
            or error_type in {"rate_limit_error", "rate_limited"}
            or any(
                marker in " ".join(
                    value.lower()
                    for value in (error_code, error_type, extracted_message)
                    if isinstance(value, str)
                )
                for marker in ("rate_limit", "rate limit", "429", "too many requests")
            )
        ):
            status_code = 429
        if not sanitized_message:
            sanitized_message = classification
        if pre_commit_retry_exhausted and retryable:
            retryable = True
        return ResponsesStreamPreCommitFailure(
            error_class=error_class,
            classification=classification,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
            error_code=error_code or event_type,
            error_type=error_type or event_type,
            status_code=status_code,
            message=sanitized_message,
            pre_commit_retry_exhausted=pre_commit_retry_exhausted,
            error_payload=error_payload,
            provider_returned=provider_returned,
            failure_origin=failure_origin,
        )

    @staticmethod
    def _find_responses_stream_recovery_failure(
        lines: List[str],
    ) -> Optional[ResponsesStreamPreCommitFailure]:
        for event_type, payload in PassThroughStreamingHandler._iter_responses_sse_events(
            lines
        ):
            error_payload = (
                PassThroughStreamingHandler._extract_responses_stream_error_payload(
                    event_type,
                    payload,
                )
            )
            if error_payload is None:
                continue
            failure = PassThroughStreamingHandler._build_responses_pre_commit_failure(
                error_payload=error_payload,
                event_type=event_type,
            )
            if failure.error_class in _RESPONSES_STREAM_REDISPATCH_CLASSES:
                return failure
        return None

    @staticmethod
    async def peek_responses_pre_commit_stream(  # noqa: PLR0915
        response: httpx.Response,
        *,
        max_buffer_bytes: int = _RESPONSES_PRE_COMMIT_MAX_BUFFER_BYTES,
        openai_alpha_capacity_retry_enabled: bool = False,
        event_pipeline: Optional[ResponsesEventPipeline] = None,
        request: Any = None,
        provider_bound_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[httpx.Response, Optional[ResponsesStreamPreCommitFailure]]:
        """Hold safe-private Responses bytes until a commit or pre-SSE failure.

        OPENAI-080: applies the Oracle-frozen semantic allowlist to classify
        each event.  Sets the monotonic replay-commit latch on the request
        *before* yielding any committed output.  When *request* is supplied
        and a provider-executable tool body is detected, dispatch commits
        at ``provider_execution_possible`` regardless of observed events.
        """
        pipeline = event_pipeline or create_responses_event_pipeline(
            retain_events=True,
        )

        # OPENAI-080: when the effective body carries provider-executable
        # tools, lock the latch at dispatch so no internal resend can
        # proceed regardless of what events arrive.
        provider_tools_detected = (
            request is not None
            and responses_effective_provider_tools_allowed(provider_bound_body)
        )
        if provider_tools_detected:
            publish_responses_replay_commitment(
                request,
                reason="provider_execution_possible",
            )

        prefix = _ResponsesPreCommitPrefixStore(
            max_buffer_bytes,
            max_storage_bytes=pipeline.max_stream_bytes,
        )
        parser = _ResponsesPreCommitSSEParser(prefix, pipeline)
        iterator: Any = None
        retired = False

        async def _close_precommit_resources() -> None:
            nonlocal retired
            if retired:
                return
            retired = True
            prefix.close()
            close_iterator = getattr(iterator, "aclose", None)
            if callable(close_iterator):
                try:
                    result = close_iterator()
                    if inspect.isawaitable(result):
                        await wait_for("resource_cleanup", result)
                except Exception:
                    pass
            close_response = getattr(response, "aclose", None)
            if callable(close_response):
                try:
                    result = close_response()
                    if inspect.isawaitable(result):
                        await wait_for("resource_cleanup", result)
                except Exception:
                    pass

        def _bound_failure(exc: BaseException) -> ResponsesStreamPreCommitFailure:
            bound_failure = ResponsesBoundFailure.from_exception(exc)
            return PassThroughStreamingHandler._build_responses_pre_commit_failure(
                error_payload=bound_failure.error_payload,
                event_type=bound_failure.code,
                openai_alpha_capacity_retry_enabled=(
                    openai_alpha_capacity_retry_enabled
                ),
            )

        try:
            iterator = response.aiter_bytes()
            while True:
                try:
                    chunk = await wait_for(
                        "upstream_precommit_read", iterator.__anext__()
                    )
                except StopAsyncIteration:
                    iterator = None
                    break
                except httpx.ReadError as exc:
                    # Only this upstream read is a transport failure. Parser,
                    # prefix storage, and later remainder iteration stay raw.
                    read_failure = ResponsesStreamPreCommitFailure(
                        error_class="stream_interrupted",
                        classification="transient_stream_interruption",
                        retryable=True,
                        error_code="stream_read_error",
                        error_type="transport",
                        message=(
                            "OpenAI Responses precommit stream read failed "
                            "before a substantive event."
                        ),
                        provider_returned=False,
                        failure_origin="local_transport",
                    )
                    read_failure.__cause__ = exc
                    return (
                        _PrefixedHttpxByteStream(response, prefix, iterator),
                        read_failure,
                    )
                if not chunk:
                    continue
                chunk = bytes(chunk)
                event_type: Optional[str] = None
                try:
                    decision, error_payload, event_type = parser.feed(chunk)
                    prefix.append(chunk)
                except (
                    ResponsesBoundFailure,
                    ResponsesEventTooLarge,
                    ResponsesStreamTooLarge,
                ) as exc:
                    bound_failure = ResponsesBoundFailure.from_exception(exc)
                    accepted_bytes = max(0, bound_failure.accepted_bytes)
                    prior_safe = bool(
                        getattr(pipeline, "held_event_count", 0)
                        or bound_failure.accepted_events
                    )
                    # OPENAI-080 (R8): when the stream carried accepted safe
                    # events and then hits a size/event bound, commit with
                    # "event_limit" and preserve the prefix.  Only discard
                    # when there is no accepted prefix at all.
                    if prior_safe:
                        if accepted_bytes:
                            try:
                                prefix.append(chunk[:accepted_bytes])
                            except (
                                ResponsesBoundFailure,
                                ResponsesEventTooLarge,
                                ResponsesStreamTooLarge,
                            ):
                                await _close_precommit_resources()
                                return response, _bound_failure(exc)
                        prefixed = _PrefixedHttpxByteStream(
                            response,
                            prefix,
                            iterator,
                            event_pipeline=pipeline,
                            bound_failure=bound_failure,
                        )
                        pipeline.attach_to_response(prefixed)
                        _maybe_set_replay_latch(
                            request, "event_limit", event_type
                        )
                        return prefixed, None
                    await _close_precommit_resources()
                    return response, _bound_failure(exc)
                if decision == "failed":
                    # OPENAI-080: a failure event commits when it carries
                    # semantic payload (text, tool data, effects), or an
                    # earlier dispatch-time capability already closed replay.
                    commitment_reason = parser.commitment_reason or (
                        get_responses_replay_commitment_reason(request)
                        if request is not None
                        else None
                    )
                    if commitment_reason is not None:
                        _maybe_set_replay_latch(
                            request, commitment_reason, event_type
                        )
                        committed = _PrefixedHttpxByteStream(
                            response,
                            prefix,
                            iterator,
                            event_pipeline=pipeline,
                        )
                        pipeline.attach_to_response(committed)
                        return committed, None
                    return (
                        _PrefixedHttpxByteStream(response, prefix, iterator),
                        PassThroughStreamingHandler._build_responses_pre_commit_failure(
                            error_payload=error_payload,
                            event_type=event_type,
                            openai_alpha_capacity_retry_enabled=(
                                openai_alpha_capacity_retry_enabled
                            ),
                        ),
                    )
                if decision == "substantive":
                    prefixed = _PrefixedHttpxByteStream(
                        response,
                        prefix,
                        iterator,
                        event_pipeline=pipeline,
                    )
                    pipeline.attach_to_response(prefixed)
                    # Set latch before yielding committed stream
                    _maybe_set_replay_latch(
                        request, parser.commitment_reason, event_type
                    )
                    return (
                        prefixed,
                        None,
                    )
                if pipeline.held_limit_exhausted:
                    prefixed = _PrefixedHttpxByteStream(
                        response,
                        prefix,
                        iterator,
                        event_pipeline=pipeline,
                    )
                    pipeline.attach_to_response(prefixed)
                    _maybe_set_replay_latch(
                        request,
                        "event_limit",
                        event_type,
                    )
                    return prefixed, None

            leftover_incomplete = pipeline.buffered_event_bytes > 0
            try:
                decision, error_payload, event_type = parser.finish()
            except (
                ResponsesBoundFailure,
                ResponsesEventTooLarge,
                ResponsesStreamTooLarge,
            ) as exc:
                await _close_precommit_resources()
                return response, _bound_failure(exc)
            if decision == "failed":
                commitment_reason = parser.commitment_reason
                if commitment_reason is not None:
                    _maybe_set_replay_latch(
                        request, commitment_reason, event_type
                    )
                return (
                    _PrefixedHttpxByteStream(response, prefix, iterator),
                    PassThroughStreamingHandler._build_responses_pre_commit_failure(
                        error_payload=error_payload,
                        event_type=event_type,
                        openai_alpha_capacity_retry_enabled=(
                            openai_alpha_capacity_retry_enabled
                        ),
                    ),
                )
            if decision in {"empty", "lifecycle"}:
                missing_terminal = (
                    PassThroughStreamingHandler._build_responses_pre_commit_failure(
                        error_payload={
                            "code": "openai_responses_stream_missing_terminal",
                            "type": "invalid_response",
                            "message": (
                                "OpenAI Responses stream ended before a substantive "
                                "or terminal event."
                            ),
                        },
                        event_type=decision,
                        openai_alpha_capacity_retry_enabled=(
                            openai_alpha_capacity_retry_enabled
                        ),
                        provider_returned=False,
                        failure_origin="local_missing_terminal",
                    )
                )
                # OPENAI-080 (R8): a truncated UTF-8 remainder after accepted
                # safe events commits with "prefix_limit".  Replay the
                # exact prefix and let the live pipeline finish.
                if prefix.byte_count > 0 and leftover_incomplete:
                    _maybe_set_replay_latch(
                        request, "prefix_limit", event_type
                    )
                    prefixed = _PrefixedHttpxByteStream(
                        response,
                        prefix,
                        iterator,
                        event_pipeline=pipeline,
                    )
                    pipeline.attach_to_response(prefixed)
                    return prefixed, None
                await _close_precommit_resources()
                return response, missing_terminal
            prefixed = _PrefixedHttpxByteStream(
                response,
                prefix,
                iterator,
                event_pipeline=pipeline,
            )
            pipeline.attach_to_response(prefixed)
            _maybe_set_replay_latch(
                request, parser.commitment_reason, event_type
            )
            return prefixed, None
        except BaseException:
            prefix.close()
            close_iterator = getattr(iterator, "aclose", None)
            if callable(close_iterator):
                try:
                    result = close_iterator()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            close_response = getattr(response, "aclose", None)
            if callable(close_response):
                try:
                    result = close_response()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            raise

    @staticmethod
    def _dedupe_done_chunks(
        *,
        chunks: List[bytes],
        lines: List[str],
    ) -> tuple[List[bytes], List[str]]:
        filtered_chunks = [
            chunk
            for chunk in chunks
            if not PassThroughStreamingHandler._is_done_chunk(chunk)
        ]
        filtered_lines = [
            line
            for line in lines
            if _strip_chunk_line(line) != "data: [DONE]"
        ]
        return filtered_chunks, filtered_lines

    @staticmethod
    def _ensure_streaming_metadata(success_handler_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(success_handler_kwargs, dict):
            return {}

        litellm_params = success_handler_kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
            success_handler_kwargs["litellm_params"] = litellm_params

        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            litellm_params["metadata"] = metadata

        return metadata

    @staticmethod
    def _sanitize_anthropic_rate_limit_headers(
        response_headers: httpx.Headers,
    ) -> Dict[str, str]:
        sanitized: Dict[str, str] = {}
        for header_name, header_value in response_headers.items():
            normalized_name = str(header_name).lower()
            if not (
                normalized_name.startswith(
                    PassThroughStreamingHandler._ANTHROPIC_RATE_LIMIT_HEADER_PREFIXES
                )
                or normalized_name
                in PassThroughStreamingHandler._ANTHROPIC_RATE_LIMIT_HEADER_NAMES
            ):
                continue
            sanitized[normalized_name] = str(header_value)
        if sanitized:
            sanitized["source"] = "anthropic_response_headers"
        return sanitized

    @staticmethod
    def _sanitize_codex_rate_limit_headers(
        response_headers: httpx.Headers,
    ) -> Dict[str, str]:
        sanitized: Dict[str, str] = {}
        for header_name, header_value in response_headers.items():
            normalized_name = str(header_name).lower()
            if not (
                normalized_name.startswith(
                    PassThroughStreamingHandler._CODEX_RATE_LIMIT_HEADER_PREFIXES
                )
                or normalized_name
                in PassThroughStreamingHandler._CODEX_RATE_LIMIT_HEADER_NAMES
            ):
                continue
            sanitized[normalized_name] = str(header_value)
        if sanitized:
            sanitized["source"] = "codex_response_headers"
        return sanitized

    @staticmethod
    def _sanitize_xai_oauth_rate_limit_headers(
        response_headers: httpx.Headers,
    ) -> Dict[str, str]:
        return PassThroughStreamingHandler._sanitize_xai_rate_limit_headers(
            response_headers,
            source=PassThroughStreamingHandler._XAI_OAUTH_RATE_LIMIT_SOURCE,
        )

    @staticmethod
    def _sanitize_xai_grok_oidc_rate_limit_headers(
        response_headers: httpx.Headers,
    ) -> Dict[str, str]:
        return PassThroughStreamingHandler._sanitize_xai_rate_limit_headers(
            response_headers,
            source=PassThroughStreamingHandler._XAI_GROK_OIDC_RATE_LIMIT_SOURCE,
        )

    @staticmethod
    def _sanitize_xai_rate_limit_headers(
        response_headers: httpx.Headers,
        *,
        source: str,
    ) -> Dict[str, str]:
        sanitized: Dict[str, str] = {}
        for header_name, header_value in response_headers.items():
            normalized_name = str(header_name).lower()
            if not (
                normalized_name.startswith(
                    PassThroughStreamingHandler._XAI_OAUTH_RATE_LIMIT_HEADER_PREFIXES
                )
                or normalized_name
                in PassThroughStreamingHandler._XAI_OAUTH_RATE_LIMIT_HEADER_NAMES
            ):
                continue
            sanitized[normalized_name] = str(header_value)
        if sanitized:
            sanitized["source"] = source
        return sanitized

    @staticmethod
    def _is_grok_native_oauth_metadata(metadata: Dict[str, Any]) -> bool:
        credential_family = str(metadata.get("credential_family") or "").lower()
        if credential_family:
            return credential_family == "xai_grok_oidc"
        route_family = str(
            metadata.get("passthrough_route_family")
            or metadata.get("route_family")
            or ""
        ).lower()
        if route_family:
            if "xai_oauth" in route_family:
                return False
            if "grok_cli" in route_family or route_family in {"grok-build", "grok_build"}:
                return True
        return metadata.get("grok_native_oauth_managed") is True

    @staticmethod
    def _is_xai_oauth_metadata(metadata: Dict[str, Any]) -> bool:
        if PassThroughStreamingHandler._is_grok_native_oauth_metadata(metadata):
            return False
        if metadata.get("xai_oauth_managed") is True:
            return True
        credential_family = str(metadata.get("credential_family") or "").lower()
        route_family = str(
            metadata.get("passthrough_route_family")
            or metadata.get("route_family")
            or ""
        ).lower()
        return (
            credential_family == "xai_oauth"
            or "xai_oauth" in route_family
            or metadata.get("xai_oauth_public_model") is not None
        )

    @staticmethod
    def _record_upstream_rate_limit_headers_metadata(
        success_handler_kwargs: Optional[Dict[str, Any]],
        *,
        response: httpx.Response,
        endpoint_type: EndpointType,
        custom_llm_provider: Optional[str],
        operational_request: Optional[Any] = None,
    ) -> None:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        if custom_llm_provider == "xai":
            if PassThroughStreamingHandler._is_grok_native_oauth_metadata(metadata):
                sanitized_headers = PassThroughStreamingHandler._sanitize_xai_grok_oidc_rate_limit_headers(
                    response.headers
                )
                if sanitized_headers:
                    metadata[
                        PassThroughStreamingHandler._XAI_GROK_OIDC_RATE_LIMIT_HEADERS_KEY
                    ] = sanitized_headers
                return
            if PassThroughStreamingHandler._is_xai_oauth_metadata(metadata):
                sanitized_headers = PassThroughStreamingHandler._sanitize_xai_oauth_rate_limit_headers(
                    response.headers
                )
                if sanitized_headers:
                    metadata[
                        PassThroughStreamingHandler._XAI_OAUTH_RATE_LIMIT_HEADERS_KEY
                    ] = sanitized_headers
                PassThroughEndpointLogging._record_validated_xai_oauth_server_account_metadata(
                    metadata,
                    operational_request,
                )
            return
        if endpoint_type == EndpointType.ANTHROPIC or custom_llm_provider == "anthropic":
            sanitized_headers = PassThroughStreamingHandler._sanitize_anthropic_rate_limit_headers(
                response.headers
            )
            if sanitized_headers:
                metadata["anthropic_response_headers"] = sanitized_headers
        if endpoint_type == EndpointType.OPENAI or custom_llm_provider == "openai":
            sanitized_headers = PassThroughStreamingHandler._sanitize_codex_rate_limit_headers(
                response.headers
            )
            if sanitized_headers:
                metadata["codex_response_headers"] = sanitized_headers

    @staticmethod
    def _prepare_streaming_metadata(
        success_handler_kwargs: Optional[Dict[str, Any]],
        *,
        response: httpx.Response,
        endpoint_type: EndpointType,
        custom_llm_provider: Optional[str],
        operational_request: Optional[Any] = None,
    ) -> Dict[str, Any]:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        PassThroughStreamingHandler._record_upstream_rate_limit_headers_metadata(
            success_handler_kwargs,
            response=response,
            endpoint_type=endpoint_type,
            custom_llm_provider=custom_llm_provider,
            operational_request=operational_request,
        )
        return metadata

    @staticmethod
    def _format_span_timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _append_stream_span(
        success_handler_kwargs: Optional[Dict[str, Any]],
        *,
        name: str,
        start_time: datetime,
        end_time: datetime,
        span_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        if not metadata:
            return

        langfuse_spans = metadata.get("langfuse_spans")
        if not isinstance(langfuse_spans, list):
            langfuse_spans = []
            metadata["langfuse_spans"] = langfuse_spans

        descriptor: Dict[str, Any] = {
            "name": name,
            "start_time": PassThroughStreamingHandler._format_span_timestamp(start_time),
            "end_time": PassThroughStreamingHandler._format_span_timestamp(end_time),
        }
        if span_metadata:
            descriptor["metadata"] = span_metadata
        langfuse_spans.append(descriptor)

    @staticmethod
    def _sync_logging_obj_model_call_details_from_kwargs(
        litellm_logging_obj: LiteLLMLoggingObj,
        kwargs: Dict[str, Any],
    ) -> None:
        model_call_details = getattr(litellm_logging_obj, "model_call_details", None)
        if not isinstance(model_call_details, dict):
            return

        for key in (
            "litellm_params",
            "standard_logging_object",
            "response_cost",
            "model",
            "custom_llm_provider",
            "passthrough_logging_payload",
            "call_type",
            "litellm_call_id",
            "completion_start_time",
            # D1-616: route rollup attachment happens after this sync runs in the
            # streaming finalize path; keep the private review-decision event on the
            # allowlist so callback-visible model_call_details carries it.
            _AAWM_PARSED_CODEX_REVIEW_DECISIONS_KWARGS_KEY,
        ):
            if key in kwargs:
                model_call_details[key] = kwargs[key]

    @staticmethod
    def _clean_streaming_logging_context_value(value: Any) -> Optional[str]:
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return None
        if not isinstance(value, (str, int, float)):
            return None

        cleaned = "".join(
            char if char.isprintable() and char not in "\r\n\t" else " "
            for char in str(value).strip()
        )
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return None
        if cleaned.lower().startswith(("bearer ", "sk-", "pk-", "xai-", "ya29.")):
            return None
        if len(cleaned) > 240:
            cleaned = cleaned[:237] + "..."
        return cleaned

    @staticmethod
    def _safe_streaming_logging_url(value: Any) -> Optional[str]:
        cleaned = PassThroughStreamingHandler._clean_streaming_logging_context_value(
            value
        )
        if not cleaned:
            return None
        parsed = urlparse(cleaned)
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            return f"{parsed.scheme}://{host}{parsed.path or '/'}"
        return cleaned

    @staticmethod
    def _first_streaming_logging_context_value(
        *values: Any,
    ) -> Optional[str]:
        for value in values:
            cleaned = PassThroughStreamingHandler._clean_streaming_logging_context_value(
                value
            )
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _build_streaming_logging_error_context(
        *,
        litellm_logging_obj: LiteLLMLoggingObj,
        response: Optional[httpx.Response],
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        custom_llm_provider: Optional[str],
        success_handler_kwargs: Optional[Dict[str, Any]],
        error_log_context: Optional[Dict[str, Any]],
        handler_branch: str,
    ) -> Dict[str, Any]:
        context = dict(error_log_context or {})
        kwargs = success_handler_kwargs if isinstance(success_handler_kwargs, dict) else {}
        litellm_params = kwargs.get("litellm_params")
        kwargs_metadata = (
            litellm_params.get("metadata")
            if isinstance(litellm_params, dict)
            else None
        )
        metadata: Dict[str, Any] = {}
        if isinstance(request_body, dict):
            for metadata_key in ("litellm_metadata", "metadata"):
                metadata_value = request_body.get(metadata_key)
                if isinstance(metadata_value, dict):
                    metadata.update(metadata_value)
        if isinstance(kwargs_metadata, dict):
            metadata.update(kwargs_metadata)

        model_call_details = getattr(litellm_logging_obj, "model_call_details", None)
        if not isinstance(model_call_details, dict):
            model_call_details = {}

        def set_default(key: str, value: Any) -> None:
            if context.get(key) is not None:
                return
            if key in {"upstream_url"}:
                cleaned_value = PassThroughStreamingHandler._safe_streaming_logging_url(
                    value
                )
            else:
                cleaned_value = PassThroughStreamingHandler._clean_streaming_logging_context_value(
                    value
                )
            if cleaned_value is not None:
                context[key] = cleaned_value

        set_default("source", "pass_through_streaming_logging")
        set_default("endpoint", url_route)
        set_default("upstream_url", url_route)
        set_default("provider", custom_llm_provider or endpoint_type.value)
        set_default(
            "model",
            PassThroughStreamingHandler._first_streaming_logging_context_value(
                request_body.get("model") if isinstance(request_body, dict) else None,
                model_call_details.get("model"),
                metadata.get("anthropic_auto_agent_selected_model"),
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("model"),
            ),
        )
        set_default(
            "model_alias",
            PassThroughStreamingHandler._first_streaming_logging_context_value(
                metadata.get("requested_model_alias"),
                metadata.get("model_alias_label"),
                metadata.get("inbound_model_alias"),
            ),
        )
        set_default(
            "route_family",
            PassThroughStreamingHandler._first_streaming_logging_context_value(
                metadata.get("passthrough_route_family"),
                metadata.get("route_family"),
                metadata.get("openai_passthrough_route_family"),
            ),
        )
        if context.get("status_code") is None:
            context["status_code"] = response.status_code if response is not None else None
        set_default(
            "trace_id",
            PassThroughStreamingHandler._first_streaming_logging_context_value(
                metadata.get("trace_id"),
                model_call_details.get("trace_id"),
            ),
        )
        set_default(
            "litellm_call_id",
            PassThroughStreamingHandler._first_streaming_logging_context_value(
                kwargs.get("litellm_call_id"),
                getattr(litellm_logging_obj, "litellm_call_id", None),
                model_call_details.get("litellm_call_id"),
            ),
        )
        set_default("callback_name", "pass_through_streaming")
        set_default("callback_phase", "post_response_stream_logging")
        set_default("handler_branch", handler_branch)
        return context

    @staticmethod
    def _capture_stream_shape(
        *,
        response: Optional[httpx.Response],
        endpoint_type: EndpointType,
        url_route: str,
        request_body: dict,
        raw_bytes: List[bytes],
        all_chunks: List[str],
        metadata: Dict[str, Any],
        custom_llm_provider: Optional[str],
        litellm_call_id: Optional[str],
    ) -> None:
        try:
            upstream_request = response.request if response is not None else None
        except RuntimeError:
            upstream_request = None
        capture_passthrough_stream_shape(
            provider=custom_llm_provider or endpoint_type.value,
            endpoint_type=endpoint_type,
            url_route=url_route,
            request_body=request_body,
            response=response,
            upstream_request=upstream_request,
            all_chunks=all_chunks,
            raw_bytes=raw_bytes,
            litellm_call_id=litellm_call_id,
            extra_metadata={
                "custom_llm_provider": custom_llm_provider,
                "is_openai_responses": metadata[
                    "aawm_stream_logging_is_openai_responses"
                ],
            },
        )

    @staticmethod
    def _annotate_stream_logging_metadata(
        metadata: Dict[str, Any],
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> None:
        metadata["aawm_stream_logging_endpoint_type"] = endpoint_type.value
        if custom_llm_provider:
            metadata["aawm_stream_logging_custom_llm_provider"] = custom_llm_provider
        metadata["aawm_stream_logging_is_openai_responses"] = (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        )

    @staticmethod
    async def chunk_processor(  # noqa: PLR0915
        response: httpx.Response,
        request_body: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        start_time: datetime,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = None,
        custom_llm_provider: Optional[str] = None,
        success_handler_kwargs: Optional[Dict[str, Any]] = None,
        upstream_wait_started_at: Optional[datetime] = None,
        upstream_wait_completed_at: Optional[datetime] = None,
        local_prepare_ms: Optional[float] = None,
        error_log_context: Optional[Dict[str, Any]] = None,
        deferred_success_holder: Optional[DeferredPassthroughSuccess] = None,
        openai_wire_trace: Optional[Any] = None,
        openai_stream_bookkeeping_state: Optional[Dict[str, Any]] = None,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
        operational_request: Optional[Any] = None,
    ):
        """
        - Yields chunks from the response
        - Collect non-empty chunks for post-processing (logging)
        - Inject cost into chunks if include_cost_in_streaming_usage is enabled
        """
        completion_bookkeeping_done = False
        responses_tracking_finalized = False
        metadata: Dict[str, Any] = {}
        transfer_identity: Dict[str, Any] = {}

        async def _finalize_transfer_if_needed(
            identity: Dict[str, Any],
            phase: str,
            *,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            if openai_wire_trace is not None:
                metadata["aawm_deferred_transfer_phase"] = phase
                if extra:
                    metadata["aawm_deferred_transfer_extra"] = dict(extra)
                return
            await wait_for(
                "transfer_registry", safe_finalize(identity, phase, extra=extra)
            )

        try:
            line_accumulator: Optional[_PassThroughStreamLineAccumulator] = None
            model_name = PassThroughStreamingHandler._extract_model_for_cost_injection(
                request_body=request_body,
                url_route=url_route,
                endpoint_type=endpoint_type,
                litellm_logging_obj=litellm_logging_obj,
            )
            is_fork_owned_responses_stream = (
                PassThroughStreamingHandler._is_openai_responses_stream(
                    endpoint_type=endpoint_type,
                    url_route=url_route,
                    custom_llm_provider=custom_llm_provider,
                )
                or PassThroughStreamingHandler._is_responses_wire_owned_stream(
                    response=response,
                    endpoint_type=endpoint_type,
                    url_route=url_route,
                    custom_llm_provider=custom_llm_provider,
                )
            )
            raw_bytes: Any = []
            is_non_openai_responses_stream = (
                endpoint_type == EndpointType.OPENAI
                and not is_fork_owned_responses_stream
                and custom_llm_provider != "openai"
                and OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                    url_route
                )
            )
            responses_event_pipeline = responses_event_pipeline or (
                responses_event_pipeline_from_response(response)
                if is_fork_owned_responses_stream
                else None
            )
            if is_fork_owned_responses_stream and responses_event_pipeline is None:
                responses_event_pipeline = create_responses_event_pipeline()
                responses_event_pipeline.attach_to_response(response)
            responses_sse_tracker = (
                OpenAIPassthroughLoggingHandler._create_responses_sse_state_tracker(
                    fallback_model=model_name or "unknown"
                )
                if is_fork_owned_responses_stream
                else None
            )
            responses_terminal_seen = False
            responses_substantive_seen = False
            held_responses_done_suffix = b""
            responses_sse_event_buffer = b""
            buffer_raw_bytes = True
            if (
                not is_fork_owned_responses_stream
                and PassThroughStreamingHandler._stream_summary_first_finalize_eligible(
                    endpoint_type=endpoint_type,
                    url_route=url_route,
                    custom_llm_provider=custom_llm_provider,
                )
            ):
                line_accumulator = _PassThroughStreamLineAccumulator()
            if is_fork_owned_responses_stream:
                # Logging/cost state comes from the attempt-scoped reducer.
                # Retain raw bytes only for explicit diagnostic capture.
                buffer_raw_bytes = (
                    PassThroughStreamingHandler._raw_stream_bytes_capture_required()
                )
            else:
                buffer_raw_bytes = (
                    PassThroughStreamingHandler._should_buffer_raw_stream_bytes(
                        line_accumulator_enabled=line_accumulator is not None,
                    )
                )
            chunk_count = 0
            total_stream_bytes = 0
            first_chunk_at: Optional[datetime] = None
            first_emitted_at: Optional[datetime] = None
            metadata = PassThroughStreamingHandler._prepare_streaming_metadata(
                success_handler_kwargs,
                response=response,
                endpoint_type=endpoint_type,
                custom_llm_provider=custom_llm_provider,
                operational_request=operational_request,
            )
            metadata["aawm_stream_raw_bytes_buffered"] = buffer_raw_bytes
            transfer_identity = extract_transfer_identity(
                request_body=request_body if isinstance(request_body, dict) else None,
                logging_obj=litellm_logging_obj,
                kwargs=success_handler_kwargs,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
                stream_path="pass_through",
            )
            await wait_for(
                "transfer_registry",
                safe_mark_phase(transfer_identity, "awaiting_upstream"),
            )
            downstream_chunk_count = 0
            downstream_byte_count = 0

            enrich_request_waits(transfer_identity)

            async def _publish_transfer_chunks(
                *,
                first_upstream: bool = False,
                first_downstream: bool = False,
                force: bool = False,
            ) -> None:
                if not force and chunk_count > 1 and chunk_count % 8 != 0:
                    return
                await wait_for(
                    "transfer_registry",
                    safe_record_chunks(
                        transfer_identity,
                        upstream_chunks=chunk_count,
                        upstream_bytes=total_stream_bytes,
                        downstream_chunks=downstream_chunk_count,
                        downstream_bytes=downstream_byte_count,
                        first_upstream=first_upstream,
                        first_downstream=first_downstream,
                    ),
                )

            def _final_wire_committed() -> bool:
                if openai_wire_trace is None:
                    return first_emitted_at is not None
                return bool(
                    getattr(openai_wire_trace, "first_body_sent", False)
                    or getattr(openai_wire_trace, "terminal_wire_committed", False)
                    or getattr(openai_wire_trace, "done_wire_committed", False)
                    or getattr(openai_wire_trace, "asgi_delivery_complete", False)
                )

            def _mark_first_emitted_chunk() -> None:
                nonlocal first_emitted_at
                if first_emitted_at is not None:
                    return
                first_emitted_at = datetime.now()
                metadata["aawm_first_emitted_chunk_ms"] = round(
                    max(0.0, (first_emitted_at - start_time).total_seconds() * 1000.0),
                    3,
                )
                if first_chunk_at is not None:
                    metadata["aawm_stream_emit_gap_ms"] = round(
                        max(
                            0.0,
                            (first_emitted_at - first_chunk_at).total_seconds()
                            * 1000.0,
                        ),
                        3,
                    )

            def _record_responses_wire_chunk(chunk: bytes) -> None:
                nonlocal downstream_chunk_count, downstream_byte_count
                if buffer_raw_bytes:
                    raw_bytes.append(chunk)
                if line_accumulator is not None:
                    line_accumulator.feed(chunk)
                downstream_chunk_count += 1
                downstream_byte_count += len(chunk)
                _mark_first_emitted_chunk()

            def _consume_responses_event(
                parsed_event: ParsedResponsesEvent,
            ) -> Optional[ResponsesStreamPreCommitFailure]:
                nonlocal responses_terminal_seen, responses_substantive_seen
                if responses_sse_tracker is None:
                    return None
                if parsed_event.is_malformed:
                    OpenAIPassthroughLoggingHandler._mark_responses_sse_parse_ambiguity(
                        responses_sse_tracker
                    )
                    return None
                if not responses_terminal_seen and parsed_event.event_type in (
                    PassThroughStreamingHandler._RESPONSES_TERMINAL_EVENTS
                ):
                    responses_terminal_seen = True
                decoded_event = parsed_event.payload
                if decoded_event is None:
                    return None
                if PassThroughStreamingHandler._is_responses_substantive_event(
                    parsed_event.event_type,
                    decoded_event,
                ):
                    responses_substantive_seen = True
                extracted_error = (
                    PassThroughStreamingHandler._extract_responses_stream_error_payload(
                        parsed_event.event_type,
                        decoded_event,
                    )
                )
                recovery_failure: Optional[ResponsesStreamPreCommitFailure] = None
                if extracted_error is not None:
                    candidate_failure = (
                        PassThroughStreamingHandler._build_responses_pre_commit_failure(
                            error_payload=extracted_error,
                            event_type=parsed_event.event_type,
                        )
                    )
                    if (
                        candidate_failure.error_class
                        in _RESPONSES_STREAM_REDISPATCH_CLASSES
                    ):
                        recovery_failure = candidate_failure
                OpenAIPassthroughLoggingHandler._consume_responses_sse_event(
                    responses_sse_tracker,
                    decoded_event,
                )
                return recovery_failure

            def _finalize_responses_tracking() -> Optional[Dict[str, Any]]:
                nonlocal responses_tracking_finalized
                nonlocal responses_sse_event_buffer, held_responses_done_suffix
                if (
                    responses_tracking_finalized
                    or responses_event_pipeline is None
                ):
                    return None
                responses_tracking_finalized = True
                if responses_sse_event_buffer or (
                    responses_event_pipeline.buffered_event_bytes
                ):
                    OpenAIPassthroughLoggingHandler._mark_responses_sse_partial_frame(
                        responses_sse_tracker
                    )
                else:
                    for parsed_event in responses_event_pipeline.finish():
                        _consume_responses_event(parsed_event)

                if responses_sse_event_buffer:
                    # Native Responses partial frames are not client-deliverable
                    # data. The outer coordinator emits one canonical
                    # incomplete terminal at EOF; do not concatenate this tail
                    # with that synthetic event.
                    responses_sse_event_buffer = b""
                    held_responses_done_suffix = b""

                synthetic_terminal: Optional[Dict[str, Any]] = None
                if _final_wire_committed() and not responses_terminal_seen:
                    synthetic_terminal = (
                        OpenAIPassthroughLoggingHandler._classify_responses_sse_clean_eof(
                            responses_sse_tracker
                        )
                    )
                metadata["aawm_stream_tracker_state"] = (
                    OpenAIPassthroughLoggingHandler._responses_sse_tracker_metadata(
                        responses_sse_tracker
                    )
                )
                return synthetic_terminal

            async def _finalize_completed_stream(
                delivered_snapshot: Optional[Dict[str, Any]] = None,
            ) -> None:
                nonlocal completion_bookkeeping_done, responses_sse_event_buffer
                nonlocal held_responses_done_suffix
                if completion_bookkeeping_done:
                    return
                if openai_wire_trace is not None and delivered_snapshot is None:
                    return
                completion_bookkeeping_done = True

                _finalize_responses_tracking()
                end_time = datetime.now()
                delivered_disposition = (
                    delivered_snapshot.get("disposition")
                    if isinstance(delivered_snapshot, dict)
                    else None
                )
                if delivered_disposition:
                    metadata["aawm_delivered_wire_disposition"] = dict(
                        delivered_snapshot
                    )
                    metadata["aawm_delivered_disposition"] = str(
                        delivered_disposition
                    ).strip().lower()
                metadata["aawm_stream_chunk_count"] = chunk_count
                metadata["aawm_stream_total_bytes"] = total_stream_bytes
                if upstream_wait_started_at is not None:
                    metadata["aawm_upstream_stream_complete_ms"] = round(
                        max(
                            0.0,
                            (end_time - upstream_wait_started_at).total_seconds()
                            * 1000.0,
                        ),
                        3,
                    )
                metadata["aawm_total_proxy_duration_ms"] = round(
                    max(0.0, (end_time - start_time).total_seconds() * 1000.0),
                    3,
                )
                PassThroughStreamingHandler._append_stream_span(
                    success_handler_kwargs,
                    name="stream.completed",
                    start_time=upstream_wait_completed_at or start_time,
                    end_time=end_time,
                    span_metadata={
                        "chunk_count": chunk_count,
                        "stream_bytes": total_stream_bytes,
                        "upstream_stream_complete_ms": metadata.get(
                            "aawm_upstream_stream_complete_ms"
                        ),
                    },
                )
                await wait_for(
                    "transfer_registry",
                    safe_mark_phase(
                        transfer_identity,
                        "finalizing",
                        extra={
                            "delivered_disposition": delivered_disposition,
                            "upstream_chunk_count": chunk_count,
                            "upstream_byte_count": total_stream_bytes,
                            "downstream_chunk_count": downstream_chunk_count,
                            "downstream_byte_count": downstream_byte_count,
                        },
                    ),
                )
                transfer_phase = {
                    "completed": "completed",
                    "failed": "failed",
                    "incomplete": "failed",
                    "cancelled": "cancelled",
                    "disconnected": "disconnected",
                }.get(
                    str(delivered_disposition or "completed").strip().lower(),
                    "failed",
                )
                async def _finalize_stream_delivery() -> None:
                    await wait_for(
                        "transfer_registry",
                        safe_finalize(
                            transfer_identity,
                            transfer_phase,
                            extra={
                                "delivered_disposition": delivered_disposition,
                                "upstream_chunk_count": chunk_count,
                                "upstream_byte_count": total_stream_bytes,
                                "downstream_chunk_count": downstream_chunk_count,
                                "downstream_byte_count": downstream_byte_count,
                            },
                        ),
                    )

                    precomputed_lines: Optional[List[str]] = None
                    if line_accumulator is not None:
                        precomputed_lines = line_accumulator.finish()

                    if openai_wire_trace is not None:
                        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
                            litellm_logging_obj=litellm_logging_obj,
                            passthrough_success_handler_obj=passthrough_success_handler_obj,
                            response=response,
                            url_route=url_route,
                            request_body=request_body or {},
                            endpoint_type=endpoint_type,
                            start_time=start_time,
                            raw_bytes=raw_bytes,
                            precomputed_lines=precomputed_lines,
                            end_time=end_time,
                            passthrough_logging_payload=passthrough_logging_payload,
                            custom_llm_provider=custom_llm_provider,
                            success_handler_kwargs=success_handler_kwargs,
                            local_prepare_ms=local_prepare_ms,
                            error_log_context=error_log_context,
                            responses_event_pipeline=responses_event_pipeline,
                            operational_request=operational_request,
                        )
                    else:
                        asyncio.create_task(
                            PassThroughStreamingHandler._route_streaming_logging_to_handler(
                                litellm_logging_obj=litellm_logging_obj,
                                passthrough_success_handler_obj=passthrough_success_handler_obj,
                                response=response,
                                url_route=url_route,
                                request_body=request_body or {},
                                endpoint_type=endpoint_type,
                                start_time=start_time,
                                raw_bytes=raw_bytes,
                                precomputed_lines=precomputed_lines,
                                end_time=end_time,
                                passthrough_logging_payload=passthrough_logging_payload,
                                custom_llm_provider=custom_llm_provider,
                                success_handler_kwargs=success_handler_kwargs,
                                local_prepare_ms=local_prepare_ms,
                                error_log_context=error_log_context,
                                responses_event_pipeline=responses_event_pipeline,
                            )
                        )

                if (
                    deferred_success_holder is not None
                    and openai_wire_trace is None
                ):
                    pending_finalization_task = (
                        deferred_success_holder.set_finalizer(
                            _finalize_stream_delivery
                        )
                    )
                    if pending_finalization_task is not None:
                        await asyncio.shield(pending_finalization_task)
                else:
                    await _finalize_stream_delivery()

            if openai_wire_trace is not None:

                async def _run_post_delivery_bookkeeping(
                    delivered_snapshot: Dict[str, Any],
                ) -> None:
                    await wait_for(
                        "terminal_bookkeeping",
                        _finalize_completed_stream(delivered_snapshot),
                    )

                if openai_stream_bookkeeping_state is not None:
                    openai_stream_bookkeeping_state[
                        "finalize_callback"
                    ] = _run_post_delivery_bookkeeping
                else:
                    openai_wire_trace.register_post_finalization_callback(
                        _run_post_delivery_bookkeeping
                    )

            def _mark_responses_terminal_pending() -> None:
                extensions = getattr(response, "extensions", None)
                if isinstance(extensions, dict):
                    extensions["aawm_openai_responses_terminal_pending"] = True

            async for chunk in wait_iterator(
                "upstream_stream_read", response.aiter_bytes()
            ):
                current_chunk_at = datetime.now()
                chunk_count += 1
                total_stream_bytes += len(chunk)
                if first_chunk_at is None:
                    first_chunk_at = current_chunk_at
                    if hasattr(litellm_logging_obj, "_update_completion_start_time"):
                        litellm_logging_obj._update_completion_start_time(
                            completion_start_time=first_chunk_at
                        )
                    if isinstance(success_handler_kwargs, dict):
                        success_handler_kwargs["completion_start_time"] = first_chunk_at
                    metadata["aawm_time_to_first_token_ms"] = round(
                        max(0.0, (first_chunk_at - start_time).total_seconds() * 1000.0),
                        3,
                    )
                    if upstream_wait_started_at is not None:
                        metadata["aawm_upstream_first_chunk_ms"] = round(
                            max(
                                0.0,
                                (first_chunk_at - upstream_wait_started_at).total_seconds()
                                * 1000.0,
                            ),
                            3,
                        )
                    PassThroughStreamingHandler._append_stream_span(
                        success_handler_kwargs,
                        name="stream.first_token",
                        start_time=upstream_wait_started_at or start_time,
                        end_time=first_chunk_at,
                        span_metadata={
                            "chunk_count": chunk_count,
                            "time_to_first_token_ms": metadata.get(
                                "aawm_time_to_first_token_ms"
                            ),
                            "upstream_first_chunk_ms": metadata.get(
                                "aawm_upstream_first_chunk_ms"
                            ),
                        },
                    )
                    await _publish_transfer_chunks(first_upstream=True, force=True)

                if responses_event_pipeline is not None:
                    limit_exception: Optional[ResponsesBoundFailure] = None
                    try:
                        admitted_events = responses_events_from_chunk(chunk)
                        if admitted_events is None:
                            parsed_events = responses_event_pipeline.feed(chunk)
                        else:
                            # The precommit wrapper already admitted this
                            # carrier. Consume its typed associations without
                            # invoking the parser or reducer again.
                            parsed_events = admitted_events
                    except (
                        ResponsesBoundFailure,
                        ResponsesEventTooLarge,
                        ResponsesStreamTooLarge,
                    ) as exception:
                        limit_exception = ResponsesBoundFailure.from_exception(
                            exception
                        )
                        parsed_events = list(limit_exception.accepted_events)

                    recovery_failure: Optional[ResponsesStreamPreCommitFailure] = None
                    delivered_event_chunks: list[bytes] = []
                    delivered_events: list[ParsedResponsesEvent] = []
                    for parsed_event in parsed_events:
                        if recovery_failure is not None:
                            break
                        candidate_failure = _consume_responses_event(parsed_event)
                        if candidate_failure is not None:
                            recovery_failure = candidate_failure
                            break
                        if parsed_event.forward_event is not None:
                            delivered_event_chunks.append(parsed_event.forward_event)
                            delivered_events.append(parsed_event)

                    if delivered_event_chunks:
                        complete_chunk = attach_responses_events(
                            b"".join(delivered_event_chunks),
                            delivered_events,
                        )
                    else:
                        complete_chunk = b""

                    if complete_chunk:
                        complete_chunk = PassThroughStreamingHandler._stamp_encrypted_reasoning_in_responses_sse_chunk(
                            complete_chunk,
                            request_body=(
                                request_body
                                if isinstance(request_body, dict)
                                else None
                            ),
                            custom_llm_provider=custom_llm_provider,
                            events=delivered_events,
                        )

                    if recovery_failure is not None and (
                        responses_substantive_seen or downstream_chunk_count > 0
                    ):
                        # The recovery event is typed and isolated; do not
                        # forward a redispatch-only failure envelope.
                        if complete_chunk:
                            _record_responses_wire_chunk(complete_chunk)
                            await _publish_transfer_chunks(
                                first_downstream=first_emitted_at is not None
                                and downstream_chunk_count == 1
                            )
                            yield complete_chunk
                        failure_context = (
                            PassThroughStreamingHandler._build_responses_stream_redispatch_failure_context(
                                failure=recovery_failure,
                                request_body=(
                                    request_body
                                    if isinstance(request_body, dict)
                                    else None
                                ),
                                success_handler_kwargs=success_handler_kwargs,
                                url_route=url_route,
                                custom_llm_provider=custom_llm_provider,
                                chunk_count=chunk_count,
                                total_stream_bytes=total_stream_bytes,
                            )
                        )
                        metadata.update(failure_context)
                        metadata["aawm_stream_interrupted"] = True
                        metadata["aawm_route_rollup_turn_suppressed"] = True
                        held_responses_done_suffix = b""
                        responses_sse_event_buffer = b""
                        responses_terminal_seen = True
                        terminal_chunks = (
                            PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
                                endpoint_type=endpoint_type,
                                url_route=url_route,
                                custom_llm_provider=custom_llm_provider,
                                failure_context=failure_context,
                                exc=recovery_failure,
                                responses_event_pipeline=responses_event_pipeline,
                            )
                        )
                        _mark_responses_terminal_pending()
                        for terminal_chunk in terminal_chunks:
                            _record_responses_wire_chunk(terminal_chunk)
                            yield terminal_chunk
                        break

                    if limit_exception is not None:
                        responses_sse_event_buffer = b""
                        held_responses_done_suffix = b""
                        if responses_sse_tracker is not None:
                            OpenAIPassthroughLoggingHandler._mark_responses_sse_partial_frame(
                                responses_sse_tracker
                            )
                        failure_kind = (
                            "streaming_upstream_event_too_large"
                            if limit_exception.kind == "event_too_large"
                            else "streaming_upstream_too_large"
                        )
                        committed_downstream = _final_wire_committed()
                        failure_context = dict(error_log_context or {})
                        failure_context.update(
                            {
                                "failure_kind": failure_kind,
                                "error_class": type(limit_exception.cause).__name__,
                                "error_code": limit_exception.code,
                                "error_message": str(limit_exception),
                                "terminal_error_payload": dict(
                                    limit_exception.error_payload
                                ),
                                "stream_failure_stage": (
                                    "stream_interrupted_after_first_byte"
                                    if committed_downstream
                                    else "stream_interrupted_before_emit"
                                ),
                                "stream_chunks_seen": chunk_count,
                                "stream_bytes_seen": total_stream_bytes,
                                "stream_hidden_retry_safe": False,
                                "aawm_stream_incomplete_reason": (
                                    limit_exception.incomplete_reason
                                ),
                                "responses_api_terminal": True,
                            }
                        )
                        if isinstance(error_log_context, dict):
                            error_log_context.update(failure_context)
                        metadata.update(failure_context)
                        metadata["aawm_stream_interrupted"] = True
                        metadata["aawm_stream_incomplete"] = True
                        metadata["aawm_route_rollup_turn_suppressed"] = True
                        if complete_chunk:
                            _record_responses_wire_chunk(complete_chunk)
                            await _publish_transfer_chunks(
                                first_downstream=first_emitted_at is not None
                                and downstream_chunk_count == 1
                            )
                            yield complete_chunk
                        terminal_chunks = (
                            PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
                                endpoint_type=endpoint_type,
                                url_route=url_route,
                                custom_llm_provider=custom_llm_provider,
                                failure_context=failure_context,
                                exc=limit_exception,
                                responses_event_pipeline=responses_event_pipeline,
                            )
                        )
                        _mark_responses_terminal_pending()
                        for terminal_chunk in terminal_chunks:
                            _record_responses_wire_chunk(terminal_chunk)
                            yield terminal_chunk
                        break

                    if not complete_chunk:
                        continue

                    if responses_terminal_seen:
                        if held_responses_done_suffix:
                            _record_responses_wire_chunk(
                                held_responses_done_suffix
                            )
                            yield held_responses_done_suffix
                            held_responses_done_suffix = b""
                        _record_responses_wire_chunk(complete_chunk)
                        await _publish_transfer_chunks(
                            first_downstream=first_emitted_at is not None
                            and downstream_chunk_count == 1
                        )
                        _mark_responses_terminal_pending()
                        yield complete_chunk
                        # The outer native Responses coordinator owns terminal
                        # and [DONE] delivery. It closes this source after
                        # delivery; GeneratorExit then runs the non-reading
                        # completion bookkeeping path below.
                        break

                    prior_held = responses_emissions_from_chunk(
                        held_responses_done_suffix
                    ) or []
                    current_emissions = responses_emissions_from_chunk(
                        complete_chunk
                    ) or [event.emission() for event in delivered_events]
                    combined_emissions = [*prior_held, *current_emissions]
                    combined_chunk = attach_responses_emissions(
                        b"",
                        combined_emissions,
                    )
                    (
                        chunk_without_done,
                        held_responses_done_suffix,
                    ) = PassThroughStreamingHandler._split_trailing_done_chunk(
                        combined_chunk
                    )
                    prefix_emissions, suffix_emissions = (
                        PassThroughStreamingHandler._partition_responses_emissions_by_count(
                            chunk_without_done,
                            held_responses_done_suffix,
                            combined_emissions,
                        )
                    )
                    if chunk_without_done and prefix_emissions:
                        chunk_without_done = attach_responses_emissions(
                            b"",
                            prefix_emissions,
                        )
                    if held_responses_done_suffix and suffix_emissions:
                        held_responses_done_suffix = attach_responses_emissions(
                            b"",
                            suffix_emissions,
                        )
                    if chunk_without_done:
                        _record_responses_wire_chunk(chunk_without_done)
                        await _publish_transfer_chunks(
                            first_downstream=first_emitted_at is not None
                            and downstream_chunk_count == 1
                        )
                        yield chunk_without_done
                    continue

                if buffer_raw_bytes:
                    raw_bytes.append(chunk)
                if line_accumulator is not None:
                    line_accumulator.feed(chunk)
                if (
                    getattr(litellm, "include_cost_in_streaming_usage", False)
                    and model_name
                ):
                    if endpoint_type == EndpointType.VERTEX_AI:
                        # Only handle streamRawPredict (uses Anthropic format)
                        if "streamRawPredict" in url_route or "rawPredict" in url_route:
                            modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                                chunk, model_name
                            )
                            if modified_chunk is not None:
                                chunk = modified_chunk
                    elif endpoint_type == EndpointType.ANTHROPIC:
                        modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                            chunk, model_name
                        )
                        if modified_chunk is not None:
                            chunk = modified_chunk

                if is_non_openai_responses_stream:
                    responses_sse_event_buffer += bytes(chunk)
                    complete_chunk, responses_sse_event_buffer = (
                        PassThroughStreamingHandler._split_responses_sse_event_buffer(
                            responses_sse_event_buffer
                        )
                    )
                    if not complete_chunk:
                        continue
                    complete_chunk = PassThroughStreamingHandler._stamp_encrypted_reasoning_in_responses_sse_chunk(
                        complete_chunk,
                        request_body=(
                            request_body if isinstance(request_body, dict) else None
                        ),
                        custom_llm_provider=custom_llm_provider,
                    )
                    downstream_chunk_count += 1
                    downstream_byte_count += len(complete_chunk)
                    _mark_first_emitted_chunk()
                    await _publish_transfer_chunks(
                        first_downstream=first_emitted_at is not None
                        and downstream_chunk_count == 1
                    )
                    yield complete_chunk
                    continue

                # OPENAI-006 / CFG-029: stamp encrypted reasoning and
                # aawm_route_identity on non-OpenAI Responses streams too
                # (e.g. xAI) so producer identity survives into Ohmypi JSONL.
                if isinstance(chunk, (bytes, bytearray)) and (
                    (
                        b"encrypted_content" in chunk
                        and b"reasoning" in chunk
                    )
                    or b"response.completed" in chunk
                    or b"response.output_item.added" in chunk
                    or b"response.output_item.done" in chunk
                ):
                    chunk = PassThroughStreamingHandler._stamp_encrypted_reasoning_in_responses_sse_chunk(
                        bytes(chunk),
                        request_body=request_body if isinstance(request_body, dict) else None,
                        custom_llm_provider=custom_llm_provider,
                    )
                downstream_chunk_count += 1
                downstream_byte_count += len(chunk)
                _mark_first_emitted_chunk()
                await _publish_transfer_chunks(
                    first_downstream=first_emitted_at is not None
                    and downstream_chunk_count == 1
                )
                yield chunk

            if responses_event_pipeline is not None:
                synthetic_terminal = _finalize_responses_tracking()
                terminal_chunks: List[bytes] = []
                if synthetic_terminal is not None:
                    event_type = synthetic_terminal["type"]
                    terminal_chunks = [
                        (
                            f"event: {event_type}\ndata: "
                            + json.dumps(
                                synthetic_terminal,
                                separators=(",", ":"),
                            )
                            + "\n\n"
                        ).encode("utf-8"),
                        b"data: [DONE]\n\n",
                    ]
                    terminal_chunks = (
                        PassThroughStreamingHandler._attach_responses_terminal_events(
                            terminal_chunks,
                            event_pipeline=responses_event_pipeline,
                            event_type=event_type,
                            payload=synthetic_terminal,
                        )
                    )

                if terminal_chunks:
                    held_responses_done_suffix = b""
                    # The native Responses coordinator closes this source
                    # after delivering the synthetic incomplete terminal. Mark
                    # the terminal before yielding so GeneratorExit takes the
                    # non-reading bookkeeping path instead of disconnecting.
                    responses_terminal_seen = True
                    _mark_responses_terminal_pending()
                    for terminal_chunk in terminal_chunks:
                        _record_responses_wire_chunk(terminal_chunk)
                        yield terminal_chunk
                elif held_responses_done_suffix:
                    _record_responses_wire_chunk(held_responses_done_suffix)
                    yield held_responses_done_suffix
                    held_responses_done_suffix = b""
            elif is_non_openai_responses_stream and responses_sse_event_buffer:
                trailing_partial = responses_sse_event_buffer
                responses_sse_event_buffer = b""
                downstream_chunk_count += 1
                downstream_byte_count += len(trailing_partial)
                _mark_first_emitted_chunk()
                await _publish_transfer_chunks(
                    first_downstream=first_emitted_at is not None
                    and downstream_chunk_count == 1
                )
                yield trailing_partial

            await wait_for("terminal_bookkeeping", _finalize_completed_stream())
        except asyncio.CancelledError:
            local_identity = (
                transfer_identity if "transfer_identity" in locals() else {}
            )
            if is_cohere_provider_name(custom_llm_provider) or is_cohere_api_host(
                getattr(response, "url", None)
            ):
                try:
                    await wait_for(
                        "resource_cleanup", aclose_upstream_response_once(response)
                    )
                except Exception:
                    pass
            await _finalize_transfer_if_needed(
                local_identity,
                "cancelled",
                extra={
                    "error_code": "cancelled",
                    "error_class": "CancelledError",
                    "upstream_chunk_count": chunk_count
                    if "chunk_count" in locals()
                    else 0,
                    "upstream_byte_count": total_stream_bytes
                    if "total_stream_bytes" in locals()
                    else 0,
                    "downstream_chunk_count": downstream_chunk_count
                    if "downstream_chunk_count" in locals()
                    else 0,
                    "downstream_byte_count": downstream_byte_count
                    if "downstream_byte_count" in locals()
                    else 0,
                },
            )
            raise
        except GeneratorExit:
            response_extensions = getattr(response, "extensions", None)
            if (
                isinstance(response_extensions, dict)
                and response_extensions.get(
                    "aawm_openai_responses_terminal_close_expected"
                )
                and response_extensions.get(
                    "aawm_openai_responses_terminal_pending"
                )
                and locals().get("responses_terminal_seen") is True
            ):
                metadata["aawm_responses_terminal_close_finalized"] = True
                await wait_for("terminal_bookkeeping", _finalize_completed_stream())
                raise
            local_identity = (
                transfer_identity if "transfer_identity" in locals() else {}
            )
            if is_cohere_provider_name(custom_llm_provider) or is_cohere_api_host(
                getattr(response, "url", None)
            ):
                try:
                    await wait_for(
                        "resource_cleanup", aclose_upstream_response_once(response)
                    )
                except Exception:
                    pass
            await _finalize_transfer_if_needed(
                local_identity,
                "disconnected",
                extra={
                    "error_code": "disconnect",
                    "error_class": "GeneratorExit",
                    "disconnect_reason": "client_disconnected",
                    "upstream_chunk_count": chunk_count
                    if "chunk_count" in locals()
                    else 0,
                    "upstream_byte_count": total_stream_bytes
                    if "total_stream_bytes" in locals()
                    else 0,
                    "downstream_chunk_count": downstream_chunk_count
                    if "downstream_chunk_count" in locals()
                    else 0,
                    "downstream_byte_count": downstream_byte_count
                    if "downstream_byte_count" in locals()
                    else 0,
                },
            )
            raise
        except Exception as e:
            local_chunk_count = chunk_count if "chunk_count" in locals() else 0
            local_total_stream_bytes = (
                total_stream_bytes if "total_stream_bytes" in locals() else 0
            )
            local_first_chunk_at = (
                first_chunk_at if "first_chunk_at" in locals() else None
            )
            local_first_emitted_at = (
                first_emitted_at if "first_emitted_at" in locals() else None
            )
            local_identity = (
                transfer_identity if "transfer_identity" in locals() else {}
            )
            local_downstream_chunk_count = (
                downstream_chunk_count if "downstream_chunk_count" in locals() else 0
            )
            local_downstream_byte_count = (
                downstream_byte_count if "downstream_byte_count" in locals() else 0
            )
            timeout_kind = None
            terminal_phase = "failed"
            error_code = "upstream_error"
            if isinstance(e, httpx.ReadTimeout):
                terminal_phase = "timed_out"
                error_code = "timeout"
                timeout_kind = "upstream_read"
            elif isinstance(e, (BrokenPipeError, ConnectionResetError)):
                terminal_phase = "disconnected"
                error_code = "disconnect"
            await _finalize_transfer_if_needed(
                local_identity,
                terminal_phase,
                extra={
                    "error_code": error_code,
                    "error_class": type(e).__name__,
                    "timeout_kind": timeout_kind,
                    "disconnect_reason": (
                        "client_disconnected"
                        if terminal_phase == "disconnected"
                        else None
                    ),
                    "upstream_chunk_count": local_chunk_count,
                    "upstream_byte_count": local_total_stream_bytes,
                    "downstream_chunk_count": local_downstream_chunk_count,
                    "downstream_byte_count": local_downstream_byte_count,
                },
            )
            exception_context = (
                PassThroughStreamingHandler._build_streaming_exception_log_context(
                    error_log_context=error_log_context,
                    exc=e,
                    chunk_count=local_chunk_count,
                    total_stream_bytes=local_total_stream_bytes,
                    first_chunk_at=local_first_chunk_at,
                    first_emitted_at=local_first_emitted_at,
                )
            )
            if isinstance(e, ResponsesBoundFailure):
                bound_failure = e
                failure_kind = (
                    "streaming_upstream_event_too_large"
                    if bound_failure.kind == "event_too_large"
                    else "streaming_upstream_too_large"
                )
                failure_context = dict(error_log_context or {})
                failure_context.update(
                    {
                        "failure_kind": failure_kind,
                        "error_class": type(bound_failure.cause).__name__,
                        "error_code": bound_failure.code,
                        "error_message": str(bound_failure),
                        "terminal_error_payload": dict(bound_failure.error_payload),
                        "stream_failure_stage": (
                            "stream_interrupted_after_first_byte"
                            if (
                                openai_wire_trace is not None
                                and (
                                    getattr(openai_wire_trace, "first_body_sent", False)
                                    or getattr(
                                        openai_wire_trace,
                                        "terminal_wire_committed",
                                        False,
                                    )
                                    or getattr(
                                        openai_wire_trace,
                                        "asgi_delivery_complete",
                                        False,
                                    )
                                )
                            )
                            or (
                                openai_wire_trace is None
                                and local_first_emitted_at is not None
                            )
                            else "stream_interrupted_before_emit"
                        ),
                        "stream_chunks_seen": local_chunk_count,
                        "stream_bytes_seen": local_total_stream_bytes,
                        "stream_hidden_retry_safe": False,
                        "aawm_stream_incomplete_reason": (
                            bound_failure.incomplete_reason
                        ),
                        "responses_api_terminal": True,
                    }
                )
                if isinstance(error_log_context, dict):
                    error_log_context.update(failure_context)
                metadata.update(failure_context)
                metadata["aawm_stream_interrupted"] = True
                metadata["aawm_stream_incomplete"] = True
                metadata["aawm_route_rollup_turn_suppressed"] = True
                terminal_chunks = (
                    PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
                        endpoint_type=endpoint_type,
                        url_route=url_route,
                        custom_llm_provider=custom_llm_provider,
                        failure_context=failure_context,
                        exc=bound_failure,
                        responses_event_pipeline=responses_event_pipeline,
                    )
                )
                _mark_responses_terminal_pending()
                for terminal_chunk in terminal_chunks:
                    if "_record_responses_wire_chunk" in locals():
                        _record_responses_wire_chunk(terminal_chunk)
                    yield terminal_chunk
                return
            if (
                isinstance(e, httpx.ReadTimeout)
                and local_first_emitted_at is not None
            ):
                terminal_chunks = (
                    await PassThroughStreamingHandler._terminalize_post_first_byte_stream_timeout(
                        exc=e,
                        litellm_logging_obj=litellm_logging_obj,
                        endpoint_type=endpoint_type,
                        url_route=url_route,
                        custom_llm_provider=custom_llm_provider,
                        start_time=start_time,
                        error_log_context=error_log_context,
                        success_handler_kwargs=success_handler_kwargs,
                        chunk_count=local_chunk_count,
                        total_stream_bytes=local_total_stream_bytes,
                        first_chunk_at=local_first_chunk_at,
                        first_emitted_at=local_first_emitted_at,
                        defer_failure_consumers=openai_wire_trace is not None,
                        responses_event_pipeline=responses_event_pipeline,
                    )
                )
                for terminal_chunk in terminal_chunks:
                    yield terminal_chunk
                return
            if (
                "held_responses_done_suffix" in locals()
                and held_responses_done_suffix
                and "_record_responses_wire_chunk" in locals()
            ):
                _record_responses_wire_chunk(held_responses_done_suffix)
                yield held_responses_done_suffix
            verbose_proxy_logger.exception(
                "Error in chunk_processor: %s",
                str(e),
                extra=exception_context,
            )
            raise

    @staticmethod
    async def _finalize_unstarted_native_stream(
        *,
        delivered_snapshot: Dict[str, Any],
        response: httpx.Response,
        request_body: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        start_time: datetime,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        passthrough_logging_payload: Optional[
            PassthroughStandardLoggingPayload
        ] = None,
        custom_llm_provider: Optional[str] = None,
        success_handler_kwargs: Optional[Dict[str, Any]] = None,
        local_prepare_ms: Optional[float] = None,
        error_log_context: Optional[Dict[str, Any]] = None,
        upstream_prefix_bytes: Optional[List[bytes]] = None,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
        operational_request: Optional[Any] = None,
    ) -> None:
        """Finish native stream bookkeeping when the body iterator never starts."""

        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        delivered_disposition = str(
            delivered_snapshot.get("delivered_disposition")
            or delivered_snapshot.get("disposition")
            or "disconnected"
        ).strip().lower()
        metadata["aawm_delivered_wire_disposition"] = dict(delivered_snapshot)
        metadata["aawm_delivered_disposition"] = delivered_disposition
        prefix_source: Any = upstream_prefix_bytes or []
        iter_prefix_chunks = getattr(prefix_source, "iter_chunks", None)
        if callable(iter_prefix_chunks):
            prefix_source = iter_prefix_chunks()
        prefix_chunks = [
            chunk
            for chunk in prefix_source
            if isinstance(chunk, (bytes, bytearray, memoryview)) and chunk
        ]
        upstream_chunk_count = len(prefix_chunks)
        upstream_byte_count = sum(len(chunk) for chunk in prefix_chunks)
        metadata["aawm_stream_chunk_count"] = upstream_chunk_count
        metadata["aawm_stream_total_bytes"] = upstream_byte_count
        metadata["aawm_stream_interrupted"] = delivered_disposition != "completed"
        metadata["aawm_route_rollup_turn_suppressed"] = True
        metadata["aawm_stream_unstarted"] = True

        transfer_identity = extract_transfer_identity(
            request_body=request_body if isinstance(request_body, dict) else None,
            logging_obj=litellm_logging_obj,
            kwargs=success_handler_kwargs,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
            stream_path="pass_through",
        )
        transfer_extra = {
            "delivered_disposition": delivered_disposition,
            "upstream_chunk_count": upstream_chunk_count,
            "upstream_byte_count": upstream_byte_count,
            "downstream_chunk_count": 0,
            "downstream_byte_count": 0,
        }
        await wait_for(
            "transfer_registry",
            safe_mark_phase(
                transfer_identity,
                "finalizing",
                extra=transfer_extra,
            ),
        )
        transfer_phase = {
            "completed": "completed",
            "failed": "failed",
            "incomplete": "failed",
            "cancelled": "cancelled",
            "disconnected": "disconnected",
        }.get(delivered_disposition, "failed")
        await wait_for(
            "transfer_registry",
            safe_finalize(
                transfer_identity,
                transfer_phase,
                extra=transfer_extra,
            ),
        )
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=litellm_logging_obj,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            response=response,
            url_route=url_route,
            request_body=request_body or {},
            endpoint_type=endpoint_type,
            start_time=start_time,
            raw_bytes=upstream_prefix_bytes or [],
            precomputed_lines=None,
            end_time=datetime.now(),
            passthrough_logging_payload=passthrough_logging_payload,
            custom_llm_provider=custom_llm_provider,
            success_handler_kwargs=success_handler_kwargs,
            local_prepare_ms=local_prepare_ms,
            error_log_context=error_log_context,
            responses_event_pipeline=(
                responses_event_pipeline
                or responses_event_pipeline_from_response(response)
            ),
            operational_request=operational_request,
        )

    @staticmethod
    async def _terminalize_post_first_byte_stream_timeout(
        *,
        exc: Exception,
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
        start_time: datetime,
        error_log_context: Optional[Dict[str, Any]],
        success_handler_kwargs: Optional[Dict[str, Any]],
        chunk_count: int,
        total_stream_bytes: int,
        first_chunk_at: Optional[datetime],
        first_emitted_at: Optional[datetime],
        defer_failure_consumers: bool = False,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> List[bytes]:
        failure_context = PassThroughStreamingHandler._build_streaming_failure_context(
            exc=exc,
            chunk_count=chunk_count,
            total_stream_bytes=total_stream_bytes,
            first_chunk_at=first_chunk_at,
            first_emitted_at=first_emitted_at,
        )

        if isinstance(error_log_context, dict):
            failure_context = {
                **error_log_context,
                **failure_context,
            }
            error_log_context.update(failure_context)

            exception_context = error_log_context
        else:
            exception_context = PassThroughStreamingHandler._build_streaming_exception_log_context(
                error_log_context=error_log_context,
                exc=exc,
                chunk_count=chunk_count,
                total_stream_bytes=total_stream_bytes,
                first_chunk_at=first_chunk_at,
                first_emitted_at=first_emitted_at,
            )

        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        if metadata.get("aawm_stream_terminal_emitted"):
            return []

        if not isinstance(success_handler_kwargs, dict) and isinstance(
            error_log_context, dict
        ):
            if error_log_context.get("aawm_stream_terminal_emitted"):
                return []

        metadata["aawm_stream_chunk_count"] = chunk_count
        metadata["aawm_stream_total_bytes"] = total_stream_bytes
        metadata["aawm_stream_interrupted"] = True
        metadata["aawm_stream_terminal_emitted"] = True
        metadata["aawm_route_rollup_turn_suppressed"] = True
        metadata.update(failure_context)
        if not isinstance(success_handler_kwargs, dict) and isinstance(
            error_log_context, dict
        ):
            error_log_context["aawm_stream_terminal_emitted"] = True

        if not defer_failure_consumers:
            _emit_aawm_terminal_error(
                {
                    "event_type": "responses_stream_terminal",
                    "endpoint": failure_context.get("endpoint"),
                    "alias_family": (
                        failure_context.get("alias_family")
                        or failure_context.get("model_family")
                        or failure_context.get("route_family")
                        or custom_llm_provider
                    ),
                    "alias_model": (
                        failure_context.get("model_alias")
                        or failure_context.get("requested_model_alias")
                        or failure_context.get("model")
                    ),
                    "selected_provider": failure_context.get("provider"),
                    "selected_model": failure_context.get("model"),
                    "selected_route": (
                        failure_context.get("route_family")
                        or failure_context.get("upstream_url")
                    ),
                    "status_code": failure_context.get("status_code"),
                    "error_code": failure_context.get("error_code")
                    or failure_context.get("failure_kind"),
                    "failure_class": failure_context.get("error_class")
                    or failure_context.get("failure_kind"),
                    "failure_phase": failure_context.get("stream_failure_stage"),
                    "attempted_provider_call": True,
                    "redispatch_required": metadata.get("redispatch_required")
                    is True,
                    "terminal_outcome": "failed",
                    "fallback_result": metadata.get("fallback_result") or "none",
                    "attempt_count": metadata.get("attempt_count"),
                    "correlation_id": (
                        metadata.get("litellm_call_id")
                        or metadata.get("request_id")
                        or metadata.get("trace_id")
                        or failure_context.get("litellm_call_id")
                        or failure_context.get("request_id")
                        or failure_context.get("trace_id")
                    ),
                },
                marker=metadata,
            )

        # The stream has already emitted bytes to the client, so this
        # cannot be retried or completed truthfully. Error intake and
        # route rollup above carry the terminal failure; do not send
        # partial chunks through the normal success callback pipeline.
        #
        # Still run the standard failure logging pipeline so Langfuse /
        # session_history / spend callbacks observe the mid-stream
        # timeout (success handlers must not run on partial streams).
        if not defer_failure_consumers:
            try:
                await litellm_logging_obj.async_failure_handler(
                    exception=exc,
                    traceback_exception=traceback.format_exc(),
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            except Exception as logging_exc:
                verbose_proxy_logger.exception(
                    "async_failure_handler failed after mid-stream ReadTimeout: %s",
                    str(logging_exc),
                    extra=exception_context,
                )

            PassThroughStreamingHandler._record_post_first_byte_stream_terminal_rollup(
                success_handler_kwargs=success_handler_kwargs,
                failure_context=failure_context,
                exc=exc,
            )

        return PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
            endpoint_type=endpoint_type,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
            failure_context=failure_context,
            exc=exc,
            responses_event_pipeline=responses_event_pipeline,
        )

    @staticmethod
    async def _terminalize_post_first_byte_responses_clean_eof(
        *,
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
        start_time: datetime,
        error_log_context: Optional[Dict[str, Any]],
        success_handler_kwargs: Optional[Dict[str, Any]],
        chunk_count: int,
        total_stream_bytes: int,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> List[bytes]:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        if metadata.get("aawm_stream_terminal_emitted"):
            return []
        if not isinstance(success_handler_kwargs, dict) and isinstance(
            error_log_context, dict
        ):
            if error_log_context.get("aawm_stream_terminal_emitted"):
                return []

        failure_context = dict(error_log_context or {})
        failure_context.update(
            {
                "failure_kind": "streaming_upstream_clean_eof",
                "stream_failure_stage": "stream_interrupted_after_first_byte",
                "stream_chunks_seen": chunk_count,
                "stream_bytes_seen": total_stream_bytes,
                "stream_hidden_retry_safe": False,
            }
        )
        if isinstance(error_log_context, dict):
            error_log_context.update(failure_context)

        metadata.update(failure_context)
        metadata["aawm_stream_chunk_count"] = chunk_count
        metadata["aawm_stream_total_bytes"] = total_stream_bytes
        metadata["aawm_stream_interrupted"] = True
        metadata["aawm_stream_incomplete"] = True
        metadata["aawm_stream_incomplete_reason"] = (
            PassThroughStreamingHandler._CLEAN_EOF_INCOMPLETE_REASON
        )
        metadata["aawm_stream_terminal_emitted"] = True
        metadata["aawm_stream_replayable"] = False
        metadata["aawm_route_rollup_turn_suppressed"] = True
        if not isinstance(success_handler_kwargs, dict) and isinstance(
            error_log_context, dict
        ):
            error_log_context.update(
                {
                    "aawm_stream_interrupted": True,
                    "aawm_stream_incomplete": True,
                    "aawm_stream_terminal_emitted": True,
                    "aawm_stream_replayable": False,
                    "aawm_route_rollup_turn_suppressed": True,
                }
            )

        clean_eof_exc = RuntimeError(
            PassThroughStreamingHandler._CLEAN_EOF_INCOMPLETE_REASON
        )
        _emit_aawm_terminal_error(
            {
                "event_type": "responses_stream_terminal",
                "endpoint": failure_context.get("endpoint"),
                "alias_family": (
                    failure_context.get("alias_family")
                    or failure_context.get("model_family")
                    or failure_context.get("route_family")
                    or custom_llm_provider
                ),
                "alias_model": (
                    failure_context.get("model_alias")
                    or failure_context.get("requested_model_alias")
                    or failure_context.get("model")
                ),
                "selected_provider": failure_context.get("provider"),
                "selected_model": failure_context.get("model"),
                "selected_route": (
                    failure_context.get("route_family")
                    or failure_context.get("upstream_url")
                ),
                "status_code": failure_context.get("status_code"),
                "error_code": "streaming_upstream_clean_eof",
                "failure_class": "streaming_upstream_clean_eof",
                "failure_phase": failure_context.get("stream_failure_stage"),
                "attempted_provider_call": True,
                "redispatch_required": metadata.get("redispatch_required") is True,
                "terminal_outcome": "incomplete",
                "fallback_result": metadata.get("fallback_result") or "none",
                "attempt_count": metadata.get("attempt_count"),
                "correlation_id": (
                    metadata.get("litellm_call_id")
                    or metadata.get("request_id")
                    or metadata.get("trace_id")
                    or failure_context.get("litellm_call_id")
                    or failure_context.get("request_id")
                    or failure_context.get("trace_id")
                ),
            },
            marker=metadata,
        )
        try:
            await litellm_logging_obj.async_failure_handler(
                exception=clean_eof_exc,
                traceback_exception=None,
                start_time=start_time,
                end_time=datetime.now(),
            )
        except Exception as logging_exc:
            verbose_proxy_logger.exception(
                "async_failure_handler failed after clean Responses EOF: %s",
                str(logging_exc),
                extra=error_log_context or failure_context,
            )

        PassThroughStreamingHandler._record_post_first_byte_stream_terminal_rollup(
            success_handler_kwargs=success_handler_kwargs,
            failure_context=failure_context,
            exc=clean_eof_exc,
        )

        return PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
            endpoint_type=endpoint_type,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
            failure_context=failure_context,
            incomplete_reason=PassThroughStreamingHandler._CLEAN_EOF_INCOMPLETE_REASON,
            responses_event_pipeline=responses_event_pipeline,
        )

    @staticmethod
    def _build_streaming_exception_log_context(
        *,
        error_log_context: Optional[Dict[str, Any]],
        exc: Exception,
        chunk_count: int,
        total_stream_bytes: int,
        first_chunk_at: Optional[datetime],
        first_emitted_at: Optional[datetime],
    ) -> Dict[str, Any]:
        failure_context = dict(error_log_context or {})
        failure_context.update(
            PassThroughStreamingHandler._build_streaming_failure_context(
                exc=exc,
                chunk_count=chunk_count,
                total_stream_bytes=total_stream_bytes,
                first_chunk_at=first_chunk_at,
                first_emitted_at=first_emitted_at,
            )
        )
        return failure_context

    @staticmethod
    def _build_streaming_failure_context(
        *,
        exc: Exception,
        chunk_count: int,
        total_stream_bytes: int,
        first_chunk_at: Optional[datetime],
        first_emitted_at: Optional[datetime],
    ) -> Dict[str, Any]:
        if first_emitted_at is not None:
            failure_stage = "stream_interrupted_after_first_byte"
        elif first_chunk_at is not None:
            failure_stage = "stream_interrupted_before_emit"
        else:
            failure_stage = "stream_interrupted_before_first_chunk"

        context: Dict[str, Any] = {
            "failure_kind": "streaming_upstream_read_failure",
            "stream_failure_stage": failure_stage,
            "stream_chunks_seen": chunk_count,
            "stream_bytes_seen": total_stream_bytes,
            "stream_hidden_retry_safe": False,
        }
        if isinstance(exc, httpx.ReadTimeout):
            context["failure_kind"] = "streaming_upstream_read_timeout"
            context["status_code"] = 504
            context["recommended_operator_action"] = (
                "Treat as terminal upstream stream timeout after bytes were already "
                "emitted; do not hidden-retry this request. Inspect provider/model "
                "health and stream progress counters before redispatching a new turn."
            )
        return context

    @staticmethod
    def _build_responses_stream_redispatch_failure_context(
        *,
        failure: ResponsesStreamPreCommitFailure,
        request_body: Optional[dict],
        success_handler_kwargs: Optional[Dict[str, Any]],
        url_route: str,
        custom_llm_provider: Optional[str],
        chunk_count: int,
        total_stream_bytes: int,
    ) -> Dict[str, Any]:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        selected_provider = (
            metadata.get("codex_auto_agent_selected_provider")
            or custom_llm_provider
            or "openai"
        )
        selected_model = (
            metadata.get("codex_auto_agent_selected_model")
            or (request_body or {}).get("model")
        )
        selected_route_family = (
            metadata.get("codex_auto_agent_selected_route_family")
            or metadata.get("route_family")
            or "codex_responses"
        )
        candidate = {
            "provider": selected_provider,
            "model": selected_model,
            "route_family": selected_route_family,
            "codex_oauth_account_label": metadata.get(
                "codex_auto_agent_selected_account_label"
            ),
            "codex_oauth_account_hash": metadata.get(
                "codex_auto_agent_selected_account_hash"
            ),
            "codex_oauth_lane_key": metadata.get(
                "codex_auto_agent_selected_account_lane"
            ),
        }
        alias_model = (
            metadata.get("model_alias_label")
            or metadata.get("requested_model_alias")
            or metadata.get("codex_auto_agent_alias")
            or (request_body or {}).get("model")
        )
        session_identity = (
            metadata.get("canonical_session_identity")
            or metadata.get("session_id")
        )
        has_continuation_state = bool(
            isinstance(request_body, dict)
            and request_body.get("previous_response_id")
        ) or alias_model == "codex-auto-review"
        if failure.error_class == "token_invalidated":
            failure_phase = (
                "token_invalidated_continuation"
                if has_continuation_state
                else "token_invalidated_stream"
            )
            message = (
                "Codex Responses stream authentication was invalidated after "
                "output began; redispatch a fresh response."
            )
        else:
            failure_phase = (
                "openai_responses_unpersisted_item_not_found_continuation"
                if has_continuation_state
                else "openai_responses_unpersisted_item_not_found_stream"
            )
            message = (
                "OpenAI Responses continuation referenced an unpersisted item "
                "(store=false). Do not continue this session; redispatch a "
                "fresh response."
            )

        from fastapi import HTTPException

        from .aawm_alias_routing import session_affinity

        try:
            session_affinity.raise_session_owner_redispatch_required(
                session_identity=(
                    str(session_identity) if session_identity is not None else None
                ),
                alias_model=(
                    str(alias_model) if alias_model is not None else None
                ),
                candidate=candidate,
                failure_phase=failure_phase,
                message=message,
                attempted_provider_call=True,
                terminal_marker=metadata,
            )
        except HTTPException as redispatch_exc:
            detail = redispatch_exc.detail
        else:
            detail = {}

        if not isinstance(detail, dict):
            detail = {}
        contract_error = detail.get("error")
        if not isinstance(contract_error, dict):
            contract_error = {
                "message": message,
                "type": "invalid_request_error",
                "code": "aawm_session_owner_redispatch_required",
            }
        terminal_error_payload = {
            "message": str(contract_error.get("message") or message),
            "type": str(contract_error.get("type") or "invalid_request_error"),
            "code": str(
                contract_error.get("code")
                or "aawm_session_owner_redispatch_required"
            ),
            "param": contract_error.get("param"),
            "redispatch_required": True,
        }
        terminal_metadata = {
            "redispatch_required": True,
            "redispatch_reason": detail.get("redispatch_reason") or failure_phase,
            "failure_phase": detail.get("failure_phase") or failure_phase,
            "attempted_provider_call": True,
            "canonical_session_identity": detail.get(
                "canonical_session_identity"
            )
            or session_identity,
        }
        return {
            "failure_kind": "streaming_session_owner_redispatch_required",
            "error_class": failure.error_class,
            "error_code": terminal_error_payload["code"],
            "error_message": terminal_error_payload["message"],
            "stream_failure_stage": "stream_interrupted_after_first_byte",
            "stream_chunks_seen": chunk_count,
            "stream_bytes_seen": total_stream_bytes,
            "stream_hidden_retry_safe": False,
            "responses_api_terminal": True,
            "model": selected_model,
            "model_alias": alias_model,
            "route_family": selected_route_family,
            "terminal_error_payload": terminal_error_payload,
            "terminal_metadata": terminal_metadata,
            "source_url": url_route,
        }

    @staticmethod
    def _attach_responses_terminal_events(
        chunks: List[bytes],
        *,
        event_pipeline: Optional[ResponsesEventPipeline],
        event_type: str,
        payload: Optional[Dict[str, Any]],
    ) -> List[bytes]:
        """Associate generated Responses terminal bytes with the attempt owner."""
        if event_pipeline is None or not chunks:
            return chunks

        carried = list(chunks)
        try:
            terminal_event = event_pipeline.adopt_event(
                payload,
                event_type=event_type,
                wire_event=carried[0],
                control=True,
            )
            carried[0] = attach_responses_events(
                carried[0],
                [terminal_event],
                replace=True,
            )
            if len(carried) > 1 and PassThroughStreamingHandler._is_done_chunk(
                carried[1]
            ):
                done_event = event_pipeline.adopt_event(
                    None,
                    is_done_marker=True,
                    wire_event=carried[1],
                    control=True,
                )
                carried[1] = attach_responses_events(
                    carried[1],
                    [done_event],
                    replace=True,
                )
        except (
            ResponsesEventRetentionTooLarge,
            ResponsesEventTooLarge,
            ResponsesStreamTooLarge,
        ):
            # Generated terminals use reserved control capacity. If even that
            # bound is exhausted, keep the already-associated prefix.
            return carried
        return carried

    @staticmethod
    def _build_post_first_byte_terminal_stream_chunks(
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
        failure_context: Dict[str, Any],
        exc: Optional[Exception] = None,
        incomplete_reason: Optional[str] = None,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> List[bytes]:
        legacy_terminal_metadata = {
            "stream_failure_stage": failure_context.get("stream_failure_stage"),
            "stream_chunks_seen": failure_context.get("stream_chunks_seen"),
            "stream_bytes_seen": failure_context.get("stream_bytes_seen"),
            "stream_hidden_retry_safe": failure_context.get("stream_hidden_retry_safe"),
            "provider": custom_llm_provider or endpoint_type.value,
            "model": failure_context.get("model"),
            "model_alias": failure_context.get("model_alias"),
            "route_family": failure_context.get("route_family"),
        }
        extra_terminal_metadata = failure_context.get("terminal_metadata")
        if isinstance(extra_terminal_metadata, dict):
            legacy_terminal_metadata = {
                **legacy_terminal_metadata,
                **extra_terminal_metadata,
            }
        if incomplete_reason is not None:
            terminal_metadata = {
                **legacy_terminal_metadata,
                "request_id": failure_context.get("request_id"),
                "litellm_call_id": failure_context.get("litellm_call_id"),
                "trace_id": failure_context.get("trace_id"),
                "session_id": failure_context.get("session_id"),
                "stream_last_emission_at": failure_context.get(
                    "stream_last_emission_at"
                ),
                "stream_idle_ms": failure_context.get("stream_idle_ms"),
            }
            payload = {
                "type": "response.incomplete",
                "response": {
                    "object": "response",
                    "status": "incomplete",
                    "error": None,
                    "incomplete_details": {"reason": incomplete_reason},
                    "output": [],
                    "metadata": terminal_metadata,
                },
            }
            chunks = [
                (
                    "event: response.incomplete\ndata: "
                    + json.dumps(payload, separators=(",", ":"))
                    + "\n\n"
                ).encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
            return PassThroughStreamingHandler._attach_responses_terminal_events(
                chunks,
                event_pipeline=responses_event_pipeline,
                event_type="response.incomplete",
                payload=payload,
            )

        custom_error_payload = failure_context.get("terminal_error_payload")
        if isinstance(custom_error_payload, dict):
            error_payload = dict(custom_error_payload)
            message = error_payload.get("message")
            if not isinstance(message, str) or not message:
                message = failure_context.get("error_message")
        else:
            message = failure_context.get("error_message")
            error_payload = {}
        if not message:
            if exc is None:
                raise ValueError("exc is required for failed stream terminal chunks")
            message = (
                "Streaming response interrupted after first byte due to upstream read "
                f"timeout: {exc}"
            )
        error_payload.setdefault("type", "proxy_stream_terminal_error")
        error_payload.setdefault(
            "code",
            failure_context.get("error_code")
            or failure_context.get("failure_kind")
            or "streaming_upstream_read_failure",
        )
        error_payload.setdefault("message", message)
        error_payload.setdefault("param", None)

        if endpoint_type == EndpointType.ANTHROPIC:
            payload = {
                "type": "error",
                "error": {
                    "type": error_payload["type"],
                    "message": message,
                },
            }
            return [
                (
                    "event: error\ndata: "
                    + json.dumps(payload, separators=(",", ":"))
                    + "\n\n"
                ).encode("utf-8")
            ]

        if failure_context.get("responses_api_terminal") or (
            endpoint_type == EndpointType.OPENAI
            and OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        ):
            payload = {
                "type": "response.failed",
                "response": {
                    "object": "response",
                    "status": "failed",
                    "error": error_payload,
                    "metadata": legacy_terminal_metadata,
                },
            }
            chunks = [
                (
                    "event: response.failed\ndata: "
                    + json.dumps(payload, separators=(",", ":"))
                    + "\n\n"
                ).encode("utf-8")
            ]
            chunks.append(b"data: [DONE]\n\n")
            return PassThroughStreamingHandler._attach_responses_terminal_events(
                chunks,
                event_pipeline=responses_event_pipeline,
                event_type="response.failed",
                payload=payload,
            )

        payload = {
            "error": error_payload,
            "aawm_stream_terminal": legacy_terminal_metadata,
        }
        return [
            (
                "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"
            ).encode("utf-8")
        ]

    @staticmethod
    def _record_post_first_byte_stream_terminal_rollup(
        *,
        success_handler_kwargs: Optional[Dict[str, Any]],
        failure_context: Dict[str, Any],
        exc: Exception,
    ) -> None:
        if not isinstance(success_handler_kwargs, dict):
            return
        litellm_params = success_handler_kwargs.get("litellm_params")
        metadata = (
            litellm_params.get("metadata")
            if isinstance(litellm_params, dict)
            and isinstance(litellm_params.get("metadata"), dict)
            else None
        )
        if not isinstance(metadata, dict):
            return
        context = metadata.get("aawm_route_rollup_context")
        if not isinstance(context, dict):
            return

        model_label = (
            failure_context.get("model_alias")
            or failure_context.get("model")
            or context.get("model_label")
            or "unknown-model"
        )
        classification = (
            failure_context.get("error_class")
            or failure_context.get("failure_kind")
        )
        detail_parts = [
            f"stream_failure_stage={failure_context.get('stream_failure_stage')}",
            f"stream_chunks_seen={failure_context.get('stream_chunks_seen')}",
            f"stream_bytes_seen={failure_context.get('stream_bytes_seen')}",
            f"failure_kind={failure_context.get('failure_kind')}",
        ]
        if classification:
            detail_parts.append(f"classification={classification}")
        detail_parts.append(f"message={exc}")
        detail = "; ".join(detail_parts)
        emit_aawm_route_status_event(
            alias_model=failure_context.get("model_alias") or model_label,
            model_label=str(model_label),
            status="Failed",
            message=detail,
        )
        record_aawm_route_rollup(
            group_header_label=str(context.get("group_header_label") or ""),
            incoming_endpoint=str(context.get("incoming_endpoint") or ""),
            outgoing_target=str(context.get("outgoing_target") or ""),
            model_label=str(model_label),
            effort=str(context.get("reasoning_effort") or "none"),
            turns=0,
            status="Failed",
            observation_context=context,
        )

    @staticmethod
    def _set_streaming_handler_branch(
        handler_branch_state: List[str],
        handler_branch: str,
    ) -> str:
        handler_branch_state[0] = handler_branch
        return handler_branch

    @staticmethod
    def _collect_streaming_logging_result(
        *,
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        response: Optional[httpx.Response],
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        raw_bytes: List[bytes],
        end_time: datetime,
        model: Optional[str],
        passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload],
        custom_llm_provider: Optional[str],
        kwargs: Dict[str, Any],
        handler_branch_state: List[str],
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> tuple[
        Optional[PassThroughEndpointLoggingResultValues],
        Dict[str, Any],
        str,
        bool,
    ]:
        set_branch = PassThroughStreamingHandler._set_streaming_handler_branch
        handler_branch = set_branch(handler_branch_state, "initial")
        standard_logging_response_object: Optional[
            PassThroughEndpointLoggingResultValues
        ] = None
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(kwargs)
        PassThroughStreamingHandler._annotate_stream_logging_metadata(
            metadata,
            endpoint_type=endpoint_type,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
        )
        PassThroughStreamingHandler._capture_stream_shape(
            response=response,
            endpoint_type=endpoint_type,
            url_route=url_route,
            request_body=request_body,
            raw_bytes=raw_bytes,
            all_chunks=all_chunks,
            metadata=metadata,
            custom_llm_provider=custom_llm_provider,
            litellm_call_id=kwargs.get("litellm_call_id")
            or getattr(litellm_logging_obj, "litellm_call_id", None),
        )

        if endpoint_type == EndpointType.OPENAI:
            handler_branch = set_branch(handler_branch_state, "openai")
            try:
                openai_passthrough_logging_handler_result = (
                    OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                        litellm_logging_obj=litellm_logging_obj,
                        passthrough_success_handler_obj=passthrough_success_handler_obj,
                        url_route=url_route,
                        request_body=request_body,
                        endpoint_type=endpoint_type,
                        start_time=start_time,
                        all_chunks=all_chunks,
                        end_time=end_time,
                        kwargs=kwargs,
                        responses_event_pipeline=responses_event_pipeline,
                    )
                )
            except TypeError:
                openai_passthrough_logging_handler_result = (
                    OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                        litellm_logging_obj=litellm_logging_obj,
                        passthrough_success_handler_obj=passthrough_success_handler_obj,
                        url_route=url_route,
                        request_body=request_body,
                        endpoint_type=endpoint_type,
                        start_time=start_time,
                        all_chunks=all_chunks,
                        end_time=end_time,
                        kwargs=kwargs,
                    )
                )
            standard_logging_response_object = (
                openai_passthrough_logging_handler_result["result"]
            )
            kwargs.update(openai_passthrough_logging_handler_result["kwargs"])
        elif endpoint_type == EndpointType.ANTHROPIC:
            handler_branch = set_branch(handler_branch_state, "anthropic")
            anthropic_passthrough_logging_handler_result = (
                AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                    passthrough_logging_payload=passthrough_logging_payload,
                    kwargs=kwargs,
                )
            )
            standard_logging_response_object = (
                anthropic_passthrough_logging_handler_result["result"]
            )
            kwargs.update(anthropic_passthrough_logging_handler_result["kwargs"])
            metadata = PassThroughStreamingHandler._ensure_streaming_metadata(kwargs)
            if metadata.get("aawm_upstream_stream_degraded") is True:
                return None, kwargs, handler_branch, True
        elif endpoint_type == EndpointType.VERTEX_AI:
            handler_branch = set_branch(handler_branch_state, "vertex")
            vertex_passthrough_logging_handler_result = (
                VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                    model=model,
                )
            )
            standard_logging_response_object = (
                vertex_passthrough_logging_handler_result["result"]
            )
            kwargs.update(vertex_passthrough_logging_handler_result["kwargs"])

        return standard_logging_response_object, kwargs, handler_branch, False

    @staticmethod
    def _build_completed_responses_body_for_route_rollup(
        *,
        all_chunks: List[str],
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> Optional[Dict[str, Any]]:
        if not PassThroughStreamingHandler._is_openai_responses_stream(
            endpoint_type=endpoint_type,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
        ):
            return None

        summary = (
            responses_event_pipeline.summary_snapshot()
            if responses_event_pipeline is not None
            else None
        )
        terminal_event_type = (
            summary.get("terminal_event_type") if isinstance(summary, dict) else None
        )
        terminal_payload = (
            summary.get("terminal_response_payload")
            if isinstance(summary, dict)
            else None
        )
        if terminal_event_type is None:
            terminal_event_type, terminal_payload = (
                OpenAIPassthroughLoggingHandler._extract_terminal_response_payload_from_stream(
                    all_chunks,
                    responses_event_pipeline=responses_event_pipeline,
                )
            )
        if (
            terminal_event_type != "response.completed"
            or not isinstance(terminal_payload, dict)
            or terminal_payload.get("status") != "completed"
        ):
            return None

        completed_output = terminal_payload.get("output")
        reconstructed_output = (
            summary.get("output")
            if isinstance(summary, dict) and isinstance(summary.get("output"), list)
            else OpenAIPassthroughLoggingHandler._reconstruct_responses_output_items_from_stream(
                all_chunks,
                responses_event_pipeline=responses_event_pipeline,
            )
        )
        merged_output = OpenAIPassthroughLoggingHandler._merge_responses_output_lists(
            completed_output if isinstance(completed_output, list) else [],
            reconstructed_output,
        )
        streamed_output_text = (
            summary.get("stream_text")
            if isinstance(summary, dict) and isinstance(summary.get("stream_text"), str)
            else OpenAIPassthroughLoggingHandler._extract_responses_api_stream_text(
                all_chunks,
                responses_event_pipeline=responses_event_pipeline,
            )
        )
        if streamed_output_text:
            assistant_messages = [
                item
                for item in merged_output
                if isinstance(item, dict)
                and item.get("type") == "message"
                and item.get("role") == "assistant"
            ]
            output_texts = [
                content_item
                for item in assistant_messages
                for content_item in (
                    item.get("content")
                    if isinstance(item.get("content"), list)
                    else []
                )
                if isinstance(content_item, dict)
                and content_item.get("type") == "output_text"
            ]
            if not output_texts:
                if len(assistant_messages) == 1:
                    assistant_messages[0]["content"] = [
                        {
                            "type": "output_text",
                            "text": streamed_output_text,
                        }
                    ]
                else:
                    merged_output.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": streamed_output_text,
                                }
                            ],
                        }
                    )

        return {
            **terminal_payload,
            "output": merged_output,
        }

    @staticmethod
    def _record_streaming_finalize_metrics(
        *,
        kwargs: Dict[str, Any],
        finalize_started_at: datetime,
        local_prepare_ms: Optional[float],
    ) -> datetime:
        finalize_completed_at = datetime.now()
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(kwargs)
        local_stream_finalize_ms = round(
            max(
                0.0,
                (finalize_completed_at - finalize_started_at).total_seconds() * 1000.0,
            ),
            3,
        )
        metadata["aawm_local_stream_finalize_ms"] = local_stream_finalize_ms
        metadata["aawm_total_proxy_overhead_ms"] = round(
            (local_prepare_ms or 0.0)
            + float(metadata.get("aawm_stream_emit_gap_ms") or 0.0)
            + local_stream_finalize_ms,
            3,
        )
        PassThroughStreamingHandler._append_stream_span(
            kwargs,
            name="proxy.post_response_finalize",
            start_time=finalize_started_at,
            end_time=finalize_completed_at,
            span_metadata={
                "duration_ms": local_stream_finalize_ms,
                "stream": True,
            },
        )
        return finalize_completed_at

    @staticmethod
    async def _dispatch_streaming_success_callbacks(
        *,
        litellm_logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: PassThroughEndpointLoggingResultValues,
        start_time: datetime,
        end_time: datetime,
        kwargs: Dict[str, Any],
        handler_branch_state: List[str],
    ) -> str:
        handler_branch = PassThroughStreamingHandler._set_streaming_handler_branch(
            handler_branch_state,
            "async_success_handler",
        )
        await litellm_logging_obj.async_success_handler(
            result=standard_logging_response_object,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )
        if litellm_logging_obj._should_run_sync_callbacks_for_async_calls() is False:
            return handler_branch

        handler_branch = PassThroughStreamingHandler._set_streaming_handler_branch(
            handler_branch_state,
            "sync_success_handler_submit",
        )
        executor.submit(
            litellm_logging_obj.success_handler,
            result=standard_logging_response_object,
            end_time=end_time,
            cache_hit=False,
            start_time=start_time,
            **kwargs,
        )
        return handler_branch

    @staticmethod
    def _reconcile_responses_stream_error_payload(
        *,
        all_chunks: List[str],
        terminal_payload: Optional[Dict[str, Any]],
        terminal_event_type: Optional[str] = None,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> Optional[Dict[str, Any]]:
        first_error: Optional[Dict[str, Any]] = None
        failed_error: Optional[Dict[str, Any]] = None
        if responses_event_pipeline is not None:
            summary = responses_event_pipeline.summary_snapshot()
            if isinstance(summary, dict):
                first_error = summary.get("first_error_payload")
                failed_error = summary.get("failed_error_payload")
        else:
            events = PassThroughStreamingHandler._iter_responses_sse_events(all_chunks)
            for event_type, payload in events:
                extracted = (
                    PassThroughStreamingHandler._extract_responses_stream_error_payload(
                        event_type,
                        payload,
                    )
                )
                if extracted is None:
                    continue
                if event_type == "error" and first_error is None:
                    first_error = extracted
                if event_type == "response.failed" and failed_error is None:
                    failed_error = extracted

        if isinstance(terminal_payload, dict) and isinstance(
            terminal_payload.get("error"), dict
        ):
            failed_error = failed_error or terminal_payload.get("error")
        elif terminal_event_type == "response.failed" and isinstance(
            terminal_payload, dict
        ):
            failed_error = failed_error or (
                PassThroughStreamingHandler._extract_responses_stream_error_payload(
                    terminal_event_type,
                    {"response": terminal_payload},
                )
            )
        if first_error is None:
            return failed_error
        if failed_error is None:
            return first_error
        first_code = first_error.get("code") if isinstance(first_error, dict) else None
        failed_code = failed_error.get("code") if isinstance(failed_error, dict) else None
        if first_code and failed_code and first_code == failed_code:
            return failed_error
        first_type = first_error.get("type") if isinstance(first_error, dict) else None
        failed_type = failed_error.get("type") if isinstance(failed_error, dict) else None
        if first_type and failed_type and first_type == failed_type:
            return failed_error
        return failed_error or first_error

    @staticmethod
    # Keep failure publication and provider evidence in one ordered finalizer.
    async def _finalize_failed_responses_stream(  # noqa: PLR0915
        *,
        litellm_logging_obj: LiteLLMLoggingObj,
        kwargs: Dict[str, Any],
        metadata: Dict[str, Any],
        all_chunks: List[str],
        request_body: dict,
        start_time: datetime,
        end_time: datetime,
        terminal_event_type: Optional[str],
        terminal_payload: Optional[Dict[str, Any]],
        handler_branch_state: List[str],
        delivered_wire_disposition: Optional[Dict[str, Any]] = None,
        openai_alpha_capacity_retry_enabled: bool = False,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
    ) -> None:
        metadata["aawm_route_rollup_turn_suppressed"] = True
        metadata["aawm_stream_interrupted"] = True
        metadata["aawm_responses_stream_failed"] = True
        delivered_disposition = (
            str(
                delivered_wire_disposition.get("delivered_disposition")
                or delivered_wire_disposition.get("disposition")
                or ""
            )
            .strip()
            .lower()
            if isinstance(delivered_wire_disposition, dict)
            else None
        )
        policy_failure_kind = (
            str(delivered_wire_disposition.get("policy_failure_kind") or "").strip()
            if isinstance(delivered_wire_disposition, dict)
            else ""
        )
        policy_failure_code = (
            str(delivered_wire_disposition.get("policy_failure_code") or "").strip()
            if isinstance(delivered_wire_disposition, dict)
            else ""
        )
        policy_failure_class = (
            str(delivered_wire_disposition.get("policy_failure_class") or "").strip()
            if isinstance(delivered_wire_disposition, dict)
            else ""
        )
        error_payload = PassThroughStreamingHandler._reconcile_responses_stream_error_payload(
            all_chunks=all_chunks,
            terminal_payload=terminal_payload,
            terminal_event_type=terminal_event_type,
            responses_event_pipeline=responses_event_pipeline,
        )
        recognized_terminal = (
            terminal_event_type in _RECOGNIZED_TERMINAL_METADATA_EVENTS
        )
        reconciled_provider_error = isinstance(error_payload, dict)
        metadata["aawm_provider_terminal_event_type"] = (
            terminal_event_type if recognized_terminal else None
        )
        if isinstance(terminal_payload, dict):
            metadata["aawm_provider_terminal_payload"] = dict(terminal_payload)
        error_class, classification, retryable = (
            PassThroughStreamingHandler._classify_responses_pre_commit_error(
                error_payload,
                openai_alpha_capacity_retry_enabled=(
                    openai_alpha_capacity_retry_enabled
                ),
            )
        )
        # Keep canonical, non-secret provider identity on the persisted marker.
        # The terminal emitter also carries these fields, but audit consumers
        # read the marker context when a client cancels a failed stream.
        route_context = metadata.get("aawm_route_rollup_context")
        if not isinstance(route_context, dict):
            route_context = {}
        selected_provider = (
            metadata.get("codex_auto_agent_selected_provider")
            or metadata.get("provider")
            or route_context.get("provider")
            or "openai"
        )
        selected_model = (
            metadata.get("codex_auto_agent_selected_model")
            or metadata.get("model")
            or request_body.get("model")
            if isinstance(request_body, dict)
            else metadata.get("model")
        )
        selected_route = (
            metadata.get("codex_auto_agent_selected_route_family")
            or metadata.get("route_family")
            or route_context.get("outgoing_target")
            or "codex_responses"
        )
        metadata.setdefault("provider", selected_provider)
        if selected_model not in (None, ""):
            metadata.setdefault("model", selected_model)
        metadata.setdefault("route_family", selected_route)
        metadata.setdefault("attempted_provider_call", True)
        if isinstance(error_payload, dict):
            payload_status = error_payload.get("status_code")
            if isinstance(payload_status, int):
                metadata.setdefault("upstream_status_code", payload_status)
        local_failure_origin: Optional[str] = None
        if policy_failure_kind or policy_failure_code or policy_failure_class:
            if isinstance(error_payload, dict):
                metadata["aawm_provider_terminal_error_payload"] = dict(
                    error_payload
                )
            error_payload = {
                "type": policy_failure_class
                or policy_failure_kind
                or "output_policy",
                "code": policy_failure_code
                or policy_failure_kind
                or "output_policy_failure",
                "message": (
                    "OpenAI Responses output policy rejected the delivered "
                    f"stream: {policy_failure_kind or policy_failure_code}"
                ),
            }
            error_class = (
                policy_failure_class
                or policy_failure_kind
                or policy_failure_code
                or "output_policy_failure"
            )
            classification = (
                policy_failure_kind
                or policy_failure_class
                or policy_failure_code
                or "output_policy_failure"
            )
            retryable = False
            local_failure_origin = "local_policy"
        elif (
            delivered_disposition
            and delivered_disposition != "completed"
            and not (
                delivered_disposition == "cancelled"
                and terminal_event_type is None
                and not policy_failure_kind
                and not policy_failure_code
                and not policy_failure_class
            )
        ):
            if error_payload is None:
                error_class = f"delivered_{delivered_disposition}"
                classification = error_class
                retryable = False
        elif (
            delivered_disposition == "cancelled"
            and terminal_event_type is None
            and not reconciled_provider_error
        ):
            error_class = "stream_interrupted"
            classification = "transient_stream_interruption"
            retryable = True
            local_failure_origin = "local_missing_terminal"
        if (
            local_failure_origin is None
            and not recognized_terminal
            and not reconciled_provider_error
        ):
            local_failure_origin = "local_missing_terminal"
        provider_authored = bool(
            recognized_terminal
            or (reconciled_provider_error and local_failure_origin is None)
        )
        metadata["provider_returned"] = provider_authored
        metadata["aawm_failure_origin"] = (
            local_failure_origin
            if local_failure_origin is not None
            else ("provider" if provider_authored else "local_missing_terminal")
        )
        sanitized_message = None
        if isinstance(error_payload, dict):
            sanitized_message = (
                OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_error_for_logging(
                    error_payload
                )
            )
        if not sanitized_message:
            sanitized_message = (
                f"delivered_disposition={delivered_disposition}"
                if delivered_disposition
                else classification
            )
        metadata["aawm_responses_stream_failure_class"] = error_class
        metadata["aawm_responses_stream_failure_classification"] = classification
        metadata["aawm_responses_stream_failure_retryable"] = retryable
        if retryable and isinstance(error_payload, dict):
            # Preserve the retry contract in the wire error for orchestrators.
            error_payload.setdefault("retryable", True)
        if delivered_disposition:
            metadata["aawm_delivered_disposition"] = delivered_disposition
            metadata["aawm_route_rollup_turn_suppressed"] = True
        if policy_failure_kind:
            metadata["aawm_policy_failure_kind"] = policy_failure_kind
        if policy_failure_code:
            metadata["aawm_policy_failure_code"] = policy_failure_code
        if policy_failure_class:
            metadata["aawm_policy_failure_class"] = policy_failure_class
        extracted_code = None
        extracted_type = None
        if isinstance(error_payload, dict):
            extracted_code, extracted_type, _extracted_message = (
                PassThroughStreamingHandler._extract_responses_stream_error_fields(
                    error_payload
                )
            )
        failure_exc = ResponsesStreamPreCommitFailure(
            error_class=error_class,
            classification=classification,
            retryable=retryable,
            error_code=extracted_code,
            error_type=extracted_type,
            error_payload=error_payload if isinstance(error_payload, dict) else None,
            message=sanitized_message,
            provider_returned=provider_authored,
            failure_origin=metadata["aawm_failure_origin"],
        )
        if policy_failure_kind:
            setattr(failure_exc, "policy_failure_kind", policy_failure_kind)
        if policy_failure_code:
            setattr(failure_exc, "policy_failure_code", policy_failure_code)
        if policy_failure_class:
            setattr(failure_exc, "policy_failure_class", policy_failure_class)
        if (
            delivered_disposition is not None
            and delivered_disposition != "completed"
            and not (
                delivered_disposition == "cancelled"
                and terminal_event_type is None
                and retryable
            )
        ):
            failure_exc.retryable = False
            failure_exc.pre_commit_retry_exhausted = True
        failure_context = {
            "failure_kind": classification,
            "stream_failure_stage": "responses_stream_failed",
            "delivered_disposition": delivered_disposition,
            "error_class": error_class,
            "policy_failure_kind": policy_failure_kind or None,
            "policy_failure_code": policy_failure_code or None,
            "policy_failure_class": policy_failure_class or None,
            "provider_terminal_event_type": (
                terminal_event_type if recognized_terminal else None
            ),
            "failure_origin": metadata.get("aawm_failure_origin"),
            "model": request_body.get("model") if isinstance(request_body, dict) else None,
        }
        metadata.update(failure_context)
        route_context = metadata.get("aawm_route_rollup_context")
        if not isinstance(route_context, dict):
            route_context = {}
        redispatch_required = metadata.get("redispatch_required") is True
        error_code = None
        if isinstance(error_payload, dict):
            error_code = error_payload.get("code") or error_payload.get("type")
        _emit_aawm_terminal_error(
            {
                "event_type": (
                    policy_failure_kind
                    or policy_failure_code
                    or (terminal_event_type if recognized_terminal else None)
                    or metadata.get("aawm_failure_origin")
                    or "local_missing_terminal"
                ),
                "provider_terminal_event_type": (
                    terminal_event_type if recognized_terminal else None
                ),
                "failure_origin": metadata.get("aawm_failure_origin"),
                "provider_returned": provider_authored,
                "endpoint": (
                    metadata.get("endpoint")
                    or route_context.get("incoming_endpoint")
                ),
                "alias_family": (
                    metadata.get("alias_family")
                    or metadata.get("model_family")
                    or metadata.get("route_family")
                    or "responses"
                ),
                "alias_model": (
                    metadata.get("model_alias")
                    or metadata.get("model_alias_label")
                    or metadata.get("requested_model_alias")
                    or request_body.get("model")
                ),
                "selected_provider": (
                    metadata.get("codex_auto_agent_selected_provider")
                    or metadata.get("provider")
                ),
                "selected_model": (
                    metadata.get("codex_auto_agent_selected_model")
                    or failure_context.get("model")
                ),
                "selected_route": (
                    metadata.get("codex_auto_agent_selected_route_family")
                    or metadata.get("route_family")
                    or route_context.get("outgoing_target")
                ),
                "status_code": metadata.get("status_code"),
                "error_code": (
                    policy_failure_code
                    or error_code
                    or classification
                ),
                "error_type": extracted_type,
                "failure_class": (
                    policy_failure_class
                    or policy_failure_kind
                    or error_class
                ),
                "failure_phase": failure_context["stream_failure_stage"],
                "attempted_provider_call": True,
                "redispatch_required": redispatch_required,
                "terminal_outcome": (
                    delivered_disposition
                    or ("redispatch_required" if redispatch_required else "failed")
                ),
                "fallback_result": metadata.get("fallback_result") or "none",
                "attempt_count": metadata.get("attempt_count"),
                "correlation_id": (
                    metadata.get("litellm_call_id")
                    or metadata.get("request_id")
                    or metadata.get("trace_id")
                ),
            },
            marker=metadata,
        )
        if delivered_disposition:
            annotate_delivered_wire_failure(
                failure_exc,
                delivered_disposition=delivered_disposition,
            )
        PassThroughStreamingHandler._sync_logging_obj_model_call_details_from_kwargs(
            litellm_logging_obj,
            kwargs,
        )
        PassThroughStreamingHandler._set_streaming_handler_branch(
            handler_branch_state,
            "async_failure_handler",
        )
        try:
            await litellm_logging_obj.async_failure_handler(
                exception=failure_exc,
                traceback_exception=None,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as logging_exc:
            verbose_proxy_logger.exception(
                "async_failure_handler failed after Responses stream failure: %s",
                str(logging_exc),
            )
        if delivered_disposition:
            if not metadata.get("aawm_delivered_disposition_failure_recorded"):
                metadata["aawm_route_rollup_turn_recorded"] = True
                metadata["aawm_delivered_disposition_failure_recorded"] = True
                record_aawm_route_rollup_failure(
                    kwargs,
                    message=(
                        f"policy_failure={policy_failure_kind or policy_failure_code}; "
                        f"delivered_disposition={delivered_disposition}"
                        if policy_failure_kind or policy_failure_code
                        else f"delivered_disposition={delivered_disposition}"
                    ),
                    status=(
                        "Incomplete"
                        if delivered_disposition == "incomplete"
                        else "Failed"
                    ),
                )
        elif not metadata.get("aawm_stream_terminal_emitted"):
            PassThroughStreamingHandler._record_post_first_byte_stream_terminal_rollup(
                success_handler_kwargs=kwargs,
                failure_context=failure_context,
                exc=failure_exc,
            )
        metadata["aawm_stream_terminal_emitted"] = True

    @staticmethod
    async def _route_streaming_logging_to_handler(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: List[bytes],
        end_time: datetime,
        response: Optional[httpx.Response] = None,
        precomputed_lines: Optional[List[str]] = None,
        model: Optional[str] = None,
        passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = None,
        custom_llm_provider: Optional[str] = None,
        success_handler_kwargs: Optional[Dict[str, Any]] = None,
        local_prepare_ms: Optional[float] = None,
        error_log_context: Optional[Dict[str, Any]] = None,
        responses_event_pipeline: Optional[ResponsesEventPipeline] = None,
        operational_request: Optional[Any] = None,
    ):
        """
        Route the logging for the collected chunks to the appropriate handler

        Supported endpoint types:
        - Anthropic
        - Vertex AI
        - OpenAI
        """
        handler_branch_state = ["initial"]
        handler_branch = handler_branch_state[0]
        try:
            finalize_started_at = datetime.now()
            all_chunks = PassThroughStreamingHandler._resolve_stream_logging_lines(
                raw_bytes=raw_bytes,
                precomputed_lines=precomputed_lines,
                endpoint_type=endpoint_type,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
                success_handler_kwargs=success_handler_kwargs,
            )
            kwargs: dict = (
                success_handler_kwargs
                if isinstance(success_handler_kwargs, dict)
                else {}
            )
            (
                standard_logging_response_object,
                kwargs,
                handler_branch,
                early_exit,
            ) = PassThroughStreamingHandler._collect_streaming_logging_result(
                litellm_logging_obj=litellm_logging_obj,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                response=response,
                url_route=url_route,
                request_body=request_body,
                endpoint_type=endpoint_type,
                start_time=start_time,
                all_chunks=all_chunks,
                raw_bytes=raw_bytes,
                end_time=end_time,
                model=model,
                passthrough_logging_payload=passthrough_logging_payload,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
                handler_branch_state=handler_branch_state,
                responses_event_pipeline=responses_event_pipeline,
            )
            if early_exit:
                return
            is_native_openai_responses_stream = (
                PassThroughStreamingHandler._is_openai_responses_stream(
                    endpoint_type=endpoint_type,
                    url_route=url_route,
                    custom_llm_provider=custom_llm_provider,
                )
            )
            delivered_wire_disposition = (
                get_delivered_wire_disposition(
                    kwargs,
                    request=operational_request,
                )
                if is_native_openai_responses_stream
                else None
            )
            skip_success_without_delivery = (
                is_native_openai_responses_stream
                and delivered_wire_disposition is None
            )
            if skip_success_without_delivery:
                verbose_proxy_logger.warning(
                    "Skipping native Responses success consumers without a delivered disposition"
                )
            if standard_logging_response_object is None:
                standard_logging_response_object = StandardPassThroughResponseObject(
                    response=f"cannot parse chunks to standard response object. Chunks={all_chunks}"
                )
            PassThroughStreamingHandler._record_streaming_finalize_metrics(
                kwargs=kwargs,
                finalize_started_at=finalize_started_at,
                local_prepare_ms=local_prepare_ms,
            )
            metadata = PassThroughStreamingHandler._ensure_streaming_metadata(kwargs)
            tracker_state = metadata.get("aawm_stream_tracker_state")
            synthetic_terminal_event_type = (
                tracker_state.get("synthetic_terminal_event_type")
                if isinstance(tracker_state, dict)
                else None
            )
            terminal_event_type, terminal_payload = (
                OpenAIPassthroughLoggingHandler._extract_terminal_response_payload_from_stream(
                    all_chunks,
                    responses_event_pipeline=responses_event_pipeline,
                )
            )
            if terminal_event_type is None and isinstance(tracker_state, dict):
                tracker_terminal = tracker_state.get("provider_terminal_event_type")
                if isinstance(tracker_terminal, str) and tracker_terminal:
                    terminal_event_type = tracker_terminal
            delivered_disposition = (
                delivered_wire_disposition.get("delivered_disposition")
                if isinstance(delivered_wire_disposition, dict)
                else None
            )
            responses_failed = (
                is_native_openai_responses_stream
                and (
                    terminal_event_type == "response.failed"
                    or (
                        isinstance(terminal_payload, dict)
                        and terminal_payload.get("status") == "failed"
                    )
                    or (
                        delivered_disposition is not None
                        and delivered_disposition != "completed"
                    )
                )
            )
            if responses_failed:
                await PassThroughStreamingHandler._finalize_failed_responses_stream(
                    litellm_logging_obj=litellm_logging_obj,
                    kwargs=kwargs,
                    metadata=metadata,
                    all_chunks=all_chunks,
                    request_body=request_body,
                    start_time=start_time,
                    end_time=end_time,
                    terminal_event_type=terminal_event_type,
                    terminal_payload=terminal_payload,
                    handler_branch_state=handler_branch_state,
                    delivered_wire_disposition=delivered_wire_disposition,
                    openai_alpha_capacity_retry_enabled=(
                        PassThroughStreamingHandler._is_openai_capacity_retry_policy_enabled()
                    ),
                    responses_event_pipeline=responses_event_pipeline,
                )
                return
            if synthetic_terminal_event_type == "response.incomplete":
                context = metadata.get("aawm_route_rollup_context")
                if isinstance(context, dict):
                    model_label = str(
                        context.get("model_label")
                        or request_body.get("model")
                        or "unknown-model"
                    )
                    message = (
                        "provider terminal omitted; synthetic response.incomplete"
                    )
                    emit_aawm_route_status_event(
                        alias_model=request_body.get("model") or model_label,
                        model_label=model_label,
                        status="Incomplete",
                        message=message,
                    )
                    record_aawm_route_rollup(
                        group_header_label=str(
                            context.get("group_header_label") or ""
                        ),
                        incoming_endpoint=str(
                            context.get("incoming_endpoint") or ""
                        ),
                        outgoing_target=str(context.get("outgoing_target") or ""),
                        model_label=model_label,
                        effort=str(context.get("reasoning_effort") or "none"),
                        turns=0,
                        status="Incomplete",
                        message=message,
                        observation_context=context,
                    )
            elif not (
                metadata.get("aawm_stream_interrupted")
                or metadata.get("aawm_route_rollup_turn_suppressed")
            ):
                record_aawm_route_rollup_turn(
                    kwargs,
                    response_body=(
                        PassThroughStreamingHandler._build_completed_responses_body_for_route_rollup(
                            all_chunks=all_chunks,
                            endpoint_type=endpoint_type,
                            url_route=url_route,
                            custom_llm_provider=custom_llm_provider,
                            responses_event_pipeline=responses_event_pipeline,
                        )
                    ),
                )
            # Sync after rollup attachment so the private codex review-decision
            # event attached by record_aawm_route_rollup_turn reaches
            # callback-visible model_call_details before success callbacks run.
            PassThroughStreamingHandler._sync_logging_obj_model_call_details_from_kwargs(
                litellm_logging_obj,
                kwargs,
            )
            if skip_success_without_delivery:
                return
            handler_branch = (
                await PassThroughStreamingHandler._dispatch_streaming_success_callbacks(
                    litellm_logging_obj=litellm_logging_obj,
                    standard_logging_response_object=standard_logging_response_object,
                    start_time=start_time,
                    end_time=end_time,
                    kwargs=kwargs,
                    handler_branch_state=handler_branch_state,
                )
            )
        except Exception as e:
            handler_branch = handler_branch_state[0]
            context = PassThroughStreamingHandler._build_streaming_logging_error_context(
                litellm_logging_obj=litellm_logging_obj,
                response=response,
                url_route=url_route,
                request_body=request_body,
                endpoint_type=endpoint_type,
                custom_llm_provider=custom_llm_provider,
                success_handler_kwargs=success_handler_kwargs,
                error_log_context=error_log_context,
                handler_branch=handler_branch,
            )
            verbose_proxy_logger.exception(
                "Error in _route_streaming_logging_to_handler: %s",
                str(e),
                extra=context,
            )

    @staticmethod
    def _extract_model_for_cost_injection(
        request_body: Optional[dict],
        url_route: str,
        endpoint_type: EndpointType,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> Optional[str]:
        """
        Extract model name for cost injection from various sources.
        """
        # Try to get model from request body
        if request_body:
            model = request_body.get("model")
            if model:
                return model

        # Try to get model from logging object
        if hasattr(litellm_logging_obj, "model_call_details"):
            model = litellm_logging_obj.model_call_details.get("model")
            if model:
                return model

        # For Vertex AI, try to extract from URL
        if endpoint_type == EndpointType.VERTEX_AI:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            if model and model != "unknown":
                return model

        return None

    @staticmethod
    def _stream_summary_first_finalize_enabled() -> bool:
        return _truthy_env_flag(
            PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV
        )

    @staticmethod
    def _stream_summary_first_finalize_eligible(
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> bool:
        if not PassThroughStreamingHandler._stream_summary_first_finalize_enabled():
            return False
        if endpoint_type == EndpointType.ANTHROPIC:
            return True
        if endpoint_type == EndpointType.OPENAI:
            return OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        return False

    @staticmethod
    def _raw_stream_bytes_capture_required() -> bool:
        """True when capture paths still need retained raw stream bytes.

        Shape-only capture is driven by decoded ``all_chunks``/lines. Full
        payload capture stores raw chunk bytes, and diagnostic payload
        manifests still record raw chunk counts/hashes. When either of those
        capture paths is enabled, keep buffering raw bytes even if
        line-accumulator finalize is active.
        """
        return (
            passthrough_full_payload_capture_enabled()
            or diagnostic_payload_capture_enabled()
        )

    @staticmethod
    def _should_buffer_raw_stream_bytes(
        *,
        line_accumulator_enabled: bool,
    ) -> bool:
        """Prefer skipping ``raw_bytes`` retention in summary-first finalize.

        When incremental line accumulation already supplies finalize logging
        lines and no debug/raw stream-shape capture path needs the raw chunks,
        drop the second full-body buffer for the stream lifetime.
        """
        if not line_accumulator_enabled:
            return True
        return PassThroughStreamingHandler._raw_stream_bytes_capture_required()

    @staticmethod
    def _resolve_stream_logging_lines(
        *,
        raw_bytes: List[bytes],
        precomputed_lines: Optional[List[str]],
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
        success_handler_kwargs: Optional[Dict[str, Any]],
    ) -> List[str]:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        if precomputed_lines is not None:
            metadata["aawm_stream_finalize_line_source"] = "incremental_summary"
            return list(precomputed_lines)
        metadata["aawm_stream_finalize_line_source"] = "raw_bytes_rebuild"
        return PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)

    @staticmethod
    def _convert_raw_bytes_to_str_lines(raw_bytes: List[bytes]) -> List[str]:
        """
        Converts a list of raw bytes into a list of string lines, similar to aiter_lines()

        Args:
            raw_bytes: List of bytes chunks from aiter.bytes()

        Returns:
            List of string lines, with each line being a complete data: {} chunk
        """
        # Combine all bytes and decode to string. Incomplete UTF-8 at a
        # stream boundary is the same T-4 condition as finish().
        combined_str = b"".join(raw_bytes).decode("utf-8", errors="ignore")

        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in combined_str.split("\n") if line.strip()]

        return lines

def _maybe_set_replay_latch(
    request: Any,
    commitment_reason: Optional[str],
    event_type: Optional[str],
) -> None:
    """Set the monotonic replay-commit latch when the parser detected a
    committing event and a Starlette request is available.

    Must be called *before* any await or yield that could expose committed
    data to the client.  Uses ``publish_responses_replay_commitment`` so the
    latch is immediately visible to ``ensure_openai_wire_replay_allowed``.
    """
    if request is None or commitment_reason is None:
        return
    try:
        publish_responses_replay_commitment(
            request,
            reason=commitment_reason,
            event_type=event_type,
        )
    except Exception:
        pass
