import asyncio
import codecs
import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_stream_shape,
    diagnostic_payload_capture_enabled,
    passthrough_full_payload_capture_enabled,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.aawm_route_logging import (
    _AAWM_PARSED_CODEX_REVIEW_DECISIONS_KWARGS_KEY,
    emit_aawm_route_status_event,
    record_aawm_route_rollup,
    record_aawm_route_rollup_turn,
)
from litellm.proxy.aawm_session_transfer.identity import extract_transfer_identity
from litellm.proxy.aawm_session_transfer.registry import (
    safe_finalize,
    safe_mark_phase,
    safe_record_chunks,
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
_RESPONSES_PRE_COMMIT_MAX_EVENTS = 16
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
_RESPONSES_ACCOUNT_EXHAUSTION_CLASSES = frozenset(
    {
        "usage_limit_reached",
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
        self.detail = {
            "error": {
                "message": self.message,
                "type": error_class,
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
        return HTTPException(
            status_code=status_code,
            detail=self.detail,
            headers=headers,
        )


class _PrefixedHttpxByteStream:
    """Replay peeked SSE bytes, then continue the original upstream iterator."""

    def __init__(
        self,
        response: httpx.Response,
        prefix: List[bytes],
        remainder: Any = None,
    ) -> None:
        self._response = response
        self._prefix = prefix
        self._remainder = remainder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    async def aiter_bytes(self):
        for chunk in self._prefix:
            if chunk:
                yield chunk
        if self._remainder is None:
            return
        async for chunk in self._remainder:
            yield chunk


class _PassThroughStreamLineAccumulator:
    """Incrementally decode raw stream bytes into non-empty SSE/log lines."""

    __slots__ = ("_decoder", "_pending", "lines")

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._pending = ""
        self.lines: List[str] = []

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._pending += self._decoder.decode(chunk)
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self.lines.append(stripped)

    def finish(self) -> List[str]:
        tail = self._decoder.decode(b"", final=True)
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
    def _is_openai_responses_stream(
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> bool:
        parsed_url = urlparse(url_route)
        return (
            endpoint_type == EndpointType.OPENAI
            and custom_llm_provider == "openai"
            and parsed_url.hostname in {"api.openai.com", "chatgpt.com"}
            and OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        )

    @staticmethod
    def _stamp_encrypted_reasoning_in_responses_sse_chunk(
        chunk: bytes,
        *,
        request_body: Optional[Dict[str, Any]],
        custom_llm_provider: Optional[str],
    ) -> bytes:
        """OPENAI-006: delegate SSE provenance stamping to the helper module."""
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            stamp_encrypted_reasoning_in_responses_sse_chunk,
        )

        return stamp_encrypted_reasoning_in_responses_sse_chunk(
            chunk,
            request_body=request_body,
            custom_llm_provider=custom_llm_provider,
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
    def _iter_responses_sse_events(
        lines: List[str],
    ) -> List[tuple[Optional[str], Optional[Dict[str, Any]]]]:
        events: List[tuple[Optional[str], Optional[Dict[str, Any]]]] = []
        pending_event_type: Optional[str] = None
        for raw_line in lines:
            line = _strip_chunk_line(raw_line)
            if not line:
                continue
            if line.startswith("event:"):
                pending_event_type = line.removeprefix("event:").strip() or None
                continue
            if not line.startswith("data:"):
                continue
            payload_text = line.removeprefix("data:").strip()
            if payload_text == "[DONE]":
                events.append(("[DONE]", None))
                pending_event_type = None
                continue
            payload: Optional[Dict[str, Any]] = None
            payload_type: Optional[str] = None
            try:
                decoded = json.loads(payload_text)
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, dict):
                payload = decoded
                raw_type = decoded.get("type")
                if isinstance(raw_type, str) and raw_type.strip():
                    payload_type = raw_type.strip()
            event_type = pending_event_type or payload_type
            events.append((event_type, payload))
            pending_event_type = None
        if pending_event_type is not None:
            events.append((pending_event_type, None))
        return events

    @staticmethod
    def _is_responses_non_substantive_lifecycle_event(
        event_type: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if event_type in _RESPONSES_NON_SUBSTANTIVE_LIFECYCLE_EVENTS:
            if not isinstance(payload, dict):
                return True
            response_payload = payload.get("response")
            if isinstance(response_payload, dict):
                if response_payload.get("status") in {"failed", "incomplete"}:
                    return False
                if response_payload.get("error") is not None:
                    return False
                output = response_payload.get("output")
                if isinstance(output, list) and any(output):
                    return False
            return True
        if event_type == "response.completed":
            return False
        return False

    @staticmethod
    def _is_responses_substantive_event(
        event_type: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if event_type is None or event_type == "[DONE]":
            return False
        if event_type in _RESPONSES_PRE_COMMIT_FAILURE_EVENTS:
            return False
        if event_type in PassThroughStreamingHandler._RESPONSES_TERMINAL_EVENTS:
            return event_type == "response.completed"
        if event_type in _RESPONSES_NON_SUBSTANTIVE_LIFECYCLE_EVENTS:
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
    def _classify_responses_pre_commit_error(
        error_payload: Optional[Dict[str, Any]],
    ) -> tuple[str, str, bool]:
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
        joined = " ".join(sorted(tokens))
        if any(
            marker in joined
            for marker in (
                "usage_limit_reached",
                "usage limit",
                "quota exceeded",
                "quota exhausted",
                "weekly limit",
            )
        ):
            return "usage_limit_reached", "usage_limit_reached", False
        if any(
            marker in joined
            for marker in (
                "server_overloaded",
                "overloaded_error",
                "model is overloaded",
                "upstream_overloaded",
                "high demand",
                "model_capacity_exhausted",
                "model at capacity",
                "upstream busy",
            )
        ):
            return "server_overloaded", "transient_capacity", True
        return "provider_terminal_error", "provider_terminal_error", False

    @staticmethod
    def _inspect_responses_pre_commit_chunks(
        chunks: List[bytes],
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        lines = PassThroughStreamingHandler._chunk_lines(chunks)
        events = PassThroughStreamingHandler._iter_responses_sse_events(lines)
        if not events:
            return "empty", None, None
        if len(events) > _RESPONSES_PRE_COMMIT_MAX_EVENTS:
            return "substantive", None, None
        first_error_payload: Optional[Dict[str, Any]] = None
        first_error_event: Optional[str] = None
        saw_lifecycle = False
        for event_type, payload in events:
            if event_type == "[DONE]":
                continue
            error_payload = (
                PassThroughStreamingHandler._extract_responses_stream_error_payload(
                    event_type,
                    payload,
                )
            )
            if error_payload is not None or event_type in _RESPONSES_PRE_COMMIT_FAILURE_EVENTS:
                if first_error_payload is None:
                    first_error_payload = error_payload or payload
                    first_error_event = event_type
                continue
            if PassThroughStreamingHandler._is_responses_non_substantive_lifecycle_event(
                event_type,
                payload,
            ):
                saw_lifecycle = True
                continue
            if PassThroughStreamingHandler._is_responses_substantive_event(
                event_type,
                payload,
            ):
                return "substantive", None, event_type
            if event_type not in {None, ""}:
                return "substantive", None, event_type
        if first_error_payload is not None or first_error_event is not None:
            return "failed", first_error_payload, first_error_event
        if saw_lifecycle:
            return "lifecycle", None, None
        return "empty", None, None

    @staticmethod
    def _build_responses_pre_commit_failure(
        *,
        error_payload: Optional[Dict[str, Any]],
        event_type: Optional[str],
        pre_commit_retry_exhausted: bool = False,
        retry_after_seconds: float = RESPONSES_PRE_COMMIT_TRANSIENT_RETRY_WAIT_SECONDS,
    ) -> ResponsesStreamPreCommitFailure:
        error_class, classification, retryable = (
            PassThroughStreamingHandler._classify_responses_pre_commit_error(
                error_payload
            )
        )
        sanitized_message = None
        error_code = None
        error_type = None
        if isinstance(error_payload, dict):
            sanitized_message = (
                OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_error_for_logging(
                    error_payload
                )
            )
            raw_code = error_payload.get("code")
            raw_type = error_payload.get("type")
            if isinstance(raw_code, str) and raw_code.strip():
                error_code = raw_code.strip()
            if isinstance(raw_type, str) and raw_type.strip():
                error_type = raw_type.strip()
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
            message=sanitized_message,
            pre_commit_retry_exhausted=pre_commit_retry_exhausted,
            error_payload=error_payload,
        )

    @staticmethod
    async def peek_responses_pre_commit_stream(
        response: httpx.Response,
        *,
        max_buffer_bytes: int = _RESPONSES_PRE_COMMIT_MAX_BUFFER_BYTES,
    ) -> tuple[httpx.Response, Optional[ResponsesStreamPreCommitFailure]]:
        """Hold lifecycle-only Responses bytes until a commit or pre-SSE failure."""
        peeked: List[bytes] = []
        buffered_bytes = 0
        iterator = response.aiter_bytes()
        while True:
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                iterator = None
                break
            if not chunk:
                continue
            peeked.append(chunk)
            buffered_bytes += len(chunk)
            if buffered_bytes > max_buffer_bytes:
                return _PrefixedHttpxByteStream(response, peeked, iterator), None
            decision, error_payload, event_type = (
                PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(peeked)
            )
            if decision == "failed":
                return (
                    _PrefixedHttpxByteStream(response, peeked, iterator),
                    PassThroughStreamingHandler._build_responses_pre_commit_failure(
                        error_payload=error_payload,
                        event_type=event_type,
                    ),
                )
            if decision == "substantive":
                return _PrefixedHttpxByteStream(response, peeked, iterator), None
        decision, error_payload, event_type = (
            PassThroughStreamingHandler._inspect_responses_pre_commit_chunks(peeked)
        )
        if decision == "failed":
            return (
                _PrefixedHttpxByteStream(response, peeked, iterator),
                PassThroughStreamingHandler._build_responses_pre_commit_failure(
                    error_payload=error_payload,
                    event_type=event_type,
                ),
            )
        return _PrefixedHttpxByteStream(response, peeked, iterator), None

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
            sanitized["source"] = "xai_oauth_response_headers"
        return sanitized

    @staticmethod
    def _is_xai_oauth_metadata(metadata: Dict[str, Any]) -> bool:
        if metadata.get("xai_oauth_managed") is True:
            return True
        if metadata.get("grok_native_oauth_managed") is True:
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
    ) -> None:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        if (
            custom_llm_provider == "xai"
            and PassThroughStreamingHandler._is_xai_oauth_metadata(metadata)
        ):
            sanitized_headers = PassThroughStreamingHandler._sanitize_xai_oauth_rate_limit_headers(
                response.headers
            )
            if sanitized_headers:
                metadata["xai_oauth_response_headers"] = sanitized_headers
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
    ) -> Dict[str, Any]:
        metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
            success_handler_kwargs
        )
        PassThroughStreamingHandler._record_upstream_rate_limit_headers_metadata(
            success_handler_kwargs,
            response=response,
            endpoint_type=endpoint_type,
            custom_llm_provider=custom_llm_provider,
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
    ):
        """
        - Yields chunks from the response
        - Collect non-empty chunks for post-processing (logging)
        - Inject cost into chunks if include_cost_in_streaming_usage is enabled
        """
        try:
            raw_bytes: List[bytes] = []
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
            )
            responses_terminal_accumulator = (
                _PassThroughStreamLineAccumulator()
                if is_fork_owned_responses_stream
                else None
            )
            responses_sse_tracker = (
                OpenAIPassthroughLoggingHandler._create_responses_sse_state_tracker(
                    fallback_model=model_name or "unknown"
                )
                if is_fork_owned_responses_stream
                else None
            )
            responses_terminal_seen = False
            held_responses_done_suffix = b""
            buffer_raw_bytes = True
            if PassThroughStreamingHandler._stream_summary_first_finalize_eligible(
                endpoint_type=endpoint_type,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
            ):
                line_accumulator = _PassThroughStreamLineAccumulator()
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
            await safe_mark_phase(transfer_identity, "awaiting_upstream")
            downstream_chunk_count = 0
            downstream_byte_count = 0

            async def _publish_transfer_chunks(
                *,
                first_upstream: bool = False,
                first_downstream: bool = False,
                force: bool = False,
            ) -> None:
                if not force and chunk_count > 1 and chunk_count % 8 != 0:
                    return
                await safe_record_chunks(
                    transfer_identity,
                    upstream_chunks=chunk_count,
                    upstream_bytes=total_stream_bytes,
                    downstream_chunks=downstream_chunk_count,
                    downstream_bytes=downstream_byte_count,
                    first_upstream=first_upstream,
                    first_downstream=first_downstream,
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

            def _consume_responses_lines(lines: List[str]) -> None:
                nonlocal responses_terminal_seen
                if responses_sse_tracker is None:
                    return
                if not responses_terminal_seen:
                    responses_terminal_seen = (
                        PassThroughStreamingHandler._responses_lines_have_terminal_event(
                            lines
                        )
                    )
                for raw_line in lines:
                    line = _strip_chunk_line(raw_line)
                    if not line.startswith("data:"):
                        continue
                    payload_text = line.removeprefix("data:").strip()
                    if payload_text == "[DONE]":
                        continue
                    try:
                        decoded_event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        OpenAIPassthroughLoggingHandler._mark_responses_sse_parse_ambiguity(
                            responses_sse_tracker
                        )
                        continue
                    OpenAIPassthroughLoggingHandler._consume_responses_sse_event(
                        responses_sse_tracker,
                        decoded_event,
                    )

            async for chunk in response.aiter_bytes():
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

                if responses_terminal_accumulator is not None:
                    chunk = PassThroughStreamingHandler._stamp_encrypted_reasoning_in_responses_sse_chunk(
                        chunk,
                        request_body=request_body if isinstance(request_body, dict) else None,
                        custom_llm_provider=custom_llm_provider,
                    )
                    responses_terminal_accumulator.feed(chunk)
                    _consume_responses_lines(responses_terminal_accumulator.lines)
                    responses_terminal_accumulator.lines.clear()

                    if responses_terminal_seen:
                        if held_responses_done_suffix:
                            _record_responses_wire_chunk(
                                held_responses_done_suffix
                            )
                            yield held_responses_done_suffix
                            held_responses_done_suffix = b""
                        _record_responses_wire_chunk(chunk)
                        await _publish_transfer_chunks(
                            first_downstream=first_emitted_at is not None
                            and downstream_chunk_count == 1
                        )
                        yield chunk
                        continue

                    (
                        chunk_without_done,
                        held_responses_done_suffix,
                    ) = PassThroughStreamingHandler._split_trailing_done_chunk(
                        held_responses_done_suffix + chunk
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

                # OPENAI-006: stamp encrypted reasoning on non-OpenAI Responses
                # streams too (e.g. xAI) so foreign provenance survives into
                # later OpenAI continuations.
                if (
                    isinstance(chunk, (bytes, bytearray))
                    and b"encrypted_content" in chunk
                    and b"reasoning" in chunk
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

            if responses_terminal_accumulator is not None:
                if responses_terminal_accumulator.has_pending_frame():
                    OpenAIPassthroughLoggingHandler._mark_responses_sse_partial_frame(
                        responses_sse_tracker
                    )
                else:
                    _consume_responses_lines(responses_terminal_accumulator.finish())

                terminal_chunks: List[bytes] = []
                if first_emitted_at is not None and not responses_terminal_seen:
                    synthetic_terminal = OpenAIPassthroughLoggingHandler._classify_responses_sse_clean_eof(
                        responses_sse_tracker
                    )
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

                if terminal_chunks:
                    held_responses_done_suffix = b""
                    for terminal_chunk in terminal_chunks:
                        _record_responses_wire_chunk(terminal_chunk)
                        yield terminal_chunk
                elif held_responses_done_suffix:
                    _record_responses_wire_chunk(held_responses_done_suffix)
                    yield held_responses_done_suffix
                    held_responses_done_suffix = b""

                metadata["aawm_stream_tracker_state"] = (
                    OpenAIPassthroughLoggingHandler._responses_sse_tracker_metadata(
                        responses_sse_tracker
                    )
                )

            # After all chunks are processed, handle post-processing
            end_time = datetime.now()
            metadata["aawm_stream_chunk_count"] = chunk_count
            metadata["aawm_stream_total_bytes"] = total_stream_bytes
            if upstream_wait_started_at is not None:
                metadata["aawm_upstream_stream_complete_ms"] = round(
                    max(0.0, (end_time - upstream_wait_started_at).total_seconds() * 1000.0),
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
            await safe_mark_phase(
                transfer_identity,
                "finalizing",
                extra={
                    "upstream_chunk_count": chunk_count,
                    "upstream_byte_count": total_stream_bytes,
                    "downstream_chunk_count": downstream_chunk_count,
                    "downstream_byte_count": downstream_byte_count,
                },
            )
            await safe_finalize(
                transfer_identity,
                "completed",
                extra={
                    "upstream_chunk_count": chunk_count,
                    "upstream_byte_count": total_stream_bytes,
                    "downstream_chunk_count": downstream_chunk_count,
                    "downstream_byte_count": downstream_byte_count,
                },
            )

            precomputed_lines: Optional[List[str]] = None
            if line_accumulator is not None:
                precomputed_lines = line_accumulator.finish()

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
                )
            )
        except asyncio.CancelledError:
            local_identity = (
                transfer_identity if "transfer_identity" in locals() else {}
            )
            await safe_finalize(
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
            local_identity = (
                transfer_identity if "transfer_identity" in locals() else {}
            )
            await safe_finalize(
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
            await safe_finalize(
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

        verbose_proxy_logger.error(
            "Streaming response interrupted after first byte in chunk_processor: %s",
            str(exc),
            extra=exception_context,
        )

        # The stream has already emitted bytes to the client, so this
        # cannot be retried or completed truthfully. Error intake and
        # route rollup above carry the terminal failure; do not send
        # partial chunks through the normal success callback pipeline.
        #
        # Still run the standard failure logging pipeline so Langfuse /
        # session_history / spend callbacks observe the mid-stream
        # timeout (success handlers must not run on partial streams).
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
        verbose_proxy_logger.error(
            "Streaming response ended after partial bytes without a terminal event",
            extra=error_log_context or failure_context,
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
    def _build_post_first_byte_terminal_stream_chunks(
        *,
        endpoint_type: EndpointType,
        url_route: str,
        custom_llm_provider: Optional[str],
        failure_context: Dict[str, Any],
        exc: Optional[Exception] = None,
        incomplete_reason: Optional[str] = None,
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
            return [
                (
                    "event: response.incomplete\ndata: "
                    + json.dumps(payload, separators=(",", ":"))
                    + "\n\n"
                ).encode("utf-8"),
                b"data: [DONE]\n\n",
            ]

        if exc is None:
            raise ValueError("exc is required for failed stream terminal chunks")
        message = (
            "Streaming response interrupted after first byte due to upstream read "
            f"timeout: {exc}"
        )
        error_payload = {
            "type": "proxy_stream_terminal_error",
            "code": failure_context.get("failure_kind")
            or "streaming_upstream_read_failure",
            "message": message,
            "param": None,
        }

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

        if endpoint_type == EndpointType.OPENAI and (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
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
            return chunks

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
    ) -> Optional[Dict[str, Any]]:
        if not PassThroughStreamingHandler._is_openai_responses_stream(
            endpoint_type=endpoint_type,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
        ):
            return None

        terminal_event_type, terminal_payload = (
            OpenAIPassthroughLoggingHandler._extract_terminal_response_payload_from_stream(
                all_chunks
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
            OpenAIPassthroughLoggingHandler._reconstruct_responses_output_items_from_stream(
                all_chunks
            )
        )
        merged_output = OpenAIPassthroughLoggingHandler._merge_responses_output_lists(
            completed_output if isinstance(completed_output, list) else [],
            reconstructed_output,
        )
        streamed_output_text = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_stream_text(
                all_chunks
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
    ) -> Optional[Dict[str, Any]]:
        events = PassThroughStreamingHandler._iter_responses_sse_events(all_chunks)
        first_error: Optional[Dict[str, Any]] = None
        failed_error: Optional[Dict[str, Any]] = None
        for event_type, payload in events:
            extracted = PassThroughStreamingHandler._extract_responses_stream_error_payload(
                event_type,
                payload,
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
    async def _finalize_failed_responses_stream(
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
    ) -> None:
        metadata["aawm_route_rollup_turn_suppressed"] = True
        metadata["aawm_stream_interrupted"] = True
        metadata["aawm_responses_stream_failed"] = True
        error_payload = PassThroughStreamingHandler._reconcile_responses_stream_error_payload(
            all_chunks=all_chunks,
            terminal_payload=terminal_payload,
        )
        error_class, classification, retryable = (
            PassThroughStreamingHandler._classify_responses_pre_commit_error(
                error_payload
            )
        )
        sanitized_message = None
        if isinstance(error_payload, dict):
            sanitized_message = (
                OpenAIPassthroughLoggingHandler._sanitize_responses_terminal_error_for_logging(
                    error_payload
                )
            )
        if not sanitized_message:
            sanitized_message = classification
        metadata["aawm_responses_stream_failure_class"] = error_class
        metadata["aawm_responses_stream_failure_classification"] = classification
        metadata["aawm_responses_stream_failure_retryable"] = retryable
        failure_exc = ResponsesStreamPreCommitFailure(
            error_class=error_class,
            classification=classification,
            retryable=retryable,
            error_payload=error_payload if isinstance(error_payload, dict) else None,
            message=sanitized_message,
        )
        failure_context = {
            "failure_kind": classification,
            "stream_failure_stage": "responses_stream_failed",
            "error_class": error_class,
            "model": request_body.get("model") if isinstance(request_body, dict) else None,
        }
        metadata.update(failure_context)
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
        if not metadata.get("aawm_stream_terminal_emitted"):
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
            )
            if early_exit:
                return
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
                    all_chunks
                )
            )
            if terminal_event_type is None and isinstance(tracker_state, dict):
                tracker_terminal = tracker_state.get("provider_terminal_event_type")
                if isinstance(tracker_terminal, str) and tracker_terminal:
                    terminal_event_type = tracker_terminal
            responses_failed = (
                PassThroughStreamingHandler._is_openai_responses_stream(
                    endpoint_type=endpoint_type,
                    url_route=url_route,
                    custom_llm_provider=custom_llm_provider,
                )
                and (
                    terminal_event_type == "response.failed"
                    or (
                        isinstance(terminal_payload, dict)
                        and terminal_payload.get("status") == "failed"
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
        # Combine all bytes and decode to string
        combined_str = b"".join(raw_bytes).decode("utf-8")

        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in combined_str.split("\n") if line.strip()]

        return lines
