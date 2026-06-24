import asyncio
import codecs
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_stream_shape,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.aawm_route_logging import record_aawm_route_rollup_turn
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.pass_through_endpoints.google_code_assist_quota import (
    sanitize_google_code_assist_quota_for_logging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardPassThroughResponseObject

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
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
    def _extract_google_code_assist_streaming_quota(
        all_chunks: List[str],
        *,
        source: str = "google_retrieve_user_quota",
    ) -> Optional[Dict[str, Any]]:
        for chunk in reversed(all_chunks):
            parsed_objects = (
                GeminiPassthroughLoggingHandler._parse_stream_chunk_json_objects(chunk)
            )
            for parsed_object in reversed(parsed_objects):
                sanitized_quota = sanitize_google_code_assist_quota_for_logging(
                    parsed_object,
                    source=source,
                )
                if sanitized_quota:
                    return sanitized_quota
        return None

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
        response: httpx.Response,
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
            context["status_code"] = response.status_code
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
        response: httpx.Response,
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
            upstream_request = response.request
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
            if PassThroughStreamingHandler._stream_summary_first_finalize_eligible(
                endpoint_type=endpoint_type,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
            ):
                line_accumulator = _PassThroughStreamLineAccumulator()
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
            # Extract model name for cost injection
            model_name = PassThroughStreamingHandler._extract_model_for_cost_injection(
                request_body=request_body,
                url_route=url_route,
                endpoint_type=endpoint_type,
                litellm_logging_obj=litellm_logging_obj,
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

                if first_emitted_at is None:
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

                yield chunk

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
                verbose_proxy_logger.error(
                    "Streaming response interrupted after first byte in chunk_processor: %s",
                    str(e),
                    extra=exception_context,
                )
                end_time = datetime.now()
                local_raw_bytes = raw_bytes if "raw_bytes" in locals() else []
                local_line_accumulator = (
                    line_accumulator if "line_accumulator" in locals() else None
                )
                precomputed_lines: Optional[List[str]] = None
                if local_line_accumulator is not None:
                    precomputed_lines = local_line_accumulator.finish()
                if isinstance(success_handler_kwargs, dict):
                    metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
                        success_handler_kwargs
                    )
                    metadata["aawm_stream_chunk_count"] = local_chunk_count
                    metadata["aawm_stream_total_bytes"] = local_total_stream_bytes
                    metadata["aawm_stream_interrupted"] = True
                    metadata.update(
                        PassThroughStreamingHandler._build_streaming_failure_context(
                            exc=e,
                            chunk_count=local_chunk_count,
                            total_stream_bytes=local_total_stream_bytes,
                            first_chunk_at=local_first_chunk_at,
                            first_emitted_at=local_first_emitted_at,
                        )
                    )
                asyncio.create_task(
                    PassThroughStreamingHandler._route_streaming_logging_to_handler(
                        litellm_logging_obj=litellm_logging_obj,
                        passthrough_success_handler_obj=passthrough_success_handler_obj,
                        response=response,
                        url_route=url_route,
                        request_body=request_body or {},
                        endpoint_type=endpoint_type,
                        start_time=start_time,
                        raw_bytes=local_raw_bytes,
                        precomputed_lines=precomputed_lines,
                        end_time=end_time,
                        passthrough_logging_payload=passthrough_logging_payload,
                        custom_llm_provider=custom_llm_provider,
                        success_handler_kwargs=success_handler_kwargs,
                        local_prepare_ms=local_prepare_ms,
                        error_log_context=error_log_context,
                    )
                )
                return
            verbose_proxy_logger.exception(
                "Error in chunk_processor: %s",
                str(e),
                extra=exception_context,
            )
            raise

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
        return context

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
        response: httpx.Response,
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

        if custom_llm_provider in {"gemini", "antigravity"}:
            if not passthrough_success_handler_obj.is_gemini_route(
                url_route, custom_llm_provider
            ):
                handler_branch = set_branch(
                    handler_branch_state,
                    "google_code_assist_control_plane",
                )
                if "retrieveUserQuota" not in url_route:
                    return None, kwargs, handler_branch, True
                handler_branch = set_branch(
                    handler_branch_state,
                    "google_code_assist_quota",
                )
                quota_source = (
                    "antigravity_retrieve_user_quota"
                    if custom_llm_provider == "antigravity"
                    else "google_retrieve_user_quota"
                )
                sanitized_quota = (
                    PassThroughStreamingHandler._extract_google_code_assist_streaming_quota(
                        all_chunks,
                        source=quota_source,
                    )
                )
                if not sanitized_quota:
                    return None, kwargs, handler_branch, True
                metadata["google_retrieve_user_quota"] = sanitized_quota
                metadata["aawm_rate_limit_observation_only"] = True
                standard_logging_response_object = StandardPassThroughResponseObject(
                    response="\n".join(all_chunks)
                )
            else:
                handler_branch = set_branch(handler_branch_state, "gemini")
                gemini_passthrough_logging_handler_result = (
                    GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks(
                        litellm_logging_obj=litellm_logging_obj,
                        passthrough_success_handler_obj=passthrough_success_handler_obj,
                        url_route=url_route,
                        request_body=request_body,
                        endpoint_type=endpoint_type,
                        start_time=start_time,
                        all_chunks=all_chunks,
                        model=model,
                        end_time=end_time,
                        kwargs=kwargs,
                        custom_llm_provider=custom_llm_provider or "gemini",
                    )
                )
                standard_logging_response_object = (
                    gemini_passthrough_logging_handler_result["result"]
                )
                kwargs.update(gemini_passthrough_logging_handler_result["kwargs"])
        elif endpoint_type == EndpointType.OPENAI:
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
    async def _route_streaming_logging_to_handler(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        response: httpx.Response,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: List[bytes],
        end_time: datetime,
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
            PassThroughStreamingHandler._sync_logging_obj_model_call_details_from_kwargs(
                litellm_logging_obj,
                kwargs,
            )
            record_aawm_route_rollup_turn(kwargs)
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
