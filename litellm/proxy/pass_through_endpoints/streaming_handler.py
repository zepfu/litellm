import asyncio
import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
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


class PassThroughStreamingHandler:
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
    async def chunk_processor(
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
    ):
        """
        - Yields chunks from the response
        - Collect non-empty chunks for post-processing (logging)
        - Inject cost into chunks if include_cost_in_streaming_usage is enabled
        """
        try:
            raw_bytes: List[bytes] = []
            chunk_count = 0
            total_stream_bytes = 0
            first_chunk_at: Optional[datetime] = None
            first_emitted_at: Optional[datetime] = None
            metadata = PassThroughStreamingHandler._ensure_streaming_metadata(
                success_handler_kwargs
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

            asyncio.create_task(
                PassThroughStreamingHandler._route_streaming_logging_to_handler(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body or {},
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    raw_bytes=raw_bytes,
                    end_time=end_time,
                    passthrough_logging_payload=passthrough_logging_payload,
                    custom_llm_provider=custom_llm_provider,
                    success_handler_kwargs=success_handler_kwargs,
                    local_prepare_ms=local_prepare_ms,
                )
            )
        except Exception as e:
            verbose_proxy_logger.error(f"Error in chunk_processor: {str(e)}")
            raise

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
        model: Optional[str] = None,
        passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = None,
        custom_llm_provider: Optional[str] = None,
        success_handler_kwargs: Optional[Dict[str, Any]] = None,
        local_prepare_ms: Optional[float] = None,
    ):
        """
        Route the logging for the collected chunks to the appropriate handler

        Supported endpoint types:
        - Anthropic
        - Vertex AI
        - OpenAI
        """
        try:
            finalize_started_at = datetime.now()
            all_chunks = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(
                raw_bytes
            )
            standard_logging_response_object: Optional[
                PassThroughEndpointLoggingResultValues
            ] = None
            kwargs: dict = copy.deepcopy(success_handler_kwargs or {})
            if custom_llm_provider == "gemini":
                if not passthrough_success_handler_obj.is_gemini_route(
                    url_route, custom_llm_provider
                ):
                    return
                gemini_passthrough_logging_handler_result = GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks(
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
                )
                standard_logging_response_object = (
                    gemini_passthrough_logging_handler_result["result"]
                )
                kwargs.update(gemini_passthrough_logging_handler_result["kwargs"])
            elif endpoint_type == EndpointType.OPENAI:
                openai_passthrough_logging_handler_result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
                standard_logging_response_object = (
                    openai_passthrough_logging_handler_result["result"]
                )
                kwargs.update(openai_passthrough_logging_handler_result["kwargs"])
            elif endpoint_type == EndpointType.ANTHROPIC:
                anthropic_passthrough_logging_handler_result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
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
                standard_logging_response_object = (
                    anthropic_passthrough_logging_handler_result["result"]
                )
                kwargs.update(anthropic_passthrough_logging_handler_result["kwargs"])
            elif endpoint_type == EndpointType.VERTEX_AI:
                vertex_passthrough_logging_handler_result = VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
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
                standard_logging_response_object = (
                    vertex_passthrough_logging_handler_result["result"]
                )
                kwargs.update(vertex_passthrough_logging_handler_result["kwargs"])
            if standard_logging_response_object is None:
                standard_logging_response_object = StandardPassThroughResponseObject(
                    response=f"cannot parse chunks to standard response object. Chunks={all_chunks}"
                )
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
            await litellm_logging_obj.async_success_handler(
                result=standard_logging_response_object,
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **kwargs,
            )
            if (
                litellm_logging_obj._should_run_sync_callbacks_for_async_calls()
                is False
            ):
                return

            executor.submit(
                litellm_logging_obj.success_handler,
                result=standard_logging_response_object,
                end_time=end_time,
                cache_hit=False,
                start_time=start_time,
                **kwargs,
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in _route_streaming_logging_to_handler: {str(e)}"
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
