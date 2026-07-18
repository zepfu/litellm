"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import contextlib
import json
import os
from collections.abc import Mapping
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

import litellm
from litellm._uuid import uuid4


@dataclass(frozen=True)
class Runtime:
    _build_anthropic_response_from_completion_adapter_response: Any
    _build_anthropic_streaming_response_from_completion_adapter_stream: Any
    _clean_codex_auth_value: Any
    _get_google_adapter_native_api_client_header: Any
    _get_google_adapter_native_user_agent: Any
    _get_google_adapter_post_tool_cooldown_seconds: Any
    _get_google_adapter_rate_limit_key: Any
    _google_code_assist_unwrapped_chunk_contains_tool_call: Any
    _is_codex_google_code_assist_empty_text_content: Any
    _mapping_or_attr_get: Any
    _normalize_google_completion_adapter_model_name: Any
    _remember_codex_google_code_assist_tool_call_name: Any
    _responses_sse_from_iterator: Any
    _sanitize_google_code_assist_tool_schema: Any
    _set_google_adapter_cooldown: Any
    _unwrap_google_code_assist_response_payload: Any
    _usage_has_no_more_than_one_output_token: Any
    verbose_proxy_logger: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _build_anthropic_response_from_completion_adapter_response=namespace[
            "_build_anthropic_response_from_completion_adapter_response"
        ],
        _build_anthropic_streaming_response_from_completion_adapter_stream=namespace[
            "_build_anthropic_streaming_response_from_completion_adapter_stream"
        ],
        _clean_codex_auth_value=namespace["_clean_codex_auth_value"],
        _get_google_adapter_native_api_client_header=namespace["_get_google_adapter_native_api_client_header"],
        _get_google_adapter_native_user_agent=namespace["_get_google_adapter_native_user_agent"],
        _get_google_adapter_post_tool_cooldown_seconds=namespace["_get_google_adapter_post_tool_cooldown_seconds"],
        _get_google_adapter_rate_limit_key=namespace["_get_google_adapter_rate_limit_key"],
        _google_code_assist_unwrapped_chunk_contains_tool_call=namespace[
            "_google_code_assist_unwrapped_chunk_contains_tool_call"
        ],
        _is_codex_google_code_assist_empty_text_content=namespace["_is_codex_google_code_assist_empty_text_content"],
        _mapping_or_attr_get=namespace["_mapping_or_attr_get"],
        _normalize_google_completion_adapter_model_name=namespace["_normalize_google_completion_adapter_model_name"],
        _remember_codex_google_code_assist_tool_call_name=namespace[
            "_remember_codex_google_code_assist_tool_call_name"
        ],
        _responses_sse_from_iterator=namespace["_responses_sse_from_iterator"],
        _sanitize_google_code_assist_tool_schema=namespace["_sanitize_google_code_assist_tool_schema"],
        _set_google_adapter_cooldown=namespace["_set_google_adapter_cooldown"],
        _unwrap_google_code_assist_response_payload=namespace["_unwrap_google_code_assist_response_payload"],
        _usage_has_no_more_than_one_output_token=namespace["_usage_has_no_more_than_one_output_token"],
        verbose_proxy_logger=namespace["verbose_proxy_logger"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


async def _iterate_google_code_assist_unwrapped_stream(
    body_iterator: Any,
    *,
    adapter_model: Optional[str] = None,
    rate_limit_key: Optional[str] = None,
) -> Any:
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

    debug_logged = False
    post_tool_cooldown_armed = False

    async def _iter_event_block_lines(event_block: str):
        nonlocal debug_logged, post_tool_cooldown_armed
        for line in event_block.splitlines():
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
            if not isinstance(parsed_chunk, dict):
                continue
            unwrapped = _runtime()._unwrap_google_code_assist_response_payload(parsed_chunk)
            if unwrapped is None:
                continue
            if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1" and not debug_logged:
                try:
                    first_candidate = None
                    candidates = unwrapped.get("candidates") if isinstance(unwrapped, dict) else None
                    if isinstance(candidates, list) and candidates:
                        first_candidate = candidates[0]
                    _runtime().verbose_proxy_logger.info(
                        "Gemini adapter stream debug: first_unwrapped_keys=%s first_candidate=%s",
                        sorted(unwrapped.keys()) if isinstance(unwrapped, dict) else type(unwrapped).__name__,
                        first_candidate,
                    )
                    debug_logged = True
                except Exception:
                    _runtime().verbose_proxy_logger.exception("Gemini adapter stream debug logging failed")
            if not post_tool_cooldown_armed and _runtime()._google_code_assist_unwrapped_chunk_contains_tool_call(
                unwrapped
            ):
                cooldown_seconds = _runtime()._get_google_adapter_post_tool_cooldown_seconds()
                if cooldown_seconds > 0:
                    await _runtime()._set_google_adapter_cooldown(
                        _runtime()._clean_codex_auth_value(rate_limit_key)
                        or _runtime()._get_google_adapter_rate_limit_key(adapter_model),
                        cooldown_seconds,
                    )
                    post_tool_cooldown_armed = True
                    _runtime().verbose_proxy_logger.debug(
                        "Google adapter post-tool cooldown armed for %.1fs",
                        cooldown_seconds,
                    )
            yield f"data: {json.dumps(unwrapped)}\n\n"

    buffer = ""
    try:
        async for raw_chunk in body_iterator:
            if isinstance(raw_chunk, bytes):
                buffer += raw_chunk.decode("utf-8")
            else:
                buffer += str(raw_chunk)

            while "\n\n" in buffer:
                event_block, buffer = buffer.split("\n\n", 1)
                async with contextlib.aclosing(_iter_event_block_lines(event_block)) as emitted_chunks:
                    async for emitted_chunk in emitted_chunks:
                        yield emitted_chunk

        if buffer.strip():
            async with contextlib.aclosing(_iter_event_block_lines(buffer)) as emitted_chunks:
                async for emitted_chunk in emitted_chunks:
                    yield emitted_chunk
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if callable(aclose):
            await aclose()


def _build_anthropic_streaming_response_from_google_code_assist_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    gemini_optional_params: dict[str, Any],
    rate_limit_key: Optional[str] = None,
) -> StreamingResponse:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    logging_obj: Any = SimpleNamespace(
        optional_params=gemini_optional_params,
        post_call=lambda **_: None,
    )
    completion_stream = ModelResponseIterator(
        streaming_response=_iterate_google_code_assist_unwrapped_stream(
            response.body_iterator,
            adapter_model=adapter_model,
            rate_limit_key=rate_limit_key,
        ),
        sync_stream=False,
        logging_obj=logging_obj,
    )
    anthropic_stream = AnthropicAdapter().translate_completion_output_params_streaming(
        completion_stream,
        model=_runtime()._normalize_google_completion_adapter_model_name(adapter_model),
        tool_name_mapping=tool_name_mapping,
    )
    return _runtime()._build_anthropic_streaming_response_from_completion_adapter_stream(anthropic_stream)


def _restore_google_adapter_tool_call_names(
    response_obj: Any,
    tool_name_mapping: dict[str, str],
    *,
    scope_key: Optional[str] = None,
) -> Any:
    if scope_key is None:
        scope_key = _runtime()._clean_codex_auth_value(tool_name_mapping.get("__aawm_scope_key__"))
    choices = getattr(response_obj, "choices", None)
    if not isinstance(choices, list):
        return response_obj
    for choice in choices:
        for message_attr in ("message", "delta"):
            message = getattr(choice, message_attr, None)
            if message is None:
                continue
            tool_calls = getattr(message, "tool_calls", None)
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                function = (
                    tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                )
                if function is None:
                    continue
                current_name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
                function_arguments = (
                    function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
                )
                original_name = tool_name_mapping.get(str(current_name or ""))
                final_name = original_name or current_name
                tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                _runtime()._remember_codex_google_code_assist_tool_call_name(
                    tool_call_id,
                    final_name,
                    function_arguments,
                    scope_key=scope_key,
                )
                if not original_name:
                    continue
                if isinstance(function, dict):
                    function["name"] = original_name
                else:
                    setattr(function, "name", original_name)
    return response_obj


async def _restore_google_adapter_tool_call_names_stream(
    completion_stream: Any,
    tool_name_mapping: dict[str, str],
    *,
    scope_key: Optional[str] = None,
) -> Any:
    async for chunk in completion_stream:
        yield _restore_google_adapter_tool_call_names(
            chunk,
            tool_name_mapping,
            scope_key=scope_key,
        )


async def _collect_google_code_assist_model_response_from_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    logging_obj: Any,
) -> Any:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
        GeminiPassthroughLoggingHandler,
    )

    all_chunks: list[str] = []
    body_iterator = response.body_iterator
    try:
        async for raw_chunk in body_iterator:
            if isinstance(raw_chunk, bytes):
                all_chunks.append(raw_chunk.decode("utf-8", errors="replace"))
            else:
                all_chunks.append(str(raw_chunk))
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if callable(aclose):
            await aclose()

    model_response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=all_chunks,
        litellm_logging_obj=logging_obj,
        model=_runtime()._normalize_google_completion_adapter_model_name(adapter_model),
        url_route="/v1internal:streamGenerateContent",
    )
    if model_response is None:
        raise HTTPException(
            status_code=502,
            detail="Google Code Assist streaming adapter could not build a complete response.",
        )
    return model_response


async def _collect_google_code_assist_response_from_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    logging_obj: Any,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )

    model_response = await _collect_google_code_assist_model_response_from_stream(
        response=response,
        adapter_model=adapter_model,
        logging_obj=logging_obj,
    )

    anthropic_response = AnthropicAdapter().translate_completion_output_params(
        model_response,
        tool_name_mapping=tool_name_mapping,
    )
    return _runtime()._build_anthropic_response_from_completion_adapter_response(anthropic_response)


def _build_codex_streaming_response_from_google_code_assist_stream(
    *,
    response: StreamingResponse,
    adapter_request: SimpleNamespace,
) -> StreamingResponse:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    logging_obj = Logging(
        model=adapter_request.google_model,
        messages=adapter_request.completion_messages,
        stream=True,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id=str(uuid4()),
        function_id="codex_google_code_assist_adapter",
    )
    logging_obj.optional_params = adapter_request.gemini_optional_params
    completion_stream = ModelResponseIterator(
        streaming_response=_iterate_google_code_assist_unwrapped_stream(
            response.body_iterator,
            adapter_model=adapter_request.google_model,
            rate_limit_key=adapter_request.google_adapter_rate_limit_key,
        ),
        sync_stream=False,
        logging_obj=logging_obj,
    )
    completion_stream = _restore_google_adapter_tool_call_names_stream(
        completion_stream,
        adapter_request.tool_name_mapping,
    )
    streamwrapper = CustomStreamWrapper(
        completion_stream=completion_stream,
        model=adapter_request.google_model,
        custom_llm_provider=litellm.LlmProviders.GEMINI.value,
        logging_obj=logging_obj,
    )
    responses_iterator = LiteLLMCompletionStreamingIterator(
        model=adapter_request.google_model,
        litellm_custom_stream_wrapper=streamwrapper,
        request_input=adapter_request.codex_request_input,
        responses_api_request=adapter_request.responses_api_request,
        custom_llm_provider=litellm.LlmProviders.GEMINI.value,
        litellm_metadata=adapter_request.litellm_metadata,
    )
    return StreamingResponse(
        _runtime()._responses_sse_from_iterator(responses_iterator),
        media_type="text/event-stream",
    )


def _build_google_debug_header_summary(headers: dict[str, Any]) -> dict[str, Any]:
    interesting_keys = (
        "authorization",
        "user-agent",
        "x-goog-api-client",
        "x-client-info",
        "x-goog-user-project",
        "origin",
        "referer",
        "accept",
    )
    summary: dict[str, Any] = {}
    for key in interesting_keys:
        value = headers.get(key) or headers.get(key.title())
        if not isinstance(value, str) or not value:
            continue
        if key == "authorization":
            summary[key] = value[:12]
        else:
            summary[key] = value
    return summary


def _build_google_adapter_native_headers(*, access_token: str, model: Optional[str], accept: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": _runtime()._get_google_adapter_native_user_agent(model),
        "x-goog-api-client": _runtime()._get_google_adapter_native_api_client_header(),
        "Accept": accept,
    }


def _is_codex_google_code_assist_empty_success_model_response(
    model_response: Any,
) -> bool:
    choices = _runtime()._mapping_or_attr_get(model_response, "choices") or []
    if not isinstance(choices, list):
        return False
    if not choices:
        return _runtime()._usage_has_no_more_than_one_output_token(
            _runtime()._mapping_or_attr_get(model_response, "usage")
        )

    for choice in choices:
        message = _runtime()._mapping_or_attr_get(choice, "message")
        if message is None:
            continue
        if not _runtime()._is_codex_google_code_assist_empty_text_content(
            _runtime()._mapping_or_attr_get(message, "content")
        ):
            return False
        if _runtime()._mapping_or_attr_get(message, "tool_calls"):
            return False
        if _runtime()._mapping_or_attr_get(message, "function_call"):
            return False

    # RR-054 #57: empty choice list / empty messages share the same usage gate.
    return _runtime()._usage_has_no_more_than_one_output_token(_runtime()._mapping_or_attr_get(model_response, "usage"))


def _sanitize_google_code_assist_request_schemas(wrapped_request_body: Any) -> int:
    sanitized_schema_fix_count = 0
    request_payload = wrapped_request_body.get("request") if isinstance(wrapped_request_body, dict) else None
    request_tools = request_payload.get("tools") if isinstance(request_payload, dict) else None
    if not isinstance(request_tools, list):
        return sanitized_schema_fix_count

    for tool_entry in request_tools:
        if not isinstance(tool_entry, dict):
            continue
        decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
        if not isinstance(decls, list):
            continue
        for declaration in decls:
            if not isinstance(declaration, dict):
                continue
            parameters = declaration.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
                declaration["parameters"] = parameters
                sanitized_schema_fix_count += 1
            sanitized_schema_fix_count += _runtime()._sanitize_google_code_assist_tool_schema(parameters)
    return sanitized_schema_fix_count
