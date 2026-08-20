"""Post-upstream literal tool-call validation for direct `/grok/v1/responses`.

Reuses Grok Composer repair. Direct Grok is not alias routing: malformed
literal tool text is a retryable 502 with content-free intake and no Redis
cooldown.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from fastapi import Response
from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_grok_literal_tool_label_transcript_text,
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    composer_repair,
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok.side_channel import (
    _normalize_grok_endpoint_path,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.aawm_runtime_error_logging import (
    schedule_persist_malformed_tool_call_detection,
)

DIRECT_GROK_RESPONSES_ADAPTER = "direct_grok_responses"
DIRECT_GROK_RESPONSES_ADAPTER_LABEL = "Direct Grok Responses"
_LITERAL_TOOL_LABEL_LINE_RE = re.compile(r"(?im)^Tool label:\s*\S")
_MALFORMED_TOOL_CALL_ERROR_CODE = "aawm_auto_agent_malformed_tool_call_text"
_MALFORMED_TOOL_CALL_FAILURE_KIND = "malformed_tool_call"


def _passthrough_host() -> Any:
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

    return lpe


def is_direct_grok_json_responses_endpoint(endpoint: str) -> bool:
    path = _normalize_grok_endpoint_path(endpoint)
    return path == "/responses" or path.startswith("/responses/")


def should_validate_direct_grok_responses(
    *,
    endpoint: str,
    request: Any,
    raw_body_passthrough: bool,
    request_body: Any,
) -> bool:
    if raw_body_passthrough:
        return False
    method = str(getattr(request, "method", "") or "").upper()
    if method not in {"POST", "PUT", "PATCH"}:
        return False
    if not isinstance(request_body, dict):
        return False
    return is_direct_grok_json_responses_endpoint(endpoint)


def _chunk_size(chunk: object) -> int:
    if isinstance(chunk, (bytes, bytearray)):
        return len(chunk)
    return len(str(chunk).encode("utf-8", errors="replace"))


def _decode_stream_chunks(chunks: list[Any]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            parts.append(chunk.decode("utf-8", errors="replace"))
        elif isinstance(chunk, str):
            parts.append(chunk)
        else:
            parts.append(str(chunk))
    return "".join(parts)


def _response_body_output_text(response_body: dict[str, Any]) -> str:
    output = response_body.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"text", "output_text"}:
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _iter_sse_event_payloads(rendered: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in rendered.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _text_has_literal_tool_label_marker(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if is_malformed_grok_literal_tool_label_transcript_text(text):
        return True
    return _LITERAL_TOOL_LABEL_LINE_RE.search(text) is not None


def _buffered_sse_has_literal_tool_label_marker(chunks: list[Any]) -> bool:
    rendered = _decode_stream_chunks(chunks)
    if "Tool label:" not in rendered:
        return False
    texts: list[str] = []
    for event in _iter_sse_event_payloads(rendered):
        delta = event.get("delta")
        if isinstance(delta, str):
            texts.append(delta)
        text = event.get("text")
        if isinstance(text, str):
            texts.append(text)
        response = event.get("response")
        if isinstance(response, dict):
            texts.append(_response_body_output_text(response))
    assembled = "".join(texts)
    if assembled:
        return _text_has_literal_tool_label_marker(assembled)
    return True


def _decode_http_response_body(body: Any) -> str:
    host = _passthrough_host()
    decode = getattr(host, "_decode_http_response_body", None)
    if callable(decode):
        return decode(body)
    return bytes(body).decode("utf-8", errors="replace")


def _adapter_model(request_body: Optional[dict[str, Any]], response_body: dict[str, Any]) -> str:
    for source in (request_body, response_body):
        if not isinstance(source, dict):
            continue
        model = source.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return "grok-build"


def _intake_context(
    *,
    host: Any,
    request: Any,
    request_body: Optional[dict[str, Any]],
    endpoint: Optional[str],
) -> dict[str, Any]:
    context = host._build_malformed_tool_call_intake_context(
        request,
        request_body,
        adapter=DIRECT_GROK_RESPONSES_ADAPTER,
        provider="xai",
        model_alias=(
            request_body.get("model")
            if isinstance(request_body, dict) and isinstance(request_body.get("model"), str)
            else None
        ),
    )
    context["redispatch_required"] = False
    context["terminal_outcome"] = "malformed_tool_call_rejected"
    context["fallback_result"] = "none"
    context["route_family"] = DIRECT_GROK_RESPONSES_ADAPTER
    if endpoint and not context.get("endpoint"):
        context["endpoint"] = endpoint
    advertised_tools = host._build_advertised_openai_function_tools_index(request_body)
    context["advertised_tool_count"] = len(advertised_tools)
    return context


def _raise_direct_grok_malformed_tool_call(
    *,
    response_body: dict[str, Any],
    request: Any,
    request_body: Optional[dict[str, Any]],
    endpoint: Optional[str] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    host = _passthrough_host()
    adapter_model = _adapter_model(request_body, response_body)
    intake_context = _intake_context(
        host=host,
        request=request,
        request_body=request_body,
        endpoint=endpoint,
    )
    try:
        schedule_persist_malformed_tool_call_detection(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=DIRECT_GROK_RESPONSES_ADAPTER,
            adapter_label=DIRECT_GROK_RESPONSES_ADAPTER_LABEL,
            intake_context=intake_context,
            stream_event_summaries=stream_event_summaries,
        )
    except Exception:
        verbose_proxy_logger.exception(
            "Failed to schedule direct Grok malformed tool-call detection intake"
        )
    diagnostic = {
        "adapter": DIRECT_GROK_RESPONSES_ADAPTER,
        "route": "direct_grok_responses",
        "adapter_model": adapter_model,
        "response_id": response_body.get("id"),
        "status": response_body.get("status"),
        "advertised_tool_count": intake_context.get("advertised_tool_count"),
    }
    if stream_event_summaries is not None:
        diagnostic["stream_events"] = stream_event_summaries
    exc = ProxyException(
        message="Direct Grok Responses returned malformed literal tool-call text.",
        type="invalid_request_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": _MALFORMED_TOOL_CALL_ERROR_CODE,
                "type": "invalid_request_error",
                "failure_kind": _MALFORMED_TOOL_CALL_FAILURE_KIND,
            },
            "failure_kind": _MALFORMED_TOOL_CALL_FAILURE_KIND,
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _repair_or_reject_response_body(
    response_body: dict[str, Any],
    *,
    request: Any,
    request_body: Optional[dict[str, Any]],
    endpoint: Optional[str] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    host = _passthrough_host()
    runtime = host._get_anthropic_grok_composer_repair_runtime()
    repaired_body = composer_repair.try_repair_literal_tool_call_response_body(
        runtime,
        response_body,
        request_body=request_body,
    )
    if isinstance(repaired_body, dict):
        return repaired_body
    if host._is_codex_auto_agent_malformed_tool_call_text_output(
        response_body
    ) or composer_repair.response_body_has_literal_tool_label_blocks(
        runtime, response_body
    ):
        _raise_direct_grok_malformed_tool_call(
            response_body=response_body,
            request=request,
            request_body=request_body,
            endpoint=endpoint,
            stream_event_summaries=stream_event_summaries,
        )
    return None


def _streaming_response_from_chunks(
    chunks: list[Any],
    *,
    response: StreamingResponse,
) -> StreamingResponse:
    async def _replay() -> Any:
        for chunk in chunks:
            yield chunk

    return StreamingResponse(
        _replay(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


async def _extend_marked_stream_until_exhausted_or_ceiling(
    peek: Any,
    *,
    max_chunks: int,
    max_bytes: int,
) -> Optional[list[Any]]:
    chunks = list(peek.buffered_chunks)
    nbytes = peek.buffered_bytes
    skip = len(peek.buffered_chunks)
    seen = 0
    async for chunk in peek.response.body_iterator:
        if seen < skip:
            seen += 1
            continue
        size = _chunk_size(chunk)
        if len(chunks) >= max_chunks or nbytes + size > max_bytes:
            return None
        chunks.append(chunk)
        nbytes += size
    return chunks


async def _validate_direct_grok_streaming_response(
    response: StreamingResponse,
    *,
    request: Any,
    request_body: Optional[dict[str, Any]],
    endpoint: Optional[str] = None,
) -> StreamingResponse:
    host = _passthrough_host()
    max_chunks = host._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS
    max_bytes = host._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES
    peek = await host._aawm_alias_streaming.peek_streaming_response(
        response,
        max_chunks=max_chunks,
        max_bytes=max_bytes,
        terminalizer=host._aawm_alias_streaming._get_stream_timeout_terminalizer(
            response
        ),
    )
    has_marker = _buffered_sse_has_literal_tool_label_marker(peek.buffered_chunks)
    if not has_marker:
        return peek.response

    collected_chunks: Optional[list[Any]]
    if peek.exhausted:
        collected_chunks = list(peek.buffered_chunks)
    elif peek.stop_reason in {"chunk_limit", "byte_limit"}:
        _raise_direct_grok_malformed_tool_call(
            response_body={
                "status": "incomplete",
                "model": _adapter_model(request_body, {}),
                "output": [],
            },
            request=request,
            request_body=request_body,
            endpoint=endpoint,
        )
        raise AssertionError("unreachable")
    else:
        collected_chunks = await _extend_marked_stream_until_exhausted_or_ceiling(
            peek,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )
        if collected_chunks is None:
            _raise_direct_grok_malformed_tool_call(
                response_body={
                    "status": "incomplete",
                    "model": _adapter_model(request_body, {}),
                    "output": [],
                },
                request=request,
                request_body=request_body,
                endpoint=endpoint,
            )
            raise AssertionError("unreachable")

    replay = _streaming_response_from_chunks(collected_chunks, response=response)
    event_summaries: list[dict[str, Any]] = []
    try:
        response_body = await host._collect_responses_response_from_stream(
            replay,
            event_summaries=event_summaries,
        )
    except Exception:
        _raise_direct_grok_malformed_tool_call(
            response_body={
                "status": "incomplete",
                "model": _adapter_model(request_body, {}),
                "output": [],
            },
            request=request,
            request_body=request_body,
            endpoint=endpoint,
            stream_event_summaries=event_summaries or None,
        )
        raise AssertionError("unreachable")

    if not isinstance(response_body, dict):
        return _streaming_response_from_chunks(collected_chunks, response=response)

    repaired_body = _repair_or_reject_response_body(
        response_body,
        request=request,
        request_body=request_body,
        endpoint=endpoint,
        stream_event_summaries=event_summaries,
    )
    if repaired_body is None:
        return _streaming_response_from_chunks(collected_chunks, response=response)
    return StreamingResponse(
        host._responses_sse_from_repaired_response_body(repaired_body),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


def _validate_direct_grok_json_response(
    response: Response,
    *,
    request: Any,
    request_body: Optional[dict[str, Any]],
    endpoint: Optional[str] = None,
) -> Response:
    try:
        response_body = json.loads(_decode_http_response_body(response.body))
    except Exception:
        return response
    if not isinstance(response_body, dict):
        return response
    repaired_body = _repair_or_reject_response_body(
        response_body,
        request=request,
        request_body=request_body,
        endpoint=endpoint,
    )
    if repaired_body is None:
        return response
    return Response(
        content=json.dumps(repaired_body),
        media_type=response.media_type or "application/json",
        status_code=response.status_code,
        headers=dict(response.headers),
    )


async def validate_direct_grok_responses_payload(
    response: Any,
    *,
    request: Any,
    request_body: Optional[dict[str, Any]] = None,
    endpoint: Optional[str] = None,
) -> Any:
    status_code = getattr(response, "status_code", None)
    try:
        if status_code is not None and int(status_code) >= 400:
            return response
    except (TypeError, ValueError):
        pass
    if isinstance(response, StreamingResponse):
        return await _validate_direct_grok_streaming_response(
            response,
            request=request,
            request_body=request_body,
            endpoint=endpoint,
        )
    if isinstance(response, Response):
        return _validate_direct_grok_json_response(
            response,
            request=request,
            request_body=request_body,
            endpoint=endpoint,
        )
    return response
