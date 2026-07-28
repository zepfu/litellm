"""Wave 6A Author B extraction: provider-neutral SSE framing, event iteration,
event summary, and streaming-response builder functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.

Owned symbols:
    _serialize_responses_adapter_response
    _responses_sse_from_iterator
    _iterate_responses_sse_events
    _mapping_or_attr_get
    _coerce_namespace_to_mapping
    _responses_event_text_key
    _responses_stream_event_summary
    _responses_repaired_output_item_id
    _responses_sse_from_repaired_response_body
    _build_anthropic_streaming_response_from_responses_stream
    _build_anthropic_streaming_response_from_completion_adapter_stream

Integration seams (resolved via host globals after install()):
    _stringify_grok_native_input_item_value  (grok normalization, not owned here)

Explicitly excluded (owned elsewhere):
    - Stream accumulation/finalization (_collect_responses_response_from_stream, etc.)
    - Custom/namespace tool restoration (_restore_adapted_*_tool_calls_*)
    - Bounded payload replay validation (_validate_alias_candidate_responses_stream_if_needed)
    - All Google-specific streaming (_build_anthropic_streaming_response_from_google_code_assist_stream, etc.)
    - Provider request preparation (_prepare_anthropic_google_completion_adapter_request, etc.)
"""

from __future__ import annotations

import codecs
import json
from inspect import isawaitable
from types import FunctionType, SimpleNamespace
from typing import Any, Callable, Optional

from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global seams: functions owned by other modules, resolved after install()
    def _stringify_grok_native_input_item_value(value: Any) -> str: ...


# Replicated constant from god module (line 345).
_AAWM_REQUEST_BODY_WALK_MAX_DEPTH = 64


_HOST_FUNCTION_NAMES = (
    "_serialize_responses_adapter_response",
    "_responses_sse_from_iterator",
    "_iterate_responses_sse_events",
    "_mapping_or_attr_get",
    "_coerce_namespace_to_mapping",
    "_responses_event_text_key",
    "_responses_stream_event_summary",
    "_responses_repaired_output_item_id",
    "_responses_sse_from_repaired_response_body",
    "_build_anthropic_streaming_response_from_responses_stream",
    "_build_anthropic_streaming_response_from_completion_adapter_stream",
)

_HOST_GLOBAL_DEFAULTS = (
    ("SimpleNamespace", SimpleNamespace),
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    for _dependency_name, _dependency in _HOST_GLOBAL_DEFAULTS:
        host_globals.setdefault(_dependency_name, _dependency)
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound


# ── Extracted functions ─────────────────────────────────────────────


def _mapping_or_attr_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_namespace_to_mapping(
    value: Any,
    *,
    _depth: int = 0,
    _max_depth: int = _AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
) -> Any:
    # RR-054 #27: reverse conversion is also depth-bounded.
    if _depth > _max_depth:
        if isinstance(value, SimpleNamespace):
            return vars(value)
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, SimpleNamespace):
        return {
            key: _coerce_namespace_to_mapping(val, _depth=_depth + 1, _max_depth=_max_depth)
            for key, val in vars(value).items()
        }
    if isinstance(value, list):
        return [_coerce_namespace_to_mapping(item, _depth=_depth + 1, _max_depth=_max_depth) for item in value]
    return value


def _serialize_responses_adapter_response(response_obj: Any) -> str:
    if hasattr(response_obj, "model_dump_json"):
        return response_obj.model_dump_json(exclude_none=True)
    if hasattr(response_obj, "json"):
        return response_obj.json(exclude_none=True)
    return json.dumps(response_obj)


async def _iterate_responses_sse_events(
    body_iterator: Any,
) -> Any:
    """Yield parsed SSE event dicts (RR-054 #27: no dict<->namespace round-trip)."""
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    async for raw_chunk in body_iterator:
        if isinstance(raw_chunk, bytes):
            buffer += decoder.decode(raw_chunk)
        else:
            buffer += str(raw_chunk)

        while "\n\n" in buffer:
            event_block, buffer = buffer.split("\n\n", 1)
            for line in event_block.splitlines():
                parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
                if parsed_chunk is not None:
                    # Prefer plain dicts; consumers already accept dict or attr form.
                    yield parsed_chunk

    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        for line in buffer.splitlines():
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
            if parsed_chunk is not None:
                yield parsed_chunk


async def _responses_sse_from_iterator(
    responses_iterator: Any,
    on_complete: Optional[Callable[[], None]] = None,
) -> Any:
    try:
        async for event in responses_iterator:
            event_type = _mapping_or_attr_get(event, "type")
            serialized = _serialize_responses_adapter_response(event)
            if isinstance(event_type, str) and event_type:
                yield f"event: {event_type}\ndata: {serialized}\n\n"
                continue
            yield f"data: {serialized}\n\n"
        if on_complete is not None:
            on_complete()
        yield "data: [DONE]\n\n"
    finally:
        close_targets = (
            responses_iterator,
            getattr(responses_iterator, "litellm_custom_stream_wrapper", None),
        )
        closed_target_ids: set[int] = set()
        for close_target in close_targets:
            if close_target is None or id(close_target) in closed_target_ids:
                continue
            closed_target_ids.add(id(close_target))
            close_fn = getattr(close_target, "aclose", None)
            if not callable(close_fn):
                close_fn = getattr(close_target, "close", None)
            if not callable(close_fn):
                continue
            try:
                close_result = close_fn()
                if isawaitable(close_result):
                    await close_result
            except BaseException:
                verbose_proxy_logger.debug(
                    "Failed to close Responses adapter stream resource",
                    exc_info=True,
                )


def _responses_event_text_key(event: Any) -> str:
    # RR-054 #27: events may be dicts or attr objects.
    item_id = _mapping_or_attr_get(event, "item_id")
    if isinstance(item_id, str) and item_id:
        return item_id
    # RR-054 #22: treat output_index=0 as valid (do not use `or` falsy fallback).
    if isinstance(event, dict) and "output_index" in event:
        output_index = event.get("output_index")
    else:
        output_index = _mapping_or_attr_get(event, "output_index")
    if isinstance(output_index, int):
        return f"output:{output_index}"
    return "output:0"


def _responses_stream_event_summary(event: Any) -> dict[str, Any]:
    event_type = _mapping_or_attr_get(event, "type")
    summary: dict[str, Any] = {"type": event_type}
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = _mapping_or_attr_get(event, "item")
        if item is not None:
            summary["item_type"] = _mapping_or_attr_get(item, "type")
            summary["item_id"] = _mapping_or_attr_get(item, "id")
            summary["item_name"] = _mapping_or_attr_get(item, "name")
        return summary
    if event_type in {
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.reasoning_summary_text.delta",
    }:
        summary["item_id"] = _mapping_or_attr_get(event, "item_id")
        text = _mapping_or_attr_get(event, "delta")
        if text is None:
            text = _mapping_or_attr_get(event, "arguments")
        if text is None:
            text = _mapping_or_attr_get(event, "text")
        if isinstance(text, str):
            summary["text_len"] = len(text)
            summary["text_preview"] = text[:200]
        return summary
    if event_type in {
        "response.completed",
        "response.failed",
        "response.incomplete",
    }:
        response_payload = _mapping_or_attr_get(event, "response")
        response_dict = _coerce_namespace_to_mapping(response_payload)
        if isinstance(response_dict, dict):
            output = response_dict.get("output") or []
            usage = response_dict.get("usage") or {}
            summary.update(
                {
                    "response_id": response_dict.get("id"),
                    "response_status": response_dict.get("status"),
                    "response_model": response_dict.get("model"),
                    "output_count": len(output) if isinstance(output, list) else 0,
                    "output_types": [item.get("type") for item in output[:20] if isinstance(item, dict)]
                    if isinstance(output, list)
                    else [],
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0) if isinstance(usage, dict) else 0,
                        "output_tokens": usage.get("output_tokens", 0) if isinstance(usage, dict) else 0,
                    },
                }
            )
    return summary


def _responses_repaired_output_item_id(item: dict[str, Any], index: int) -> str:
    for key in ("id", "call_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"item_{index}"


async def _responses_sse_from_repaired_response_body(
    response_body: dict[str, Any],
) -> Any:
    output = response_body.get("output")
    if not isinstance(output, list):
        output = []
    for index, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        item_id = _responses_repaired_output_item_id(item, index)
        yield (
            "event: response.output_item.added\n"
            + "data: "
            + json.dumps(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        if item.get("type") == "function_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = _stringify_grok_native_input_item_value(arguments)  # noqa: F821
            yield (
                "event: response.function_call_arguments.done\n"
                + "data: "
                + json.dumps(
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item_id,
                        "output_index": index,
                        "arguments": arguments,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        yield (
            "event: response.output_item.done\n"
            + "data: "
            + json.dumps(
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
    yield (
        "event: response.completed\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": response_body,
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


def _build_anthropic_streaming_response_from_responses_stream(
    response: StreamingResponse,
    *,
    model: str,
    request_body: Optional[dict[str, Any]] = None,
    reject_empty_success: bool = False,
    use_codex_native_tools: bool = False,
) -> StreamingResponse:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
        AnthropicResponsesStreamWrapper,
    )

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_iterate_responses_sse_events(response.body_iterator),
        model=model,
        request_body=request_body,
        reject_empty_success=reject_empty_success,
        use_codex_native_tools=use_codex_native_tools,
    )
    return StreamingResponse(
        wrapper.async_anthropic_sse_wrapper(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type="text/event-stream",
    )


def _build_anthropic_streaming_response_from_completion_adapter_stream(
    response_stream: Any,
) -> StreamingResponse:
    return StreamingResponse(
        response_stream,
        media_type="text/event-stream",
    )
