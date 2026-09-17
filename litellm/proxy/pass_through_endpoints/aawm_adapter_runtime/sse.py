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
    - Streaming response builders for providers not retained in this runtime
    - Provider request preparation for providers not retained in this runtime
"""

from __future__ import annotations

import codecs
import json
from inspect import isawaitable
from types import FunctionType, SimpleNamespace
from typing import Any, Callable, Optional

from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.types.llms.openai import RESPONSES_API_TERMINAL_STREAM_EVENTS

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
    "_coerce_sequence_number",
    "_ensure_responses_sse_sequence_number",
    "_event_sequence_number",
    "_ensure_reasoning_item_summary",
    "_ensure_response_text_format",
    "_ensure_response_usage_details",
    "_ensure_responses_event_indexes",
    "_ensure_grok_responses_sse_compat",
    "_reattach_sequence_number_json",
    "_responses_event_text_key",
    "_responses_stream_event_summary",
    "_responses_repaired_output_item_id",
    "_responses_sse_from_repaired_response_body",
    "_build_anthropic_streaming_response_from_responses_stream",
    "_build_anthropic_streaming_response_from_completion_adapter_stream",
)

_GROK_SSE_DEFAULT_INDEX_FIELDS: dict[str, tuple[str, ...]] = {
    "response.reasoning_summary_text.delta": ("summary_index",),
    "response.reasoning_summary_text.done": ("summary_index",),
    "response.reasoning_summary_part.added": ("summary_index",),
    "response.reasoning_summary_part.done": ("summary_index",),
    "response.output_text.delta": ("content_index",),
    "response.output_text.done": ("content_index",),
    "response.output_text.annotation.added": ("content_index", "annotation_index"),
    "response.content_part.added": ("content_index",),
    "response.content_part.done": ("content_index",),
    "response.refusal.delta": ("content_index",),
    "response.refusal.done": ("content_index",),
}

_HOST_GLOBAL_DEFAULTS = (
    ("SimpleNamespace", SimpleNamespace),
    ("RESPONSES_API_TERMINAL_STREAM_EVENTS", RESPONSES_API_TERMINAL_STREAM_EVENTS),
    ("_GROK_SSE_DEFAULT_INDEX_FIELDS", _GROK_SSE_DEFAULT_INDEX_FIELDS),
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


def _coerce_sequence_number(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _ensure_responses_sse_sequence_number(
    payload: Any,
    *,
    sequence_number: int,
) -> Any:
    """Grok Build requires OpenAI Responses SSE `sequence_number` on every event.

    Adapter-produced events often omit it (LiteLLM's ResponseCreatedEvent type
    does not declare the field). Preserve an existing non-negative int; otherwise
    stamp the caller-assigned counter. Extra fields are allowed on
    BaseLiteLLMOpenAIResponseObject.
    """

    existing = _coerce_sequence_number(_mapping_or_attr_get(payload, "sequence_number"))
    if existing is None and hasattr(payload, "__dict__"):
        existing = _coerce_sequence_number(payload.__dict__.get("sequence_number"))
    assigned = existing if existing is not None else sequence_number
    if isinstance(payload, dict):
        payload["sequence_number"] = assigned
        return payload
    try:
        setattr(payload, "sequence_number", assigned)
    except Exception:
        verbose_proxy_logger.debug(
            "Failed to stamp sequence_number on Responses SSE event",
            extra={"event_type": str(_mapping_or_attr_get(payload, "type"))},
        )
        if hasattr(payload, "__dict__"):
            payload.__dict__["sequence_number"] = assigned
    return payload


def _event_sequence_number(response_obj: Any) -> Optional[int]:
    seq = _coerce_sequence_number(_mapping_or_attr_get(response_obj, "sequence_number"))
    if seq is not None:
        return seq
    if hasattr(response_obj, "__dict__"):
        return _coerce_sequence_number(response_obj.__dict__.get("sequence_number"))
    return None


def _ensure_reasoning_item_summary(item: Any) -> bool:
    """Grok Build requires `summary` on Responses reasoning items.

    Adapter reasoning items often omit it. Stamp an empty list without
    overwriting an existing summary.
    """

    if not isinstance(item, dict):
        return False
    if item.get("type") != "reasoning":
        return False
    if "summary" in item:
        return False
    item["summary"] = []
    return True


def _ensure_response_text_format(response: Any) -> bool:
    """Grok Build requires `text.format` on Responses envelopes.

    Adapter `response.completed` events often emit `"text": {}`. Stamp the
    OpenAI default `{type: text}` format without overwriting an existing one.
    """

    if not isinstance(response, dict):
        return False
    text = response.get("text")
    if text is None:
        response["text"] = {"format": {"type": "text"}}
        return True
    if not isinstance(text, dict):
        return False
    if "format" in text:
        return False
    text["format"] = {"type": "text"}
    response["text"] = text
    return True


def _ensure_response_usage_details(response: Any) -> bool:
    """Grok Build requires `usage.input_tokens_details` on Responses envelopes."""

    if not isinstance(response, dict):
        return False
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return False
    changed = False
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        usage["input_tokens_details"] = {"cached_tokens": 0}
        changed = True
    elif "cached_tokens" not in details:
        details["cached_tokens"] = 0
        usage["input_tokens_details"] = details
        changed = True
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        usage["output_tokens_details"] = {"reasoning_tokens": 0}
        changed = True
    elif "reasoning_tokens" not in output_details:
        output_details["reasoning_tokens"] = 0
        usage["output_tokens_details"] = output_details
        changed = True
    response["usage"] = usage
    return changed


def _ensure_responses_event_indexes(payload: dict[str, Any]) -> bool:
    """Stamp missing integer indexes Grok Build requires on Responses SSE events."""

    event_type = payload.get("type")
    if not isinstance(event_type, str):
        return False
    fields = _GROK_SSE_DEFAULT_INDEX_FIELDS.get(event_type)
    if not fields:
        return False
    changed = False
    for field in fields:
        if field in payload:
            continue
        payload[field] = 0
        changed = True
    return changed


def _ensure_grok_responses_sse_compat(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("item")
    if isinstance(item, dict):
        _ensure_reasoning_item_summary(item)
        payload["item"] = item
    response = payload.get("response")
    if isinstance(response, dict):
        _ensure_response_text_format(response)
        _ensure_response_usage_details(response)
        output = response.get("output")
        if isinstance(output, list):
            for entry in output:
                _ensure_reasoning_item_summary(entry)
    _ensure_responses_event_indexes(payload)
    return payload


def _reattach_sequence_number_json(serialized: str, seq: Optional[int]) -> str:
    try:
        payload = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError):
        return serialized
    if not isinstance(payload, dict):
        return serialized
    changed = False
    if seq is not None and payload.get("sequence_number") != seq:
        payload["sequence_number"] = seq
        changed = True
    before = json.dumps(payload, sort_keys=True)
    _ensure_grok_responses_sse_compat(payload)
    after = json.dumps(payload, sort_keys=True)
    if not changed and before == after:
        return serialized
    return json.dumps(payload)


def _serialize_responses_adapter_response(response_obj: Any) -> str:
    """Serialize one Responses SSE event, keeping Grok-required wire fields.

    Pydantic `model_dump_json(exclude_none=True)` drops extra `__dict__` values
    that adapter iterators stamp outside declared fields. Grok Build requires
    `sequence_number` on every event and `summary` on reasoning items.
    """

    seq = _event_sequence_number(response_obj)
    if hasattr(response_obj, "model_dump_json"):
        return _reattach_sequence_number_json(
            response_obj.model_dump_json(exclude_none=True),
            seq,
        )
    if hasattr(response_obj, "json"):
        return _reattach_sequence_number_json(
            response_obj.json(exclude_none=True),
            seq,
        )
    if isinstance(response_obj, dict):
        payload = dict(response_obj)
        if seq is not None:
            payload["sequence_number"] = seq
        _ensure_grok_responses_sse_compat(payload)
        return json.dumps(payload)
    return json.dumps(response_obj)


async def _iter_sse_event_blocks_with_separator(body_iterator: Any):
    """Yield raw SSE event blocks from an async body iterator.

    Supports LF (`\n`), CRLF (`\r\n`), and CR (`\r`) line endings even when
    boundaries split across chunks. Uses an incremental UTF-8 decoder so split
    code points are preserved and flushes a final unterminated block at EOF.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    trailing_cr = False

    async for raw_chunk in body_iterator:
        text_chunk = decoder.decode(raw_chunk) if isinstance(raw_chunk, bytes) else str(raw_chunk)
        if not text_chunk:
            continue

        if trailing_cr:
            text_chunk = f"\r{text_chunk}"
            trailing_cr = False

        if text_chunk.endswith("\r"):
            text_chunk = text_chunk[:-1]
            trailing_cr = True

        if text_chunk:
            buffer += text_chunk.replace("\r\n", "\n").replace("\r", "\n")

        while "\n\n" in buffer:
            event_block, buffer = buffer.split("\n\n", 1)
            yield event_block, True

    tail = decoder.decode(b"", final=True)
    if trailing_cr:
        tail = f"\r{tail}"

    if tail:
        buffer += tail.replace("\r\n", "\n").replace("\r", "\n")

    while "\n\n" in buffer:
        event_block, buffer = buffer.split("\n\n", 1)
        yield event_block, True

    if buffer:
        yield buffer, False


async def _iter_sse_event_blocks(body_iterator: Any):
    """Yield raw SSE event blocks from an async body iterator."""
    async for event_block, _ in _iter_sse_event_blocks_with_separator(body_iterator):
        yield event_block


async def _iterate_responses_sse_events(
    body_iterator: Any,
    _iter_sse_blocks=_iter_sse_event_blocks,
) -> Any:
    """Yield parsed SSE event dicts (RR-054 #27: no dict<->namespace round-trip)."""
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

    async for event_block in _iter_sse_blocks(body_iterator):
        for line in event_block.splitlines():
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
            if parsed_chunk is not None:
                # Prefer plain dicts; consumers already accept dict or attr form.
                yield parsed_chunk


async def _responses_sse_from_iterator(
    responses_iterator: Any,
    on_complete: Optional[Callable[[], None]] = None,
    on_stream_error: Optional[Callable[[Exception], Optional[str]]] = None,
    *,
    request_body: Optional[dict[str, Any]] = None,
) -> Any:
    has_emitted = False
    has_terminal = False
    last_event: Any = None
    sse_sequence = 0

    def _identity_request_body() -> Optional[dict[str, Any]]:
        if isinstance(request_body, dict):
            return request_body
        metadata = getattr(responses_iterator, "litellm_metadata", None)
        if isinstance(metadata, dict) and metadata:
            return {"litellm_metadata": metadata}
        return None

    def _stamp_sse_text(sse_text: str) -> str:
        identity_body = _identity_request_body()
        if not sse_text or not identity_body:
            return sse_text
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            stamp_route_identity_in_sse_chunk,
        )

        stamped = stamp_route_identity_in_sse_chunk(
            sse_text,
            request_body=identity_body,
        )
        return stamped if isinstance(stamped, str) else sse_text

    def _injected_terminal_sse(*, event_type: str, status: str) -> str:
        nonlocal sse_sequence
        last_response = (
            _mapping_or_attr_get(last_event, "response")
            if last_event is not None
            else None
        )
        response_payload: dict[str, Any] = {
            "object": "response",
            "status": status,
            "output": [],
        }
        last_id = (
            _mapping_or_attr_get(last_response, "id")
            if last_response is not None
            else None
        )
        if last_id is None and last_event is not None:
            last_id = _mapping_or_attr_get(last_event, "id")
        if isinstance(last_id, str) and last_id:
            response_payload["id"] = last_id
        created_at = None
        if last_response is not None:
            created_at = _mapping_or_attr_get(last_response, "created_at")
            if created_at is None:
                created_at = _mapping_or_attr_get(last_response, "created")
        if created_at is None and last_event is not None:
            created_at = _mapping_or_attr_get(last_event, "created_at")
            if created_at is None:
                created_at = _mapping_or_attr_get(last_event, "created")
        try:
            response_payload["created_at"] = int(created_at)
        except (TypeError, ValueError):
            response_payload["created_at"] = int(__import__("time").time())
        last_model = (
            _mapping_or_attr_get(last_response, "model")
            if last_response is not None
            else None
        )
        if last_model is None and last_event is not None:
            last_model = _mapping_or_attr_get(last_event, "model")
        if isinstance(last_model, str) and last_model:
            response_payload["model"] = last_model
        injected = {
            "type": event_type,
            "response": response_payload,
        }
        sse_sequence += 1
        _ensure_responses_sse_sequence_number(injected, sequence_number=sse_sequence)
        serialized = _serialize_responses_adapter_response(injected)
        return f"event: {event_type}\ndata: {serialized}\n\n"

    try:
        async for event in responses_iterator:
            has_emitted = True
            last_event = event
            event_type = _mapping_or_attr_get(event, "type")
            if hasattr(event_type, "value"):
                event_type = event_type.value
            if (
                isinstance(event_type, str)
                and event_type in RESPONSES_API_TERMINAL_STREAM_EVENTS
            ):
                has_terminal = True
            existing_seq = _coerce_sequence_number(
                _mapping_or_attr_get(event, "sequence_number")
            )
            if existing_seq is None:
                sse_sequence += 1
                assigned_seq = sse_sequence
            else:
                assigned_seq = existing_seq
                if existing_seq > sse_sequence:
                    sse_sequence = existing_seq
            _ensure_responses_sse_sequence_number(
                event, sequence_number=assigned_seq
            )
            serialized = _serialize_responses_adapter_response(event)
            if isinstance(event_type, str) and event_type:
                yield _stamp_sse_text(
                    f"event: {event_type}\ndata: {serialized}\n\n",
                )
                continue
            yield _stamp_sse_text(f"data: {serialized}\n\n")
        if not has_terminal:
            yield _stamp_sse_text(
                _injected_terminal_sse(
                    event_type="response.completed",
                    status="completed",
                ),
            )
        if on_complete is not None:
            on_complete()
        yield "data: [DONE]\n\n"
    except Exception as stream_exc:
        if not has_emitted:
            raise
        if on_stream_error is not None:
            terminal_event = on_stream_error(stream_exc)
            if terminal_event is not None:
                yield terminal_event
                return
        # Post-created iterator errors are real adapter failures. Do not
        # paper them over as empty Ohmypi response.completed success.
        raise
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
    *,
    request_body: Optional[dict[str, Any]] = None,
) -> Any:
    if isinstance(response_body, dict) and isinstance(request_body, dict):
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            build_producer_provenance_from_egress_context,
            stamp_route_identity_in_response,
        )

        stamp_route_identity_in_response(
            response_body,
            build_producer_provenance_from_egress_context(
                request_body=request_body,
            ),
        )
    output = response_body.get("output")
    if not isinstance(output, list):
        output = []
    sse_sequence = 0

    def _emit(event_type: str, payload: dict[str, Any]) -> str:
        nonlocal sse_sequence
        sse_sequence += 1
        event = dict(payload)
        event["type"] = event_type
        _ensure_responses_sse_sequence_number(event, sequence_number=sse_sequence)
        serialized = _serialize_responses_adapter_response(event)
        return f"event: {event_type}\ndata: {serialized}\n\n"

    for index, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        item_id = _responses_repaired_output_item_id(item, index)
        yield _emit(
            "response.output_item.added",
            {"output_index": index, "item": item},
        )
        if item.get("type") == "function_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = _stringify_grok_native_input_item_value(arguments)  # noqa: F821
            yield _emit(
                "response.function_call_arguments.done",
                {
                    "item_id": item_id,
                    "output_index": index,
                    "arguments": arguments,
                },
            )
        yield _emit(
            "response.output_item.done",
            {"output_index": index, "item": item},
        )
    yield _emit(
        "response.completed",
        {"response": response_body},
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
