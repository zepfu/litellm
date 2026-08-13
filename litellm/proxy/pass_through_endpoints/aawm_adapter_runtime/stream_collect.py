"""Wave 6A Author E extraction: stream collection pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.

Explicitly excluded from this module:
- SSE framing / event iteration (_iterate_responses_sse_events)
- Validation replay
- Provider-specific / Google streaming
- Request building

Integration seams (resolved via install() rebinding to host globals):
- _iterate_responses_sse_events: SSE event iteration (framing excluded)
- _responses_stream_event_summary: event summary construction
- _responses_event_text_key: text dedup key derivation
- _coerce_namespace_to_mapping: namespace-to-dict coercion
- _mapping_or_attr_get: dict/attr accessor
- RESPONSES_API_TERMINAL_STREAM_EVENTS: terminal event set
- HTTPException: payload error construction (fastapi)
- StreamingResponse: type annotation (starlette)
"""

from __future__ import annotations

from typing import Any, Optional

from types import FunctionType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global functions (bound via install())
    def _iterate_responses_sse_events(body_iterator: Any) -> Any: ...
    def _responses_stream_event_summary(event: Any) -> dict[str, Any]: ...
    def _responses_event_text_key(event: Any) -> str: ...
    def _coerce_namespace_to_mapping(value: Any, **kw: Any) -> Any: ...
    def _mapping_or_attr_get(obj: Any, key: str, default: Any = None) -> Any: ...

    # Host-global constants
    RESPONSES_API_TERMINAL_STREAM_EVENTS: frozenset[str]

    # Host-global classes
    class HTTPException(Exception):
        def __init__(self, *, status_code: int, detail: Any) -> None: ...

    class StreamingResponse:
        body_iterator: Any


_HOST_FUNCTION_NAMES = (
    "_responses_output_stream_key",
    "_merge_responses_output_lists",
    "_responses_output_has_message_text",
    "_build_collected_responses_text_output_item",
    "_record_collected_responses_output_item_event",
    "_record_collected_responses_arguments_event",
    "_finalize_collected_responses_stream_response",
    "_build_empty_success_responses_diagnostic",
    "_collect_responses_response_from_stream",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
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
    # Pure local helper used by rebound OPENAI-007 identity keying.
    host_globals["_function_call_identity_token"] = _mod["_function_call_identity_token"]


# -- Extracted functions -------------------------------------------------


def _function_call_identity_token(value: Any) -> Optional[str]:
    """Return a non-blank function_call identity token without stripping bytes.

    OPENAI-007: provider ``call_id`` values are byte-preserving collection keys.
    Whitespace-only placeholders are still ignored; every other non-empty string
    keeps its exact bytes so ``call_ws`` and `` call_ws`` remain distinct.
    """
    if not isinstance(value, str) or not value:
        return None
    if not value.strip():
        return None
    return value


def _responses_output_stream_key(
    *,
    item: Optional[dict[str, Any]] = None,
    output_index: Any = None,
    item_id: Any = None,
    fallback_index: Optional[int] = None,
) -> str:
    item_type = None
    if isinstance(item, dict):
        raw_type = item.get("type")
        if isinstance(raw_type, str) and raw_type.strip():
            item_type = raw_type.strip()
        # OPENAI-007: function_call provider call_id / fc_* item id are
        # byte-preserving identity keys. Do not strip or collapse byte-distinct
        # values such as "call_ws" vs " call_ws".
        if item_type == "function_call":
            for key in ("call_id", "id"):
                token = _function_call_identity_token(item.get(key))
                if token is not None:
                    return token
        else:
            for key in ("call_id", "id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    # Arguments events typically carry item_id only. Preserve non-blank bytes so
    # generated fc_* ids with surrounding whitespace cannot collapse.
    token = _function_call_identity_token(item_id)
    if token is not None:
        return token
    # RR-054 #46: include type in synthetic keys so distinct item types at the same
    # fallback index do not silently merge.
    type_prefix = f"{item_type}:" if item_type else ""
    if isinstance(output_index, int):
        return f"{type_prefix}output:{output_index}"
    if fallback_index is not None:
        return f"{type_prefix}fallback:{fallback_index}"
    return f"{type_prefix}fallback:0"


def _merge_responses_output_lists(
    completed_output: Optional[list[dict[str, Any]]],
    streamed_output: Optional[list[dict[str, Any]]],
    *,
    streamed_ordered_keys: Optional[list[str]] = None,
    key_aliases: Optional[dict[str, str]] = None,
    key_by_output_index: Optional[dict[int, str]] = None,
) -> list[dict[str, Any]]:
    merged_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    aliases = dict(key_aliases or {})
    index_keys = dict(key_by_output_index or {})

    for index, item in enumerate(streamed_output or []):
        if not isinstance(item, dict):
            continue
        key = (
            streamed_ordered_keys[index]
            if streamed_ordered_keys is not None and index < len(streamed_ordered_keys)
            else _responses_output_stream_key(item=item, fallback_index=index)
        )
        if key not in ordered_keys:
            ordered_keys.append(key)
        merged_by_key[key] = dict(item)
        index_keys.setdefault(index, key)
        item_type = item.get("type")
        if item_type == "function_call":
            for alias in (item.get("id"), item.get("call_id")):
                token = _function_call_identity_token(alias)
                if token is not None:
                    aliases[token] = key
        else:
            for alias in (item.get("id"), item.get("call_id")):
                if isinstance(alias, str) and alias.strip():
                    aliases[alias.strip()] = key

    for index, item in enumerate(completed_output or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            item_aliases = [
                token
                for token in (
                    _function_call_identity_token(item.get("call_id")),
                    _function_call_identity_token(item.get("id")),
                )
                if token is not None
            ]
        else:
            item_aliases = [
                value.strip()
                for value in (item.get("call_id"), item.get("id"))
                if isinstance(value, str) and value.strip()
            ]
        terminal_key: Optional[str] = next(
            (aliases[value] for value in item_aliases if value in aliases),
            None,
        )
        if terminal_key is None:
            terminal_key = index_keys.get(index)
        if terminal_key is None and streamed_ordered_keys is not None and index < len(streamed_ordered_keys):
            terminal_key = streamed_ordered_keys[index]
        if terminal_key is None and index < len(ordered_keys):
            terminal_key = ordered_keys[index]
        if terminal_key is None:
            terminal_key = _responses_output_stream_key(
                item=item,
                fallback_index=len(ordered_keys),
            )
        if terminal_key not in ordered_keys:
            ordered_keys.append(terminal_key)
        existing = merged_by_key.get(terminal_key, {})
        merged_item = {**existing, **item}
        if "arguments" in existing and "arguments" not in item:
            merged_item["arguments"] = existing["arguments"]
        merged_by_key[terminal_key] = merged_item
        for alias in item_aliases:
            aliases[alias] = terminal_key

    return [merged_by_key[key] for key in ordered_keys if key in merged_by_key]


def _responses_output_has_message_text(output: Any) -> bool:
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") in {"output_text", "text"}
                and isinstance(part.get("text"), str)
                and part["text"]
            ):
                return True
    return False


def _build_collected_responses_text_output_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_adapter_0",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _record_collected_responses_output_item_event(
    *,
    event: Any,
    output_items: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    key_aliases: dict[str, str],
    key_by_output_index: dict[int, str],
) -> None:
    item = _coerce_namespace_to_mapping(_mapping_or_attr_get(event, "item"))
    if not isinstance(item, dict):
        return

    output_index = _mapping_or_attr_get(event, "output_index")
    raw_key = _responses_output_stream_key(
        item=item,
        output_index=output_index,
        fallback_index=len(ordered_keys),
    )
    if isinstance(output_index, int) and output_index in key_by_output_index:
        key = key_by_output_index[output_index]
    else:
        key = key_aliases.get(raw_key, raw_key)
    if key not in ordered_keys:
        ordered_keys.append(key)

    existing = output_items.get(key, {})
    merged_item = {**existing, **item}
    if "arguments" in existing and "arguments" not in item:
        merged_item["arguments"] = existing["arguments"]
    output_items[key] = merged_item

    if isinstance(output_index, int):
        key_by_output_index[output_index] = key
    item_type = item.get("type")
    if item_type == "function_call":
        for alias in (raw_key, item.get("id"), item.get("call_id")):
            token = _function_call_identity_token(alias)
            if token is not None:
                key_aliases[token] = key
    else:
        for alias in (raw_key, item.get("id"), item.get("call_id")):
            if isinstance(alias, str) and alias.strip():
                key_aliases[alias.strip()] = key


def _record_collected_responses_arguments_event(
    *,
    event: Any,
    event_type: str,
    output_items: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    key_aliases: dict[str, str],
    key_by_output_index: dict[int, str],
) -> None:
    item_id = _mapping_or_attr_get(event, "item_id")
    output_index = _mapping_or_attr_get(event, "output_index")
    raw_key = _responses_output_stream_key(
        output_index=output_index,
        item_id=item_id,
        fallback_index=len(ordered_keys),
    )
    if isinstance(output_index, int) and output_index in key_by_output_index:
        key = key_by_output_index[output_index]
    else:
        key = key_aliases.get(raw_key, raw_key)
    if key not in ordered_keys:
        ordered_keys.append(key)

    existing = output_items.get(key, {})
    if not existing:
        item_type = "mcp_call" if "mcp_call" in event_type else "function_call"
        # Preserve Responses item id on `id` only. Do not synthesize call_id from
        # item_id: function_call.call_id is the exclusive upstream provider id and
        # may intentionally differ from the fc_* item id (OPENAI-007).
        existing = {"type": item_type}
        if item_type == "function_call":
            token = _function_call_identity_token(item_id)
            if token is not None:
                existing["id"] = token
        elif isinstance(item_id, str) and item_id.strip():
            existing["id"] = item_id.strip()

    value = _mapping_or_attr_get(event, "arguments")
    if not isinstance(value, str):
        value = _mapping_or_attr_get(event, "delta")
    if isinstance(value, str):
        if event_type.endswith(".delta"):
            existing["arguments"] = f"{existing.get('arguments', '')}{value}"
        else:
            existing["arguments"] = value

    output_items[key] = existing
    if isinstance(output_index, int):
        key_by_output_index[output_index] = key
    if existing.get("type") == "function_call":
        token = _function_call_identity_token(item_id)
        if token is not None:
            key_aliases[token] = key
    elif isinstance(item_id, str) and item_id.strip():
        key_aliases[item_id.strip()] = key


def _finalize_collected_responses_stream_response(
    *,
    response_dict: dict[str, Any],
    output_text_parts: list[str],
    output_items: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    key_aliases: dict[str, str],
    key_by_output_index: dict[int, str],
) -> dict[str, Any]:
    streamed_output = [output_items[key] for key in ordered_keys if key in output_items]
    completed_output = response_dict.get("output")
    if (
        output_text_parts
        and not _responses_output_has_message_text(streamed_output)
        and not _responses_output_has_message_text(completed_output)
    ):
        streamed_output.append(_build_collected_responses_text_output_item("".join(output_text_parts)))
    if streamed_output:
        response_dict["output"] = _merge_responses_output_lists(
            completed_output if isinstance(completed_output, list) else [],
            streamed_output,
            streamed_ordered_keys=ordered_keys,
            key_aliases=key_aliases,
            key_by_output_index=key_by_output_index,
        )
    elif not response_dict.get("output") and output_text_parts:
        response_dict["output"] = [_build_collected_responses_text_output_item("".join(output_text_parts))]
    return response_dict


def _build_empty_success_responses_diagnostic(
    *,
    response_body: dict[str, Any],
    diagnostic_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    output = response_body.get("output") or []
    usage = response_body.get("usage") or {}
    diagnostic = {
        "id": response_body.get("id"),
        "status": response_body.get("status"),
        "model": response_body.get("model"),
        "output_count": len(output) if isinstance(output, list) else 0,
        "output_types": [item.get("type") for item in output[:20] if isinstance(item, dict)]
        if isinstance(output, list)
        else [],
        "usage": usage if isinstance(usage, dict) else {},
        "error": response_body.get("error"),
        "incomplete_details": response_body.get("incomplete_details"),
    }
    if diagnostic_context:
        diagnostic["context"] = diagnostic_context
    return diagnostic


async def _collect_responses_response_from_stream(
    response: "StreamingResponse",
    *,
    event_summaries: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    output_text_parts: list[str] = []
    text_done_keys_seen: set[str] = set()
    output_items: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    key_aliases: dict[str, str] = {}
    key_by_output_index: dict[int, str] = {}
    terminal_response_dict: Optional[dict[str, Any]] = None
    event_iterator = _iterate_responses_sse_events(response.body_iterator)
    try:
        async for event in event_iterator:
            # RR-054 #27: stream events are plain dicts (or attr objects).
            event_type = _mapping_or_attr_get(event, "type")
            if event_summaries is not None and len(event_summaries) < 50:
                event_summaries.append(_responses_stream_event_summary(event))
            if event_type in {
                "response.output_item.added",
                "response.output_item.done",
            }:
                _record_collected_responses_output_item_event(
                    event=event,
                    output_items=output_items,
                    ordered_keys=ordered_keys,
                    key_aliases=key_aliases,
                    key_by_output_index=key_by_output_index,
                )
            if event_type == "response.output_text.delta":
                delta = _mapping_or_attr_get(event, "delta")
                if isinstance(delta, str):
                    output_text_parts.append(delta)
                    text_done_keys_seen.add(_responses_event_text_key(event))
            if event_type == "response.output_text.done":
                text = _mapping_or_attr_get(event, "text")
                text_key = _responses_event_text_key(event)
                if isinstance(text, str) and text and text_key not in text_done_keys_seen:
                    output_text_parts.append(text)
                    text_done_keys_seen.add(text_key)
            if event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
            }:
                _record_collected_responses_arguments_event(
                    event=event,
                    event_type=event_type,
                    output_items=output_items,
                    ordered_keys=ordered_keys,
                    key_aliases=key_aliases,
                    key_by_output_index=key_by_output_index,
                )
            if event_type in RESPONSES_API_TERMINAL_STREAM_EVENTS:  # noqa: F821
                response_payload = _mapping_or_attr_get(event, "response")
                if response_payload is None:
                    continue
                response_dict = _coerce_namespace_to_mapping(response_payload)
                if isinstance(response_dict, dict):
                    terminal_response_dict = response_dict
    finally:
        if terminal_response_dict is None:
            await event_iterator.aclose()
            body_iterator = getattr(response, "body_iterator", None)
            aclose = getattr(body_iterator, "aclose", None)
            if callable(aclose):
                await aclose()
    if terminal_response_dict is not None:
        return _finalize_collected_responses_stream_response(
            response_dict=terminal_response_dict,
            output_text_parts=output_text_parts,
            output_items=output_items,
            ordered_keys=ordered_keys,
            key_aliases=key_aliases,
            key_by_output_index=key_by_output_index,
        )
    raise HTTPException(
        status_code=502,
        detail="OpenAI Responses stream completed without a response payload.",
    )
